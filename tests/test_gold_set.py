"""Tests for image-rule gold-set construction and metrics."""

from __future__ import annotations

from app.gold_set import _round_robin, build_gold_set, evaluate_gold_set


METRIC_ROWS = [
    {"episode_id": "ep0", "frame_index": 0, "timestamp_s": 0.0, "evidence_path": "a"},
    {"episode_id": "ep0", "frame_index": 1, "timestamp_s": 1.0, "evidence_path": "b"},
    {"episode_id": "ep0", "frame_index": 2, "timestamp_s": 2.0, "evidence_path": "c"},
    {"episode_id": "ep0", "frame_index": 3, "timestamp_s": 3.0, "evidence_path": "d"},
]

HIT_ROWS = [
    {"rule_id": "image_blur", "episode_id": "ep0", "frame_index": 1, "timestamp_s": 1.0},
]


def test_build_gold_set_includes_hit_and_control() -> None:
    rows, metadata = build_gold_set(
        METRIC_ROWS,
        HIT_ROWS,
        max_hits_per_rule=1,
        max_controls_per_rule=1,
    )

    assert len(rows) == 2
    assert {row["auto_hit"] for row in rows} == {True, False}
    source = metadata["source_counts"]["image_blur"]
    assert source["source_frame_count"] == 4
    assert source["source_auto_hit_count"] == 1
    assert source["selected_auto_hit_count"] == 1
    assert source["auto_hit_sampling"] == "census"


def test_build_gold_set_round_robins_across_episodes() -> None:
    metric_rows = [
        {"episode_id": f"ep{index}", "frame_index": frame, "timestamp_s": frame}
        for index in range(3)
        for frame in range(4)
    ]
    hit_rows = [
        {
            "rule_id": "image_blur",
            "episode_id": f"ep{index}",
            "frame_index": frame,
            "timestamp_s": frame,
        }
        for index in range(3)
        for frame in range(4)
    ]

    rows = _round_robin(hit_rows, 6)

    assert [row["episode_id"] for row in rows] == [
        "ep0", "ep1", "ep2", "ep0", "ep1", "ep2",
    ]

    sampled_rows, sampled_meta = build_gold_set(
        metric_rows,
        hit_rows,
        max_hits_per_rule=6,
        max_controls_per_rule=0,
        census_threshold=12,
    )
    census_rows, census_meta = build_gold_set(
        metric_rows,
        hit_rows,
        max_hits_per_rule=6,
        max_controls_per_rule=0,
        census_threshold=13,
    )
    assert len(sampled_rows) == 6
    assert sampled_meta["source_counts"]["image_blur"]["auto_hit_sampling"] == "round_robin_sample"
    assert len(census_rows) == 12
    assert census_meta["source_counts"]["image_blur"]["auto_hit_sampling"] == "census"


def test_gold_set_metrics_require_review() -> None:
    rows, metadata = build_gold_set(METRIC_ROWS, HIT_ROWS)

    metrics = evaluate_gold_set(rows, metadata["source_counts"])

    assert metrics["evaluated_rows"] == 0
    assert metrics["review_completion"] == 0
    assert metrics["precision"] is None
    assert metrics["recall"] is None
    assert metrics["f1"] is None
    assert metrics["recall_f1_null_reason"] == "sampling design does not cover false negatives"
    assert metrics["per_rule"]["image_blur"]["automation_coverage"] == 0.75


def test_gold_set_metrics_compute_confusion_matrix() -> None:
    rows, metadata = build_gold_set(
        METRIC_ROWS,
        HIT_ROWS,
        max_hits_per_rule=1,
        max_controls_per_rule=3,
    )
    for row in rows:
        if row["auto_hit"]:
            row["is_true_positive"] = "1"
        else:
            row["is_true_positive"] = "0"

    metrics = evaluate_gold_set(rows, metadata["source_counts"])
    rule = metrics["per_rule"]["image_blur"]

    assert rule["reviewed_sample_counts"] == {"tn": 3, "tp": 1}
    assert rule["precision"] == 1.0
    assert rule["recall"] is None
    assert rule["fpr_estimate"] == 0.0
    assert rule["f1"] is None
    assert metrics["recall"] is None
    assert metrics["f1"] is None


def test_gold_set_metrics_handle_false_positive_and_miss() -> None:
    rows, _ = build_gold_set(METRIC_ROWS, HIT_ROWS)
    for row in rows:
        row["human_label"] = not bool(row["auto_hit"])
    # The selected hit is a false positive, the selected control is a true
    # negative, and the appended row is a false negative.
    rows.append({
        "rule_id": "image_blur",
        "episode_id": "ep0",
        "frame_index": 2,
        "timestamp_s": 2.0,
        "evidence_path": "c",
        "auto_hit": False,
        "human_label": True,
        "is_true_positive": "",
        "note": "",
    })

    metrics = evaluate_gold_set(rows)
    matrix = metrics["reviewed_sample_counts"]

    assert matrix["fp"] >= 1
    assert matrix["fn"] >= 1
    assert metrics["precision"] < 1.0
    assert metrics["recall"] is None


def test_fpr_is_weighted_by_source_hit_and_non_hit_counts() -> None:
    rows = [
        {
            "rule_id": "image_blur",
            "episode_id": "ep0",
            "frame_index": 0,
            "timestamp_s": 0.0,
            "evidence_path": "",
            "auto_hit": True,
            "human_label": False,
            "is_true_positive": False,
            "note": "",
        },
        {
            "rule_id": "image_blur",
            "episode_id": "ep0",
            "frame_index": 1,
            "timestamp_s": 1.0,
            "evidence_path": "",
            "auto_hit": False,
            "human_label": True,
            "is_true_positive": True,
            "note": "",
        },
    ]
    source_counts = {
        "image_blur": {
            "source_frame_count": 100,
            "source_auto_hit_count": 10,
        }
    }

    metrics = evaluate_gold_set(rows, source_counts)
    rule = metrics["per_rule"]["image_blur"]

    # Among 10 source hits, the reviewed 0% precision estimates 10 false
    # positives. Among 90 non-hits, the reviewed control implies 90 false
    # negatives, leaving zero estimated true negatives.
    assert rule["precision"] == 0.0
    assert rule["estimated_false_positive_count"] == 10.0
    assert rule["estimated_true_negative_count"] == 0.0
    assert rule["fpr_estimate"] == 1.0
