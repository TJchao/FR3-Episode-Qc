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
    assert metadata["source_counts"]["image_blur"] == {
        "source_frame_count": 4,
        "source_auto_hit_count": 1,
    }


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


def test_gold_set_metrics_require_review() -> None:
    rows, metadata = build_gold_set(METRIC_ROWS, HIT_ROWS)

    metrics = evaluate_gold_set(rows, metadata["source_counts"])

    assert metrics["evaluated_rows"] == 0
    assert metrics["review_completion"] == 0
    assert metrics["precision"] is None
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

    assert rule["confusion_matrix"] == {"tn": 3, "tp": 1}
    assert rule["precision"] == 1.0
    assert rule["recall"] == 1.0
    assert rule["fpr"] == 0.0
    assert metrics["f1"] == 1.0


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
    matrix = metrics["confusion_matrix"]

    assert matrix["fp"] >= 1
    assert matrix["fn"] >= 1
    assert metrics["precision"] < 1.0
    assert metrics["recall"] < 1.0
