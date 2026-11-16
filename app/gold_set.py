"""Build and evaluate a lightweight image-rule gold set."""

from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


GOLD_FIELDS = (
    "rule_id",
    "episode_id",
    "frame_index",
    "timestamp_s",
    "evidence_path",
    "auto_hit",
    "human_label",
    "is_true_positive",
    "note",
)


def parse_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "positive", "defect", "hit"}:
        return True
    if text in {"0", "false", "no", "n", "negative", "normal", "miss"}:
        return False
    return None


def _round_robin(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_episode[str(row.get("episode_id"))].append(row)
    selected: list[dict[str, Any]] = []
    queues = [by_episode[key] for key in sorted(by_episode)]
    while len(selected) < limit:
        added = False
        for queue in queues:
            if len(selected) >= limit:
                break
            if queue:
                selected.append(queue.pop(0))
                added = True
        if not added:
            break
    return selected


def build_gold_set(
    metric_rows: list[dict[str, Any]],
    hit_rows: list[dict[str, Any]],
    *,
    max_hits_per_rule: int = 50,
    max_controls_per_rule: int = 50,
    census_threshold: int = 30,
    seed: int = 7,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Create a review template with auto hits and deterministic controls."""
    rng = random.Random(seed)
    hit_keys = {
        (str(row.get("rule_id")), str(row.get("episode_id")), int(row.get("frame_index", -1)))
        for row in hit_rows
    }
    rules = sorted({str(row.get("rule_id")) for row in hit_rows})
    output: list[dict[str, Any]] = []
    source_counts: dict[str, dict[str, int]] = {}

    shuffled_metrics = list(metric_rows)
    rng.shuffle(shuffled_metrics)
    for rule in rules:
        hits = [
            row for row in hit_rows
            if str(row.get("rule_id")) == rule
        ]
        controls = [
            row for row in shuffled_metrics
            if (rule, str(row.get("episode_id")), int(row.get("frame_index", -1)))
            not in hit_keys
        ]
        use_census = len(hits) < census_threshold
        selected_hits = hits if use_census else _round_robin(hits, max_hits_per_rule)
        selected_controls = _round_robin(controls, max_controls_per_rule)
        for row in selected_hits:
            output.append({
                "rule_id": rule,
                "episode_id": row.get("episode_id"),
                "frame_index": row.get("frame_index"),
                "timestamp_s": row.get("timestamp_s"),
                "evidence_path": row.get("evidence_path"),
                "auto_hit": True,
                "human_label": "",
                "is_true_positive": "",
                "note": "",
            })
        for row in selected_controls:
            output.append({
                "rule_id": rule,
                "episode_id": row.get("episode_id"),
                "frame_index": row.get("frame_index"),
                "timestamp_s": row.get("timestamp_s"),
                "evidence_path": row.get("evidence_path"),
                "auto_hit": False,
                "human_label": "",
                "is_true_positive": "",
                "note": "",
            })
        source_counts[rule] = {
            "source_frame_count": len(metric_rows),
            "source_auto_hit_count": len(hits),
            "selected_auto_hit_count": len(selected_hits),
            "auto_hit_sampling": "census" if use_census else "round_robin_sample",
        }

    output.sort(key=lambda row: (
        str(row["rule_id"]),
        str(row["episode_id"]),
        int(row["frame_index"]),
        -int(bool(row["auto_hit"])),
    ))
    metadata = {
        "sampling": (
            "rules with fewer than 30 auto hits use hit census; larger rules and "
            "controls use round-robin sampling by episode"
        ),
        "census_threshold": census_threshold,
        "rules": rules,
        "source_counts": source_counts,
        "rows": len(output),
        "requires_human_review": True,
    }
    return output, metadata


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def evaluate_gold_set(
    rows: list[dict[str, Any]],
    source_counts: dict[str, dict[str, int]] | None = None,
) -> dict[str, Any]:
    """Compute reviewed precision and source-weighted FPR estimates."""
    source_counts = source_counts or {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("rule_id"))].append(row)

    per_rule: dict[str, Any] = {}
    all_counts = Counter()
    weighted_false_positives: list[float] = []
    weighted_true_negatives: list[float] = []
    for rule, rule_rows in sorted(grouped.items()):
        counts = Counter()
        hit_counts = Counter()
        control_counts = Counter()
        ambiguous = 0
        for row in rule_rows:
            truth = parse_bool(row.get("is_true_positive"))
            if truth is None:
                truth = parse_bool(row.get("human_label"))
            auto = parse_bool(row.get("auto_hit"))
            if truth is None or auto is None:
                ambiguous += 1
                continue
            if auto and truth:
                counts["tp"] += 1
                hit_counts["tp"] += 1
            elif auto and not truth:
                counts["fp"] += 1
                hit_counts["fp"] += 1
            elif not auto and truth:
                counts["fn"] += 1
                control_counts["fn"] += 1
            else:
                counts["tn"] += 1
                control_counts["tn"] += 1

        all_counts.update(counts)
        source = source_counts.get(rule, {})
        source_frames = int(source.get("source_frame_count", 0))
        source_hits = int(source.get("source_auto_hit_count", 0))
        reviewed_hit_count = sum(hit_counts.values())
        reviewed_control_count = sum(control_counts.values())
        precision = _safe_ratio(hit_counts["tp"], reviewed_hit_count)
        weighted_fpr = None
        estimated_tp = None
        estimated_fp = None
        estimated_fn = None
        estimated_tn = None
        if source_frames and reviewed_control_count:
            source_non_hits = source_frames - source_hits
            true_hit_fraction = (
                hit_counts["tp"] / reviewed_hit_count
                if reviewed_hit_count
                else 0.0
            )
            false_non_hit_fraction = (
                control_counts["fn"] / reviewed_control_count
                if reviewed_control_count
                else 0.0
            )
            estimated_tp = source_hits * true_hit_fraction
            estimated_fp = source_hits * (1.0 - true_hit_fraction)
            estimated_fn = source_non_hits * false_non_hit_fraction
            estimated_tn = source_non_hits - estimated_fn
            weighted_false_positives.append(estimated_fp)
            weighted_true_negatives.append(estimated_tn)
            denominator = estimated_fp + estimated_tn
            weighted_fpr = (
                round(estimated_fp / denominator, 6) if denominator > 0 else None
            )
        result: dict[str, Any] = {
            "rows": len(rule_rows),
            "evaluated_rows": sum(counts.values()),
            "ambiguous_or_unreviewed": ambiguous,
            "review_completion": _safe_ratio(sum(counts.values()), len(rule_rows)),
            "reviewed_sample_counts": dict(sorted(counts.items())),
            "precision": precision,
            "fpr_estimate": weighted_fpr,
            "recall": None,
            "f1": None,
            "recall_f1_null_reason": (
                "sampling design does not cover false negatives"
            ),
            "source_frame_count": source_frames or None,
            "source_auto_hit_count": source_hits or None,
            "automation_coverage": None,
            "automation_coverage_definition": (
                "1 - review candidates / sampled frames; assumes every auto hit needs review"
            ),
            "fpr_estimate_method": (
                "case-control weighted estimate using source hit/non-hit counts"
            ),
            "estimated_true_positive_count": round(estimated_tp, 6) if estimated_tp is not None else None,
            "estimated_false_positive_count": round(estimated_fp, 6) if estimated_fp is not None else None,
            "estimated_false_negative_count": round(estimated_fn, 6) if estimated_fn is not None else None,
            "estimated_true_negative_count": round(estimated_tn, 6) if estimated_tn is not None else None,
        }
        if source_frames:
            result["automation_coverage"] = round(1 - source_hits / source_frames, 6)
        per_rule[rule] = result

    overall_precision = _safe_ratio(all_counts["tp"], all_counts["tp"] + all_counts["fp"])
    total_weighted_fp = sum(weighted_false_positives)
    total_weighted_tn = sum(weighted_true_negatives)
    overall_weighted_fpr = (
        round(total_weighted_fp / (total_weighted_fp + total_weighted_tn), 6)
        if total_weighted_fp + total_weighted_tn > 0
        else None
    )
    null_reason = "sampling design does not cover false negatives"
    return {
        "available": sum(all_counts.values()) > 0,
        "requires_human_review": sum(all_counts.values()) == 0,
        "rows": len(rows),
        "evaluated_rows": sum(all_counts.values()),
        "ambiguous_or_unreviewed": len(rows) - sum(all_counts.values()),
        "review_completion": _safe_ratio(sum(all_counts.values()), len(rows)),
        "reviewed_sample_counts": dict(sorted(all_counts.items())),
        "precision": overall_precision,
        "fpr_estimate": overall_weighted_fpr,
        "recall": None,
        "f1": None,
        "recall_f1_null_reason": null_reason,
        "metric_boundary": (
            "Precision is measured on reviewed auto hits. FPR is a case-control "
            "weighted estimate using source hit/non-hit counts. Recall and F1 "
            "are null because this sampling design does not cover all false negatives."
        ),
        "per_rule": per_rule,
    }


def write_gold_set(
    rows: list[dict[str, Any]],
    metadata: dict[str, Any],
    output_csv: str | Path,
    metadata_path: str | Path,
) -> None:
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8-sig", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=GOLD_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    Path(metadata_path).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def read_gold_set(
    csv_path: str | Path,
    metadata_path: str | Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with Path(csv_path).open(encoding="utf-8-sig", newline="") as file_handle:
        rows = list(csv.DictReader(file_handle))
    metadata: dict[str, Any] = {}
    if metadata_path is not None and Path(metadata_path).is_file():
        metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    return rows, metadata
