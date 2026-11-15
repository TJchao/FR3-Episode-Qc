"""Build and evaluate a lightweight image-rule gold set."""

from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


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
        selected_hits = _round_robin(hits, max_hits_per_rule)
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
        }

    output.sort(key=lambda row: (
        str(row["rule_id"]),
        str(row["episode_id"]),
        int(row["frame_index"]),
        -int(bool(row["auto_hit"])),
    ))
    metadata = {
 "sampling": "round-robin by episode; equal controls up to the configured cap",
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
    """Compute per-rule confusion matrices after human labels are filled."""
    source_counts = source_counts or {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("rule_id"))].append(row)

    per_rule: dict[str, Any] = {}
    all_counts = Counter()
    for rule, rule_rows in sorted(grouped.items()):
        counts = Counter()
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
            elif auto and not truth:
                counts["fp"] += 1
            elif not auto and truth:
                counts["fn"] += 1
            else:
                counts["tn"] += 1

        all_counts.update(counts)
        source = source_counts.get(rule, {})
        source_frames = int(source.get("source_frame_count", 0))
        source_hits = int(source.get("source_auto_hit_count", 0))
        result: dict[str, Any] = {
            "rows": len(rule_rows),
            "evaluated_rows": sum(counts.values()),
            "ambiguous_or_unreviewed": ambiguous,
            "review_completion": _safe_ratio(sum(counts.values()), len(rule_rows)),
            "confusion_matrix": dict(sorted(counts.items())),
            "precision": _safe_ratio(counts["tp"], counts["tp"] + counts["fp"]),
            "recall": _safe_ratio(counts["tp"], counts["tp"] + counts["fn"]),
            "fpr": _safe_ratio(counts["fp"], counts["fp"] + counts["tn"]),
            "f1": None,
            "source_frame_count": source_frames or None,
            "source_auto_hit_count": source_hits or None,
            "automation_coverage": None,
            "automation_coverage_definition": (
                "1 - review candidates / sampled frames; assumes every auto hit needs review"
            ),
        }
        precision = result["precision"]
        recall = result["recall"]
        if precision is not None and recall is not None and precision + recall > 0:
            result["f1"] = round(2 * precision * recall / (precision + recall), 6)
        if source_frames:
            result["automation_coverage"] = round(1 - source_hits / source_frames, 6)
        per_rule[rule] = result

    overall_precision = _safe_ratio(all_counts["tp"], all_counts["tp"] + all_counts["fp"])
    overall_recall = _safe_ratio(all_counts["tp"], all_counts["tp"] + all_counts["fn"])
    return {
        "available": sum(all_counts.values()) > 0,
        "requires_human_review": sum(all_counts.values()) == 0,
        "rows": len(rows),
        "evaluated_rows": sum(all_counts.values()),
        "ambiguous_or_unreviewed": len(rows) - sum(all_counts.values()),
        "review_completion": _safe_ratio(sum(all_counts.values()), len(rows)),
        "confusion_matrix": dict(sorted(all_counts.items())),
        "precision": overall_precision,
        "recall": overall_recall,
        "fpr": _safe_ratio(all_counts["fp"], all_counts["fp"] + all_counts["tn"]),
        "f1": (
            round(2 * overall_precision * overall_recall / (overall_precision + overall_recall), 6)
            if overall_precision is not None
            and overall_recall is not None
            and overall_precision + overall_recall > 0
            else None
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
