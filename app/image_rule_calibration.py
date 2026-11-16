"""Calibrate image-rule thresholds against a reviewed gold set."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from app.gold_set import parse_bool
from app.image_rules import hamming_distance


def read_csv_rows(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as file_handle:
        return list(csv.DictReader(file_handle))


def read_metric_rows(path: str | Path) -> list[dict[str, Any]]:
    return pq.read_table(path).to_pylist()


def _counts(pairs: list[tuple[bool, bool]]) -> dict[str, int]:
    return {
        "tp": sum(flag and truth for flag, truth in pairs),
        "fp": sum(flag and not truth for flag, truth in pairs),
        "fn": sum(not flag and truth for flag, truth in pairs),
        "tn": sum(not flag and not truth for flag, truth in pairs),
    }


def _summary(pairs: list[tuple[bool, bool]]) -> dict[str, Any]:
    counts = _counts(pairs)
    flagged = counts["tp"] + counts["fp"]
    return {
        **counts,
        "flagged_reviewed_rows": flagged,
        "sample_precision": round(counts["tp"] / flagged, 6) if flagged else None,
        "boundary": (
            "Precision applies only to reviewed gold-set rows. It is not a "
            "population recall estimate."
        ),
    }


def _reviewed_truth(row: dict[str, Any]) -> bool | None:
    truth = parse_bool(row.get("is_true_positive"))
    if truth is None:
        truth = parse_bool(row.get("human_label"))
    return truth


def _best_threshold(rows: dict[str, dict[str, Any]]) -> str:
    eligible = [
        (name, row)
        for name, row in rows.items()
        if row["sample_precision"] is not None
    ]
    if not eligible:
        return "no threshold flagged reviewed rows"
    return max(
        eligible,
        key=lambda item: (
            item[1]["sample_precision"],
            item[1]["flagged_reviewed_rows"],
        ),
    )[0]


def calibrate(
    gold_rows: list[dict[str, Any]],
    metric_rows: list[dict[str, Any]],
    *,
    blur_percentiles: tuple[float, ...] = (2.0, 5.0, 10.0),
    static_hamming_thresholds: tuple[int, ...] = (0, 2, 4, 6, 8),
) -> dict[str, Any]:
    metrics_by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in metric_rows:
        metrics_by_episode[str(row["episode_id"])].append(row)
    for rows in metrics_by_episode.values():
        rows.sort(key=lambda row: int(row["frame_index"]))
    lookup = {
        (str(row["episode_id"]), int(row["frame_index"])): row
        for rows in metrics_by_episode.values()
        for row in rows
    }

    blur_rules: dict[str, Any] = {}
    blur_rows = [row for row in gold_rows if row.get("rule_id") == "image_blur"]
    for percentile in blur_percentiles:
        pairs: list[tuple[bool, bool]] = []
        for gold_row in blur_rows:
            episode_id = str(gold_row["episode_id"])
            frame_index = int(gold_row["frame_index"])
            metric = lookup[(episode_id, frame_index)]
            episode_values = [float(row["laplacian_variance"]) for row in metrics_by_episode[episode_id]]
            threshold = float(np.percentile(episode_values, percentile))
            truth = _reviewed_truth(gold_row)
            if truth is None:
                continue
            pairs.append((float(metric["laplacian_variance"]) < threshold, truth))
        blur_rules[f"P{percentile:g}"] = _summary(pairs)

    static_rules: dict[str, Any] = {}
    static_rows = [row for row in gold_rows if row.get("rule_id") == "image_static_candidate"]
    for threshold in static_hamming_thresholds:
        pairs = []
        for gold_row in static_rows:
            episode_id = str(gold_row["episode_id"])
            frame_index = int(gold_row["frame_index"])
            position = next(
                index
                for index, row in enumerate(metrics_by_episode[episode_id])
                if int(row["frame_index"]) == frame_index
            )
            current = metrics_by_episode[episode_id][position]
            previous = (
                metrics_by_episode[episode_id][position - 1]
                if position > 0
                else None
            )
            distance = (
                hamming_distance(str(previous["phash"]), str(current["phash"]))
                if previous is not None
                else 10**6
            )
            truth = _reviewed_truth(gold_row)
            if truth is None:
                continue
            pairs.append((distance <= threshold, truth))
        static_rules[f"H{threshold}"] = _summary(pairs)

    return {
        "reviewed_rows": len(gold_rows),
        "blur_percentile": blur_rules,
        "static_hamming": static_rules,
        "recommended_interpretation": {
            "image_blur": (
                f"{_best_threshold(blur_rules)} has the highest reviewed-sample "
                "precision in this gold set; validate on another batch before "
                "changing a production policy."
            ),
            "image_static_candidate": (
                f"{_best_threshold(static_rules)} has the highest reviewed-sample "
                "precision in this gold set; validate on another batch before "
                "changing a production policy."
            ),
        },
        "boundary": (
            "All results are conditional on the reviewed gold-set sample. "
            "Recall and population FPR require a different sampling design."
        ),
    }


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Image Rule Threshold Calibration",
        "",
        f"Reviewed rows: `{summary['reviewed_rows']}`",
        "",
        "## Blur Percentile",
        "",
        "| Threshold | Flagged | TP | FP | FN | TN | Sample precision |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in summary["blur_percentile"].items():
        lines.append(
            f"| `{name}` | {row['flagged_reviewed_rows']} | {row['tp']} | "
            f"{row['fp']} | {row['fn']} | {row['tn']} | {row['sample_precision']} |"
        )
    lines.extend([
        "",
        "## Static pHash Hamming",
        "",
        "| Threshold | Flagged | TP | FP | FN | TN | Sample precision |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for name, row in summary["static_hamming"].items():
        lines.append(
            f"| `{name}` | {row['flagged_reviewed_rows']} | {row['tp']} | "
            f"{row['fp']} | {row['fn']} | {row['tn']} | {row['sample_precision']} |"
        )
    lines.extend([
        "",
        summary["boundary"],
        "",
    ])
    return "\n".join(lines)
