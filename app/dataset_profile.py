"""Aggregate LeRobot QC, motion, image, and embedding reports."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from app.motion_profile import build_motion_profile
from app.profile import ProfileError, load_profile
from app.readers.lerobot import (
    discover_layout,
    inspect_layout,
    iter_selected_episode_arrays,
    parse_episode_selection,
)


def _values_summary(values: list[float | int | None]) -> dict[str, float | None]:
    valid = [float(value) for value in values if value is not None and np.isfinite(value)]
    if not valid:
        return {
            "count": 0,
            "min": None,
            "mean": None,
            "p50": None,
            "p95": None,
            "max": None,
        }
    array = np.asarray(valid)
    return {
        "count": len(valid),
        "min": round(float(np.min(array)), 6),
        "mean": round(float(np.mean(array)), 6),
        "p50": round(float(np.percentile(array, 50)), 6),
        "p95": round(float(np.percentile(array, 95)), 6),
        "max": round(float(np.max(array)), 6),
    }


def _optional_parquet(path: str | Path | None) -> pa.Table | None:
    if path is None:
        return None
    path = Path(path)
    return pq.read_table(path) if path.is_file() else None


def _optional_json(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _motion_summary(motions: dict[str, Any], fps: float | None) -> dict[str, Any]:
    result: dict[str, Any] = {}
    metric_names = (
        "velocity_p50",
        "velocity_p95",
        "velocity_p99",
        "acceleration_p95",
        "jerk_p95",
        "relative_static_tail_ratio",
        "motion_range",
        "active_dimension_ratio",
    )

    profiles = [value for value in motions.values() if isinstance(value, dict)]
    result["all_dimensions"] = {
        name: _values_summary([profile.get(name) for profile in profiles])
        for name in metric_names
    }
    group_names = sorted({
        group
        for profile in profiles
        for group in profile.get("groups", {})
    })
    result["groups"] = {
        group: {
            name: _values_summary([
                profile.get("groups", {}).get(group, {}).get(name)
                for profile in profiles
            ])
            for name in metric_names
        }
        for group in group_names
    }
    result["state_unit"] = next(
        (profile.get("state_unit") for profile in profiles if profile.get("state_unit")),
        "unknown",
    )
    resolved_fps = float(fps or 0.0)
    result["finite_difference_boundary"] = {
        "differentiation_order": 3,
        "fps": resolved_fps,
        "theoretical_noise_gain": round(resolved_fps**3, 6) if resolved_fps else None,
        "jerk_noise_floor": None,
        "jerk_noise_floor_reason": (
            "not estimated: no verified static segments or sensor noise model; "
            "absolute jerk is not comparable across datasets"
        ),
        "interpretation": (
            "Jerk is used only for within-dataset relative ranking because "
            "third-order finite differencing amplifies noise."
        ),
    }
    if "left_arm" in result["groups"] and "right_hand" in result["groups"]:
        left_values = [
            profile.get("groups", {}).get("left_arm", {}).get("velocity_p95")
            for profile in profiles
        ]
        right_values = [
            profile.get("groups", {}).get("right_hand", {}).get("velocity_p95")
            for profile in profiles
        ]
        paired = [
            (float(left), float(right))
            for left, right in zip(left_values, right_values)
            if left is not None and right is not None and np.isfinite(left) and np.isfinite(right)
        ]
        ratios = [right / left for left, right in paired if left > 1e-9]
        result["group_comparison"] = {
            "metric": "velocity_p95",
            "left_group": "left_arm",
            "right_group": "right_hand",
            "episode_count": len(paired),
            "right_hand_greater_count": sum(right > left for left, right in paired),
            "paired_ratio_p50": (
                round(float(np.percentile(ratios, 50)), 6) if ratios else None
            ),
            "boundary": (
                "This compares the same episode across groups. It avoids comparing "
                "separate maxima from different episodes."
            ),
        }
    return result


def _image_summary(
    image_metrics_path: str | Path | None,
    image_hits_path: str | Path | None,
) -> dict[str, Any]:
    table = _optional_parquet(image_metrics_path)
    if table is None:
        return {"available": False}
    result: dict[str, Any] = {"available": True, "sample_count": table.num_rows}
    if "episode_id" in table.column_names:
        result["episode_count"] = len(set(table["episode_id"].to_pylist()))
    for field in (
        "brightness_mean",
        "brightness_std",
        "highlight_ratio",
        "shadow_ratio",
        "laplacian_variance",
    ):
        if field in table.column_names:
            values = table[field].to_pylist()
            result[field] = _values_summary(values)
    hits_path = Path(image_hits_path or "")
    if hits_path.is_file():
        with hits_path.open(encoding="utf-8-sig", newline="") as file_handle:
            hits = list(csv.DictReader(file_handle))
        result["rule_hit_counts"] = dict(sorted(Counter(
            row.get("rule_id", "") for row in hits
        ).items()))
        result["rule_hit_rows"] = len(hits)
    return result


def _rule_coverage(episodes: dict[int, Any]) -> dict[str, Any]:
    """Report which low-dimensional rules were executable for this source format."""
    array_fields: set[str] = set()
    metadata_fields: set[str] = set()
    for episode in episodes.values():
        array_fields.update(episode.arrays)
        metadata_fields.update(episode.metadata)

    requirements = (
        ("episode_length_metadata", "metadata", "expected_frame_count"),
        ("timestamp_monotonic", "array", "timestamp_s"),
        ("state_shape", "array", "state"),
        ("state_finite", "array", "state"),
        ("action_shape", "array", "action"),
        ("action_finite", "array", "action"),
        ("camera_age_p95", "array", "camera_age_ms"),
        ("robot_time_monotonic", "array", "robot_time_ns"),
        ("rgb_frame_count", "metadata", "rgb_frame_count"),
        ("depth_frame_count", "metadata", "depth_frame_count"),
    )
    evaluated: list[str] = []
    skipped: list[dict[str, str]] = []
    for rule, scope, field_name in requirements:
        available = (
            field_name in array_fields if scope == "array"
            else field_name in metadata_fields
        )
        if available:
            evaluated.append(rule)
        else:
            skipped.append({
                "rule": rule,
                "reason": f"missing_field: {field_name}",
            })
    return {
        "rules_evaluated": len(evaluated),
        "evaluated_rules": evaluated,
        "rules_skipped": skipped,
        "interpretation": (
            "Pass means all executable rules passed. Missing fields are skipped, "
            "not implicitly passed."
        ),
    }


def _profile_records(
    dataset_dir: str | Path,
    profile_name: str,
    episode_selection: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    layout = discover_layout(dataset_dir)
    inspect_report = inspect_layout(layout)
    try:
        profile = load_profile(profile_name)
    except ProfileError:
        profile = {
            "state_dim": inspect_report.get("state_shape", [None])[-1],
            "action_dim": inspect_report.get("action_shape", [None])[-1],
            "fps": inspect_report.get("fps"),
            "motion": {},
        }

    total = int(layout.info.get("total_episodes") or 0)
    if total <= 0:
        total = max(
            int(value)
            for value in pq.read_table(
                layout.episode_files[0], columns=["episode_index"]
            )["episode_index"].to_pylist()
        ) + 1
    selected = parse_episode_selection(episode_selection, total)
    episode_rows, episodes = iter_selected_episode_arrays(layout, selected)
    rule_coverage = _rule_coverage(episodes)

    records: list[dict[str, Any]] = []
    motions: dict[str, Any] = {}
    errors_by_rule = Counter()
    for index in selected:
        episode = episodes.get(index)
        if episode is None:
            continue
        arrays = episode.arrays
        metadata = episode.metadata
        expected = metadata.get("expected_frame_count")
        actual = int(arrays.get("timestamp_s", np.array([])).shape[0])
        errors: list[str] = []
        if expected is not None and actual != int(expected):
            errors.append(
                "LeRobot episode length mismatch: "
                f"metadata={int(expected)}, data={actual}"
            )
        timestamp = arrays.get("timestamp_s", np.array([]))
        if timestamp.ndim != 1 or not np.all(np.diff(timestamp) > 0):
            errors.append("timestamp_s is not strictly increasing")
        for field in ("state", "action"):
            values = arrays.get(field)
            if values is None or values.ndim != 2:
                errors.append(f"{field} shape must be 2-D")
            elif not np.all(np.isfinite(values)):
                errors.append(f"{field} contains NaN or Inf")

        motion = build_motion_profile(
            arrays["state"],
            arrays["action"],
            fps=float(metadata.get("fps") or profile.get("fps") or 1.0),
            indices=profile.get("motion", {}).get("indices"),
            state_names=metadata.get("state_names"),
            groups=profile.get("motion", {}).get("groups"),
            state_unit=profile.get("motion", {}).get("unit"),
        )
        episode_id = str(metadata.get("episode_id") or f"episode_{index:06d}")
        motions[episode_id] = motion
        error_types: set[str] = set()
        for error in errors:
            if "episode length mismatch" in error:
                error_types.add("lerobot_length_mismatch")
            elif "timestamp_s" in error:
                error_types.add("timestamp_backwards")
            elif "NaN" in error:
                error_types.add("nan_or_inf")
            elif "state shape" in error:
                error_types.add("state_shape_error")
            elif "action shape" in error:
                error_types.add("action_shape_error")
            else:
                error_types.add("unknown")
        error_types = sorted(error_types)
        errors_by_rule.update(error_types)
        records.append({
            "episode_id": episode_id,
            "status": "passed" if not errors else "failed",
            "severity": "none" if not errors else "fatal",
            "error_types": ";".join(error_types),
            "error_count": len(errors),
            "camera_age_p95_ms": None,
            "frame_count": actual,
            "episode_hash": "",
            "dataset_version": "",
            "git_commit": "",
            "created_at": "",
            "duration_s": round(actual / float(metadata.get("fps") or 1.0), 6),
        })
    return records, motions, {
        "inspect": inspect_report,
        "error_type_counts": dict(sorted(errors_by_rule.items())),
        "rule_coverage": rule_coverage,
    }


def build_dataset_profile(
    dataset_dir: str | Path,
    *,
    profile_name: str,
    episode_selection: str,
    image_metrics_path: str | Path | None = None,
    image_hits_path: str | Path | None = None,
    embedding_summary_path: str | Path | None = None,
    gold_set_metrics_path: str | Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    records, motions, low_dim = _profile_records(
        dataset_dir, profile_name, episode_selection
    )
    frame_counts = [record["frame_count"] for record in records]
    durations = [record["duration_s"] for record in records]
    severity_counts = Counter(record["severity"] for record in records)
    status_counts = Counter(record["status"] for record in records)

    embedding_summary = _optional_json(embedding_summary_path)
    gold_metrics = _optional_json(gold_set_metrics_path)
    image_summary = _image_summary(image_metrics_path, image_hits_path)
    profile: dict[str, Any] = {
        "dataset": {
            "dir": str(Path(dataset_dir).resolve()),
            "robot_type": low_dim["inspect"].get("robot_type"),
            "codebase_version": low_dim["inspect"].get("codebase_version"),
            "fps": low_dim["inspect"].get("fps"),
            "state_shape": low_dim["inspect"].get("state_shape"),
            "action_shape": low_dim["inspect"].get("action_shape"),
            "selected_episodes": len(records),
        },
        "quality": {
            "status_counts": dict(sorted(status_counts.items())),
            "severity_counts": dict(sorted(severity_counts.items())),
            "error_type_counts": low_dim["error_type_counts"],
            "rules_evaluated": low_dim["rule_coverage"]["rules_evaluated"],
            "rules_skipped": low_dim["rule_coverage"]["rules_skipped"],
            "rule_coverage_interpretation": low_dim["rule_coverage"]["interpretation"],
            "frame_count": _values_summary(frame_counts),
            "duration_s": _values_summary(durations),
        },
        "motion": _motion_summary(
            motions,
            fps=low_dim["inspect"].get("fps"),
        ),
        "image": image_summary,
        "embedding": embedding_summary or {"available": False},
        "gold_set": gold_metrics or {"available": False, "requires_human_review": True},
        "coverage": {
            "low_dimensional_episodes": len(records),
            "image_episodes": image_summary.get("episode_count", 0),
            "embedding_episodes": (
                embedding_summary.get("episode_count", 0)
                if embedding_summary else 0
            ),
            "gold_set_review_rows": gold_metrics.get("rows", 0) if gold_metrics else 0,
        },
    }
    return profile, records, motions
