"""Stable error codes and severity classification for episode QC."""

from __future__ import annotations

from typing import Any


SEVERITY_ORDER = {
    "fatal": 3,
    "repairable": 2,
    "warning": 1,
    "none": 0,
}

ERROR_RULES = (
    ("metadata.json is missing", "missing_metadata"),
    ("metadata.json cannot be read", "invalid_metadata"),
    ("metadata.json must contain a JSON object", "invalid_metadata"),
    ("metadata missing fields", "missing_metadata_field"),
    ("metadata episode_id", "invalid_episode_id"),
    ("must be a positive integer", "invalid_metadata_field"),
    ("metadata schema_version", "schema_version_mismatch"),
    ("metadata state_dim", "state_dim_mismatch"),
    ("metadata action_dim", "action_dim_mismatch"),
    ("frames.npz is missing", "missing_frames"),
    ("frames.npz cannot be read", "invalid_frames"),
    ("frames.npz missing fields", "missing_npz_field"),
    ("field length mismatch", "field_length_mismatch"),
    ("robot_time_ns is not strictly increasing", "robot_time_backwards"),
    ("timestamp_s is not strictly increasing", "timestamp_backwards"),
    ("action contains missing frames", "action_missing"),
    ("contains NaN or Inf", "nan_or_inf"),
    ("state shape", "state_shape_error"),
    ("action shape", "action_shape_error"),
    ("camera_age_ms shape", "camera_age_shape_error"),
    ("camera age P95", "camera_age_p95_exceeded"),
    ("metadata frame_count", "frame_count_mismatch"),
    ("rgb_frame_count", "rgb_count_mismatch"),
    ("depth_frame_count", "depth_count_mismatch"),
    ("episode path is outside the allowed root", "path_outside_root"),
    ("episode path does not exist", "path_not_found"),
)


def classify_error(message: str) -> str:
    """Map a human-readable QC error to a stable error code."""
    for text, code in ERROR_RULES:
        if text in message:
            return code
    return "unknown"


def severity_for_error(code: str, report: dict[str, Any]) -> str:
    """Classify one error code without changing the pass/fail status."""
    if code in {
        "action_missing",
        "frame_count_mismatch",
        "rgb_count_mismatch",
        "depth_count_mismatch",
    }:
        return "repairable"

    if code == "camera_age_p95_exceeded":
        p95_ms = report.get("camera_age_p95_ms")
        if isinstance(p95_ms, (int, float)) and p95_ms < 100.0:
            return "warning"
        return "fatal"

    return "fatal"


def highest_severity(severities: list[str]) -> str:
    """Return the most severe level present in a report."""
    if not severities:
        return "none"
    return max(severities, key=lambda severity: SEVERITY_ORDER.get(severity, 3))
