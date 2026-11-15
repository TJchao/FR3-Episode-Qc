"""Motion statistics for multi-embodiment state streams."""

from __future__ import annotations

from typing import Any

import numpy as np


def build_motion_profile(
    state: np.ndarray,
    action: np.ndarray,
    fps: float,
    indices: list[int] | None = None,
) -> dict[str, Any]:
    """Return descriptive motion statistics without applying pass/fail rules.

    Humanoid state vectors can mix joint angles, finger joints, base pose, and
    IMU values. Until a profile declares semantic indices, these values are used
    only for dataset profiling rather than error classification.
    """
    if fps <= 0:
        raise ValueError("fps must be positive")
    if state.ndim != 2:
        raise ValueError("state must be a 2-D array")

    selected = state[:, indices] if indices else state
    finite = np.isfinite(selected).all(axis=1)
    selected = selected[finite]
    if selected.shape[0] < 2:
        return {
            "frame_count": int(state.shape[0]),
            "valid_frame_count": int(selected.shape[0]),
            "velocity_p50": None,
            "velocity_p95": None,
            "velocity_p99": None,
            "acceleration_p95": None,
            "jerk_p95": None,
            "static_ratio": None,
            "motion_range": None,
            "active_dimension_ratio": None,
        }

    dt = 1.0 / float(fps)
    velocity = np.diff(selected, axis=0) / dt
    speed = np.linalg.norm(velocity, axis=1)
    acceleration = np.diff(velocity, axis=0) / dt
    accel_norm = np.linalg.norm(acceleration, axis=1)
    jerk = np.diff(acceleration, axis=0) / dt
    jerk_norm = np.linalg.norm(jerk, axis=1)

    frame_motion = np.linalg.norm(np.diff(selected, axis=0), axis=1)
    static_threshold = max(float(np.percentile(frame_motion, 10)), 1e-8)
    static_ratio = float(np.mean(frame_motion <= static_threshold))

    def percentile_or_none(values: np.ndarray, percentile: float) -> float | None:
        return float(np.percentile(values, percentile)) if len(values) else None

    return {
        "frame_count": int(state.shape[0]),
        "valid_frame_count": int(selected.shape[0]),
        "velocity_p50": percentile_or_none(speed, 50),
        "velocity_p95": percentile_or_none(speed, 95),
        "velocity_p99": percentile_or_none(speed, 99),
        "acceleration_p95": percentile_or_none(accel_norm, 95),
        "jerk_p95": percentile_or_none(jerk_norm, 95),
        "static_ratio": static_ratio,
        "motion_range": float(np.ptp(selected, axis=0).sum()),
        "active_dimension_ratio": float(np.mean(np.ptp(selected, axis=0) > 1e-8)),
    }
