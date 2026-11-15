"""Motion statistics for multi-embodiment state streams."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def infer_motion_groups(state_names: Sequence[str]) -> dict[str, list[int]]:
    """Infer coarse arm/hand groups from common Unitree joint names."""
    groups: dict[str, list[int]] = {
        "left_arm": [],
        "right_arm": [],
        "left_hand": [],
        "right_hand": [],
    }
    for index, raw_name in enumerate(state_names):
        name = raw_name.replace("_", "")
        if "LeftHand" in name:
            groups["left_hand"].append(index)
        elif "RightHand" in name:
            groups["right_hand"].append(index)
        elif name.startswith("kLeft"):
            groups["left_arm"].append(index)
        elif name.startswith("kRight"):
            groups["right_arm"].append(index)
    return {name: indices for name, indices in groups.items() if indices}


def _profile_one(
    values: np.ndarray,
    frame_count: int,
    fps: float,
) -> dict[str, Any]:
    finite = np.isfinite(values).all(axis=1)
    selected = values[finite]
    if selected.shape[0] < 2:
        return {
            "frame_count": frame_count,
            "valid_frame_count": int(selected.shape[0]),
            "velocity_p50": None,
            "velocity_p95": None,
            "velocity_p99": None,
            "acceleration_p95": None,
            "jerk_p95": None,
            "relative_static_tail_ratio": None,
            "motion_range": None,
            "active_dimension_ratio": None,
        }

    dt = 1.0 / float(fps)
    velocity = np.diff(selected, axis=0) / dt
    speed = np.linalg.norm(velocity, axis=1)
    acceleration = np.diff(velocity, axis=0) / dt
    acceleration_speed = np.linalg.norm(acceleration, axis=1)
    jerk = np.diff(acceleration, axis=0) / dt
    jerk_speed = np.linalg.norm(jerk, axis=1)
    frame_motion = np.linalg.norm(np.diff(selected, axis=0), axis=1)
    static_threshold = max(float(np.percentile(frame_motion, 10)), 1e-8)

    def percentile_or_none(items: np.ndarray, percentile: float) -> float | None:
        return float(np.percentile(items, percentile)) if len(items) else None

    return {
        "frame_count": frame_count,
        "valid_frame_count": int(selected.shape[0]),
        "velocity_p50": percentile_or_none(speed, 50),
        "velocity_p95": percentile_or_none(speed, 95),
        "velocity_p99": percentile_or_none(speed, 99),
        "acceleration_p95": percentile_or_none(acceleration_speed, 95),
        "jerk_p95": percentile_or_none(jerk_speed, 95),
        "relative_static_tail_ratio": float(
            np.mean(frame_motion <= static_threshold)
        ),
        "motion_range": float(np.ptp(selected, axis=0).sum()),
        "active_dimension_ratio": float(
            np.mean(np.ptp(selected, axis=0) > 1e-8)
        ),
    }


def build_motion_profile(
    state: np.ndarray,
    action: np.ndarray,
    fps: float,
    indices: list[int] | None = None,
    *,
    state_names: Sequence[str] | None = None,
    groups: Mapping[str, list[int]] | None = None,
    state_unit: str | None = None,
) -> dict[str, Any]:
    """Return descriptive motion statistics without applying pass/fail rules.

    Humanoid state vectors can mix arm joints, finger joints, base pose, and
    IMU values with different physical scales. Until a profile declares exact
    units, values are reported in ``state_unit/s`` rather than guessing radians.
    """
    if fps <= 0:
        raise ValueError("fps must be positive")
    if state.ndim != 2:
        raise ValueError("state must be a 2-D array")

    if indices:
        selected = state[:, indices]
        selected_names = [state_names[i] for i in indices] if state_names else None
    else:
        selected = state
        selected_names = list(state_names) if state_names else None

    resolved_unit = state_unit or "unknown"
    profile = _profile_one(selected, state.shape[0], fps)
    profile.update({
        "state_unit": resolved_unit,
        "velocity_unit": f"{resolved_unit}/s",
        "acceleration_unit": f"{resolved_unit}/s^2",
        "jerk_unit": f"{resolved_unit}/s^3",
        "aggregation": "selected_dimensions" if indices else "all_dimensions",
        "group_definitions": {},
    })

    if groups is None and selected_names is not None:
        groups = infer_motion_groups(selected_names)
    if groups:
        group_profiles: dict[str, dict[str, Any]] = {}
        definitions: dict[str, dict[str, Any]] = {}
        for name, group_indices in groups.items():
            if not group_indices or max(group_indices) >= selected.shape[1]:
                continue
            group_profiles[name] = _profile_one(
                selected[:, group_indices],
                selected.shape[0],
                fps,
            )
            definitions[name] = {
                "indices": group_indices,
                "names": (
                    [selected_names[index] for index in group_indices]
                    if selected_names
                    else None
                ),
            }
        profile["groups"] = group_profiles
        profile["group_definitions"] = definitions

    return profile
