"""Tests for embodiment-agnostic motion profiling."""

from __future__ import annotations

import numpy as np

from app.motion_profile import build_motion_profile


def test_motion_profile_reports_core_statistics() -> None:
    state = np.linspace(0.0, 1.0, num=10).reshape(-1, 1)
    action = np.zeros((10, 2))

    profile = build_motion_profile(state, action, fps=10.0, indices=[0])

    assert profile["frame_count"] == 10
    assert profile["valid_frame_count"] == 10
    assert profile["velocity_p95"] > 0
    assert profile["jerk_p95"] >= 0
    assert profile["active_dimension_ratio"] == 1.0


def test_single_frame_returns_null_statistics() -> None:
    profile = build_motion_profile(np.zeros((1, 3)), np.zeros((1, 2)), fps=10.0)

    assert profile["frame_count"] == 1
    assert profile["velocity_p95"] is None


def test_non_finite_rows_are_excluded_from_statistics() -> None:
    state = np.zeros((4, 2))
    state[2, 0] = np.nan
    profile = build_motion_profile(state, np.zeros((4, 2)), fps=10.0)

    assert profile["valid_frame_count"] == 3
