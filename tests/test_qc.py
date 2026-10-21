"""Tests for the episode quality-check rules."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.constants import ACTION_DIM, STATE_DIM
from app.qc import check_episode


EXAMPLES_ROOT = Path(__file__).resolve().parents[1] / "examples"


EXPECTED_ERRORS = {
    "ep_missing_metadata": "metadata.json is missing",
    "ep_missing_npz_field": "frames.npz missing fields: action",
    "ep_missing_metadata_field": "metadata missing fields: episode_id",
    "ep_length_mismatch": "field length mismatch",
    "ep_timestamp_backwards": "timestamp_s is not strictly increasing",
    "ep_robot_time_backwards": "robot_time_ns is not strictly increasing",
    "ep_nan_value": "state contains NaN or Inf",
    "ep_inf_value": "action contains NaN or Inf",
    "ep_state_shape_error": f"state shape must be (128, {STATE_DIM})",
    "ep_action_shape_error": f"action shape must be (128, {ACTION_DIM})",
    "ep_camera_age_p95": "camera age P95 must be below 50.0ms",
    "ep_frame_count_mismatch": "metadata frame_count is 129, actual frames are 128",
    "ep_rgb_count_mismatch": "rgb_frame_count is 127, actual frames are 128",
    "ep_depth_count_mismatch": "depth_frame_count is 130, actual frames are 128",
}


def test_valid_episode_passes_all_checks() -> None:
    report = check_episode(EXAMPLES_ROOT / "ep_valid")

    assert report["status"] == "passed"
    assert report["errors"] == []
    assert report["warnings"] == []
    assert report["episode_id"] == "ep_valid"
    assert report["frame_count"] == 128
    assert report["camera_age_p95_ms"] is not None
    assert report["camera_age_p95_ms"] < 50.0
    assert report["episode_hash"].startswith("sha256:")


@pytest.mark.parametrize(
    ("episode_name", "expected_error"),
    sorted(EXPECTED_ERRORS.items()),
)
def test_invalid_episode_fails_for_expected_reason(
    episode_name: str,
    expected_error: str,
) -> None:
    report = check_episode(EXAMPLES_ROOT / episode_name)

    assert report["status"] == "failed"
    assert any(expected_error in error for error in report["errors"])


def test_allowed_root_rejects_outside_path(tmp_path: Path) -> None:
    outside_episode = tmp_path / "outside"
    outside_episode.mkdir()

    report = check_episode(outside_episode, EXAMPLES_ROOT)

    assert report["status"] == "failed"
    assert "episode path is outside the allowed root" in report["errors"]
