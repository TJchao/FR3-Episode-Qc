"""Tests for robot-specific QC profiles."""

from __future__ import annotations

import json

from app.profile import DEFAULT_PROFILE, ProfileError, load_profile, validate_profile


def test_loads_fr3_profile_from_config() -> None:
    profile = load_profile("fr3")

    assert profile["state_dim"] == 8
    assert profile["action_dim"] == 7
    assert profile["camera_age_p95_limit_ms"] == 50.0


def test_missing_profile_raises_profile_error(tmp_path) -> None:
    try:
        load_profile("unknown_robot", config_dir=tmp_path)
    except ProfileError as exc:
        assert "profile not found" in str(exc)
    else:
        raise AssertionError("expected ProfileError")


def test_profile_allows_null_dimensions_and_camera_age(tmp_path) -> None:
    path = tmp_path / "robot.json"
    path.write_text(
        json.dumps({
            "robot_type": "robot",
            "schema_version": "v2",
            "state_dim": None,
            "action_dim": None,
            "camera_age_p95_limit_ms": None,
        }),
        encoding="utf-8",
    )

    profile = load_profile("robot", config_dir=tmp_path)

    assert profile["state_dim"] is None
    assert profile["camera_age_p95_limit_ms"] is None


def test_default_profile_remains_fr3() -> None:
    validate_profile(DEFAULT_PROFILE)
    assert DEFAULT_PROFILE["robot_type"] == "fr3"
