"""Robot-specific QC profiles.

Profiles keep embodiment-specific schema dimensions and thresholds out of the
core checker. FR3 remains the default for the legacy NPZ interface; LeRobot
datasets can use a profile or derive dimensions from ``meta/info.json``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ProfileError(ValueError):
    """Raised when a robot profile is missing or invalid."""


DEFAULT_PROFILE: dict[str, Any] = {
    "robot_type": "fr3",
    "schema_version": "fr3-episode-v1",
    "state_dim": 8,
    "action_dim": 7,
    "fps": 10,
    "camera_age_p95_limit_ms": 50.0,
    "motion": {"enabled": True, "indices": None},
    "image": {"enabled": True, "sample_fps": 1.0},
}


def validate_profile(profile: dict[str, Any]) -> None:
    required = ("robot_type", "schema_version", "state_dim", "action_dim")
    missing = [key for key in required if key not in profile]
    if missing:
        raise ProfileError(f"profile missing fields: {', '.join(missing)}")

    for key in ("state_dim", "action_dim"):
        value = profile.get(key)
        if value is not None and (not isinstance(value, int) or value <= 0):
            raise ProfileError(f"profile {key} must be a positive integer or null")

    camera_limit = profile.get("camera_age_p95_limit_ms", 50.0)
    if camera_limit is not None and (
        not isinstance(camera_limit, (int, float)) or camera_limit <= 0
    ):
        raise ProfileError("camera_age_p95_limit_ms must be positive or null")


def load_profile(name: str, config_dir: str | Path | None = None) -> dict[str, Any]:
    """Load a JSON profile by robot type.

    JSON is intentional here: it avoids adding a YAML dependency to the core
    service while still keeping thresholds outside source code.
    """
    root = (
        Path(config_dir)
        if config_dir is not None
        else Path(__file__).resolve().parents[1] / "config" / "profiles"
    )
    path = root.expanduser() / f"{name}.json"
    if not path.is_file():
        raise ProfileError(f"profile not found: {name}")
    try:
        profile = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProfileError(f"profile {name} cannot be read: {exc}") from exc
    if not isinstance(profile, dict):
        raise ProfileError(f"profile {name} must contain a JSON object")
    validate_profile(profile)
    return profile


def resolve_profile(metadata: dict[str, Any]) -> dict[str, Any]:
    """Resolve a profile while preserving legacy FR3 behavior."""
    profile = dict(DEFAULT_PROFILE)
    robot_type = metadata.get("robot_type", "fr3")
    if robot_type == "fr3":
        return profile

    try:
        loaded = load_profile(str(robot_type))
    except ProfileError:
        # LeRobot datasets may use arbitrary embodiment names. Dimensions are
        # derived from info.json by the reader when this happens.
        loaded = {
            "robot_type": robot_type,
            "schema_version": metadata.get("schema_version", "unknown"),
            "state_dim": None,
            "action_dim": None,
            "fps": metadata.get("fps"),
            "camera_age_p95_limit_ms": None,
            "motion": {"enabled": True, "indices": None},
            "image": {"enabled": True, "sample_fps": 1.0},
        }
    return loaded
