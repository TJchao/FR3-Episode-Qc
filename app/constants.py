"""Shared constants for synthetic data generation and episode checks."""

from __future__ import annotations

STATE_DIM = 8
ACTION_DIM = 7
CAMERA_AGE_P95_LIMIT_MS = 50.0
SCHEMA_VERSION = "fr3-episode-v1"

REQUIRED_NPZ_FIELDS = (
    "timestamp_s",
    "robot_time_ns",
    "camera_time_ns",
    "camera_age_ms",
    "state",
    "action",
)

REQUIRED_METADATA_FIELDS = (
    "episode_id",
    "frame_count",
    "fps",
    "state_dim",
    "action_dim",
    "rgb_frame_count",
    "depth_frame_count",
    "schema_version",
)

