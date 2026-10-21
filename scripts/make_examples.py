#!/usr/bin/env python3
"""Generate synthetic FR3 episodes for quality-check testing."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.constants import (
    ACTION_DIM,
    CAMERA_AGE_P95_LIMIT_MS,
    SCHEMA_VERSION,
    STATE_DIM,
)


def _base_arrays(frame_count: int) -> dict[str, np.ndarray]:
    """Create deterministic, valid arrays for one episode."""
    timestamps_s = np.arange(frame_count, dtype=np.float32) / 10.0
    robot_time_ns = np.arange(frame_count, dtype=np.int64) * 100_000_000
    camera_age_ms = 24.0 + 8.0 * np.sin(
        np.arange(frame_count, dtype=np.float32) / 7.0
    )
    camera_time_ns = robot_time_ns - (camera_age_ms * 1_000_000).astype(np.int64)

    step = np.arange(frame_count, dtype=np.float32)
    state = np.column_stack(
        [
            np.sin(step / 11.0 + index) * 0.2
            for index in range(STATE_DIM)
        ]
    ).astype(np.float32)
    action = np.column_stack(
        [
            np.cos(step / 13.0 + index) * 0.05
            for index in range(ACTION_DIM)
        ]
    ).astype(np.float32)

    return {
        "timestamp_s": timestamps_s,
        "robot_time_ns": robot_time_ns,
        "camera_time_ns": camera_time_ns,
        "camera_age_ms": camera_age_ms.astype(np.float32),
        "state": state,
        "action": action,
    }


def _base_metadata(
    episode_id: str,
    frame_count: int,
) -> dict[str, str | int]:
    return {
        "episode_id": episode_id,
        "frame_count": frame_count,
        "fps": 10,
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "rgb_frame_count": frame_count,
        "depth_frame_count": frame_count,
        "schema_version": SCHEMA_VERSION,
    }


def _write_episode(
    root: Path,
    episode_id: str,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, str | int] | None,
) -> Path:
    episode_dir = root / episode_id
    episode_dir.mkdir(parents=True, exist_ok=True)

    frames_path = episode_dir / "frames.npz"
    metadata_path = episode_dir / "metadata.json"

    np.savez_compressed(frames_path, **arrays)
    metadata_path.unlink(missing_ok=True)
    if metadata is not None:
        metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    return episode_dir


def generate_examples(root: Path, frame_count: int) -> list[Path]:
    """Generate one valid episode and isolated failure cases."""
    generated: list[Path] = []

    arrays = _base_arrays(frame_count)
    metadata = _base_metadata("ep_valid", frame_count)
    generated.append(_write_episode(root, "ep_valid", arrays, metadata))

    missing_metadata_arrays = _base_arrays(frame_count)
    generated.append(
        _write_episode(
            root,
            "ep_missing_metadata",
            missing_metadata_arrays,
            None,
        )
    )

    missing_field_arrays = _base_arrays(frame_count)
    missing_field_arrays.pop("action")
    generated.append(
        _write_episode(
            root,
            "ep_missing_npz_field",
            missing_field_arrays,
            _base_metadata("ep_missing_npz_field", frame_count),
        )
    )

    missing_metadata_field = _base_metadata(
        "ep_missing_metadata_field",
        frame_count,
    )
    missing_metadata_field.pop("episode_id")
    generated.append(
        _write_episode(
            root,
            "ep_missing_metadata_field",
            _base_arrays(frame_count),
            missing_metadata_field,
        )
    )

    length_mismatch_arrays = _base_arrays(frame_count)
    length_mismatch_arrays["camera_age_ms"] = (
        length_mismatch_arrays["camera_age_ms"][:-1]
    )
    generated.append(
        _write_episode(
            root,
            "ep_length_mismatch",
            length_mismatch_arrays,
            _base_metadata("ep_length_mismatch", frame_count),
        )
    )

    timestamp_arrays = _base_arrays(frame_count)
    timestamp_arrays["robot_time_ns"][frame_count // 2] -= 1
    timestamp_arrays["timestamp_s"][frame_count // 2] -= 0.1
    generated.append(
        _write_episode(
            root,
            "ep_timestamp_backwards",
            timestamp_arrays,
            _base_metadata("ep_timestamp_backwards", frame_count),
        )
    )

    robot_time_arrays = _base_arrays(frame_count)
    midpoint = frame_count // 2
    robot_time_arrays["robot_time_ns"][midpoint] = (
        robot_time_arrays["robot_time_ns"][midpoint - 1] - 1
    )
    generated.append(
        _write_episode(
            root,
            "ep_robot_time_backwards",
            robot_time_arrays,
            _base_metadata("ep_robot_time_backwards", frame_count),
        )
    )

    nan_arrays = _base_arrays(frame_count)
    nan_arrays["state"][frame_count // 3, 0] = np.nan
    generated.append(
        _write_episode(
            root,
            "ep_nan_value",
            nan_arrays,
            _base_metadata("ep_nan_value", frame_count),
        )
    )

    inf_arrays = _base_arrays(frame_count)
    inf_arrays["action"][frame_count // 3, 1] = np.inf
    generated.append(
        _write_episode(
            root,
            "ep_inf_value",
            inf_arrays,
            _base_metadata("ep_inf_value", frame_count),
        )
    )

    state_shape_arrays = _base_arrays(frame_count)
    state_shape_arrays["state"] = state_shape_arrays["state"][:, :-1]
    generated.append(
        _write_episode(
            root,
            "ep_state_shape_error",
            state_shape_arrays,
            _base_metadata("ep_state_shape_error", frame_count),
        )
    )

    action_shape_arrays = _base_arrays(frame_count)
    action_shape_arrays["action"] = np.column_stack(
        [
            action_shape_arrays["action"],
            np.zeros(frame_count, dtype=np.float32),
        ]
    )
    generated.append(
        _write_episode(
            root,
            "ep_action_shape_error",
            action_shape_arrays,
            _base_metadata("ep_action_shape_error", frame_count),
        )
    )

    camera_age_arrays = _base_arrays(frame_count)
    camera_age_arrays["camera_age_ms"][frame_count // 2 :] = (
        CAMERA_AGE_P95_LIMIT_MS + 30.0
    )
    generated.append(
        _write_episode(
            root,
            "ep_camera_age_p95",
            camera_age_arrays,
            _base_metadata("ep_camera_age_p95", frame_count),
        )
    )

    frame_count_metadata = _base_metadata(
        "ep_frame_count_mismatch",
        frame_count,
    )
    frame_count_metadata["frame_count"] = frame_count + 1
    generated.append(
        _write_episode(
            root,
            "ep_frame_count_mismatch",
            _base_arrays(frame_count),
            frame_count_metadata,
        )
    )

    rgb_metadata = _base_metadata("ep_rgb_count_mismatch", frame_count)
    rgb_metadata["rgb_frame_count"] = frame_count - 1
    generated.append(
        _write_episode(
            root,
            "ep_rgb_count_mismatch",
            _base_arrays(frame_count),
            rgb_metadata,
        )
    )

    depth_metadata = _base_metadata("ep_depth_count_mismatch", frame_count)
    depth_metadata["depth_frame_count"] = frame_count + 2
    generated.append(
        _write_episode(
            root,
            "ep_depth_count_mismatch",
            _base_arrays(frame_count),
            depth_metadata,
        )
    )

    return generated


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "examples",
        help="Directory in which synthetic episodes are written.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=128,
        help="Number of frames in each generated episode.",
    )
    args = parser.parse_args()

    if args.frames < 2:
        parser.error("--frames must be at least 2")

    args.output.mkdir(parents=True, exist_ok=True)
    generated = generate_examples(args.output, args.frames)

    for episode_dir in generated:
        print(f"created {episode_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
