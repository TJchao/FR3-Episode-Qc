"""Conservative repair for missing action frames in FR3 episodes.

The repair never interpolates and never modifies the source episode. A missing
action frame is represented by an all-NaN action row. Missing frames at an
episode boundary are cropped; a missing frame in the middle splits the episode
so a 200 ms hole is not presented to a policy as two adjacent 100 ms steps.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from app.constants import ACTION_DIM
from app.qc import (
    _episode_hash,
    _load_arrays,
    _load_metadata,
    check_episode,
    resolve_episode_path,
)


REPAIR_SCHEMA_VERSION = "fr3-repair-v1"
MIN_REPAIRED_FRAMES = 2


class RepairError(Exception):
    """Raised when an episode cannot be repaired conservatively."""


def _contiguous_true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return inclusive start/end index pairs for contiguous True values."""
    if mask.ndim != 1 or mask.size == 0:
        return []

    padded = np.concatenate(([False], mask, [False])).astype(np.int8)
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def _json_frame_record(
    arrays: dict[str, np.ndarray],
    frame_index: int,
) -> dict[str, Any]:
    timestamps = arrays["timestamp_s"]
    previous_index = frame_index - 1 if frame_index > 0 else None
    next_index = frame_index + 1 if frame_index + 1 < len(timestamps) else None

    return {
        "frame_index": frame_index,
        "timestamp_s": float(timestamps[frame_index]),
        "previous_frame_index": previous_index,
        "previous_timestamp_s": (
            float(timestamps[previous_index]) if previous_index is not None else None
        ),
        "next_frame_index": next_index,
        "next_timestamp_s": (
            float(timestamps[next_index]) if next_index is not None else None
        ),
        "timestamp_delta_to_previous_s": (
            float(timestamps[frame_index] - timestamps[previous_index])
            if previous_index is not None
            else None
        ),
        "timestamp_delta_to_next_s": (
            float(timestamps[next_index] - timestamps[frame_index])
            if next_index is not None
            else None
        ),
    }


def _build_manifest_operations(
    arrays: dict[str, np.ndarray],
    missing_segments: list[tuple[int, int]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    operations: list[dict[str, Any]] = []
    dropped_frames: list[dict[str, Any]] = []
    split_points: list[dict[str, Any]] = []
    frame_count = len(arrays["action"])

    for start, end in missing_segments:
        indices = list(range(start, end + 1))
        dropped_frames.extend(
            _json_frame_record(arrays, frame_index) for frame_index in indices
        )

        if start == 0:
            operations.append(
                {
                    "operation": "crop",
                    "side": "start",
                    "reason": "action_missing_boundary",
                    "dropped_frame_indices": indices,
                }
            )
        elif end == frame_count - 1:
            operations.append(
                {
                    "operation": "crop",
                    "side": "end",
                    "reason": "action_missing_boundary",
                    "dropped_frame_indices": indices,
                }
            )
        else:
            left_index = start - 1
            right_index = end + 1
            split = {
                "reason": "action_missing_middle",
                "dropped_frame_indices": indices,
                "left_last_frame_index": left_index,
                "right_first_frame_index": right_index,
                "left_last_timestamp_s": float(arrays["timestamp_s"][left_index]),
                "right_first_timestamp_s": float(arrays["timestamp_s"][right_index]),
                "timestamp_gap_s": float(
                    arrays["timestamp_s"][right_index]
                    - arrays["timestamp_s"][left_index]
                ),
            }
            split_points.append(split)
            operations.append({"operation": "split", **split})

    return operations, dropped_frames, split_points


def _write_repaired_part(
    part_dir: Path,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
    indices: np.ndarray,
    part_number: int,
    source_episode_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    part_dir.mkdir(parents=True, exist_ok=False)

    repaired_arrays = {
        name: array[indices] for name, array in arrays.items()
    }
    repaired_metadata = dict(metadata)
    repaired_metadata["episode_id"] = f"{source_episode_id}_part_{part_number:03d}"
    repaired_metadata["frame_count"] = int(len(indices))
    for count_field in ("rgb_frame_count", "depth_frame_count"):
        if count_field in repaired_metadata:
            repaired_metadata[count_field] = int(len(indices))

    np.savez_compressed(part_dir / "frames.npz", **repaired_arrays)
    (part_dir / "metadata.json").write_text(
        json.dumps(repaired_metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    report = check_episode(part_dir, part_dir.parent)
    return repaired_metadata, report


def repair_episode(
    episode_path: str | Path,
    output_dir: str | Path | None = None,
    allowed_root: str | Path | None = None,
) -> dict[str, Any]:
    """Repair all-NaN action rows by cropping boundaries or splitting episodes."""
    source = resolve_episode_path(episode_path, allowed_root)
    source_hash = _episode_hash(source)

    metadata, metadata_error = _load_metadata(source)
    if metadata_error is not None or metadata is None:
        raise RepairError(f"cannot repair episode: {metadata_error}")

    arrays, arrays_error = _load_arrays(source)
    if arrays_error is not None or arrays is None:
        raise RepairError(f"cannot repair episode: {arrays_error}")

    if "action" not in arrays:
        raise RepairError("cannot repair episode: action field is missing")
    action = arrays["action"]
    if action.ndim != 2 or action.shape[1] != ACTION_DIM:
        raise RepairError("cannot repair episode: action shape is invalid")

    missing_rows = np.all(np.isnan(action), axis=1)
    invalid_outside_missing = ~np.isfinite(action) & ~missing_rows[:, None]
    if bool(np.any(invalid_outside_missing)):
        raise RepairError(
            "action has partial NaN or Inf values; only all-NaN rows are repairable"
        )
    if not bool(np.any(missing_rows)):
        raise RepairError("episode has no all-NaN action frames to repair")

    for optional_field in ("rgb_timestamps_ns", "depth_timestamps_ns"):
        if optional_field in arrays and arrays[optional_field].shape[0] != len(action):
            raise RepairError(
                f"cannot repair episode: {optional_field} length does not match action"
            )

    retained_runs = _contiguous_true_runs(~missing_rows)
    if not retained_runs:
        raise RepairError("all action frames are missing; nothing remains to repair")
    if any(end - start + 1 < MIN_REPAIRED_FRAMES for start, end in retained_runs):
        raise RepairError(
            f"repair would create an episode shorter than {MIN_REPAIRED_FRAMES} frames"
        )

    output_root = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else source.parent / f"{source.name}.repaired"
    )
    if output_root.exists():
        raise FileExistsError(f"repair output already exists: {output_root}")
    if output_root == source:
        raise RepairError("repair output must not be the source episode directory")

    missing_segments = _contiguous_true_runs(missing_rows)
    operations, dropped_frames, split_points = _build_manifest_operations(
        arrays,
        missing_segments,
    )

    output_root.mkdir(parents=True, exist_ok=False)
    source_episode_id = str(metadata.get("episode_id") or source.name)
    repaired_episodes: list[dict[str, Any]] = []

    for part_number, (start, end) in enumerate(retained_runs):
        indices = np.arange(start, end + 1, dtype=np.int64)
        part_dir = output_root / f"part_{part_number:03d}"
        repaired_metadata, report = _write_repaired_part(
            part_dir,
            arrays,
            metadata,
            indices,
            part_number,
            source_episode_id,
        )
        repaired_episodes.append(
            {
                "episode_id": repaired_metadata["episode_id"],
                "path": str(part_dir),
                "episode_hash": _episode_hash(part_dir),
                "frame_count": int(len(indices)),
                "qc_report": report,
            }
        )

    all_passed = all(
        episode["qc_report"]["status"] == "passed"
        for episode in repaired_episodes
    )
    manifest: dict[str, Any] = {
        "schema_version": REPAIR_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_episode_id": source_episode_id,
        "source_path": str(source),
        "source_hash": source_hash,
        "repair_policy": "action_missing_crop_or_split_v1",
        "repair_rules": [
            "action_missing_boundary_crop",
            "action_missing_middle_split",
        ],
        "constraints": {
            "interpolation_allowed": False,
            "source_modified": False,
            "minimum_repaired_frames": MIN_REPAIRED_FRAMES,
            "image_repair_implemented": False,
        },
        "operations": operations,
        "dropped_frames": dropped_frames,
        "split_points": split_points,
        "repaired_hashes": [
            episode["episode_hash"] for episode in repaired_episodes
        ],
        "repaired_episodes": repaired_episodes,
        "result": {
            "status": "passed" if all_passed else "failed",
            "episode_count": len(repaired_episodes),
        },
    }

    manifest_path = output_root / "repair_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode_path", help="Path to one episode directory")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Repair output directory; defaults to <episode>.repaired",
    )
    parser.add_argument(
        "--allowed-root",
        default=None,
        help="Restrict episode paths to this root directory",
    )
    return parser


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()

    try:
        manifest = repair_episode(
            args.episode_path,
            output_dir=args.output,
            allowed_root=args.allowed_root,
        )
    except (RepairError, FileExistsError) as exc:
        parser.error(str(exc))
        return 2

    print(
        json.dumps(
            {
                "manifest_path": str(
                    Path(args.output or f"{args.episode_path}.repaired").resolve()
                    / "repair_manifest.json"
                ),
                "result": manifest["result"],
                "repaired_episodes": [
                    {
                        "episode_id": episode["episode_id"],
                        "path": episode["path"],
                        "episode_hash": episode["episode_hash"],
                    }
                    for episode in manifest["repaired_episodes"]
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if manifest["result"]["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
