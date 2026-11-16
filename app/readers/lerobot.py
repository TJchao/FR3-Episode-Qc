"""LeRobot dataset inspection and episode loading."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from app.readers.base import EpisodeData


class LeRobotDatasetError(RuntimeError):
    """Raised when a directory is not a readable LeRobot dataset."""


@dataclass(frozen=True)
class LeRobotLayout:
    root: Path
    info: dict[str, Any]
    episode_files: list[Path]
    data_files: list[Path]
    video_files: list[Path]


def _require_file(path: Path, message: str) -> None:
    if not path.is_file():
        raise LeRobotDatasetError(message)


def discover_layout(dataset_dir: str | Path) -> LeRobotLayout:
    root = Path(dataset_dir).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise LeRobotDatasetError(f"dataset directory is not a directory: {root}")

    info_path = root / "meta" / "info.json"
    _require_file(info_path, f"LeRobot dataset is missing meta/info.json at {info_path}")
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LeRobotDatasetError(f"meta/info.json cannot be read: {exc}") from exc
    if not isinstance(info, dict):
        raise LeRobotDatasetError("meta/info.json must contain a JSON object")

    episode_files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    data_files = sorted((root / "data").rglob("*.parquet"))
    video_files = sorted(
        path for path in (root / "videos").rglob("*") if path.suffix in {".mp4", ".webm"}
    )
    if not episode_files:
        raise LeRobotDatasetError("LeRobot dataset is missing episode parquet files")
    if not data_files:
        raise LeRobotDatasetError("LeRobot dataset is missing data parquet files")

    return LeRobotLayout(
        root=root,
        info=info,
        episode_files=episode_files,
        data_files=data_files,
        video_files=video_files,
    )


def inspect_layout(layout: LeRobotLayout) -> dict[str, Any]:
    features = layout.info.get("features", {})
    state_shape = features.get("observation.state", {}).get("shape")
    action_shape = features.get("action", {}).get("shape")

    episode_columns: list[str] = []
    data_columns: list[str] = []
    for files, target in (
        (layout.episode_files, episode_columns),
        (layout.data_files, data_columns),
    ):
        if files:
            target.extend(pq.ParquetFile(files[0]).schema_arrow.names)

    return {
        "dataset_dir": str(layout.root),
        "robot_type": layout.info.get("robot_type"),
        "codebase_version": layout.info.get("codebase_version"),
        "total_episodes": layout.info.get("total_episodes"),
        "total_frames": layout.info.get("total_frames"),
        "fps": layout.info.get("fps"),
        "state_shape": state_shape,
        "action_shape": action_shape,
        "episode_files": [str(path.relative_to(layout.root)) for path in layout.episode_files],
        "data_files": [str(path.relative_to(layout.root)) for path in layout.data_files],
        "episode_columns": episode_columns,
        "data_columns": data_columns,
        "video_files": [str(path.relative_to(layout.root)) for path in layout.video_files],
        "video_keys": sorted(
            {path.parent.parent.relative_to(layout.root / "videos").as_posix()
             for path in layout.video_files}
        ),
    }


def parse_episode_selection(selection: str, available_count: int) -> list[int]:
    selected: set[int] = set()
    for item in selection.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            try:
                start, end = int(start_text), int(end_text)
            except ValueError as exc:
                raise ValueError(f"invalid episode range: {item}") from exc
            if start < 0 or end < start:
                raise ValueError(f"invalid episode range: {item}")
            selected.update(range(start, end + 1))
        else:
            try:
                selected.add(int(item))
            except ValueError as exc:
                raise ValueError(f"invalid episode index: {item}") from exc
    selected = {index for index in selected if index < available_count}
    if not selected:
        raise ValueError("episode selection is empty")
    return sorted(selected)


def _feature_names(feature: dict[str, Any]) -> list[str]:
    names = feature.get("names")
    if not names:
        return []
    first = names[0] if isinstance(names, list) else names
    if isinstance(first, list):
        return [str(name) for name in first]
    return [str(name) for name in names]


def iter_selected_episode_arrays(
    layout: LeRobotLayout,
    selected: list[int],
) -> tuple[dict[int, dict[str, Any]], dict[int, EpisodeData]]:
    """Stream selected episodes without converting every Arrow row to Python."""
    episode_rows: dict[int, dict[str, Any]] = {}
    for episode_path in layout.episode_files:
        episode_table = pq.read_table(episode_path)
        for row in episode_table.to_pylist():
            index = int(row["episode_index"])
            if index in selected:
                episode_rows[index] = row
    missing_metadata = [
        index for index in selected if index not in episode_rows
    ]
    if missing_metadata:
        preview = ", ".join(str(index) for index in missing_metadata[:10])
        raise LeRobotDatasetError(
            f"LeRobot episode metadata is missing: {preview}"
        )

    features = layout.info.get("features", {})
    state_dim = int(features.get("observation.state", {}).get("shape", [0])[-1] or 0)
    action_dim = int(features.get("action", {}).get("shape", [0])[-1] or 0)
    schema_version = str(layout.info.get("codebase_version") or "unknown")
    state_names = _feature_names(features.get("observation.state", {}))
    action_names = _feature_names(features.get("action", {}))
    episodes: dict[int, EpisodeData] = {}
    found_episodes: set[int] = set()

    for data_path in layout.data_files:
        table = pq.read_table(
            data_path,
            columns=[
                "episode_index",
                "timestamp",
                "frame_index",
                "index",
                "task_index",
                "observation.state",
                "action",
            ],
        )
        episode_values = table["episode_index"].to_numpy(zero_copy_only=False)
        positions_by_episode: dict[int, list[int]] = {
            index: [] for index in selected if index not in found_episodes
        }
        for position, raw_index in enumerate(episode_values):
            index = int(raw_index)
            if index in positions_by_episode:
                positions_by_episode[index].append(position)

        timestamps_all = np.asarray(
            table["timestamp"].to_numpy(zero_copy_only=False), dtype=np.float64
        )
        frame_indices_all = np.asarray(
            table["frame_index"].to_numpy(zero_copy_only=False), dtype=np.int64
        )
        global_indices_all = np.asarray(
            table["index"].to_numpy(zero_copy_only=False), dtype=np.int64
        )
        task_indices_all = np.asarray(
            table["task_index"].to_numpy(zero_copy_only=False), dtype=np.int64
        )
        state_array = table["observation.state"].combine_chunks()
        action_array = table["action"].combine_chunks()
        state_values = np.asarray(
            state_array.values.to_numpy(zero_copy_only=False), dtype=np.float32
        )
        action_values = np.asarray(
            action_array.values.to_numpy(zero_copy_only=False), dtype=np.float32
        )
        state_offsets = state_array.offsets.to_numpy(zero_copy_only=False)
        action_offsets = action_array.offsets.to_numpy(zero_copy_only=False)

        for index, positions in sorted(positions_by_episode.items()):
            if not positions:
                continue
            positions_array = np.asarray(positions, dtype=np.int64)
            found_episodes.add(index)
            timestamps = timestamps_all[positions_array]
            order = np.argsort(timestamps, kind="stable")
            positions_array = positions_array[order]
            timestamps = timestamps[order]

            state_parts = [
                state_values[state_offsets[pos]:state_offsets[pos + 1]]
                for pos in positions_array
            ]
            action_parts = [
                action_values[action_offsets[pos]:action_offsets[pos + 1]]
                for pos in positions_array
            ]
            state = (
                np.concatenate(state_parts).reshape(len(positions_array), state_dim)
                if state_parts
                else np.empty((0, state_dim), dtype=np.float32)
            )
            action = (
                np.concatenate(action_parts).reshape(len(positions_array), action_dim)
                if action_parts
                else np.empty((0, action_dim), dtype=np.float32)
            )
            episodes[index] = EpisodeData(
                metadata={
                    "episode_id": f"episode_{index:06d}",
                    "robot_type": layout.info.get("robot_type", "unknown"),
                    "frame_count": len(positions_array),
                    "fps": layout.info.get("fps"),
                    "state_dim": state_dim or None,
                    "action_dim": action_dim or None,
                    "schema_version": schema_version,
                    "expected_frame_count": episode_rows[index].get("length"),
                    "source": "lerobot",
                    "state_names": state_names,
                    "action_names": action_names,
                },
                arrays={
                    "timestamp_s": timestamps,
                    "frame_index": frame_indices_all[positions_array],
                    "episode_index": np.full(len(positions_array), index, dtype=np.int64),
                    "index": global_indices_all[positions_array],
                    "task_index": task_indices_all[positions_array],
                    "state": state,
                    "action": action,
                },
            )

    for index in selected:
        if index in found_episodes:
            continue
        episodes[index] = EpisodeData(
            metadata={
                "episode_id": f"episode_{index:06d}",
                "robot_type": layout.info.get("robot_type", "unknown"),
                "frame_count": 0,
                "fps": layout.info.get("fps"),
                "state_dim": state_dim or None,
                "action_dim": action_dim or None,
                "schema_version": schema_version,
                "expected_frame_count": episode_rows.get(index, {}).get("length"),
                "state_names": state_names,
                "action_names": action_names,
                "source": "lerobot",
                "errors": [f"LeRobot feature is missing: data for episode {index}"],
            },
            arrays={},
            image_source=None,
        )

    return episode_rows, episodes


def video_path_for_episode(
    layout: LeRobotLayout,
    episode_row: dict[str, Any],
    camera_key: str | None,
) -> Path | None:
    if not layout.video_files:
        return None
    candidates = layout.video_files
    if camera_key:
        normalized = camera_key.replace(".", "_")
        candidates = [
            path for path in candidates
            if camera_key in path.as_posix() or normalized in path.as_posix()
        ]
    if not candidates:
        return None

    file_index = None
    for key, value in episode_row.items():
        normalized = key.replace("/", "_").replace(".", "_")
        if camera_key and camera_key.replace(".", "_") not in normalized:
            continue
        if normalized.endswith("file_index"):
            try:
                file_index = int(value)
            except (TypeError, ValueError):
                pass
    if file_index is not None and 0 <= file_index < len(candidates):
        path = candidates[file_index]
    else:
        path = candidates[0]
    return path


def video_offset_for_episode(
    episode_row: dict[str, Any],
    camera_key: str | None,
) -> float:
    """Return the video-local start timestamp for one episode/camera pair."""
    if not camera_key:
        return 0.0
    normalized_camera = camera_key.replace(".", "_")
    for key, value in episode_row.items():
        normalized = key.replace("/", "_").replace(".", "_")
        if (
            normalized.endswith("from_timestamp")
            and normalized_camera in normalized
        ):
            try:
                return float(value)
            except (TypeError, ValueError):
                return 0.0
    return 0.0
