"""LeRobot v3 dataset inspection and episode loading."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
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


def _tables_by_episode(
    files: list[Path],
    selected: list[int],
) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {index: [] for index in selected}
    mask_values = selected
    for path in files:
        table = pq.read_table(path)
        if "episode_index" not in table.column_names:
            raise LeRobotDatasetError(
                f"LeRobot feature is missing: episode_index in {path.relative_to(files[0].parents[2])}"
            )
        try:
            episode_index = pc.cast(table["episode_index"], pa.int64())
            value_set = pa.array(mask_values, type=pa.int64())
            table = table.filter(pc.is_in(episode_index, value_set=value_set))
        except Exception as exc:  # PyArrow raises several arrow-specific types.
            raise LeRobotDatasetError(f"cannot filter episode_index in {path}: {exc}") from exc
        for row in table.to_pylist():
            episode_index = int(row["episode_index"])
            if episode_index in grouped:
                grouped[episode_index].append(row)
    return grouped


def load_selected_episodes(
    layout: LeRobotLayout,
    selected: list[int],
) -> tuple[dict[int, dict[str, Any]], dict[int, EpisodeData], dict[int, list[dict[str, Any]]]]:
    episode_rows_list = _tables_by_episode(layout.episode_files, selected)
    data_rows = _tables_by_episode(layout.data_files, selected)

    episode_rows: dict[int, dict[str, Any]] = {}
    for index, rows in episode_rows_list.items():
        if not rows:
            raise LeRobotDatasetError(f"LeRobot episode metadata is missing: {index}")
        row = rows[0]
        row["episode_index"] = int(row.get("episode_index", index))
        episode_rows[index] = row

    episodes: dict[int, EpisodeData] = {}
    raw_rows: dict[int, list[dict[str, Any]]] = {}
    features = layout.info.get("features", {})
    state_dim = features.get("observation.state", {}).get("shape", [None])[-1]
    action_dim = features.get("action", {}).get("shape", [None])[-1]
    fps = layout.info.get("fps")

    for index, rows in data_rows.items():
        if not rows:
            episodes[index] = EpisodeData(
                metadata={
                    "episode_id": f"episode_{index:06d}",
                    "robot_type": "unknown",
                    "frame_count": 0,
                    "fps": fps,
                    "state_dim": state_dim,
                    "action_dim": action_dim,
                    "schema_version": "lerobot-v3",
                    "source": "lerobot",
                    "errors": [f"LeRobot feature is missing: data for episode {index}"],
                },
                arrays={},
                image_source=None,
            )
            continue

        rows.sort(key=lambda row: float(row.get("timestamp", 0.0)))
        arrays: dict[str, np.ndarray] = {}
        for field, arrow_name in (
            ("timestamp_s", "timestamp"),
            ("frame_index", "frame_index"),
            ("episode_index", "episode_index"),
            ("index", "index"),
            ("task_index", "task_index"),
            ("state", "observation.state"),
            ("action", "action"),
        ):
            if arrow_name not in rows[0]:
                continue
            arrays[field] = np.asarray([row[arrow_name] for row in rows], dtype=float)

        if "state" in arrays:
            arrays["state"] = arrays["state"].reshape(len(rows), -1)
        if "action" in arrays:
            arrays["action"] = arrays["action"].reshape(len(rows), -1)

        metadata = {
            "episode_id": f"episode_{index:06d}",
            "robot_type": layout.info.get("robot_type", "unknown"),
            "frame_count": len(rows),
            "fps": fps,
            "state_dim": state_dim,
            "action_dim": action_dim,
            "schema_version": "lerobot-v3",
            "source": "lerobot",
        }
        episodes[index] = EpisodeData(
            metadata=metadata,
            arrays=arrays,
            image_source=None,
        )
        raw_rows[index] = rows
    return episode_rows, episodes, raw_rows


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
    from_timestamp = 0.0
    for key, value in episode_row.items():
        normalized = key.replace("/", "_").replace(".", "_")
        if camera_key and camera_key.replace(".", "_") not in normalized:
            continue
        if normalized.endswith("file_index"):
            try:
                file_index = int(value)
            except (TypeError, ValueError):
                pass
        if normalized.endswith("from_timestamp"):
            try:
                from_timestamp = float(value)
            except (TypeError, ValueError):
                pass

    if file_index is not None and 0 <= file_index < len(candidates):
        path = candidates[file_index]
    else:
        path = candidates[0]
    return path
