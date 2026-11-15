"""Tests for LeRobot dataset inspection, loading, and QC reports."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from app.readers.lerobot import (
    discover_layout,
    inspect_layout,
    load_selected_episodes,
    parse_episode_selection,
)
from app.release import evaluate_gate
from app.readers.video import VideoImageSource
from tools.qc_lerobot import run_qc


def _write_dataset(root: Path) -> None:
    meta = root / "meta"
    episodes = meta / "episodes" / "chunk-000"
    data_dir = root / "data" / "chunk-000"
    videos = root / "videos" / "observation.images.wrist" / "chunk-000"
    episodes.mkdir(parents=True)
    data_dir.mkdir(parents=True)
    videos.mkdir(parents=True)

    (meta / "info.json").write_text(
        json.dumps({
            "codebase_version": "v3.0",
            "robot_type": "unitree_g1",
            "total_episodes": 2,
            "total_frames": 8,
            "fps": 4.0,
            "features": {
                "observation.state": {"dtype": "float32", "shape": [3]},
                "action": {"dtype": "float32", "shape": [2]},
            },
        }),
        encoding="utf-8",
    )
    pq.write_table(
        pa.table({
            "episode_index": pa.array([0, 1], type=pa.int64()),
            "length": pa.array([4, 4], type=pa.int64()),
            "tasks": pa.array([["pick"], ["pick"]], type=pa.list_(pa.string())),
            "observation.images.wrist_file_index": pa.array([0, 0], type=pa.int64()),
            "observation.images.wrist_from_timestamp": pa.array([0.0, 1.0], type=pa.float32()),
        }),
        episodes / "file-000.parquet",
    )

    rows = {
        "timestamp": [],
        "frame_index": [],
        "episode_index": [],
        "index": [],
        "task_index": [],
        "observation.state": [],
        "action": [],
    }
    for episode_index in range(2):
        for frame_index in range(4):
            rows["timestamp"].append(episode_index + frame_index * 0.25)
            rows["frame_index"].append(frame_index)
            rows["episode_index"].append(episode_index)
            rows["index"].append(episode_index * 4 + frame_index)
            rows["task_index"].append(0)
            rows["observation.state"].append([frame_index * 0.1, 0.2, 0.3])
            rows["action"].append([0.01, -0.01])
    pq.write_table(
        pa.table({
            "timestamp": pa.array(rows["timestamp"], type=pa.float32()),
            "frame_index": pa.array(rows["frame_index"], type=pa.int64()),
            "episode_index": pa.array(rows["episode_index"], type=pa.int64()),
            "index": pa.array(rows["index"], type=pa.int64()),
            "task_index": pa.array(rows["task_index"], type=pa.int64()),
            "observation.state": pa.array(rows["observation.state"], type=pa.list_(pa.float32())),
            "action": pa.array(rows["action"], type=pa.list_(pa.float32())),
        }),
        data_dir / "file-000.parquet",
    )

    _write_black_video(videos / "file-000.mp4")


def _write_black_video(path: Path) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        4.0,
        (16, 16),
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV cannot write the synthetic MP4 fixture")
    try:
        for _ in range(8):
            writer.write(np.zeros((16, 16, 3), dtype=np.uint8))
    finally:
        writer.release()


def test_parse_episode_selection_ranges_and_values() -> None:
    assert parse_episode_selection("0,2-3", 5) == [0, 2, 3]


def test_inspect_and_load_synthetic_lerobot_dataset(tmp_path: Path) -> None:
    root = tmp_path / "unitree"
    _write_dataset(root)

    layout = discover_layout(root)
    report = inspect_layout(layout)
    assert report["total_episodes"] == 2
    assert report["state_shape"] == [3]
    assert "observation.state" in report["data_columns"]

    episode_rows, episodes, _ = load_selected_episodes(layout, [0, 1])
    assert episode_rows[0]["length"] == 4
    assert episodes[0].arrays["state"].shape == (4, 3)
    assert episodes[1].arrays["action"].shape == (4, 2)


def test_qc_lerobot_outputs_low_dim_motion_and_report(tmp_path: Path) -> None:
    root = tmp_path / "unitree"
    _write_dataset(root)
    output = tmp_path / "reports"

    summary = run_qc(
        root,
        profile_name="unitree_g1",
        episode_selection="0-1",
        camera_key=None,
        output_dir=output,
        dataset_version="synthetic-lerobot-v1",
        git_commit="test",
    )

    assert summary["checked_episodes"] == 2
    assert summary["passed"] == 2
    assert summary["failed"] == 0
    assert (output / "qc_results.parquet").is_file()
    assert (output / "qc_results.csv").is_file()
    assert (output / "motion_profile.json").is_file()
    assert (output / "inspect_report.json").is_file()

    table = pq.read_table(output / "qc_results.parquet")
    assert table.num_rows == 2
    assert table["severity"].to_pylist() == ["none", "none"]


def test_video_source_reads_single_frame_without_loading_video(
    tmp_path: Path,
) -> None:
    root = tmp_path / "unitree"
    _write_dataset(root)
    video = next((root / "videos").rglob("*.mp4"))

    frame = VideoImageSource(video).frame_at(0.0)

    assert frame is not None
    assert frame.shape == (16, 16, 3)


def test_black_frame_is_fatal_and_blocks_release_gate(tmp_path: Path) -> None:
    root = tmp_path / "unitree"
    _write_dataset(root)
    output = tmp_path / "reports"

    summary = run_qc(
        root,
        profile_name="unitree_g1",
        episode_selection="0",
        camera_key="observation.images.wrist",
        output_dir=output,
        dataset_version="synthetic-image-v1",
        git_commit="test",
        sample_fps=4.0,
        save_evidence=True,
    )

    assert summary["checked_episodes"] == 1
    assert summary["failed"] == 1
    assert summary["severity_counts"]["fatal"] == 1
    assert summary["image_samples"] >= 1
    assert "image_freeze" in summary["error_type_counts"]
    assert (output / "image_review.csv").is_file()
    assert list((output / "evidence_frames" / "episode_000000").glob("*.jpg"))

    records = pq.read_table(output / "qc_results.parquet").to_pylist()
    assert "image_black_white" in records[0]["error_types"].split(";")

    gate = evaluate_gate(records, fail_on={"fatal"})
    assert gate["passed"] is False
    assert gate["blocked_by"] == ["fatal"]
