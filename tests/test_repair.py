"""Tests for conservative action-frame repair."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from app.qc import _episode_hash
from app.qc import check_episode
from app.repair import RepairError, repair_episode
from scripts.make_examples import _base_arrays, _base_metadata


def _write_episode(
    root: Path,
    episode_id: str,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, object],
) -> Path:
    episode_dir = root / episode_id
    episode_dir.mkdir(parents=True)
    np.savez_compressed(episode_dir / "frames.npz", **arrays)
    (episode_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    return episode_dir


def test_boundary_action_missing_is_cropped_and_rechecked(
    tmp_path: Path,
) -> None:
    arrays = _base_arrays(128)
    arrays["action"][0, :] = np.nan
    episode = _write_episode(
        tmp_path,
        "ep_action_missing_start",
        arrays,
        _base_metadata("ep_action_missing_start", 128),
    )
    source_hash = _episode_hash(episode)
    qc_report = check_episode(episode)
    assert qc_report["status"] == "failed"
    assert qc_report["severity"] == "repairable"
    assert qc_report["error_details"][0]["code"] == "action_missing"

    manifest = repair_episode(episode)

    assert manifest["source_hash"] == source_hash
    assert manifest["result"] == {"status": "passed", "episode_count": 1}
    assert len(manifest["repaired_hashes"]) == 1
    assert manifest["repaired_episodes"][0]["frame_count"] == 127
    assert manifest["repaired_episodes"][0]["qc_report"]["status"] == "passed"
    assert manifest["operations"] == [
        {
            "operation": "crop",
            "side": "start",
            "reason": "action_missing_boundary",
            "dropped_frame_indices": [0],
        }
    ]
    assert manifest["dropped_frames"][0]["frame_index"] == 0
    assert manifest["dropped_frames"][0]["timestamp_delta_to_next_s"] == pytest.approx(0.1)
    assert manifest["split_points"] == []
    assert _episode_hash(episode) == source_hash


def test_middle_action_missing_splits_episode_at_the_hole(
    tmp_path: Path,
) -> None:
    arrays = _base_arrays(128)
    arrays["action"][64, :] = np.nan
    episode = _write_episode(
        tmp_path,
        "ep_action_missing_middle",
        arrays,
        _base_metadata("ep_action_missing_middle", 128),
    )
    source_hash = _episode_hash(episode)

    manifest = repair_episode(episode)

    assert manifest["result"] == {"status": "passed", "episode_count": 2}
    assert [item["frame_count"] for item in manifest["repaired_episodes"]] == [64, 63]
    assert all(
        item["qc_report"]["status"] == "passed"
        for item in manifest["repaired_episodes"]
    )
    assert manifest["operations"][0]["operation"] == "split"
    assert manifest["split_points"][0]["left_last_frame_index"] == 63
    assert manifest["split_points"][0]["right_first_frame_index"] == 65
    assert manifest["split_points"][0]["timestamp_gap_s"] == pytest.approx(0.2)
    assert _episode_hash(episode) == source_hash


def test_repair_preserves_reserved_image_timestamp_fields(
    tmp_path: Path,
) -> None:
    arrays = _base_arrays(20)
    arrays["action"][10, :] = np.nan
    arrays["rgb_timestamps_ns"] = arrays["camera_time_ns"].copy()
    arrays["depth_timestamps_ns"] = arrays["camera_time_ns"].copy()
    episode = _write_episode(
        tmp_path,
        "ep_with_image_timestamps",
        arrays,
        _base_metadata("ep_with_image_timestamps", 20),
    )

    manifest = repair_episode(episode)

    assert manifest["result"]["status"] == "passed"
    for part in manifest["repaired_episodes"]:
        with np.load(Path(part["path"]) / "frames.npz", allow_pickle=False) as npz:
            assert "rgb_timestamps_ns" in npz.files
            assert "depth_timestamps_ns" in npz.files
            assert np.array_equal(npz["rgb_timestamps_ns"], npz["depth_timestamps_ns"])


def test_partial_action_nan_is_not_repaired(tmp_path: Path) -> None:
    arrays = _base_arrays(16)
    arrays["action"][8, 0] = np.nan
    episode = _write_episode(
        tmp_path,
        "ep_partial_action_nan",
        arrays,
        _base_metadata("ep_partial_action_nan", 16),
    )

    with pytest.raises(RepairError, match="only all-NaN rows are repairable"):
        repair_episode(episode)


def test_repair_requires_a_missing_action_row(tmp_path: Path) -> None:
    episode = _write_episode(
        tmp_path,
        "ep_valid_no_repair",
        _base_arrays(16),
        _base_metadata("ep_valid_no_repair", 16),
    )

    with pytest.raises(RepairError, match="no all-NaN action frames"):
        repair_episode(episode)


def test_repair_refuses_to_overwrite_existing_output(tmp_path: Path) -> None:
    arrays = _base_arrays(16)
    arrays["action"][0, :] = np.nan
    episode = _write_episode(
        tmp_path,
        "ep_action_missing",
        arrays,
        _base_metadata("ep_action_missing", 16),
    )
    output = tmp_path / "existing-output"
    output.mkdir()

    with pytest.raises(FileExistsError):
        repair_episode(episode, output_dir=output)
