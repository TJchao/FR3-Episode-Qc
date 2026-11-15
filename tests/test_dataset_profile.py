"""Tests for batch-level LeRobot dataset profiling."""

from __future__ import annotations

from pathlib import Path

from app.dataset_profile import build_dataset_profile
from tests.test_reader_lerobot import _write_dataset


def test_dataset_profile_summarizes_all_selected_episodes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "unitree"
    _write_dataset(root)

    profile, records, motions = build_dataset_profile(
        root,
        profile_name="unitree_g1",
        episode_selection="0-1",
    )

    assert len(records) == 2
    assert all(record["status"] == "passed" for record in records)
    assert profile["dataset"]["selected_episodes"] == 2
    assert profile["dataset"]["robot_type"] == "unitree_g1"
    assert profile["quality"]["frame_count"]["count"] == 2
    assert set(motions) == {"episode_000000", "episode_000001"}
    assert profile["motion"]["all_dimensions"]["velocity_p95"]["count"] == 2


def test_dataset_profile_reports_missing_image_and_gold_set(
    tmp_path: Path,
) -> None:
    root = tmp_path / "unitree"
    _write_dataset(root)

    profile, _, _ = build_dataset_profile(
        root,
        profile_name="unitree_g1",
        episode_selection="0",
    )

    assert profile["image"] == {"available": False}
    assert profile["embedding"] == {"available": False}
    assert profile["gold_set"]["requires_human_review"] is True
