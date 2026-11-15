"""Tests for handcrafted visual embeddings and coverage analysis."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.embedding import (
    build_descriptor,
    compute_coverage,
    mean_episode_descriptor,
    pca_projection,
)
from tests.test_reader_lerobot import _write_dataset
from tools.embedding_coverage import run as run_embedding_coverage


def test_descriptor_is_deterministic_and_normalized() -> None:
    rng = np.random.default_rng(3)
    frame = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)

    descriptor = build_descriptor(frame)

    assert descriptor.dtype == np.float32
    assert np.isclose(np.linalg.norm(descriptor), 1.0, atol=1e-5)
    assert np.array_equal(descriptor, build_descriptor(frame.copy()))


def test_mean_episode_descriptor_preserves_normalization() -> None:
    rng = np.random.default_rng(4)
    descriptors = [rng.normal(size=32).astype(np.float32) for _ in range(4)]

    result = mean_episode_descriptor(descriptors)

    assert result.shape == (32,)
    assert np.isclose(np.linalg.norm(result), 1.0, atol=1e-5)


def test_coverage_detects_relative_low_density_and_duplicate_candidates() -> None:
    base = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    another = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    embeddings = np.stack([base, base, another])

    rows, pairs, summary = compute_coverage(
        embeddings,
        ["ep0", "ep1", "ep2"],
        k=1,
        near_duplicate_similarity=0.999,
    )

    assert len(rows) == 3
    assert pairs == [{
        "episode_a": "ep0",
        "episode_b": "ep1",
        "cosine_similarity": 1.0,
        "threshold": 0.999,
        "suggested_action": "human_review",
    }]
    assert summary["near_duplicate_candidate_pairs"] == 1
    assert summary["method"] == "handcrafted_visual_v1"
    assert summary["pairwise_similarity"]["count"] == 3
    assert summary["pairwise_similarity"]["max"] == 1.0
    assert summary["closest_pair"]["episode_a"] == "ep0"


def test_pca_projection_returns_two_dimensions() -> None:
    rng = np.random.default_rng(5)
    embeddings = rng.normal(size=(8, 12)).astype(np.float32)

    coordinates, explained = pca_projection(embeddings)

    assert coordinates.shape == (8, 2)
    assert len(explained) == 2


def test_embedding_coverage_runs_on_synthetic_lerobot_dataset(
    tmp_path: Path,
) -> None:
    root = tmp_path / "unitree"
    _write_dataset(root)

    summary = run_embedding_coverage(
        root,
        episode_selection="0-1",
        camera_key="observation.images.wrist",
        output_dir=tmp_path / "embedding",
        frames_per_episode=2,
        k=1,
        near_duplicate_similarity=0.999,
    )

    assert summary["decoded_episodes"] == 2
    assert (tmp_path / "embedding" / "embeddings.npy").is_file()
    assert (tmp_path / "embedding" / "coverage_density.csv").is_file()
    assert (tmp_path / "embedding" / "coverage_pca.svg").is_file()
