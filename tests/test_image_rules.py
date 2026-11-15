"""Boundary tests for sampled image rules."""

from __future__ import annotations

import numpy as np

from app.image_rules import evaluate_image_metrics, image_metrics, perceptual_hash


def _metric(image: np.ndarray, index: int = 0) -> object:
    return image_metrics(image, index, float(index) / 10.0)


def test_black_frame_is_detected_at_boundary() -> None:
    dark = np.full((32, 32), 7, dtype=np.uint8)
    metric = _metric(dark)

    issues = evaluate_image_metrics([metric])

    assert issues[0]["rule_id"] == "image_black_white"
    assert issues[0]["severity"] == "fatal"


def test_dark_textured_frame_is_not_black() -> None:
    image = np.random.default_rng(0).integers(0, 32, size=(32, 32), dtype=np.uint8)
    metric = _metric(image)

    issues = evaluate_image_metrics([metric])

    assert all(issue["rule_id"] != "image_black_white" for issue in issues)


def test_overexposure_is_warning() -> None:
    image = np.full((32, 32), 255, dtype=np.uint8)
    metric = _metric(image)

    issues = evaluate_image_metrics([metric])

    assert any(issue["rule_id"] == "image_black_white" for issue in issues)
    assert all(issue["severity"] != "fatal" or issue["rule_id"] != "image_exposure" for issue in issues)


def test_relative_blur_marks_lowest_tail_as_warning() -> None:
    rng = np.random.default_rng(1)
    sharp = _metric(rng.integers(0, 255, size=(64, 64), dtype=np.uint8), 0)
    blurry_image = np.ones((64, 64), dtype=np.uint8)
    blurry_image[::8] = 128
    blurry = _metric(blurry_image, 1)

    issues = evaluate_image_metrics([sharp, blurry], blur_percentile=50.0)

    blur_issues = [issue for issue in issues if issue["rule_id"] == "image_blur"]
    assert blur_issues
    assert blur_issues[0]["severity"] == "warning"


def test_perceptual_hash_is_stable_and_sensitive() -> None:
    rng = np.random.default_rng(2)
    image = rng.integers(0, 255, size=(64, 64), dtype=np.uint8).astype(np.uint8)
    changed = image.copy()
    changed[:8] = 255 - changed[:8]

    assert perceptual_hash(image) == perceptual_hash(image.copy())
    assert perceptual_hash(image) != perceptual_hash(changed)
