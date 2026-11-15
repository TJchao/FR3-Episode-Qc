"""Tests for image-rule threshold calibration."""

from __future__ import annotations

from app.image_rule_calibration import calibrate


def test_calibration_compares_blur_and_static_thresholds() -> None:
    metric_rows = [
        {
            "episode_id": "ep0",
            "frame_index": frame_index,
            "laplacian_variance": 100.0 - frame_index,
            "phash": f"{frame_index:064b}",
        }
        for frame_index in range(4)
    ]
    gold_rows = [
        {
            "rule_id": "image_blur",
            "episode_id": "ep0",
            "frame_index": 3,
            "is_true_positive": "true",
            "human_label": "",
        },
        {
            "rule_id": "image_static_candidate",
            "episode_id": "ep0",
            "frame_index": 2,
            "is_true_positive": "true",
            "human_label": "",
        },
    ]

    summary = calibrate(
        gold_rows,
        metric_rows,
        blur_percentiles=(2.0, 50.0),
        static_hamming_thresholds=(2, 4),
    )

    assert summary["reviewed_rows"] == 2
    assert summary["blur_percentile"]["P50"]["tp"] == 1
    assert summary["blur_percentile"]["P50"]["sample_precision"] == 1.0
    assert summary["static_hamming"]["H2"]["tp"] == 1
    assert summary["static_hamming"]["H2"]["fp"] == 0
