"""Tests for the reusable gold-set HTML reviewer."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.gold_set_review import (
    build_review_items,
    write_review_html,
)


def _write_frame(path: Path, value: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = np.full((48, 64, 3), value, dtype=np.uint8)
    frame[value : value + 8] = 255 - value
    encoded, buffer = cv2.imencode(".jpg", frame)
    if not encoded:
        raise RuntimeError(f"cannot create test frame: {path}")
    path.write_bytes(buffer.tobytes())
    return path


def _metric_rows(root: Path) -> list[dict]:
    rows = []
    for frame_index, value in ((0, 30), (30, 90), (60, 150)):
        path = _write_frame(
            root / "evidence_frames" / "ep0" / f"frame_{frame_index:08d}.jpg",
            value,
        )
        rows.append({
            "episode_id": "ep0",
            "frame_index": frame_index,
            "timestamp_s": frame_index / 30.0,
            "brightness_mean": float(value),
            "brightness_std": 10.0,
            "highlight_ratio": 0.01,
            "shadow_ratio": 0.02,
            "laplacian_variance": 100.0 + frame_index,
            "phash": f"{frame_index:064b}",
            "evidence_path": str(path),
        })
    return rows


def _gold_rows() -> list[dict]:
    return [{
        "rule_id": "image_static_candidate",
        "episode_id": "ep0",
        "frame_index": "30",
        "timestamp_s": "1.0",
        "evidence_path": "unused",
        "auto_hit": "True",
        "human_label": "",
        "is_true_positive": "",
        "note": "",
    }]


def test_review_items_attach_previous_and_current_frames(
    tmp_path: Path,
) -> None:
    metrics = _metric_rows(tmp_path / "source")
    hits = [{
        "rule_id": "image_static_candidate",
        "episode_id": "ep0",
        "frame_index": "30",
        "timestamp_s": "1.0",
        "value": "2",
        "threshold": "4",
        "message": "sampled adjacent frames are visually static",
    }]

    items, summary = build_review_items(
        _gold_rows(),
        metrics,
        hits,
        output_dir=tmp_path / "review",
        copy_evidence=True,
    )

    assert summary["input_rows"] == 1
    assert summary["review_items"] == 1
    assert summary["evidence_references"] == 2
    assert summary["unique_evidence_files"] == 2
    assert items[0]["previous_frame_index"] == 0
    assert items[0]["current_uri"] == "evidence/ep0/current_00000030.jpg"
    assert items[0]["previous_uri"] == "evidence/ep0/previous_00000000.jpg"
    assert items[0]["sampled_hamming_distance"] == 4
    assert items[0]["auto_rule_value"] == 2.0
    assert (tmp_path / "review" / "evidence" / "ep0" / "current_00000030.jpg").is_file()


def test_review_html_embeds_payload_and_guidance(
    tmp_path: Path,
) -> None:
    metrics = _metric_rows(tmp_path / "source")
    items, summary = build_review_items(
        _gold_rows(),
        metrics,
        [],
        output_dir=tmp_path / "review",
        copy_evidence=False,
    )
    html_path = tmp_path / "review" / "gold_set_review.html"
    write_review_html(items, output_path=html_path, summary=summary)
    text = html_path.read_text(encoding="utf-8")

    assert "gold-set-review:" in text
    assert "Static candidate：对比前一帧和当前帧" in text
    assert "file:///" in text
    assert "exportCsv" in text
    assert "localStorage" in text
    assert "ruleFilter.value=previousRule" in text
    assert "1 / ${r.review_guidance.true}" in text
    assert "2 / ${r.review_guidance.false}" in text


def test_review_items_without_copy_use_original_paths(
    tmp_path: Path,
) -> None:
    metrics = _metric_rows(tmp_path / "source")
    items, summary = build_review_items(
        _gold_rows(),
        metrics,
        [],
        output_dir=tmp_path / "review",
        copy_evidence=False,
    )

    assert summary["evidence_references"] == 2
    assert summary["unique_evidence_files"] == 2
    assert items[0]["current_uri"].startswith("file:///")
    assert not (tmp_path / "review" / "evidence").exists()
