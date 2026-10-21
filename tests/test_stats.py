"""Tests for DuckDB summaries over QC Parquet reports."""

from __future__ import annotations

from pathlib import Path

from app.qc_batch import run_batch
from app.stats import summarize_qc_results


EXAMPLES_ROOT = Path(__file__).resolve().parents[1] / "examples"


def test_summarize_qc_results(tmp_path: Path) -> None:
    parquet_output = tmp_path / "qc_results.parquet"
    run_batch(
        input_dir=EXAMPLES_ROOT,
        csv_output=tmp_path / "qc_results.csv",
        parquet_output=parquet_output,
        dataset_version="synthetic-v1",
        git_commit="test-commit",
    )

    summary = summarize_qc_results(parquet_output)

    assert summary["total_episodes"] == 15
    assert summary["status_counts"] == {"failed": 14, "passed": 1}
    assert summary["error_type_counts"]["camera_age_p95_exceeded"] == 1
    assert summary["dataset_version_counts"] == {"synthetic-v1": 15}

    camera_by_status = {
        item["status"]: item for item in summary["camera_age_by_status"]
    }
    assert camera_by_status["failed"]["episode_count"] == 14
    assert camera_by_status["failed"]["measured_episode_count"] == 13
    assert camera_by_status["passed"]["episode_count"] == 1
    assert camera_by_status["passed"]["measured_episode_count"] == 1
    assert camera_by_status["failed"]["avg_camera_age_p95_ms"] > 0
