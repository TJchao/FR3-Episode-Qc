"""Tests for batch episode QC and Parquet report generation."""

from __future__ import annotations

import csv
from pathlib import Path

import pyarrow.parquet as pq

from app.qc_batch import build_record, classify_error, run_batch


EXAMPLES_ROOT = Path(__file__).resolve().parents[1] / "examples"


def test_batch_qc_writes_csv_and_parquet_reports(tmp_path: Path) -> None:
    csv_output = tmp_path / "reports" / "qc_results.csv"
    parquet_output = tmp_path / "reports" / "qc_results.parquet"

    summary = run_batch(
        input_dir=EXAMPLES_ROOT,
        csv_output=csv_output,
        parquet_output=parquet_output,
        dataset_version="synthetic-v1",
        git_commit="test-commit",
    )

    assert summary["checked_episodes"] == 15
    assert summary["passed"] == 1
    assert summary["failed"] == 14
    assert summary["pass_rate"] == round(1 / 15, 4)
    assert summary["error_type_counts"]["camera_age_p95_exceeded"] == 1
    assert summary["error_type_counts"]["missing_metadata"] == 1
    assert csv_output.is_file()
    assert parquet_output.is_file()

    with csv_output.open(encoding="utf-8-sig", newline="") as csv_file:
        rows = {
            row["episode_id"]: row
            for row in csv.DictReader(csv_file)
        }

    assert rows["ep_valid"]["status"] == "passed"
    assert rows["ep_valid"]["error_types"] == ""
    assert rows["ep_camera_age_p95"]["error_types"] == "camera_age_p95_exceeded"
    assert rows["ep_valid"]["dataset_version"] == "synthetic-v1"
    assert rows["ep_valid"]["git_commit"] == "test-commit"

    table = pq.read_table(parquet_output)
    assert table.num_rows == 15
    assert set(table.column_names) >= {
        "episode_id",
        "status",
        "error_types",
        "camera_age_p95_ms",
        "episode_hash",
        "dataset_version",
    }


def test_batch_qc_counts_each_error_type_once_per_episode(
    tmp_path: Path,
) -> None:
    summary = run_batch(
        input_dir=EXAMPLES_ROOT,
        csv_output=tmp_path / "qc.csv",
        parquet_output=tmp_path / "qc.parquet",
    )

    assert summary["error_type_counts"]["timestamp_backwards"] == 1
    assert summary["error_type_counts"]["robot_time_backwards"] == 1
    assert summary["error_type_counts"]["nan_or_inf"] == 2


def test_build_record_deduplicates_error_types() -> None:
    record = build_record(
        {
            "episode_id": "ep_duplicate",
            "status": "failed",
            "errors": [
                "state contains NaN or Inf",
                "action contains NaN or Inf",
            ],
            "camera_age_p95_ms": 31.917,
            "frame_count": 128,
            "episode_hash": "sha256:test",
        },
        dataset_version="v1",
        git_commit="test",
        created_at="2026-10-21T00:00:00+00:00",
    )

    assert record["error_types"] == "nan_or_inf"
    assert record["error_count"] == 2


def test_classify_unknown_error_is_explicit() -> None:
    assert classify_error("an unexpected failure") == "unknown"
