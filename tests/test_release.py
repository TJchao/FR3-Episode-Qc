"""Tests for the dataset release report and release gate."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq

from app.qc_batch import run_batch
from app.release import (
    evaluate_gate,
    generate_report,
    load_records,
    render_html,
    render_markdown,
    summarize_records,
)

EXAMPLES_ROOT = Path(__file__).resolve().parents[1] / "examples"


def _build_parquet(tmp_path: Path) -> Path:
    parquet_output = tmp_path / "reports" / "qc_results.parquet"
    run_batch(
        input_dir=EXAMPLES_ROOT,
        csv_output=tmp_path / "reports" / "qc_results.csv",
        parquet_output=parquet_output,
        dataset_version="synthetic-v1",
        git_commit="test-commit",
    )
    return parquet_output


def test_load_records_from_parquet(tmp_path: Path) -> None:
    parquet_path = _build_parquet(tmp_path)
    records = load_records(parquet_path)

    assert len(records) == 15
    by_id = {row["episode_id"]: row for row in records}
    assert by_id["ep_valid"]["status"] == "passed"
    assert by_id["ep_valid"]["severity"] == "none"
    assert by_id["ep_valid"]["error_types"] == []
    assert by_id["ep_camera_age_p95"]["severity"] == "warning"
    assert (
        "camera_age_p95_exceeded" in by_id["ep_camera_age_p95"]["error_types"]
    )
    assert by_id["ep_nan_value"]["error_types"] == ["nan_or_inf"]
    assert by_id["ep_valid"]["dataset_version"] == "synthetic-v1"
    assert by_id["ep_valid"]["git_commit"] == "test-commit"


def test_load_records_from_sqlite(tmp_path: Path) -> None:
    from app.db import DEFAULT_DB_PATH, insert_qc_run
    from app.qc import check_episode

    db_path = tmp_path / "history.db"
    report = check_episode(EXAMPLES_ROOT / "ep_nan_value", EXAMPLES_ROOT)
    insert_qc_run(
        report,
        dataset_version="sqlite-v1",
        git_commit="sqlite-commit",
        db_path=db_path,
    )

    records = load_records(db_path)
    assert len(records) == 1
    row = records[0]
    assert row["episode_id"] == report["episode_id"]
    assert row["status"] == "failed"
    assert row["severity"] == "fatal"
    assert "nan_or_inf" in row["error_types"]
    assert row["dataset_version"] == "sqlite-v1"
    assert row["git_commit"] == "sqlite-commit"


def test_sqlite_history_requires_unique_dataset_version(tmp_path: Path) -> None:
    from app.db import insert_qc_run
    from app.qc import check_episode

    bad = check_episode(EXAMPLES_ROOT / "ep_nan_value", EXAMPLES_ROOT)
    good = check_episode(EXAMPLES_ROOT / "ep_valid", EXAMPLES_ROOT)
    db_path = tmp_path / "history.db"
    insert_qc_run(bad, dataset_version="bad-v1", git_commit="c1", db_path=db_path)
    insert_qc_run(good, dataset_version="good-v1", git_commit="c1", db_path=db_path)

    try:
        load_records(db_path)
    except ValueError as exc:
        assert "multiple dataset versions" in str(exc)
        assert "bad-v1" in str(exc)
        assert "good-v1" in str(exc)
    else:
        raise AssertionError("expected mixed-version input to fail")


def test_sqlite_history_filters_version_and_keeps_latest_episode(tmp_path: Path) -> None:
    from app.db import insert_qc_run
    from app.qc import check_episode

    failed = check_episode(EXAMPLES_ROOT / "ep_nan_value", EXAMPLES_ROOT)
    passed = check_episode(EXAMPLES_ROOT / "ep_valid", EXAMPLES_ROOT)
    failed["episode_id"] = "ep_same"
    passed["episode_id"] = "ep_same"
    db_path = tmp_path / "history.db"
    insert_qc_run(failed, dataset_version="v1", git_commit="c1", db_path=db_path)
    insert_qc_run(passed, dataset_version="v1", git_commit="c1", db_path=db_path)

    rows = load_records(db_path, dataset_version="v1")
    assert len(rows) == 1
    assert rows[0]["episode_id"] == passed["episode_id"]
    assert rows[0]["status"] == "passed"


def test_duplicate_episode_ids_are_rejected(tmp_path: Path) -> None:
    rows = [
        {
            "episode_id": "ep_a",
            "status": "passed",
            "severity": "none",
            "error_types": [],
            "error_count": 0,
            "camera_age_p95_ms": 1.0,
            "frame_count": 2,
            "episode_hash": "a",
            "dataset_version": "v1",
            "git_commit": "c1",
            "created_at": "now",
        },
        {
            "episode_id": "ep_a",
            "status": "failed",
            "severity": "fatal",
            "error_types": ["nan_or_inf"],
            "error_count": 1,
            "camera_age_p95_ms": 1.0,
            "frame_count": 2,
            "episode_hash": "b",
            "dataset_version": "v1",
            "git_commit": "c1",
            "created_at": "now",
        },
    ]
    from app.release import _validate_records

    try:
        _validate_records(rows, None)
    except ValueError as exc:
        assert "duplicate episode IDs" in str(exc)
    else:
        raise AssertionError("expected duplicate episode IDs to fail")


def test_summarize_records_core_stats(tmp_path: Path) -> None:
    parquet_path = _build_parquet(tmp_path)
    records = load_records(parquet_path)
    stats = summarize_records(records)

    assert stats["total_episodes"] == 15
    assert stats["passed"] == 1
    assert stats["failed"] == 14
    assert stats["pass_rate"] == round(1 / 15, 4)
    assert stats["severity_counts"] == {
        "fatal": 10,
        "repairable": 3,
        "warning": 1,
        "none": 1,
    }
    assert stats["error_type_counts"]["nan_or_inf"] == 2
    assert stats["camera_age_p95"]["count"] > 0
    assert stats["camera_age_p95"]["p50_ms"] is not None
    assert stats["camera_age_p95"]["p95_ms"] is not None
    assert stats["camera_age_p95"]["p99_ms"] is not None
    assert stats["frame_count"]["count"] == 15
    # Episodes with missing/invalid metadata are recorded with frame_count 0;
    # this is intentional so a report cannot hide broken episodes.
    assert stats["frame_count"]["min"] == 0
    assert stats["frame_count"]["max"] > 0
    assert len(stats["failed_episodes"]) == 14
    assert all(episode["severity"] in {"fatal", "repairable", "warning"}
               for episode in stats["failed_episodes"])


def test_render_markdown_contains_core_sections(tmp_path: Path) -> None:
    parquet_path = _build_parquet(tmp_path)
    records = load_records(parquet_path)
    stats = summarize_records(records)
    markdown = render_markdown(
        stats,
        source=str(parquet_path),
        dataset_version="synthetic-v1",
        git_commit="test-commit",
        generated_at="2026-11-04T00:00:00+00:00",
    )

    assert "# Dataset QC Report" in markdown
    assert "| Total episodes | 15 |" in markdown
    assert "| Passed | 1 |" in markdown
    assert "| Pass rate | 6.67% |" in markdown
    assert "## Severity" in markdown
    assert "| fatal | 10 |" in markdown
    assert "## Error Types (Top)" in markdown
    assert "## Camera Age P95 (ms)" in markdown
    assert "## Frame Count" in markdown
    assert "## Failed Episodes" in markdown
    assert "ep_nan_value" in markdown
    assert "## Repair Manifests" in markdown


def test_render_html_contains_table(tmp_path: Path) -> None:
    parquet_path = _build_parquet(tmp_path)
    records = load_records(parquet_path)
    stats = summarize_records(records)
    page = render_html(
        stats,
        source=str(parquet_path),
        dataset_version="synthetic-v1",
        git_commit="test-commit",
        generated_at="2026-11-04T00:00:00+00:00",
    )

    assert "<!DOCTYPE html>" in page
    assert "<table>" in page
    assert "Dataset QC Report" in page
    assert "Total episodes" in page
    assert "ep_nan_value" in page


def test_gate_passes_when_no_fail_level_present(tmp_path: Path) -> None:
    parquet_path = _build_parquet(tmp_path)
    records = load_records(parquet_path)

    gate = evaluate_gate(
        records,
        fail_on=["fatal"],
        dataset_version="v1",
        git_commit="unknown",
    )
    assert gate["passed"] is False
    assert gate["blocked_by"] == ["fatal"]
    assert gate["dataset_version"] == "v1"
    assert gate["git_commit"] == "unknown"
    assert gate["generated_at"]

    # The source commit takes precedence over the current repository commit.
    probed = evaluate_gate(records, fail_on=["fatal"])
    assert probed["git_commit"] == "test-commit"
    assert probed["source_git_commit"] == "test-commit"

    gate_none = evaluate_gate(
        records,
        fail_on=[],
        dataset_version="synthetic-v1",
    )
    assert gate_none["passed"] is True
    assert gate_none["blocked_by"] == []


def test_gate_blocks_repairable_only_when_requested(tmp_path: Path) -> None:
    parquet_path = _build_parquet(tmp_path)
    records = load_records(parquet_path)

    gate_fatal = evaluate_gate(records, fail_on=["fatal"])
    assert gate_fatal["blocked_by"] == ["fatal"]

    gate_both = evaluate_gate(records, fail_on=["fatal", "repairable"])
    assert gate_both["blocked_by"] == ["fatal", "repairable"]

    gate_repairable_only = evaluate_gate(records, fail_on=["repairable"])
    assert gate_repairable_only["blocked_by"] == ["repairable"]

    gate_warning = evaluate_gate(records, fail_on=["warning"])
    assert gate_warning["blocked_by"] == ["warning"]


def test_gate_priority_order_is_fatal_first(tmp_path: Path) -> None:
    parquet_path = _build_parquet(tmp_path)
    records = load_records(parquet_path)

    gate = evaluate_gate(records, fail_on=["warning", "fatal", "repairable"])
    assert gate["blocked_by"] == ["fatal", "repairable", "warning"]


def test_generate_report_writes_markdown_and_optional_html(tmp_path: Path) -> None:
    parquet_path = _build_parquet(tmp_path)
    records = load_records(parquet_path)
    output = tmp_path / "qc_report.md"

    result = generate_report(
        records,
        source=str(parquet_path),
        output=output,
        include_html=True,
    )

    assert output.is_file()
    assert output.with_suffix(".html").is_file()
    assert result["output_markdown"] == str(output)
    assert result["summary"]["total_episodes"] == 15

    content = output.read_text(encoding="utf-8")
    assert "Dataset QC Report" in content
    assert "synthetic-v1" in content


def test_generate_report_links_repair_manifest(tmp_path: Path) -> None:
    manifest = EXAMPLES_ROOT / "repair_manifest_sample.json"
    parquet_path = _build_parquet(tmp_path)
    records = load_records(parquet_path)
    output = tmp_path / "qc_report.md"

    generate_report(
        records,
        source=str(parquet_path),
        output=output,
        repair_manifest_paths=[str(manifest)],
    )

    content = output.read_text(encoding="utf-8")
    assert "repair_manifest_sample.json" in content
    assert "2 repaired episode(s)" in content


def test_generate_report_missing_manifest_is_disclosed(tmp_path: Path) -> None:
    parquet_path = _build_parquet(tmp_path)
    records = load_records(parquet_path)
    output = tmp_path / "qc_report.md"

    generate_report(
        records,
        source=str(parquet_path),
        output=output,
        repair_manifest_paths=[str(tmp_path / "missing.json")],
    )

    content = output.read_text(encoding="utf-8")
    assert "not found" in content


def test_gate_cli_semantics_exit_code() -> None:
    """The gate JSON contains the fail_on set that produced the decision."""
    records = [
        {
            "episode_id": "ep_a",
            "status": "failed",
            "severity": "fatal",
            "error_types": ["nan_or_inf"],
        }
    ]
    gate = evaluate_gate(records, fail_on=["fatal"])
    gate["fail_on"] = ["fatal"]
    assert gate["passed"] is False
    assert gate["blocked_by"] == ["fatal"]


def test_gate_fails_closed_on_empty_input() -> None:
    gate = evaluate_gate([], fail_on=["fatal"])
    assert gate["passed"] is False
    assert "empty_input" in gate["blocking_reasons"]


def test_gate_detects_lineage_mismatch() -> None:
    records = [
        {
            "episode_id": "ep_a",
            "status": "passed",
            "severity": "none",
            "error_types": [],
            "dataset_version": "v1",
            "git_commit": "record-commit",
        }
    ]
    gate = evaluate_gate(
        records,
        fail_on=["fatal"],
        dataset_version="v2",
        git_commit="override-commit",
    )
    assert gate["passed"] is False
    assert "dataset_version_mismatch" in gate["blocking_reasons"]
    assert "git_commit_mismatch" in gate["blocking_reasons"]
