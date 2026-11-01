"""Batch episode quality checks and report generation."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from app.qc import check_episode
from app.errors import classify_error


CSV_FIELDS = (
    "episode_id",
    "status",
    "severity",
    "error_types",
    "error_count",
    "camera_age_p95_ms",
    "frame_count",
    "episode_hash",
    "dataset_version",
    "git_commit",
    "created_at",
)

PARQUET_SCHEMA = pa.schema(
    [
        pa.field("episode_id", pa.string(), nullable=False),
        pa.field("status", pa.string(), nullable=False),
        pa.field("severity", pa.string(), nullable=False),
        pa.field("error_types", pa.string(), nullable=False),
        pa.field("error_count", pa.int32(), nullable=False),
        pa.field("camera_age_p95_ms", pa.float64()),
        pa.field("frame_count", pa.int64(), nullable=False),
        pa.field("episode_hash", pa.string(), nullable=False),
        pa.field("dataset_version", pa.string(), nullable=False),
        pa.field("git_commit", pa.string(), nullable=False),
        pa.field("created_at", pa.string(), nullable=False),
    ]
)

def build_record(
    report: dict[str, Any],
    dataset_version: str,
    git_commit: str,
    created_at: str,
) -> dict[str, Any]:
    errors = report.get("errors", [])
    error_types = sorted({classify_error(error) for error in errors})

    return {
        "episode_id": str(report.get("episode_id") or ""),
        "status": str(report.get("status") or "failed"),
        "severity": str(report.get("severity") or "fatal"),
        "error_types": ";".join(error_types),
        "error_count": len(errors),
        "camera_age_p95_ms": report.get("camera_age_p95_ms"),
        "frame_count": int(report.get("frame_count") or 0),
        "episode_hash": str(report.get("episode_hash") or ""),
        "dataset_version": dataset_version,
        "git_commit": git_commit,
        "created_at": created_at,
    }


def _write_csv(records: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(records)


def _write_parquet(records: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(records, schema=PARQUET_SCHEMA)
    pq.write_table(table, output, compression="snappy")


def run_batch(
    input_dir: Path,
    csv_output: Path,
    parquet_output: Path,
    allowed_root: Path | None = None,
    dataset_version: str = "unversioned",
    git_commit: str = "unknown",
) -> dict[str, Any]:
    """Check every episode directory and write CSV and Parquet reports."""
    root = input_dir.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(f"input directory is not a directory: {root}")

    effective_root = (
        allowed_root.expanduser().resolve(strict=True)
        if allowed_root is not None
        else root
    )
    episode_dirs = sorted(
        [path for path in root.iterdir() if path.is_dir()],
        key=lambda path: path.name,
    )

    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    records = [
        build_record(
            check_episode(path, effective_root),
            dataset_version=dataset_version,
            git_commit=git_commit,
            created_at=created_at,
        )
        for path in episode_dirs
    ]

    csv_path = csv_output.expanduser().resolve()
    parquet_path = parquet_output.expanduser().resolve()
    _write_csv(records, csv_path)
    _write_parquet(records, parquet_path)

    passed = sum(record["status"] == "passed" for record in records)
    failed = len(records) - passed
    severity_counts = Counter(record["severity"] for record in records)
    error_counts = Counter(
        error_type
        for record in records
        for error_type in record["error_types"].split(";")
        if error_type
    )

    return {
        "checked_episodes": len(records),
        "passed": passed,
        "failed": failed,
        "pass_rate": round(passed / len(records), 4) if records else 0.0,
        "severity_counts": dict(sorted(severity_counts.items())),
        "error_type_counts": dict(sorted(error_counts.items())),
        "csv_output": str(csv_path),
        "parquet_output": str(parquet_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing episode folders.",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=Path("reports/qc_results.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--parquet-output",
        type=Path,
        default=Path("reports/qc_results.parquet"),
        help="Output Parquet path.",
    )
    parser.add_argument(
        "--allowed-root",
        type=Path,
        default=None,
        help="Restrict episode paths to this root directory.",
    )
    parser.add_argument(
        "--dataset-version",
        default="unversioned",
        help="Dataset version recorded in the output reports.",
    )
    parser.add_argument(
        "--git-commit",
        default="unknown",
        help="Git commit recorded in the output reports.",
    )
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help="Return exit code 1 when any episode fails QC.",
    )

    args = parser.parse_args()

    try:
        summary = run_batch(
            input_dir=args.input_dir,
            csv_output=args.csv_output,
            parquet_output=args.parquet_output,
            allowed_root=args.allowed_root,
            dataset_version=args.dataset_version,
            git_commit=args.git_commit,
        )
    except (FileNotFoundError, NotADirectoryError) as exc:
        parser.error(str(exc))
        return 2

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.fail_on_error and summary["failed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
