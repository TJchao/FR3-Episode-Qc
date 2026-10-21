"""SQLite persistence for episode quality-check runs."""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_DB_PATH = Path(__file__).resolve().parents[1] / "db" / "qc_history.db"


def _resolve_db_path(db_path: str | Path | None) -> Path:
    if db_path is not None:
        return Path(db_path).expanduser().resolve()

    configured_path = os.getenv("QC_DB_PATH")
    if configured_path:
        return Path(configured_path).expanduser().resolve()

    return DEFAULT_DB_PATH


def _connect(db_path: str | Path | None = None) -> sqlite3.Connection:
    path = _resolve_db_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(path, timeout=30.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS qc_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            episode_id TEXT NOT NULL,
            episode_hash TEXT NOT NULL,
            dataset_version TEXT NOT NULL DEFAULT 'unversioned',
            git_commit TEXT NOT NULL DEFAULT 'unknown',
            status TEXT NOT NULL,
            report_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    return connection


def _serialize_report(report: dict[str, Any]) -> str:
    return json.dumps(
        report,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _deserialize_report(report_json: str) -> dict[str, Any]:
    return json.loads(report_json)


def _row_to_record(row: sqlite3.Row) -> dict[str, Any]:
    record = dict(row)
    record["report"] = _deserialize_report(record.pop("report_json"))
    return record


def insert_qc_run(
    report: dict[str, Any],
    dataset_version: str = "unversioned",
    git_commit: str = "unknown",
    db_path: str | Path | None = None,
) -> int:
    """Insert one QC report and return the generated run id."""
    required_fields = ("episode_id", "episode_hash", "status")
    missing_fields = [
        field for field in required_fields if report.get(field) is None
    ]
    if missing_fields:
        raise ValueError(
            "QC report missing required fields: " + ", ".join(missing_fields)
        )

    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with _connect(db_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO qc_runs (
                episode_id,
                episode_hash,
                dataset_version,
                git_commit,
                status,
                report_json,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(report["episode_id"]),
                str(report["episode_hash"]),
                dataset_version,
                git_commit,
                str(report["status"]),
                _serialize_report(report),
                created_at,
            ),
        )
        return int(cursor.lastrowid)


def list_qc_runs(
    limit: int = 100,
    status: str | None = None,
    dataset_version: str | None = None,
    db_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Return recent QC runs, optionally filtered, newest first."""
    if limit <= 0:
        raise ValueError("limit must be a positive integer")
    if status is not None and status not in {"passed", "failed"}:
        raise ValueError("status must be 'passed' or 'failed'")
    if dataset_version is not None and not dataset_version.strip():
        raise ValueError("dataset_version must be a non-empty string")

    conditions: list[str] = []
    parameters: list[str | int] = []
    if status is not None:
        conditions.append("status = ?")
        parameters.append(status)
    if dataset_version is not None:
        conditions.append("dataset_version = ?")
        parameters.append(dataset_version)

    where_sql = " WHERE " + " AND ".join(conditions) if conditions else ""

    with _connect(db_path) as connection:
        rows = connection.execute(
            f"""
            SELECT
                id,
                episode_id,
                episode_hash,
                dataset_version,
                git_commit,
                status,
                report_json,
                created_at
            FROM qc_runs
            {where_sql}
            ORDER BY id DESC
            LIMIT ?
            """,
            (*parameters, limit),
        ).fetchall()

    return [_row_to_record(row) for row in rows]


def get_qc_run(
    run_id: int,
    db_path: str | Path | None = None,
) -> dict[str, Any] | None:
    """Return one QC run by id, or None when it does not exist."""
    with _connect(db_path) as connection:
        row = connection.execute(
            """
            SELECT
                id,
                episode_id,
                episode_hash,
                dataset_version,
                git_commit,
                status,
                report_json,
                created_at
            FROM qc_runs
            WHERE id = ?
            """,
            (run_id,),
        ).fetchone()

    return _row_to_record(row) if row is not None else None
