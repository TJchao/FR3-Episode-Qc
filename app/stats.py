"""DuckDB summaries for generated episode QC Parquet reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import duckdb


def summarize_qc_results(parquet_path: Path) -> dict[str, Any]:
    """Compute status, failure-type, latency, and version summaries."""
    path = parquet_path.expanduser().resolve(strict=True)
    connection = duckdb.connect(database=":memory:")

    status_rows = connection.execute(
        """
        SELECT status, count(*)::INTEGER AS count
        FROM read_parquet(?)
        GROUP BY status
        ORDER BY status
        """,
        [str(path)],
    ).fetchall()

    error_rows = connection.execute(
        """
        SELECT error_type, count(*)::INTEGER AS count
        FROM (
            SELECT unnest(string_split(error_types, ';')) AS error_type
            FROM read_parquet(?)
            WHERE error_types IS NOT NULL AND error_types <> ''
        )
        GROUP BY error_type
        ORDER BY count DESC, error_type
        """,
        [str(path)],
    ).fetchall()

    camera_rows = connection.execute(
        """
        SELECT
            status,
            count(*)::INTEGER AS episode_count,
            count(camera_age_p95_ms)::INTEGER AS measured_episode_count,
            CAST(round(avg(camera_age_p95_ms), 3) AS DOUBLE) AS avg_camera_age_p95_ms,
            CAST(min(camera_age_p95_ms) AS DOUBLE) AS min_camera_age_p95_ms,
            CAST(max(camera_age_p95_ms) AS DOUBLE) AS max_camera_age_p95_ms
        FROM read_parquet(?)
        GROUP BY status
        ORDER BY status
        """,
        [str(path)],
    ).fetchall()

    version_rows = connection.execute(
        """
        SELECT dataset_version, count(*)::INTEGER AS count
        FROM read_parquet(?)
        GROUP BY dataset_version
        ORDER BY dataset_version
        """,
        [str(path)],
    ).fetchall()

    status_counts = {status: count for status, count in status_rows}
    total_episodes = sum(status_counts.values())

    return {
        "total_episodes": total_episodes,
        "status_counts": status_counts,
        "error_type_counts": {
            error_type: count for error_type, count in error_rows
        },
        "camera_age_by_status": [
            {
                "status": status,
                "episode_count": episode_count,
                "measured_episode_count": measured_episode_count,
                "avg_camera_age_p95_ms": avg_ms,
                "min_camera_age_p95_ms": min_ms,
                "max_camera_age_p95_ms": max_ms,
            }
            for (
                status,
                episode_count,
                measured_episode_count,
                avg_ms,
                min_ms,
                max_ms,
            ) in camera_rows
        ],
        "dataset_version_counts": {
            dataset_version: count for dataset_version, count in version_rows
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "parquet_path",
        type=Path,
        help="Path to a qc_results.parquet file.",
    )
    args = parser.parse_args()

    try:
        summary = summarize_qc_results(args.parquet_path)
    except FileNotFoundError as exc:
        parser.error(str(exc))
        return 2

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
