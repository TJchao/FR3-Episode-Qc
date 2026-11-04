"""Dataset release report and gate for episode QC results.

Turns a batch QC output (``qc_results.parquet`` or the SQLite history database)
into a human-readable dataset report and an explicit release decision
(``release_gate.json``) that records whether a dataset version may enter
training.

Two subcommands::

    python -m app.release report <qc_results.parquet|qc_history.db> [--html]
    python -m app.release gate   <qc_results.parquet|qc_history.db> --fail-on fatal

The gate command exits 1 when the requested severity level is present, which
makes it usable as a CI gate for dataset releases.
"""

from __future__ import annotations

import argparse
import html
import json
import sqlite3
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow.parquet as pq

from app.errors import classify_error

SEVERITY_LEVELS = ("fatal", "repairable", "warning", "none")


def load_records(
    source: str | Path,
    dataset_version: str | None = None,
) -> list[dict[str, Any]]:
    """Load QC records from a parquet report or a SQLite history database.

    Both inputs are normalised to one record shape::

        episode_id, status, severity, error_types (list[str]),
        error_count, camera_age_p95_ms, frame_count, episode_hash,
        dataset_version, git_commit, created_at
    """
    path = Path(source).expanduser().resolve(strict=True)
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return _validate_records(_records_from_parquet(path), dataset_version)
    if suffix in {".db", ".sqlite", ".sqlite3"}:
        return _validate_records(_records_from_sqlite(path, dataset_version), dataset_version)
    raise ValueError(
        f"unsupported source: {path.name} "
        "(expected qc_results.parquet or a SQLite database)"
    )


def _records_from_parquet(path: Path) -> list[dict[str, Any]]:
    table = pq.read_table(path)
    records = []
    for row in table.to_pylist():
        records.append(
            {
                "episode_id": str(row.get("episode_id") or ""),
                "status": str(row.get("status") or "failed"),
                "severity": str(row.get("severity") or "fatal"),
                "error_types": [
                    item
                    for item in str(row.get("error_types") or "").split(";")
                    if item
                ],
                "error_count": int(row.get("error_count") or 0),
                "camera_age_p95_ms": row.get("camera_age_p95_ms"),
                "frame_count": int(row.get("frame_count") or 0),
                "episode_hash": str(row.get("episode_hash") or ""),
                "dataset_version": str(row.get("dataset_version") or "unversioned"),
                "git_commit": str(row.get("git_commit") or "unknown"),
                "created_at": str(row.get("created_at") or ""),
            }
        )
    return records


def _records_from_sqlite(
    path: Path,
    dataset_version: str | None,
) -> list[dict[str, Any]]:
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    conditions = []
    parameters: list[str] = []
    if dataset_version is not None:
        conditions.append("dataset_version = ?")
        parameters.append(dataset_version)
    where_sql = " WHERE " + " AND ".join(conditions) if conditions else ""
    rows = connection.execute(
        f"""
        WITH ranked AS (
            SELECT *,
                   ROW_NUMBER() OVER (
                       PARTITION BY episode_id
                       ORDER BY id DESC
                   ) AS row_number
            FROM qc_runs{where_sql}
        )
        SELECT episode_id, episode_hash, dataset_version, git_commit,
               status, report_json, created_at
        FROM ranked
        WHERE row_number = 1
        ORDER BY episode_id
        """,
        parameters,
    ).fetchall()
    connection.close()

    records = []
    for row in rows:
        report = json.loads(row["report_json"])
        errors = list(report.get("errors") or [])
        error_types = sorted(
            {classify_error(error) for error in errors} - {"unknown"}
        )
        records.append(
            {
                "episode_id": str(row["episode_id"]),
                "status": str(row["status"]),
                "severity": str(report.get("severity") or "fatal"),
                "error_types": error_types,
                "error_count": len(errors),
                "camera_age_p95_ms": report.get("camera_age_p95_ms"),
                "frame_count": int(report.get("frame_count") or 0),
                "episode_hash": str(row["episode_hash"]),
                "dataset_version": str(row["dataset_version"]),
                "git_commit": str(row["git_commit"]),
                "created_at": str(row["created_at"]),
            }
        )
    return records


def _validate_records(
    records: list[dict[str, Any]],
    dataset_version: str | None,
) -> list[dict[str, Any]]:
    """Filter to one release version and reject ambiguous episode lineage."""
    rows = records
    if dataset_version is not None:
        rows = [
            row
            for row in rows
            if row.get("dataset_version") == dataset_version
        ]

    versions = sorted(
        {
            row.get("dataset_version") or "unversioned"
            for row in rows
        }
    )
    if len(versions) > 1:
        raise ValueError(
            "source contains multiple dataset versions; "
            "pass --dataset-version with one of: " + ", ".join(versions)
        )

    episode_id_counts = Counter(
        row.get("episode_id") or "" for row in rows
    )
    duplicates = sorted(
        episode_id for episode_id, count in episode_id_counts.items()
        if episode_id and count > 1
    )
    if duplicates:
        preview = ", ".join(duplicates[:10])
        raise ValueError(f"source contains duplicate episode IDs: {preview}")

    commits = sorted({row.get("git_commit") or "unknown" for row in rows})
    if len(commits) > 1:
        raise ValueError(
            "source contains multiple git commits; regenerate the batch report "
            "from one commit or provide an explicit release commit"
        )

    return rows


def summarize_records(
    records: Iterable[dict[str, Any]],
    top_n: int = 10,
) -> dict[str, Any]:
    """Compute the aggregate statistics shown in the dataset report."""
    rows = list(records)
    total = len(rows)
    passed = sum(row["status"] == "passed" for row in rows)
    failed = total - passed

    severity_counts = Counter(row["severity"] for row in rows)
    error_counts = Counter(
        error_type
        for row in rows
        for error_type in row.get("error_types") or []
    )
    top_errors = dict(error_counts.most_common(top_n))

    camera_values = sorted(
        value
        for row in rows
        for value in [row.get("camera_age_p95_ms")]
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    )
    frame_values = [int(row["frame_count"]) for row in rows]

    def percentile(values: list[float], q: float) -> float | None:
        if not values:
            return None
        return round(float(np.percentile(np.asarray(values, dtype=float), q)), 3)

    return {
        "total_episodes": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": round(passed / total, 4) if total else 0.0,
        "severity_counts": {
            level: severity_counts.get(level, 0) for level in SEVERITY_LEVELS
        },
        "error_type_counts": top_errors,
        "camera_age_p95": {
            "count": len(camera_values),
            "p50_ms": percentile(camera_values, 50),
            "p95_ms": percentile(camera_values, 95),
            "p99_ms": percentile(camera_values, 99),
        },
        "frame_count": {
            "count": len(frame_values),
            "min": min(frame_values) if frame_values else 0,
            "max": max(frame_values) if frame_values else 0,
            "mean": round(float(np.mean(frame_values)), 2) if frame_values else 0.0,
            "p50": percentile([float(v) for v in frame_values], 50),
        },
        "failed_episodes": [
            {
                "episode_id": row["episode_id"],
                "severity": row["severity"],
                "error_types": row.get("error_types") or [],
                "camera_age_p95_ms": row.get("camera_age_p95_ms"),
                "frame_count": row["frame_count"],
            }
            for row in rows
            if row["status"] != "passed"
        ],
    }


def _manifest_summary(manifest_paths: list[str]) -> list[dict[str, Any]]:
    summaries = []
    for raw in manifest_paths:
        path = Path(raw).expanduser().resolve()
        if not path.is_file():
            summaries.append({"path": str(path), "error": "not found"})
            continue
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            summaries.append({"path": str(path), "error": "unreadable"})
            continue
        repaired = manifest.get("repaired_episodes") or []
        summaries.append(
            {
                "path": str(path),
                "source_episode_id": manifest.get("source_episode_id"),
                "repair_policy": manifest.get("repair_policy"),
                "repaired_episode_count": len(repaired),
                "dropped_frames": len(manifest.get("dropped_frames") or []),
                "result_status": (manifest.get("result") or {}).get("status"),
            }
        )
    return summaries


def render_markdown(
    stats: dict[str, Any],
    source: str,
    dataset_version: str,
    git_commit: str,
    generated_at: str,
    manifest_summaries: list[dict[str, Any]] | None = None,
) -> str:
    """Render the dataset QC report as Markdown."""
    lines = [
        "# Dataset QC Report",
        "",
        f"- Source: `{source}`",
        f"- Dataset version: `{dataset_version}`",
        f"- Git commit: `{git_commit}`",
        f"- Generated at: `{generated_at}`",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Total episodes | {stats['total_episodes']} |",
        f"| Passed | {stats['passed']} |",
        f"| Failed | {stats['failed']} |",
        f"| Pass rate | {stats['pass_rate']:.2%} |",
        "",
        "## Severity",
        "",
        "| Severity | Episodes |",
        "|---|---|",
    ]
    for level in SEVERITY_LEVELS:
        lines.append(f"| {level} | {stats['severity_counts'][level]} |")

    lines += ["", "## Error Types (Top)", "", "| Error type | Episodes |", "|---|---|"]
    if stats["error_type_counts"]:
        for error_type, count in stats["error_type_counts"].items():
            lines.append(f"| {error_type} | {count} |")
    else:
        lines.append("| _(none)_ | 0 |")

    camera = stats["camera_age_p95"]
    frames = stats["frame_count"]
    lines += [
        "",
        "## Camera Age P95 (ms)",
        "",
        "| Statistic | Value |",
        "|---|---|",
        f"| Measured episodes | {camera['count']} |",
        f"| P50 | {camera['p50_ms']} |",
        f"| P95 | {camera['p95_ms']} |",
        f"| P99 | {camera['p99_ms']} |",
        "",
        "## Frame Count",
        "",
        "| Statistic | Value |",
        "|---|---|",
        f"| Episodes | {frames['count']} |",
        f"| Min | {frames['min']} |",
        f"| Max | {frames['max']} |",
        f"| Mean | {frames['mean']} |",
        f"| P50 | {frames['p50']} |",
        "",
        "## Failed Episodes",
        "",
    ]
    failed = stats["failed_episodes"]
    if failed:
        lines += [
            "| Episode | Severity | Error types | Camera age P95 (ms) | Frames |",
            "|---|---|---|---|---|",
        ]
        for episode in failed:
            lines.append(
                "| {} | {} | {} | {} | {} |".format(
                    episode["episode_id"],
                    episode["severity"],
                    "; ".join(episode["error_types"]) or "—",
                    episode["camera_age_p95_ms"]
                    if episode["camera_age_p95_ms"] is not None
                    else "—",
                    episode["frame_count"],
                )
            )
    else:
        lines.append("_(no failed episodes)_")

    lines += ["", "## Repair Manifests", ""]
    if manifest_summaries:
        for item in manifest_summaries:
            if "error" in item:
                lines.append(f"- {item['path']} ({item['error']})")
            else:
                lines.append(
                    f"- `{item['path']}` — policy `{item['repair_policy']}`, "
                    f"{item['repaired_episode_count']} repaired episode(s), "
                    f"{item['dropped_frames']} dropped frame(s), result "
                    f"`{item['result_status']}`"
                )
    else:
        lines.append("_(no repair manifests found)_")

    return "\n".join(lines) + "\n"


def _escape(value: Any) -> str:
    return html.escape(str(value))


def render_html(
    stats: dict[str, Any],
    source: str,
    dataset_version: str,
    git_commit: str,
    generated_at: str,
    manifest_summaries: list[dict[str, Any]] | None = None,
) -> str:
    """Render the dataset QC report as a self-contained HTML page."""
    camera = stats["camera_age_p95"]
    frames = stats["frame_count"]

    def table(headers: list[str], rows: list[list[Any]]) -> str:
        body = "".join(
            "<tr>" + "".join(f"<td>{_escape(cell)}</td>" for cell in row) + "</tr>"
            for row in rows
        )
        head = "".join(f"<th>{_escape(header)}</th>" for header in headers)
        return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"

    failed_rows = [
        [
            episode["episode_id"],
            episode["severity"],
            "; ".join(episode["error_types"]) or "—",
            episode["camera_age_p95_ms"]
            if episode["camera_age_p95_ms"] is not None
            else "—",
            episode["frame_count"],
        ]
        for episode in stats["failed_episodes"]
    ]

    manifest_items = []
    if manifest_summaries:
        for item in manifest_summaries:
            if "error" in item:
                manifest_items.append(f"<li>{_escape(item['path'])} ({item['error']})</li>")
            else:
                manifest_items.append(
                    "<li><code>{}</code> — policy <code>{}</code>, {} repaired "
                    "episode(s), {} dropped frame(s), result <code>{}</code></li>".format(
                        _escape(item["path"]),
                        _escape(item["repair_policy"]),
                        item["repaired_episode_count"],
                        item["dropped_frames"],
                        _escape(item["result_status"]),
                    )
                )
    manifest_html = "<ul>" + "".join(manifest_items) + "</ul>" if manifest_items else "<p>_(no repair manifests found)_</p>"

    error_rows = [
        [error_type, count] for error_type, count in stats["error_type_counts"].items()
    ]
    if not error_rows:
        error_rows = [["_(none)_", 0]]

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Dataset QC Report</title>
<style>
body {{ font-family: -apple-system, "Segoe UI", Roboto, sans-serif; margin: 2rem auto; max-width: 900px; color: #1f2328; }}
h1, h2 {{ border-bottom: 1px solid #d0d7de; padding-bottom: .3rem; }}
table {{ border-collapse: collapse; width: 100%; margin: .5rem 0 1.5rem; }}
th, td {{ border: 1px solid #d0d7de; padding: .4rem .6rem; text-align: left; }}
th {{ background: #f6f8fa; }}
code {{ background: #f6f8fa; padding: .1rem .3rem; border-radius: 4px; }}
.meta {{ color: #57606a; }}
</style>
</head>
<body>
<h1>Dataset QC Report</h1>
<p class="meta">Source: <code>{_escape(source)}</code> · Dataset version: <code>{_escape(dataset_version)}</code> · Git commit: <code>{_escape(git_commit)}</code> · Generated at: <code>{_escape(generated_at)}</code></p>

<h2>Summary</h2>
{table(["Metric", "Value"], [["Total episodes", stats["total_episodes"]], ["Passed", stats["passed"]], ["Failed", stats["failed"]], ["Pass rate", f"{stats['pass_rate']:.2%}"]])}

<h2>Severity</h2>
{table(["Severity", "Episodes"], [[level, stats["severity_counts"][level]] for level in SEVERITY_LEVELS])}

<h2>Error Types (Top)</h2>
{table(["Error type", "Episodes"], error_rows)}

<h2>Camera Age P95 (ms)</h2>
{table(["Statistic", "Value"], [["Measured episodes", camera["count"]], ["P50", camera["p50_ms"]], ["P95", camera["p95_ms"]], ["P99", camera["p99_ms"]]])}

<h2>Frame Count</h2>
{table(["Statistic", "Value"], [["Episodes", frames["count"]], ["Min", frames["min"]], ["Max", frames["max"]], ["Mean", frames["mean"]], ["P50", frames["p50"]]])}

<h2>Failed Episodes</h2>
{"<p>_(no failed episodes)_</p>" if not failed_rows else table(["Episode", "Severity", "Error types", "Camera age P95 (ms)", "Frames"], failed_rows)}

<h2>Repair Manifests</h2>
{manifest_html}
</body>
</html>
"""


def generate_report(
    records: list[dict[str, Any]],
    source: str,
    output: Path,
    include_html: bool = False,
    top_n: int = 10,
    repair_manifest_paths: list[str] | None = None,
) -> dict[str, Any]:
    """Write the Markdown (and optionally HTML) dataset report."""
    stats = summarize_records(records, top_n=top_n)
    dataset_version = _common_dataset_version(records)
    git_commit = _common_git_commit(records)
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    manifest_summaries = _manifest_summary(repair_manifest_paths or [])

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        render_markdown(
            stats,
            source=source,
            dataset_version=dataset_version,
            git_commit=git_commit,
            generated_at=generated_at,
            manifest_summaries=manifest_summaries,
        ),
        encoding="utf-8",
    )

    html_path = output.with_suffix(".html")
    if include_html:
        html_path.write_text(
            render_html(
                stats,
                source=source,
                dataset_version=dataset_version,
                git_commit=git_commit,
                generated_at=generated_at,
                manifest_summaries=manifest_summaries,
            ),
            encoding="utf-8",
        )

    return {
        "output_markdown": str(output),
        "output_html": str(html_path) if include_html else None,
        "summary": stats,
        "dataset_version": dataset_version,
        "git_commit": git_commit,
        "generated_at": generated_at,
    }


def _common_dataset_version(records: list[dict[str, Any]]) -> str:
    versions = [row.get("dataset_version") or "unversioned" for row in records]
    if not versions:
        return "unknown"
    return versions[0] if len(set(versions)) == 1 else "mixed"


def _common_git_commit(records: list[dict[str, Any]]) -> str:
    commits = [row.get("git_commit") or "unknown" for row in records]
    if not commits:
        return "unknown"
    return commits[0] if len(set(commits)) == 1 else "mixed"


def _git_head() -> str | None:
    """Return the current short git commit, or None when unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def evaluate_gate(
    records: list[dict[str, Any]],
    fail_on: Iterable[str],
    dataset_version: str | None = None,
    git_commit: str | None = None,
) -> dict[str, Any]:
    """Decide whether a dataset version may enter training.

    ``fail_on`` is the set of severity levels that block the release (for
    example ``{"fatal", "repairable"}``). ``blocked_by`` lists the requested
    levels that are actually present in the records.
    """
    fail_levels = [level for level in SEVERITY_LEVELS[:3] if level in fail_on]
    severity_counts = Counter(row.get("severity") for row in records)
    blocked_by = [
        level for level in fail_levels if severity_counts.get(level, 0) > 0
    ]

    resolved_version = dataset_version or _common_dataset_version(records)
    record_version = _common_dataset_version(records)
    record_commit = _common_git_commit(records)
    resolved_commit = (
        git_commit
        or (_git_head() if record_commit == "unknown" else record_commit)
        or "unknown"
    )

    blocking_reasons = list(blocked_by)
    if not records:
        blocking_reasons.append("empty_input")
    if record_version == "mixed":
        blocking_reasons.append("dataset_version_mismatch")
    if dataset_version and record_version not in {"unknown", "mixed"} and dataset_version != record_version:
        blocking_reasons.append("dataset_version_mismatch")
    if record_commit == "mixed":
        blocking_reasons.append("git_commit_mismatch")
    if git_commit and record_commit not in {"unknown", "mixed"} and git_commit != record_commit:
        blocking_reasons.append("git_commit_mismatch")

    invalid_severity_count = sum(
        row.get("severity") not in SEVERITY_LEVELS for row in records
    )
    if invalid_severity_count:
        blocking_reasons.append("invalid_severity")

    return {
        "dataset_version": resolved_version,
        "git_commit": resolved_commit,
        "source_dataset_version": record_version,
        "source_git_commit": record_commit,
        "passed": not blocking_reasons,
        "blocked_by": blocked_by,
        "blocking_reasons": blocking_reasons,
        "severity_counts": {
            level: severity_counts.get(level, 0) for level in SEVERITY_LEVELS
        },
        "invalid_severity_count": invalid_severity_count,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m app.release", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    report_parser = subparsers.add_parser(
        "report", help="Generate the dataset QC report (Markdown + optional HTML)."
    )
    report_parser.add_argument(
        "source",
        type=Path,
        help="qc_results.parquet or a SQLite history database.",
    )
    report_parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/qc_report.md"),
        help="Markdown report output path.",
    )
    report_parser.add_argument(
        "--html",
        action="store_true",
        help="Also write a self-contained HTML report next to the Markdown output.",
    )
    report_parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top error types to include (default 10).",
    )
    report_parser.add_argument(
        "--repair-manifest",
        nargs="*",
        default=[],
        help="Optional repair_manifest.json path(s) to link from the report.",
    )
    report_parser.add_argument(
        "--dataset-version",
        help="Select one dataset version when the source contains multiple versions.",
    )

    gate_parser = subparsers.add_parser(
        "gate",
        help="Evaluate the release gate and write release_gate.json.",
    )
    gate_parser.add_argument(
        "source",
        type=Path,
        help="qc_results.parquet or a SQLite history database.",
    )
    gate_parser.add_argument(
        "--fail-on",
        action="append",
        choices=["fatal", "repairable", "warning"],
        default=[],
        help=(
            "Severity level(s) that block the release; repeatable. "
            "Defaults to fatal when not provided."
        ),
    )
    gate_parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/release_gate.json"),
        help="release_gate.json output path.",
    )
    gate_parser.add_argument(
        "--dataset-version",
        help="Select one dataset version when the source contains multiple versions.",
    )
    gate_parser.add_argument("--git-commit", help="Override the git commit.")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    try:
        records = load_records(args.source, dataset_version=args.dataset_version)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
        return 2

    if args.command == "report":
        try:
            result = generate_report(
                records,
                source=str(args.source),
                output=args.output,
                include_html=args.html,
                top_n=args.top_n,
                repair_manifest_paths=args.repair_manifest,
            )
        except OSError as exc:
            parser.error(str(exc))
            return 2
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
        print(f"Markdown report: {result['output_markdown']}")
        if result["output_html"]:
            print(f"HTML report: {result['output_html']}")
        return 0

    fail_on = args.fail_on or ["fatal"]
    gate = evaluate_gate(
        records,
        fail_on=fail_on,
        dataset_version=args.dataset_version,
        git_commit=args.git_commit,
    )
    gate["fail_on"] = fail_on
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(gate, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(gate, ensure_ascii=False, indent=2))
    print(f"Gate file: {args.output}")
    return 0 if gate["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
