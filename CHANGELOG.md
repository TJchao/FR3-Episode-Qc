# Changelog

## v0.3.0 - 2026-11-04

- Added dataset release reports in Markdown and self-contained HTML.
- Added `release_gate.json` with repeatable `--fail-on` severity controls.
- Added fail-closed checks for empty input, mixed dataset versions, mixed git commits, duplicate episode IDs, and lineage mismatches.
- Added SQLite `--dataset-version` snapshots and latest-run-per-episode selection.
- Added camera-age P50/P95/P99 and frame-count statistics to release reports.
- Added repair-manifest summaries.
- Added report and gate regression tests.
- CI now verifies the expected blocked release exit code and uploads release artifacts.

## v0.2.0 - 2026-11-01

- Added `fatal` / `repairable` / `warning` severity classification.
- Added stable error codes.
- Added conservative repair for all-NaN action rows.
- Added boundary cropping and middle-episode splitting.
- Added `repair_manifest.json` with source hashes, repaired hashes, dropped frames, timestamp deltas, and post-repair QC results.
- Kept source episodes read-only.

## v0.1.0 - 2026-10-21

- Initial public release.
- Added episode QC CLI and FastAPI service.
- Added SQLite history and status / severity / dataset-version filtering.
- Added batch CSV and Parquet reports.
- Added DuckDB summaries.
- Added Docker and GitHub Actions CI.
