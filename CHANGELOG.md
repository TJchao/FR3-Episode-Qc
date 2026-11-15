# Changelog

## Unreleased

- Added robot-specific JSON profiles for FR3 and Unitree G1.
- Added LeRobot dataset inspection, selected-episode loading, and low-dimensional QC.
- Added seek-based sampled video decoding with an ffmpeg fallback.
- Added black/white-frame, exposure, relative blur, pHash duplicate, and freeze-candidate checks.
- Added multi-dimensional motion profiling without premature pass/fail classification.
- Added image metrics and lightweight gold-set review output.
- Added optional sampled-frame evidence output for human image-rule review.
- Added a sanitized validation report for the official Unitree G1 Dex3 LeRobot dataset.
- Renamed adjacent-frame pHash findings to `image_static_candidate` to avoid confusing static intervals with cross-episode duplicate detection.
- Added image rule-hit versus unique-frame counts, pHash threshold sweep, grouped Unitree motion statistics, explicit unknown units, LeRobot schema version inspection, and episode-length mismatch checks.
- Added gold-set template construction and precision / recall / FPR evaluation.
- Added dependency-free handcrafted visual embeddings, kNN coverage density, cross-episode duplicate candidates, and PCA visualization.
- Added batch-level dataset profiling over low-dimensional QC, motion, sampled image metrics, embedding coverage, and gold-set status.
- Added tests for profiles, image boundaries, motion statistics, LeRobot loading, video decoding, and fatal-image release gating.

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
