# fr3-episode-qc

[![CI](https://github.com/TJchao/FR3-Episode-Qc/actions/workflows/ci.yml/badge.svg)](https://github.com/TJchao/FR3-Episode-Qc/actions/workflows/ci.yml)

Lightweight episode quality checks for FR3 robot-learning data.

This project turns the episode QC workflow used in a Franka Research3 grasping project into a small, reproducible CLI and HTTP service. It checks field completeness, temporal consistency, action/state dimensions, camera latency, RGB/Depth frame consistency, and stores each QC result in SQLite.

The original robot dataset is not included. It contains lab data and is not approved for publication. This repository uses deterministic synthetic episodes that exercise every QC rule.

## Scope

Included:

- Episode QC CLI and FastAPI service
- Error severity classification and conservative action-frame repair
- SQLite QC history with `status` and `dataset_version` filtering
- Batch QC reports in CSV and Parquet, plus DuckDB summaries
- Docker Compose deployment and GitHub Actions CI

Not included:

- Real robot data
- Distributed data processing
- Production-grade lineage or data versioning
- Message queues, Kubernetes, or a multi-node data platform

## Data Schema

Each episode is a directory containing:

```text
episode/
  frames.npz
  metadata.json
```

`frames.npz` fields:

| Field | Shape | Description |
| --- | --- | --- |
| `timestamp_s` | `(N,)` | Episode timestamps in seconds |
| `robot_time_ns` | `(N,)` | Robot-side timestamps in nanoseconds |
| `camera_time_ns` | `(N,)` | Camera timestamps in nanoseconds |
| `camera_age_ms` | `(N,)` | Camera frame age at policy/control time |
| `state` | `(N, 8)` | Robot state used by the policy |
| `action` | `(N, 7)` | Robot action command |

Optional fields reserved for future image repair:

| Field | Shape | Description |
| --- | --- | --- |
| `rgb_timestamps_ns` | `(N,)` | Acquisition timestamps for RGB frames |
| `depth_timestamps_ns` | `(N,)` | Acquisition timestamps for Depth frames |

The current repair implementation does not copy or synthesize image frames. These fields document the timestamp contract that a future nearest-frame image repair would use.

Required `metadata.json` fields:

```json
{
  "episode_id": "ep_valid",
  "frame_count": 128,
  "fps": 10,
  "state_dim": 8,
  "action_dim": 7,
  "rgb_frame_count": 128,
  "depth_frame_count": 128,
  "schema_version": "fr3-episode-v1"
}
```

## QC Rules

An episode fails when any of the following is true:

- `frames.npz` or `metadata.json` is missing
- A required NPZ or metadata field is missing
- Required fields have different first-dimension lengths
- `robot_time_ns` or `timestamp_s` is not strictly increasing
- Any numeric field contains NaN or Inf
- `state` is not `(N, 8)`
- `action` is not `(N, 7)`
- `camera_age_ms` P95 is 50 ms or higher
- `metadata.frame_count` does not match the actual frame count
- `rgb_frame_count` or `depth_frame_count` does not match the actual frame count
- Metadata dimensions or schema version do not match the expected schema

The report includes:

```json
{
  "episode_id": "ep_valid",
  "status": "passed",
  "errors": [],
  "warnings": [],
  "error_details": [],
  "severity": "none",
  "frame_count": 128,
  "camera_age_p95_ms": 31.917,
  "episode_hash": "sha256:..."
}
```

`episode_hash` is a combined SHA-256 hash of `frames.npz` and `metadata.json`, allowing the same episode content to be recognized across QC runs.

### Severity Classification

Each error receives a stable code and one of three severity levels:

- `fatal`: schema, shape, non-finite value, timestamp ordering, or path errors that make an episode unsafe for training.
- `repairable`: all-NaN action rows, or frame/image count mismatches that require controlled repair or manual review.
- `warning`: camera-age P95 between 50 and 100 ms. The episode still fails, but the issue is less severe than a schema or timestamp break.

The overall `severity` is the highest severity present. A passed episode has `severity: "none"`.

## Architecture

```mermaid
flowchart LR
    A[Episode directory] --> B[QC core]
    B --> C[JSON report]
    B --> D[(SQLite history)]
    E[FastAPI] --> B
    E --> D
    A --> H[Batch CLI]
    H --> I[CSV report]
    H --> J[Parquet report]
    J --> K[DuckDB summary]
    A --> L[Repair CLI]
    L --> M[.repaired parts]
    L --> N[repair_manifest.json]
    M --> B
    F[Docker Compose] --> E
    G[Synthetic examples] --> A
```

## Data Synchronization Mechanism

Robot-learning episodes are normalized after several upstream streams are recorded. Camera, robot state, operator input, and the data recorder can run at different rates, so the useful episode boundary is not each producer's native rate; it is the aligned observation-action stream used by training and deployment.

This project focuses on validating that normalized stream:

- `timestamp_s` and `robot_time_ns` must be strictly increasing, exposing dropped, duplicated, or incorrectly ordered records.
- `camera_age_ms` measures how old the selected camera observation is at the policy/control time. Its P95 must stay below the configured threshold, which catches stale images without requiring every camera frame to arrive at exactly the same time as the robot state.
- RGB and Depth frame counts must match the episode frame count, catching partial writes or camera pipeline failures.
- State and action dimensions must match the configured schema, preventing silent train/deploy interface changes.

The synthetic examples use a small 10 Hz normalized episode stream for demonstration. The camera-age threshold and schema dimensions are shared between data generation and validation, so test data and QC rules cannot drift apart.

## Local Setup

```bash
python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

Generate synthetic examples:

```bash
python scripts/make_examples.py
```

## CLI

```bash
python -m app.qc examples/ep_valid
python -m app.qc examples/ep_camera_age_p95
```

The command prints a JSON report. It exits with code `0` when the episode passes and `1` when it fails, so it can be used in a shell pipeline.

To restrict episode paths to a specific root:

```bash
python -m app.qc examples/ep_valid --allowed-root examples
```

Check every episode under a directory and write CSV and Parquet reports:

```bash
python -m app.qc_batch examples \
  --csv-output reports/qc_results.csv \
  --parquet-output reports/qc_results.parquet \
  --dataset-version synthetic-v1 \
  --git-commit "$(git rev-parse --short HEAD)"
```

The batch command prints the number of checked, passed, and failed episodes plus a structured count of each error type. Add `--fail-on-error` when it is used as a pipeline gate.

## Conservative Repair

The repair CLI handles one narrow failure mode: an action row in which all seven dimensions are NaN. This represents a missing action label. Partial NaN, Inf, malformed action shapes, and missing image data are not repaired.

```bash
python -m app.repair path/to/episode
```

The source episode is read-only. Output is written to `<episode>.repaired/` and contains one or more `part_###` episode directories plus `repair_manifest.json`.

Repair rules:

- Missing action frames at the start or end are cropped.
- A missing action frame in the middle splits the episode into two parts, avoiding a 200 ms timestamp hole being presented as two adjacent 100 ms steps.
- No action or state values are interpolated.
- Each repaired part is rerun through QC before the manifest is marked as passed.
- The manifest records the source hash, repaired hashes, repair policy, dropped frame indices, timestamp deltas, and split points.

A sanitized example manifest is available at [`examples/repair_manifest_sample.json`](examples/repair_manifest_sample.json).

Summarize the Parquet report with DuckDB:

```bash
python -m app.stats reports/qc_results.parquet
```

The summary includes status counts, error-type counts, camera-age statistics by status, and dataset-version counts.

## API

Start the development server:

```bash
uvicorn app.main:app --reload
```

Interactive documentation is available at:

```text
http://127.0.0.1:8000/docs
```

Check one episode:

```bash
curl -X POST http://127.0.0.1:8000/qc \
  -H "Content-Type: application/json" \
  -d "{\"episode_path\": \"examples/ep_valid\"}"
```

Read recent QC runs:

```bash
curl http://127.0.0.1:8000/runs
```

Filter runs by status, severity, or dataset version:

```bash
curl "http://127.0.0.1:8000/runs?status=failed"
curl "http://127.0.0.1:8000/runs?dataset_version=synthetic-v1"
curl "http://127.0.0.1:8000/runs?status=failed&dataset_version=synthetic-v1"
curl "http://127.0.0.1:8000/runs?severity=repairable"
```

Read one run:

```bash
curl http://127.0.0.1:8000/runs/1
```

The service restricts episode access to `QC_ALLOWED_ROOT` when that environment variable is set. It uses `QC_DB_PATH` for the SQLite database when provided.

## Docker

```bash
docker compose up --build
```

The service then listens on:

```text
http://127.0.0.1:8000
```

In Docker Compose, synthetic episodes are mounted read-only at `/data`, and SQLite history is persisted in a named volume.

Example request inside Docker:

```bash
curl -X POST http://127.0.0.1:8000/qc \
  -H "Content-Type: application/json" \
  -d "{\"episode_path\": \"/data/ep_valid\"}"
```

## Tests

```bash
pytest
```

The tests cover:

- The valid synthetic episode
- Each isolated invalid episode
- Severity classification for fatal, repairable, and warning errors
- Boundary cropping and middle splitting for all-NaN action rows
- Repair manifests, source hashes, repaired hashes, and post-repair QC
- Batch QC, structured error classification, CSV, and Parquet outputs
- DuckDB summaries over generated Parquet reports
- Path-root restrictions
- API health and QC endpoints
- API filtering by status and dataset version
- Request validation and HTTP error codes
- SQLite history insertion and lookup

GitHub Actions runs the Python tests and builds the Docker image on every push and pull request.

## Design Notes

- The QC threshold and dimensions are shared between data generation and validation, avoiding drift between test data and checks.
- Paths are resolved before access and constrained to an allowed root.
- NPZ loading disables pickle for safety.
- SQLite is intentionally minimal; it records report history without introducing an ORM or a separate service.
- SQLite is suitable for this single-service, small-batch laboratory workflow. Under many concurrent writers, its database-level write lock can become a bottleneck; a production multi-worker deployment would use PostgreSQL or another external database service.
- Parquet and DuckDB provide a small, local analytics path without introducing a distributed processing framework.
- The repository is a small engineering demonstration, not a claim of large-scale robot-data infrastructure.

## License

This project is released under the [MIT License](LICENSE).
