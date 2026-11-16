# fr3-episode-qc

[![CI](https://github.com/TJchao/FR3-Episode-Qc/actions/workflows/ci.yml/badge.svg)](https://github.com/TJchao/FR3-Episode-Qc/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/TJchao/FR3-Episode-Qc)](https://github.com/TJchao/FR3-Episode-Qc/releases)
[![License](https://img.shields.io/github/license/TJchao/FR3-Episode-Qc)](LICENSE)

`fr3-episode-qc` is a reproducible quality, profiling, and release-gate toolkit for FR3 episodes and LeRobot v3 robot-learning datasets.

## Business Background

Robot-learning datasets are expensive to collect and easy to corrupt silently: timestamps drift, camera frames lag, action semantics change, and repeated scenes increase storage without improving policy coverage. This project turns those operational risks into explicit checks, review queues, calibrated thresholds, and release decisions.

The original FR3 robot dataset is not included because it contains unpublished laboratory data. Deterministic synthetic fixtures exercise every QC rule, while the public Unitree G1 LeRobot dataset provides integration and reporting evidence.

## Completed Capabilities

- FR3 NPZ episode QC: schema, timestamp monotonicity, NaN/Inf, state/action dimensions, camera-age P95, and RGB/Depth frame-count consistency.
- Conservative repair: source episodes remain read-only; all-NaN action rows are cropped or split, and every repair writes a manifest.
- LeRobot v3 support: dataset inspection, selected-episode loading, robot profiles, sampled video decoding, and per-episode reports.
- Sampled image QC: black/white frames, exposure ratios, relative blur, adjacent-frame static candidates, and freeze candidates that combine visual stasis with robot-state motion.
- Gold-set workflow: local HTML review, source-count-weighted precision/FPR reporting, and threshold calibration.
- Visual coverage: dependency-free 216-D handcrafted episode embeddings, original-space kNN density, PCA visualization, and connected duplicate-candidate groups.
- Dataset profiling: low-dimensional quality, image metrics, grouped motion statistics, coverage disclosures, and gold-set status.
- Governance: SQLite history, CSV/Parquet outputs, DuckDB summaries, Markdown/HTML release reports, severity-aware release gates, Docker, and GitHub Actions.

## P1.3 Threshold Impact

Both runs use episodes 0-85, the left-wrist camera, 1 fps sampling, and 2,806 sampled frames.

| Metric | P5 / Hamming 4 baseline | P2 / Hamming 2 calibrated | Change |
|---|---:|---:|---:|
| blur hits | 175 | 86 | -50.9% |
| freeze hits | 121 | 121 | 0.0% |
| static-candidate hits | 684 | 338 | -50.6% |
| total rule hits | 980 | 545 | -44.4% |
| unique flagged frames | 855 | 423 | -50.5% |
| review-candidate rate | 34.93% | 19.42% | -15.51 pp |

The calibrated policy came from a 120-row human gold set:

```text
review completion: 100%
overall precision: 53.3333%
weighted FPR estimate: 6.6404%
recall: null
F1: null
```

Recall and F1 are intentionally null because the balanced hit/control sample does not cover all false negatives.

## Key Experimental Findings

- The 301-episode low-dimensional profile passed all six executable rules. Four rules were skipped because LeRobot omitted `camera_age_ms`, `robot_time_ns`, `rgb_frame_count`, and `depth_frame_count`; skipped is not equivalent to passed.
- Frame count has a clear long tail: min 666, mean 934.20598, p95 1131, max 6791. Rule success does not imply a healthy duration distribution.
- Right-hand velocity_p95 exceeds left-arm velocity_p95 in 197/301 episodes, with a paired ratio p50 of 1.230204; humanoid motion should therefore be profiled by arm/hand groups.
- The 86-episode embedding analysis covers 3,655 episode pairs. Similarity ranges from 0.796085 to 0.991590, indicating strong same-task visual homogeneity.
- At similarity 0.99 there are two duplicate-candidate groups involving four episodes; at 0.98 there are seven connected groups involving 61 episodes. These are review candidates, not automatic deletions.

## QuickStart

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest

# FR3 NPZ episode QC
python -m app.qc examples/ep_valid

# Inspect a LeRobot v3 dataset before selecting a profile
python -m tools.inspect_lerobot path/to/lerobot_dataset

# Low-dimensional QC, sampled image QC, motion profiling, and reports
python -m tools.qc_lerobot path/to/lerobot_dataset \
  --profile unitree_g1 \
  --episodes 0-5 \
  --camera-key observation.images.cam_left_wrist \
  --sample-fps 1 \
  --save-evidence \
  --output reports/unitree_g1

# Build a review queue and a local HTML reviewer
python -m tools.build_gold_set \
  --image-metrics reports/unitree_g1/image_metrics.parquet \
  --image-hits reports/unitree_g1/image_rule_hits.csv \
  --output reports/unitree_g1/gold_set.csv \
  --metadata reports/unitree_g1/gold_set_meta.json

python -m tools.build_gold_set_review \
  --gold-set reports/unitree_g1/gold_set.csv \
  --image-metrics reports/unitree_g1/image_metrics.parquet \
  --image-hits reports/unitree_g1/image_rule_hits.csv \
  --output-dir reports/unitree_g1/gold_set_review
```

After reviewing and exporting `gold_set_reviewed.csv`, evaluate it with `python -m tools.evaluate_gold_set` and calibrate thresholds with `python -m tools.calibrate_image_rules`.

## Documentation

- [P1.3 calibrated thresholds and coverage](docs/unitree_g1_p13_calibrated_thresholds_and_coverage.md)
- [P1.3 embedding coverage report](docs/unitree_embedding_0_85_p13_report.md)
- [P1.3 dataset profile report](docs/unitree_dataset_profile_p13_report.md)
- [P1 gold-set, embedding, and profiling report](docs/unitree_g1_p1_gold_embedding_profile.md)
- [P0 Unitree validation report](docs/unitree_g1_p0_validation.md)
- [Engineering and upgrade narrative](docs/qc_p0_upgrade_narrative.md)

## Scope And Limitations

Included:

- Offline and near-line QC; no robot control and no policy training.
- FR3 NPZ and LeRobot v3 input formats.
- Sampled image QC and human gold-set calibration.
- Public-dataset validation evidence.
- Release reports and severity-aware gates.

Not included:

- The original FR3 robot dataset or other proprietary data.
- Distributed processing, multi-node orchestration, or a production data platform.
- Full-image semantic understanding, automatic image repair, or automatic duplicate deletion.
- Full 301-episode image and embedding coverage; the current video subset covers 86/301 episodes.
- Claims of population recall or calibrated production deletion thresholds.

## Roadmap

1. Complete left-wrist image and embedding coverage for all 301 Unitree episodes.
2. Add rule-based task segmentation, grouped train/validation/test splitting, and leakage checks.
3. Add an idempotent batch runner with stage states, retries, resume, and processing-time monitoring.

## Technology Stack

Python, NumPy, OpenCV, PyArrow/Parquet, DuckDB, SQLite, FastAPI, pytest, Docker, and GitHub Actions.
