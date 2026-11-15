# Unitree G1 P0 Validation Report

Date: 2026-11-15  
Tool commit under validation: `ce7cf1e`  
Dataset: [`unitreerobotics/G1_Dex3_BlockStacking_Dataset`](https://huggingface.co/datasets/unitreerobotics/G1_Dex3_BlockStacking_Dataset)  
License: Apache-2.0  

## Scope

This validation used a minimal public subset rather than redistributing the dataset:

```text
meta/info.json
meta/stats.json
meta/episodes/chunk-000/file-000.parquet
data/chunk-000/file-000.parquet
videos/observation.images.cam_left_wrist/chunk-000/file-000.mp4
```

No Unitree video or Parquet data is committed to this repository. Only local generated reports and evidence frames are retained outside Git.

## Dataset Schema

`tools.inspect_lerobot` reported:

```text
total_episodes: 301
total_frames: 281196
fps: 30.0
observation.state shape: [28]
action shape: [28]
data columns:
  observation.state
  action
  timestamp
  frame_index
  episode_index
  index
  task_index
```

The dataset also declares four video streams:

```text
observation.images.cam_left_high
observation.images.cam_left_wrist
observation.images.cam_right_high
observation.images.cam_right_wrist
```

This validation used only `observation.images.cam_left_wrist`.

## Low-dimensional QC

Command:

```bash
python -m tools.qc_lerobot .tmp/unitree \
  --profile unitree_g1 \
  --episodes 0-5 \
  --output .tmp/reports/unitree_g1_p0 \
  --dataset-version unitree-g1-dex3-block-stacking-subset \
  --git-commit ce7cf1e
```

Result:

```text
checked episodes: 6
passed: 6
failed: 0
severity: none for all 6 episodes
frame count range: 962-1196
mean frame count: 1096.67
```

Checks covered:

```text
timestamp strict monotonicity
NaN / Inf
state shape
action shape
episode length consistency
```

`camera_age_ms` is not present in this LeRobot dataset, so that check was intentionally skipped rather than fabricating the field.

## Motion Profile

For episode 0:

```text
frame_count: 1173
valid_frame_count: 1173
velocity_p50: 0.6080
velocity_p95: 3.2032
velocity_p99: 5.9684
acceleration_p95: 28.6540
jerk_p95: 1139.7160
static_ratio: 0.1007
motion_range: 25.8458
active_dimension_ratio: 1.0
```

Motion values are descriptive only. They are not marked as errors because the 28 humanoid state dimensions mix semantic types and units that are not declared in the public metadata.

## Sampled Video QC

Command:

```bash
python -m tools.qc_lerobot .tmp/unitree \
  --profile unitree_g1 \
  --episodes 0 \
  --camera-key observation.images.cam_left_wrist \
  --sample-fps 1 \
  --save-evidence \
  --output .tmp/reports/unitree_g1_ep0_video_evidence \
  --dataset-version unitree-g1-dex3-block-stacking-ep0 \
  --git-commit ce7cf1e
```

Result:

```text
sampled frames: 40
issue count: 11
overall severity: repairable
```

Issue distribution:

| Rule | Count | Severity | Interpretation |
|---|---:|---|---|
| `image_blur` | 2 | warning | Relative lowest-tail blur candidates |
| `image_duplicate` | 7 | warning | Consecutive sampled pHash near-duplicates |
| `image_freeze` | 2 | repairable | Near-identical images while state motion exceeds threshold |

Candidate frames:

```text
image_blur:      540, 870
image_duplicate: 60, 90, 810, 1020, 1050, 1110, 1170
image_freeze:    60, 90
```

All 40 sampled evidence frames were saved locally for review. `image_review.csv` links each issue to its evidence frame and provides empty human-review fields:

```text
human_label
is_true_positive
note
```

## Important Boundary

The blur and duplicate rules are relative rules. They intentionally select the low sharpness tail and highly similar sampled pairs within this episode; they do not yet claim a production false-positive rate. The freeze rule uses both image similarity and state motion, but frames 60 and 90 still require human review because a wrist camera can remain visually similar while other parts of the humanoid move.

No image frame is repaired and no image is committed to this repository.

## Release Gate

The six-episode low-dimensional subset passed `--fail-on fatal`. The video episode is classified `repairable`, so a stricter production gate can block it with:

```bash
python -m app.release gate \
  .tmp/reports/unitree_g1_ep0_video_evidence/qc_results.parquet \
  --fail-on fatal \
  --fail-on repairable
```

This distinction is deliberate: fatal errors block immediately, while relative-rule candidates require review before becoming release policy.
