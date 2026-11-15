# Unitree G1 P0 Validation Report

Date: 2026-11-15  
Tool commits under validation: `ce7cf1e` plus the P0.1 correction branch
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
codebase_version: v3.0
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
velocity_p50: 0.6080 unknown/s
velocity_p95: 3.2032 unknown/s
velocity_p99: 5.9684 unknown/s
acceleration_p95: 28.6540 unknown/s^2
jerk_p95: 1139.7160 unknown/s^3
relative_static_tail_ratio: 0.1007
motion_range: 25.8458 unknown
active_dimension_ratio: 1.0
groups: left_arm, right_arm, left_hand, right_hand
```

Motion values are descriptive only. They are not marked as errors because the dataset does not declare a physical state unit. Values are therefore reported in `unknown/s`, `unknown/s^2`, and `unknown/s^3`; they are not claimed to be radians. Arm and finger groups are reported separately because their magnitudes and scales can differ.

## Sampled Video QC

Result:

```text
sampled frames: 40
image rule hits: 11
unique flagged frames: 9
overall severity: repairable
```

Rule-hit distribution:

| Rule | Hits | Severity | Interpretation |
|---|---:|---|---|
| `image_blur` | 2 | warning | Relative lowest-tail rule hits; P5 over 40 sampled frames |
| `image_static_candidate` | 7 | warning | Consecutive sampled frames are visually static |
| `image_freeze` | 2 | repairable | Visual stasis while relative state motion exceeds threshold |

Candidate frames:

```text
image_blur:              540, 870
image_static_candidate: 60, 90, 810, 1020, 1050, 1110, 1170
image_freeze:            60, 90
```

The 11 hits involve 9 unique frame indices because the two freeze candidates at frames 60 and 90 overlap with static candidates. These are rule hits and review candidates, not confirmed defects.

Adjacent-frame pHash threshold sweep:

| Hamming threshold | Hit count |
|---:|---:|
| 2 | 2 |
| 4 | 7 |
| 6 | 8 |

All 40 sampled evidence frames were saved locally for review. `image_review.csv` links each hit to its evidence frame and provides empty human-review fields:

```text
human_label
is_true_positive
note
```

## Important Boundary

The blur and static-candidate rules are relative rules. They intentionally select the low sharpness tail and visually similar adjacent sampled pairs within this episode; they do not claim a production false-positive rate. The static-candidate rule is not cross-episode duplicate detection. The freeze rule uses both image similarity and state motion, but frames 60 and 90 still require human review because a wrist camera can remain visually similar while other parts of the humanoid move.

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
