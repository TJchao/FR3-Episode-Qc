# Unitree G1 P1 Gold Set, Embedding, and Dataset Profile

Date: 2026-11-15  
Dataset: `unitreerobotics/G1_Dex3_BlockStacking_Dataset`  
License: Apache-2.0  
Branch: `codex/qc-p0-lerobot-image`  

## Scope

This P1 pass adds three capabilities without committing the public dataset:

```text
1. Gold-set template and metric evaluation
2. Dependency-free visual embedding coverage
3. Batch-level dataset profile
```

## 1. Gold Set

Image QC was run over episodes 0-9 using the left-wrist camera at 1 fps.

```text
episodes: 10
sampled frames: 363
rule hits: 148
unique flagged frames: 124
```

Rule-hit distribution:

| Rule | Hits |
|---|---:|
| `image_blur` | 21 |
| `image_freeze` | 23 |
| `image_static_candidate` | 104 |

The review template contains 124 rows:

```text
21 blur hits + 20 blur controls
23 freeze hits + 20 freeze controls
20 static-candidate hits + 20 static-candidate controls
```

Rules with fewer than 30 auto hits use a hit census. Therefore all 21 blur hits and all 23 freeze hits are included. The 104 static-candidate hits use round-robin sampling across episodes. Controls are also round-robin sampled. A reviewer fills `human_label` or `is_true_positive`; until then, precision and the FPR estimate remain null.

Policy-based automation coverage, where every auto hit requires human review:

| Rule | Review candidates / sampled frames | Automation coverage |
|---|---:|---:|
| `image_blur` | 21 / 363 | 94.2149% |
| `image_freeze` | 23 / 363 | 93.6639% |
| `image_static_candidate` | 104 / 363 | 71.3499% |

This automation coverage is a review-work metric, not an accuracy claim.

Metric boundary:

```text
Precision: measurable after hit review
FPR: estimated from reviewed controls
Recall: null
F1: null
Reason: sampling design does not cover false negatives
```

## 2. Handcrafted Embedding Coverage

Method:

```text
handcrafted_visual_v1
```

Each episode embedding concatenates:

```text
HSV color histogram
4x4 low-frequency DCT layout
8-bin gradient orientation histogram
64-bit pHash signature
```

The final descriptor is 216-D and L2-normalized. Episodes 0-9 were represented by six uniformly sampled left-wrist frames each.

```text
episodes: 10
frames: 60
embedding dimension: 216
k for kNN: 3
near-duplicate similarity threshold: 0.99
```

Result:

```text
relative low-density tail: episode_000004
near-duplicate candidate pairs at 0.99: 0
pairwise cosine similarity range: 0.856672 - 0.981875
closest pair: episode_000001 and episode_000002 at 0.981875
margin below duplicate threshold: 0.008125
```

`episode_000004` is only the relative low-density tail of this ten-episode subset. It is not a confirmed data gap. The 0.99 threshold is uncalibrated, and zero candidates is not proof that no duplicates exist. Across 45 episode pairs, similarity ranges from 0.856672 to 0.981875, with mean 0.931309. The closest pair misses the threshold by only 0.008125. This suggests that the same-task ten-episode subset is visually homogeneous and that the duplicate threshold needs calibration by reviewed pairs. It does not yet prove that all 301 episodes have low diversity.

PCA is emitted only as an SVG visualization. Coverage density is computed in the original 216-D space.

## 3. Dataset Profile

Low-dimensional profiling covered all 301 episodes:

```text
selected episodes: 301
passed: 301
failed: 0
total frames: 281196
fps: 30
state/action dimension: 28
```

Rule coverage:

```text
rules evaluated: 6
rules skipped: 4
```

Skipped rules:

| Rule | Reason |
|---|---|
| `camera_age_p95` | missing field `camera_age_ms` |
| `robot_time_monotonic` | missing field `robot_time_ns` |
| `rgb_frame_count` | missing field `rgb_frame_count` |
| `depth_frame_count` | missing field `depth_frame_count` |

Therefore `301/301 passed` means that all executable low-dimensional rules passed. Missing fields are skipped, not implicitly passed.

Analysis coverage:

```text
low-dimensional profile: 301 / 301 episodes
sampled image QC: 10 / 301 episodes
embedding coverage: 10 / 301 episodes
gold-set review rows: 124
```

Frame-count distribution:

| Statistic | Frames | Duration |
|---|---:|---:|
| min | 666 | 22.2 s |
| mean | 934.21 | 31.14 s |
| p50 | 892 | 29.73 s |
| p95 | 1131 | 37.7 s |
| max | 6791 | 226.37 s |

All-dimension motion summary, in unknown state units:

| Metric | Mean | P95 | Max |
|---|---:|---:|---:|
| velocity_p95 | 2.7805 | 3.1710 | 10.9396 |
| acceleration_p95 | 23.6088 | 28.5250 | 111.1336 |
| jerk_p95 | 903.3211 | 1071.5720 | 4057.0076 |

Arm and hand groups are also reported separately. The dataset does not declare the physical state unit, so the report uses `unknown/s`, `unknown/s^2`, and `unknown/s^3`.

### Finding 1: Arm and Hand Motion Need Separate Profiles

Using the same episode as the pairing unit, `right_hand.velocity_p95` exceeds `left_arm.velocity_p95` in 197 of 301 episodes. The paired right-hand/left-arm ratio has a p50 of 1.230204. This is more valid than comparing the global left-arm maximum with the global right-hand maximum, which can come from different episodes.

The result still supports group-specific profiling: arm and finger branches have different roles and scales, and a single 28-dimensional aggregate can hide branch-specific behavior.

### Finding 2: Rule Pass Does Not Mean Distribution Health

Frame count has a clear long tail:

```text
min: 666
mean: 934.20598
p95: 1131
max: 6791
max / mean: 7.27
max / p95: 6.00
max / min: 10.20
```

All executable format rules pass, but the duration distribution is not homogeneous. This is why a profile layer is needed in addition to pass/fail schema QC.

### Finding 3: Jerk Has a Noise Boundary

At 30 Hz, third-order finite differencing has a theoretical noise gain of `30^3 = 27000`. The dataset provides no verified static segments and no sensor noise model, so `jerk_noise_floor` is null.

Absolute jerk values are therefore not comparable across datasets. Jerk is used only for relative ranking within this dataset and preprocessing configuration.

## 4. Boundary

This P1 profile does not claim:

```text
human-verified image precision or recall
semantic CLIP/DINOv2 coverage
calibrated duplicate thresholds
absolute visual blind-spot detection
```

It does provide reproducible review queues, original-space kNN density, cross-episode duplicate candidates, and a batch-level summary suitable for gold-set calibration and later model-based embedding upgrades.
