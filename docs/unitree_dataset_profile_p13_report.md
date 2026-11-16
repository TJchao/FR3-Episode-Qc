# Unitree G1 Dataset Profile Report

Date: 2026-11-16  
Dataset: `unitreerobotics/G1_Dex3_BlockStacking_Dataset`  
Robot: Unitree G1  
LeRobot codebase version: v3.0  
Checkpoint-independent data profile

## Coverage

```text
low-dimensional profile: 301 / 301 episodes
sampled image QC: 86 / 301 episodes
embedding coverage: 86 / 301 episodes
human gold set: 120 reviewed rows
```

## Low-Dimensional Quality

The profile executed six rules across all 301 episodes. All 301 passed the executable rules.

Four rules were skipped because the source format does not provide their required fields:

| Skipped rule | Missing field |
|---|---|
| `camera_age_p95` | `camera_age_ms` |
| `robot_time_monotonic` | `robot_time_ns` |
| `rgb_frame_count` | `rgb_frame_count` |
| `depth_frame_count` | `depth_frame_count` |

Therefore, `301/301 passed` means that executable rules passed. It does not mean that missing dimensions were implicitly validated.

Frame and duration distributions:

| Statistic | Frames | Duration |
|---|---:|---:|
| min | 666 | 22.2 s |
| mean | 934.20598 | 31.140199 s |
| p50 | 892 | 29.733333 s |
| p95 | 1131 | 37.7 s |
| max | 6791 | 226.366667 s |

The max-to-p95 frame-count ratio is 6.00 and max-to-min is 10.20. This long tail is a distribution finding; it is not a schema failure.

## Image Quality Review

The calibrated policy uses blur P2 and static pHash Hamming 2 as primary review thresholds. P5 and Hamming 4 remain broad sensitivity bands.

| Rule | Baseline hits | Calibrated hits |
|---|---:|---:|
| `image_blur` | 175 | 86 |
| `image_freeze` | 121 | 121 |
| `image_static_candidate` | 684 | 338 |
| total | 980 | 545 |

Unique flagged frames fell from 855 to 423. The review-candidate rate fell from 34.93% to 19.42% of sampled frames.

The 120-row gold set produced:

```text
review completion: 100%
overall precision: 53.3333%
weighted FPR estimate: 6.6404%
recall: null
F1: null
```

Recall and F1 remain null because the balanced hit/control design does not cover all false negatives.

## Motion Findings

The state unit is not declared by the public dataset, so motion is reported as `unknown/s`, `unknown/s^2`, and `unknown/s^3`.

Across 301 episodes, right-hand velocity_p95 exceeds left-arm velocity_p95 in 197 episodes. The paired right-hand/left-arm ratio has a p50 of 1.230204. This comparison uses the same episode for both groups and supports separate arm/hand profiling.

At 30 Hz, third-order finite differencing has a theoretical noise gain of 27,000. No verified static segments or sensor-noise model are available, so `jerk_noise_floor` is null. Absolute jerk is used only for within-dataset relative ranking.

## Embedding Summary

The 86-episode handcrafted embedding analysis found:

```text
pairwise similarity range: 0.796085 - 0.991590
relative low-density tail: 9 episodes
0.99 duplicate groups: 2 groups, 4 episodes
0.98 duplicate groups: 7 groups, 61 episodes
```

These groups are review candidates, not deletion decisions.

## Boundary

This profile does not claim production-platform operation, distributed processing, semantic image understanding, or full-dataset image coverage. Its purpose is to expose executable-rule coverage, review workload, data-distribution findings, and the next collection priorities.
