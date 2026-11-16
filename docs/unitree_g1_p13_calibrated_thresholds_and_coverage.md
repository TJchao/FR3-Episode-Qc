# Unitree G1 P1.3 Calibrated Thresholds and Coverage

Date: 2026-11-16  
Dataset: `unitreerobotics/G1_Dex3_BlockStacking_Dataset`  
License: Apache-2.0  
Branch: `codex/qc-p0-lerobot-image`

## Scope

P1.3 applies the 120-row human gold-set result to the available 86-episode left-wrist subset and expands visual coverage analysis.

```text
1. Apply calibrated blur and static thresholds
2. Re-run image QC over episodes 0-85
3. Expand handcrafted visual embeddings from 10 to 86 episodes
4. Build cross-episode duplicate-candidate groups using connected components
```

No P2/P3 features were added in this pass.

## 1. Calibrated Threshold Policy

The Unitree profile now uses:

```text
blur primary: P2
blur broad observation band: P5
static primary: pHash Hamming <= 2
static broad observation band: Hamming <= 4
freeze: unchanged
```

Policy source:

```text
120-row human gold set, 2026-11-16
```

On the reviewed gold-set sample:

| Rule / threshold | Sample precision |
|---|---:|
| blur P2 | 55.5556% |
| blur P5 | 40.0000% |
| static Hamming 2 | 75.0000% |
| static Hamming 4 | 50.0000% |

Primary thresholds drive review hits. Broad thresholds are sensitivity-only and do not by themselves imply a defect.

## 2. Image QC: Before and After

Both runs use:

```text
episodes: 0-85
camera: observation.images.cam_left_wrist
sampling: 1 fps
sampled frames: 2806
```

| Metric | P5 / Hamming 4 baseline | P2 / Hamming 2 calibrated | Change |
|---|---:|---:|---:|
| blur hits | 175 | 86 | -50.9% |
| freeze hits | 121 | 121 | 0.0% |
| static hits | 684 | 338 | -50.6% |
| total rule hits | 980 | 545 | -44.4% |
| unique flagged frames | 855 | 423 | -50.5% |
| review-candidate rate | 34.93% | 19.42% | -15.51 percentage points |

Static bands:

| Band | Hits |
|---|---:|
| High confidence, Hamming <= 2 | 338 |
| Broad-only, Hamming 3-4 | 346 |
| Broad total, Hamming <= 4 | 684 |

Blur bands:

| Band | Hits |
|---|---:|
| Primary, P2 | 86 |
| Broad, P5 | 175 |

The calibrated run still reports all 86 episodes as failed because every episode has at least one relative-rule candidate. This is a review-policy result, not a final dataset-quality failure rate.

## 3. Visual Embedding Coverage

Method:

```text
handcrafted_visual_v1
```

Coverage was expanded from 10 to all 86 locally available episodes:

```text
episodes: 86
frames per episode: 6
sampled frames: 516
embedding dimension: 216
k for kNN: 3
```

Pairwise cosine similarity over 3,655 episode pairs:

| Statistic | Similarity |
|---|---:|
| min | 0.796085 |
| mean | 0.936630 |
| p50 | 0.942114 |
| p90 | 0.971497 |
| max | 0.991590 |

Relative low-density tail:

```text
episode_000003
episode_000004
episode_000006
episode_000024
episode_000026
episode_000040
episode_000051
episode_000061
episode_000063
```

This is the lowest-density 10% tail of the 86-episode subset. It is not a confirmed data gap.

## 4. Cross-episode Duplicate Groups

Duplicate groups are connected components over episode-embedding similarity edges. A group can contain chained near-neighbors, so its interpretation is stricter than an isolated pair.

### High-confidence threshold 0.99

| Group | Members | Size | Mean intra similarity |
|---|---|---:|---:|
| `t0.990_g001` | episode 42, episode 47 | 2 | 0.991590 |
| `t0.990_g002` | episode 64, episode 66 | 2 | 0.990552 |

Result:

```text
groups: 2
duplicate-candidate episodes: 4
edges: 2
```

### Broad threshold 0.98

| Group | Size | Interpretation |
|---|---:|---|
| `t0.980_g001` | 2 | Close pair |
| `t0.980_g002` | 41 | Large chained same-view cluster |
| `t0.980_g003` | 2 | Close pair |
| `t0.980_g004` | 2 | Close pair |
| `t0.980_g005` | 2 | Close pair |
| `t0.980_g006` | 2 | Close pair |
| `t0.980_g007` | 10 | Later-episode connected cluster |

Result:

```text
groups: 7
duplicate-candidate episodes: 61
edges: 98
```

The 41-member component at 0.98 does not mean 41 episodes are interchangeable. It means a chain of local similarities connects them, which is expected for a same-task, same-camera subset. The useful interpretation is visual homogeneity and potential redundancy, not automatic deletion.

### Sensitivity 0.995

```text
groups: 0
edges: 0
```

## 5. Boundary

This P1.3 report does not claim:

```text
population recall
semantic CLIP/DINOv2 coverage
calibrated duplicate-deletion decisions
full 301-episode image coverage
```

Current coverage:

```text
low-dimensional profile: 301 / 301 episodes
image QC: 86 / 301 episodes
embedding coverage: 86 / 301 episodes
gold set: 120 reviewed rows
```

The duplicate threshold remains uncalibrated. Groups at 0.98 and 0.99 are review candidates. Human review should decide whether to keep all episodes, stratify-sample the large cluster, or target new viewpoints and object configurations.
