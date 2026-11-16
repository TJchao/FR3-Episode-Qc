# Unitree G1 Embedding Coverage Report

Date: 2026-11-16  
Dataset: `unitreerobotics/G1_Dex3_BlockStacking_Dataset`  
Method: `handcrafted_visual_v1`  
Camera: `observation.images.cam_left_wrist`

## Scope

The locally available left-wrist video file covers episodes 0-85. This report uses six uniformly sampled frames per episode.

```text
episodes analyzed: 86 / 301
frames analyzed: 516
embedding dimension: 216
k for kNN: 3
pairwise episode pairs: 3,655
```

The descriptor concatenates an HSV color histogram, a 4x4 DCT layout vector, an 8-bin gradient-orientation histogram, and a 64-bit pHash signature. It is intentionally lightweight and dependency-free; it is not a semantic CLIP or DINOv2 embedding.

## Pairwise Similarity

| Statistic | Cosine similarity |
|---|---:|
| min | 0.796085 |
| mean | 0.936630 |
| p50 | 0.942114 |
| p90 | 0.971497 |
| max | 0.991590 |

The high mean similarity is expected for a same-task, same-camera subset. It indicates limited visual diversity, but it is not by itself a data-quality defect.

## Relative Low-Density Tail

The lowest-density 10% tail contains nine episodes:

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

These episodes are candidates for coverage review. They are not confirmed data gaps.

## Duplicate-Candidate Groups

Groups are connected components over episode-embedding similarity edges. They are review candidates, not deletion decisions.

### Similarity threshold 0.99

| Group | Episodes | Size | Mean intra similarity |
|---|---|---:|---:|
| `t0.990_g001` | episode 42, episode 47 | 2 | 0.991590 |
| `t0.990_g002` | episode 64, episode 66 | 2 | 0.990552 |

```text
edges: 2
groups: 2
duplicate-candidate episodes: 4
```

### Similarity threshold 0.98

| Group | Size | Interpretation |
|---|---:|---|
| `t0.980_g001` | 2 | Close pair |
| `t0.980_g002` | 41 | Large chained same-view component |
| `t0.980_g003` | 2 | Close pair |
| `t0.980_g004` | 2 | Close pair |
| `t0.980_g005` | 2 | Close pair |
| `t0.980_g006` | 2 | Close pair |
| `t0.980_g007` | 10 | Later-episode connected component |

```text
edges: 98
groups: 7
duplicate-candidate episodes: 61
```

The 41-member component does not imply that all 41 episodes are interchangeable. Local similarities can chain through a dense region. The actionable interpretation is visual homogeneity and possible redundancy; a stratified sample or new-viewpoint acquisition should be considered after human review.

### Similarity threshold 0.995

```text
edges: 0
groups: 0
duplicate-candidate episodes: 0
```

## Boundary

This report does not claim semantic coverage, population recall, or a calibrated duplicate-deletion threshold. Image QC and embedding coverage both remain 86/301 episodes. The remaining 215 episodes require the other three left-wrist video files.
