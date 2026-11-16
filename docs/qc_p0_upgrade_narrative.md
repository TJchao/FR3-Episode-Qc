# QC P0 Upgrade Narrative

Date: 2026-11-15

## Summary

I extended the FR3 episode QC tool from a single-robot format checker into a profile-based embodied-data checker that can inspect LeRobot datasets, split selected episodes by `episode_index`, profile multi-dimensional humanoid state, sample one camera stream, and run lightweight image rules without breaking the original FR3 workflow.

## Main Upgrades

```text
Robot profile
LeRobot inspect / episode reader
Low-dimensional schema and temporal QC
Sampled video decoding
Black / white frame detection
Exposure and relative blur checks
pHash near-duplicate detection
Image-freeze + state-motion dual check
Motion statistics
QC CSV / Parquet reports
Release-gate-compatible severity output
```

## Design Decisions

### LeRobot is treated as a dataset

The existing `app.qc.check_episode()` interface remains unchanged for FR3 NPZ episodes. LeRobot uses `meta/info.json`, episode metadata Parquet, data Parquet, and per-camera video files, so it gets separate `tools.inspect_lerobot` and `tools.qc_lerobot` entry points. This avoids forcing fake `camera_age_ms` or `robot_time_ns` fields into third-party data.

### Schema inspection precedes profile creation

The tool does not assume that every Unitree G1 dataset has a fixed 28-D state and action. `inspect_lerobot` reads the actual feature shapes, Parquet columns, and video keys. `unitree_g1.json` uses null dimensions by default and can be filled from inspected data.

### Motion is profiled before it is judged

Humanoid state streams can mix joint angles, finger joints, base pose, IMU values, and force signals. These dimensions do not share one physical scale. P0 reports velocity, acceleration, jerk, motion range, static ratio, and active-dimension ratio, but does not classify motion outliers until a profile declares semantic indices and units.

### Freeze needs image and state evidence

A static image is not necessarily a frozen camera. The rule only becomes a repairable freeze candidate when consecutive sampled frames are near-identical while robot-state motion exceeds a relative threshold.

```text
image unchanged + state moving => freeze candidate
image unchanged + state static => possibly valid static segment
```

### Image review is part of the output

Black and white frames have clear absolute meanings. Relative blur and pHash duplicate rules always select a tail from the current batch and require review before becoming production thresholds. The tool emits `image_review.csv` with `human_label`, `is_true_positive`, and `note` fields for calibration.

## Real Issues Encountered

### 1. PyArrow `is_in` requires an Arrow value set

Initial code passed a Python list to `pyarrow.compute.is_in`. PyArrow rejected it with:

```text
TypeError: "[0, 1]" is not a valid value set
```

Fix: build `pa.array(selected, type=pa.int64())` and cast the episode column before filtering.

### 2. Motion statistics can become empty after NaN filtering

Filtering non-finite rows can leave too few samples for jerk. Calling `np.percentile` on an empty array raised an index error. The fix returns `None` for a statistic when its input array is empty.

### 3. Video decoding depends on the local codec build

OpenCV reads MP4 in the local environment, but a future AV1 file may fail depending on the wheel build. The reader therefore has an ffmpeg CLI fallback and records `image_decoder_unavailable` when no decoder is available. For ffmpeg raw output, dimensions come from the LeRobot feature shape when available.

### 4. A black synthetic video intentionally triggers multiple rules

An all-black video triggers black-frame, exposure, blur, duplicate, and freeze rules. This is expected: black-frame is fatal, while the others are warning or repairable. The important regression is that overall severity is fatal and the existing release gate blocks it.

### 5. Local sandbox blocked the default pytest temporary directory

The managed execution environment denied access to the default Windows pytest temp root. Tests were run with:

```bash
pytest -q -p no:cacheprovider --basetemp .tmp/pytest
```

The parent `.tmp` directory must exist. This did not require changing test semantics.

### 6. Public dataset access required an explicit terminal proxy

The browser could reach GitHub and Hugging Face, but command-line tools initially bypassed the local proxy. Git and Python downloads were made explicit through `127.0.0.1:7890`. The official Unitree G1 Dex3 block-stacking subset was then downloaded and validated.

The real dataset was not committed; only the sanitized validation report is stored in `docs/unitree_g1_p0_validation.md`.

### P0.1 correction: separate rule hits from confirmed defects

The first real-video report used the phrase "11 image issues". That wording was too strong. The P5 blur rule necessarily selects a tail from the sampled batch, and adjacent pHash similarity identifies visually static intervals rather than cross-episode duplicates.

The corrected report therefore uses:

```text
image_rule_hits = 11
unique_flagged_frames = 9
image_static_candidate instead of image_duplicate
relative_static_tail_ratio instead of static_ratio
```

It also reports pHash Hamming thresholds 2, 4, and 6. On episode 0, the hit count changes from 2 to 7 to 8. This is threshold-calibration evidence, not a claim about true duplicates.

### P0.1 correction: do not call adjacent similarity cross-episode duplicate detection

The original rule ID `image_duplicate` was misleading. It only compares adjacent frames in the sampled sequence, typically one second apart:

```text
frame[t] vs frame[t+1]
```

Therefore seven hits mean "seven adjacent sampled pairs are visually static", not "seven episodes are duplicates". The rule was renamed to `image_static_candidate`. Cross-episode duplicate detection must compare episode-level representative frames or embeddings and remains future work.

### P0.1 correction: report relative rules by construction

The blur rule uses the fifth percentile of the current episode's sampled Laplacian scores. With 40 samples, two hits are expected by construction. They are review candidates, not confirmed blur defects.

The corrected summary therefore reports:

```text
rule hits
unique flagged frames
per-rule hit counts
rule interpretation
```

This distinction prevents a reviewer from mistaking a percentile artifact for a measured defect rate.

### P0.1 correction: split Unitree motion groups and preserve unit uncertainty

The public dataset provides joint names but does not explicitly declare the physical unit. The report therefore uses:

```text
state_unit: unknown
velocity_unit: unknown/s
acceleration_unit: unknown/s^2
jerk_unit: unknown/s^3
```

It does not claim radians. Motion is also split into:

```text
left arm: indices 0-6
right arm: indices 7-13
left hand: indices 14-20
right hand: indices 21-27
```

The all-dimension aggregate remains visible, but group statistics prevent arm motion from hiding or dominating finger behavior.

### P0.1 correction: rename the relative static metric

The old `static_ratio` compared frame motion with the episode's tenth percentile. By construction it selects approximately the lowest-motion tail, so calling it a physical static ratio was misleading. It is now `relative_static_tail_ratio`.

A physical static ratio requires a profile-declared threshold in known state units.

### P0.1 correction: derive LeRobot schema and validate declared length

Two reader contract issues were fixed:

```text
1. schema_version now comes from meta/info.json codebase_version
2. actual data-row count is compared with meta/episodes.length
```

The first implementation synthesized `frame_count` from the data rows and then compared metadata with itself, so it could not detect an episode-length mismatch. The corrected reader preserves the declared length as `expected_frame_count` and reports `lerobot_length_mismatch` as fatal.

### P0.1 correction: make threshold behavior inspectable

The adjacent-frame pHash threshold is now configurable through the robot profile and CLI. The report includes a 2/4/6 Hamming-threshold sweep:

```text
threshold 2: 2 hits
threshold 4: 7 hits
threshold 6: 8 hits
```

The default threshold 4 is provisional until a human-reviewed gold set measures false positives and misses.

### P0.1 performance fixes

Two performance issues were addressed before moving to full-frame analysis:

```text
1. freeze motion comparisons are precomputed once, removing repeated O(n²) work
2. VideoImageSource reuses an OpenCV capture for monotonically increasing seeks
```

The source still falls back to ffmpeg and reopens the capture when seeking backward or after a decode failure. This avoids reopening the 497 MB AV1 file for every sampled frame while preserving robust decoding behavior.

### P0.1 regression coverage

The P0.1 test suite now covers:

```text
schema version inspection
declared episode length mismatch
adjacent-frame static-candidate naming
rule-hit versus unique-frame counts
pHash threshold sweep
Unitree arm/hand motion grouping
unknown unit labels
monotonic video-source reuse
CLI release gate blocking a fatal black-frame episode
```

Current verification: 77 tests passed.

## P1 extension: gold set, embedding, and profile

After P0.1, three batch-analysis capabilities were added.

### Gold set without fabricating labels

The tool generates a balanced review queue from auto hits and deterministic controls. The initial 10-episode pilot produced 363 frames and 148 rule hits. It was then expanded to all 86 episodes covered by the locally available left-wrist video file, producing 2806 frames and 980 rule hits. The expanded gold-set template contains 120 rows: 20 round-robin hit samples and 20 controls for each rule.

All 120 rows were reviewed, followed by an 11-row second pass for objectively ambiguous labels. Final reviewed precision is 53.3333% overall, with a case-control weighted FPR estimate of 6.6404%. Per-rule precision is 40% for blur, 70% for freeze, and 50% for static candidates. Recall and F1 remain null because the sampling design does not cover all false negatives.

Threshold calibration found higher sample precision at blur P2 than P5 or P10, and at static pHash Hamming 2 than Hamming 4. These are review-priority recommendations, not population recall claims. Policy-based automation coverage remains:

```text
blur: 93.7634%
freeze: 95.6878%
static candidate: 75.6237%
```

### Dependency-free embedding coverage

Installing CLIP or DINOv2 would add a large model dependency before the review loop was trustworthy. The P1 implementation therefore uses `handcrafted_visual_v1`: HSV histogram, low-frequency DCT, gradient orientation, and pHash information in a 216-D normalized descriptor.

For episodes 0-9, episode 4 was the relative low-density tail and no pair exceeded the uncalibrated 0.99 near-duplicate threshold. The report explicitly says that this is not semantic coverage and not proof of no duplicates.
The pairwise similarity range is 0.856672-0.981875; the closest pair misses the threshold by 0.008125, indicating visual homogeneity in this same-task subset and the need for threshold calibration.

### Batch profile over all 301 episodes

A PyArrow streaming path was added so all 301 episodes / 281196 frames can be profiled without converting every row to Python. The full low-dimensional profile completed in about 2.7 seconds and reported 301/301 passed. The output combines quality, frame and duration distributions, grouped motion statistics, image metrics, embedding coverage, and gold-set review status.
The report explicitly distinguishes executed rules from skipped rules. Six low-dimensional rules were executable and four were skipped because the source format lacked `camera_age_ms`, `robot_time_ns`, `rgb_frame_count`, and `depth_frame_count`. Thus 301/301 passed means executable rules passed, not that missing dimensions were implicitly validated.

### P1.3: apply the calibration and expand visual coverage

The reviewed thresholds were applied to the full locally available 86-episode image subset. Rule hits fell from 980 to 545, and unique flagged frames fell from 855 to 423. Static high-confidence Hamming-2 hits were 338; the broad Hamming-4 band had 684 hits. This converts the gold-set finding into a measurable review-work reduction.

Visual coverage was expanded from 10 to 86 episodes using six sampled frames per episode. The 3,655 pairwise similarities ranged from 0.796085 to 0.991590. Connected duplicate grouping found two 0.99 groups involving four episodes, and a broad 0.98 graph with seven components covering 61 episodes. These are redundancy and homogeneity signals, not automatic deletion decisions.

### 7. OpenCV image writing failed on a non-ASCII Windows path

`cv2.imwrite` failed when the temporary path contained Chinese characters. The fix uses `cv2.imencode` followed by `Path.write_bytes`, which avoids OpenCV's path encoding limitation while preserving the same JPEG output.

## Verification

```text
74 tests passed
Original 57 tests remained green
Synthetic LeRobot dataset inspection passed
Two-episode low-dimensional QC passed
MP4 single-frame seek passed
Black-frame episode produced fatal severity
Release gate blocked the fatal episode
Motion profile JSON was generated
Official Unitree G1 dataset schema inspection passed
Six real Unitree G1 episodes passed low-dimensional QC
One real Unitree G1 wrist-camera episode was sampled at 1 fps
40 evidence frames and 11 image-rule candidates were recorded
```

## Interview Phrasing

> I kept the original FR3 QC contract unchanged and added a separate LeRobot dataset path. First, an inspector reads the real `info.json`, Parquet columns, feature shapes, and video keys instead of guessing a humanoid schema. Then selected episodes are grouped by `episode_index`, state/action dimensions and timestamps are checked, motion is profiled without premature error classification, and one camera stream is sampled. Image rules cover absolute black/white frames, exposure, relative blur, pHash near-duplicates, and freeze candidates based on both image similarity and state motion. Results remain compatible with the existing severity and release-gate schema, so a fatal black frame blocks release. Relative image rules also emit a review CSV because percentile-based rules need human calibration before I claim a production false-positive rate.
