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
