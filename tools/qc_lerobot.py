"""Run low-dimensional, sampled-image, and motion QC on LeRobot episodes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from app.errors import classify_error, highest_severity, severity_for_error
from app.image_rules import (
    evaluate_freeze,
    evaluate_image_metrics,
    gray,
    hamming_distance,
    image_metrics,
)
from app.motion_profile import build_motion_profile
from app.profile import ProfileError, load_profile
from app.readers.lerobot import (
    discover_layout,
    inspect_layout,
    iter_selected_episode_arrays,
    parse_episode_selection,
    video_offset_for_episode,
    video_path_for_episode,
)
from app.readers.video import VideoImageSource


REPORT_FIELDS = (
    "episode_id",
    "status",
    "severity",
    "error_types",
    "error_count",
    "camera_age_p95_ms",
    "frame_count",
    "episode_hash",
    "dataset_version",
    "git_commit",
    "created_at",
)


def _episode_hash(episode_index: int, arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    digest.update(f"episode-{episode_index}".encode("utf-8"))
    for name in sorted(arrays):
        digest.update(name.encode("utf-8"))
        digest.update(np.asarray(arrays[name], dtype=float).tobytes())
    return f"sha256:{digest.hexdigest()}"


def _validate_low_dim(
    arrays: dict[str, np.ndarray],
    state_dim: int | None,
    action_dim: int | None,
    expected_frame_count: int | None = None,
) -> list[str]:
    errors: list[str] = []
    required = ("timestamp_s", "state", "action")
    for field in required:
        if field not in arrays:
            errors.append(f"LeRobot feature is missing: {field}")
    if errors:
        return errors

    actual_frame_count = int(arrays["timestamp_s"].shape[0])
    if (
        expected_frame_count is not None
        and actual_frame_count != expected_frame_count
    ):
        errors.append(
            "LeRobot episode length mismatch: "
            f"metadata={expected_frame_count}, data={actual_frame_count}"
        )

    timestamp = np.asarray(arrays["timestamp_s"], dtype=float)
    if timestamp.ndim != 1 or not np.all(np.diff(timestamp) > 0):
        errors.append("timestamp_s is not strictly increasing")

    state = np.asarray(arrays["state"], dtype=float)
    action = np.asarray(arrays["action"], dtype=float)
    for name, values, expected_dim in (
        ("state", state, state_dim),
        ("action", action, action_dim),
    ):
        if values.ndim != 2:
            errors.append(f"{name} shape must be 2-D")
        elif expected_dim is not None and values.shape[1] != expected_dim:
            errors.append(
                f"{name} shape must be ({len(values)}, {expected_dim}), got {values.shape}"
            )
        if not np.all(np.isfinite(values)):
            errors.append(f"{name} contains NaN or Inf")
    return errors


def _sample_indices(frame_count: int, fps: float | None, sample_fps: float) -> list[int]:
    if frame_count <= 0:
        return []
    effective_fps = float(fps or 1.0)
    step = max(1, int(round(effective_fps / max(sample_fps, 0.01))))
    return list(range(0, frame_count, step))


def _static_threshold_sweep(
    image_rows: list[dict[str, Any]],
    thresholds: tuple[int, ...],
) -> dict[str, int]:
    by_episode: dict[Any, list[dict[str, Any]]] = {}
    for row in image_rows:
        by_episode.setdefault(row["episode_id"], []).append(row)
    counts = {str(threshold): 0 for threshold in thresholds}
    for rows in by_episode.values():
        rows.sort(key=lambda row: int(row["frame_index"]))
        for left, right in zip(rows[:-1], rows[1:]):
            distance = hamming_distance(str(left["phash"]), str(right["phash"]))
            for threshold in thresholds:
                if distance <= threshold:
                    counts[str(threshold)] += 1
    return counts


def _blur_band_count(
    image_rows: list[dict[str, Any]],
    percentile: float,
) -> int:
    by_episode: dict[Any, list[dict[str, Any]]] = {}
    for row in image_rows:
        by_episode.setdefault(row["episode_id"], []).append(row)
    count = 0
    for rows in by_episode.values():
        threshold = float(np.percentile(
            [float(row["laplacian_variance"]) for row in rows],
            percentile,
        ))
        count += sum(
            float(row["laplacian_variance"]) < threshold
            for row in rows
        )
    return count


def _write_reports(
    records: list[dict[str, Any]],
    image_rows: list[dict[str, Any]],
    issues: list[dict[str, Any]],
    motions: dict[str, Any],
    output_dir: Path,
    dataset_version: str,
    git_commit: str,
    threshold_sweep: dict[str, int],
    threshold_policy: dict[str, Any],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for record in records:
        record.update({
            "dataset_version": dataset_version,
            "git_commit": git_commit,
            "created_at": created_at,
        })

    with (output_dir / "qc_results.csv").open("w", encoding="utf-8-sig", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        writer.writerows(records)

    schema = pa.schema([
        pa.field("episode_id", pa.string(), nullable=False),
        pa.field("status", pa.string(), nullable=False),
        pa.field("severity", pa.string(), nullable=False),
        pa.field("error_types", pa.string(), nullable=False),
        pa.field("error_count", pa.int32(), nullable=False),
        pa.field("camera_age_p95_ms", pa.float64()),
        pa.field("frame_count", pa.int64(), nullable=False),
        pa.field("episode_hash", pa.string(), nullable=False),
        pa.field("dataset_version", pa.string(), nullable=False),
        pa.field("git_commit", pa.string(), nullable=False),
        pa.field("created_at", pa.string(), nullable=False),
    ])
    pq.write_table(
        pa.Table.from_pylist(records, schema=schema),
        output_dir / "qc_results.parquet",
        compression="snappy",
    )

    if image_rows:
        pq.write_table(
            pa.Table.from_pylist(image_rows),
            output_dir / "image_metrics.parquet",
            compression="snappy",
        )
    if issues:
        fields = sorted({key for issue in issues for key in issue})
        with (output_dir / "image_rule_hits.csv").open("w", encoding="utf-8-sig", newline="") as file_handle:
            writer = csv.DictWriter(file_handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(issues)
        review_fields = [
            "rule_id",
            "episode_id",
            "frame_index",
            "timestamp_s",
            "evidence_path",
            "auto_label",
            "human_label",
            "is_true_positive",
            "note",
        ]
        with (output_dir / "image_review.csv").open("w", encoding="utf-8-sig", newline="") as file_handle:
            writer = csv.DictWriter(file_handle, fieldnames=review_fields)
            writer.writeheader()
            for issue in issues:
                writer.writerow({
                    "rule_id": issue.get("rule_id"),
                    "episode_id": issue.get("episode_id"),
                    "frame_index": issue.get("frame_index"),
                    "timestamp_s": issue.get("timestamp_s"),
                    "evidence_path": issue.get("evidence_path"),
                    "auto_label": issue.get("message"),
                    "human_label": "",
                    "is_true_positive": "",
                    "note": "",
                })
    (output_dir / "motion_profile.json").write_text(
        json.dumps(motions, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    severity_counts = Counter(record["severity"] for record in records)
    error_counts = Counter(
        error_type
        for record in records
        for error_type in record["error_types"].split(";")
        if error_type
    )
    rule_counts = Counter(str(issue.get("rule_id")) for issue in issues)
    unique_flagged_frames = len({
        (issue.get("episode_id"), issue.get("frame_index"))
        for issue in issues
    })
    return {
        "checked_episodes": len(records),
        "passed": sum(record["status"] == "passed" for record in records),
        "failed": sum(record["status"] == "failed" for record in records),
        "severity_counts": dict(sorted(severity_counts.items())),
        "error_type_counts": dict(sorted(error_counts.items())),
        "image_samples": len(image_rows),
        "image_rule_hits": len(issues),
        "unique_flagged_frames": unique_flagged_frames,
        "image_rule_hit_counts": dict(sorted(rule_counts.items())),
        "static_threshold_sweep": threshold_sweep,
        "threshold_policy": threshold_policy,
        "blur_band_counts": {
            "primary": _blur_band_count(
                image_rows,
                float(threshold_policy["blur_primary_percentile"]),
            ),
            "broad": _blur_band_count(
                image_rows,
                float(threshold_policy["blur_broad_percentile"]),
            ),
        },
        "static_band_counts": {
            "high_confidence": threshold_sweep[
                str(threshold_policy["static_primary_hamming"])
            ],
            "broad_only": (
                threshold_sweep[str(threshold_policy["static_broad_hamming"])]
                - threshold_sweep[str(threshold_policy["static_primary_hamming"])]
            ),
            "broad_total": threshold_sweep[
                str(threshold_policy["static_broad_hamming"])
            ],
        },
        "rule_interpretation": {
            "image_blur": "relative percentile rule hit; not an absolute defect claim",
            "image_static_candidate": "adjacent sampled frames are visually static; not cross-episode duplicate detection",
            "image_freeze": "visual stasis with relative state motion; requires human review",
        },
        "output_dir": str(output_dir),
    }


def run_qc(
    dataset_dir: str | Path,
    *,
    profile_name: str,
    episode_selection: str,
    camera_key: str | None,
    output_dir: str | Path,
    sample_fps: float = 1.0,
    dataset_version: str = "unversioned",
    git_commit: str = "unknown",
    save_evidence: bool = False,
    static_hamming: int | None = None,
    blur_percentile: float | None = None,
) -> dict[str, Any]:
    layout = discover_layout(dataset_dir)
    inspect_report = inspect_layout(layout)
    try:
        profile = load_profile(profile_name)
    except ProfileError:
        profile = {
            "robot_type": profile_name,
            "schema_version": inspect_report.get("codebase_version") or "unknown",
            "state_dim": inspect_report.get("state_shape", [None])[-1],
            "action_dim": inspect_report.get("action_shape", [None])[-1],
            "fps": inspect_report.get("fps"),
            "camera_age_p95_limit_ms": None,
        }

    total_episodes = int(layout.info.get("total_episodes") or 0)
    if total_episodes <= 0:
        # Derive from episode parquet only when older metadata omits the count.
        episode_table = pq.read_table(layout.episode_files[0])
        total_episodes = max(int(episode_table["episode_index"].to_pylist() or [0])) + 1
    selected = parse_episode_selection(episode_selection, total_episodes)
    episode_rows, episodes = iter_selected_episode_arrays(layout, selected)
    image_config = dict(profile.get("image", {}))
    resolved_static_hamming = int(
        static_hamming if static_hamming is not None else image_config.get("static_hamming", 4)
    )
    resolved_blur_percentile = float(
        blur_percentile
        if blur_percentile is not None
        else image_config.get("blur_percentile", 5.0)
    )
    resolved_broad_blur_percentile = float(
        image_config.get("blur_broad_percentile", resolved_blur_percentile + 3.0)
    )
    resolved_broad_static_hamming = int(
        image_config.get("static_broad_hamming", resolved_static_hamming + 2)
    )
    threshold_policy = {
        "blur_primary_percentile": resolved_blur_percentile,
        "blur_broad_percentile": resolved_broad_blur_percentile,
        "static_primary_hamming": resolved_static_hamming,
        "static_broad_hamming": resolved_broad_static_hamming,
        "source": image_config.get(
            "threshold_policy_source",
            "default profile policy",
        ),
        "interpretation": (
            "Primary thresholds drive review hits; broad thresholds are "
            "reported for sensitivity only."
        ),
    }

    camera_shape = None
    if camera_key:
        feature = layout.info.get("features", {}).get(camera_key, {})
        shape = feature.get("shape", [])
        if len(shape) >= 2:
            # OpenCV shape is (height, width); LeRobot feature shapes may use
            # either CHW or HWC. Select the two largest spatial values.
            spatial = sorted(map(int, shape[:3]))[-2:]
            camera_shape = (spatial[1], spatial[0])

    records: list[dict[str, Any]] = []
    image_rows: list[dict[str, Any]] = []
    evidence_by_frame: dict[tuple[str | None, int], str | None] = {}
    issues: list[dict[str, Any]] = []
    motions: dict[str, Any] = {}
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    for episode_index in selected:
        episode = episodes[episode_index]
        arrays = episode.arrays
        metadata = episode.metadata
        errors = list(metadata.get("errors", []))
        errors.extend(_validate_low_dim(
            arrays,
            state_dim=profile.get("state_dim") or metadata.get("state_dim"),
            action_dim=profile.get("action_dim") or metadata.get("action_dim"),
            expected_frame_count=metadata.get("expected_frame_count"),
        ))

        motion: dict[str, Any] = {}
        if "state" in arrays and "action" in arrays:
            try:
                motion = build_motion_profile(
                    arrays["state"],
                    arrays["action"],
                    fps=float(metadata.get("fps") or profile.get("fps") or 1.0),
                    indices=profile.get("motion", {}).get("indices"),
                    state_names=metadata.get("state_names"),
                    groups=profile.get("motion", {}).get("groups"),
                    state_unit=profile.get("motion", {}).get("unit"),
                )
            except (ValueError, ZeroDivisionError) as exc:
                motion = {"error": str(exc)}
        motions[str(metadata.get("episode_id") or episode_index)] = motion

        camera_path = None
        if camera_key:
            camera_path = video_path_for_episode(layout, episode_rows[episode_index], camera_key)
        sampled = _sample_indices(
            int(arrays.get("timestamp_s", np.array([])).shape[0] or 0),
            metadata.get("fps"),
            sample_fps,
        )
        if camera_path is None:
            if camera_key:
                errors.append("image decoder unavailable")
        else:
            source = VideoImageSource(
                camera_path,
                width=camera_shape[1] if camera_shape else None,
                height=camera_shape[0] if camera_shape else None,
            )
            offset = video_offset_for_episode(
                episode_rows[episode_index],
                camera_key,
            )
            metrics = []
            evidence_dir = (
                output_path
                / "evidence_frames"
                / str(metadata.get("episode_id") or f"episode_{episode_index:06d}")
            )
            if save_evidence:
                evidence_dir.mkdir(parents=True, exist_ok=True)
            for frame_index in sampled:
                timestamp = float(arrays["timestamp_s"][frame_index]) if "timestamp_s" in arrays else frame_index
                frame = source.frame_at(offset + timestamp)
                if frame is None:
                    errors.append("image decoder unavailable")
                    break
                metric = image_metrics(gray(frame), frame_index, timestamp)
                metrics.append(metric)
                evidence_path = None
                if save_evidence:
                    evidence_path = evidence_dir / f"frame_{frame_index:08d}.jpg"
                    encoded, buffer = cv2.imencode(".jpg", frame)
                    if not encoded:
                        raise OSError(f"cannot write image evidence frame: {evidence_path}")
                    evidence_path.write_bytes(buffer.tobytes())
                image_rows.append({
                    "episode_id": metadata.get("episode_id"),
                    "frame_index": metric.frame_index,
                    "timestamp_s": metric.timestamp_s,
                    "brightness_mean": metric.brightness_mean,
                    "brightness_std": metric.brightness_std,
                    "highlight_ratio": metric.highlight_ratio,
                    "shadow_ratio": metric.shadow_ratio,
                    "laplacian_variance": metric.laplacian_variance,
                    "phash": metric.phash,
                    "evidence_path": str(evidence_path) if evidence_path else None,
                })
                evidence_by_frame[
                    (
                        str(metadata.get("episode_id") or f"episode_{episode_index:06d}"),
                        metric.frame_index,
                    )
                ] = str(evidence_path) if evidence_path else None
            episode_issues = evaluate_image_metrics(
                metrics,
                blur_percentile=resolved_blur_percentile,
                static_hamming=resolved_static_hamming,
            )
            if "state" in arrays:
                episode_issues.extend(evaluate_freeze(
                    metrics,
                    arrays["state"],
                    hamming_threshold=int(image_config.get("freeze_hamming", 2)),
                    motion_quantile=float(
                        image_config.get("freeze_motion_quantile", 50.0)
                    ),
                ))
            for issue in episode_issues:
                errors.append(str(issue["message"]))
                issue = dict(issue)
                issue["episode_id"] = metadata.get("episode_id")
                issue["evidence_path"] = evidence_by_frame.get(
                    (issue["episode_id"], int(issue["frame_index"]))
                )
                issues.append(issue)

        error_details = [
            {
                "code": classify_error(error),
                "message": error,
                "severity": severity_for_error(classify_error(error), {}),
            }
            for error in errors
        ]
        severity = highest_severity([detail["severity"] for detail in error_details])
        error_types = sorted({detail["code"] for detail in error_details})
        records.append({
            "episode_id": str(metadata.get("episode_id") or f"episode_{episode_index:06d}"),
            "status": "passed" if not errors else "failed",
            "severity": severity,
            "error_types": ";".join(error_types),
            "error_count": len(errors),
            "camera_age_p95_ms": None,
            "frame_count": int(metadata.get("frame_count") or 0),
            "episode_hash": _episode_hash(episode_index, arrays),
            "dataset_version": dataset_version,
            "git_commit": git_commit,
            "created_at": "",
        })

    (output_path / "inspect_report.json").write_text(
        json.dumps(inspect_report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = _write_reports(
        records,
        image_rows,
        issues,
        motions,
        output_path,
        dataset_version,
        git_commit,
        _static_threshold_sweep(
            image_rows,
            tuple(sorted({
                resolved_static_hamming,
                resolved_broad_static_hamming,
                6,
            })),
        ),
        threshold_policy,
    )
    (output_path / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--profile", default="unitree_g1")
    parser.add_argument("--episodes", default="0")
    parser.add_argument("--camera-key", default=None)
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument(
        "--static-hamming",
        type=int,
        default=None,
        help="Adjacent-frame pHash Hamming threshold; default comes from the profile.",
    )
    parser.add_argument(
        "--blur-percentile",
        type=float,
        default=None,
        help="Relative blur percentile; default comes from the profile.",
    )
    parser.add_argument(
        "--save-evidence",
        action="store_true",
        help="Save sampled image frames for human review.",
    )
    parser.add_argument("--output", type=Path, default=Path("reports/lerobot_qc"))
    parser.add_argument("--dataset-version", default="unversioned")
    parser.add_argument("--git-commit", default="unknown")
    args = parser.parse_args()

    summary = run_qc(
        args.dataset_dir,
        profile_name=args.profile,
        episode_selection=args.episodes,
        camera_key=args.camera_key,
        output_dir=args.output,
        sample_fps=args.sample_fps,
        dataset_version=args.dataset_version,
        git_commit=args.git_commit,
        save_evidence=args.save_evidence,
        static_hamming=args.static_hamming,
        blur_percentile=args.blur_percentile,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
