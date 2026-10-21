"""Episode quality-check core logic and CLI."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from app.constants import (
    ACTION_DIM,
    CAMERA_AGE_P95_LIMIT_MS,
    REQUIRED_METADATA_FIELDS,
    REQUIRED_NPZ_FIELDS,
    SCHEMA_VERSION,
    STATE_DIM,
)


class EpisodePathError(Exception):
    """Base error for invalid episode paths."""

    status_code = 400


class EpisodePathNotFound(EpisodePathError):
    """Requested episode path does not exist."""

    status_code = 404


class EpisodePathForbidden(EpisodePathError):
    """Requested episode path is outside the allowed root."""

    status_code = 403


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _episode_hash(episode_dir: Path) -> str:
    """Hash the episode payload and metadata together."""
    digest = hashlib.sha256()
    for filename in ("frames.npz", "metadata.json"):
        path = episode_dir / filename
        if not path.exists():
            continue
        digest.update(filename.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256_file(path)))
    return f"sha256:{digest.hexdigest()}"


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def resolve_episode_path(
    episode_path: str | Path,
    allowed_root: str | Path | None,
) -> Path:
    """Resolve an episode path and enforce an optional allowed root."""
    try:
        resolved = Path(episode_path).expanduser().resolve(strict=True)
    except FileNotFoundError:
        raise EpisodePathNotFound("episode path does not exist") from None
    except OSError as exc:
        raise EpisodePathError(f"cannot resolve episode path: {exc}") from exc

    if not resolved.is_dir():
        raise EpisodePathError("episode path is not a directory")

    if allowed_root is not None:
        try:
            root = Path(allowed_root).expanduser().resolve(strict=True)
        except FileNotFoundError:
            raise EpisodePathError("allowed root does not exist") from None
        except OSError as exc:
            raise EpisodePathError(f"cannot resolve allowed root: {exc}") from exc

        if not root.is_dir():
            raise EpisodePathError("allowed root is not a directory")
        if not resolved.is_relative_to(root):
            raise EpisodePathForbidden(
                "episode path is outside the allowed root"
            )

    return resolved


def _load_metadata(
    episode_dir: Path,
) -> tuple[dict[str, Any] | None, str | None]:
    metadata_path = episode_dir / "metadata.json"
    if not metadata_path.exists():
        return None, "metadata.json is missing"

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, f"metadata.json cannot be read: {exc}"

    if not isinstance(metadata, dict):
        return None, "metadata.json must contain a JSON object"
    return metadata, None


def _load_arrays(
    episode_dir: Path,
) -> tuple[dict[str, np.ndarray] | None, str | None]:
    frames_path = episode_dir / "frames.npz"
    if not frames_path.exists():
        return None, "frames.npz is missing"

    try:
        with np.load(frames_path, allow_pickle=False) as npz:
            arrays = {name: np.asarray(npz[name]) for name in npz.files}
    except (OSError, ValueError) as exc:
        return None, f"frames.npz cannot be read: {exc}"

    return arrays, None


def _validate_metadata(
    metadata: dict[str, Any],
    report: dict[str, Any],
) -> None:
    missing_fields = [
        field for field in REQUIRED_METADATA_FIELDS if field not in metadata
    ]
    if missing_fields:
        report["errors"].append(
            "metadata missing fields: " + ", ".join(missing_fields)
        )

    if "episode_id" in metadata:
        if not isinstance(metadata["episode_id"], str) or not metadata["episode_id"].strip():
            report["errors"].append("metadata episode_id must be a non-empty string")
        else:
            report["episode_id"] = metadata["episode_id"]

    int_fields = (
        "frame_count",
        "fps",
        "state_dim",
        "action_dim",
        "rgb_frame_count",
        "depth_frame_count",
    )
    for field in int_fields:
        if field in metadata and not _is_positive_int(metadata[field]):
            report["errors"].append(f"metadata {field} must be a positive integer")

    if "schema_version" in metadata and metadata["schema_version"] != SCHEMA_VERSION:
        report["errors"].append(
            f"metadata schema_version must be {SCHEMA_VERSION}"
        )

    if metadata.get("state_dim") != STATE_DIM:
        report["errors"].append(f"metadata state_dim must be {STATE_DIM}")
    if metadata.get("action_dim") != ACTION_DIM:
        report["errors"].append(f"metadata action_dim must be {ACTION_DIM}")


def _validate_arrays(
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any] | None,
    report: dict[str, Any],
) -> None:
    missing_fields = [
        field for field in REQUIRED_NPZ_FIELDS if field not in arrays
    ]
    if missing_fields:
        report["errors"].append(
            "frames.npz missing fields: " + ", ".join(missing_fields)
        )

    present_fields = [
        field for field in REQUIRED_NPZ_FIELDS if field in arrays
    ]
    lengths: dict[str, int] = {}
    for field in present_fields:
        array = arrays[field]
        if array.ndim < 1:
            report["errors"].append(f"{field} must be an array")
            continue
        lengths[field] = int(array.shape[0])

    if lengths and len(set(lengths.values())) != 1:
        report["errors"].append(f"field length mismatch: {lengths}")
        return

    if not lengths:
        return

    frame_count = next(iter(lengths.values()))
    report["frame_count"] = frame_count

    if "robot_time_ns" in arrays and not np.all(
        np.diff(arrays["robot_time_ns"]) > 0
    ):
        report["errors"].append("robot_time_ns is not strictly increasing")

    if "timestamp_s" in arrays and not np.all(
        np.diff(arrays["timestamp_s"]) > 0
    ):
        report["errors"].append("timestamp_s is not strictly increasing")

    for field in present_fields:
        if not np.all(np.isfinite(arrays[field])):
            report["errors"].append(f"{field} contains NaN or Inf")

    if "state" in arrays:
        expected_state_shape = (frame_count, STATE_DIM)
        if arrays["state"].shape != expected_state_shape:
            report["errors"].append(
                f"state shape must be {expected_state_shape}, got {arrays['state'].shape}"
            )

    if "action" in arrays:
        expected_action_shape = (frame_count, ACTION_DIM)
        if arrays["action"].shape != expected_action_shape:
            report["errors"].append(
                f"action shape must be {expected_action_shape}, got {arrays['action'].shape}"
            )

    if "camera_age_ms" in arrays:
        camera_age = arrays["camera_age_ms"]
        if camera_age.ndim != 1 or camera_age.shape[0] != frame_count:
            report["errors"].append(
                f"camera_age_ms shape must be ({frame_count},), got {camera_age.shape}"
            )
        elif np.all(np.isfinite(camera_age)):
            p95_ms = float(np.percentile(camera_age, 95))
            report["camera_age_p95_ms"] = round(p95_ms, 3)
            if p95_ms >= CAMERA_AGE_P95_LIMIT_MS:
                report["errors"].append(
                    f"camera age P95 must be below {CAMERA_AGE_P95_LIMIT_MS}ms"
                )

    if metadata is not None:
        expected_count = metadata.get("frame_count")
        if _is_positive_int(expected_count) and expected_count != frame_count:
            report["errors"].append(
                f"metadata frame_count is {expected_count}, actual frames are {frame_count}"
            )

        rgb_count = metadata.get("rgb_frame_count")
        if _is_positive_int(rgb_count) and rgb_count != frame_count:
            report["errors"].append(
                f"rgb_frame_count is {rgb_count}, actual frames are {frame_count}"
            )

        depth_count = metadata.get("depth_frame_count")
        if _is_positive_int(depth_count) and depth_count != frame_count:
            report["errors"].append(
                f"depth_frame_count is {depth_count}, actual frames are {frame_count}"
            )


def check_episode(
    episode_path: str | Path,
    allowed_root: str | Path | None = None,
) -> dict[str, Any]:
    """Check one synthetic FR3 episode and return a structured report."""
    report: dict[str, Any] = {
        "episode_id": None,
        "status": "failed",
        "errors": [],
        "warnings": [],
        "frame_count": 0,
        "camera_age_p95_ms": None,
        "episode_hash": None,
    }

    try:
        resolved_path = resolve_episode_path(episode_path, allowed_root)
    except EpisodePathError as exc:
        report["errors"].append(str(exc))
        return report

    report["episode_id"] = resolved_path.name
    report["episode_hash"] = _episode_hash(resolved_path)

    metadata, metadata_error = _load_metadata(resolved_path)
    if metadata_error is not None:
        report["errors"].append(metadata_error)

    arrays, arrays_error = _load_arrays(resolved_path)
    if arrays_error is not None:
        report["errors"].append(arrays_error)

    if metadata is not None:
        _validate_metadata(metadata, report)

    if arrays is not None:
        _validate_arrays(arrays, metadata, report)

    report["status"] = "passed" if not report["errors"] else "failed"
    return report


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode_path", help="Path to one episode directory")
    parser.add_argument(
        "--allowed-root",
        default=None,
        help="Restrict episode paths to this root directory",
    )
    return parser


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()
    report = check_episode(args.episode_path, args.allowed_root)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
