"""Compute lightweight episode visual embeddings and coverage density."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from app.embedding import (
    EMBEDDING_METHOD,
    build_descriptor,
    compute_coverage,
    compute_duplicate_groups,
    mean_episode_descriptor,
    pca_projection,
    write_pca_svg,
)
from app.readers.lerobot import (
    discover_layout,
    inspect_layout,
    iter_selected_episode_arrays,
    parse_episode_selection,
    video_offset_for_episode,
    video_path_for_episode,
)
from app.readers.video import VideoImageSource


def run(
    dataset_dir: str | Path,
    *,
    episode_selection: str,
    camera_key: str,
    output_dir: str | Path,
    frames_per_episode: int,
    k: int,
    near_duplicate_similarity: float,
) -> dict[str, Any]:
    layout = discover_layout(dataset_dir)
    inspect_report = inspect_layout(layout)
    total = int(layout.info.get("total_episodes") or 0)
    selected = parse_episode_selection(episode_selection, total)
    episode_rows, episodes = iter_selected_episode_arrays(layout, selected)
    feature = layout.info.get("features", {}).get(camera_key, {})
    shape = feature.get("shape", [])
    spatial = sorted(map(int, shape[:3]))[-2:] if len(shape) >= 2 else []
    width = spatial[1] if spatial else None
    height = spatial[0] if spatial else None

    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    source_by_path: dict[Path, VideoImageSource] = {}
    descriptors: list[np.ndarray] = []
    episode_ids: list[str] = []
    frame_rows: list[dict[str, Any]] = []
    skipped: list[str] = []

    for index in selected:
        episode = episodes[index]
        metadata = episode.metadata
        episode_id = str(metadata.get("episode_id") or f"episode_{index:06d}")
        video_path = video_path_for_episode(layout, episode_rows[index], camera_key)
        if video_path is None:
            skipped.append(episode_id)
            continue
        source = source_by_path.get(video_path)
        if source is None:
            source = VideoImageSource(video_path, width=width, height=height)
            source_by_path[video_path] = source

        frame_count = int(len(episode.arrays.get("timestamp_s", [])))
        if frame_count == 0 or frames_per_episode <= 0:
            skipped.append(episode_id)
            continue
        chosen = np.linspace(0, frame_count - 1, frames_per_episode, dtype=int)
        episode_descriptors: list[np.ndarray] = []
        offset = video_offset_for_episode(episode_rows[index], camera_key)
        for frame_index in chosen:
            timestamp = float(episode.arrays["timestamp_s"][frame_index])
            frame = source.frame_at(offset + timestamp)
            if frame is None:
                continue
            descriptor = build_descriptor(frame)
            episode_descriptors.append(descriptor)
            frame_rows.append({
                "episode_id": episode_id,
                "frame_index": int(frame_index),
                "timestamp_s": timestamp,
                "descriptor_norm": round(float(np.linalg.norm(descriptor)), 6),
            })
        if not episode_descriptors:
            skipped.append(episode_id)
            continue
        descriptors.append(mean_episode_descriptor(episode_descriptors))
        episode_ids.append(episode_id)

    for source in source_by_path.values():
        source.close()
    if len(episode_ids) < 2:
        raise RuntimeError("embedding coverage requires at least two decodable episodes")

    embedding_matrix = np.stack(descriptors, axis=0).astype(np.float32)
    density_rows, duplicate_rows, summary = compute_coverage(
        embedding_matrix,
        episode_ids,
        k=k,
        near_duplicate_similarity=near_duplicate_similarity,
    )
    duplicate_group_rows, duplicate_group_summary = compute_duplicate_groups(
        embedding_matrix,
        episode_ids,
    )
    summary.update({
        "dataset_dir": str(Path(dataset_dir).resolve()),
        "camera_key": camera_key,
        "frames_per_episode": frames_per_episode,
        "requested_episodes": len(selected),
        "decoded_episodes": len(episode_ids),
        "skipped_episodes": skipped,
        "codebase_version": inspect_report.get("codebase_version"),
        "duplicate_groups": duplicate_group_summary,
    })

    np.save(output_path / "embeddings.npy", embedding_matrix)
    with (output_path / "episode_embeddings.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file_handle:
        writer = csv.DictWriter(
            file_handle,
            fieldnames=["episode_id", "embedding_index", "method"],
        )
        writer.writeheader()
        for index, episode_id in enumerate(episode_ids):
            writer.writerow({
                "episode_id": episode_id,
                "embedding_index": index,
                "method": EMBEDDING_METHOD,
            })
    with (output_path / "frame_embeddings.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file_handle:
        writer = csv.DictWriter(
            file_handle,
            fieldnames=["episode_id", "frame_index", "timestamp_s", "descriptor_norm"],
        )
        writer.writeheader()
        writer.writerows(frame_rows)
    with (output_path / "coverage_density.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file_handle:
        writer = csv.DictWriter(
            file_handle,
            fieldnames=sorted({key for row in density_rows for key in row}),
        )
        writer.writeheader()
        writer.writerows(density_rows)
    with (output_path / "visual_duplicate_candidates.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file_handle:
        writer = csv.DictWriter(
            file_handle,
            fieldnames=["episode_a", "episode_b", "cosine_similarity", "threshold", "suggested_action"],
        )
        writer.writeheader()
        writer.writerows(duplicate_rows)
    if duplicate_group_rows:
        group_fields = [
            "threshold",
            "group_id",
            "group_size",
            "member_episodes",
            "mean_intra_similarity",
            "min_intra_similarity",
            "max_intra_similarity",
            "suggested_action",
        ]
        with (output_path / "visual_duplicate_groups.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as file_handle:
            writer = csv.DictWriter(file_handle, fieldnames=group_fields)
            writer.writeheader()
            writer.writerows(duplicate_group_rows)

    coordinates, explained = pca_projection(embedding_matrix)
    low_density_ids = {
        row["episode_id"]
        for row in density_rows
        if row["is_relative_low_density_tail"]
    }
    write_pca_svg(
        output_path / "coverage_pca.svg",
        coordinates,
        episode_ids,
        explained,
        low_density_ids,
    )
    (output_path / "coverage_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--camera-key", required=True)
    parser.add_argument("--frames-per-episode", type=int, default=6)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--near-duplicate-similarity", type=float, default=0.99)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = run(
        args.dataset_dir,
        episode_selection=args.episodes,
        camera_key=args.camera_key,
        output_dir=args.output,
        frames_per_episode=args.frames_per_episode,
        k=args.k,
        near_duplicate_similarity=args.near_duplicate_similarity,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
