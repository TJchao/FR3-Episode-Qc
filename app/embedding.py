"""Handcrafted visual embeddings and episode coverage analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.image_rules import perceptual_hash


EMBEDDING_METHOD = "handcrafted_visual_v1"


@dataclass(frozen=True)
class EpisodeEmbedding:
    episode_id: str
    frame_indices: list[int]
    timestamps_s: list[float]
    descriptor: np.ndarray


def build_descriptor(frame: np.ndarray) -> np.ndarray:
    """Build a deterministic, lightweight visual descriptor.

    This is intentionally not a semantic CLIP/DINO embedding. It combines HSV
    color, DCT layout, gradient orientation, and pHash information for a fast,
    dependency-free visual coverage prototype.
    """
    if frame.ndim == 2:
        gray = frame
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    elif frame.ndim == 3 and frame.shape[2] in (3, 4):
        if frame.shape[2] == 4:
            frame = frame[:, :, :3]
        bgr = frame
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError("frame must be HxW or HxWx3/HxWx4")

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    color_hist = cv2.calcHist(
        [hsv],
        [0, 1, 2],
        None,
        [8, 4, 4],
        [0, 180, 0, 256, 0, 256],
    ).flatten()

    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(np.float32(resized))[:4, :4].flatten()

    gx = cv2.Sobel(resized, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(resized, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx * gx + gy * gy)
    angle = ((np.arctan2(gy, gx) + np.pi) * (8 / (2 * np.pi))).astype(np.int32)
    angle[angle >= 8] = 0
    orientation = np.bincount(
        angle.flatten(),
        weights=magnitude.flatten(),
        minlength=8,
    )

    bits = np.asarray([1 if char == "1" else -1 for char in perceptual_hash(gray)], dtype=np.float32)
    descriptor = np.concatenate(
        [color_hist, dct, orientation, bits]
    ).astype(np.float32)
    norm = float(np.linalg.norm(descriptor))
    if norm > 1e-8:
        descriptor /= norm
    return descriptor


def mean_episode_descriptor(descriptors: list[np.ndarray]) -> np.ndarray:
    if not descriptors:
        raise ValueError("descriptors must not be empty")
    result = np.mean(np.stack(descriptors, axis=0), axis=0)
    norm = float(np.linalg.norm(result))
    return result / norm if norm > 1e-8 else result


def compute_coverage(
    embeddings: np.ndarray,
    episode_ids: list[str],
    *,
    k: int = 3,
    near_duplicate_similarity: float = 0.99,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    count = len(episode_ids)
    if embeddings.shape[0] != count:
        raise ValueError("embedding count and episode id count differ")
    if count < 2:
        raise ValueError("coverage analysis needs at least two episodes")

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / np.maximum(norms, 1e-8)
    similarity = normalized @ normalized.T
    upper_left, upper_right = np.triu_indices(count, k=1)
    pairwise_similarities = similarity[upper_left, upper_right]
    closest_offset = int(np.argmax(pairwise_similarities))
    closest_left = int(upper_left[closest_offset])
    closest_right = int(upper_right[closest_offset])
    closest_similarity = float(pairwise_similarities[closest_offset])
    np.fill_diagonal(similarity, -np.inf)
    resolved_k = min(k, count - 1)
    nearest_indices = np.argsort(-similarity, axis=1)[:, :resolved_k]

    rows: list[dict[str, Any]] = []
    for index, episode_id in enumerate(episode_ids):
        neighbours = nearest_indices[index]
        similarities = similarity[index, neighbours]
        distances = 1.0 - similarities
        mean_distance = float(np.mean(distances))
        rows.append({
            "episode_id": episode_id,
            "k": resolved_k,
            "nearest_episodes": ";".join(episode_ids[int(i)] for i in neighbours),
            "nearest_similarities": ";".join(f"{float(v):.6f}" for v in similarities),
            "knn_distance_mean": round(mean_distance, 6),
            "knn_distance_p90_within_k": round(float(np.percentile(distances, 90)), 6),
            "max_similarity": round(float(np.max(similarities)), 6),
        })

    threshold = float(np.percentile([row["knn_distance_mean"] for row in rows], 90))
    for row in rows:
        row["low_density_threshold"] = round(threshold, 6)
        row["is_relative_low_density_tail"] = bool(
            row["knn_distance_mean"] >= threshold
        )

    pairs: list[dict[str, Any]] = []
    for left in range(count):
        for right in range(left + 1, count):
            value = float(similarity[left, right])
            if value >= near_duplicate_similarity:
                pairs.append({
                    "episode_a": episode_ids[left],
                    "episode_b": episode_ids[right],
                    "cosine_similarity": round(value, 6),
                    "threshold": near_duplicate_similarity,
                    "suggested_action": "human_review",
                })

    summary = {
        "method": EMBEDDING_METHOD,
        "episode_count": count,
        "embedding_dimension": int(embeddings.shape[1]),
        "k": resolved_k,
        "near_duplicate_similarity_threshold": near_duplicate_similarity,
        "threshold_boundary": (
            "The similarity threshold is uncalibrated; pairs are review candidates only."
        ),
        "relative_low_density_tail_count": sum(
            row["is_relative_low_density_tail"] for row in rows
        ),
        "near_duplicate_candidate_pairs": len(pairs),
        "pairwise_similarity": {
            "count": int(len(pairwise_similarities)),
            "min": round(float(np.min(pairwise_similarities)), 6),
            "mean": round(float(np.mean(pairwise_similarities)), 6),
            "p50": round(float(np.percentile(pairwise_similarities, 50)), 6),
            "p90": round(float(np.percentile(pairwise_similarities, 90)), 6),
            "max": round(closest_similarity, 6),
        },
        "closest_pair": {
            "episode_a": episode_ids[closest_left],
            "episode_b": episode_ids[closest_right],
            "cosine_similarity": round(closest_similarity, 6),
            "margin_below_duplicate_threshold": round(
                near_duplicate_similarity - closest_similarity,
                6,
            ),
        },
        "coverage_boundary": (
            "kNN distance is computed in the original handcrafted embedding space; "
            "PCA is visualization only."
        ),
    }
    return rows, pairs, summary


def compute_duplicate_groups(
    embeddings: np.ndarray,
    episode_ids: list[str],
    *,
    thresholds: tuple[float, ...] = (0.98, 0.99, 0.995),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build connected duplicate-candidate groups at multiple thresholds."""
    count = len(episode_ids)
    if embeddings.shape[0] != count:
        raise ValueError("embedding count and episode id count differ")
    if count < 2:
        raise ValueError("duplicate grouping needs at least two episodes")

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / np.maximum(norms, 1e-8)
    similarity = normalized @ normalized.T
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "thresholds": list(thresholds),
        "boundary": (
            "Groups are connected components of uncalibrated similarity edges. "
            "They are review candidates, not deletion decisions."
        ),
    }

    for threshold in thresholds:
        parent = list(range(count))

        def find(node: int) -> int:
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        def union(left: int, right: int) -> None:
            left_root = find(left)
            right_root = find(right)
            if left_root != right_root:
                parent[right_root] = left_root

        edge_count = 0
        for left in range(count):
            for right in range(left + 1, count):
                if float(similarity[left, right]) >= threshold:
                    union(left, right)
                    edge_count += 1

        components: dict[int, list[int]] = {}
        for index in range(count):
            components.setdefault(find(index), []).append(index)

        group_index = 0
        group_count = 0
        duplicate_episode_count = 0
        for members in sorted(
            components.values(),
            key=lambda values: (min(episode_ids[index] for index in values), len(values)),
        ):
            if len(members) < 2:
                continue
            group_index += 1
            group_count += 1
            duplicate_episode_count += len(members)
            pairwise = [
                float(similarity[left, right])
                for offset, left in enumerate(members[:-1])
                for right in members[offset + 1 :]
            ]
            rows.append({
                "threshold": threshold,
                "group_id": f"t{threshold:.3f}_g{group_index:03d}",
                "group_size": len(members),
                "member_episodes": ";".join(
                    episode_ids[index] for index in members
                ),
                "mean_intra_similarity": round(float(np.mean(pairwise)), 6),
                "min_intra_similarity": round(float(np.min(pairwise)), 6),
                "max_intra_similarity": round(float(np.max(pairwise)), 6),
                "suggested_action": "human_review",
            })

        summary[str(threshold)] = {
            "edge_count": edge_count,
            "group_count": group_count,
            "duplicate_episode_count": duplicate_episode_count,
        }

    return rows, summary


def pca_projection(embeddings: np.ndarray) -> tuple[np.ndarray, list[float]]:
    centered = embeddings - np.mean(embeddings, axis=0, keepdims=True)
    covariance = centered @ centered.T / max(len(centered) - 1, 1)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    values = np.maximum(values[order[:2]], 0.0)
    coordinates = vectors[:, order[:2]]
    explained = values / max(float(np.sum(values)), 1e-12)
    return coordinates, [round(float(value), 6) for value in explained[:2]]


def write_pca_svg(
    path: str | Path,
    coordinates: np.ndarray,
    episode_ids: list[str],
    explained_variance: list[float],
    low_density_ids: set[str],
    *,
    width: int = 900,
    height: int = 620,
) -> None:
    if len(coordinates) == 0:
        raise ValueError("coordinates must not be empty")
    left, top, right, bottom = 70, 30, width - 30, height - 45
    x_min, x_max = float(np.min(coordinates[:, 0])), float(np.max(coordinates[:, 0]))
    y_min, y_max = float(np.min(coordinates[:, 1])), float(np.max(coordinates[:, 1]))
    x_span = x_max - x_min if x_max > x_min else 1.0
    y_span = y_max - y_min if y_max > y_min else 1.0
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#666"/>',
        f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#666"/>',
        "<style>text{font-family:Arial,sans-serif;font-size:11px;fill:#222}</style>",
    ]
    for index, (x_value, y_value) in enumerate(coordinates):
        x = left + (float(x_value) - x_min) / x_span * (right - left)
        y = bottom - (float(y_value) - y_min) / y_span * (bottom - top)
        episode_id = episode_ids[index]
        fill = "#c0392b" if episode_id in low_density_ids else "#2471a3"
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{fill}">')
        parts.append(f"<title>{episode_id}</title></circle>")
    label = "PCA visualization only; coverage uses original-space kNN"
    variance = ", ".join(f"PC{i + 1}: {value:.1%}" for i, value in enumerate(explained_variance))
    parts.append(f'<text x="{left}" y="{height - 20}">{label} ({variance})</text>')
    parts.append("</svg>")
    Path(path).write_text("\n".join(parts), encoding="utf-8")
