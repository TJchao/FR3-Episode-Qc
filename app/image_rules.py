"""Lightweight sampled-image quality rules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np


@dataclass(frozen=True)
class ImageMetric:
    frame_index: int
    timestamp_s: float
    brightness_mean: float
    brightness_std: float
    highlight_ratio: float
    shadow_ratio: float
    laplacian_variance: float
    phash: str


def gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError("image must be HxW or HxWx3/HxWx4")
    if image.shape[2] == 4:
        image = image[:, :, :3]
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def perceptual_hash(image: np.ndarray, hash_size: int = 8) -> str:
    resized = cv2.resize(image, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(np.float32(resized))
    low = dct[:hash_size, :hash_size]
    median = np.median(low[1:, 1:])
    bits = (low > median).flatten()
    return "".join("1" if value else "0" for value in bits)


def hamming_distance(left: str, right: str) -> int:
    if len(left) != len(right):
        return len(left)
    return sum(a != b for a, b in zip(left, right))


def image_metrics(
    image: np.ndarray,
    frame_index: int,
    timestamp_s: float,
) -> ImageMetric:
    pixels = image.astype(np.float32)
    highlight = float(np.mean(pixels >= 250))
    shadow = float(np.mean(pixels <= 5))
    laplacian = cv2.Laplacian(image, cv2.CV_32F)
    return ImageMetric(
        frame_index=int(frame_index),
        timestamp_s=float(timestamp_s),
        brightness_mean=float(np.mean(pixels)),
        brightness_std=float(np.std(pixels)),
        highlight_ratio=highlight,
        shadow_ratio=shadow,
        laplacian_variance=float(laplacian.var()),
        phash=perceptual_hash(image),
    )


def evaluate_image_metrics(
    metrics: list[ImageMetric],
    *,
    black_mean: float = 8.0,
    white_mean: float = 247.0,
    low_std: float = 5.0,
    exposure_ratio: float = 0.20,
    blur_percentile: float = 5.0,
    duplicate_hamming: int = 4,
) -> list[dict[str, object]]:
    issues: list[dict[str, object]] = []
    if not metrics:
        return issues

    blur_threshold = float(np.percentile(
        [metric.laplacian_variance for metric in metrics],
        blur_percentile,
    ))

    for metric in metrics:
        if metric.brightness_mean < black_mean and metric.brightness_std < low_std:
            issues.append({
                "rule_id": "image_black_white",
                "severity": "fatal",
                "frame_index": metric.frame_index,
                "timestamp_s": metric.timestamp_s,
                "value": metric.brightness_mean,
                "threshold": black_mean,
                "message": "image black or white frame",
            })
        elif metric.brightness_mean > white_mean and metric.brightness_std < low_std:
            issues.append({
                "rule_id": "image_black_white",
                "severity": "fatal",
                "frame_index": metric.frame_index,
                "timestamp_s": metric.timestamp_s,
                "value": metric.brightness_mean,
                "threshold": white_mean,
                "message": "image black or white frame",
            })

        if metric.highlight_ratio > exposure_ratio or metric.shadow_ratio > exposure_ratio:
            issues.append({
                "rule_id": "image_exposure",
                "severity": "warning",
                "frame_index": metric.frame_index,
                "timestamp_s": metric.timestamp_s,
                "value": max(metric.highlight_ratio, metric.shadow_ratio),
                "threshold": exposure_ratio,
                "message": "image exposure ratio exceeded",
            })

        if metric.laplacian_variance < blur_threshold:
            issues.append({
                "rule_id": "image_blur",
                "severity": "warning",
                "frame_index": metric.frame_index,
                "timestamp_s": metric.timestamp_s,
                "value": metric.laplacian_variance,
                "threshold": blur_threshold,
                "message": "relative image blur threshold exceeded",
            })

    for left, right in zip(metrics[:-1], metrics[1:]):
        distance = hamming_distance(left.phash, right.phash)
        if distance <= duplicate_hamming:
            issues.append({
                "rule_id": "image_duplicate",
                "severity": "warning",
                "frame_index": right.frame_index,
                "timestamp_s": right.timestamp_s,
                "value": distance,
                "threshold": duplicate_hamming,
                "message": "sampled image pair is near-duplicate",
            })

    return issues


def evaluate_freeze(
    metrics: list[ImageMetric],
    state: np.ndarray,
    *,
    hamming_threshold: int = 2,
    motion_quantile: float = 50.0,
) -> list[dict[str, object]]:
    """Detect image inactivity that contradicts robot-state motion.

    A static camera view is not automatically a frozen frame. The rule only
    fires when consecutive sampled images are nearly identical while the robot
    state moves more than the configured relative threshold.
    """
    if state.ndim != 2 or len(metrics) < 2:
        return []

    issues: list[dict[str, object]] = []
    frame_indices = [metric.frame_index for metric in metrics]
    for left, right in zip(metrics[:-1], metrics[1:]):
        if left.frame_index >= state.shape[0] or right.frame_index >= state.shape[0]:
            continue
        distance = hamming_distance(left.phash, right.phash)
        motion = float(np.linalg.norm(
            state[right.frame_index].astype(float)
            - state[left.frame_index].astype(float)
        ))
        comparable_motions = []
        for index in frame_indices[:-1]:
            next_indices = [value for value in frame_indices if value > index]
            if not next_indices:
                continue
            next_index = min(next_indices)
            comparable_motions.append(float(np.linalg.norm(
                state[next_index].astype(float) - state[index].astype(float)
            )))
        if not comparable_motions:
            continue
        threshold = max(float(np.percentile(comparable_motions, motion_quantile)), 1e-8)
        if distance <= hamming_threshold and motion > threshold:
            issues.append({
                "rule_id": "image_freeze",
                "severity": "repairable",
                "frame_index": right.frame_index,
                "timestamp_s": right.timestamp_s,
                "value": motion,
                "threshold": threshold,
                "message": "sampled image freeze while state motion",
            })
    return issues
