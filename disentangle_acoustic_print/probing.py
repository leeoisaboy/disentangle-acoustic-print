from __future__ import annotations

from typing import Sequence

import numpy as np


def rank_auc(positive: Sequence[float], negative: Sequence[float]) -> float:
    """Rank AUC with deterministic tie handling."""
    positives = np.asarray(positive, dtype=np.float64)
    negatives = np.asarray(negative, dtype=np.float64)
    if not len(positives) or not len(negatives):
        raise ValueError("AUC needs both positive and negative samples")
    greater = np.sum(positives[:, None] > negatives[None, :])
    ties = np.sum(positives[:, None] == negatives[None, :])
    return float((greater + 0.5 * ties) / (len(positives) * len(negatives)))


def calibrate_by_group(raw_scores: Sequence[float], group_ids: Sequence[str]) -> np.ndarray:
    """Remove group-level response bias with within-group standardization."""
    values = np.asarray(raw_scores, dtype=np.float64)
    groups = np.asarray(group_ids)
    if values.shape != groups.shape:
        raise ValueError("scores and groups must have equal shape")
    output = np.zeros_like(values)
    for group in np.unique(groups):
        selected = groups == group
        block = values[selected]
        output[selected] = (block - block.mean()) / max(block.std(), 1e-12)
    return output


def layerwise_probe(
    features: np.ndarray,
    content_ids: Sequence[str],
    speaker_ids: Sequence[str],
) -> dict[str, np.ndarray | float]:
    """Estimate content and speaker variance carried by every feature column."""
    matrix = np.asarray(features, dtype=np.float64)
    content = np.asarray(content_ids)
    speaker = np.asarray(speaker_ids)
    if matrix.ndim != 2 or not (len(matrix) == len(content) == len(speaker)):
        raise ValueError("features must be [samples, dimensions] with aligned labels")
    grand_mean = matrix.mean(axis=0)
    content_variance = np.zeros(matrix.shape[1])
    speaker_variance = np.zeros(matrix.shape[1])
    for value in np.unique(content):
        block = matrix[content == value]
        content_variance += len(block) * (block.mean(axis=0) - grand_mean) ** 2
    for value in np.unique(speaker):
        block = matrix[speaker == value]
        speaker_variance += len(block) * (block.mean(axis=0) - grand_mean) ** 2
    total_variance = np.sum((matrix - grand_mean) ** 2, axis=0)
    residual = np.maximum(total_variance - content_variance - speaker_variance, 0.0)
    ratio = content_variance / np.maximum(speaker_variance, 1e-12)
    return {
        "content_variance": content_variance,
        "speaker_variance": speaker_variance,
        "residual_variance": residual,
        "content_to_speaker_ratio": ratio,
        "speaker_leak_index": speaker_variance / np.maximum(content_variance + speaker_variance, 1e-12),
        "n_samples": float(len(matrix)),
    }
