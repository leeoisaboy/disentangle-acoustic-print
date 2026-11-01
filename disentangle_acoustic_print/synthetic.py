from __future__ import annotations

from typing import Sequence

import numpy as np

from .geometry import smooth_time_warp


def l2_rows(matrix: np.ndarray) -> np.ndarray:
    array = np.asarray(matrix, dtype=np.float64)
    return array / np.maximum(np.linalg.norm(array, axis=1, keepdims=True), 1e-12)


def make_cross_voice_dataset(
    content_count: int = 4,
    speaker_count: int = 3,
    repeats: int = 3,
    frames: int = 12,
    dimensions: int = 8,
    seed: int = 17,
    warp_strength: float = 0.25,
    rotation_strength: float = 0.08,
) -> dict[str, object]:
    """Create deterministic trajectories with content and speaker nuisance axes.

    In addition to a constant speaker offset, every voice receives a smooth
    local speaking-rate warp and a weak orthogonal rotation. These nuisance
    factors make first-difference removal less artificially perfect and provide
    a validation surface for arc-length resampling and speaker projection.
    """
    if min(content_count, speaker_count, repeats, frames, dimensions) < 1:
        raise ValueError("all dataset dimensions must be positive")
    if frames < 3:
        raise ValueError("synthetic trajectories need at least three frames")
    if warp_strength < 0.0 or rotation_strength < 0.0:
        raise ValueError("warp and rotation strengths must be non-negative")
    random = np.random.default_rng(seed)
    sequences = []
    content_ids = []
    speaker_ids = []
    content_vectors = l2_rows(random.normal(size=(content_count, dimensions)))
    temporal_profiles = [
        np.linspace(0.0, 1.0, frames) ** (0.6 + 0.5 * index)
        for index in range(content_count)
    ]
    speaker_offsets = random.normal(scale=0.35, size=(speaker_count, dimensions))
    speaker_rotations = [
        _weak_orthogonal_rotation(
            dimensions,
            rotation_strength,
            np.random.default_rng(seed + 7919 * (index + 1)),
        )
        for index in range(speaker_count)
    ]
    for content_index in range(content_count):
        for speaker_index in range(speaker_count):
            for repeat_index in range(repeats):
                profile = temporal_profiles[content_index].reshape(-1, 1)
                base = (
                    1.6 * profile * content_vectors[content_index].reshape(1, -1)
                    + speaker_offsets[speaker_index].reshape(1, -1)
                )
                if warp_strength > 0.0:
                    warp_center = float(np.clip(0.5 + warp_strength, 0.1, 0.9))
                    base = smooth_time_warp(
                        base,
                        strength=warp_center,
                        seed=seed + 1009 * (speaker_index + 1),
                    )
                trajectory = base @ speaker_rotations[speaker_index].T
                trajectory += random.normal(scale=0.03, size=(frames, dimensions))
                sequences.append(trajectory)
                content_ids.append(f"phrase-{content_index}")
                speaker_ids.append(f"voice-{speaker_index}")
    return {
        "sequences": sequences,
        "content_ids": content_ids,
        "speaker_ids": speaker_ids,
    }


def _weak_orthogonal_rotation(
    dimensions: int,
    strength: float,
    random: np.random.Generator,
) -> np.ndarray:
    """Return an orthogonal Cayley perturbation of the identity matrix."""
    if strength == 0.0:
        return np.eye(dimensions, dtype=np.float64)
    skew = random.normal(size=(dimensions, dimensions))
    skew = (skew - skew.T) / 2.0
    spectral_norm = float(np.linalg.norm(skew, ord=2))
    if spectral_norm <= 1e-12:
        return np.eye(dimensions, dtype=np.float64)
    skew /= spectral_norm
    identity = np.eye(dimensions, dtype=np.float64)
    return np.linalg.solve(identity + strength * skew, identity - strength * skew)


def make_stream(
    length: int = 45,
    dimensions: int = 8,
    target: Sequence[float] | None = None,
    insert_start: int = 20,
    target_length: int = 12,
    seed: int = 17,
) -> np.ndarray:
    """Create a background trajectory containing one marked temporal event."""
    if length < insert_start + target_length:
        raise ValueError("stream must be long enough for the target window")
    random = np.random.default_rng(seed)
    if target is None:
        target = np.asarray(random.normal(size=(dimensions,)))
    target_array = np.asarray(target, dtype=np.float64)
    if target_array.ndim != 1 or len(target_array) != dimensions:
        raise ValueError("target must be a one-dimensional vector with matching dimension")
    target_array = target_array / max(float(np.linalg.norm(target_array)), 1e-12)
    profile = np.sin(np.linspace(0.0, np.pi, target_length)).reshape(-1, 1)
    stream = 0.02 * random.normal(size=(length, dimensions))
    stream[insert_start : insert_start + target_length] += (
        1.5 * profile * target_array.reshape(1, -1)
    )
    return stream
