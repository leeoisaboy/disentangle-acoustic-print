from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


EPSILON = 1e-12


def l2_normalize(values: np.ndarray, axis: int = -1) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    norm = np.linalg.norm(array, axis=axis, keepdims=True)
    return array / np.maximum(norm, EPSILON)


def _linear_resample(values: np.ndarray, output_length: int) -> np.ndarray:
    frames = len(values)
    if frames == output_length:
        return values.copy()
    positions = np.linspace(0.0, frames - 1.0, output_length)
    lower = np.floor(positions).astype(int)
    upper = np.minimum(lower + 1, frames - 1)
    weight = (positions - lower).reshape((-1,) + (1,) * (values.ndim - 1))
    return values[lower] * (1.0 - weight) + values[upper] * weight


def _interpolate_rows(values: np.ndarray, positions: np.ndarray) -> np.ndarray:
    lower = np.floor(positions).astype(int)
    upper = np.minimum(lower + 1, len(values) - 1)
    weight = (positions - lower).reshape((-1,) + (1,) * (values.ndim - 1))
    return values[lower] * (1.0 - weight) + values[upper] * weight


def _smooth_and_trim_trajectory(
    array: np.ndarray,
    *,
    trim_silence: bool = False,
    trim_quantile: float = 0.15,
    smooth_alpha: float = 0.0,
) -> np.ndarray:
    """Optional zero-phase 1st-order smoothing and lightweight endpoint trimming guard.

    Mitigates high-frequency jitter and leading/trailing silence in noisy real audio
    that otherwise inject false chord length into ``arc_length_resample``.
    """
    out = array
    if len(out) >= 4 and 0.0 < smooth_alpha < 1.0:
        fwd = out.copy()
        for t in range(1, len(fwd)):
            fwd[t] = smooth_alpha * out[t] + (1.0 - smooth_alpha) * fwd[t - 1]
        bwd = fwd.copy()
        for t in range(len(bwd) - 2, -1, -1):
            bwd[t] = smooth_alpha * fwd[t] + (1.0 - smooth_alpha) * bwd[t + 1]
        out = bwd
    if trim_silence and len(out) >= 6:
        edge_ref = 0.5 * (out[0] + out[-1])
        energy = np.linalg.norm(out - edge_ref, axis=1)
        steps = np.linalg.norm(np.diff(out, axis=0), axis=1)
        motion = np.concatenate(([steps[0]], steps))
        score = energy / (float(np.max(energy)) + EPSILON) + motion / (float(np.max(motion)) + EPSILON)
        thresh = float(np.quantile(score, trim_quantile)) + 0.15 * float(np.max(score) - np.min(score))
        active = np.where(score >= thresh)[0]
        if len(active) >= 4:
            s0 = max(0, int(active[0]) - 1)
            s1 = min(len(out), int(active[-1]) + 2)
            if s1 - s0 >= max(4, int(0.3 * len(out))):
                out = out[s0:s1]
    return out


def arc_length_resample(
    frames: np.ndarray,
    output_length: int,
    *,
    trim_silence: bool = False,
    trim_quantile: float = 0.15,
    smooth_alpha: float = 0.0,
) -> np.ndarray:
    """Resample a trajectory at equal increments of cumulative chord length.

    The parameterization depends on the geometry of the path rather than on the
    original frame indices. For a continuously sampled curve it is invariant to
    any monotonic time warp, while remaining deterministic and train-free.

    Set ``trim_silence=True`` and ``smooth_alpha=0.5`` on noisy/padded real speech
    to suppress false arc length from jitter and leading/trailing silence.
    """
    array = np.asarray(frames, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("frames must have shape [time, dimension]")
    if len(array) < 2:
        raise ValueError("arc-length resampling needs at least two frames")
    if output_length < 2:
        raise ValueError("output_length must be at least two")
    if not np.all(np.isfinite(array)):
        raise ValueError("frames must contain finite values")
    if trim_silence or (0.0 < smooth_alpha < 1.0):
        array = _smooth_and_trim_trajectory(
            array,
            trim_silence=trim_silence,
            trim_quantile=trim_quantile,
            smooth_alpha=smooth_alpha,
        )
    steps = np.linalg.norm(np.diff(array, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(steps)))
    total = float(cumulative[-1])
    if total <= EPSILON:
        return _linear_resample(array, output_length)
    targets = np.linspace(0.0, total, output_length)
    positions = np.interp(targets, cumulative, np.arange(len(array), dtype=np.float64))
    return _interpolate_rows(array, positions)


@dataclass(frozen=True)
class SpeakerSubspaceProjector:
    """Remove a low-rank speaker subspace estimated from speaker centroids.

    Supports both explicit ``rank`` and automatic explained-variance rank selection
    (``variance_threshold`` + ``max_rank``) with optional soft-projection ``shrinkage``
    in ``(0, 1]`` to preserve phonetic content on real speech representations.
    """

    mean: np.ndarray
    basis: np.ndarray
    explained_variance_ratio: np.ndarray
    n_speakers: int
    shrinkage: float = 1.0

    @classmethod
    def fit(
        cls,
        sequences: Sequence[np.ndarray],
        speaker_ids: Sequence[str],
        rank: int | None = None,
        variance_threshold: float = 0.95,
        *,
        max_rank: int | None = None,
        shrinkage: float = 1.0,
    ) -> "SpeakerSubspaceProjector":
        if len(sequences) != len(speaker_ids):
            raise ValueError("sequences and speaker ids must have equal length")
        if not sequences:
            raise ValueError("speaker projection needs at least one sequence")
        if not 0.0 <= shrinkage <= 1.0:
            raise ValueError("shrinkage must be in [0, 1]")
        speakers = np.asarray(speaker_ids)
        speaker_means = []
        for speaker in np.unique(speakers):
            frames = np.concatenate(
                [np.asarray(sequence, dtype=np.float64) for sequence, label in zip(sequences, speakers) if label == speaker],
                axis=0,
            )
            if frames.ndim != 2:
                raise ValueError("each sequence must have shape [time, dimension]")
            speaker_means.append(frames.mean(axis=0))
        matrix = np.stack(speaker_means)
        if matrix.shape[0] < 2:
            raise ValueError("speaker subspace estimation needs at least two speakers")
        mean = matrix.mean(axis=0)
        centered = matrix - mean
        covariance = centered.T @ centered / float(len(centered) - 1)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.maximum(eigenvalues[order], 0.0)
        eigenvectors = eigenvectors[:, order]
        total = float(eigenvalues.sum())
        if total <= EPSILON:
            selected = 0
            ratios = np.zeros(0, dtype=np.float64)
        else:
            ratios = eigenvalues / total
            if rank is None:
                cumulative = np.cumsum(ratios)
                selected = int(np.searchsorted(cumulative, variance_threshold) + 1)
            else:
                selected = int(rank)
            if max_rank is not None:
                selected = min(selected, int(max_rank))
            selected = max(0, min(selected, len(eigenvalues), len(centered) - 1))
        return cls(
            mean=mean,
            basis=np.ascontiguousarray(eigenvectors[:, :selected]),
            explained_variance_ratio=ratios[:selected],
            n_speakers=len(speaker_means),
            shrinkage=float(shrinkage),
        )

    def transform(self, frames: np.ndarray) -> np.ndarray:
        array = np.asarray(frames, dtype=np.float64)
        if array.ndim != 2:
            raise ValueError("frames must have shape [time, dimension]")
        if array.shape[1] != len(self.mean):
            raise ValueError("projector dimension does not match frames")
        if not self.basis.size or self.shrinkage <= 0.0:
            return array.copy()
        centered = array - self.mean
        projected = centered - self.shrinkage * ((centered @ self.basis) @ self.basis.T)
        return projected + self.mean


@dataclass(frozen=True)
class TemporalDecomposition:
    centroid: np.ndarray
    delta: np.ndarray
    residual: np.ndarray
    speed: np.ndarray
    acceleration: np.ndarray
    turning: np.ndarray

    @property
    def frames(self) -> int:
        return int(self.speed.shape[0])


def decompose(frames: np.ndarray) -> TemporalDecomposition:
    """Split a trajectory into static, delta, residual, and velocity views."""
    trajectory = np.asarray(frames, dtype=np.float64)
    if trajectory.ndim != 2:
        raise ValueError("frames must have shape [time, dimension]")
    if len(trajectory) < 2:
        raise ValueError("temporal decomposition needs at least two frames")
    centroid = trajectory.mean(axis=0)
    delta = np.diff(trajectory, axis=0)
    residual = trajectory - centroid
    speed = np.linalg.norm(delta, axis=1)
    acceleration = np.diff(delta, axis=0)
    acceleration_norm = np.linalg.norm(acceleration, axis=1)
    unit_delta = l2_normalize(delta)
    turning = np.sum(unit_delta[:-1] * unit_delta[1:], axis=1)
    return TemporalDecomposition(
        centroid=centroid,
        delta=delta,
        residual=residual,
        speed=speed,
        acceleration=acceleration_norm,
        turning=turning,
    )


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    """Compute cosine similarity after flattening both arrays."""
    left_flat = np.asarray(left, dtype=np.float64).reshape(-1)
    right_flat = np.asarray(right, dtype=np.float64).reshape(-1)
    if left_flat.shape != right_flat.shape:
        raise ValueError("cosine inputs must have equal shape")
    denominator = max(
        float(np.linalg.norm(left_flat) * np.linalg.norm(right_flat)), EPSILON
    )
    return float(np.dot(left_flat, right_flat) / denominator)


def reverse(frames: np.ndarray) -> np.ndarray:
    return np.asarray(frames)[::-1].copy()


def half_swap(frames: np.ndarray) -> np.ndarray:
    array = np.asarray(frames)
    middle = len(array) // 2
    return np.concatenate((array[middle:], array[:middle]), axis=0)


def pair_swap(frames: np.ndarray) -> np.ndarray:
    array = np.asarray(frames).copy()
    even = len(array) - len(array) % 2
    array[:even:2], array[1:even:2] = array[1:even:2], array[:even:2].copy()
    return array


def smooth_time_warp(frames: np.ndarray, strength: float = 0.6, seed: int = 17) -> np.ndarray:
    """Apply a deterministic monotone warp; zero strength is the identity."""
    array = np.asarray(frames, dtype=np.float64)
    if len(array) < 3 or abs(strength) <= EPSILON:
        return array.copy()
    knots = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    mapped = np.array(
        [0.0, 0.5 * strength, strength, strength + 0.5 * (1.0 - strength), 1.0]
    )
    mapped[1:-1] += np.random.default_rng(seed).normal(0.0, 0.04, size=3)
    mapped[0], mapped[-1] = 0.0, 1.0
    mapped = np.sort(np.clip(mapped, 0.0, 1.0))
    positions = np.interp(np.arange(len(array)), knots * (len(array) - 1), mapped * (len(array) - 1))
    lower = np.floor(positions).astype(int)
    upper = np.minimum(lower + 1, len(array) - 1)
    weight = (positions - lower).reshape((-1, 1))
    return array[lower] * (1.0 - weight) + array[upper] * weight


def frame_shuffle(frames: np.ndarray, seed: int = 17) -> np.ndarray:
    array = np.asarray(frames)
    return array[np.random.default_rng(seed).permutation(len(array))].copy()


TEMPORAL_CONTROLS = {
    "original": lambda frames, seed: np.asarray(frames).copy(),
    "reverse": lambda frames, seed: reverse(frames),
    "half_swap": lambda frames, seed: half_swap(frames),
    "pair_swap": lambda frames, seed: pair_swap(frames),
    "time_warp": lambda frames, seed: smooth_time_warp(frames, seed=seed),
    "frame_shuffle": lambda frames, seed: frame_shuffle(frames, seed=seed),
}


def static_descriptor(frames: np.ndarray) -> np.ndarray:
    return l2_normalize(decompose(frames).centroid)


def delta_descriptor(
    frames: np.ndarray,
    output_length: int = 8,
    resample: str = "arc",
) -> np.ndarray:
    if resample == "arc":
        sampled = arc_length_resample(frames, output_length + 1)
        delta = np.diff(sampled, axis=0)
    elif resample == "linear":
        delta = _linear_resample(decompose(frames).delta, output_length)
    else:
        raise ValueError("resample must be 'arc' or 'linear'")
    return l2_normalize(delta.reshape(-1))


def residual_descriptor(
    frames: np.ndarray,
    output_length: int = 8,
    resample: str = "arc",
) -> np.ndarray:
    if resample == "arc":
        sampled = arc_length_resample(frames, output_length)
    elif resample == "linear":
        sampled = _linear_resample(decompose(frames).residual, output_length)
    else:
        raise ValueError("resample must be 'arc' or 'linear'")
    residual = sampled - sampled.mean(axis=0)
    return l2_normalize(residual.reshape(-1))


def velocity_descriptor(frames: np.ndarray, output_length: int = 8) -> np.ndarray:
    decomposition = decompose(frames)
    speed = decomposition.speed.reshape(-1, 1)
    return l2_normalize(_linear_resample(speed, output_length).reshape(-1))


DESCRIPTORS = {
    "static": static_descriptor,
    "delta": delta_descriptor,
    "residual": residual_descriptor,
    "velocity": velocity_descriptor,
}


def _mean_cosine(left: np.ndarray, right: np.ndarray) -> float:
    left_unit = l2_normalize(left)
    right_unit = l2_normalize(right)
    return float(np.mean(np.sum(left_unit * right_unit, axis=-1)))


def disentanglement_report(
    sequences: Sequence[np.ndarray],
    content_ids: Sequence[str],
    speaker_ids: Sequence[str],
    descriptors: Iterable[str] = DESCRIPTORS.keys(),
    projector: SpeakerSubspaceProjector | None = None,
) -> dict[str, dict[str, float]]:
    """Compare content stability against speaker leakage for each view."""
    content_array = np.asarray(content_ids)
    speaker_array = np.asarray(speaker_ids)
    if not (len(sequences) == len(content_array) == len(speaker_array)):
        raise ValueError("sequence, content, and speaker arrays must have equal length")
    output: dict[str, dict[str, float]] = {}
    prepared = [projector.transform(sequence) if projector is not None else sequence for sequence in sequences]
    for descriptor_name in descriptors:
        descriptor = DESCRIPTORS[descriptor_name]
        vectors = np.stack([descriptor(sequence) for sequence in prepared])
        same_content_across_speaker = []
        same_speaker_different_content = []
        same_content_same_speaker = []
        for left in range(len(vectors)):
            for right in range(left + 1, len(vectors)):
                cosine = float(vectors[left] @ vectors[right])
                if content_array[left] == content_array[right]:
                    if speaker_array[left] == speaker_array[right]:
                        same_content_same_speaker.append(cosine)
                    else:
                        same_content_across_speaker.append(cosine)
                elif speaker_array[left] == speaker_array[right]:
                    same_speaker_different_content.append(cosine)
        cross_voice = float(np.mean(same_content_across_speaker)) if same_content_across_speaker else 0.0
        speaker_leak = float(np.mean(same_speaker_different_content)) if same_speaker_different_content else 0.0
        within = float(np.mean(same_content_same_speaker)) if same_content_same_speaker else cross_voice
        output[descriptor_name] = {
            "cross_voice_content_similarity": cross_voice,
            "same_speaker_other_content_similarity": speaker_leak,
            "within_speaker_content_similarity": within,
            "content_to_leak_margin": cross_voice - speaker_leak,
            "pairs_cross_voice": len(same_content_across_speaker),
            "pairs_speaker_leak": len(same_speaker_different_content),
        }
    return output


def paired_bootstrap(left: Sequence[float], right: Sequence[float], bootstrap: int = 4000, seed: int = 17) -> dict[str, float]:
    differences = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    if not len(differences):
        raise ValueError("paired bootstrap needs at least one paired sample")
    random = np.random.default_rng(seed)
    samples = random.integers(0, len(differences), size=(bootstrap, len(differences)))
    distribution = differences[samples].mean(axis=1)
    return {
        "n": len(differences),
        "delta": float(differences.mean()),
        "ci_low": float(np.percentile(distribution, 2.5)),
        "ci_high": float(np.percentile(distribution, 97.5)),
        "probability_positive": float(np.mean(distribution > 0.0)),
    }
