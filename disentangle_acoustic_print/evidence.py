from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np

from .geometry import cosine, decompose, l2_normalize
from .probing import calibrate_by_group


@dataclass(frozen=True)
class StructuredEvidence:
    response: float
    change_position: float
    primary_peak_position: float
    primary_peak_strength: float
    secondary_peak_position: float
    secondary_peak_strength: float
    speed_mean: float
    speed_std: float
    turning_mean: float
    turning_std: float
    boundary: np.ndarray

    def to_dict(self, include_boundary: bool = False) -> dict[str, object]:
        output = asdict(self)
        output["boundary"] = self.boundary.tolist()
        if not include_boundary:
            output.pop("boundary")
        return output


def _resample(values: np.ndarray, output_length: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(array):
        return np.zeros(output_length, dtype=np.float64)
    positions = np.linspace(0.0, len(array) - 1.0, output_length)
    lower = np.floor(positions).astype(int)
    upper = np.minimum(lower + 1, len(array) - 1)
    weight = positions - lower
    return array[lower] * (1.0 - weight) + array[upper] * weight


def select_peak_indices(
    values: np.ndarray,
    count: int = 2,
    min_distance: int = 2,
) -> list[int]:
    """Return separated local maxima in descending order of strength."""
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(array):
        return []
    if count < 1:
        raise ValueError("count must be at least one")
    min_distance = max(1, int(min_distance))
    candidates: list[int] = []
    if len(array) == 1:
        candidates = [0]
    else:
        if array[0] >= array[1]:
            candidates.append(0)
        for index in range(1, len(array) - 1):
            if array[index] >= array[index - 1] and array[index] > array[index + 1]:
                candidates.append(index)
        if array[-1] >= array[-2]:
            candidates.append(len(array) - 1)
    if not candidates:
        candidates = [int(np.argmax(array))]
    order = sorted(candidates, key=lambda index: (-array[index], index))
    selected: list[int] = []
    for index in order:
        if all(abs(index - previous) >= min_distance for previous in selected):
            selected.append(index)
        if len(selected) == count:
            break
    return selected


def _change_position(speed: np.ndarray) -> float:
    energy = np.asarray(speed, dtype=np.float64).reshape(-1)
    if len(energy) < 2:
        return 0.0
    centered = energy - float(energy.mean())
    cumulative = np.cumsum(centered)
    index = int(np.argmax(np.abs(cumulative)))
    return float(index / max(len(energy) - 1, 1))


def extract_evidence(frames: np.ndarray, boundary_length: int = 8) -> StructuredEvidence:
    """Summarize trajectory evidence without learning or sequence alignment."""
    decomposition = decompose(frames)
    speed = decomposition.speed
    turning = decomposition.turning
    speed_scale = max(float(np.max(speed)), 1e-12)
    min_distance = max(2, int(np.floor(0.15 * len(speed))))
    peaks = select_peak_indices(speed, count=2, min_distance=min_distance)
    positions = np.asarray(peaks, dtype=float) / max(len(speed) - 1, 1)
    strengths = speed[peaks] / speed_scale
    boundary = l2_normalize(_resample(speed, boundary_length))
    return StructuredEvidence(
        response=float(np.mean(speed)),
        change_position=_change_position(speed),
        primary_peak_position=float(positions[0]),
        primary_peak_strength=float(strengths[0]),
        secondary_peak_position=float(positions[1]) if len(positions) > 1 else 0.0,
        secondary_peak_strength=float(strengths[1]) if len(strengths) > 1 else 0.0,
        speed_mean=float(np.mean(speed)),
        speed_std=float(np.std(speed)),
        turning_mean=float(np.mean(turning)) if len(turning) else 0.0,
        turning_std=float(np.std(turning)) if len(turning) else 0.0,
        boundary=boundary,
    )


SCALAR_NAMES = (
    "response",
    "change_position",
    "primary_peak_position",
    "primary_peak_strength",
    "secondary_peak_position",
    "secondary_peak_strength",
    "speed_mean",
    "speed_std",
    "turning_mean",
    "turning_std",
)


def evidence_vector(evidence: StructuredEvidence, boundary_weight: float = 0.35) -> np.ndarray:
    values = np.asarray([getattr(evidence, name) for name in SCALAR_NAMES])
    return np.concatenate((values, evidence.boundary * boundary_weight))


def evidence_distance(
    left: StructuredEvidence,
    right: StructuredEvidence,
    boundary_weight: float = 0.35,
) -> float:
    return float(np.linalg.norm(evidence_vector(left, boundary_weight) - evidence_vector(right, boundary_weight)))


@dataclass(frozen=True)
class EvidenceCalibrator:
    """Per-dimension z-score calibration for structured evidence vectors."""

    mean: np.ndarray
    scale: np.ndarray
    boundary_weight: float = 0.35

    @classmethod
    def from_vectors(
        cls,
        vectors: np.ndarray,
        boundary_weight: float = 0.35,
    ) -> "EvidenceCalibrator":
        matrix = np.asarray(vectors, dtype=np.float64)
        if matrix.ndim != 2 or not len(matrix):
            raise ValueError("evidence vectors must be a non-empty [samples, dimensions] matrix")
        mean = matrix.mean(axis=0)
        scale = matrix.std(axis=0)
        scale = np.where(scale > 1e-8, scale, 1.0)
        return cls(mean=mean, scale=scale, boundary_weight=boundary_weight)

    @classmethod
    def fit(
        cls,
        evidences: Sequence[StructuredEvidence],
        boundary_weight: float = 0.35,
    ) -> "EvidenceCalibrator":
        if not evidences:
            raise ValueError("evidence calibration needs at least one sample")
        return cls.from_vectors(
            np.stack([evidence_vector(item, boundary_weight) for item in evidences]),
            boundary_weight,
        )

    def transform_vector(self, vector: np.ndarray) -> np.ndarray:
        return (np.asarray(vector, dtype=np.float64) - self.mean) / self.scale

    def transform(self, evidence: StructuredEvidence) -> np.ndarray:
        return self.transform_vector(evidence_vector(evidence, self.boundary_weight))

    def distance(self, left: StructuredEvidence, right: StructuredEvidence) -> float:
        return float(np.linalg.norm(self.transform(left) - self.transform(right)))


def calibrated_evidence_distance(
    left: StructuredEvidence,
    right: StructuredEvidence,
    calibrator: EvidenceCalibrator,
) -> float:
    return calibrator.distance(left, right)


def calibrate_response(responses: Sequence[float], group_ids: Sequence[str]) -> np.ndarray:
    return calibrate_by_group(responses, group_ids)


def boundary_similarity(left: StructuredEvidence, right: StructuredEvidence) -> float:
    return cosine(left.boundary, right.boundary)
