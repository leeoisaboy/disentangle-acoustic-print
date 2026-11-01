from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Sequence

import numpy as np

from .evidence import EvidenceCalibrator, extract_evidence, evidence_vector
from .geometry import (
    SpeakerSubspaceProjector,
    _linear_resample,
    arc_length_resample,
    decompose,
    l2_normalize,
)


@dataclass(frozen=True)
class FingerprintMatch:
    key: str
    score: float
    start: int
    end: int
    fingerprint_score: float | None = None
    evidence_distance: float | None = None


def fingerprint(
    frames: np.ndarray,
    output_length: int = 8,
    resample: str = "arc",
) -> np.ndarray:
    """Build a unit-norm view of static and temporal trajectory parts."""
    if output_length < 2:
        raise ValueError("output_length must be at least two")
    trajectory = _validate(frames)
    decomposition = decompose(trajectory)
    static = l2_normalize(decomposition.centroid)
    if resample == "arc":
        sampled = arc_length_resample(trajectory, output_length + 1)
        delta = np.diff(sampled, axis=0).reshape(-1)
    elif resample == "linear":
        delta = _linear_resample(decomposition.delta, output_length).reshape(-1)
    else:
        raise ValueError("resample must be 'arc' or 'linear'")
    delta = l2_normalize(delta)
    speed = l2_normalize(
        _linear_resample(
            decomposition.speed.reshape(-1, 1), output_length
        ).reshape(-1)
    )
    acceleration = l2_normalize(
        _linear_resample(
            decomposition.acceleration.reshape(-1, 1), max(output_length - 1, 2)
        ).reshape(-1)
    )
    weights = np.sqrt(np.asarray([0.45, 0.20, 0.20, 0.15], dtype=np.float64))
    combined = np.concatenate(
        (
            weights[0] * static,
            weights[1] * delta,
            weights[2] * speed,
            weights[3] * acceleration,
        )
    )
    return l2_normalize(combined)


def _validate(frames: np.ndarray) -> np.ndarray:
    array = np.asarray(frames, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("frames must have shape [time, dimension]")
    if len(array) < 2:
        raise ValueError("fingerprints need at least two frames")
    if not np.all(np.isfinite(array)):
        raise ValueError("frames must contain finite values")
    return array


def temporal_iou(
    left_start: int,
    left_end: int,
    right_start: int,
    right_end: int,
) -> float:
    """Intersection-over-union for two half-open time intervals."""
    overlap = max(0, min(left_end, right_end) - max(left_start, right_start))
    union = max(left_end, right_end) - min(left_start, right_start)
    return float(overlap / union) if union > 0 else 0.0


class EnrollmentBank:
    """A template bank with cosine retrieval and optional evidence reranking."""

    def __init__(
        self,
        output_length: int = 8,
        projector: SpeakerSubspaceProjector | None = None,
        evidence_weight: float = 0.0,
        boundary_weight: float = 0.35,
        resample: str = "arc",
    ):
        if output_length < 2:
            raise ValueError("output_length must be at least two")
        if not 0.0 <= evidence_weight <= 1.0:
            raise ValueError("evidence_weight must be in [0, 1]")
        if resample not in ("arc", "linear"):
            raise ValueError("resample must be 'arc' or 'linear'")
        self.output_length = output_length
        self.projector = projector
        self.evidence_weight = float(evidence_weight)
        self.boundary_weight = float(boundary_weight)
        self.resample = resample
        self._keys: list[str] = []
        self._means: list[np.ndarray] = []
        self._templates: list[np.ndarray] = []
        self._counts: list[int] = []
        self._evidence_means: list[np.ndarray] = []
        self._evidence_samples: list[np.ndarray] = []
        self._evidence_cache: tuple[EvidenceCalibrator, np.ndarray, float] | None = None

    def _prepare(self, frames: np.ndarray) -> np.ndarray:
        array = _validate(frames)
        return self.projector.transform(array) if self.projector is not None else array

    def enroll(self, key: str, frames: Iterable[np.ndarray]) -> None:
        prepared = [self._prepare(sequence) for sequence in frames]
        if not prepared:
            raise ValueError("enrollment needs at least one trajectory")
        vectors = np.stack(
            [fingerprint(sequence, self.output_length, self.resample) for sequence in prepared]
        )
        evidence = np.stack(
            [evidence_vector(extract_evidence(sequence), self.boundary_weight) for sequence in prepared]
        )
        batch_mean = vectors.mean(axis=0)
        batch_evidence = evidence.mean(axis=0)
        count = len(prepared)
        if key in self._keys:
            index = self._keys.index(key)
            total = self._counts[index] + count
            mean = (self._means[index] * self._counts[index] + batch_mean * count) / total
            evidence_mean = (
                self._evidence_means[index] * self._counts[index]
                + batch_evidence * count
            ) / total
            self._means[index] = mean
            self._templates[index] = l2_normalize(mean)
            self._counts[index] = total
            self._evidence_means[index] = evidence_mean
            self._evidence_samples.extend(evidence)
            self._evidence_cache = None
            return
        self._keys.append(key)
        self._means.append(batch_mean)
        self._templates.append(l2_normalize(batch_mean))
        self._counts.append(count)
        self._evidence_means.append(batch_evidence)
        self._evidence_samples.extend(evidence)
        self._evidence_cache = None

    def __len__(self) -> int:
        return len(self._keys)

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(self._keys)

    def count(self, key: str) -> int:
        if key not in self._keys:
            raise KeyError(key)
        return int(self._counts[self._keys.index(key)])

    def matrix(self) -> np.ndarray:
        if not self._templates:
            raise ValueError("enrollment bank is empty")
        return np.stack(self._templates)

    def _calibrated_evidence(
        self,
        prepared: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        if self._evidence_cache is None:
            calibrator = EvidenceCalibrator.from_vectors(
                np.stack(self._evidence_samples), self.boundary_weight
            )
            templates = np.stack(
                [calibrator.transform_vector(vector) for vector in self._evidence_means]
            )
            if len(templates) > 1:
                upper = np.triu_indices(len(templates), 1)
                pair_distances = np.linalg.norm(
                    templates[upper[0]] - templates[upper[1]], axis=1
                )
                scale = float(np.median(pair_distances))
            else:
                scale = 1.0
            scale = max(scale, 1e-8)
            self._evidence_cache = (calibrator, templates, scale)
        calibrator, templates, scale = self._evidence_cache
        query = calibrator.transform(extract_evidence(prepared))
        distances = np.linalg.norm(templates - query, axis=1)
        return np.minimum(distances / scale, 1.0), scale

    def search(
        self,
        frames: np.ndarray,
        top_k: int = 1,
        evidence_weight: float | None = None,
    ) -> list[FingerprintMatch]:
        if top_k < 1:
            raise ValueError("top_k must be at least one")
        prepared = self._prepare(frames)
        query = fingerprint(prepared, self.output_length, self.resample)
        fingerprint_scores = self.matrix() @ query
        weight = self.evidence_weight if evidence_weight is None else float(evidence_weight)
        if not 0.0 <= weight <= 1.0:
            raise ValueError("evidence_weight must be in [0, 1]")
        evidence_distances = None
        scores = fingerprint_scores
        if weight > 0.0:
            evidence_distances, _ = self._calibrated_evidence(prepared)
            scores = (1.0 - weight) * fingerprint_scores - weight * evidence_distances
        order = np.argsort(-scores, kind="stable")[:top_k]
        return [
            FingerprintMatch(
                self._keys[index],
                float(scores[index]),
                0,
                len(prepared),
                fingerprint_score=float(fingerprint_scores[index]),
                evidence_distance=(
                    float(evidence_distances[index])
                    if evidence_distances is not None
                    else None
                ),
            )
            for index in order
        ]


class OnlineMatcher:
    """Multi-scale sliding-window retrieval with temporal non-maximum suppression."""

    def __init__(
        self,
        bank: EnrollmentBank,
        window_length: int | None = None,
        hop: int = 1,
        *,
        window_lengths: Sequence[int] | None = None,
        scales: Sequence[float] | None = None,
        iou_threshold: float = 0.4,
    ):
        if hop < 1:
            raise ValueError("hop must be at least one")
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be in [0, 1]")
        if window_lengths is not None and scales is not None:
            raise ValueError("provide window_lengths or scales, not both")
        if window_lengths is None:
            if window_length is None:
                raise ValueError("window_length or window_lengths is required")
            if window_length < 2:
                raise ValueError("window_length must be at least two")
            factors = (1.0,) if scales is None else tuple(float(value) for value in scales)
            lengths = {max(2, int(round(window_length * factor))) for factor in factors}
        else:
            lengths = {int(value) for value in window_lengths}
            if window_length is not None:
                lengths.add(int(window_length))
        if not lengths or min(lengths) < 2:
            raise ValueError("window lengths must be at least two")
        self.bank = bank
        self.window_length = int(window_length) if window_length is not None else max(lengths)
        self.window_lengths = tuple(sorted(lengths))
        self.hop = hop
        self.iou_threshold = float(iou_threshold)

    def scan(self, frames: np.ndarray) -> list[FingerprintMatch]:
        stream = _validate(frames)
        usable = [length for length in self.window_lengths if len(stream) >= length]
        if not usable:
            raise ValueError("stream is shorter than every matching window")
        matches: list[FingerprintMatch] = []
        for length in usable:
            for start in range(0, len(stream) - length + 1, self.hop):
                best = self.bank.search(stream[start : start + length], top_k=1)[0]
                matches.append(replace(best, start=start, end=start + length))
        return matches

    def match(
        self,
        frames: np.ndarray,
        top_k: int = 1,
        iou_threshold: float | None = None,
    ) -> list[FingerprintMatch]:
        if top_k < 1:
            raise ValueError("top_k must be at least one")
        threshold = self.iou_threshold if iou_threshold is None else float(iou_threshold)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("iou_threshold must be in [0, 1]")
        matches = self.scan(frames)
        matches.sort(key=lambda item: (item.score, -item.start), reverse=True)
        selected: list[FingerprintMatch] = []
        for match in matches:
            if all(
                temporal_iou(match.start, match.end, kept.start, kept.end) <= threshold
                for kept in selected
            ):
                selected.append(match)
            if len(selected) == top_k:
                break
        return selected


def evaluate_top_one(
    bank: EnrollmentBank,
    queries: Sequence[np.ndarray],
    labels: Sequence[str],
    evidence_weight: float | None = None,
) -> float:
    if len(queries) != len(labels):
        raise ValueError("queries and labels must have equal length")
    hits = [
        bool(bank.search(query, evidence_weight=evidence_weight)[0].key == label)
        for query, label in zip(queries, labels)
    ]
    return float(np.mean(hits)) if hits else 0.0
