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
    calibrated_z: float | None = None


@dataclass(frozen=True)
class BackgroundCalibration:
    """Per-keyword background score distribution and dynamic FA/hr trigger threshold."""

    key: str
    mean: float
    std: float
    threshold_z: float
    threshold_raw: float
    target_fa_per_hour: float
    n_windows: int


def fingerprint(
    frames: np.ndarray,
    output_length: int = 8,
    resample: str = "arc",
    *,
    trim_silence: bool = False,
    smooth_alpha: float = 0.0,
    preset: str = "legacy",
) -> np.ndarray:
    """Build a unit-norm view of static and temporal trajectory parts.

    Parameters
    ----------
    preset:
        ``"legacy"`` (default) uses ``(0.45 static, 0.20 delta, 0.20 speed, 0.15 accel)``
        for full compatibility with the 4-track Studio barcode.
        ``"content"`` uses ``(0.45 static, 0.35 arc-resampled residual, 0.20 delta)``
        validated on real multi-dialect speech.
    """
    if output_length < 2:
        raise ValueError("output_length must be at least two")
    trajectory = _validate(frames)
    decomposition = decompose(trajectory)
    static = l2_normalize(decomposition.centroid)
    if resample == "arc":
        sampled = arc_length_resample(
            trajectory,
            output_length + 1,
            trim_silence=trim_silence,
            smooth_alpha=smooth_alpha,
        )
        delta = np.diff(sampled, axis=0).reshape(-1)
    elif resample == "linear":
        delta = _linear_resample(decomposition.delta, output_length).reshape(-1)
    else:
        raise ValueError("resample must be 'arc' or 'linear'")
    delta = l2_normalize(delta)

    if preset == "content":
        res_source = (
            arc_length_resample(
                decomposition.residual,
                output_length,
                trim_silence=trim_silence,
                smooth_alpha=smooth_alpha,
            )
            if resample == "arc"
            else _linear_resample(decomposition.residual, output_length)
        )
        residual = l2_normalize(res_source.reshape(-1))
        weights = np.sqrt(np.asarray([0.45, 0.35, 0.20], dtype=np.float64))
        combined = np.concatenate((weights[0] * static, weights[1] * residual, weights[2] * delta))
        return l2_normalize(combined)
    if preset != "legacy":
        raise ValueError("preset must be 'legacy' or 'content'")

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
    """A template bank with cosine retrieval, background calibration, and optional evidence reranking."""

    def __init__(
        self,
        output_length: int = 8,
        projector: SpeakerSubspaceProjector | None = None,
        evidence_weight: float = 0.0,
        boundary_weight: float = 0.35,
        resample: str = "arc",
        *,
        trim_silence: bool = False,
        smooth_alpha: float = 0.0,
        preset: str = "legacy",
    ):
        if output_length < 2:
            raise ValueError("output_length must be at least two")
        if not 0.0 <= evidence_weight <= 1.0:
            raise ValueError("evidence_weight must be in [0, 1]")
        if resample not in ("arc", "linear"):
            raise ValueError("resample must be 'arc' or 'linear'")
        if preset not in ("legacy", "content"):
            raise ValueError("preset must be 'legacy' or 'content'")
        self.output_length = output_length
        self.projector = projector
        self.evidence_weight = float(evidence_weight)
        self.boundary_weight = float(boundary_weight)
        self.resample = resample
        self.trim_silence = bool(trim_silence)
        self.smooth_alpha = float(smooth_alpha)
        self.preset = preset
        self._keys: list[str] = []
        self._means: list[np.ndarray] = []
        self._templates: list[np.ndarray] = []
        self._counts: list[int] = []
        self._durations: dict[str, list[int]] = {}
        self._evidence_means: list[np.ndarray] = []
        self._evidence_samples: list[np.ndarray] = []
        self._evidence_cache: tuple[EvidenceCalibrator, np.ndarray, float] | None = None
        self._background_calibration: dict[str, BackgroundCalibration] = {}

    def _prepare(self, frames: np.ndarray) -> np.ndarray:
        array = _validate(frames)
        return self.projector.transform(array) if self.projector is not None else array

    def _embed(self, prepared: np.ndarray) -> np.ndarray:
        return fingerprint(
            prepared,
            self.output_length,
            self.resample,
            trim_silence=self.trim_silence,
            smooth_alpha=self.smooth_alpha,
            preset=self.preset,
        )

    def enroll(self, key: str, frames: Iterable[np.ndarray]) -> None:
        prepared = [self._prepare(sequence) for sequence in frames]
        if not prepared:
            raise ValueError("enrollment needs at least one trajectory")
        vectors = np.stack([self._embed(sequence) for sequence in prepared])
        evidence = np.stack(
            [evidence_vector(extract_evidence(sequence), self.boundary_weight) for sequence in prepared]
        )
        self._durations.setdefault(key, []).extend(len(seq) for seq in prepared)
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

    def median_template_length(self, key: str | None = None) -> int:
        """Return median enrolled trajectory length (in frames) for one key or the full bank."""
        if key is not None:
            if key not in self._durations or not self._durations[key]:
                raise KeyError(key)
            return max(2, int(round(float(np.median(self._durations[key])))))
        all_durs = [d for vals in self._durations.values() for d in vals]
        if not all_durs:
            raise ValueError("enrollment bank is empty")
        return max(2, int(round(float(np.median(all_durs)))))

    @property
    def is_calibrated(self) -> bool:
        return bool(self._background_calibration) and all(
            k in self._background_calibration for k in self._keys
        )

    @property
    def background_calibration(self) -> dict[str, BackgroundCalibration]:
        return dict(self._background_calibration)

    def calibrated_score(self, key: str, raw_score: float) -> float:
        cal = self._background_calibration.get(key)
        if cal is None:
            return float(raw_score)
        return float((raw_score - cal.mean) / max(cal.std, 1e-6))

    def threshold_for(self, key: str) -> float | None:
        cal = self._background_calibration.get(key)
        return cal.threshold_z if cal is not None else None

    def calibrate_background(
        self,
        neg_streams: Sequence[np.ndarray],
        *,
        window_lengths: Sequence[int] | None = None,
        scales: Sequence[float] = (0.75, 1.0, 1.25),
        hop: int = 2,
        target_fa_per_hour: float = 1.0,
        frame_rate_hz: float = 50.0,
    ) -> dict[str, BackgroundCalibration]:
        """Estimate per-keyword background distributions and dynamic FA/hr thresholds.

        Scans unlabeled background streams (containing no target keywords) to compute
        each keyword's background score mean, standard deviation, and operating z-score
        threshold for ``target_fa_per_hour``.
        """
        if not self._keys:
            raise ValueError("enroll at least one keyword before calibrating background")
        if not neg_streams:
            raise ValueError("neg_streams must contain at least one background stream")
        if target_fa_per_hour < 0.0:
            raise ValueError("target_fa_per_hour must be non-negative")

        if window_lengths is None:
            med = self.median_template_length()
            wlens = tuple(sorted({max(2, int(round(med * float(s)))) for s in scales}))
        else:
            wlens = tuple(sorted({max(2, int(w)) for w in window_lengths}))

        bank_mat = self.matrix()
        all_blocks: list[np.ndarray] = []
        total_frames = 0
        for raw_stream in neg_streams:
            stream = self._prepare(raw_stream)
            total_frames += len(stream)
            fps_list: list[np.ndarray] = []
            for wl in wlens:
                if len(stream) < wl:
                    continue
                for start in range(0, len(stream) - wl + 1, max(1, int(hop))):
                    fps_list.append(self._embed(stream[start : start + wl]))
            if fps_list:
                Q = np.stack(fps_list, axis=0)
                all_blocks.append(Q @ bank_mat.T)

        if not all_blocks:
            raise ValueError("background streams are shorter than every calibration window")

        scores_mat = np.vstack(all_blocks)  # [N_win, K]
        total_hours = max(total_frames / max(float(frame_rate_hz), 1.0) / 3600.0, 1e-6)
        allowed_fa = int(np.floor(target_fa_per_hour * total_hours))
        n_win = scores_mat.shape[0]

        cal_map: dict[str, BackgroundCalibration] = {}
        for idx, key in enumerate(self._keys):
            col = scores_mat[:, idx]
            mu = float(np.mean(col))
            sigma = max(float(np.std(col)), 1e-6)
            z_col = np.sort((col - mu) / sigma)[::-1]
            if allowed_fa < len(z_col):
                th_z = float(z_col[allowed_fa])
            else:
                th_z = float(z_col[-1])
            cal_map[key] = BackgroundCalibration(
                key=key,
                mean=mu,
                std=sigma,
                threshold_z=th_z,
                threshold_raw=float(mu + th_z * sigma),
                target_fa_per_hour=float(target_fa_per_hour),
                n_windows=int(n_win),
            )
        self._background_calibration = cal_map
        return dict(cal_map)

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
        use_calibration: bool = False,
    ) -> list[FingerprintMatch]:
        if top_k < 1:
            raise ValueError("top_k must be at least one")
        prepared = self._prepare(frames)
        query = self._embed(prepared)
        fingerprint_scores = self.matrix() @ query
        weight = self.evidence_weight if evidence_weight is None else float(evidence_weight)
        if not 0.0 <= weight <= 1.0:
            raise ValueError("evidence_weight must be in [0, 1]")
        evidence_distances = None
        scores = fingerprint_scores
        if weight > 0.0:
            evidence_distances, _ = self._calibrated_evidence(prepared)
            scores = (1.0 - weight) * fingerprint_scores - weight * evidence_distances

        cal_zs: np.ndarray | None = None
        if use_calibration and self.is_calibrated:
            cal_zs = np.asarray(
                [self.calibrated_score(self._keys[i], float(scores[i])) for i in range(len(self._keys))],
                dtype=np.float64,
            )
            ranking_scores = cal_zs
        else:
            ranking_scores = scores

        order = np.argsort(-ranking_scores, kind="stable")[:top_k]
        return [
            FingerprintMatch(
                self._keys[index],
                float(ranking_scores[index]),
                0,
                len(prepared),
                fingerprint_score=float(fingerprint_scores[index]),
                evidence_distance=(
                    float(evidence_distances[index])
                    if evidence_distances is not None
                    else None
                ),
                calibrated_z=float(cal_zs[index]) if cal_zs is not None else None,
            )
            for index in order
        ]


class OnlineMatcher:
    """Multi-scale sliding-window retrieval with temporal non-maximum suppression.

    Supports both frame-based (``window_length``, ``hop``) and frame-rate-agnostic
    millisecond sizing (``window_ms``, ``hop_ms``, ``frame_rate_hz``), as well as
    ``window_length="auto"`` derived from the enrolled template median duration.
    """

    def __init__(
        self,
        bank: EnrollmentBank,
        window_length: int | str | None = None,
        hop: int | None = None,
        *,
        window_lengths: Sequence[int] | None = None,
        scales: Sequence[float] | None = None,
        iou_threshold: float = 0.4,
        frame_rate_hz: float | None = None,
        window_ms: float | None = None,
        hop_ms: float | None = None,
        use_calibration: bool = False,
    ):
        if hop is None:
            if hop_ms is not None and frame_rate_hz is not None:
                hop = max(1, int(round(float(hop_ms) * float(frame_rate_hz) / 1000.0)))
            else:
                hop = 1
        if hop < 1:
            raise ValueError("hop must be at least one")
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be in [0, 1]")
        if window_lengths is not None and scales is not None:
            raise ValueError("provide window_lengths or scales, not both")

        resolved_wl: int | None = None
        if isinstance(window_length, str):
            if window_length != "auto":
                raise ValueError("window_length string must be 'auto'")
            resolved_wl = bank.median_template_length()
        elif window_length is not None:
            resolved_wl = int(window_length)
        elif window_ms is not None and frame_rate_hz is not None:
            resolved_wl = max(2, int(round(float(window_ms) * float(frame_rate_hz) / 1000.0)))

        if window_lengths is None:
            if resolved_wl is None:
                raise ValueError("window_length, window_lengths, or (window_ms + frame_rate_hz) is required")
            if resolved_wl < 2:
                raise ValueError("window_length must be at least two")
            factors = (1.0,) if scales is None else tuple(float(value) for value in scales)
            lengths = {max(2, int(round(resolved_wl * factor))) for factor in factors}
        else:
            lengths = {int(value) for value in window_lengths}
            if resolved_wl is not None:
                lengths.add(int(resolved_wl))
        if not lengths or min(lengths) < 2:
            raise ValueError("window lengths must be at least two")
        self.bank = bank
        self.window_length = int(resolved_wl) if resolved_wl is not None else max(lengths)
        self.window_lengths = tuple(sorted(lengths))
        self.hop = int(hop)
        self.iou_threshold = float(iou_threshold)
        self.frame_rate_hz = float(frame_rate_hz) if frame_rate_hz is not None else None
        self.use_calibration = bool(use_calibration)

    def scan(self, frames: np.ndarray) -> list[FingerprintMatch]:
        stream = _validate(frames)
        usable = [length for length in self.window_lengths if len(stream) >= length]
        if not usable:
            raise ValueError("stream is shorter than every matching window")
        matches: list[FingerprintMatch] = []
        for length in usable:
            for start in range(0, len(stream) - length + 1, self.hop):
                best = self.bank.search(
                    stream[start : start + length],
                    top_k=1,
                    use_calibration=self.use_calibration,
                )[0]
                matches.append(replace(best, start=start, end=start + length))
        return matches

    def match(
        self,
        frames: np.ndarray,
        top_k: int = 1,
        iou_threshold: float | None = None,
        min_score: float | None = None,
    ) -> list[FingerprintMatch]:
        if top_k < 1:
            raise ValueError("top_k must be at least one")
        threshold = self.iou_threshold if iou_threshold is None else float(iou_threshold)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("iou_threshold must be in [0, 1]")
        matches = self.scan(frames)
        if min_score is not None:
            matches = [m for m in matches if m.score >= min_score]
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
