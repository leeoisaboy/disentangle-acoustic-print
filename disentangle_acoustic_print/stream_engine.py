"""Real-time rolling buffer and streaming engine for Acoustic Print Studio.

Maintains a fixed-capacity rolling trajectory and waveform buffer (e.g. 4 seconds
= 200 frames at 50 fps), supports 3-second dual-tempo voice enrollment, speaker
subspace projection fitting, and sub-millisecond sliding-window telemetry updates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Iterable, Sequence

import numpy as np

from .audio_bridge import HiddenStateBridge, MelDeltaExtractor, synthesize_phrase_waveform
from .fingerprint import EnrollmentBank, OnlineMatcher
from .geometry import SpeakerSubspaceProjector, smooth_time_warp
from .synthetic import make_cross_voice_dataset
from .telemetry import build_telemetry_packet


@dataclass
class RollingAcousticStream:
    """Rolling ring-buffer engine coupling audio/feature streams with ``OnlineMatcher``."""

    max_frames: int = 120
    dimension: int = 16
    output_length: int = 8
    evidence_weight: float = 0.0
    window_length: int = 16
    scales: tuple[float, ...] = (0.75, 1.0, 1.25)
    hop: int = 2
    iou_threshold: float = 0.4
    match_threshold: float = 0.42
    frame_rate_hz: float = 50.0
    window_ms: float | None = None
    hop_ms: float | None = None
    extractor: MelDeltaExtractor | HiddenStateBridge | None = None

    _frames: np.ndarray = field(init=False, repr=False)
    _waveform: np.ndarray = field(init=False, repr=False)
    bank: EnrollmentBank = field(init=False)
    matcher: OnlineMatcher = field(init=False)
    projector: SpeakerSubspaceProjector | None = field(default=None, init=False)
    _speaker_Calibration_sequences: list[np.ndarray] = field(default_factory=list, init=False, repr=False)
    _speaker_calibration_ids: list[str] = field(default_factory=list, init=False, repr=False)
    _enrolled_raw: dict[str, list[np.ndarray]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.max_frames < 12:
            raise ValueError("max_frames must be at least 12")
        if self.extractor is None:
            n_mels = max(2, self.dimension // 2)
            hop_ms = 1000.0 / max(self.frame_rate_hz, 1.0)
            self.extractor = MelDeltaExtractor(
                n_mels=n_mels,
                hop_ms=hop_ms,
                include_delta=(self.dimension % 2 == 0),
                normalize_utterance=False,
            )
            self.dimension = self.extractor.dimension
            self.frame_rate_hz = self.extractor.frame_rate_hz
        elif hasattr(self.extractor, "frame_rate_hz"):
            self.frame_rate_hz = float(self.extractor.frame_rate_hz)

        if self.window_ms is not None:
            self.window_length = max(2, int(round(self.window_ms * self.frame_rate_hz / 1000.0)))
        if self.hop_ms is not None:
            self.hop = max(1, int(round(self.hop_ms * self.frame_rate_hz / 1000.0)))

        rng = np.random.default_rng(17)
        self._frames = 0.015 * rng.normal(size=(self.max_frames, self.dimension))
        self._waveform = 0.01 * rng.normal(size=(self.max_frames,))
        self._rebuild_bank_and_matcher()

    def _rebuild_bank_and_matcher(self) -> None:
        self.bank = EnrollmentBank(
            output_length=self.output_length,
            projector=self.projector,
            evidence_weight=self.evidence_weight,
            resample="arc",
        )
        for key, sequences in self._enrolled_raw.items():
            if sequences:
                self.bank.enroll(key, sequences)
        self.matcher = OnlineMatcher(
            self.bank,
            window_length=self.window_length,
            scales=self.scales,
            hop=self.hop,
            iou_threshold=self.iou_threshold,
        )

    @property
    def frames(self) -> np.ndarray:
        return self._frames.copy()

    @property
    def waveform(self) -> np.ndarray:
        return self._waveform.copy()

    def reset_buffer(self, seed: int = 17) -> None:
        """Reset the rolling buffer to a quiet background floor."""
        rng = np.random.default_rng(seed)
        self._frames = 0.015 * rng.normal(size=(self.max_frames, self.dimension))
        self._waveform = 0.01 * rng.normal(size=(self.max_frames,))

    def enroll_trajectories(
        self,
        key: str,
        sequences: Iterable[np.ndarray],
        speaker_id: str | None = None,
    ) -> dict[str, Any]:
        """Enroll one or more ``[time, dimension]`` trajectories under ``key``."""
        cleaned = []
        for seq in sequences:
            arr = np.asarray(seq, dtype=np.float64)
            if arr.ndim != 2 or arr.shape[1] != self.dimension or len(arr) < 2:
                raise ValueError(
                    f"enrollment trajectory must have shape [time >= 2, {self.dimension}]"
                )
            cleaned.append(arr)
            if speaker_id is not None:
                self._speaker_Calibration_sequences.append(arr)
                self._speaker_calibration_ids.append(speaker_id)

        if not cleaned:
            raise ValueError("enroll_trajectories requires at least one sequence")

        self._enrolled_raw.setdefault(key, []).extend(cleaned)
        self.bank.enroll(key, cleaned)
        return {
            "key": key,
            "enrolled_count": self.bank.count(key),
            "all_keys": list(self.bank.keys),
        }

    def enroll_audio(
        self,
        key: str,
        audio_chunks: Sequence[np.ndarray],
        sample_rate: int | None = None,
        speaker_id: str | None = None,
    ) -> dict[str, Any]:
        """Mode 1 (3-second real-time voice registration): extract and enroll normal/slow takes."""
        assert self.extractor is not None
        trajectories = [
            self.extractor.extract(chunk, sample_rate=sample_rate)
            for chunk in audio_chunks
        ]
        return self.enroll_trajectories(key, trajectories, speaker_id=speaker_id)

    def fit_speaker_projector(
        self,
        sequences: Sequence[np.ndarray] | None = None,
        speaker_ids: Sequence[str] | None = None,
        rank: int | None = 2,
    ) -> SpeakerSubspaceProjector:
        """Fit and activate ``SpeakerSubspaceProjector`` and re-index all enrolled templates."""
        seqs = list(sequences) if sequences is not None else self._speaker_Calibration_sequences
        spk_ids = list(speaker_ids) if speaker_ids is not None else self._speaker_calibration_ids
        self.projector = SpeakerSubspaceProjector.fit(seqs, spk_ids, rank=rank)
        self._rebuild_bank_and_matcher()
        return self.projector

    def toggle_speaker_projector(self, enabled: bool) -> bool:
        """Enable or disable the fitted ``SpeakerSubspaceProjector`` on the active bank."""
        if enabled and self.projector is None and len(set(self._speaker_calibration_ids)) >= 2:
            self.fit_speaker_projector()
        elif not enabled:
            self.bank.projector = None
        else:
            self.bank.projector = self.projector
        self._rebuild_bank_and_matcher()
        return self.bank.projector is not None

    def push_frames(
        self,
        new_frames: np.ndarray,
        new_waveform: np.ndarray | None = None,
        focus_window: tuple[int, int] | None = None,
    ) -> dict[str, Any]:
        """Append ``[chunk_time, dimension]`` frames into the rolling buffer and emit telemetry."""
        t0 = time.perf_counter()
        arr = np.asarray(new_frames, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != self.dimension:
            raise ValueError(f"new_frames must have shape [time, {self.dimension}]")

        n_new = len(arr)
        if new_waveform is None:
            speed_step = np.linalg.norm(np.diff(arr, axis=0, prepend=arr[:1]), axis=1)
            wave_chunk = np.tanh(arr[:, 0] * 0.6) * (0.25 + 0.75 * np.tanh(speed_step))
        else:
            wave_chunk = np.asarray(new_waveform, dtype=np.float64).reshape(-1)
            if len(wave_chunk) != n_new:
                wave_chunk = np.interp(
                    np.linspace(0.0, len(wave_chunk) - 1.0, n_new),
                    np.arange(len(wave_chunk), dtype=np.float64),
                    wave_chunk,
                )

        if n_new >= self.max_frames:
            self._frames = arr[-self.max_frames :].copy()
            self._waveform = wave_chunk[-self.max_frames :].copy()
        else:
            self._frames = np.vstack((self._frames[n_new:], arr))
            self._waveform = np.concatenate((self._waveform[n_new:], wave_chunk))

        telemetry = build_telemetry_packet(
            self._frames,
            waveform_chunk=self._waveform,
            matcher=self.matcher,
            bank=self.bank,
            projector=self.bank.projector,
            output_length=self.output_length,
            match_threshold=self.match_threshold,
            focus_window=focus_window,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        telemetry["compute_latency_ms"] = round(elapsed_ms, 4)
        return telemetry

    def push_audio_chunk(
        self,
        samples: np.ndarray,
        sample_rate: int | None = None,
    ) -> dict[str, Any]:
        """Push a raw PCM audio chunk (e.g. 100ms microphone slice or WAV buffer) and emit telemetry."""
        assert self.extractor is not None
        frames, envelope = self.extractor.extract_with_envelope(samples, sample_rate=sample_rate)
        return self.push_frames(frames, new_waveform=envelope)

    def load_default_dialect_presets(self, seed: int = 17) -> dict[str, Any]:
        """Populate default cross-voice dialect/keyword templates and speaker subspace."""
        dataset = make_cross_voice_dataset(
            content_count=4,
            speaker_count=3,
            repeats=3,
            frames=self.window_length,
            dimensions=self.dimension,
            seed=seed,
            warp_strength=0.28,
            rotation_strength=0.08,
        )
        label_map = {
            "phrase-0": "吴语·侬好 (Wu Dialect)",
            "phrase-1": "粤语·唔该 (Cantonese)",
            "phrase-2": "川渝·巴适 (Sichuanese)",
            "phrase-3": "闽南·多谢 (Hokkien)",
        }
        self._enrolled_raw.clear()
        self._speaker_Calibration_sequences = list(dataset["sequences"])
        self._speaker_calibration_ids = list(dataset["speaker_ids"])
        self.projector = SpeakerSubspaceProjector.fit(
            self._speaker_Calibration_sequences,
            self._speaker_calibration_ids,
            rank=2,
        )
        self._rebuild_bank_and_matcher()
        for seq, cid in zip(dataset["sequences"], dataset["content_ids"]):
            mapped_key = label_map.get(str(cid), str(cid))
            self._enrolled_raw.setdefault(mapped_key, []).append(seq)
            self.bank.enroll(mapped_key, [seq])

        return {
            "enrolled_keys": list(self.bank.keys),
            "speaker_count": self.projector.n_speakers,
            "subspace_rank": int(self.projector.basis.shape[1]),
        }

    def inject_synthetic_utterance(
        self,
        key: str | None = None,
        speaking_rate: float = 1.0,
        speaker_nuisance: float = 0.35,
        seed: int = 23,
    ) -> dict[str, Any]:
        """Inject a target dialect phrase (with optional speaking-rate warp & voice shift) into the stream."""
        if not self._enrolled_raw:
            self.load_default_dialect_presets(seed=seed)

        target_key = key if (key in self._enrolled_raw) else next(iter(self._enrolled_raw.keys()))
        base_seq = self._enrolled_raw[target_key][0].copy()

        rng = np.random.default_rng(seed)
        # Apply time-stretching / speaking-rate change
        stretched_len = max(6, int(round(len(base_seq) / max(speaking_rate, 0.35))))
        positions = np.linspace(0.0, len(base_seq) - 1.0, stretched_len)
        lower = np.floor(positions).astype(int)
        upper = np.minimum(lower + 1, len(base_seq) - 1)
        weight = (positions - lower).reshape(-1, 1)
        utterance = base_seq[lower] * (1.0 - weight) + base_seq[upper] * weight

        if abs(speaking_rate - 1.0) > 0.05:
            warp_val = float(np.clip(0.5 + 0.25 * (speaking_rate - 1.0), 0.15, 0.85))
            utterance = smooth_time_warp(utterance, strength=warp_val, seed=seed)

        # Inject speaker nuisance in the projector's subspace if available
        if speaker_nuisance > 0.0 and self.projector is not None and self.projector.basis.size > 0:
            nuisance_vec = self.projector.basis[:, 0] * float(speaker_nuisance) * 1.8
            utterance = utterance + nuisance_vec.reshape(1, -1)

        pad_before = 0.02 * rng.normal(size=(8, self.dimension))
        pad_after = 0.02 * rng.normal(size=(10, self.dimension))
        chunk = np.vstack((pad_before, utterance, pad_after))
        return self.push_frames(chunk)
