"""Real-time audio-to-trajectory bridge for Acoustic Print Studio.

Provides a zero-dependency NumPy ``MelDeltaExtractor`` (converting 16 kHz PCM
streams or ``.wav`` files into ``[time, dimension]`` trajectories in < 1 ms)
alongside an optional ``HiddenStateBridge`` for pretrained encoders (WavLM,
HuBERT, Whisper) when PyTorch/Transformers are installed.
"""

from __future__ import annotations

from dataclasses import dataclass
import io
from pathlib import Path
from typing import Callable, Protocol
import wave

import numpy as np

from .geometry import EPSILON, l2_normalize


class FrameExtractor(Protocol):
    """Protocol for converting a 1D mono waveform into ``[time, dimension]`` frames."""

    @property
    def sample_rate(self) -> int: ...

    @property
    def dimension(self) -> int: ...

    def extract(self, samples: np.ndarray, sample_rate: int | None = None) -> np.ndarray: ...

    def extract_with_envelope(
        self,
        samples: np.ndarray,
        sample_rate: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]: ...


def _hz_to_mel(freq_hz: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(freq_hz, dtype=np.float64) / 700.0)


def _mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10.0 ** (np.asarray(mel, dtype=np.float64) / 2595.0) - 1.0)


def build_mel_filterbank(
    sample_rate: int = 16000,
    n_fft: int = 512,
    n_mels: int = 16,
    fmin: float = 80.0,
    fmax: float | None = None,
) -> np.ndarray:
    """Construct a triangular Mel filterbank matrix of shape ``[n_mels, n_fft // 2 + 1]``."""
    if n_mels < 2:
        raise ValueError("n_mels must be at least two")
    if n_fft < 16:
        raise ValueError("n_fft must be at least 16")
    nyquist = sample_rate / 2.0
    high = min(float(fmax) if fmax is not None else nyquist, nyquist)
    if not (0.0 <= fmin < high):
        raise ValueError("fmin must be non-negative and strictly less than fmax")

    mel_points = np.linspace(_hz_to_mel(fmin), _hz_to_mel(high), n_mels + 2)
    hz_points = _mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
    num_Freq = n_fft // 2 + 1
    bins = np.clip(bins, 0, num_Freq - 1)

    weights = np.zeros((n_mels, num_Freq), dtype=np.float64)
    for m in range(1, n_mels + 1):
        left, center, right = bins[m - 1], bins[m], bins[m + 1]
        if center <= left:
            center = min(left + 1, num_Freq - 1)
        if right <= center:
            right = min(center + 1, num_Freq - 1)
        for k in range(left, center):
            weights[m - 1, k] = (k - left) / max(center - left, 1)
        for k in range(center, right + 1):
            weights[m - 1, k] = (right - k) / max(right - center, 1)
    return weights


def resample_waveform(
    samples: np.ndarray,
    orig_sr: int,
    target_sr: int = 16000,
) -> np.ndarray:
    """Linearly resample a 1D waveform when sample rates differ."""
    if orig_sr <= 0 or target_sr <= 0:
        raise ValueError("sample rates must be positive")
    waveform = np.asarray(samples, dtype=np.float64).reshape(-1)
    if orig_sr == target_sr or len(waveform) < 2:
        return waveform.copy()
    target_len = max(2, int(round(len(waveform) * target_sr / orig_sr)))
    positions = np.linspace(0.0, len(waveform) - 1.0, target_len)
    return np.interp(positions, np.arange(len(waveform), dtype=np.float64), waveform)


def decode_wav_bytes(data: bytes | Path | str) -> tuple[np.ndarray, int]:
    """Decode a mono or multi-channel PCM ``.wav`` file or byte buffer into float64 ``[-1, 1]``."""
    if isinstance(data, (str, Path)):
        raw_bytes = Path(data).read_bytes()
    else:
        raw_bytes = bytes(data)

    with wave.open(io.BytesIO(raw_bytes), "rb") as wf:
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        pcm = wf.readframes(n_frames)

    if sampwidth == 1:
        # 8-bit unsigned PCM
        audio = (np.frombuffer(pcm, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
    elif sampwidth == 2:
        audio = np.frombuffer(pcm, dtype="<i2").astype(np.float64) / 32768.0
    elif sampwidth == 4:
        raw_i32 = np.frombuffer(pcm, dtype="<i4").astype(np.float64)
        # Support either 32-bit int PCM or IEEE float32 WAV
        if np.max(np.abs(raw_i32)) > 2.0:
            audio = raw_i32 / 2147483648.0
        else:
            audio = np.frombuffer(pcm, dtype="<f4").astype(np.float64)
    else:
        raise ValueError(f"unsupported WAV sample width: {sampwidth} bytes")

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio, int(sample_rate)


def encode_wav_bytes(samples: np.ndarray, sample_rate: int = 16000) -> bytes:
    """Encode a 1D float waveform in ``[-1, 1]`` into 16-bit PCM WAV bytes."""
    waveform = np.asarray(samples, dtype=np.float64).reshape(-1)
    clipped = np.clip(waveform, -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm16.tobytes())
    return buffer.getvalue()


class FeatureFrontend:
    """Unified frame-rate-aware frontend contract returning ``(frames, frame_rate_hz)``."""

    @property
    def frame_rate_hz(self) -> float:
        raise NotImplementedError

    @property
    def dimension(self) -> int:
        raise NotImplementedError

    def extract(self, samples: np.ndarray, sample_rate: int | None = None) -> np.ndarray:
        frames, _ = self.extract_with_rate(samples, sample_rate=sample_rate)
        return frames

    def extract_with_rate(
        self,
        samples: np.ndarray,
        sample_rate: int | None = None,
    ) -> tuple[np.ndarray, float]:
        raise NotImplementedError


@dataclass
class MelDeltaExtractor(FeatureFrontend):
    """CPU-only sub-millisecond acoustic feature extractor producing ``[time, dimension]``.

    Converts 16 kHz audio into windowed frames (default hop 20 ms = 50 fps),
    computing log-Mel filterbank energies and optionally appending first-order
    articulatory delta features so that hidden trajectories reflect both formant
    place and rapid consonant transitions.

    Set ``normalize_utterance=False`` for strict chunk-to-offline streaming parity.
    """

    sample_rate: int = 16000
    frame_ms: float = 25.0
    hop_ms: float = 20.0
    n_mels: int = 8
    include_delta: bool = True
    pre_emphasis: float = 0.97
    normalize_utterance: bool = True

    def __post_init__(self) -> None:
        if self.sample_rate < 1000:
            raise ValueError("sample_rate must be at least 1000 Hz")
        if self.frame_ms <= 0.0 or self.hop_ms <= 0.0:
            raise ValueError("frame_ms and hop_ms must be positive")
        self.frame_length = max(16, int(round(self.sample_rate * self.frame_ms / 1000.0)))
        self.hop_length = max(8, int(round(self.sample_rate * self.hop_ms / 1000.0)))
        self.n_fft = max(256, 1 << (self.frame_length - 1).bit_length())
        self._window = np.hanning(self.frame_length).astype(np.float64)
        self._mel_basis = build_mel_filterbank(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
        )

    @property
    def frame_rate_hz(self) -> float:
        return float(self.sample_rate / self.hop_length)

    @property
    def dimension(self) -> int:
        return self.n_mels * 2 if self.include_delta else self.n_mels

    def _frame_signal(self, waveform: np.ndarray) -> np.ndarray:
        if len(waveform) < self.frame_length:
            padded = np.zeros(self.frame_length + self.hop_length, dtype=np.float64)
            padded[: len(waveform)] = waveform
            waveform = padded
        n_frames = 1 + max(0, (len(waveform) - self.frame_length) // self.hop_length)
        if n_frames < 2:
            padded = np.pad(waveform, (0, self.frame_length + self.hop_length - len(waveform)))
            n_frames = 2
            waveform = padded
        indices = (
            np.arange(self.frame_length)[None, :]
            + np.arange(n_frames)[:, None] * self.hop_length
        )
        return waveform[indices]

    def extract_with_envelope(
        self,
        samples: np.ndarray,
        sample_rate: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(frames [T, D], waveform_envelope [T])`` from raw audio samples."""
        raw = np.asarray(samples, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(raw)):
            raise ValueError("audio samples must contain finite values")
        if sample_rate is not None and sample_rate != self.sample_rate:
            raw = resample_waveform(raw, int(sample_rate), self.sample_rate)

        raw_frames = self._frame_signal(raw)
        # Peak-preserving signed RMS envelope per frame for oscilloscope rendering
        rms = np.sqrt(np.mean(raw_frames**2, axis=1) + EPSILON)
        peak_idx = np.argmax(np.abs(raw_frames), axis=1)
        signs = np.sign(raw_frames[np.arange(len(raw_frames)), peak_idx])
        signs = np.where(signs == 0.0, 1.0, signs)
        envelope = rms * signs

        # Pre-emphasis for spectral balance
        emphasized = np.empty_like(raw)
        if len(raw) > 0:
            emphasized[0] = raw[0]
            if len(raw) > 1:
                emphasized[1:] = raw[1:] - self.pre_emphasis * raw[:-1]
        emph_frames = self._frame_signal(emphasized) * self._window[None, :]
        spectrum = np.abs(np.fft.rfft(emph_frames, n=self.n_fft, axis=1)) ** 2
        mel_energy = spectrum @ self._mel_basis.T
        log_mel = np.log(np.maximum(mel_energy, 1e-8))
        if self.normalize_utterance:
            log_mel = log_mel - np.mean(log_mel, axis=0, keepdims=True)

        if not self.include_delta:
            return log_mel, envelope

        delta = np.zeros_like(log_mel)
        delta[1:] = np.diff(log_mel, axis=0)
        delta[0] = delta[1]
        features = np.concatenate((log_mel, delta), axis=1)
        return features, envelope

    def extract(self, samples: np.ndarray, sample_rate: int | None = None) -> np.ndarray:
        """Extract ``[time, dimension]`` acoustic trajectory from 1D audio samples."""
        features, _ = self.extract_with_envelope(samples, sample_rate=sample_rate)
        return features

    def extract_with_rate(
        self,
        samples: np.ndarray,
        sample_rate: int | None = None,
    ) -> tuple[np.ndarray, float]:
        """Return ``(frames [T, D], frame_rate_hz)``."""
        return self.extract(samples, sample_rate=sample_rate), self.frame_rate_hz


class HiddenStateBridge(FeatureFrontend):
    """Adapter that wraps an optional neural encoder (WavLM / HuBERT / Whisper) or callable.

    Falls back cleanly to ``MelDeltaExtractor`` when no neural callable is supplied,
    ensuring zero-threshold CPU execution while supporting deep hidden-state streams.
    """

    def __init__(
        self,
        encoder_fn: Callable[[np.ndarray, int], np.ndarray] | None = None,
        fallback: MelDeltaExtractor | None = None,
        target_dimension: int | None = None,
        frame_rate_hz: float | None = None,
    ) -> None:
        self.encoder_fn = encoder_fn
        self.fallback = fallback or MelDeltaExtractor()
        self.target_dimension = target_dimension
        self._frame_rate_hz = float(frame_rate_hz) if frame_rate_hz is not None else self.fallback.frame_rate_hz

    @property
    def sample_rate(self) -> int:
        return self.fallback.sample_rate

    @property
    def frame_rate_hz(self) -> float:
        return self._frame_rate_hz

    @property
    def dimension(self) -> int:
        return int(self.target_dimension) if self.target_dimension is not None else self.fallback.dimension

    def extract_with_envelope(
        self,
        samples: np.ndarray,
        sample_rate: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        fallback_frames, envelope = self.fallback.extract_with_envelope(
            samples, sample_rate=sample_rate
        )
        if self.encoder_fn is None:
            frames = fallback_frames
        else:
            sr = int(sample_rate or self.sample_rate)
            encoded = np.asarray(self.encoder_fn(samples, sr), dtype=np.float64)
            if encoded.ndim != 2 or len(encoded) < 2:
                raise ValueError("encoder_fn must return a [time, dimension] matrix with time >= 2")
            frames = encoded
            if len(envelope) != len(frames):
                envelope = np.interp(
                    np.linspace(0.0, len(envelope) - 1.0, len(frames)),
                    np.arange(len(envelope), dtype=np.float64),
                    envelope,
                )

        if self.target_dimension is not None and frames.shape[1] != self.target_dimension:
            # Deterministic orthogonal random-projection compression/expansion
            rng = np.random.default_rng(17 + frames.shape[1] * 31 + self.target_dimension)
            proj = rng.normal(size=(frames.shape[1], self.target_dimension))
            proj = l2_normalize(proj, axis=0)
            frames = frames @ proj
        return frames, envelope

    def extract(self, samples: np.ndarray, sample_rate: int | None = None) -> np.ndarray:
        frames, _ = self.extract_with_envelope(samples, sample_rate=sample_rate)
        return frames

    def extract_with_rate(
        self,
        samples: np.ndarray,
        sample_rate: int | None = None,
    ) -> tuple[np.ndarray, float]:
        return self.extract(samples, sample_rate=sample_rate), self.frame_rate_hz


class OnnxEncoderFrontend(FeatureFrontend):
    """Optional ONNX Runtime speech encoder adapter (requires ``onnxruntime``)."""

    def __init__(
        self,
        model_path: str,
        *,
        sample_rate: int = 16000,
        frame_rate_hz: float = 50.0,
        dimension: int = 768,
        input_name: str | None = None,
        output_index: int = 0,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "OnnxEncoderFrontend requires onnxruntime. Install with: pip install 'disentangle-acoustic-print[onnx]'"
            ) from exc
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = input_name or self.session.get_inputs()[0].name
        self.output_index = int(output_index)
        self.sample_rate = int(sample_rate)
        self._frame_rate_hz = float(frame_rate_hz)
        self._dimension = int(dimension)

    @property
    def frame_rate_hz(self) -> float:
        return self._frame_rate_hz

    @property
    def dimension(self) -> int:
        return self._dimension

    def extract_with_rate(
        self,
        samples: np.ndarray,
        sample_rate: int | None = None,
    ) -> tuple[np.ndarray, float]:
        raw = np.asarray(samples, dtype=np.float32).reshape(-1)
        if sample_rate is not None and int(sample_rate) != self.sample_rate:
            raw = resample_waveform(raw.astype(np.float64), int(sample_rate), self.sample_rate).astype(np.float32)
        outputs = self.session.run(None, {self.input_name: raw[None, :]})
        frames = np.asarray(outputs[self.output_index], dtype=np.float64)
        if frames.ndim == 3:
            frames = frames[0]
        return frames, self._frame_rate_hz


class PyTorchSSLFrontend(FeatureFrontend):
    """Optional PyTorch / Hugging Face SSL encoder adapter (WavLM / HuBERT / Whisper)."""

    def __init__(
        self,
        model_name_or_path: str,
        *,
        layer: int = 14,
        sample_rate: int = 16000,
        frame_rate_hz: float = 50.0,
        device: str = "cpu",
    ) -> None:
        try:
            import torch
            from transformers import AutoModel
        except ImportError as exc:
            raise RuntimeError(
                "PyTorchSSLFrontend requires torch and transformers. Install with: pip install 'disentangle-acoustic-print[torch]'"
            ) from exc
        self._torch = torch
        self.device = torch.device(device)
        self.model = AutoModel.from_pretrained(model_name_or_path).to(self.device).eval()
        self.layer = int(layer)
        self.sample_rate = int(sample_rate)
        self._frame_rate_hz = float(frame_rate_hz)
        self._dimension = int(getattr(self.model.config, "hidden_size", 1024))

    @property
    def frame_rate_hz(self) -> float:
        return self._frame_rate_hz

    @property
    def dimension(self) -> int:
        return self._dimension

    def extract_with_rate(
        self,
        samples: np.ndarray,
        sample_rate: int | None = None,
    ) -> tuple[np.ndarray, float]:
        raw = np.asarray(samples, dtype=np.float32).reshape(-1)
        if sample_rate is not None and int(sample_rate) != self.sample_rate:
            raw = resample_waveform(raw.astype(np.float64), int(sample_rate), self.sample_rate).astype(np.float32)
        with self._torch.no_grad():
            inp = self._torch.from_numpy(raw[None, :]).to(self.device)
            out = self.model(inp, output_hidden_states=True)
            hs = out.hidden_states[self.layer][0].float().cpu().numpy().astype(np.float64)
        return hs, self._frame_rate_hz


def synthesize_phrase_waveform(
    phrase: str = "dialect-alpha",
    duration_sec: float = 1.2,
    sample_rate: int = 16000,
    speaking_rate: float = 1.0,
    pitch_shift: float = 1.0,
    seed: int = 17,
) -> np.ndarray:
    """Synthesize a realistic multi-formant syllable waveform for studio testing and demos.

    Produces distinct consonant bursts, vowel nuclei, and speaking-rate variations so
    ``MelDeltaExtractor`` yields repeatable cross-voice and cross-tempo trajectories.
    """
    if duration_sec <= 0.1 or speaking_rate <= 0.0:
        raise ValueError("duration_sec and speaking_rate must be positive")
    effective_duration = float(np.clip(duration_sec / speaking_rate, 0.25, 6.0))
    n_samples = max(640, int(round(effective_duration * sample_rate)))
    t = np.linspace(0.0, effective_duration, n_samples, endpoint=False)
    tau = np.linspace(0.0, 1.0, n_samples, endpoint=False)

    # Deterministic phrase signature from phrase string
    phrase_hash = sum((idx + 1) * ord(ch) for idx, ch in enumerate(phrase))
    rng = np.random.default_rng(seed + phrase_hash * 97)

    f0_base = (125.0 + (phrase_hash % 55)) * float(pitch_shift)
    f1 = 420.0 + ((phrase_hash * 13) % 320)
    f2 = 1350.0 + ((phrase_hash * 29) % 950)
    f3 = 2450.0 + ((phrase_hash * 53) % 700)

    # Two-syllable articulatory envelope with a consonant burst transition
    burst_center = 0.22 + 0.08 * ((phrase_hash % 5) / 4.0)
    second_center = 0.68 + 0.06 * (((phrase_hash // 5) % 5) / 4.0)
    env1 = np.exp(-0.5 * ((tau - burst_center) / 0.11) ** 2)
    env2 = 0.82 * np.exp(-0.5 * ((tau - second_center) / 0.14) ** 2)
    plosive_gate = np.exp(-0.5 * ((tau - (burst_center - 0.09)) / 0.025) ** 2)

    phase0 = 2.0 * np.pi * f0_base * (t + 0.015 * np.sin(2.0 * np.pi * 3.0 * tau))
    voiced = (
        0.55 * np.sin(phase0)
        + 0.40 * np.sin(2.0 * np.pi * f1 * t + 0.3 * np.sin(phase0))
        + 0.28 * np.sin(2.0 * np.pi * f2 * t * (1.0 + 0.12 * (tau - 0.5)))
        + 0.14 * np.sin(2.0 * np.pi * f3 * t)
    )
    burst = rng.normal(scale=0.35, size=n_samples) * plosive_gate
    waveform = (env1 + env2) * voiced + burst + rng.normal(scale=0.008, size=n_samples)
    peak = max(float(np.max(np.abs(waveform))), EPSILON)
    return (0.85 * waveform / peak).astype(np.float64)
