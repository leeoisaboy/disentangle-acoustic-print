# Data Contract

## Trajectory & Feature Frontends

| Field | Type | Shape | Requirement |
| --- | --- | --- | --- |
| `frames` | `np.ndarray`, `float64` | `[time, dimension]` | At least 2 time steps; finite values |
| `frame_rate_hz` | `float` | scalar | Frames per second (e.g., `50.0` for WavLM/Whisper, `100.0` for 10 ms Log-Mel) |

Each row is one ordered acoustic latent frame. Core operators are pure NumPy and model-agnostic. Implement `FeatureFrontend.extract_with_rate(samples, sample_rate) -> (frames, frame_rate_hz)` (or use `MelDeltaExtractor`, `HiddenStateBridge`, `OnnxEncoderFrontend`, or `PyTorchSSLFrontend`) to plug in any frontend without changing core code.

Use `arc_length_resample(frames, output_length, trim_silence=True, smooth_alpha=0.5)` on noisy or silence-padded real audio so speaking-rate shifts and leading/trailing noise do not distort the temporal descriptor.

```python
import numpy as np
from disentangle_acoustic_print import MelDeltaExtractor, extract_evidence

frontend = MelDeltaExtractor(n_mels=80, hop_ms=10.0, include_delta=False, normalize_utterance=False)
frames, fps = frontend.extract_with_rate(waveform, sample_rate=16000)
evidence = extract_evidence(frames)
```

## Dataset Arrays

For `disentanglement_report` and `layerwise_probe`:

```python
sequences: list[np.ndarray]        # each item has shape [time, dimension]
content_ids: list[str]             # phrase or phonetic-label identifier
speaker_ids: list[str]             # voice identifier
```

The three sequences must have equal length. Variable time length is supported; hidden dimension must remain fixed.

For `SpeakerSubspaceProjector.fit(sequences, speaker_ids)`, fit only on enrollment or development voices that are allowed to inform the nuisance subspace. Do not fit it on the query speakers being evaluated.

## Enrollment

One enrollment key can receive multiple trajectories:

```python
bank.enroll("phrase-a", [sample_1, sample_2, sample_3])
```

Repeated calls for the same key use cumulative sample counts:

```python
bank.enroll("phrase-a", [sample_1])
bank.enroll("phrase-a", [sample_2, sample_3])  # equivalent to one batch of three
```

Samples for one key should share the extractor, layer, dimension, normalization policy, and frame-rate range. A query must use the same setup as enrollment. The bank stores both a unit-norm fingerprint and a structured-evidence vector. Set `evidence_weight` in `[0, 1]` to trade cosine retrieval against z-scored evidence distance.

## Online Matching & Background Calibration

`OnlineMatcher` accepts either frame-based `window_lengths=(9, 12, 15)` / `window_length=12`, template-adaptive `window_length="auto"`, or frame-rate-agnostic millisecond sizing (`window_ms=400.0, hop_ms=40.0, frame_rate_hz=50.0`) with `scales=(0.75, 1.0, 1.25)`. Returned windows are filtered with temporal IoU NMS (`iou_threshold` in `[0, 1]`).

For open continuous streams, call `bank.calibrate_background(neg_streams, target_fa_per_hour=1.0, frame_rate_hz=50.0)` on unlabeled background audio to compute per-keyword z-score normalization and dynamic FA/hr trigger thresholds, then enable `OnlineMatcher(..., use_calibration=True)`.

Note: Real-data ablations across Shanghainese, multi-dialect Mandarin, and English show that `evidence_weight=0.0` (pure geometric fingerprint cosine) is optimal for online retrieval; `extract_evidence()` is retained as the white-box diagnostic and 4-track telemetry layer.

## Recommended Splits

Use disjoint voice sets for enrollment and query evaluation when measuring cross-voice robustness. Keep repeated utterances from one voice together in a split to avoid inflated scores.

## Deterministic Demo

`make_cross_voice_dataset` and `make_stream` generate synthetic NumPy trajectories with fixed seeds. Cross-voice data includes a speaker offset, a speaker-specific smooth time warp, and a weak orthogonal rotation. `warp_strength` is a non-negative midpoint displacement from the linear schedule; zero disables time warping. `rotation_strength` controls the orthogonal Cayley perturbation. The data are intended for API checks and documentation examples, not as evidence about real speech.
