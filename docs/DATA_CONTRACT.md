# Data Contract

## Trajectory

| Field | Type | Shape | Requirement |
| --- | --- | --- | --- |
| `frames` | `np.ndarray`, `float64` | `[time, dimension]` | At least 2 time steps; finite values |

Each row is one ordered acoustic latent frame. The package does not decode audio or assume a particular model. Keep the extractor, layer index, frame rate, and normalization policy fixed within one experiment.

Use `arc_length_resample(frames, output_length)` when speaking-rate changes should not alter the temporal descriptor. It samples the path at equal cumulative chord length. The original chronological order must still be correct.

```python
import numpy as np
from disentangle_acoustic_print import extract_evidence

frames = np.load("trajectory.npy")
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

## Online Matching

`OnlineMatcher` accepts either `window_lengths=(9, 12, 15)` or `scales=(0.75, 1.0, 1.25)` relative to `window_length`. Returned windows are filtered with temporal IoU NMS. Use `iou_threshold` in `[0, 1]`; `top_k` then means distinct events rather than adjacent offsets.

## Recommended Splits

Use disjoint voice sets for enrollment and query evaluation when measuring cross-voice robustness. Keep repeated utterances from one voice together in a split to avoid inflated scores.

## Deterministic Demo

`make_cross_voice_dataset` and `make_stream` generate synthetic NumPy trajectories with fixed seeds. Cross-voice data includes a speaker offset, a speaker-specific smooth time warp, and a weak orthogonal rotation. `warp_strength` is a non-negative midpoint displacement from the linear schedule; zero disables time warping. `rotation_strength` controls the orthogonal Cayley perturbation. The data are intended for API checks and documentation examples, not as evidence about real speech.
