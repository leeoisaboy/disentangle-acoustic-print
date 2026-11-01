<p align="center">
  <a href="https://github.com/leeoisaboy/disentangle-acoustic-print">
    <img src="docs/assets/title_banner.en.png" alt="Disentangle-Acoustic-Print" />
  </a>
</p>

<p align="center">
  <a href="README.md"><img alt="中文" src="https://img.shields.io/badge/%E4%B8%AD%E6%96%87-README-d73a49?style=flat-square"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-3776ab?style=flat-square">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-555555?style=flat-square">
  <img alt="Tests" src="https://img.shields.io/badge/tests-20%20passing-2ea44f?style=flat-square">
</p>

<h1 align="center">Disentangle-Acoustic-Print</h1>

<p align="center"><strong>Turn a cross-voice speech representation into interpretable temporal views.</strong></p>

<img align="right" width="210" src="docs/assets/logo.png" alt="Disentangle-Acoustic-Print logo" />

## Why This Project

A dialect command changes when another person says it. The words stay the same, while timbre and speaking rate move. A single averaged vector can absorb the speaker into the lexical representation. A pure frame-difference descriptor can move in the opposite direction, because every voice has a different rhythm.

This project measures that trade-off. It splits a frozen hidden-state trajectory into static, delta, residual, and velocity views. Structured evidence and multi-scale online matching add boundary and timing cues. The entire pipeline uses NumPy and trains no additional matcher.

## Questions

1. Which view preserves content across speakers, the static centroid or the first-order motion?
2. Can arc-length parameterization remove smooth speaking-rate changes from a temporal descriptor?
3. Can a low-rank orthogonal projection reduce timbre leakage when speaker variation is a rotation?
4. When does structured evidence improve acoustic retrieval, and when does it add noise?
5. How can an online scan return separate events instead of adjacent copies of one peak?

## What The Evidence Says

The deterministic fixtures test numerical behavior and API contracts. They do not stand in for real dialect speech. The following values come from earlier local exploration logs.

| Probe | Result | Reading |
| --- | ---: | --- |
| Real temporal-shape Top-1 | 21.32% | Local motion carries phrase information |
| Frame-shuffle control | 14.60% | Breaking order costs 6.72 points |
| Single-vector retrieval bound | 58.05% | Static and velocity reference point |
| Structured-evidence oracle bound | 70.43% to 71.76% | Evidence summaries leave visible headroom |

The shuffle control shows that order matters. The oracle gap shows that structured evidence is worth studying, while also warning that an oracle number is not an online gain.

This release makes the following engineering claims directly testable.

- Temporal views can use equal-arc-length resampling by default.
- Zero-mean CUSUM and separated local peaks replace degenerate summary statistics.
- Query and template fingerprints share the same L2-normalized geometry.
- Incremental enrollment uses count-weighted means and invalidates cached evidence calibration.
- Multi-scale scans pass through temporal IoU NMS before returning top-k events.

## Method

<p align="center">
  <img src="docs/assets/pipeline.en.png" alt="Pipeline from hidden-state trajectory to online localization" />
</p>

For a trajectory $H \in \mathbb{R}^{T \times D}$, the library exposes four views.

| View | Construction | Purpose |
| --- | --- | --- |
| Static centroid | $\mu_H = T^{-1}\sum_t H_t$ | Cross-voice lexical anchor |
| Delta shape | $\Delta H_t = H_{t+1}-H_t$ after optional arc-length resampling | Articulatory transition direction |
| Residual path | $H_t-\mu_H$ | Motion relative to the phrase center |
| Velocity profile | $\lVert \Delta H_t \rVert_2$ | Local energy, duration, and tempo evidence |

`StructuredEvidence` stores response, a CUSUM change point, two separated peaks, speed statistics, turning statistics, and a normalized boundary contour. Evidence vectors can be z-scored against enrollment samples and fused with fingerprint cosine scores.

## Quickstart

```bash
python -m venv .venv
python -m pip install -e .
```

```python
from disentangle_acoustic_print import (
    EnrollmentBank,
    OnlineMatcher,
    SpeakerSubspaceProjector,
    make_cross_voice_dataset,
    make_stream,
)

dataset = make_cross_voice_dataset(dimensions=16)
projector = SpeakerSubspaceProjector.fit(
    dataset["sequences"],
    dataset["speaker_ids"],
    rank=2,
)

frames = dataset["sequences"][0]
bank = EnrollmentBank(projector=projector, evidence_weight=0.2)
bank.enroll("phrase-a", [frames])
print(bank.search(frames)[0])

stream = make_stream(
    dimensions=16,
    target=frames.mean(axis=0),
    insert_start=10,
    target_length=len(frames),
)
locator = OnlineMatcher(
    bank,
    window_length=len(frames),
    scales=(0.75, 1.0, 1.25),
    hop=2,
    iou_threshold=0.4,
)
print(locator.match(stream)[0])
```

```bash
disentangle-acoustic-print demo
disentangle-acoustic-print evidence
disentangle-acoustic-print fingerprint
disentangle-acoustic-print online
```

## Why It Matters

The package is a diagnostic harness for speech representations. Swap a layer, a model, or a pooling rule, then inspect content stability, speaker leakage, and temporal robustness separately. `layerwise_probe` reports those components per feature column.

It is also a low-resource retrieval baseline. A new word needs a small enrollment set. Acoustic localization proposes candidates, structured evidence adds boundary and timing cues, and the online matcher finds events in a continuous stream.

The project keeps its failure cases visible. Deterministic fixtures, numerical contracts, and negative results are part of the deliverable. A retrieval baseline that explains failure is more useful than a demo that only succeeds once.

## Scope

- The public API accepts `[time, dimension]` hidden-state trajectories.
- The repository ships no audio decoder, model weights, private manifests, or credentials.
- Synthetic fixtures test interfaces and numerical behavior, not real Wu or Mandarin accuracy.
- Fit `SpeakerSubspaceProjector` only on speakers allowed into enrollment or development analysis.
- Tune window scales and temporal IoU for the target frame rate, word length, and event density.
- Oracle bounds from historical logs are not online Top-1 results.

## Repository Layout

```text
disentangle_acoustic_print/
  geometry.py    temporal decomposition, arc-length resampling, speaker projection
  probing.py     layer probes, ranking metrics, grouped calibration
  evidence.py    CUSUM change point, peaks, structured evidence calibration
  fingerprint.py unit fingerprints, hybrid bank, multi-scale online locator
  synthetic.py   deterministic fixtures with tempo warp and orthogonal rotation
  cli.py         JSON demos and command-line entry points
docs/
  DATA_CONTRACT.md
  MODULE_MAP.md
  assets/        banners, logo, pipeline art, mascot
scripts/
  make_assets.py render README assets
tests/
  20 numerical and API regression tests
```

## Development

```bash
python -m compileall -q .
python -m pytest -q
```

See [docs/DATA_CONTRACT.md](docs/DATA_CONTRACT.md) for input requirements and [docs/MODULE_MAP.md](docs/MODULE_MAP.md) for module ownership.

## License

MIT. Audio, model weights, and derived features retain their original licenses.

**A useful acoustic retrieval baseline should explain both its hits and its failures.**
