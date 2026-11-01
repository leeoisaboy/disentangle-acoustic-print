# Module Map

The implementation consolidates the original exploration branches into four stable layers.

| Public module | Consolidated concern | Original exploration focus |
| --- | --- | --- |
| `geometry.py` | Trajectory algebra, arc-length resampling, speaker projection, temporal controls, cross-voice consistency | `v82`, `v90`, `v90c` |
| `probing.py` | Rank AUC, response calibration, variance attribution | `v82`, `v90c` |
| `evidence.py` | CUSUM change point, separated peaks, evidence calibration | `v89`, `v90b`, `v91` |
| `fingerprint.py` | Unit fingerprint, hybrid enrollment bank, multi-scale online locator with temporal NMS | `v89`, `v92`, `v93` |

Only portable NumPy logic is included. Private manifests, audio assets, extractor scripts, credentials, and infrastructure code remain outside this repository.

## API Selection

- Use `decompose` when explaining static and temporal views.
- Use `arc_length_resample` when speaker tempo changes should not alter the temporal descriptor.
- Use `SpeakerSubspaceProjector` when enrollment voices provide enough samples to estimate a low-rank voice subspace.
- Use `disentanglement_report` when comparing content stability with speaker leakage.
- Use `layerwise_probe` when comparing candidate representation layers.
- Use `extract_evidence` when a single vector needs interpretable motion summaries.
- Use `EvidenceCalibrator` when scalar evidence dimensions have different units or dynamic ranges.
- Use `EnrollmentBank` for fixed-shape retrieval, with optional evidence fusion.
- Use `OnlineMatcher` for multi-scale sliding-window localization with temporal NMS.
