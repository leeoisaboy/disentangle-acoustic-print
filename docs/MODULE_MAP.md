# Module Map

The implementation consolidates the core geometric operators and the non-invasive **Acoustic Print Studio** interactive layer.

| Public module | Consolidated concern | Original / Studio focus |
| --- | --- | --- |
| `geometry.py` | Trajectory algebra, VAD/smoothing-guarded `arc_length_resample`, soft low-rank `SpeakerSubspaceProjector`, cross-voice consistency | Core SDK (NumPy-only) |
| `probing.py` | Rank AUC, response calibration, `layerwise_probe` content-to-speaker variance attribution | Core SDK (NumPy-only) |
| `evidence.py` | CUSUM change point, separated peaks, evidence calibration (white-box diagnostic layer) | Core SDK + Diagnostics |
| `fingerprint.py` | Unit fingerprint (`legacy` / `content`), `EnrollmentBank` with `calibrate_background()`, ms/auto-window `OnlineMatcher` with temporal NMS | Core SDK (NumPy-only) |
| `audio_bridge.py` | Unified `FeatureFrontend` protocol, streaming-safe `MelDeltaExtractor`, `HiddenStateBridge`, optional `OnnxEncoderFrontend` & `PyTorchSSLFrontend` | Frontend Adapters |
| `telemetry.py` | 4-track white-box telemetry serializer (`build_telemetry_packet`) | Acoustic Print Studio |
| `stream_engine.py` | Frame-rate-agnostic `RollingAcousticStream` ring buffer, dual-tempo enrollment, live sliding-window matching | Acoustic Print Studio |
| `benchmarks/` | Public multi-dialect benchmark suite & 4-gate automated release verifier (`verify_release_gates.py`) | Real-Data Verification |
| `studio/app.py` | Plan A FastAPI WebSocket + REST server and unified CLI launcher | Acoustic Print Studio |
| `studio/gradio_app.py` | Plan B Gradio streaming studio for Hugging Face Spaces | Acoustic Print Studio |
| `studio/index.html` | Light, progressive-disclosure HTML5 Canvas 4-track white-box oscilloscope | Acoustic Print Studio |
| `studio/static/` | Local CSS design system and deterministic Canvas renderer/capture API | Acoustic Print Studio |

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
- Use `MelDeltaExtractor` or `HiddenStateBridge` to convert raw 16 kHz audio or `.wav` bytes into `[time, dimension]` trajectories.
- Use `RollingAcousticStream` and `build_telemetry_packet` to drive the 4-track white-box oscilloscope.
