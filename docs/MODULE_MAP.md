# Module Map

The implementation consolidates the core geometric operators and the non-invasive **Acoustic Print Studio** interactive layer.

| Public module | Consolidated concern | Original / Studio focus |
| --- | --- | --- |
| `geometry.py` | Trajectory algebra, arc-length resampling, speaker projection, temporal controls, cross-voice consistency | `v82`, `v90`, `v90c` |
| `probing.py` | Rank AUC, response calibration, variance attribution | `v82`, `v90c` |
| `evidence.py` | CUSUM change point, separated peaks, evidence calibration | `v89`, `v90b`, `v91` |
| `fingerprint.py` | Unit fingerprint, hybrid enrollment bank, multi-scale online locator with temporal NMS | `v89`, `v92`, `v93` |
| `audio_bridge.py` | Sub-millisecond 16 kHz `MelDeltaExtractor`, `.wav` codec, and optional `HiddenStateBridge` | Acoustic Print Studio |
| `telemetry.py` | 4-track white-box telemetry serializer (`build_telemetry_packet`) | Acoustic Print Studio |
| `stream_engine.py` | `RollingAcousticStream` ring buffer, 3-second dual-tempo enrollment, live sliding-window matching | Acoustic Print Studio |
| `studio/app.py` | Plan A FastAPI WebSocket + REST server and unified CLI launcher | Acoustic Print Studio |
| `studio/gradio_app.py` | Plan B Gradio Streaming dashboard for Hugging Face Spaces | Acoustic Print Studio |
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
