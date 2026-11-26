"""Acoustic Print Studio backend server (FastAPI WebSocket + Gradio launcher).

Supports both:
- Plan A (60fps HTML5 Canvas + FastAPI WebSocket & REST backend):
    ``python -m studio.app --mode fastapi --port 8000``
- Plan B (Gradio Streaming interface for Hugging Face Spaces):
    ``python -m studio.app --mode gradio --port 7860``
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import numpy as np

from disentangle_acoustic_print.audio_bridge import (
    decode_wav_bytes,
    synthesize_phrase_waveform,
)
from disentangle_acoustic_print.stream_engine import RollingAcousticStream


STUDIO_DIR = Path(__file__).resolve().parent
INDEX_HTML_PATH = STUDIO_DIR / "index.html"
STATIC_DIR = STUDIO_DIR / "static"


def inline_studio_html() -> str:
    """Return the Studio page with local CSS and JS inlined for ``srcdoc`` iframes."""
    html = INDEX_HTML_PATH.read_text(encoding="utf-8")
    css_path = STATIC_DIR / "studio.css"
    js_path = STATIC_DIR / "studio.js"
    if css_path.exists():
        css_text = css_path.read_text(encoding="utf-8")
        html = re.sub(
            r'<link\s+rel="stylesheet"\s+href="static/studio\.css(?:\?[^"]*)?"\s*/?>',
            lambda _: f"<style>{css_text}</style>",
            html,
            count=1,
        )
    if js_path.exists():
        js_text = js_path.read_text(encoding="utf-8")
        html = re.sub(
            r'<script\s+src="static/studio\.js(?:\?[^"]*)?"(?:\s+defer)?></script>',
            lambda _: f"<script>{js_text}</script>",
            html,
            count=1,
        )
    return html


class StudioSession:
    """Stateful studio session wrapping ``RollingAcousticStream`` for web handlers."""

    def __init__(self, seed: int = 17) -> None:
        self.engine = RollingAcousticStream(
            max_frames=120,
            dimension=16,
            output_length=8,
            evidence_weight=0.0,
            window_length=16,
            scales=(0.75, 1.0, 1.25),
            hop=2,
            iou_threshold=0.4,
            match_threshold=0.42,
        )
        self.preset_info = self.engine.load_default_dialect_presets(seed=seed)

    def current_telemetry(self) -> dict[str, Any]:
        return self.engine.push_frames(np.zeros((0, self.engine.dimension)))

    def handle_pcm_chunk(
        self,
        samples: list[float] | np.ndarray,
        sample_rate: int = 16000,
    ) -> dict[str, Any]:
        arr = np.asarray(samples, dtype=np.float64).reshape(-1)
        return self.engine.push_audio_chunk(arr, sample_rate=sample_rate)

    def handle_wav_upload(self, raw_bytes: bytes) -> dict[str, Any]:
        audio, sr = decode_wav_bytes(raw_bytes)
        return self.engine.push_audio_chunk(audio, sample_rate=sr)

    def enroll_dual_tempo(
        self,
        key: str,
        normal_samples: list[float] | np.ndarray | None = None,
        slow_samples: list[float] | np.ndarray | None = None,
        sample_rate: int = 16000,
        speaker_id: str = "studio-user",
    ) -> dict[str, Any]:
        if normal_samples is None or slow_samples is None:
            normal_wave = synthesize_phrase_waveform(
                phrase=key, duration_sec=1.2, sample_rate=sample_rate, speaking_rate=1.0
            )
            slow_wave = synthesize_phrase_waveform(
                phrase=key, duration_sec=1.2, sample_rate=sample_rate, speaking_rate=0.68
            )
        else:
            normal_wave = np.asarray(normal_samples, dtype=np.float64).reshape(-1)
            slow_wave = np.asarray(slow_samples, dtype=np.float64).reshape(-1)

        info = self.engine.enroll_audio(
            key=key,
            audio_chunks=[normal_wave, slow_wave],
            sample_rate=sample_rate,
            speaker_id=speaker_id,
        )
        telemetry = self.engine.push_audio_chunk(normal_wave, sample_rate=sample_rate)
        return {"enrollment": info, "telemetry": telemetry}

    def inject_demo_phrase(
        self,
        key: str | None = None,
        speaking_rate: float = 1.0,
        speaker_nuisance: float = 0.35,
        seed: int = 29,
    ) -> dict[str, Any]:
        return self.engine.inject_synthetic_utterance(
            key=key,
            speaking_rate=speaking_rate,
            speaker_nuisance=speaker_nuisance,
            seed=seed,
        )


def create_fastapi_app(session: StudioSession | None = None) -> Any:
    """Create the FastAPI application with WebSocket and REST endpoints for Plan A."""
    try:
        from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
        from fastapi.responses import HTMLResponse, JSONResponse
    except ImportError as exc:
        raise RuntimeError(
            "FastAPI is required for --mode fastapi. Install with: pip install fastapi uvicorn"
        ) from exc
    from fastapi.staticfiles import StaticFiles

    studio = session or StudioSession()
    app = FastAPI(
        title="Acoustic Print Studio",
        description="Real-time 4-Track White-Box Acoustic Oscilloscope",
        version="0.3.0",
    )
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        if INDEX_HTML_PATH.exists():
            return INDEX_HTML_PATH.read_text(encoding="utf-8")
        return "<h1>Acoustic Print Studio</h1><p>studio/index.html not found.</p>"

    @app.get("/api/state")
    async def get_state() -> JSONResponse:
        return JSONResponse(studio.current_telemetry())

    @app.post("/api/inject")
    async def inject_event(payload: dict[str, Any]) -> JSONResponse:
        telemetry = studio.inject_demo_phrase(
            key=payload.get("key"),
            speaking_rate=float(payload.get("speaking_rate", 1.0)),
            speaker_nuisance=float(payload.get("speaker_nuisance", 0.35)),
            seed=int(payload.get("seed", 29)),
        )
        return JSONResponse(telemetry)

    @app.post("/api/enroll")
    async def enroll_phrase(payload: dict[str, Any]) -> JSONResponse:
        key = str(payload.get("key", "自定义口令")).strip() or "自定义口令"
        result = studio.enroll_dual_tempo(
            key=key,
            normal_samples=payload.get("normal_samples"),
            slow_samples=payload.get("slow_samples"),
            sample_rate=int(payload.get("sample_rate", 16000)),
            speaker_id=str(payload.get("speaker_id", "studio-user")),
        )
        return JSONResponse(result)

    @app.post("/api/projector")
    async def toggle_projector(payload: dict[str, Any]) -> JSONResponse:
        enabled = bool(payload.get("enabled", True))
        studio.engine.toggle_speaker_projector(enabled)
        return JSONResponse(studio.current_telemetry())

    @app.post("/api/upload_wav")
    async def upload_wav(file: UploadFile = File(...)) -> JSONResponse:
        raw_bytes = await file.read()
        telemetry = studio.handle_wav_upload(raw_bytes)
        return JSONResponse(telemetry)

    @app.websocket("/ws/telemetry")
    async def websocket_telemetry(ws: WebSocket) -> None:
        await ws.accept()
        await ws.send_json(studio.current_telemetry())
        try:
            while True:
                msg = await ws.receive_json()
                action = msg.get("action", "pcm")
                if action == "pcm":
                    packet = studio.handle_pcm_chunk(
                        msg.get("samples", []),
                        sample_rate=int(msg.get("sample_rate", 16000)),
                    )
                elif action == "inject":
                    packet = studio.inject_demo_phrase(
                        key=msg.get("key"),
                        speaking_rate=float(msg.get("speaking_rate", 1.0)),
                        speaker_nuisance=float(msg.get("speaker_nuisance", 0.35)),
                    )
                elif action == "enroll":
                    res = studio.enroll_dual_tempo(
                        key=str(msg.get("key", "新方言口令")),
                        normal_samples=msg.get("normal_samples"),
                        slow_samples=msg.get("slow_samples"),
                        sample_rate=int(msg.get("sample_rate", 16000)),
                    )
                    packet = res["telemetry"]
                elif action == "projector":
                    studio.engine.toggle_speaker_projector(bool(msg.get("enabled", True)))
                    packet = studio.current_telemetry()
                else:
                    packet = studio.current_telemetry()
                await ws.send_json(packet)
        except WebSocketDisconnect:
            return

    return app


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="acoustic-print-studio")
    parser.add_argument(
        "--mode",
        choices=("fastapi", "gradio", "dump-telemetry"),
        default="fastapi",
        help="Launch Plan A (FastAPI + HTML5 Canvas), Plan B (Gradio Spaces), or dump a telemetry JSON snapshot",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    if args.mode == "dump-telemetry":
        import json

        session = StudioSession()
        packet = session.inject_demo_phrase(speaking_rate=1.3, speaker_nuisance=0.4)
        print(json.dumps(packet, ensure_ascii=False, indent=2))
        return

    if args.mode == "gradio":
        from .gradio_app import launch_gradio_studio

        launch_gradio_studio(server_name=args.host, server_port=args.port)
        return

    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError(
            "uvicorn is required to serve FastAPI. Install with: pip install fastapi uvicorn"
        ) from exc

    app = create_fastapi_app()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
