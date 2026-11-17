"""Plan B: Gradio Streaming application for Hugging Face Spaces deployment.

Embeds the interactive 4-Track HTML5 Canvas Oscilloscope alongside Gradio
streaming audio / file upload / dual-tempo enrollment controls so the project
can be deployed to Hugging Face Spaces with a single ``app.py`` entrypoint.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from .app import StudioSession, inline_studio_html


def build_gradio_blocks(session: StudioSession | None = None) -> Any:
    """Construct a Gradio ``Blocks`` interface wrapping ``StudioSession``."""
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError(
            "Gradio is required for Plan B. Install with: pip install gradio"
        ) from exc

    studio = session or StudioSession()
    initial_packet = studio.inject_demo_phrase(speaking_rate=1.2, speaker_nuisance=0.35)

    def _format_summary(packet: dict[str, Any]) -> str:
        matches = packet.get("active_matches", [])
        proj = packet.get("speaker_projection", {})
        ev = packet.get("evidence", {})
        top_str = (
            f"命中口令: **{matches[0]['key']}** | 综合分: `{matches[0]['score']:.3f}` "
            f"(指纹分 `{matches[0]['fingerprint_score']:.3f}`, 证据距离 `{matches[0]['evidence_distance']:.3f}`) "
            f"区间 `[{matches[0]['start']}, {matches[0]['end']}]`"
            if matches
            else "当前窗口无超过阈值的目标口令命中"
        )
        return (
            f"### 实时白盒遥测摘要\n"
            f"- **多尺度捕获 (Track 1)**: {top_str}\n"
            f"- **运动学证据 (Track 2)**: CUSUM 变点 `{ev.get('change_position', 0.0):.2f}` | "
            f"主峰 `{ev.get('primary_peak_position', 0.0):.2f}` | 次峰 `{ev.get('secondary_peak_position', 0.0):.2f}`\n"
            f"- **音色正交剥离 (Track 3)**: {proj.get('status_text', '')}\n"
            f"- **单次滑窗解耦耗时**: `{packet.get('compute_latency_ms', 0.25):.3f} ms`"
        )

    def on_audio_input(audio_tuple: tuple[int, np.ndarray] | None) -> tuple[str, dict[str, Any]]:
        if audio_tuple is None:
            pkt = studio.current_telemetry()
            return _format_summary(pkt), pkt
        sr, data = audio_tuple
        arr = np.asarray(data, dtype=np.float64)
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        max_val = float(np.max(np.abs(arr))) if len(arr) else 1.0
        if max_val > 1.5:
            arr = arr / 32768.0
        pkt = studio.handle_pcm_chunk(arr, sample_rate=int(sr))
        return _format_summary(pkt), pkt

    def on_inject(
        phrase_key: str,
        speaking_rate: float,
        speaker_nuisance: float,
        projector_on: bool,
    ) -> tuple[str, dict[str, Any]]:
        studio.engine.toggle_speaker_projector(projector_on)
        pkt = studio.inject_demo_phrase(
            key=phrase_key,
            speaking_rate=float(speaking_rate),
            speaker_nuisance=float(speaker_nuisance),
        )
        return _format_summary(pkt), pkt

    def on_enroll(
        new_key: str,
        normal_audio: tuple[int, np.ndarray] | None,
        slow_audio: tuple[int, np.ndarray] | None,
    ) -> tuple[str, dict[str, Any]]:
        key_clean = (new_key or "自定义方言口令").strip()
        norm_arr = None
        slow_arr = None
        sr = 16000
        if normal_audio is not None:
            sr, raw_n = normal_audio
            norm_arr = np.asarray(raw_n, dtype=np.float64).reshape(-1)
        if slow_audio is not None:
            sr, raw_s = slow_audio
            slow_arr = np.asarray(raw_s, dtype=np.float64).reshape(-1)
        res = studio.enroll_dual_tempo(
            key=key_clean,
            normal_samples=norm_arr,
            slow_samples=slow_arr,
            sample_rate=int(sr),
        )
        pkt = res["telemetry"]
        return _format_summary(pkt), pkt

    canvas_iframe = ""
    try:
        escaped = inline_studio_html().replace('"', "&quot;")
        canvas_iframe = (
            f'<iframe srcdoc="{escaped}" '
            f'style="width:100%;height:1080px;border:1px solid #e5e5ea;border-radius:20px;" '
            f'allow="microphone"></iframe>'
        )
    except FileNotFoundError:
        canvas_iframe = ""

    with gr.Blocks(title="Acoustic Print Studio - 四轨白盒声学示波器") as demo:
        gr.Markdown(
            "## Acoustic Print Studio — 四轨同步白盒声学示波器\n"
            "零神经网络解码开销（纯 NumPy 几何解耦 `< 1 ms`），支持跨语速等弧长重采样与说话人正交子空间剥离。"
        )
        if canvas_iframe:
            gr.HTML(canvas_iframe)

        with gr.Row():
            with gr.Column(scale=1):
                phrase_dropdown = gr.Dropdown(
                    choices=list(studio.engine.bank.keys),
                    value=studio.engine.bank.keys[0],
                    label="目标口令 / 方言模板",
                )
                rate_slider = gr.Slider(
                    0.5, 2.0, value=1.25, step=0.05, label="语速倍率 (Speaking Rate Warp)"
                )
                nuisance_slider = gr.Slider(
                    0.0, 1.0, value=0.38, step=0.02, label="跨说话人音色偏移强度 (Speaker Nuisance)"
                )
                proj_checkbox = gr.Checkbox(
                    value=True, label="启用 SpeakerSubspaceProjector 正交音色剥离"
                )
                inject_btn = gr.Button("注入测试语流并触发四轨解耦", variant="primary")

            with gr.Column(scale=1):
                audio_in = gr.Audio(
                    sources=["microphone", "upload"],
                    type="numpy",
                    label="实时麦克风流 / 上传 .wav 音频",
                )
                with gr.Accordion("模式一：3 秒双语速实时注册新口令", open=False):
                    enroll_name = gr.Textbox(value="湘语·恰饭 (Hunanese)", label="新口令标签")
                    enroll_norm = gr.Audio(sources=["microphone", "upload"], type="numpy", label="正常语速样本 (可选，留空自动合成)")
                    enroll_slow = gr.Audio(sources=["microphone", "upload"], type="numpy", label="慢速拉长样本 (可选，留空自动合成)")
                    enroll_btn = gr.Button("合成 EnrollmentBank 指纹模板")

        summary_md = gr.Markdown(value=_format_summary(initial_packet))
        telemetry_json = gr.JSON(value=initial_packet, label="实时白盒遥测包 (telemetry.py 输出)")

        inject_btn.click(
            on_inject,
            inputs=[phrase_dropdown, rate_slider, nuisance_slider, proj_checkbox],
            outputs=[summary_md, telemetry_json],
        )
        audio_in.change(
            on_audio_input,
            inputs=[audio_in],
            outputs=[summary_md, telemetry_json],
        )
        enroll_btn.click(
            on_enroll,
            inputs=[enroll_name, enroll_norm, enroll_slow],
            outputs=[summary_md, telemetry_json],
        )

    return demo


def launch_gradio_studio(server_name: str = "127.0.0.1", server_port: int = 7860) -> None:
    demo = build_gradio_blocks()
    demo.launch(server_name=server_name, server_port=server_port)


if __name__ == "__main__":
    launch_gradio_studio()
