"""White-box telemetry serializer for the 4-Track Acoustic Print Studio.

Packages intermediate geometric states from ``decompose``, ``extract_evidence``,
``arc_length_resample``, ``SpeakerSubspaceProjector``, ``fingerprint``, and
``OnlineMatcher`` into a lightweight JSON-serializable dictionary ready for
60fps HTML5 Canvas or Gradio rendering without modifying core library modules.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .evidence import SCALAR_NAMES, extract_evidence, select_peak_indices
from .fingerprint import EnrollmentBank, OnlineMatcher, fingerprint
from .geometry import (
    EPSILON,
    SpeakerSubspaceProjector,
    _linear_resample,
    arc_length_resample,
    cosine,
    decompose,
    l2_normalize,
)


FINGERPRINT_WEIGHTS = (0.45, 0.20, 0.20, 0.15)


def compute_arc_sample_positions(frames: np.ndarray, output_length: int = 8) -> dict[str, list[float]]:
    """Compute both linear and equal-arc-length interpolation coordinates on ``[0, T - 1]``.

    Reveals how ``arc_length_resample`` stretches ticks during slow vowels and
    compresses ticks during rapid consonant transitions.
    """
    array = np.asarray(frames, dtype=np.float64)
    if array.ndim != 2 or len(array) < 2:
        raise ValueError("frames must have shape [time, dimension] with time >= 2")
    if output_length < 2:
        raise ValueError("output_length must be at least two")

    t_len = len(array)
    linear_positions = np.linspace(0.0, float(t_len - 1), output_length)
    steps = np.linalg.norm(np.diff(array, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(steps)))
    total = float(cumulative[-1])
    if total <= EPSILON:
        arc_positions = linear_positions.copy()
    else:
        targets = np.linspace(0.0, total, output_length)
        arc_positions = np.interp(targets, cumulative, np.arange(t_len, dtype=np.float64))

    return {
        "linear_positions": linear_positions.tolist(),
        "arc_positions": arc_positions.tolist(),
        "linear_normalized": (linear_positions / max(t_len - 1, 1)).tolist(),
        "arc_normalized": (arc_positions / max(t_len - 1, 1)).tolist(),
        "cumulative_arc_length": cumulative.tolist(),
        "total_arc_length": total,
    }


def compute_cusum_curve(speed: np.ndarray) -> dict[str, Any]:
    """Compute the zero-mean CUSUM curve and exact change-point index for Track 2."""
    energy = np.asarray(speed, dtype=np.float64).reshape(-1)
    if len(energy) < 2:
        return {
            "curve": [0.0],
            "normalized_curve": [0.0],
            "change_index": 0,
            "change_position": 0.0,
            "extremum_value": 0.0,
        }
    centered = energy - float(energy.mean())
    cumulative = np.cumsum(centered)
    change_index = int(np.argmax(np.abs(cumulative)))
    change_position = float(change_index / max(len(energy) - 1, 1))
    scale = max(float(np.max(np.abs(cumulative))), EPSILON)
    return {
        "curve": cumulative.tolist(),
        "normalized_curve": (cumulative / scale).tolist(),
        "change_index": change_index,
        "change_position": change_position,
        "extremum_value": float(cumulative[change_index]),
    }


def compute_speaker_stripping_metrics(
    frames: np.ndarray,
    projector: SpeakerSubspaceProjector | None,
) -> dict[str, Any]:
    """Compute Frobenius energy stripped by ``SpeakerSubspaceProjector`` for Track 3."""
    array = np.asarray(frames, dtype=np.float64)
    if projector is None:
        return {
            "enabled": False,
            "removed_ratio": 0.0,
            "retained_ratio": 1.0,
            "removed_percent": 0.0,
            "retained_percent": 100.0,
            "n_speakers": 0,
            "subspace_rank": 0,
            "status_text": "未启用说话人正交投影（保留 100% 原始声学轨迹）",
            "projected_frames": array,
        }

    projected = projector.transform(array)
    diff_norm = float(np.linalg.norm(array - projected, ord="fro"))
    base_norm = max(float(np.linalg.norm(array, ord="fro")), EPSILON)
    removed_ratio = float(np.clip(diff_norm / base_norm, 0.0, 1.0))
    retained_ratio = float(1.0 - removed_ratio)
    removed_pct = round(removed_ratio * 100.0, 1)
    retained_pct = round(retained_ratio * 100.0, 1)
    return {
        "enabled": True,
        "removed_ratio": removed_ratio,
        "retained_ratio": retained_ratio,
        "removed_percent": removed_pct,
        "retained_percent": retained_pct,
        "n_speakers": int(projector.n_speakers),
        "subspace_rank": int(projector.basis.shape[1]) if projector.basis.ndim == 2 else 0,
        "status_text": f"当前已滤除 {removed_pct:.1f}% 的个人音色干扰，保留 {retained_pct:.1f}% 的纯发音内容",
        "projected_frames": projected,
    }


def decompose_fingerprint_barcode(
    frames: np.ndarray,
    output_length: int = 8,
    resample: str = "arc",
    bank: EnrollmentBank | None = None,
) -> dict[str, Any]:
    """Deconstruct ``fingerprint()`` into its 4 weighted partitions for Track 4 Barcode view."""
    trajectory = np.asarray(frames, dtype=np.float64)
    decomposition = decompose(trajectory)
    static_raw = l2_normalize(decomposition.centroid)
    if resample == "arc":
        sampled = arc_length_resample(trajectory, output_length + 1)
        delta_raw = l2_normalize(np.diff(sampled, axis=0).reshape(-1))
    else:
        delta_raw = l2_normalize(
            _linear_resample(decomposition.delta, output_length).reshape(-1)
        )
    speed_raw = l2_normalize(
        _linear_resample(decomposition.speed.reshape(-1, 1), output_length).reshape(-1)
    )
    accel_len = max(output_length - 1, 2)
    accel_raw = l2_normalize(
        _linear_resample(decomposition.acceleration.reshape(-1, 1), accel_len).reshape(-1)
    )

    weights = np.sqrt(np.asarray(FINGERPRINT_WEIGHTS, dtype=np.float64))
    seg_static = weights[0] * static_raw
    seg_delta = weights[1] * delta_raw
    seg_speed = weights[2] * speed_raw
    seg_accel = weights[3] * accel_raw
    combined = np.concatenate((seg_static, seg_delta, seg_speed, seg_accel))
    norm_factor = max(float(np.linalg.norm(combined)), EPSILON)
    full_vector = combined / norm_factor

    s0 = 0
    s1 = s0 + len(seg_static)
    s2 = s1 + len(seg_delta)
    s3 = s2 + len(seg_speed)
    s4 = s3 + len(seg_accel)

    segments = [
        {
            "name": "static",
            "label": "0.45 × Static 静态质心",
            "weight": 0.45,
            "start": s0,
            "end": s1,
            "values": full_vector[s0:s1].tolist(),
        },
        {
            "name": "delta",
            "label": "0.20 × Delta 形态变化",
            "weight": 0.20,
            "start": s1,
            "end": s2,
            "values": full_vector[s1:s2].tolist(),
        },
        {
            "name": "speed",
            "label": "0.20 × Speed 速度分布",
            "weight": 0.20,
            "start": s2,
            "end": s3,
            "values": full_vector[s2:s3].tolist(),
        },
        {
            "name": "acceleration",
            "label": "0.15 × Accel 加速度",
            "weight": 0.15,
            "start": s3,
            "end": s4,
            "values": full_vector[s3:s4].tolist(),
        },
    ]

    template_comparisons: list[dict[str, Any]] = []
    if bank is not None and len(bank) > 0:
        matrix = bank.matrix()
        for idx, key in enumerate(bank.keys):
            tmpl = matrix[idx]
            if len(tmpl) != len(full_vector):
                continue
            seg_sims = {
                seg["name"]: cosine(
                    full_vector[seg["start"] : seg["end"]],
                    tmpl[seg["start"] : seg["end"]],
                )
                for seg in segments
            }
            template_comparisons.append(
                {
                    "key": key,
                    "count": bank.count(key),
                    "overall_cosine": float(tmpl @ full_vector),
                    "vector": tmpl.tolist(),
                    "segment_similarities": seg_sims,
                }
            )
        template_comparisons.sort(key=lambda item: item["overall_cosine"], reverse=True)

    return {
        "vector": full_vector.tolist(),
        "dimension": len(full_vector),
        "segments": segments,
        "templates": template_comparisons,
    }


@dataclass(frozen=True)
class TelemetryPacket:
    """Structured white-box telemetry snapshot for the 4-track acoustic oscilloscope."""

    frame_count: int
    dimension: int
    waveform_chunk: list[float]
    frame_speed: list[float]
    frame_speed_normalized: list[float]
    active_matches: list[dict[str, Any]]
    kinematics: dict[str, Any]
    evidence: dict[str, Any]
    arc_ruler: dict[str, Any]
    speaker_projection: dict[str, Any]
    barcode: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_telemetry_packet(
    frames: np.ndarray,
    *,
    waveform_chunk: np.ndarray | None = None,
    matcher: OnlineMatcher | None = None,
    bank: EnrollmentBank | None = None,
    projector: SpeakerSubspaceProjector | None = None,
    output_length: int = 8,
    top_k: int = 3,
    match_threshold: float = -1.0,
    focus_window: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Assemble a complete 4-track white-box telemetry dictionary in one pass.

    Parameters
    ----------
    frames:
        ``[time, dimension]`` hidden-state or acoustic trajectory buffer.
    waveform_chunk:
        Optional raw waveform or frame-aligned amplitude envelope. If omitted,
        a signed projection along the primary trajectory axis is synthesized.
    matcher:
        Optional ``OnlineMatcher`` for multi-scale sliding-window event capture.
    bank:
        Optional ``EnrollmentBank`` (inferred from ``matcher.bank`` if omitted).
    projector:
        Optional ``SpeakerSubspaceProjector`` (inferred from ``bank.projector`` if omitted).
    output_length:
        Number of arc-length resample ticks and fingerprint steps (default 8).
    top_k:
        Maximum number of NMS-filtered active matches to return.
    match_threshold:
        Minimum hybrid score required to display a match box on Track 1.
    focus_window:
        Optional ``(start, end)`` frame slice for localized barcode/evidence inspection;
        defaults to the top active match window if one exists, or the full buffer.
    """
    raw_frames = np.asarray(frames, dtype=np.float64)
    if raw_frames.ndim != 2 or len(raw_frames) < 3:
        raise ValueError("telemetry requires a [time, dimension] trajectory with at least 3 frames")

    active_bank = bank or (matcher.bank if matcher is not None else None)
    active_projector = projector or (active_bank.projector if active_bank is not None else None)

    # Track 3: Speaker subspace stripping gauge
    projection_info = compute_speaker_stripping_metrics(raw_frames, active_projector)
    working_frames = projection_info.pop("projected_frames")

    # Track 1 & Track 2: Full-stream decomposition
    decomp = decompose(working_frames)
    speed = decomp.speed
    # Pad frame_speed from length T-1 to T so every frame has an instantaneous speed color
    frame_speed = np.concatenate(([speed[0]], speed))
    speed_min = float(np.min(frame_speed))
    speed_span = max(float(np.max(frame_speed) - speed_min), EPSILON)
    frame_speed_norm = (frame_speed - speed_min) / speed_span

    if waveform_chunk is None:
        # Synthesize a frame-synchronous waveform envelope from the first principal axis + speed
        centered = raw_frames - raw_frames.mean(axis=0, keepdims=True)
        carrier = centered[:, 0]
        carrier_scale = max(float(np.max(np.abs(carrier))), EPSILON)
        waveform = (carrier / carrier_scale) * (0.35 + 0.65 * frame_speed_norm)
    else:
        waveform = np.asarray(waveform_chunk, dtype=np.float64).reshape(-1)

    # Track 1: Multi-scale OnlineMatcher + temporal IoU NMS highlights
    active_matches: list[dict[str, Any]] = []
    if matcher is not None and active_bank is not None and len(active_bank) > 0:
        usable_lengths = [w for w in matcher.window_lengths if len(raw_frames) >= w]
        if usable_lengths:
            matches = matcher.match(raw_frames, top_k=top_k)
            for item in matches:
                if item.score >= match_threshold:
                    active_matches.append(
                        {
                            "key": item.key,
                            "score": float(item.score),
                            "start": int(item.start),
                            "end": int(item.end),
                            "start_ratio": float(item.start / max(len(raw_frames) - 1, 1)),
                            "end_ratio": float(item.end / max(len(raw_frames), 1)),
                            "fingerprint_score": (
                                float(item.fingerprint_score)
                                if item.fingerprint_score is not None
                                else float(item.score)
                            ),
                            "evidence_distance": (
                                float(item.evidence_distance)
                                if item.evidence_distance is not None
                                else 0.0
                            ),
                        }
                    )

    # Choose inspection slice for detailed evidence/barcode (either focus_window, top match, or full buffer)
    if focus_window is not None:
        w_start, w_end = max(0, int(focus_window[0])), min(len(working_frames), int(focus_window[1]))
        if w_end - w_start < 3:
            w_start, w_end = 0, len(working_frames)
    elif active_matches:
        w_start = int(active_matches[0]["start"])
        w_end = int(active_matches[0]["end"])
        if w_end - w_start < 3:
            w_start, w_end = 0, len(working_frames)
    else:
        w_start, w_end = 0, len(working_frames)

    focus_frames = working_frames[w_start:w_end]

    # Track 2: Kinematics curves (speed, peaks, CUSUM change point, turning angle)
    min_peak_dist = max(2, int(np.floor(0.15 * len(speed))))
    peaks = select_peak_indices(speed, count=2, min_distance=min_peak_dist)
    primary_idx = int(peaks[0]) if len(peaks) > 0 else 0
    secondary_idx = int(peaks[1]) if len(peaks) > 1 else primary_idx
    cusum_info = compute_cusum_curve(speed)

    kinematics = {
        "speed_curve": speed.tolist(),
        "acceleration_curve": decomp.acceleration.tolist(),
        "turning_curve": decomp.turning.tolist(),
        "primary_peak_index": primary_idx,
        "primary_peak_position": float(primary_idx / max(len(speed) - 1, 1)),
        "primary_peak_value": float(speed[primary_idx]) if len(speed) else 0.0,
        "secondary_peak_index": secondary_idx,
        "secondary_peak_position": float(secondary_idx / max(len(speed) - 1, 1)),
        "secondary_peak_value": float(speed[secondary_idx]) if len(speed) else 0.0,
        "cusum": cusum_info,
    }

    # Structured evidence & 10-D radar scalars
    ev = extract_evidence(focus_frames)
    ev_dict = ev.to_dict(include_boundary=True)
    ev_dict["scalar_names"] = list(SCALAR_NAMES)
    ev_dict["focus_window"] = [w_start, w_end]

    # Track 3: Equal-arc-length resampling ruler on the active window / stream
    arc_ruler = compute_arc_sample_positions(working_frames, output_length=output_length + 1)

    # Track 4: 4-partition weighted fingerprint barcode & template comparison
    resample_mode = active_bank.resample if active_bank is not None else "arc"
    barcode = decompose_fingerprint_barcode(
        focus_frames,
        output_length=output_length,
        resample=resample_mode,
        bank=active_bank,
    )
    barcode["focus_window"] = [w_start, w_end]

    packet = TelemetryPacket(
        frame_count=int(len(raw_frames)),
        dimension=int(raw_frames.shape[1]),
        waveform_chunk=waveform.tolist(),
        frame_speed=frame_speed.tolist(),
        frame_speed_normalized=frame_speed_norm.tolist(),
        active_matches=active_matches,
        kinematics=kinematics,
        evidence=ev_dict,
        arc_ruler=arc_ruler,
        speaker_projection=projection_info,
        barcode=barcode,
    )
    return packet.to_dict()
