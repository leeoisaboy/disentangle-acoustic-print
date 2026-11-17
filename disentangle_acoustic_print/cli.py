from __future__ import annotations

import argparse
import json
from typing import Sequence

import numpy as np

from .evidence import extract_evidence
from .fingerprint import EnrollmentBank, OnlineMatcher, fingerprint
from .geometry import decompose, disentanglement_report
from .probing import layerwise_probe
from .stream_engine import RollingAcousticStream
from .synthetic import make_cross_voice_dataset, make_stream


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def _demo(seed: int) -> dict[str, object]:
    dataset = make_cross_voice_dataset(seed=seed)
    bank = EnrollmentBank()
    train_sequences = dataset["sequences"]
    train_content = dataset["content_ids"]
    train_speakers = dataset["speaker_ids"]
    for index, content_id in enumerate(train_content):
        bank.enroll(content_id, [train_sequences[index]])
    report = disentanglement_report(
        train_sequences,
        train_content,
        train_speakers,
        descriptors=("static", "delta", "residual", "velocity"),
    )
    probe = layerwise_probe(
        np.stack([fingerprint(sequence) for sequence in train_sequences]),
        train_content,
        train_speakers,
    )
    evidence = extract_evidence(train_sequences[0])
    return {
        "seed": seed,
        "sample_count": len(train_sequences),
        "enrolled_keys": list(bank.keys),
        "disentanglement": report,
        "layer_probe": {key: np.asarray(value).tolist() for key, value in probe.items()},
        "structured_evidence": evidence.to_dict(include_boundary=True),
    }


def _bank_demo(seed: int) -> dict[str, object]:
    random = np.random.default_rng(seed)
    bank = EnrollmentBank(evidence_weight=0.2)
    directions = random.normal(size=(3, 8))
    for label, direction in zip(("door", "window", "lamp"), directions):
        profile = np.sin(np.linspace(0.0, np.pi, 12)).reshape(-1, 1)
        for offset in (-0.2, 0.0, 0.2):
            sequence = 1.4 * profile * direction.reshape(1, -1)
            sequence += offset + random.normal(scale=0.02, size=(12, 8))
            bank.enroll(label, [sequence])
    query_direction = directions[1]
    query = 1.4 * np.sin(np.linspace(0.0, np.pi, 12)).reshape(-1, 1) * query_direction.reshape(1, -1)
    query += 0.1
    return {
        "evidence_weight": bank.evidence_weight,
        "query_ranking": [match.__dict__ for match in bank.search(query, top_k=3)],
    }


def _decompose_demo(seed: int) -> dict[str, object]:
    dataset = make_cross_voice_dataset(seed=seed)
    view = decompose(dataset["sequences"][0])
    return {
        "frames": view.frames,
        "dimension": int(view.centroid.shape[0]),
        "centroid_norm": float(np.linalg.norm(view.centroid)),
        "mean_speed": float(view.speed.mean()),
        "mean_acceleration": float(view.acceleration.mean()),
        "mean_turning": float(view.turning.mean()),
    }


def _telemetry_demo(seed: int) -> dict[str, object]:
    engine = RollingAcousticStream(max_frames=72, dimension=16, window_length=14)
    engine.load_default_dialect_presets(seed=seed)
    return engine.inject_synthetic_utterance(
        key="吴语·侬好 (Wu Dialect)",
        speaking_rate=1.25,
        speaker_nuisance=0.4,
        seed=seed,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="disentangle-acoustic-print")
    parser.add_argument("--seed", type=int, default=17)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("demo", help="run the deterministic synthetic pipeline")
    subparsers.add_parser("decompose", help="show decomposition sample statistics")
    subparsers.add_parser("probe", help="estimate content and speaker variance")
    subparsers.add_parser("evidence", help="extract structured temporal evidence")
    subparsers.add_parser("fingerprint", help="build and search a synthetic template bank")
    subparsers.add_parser("online", help="locate a synthetic event with sliding windows")
    subparsers.add_parser("telemetry", help="emit a 4-track white-box oscilloscope telemetry packet")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "demo":
        output = _demo(args.seed)
    elif args.command == "decompose":
        output = _decompose_demo(args.seed)
    elif args.command == "probe":
        output = {"layer_probe": _demo(args.seed)["layer_probe"]}
    elif args.command == "evidence":
        dataset = make_cross_voice_dataset(seed=args.seed)
        output = extract_evidence(dataset["sequences"][0]).to_dict(include_boundary=True)
    elif args.command == "fingerprint":
        output = _bank_demo(args.seed)
    elif args.command == "telemetry":
        output = _telemetry_demo(args.seed)
    else:
        random = np.random.default_rng(args.seed)
        target = random.normal(size=8)
        event = 1.4 * np.sin(np.linspace(0.0, np.pi, 12)).reshape(-1, 1) * target.reshape(1, -1)
        bank = EnrollmentBank()
        bank.enroll("event", [event])
        stream = make_stream(target=target, seed=args.seed)
        matcher = OnlineMatcher(
            bank,
            window_length=12,
            scales=(0.75, 1.0, 1.25),
            hop=2,
            iou_threshold=0.4,
        )
        matches = matcher.match(stream, top_k=3)
        output = {
            "window_lengths": list(matcher.window_lengths),
            "iou_threshold": matcher.iou_threshold,
            "matches": [match.__dict__ for match in matches],
        }
    print(_json(output))
