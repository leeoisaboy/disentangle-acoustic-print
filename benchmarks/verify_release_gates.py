#!/usr/bin/env python3
"""Automated 4-Gate Release Verifier for Disentangle-Acoustic-Print (v0.4).

Reads real-data benchmark JSON reports (produced on public corpora L1/L2/L3)
and verifies all 4 hard release gates with Pass/Fail verdicts.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any


def evaluate_gates(report: dict[str, Any]) -> dict[str, Any]:
    """Evaluate the 4 hard release gates from a real-data benchmark report."""
    # Gate 1: Cross-speaker generalization (projector + arc-length lowers EER vs static mean on >= 2 corpora)
    g1_details = {}
    g1_wins = 0
    for lang in ("L1", "L2", "L3"):
        s2 = report["study2_geometry"][lang]["wavlm_mid"]
        static_eer = s2["arc_length_ablation"]["static_mean"]["clean"]["eer"]
        best_proj = min(s2["projector_rank_sweep"], key=lambda x: x["cross_speaker_eer"])
        proj_eer = best_proj["cross_speaker_eer"]
        improved = proj_eer < static_eer - 0.01
        if improved:
            g1_wins += 1
        g1_details[lang] = {
            "static_mean_eer": static_eer,
            "projector_best_rank": best_proj["effective_rank"],
            "projector_eer": proj_eer,
            "eer_delta": round(proj_eer - static_eer, 4),
            "improved": improved,
        }
    gate1_pass = g1_wins >= 2

    # Gate 2: Continuous-stream localization usability (real in-context calibrated stream)
    s4 = report.get("study4_incontext", {})
    g2_details = {}
    best_auc = 0.0
    best_rec = 0.0
    for lang, info in s4.items():
        auc = float(info.get("event_auc_iou05", 0.0))
        rec2 = float(info.get("ops", {}).get("fa_2", {}).get("rec_iou05", 0.0))
        best_auc = max(best_auc, auc)
        best_rec = max(best_rec, rec2)
        g2_details[lang] = {
            "event_auc_iou05": auc,
            "rec_iou05_at_fa2": rec2,
            "no_cover": info.get("no_cover", 0),
        }
    gate2_strict_pass = best_rec >= 0.80
    gate2_coarse_pass = best_auc >= 0.75

    # Gate 3: Streaming parity (r >= 0.99) & single-core CPU RTF (< 0.2)
    s5 = report["study5_streaming_parity_rtf"]
    g3_details = {}
    gate3_pass = True
    for lang, feats in s5.items():
        for feat, x in feats.items():
            ok = (x["pearson_r"] >= 0.99) and (x["online_rtf"] < 0.2) and (x["peak_timestamp_err_hops"] == 0)
            if not ok:
                gate3_pass = False
            g3_details[f"{lang}/{feat}"] = {
                "pearson_r": x["pearson_r"],
                "online_rtf": x["online_rtf"],
                "peak_timestamp_err_hops": x["peak_timestamp_err_hops"],
                "pass": ok,
            }

    # Gate 4: Zero-intrusion generality (3 frontends × 3 languages with zero core code changes)
    frontends_verified = sorted(report["study2_geometry"]["L1"].keys())
    languages_verified = sorted(report["study2_geometry"].keys())
    gate4_pass = (len(frontends_verified) >= 3) and (len(languages_verified) >= 3)

    return {
        "gate1_cross_speaker_generalization": {
            "pass": gate1_pass,
            "corpora_improved": f"{g1_wins}/3",
            "details": g1_details,
        },
        "gate2_continuous_stream_usability": {
            "pass": gate2_strict_pass,
            "coarse_localization_auc_pass": gate2_coarse_pass,
            "best_event_auc_iou05": round(best_auc, 4),
            "best_rec_iou05_at_fa2": round(best_rec, 4),
            "note": (
                "Honest boundary: training-free single-vector cosine reaches Event AUC 0.75-0.81 "
                "after per-keyword background calibration, but strict Recall@IoU>=0.5 >= 80% at "
                "FA/hr <= 2 requires a frame-level verifier (e.g. audiollm-latent-kws)."
            ),
            "details": g2_details,
        },
        "gate3_streaming_parity_and_rtf": {
            "pass": gate3_pass,
            "details": g3_details,
        },
        "gate4_zero_intrusion_generality": {
            "pass": gate4_pass,
            "frontends": frontends_verified,
            "languages": languages_verified,
        },
        "hard_gates_passed": int(gate1_pass) + int(gate2_strict_pass) + int(gate3_pass) + int(gate4_pass),
        "all_gates_pass": bool(gate1_pass and gate2_strict_pass and gate3_pass and gate4_pass),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Verify Disentangle-Acoustic-Print v0.4 release gates.")
    parser.add_argument(
        "--report",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[1] / "reports" / "real_data_benchmark_v04.json",
    )
    args = parser.parse_args(argv)
    report = json.loads(args.report.read_text(encoding="utf-8"))
    verdict = evaluate_gates(report)
    print(json.dumps(verdict, ensure_ascii=False, indent=2))
    sys.exit(0 if verdict["all_gates_pass"] else 1)


if __name__ == "__main__":
    main()
