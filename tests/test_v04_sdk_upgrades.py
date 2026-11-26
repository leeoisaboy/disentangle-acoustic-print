from __future__ import annotations

import pathlib
import unittest

import numpy as np

from disentangle_acoustic_print import (
    BackgroundCalibration,
    EnrollmentBank,
    FeatureFrontend,
    MelDeltaExtractor,
    OnlineMatcher,
    RollingAcousticStream,
    SpeakerSubspaceProjector,
    arc_length_resample,
    fingerprint,
    make_cross_voice_dataset,
    synthesize_phrase_waveform,
)


class SdkV04Tests(unittest.TestCase):
    def test_arc_length_vad_smoothing_suppresses_noisy_padding(self) -> None:
        rng = np.random.default_rng(42)
        t = np.linspace(0.0, np.pi, 24)
        clean = np.stack((np.sin(t), np.cos(2.0 * t), np.sin(3.0 * t)), axis=1)
        pad = 0.04 * rng.normal(size=(12, 3))
        noisy_padded = np.vstack((pad, clean + 0.03 * rng.normal(size=clean.shape), pad))

        ref = arc_length_resample(clean, 10)
        raw = arc_length_resample(noisy_padded, 10)
        guarded = arc_length_resample(noisy_padded, 10, trim_silence=True, smooth_alpha=0.5)
        self.assertLess(
            float(np.linalg.norm(guarded - ref)),
            float(np.linalg.norm(raw - ref)),
        )

    def test_projector_auto_rank_max_rank_and_soft_shrinkage(self) -> None:
        dataset = make_cross_voice_dataset(dimensions=10, seed=11)
        hard = SpeakerSubspaceProjector.fit(
            dataset["sequences"],
            dataset["speaker_ids"],
            variance_threshold=0.85,
            max_rank=2,
            shrinkage=1.0,
        )
        soft = SpeakerSubspaceProjector.fit(
            dataset["sequences"],
            dataset["speaker_ids"],
            variance_threshold=0.85,
            max_rank=2,
            shrinkage=0.5,
        )
        self.assertLessEqual(hard.basis.shape[1], 2)
        self.assertEqual(soft.shrinkage, 0.5)
        sample = dataset["sequences"][0]
        diff_hard = float(np.linalg.norm(sample - hard.transform(sample)))
        diff_soft = float(np.linalg.norm(sample - soft.transform(sample)))
        self.assertAlmostEqual(diff_soft, 0.5 * diff_hard, places=6)

    def test_content_preset_fingerprint_is_unit_norm(self) -> None:
        rng = np.random.default_rng(9)
        frames = np.cumsum(rng.normal(scale=0.25, size=(18, 8)), axis=0)
        vec = fingerprint(frames, output_length=8, preset="content")
        self.assertAlmostEqual(float(np.linalg.norm(vec)), 1.0, places=10)

    def test_feature_frontend_contract_and_streaming_parity(self) -> None:
        frontend = MelDeltaExtractor(
            sample_rate=16000,
            frame_ms=25.0,
            hop_ms=20.0,
            n_mels=8,
            include_delta=False,
            normalize_utterance=False,
        )
        self.assertIsInstance(frontend, FeatureFrontend)
        self.assertAlmostEqual(frontend.frame_rate_hz, 50.0, places=4)

        wave = synthesize_phrase_waveform("parity-check", duration_sec=2.0, sample_rate=16000)
        frames, fps = frontend.extract_with_rate(wave, sample_rate=16000)
        self.assertAlmostEqual(fps, 50.0, places=4)
        self.assertEqual(frames.shape[1], 8)

        # Verify offline vs incremental chunk scoring parity >= 0.99
        bank = EnrollmentBank(output_length=8, evidence_weight=0.0)
        bank.enroll("target", [frames[15:35]])
        bank.enroll("other", [frames[50:70]])
        matcher = OnlineMatcher(
            bank,
            window_ms=400.0,
            hop_ms=40.0,
            frame_rate_hz=fps,
            scales=(0.75, 1.0, 1.25),
        )
        max_w = max(matcher.window_lengths)
        hop = matcher.hop
        off_scores = [
            max(bank.search(frames[end - w : end], top_k=1)[0].score for w in matcher.window_lengths)
            for end in range(max_w, len(frames) + 1, hop)
        ]
        # Incremental 100ms chunk accumulation
        chunk_len = 5
        buf = np.zeros((0, frames.shape[1]), dtype=np.float64)
        next_end = max_w
        on_scores = []
        for pos in range(0, len(frames), chunk_len):
            buf = np.vstack((buf, frames[pos : pos + chunk_len])) if len(buf) else frames[pos : pos + chunk_len]
            while len(buf) >= next_end:
                on_scores.append(
                    max(bank.search(buf[next_end - w : next_end], top_k=1)[0].score for w in matcher.window_lengths)
                )
                next_end += hop
        corr = float(np.corrcoef(off_scores, on_scores)[0, 1])
        self.assertGreaterEqual(corr, 0.9999)

    def test_background_calibration_and_auto_window_matcher(self) -> None:
        rng = np.random.default_rng(123)
        bank = EnrollmentBank(output_length=8, evidence_weight=0.0)
        t = np.linspace(0.0, np.pi, 16).reshape(-1, 1)
        d1 = rng.normal(size=(1, 6))
        d2 = rng.normal(size=(1, 6))
        bank.enroll("kw-alpha", [np.sin(t) * d1, np.sin(t) * d1 + 0.05])
        bank.enroll("kw-beta", [np.sin(t) * d2])
        self.assertEqual(bank.median_template_length(), 16)

        neg_streams = [np.cumsum(0.1 * rng.normal(size=(80, 6)), axis=0) for _ in range(4)]
        cal = bank.calibrate_background(
            neg_streams,
            hop=4,
            target_fa_per_hour=1.0,
            frame_rate_hz=50.0,
        )
        self.assertTrue(bank.is_calibrated)
        self.assertIsInstance(cal["kw-alpha"], BackgroundCalibration)
        self.assertIsNotNone(bank.threshold_for("kw-alpha"))

        matcher = OnlineMatcher(
            bank,
            window_length="auto",
            scales=(0.75, 1.0, 1.25),
            hop=2,
            use_calibration=True,
        )
        stream = 0.05 * rng.normal(size=(60, 6))
        stream[20:36] += np.sin(t) * d1
        matches = matcher.match(stream, top_k=1)
        self.assertEqual(matches[0].key, "kw-alpha")
        self.assertIsNotNone(matches[0].calibrated_z)
        self.assertGreater(matches[0].calibrated_z or 0.0, 2.0)

    def test_rolling_stream_ms_config_and_zero_evidence_default(self) -> None:
        engine = RollingAcousticStream(
            max_frames=100,
            dimension=16,
            frame_rate_hz=50.0,
            window_ms=320.0,
            hop_ms=40.0,
        )
        self.assertEqual(engine.evidence_weight, 0.0)
        self.assertEqual(engine.window_length, 16)
        self.assertEqual(engine.hop, 2)

    def test_repo_has_no_private_paths_or_ctc_references(self) -> None:
        root = pathlib.Path(__file__).resolve().parents[1]
        pkg = root / "disentangle_acoustic_print"
        forbidden = ("/home/", "ctc_loss", "CTCLoss")
        for py in pkg.rglob("*.py"):
            text = py.read_text(encoding="utf-8")
            for token in forbidden:
                self.assertNotIn(token, text, f"Forbidden token {token!r} in {py}")


if __name__ == "__main__":
    unittest.main()
