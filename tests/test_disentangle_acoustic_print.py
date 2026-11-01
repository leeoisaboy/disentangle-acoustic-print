from __future__ import annotations

import contextlib
import json
from io import StringIO
import unittest
from unittest.mock import patch

import numpy as np

from disentangle_acoustic_print import (
    EnrollmentBank,
    EvidenceCalibrator,
    OnlineMatcher,
    SpeakerSubspaceProjector,
    arc_length_resample,
    decompose,
    disentanglement_report,
    extract_evidence,
    evidence_distance,
    fingerprint,
    make_cross_voice_dataset,
    make_stream,
    select_peak_indices,
    temporal_iou,
)
from disentangle_acoustic_print.cli import main
from disentangle_acoustic_print.geometry import (
    frame_shuffle,
    half_swap,
    pair_swap,
    reverse,
    smooth_time_warp,
)
from disentangle_acoustic_print.probing import layerwise_probe, rank_auc


class LibraryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = np.random.default_rng(17)
        self.frames = np.cumsum(self.rng.normal(scale=0.3, size=(16, 6)), axis=0)

    def test_decomposition_shapes(self) -> None:
        result = decompose(self.frames)
        self.assertEqual(result.centroid.shape, (6,))
        self.assertEqual(result.delta.shape, (15, 6))
        self.assertEqual(result.residual.shape, (16, 6))
        self.assertEqual(result.speed.shape, (15,))
        self.assertEqual(result.acceleration.shape, (14,))
        self.assertEqual(result.turning.shape, (14,))

    def test_permutation_controls_preserve_multiset(self) -> None:
        for control in (reverse, half_swap, pair_swap, frame_shuffle):
            edited = control(self.frames)
            np.testing.assert_allclose(
                np.sort(edited, axis=0),
                np.sort(self.frames, axis=0),
            )

    def test_cross_voice_report(self) -> None:
        dataset = make_cross_voice_dataset()
        report = disentanglement_report(
            dataset["sequences"], dataset["content_ids"], dataset["speaker_ids"]
        )
        self.assertIn("delta", report)
        self.assertGreaterEqual(report["delta"]["pairs_cross_voice"], 30)

    def test_synthetic_speaker_nuisance_is_deterministic_and_active(self) -> None:
        plain = make_cross_voice_dataset(
            seed=5,
            warp_strength=0.0,
            rotation_strength=0.0,
        )["sequences"]
        warped = make_cross_voice_dataset(
            seed=5,
            warp_strength=0.35,
            rotation_strength=0.0,
        )["sequences"]
        rotated = make_cross_voice_dataset(
            seed=5,
            warp_strength=0.0,
            rotation_strength=0.3,
        )["sequences"]
        repeated = make_cross_voice_dataset(
            seed=5,
            warp_strength=0.35,
            rotation_strength=0.0,
        )["sequences"]
        self.assertFalse(np.allclose(plain[0], warped[0]))
        self.assertFalse(np.allclose(plain[0], rotated[0]))
        for left, right in zip(warped, repeated):
            np.testing.assert_allclose(left, right)

    def test_zero_time_warp_is_identity(self) -> None:
        np.testing.assert_array_equal(
            smooth_time_warp(self.frames, strength=0.0, seed=9),
            self.frames,
        )

    def test_evidence_prefers_matching_shape(self) -> None:
        aligned = np.sin(np.linspace(0.0, np.pi, 12)).reshape(-1, 1) * np.ones((1, 4))
        shifted = np.cos(np.linspace(0.0, np.pi, 12)).reshape(-1, 1) * np.ones((1, 4))
        self.assertLess(
            evidence_distance(extract_evidence(aligned), extract_evidence(aligned + 0.01)),
            evidence_distance(extract_evidence(aligned), extract_evidence(shifted)),
        )

    def test_change_position_uses_zero_mean_cusum(self) -> None:
        fast_first = np.asarray([3, 3, 3, 1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
        fast_last = fast_first[::-1].copy()

        def trajectory(speed: np.ndarray) -> np.ndarray:
            steps = np.stack((speed, np.zeros_like(speed)), axis=1)
            return np.vstack((np.zeros(2), np.cumsum(steps, axis=0)))

        early = extract_evidence(trajectory(fast_first)).change_position
        late = extract_evidence(trajectory(fast_last)).change_position
        self.assertLess(early, 0.35)
        self.assertGreater(late, 0.65)

    def test_peak_selection_enforces_minimum_distance(self) -> None:
        peaks = select_peak_indices(
            np.asarray([0.0, 1.0, 0.2, 4.0, 3.0, 0.1, 2.0, 0.0]),
            count=2,
            min_distance=3,
        )
        self.assertEqual(len(peaks), 2)
        self.assertEqual(peaks[0], 3)
        self.assertGreaterEqual(abs(peaks[0] - peaks[1]), 3)

    def test_layerwise_probe_keeps_per_column_variance(self) -> None:
        random = np.random.default_rng(7)
        rows = []
        contents = []
        speakers = []
        for content in range(2):
            for speaker in range(2):
                for _ in range(3):
                    rows.append([2.0 * content, 3.0 * speaker])
                    contents.append(f"phrase-{content}")
                    speakers.append(f"voice-{speaker}")
        features = np.asarray(rows) + random.normal(scale=0.05, size=(len(rows), 2))
        probe = layerwise_probe(features, contents, speakers)
        self.assertGreater(probe["content_variance"][0], probe["content_variance"][1])
        self.assertGreater(probe["speaker_variance"][1], probe["speaker_variance"][0])
        self.assertTrue(np.any(probe["residual_variance"] > 0.0))

    def test_fingerprint_retrieval(self) -> None:
        directions = self.rng.normal(size=(3, 5))
        bank = EnrollmentBank()
        for label, direction in zip(("alpha", "beta", "gamma"), directions):
            profile = np.sin(np.linspace(0.0, np.pi, 12)).reshape(-1, 1)
            bank.enroll(label, [profile * direction.reshape(1, -1)])
        query = np.sin(np.linspace(0.0, np.pi, 12)).reshape(-1, 1) * directions[1].reshape(1, -1)
        self.assertEqual(bank.search(query)[0].key, "beta")

    def test_fingerprint_is_unit_norm(self) -> None:
        vector = fingerprint(self.frames)
        self.assertAlmostEqual(float(np.linalg.norm(vector)), 1.0, places=10)

    def test_repeated_enrollment_uses_sample_weighted_mean(self) -> None:
        directions = self.rng.normal(size=(3, 5))
        profile = np.sin(np.linspace(0.0, np.pi, 12)).reshape(-1, 1)
        samples = [profile * direction.reshape(1, -1) for direction in directions]
        sequential = EnrollmentBank()
        sequential.enroll("phrase", [samples[0]])
        sequential.enroll("phrase", samples[1:])
        combined = EnrollmentBank()
        combined.enroll("phrase", samples)
        self.assertEqual(sequential.count("phrase"), 3)
        np.testing.assert_allclose(sequential.matrix(), combined.matrix(), atol=1e-12)

    def test_speaker_projection_removes_centroid_subspace(self) -> None:
        sequences = []
        speakers = []
        profile = np.linspace(0.0, 1.0, 10).reshape(-1, 1)
        for index, offset in enumerate((0.0, 2.0, 4.0)):
            sequences.append(profile * np.asarray([[0.4, 0.8]]) + np.asarray([[offset, 0.0]]))
            speakers.append(f"voice-{index}")
        projector = SpeakerSubspaceProjector.fit(sequences, speakers)
        cleaned = projector.transform(sequences[2])
        np.testing.assert_allclose(cleaned[:, 0], projector.mean[0], atol=1e-10)
        np.testing.assert_allclose(cleaned[:, 1], sequences[2][:, 1], atol=1e-10)

    def test_arc_length_resampling_is_time_warp_stable(self) -> None:
        from disentangle_acoustic_print.geometry import smooth_time_warp

        positions = np.linspace(0.0, 1.0, 41)
        curve = np.stack((np.sin(2.0 * np.pi * positions), np.cos(3.0 * np.pi * positions)), axis=1)
        warped = smooth_time_warp(curve, strength=0.45, seed=9)
        np.testing.assert_allclose(
            arc_length_resample(curve, 20),
            arc_length_resample(warped, 20),
            atol=0.02,
        )

    def test_online_matcher_locates_event(self) -> None:
        target = self.rng.normal(size=(6,))
        event = np.sin(np.linspace(0.0, np.pi, 12)).reshape(-1, 1) * target.reshape(1, -1)
        bank = EnrollmentBank()
        bank.enroll("event", [event])
        stream = make_stream(dimensions=6, target=target, insert_start=22, seed=17)
        match = OnlineMatcher(bank, window_length=12, hop=2).match(stream)[0]
        self.assertEqual(match.key, "event")
        self.assertLessEqual(abs(match.start - 22), 3)

    def test_online_matcher_multiscale_nms_returns_disjoint_windows(self) -> None:
        target = self.rng.normal(size=(6,))
        event = np.sin(np.linspace(0.0, np.pi, 12)).reshape(-1, 1) * target.reshape(1, -1)
        bank = EnrollmentBank()
        bank.enroll("event", [event])
        stream = 0.02 * self.rng.normal(size=(70, 6))
        for start in (10, 44):
            stream[start : start + 12] += event
        matcher = OnlineMatcher(
            bank,
            window_lengths=(9, 12, 15),
            hop=1,
            iou_threshold=0.4,
        )
        matches = matcher.match(stream, top_k=3)
        self.assertEqual(matches[0].key, "event")
        for left in range(len(matches)):
            for right in range(left + 1, len(matches)):
                self.assertLessEqual(
                    temporal_iou(
                        matches[left].start,
                        matches[left].end,
                        matches[right].start,
                        matches[right].end,
                    ),
                    0.4,
                )

    def test_hybrid_evidence_search_is_available(self) -> None:
        bank = EnrollmentBank(evidence_weight=0.25)
        for index, frequency in enumerate((1.0, 2.0, 3.0)):
            phase = np.linspace(0.0, np.pi, 12)
            sequence = np.stack(
                (np.sin(frequency * phase), np.cos(frequency * phase)),
                axis=1,
            )
            bank.enroll(f"word-{index}", [sequence])
        phase = np.linspace(0.0, np.pi, 12)
        query = np.stack((np.sin(2.0 * phase), np.cos(2.0 * phase)), axis=1)
        match = bank.search(query)[0]
        self.assertIsNotNone(match.fingerprint_score)
        self.assertIsNotNone(match.evidence_distance)

    def test_hybrid_calibration_cache_invalidates_on_enroll(self) -> None:
        bank = EnrollmentBank(evidence_weight=0.25)
        for index, frequency in enumerate((1.0, 2.0, 3.0)):
            phase = np.linspace(0.0, np.pi, 12)
            sequence = np.stack(
                (np.sin(frequency * phase), np.cos(frequency * phase)),
                axis=1,
            )
            bank.enroll(f"word-{index}", [sequence])
        phase = np.linspace(0.0, np.pi, 12)
        query = np.stack((np.sin(2.0 * phase), np.cos(2.0 * phase)), axis=1)
        original = EvidenceCalibrator.from_vectors
        with patch.object(EvidenceCalibrator, "from_vectors", wraps=original) as fit:
            first = bank.search(query)
            second = bank.search(query)
            self.assertEqual(fit.call_count, 1)
            np.testing.assert_allclose(first[0].score, second[0].score)
            bank.enroll("word-3", [query])
            self.assertIsNone(bank._evidence_cache)
            bank.search(query)
            self.assertEqual(fit.call_count, 2)

    def test_rank_auc_extremes(self) -> None:
        self.assertEqual(rank_auc([0.8, 0.9], [0.1, 0.2]), 1.0)

    def test_cli_demo_outputs_json(self) -> None:
        buffer = StringIO()
        with contextlib.redirect_stdout(buffer):
            main(["demo"])
        value = json.loads(buffer.getvalue())
        self.assertEqual(value["sample_count"], 36)


if __name__ == "__main__":
    unittest.main()
