"""Cross-voice temporal disentanglement and structured acoustic fingerprints."""

from .evidence import (
    EvidenceCalibrator,
    StructuredEvidence,
    boundary_similarity,
    calibrate_response,
    calibrated_evidence_distance,
    evidence_distance,
    evidence_vector,
    extract_evidence,
    select_peak_indices,
)
from .fingerprint import (
    EnrollmentBank,
    FingerprintMatch,
    OnlineMatcher,
    fingerprint,
    temporal_iou,
)
from .geometry import (
    SpeakerSubspaceProjector,
    TemporalDecomposition,
    arc_length_resample,
    decompose,
    disentanglement_report,
    paired_bootstrap,
)
from .probing import calibrate_by_group, layerwise_probe, rank_auc
from .synthetic import make_cross_voice_dataset, make_stream

__all__ = [
    "TemporalDecomposition",
    "SpeakerSubspaceProjector",
    "arc_length_resample",
    "decompose",
    "StructuredEvidence",
    "EvidenceCalibrator",
    "calibrated_evidence_distance",
    "extract_evidence",
    "select_peak_indices",
    "EnrollmentBank",
    "FingerprintMatch",
    "OnlineMatcher",
    "fingerprint",
    "temporal_iou",
    "make_cross_voice_dataset",
    "make_stream",
    "disentanglement_report",
    "paired_bootstrap",
    "calibrate_by_group",
    "layerwise_probe",
    "rank_auc",
    "boundary_similarity",
    "calibrate_response",
    "evidence_distance",
    "evidence_vector",
]
