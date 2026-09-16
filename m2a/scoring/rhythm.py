"""Rhythmic adherence metrics: Onset F-measure and DTW envelope alignment."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple
import librosa
import numpy as np
from pydantic import BaseModel, Field
from scipy.spatial.distance import cdist


class RhythmScoreResult(BaseModel):
    """Evaluation result for rhythmic alignment and onset preservation."""

    f_measure: float = Field(ge=0.0, le=1.0, description="Onset F1 score within tolerance window.")
    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)
    tolerance_ms: float
    dtw_cost: float = Field(ge=0.0, description="Normalized Dynamic Time Warping alignment distance.")


def extract_audio_onsets(
    audio_path: Path | str,
    sr: int = 48000,
    hop_length: int = 512,
    backtrack: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract onset timestamps and normalized onset strength envelope from an audio file."""
    y, sample_rate = librosa.load(str(audio_path), sr=sr, mono=True)
    if len(y) == 0:
        return np.array([], dtype=np.float64), np.zeros(1, dtype=np.float64)

    onset_env = librosa.onset.onset_strength(y=y, sr=sample_rate, hop_length=hop_length)
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sample_rate,
        hop_length=hop_length,
        backtrack=backtrack,
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sample_rate, hop_length=hop_length)
    return onset_times, onset_env


def compute_onset_f_measure(
    reference_onsets: np.ndarray,
    candidate_onsets: np.ndarray,
    tolerance_ms: float = 50.0,
) -> Tuple[float, float, float]:
    """Calculate bipartite precision, recall, and F-measure given a tolerance window.

    Matches each reference onset to at most one candidate onset within tolerance.
    """
    if len(reference_onsets) == 0 and len(candidate_onsets) == 0:
        return 1.0, 1.0, 1.0
    if len(reference_onsets) == 0 or len(candidate_onsets) == 0:
        return 0.0, 0.0, 0.0

    tol_sec = tolerance_ms / 1000.0
    cost_matrix = cdist(reference_onsets[:, None], candidate_onsets[:, None])

    matched_ref = set()
    matched_cand = set()

    # Greedy bipartite matching on smallest temporal deviation
    indices = np.unravel_index(np.argsort(cost_matrix, axis=None), cost_matrix.shape)
    for r_idx, c_idx in zip(indices[0], indices[1]):
        if cost_matrix[r_idx, c_idx] > tol_sec:
            break
        if r_idx not in matched_ref and c_idx not in matched_cand:
            matched_ref.add(r_idx)
            matched_cand.add(c_idx)

    tp = len(matched_ref)
    fp = len(candidate_onsets) - tp
    fn = len(reference_onsets) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return float(f1), float(precision), float(recall)


def compute_dtw_envelope_cost(env_ref: np.ndarray, env_cand: np.ndarray) -> float:
    """Compute normalized Dynamic Time Warping alignment cost between onset strength envelopes."""
    if len(env_ref) == 0 or len(env_cand) == 0:
        return 1.0

    norm_ref = env_ref / (np.linalg.norm(env_ref) + 1e-8)
    norm_cand = env_cand / (np.linalg.norm(env_cand) + 1e-8)

    d, _ = librosa.sequence.dtw(norm_ref, norm_cand, metric="euclidean")
    normalized_cost = float(d[-1, -1] / (len(norm_ref) + len(norm_cand)))
    return normalized_cost


def evaluate_rhythm(
    reference_audio_path: Path | str,
    candidate_audio_path: Path | str,
    role: str = "comp",
    custom_tolerance_ms: Optional[float] = None,
) -> RhythmScoreResult:
    """Evaluate candidate audio against reference dry stem for rhythmic adherence."""
    if custom_tolerance_ms is not None:
        tolerance_ms = custom_tolerance_ms
    else:
        tolerance_ms = 25.0 if role == "drums" else 50.0

    ref_onsets, ref_env = extract_audio_onsets(reference_audio_path)
    cand_onsets, cand_env = extract_audio_onsets(candidate_audio_path)

    f1, prec, rec = compute_onset_f_measure(ref_onsets, cand_onsets, tolerance_ms=tolerance_ms)
    dtw_cost = compute_dtw_envelope_cost(ref_env, cand_env)

    return RhythmScoreResult(
        f_measure=f1,
        precision=prec,
        recall=rec,
        tolerance_ms=tolerance_ms,
        dtw_cost=dtw_cost,
    )