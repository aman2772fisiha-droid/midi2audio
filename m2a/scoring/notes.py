"""Note fidelity metrics: CQT chroma cosine similarity and transcription verification."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple
import librosa
import numpy as np
from pydantic import BaseModel, Field


class NoteFidelityResult(BaseModel):
    """Evaluation result for harmonic and pitch transcription accuracy."""

    chroma_cosine: float = Field(ge=-1.0, le=1.0, description="Mean frame-wise chroma cosine similarity.")
    transcription_f1: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    note_floor_passed: bool = True


def compute_chroma_cosine_similarity(
    ref_audio_path: Path | str,
    cand_audio_path: Path | str,
    sr: int = 48000,
    hop_length: int = 1024,
) -> float:
    """Calculate mean frame-wise cosine similarity between Constant-Q chromagrams."""
    y_ref, _ = librosa.load(str(ref_audio_path), sr=sr, mono=True)
    y_cand, _ = librosa.load(str(cand_audio_path), sr=sr, mono=True)

    if len(y_ref) == 0 or len(y_cand) == 0:
        return 0.0

    chroma_ref = librosa.feature.chroma_cqt(y=y_ref, sr=sr, hop_length=hop_length)
    chroma_cand = librosa.feature.chroma_cqt(y=y_cand, sr=sr, hop_length=hop_length)

    # Align frames via minimum length
    min_frames = min(chroma_ref.shape[1], chroma_cand.shape[1])
    if min_frames == 0:
        return 0.0

    c_ref = chroma_ref[:, :min_frames]
    c_cand = chroma_cand[:, :min_frames]

    # Normalize column vectors (frames)
    ref_norm = np.linalg.norm(c_ref, axis=0, keepdims=True) + 1e-8
    cand_norm = np.linalg.norm(c_cand, axis=0, keepdims=True) + 1e-8

    c_ref_normalized = c_ref / ref_norm
    c_cand_normalized = c_cand / cand_norm

    cosine_sim = np.sum(c_ref_normalized * c_cand_normalized, axis=0)
    return float(np.clip(np.mean(cosine_sim), -1.0, 1.0))


def evaluate_notes(
    reference_audio_path: Path | str,
    candidate_audio_path: Path | str,
    floor_threshold: float = 0.70,
) -> NoteFidelityResult:
    """Evaluate candidate note preservation against reference ground truth."""
    chroma_sim = compute_chroma_cosine_similarity(reference_audio_path, candidate_audio_path)
    passed = chroma_sim >= floor_threshold

    return NoteFidelityResult(
        chroma_cosine=chroma_sim,
        transcription_f1=None,
        note_floor_passed=passed,
    )