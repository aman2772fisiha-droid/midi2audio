"""Stage 6: Delay compensation via normalized cross-correlation against dry ground truth."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple
import librosa
import numpy as np
import scipy.signal
import soundfile as sf
from pydantic import BaseModel


class AlignmentResult(BaseModel):
    """Delay estimation and phase compensation metrics."""

    delay_samples: int
    delay_ms: float
    correlation_peak: float
    output_aligned_path: str


def compute_cross_correlation_delay(
    reference_audio: np.ndarray,
    candidate_audio: np.ndarray,
    sr: int = 48000,
    max_lag_ms: float = 150.0,
) -> Tuple[int, float, float]:
    """Calculate sample lag between reference and candidate using cross-correlation.

    Returns positive lag when candidate lags behind reference (started later).
    """
    ref_mono = np.mean(reference_audio, axis=0) if reference_audio.ndim > 1 else reference_audio
    cand_mono = np.mean(candidate_audio, axis=0) if candidate_audio.ndim > 1 else candidate_audio

    # Window up to the first 10 seconds for latency alignment
    max_samples = min(len(ref_mono), len(cand_mono), sr * 10)
    ref_sig = ref_mono[:max_samples]
    cand_sig = cand_mono[:max_samples]

    ref_sig = ref_sig - np.mean(ref_sig)
    cand_sig = cand_sig - np.mean(cand_sig)

    ref_norm = np.linalg.norm(ref_sig) + 1e-8
    cand_norm = np.linalg.norm(cand_sig) + 1e-8

    # Correlate candidate against reference
    corr = scipy.signal.correlate(cand_sig / cand_norm, ref_sig / ref_norm, mode="full")
    lags = scipy.signal.correlation_lags(len(cand_sig), len(ref_sig), mode="full")

    max_lag_samples = int(round((max_lag_ms / 1000.0) * sr))
    valid_mask = np.abs(lags) <= max_lag_samples

    valid_corr = corr[valid_mask]
    valid_lags = lags[valid_mask]

    peak_idx = int(np.argmax(valid_corr))
    best_lag_samples = int(valid_lags[peak_idx])
    peak_val = float(valid_corr[peak_idx])
    best_lag_ms = float((best_lag_samples / sr) * 1000.0)

    return best_lag_samples, best_lag_ms, peak_val


def align_restyle_stem_to_reference(
    reference_path: Path | str,
    candidate_path: Path | str,
    output_path: Path | str,
    sr: int = 48000,
) -> AlignmentResult:
    """Shift candidate audio to cancel latency introduced by generative endpoints."""
    ref_audio, _ = librosa.load(str(reference_path), sr=sr, mono=False)
    cand_audio, _ = librosa.load(str(candidate_path), sr=sr, mono=False)

    if cand_audio.ndim == 1:
        cand_audio = np.vstack((cand_audio, cand_audio))

    lag_samples, lag_ms, peak = compute_cross_correlation_delay(ref_audio, cand_audio, sr=sr)

    # Compensate delay: lag > 0 means candidate lags reference (trim start)
    if lag_samples > 0:
        aligned = cand_audio[:, lag_samples:]
    elif lag_samples < 0:
        pad = np.zeros((cand_audio.shape[0], abs(lag_samples)), dtype=cand_audio.dtype)
        aligned = np.hstack((pad, cand_audio))
    else:
        aligned = cand_audio

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_file), aligned.T, sr)

    return AlignmentResult(
        delay_samples=lag_samples,
        delay_ms=round(lag_ms, 2),
        correlation_peak=round(peak, 4),
        output_aligned_path=str(out_file),
    )