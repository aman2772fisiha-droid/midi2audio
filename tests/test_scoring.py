"""Synthetic corruption suites proving monotonic metric degradation and hard floor gating."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple
import numpy as np
import pytest
import soundfile as sf
from m2a.scoring.clap_sim import ClapScorer
from m2a.scoring.combine import CandidateScorer
from m2a.scoring.fad import compute_frechet_distance
from m2a.scoring.notes import compute_chroma_cosine_similarity
from m2a.scoring.rhythm import compute_onset_f_measure, evaluate_rhythm


@pytest.fixture
def click_track_pair(tmp_path: Path) -> Tuple[Path, Path]:
    """Create a 4-beat percussive reference audio file and clean duplicate candidate."""
    sr = 48000
    dur = 2.0
    total_samples = int(sr * dur)
    audio = np.zeros(total_samples, dtype=np.float32)

    # 4 distinct transient pulses at 0.0s, 0.5s, 1.0s, 1.5s
    for t in [0.0, 0.5, 1.0, 1.5]:
        idx = int(t * sr)
        audio[idx : idx + 200] = np.sin(2 * np.pi * 1000.0 * np.arange(200) / sr)

    ref_path = tmp_path / "ref.wav"
    cand_path = tmp_path / "cand.wav"
    sf.write(str(ref_path), audio, sr)
    sf.write(str(cand_path), audio, sr)
    return ref_path, cand_path


def test_onset_f_measure_monotonic_degradation():
    """Verify onset F-measure strictly degrades as temporal shift increases."""
    ref_onsets = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    tolerance_ms = 50.0

    shifts = [0.0, 15.0, 30.0, 45.0, 60.0]  # ms
    scores = []

    for s in shifts:
        cand_onsets = ref_onsets + (s / 1000.0)
        f1, _, _ = compute_onset_f_measure(ref_onsets, cand_onsets, tolerance_ms=tolerance_ms)
        scores.append(f1)

    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1], f"Non-monotonic drop at index {i}: {scores}"
    assert scores[-1] == 0.0, "Offset exceeding tolerance must yield F1 of 0.0"


def test_chroma_cosine_monotonic_degradation_on_pitch_shift(tmp_path: Path):
    """Verify chroma cosine similarity decreases monotonically under pitch deviations."""
    sr = 48000
    t = np.arange(int(sr * 1.5)) / sr

    # Base A4 (440 Hz)
    sig_base = 0.8 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
    ref_file = tmp_path / "base_440.wav"
    sf.write(str(ref_file), sig_base, sr)

    # Frequency corruptions
    test_freqs = [440.0, 450.0, 480.0, 550.0, 660.0]
    similarities = []

    for freq in test_freqs:
        sig_test = 0.8 * np.sin(2 * np.pi * freq * t).astype(np.float32)
        cand_file = tmp_path / f"test_{int(freq)}.wav"
        sf.write(str(cand_file), sig_test, sr)
        sim = compute_chroma_cosine_similarity(ref_file, cand_file, sr=sr)
        similarities.append(sim)

    for i in range(len(similarities) - 1):
        assert similarities[i] >= similarities[i + 1] - 0.05, f"Chroma sim did not degrade monotonically: {similarities}"


def test_frechet_distance_identity_and_perturbation():
    """Verify FAD evaluates to 0.0 on identical distributions and increases with offset."""
    rng = np.random.default_rng(123)
    emb_ref = rng.normal(0.0, 1.0, size=(100, 32))
    mu_ref = np.mean(emb_ref, axis=0)
    sigma_ref = np.cov(emb_ref, rowvar=False)

    fad_zero = compute_frechet_distance(mu_ref, sigma_ref, mu_ref, sigma_ref)
    assert pytest.approx(fad_zero, abs=1e-5) == 0.0

    fad_small = compute_frechet_distance(mu_ref, sigma_ref, mu_ref + 0.5, sigma_ref)
    fad_large = compute_frechet_distance(mu_ref, sigma_ref, mu_ref + 2.0, sigma_ref)

    assert fad_large > fad_small > fad_zero


def test_candidate_scorer_hard_floor_rejection(click_track_pair: Tuple[Path, Path], tmp_path: Path):
    """Confirm candidates failing rhythmic or harmonic floors receive a final composite score of 0.0."""
    ref_path, cand_path = click_track_pair
    clap = ClapScorer()
    scorer = CandidateScorer(clap_scorer=clap)

    # Baseline perfect candidate
    perfect_res = scorer.score_candidate(
        reference_audio_path=ref_path,
        candidate_audio_path=cand_path,
        role="drums",
        prompt="Tight studio drums",
    )
    assert perfect_res.passed_hard_floors is True
    assert perfect_res.composite_score > 0.8

    # Create corrupted candidate with +60ms latency shift (violates 25ms drum floor)
    sr = 48000
    y, _ = sf.read(str(cand_path))
    shifted = np.pad(y, (int(0.060 * sr), 0))[: len(y)]
    corrupted_path = tmp_path / "corrupted_drum.wav"
    sf.write(str(corrupted_path), shifted, sr)

    failed_res = scorer.score_candidate(
        reference_audio_path=ref_path,
        candidate_audio_path=corrupted_path,
        role="drums",
        prompt="Tight studio drums",
    )

    assert failed_res.rhythm_floor_passed is False
    assert failed_res.passed_hard_floors is False
    assert failed_res.composite_score == 0.0
    assert "fell below hard floor" in failed_res.rejection_reason