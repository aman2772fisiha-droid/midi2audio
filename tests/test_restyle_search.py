"""Verification tests for successive halving, chunking, and financial circuit breakers."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
import soundfile as sf
from m2a.restyle.base import CandidateResult, EndpointErrorType, EndpointException, RestyleEndpoint
from m2a.restyle.chunking import rejoin_chunks_equal_power, split_audio_into_chunks
from m2a.restyle.stable_audio import StableAudioEndpoint
from m2a.scoring.clap_sim import ClapScorer
from m2a.scoring.combine import CandidateScorer
from m2a.search.loop import BudgetTracker, SuccessiveHalvingOptimizer


@pytest.fixture
def synthetic_dry_stem(tmp_path: Path) -> Path:
    """Generate a clean 4-second dry audio stem with clear rhythmic transients at 48 kHz."""
    sr = 48000
    duration = 4.0
    total_samples = int(sr * duration)
    audio = np.zeros(total_samples, dtype=np.float32)

    # 8 clear comping chord hits (every 0.5s) with exponential decay envelope
    for beat in range(8):
        start_idx = int(beat * 0.5 * sr)
        hit_len = int(0.35 * sr)
        t = np.arange(hit_len) / sr
        env = np.exp(-t * 12.0)
        # Harmonically rich dual-tone chord
        hit = (0.5 * np.sin(2 * np.pi * 440.0 * t) + 0.3 * np.sin(2 * np.pi * 554.37 * t)) * env
        audio[start_idx : start_idx + hit_len] = hit.astype(np.float32)

    stereo = np.vstack((audio, audio)).T
    stem_path = tmp_path / "dry_comp_stem.wav"
    sf.write(str(stem_path), stereo, sr)
    return stem_path


def test_budget_circuit_breaker_halts_execution(synthetic_dry_stem: Path):
    """Prove the optimizer halts when max_endpoint_calls or max_usd is hit."""
    # Set strict budget limit: 3 calls max
    budget = BudgetTracker(max_calls=3, max_usd=1.00)
    endpoint = StableAudioEndpoint(config={}, api_key="mock_key")
    scorer = CandidateScorer(clap_scorer=ClapScorer())

    optimizer = SuccessiveHalvingOptimizer(
        endpoint=endpoint,
        scorer=scorer,
        budget=budget,
        coarse_strengths=[0.6, 0.7, 0.8, 0.9],
        coarse_seeds=[42],
    )

    res = optimizer.optimize_stem(synthetic_dry_stem, role="comp")

    assert res.total_calls == 3
    assert res.budget_exhausted is True
    assert len(res.all_evaluated) == 3


def test_bar_aligned_chunking_and_equal_power_rejoin(synthetic_dry_stem: Path, tmp_path: Path):
    """Confirm chunk splitting on bar timestamps and smooth equal-power crossfading."""
    # 4 bars spanning 4.0 seconds (1.0 sec per bar)
    bar_timestamps = [0.0, 1.0, 2.0, 3.0, 4.0]

    chunks = split_audio_into_chunks(
        audio_path=synthetic_dry_stem,
        bar_timestamps=bar_timestamps,
        bars_per_chunk=2,
        overlap_bars=1,
        output_dir=tmp_path / "chunks",
    )

    assert len(chunks) > 1
    for chunk in chunks:
        assert Path(chunk.audio_path).is_file()

    # Rejoin using equal-power crossfade
    rejoined_path = tmp_path / "rejoined.wav"
    out_file = rejoin_chunks_equal_power(
        chunk_paths=[c.audio_path for c in chunks],
        crossfade_duration_sec=0.1,
        output_path=rejoined_path,
        sr=48000,
    )

    assert out_file.is_file()
    y_rejoined, sr = sf.read(str(out_file))
    assert sr == 48000
    assert len(y_rejoined) > 0


def test_successive_halving_selects_best_candidate(synthetic_dry_stem: Path):
    """Confirm the optimizer returns a winning candidate passing all quality gates."""
    budget = BudgetTracker(max_calls=15, max_usd=5.00)
    endpoint = StableAudioEndpoint(config={}, api_key="mock_key")
    scorer = CandidateScorer(clap_scorer=ClapScorer())

    optimizer = SuccessiveHalvingOptimizer(
        endpoint=endpoint,
        scorer=scorer,
        budget=budget,
        coarse_strengths=[0.60, 0.75],
        coarse_seeds=[42],
    )

    res = optimizer.optimize_stem(synthetic_dry_stem, role="comp")

    assert res.best_candidate is not None
    assert res.best_candidate.evaluation.passed_hard_floors is True
    assert res.best_candidate.evaluation.composite_score > 0.0