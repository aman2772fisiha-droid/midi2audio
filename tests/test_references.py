"""Verification tests for audio groove extraction, style card distillation, alignment, and mastering."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple
import librosa
import numpy as np
import pytest
import soundfile as sf
from m2a.groove.extract import extract_groove_template_from_audio
from m2a.mixmaster.align import align_restyle_stem_to_reference, compute_cross_correlation_delay
from m2a.mixmaster.master import audit_audio_levels, master_against_reference
from m2a.mixmaster.mix import mix_stems_to_stereo
from m2a.prompts.stylecard import distill_house_style_card


@pytest.fixture
def synthetic_groove_recording(tmp_path: Path) -> Path:
    """Create a 120 BPM drum loop with an exact 8th-note swing and known microtiming."""
    sr = 48000
    bpm = 120.0
    sec_per_16th = (60.0 / bpm) / 4.0  # 0.125s
    duration = 4.0  # 2 bars
    audio = np.zeros(int(sr * duration), dtype=np.float32)

    # Place 16th-note transient impulses across 2 bars
    for i in range(32):
        t = i * sec_per_16th
        # Add swing delay on odd 8th subdivisions (slots 2, 6, 10, 14, ...)
        if (i % 4) == 2:
            t += 0.020  # +20ms swing delay

        idx = int(t * sr)
        if idx < len(audio) - 100:
            audio[idx : idx + 100] = np.hanning(100)

    file_path = tmp_path / "swing_groove.wav"
    sf.write(str(file_path), audio, sr)
    return file_path


@pytest.fixture
def reference_audio_file(tmp_path: Path) -> Path:
    """Generate reference stereo master file."""
    sr = 48000
    t = np.arange(int(sr * 3.0)) / sr
    # Band-limited musical composite
    l_chan = 0.4 * np.sin(2 * np.pi * 100.0 * t) + 0.3 * np.sin(2 * np.pi * 1000.0 * t)
    r_chan = 0.4 * np.sin(2 * np.pi * 100.0 * t) + 0.3 * np.sin(2 * np.pi * 1200.0 * t)
    stereo = np.vstack((l_chan, r_chan)).T
    ref_path = tmp_path / "reference_master.wav"
    sf.write(str(ref_path), stereo, sr)
    return ref_path


def test_groove_extraction_recovers_tempo_and_swing(synthetic_groove_recording: Path):
    """Verify groove extraction recovers tempo and detects positive swing deviation."""
    template = extract_groove_template_from_audio(synthetic_groove_recording)

    assert pytest.approx(template.tempo_bpm, abs=2.0) == 120.0
    assert len(template.by_position_16) == 16
    assert len(template.accent_map_16) == 16
    assert template.swing_ratio_estimated > 0.52


def test_house_style_card_distillation(reference_audio_file: Path):
    """Confirm distillation extracts acoustic profiles and produces valid prompt descriptors."""
    style_card = distill_house_style_card([reference_audio_file], genre_hint="Vintage 70s Soul")

    assert style_card.target_genre_era == "Vintage 70s Soul"
    assert style_card.acoustic_summary.spectral_centroid_hz > 0.0
    assert len(style_card.key_descriptors) >= 2
    assert "drums" in style_card.instrumentation_cues
    assert len(style_card.negative_prompts) > 0


def test_cross_correlation_delay_alignment(tmp_path: Path):
    """Verify cross-correlation correctly computes and compensates positive millisecond lag."""
    sr = 48000
    t = np.arange(int(sr * 2.0)) / sr
    # Modulated transient signal (avoids periodic sine ambiguity)
    envelope = np.exp(-t * 5.0)
    sig = (0.5 * np.sin(2 * np.pi * 300.0 * t) * envelope).astype(np.float32)

    # Shift candidate later by 1200 samples (25 ms)
    lag_samples_ground_truth = 1200
    cand_sig = np.pad(sig, (lag_samples_ground_truth, 0))[: len(sig)]

    ref_path = tmp_path / "ref_lead.wav"
    cand_path = tmp_path / "cand_lead.wav"
    out_aligned = tmp_path / "aligned_lead.wav"

    sf.write(str(ref_path), sig, sr)
    sf.write(str(cand_path), cand_sig, sr)

    res = align_restyle_stem_to_reference(ref_path, cand_path, out_aligned, sr=sr)

    # Recovered lag must match ground truth within 5 samples
    assert abs(res.delay_samples - lag_samples_ground_truth) <= 5
    assert pytest.approx(res.delay_ms, abs=0.5) == 25.0
    assert Path(res.output_aligned_path).is_file()


def test_multi_stem_mix_and_master_pipeline(reference_audio_file: Path, tmp_path: Path):
    """Test stem mixing and mastering against reference with audio level verification."""
    sr = 48000
    t = np.arange(int(sr * 2.0)) / sr

    # Create synthetic stems
    drums_audio = (0.6 * np.sin(2 * np.pi * 80.0 * t)).astype(np.float32)
    comp_audio = (0.3 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)

    drums_path = tmp_path / "stem_drums.wav"
    comp_path = tmp_path / "stem_comp.wav"
    sf.write(str(drums_path), drums_audio, sr)
    sf.write(str(comp_path), comp_audio, sr)

    # 1. Mix Stems
    mixed_path = tmp_path / "rough_mix.wav"
    mix_stems_to_stereo([(drums_path, "drums"), (comp_path, "comp")], mixed_path, sr=sr)
    assert mixed_path.is_file()

    # 2. Master Against Reference
    final_master_path = tmp_path / "final_master.wav"
    audit = master_against_reference(mixed_path, reference_audio_file, final_master_path, sr=sr)

    assert Path(audit.output_master_path).is_file()
    assert audit.is_clipping is False
    assert audit.true_peak_dbfs <= 0.0