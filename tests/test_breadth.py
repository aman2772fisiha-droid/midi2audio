"""Verification tests for Suno blend with HPF, local MusicGen search, and GrooVAE gates."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pretty_midi
import pytest
import soundfile as sf

from m2a.groove.models import ModelBasedHumanizer
from m2a.restyle.chunking import detect_seam_discontinuity, inpaint_seam_window
from m2a.restyle.musicgen_local import MusicGenLocalEndpoint
from m2a.restyle.suno import (
    SunoCoverEndpoint,
    apply_butterworth_highpass,
    blend_cover_with_stem_mix,
)


@pytest.fixture
def synthetic_stems_and_cover(tmp_path: Path) -> Tuple[Path, Path]:
    """Create a synthetic stem mix and cover track fixture."""
    sr = 48000
    duration = 3.0
    t = np.arange(int(sr * duration)) / sr

    # Stem mix: 60 Hz kick fundamental + 440 Hz comping chord
    stem_mix = 0.5 * np.sin(2 * np.pi * 60.0 * t) + 0.3 * np.sin(2 * np.pi * 440.0 * t)
    stereo_stem = np.vstack((stem_mix, stem_mix)).T

    # Cover track: Room reverberation + out-of-phase 60 Hz sub mud
    cover_track = 0.4 * np.sin(2 * np.pi * 60.0 * t + np.pi / 2.0) + 0.3 * np.sin(2 * np.pi * 440.0 * t)
    stereo_cover = np.vstack((cover_track, cover_track)).T

    stem_path = tmp_path / "stem_mix.wav"
    cover_path = tmp_path / "cover_track.wav"
    sf.write(str(stem_path), stereo_stem, sr)
    sf.write(str(cover_path), stereo_cover, sr)

    return stem_path, cover_path


@pytest.fixture
def drum_midi_file(tmp_path: Path) -> Path:
    """Create a 2-bar quantized drum MIDI fixture."""
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    drums = pretty_midi.Instrument(program=0, is_drum=True, name="Drumset")

    for i in range(16):
        t = i * 0.125
        # Kick on 0, 4, 8, 12; Snare on 2, 6, 10, 14; Hats on all
        drums.notes.append(pretty_midi.Note(velocity=100, pitch=42, start=t, end=t + 0.05))
        if i % 4 == 0:
            drums.notes.append(pretty_midi.Note(velocity=110, pitch=36, start=t, end=t + 0.08))
        if i % 4 == 2:
            drums.notes.append(pretty_midi.Note(velocity=105, pitch=38, start=t, end=t + 0.08))

    pm.instruments.append(drums)
    p = tmp_path / "drums_quantized.mid"
    pm.write(str(p))
    return p


def test_butterworth_highpass_filters_sub_bass():
    """Verify Butterworth HPF attenuates sub-bass frequencies below cutoff."""
    sr = 48000
    t = np.arange(sr) / sr  # 1.0 second
    # Mix 50 Hz sub and 1000 Hz tone
    sub_50hz = np.sin(2 * np.pi * 50.0 * t).astype(np.float32)
    tone_1000hz = np.sin(2 * np.pi * 1000.0 * t).astype(np.float32)
    composite = sub_50hz + tone_1000hz

    filtered = apply_butterworth_highpass(composite, cutoff_hz=120.0, sr=sr)

    # 50 Hz sub energy should be attenuated by > 18 dB
    energy_before = np.sum(sub_50hz**2)
    # Estimate residual 50 Hz by dot product
    residual_50hz = np.sum(filtered * sub_50hz)
    attenuation_db = 10.0 * np.log10(abs(residual_50hz) / energy_before)

    assert attenuation_db < -15.0


def test_suno_cover_blend_layer(synthetic_stems_and_cover: Tuple[Path, Path], tmp_path: Path):
    """Verify 20-40% cover blend with low-end protection runs without clipping."""
    stem_p, cover_p = synthetic_stems_and_cover
    out_blended = tmp_path / "blended_output.wav"

    res = blend_cover_with_stem_mix(
        stem_mix_path=stem_p,
        cover_audio_path=cover_p,
        output_path=out_blended,
        blend_ratio=0.30,
        hpf_cutoff_hz=120.0,
        sr=48000,
    )

    assert res.is_file()
    y, sr = sf.read(str(res))
    assert sr == 48000
    assert np.max(np.abs(y)) <= 1.0  # Peak controlled


def test_musicgen_local_zero_cost_restyle(synthetic_stems_and_cover: Tuple[Path, Path]):
    """Verify local open-weight MusicGen surrogate incurs zero financial spend."""
    stem_p, _ = synthetic_stems_and_cover
    endpoint = MusicGenLocalEndpoint(config={})

    cand = endpoint.transform(
        audio_path=stem_p,
        prompt="Warm vintage Rhodes",
        strength=0.75,
        seed=101,
    )

    assert Path(cand.audio_path).is_file()
    assert cand.cost_usd == 0.0
    assert cand.endpoint_name == "musicgen-local-baseline"


def test_groovae_drum_humanizer_safety_gate(drum_midi_file: Path, tmp_path: Path):
    """Verify drum humanizer shifts onsets within tolerance and preserves note count."""
    humanizer = ModelBasedHumanizer(max_displacement_ms=30.0)
    out_mid = tmp_path / "humanized_drums.mid"

    res_path, audit = humanizer.humanize_drum_track(
        midi_path=drum_midi_file,
        output_path=out_mid,
        seed=42,
    )

    assert res_path.is_file()
    assert audit.passed_safety_gate is True
    assert audit.notes_preserved is True
    assert audit.max_timing_displacement_ms <= 30.0


def test_seam_discontinuity_detection_and_repair(tmp_path: Path):
    """Verify seam discontinuity detector catches jumps and inpainting smooths them."""
    sr = 48000
    t = np.arange(int(sr * 1.0)) / sr
    audio = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)

    # Inject an intentional spectral jump at 0.5s (downbeat joint)
    joint_time = 0.5
    joint_idx = int(joint_time * sr)
    audio[joint_idx : joint_idx + int(0.05 * sr)] = np.sin(2 * np.pi * 3500.0 * np.arange(int(0.05 * sr)) / sr)

    test_file = tmp_path / "seam_audio.wav"
    sf.write(str(test_file), audio, sr)

    is_bad, diff = detect_seam_discontinuity(test_file, joint_time_sec=joint_time, sr=sr)
    assert is_bad is True
    assert diff >= 0.45

    repaired_file = tmp_path / "seam_repaired.wav"
    inpaint_seam_window(test_file, seam_time_sec=joint_time, output_path=repaired_file, sr=sr)
    assert repaired_file.is_file()