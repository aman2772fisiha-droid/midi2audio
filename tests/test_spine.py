"""Property-based verification and deterministic tests for the M1 spine."""

from __future__ import annotations

import tempfile
from pathlib import Path
import numpy as np
import pretty_midi
import pytest
from m2a.analysis import analyze_midi
from m2a.groove.apply import apply_groove_to_midi
from m2a.groove.spec import GrooveSpec, SwingSpec, TrackMicrotiming
from m2a.orchestrator import PipelineOrchestrator


@pytest.fixture
def quantized_midi_file(tmp_path: Path) -> Path:
    """Generate an exact, quantized 4-bar 120 BPM MIDI fixture."""
    print("FUNC-QMF")
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)

    # Track 0: Drums (Kick on 1, Snare on 2 & 4, Hi-Hat on all 16ths)
    drums = pretty_midi.Instrument(program=0, is_drum=True, name="Standard Kit")
    sec_per_16th = 0.125  # 120 BPM -> 0.5s/beat -> 0.125s/16th

    for bar in range(4):
        bar_offset = bar * 2.0
        # Kick on 1 and 3
        drums.notes.append(pretty_midi.Note(100, 36, bar_offset + 0.0, bar_offset + 0.1))
        drums.notes.append(pretty_midi.Note(100, 36, bar_offset + 1.0, bar_offset + 1.1))
        # Snare on 2 and 4
        drums.notes.append(pretty_midi.Note(95, 38, bar_offset + 0.5, bar_offset + 0.6))
        drums.notes.append(pretty_midi.Note(95, 38, bar_offset + 1.5, bar_offset + 1.6))
        # Hats on all 16ths with flat velocity
        for step in range(16):
            t = bar_offset + (step * sec_per_16th)
            drums.notes.append(pretty_midi.Note(80, 42, t, t + 0.05))

    pm.instruments.append(drums)

    # Track 1: Bass
    bass = pretty_midi.Instrument(program=33, is_drum=False, name="Fingered Bass")
    for bar in range(4):
        bar_offset = bar * 2.0
        bass.notes.append(pretty_midi.Note(90, 36, bar_offset + 0.0, bar_offset + 0.4))
        bass.notes.append(pretty_midi.Note(90, 39, bar_offset + 0.5, bar_offset + 0.9))
        bass.notes.append(pretty_midi.Note(90, 41, bar_offset + 1.0, bar_offset + 1.4))
        bass.notes.append(pretty_midi.Note(90, 36, bar_offset + 1.5, bar_offset + 1.9))
    pm.instruments.append(bass)

    midi_path = tmp_path / "test_quantized.mid"
    pm.write(str(midi_path))
    return midi_path


def test_groove_spec_validation():
    """Verify strict Pydantic contract validation for symbolic GrooveSpec parameters.

    Task:
        Enforces runtime boundary and dimensional safety on `SwingSpec` and
        `TrackMicrotiming` models prior to symbolic MIDI transformation in Stage 1[cite: 1].
        Ensures illegal configurations cannot enter `apply_groove_to_midi()`[cite: 1],
        preventing downstream metrical binning crashes (`IndexError`) and acoustic
        smearing.

    Expected Schema Constraints & Values:
        1. Swing Ratio (`SwingSpec.ratio`):
           - Invariant: Strictly bounded in [0.50, 0.75][cite: 1].
           - 0.50 represents rigid, straight subdivision timing[cite: 1].
           - 0.66 represents exact 2:1 ternary triplet shuffle[cite: 1].
           - 0.75 represents hard dotted-eighth swing boundary[cite: 1].
           - Values < 0.50 (anticipatory rush) or > 0.75 (extreme distortion)
             must raise `pydantic.ValidationError` (subclass of `ValueError`)[cite: 1].

        2. Swing Subdivision (`SwingSpec.subdivision`):
           - Invariant: `Literal[8, 16]`[cite: 1].
           - Rejects arbitrary subdivisions (e.g., 12, 32)[cite: 1].

        3. 16th-Position Microtiming Array (`TrackMicrotiming.by_position_16`):
           - Invariant: Fixed-size 1D array of floats with length == 16[cite: 1].
           - Corresponds to the 16 sixteenth-note metrical slots of a 4/4 bar:
             Slot 0 (Beat 1), Slot 4 (Beat 2 downbeat), Slot 8 (Beat 3),
             Slot 12 (Beat 4)[cite: 1].
           - Values represent systematic offset displacements in milliseconds (-50.0 ms to +50.0 ms).
           - Passing arrays with length != 16 must raise `ValueError` via
             validator `@field_validator("by_position_16")`[cite: 1].

        4. Stochastic Humanization Jitter (`TrackMicrotiming.jitter_sd`):
           - Invariant: Gaussian standard deviation sigma bounded in [0.0, 4.0] ms[cite: 1].
           - Values > 4.0 ms violate the tight pocket constraint and raise `ValueError`[cite: 1].
    """
    # 1. Positive Verification: Legal configurations pass at exact boundaries
    spec_straight = SwingSpec(ratio=0.50, subdivision=8)
    assert spec_straight.ratio == 0.50

    spec_triplet = SwingSpec(ratio=0.75, subdivision=16)
    assert spec_triplet.ratio == 0.75

    valid_timing = TrackMicrotiming(
        by_position_16=[0.0] * 16,
        global_offset=5.0,
        jitter_sd=2.0,
    )
    assert len(valid_timing.by_position_16) == 16
    assert valid_timing.jitter_sd == 2.0

    # 2. Negative Verification: Swing ratio out-of-bounds rejection
    with pytest.raises(ValueError, match="less than or equal to 0.75"):
        SwingSpec(ratio=0.85)

    with pytest.raises(ValueError, match="greater than or equal to 0.5"):
        SwingSpec(ratio=0.40)

    # 3. Negative Verification: Subdivision literal constraint
    with pytest.raises(ValueError):
        SwingSpec(subdivision=12)

    # 4. Negative Verification: Array length violations (!= 16)
    with pytest.raises(ValueError, match="must have exactly 16 offsets, received 8"):
        TrackMicrotiming(by_position_16=[0.0] * 8)

    with pytest.raises(ValueError, match="must have exactly 16 offsets, received 20"):
        TrackMicrotiming(by_position_16=[0.0] * 20)

    # 5. Negative Verification: Unbounded jitter standard deviation (> 4.0 ms)
    with pytest.raises(ValueError, match="less than or equal to 4"):
        TrackMicrotiming(by_position_16=[0.0] * 16, jitter_sd=10.0)
    """Verify pydantic raises strict validation errors on malformed specs

    and accepts valid configurations at the boundaries.


    """
    # 1. POSITIVE CHECKS: Verify legal boundaries pass cleanly
    spec_straight = SwingSpec(ratio=0.50, subdivision=8)
    assert spec_straight.ratio == 0.50

    spec_triplet = SwingSpec(ratio=0.75, subdivision=16)
    assert spec_triplet.ratio == 0.75

    valid_timing = TrackMicrotiming(by_position_16=[0.0] * 16, global_offset=5.0, jitter_sd=2.0)
    assert len(valid_timing.by_position_16) == 16
    assert valid_timing.jitter_sd == 2.0

    # 2. NEGATIVE CHECKS: Upper & lower swing ratio bounds
    with pytest.raises(ValueError, match="less than or equal to 0.75"):
        SwingSpec(ratio=0.85)

    with pytest.raises(ValueError, match="greater than or equal to 0.5"):
        SwingSpec(ratio=0.40)

    # 3. NEGATIVE CHECKS: Subdivision enum constraints
    with pytest.raises(ValueError):
        SwingSpec(subdivision=12)  # Literal[8, 16] only

    # 4. NEGATIVE CHECKS: Array length constraints
    with pytest.raises(ValueError, match="must have exactly 16 offsets, received 8"):
        TrackMicrotiming(by_position_16=[0.0] * 8)

    with pytest.raises(ValueError, match="must have exactly 16 offsets, received 20"):
        TrackMicrotiming(by_position_16=[0.0] * 20)

    # 5. NEGATIVE CHECKS: Jitter standard deviation limit (max 4.0 ms)
    with pytest.raises(ValueError, match="less than or equal to 4"):
        TrackMicrotiming(by_position_16=[0.0] * 16, jitter_sd=10.0)


def test_deterministic_microtiming_offset_within_one_tick(quantized_midi_file: Path, tmp_path: Path):
    """Prove that applied microtiming matches the spec within 1 tick (strict non-negotiable)."""
    print("FUNC-TDMOWOT")
    analysis = analyze_midi(quantized_midi_file, {})

    spec = GrooveSpec(
        swing=SwingSpec(ratio=0.5),  # Straight, test only microtiming offsets
        microtiming_ms={
            "drums": TrackMicrotiming(
                by_position_16=[0.0, 5.0, 10.0, -5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                global_offset=0.0,
                jitter_sd=0.0,  # 0 jitter for deterministic verification
            )
        },
    )

    out_midi_path = tmp_path / "expressive.mid"
    apply_groove_to_midi(quantized_midi_file, spec, analysis, out_midi_path, seed=42)

    # Re-read and check offsets
    pm_orig = pretty_midi.PrettyMIDI(str(quantized_midi_file))
    pm_expr = pretty_midi.PrettyMIDI(str(out_midi_path))

    drums_orig = pm_orig.instruments[0]
    drums_expr = pm_expr.instruments[0]

    # Calculate tick duration at 120 BPM: 0.5s / 480 ppq ≈ 1.0416 ms
    tick_dur_sec = (60.0 / 120.0) / pm_orig.resolution

    for n_orig, n_expr in zip(drums_orig.notes, drums_expr.notes):
        if n_orig.pitch == 42:  # Hi-hat check
            pos_16 = int(round((n_orig.start % 2.0) / 0.125)) % 16
            expected_offset_sec = spec.microtiming_ms["drums"].by_position_16[pos_16] / 1000.0
            actual_offset_sec = n_expr.start - n_orig.start
            assert abs(actual_offset_sec - expected_offset_sec) <= (tick_dur_sec + 1e-4)


def test_end_to_end_spine_pipeline(quantized_midi_file: Path, tmp_path: Path):
    """Verify execution of Orchestrator across Stages 0, 1, and 2."""
    print("FUNC-TETSP")
    config = {
        "pipeline": {
            "sample_rate": 48000,
            "headroom_dbfs": -12.0,
            "artifacts_dir": str(tmp_path / "artifacts"),
            "soundfont_path": "non_existent.sf2",  # Triggers pure DSP fallback
        },
        "stage0_analysis": {},
        "stage1_groove": {
            "swing": {"ratio": 0.60, "subdivision": 8},
        },
    }

    orchestrator = PipelineOrchestrator(config)
    results = orchestrator.run_deterministic_spine(quantized_midi_file)

    assert Path(results["groove_midi_path"]).is_file()
    assert len(results["stems"]) == 2
    for stem in results["stems"]:
        assert Path(stem["audio_path"]).is_file()
        assert stem["sample_rate"] == 48000