"""Stage 1: Extract empirical groove templates and microtiming profiles from reference audio."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple
import librosa
import numpy as np
from pydantic import BaseModel, Field


class ExtractedGrooveTemplate(BaseModel):
    """Empirical microtiming profile and accent map extracted from reference audio."""

    tempo_bpm: float
    by_position_16: List[float] = Field(
        description="Median deviation in ms per 16th-note metrical position relative to regular grid."
    )
    spread_ms_16: List[float] = Field(
        description="Interquartile spread/standard deviation in ms per 16th-note slot."
    )
    accent_map_16: List[float] = Field(
        description="Normalized RMS energy profile per 16th-note position."
    )
    swing_ratio_estimated: float = Field(
        ge=0.45, le=0.75, description="Empirical 8th-note swing ratio measured from audio."
    )


def extract_groove_template_from_audio(
    audio_path: Path | str,
    sr: int = 48000,
    hop_length: int = 512,
) -> ExtractedGrooveTemplate:
    """Analyze audio to recover metrical grid, onset deviations, and per-slot dynamic accents."""
    y, sample_rate = librosa.load(str(audio_path), sr=sr, mono=True)
    if len(y) == 0:
        raise ValueError("Cannot extract groove from empty audio file.")

    # 1. Estimate tempo and beat frames
    tempo_est, beat_frames = librosa.beat.beat_track(y=y, sr=sample_rate, hop_length=hop_length)
    bpm = float(np.atleast_1d(tempo_est)[0])
    beat_times = librosa.frames_to_time(beat_frames, sr=sample_rate, hop_length=hop_length)

    if len(beat_times) < 2:
        bpm = 120.0
        sec_per_beat = 0.5
    else:
        sec_per_beat = float(np.median(np.diff(beat_times)))
        bpm = float(60.0 / sec_per_beat)

    sec_per_16th = sec_per_beat / 4.0
    sec_per_bar = sec_per_beat * 4.0

    # 2. Extract onsets and envelope strengths
    onset_env = librosa.onset.onset_strength(y=y, sr=sample_rate, hop_length=hop_length)
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sample_rate,
        hop_length=hop_length,
        backtrack=True,
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sample_rate, hop_length=hop_length)

    # 3. Bin onsets into 16 metrical positions
    slot_deviations: List[List[float]] = [[] for _ in range(16)]
    slot_energies: List[List[float]] = [[] for _ in range(16)]

    for t_onset in onset_times:
        time_in_bar = t_onset % sec_per_bar
        raw_pos_16 = time_in_bar / sec_per_16th
        pos_16 = int(round(raw_pos_16)) % 16

        nominal_grid_time = (int(t_onset // sec_per_bar) * sec_per_bar) + (pos_16 * sec_per_16th)
        dev_ms = (t_onset - nominal_grid_time) * 1000.0

        # Discard spurious outlier detections beyond half a 16th interval
        if abs(dev_ms) < (sec_per_16th * 500.0):
            frame_idx = min(len(onset_env) - 1, int(round(librosa.time_to_frames(t_onset, sr=sample_rate, hop_length=hop_length))))
            energy = float(onset_env[frame_idx])
            slot_deviations[pos_16].append(dev_ms)
            slot_energies[pos_16].append(energy)

    # 4. Compute median offsets, spread, and accent profile
    by_pos_16 = []
    spread_16 = []
    accent_16 = []

    for pos in range(16):
        devs = slot_deviations[pos]
        energies = slot_energies[pos]

        if devs:
            by_pos_16.append(float(np.median(devs)))
            spread_16.append(float(np.std(devs) if len(devs) > 1 else 0.0))
        else:
            by_pos_16.append(0.0)
            spread_16.append(0.0)

        accent_16.append(float(np.mean(energies)) if energies else 0.5)

    # Normalize accent profile
    max_accent = max(accent_16) if max(accent_16) > 0 else 1.0
    accent_map = [round(float(a / max_accent), 3) for a in accent_16]

    # 5. Estimate 8th-note swing ratio from odd 8th-note offsets (positions 2, 6, 10, 14)
    odd_8th_offsets_ms = [by_pos_16[2], by_pos_16[6], by_pos_16[10], by_pos_16[14]]
    mean_odd_delay_sec = float(np.mean(odd_8th_offsets_ms)) / 1000.0
    eighth_sec = sec_per_beat / 2.0
    # ratio = 0.5 + (delay / (2 * eighth_sec))
    estimated_swing = float(np.clip(0.5 + (mean_odd_delay_sec / (2.0 * max(1e-4, eighth_sec))), 0.45, 0.75))

    return ExtractedGrooveTemplate(
        tempo_bpm=round(bpm, 2),
        by_position_16=[round(v, 2) for v in by_pos_16],
        spread_ms_16=[round(v, 2) for v in spread_16],
        accent_map_16=accent_map,
        swing_ratio_estimated=round(estimated_swing, 3),
    )