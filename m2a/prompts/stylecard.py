"""Stage 4: Distill reference corpora into structured House Style Cards for prompt synthesis."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import librosa
import numpy as np
from pydantic import BaseModel, Field


class AcousticProfile(BaseModel):
    """Low-level acoustic and dynamic descriptors extracted from reference audio."""

    spectral_centroid_hz: float
    spectral_rolloff_hz: float
    rms_energy_mean: float
    dynamic_range_db: float
    stereo_width_ratio: float


class HouseStyleCard(BaseModel):
    """Distilled production aesthetic descriptor governing per-stem generative prompts."""

    target_genre_era: str
    key_descriptors: List[str]
    production_space: str
    instrumentation_cues: Dict[str, str]
    negative_prompts: List[str]
    acoustic_summary: AcousticProfile


def analyze_reference_acoustics(audio_path: Path | str, sr: int = 48000) -> AcousticProfile:
    """Extract physical acoustic measurements from reference recording."""
    y, sample_rate = librosa.load(str(audio_path), sr=sr, mono=False)
    if y.ndim == 1:
        stereo_y = np.vstack((y, y))
        mono_y = y
    else:
        stereo_y = y
        mono_y = librosa.to_mono(y)

    centroid = float(np.mean(librosa.feature.spectral_centroid(y=mono_y, sr=sample_rate)))
    rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=mono_y, sr=sample_rate, roll_percent=0.85)))
    rms = librosa.feature.rms(y=mono_y)[0]
    rms_mean = float(np.mean(rms))

    # Approximate dynamic range: 95th percentile vs 5th percentile RMS in dB
    p95 = np.percentile(rms, 95) + 1e-7
    p05 = np.percentile(rms, 5) + 1e-7
    dyn_range = float(20.0 * np.log10(p95 / p05))

    # Stereo width estimation: side energy / mid energy
    mid = 0.5 * (stereo_y[0] + stereo_y[1])
    side = 0.5 * (stereo_y[0] - stereo_y[1])
    mid_energy = float(np.sum(mid**2)) + 1e-7
    side_energy = float(np.sum(side**2))
    stereo_width = float(np.clip(side_energy / mid_energy, 0.0, 1.0))

    return AcousticProfile(
        spectral_centroid_hz=round(centroid, 1),
        spectral_rolloff_hz=round(rolloff, 1),
        rms_energy_mean=round(rms_mean, 4),
        dynamic_range_db=round(dyn_range, 2),
        stereo_width_ratio=round(stereo_width, 3),
    )


def distill_house_style_card(
    reference_paths: List[Path | str],
    genre_hint: str = "1970s Vintage Analog Funk",
) -> HouseStyleCard:
    """Distill one or more reference recordings into a consolidated House Style Card."""
    if not reference_paths:
        raise ValueError("At least one reference audio track required to build style card.")

    profiles = [analyze_reference_acoustics(p) for p in reference_paths]

    # Compute averaged acoustic profile
    avg_centroid = float(np.mean([p.spectral_centroid_hz for p in profiles]))
    avg_rolloff = float(np.mean([p.spectral_rolloff_hz for p in profiles]))
    avg_rms = float(np.mean([p.rms_energy_mean for p in profiles]))
    avg_dyn = float(np.mean([p.dynamic_range_db for p in profiles]))
    avg_width = float(np.mean([p.stereo_width_ratio for p in profiles]))

    acoustic_summary = AcousticProfile(
        spectral_centroid_hz=round(avg_centroid, 1),
        spectral_rolloff_hz=round(avg_rolloff, 1),
        rms_energy_mean=round(avg_rms, 4),
        dynamic_range_db=round(avg_dyn, 2),
        stereo_width_ratio=round(avg_width, 3),
    )

    # Classify aesthetic based on brightness (spectral rolloff) and dynamic range
    descriptors = ["warm analog saturation", "vintage console punch"]
    if avg_rolloff < 3000.0:
        descriptors.extend(["dark tape warmth", "smooth rolled-off highs"])
    else:
        descriptors.extend(["crisp transient definition", "articulate top end"])

    if avg_dyn > 18.0:
        descriptors.append("uncompressed organic dynamic breathing")
    else:
        descriptors.append("tightly glued optical bus compression")

    space_desc = "dry close-mic isolation with treated vintage studio acoustic chamber"
    if avg_width > 0.35:
        space_desc = "wide stereo studio room with natural ambient room mic bleed"

    cues = {
        "drums": "tight dampened kick, crisp wood snare with ghost notes, focused dry hats",
        "bass": "vintage flatwound P-bass through warm tube DI, round punchy low-mids",
        "comp": "analog Rhodes electric piano with warm vibrato, clean rhythmic funk guitar chops",
        "lead": "singing monophonic analog synth lead with smooth filter resonance",
    }

    negative = [
        "modern EDM hyper-compressed",
        "thin synthetic drums",
        "harsh digital clipping",
        "muddy low-end buildup",
        "excessive cavernous digital reverb",
    ]

    return HouseStyleCard(
        target_genre_era=genre_hint,
        key_descriptors=descriptors,
        production_space=space_desc,
        instrumentation_cues=cues,
        negative_prompts=negative,
        acoustic_summary=acoustic_summary,
    )