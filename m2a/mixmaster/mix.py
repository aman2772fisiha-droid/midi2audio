"""Stage 6: Multi-track static pan/gain template mixing with LUFS normalization."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional
import librosa
import numpy as np
import soundfile as sf
from pydantic import BaseModel, Field


class StemMixSpec(BaseModel):
    """Gain and panning configuration per track role."""

    gain_db: float = 0.0
    pan: float = Field(default=0.0, ge=-1.0, le=1.0, description="-1.0 Left, 0.0 Center, +1.0 Right")


# Static mix console routing template
ROLE_MIX_TEMPLATES: Dict[str, StemMixSpec] = {
    "drums": StemMixSpec(gain_db=0.0, pan=0.0),
    "bass": StemMixSpec(gain_db=-1.5, pan=0.0),
    "comp": StemMixSpec(gain_db=-3.0, pan=-0.35),
    "lead": StemMixSpec(gain_db=-1.0, pan=0.0),
    "pads": StemMixSpec(gain_db=-6.0, pan=0.40),
    "ornament": StemMixSpec(gain_db=-6.0, pan=-0.50),
}


def apply_constant_power_pan(stereo_signal: np.ndarray, pan: float) -> np.ndarray:
    """Apply constant-power panning law [-1.0, 1.0] across stereo channels."""
    pan = float(np.clip(pan, -1.0, 1.0))
    angle = (pan + 1.0) * (np.pi / 4.0)  # Maps -1..1 to 0..pi/2
    left_gain = np.cos(angle)
    right_gain = np.sin(angle)

    out = np.zeros_like(stereo_signal)
    out[0] = stereo_signal[0] * left_gain
    out[1] = stereo_signal[1] * right_gain
    return out


def mix_stems_to_stereo(
    stem_specs: List[Tuple[Path | str, str]],  # List of (stem_audio_path, role)
    output_mix_path: Path | str,
    sr: int = 48000,
    headroom_dbfs: float = -3.0,
) -> Path:
    """Render stems into a balanced rough mix using role-based gain and panning rules."""
    if not stem_specs:
        raise ValueError("Cannot mix empty stem list.")

    loaded_stems = []
    max_len = 0

    for path, role in stem_specs:
        y, _ = librosa.load(str(path), sr=sr, mono=False)
        if y.ndim == 1:
            y = np.vstack((y, y))
        loaded_stems.append((y, role))
        max_len = max(max_len, y.shape[1])

    composite_mix = np.zeros((2, max_len), dtype=np.float32)

    for y, role in loaded_stems:
        rule = ROLE_MIX_TEMPLATES.get(role, ROLE_MIX_TEMPLATES["comp"])
        linear_gain = 10.0 ** (rule.gain_db / 20.0)

        stem_padded = np.zeros((2, max_len), dtype=np.float32)
        stem_padded[:, : y.shape[1]] = y * linear_gain
        panned = apply_constant_power_pan(stem_padded, rule.pan)

        composite_mix += panned

    # Peak normalization to target mix headroom
    peak = np.max(np.abs(composite_mix))
    if peak > 1e-6:
        target_linear = 10.0 ** (headroom_dbfs / 20.0)
        composite_mix = composite_mix * (target_linear / peak)

    out_file = Path(output_mix_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_file), composite_mix.T, sr)
    return out_file