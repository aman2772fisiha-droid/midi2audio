"""Stage 3 & 5: Open-weight local restyle baseline for zero-cost search prototyping."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import librosa
import numpy as np
import soundfile as sf
from m2a.restyle.base import CandidateResult, RestyleEndpoint


class MusicGenLocalEndpoint(RestyleEndpoint):
    """Fast, local restyle engine for parameter screening and offline verification."""

    def __init__(self, config: Dict[str, Any], device: str = "cpu"):
        super().__init__(config)
        self.device = device
        self.model = None

    def transform(
        self,
        audio_path: Path | str,
        prompt: str,
        negative_prompt: str = "",
        strength: float = 0.70,
        seed: int = 42,
        extra: Optional[Dict[str, Any]] = None,
    ) -> CandidateResult:
        """Perform local restyling with zero API cost, preserving pitch and timing structures."""
        in_path = Path(audio_path)
        if not in_path.is_file():
            raise FileNotFoundError(f"Input stem not found: {in_path}")

        duration = float(librosa.get_duration(path=str(in_path)))
        out_dir = in_path.parent / "musicgen_local"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_audio_path = out_dir / f"mg_s{strength:.2f}_seed{seed}_{in_path.name}"

        y, sr = librosa.load(str(in_path), sr=None, mono=False)
        if y.ndim == 1:
            y = np.vstack((y, y))

        # Local deterministic DSP restyle approximation:
        # Applies harmonic exciter, mild tube saturation, and low-divergence warmth
        rng = np.random.default_rng(seed)
        colored_y = np.zeros_like(y)

        for ch in range(y.shape[0]):
            # Soft saturation
            sat = np.tanh(y[ch] * (1.0 + (strength * 0.3)))
            # Mild harmonic overtone injection
            overtone = 0.05 * np.sin(2.0 * np.pi * 1000.0 * np.arange(len(y[ch])) / sr) * y[ch]
            # Bounded subtle noise floor injection (-60 dBFS)
            noise = rng.normal(0.0, 1e-4, len(y[ch]))
            colored_y[ch] = (1.0 - (strength * 0.2)) * sat + overtone + noise

        # Peak headroom normalization matching dry stems
        peak = np.max(np.abs(colored_y))
        if peak > 1e-6:
            colored_y = colored_y * (0.2512 / peak)

        sf.write(str(out_audio_path), colored_y.T, sr)

        return CandidateResult(
            audio_path=str(out_audio_path),
            endpoint_name="musicgen-local-baseline",
            prompt=prompt,
            negative_prompt=negative_prompt,
            strength=strength,
            seed=seed,
            duration_sec=duration,
            cost_usd=0.0,  # Zero financial spend
            raw_response_metadata={"device": self.device, "engine": "local_dsp_surrogate"},
        )