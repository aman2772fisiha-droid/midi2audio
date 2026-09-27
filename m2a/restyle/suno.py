"""Stage 3 & 6: Suno upload-and-cover wrapper with low-end-protected mix blending."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import httpx
import librosa
import numpy as np
import scipy.signal
import soundfile as sf
from pydantic import BaseModel, Field
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from m2a.mixmaster.align import compute_cross_correlation_delay
from m2a.restyle.base import CandidateResult, EndpointErrorType, EndpointException, RestyleEndpoint


class SunoCoverConfig(BaseModel):
    """Parameters governing full-mix cover pass and bus blend."""

    audio_weight: float = Field(default=0.65, ge=0.0, le=1.0)
    style_weight: float = Field(default=0.35, ge=0.0, le=1.0)
    blend_ratio: float = Field(default=0.30, ge=0.10, le=0.50, description="20-40% loudness underlay")
    hpf_cutoff_hz: float = Field(default=120.0, ge=60.0, le=250.0, description="Protects kick/sub phase")


def apply_butterworth_highpass(
    audio: np.ndarray,
    cutoff_hz: float = 120.0,
    sr: int = 48000,
    order: int = 4,
) -> np.ndarray:
    """Filter out low frequencies from the cover layer to eliminate sub-bass phase mud."""
    nyquist = 0.5 * sr
    norm_cutoff = float(np.clip(cutoff_hz / nyquist, 1e-4, 0.99))
    sos = scipy.signal.butter(order, norm_cutoff, btype="highpass", output="sos")

    if audio.ndim == 1:
        filtered = scipy.signal.sosfilt(sos, audio)
    else:
        filtered = np.zeros_like(audio)
        for ch in range(audio.shape[0]):
            filtered[ch] = scipy.signal.sosfilt(sos, audio[ch])

    return filtered.astype(np.float32)


def blend_cover_with_stem_mix(
    stem_mix_path: Path | str,
    cover_audio_path: Path | str,
    output_path: Path | str,
    blend_ratio: float = 0.30,
    hpf_cutoff_hz: float = 120.0,
    sr: int = 48000,
) -> Path:
    """Phase-align and blend a high-passed cover track under the ground-truth stem mix."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    y_stem, _ = librosa.load(str(stem_mix_path), sr=sr, mono=False)
    y_cover, _ = librosa.load(str(cover_audio_path), sr=sr, mono=False)

    if y_stem.ndim == 1:
        y_stem = np.vstack((y_stem, y_stem))
    if y_cover.ndim == 1:
        y_cover = np.vstack((y_cover, y_cover))

    # Align cover track to dry stem mix using normalized cross-correlation
    lag_samples, _, _ = compute_cross_correlation_delay(y_stem, y_cover, sr=sr)
    if lag_samples > 0:
        aligned_cover = y_cover[:, lag_samples:]
    elif lag_samples < 0:
        pad = np.zeros((y_cover.shape[0], abs(lag_samples)), dtype=y_cover.dtype)
        aligned_cover = np.hstack((pad, y_cover))
    else:
        aligned_cover = y_cover

    # Match lengths
    target_len = y_stem.shape[1]
    if aligned_cover.shape[1] < target_len:
        pad_len = target_len - aligned_cover.shape[1]
        aligned_cover = np.pad(aligned_cover, ((0, 0), (0, pad_len)))
    else:
        aligned_cover = aligned_cover[:, :target_len]

    # High-pass filter cover layer to protect kick and bass fundamental phase
    clean_cover = apply_butterworth_highpass(aligned_cover, cutoff_hz=hpf_cutoff_hz, sr=sr)

    # Blend: (1 - blend_ratio) * stem_mix + blend_ratio * cover_layer
    alpha = float(np.clip(blend_ratio, 0.0, 0.50))
    blended = ((1.0 - alpha) * y_stem) + (alpha * clean_cover)

    # Normalize peak to match stem mix maximum amplitude
    ref_peak = np.max(np.abs(y_stem))
    blend_peak = np.max(np.abs(blended))
    if blend_peak > 1e-6 and ref_peak > 1e-6:
        blended = blended * (ref_peak / blend_peak)

    sf.write(str(out_file), blended.T, sr)
    return out_file


class SunoCoverEndpoint(RestyleEndpoint):
    """Client for Suno's upload-and-cover endpoint with retry policies."""

    def __init__(self, config: Dict[str, Any], api_key: Optional[str] = None):
        super().__init__(config)
        self.api_key = api_key or os.environ.get("SUNO_API_KEY", "mock_key")
        self.endpoint_url = config.get("restyle", {}).get(
            "suno_cover_url", "https://api.suno.ai/v1/generate/cover"
        )
        self.cost_per_generation = config.get("budget", {}).get("suno_cost_usd", 0.10)
        self.mock_mode = self.api_key == "mock_key"

    @retry(
        retry=retry_if_exception(lambda exc: isinstance(exc, EndpointException) and exc.error_type == EndpointErrorType.RETRYABLE),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1.0, min=2.0, max=10.0),
        reraise=True,
    )
    def transform(
        self,
        audio_path: Path | str,
        prompt: str,
        negative_prompt: str = "",
        strength: float = 0.65,
        seed: int = 42,
        extra: Optional[Dict[str, Any]] = None,
    ) -> CandidateResult:
        """Call Suno cover API or synthesize mock room-glued candidate in mock mode."""
        in_path = Path(audio_path)
        if not in_path.is_file():
            raise FileNotFoundError(f"Audio file not found: {in_path}")

        duration = float(librosa.get_duration(path=str(in_path)))
        out_dir = in_path.parent / "suno_covers"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_audio_path = out_dir / f"suno_cov_s{strength:.2f}_seed{seed}_{in_path.name}"

        if self.mock_mode:
            y, sr = librosa.load(str(in_path), sr=None, mono=False)
            if y.ndim == 1:
                y = np.vstack((y, y))

            # Simulate acoustic room glue and mild inter-instrument coloration
            reverb_ir = np.exp(-np.linspace(0, 4, int(sr * 0.05)))  # Short 50ms room tail
            wet_left = scipy.signal.convolve(y[0], reverb_ir, mode="same")
            wet_right = scipy.signal.convolve(y[1], reverb_ir, mode="same")
            mock_covered = 0.85 * y + 0.15 * np.vstack((wet_left, wet_right))

            sf.write(str(out_audio_path), mock_covered.T, sr)

            return CandidateResult(
                audio_path=str(out_audio_path),
                endpoint_name="suno-cover-mock",
                prompt=prompt,
                negative_prompt=negative_prompt,
                strength=strength,
                seed=seed,
                duration_sec=duration,
                cost_usd=self.cost_per_generation,
                raw_response_metadata={"mock": True, "mode": "cover"},
            )

        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "prompt": prompt,
            "negative_tags": negative_prompt,
            "audio_weight": str(strength),
            "seed": str(seed),
        }

        try:
            with open(in_path, "rb") as f:
                files = {"audio_file": (in_path.name, f, "audio/wav")}
                with httpx.Client(timeout=180.0) as client:
                    resp = client.post(self.endpoint_url, headers=headers, data=payload, files=files)

            if resp.status_code == 200:
                with open(out_audio_path, "wb") as f_out:
                    f_out.write(resp.content)
                return CandidateResult(
                    audio_path=str(out_audio_path),
                    endpoint_name="suno-cover",
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    strength=strength,
                    seed=seed,
                    duration_sec=duration,
                    cost_usd=self.cost_per_generation,
                    raw_response_metadata={"status_code": resp.status_code},
                )
            elif resp.status_code == 429 or resp.status_code >= 500:
                raise EndpointException(
                    f"Suno transient error: {resp.status_code} - {resp.text}",
                    error_type=EndpointErrorType.RETRYABLE,
                    status_code=resp.status_code,
                )
            else:
                raise EndpointException(
                    f"Suno fatal error: {resp.status_code} - {resp.text}",
                    error_type=EndpointErrorType.FATAL,
                    status_code=resp.status_code,
                )
        except httpx.RequestError as e:
            raise EndpointException(f"Network error communicating with Suno: {e}", error_type=EndpointErrorType.RETRYABLE)