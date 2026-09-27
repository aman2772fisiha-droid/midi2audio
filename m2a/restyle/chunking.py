"""Stage 3: Section-aware chunking, bar-boundary overlap, and equal-power crossfading."""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Tuple
import librosa
import numpy as np
import soundfile as sf
from pydantic import BaseModel


class AudioChunk(BaseModel):
    """Metadata representing a bar-aligned audio segment with overlap margins."""

    chunk_idx: int
    start_bar: int
    end_bar: int
    start_time_sec: float
    end_time_sec: float
    overlap_start_sec: float
    overlap_end_sec: float
    audio_path: str


def compute_equal_power_crossfade(n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate equal-power crossfade gain curves (fade-out, fade-in)."""
    t = np.linspace(0.0, np.pi / 2.0, n_samples)
    fade_out = np.cos(t).astype(np.float32)
    fade_in = np.sin(t).astype(np.float32)
    return fade_out, fade_in


def split_audio_into_chunks(
    audio_path: Path | str,
    bar_timestamps: List[float],
    bars_per_chunk: int = 16,
    overlap_bars: int = 2,
    output_dir: Path | str = "artifacts/chunks",
) -> List[AudioChunk]:
    """Split an audio file into overlapping chunks aligned strictly to bar boundaries."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    y, sr = librosa.load(str(audio_path), sr=None, mono=False)
    if y.ndim == 1:
        y = np.vstack((y, y))

    total_bars = len(bar_timestamps) - 1
    chunks: List[AudioChunk] = []
    chunk_idx = 0

    curr_start_bar = 0
    while curr_start_bar < total_bars:
        curr_end_bar = min(curr_start_bar + bars_per_chunk, total_bars)

        # Apply overlap margins
        ov_start_bar = max(0, curr_start_bar - overlap_bars) if curr_start_bar > 0 else 0
        ov_end_bar = min(total_bars, curr_end_bar + overlap_bars)

        t_start = bar_timestamps[curr_start_bar]
        t_end = bar_timestamps[curr_end_bar]
        t_ov_start = bar_timestamps[ov_start_bar]
        t_ov_end = bar_timestamps[ov_end_bar]

        s_start = int(round(t_ov_start * sr))
        s_end = int(round(t_ov_end * sr))
        chunk_y = y[:, s_start:s_end]

        chunk_path = out_dir / f"chunk_{chunk_idx:03d}.wav"
        sf.write(str(chunk_path), chunk_y.T, sr)

        chunks.append(
            AudioChunk(
                chunk_idx=chunk_idx,
                start_bar=curr_start_bar,
                end_bar=curr_end_bar,
                start_time_sec=t_start,
                end_time_sec=t_end,
                overlap_start_sec=t_ov_start,
                overlap_end_sec=t_ov_end,
                audio_path=str(chunk_path),
            )
        )

        if curr_end_bar >= total_bars:
            break
        curr_start_bar += bars_per_chunk - overlap_bars
        chunk_idx += 1

    return chunks


def rejoin_chunks_equal_power(
    chunk_paths: List[Path | str],
    crossfade_duration_sec: float,
    output_path: Path | str,
    sr: int = 48000,
) -> Path:
    """Recombine chunks using equal-power crossfades across adjacent overlapping boundaries."""
    if not chunk_paths:
        raise ValueError("Cannot stitch empty chunk list.")

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    loaded_chunks = []
    for cp in chunk_paths:
        audio, file_sr = librosa.load(str(cp), sr=sr, mono=False)
        if audio.ndim == 1:
            audio = np.vstack((audio, audio))
        loaded_chunks.append(audio)

    combined = loaded_chunks[0]
    cf_samples = int(round(crossfade_duration_sec * sr))

    fade_out, fade_in = compute_equal_power_crossfade(cf_samples)
    fade_out_2d = fade_out[None, :]
    fade_in_2d = fade_in[None, :]

    for i in range(1, len(loaded_chunks)):
        next_chunk = loaded_chunks[i]
        if combined.shape[1] < cf_samples or next_chunk.shape[1] < cf_samples:
            # Fallback direct concatenation if chunk shorter than crossfade window
            combined = np.hstack((combined, next_chunk))
            continue

        head = combined[:, :-cf_samples]
        overlap_a = combined[:, -cf_samples:] * fade_out_2d
        overlap_b = next_chunk[:, :cf_samples] * fade_in_2d
        crossfaded = overlap_a + overlap_b
        tail = next_chunk[:, cf_samples:]

        combined = np.hstack((head, crossfaded, tail))

    sf.write(str(out_path), combined.T, sr)
    return out_path

def detect_seam_discontinuity(
    audio_path: Path | str,
    joint_time_sec: float,
    window_ms: float = 80.0,
    sr: int = 48000,
    spectral_diff_threshold: float = 0.45,
) -> Tuple[bool, float]:
    """Detect audible spectral discontinuities or phase jumps across chunk joints."""
    y, sample_rate = librosa.load(str(audio_path), sr=sr, mono=True)
    joint_sample = int(round(joint_time_sec * sample_rate))
    win_samples = int(round((window_ms / 1000.0) * sample_rate))

    left_start = max(0, joint_sample - win_samples)
    right_end = min(len(y), joint_sample + win_samples)

    if joint_sample - left_start < win_samples // 2 or right_end - joint_sample < win_samples // 2:
        return False, 0.0

    seg_left = y[left_start:joint_sample]
    seg_right = y[joint_sample:right_end]

    # Compute short-time spectral centroid and energy across joint
    centroid_left = float(np.mean(librosa.feature.spectral_centroid(y=seg_left, sr=sample_rate)))
    centroid_right = float(np.mean(librosa.feature.spectral_centroid(y=seg_right, sr=sample_rate)))

    mean_c = 0.5 * (centroid_left + centroid_right) + 1e-6
    relative_spectral_jump = float(abs(centroid_left - centroid_right) / mean_c)

    is_discontinuous = relative_spectral_jump >= spectral_diff_threshold
    return is_discontinuous, round(relative_spectral_jump, 3)


def inpaint_seam_window(
    audio_path: Path | str,
    seam_time_sec: float,
    output_path: Path | str,
    repair_window_ms: float = 60.0,
    sr: int = 48000,
) -> Path:
    """Repair an audible joint discontinuity using smooth cosine window smoothing."""
    y, sample_rate = librosa.load(str(audio_path), sr=sr, mono=False)
    if y.ndim == 1:
        y = np.vstack((y, y))

    seam_sample = int(round(seam_time_sec * sample_rate))
    half_win = int(round(((repair_window_ms / 2.0) / 1000.0) * sample_rate))

    start = max(0, seam_sample - half_win)
    end = min(y.shape[1], seam_sample + half_win)
    win_len = end - start

    if win_len > 4:
        # Smooth transient discontinuity with a raised-cosine taper
        taper = np.hanning(win_len).astype(np.float32)
        for ch in range(y.shape[0]):
            y[ch, start:end] = y[ch, start:end] * taper + (1.0 - taper) * np.mean(y[ch, start:end])

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_p), y.T, sample_rate)
    return out_p