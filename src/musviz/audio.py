from __future__ import annotations

import math
import subprocess
from pathlib import Path

import numpy as np

EPSILON = 1e-8


def decode_audio_mono_f32(audio_path: Path, sample_rate: int) -> np.ndarray:
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(audio_path),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "f32le",
        "-",
    ]
    result = subprocess.run(cmd, capture_output=True, check=False)
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace")
        raise SystemExit(f"ffmpeg failed to decode audio:\n{stderr}")

    audio = np.frombuffer(result.stdout, dtype=np.float32).copy()
    if audio.size == 0:
        raise SystemExit("No audio samples decoded from input file")
    return np.clip(audio, -1.0, 1.0)


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(values, kernel, mode="same")


def robust_normalize(values: np.ndarray) -> np.ndarray:
    lo = np.percentile(values, 5)
    hi = np.percentile(values, 95)
    scale = hi - lo
    if scale < EPSILON:
        return np.zeros_like(values)
    return np.clip((values - lo) / scale, 0.0, 1.0).astype(np.float32)


def lift_levels(values: np.ndarray, floor: float, gamma: float) -> np.ndarray:
    lifted = floor + (1.0 - floor) * np.power(np.clip(values, 0.0, 1.0), gamma)
    return np.clip(lifted, 0.0, 1.0).astype(np.float32)


def extract_features(
    audio: np.ndarray,
    sample_rate: int,
    fps: int,
    mode: str,
) -> dict[str, np.ndarray]:
    frame_samples = max(1, int(sample_rate / fps))
    n_frames = int(math.ceil(audio.size / frame_samples))

    padded = np.pad(audio, (0, n_frames * frame_samples - audio.size))
    framed = padded.reshape(n_frames, frame_samples)

    rms = np.sqrt(np.mean(framed * framed, axis=1) + EPSILON)
    energy = robust_normalize(moving_average(rms, max(2, fps // 12)))

    onset = np.maximum(0.0, np.diff(np.r_[energy[:1], energy]))
    beat = robust_normalize(moving_average(onset, max(2, fps // 8)))

    n_fft = 1024
    window = np.hanning(n_fft).astype(np.float32)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)

    low_mask = (freqs >= 20) & (freqs < 180)
    mid_mask = (freqs >= 180) & (freqs < 2200)
    high_mask = (freqs >= 2200) & (freqs < 10000)

    if not np.any(low_mask):
        low_mask[:6] = True
    if not np.any(mid_mask):
        mid_mask[6:40] = True
    if not np.any(high_mask):
        high_mask[40:] = True

    low = np.zeros(n_frames, dtype=np.float32)
    mid = np.zeros(n_frames, dtype=np.float32)
    high = np.zeros(n_frames, dtype=np.float32)

    for i in range(n_frames):
        start = i * frame_samples
        chunk = audio[start : start + n_fft]
        if chunk.size < n_fft:
            chunk = np.pad(chunk, (0, n_fft - chunk.size))
        spectrum = np.abs(np.fft.rfft(chunk * window)).astype(np.float32)
        low[i] = float(np.mean(spectrum[low_mask]))
        mid[i] = float(np.mean(spectrum[mid_mask]))
        high[i] = float(np.mean(spectrum[high_mask]))

    low = robust_normalize(moving_average(low, max(2, fps // 16)))
    mid = robust_normalize(moving_average(mid, max(2, fps // 18)))
    high = robust_normalize(moving_average(high, max(2, fps // 20)))

    low_onset = np.maximum(0.0, np.diff(np.r_[low[:1], low]))
    mid_onset = np.maximum(0.0, np.diff(np.r_[mid[:1], mid]))

    kick = robust_normalize(moving_average(low_onset, max(2, fps // 10)))
    snare = robust_normalize(moving_average(mid_onset, max(2, fps // 11)))
    transient = robust_normalize(0.62 * kick + 0.38 * snare)

    if mode == "hiphop":
        beat = robust_normalize(0.20 * beat + 0.50 * kick + 0.30 * snare)
        bass = robust_normalize(0.88 * low + 0.68 * kick)
        beat = lift_levels(beat, floor=0.12, gamma=0.74)
        bass = lift_levels(bass, floor=0.14, gamma=0.72)
        transient = lift_levels(transient, floor=0.10, gamma=0.78)
    else:
        beat = robust_normalize(0.35 * beat + 0.65 * kick)
        bass = robust_normalize(0.75 * low + 0.55 * kick)
        beat = lift_levels(beat, floor=0.08, gamma=0.82)
        bass = lift_levels(bass, floor=0.10, gamma=0.80)
        transient = lift_levels(transient, floor=0.07, gamma=0.84)

    return {
        "energy": energy,
        "beat": beat,
        "bass": bass,
        "low": low,
        "mid": mid,
        "high": high,
        "kick": kick,
        "snare": snare,
        "transient": transient,
        "frames": np.array([n_frames], dtype=np.int64),
    }
