#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.2.0"]
# ///

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

EPSILON = 1e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render an offline Rabbithole-style music visualisation and mux it "
            "with the input track into an MP4."
        )
    )
    parser.add_argument("input_audio", type=Path, help="Path to audio file")
    parser.add_argument("output_video", type=Path, help="Path to output .mp4 file")
    parser.add_argument("--width", type=int, default=1280, help="Video width")
    parser.add_argument("--height", type=int, default=720, help="Video height")
    parser.add_argument("--fps", type=int, default=60, help="Frames per second")
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=22050,
        help="Audio decoding rate for feature extraction",
    )
    parser.add_argument(
        "--preset",
        default="medium",
        help="x264 preset for ffmpeg output (eg. veryfast, medium, slow)",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=18,
        help="x264 CRF quality (lower is better quality)",
    )
    return parser.parse_args()


def ensure_binary(name: str) -> None:
    if shutil.which(name):
        return
    msg = (
        f"Required tool '{name}' is not installed. "
        "Install with brewx or run via pkgx."
    )
    raise SystemExit(msg)


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


def extract_features(audio: np.ndarray, sample_rate: int, fps: int) -> dict[str, np.ndarray]:
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
    kick = robust_normalize(moving_average(low_onset, max(2, fps // 10)))
    beat = robust_normalize(0.35 * beat + 0.65 * kick)
    bass = robust_normalize(0.75 * low + 0.55 * kick)
    beat = lift_levels(beat, floor=0.08, gamma=0.82)
    bass = lift_levels(bass, floor=0.10, gamma=0.80)

    return {
        "energy": energy,
        "beat": beat,
        "bass": bass,
        "low": low,
        "mid": mid,
        "high": high,
        "frames": np.array([n_frames], dtype=np.int64),
    }


def hsv_to_rgb(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    h6 = h[..., None] * 6.0
    offsets = np.array([0.0, 4.0, 2.0], dtype=np.float32)
    k = np.mod(h6 + offsets, 6.0)
    rgb = np.clip(np.abs(k - 3.0) - 1.0, 0.0, 1.0)
    rgb = rgb * rgb * (3.0 - 2.0 * rgb)
    return v[..., None] * ((1.0 - s[..., None]) + s[..., None] * rgb)


def build_grid(width: int, height: int) -> dict[str, np.ndarray]:
    xs = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    ys = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")

    aspect = width / max(1, height)
    x = xx * aspect
    y = yy

    radius = np.sqrt(x * x + y * y).astype(np.float32)
    angle = np.arctan2(y, x).astype(np.float32)

    vignette = np.clip(1.25 - radius * 0.9, 0.0, 1.0).astype(np.float32)

    return {
        "x": x,
        "y": y,
        "radius": radius,
        "angle": angle,
        "vignette": vignette,
    }


def render_frame(
    grid: dict[str, np.ndarray],
    t: float,
    energy: float,
    beat: float,
    bass: float,
    low: float,
    mid: float,
    high: float,
) -> np.ndarray:
    x = grid["x"]
    y = grid["y"]
    radius = grid["radius"]
    angle = grid["angle"]
    vignette = grid["vignette"]

    time_scale = 0.38 + bass * 0.28 + beat * 0.18
    tt = t * time_scale

    wobble = 0.045 * np.sin(angle * 2.2 + tt * (0.5 + bass * 1.0))
    depth = 1.0 / (radius + 0.42 + wobble)

    twist = angle + depth * (0.20 + bass * 2.6) + tt * (0.35 + bass * 0.9)
    rings = np.sin(depth * (10.0 + bass * 22.0) - tt * (2.3 + bass * 1.3))
    spokes = np.sin(twist * (6.0 + high * 10.0 + bass * 7.0))

    drift = np.sin(tt * 0.45)
    ripples = np.sin(
        (x * np.cos(drift) + y * np.sin(drift)) * (8.0 + mid * 10.0 + bass * 12.0)
        + tt * (0.9 + bass * 1.8)
    )

    pattern = 0.40 + 0.40 * rings + 0.15 * spokes + 0.11 * ripples
    pattern += 0.20 * np.sin(depth * 2.0 - tt * 1.1 + beat * 9.0 + bass * 6.0)
    pattern = np.clip(pattern, 0.0, 1.0)

    hue = (
        0.55
        + 0.16 * np.sin(depth * 0.55 - tt * 0.30)
        + 0.08 * np.sin(tt * 0.42 + beat * 3.2)
        + high * 0.05
        + bass * 0.04
    ) % 1.0

    sat = np.clip(0.58 + 0.18 * np.abs(spokes) + bass * 0.24 + high * 0.10, 0.0, 1.0)

    center_radius = 0.22 + bass * 0.12 + beat * 0.08
    center_gate = np.clip((radius - center_radius) / 0.12, 0.0, 1.0)

    val = pattern * (0.48 + bass * 0.70 + energy * 0.14) + 0.35 * beat + 0.22 * bass
    val = val * center_gate * vignette
    glow = np.power(np.clip(pattern - 0.74, 0.0, 1.0), 2.0) * (0.12 + beat * 0.45 + bass * 0.20)
    edge_detail = np.power(np.clip(np.abs(spokes), 0.0, 1.0), 1.35) * (
        0.14 + bass * 0.14 + beat * 0.10
    )
    val = np.clip(val + glow + edge_detail, 0.0, 1.0)
    val = np.clip((val - 0.5) * 1.32 + 0.5, 0.0, 1.0)

    rgb = hsv_to_rgb(hue.astype(np.float32), sat.astype(np.float32), val.astype(np.float32))
    blur = (
        rgb
        + np.roll(rgb, 1, axis=0)
        + np.roll(rgb, -1, axis=0)
        + np.roll(rgb, 1, axis=1)
        + np.roll(rgb, -1, axis=1)
    ) / 5.0
    sharp_amount = 0.78 + 0.18 * beat + 0.10 * bass
    rgb = np.clip(rgb + sharp_amount * (rgb - blur), 0.0, 1.0)
    return (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)


def start_encoder(
    output_video: Path,
    input_audio: Path,
    width: int,
    height: int,
    fps: int,
    preset: str,
    crf: int,
) -> subprocess.Popen[bytes]:
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-i",
        str(input_audio),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        str(output_video),
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def format_seconds(seconds: float) -> str:
    minutes = int(seconds // 60)
    rem = seconds - minutes * 60
    return f"{minutes:02d}:{rem:05.2f}"


def main() -> None:
    args = parse_args()
    ensure_binary("ffmpeg")

    if args.width < 64 or args.height < 64:
        raise SystemExit("Video dimensions should both be at least 64")
    if args.fps < 12:
        raise SystemExit("Use --fps of at least 12 for acceptable motion")

    if not args.input_audio.exists():
        raise SystemExit(f"Input audio file not found: {args.input_audio}")

    print("Decoding audio...", file=sys.stderr)
    audio = decode_audio_mono_f32(args.input_audio, args.sample_rate)

    print("Extracting audio features...", file=sys.stderr)
    features = extract_features(audio, args.sample_rate, args.fps)
    n_frames = int(features["frames"][0])
    length_seconds = n_frames / args.fps

    grid = build_grid(args.width, args.height)
    encoder = start_encoder(
        args.output_video,
        args.input_audio,
        args.width,
        args.height,
        args.fps,
        args.preset,
        args.crf,
    )

    if encoder.stdin is None:
        raise SystemExit("Failed to open encoder stdin")

    print(
        (
            f"Rendering {n_frames} frames "
            f"({format_seconds(length_seconds)}) at {args.width}x{args.height}@{args.fps}"
        ),
        file=sys.stderr,
    )

    try:
        for i in range(n_frames):
            t = i / args.fps
            frame = render_frame(
                grid,
                t=t,
                energy=float(features["energy"][i]),
                beat=float(features["beat"][i]),
                bass=float(features["bass"][i]),
                low=float(features["low"][i]),
                mid=float(features["mid"][i]),
                high=float(features["high"][i]),
            )
            try:
                encoder.stdin.write(frame.tobytes())
            except BrokenPipeError:
                break

            if i % max(1, args.fps) == 0 or i + 1 == n_frames:
                pct = ((i + 1) / n_frames) * 100.0
                print(f"\rRendering: {pct:5.1f}%", end="", file=sys.stderr)
        print("", file=sys.stderr)
    finally:
        encoder.stdin.close()

    code = encoder.wait()
    if code != 0:
        raise SystemExit("ffmpeg failed while encoding output video")

    print(f"Wrote {args.output_video}")


if __name__ == "__main__":
    main()
