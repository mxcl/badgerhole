from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from musviz.audio import decode_audio_mono_f32, extract_features
from musviz.encode import start_encoder
from musviz.render import build_grid, render_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render an offline Rabbithole-style music visualisation and mux it "
            "with the input track into an MP4."
        )
    )
    parser.add_argument("input_audio", type=Path, help="Path to audio file")
    parser.add_argument("output_video", type=Path, help="Path to output .mp4 file")
    parser.add_argument(
        "--mode",
        choices=["auto", "rabbithole", "hiphop"],
        default="auto",
        help="Visual style mode. auto picks hiphop for hip-hop named tracks.",
    )
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


def resolve_mode(requested_mode: str, input_audio: Path) -> str:
    if requested_mode != "auto":
        return requested_mode
    name = input_audio.name.lower()
    if "hip hop" in name or "hip-hop" in name or "hiphop" in name:
        return "hiphop"
    return "rabbithole"


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

    mode = resolve_mode(args.mode, args.input_audio)
    print(f"Mode: {mode}", file=sys.stderr)

    print("Decoding audio...", file=sys.stderr)
    audio = decode_audio_mono_f32(args.input_audio, args.sample_rate)

    print("Extracting audio features...", file=sys.stderr)
    features = extract_features(audio, args.sample_rate, args.fps, mode)
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
                mode,
                grid,
                t=t,
                energy=float(features["energy"][i]),
                beat=float(features["beat"][i]),
                bass=float(features["bass"][i]),
                low=float(features["low"][i]),
                mid=float(features["mid"][i]),
                high=float(features["high"][i]),
                kick=float(features["kick"][i]),
                snare=float(features["snare"][i]),
                transient=float(features["transient"][i]),
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
