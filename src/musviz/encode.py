from __future__ import annotations

import subprocess
from pathlib import Path


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
