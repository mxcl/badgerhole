# musviz

`musviz` creates an offline music visualisation inspired by Sonique's
Rabbithole-style tunnel effects. It is not realtime; it renders frames and
produces an `.mp4` containing both visuals and the input audio.

## Requirements

- `ffmpeg`
- `uv` (used to run the script with ephemeral dependencies)

If tools are missing, install with `brewx` or run them via `pkgx`.

## Usage

```sh
./musviz.py input.mp3 output.mp4
```

Optional quality/performance settings:

```sh
./musviz.py input.flac output.mp4 \
  --width 1920 --height 1080 --fps 60 --preset slow --crf 16
```

Faster draft render:

```sh
./musviz.py input.mp3 draft.mp4 --width 960 --height 540 --fps 30 \
  --preset veryfast --crf 22
```

## Notes

- Output is H.264 video + AAC audio in an MP4 container.
- Visuals react to per-frame RMS, onset energy, and low/mid/high frequency
  bands extracted from the track.
- Rendering time depends on resolution, FPS, and track length.
