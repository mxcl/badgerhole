# Badgerhole

`badgerhole` creates an offline music visualisation inspired by Sonique's
Rabbithole-style tunnel effects. It is not realtime; it renders frames and
produces an `.mp4` containing both visuals and the input audio.

## Examples

* [Country](https://www.youtube.com/watch?v=abAwojOvsqY)
* [Trance](https://www.youtube.com/embed/dOfkiYO1I18)

## Requirements

- `ffmpeg`
- `uv` (used to run the script with ephemeral dependencies)

## Usage

```sh
./badgerhole.py input.mp3 output.mp4
```

Hip-hop tuned mode:

```sh
./badgerhole.py "Forces Join (Hip Hop).mp3" out.mp4 --mode hiphop
```

`--mode auto` is the default and picks `hiphop` when the input filename contains
`hip hop`, `hip-hop`, or `hiphop`; otherwise it uses `rabbithole`.

Optional quality/performance settings:

```sh
./badgerhole.py input.flac output.mp4 \
  --width 1920 --height 1080 --fps 60 --preset slow --crf 16
```

Faster draft render:

```sh
./badgerhole.py input.mp3 draft.mp4 --width 960 --height 540 --fps 30 \
  --preset veryfast --crf 22
```

## Notes

- Output is H.264 video + AAC audio in an MP4 container.
- Visuals react to per-frame RMS, onset energy, and low/mid/high frequency
  bands extracted from the track.
- Project code is split across `src/badgerhole/`:
  - `cli.py` argument parsing and render loop
  - `audio.py` audio decode and feature extraction
  - `render.py` frame generation for each mode
  - `encode.py` ffmpeg output pipeline
- Rendering time depends on resolution, FPS, and track length.
