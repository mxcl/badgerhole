from __future__ import annotations

import numpy as np


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


def _unsharp(rgb: np.ndarray, amount: float) -> np.ndarray:
    padded = np.pad(rgb, ((1, 1), (1, 1), (0, 0)), mode="edge")
    blur = (
        padded[1:-1, 1:-1]
        + padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
    ) / 5.0
    return np.clip(rgb + amount * (rgb - blur), 0.0, 1.0)


def _render_rabbithole(
    grid: dict[str, np.ndarray],
    t: float,
    energy: float,
    beat: float,
    bass: float,
    mid: float,
    high: float,
    transient: float,
) -> np.ndarray:
    x = grid["x"]
    y = grid["y"]
    radius = grid["radius"]
    angle = grid["angle"]
    vignette = grid["vignette"]

    time_scale = 0.36 + bass * 0.26 + beat * 0.16
    tt = t * time_scale

    wobble = 0.030 * np.sin(angle * 2.0 + tt * (0.5 + bass * 1.0))
    wobble += 0.015 * np.sin(angle * 4.0 - tt * (0.3 + bass * 0.8))
    depth = 1.0 / (radius + 0.42 + wobble)

    twist = angle + depth * (0.20 + bass * 2.6) + tt * (0.35 + bass * 0.9)
    rings = np.sin(depth * (10.0 + bass * 22.0) - tt * (2.3 + bass * 1.3))
    spoke_lobes = int(round(8.0 + high * 8.0 + bass * 8.0))
    spokes = np.sin(twist * max(3, spoke_lobes))

    drift = np.sin(tt * 0.45)
    ripples = np.sin(
        (x * np.cos(drift) + y * np.sin(drift)) * (8.0 + mid * 10.0 + bass * 12.0)
        + tt * (0.9 + bass * 1.8)
    )

    fine = np.sin(depth * (22.0 + bass * 18.0) + twist * 1.7 - tt * (0.8 + beat * 1.2))
    pattern = 0.35 + 0.42 * rings + 0.17 * spokes + 0.09 * ripples + 0.10 * fine
    pattern += 0.18 * np.sin(depth * 2.0 - tt * 1.1 + beat * 9.0 + bass * 6.0)
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
    glow = np.power(np.clip(pattern - 0.80, 0.0, 1.0), 2.0) * (
        0.06 + beat * 0.22 + bass * 0.10
    )
    edge_detail = np.power(np.clip(np.abs(spokes), 0.0, 1.0), 1.35) * (
        0.22 + bass * 0.16 + beat * 0.12 + transient * 0.08
    )
    val = np.clip(val + glow + edge_detail, 0.0, 1.0)
    val = np.clip((val - 0.5) * 1.44 + 0.5, 0.0, 1.0)

    rgb = hsv_to_rgb(hue.astype(np.float32), sat.astype(np.float32), val.astype(np.float32))
    sharp_amount = 1.06 + 0.18 * beat + 0.10 * bass
    rgb = _unsharp(rgb, sharp_amount)
    return (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)


def _render_hiphop(
    grid: dict[str, np.ndarray],
    t: float,
    energy: float,
    beat: float,
    bass: float,
    mid: float,
    high: float,
    snare: float,
    transient: float,
) -> np.ndarray:
    x = grid["x"]
    y = grid["y"]
    radius = grid["radius"]
    angle = grid["angle"]
    vignette = grid["vignette"]

    time_scale = 0.26 + bass * 0.20 + transient * 0.14
    tt = t * time_scale

    wobble = 0.020 * np.sin(angle * 2.0 - tt * (0.4 + bass * 0.7))
    wobble += 0.012 * np.sin(angle * 6.0 + tt * (0.3 + snare * 0.9))
    depth = 1.0 / (radius + 0.38 + wobble)

    spoke_lobes = max(6, int(round(10.0 + bass * 10.0 + transient * 6.0)))
    sector_lobes = max(4, int(round(5.0 + snare * 9.0)))

    twist = angle + depth * (0.18 + bass * 2.2) + tt * (0.26 + bass * 0.75)
    rings = np.sin(depth * (8.0 + bass * 18.0) - tt * (1.7 + bass * 1.2))
    spokes = np.sin(twist * spoke_lobes)
    sectors = np.sin(angle * sector_lobes - tt * (0.5 + snare * 0.8))

    rail = np.sin(
        (x * np.cos(tt * 0.3) + y * np.sin(tt * 0.3))
        * (7.0 + mid * 8.0 + bass * 8.0)
        + tt * (0.9 + transient * 1.3)
    )

    rings_hard = np.power(np.clip(0.5 + 0.5 * rings, 0.0, 1.0), 0.62)
    spokes_hard = np.power(np.clip(np.abs(spokes), 0.0, 1.0), 0.50)
    sectors_hard = np.power(np.clip(0.5 + 0.5 * sectors, 0.0, 1.0), 0.75)

    pattern = 0.24 + 0.45 * rings_hard + 0.33 * spokes_hard + 0.10 * sectors_hard
    pattern += 0.09 * np.power(np.clip(0.5 + 0.5 * rail, 0.0, 1.0), 1.1)
    pattern += 0.18 * np.sin(depth * (18.0 + bass * 14.0) - tt * 2.2 + snare * 6.5)
    pattern = np.clip(pattern, 0.0, 1.0)

    hue = (
        0.58
        + 0.05 * np.sin(depth * 0.45 - tt * 0.28)
        - 0.08 * bass
        + 0.05 * snare
        + 0.03 * high
    ) % 1.0
    sat = np.clip(0.64 + 0.18 * bass + 0.18 * transient + 0.06 * np.abs(sectors), 0.0, 1.0)

    center_radius = 0.25 + bass * 0.09 + beat * 0.06
    center_gate = np.clip((radius - center_radius) / 0.10, 0.0, 1.0)

    val = pattern * (0.42 + bass * 0.74 + transient * 0.30) + 0.20 * beat + 0.20 * snare
    val = val * center_gate * vignette
    glow = np.power(np.clip(pattern - 0.86, 0.0, 1.0), 2.1) * (0.03 + transient * 0.09)
    edge_detail = np.power(spokes_hard, 1.2) * (0.32 + transient * 0.14 + bass * 0.14)
    val = np.clip(val + glow + edge_detail, 0.0, 1.0)
    val = np.clip((val - 0.5) * 1.58 + 0.5, 0.0, 1.0)

    rgb = hsv_to_rgb(hue.astype(np.float32), sat.astype(np.float32), val.astype(np.float32))
    sharp_amount = 1.30 + 0.16 * transient + 0.06 * bass
    rgb = _unsharp(rgb, sharp_amount)
    return (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)


def render_frame(
    mode: str,
    grid: dict[str, np.ndarray],
    t: float,
    energy: float,
    beat: float,
    bass: float,
    low: float,
    mid: float,
    high: float,
    kick: float,
    snare: float,
    transient: float,
) -> np.ndarray:
    del low, kick
    if mode == "hiphop":
        return _render_hiphop(
            grid,
            t,
            energy,
            beat,
            bass,
            mid,
            high,
            snare,
            transient,
        )
    return _render_rabbithole(
        grid,
        t,
        energy,
        beat,
        bass,
        mid,
        high,
        transient,
    )
