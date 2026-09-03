"""Simple independent RGB landmark measurements.

The masks here are deliberately plain arithmetic on the raw array. They must
never call a production colour-conversion helper, because a client that
implements a channel order must not also define the oracle for it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Centroid:
    x: float
    y: float
    pixels: int


COLOR_NAMES = ("red", "blue", "yellow", "green", "white")


def _masks(frame_rgb: np.ndarray) -> dict:
    r = frame_rgb[:, :, 0].astype(np.int16)
    g = frame_rgb[:, :, 1].astype(np.int16)
    b = frame_rgb[:, :, 2].astype(np.int16)
    return {
        "red": (r > 150) & (g < 100) & (b < 100),
        "blue": (b > 150) & (r < 100) & (g < 100),
        "yellow": (r > 150) & (g > 150) & (b < 100),
        "green": (g > 150) & (r < 100) & (b < 100),
        "white": (r > 210) & (g > 210) & (b > 210),
    }


def color_centroid(frame_rgb: np.ndarray, color: str,
                   *, minimum_pixels: int = 20) -> Centroid:
    if frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
        raise AssertionError(f"expected HxWx3 RGB frame, got {frame_rgb.shape}")
    masks = _masks(frame_rgb)
    if color not in masks:
        raise ValueError(f"unknown calibration color {color!r}")
    ys, xs = np.nonzero(masks[color])
    if len(xs) < minimum_pixels:
        raise AssertionError(f"{color} marker has {len(xs)} pixels, expected {minimum_pixels}+")
    return Centroid(float(xs.mean()), float(ys.mean()), int(len(xs)))


def patch_mean_rgb(frame_rgb: np.ndarray, x: float, y: float,
                   *, half: int = 4) -> tuple[float, float, float]:
    """Mean R, G, B of a small patch centred on (x, y).

    Used to prove channel order at a known landmark independently of the
    colour masks, so a frame whose R and B planes are exchanged fails even if
    the exchanged landmarks happen to land in plausible regions.
    """
    h, w = frame_rgb.shape[:2]
    cx, cy = int(round(x)), int(round(y))
    x0, x1 = max(0, cx - half), min(w, cx + half + 1)
    y0, y1 = max(0, cy - half), min(h, cy + half + 1)
    patch = frame_rgb[y0:y1, x0:x1].astype(np.float64)
    if patch.size == 0:
        raise AssertionError(f"patch at ({x}, {y}) is outside the {w}x{h} frame")
    return tuple(float(v) for v in patch.reshape(-1, 3).mean(axis=0))


def horizon_row(frame_rgb: np.ndarray, column: int) -> int:
    """Lowest image row still classified as sky in ``column``.

    Sky is the only large region whose blue channel dominates red, so this is a
    cheap independent measure of the horizon height that needs no production
    camera model. A smaller row number means the horizon sits higher up.
    """
    r = frame_rgb[:, column, 0].astype(np.int16)
    b = frame_rgb[:, column, 2].astype(np.int16)
    sky = (b > r + 25) & (b > 120)
    rows = np.nonzero(sky)[0]
    if len(rows) == 0:
        raise AssertionError(f"no sky pixels in column {column}")
    return int(rows.max())
