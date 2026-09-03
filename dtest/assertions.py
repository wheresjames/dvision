"""High-level independent assertions with optional failure artifacts.

Assertion messages name the violated boundary ("yaw-right decreased compass
heading", "blue landmark expected right, observed left") so a CI failure can
be located without an interactive reproduction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dtest.artifacts import save_failure_bundle
from dtest.calibration_scene import (
    CALIBRATION_FIXTURE,
    CENTER_X,
    EXPECTED_REGIONS,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    MINIMUM_MARKER_PIXELS,
)
from dtest.color_probe import color_centroid, patch_mean_rgb


def _centroids(frame_rgb: np.ndarray) -> dict:
    return {name: color_centroid(frame_rgb, name) for name in EXPECTED_REGIONS}


def assert_calibration_orientation(frame_rgb: np.ndarray, *,
                                   artifact_dir: Path | None = None) -> None:
    """Assert shape, horizontal/vertical orientation, and RGB channel order.

    Covers, independently of one another:

    * the frame shape, so a transpose cannot pass silently;
    * left/right ordering (red west of the nose, blue east of it);
    * top/bottom ordering (the elevated yellow panel above the short green one);
    * the neutral white panel just right of the forward axis;
    * coarse literal regions for every landmark;
    * channel order sampled at the red and blue panels themselves.
    """
    details: dict = {"shape": list(frame_rgb.shape)}
    try:
        assert frame_rgb.shape == (FRAME_HEIGHT, FRAME_WIDTH, 3), (
            f"frame shape {frame_rgb.shape} is not the "
            f"{FRAME_HEIGHT}x{FRAME_WIDTH} RGB24 contract"
        )
        marks = _centroids(frame_rgb)
        details.update({name: vars(c) for name, c in marks.items()})

        for name, centroid in marks.items():
            assert centroid.pixels >= MINIMUM_MARKER_PIXELS, (
                f"{name} landmark mask is only {centroid.pixels}px; expected "
                f"{MINIMUM_MARKER_PIXELS}+ (lighting or material regression)"
            )

        red, blue = marks["red"], marks["blue"]
        yellow, green = marks["yellow"], marks["green"]
        white = marks["white"]

        assert red.x < CENTER_X, (
            f"red landmark expected left, observed x={red.x:.1f} (centre {CENTER_X:.0f})"
        )
        assert blue.x > CENTER_X, (
            f"blue landmark expected right, observed x={blue.x:.1f} (centre {CENTER_X:.0f})"
        )
        assert red.x < white.x < blue.x, (
            "landmark left-to-right order is wrong: "
            f"red={red.x:.1f}, white={white.x:.1f}, blue={blue.x:.1f}"
        )
        assert yellow.y < green.y, (
            "elevated yellow landmark expected above green, observed "
            f"yellow y={yellow.y:.1f}, green y={green.y:.1f}"
        )

        for name, (x_lo, x_hi, y_lo, y_hi) in EXPECTED_REGIONS.items():
            c = marks[name]
            assert x_lo <= c.x <= x_hi and y_lo <= c.y <= y_hi, (
                f"{name} landmark centroid ({c.x:.1f}, {c.y:.1f}) is outside its "
                f"expected region x[{x_lo:.0f}, {x_hi:.0f}] y[{y_lo:.0f}, {y_hi:.0f}]"
            )

        assert_channel_order(frame_rgb, marks=marks, details=details)
    except Exception as exc:
        if artifact_dir is not None:
            details["error"] = repr(exc)
            observed = {}
            for name in EXPECTED_REGIONS:
                try:
                    c = color_centroid(frame_rgb, name)
                except Exception:
                    continue
                observed[name] = (c.x, c.y)
            save_failure_bundle(
                Path(artifact_dir), frame_rgb=frame_rgb, details=details,
                observed=observed, expected=EXPECTED_REGIONS,
                fixture=CALIBRATION_FIXTURE,
            )
        raise


def assert_channel_order(frame_rgb: np.ndarray, *, marks: dict | None = None,
                         details: dict | None = None) -> None:
    """Prove RGB24 channel order by sampling the red and blue panels."""
    marks = marks if marks is not None else _centroids(frame_rgb)
    red_patch = patch_mean_rgb(frame_rgb, marks["red"].x, marks["red"].y)
    blue_patch = patch_mean_rgb(frame_rgb, marks["blue"].x, marks["blue"].y)
    if details is not None:
        details["red_patch_rgb"] = list(red_patch)
        details["blue_patch_rgb"] = list(blue_patch)
    assert red_patch[0] > red_patch[2] + 40.0, (
        "red landmark patch does not lead in the R channel; frame is not RGB24: "
        f"rgb={tuple(round(v) for v in red_patch)}"
    )
    assert blue_patch[2] > blue_patch[0] + 40.0, (
        "blue landmark patch does not lead in the B channel; frame is not RGB24: "
        f"rgb={tuple(round(v) for v in blue_patch)}"
    )


def assert_landmark_moves(before_rgb: np.ndarray, after_rgb: np.ndarray,
                          color: str, direction: str, *, minimum_px: float = 10.0,
                          artifact_dir: Path | None = None) -> None:
    before = color_centroid(before_rgb, color)
    after = color_centroid(after_rgb, color)
    delta = after.x - before.x
    try:
        if direction == "left":
            assert delta <= -minimum_px, (
                f"{color} landmark expected to move left, observed {delta:+.1f}px "
                f"({before.x:.1f} -> {after.x:.1f})"
            )
        elif direction == "right":
            assert delta >= minimum_px, (
                f"{color} landmark expected to move right, observed {delta:+.1f}px "
                f"({before.x:.1f} -> {after.x:.1f})"
            )
        else:
            raise ValueError(f"unknown direction {direction!r}")
    except Exception as exc:
        if artifact_dir is not None:
            save_failure_bundle(
                Path(artifact_dir),
                frames={"before": before_rgb, "after": after_rgb},
                frame_rgb=after_rgb,
                observed={f"{color} before": (before.x, before.y),
                          f"{color} after": (after.x, after.y)},
                details={
                    "color": color,
                    "expected_direction": direction,
                    "observed_direction": "right" if delta > 0 else "left",
                    "delta_x_px": delta,
                    "minimum_px": minimum_px,
                    "error": repr(exc),
                },
                fixture=CALIBRATION_FIXTURE,
            )
        raise


def assert_heading_change(after_deg: float, before_deg: float, direction: str,
                          *, minimum_deg: float = 5.0) -> None:
    """Assert a semantic yaw direction using a normalized circular difference."""
    from dtest.contract import circular_delta_deg

    delta = circular_delta_deg(after_deg, before_deg)
    if direction == "right":
        assert delta >= minimum_deg, (
            f"yaw-right changed compass heading by {delta:+.2f} deg "
            f"({before_deg:.2f} -> {after_deg:.2f}); expected an increase of "
            f"at least {minimum_deg:.1f} deg"
        )
    elif direction == "left":
        assert delta <= -minimum_deg, (
            f"yaw-left changed compass heading by {delta:+.2f} deg "
            f"({before_deg:.2f} -> {after_deg:.2f}); expected a decrease of "
            f"at least {minimum_deg:.1f} deg"
        )
    else:
        raise ValueError(f"unknown direction {direction!r}")
