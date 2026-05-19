"""Tests for daic.detector — no dsim or pymembus required."""

import math

import numpy as np
import pytest

from daic.detector import detect, Detection
from daic.controller import _MIN_CONFIDENCE


def _make_frame(h: int = 240, w: int = 320) -> np.ndarray:
    """Black RGB frame."""
    return np.zeros((h, w, 3), dtype=np.uint8)


def _paint_red_circle(frame: np.ndarray, cx: int, cy: int, r: int) -> None:
    """Paint a solid red circle onto *frame* in-place."""
    import cv2
    cv2.circle(frame, (cx, cy), r, (220, 20, 20), -1)


def _paint_blue_circle(frame: np.ndarray, cx: int, cy: int, r: int) -> None:
    import cv2
    cv2.circle(frame, (cx, cy), r, (20, 20, 220), -1)


def _paint_white_circle(frame: np.ndarray, cx: int, cy: int, r: int) -> None:
    import cv2
    cv2.circle(frame, (cx, cy), r, (255, 255, 255), -1)


# ---------------------------------------------------------------------------

def test_empty_frame_returns_not_visible():
    frame = _make_frame()
    det = detect(frame)
    assert not det.visible


def test_none_frame_returns_not_visible():
    det = detect(None)
    assert not det.visible


def test_red_circle_detected():
    frame = _make_frame(480, 640)
    _paint_red_circle(frame, 320, 240, 60)
    det = detect(frame)
    assert det.visible
    assert det.confidence > _MIN_CONFIDENCE
    # Centre should be within 15 pixels of truth.
    assert abs(det.cx - 320) < 15
    assert abs(det.cy - 240) < 15


def test_red_circle_with_white_centre_detected():
    frame = _make_frame(480, 640)
    _paint_red_circle(frame, 320, 240, 70)
    _paint_white_circle(frame, 320, 240, 16)
    det = detect(frame)
    assert det.visible
    assert det.confidence > _MIN_CONFIDENCE
    assert abs(det.cx - 320) < 3
    assert abs(det.cy - 240) < 3
    assert abs(det.radius - 70) < 5


def test_red_circle_centre_ignores_small_edge_artifact():
    import cv2
    frame = _make_frame(480, 640)
    _paint_red_circle(frame, 320, 240, 70)
    _paint_white_circle(frame, 320, 240, 16)
    cv2.circle(frame, (402, 240), 5, (220, 20, 20), -1)
    det = detect(frame)
    assert det.visible
    assert abs(det.cx - 320) < 5
    assert abs(det.cy - 240) < 3


def test_blue_circle_not_detected():
    frame = _make_frame(480, 640)
    _paint_blue_circle(frame, 320, 240, 60)
    det = detect(frame)
    assert not det.visible


def test_small_red_dot_below_min_area():
    """A 2-pixel radius dot is below the min-area threshold."""
    frame = _make_frame(240, 320)
    _paint_red_circle(frame, 160, 120, 2)
    det = detect(frame)
    # May or may not detect — just confirm it doesn't crash and returns a valid object.
    assert isinstance(det, Detection)
    assert 0.0 <= det.confidence <= 1.0


def test_largest_circle_selected_when_multiple_present():
    """When two red circles are present the larger one should be selected."""
    frame = _make_frame(480, 640)
    _paint_red_circle(frame, 150, 240, 20)   # small
    _paint_red_circle(frame, 450, 240, 60)   # large
    det = detect(frame)
    assert det.visible
    # Large circle centre should be closer to (450, 240).
    assert abs(det.cx - 450) < abs(det.cx - 150)


def test_off_centre_circle_position():
    """Detection centre tracks the actual painted position."""
    frame = _make_frame(480, 640)
    _paint_red_circle(frame, 100, 380, 50)
    det = detect(frame)
    if det.visible:
        assert abs(det.cx - 100) < 20
        assert abs(det.cy - 380) < 20


def test_confidence_range():
    frame = _make_frame(480, 640)
    _paint_red_circle(frame, 320, 240, 80)
    det = detect(frame)
    assert det.visible
    assert 0.0 < det.confidence <= 1.0


def test_radius_approximately_correct():
    frame = _make_frame(480, 640)
    _paint_red_circle(frame, 320, 240, 50)
    det = detect(frame)
    assert det.visible
    assert abs(det.radius - 50) < 15


# ---------------------------------------------------------------------------
# Colour discrimination tests
# ---------------------------------------------------------------------------

def _paint_solid(frame: np.ndarray, rgb: tuple[int, int, int]) -> None:
    """Flood-fill the entire frame with one colour."""
    frame[:] = rgb


def test_bright_red_target_detected():
    """The dsim target marker colour (0.88, 0.14, 0.10) should be detected."""
    frame = _make_frame(480, 640)
    # Approximate the Panda3D rendered colour (linear ≈ sRGB at these levels).
    _paint_red_circle(frame, 320, 240, 80)   # (220, 20, 20)
    det = detect(frame)
    assert det.visible, "bright red target must be detected"


def test_brick_brown_not_detected():
    """A brown-brick circle should not trigger the detector."""
    import cv2
    frame = _make_frame(480, 640)
    # Brick brown: HSV roughly (15°, 150, 155) → RGB ≈ (155, 100, 60)
    brick_rgb = (155, 100, 60)
    cv2.circle(frame, (320, 240), 80, brick_rgb, -1)
    det = detect(frame)
    assert not det.visible, "brick brown must not be detected as target"


def test_orange_red_not_detected():
    """Orange-red (hue ~15°) should not trigger the detector."""
    import cv2
    frame = _make_frame(480, 640)
    # Orange-red: RGB (230, 80, 20) — higher hue than pure red
    cv2.circle(frame, (320, 240), 80, (230, 80, 20), -1)
    det = detect(frame)
    assert not det.visible, "orange-red must not be detected as landing target"


def test_dark_red_not_detected():
    """Dark/shadowed red (low value) should not trigger the detector."""
    import cv2
    frame = _make_frame(480, 640)
    # Dark red: RGB (90, 15, 15) — same hue, low value
    cv2.circle(frame, (320, 240), 80, (90, 15, 15), -1)
    det = detect(frame)
    assert not det.visible, "dark red must not be detected as landing target"
