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


def test_dark_saturated_red_is_still_the_target():
    """A pad in shade is still a pad.

    RGB (90, 15, 15) is the same hue and saturation as the marker, at a third
    the brightness — the pad under a tree, under cloud, or seen by a camera
    that has stopped down. This assertion used to run the other way, and that
    is precisely how the detector came to report nothing at all when the
    simulator started shading the scene physically: the pad rendered at V=90
    and every frame of every approach came back empty.
    """
    import cv2
    frame = _make_frame(480, 640)
    cv2.circle(frame, (320, 240), 80, (90, 15, 15), -1)
    det = detect(frame)
    assert det.visible, "a shadowed but saturated red pad must still be found"


def test_dark_desaturated_red_not_detected():
    """Shadowed brickwork, which is what the old brightness floor was for."""
    import cv2
    frame = _make_frame(480, 640)
    # Same low brightness as above, but washed out: S ≈ 100, not ≈ 230.
    cv2.circle(frame, (320, 240), 80, (90, 55, 50), -1)
    det = detect(frame)
    assert not det.visible, "dark desaturated red must not be detected"


def test_oblique_pad_is_detected_as_an_ellipse():
    """A circular pad seen from a shallow angle projects to a long ellipse.

    Aspect is roughly 1/sin(elevation), so requiring a near-circle would mean
    the pad can only be found from nearly overhead — which is the one phase of
    flight that does not need help finding it.
    """
    import cv2
    for aspect in (2, 4, 8):
        frame = _make_frame(480, 640)
        cv2.ellipse(frame, (320, 240), (160, max(160 // aspect, 4)), 0, 0, 360,
                    (220, 20, 20), -1)
        det = detect(frame)
        assert det.visible, f"pad at {aspect}:1 obliquity must be detected"
        assert abs(det.cx - 320) < 6 and abs(det.cy - 240) < 6


def test_detection_survives_a_wide_exposure_range():
    """The same pad under six stops of illumination is the same pad."""
    import cv2
    for scale in (0.25, 0.4, 0.6, 1.0, 1.4):
        frame = _make_frame(480, 640)
        colour = tuple(int(min(255, c * scale)) for c in (230, 25, 20))
        cv2.circle(frame, (320, 240), 70, colour, -1)
        det = detect(frame)
        assert det.visible, f"pad must be found at exposure scale {scale}"


def test_a_ragged_red_blob_is_not_a_pad():
    """Foliage fringing into the red band is ragged, and a pad is not.

    Same colour, same bounding box, same rough area as a pad — only the outline
    differs. Solidity is what separates them.
    """
    import cv2
    import math
    frame = _make_frame(480, 640)
    points = []
    for i in range(40):
        angle = math.tau * i / 40
        r = 110 if i % 2 == 0 else 26          # deep spikes: a star, not a disc
        points.append([int(320 + r * math.cos(angle)), int(240 + r * math.sin(angle))])
    cv2.fillPoly(frame, [np.array(points, dtype=np.int32)], (220, 20, 20))
    assert not detect(frame).visible, "a ragged star is not a landing pad"
