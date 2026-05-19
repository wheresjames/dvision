"""OpenCV-based red landing target detector.

Converts an RGB frame to HSV and thresholds for red hue (which wraps around 0°
in HSV, so two ranges are needed).  The largest sufficiently-circular contour
is selected as the landing target.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


# HSV thresholds for bright red.  Red wraps around 0° in OpenCV HSV (0-180
# scale), so two bands are required.
#
# The target marker is rendered at RGB (0.88, 0.14, 0.10) ≈ HSV (2°, 225, 224)
# — pure red, very saturated, very bright.
#
# Brown/brick texture sits at HSV ~(12-20°, 130-160, 100-170).
# The three constraints that separate them:
#   H  : 0-6 and 172-180  (pure red only; brick hue is 12+)
#   S  : ≥ 160             (highly saturated; brick is 130-160, often below)
#   V  : ≥ 120             (bright; eliminates shadowed/dark reds)
_RED_LO1 = np.array([0,   160, 120], dtype=np.uint8)
_RED_HI1 = np.array([6,   255, 255], dtype=np.uint8)
_RED_LO2 = np.array([172, 160, 120], dtype=np.uint8)
_RED_HI2 = np.array([180, 255, 255], dtype=np.uint8)

# Minimum contour area in pixels to be considered a candidate.
_MIN_AREA = 30.0

# Minimum circularity score (4π·area / perimeter²) to accept a contour.
_MIN_CIRCULARITY = 0.45


@dataclass
class Detection:
    """Result of one detector call."""
    visible: bool
    cx: float        # centre x in pixels (0 if not visible)
    cy: float        # centre y in pixels (0 if not visible)
    radius: float    # approximate radius in pixels (0 if not visible)
    confidence: float  # 0.0–1.0 based on circularity and area

    @property
    def as_dict(self) -> dict:
        return {
            "visible": self.visible,
            "cx": self.cx,
            "cy": self.cy,
            "radius": self.radius,
            "confidence": self.confidence,
        }


_NULL = Detection(visible=False, cx=0.0, cy=0.0, radius=0.0, confidence=0.0)


def detect(frame_rgb: np.ndarray) -> Detection:
    """Detect the red landing target in *frame_rgb* (H×W×3, uint8, RGB).

    Returns a Detection with visible=False when no target is found.
    """
    if frame_rgb is None or frame_rgb.size == 0:
        return _NULL

    bgr = cv2.cvtColor(np.ascontiguousarray(frame_rgb), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    mask = cv2.bitwise_or(
        cv2.inRange(hsv, _RED_LO1, _RED_HI1),
        cv2.inRange(hsv, _RED_LO2, _RED_HI2),
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return _NULL

    best: tuple | None = None  # (score, cx, cy, radius, circularity)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < _MIN_AREA:
            continue
        perim = cv2.arcLength(cnt, True)
        if perim < 1e-6:
            continue
        circularity = 4.0 * np.pi * area / (perim * perim)
        if circularity < _MIN_CIRCULARITY:
            continue
        moments = cv2.moments(cnt)
        if abs(moments["m00"]) > 1e-6:
            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]
        else:
            (cx, cy), _ = cv2.minEnclosingCircle(cnt)
        _, r = cv2.minEnclosingCircle(cnt)
        score = circularity * area
        if best is None or score > best[0]:
            best = (score, cx, cy, r, circularity)

    if best is None:
        return _NULL

    _, cx, cy, r, circ = best
    # Confidence: blend circularity (shape quality) with a mild area bonus.
    confidence = float(min(1.0, circ * 1.1))
    return Detection(visible=True, cx=float(cx), cy=float(cy),
                     radius=float(r), confidence=confidence)
