"""OpenCV-based red landing target detector.

The target is found as a **reddish, roughly elliptical blob**: a chromatic gate
picks red pixels, and a shape test then decides whether the blob could be a
circular pad seen from some viewpoint. Both halves are deliberately chosen to
survive a real camera looking at a real pad.

Why the gate is chromatic and not photometric
---------------------------------------------
Hue and saturation are properties of the *surface*; value is a property of the
*illumination* on it. A red pad in open sun, in overcast, in a tree's shadow, or
under a camera that has just stopped down all produce the same hue and much the
same saturation, and wildly different values. So the pad is identified by hue
and saturation, and value is used only as a noise floor: below `_MIN_VALUE` a
pixel is dark enough that its hue is quantisation noise rather than colour.

This is not a theoretical concern. An earlier version of this file required
V >= 120, which is a statement that the pad is brightly lit. When the simulator
gained a physically-shaded scene the pad rendered at V = 90 -- hue and
saturation perfect, brightness merely lower -- and the detector reported nothing
at all on every frame of every approach. A pad in shade is still a pad.

Why the shape test fits an ellipse
----------------------------------
A circular pad is only circular from directly above. From a drone on approach it
is an ellipse whose aspect ratio is about 1/sin(elevation angle): at 9 m range
and 1 m altitude that is roughly 9:1. Testing circularity therefore amounts to
testing that the drone is nearly overhead, which is exactly the phase of flight
where the pad is easiest to find anyway. Instead the blob is fitted with an
ellipse and judged on how well it *fills* that ellipse, which is invariant to
viewing angle, plus solidity and extent, which reject the ragged, fragmented
blobs that foliage and brickwork produce.

What still gets rejected, and why
---------------------------------
Brick and foliage are the two confusers this scene actually contains, and both
are separable on hue: brick sits at H 12-20 and the tree canopy at H ~8, while
the pad sits at H 0-2. Neither survives the hue gate. Shadowed brick is also
below the saturation floor. Sensor noise and heavy JPEG chroma artefacts can
still fringe a canopy edge into the red band, but such blobs are ragged and
flicker frame to frame; the planner requires several consecutive detections
before it acts on one.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


# --- Chromatic gate -------------------------------------------------------
#
# Red wraps around 0 degrees, so OpenCV's 0-180 hue scale needs two bands. The
# pad measures H 0-2 in the simulator under both scene presets; the tree canopy
# measures H ~8 and brick H 12-20, so the band is kept tight.
_HUE_HALF_WIDTH = 5
# The pad measures S 227-250. Reddish background pixels in this scene have a
# median S near 80 and a 99th percentile near 213, so 175 clears the background
# comfortably while leaving room for the saturation loss a gamma or white
# balance shift causes.
_MIN_SATURATION = 175
# Not an illumination gate: below this, hue is quantisation noise.
_MIN_VALUE = 30

# --- Shape test -----------------------------------------------------------
#
# Minimum blob area in pixels. Smaller than this the pad is a handful of pixels
# and its shape cannot be judged, which is where false alarms live.
_MIN_AREA = 25.0
# Below this area an ellipse fit is not meaningful; extent alone is used.
_SHAPE_MIN_AREA = 60.0
# Contour area over fitted-ellipse area. A filled ellipse scores 1.0. Observed
# pads sit at 0.91 and above; ragged foliage fragments sit near 0.68.
_MIN_FILL, _MAX_FILL = 0.82, 1.60
# Contour area over convex-hull area. Convex blobs score near 1.0; the pad's 1st
# percentile is 0.70, while foliage fragments have a median near 0.46.
_MIN_SOLIDITY = 0.70
# Blob area over bounding-box area. An ellipse scores pi/4 = 0.785.
_MIN_EXTENT = 0.50
# 1/sin(elevation): 14 corresponds to about 4 degrees above the horizon, beyond
# which the pad is a line and its centre is not usefully localised.
_MAX_ASPECT = 14.0


@dataclass
class Detection:
    """Result of one detector call."""
    visible: bool
    cx: float        # centre x in pixels (0 if not visible)
    cy: float        # centre y in pixels (0 if not visible)
    radius: float    # approximate radius in pixels (0 if not visible)
    confidence: float  # 0.0–1.0 based on shape quality and viewing obliquity

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


def _red_mask(frame_rgb: np.ndarray) -> np.ndarray:
    """Pixels whose hue and saturation say "painted red", at any brightness."""
    bgr = cv2.cvtColor(np.ascontiguousarray(frame_rgb), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    low = np.array([0, _MIN_SATURATION, _MIN_VALUE], dtype=np.uint8)
    high = np.array([_HUE_HALF_WIDTH, 255, 255], dtype=np.uint8)
    wrapped_low = np.array([180 - _HUE_HALF_WIDTH, _MIN_SATURATION, _MIN_VALUE],
                           dtype=np.uint8)
    wrapped_high = np.array([180, 255, 255], dtype=np.uint8)
    mask = cv2.bitwise_or(cv2.inRange(hsv, low, high),
                          cv2.inRange(hsv, wrapped_low, wrapped_high))
    # Close only. An opening step would erase the pad at range, where it is a
    # few pixels tall, which is the distance the search phase needs it at.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)


def detect(frame_rgb: np.ndarray) -> Detection:
    """Detect the red landing target in *frame_rgb* (H×W×3, uint8, RGB).

    Returns a Detection with visible=False when no target is found.
    """
    if frame_rgb is None or frame_rgb.size == 0:
        return _NULL

    height, width = frame_rgb.shape[:2]
    mask = _red_mask(frame_rgb)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)

    best: tuple | None = None
    for index in range(1, count):
        area = float(stats[index, cv2.CC_STAT_AREA])
        if area < _MIN_AREA:
            continue
        box_x = int(stats[index, cv2.CC_STAT_LEFT])
        box_y = int(stats[index, cv2.CC_STAT_TOP])
        box_w = int(stats[index, cv2.CC_STAT_WIDTH])
        box_h = int(stats[index, cv2.CC_STAT_HEIGHT])
        if area / max(box_w * box_h, 1) < _MIN_EXTENT:
            continue
        # A pad running off the edge of the frame is a partial shape, so the
        # ellipse tests below cannot be applied to it. This happens on every
        # final descent, when the pad fills the bottom of the image.
        clipped = (box_x <= 0 or box_y <= 0
                   or box_x + box_w >= width or box_y + box_h >= height)

        cx, cy = centroids[index]
        radius = 0.5 * max(box_w, box_h)
        fill = aspect = None

        blob = (labels[box_y:box_y + box_h, box_x:box_x + box_w] == index)
        contours, _ = cv2.findContours(blob.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)
        contour = max(contours, key=cv2.contourArea) if contours else None
        if contour is not None and len(contour) >= 5:
            hull_area = cv2.contourArea(cv2.convexHull(contour))
            if hull_area > 0 and area / hull_area < _MIN_SOLIDITY:
                continue
            if area >= _SHAPE_MIN_AREA:
                (ex, ey), (d1, d2), _ = cv2.fitEllipse(contour)
                major, minor = max(d1, d2), min(d1, d2)
                if minor > 1e-6:
                    fill = area / (np.pi * 0.25 * major * minor)
                    aspect = major / minor
                    cx, cy, radius = box_x + ex, box_y + ey, 0.5 * major

        if aspect is not None and aspect > _MAX_ASPECT:
            continue
        if fill is not None and not clipped and not (_MIN_FILL <= fill <= _MAX_FILL):
            continue

        score = area * (min(fill, 1.0) if fill is not None else 1.0)
        if best is None or score > best[0]:
            best = (score, cx, cy, radius, fill, aspect)

    if best is None:
        return _NULL

    _, cx, cy, radius, fill, aspect = best
    # Confidence blends how cleanly the blob fills its ellipse with how oblique
    # the view is: a pad seen nearly edge-on is a real detection whose centre is
    # a poor position estimate, and the servo should weigh it accordingly.
    quality = 1.0 if fill is None else min(fill, 1.0)
    confidence = quality / max(1.0, (aspect or 1.0) / 4.0)
    return Detection(visible=True, cx=float(cx), cy=float(cy),
                     radius=float(radius),
                     confidence=float(min(1.0, max(0.0, confidence))))
