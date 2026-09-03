"""Optical-flow obstacle risk for daic.

The detector estimates image-space expansion between consecutive RGB frames.
When the drone moves toward a nearby wall, texture flows radially outward from
the image centre; this module converts that expansion into the same sector-risk
shape used by the SLAM obstacle detectors.
"""

from __future__ import annotations

import math
import time
from collections import deque
from typing import Any

import numpy as np

from daic.orb_slam3_detector import ObstacleSectors, _NULL_SECTORS


_FLOW_W = 160
_FLOW_H = 120
_MIN_FLOW_MAG_PX = 0.08
_START_EXPANSION_PX = 0.18
_FULL_EXPANSION_PX = 1.60
_MIN_VALID_PIXELS = 120
_CONF_FULL_PIXELS = 1600
_PERSIST_DECAY = 0.86
_PERSIST_CONF_DECAY = 0.92
_MIN_RANGE_SPEED_MPS = 0.06
_MIN_TTC_RATIO = 0.002
_MIN_RANGE_M = 0.6
_MAX_RANGE_M = 8.0
# Percentile of the per-pixel divergence used as the sector's divergence.
# The sector images a mix of surfaces; the *nearest* one has the largest
# divergence, and that is the one obstacle avoidance cares about, so this is a
# robust max rather than a central tendency. Calibrated against dsim ground
# truth on a straight-in wall approach at 0.35/0.6/1.0 m/s: the estimate is
# near-unbiased at contact range (median est/true 1.08 within 1 m) and
# conservatively short beyond it (0.86 at 1-2 m, 0.75 at 2-3 m). That asymmetry
# is deliberate - reading a wall as nearer than it is brakes early, reading it
# as further drives into it. Lower percentiles bias long and dangerous: the
# original 45th read 1.41x too far inside 1 m.
_DIVERGENCE_PCT = 98.0
# Below this the reading is more often the floor parallaxing beneath the
# pitched-forward camera than a navigable obstacle. Kept well under the
# calibrated contact-range accuracy so a genuine close wall still maps.
_MIN_PLAUSIBLE_RANGE_M = 0.6

# Per-sector temporal median over the range estimate. The per-tick estimate is
# accurate in the mean but scatters by roughly +/-50% frame to frame, which is
# enough that consecutive marks land in different grid cells and the local map's
# flow-confirmation window (which needs repeat hits on one cell) never
# confirms - so an accurate estimator maps *less* of the wall than a broken one
# that returned a constant. Distance to a wall changes smoothly, so a short
# median is well justified: at 30 fps this spans 0.23 s, about 0.14 m of travel
# at cruise, while cutting the cell-to-cell jitter that blocks confirmation.
_RANGE_MEDIAN_TICKS = 7
# Drop the history for a sector that has gone this long without a fresh
# reading, so a stale distance cannot be revived after the drone has moved on.
_RANGE_HISTORY_GAP_TICKS = 10

_SECTOR_RANGE_ATTRS = (
    "front_range_m", "front_left_range_m", "front_right_range_m",
    "left_range_m", "right_range_m",
)


class OpticalFlowAvoidance:
    """Dense optical-flow expansion detector."""

    def __init__(self) -> None:
        self._prev_gray: np.ndarray | None = None
        self._prev_t: float | None = None
        self._forward_speed_mps = 0.0
        self._persisted = _NULL_SECTORS
        self._status_text = "initialising"
        self._range_hist: dict[str, deque] = {
            attr: deque(maxlen=_RANGE_MEDIAN_TICKS) for attr in _SECTOR_RANGE_ATTRS
        }
        self._range_last_tick: dict[str, int] = {}
        self._tick = 0

    @property
    def status_text(self) -> str:
        return self._status_text

    def reset(self) -> None:
        self._prev_gray = None
        self._prev_t = None
        self._persisted = _NULL_SECTORS
        self._status_text = "reset"
        for hist in self._range_hist.values():
            hist.clear()
        self._range_last_tick.clear()

    def set_motion_from_status(self, status: dict[str, Any]) -> None:
        """Update body-forward speed used to turn TTC into range."""
        vx = _try_float(status.get("drone.vx_mps"))
        vy = _try_float(status.get("drone.vy_mps"))
        yaw = _try_float(status.get("drone.heading_deg"))
        if None in (vx, vy, yaw):
            self._forward_speed_mps = 0.0
            return
        self._forward_speed_mps = _body_forward_speed(
            float(vx), float(vy), float(yaw),
        )

    def _smooth_ranges(self, sectors: ObstacleSectors) -> ObstacleSectors:
        """Replace each sector range with a short running median.

        See _RANGE_MEDIAN_TICKS: the raw per-tick estimate is accurate in the
        mean but too jittery for the local map to confirm a cell.
        """
        self._tick += 1
        smoothed: dict[str, float | None] = {}
        for attr in _SECTOR_RANGE_ATTRS:
            hist = self._range_hist[attr]
            last = self._range_last_tick.get(attr)
            if last is not None and self._tick - last > _RANGE_HISTORY_GAP_TICKS:
                hist.clear()
            value = getattr(sectors, attr, None)
            if value is not None and math.isfinite(float(value)):
                hist.append(float(value))
                self._range_last_tick[attr] = self._tick
            smoothed[attr] = _median(list(hist)) if hist else None
        return ObstacleSectors(
            front=sectors.front,
            front_left=sectors.front_left,
            front_right=sectors.front_right,
            left=sectors.left,
            right=sectors.right,
            confidence=sectors.confidence,
            method=sectors.method,
            **smoothed,
        )

    def detect_obstacles(self, frame_rgb: np.ndarray) -> ObstacleSectors:
        now = time.monotonic()
        gray = _to_gray_small(frame_rgb)
        if self._prev_gray is None:
            self._prev_gray = gray
            self._prev_t = now
            self._status_text = "priming"
            return self._persisted

        prev = self._prev_gray
        prev_t = self._prev_t
        self._prev_gray = gray
        self._prev_t = now

        cv2 = _cv2()
        flow = cv2.calcOpticalFlowFarneback(
            prev,
            gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=17,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        dt_s = (now - prev_t) if prev_t is not None else None
        instant = _flow_to_sectors(flow, self._forward_speed_mps, dt_s)
        instant = self._smooth_ranges(instant)
        sectors = _persist_sectors(self._persisted, instant)
        self._persisted = sectors
        self._status_text = (
            f"{sectors.method} f={sectors.front:.2f} "
            f"l={max(sectors.front_left, sectors.left):.2f} "
            f"r={max(sectors.front_right, sectors.right):.2f}"
        )
        return sectors


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def fuse_obstacle_sectors(*items: ObstacleSectors) -> ObstacleSectors:
    """Combine independent obstacle estimates by taking max risk per sector."""
    valid = [s for s in items if s.confidence > 0.0]
    if not valid:
        return _NULL_SECTORS
    method = "+".join(s.method for s in valid if s.method and s.method != "none")
    return ObstacleSectors(
        front=max(s.front for s in valid),
        front_left=max(s.front_left for s in valid),
        front_right=max(s.front_right for s in valid),
        left=max(s.left for s in valid),
        right=max(s.right for s in valid),
        confidence=max(s.confidence for s in valid),
        method=method or "vision",
        front_range_m=_nearest_range(valid, "front_range_m"),
        front_left_range_m=_nearest_range(valid, "front_left_range_m"),
        front_right_range_m=_nearest_range(valid, "front_right_range_m"),
        left_range_m=_nearest_range(valid, "left_range_m"),
        right_range_m=_nearest_range(valid, "right_range_m"),
    )


def _persist_sectors(previous: ObstacleSectors,
                     current: ObstacleSectors) -> ObstacleSectors:
    """Keep recent obstacle risk alive through intermittent flow dropouts."""
    decayed = ObstacleSectors(
        front=previous.front * _PERSIST_DECAY,
        front_left=previous.front_left * _PERSIST_DECAY,
        front_right=previous.front_right * _PERSIST_DECAY,
        left=previous.left * _PERSIST_DECAY,
        right=previous.right * _PERSIST_DECAY,
        confidence=previous.confidence * _PERSIST_CONF_DECAY,
        method="flow:persist",
        # Persisted risk is a fading memory, not a fresh measurement — carrying
        # its range estimate forward verbatim lets a single stale close-range
        # reading "stick" to the drone indefinitely (fuse_obstacle_sectors
        # always prefers the nearest of the valid ranges). Only a live
        # detection should claim a distance; persistence keeps risk alive
        # through brief dropouts without pretending to know how far away it is.
        front_range_m=None,
        front_left_range_m=None,
        front_right_range_m=None,
        left_range_m=None,
        right_range_m=None,
    )
    fused = fuse_obstacle_sectors(decayed, current)
    if fused.confidence < 0.05:
        return _NULL_SECTORS
    if current.confidence > 0.0 and decayed.confidence > 0.0:
        method = f"{current.method}+persist"
    else:
        method = fused.method
    return ObstacleSectors(
        front=fused.front,
        front_left=fused.front_left,
        front_right=fused.front_right,
        left=fused.left,
        right=fused.right,
        confidence=fused.confidence,
        method=method,
        front_range_m=fused.front_range_m,
        front_left_range_m=fused.front_left_range_m,
        front_right_range_m=fused.front_right_range_m,
        left_range_m=fused.left_range_m,
        right_range_m=fused.right_range_m,
    )


def _to_gray_small(frame_rgb: np.ndarray) -> np.ndarray:
    cv2 = _cv2()
    bgr = np.ascontiguousarray(frame_rgb[:, :, ::-1])
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (_FLOW_W, _FLOW_H), interpolation=cv2.INTER_AREA)


def _cv2():
    import cv2

    return cv2


def _flow_to_sectors(flow: np.ndarray, forward_speed_mps: float = 0.0,
                     dt_s: float | None = None) -> ObstacleSectors:
    # Optical "expansion" toward an obstacle requires the camera to translate
    # toward the scene. Yaw-scanning (forward_speed_mps ~= 0, the dominant
    # SEARCH-state motion) sweeps the off-axis-mounted camera through a small
    # arc, and the nearby floor parallaxes hard against the distant scene --
    # producing flow with the same radially-symmetric, ~zero-median signature
    # as genuine expansion (median subtraction can't tell them apart). Gating
    # on forward speed is the only reliable split: a benchmark run showed front
    # risk reads 0.86 median while forward_speed_mps < 0.06 (floor-parallax
    # noise) vs. 0.14 median once it isn't (real signal). `_persist_sectors`
    # still carries a genuine close-range reading through brief stops.
    if forward_speed_mps < _MIN_RANGE_SPEED_MPS:
        return ObstacleSectors(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "flow:no_translation")

    h, w = flow.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    dx = xx - cx
    dy = yy - cy
    radius = np.sqrt(dx * dx + dy * dy) + 1.0

    fx = flow[:, :, 0]
    fy = flow[:, :, 1]
    mag = np.sqrt(fx * fx + fy * fy)
    expansion = (fx * dx + fy * dy) / radius

    valid = mag >= _MIN_FLOW_MAG_PX
    if int(valid.sum()) < _MIN_VALID_PIXELS:
        return ObstacleSectors(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "flow:low_motion")

    confidence = min(1.0, float(valid.sum()) / _CONF_FULL_PIXELS)

    x_norm = xx / max(float(w - 1), 1.0)
    y_norm = yy / max(float(h - 1), 1.0)
    # Ignore the very top and bottom of the image. The bottom is mostly floor in
    # dsim's pitched-forward camera and otherwise dominates TTC as a permanently
    # close surface, which makes walls at different ranges collapse together.
    roi = valid & (y_norm > 0.18) & (y_norm < 0.52)

    range_scale_m = _range_scale_m(forward_speed_mps, dt_s)
    ratio = expansion / radius

    front_mask = roi & (x_norm >= 0.35) & (x_norm <= 0.65)
    front_left_mask = roi & (x_norm >= 0.18) & (x_norm < 0.45)
    front_right_mask = roi & (x_norm > 0.55) & (x_norm <= 0.82)
    left_mask = roi & (x_norm < 0.28)
    right_mask = roi & (x_norm > 0.72)

    front, front_range = _sector_metrics(expansion, ratio, front_mask, range_scale_m)
    front_left, front_left_range = _sector_metrics(expansion, ratio, front_left_mask, range_scale_m)
    front_right, front_right_range = _sector_metrics(expansion, ratio, front_right_mask, range_scale_m)
    left, left_range = _sector_metrics(expansion, ratio, left_mask, range_scale_m)
    right, right_range = _sector_metrics(expansion, ratio, right_mask, range_scale_m)

    return ObstacleSectors(
        front=front,
        front_left=front_left,
        front_right=front_right,
        left=left,
        right=right,
        confidence=confidence,
        method="flow:expansion",
        front_range_m=front_range,
        front_left_range_m=front_left_range,
        front_right_range_m=front_right_range,
        left_range_m=left_range,
        right_range_m=right_range,
    )


def _sector_risk(expansion: np.ndarray, mask: np.ndarray) -> float:
    vals = expansion[mask]
    vals = vals[vals > 0.0]
    if vals.size < 25:
        return 0.0
    # Use upper-percentile expansion so a textured obstacle region can trigger
    # without requiring the whole sector to move coherently.
    score = float(np.percentile(vals, 85))
    risk = (score - _START_EXPANSION_PX) / (_FULL_EXPANSION_PX - _START_EXPANSION_PX)
    return float(np.clip(risk, 0.0, 1.0))


def _sector_metrics(expansion: np.ndarray, ratio: np.ndarray,
                    mask: np.ndarray,
                    range_scale_m: float | None) -> tuple[float, float | None]:
    risk = _sector_risk(expansion, mask)
    if range_scale_m is None or risk <= 0.0:
        return risk, None

    vals = ratio[mask]
    vals = vals[vals >= _MIN_TTC_RATIO]
    if vals.size < 25:
        return risk, None

    divergence = float(np.percentile(vals, _DIVERGENCE_PCT))
    if divergence < _MIN_TTC_RATIO:
        return risk, None
    # Radial flow from pure translation toward a surface at distance Z is
    # f_r = r * (V/Z) * dt, so ratio = f_r / r = V*dt/Z and Z = V*dt / ratio.
    # range_scale_m is V*dt, so this is the range directly - it needs no
    # tuning gain. An earlier 8x multiplier here drove every reading into the
    # _MAX_RANGE_M clamp, so the map reported a constant 8 m all the way to
    # impact; see the calibration note on _DIVERGENCE_PCT.
    raw_range_m = range_scale_m / divergence
    # Judge plausibility on the raw measurement, before clamping. Clamping
    # first would lift an implausible reading up to _MIN_RANGE_M and, whenever
    # that floor is >= _MIN_PLAUSIBLE_RANGE_M, the guard below could never fire.
    if raw_range_m < _MIN_PLAUSIBLE_RANGE_M:
        # A TTC range this short is far more often the floor parallaxing
        # close beneath the pitched-forward camera during genuine forward
        # translation (the same near-field surface _flow_to_sectors's ROI
        # comment names as a "permanently close surface") than a real
        # navigable obstacle -- anything that close to a route-relevant wall
        # would already be inside direct collision-braking range. Keep the
        # risk (it still drives avoidance) but drop the implausible distance
        # so the local map falls back to its conservative default instead of
        # planting a phantom wall in the drone's own grid cell.
        return risk, None
    return risk, float(np.clip(raw_range_m, _MIN_RANGE_M, _MAX_RANGE_M))


def _range_scale_m(forward_speed_mps: float, dt_s: float | None) -> float | None:
    if dt_s is None or dt_s <= 0.0:
        return None
    if forward_speed_mps < _MIN_RANGE_SPEED_MPS:
        return None
    return forward_speed_mps * dt_s


def _body_forward_speed(vx_mps: float, vy_mps: float,
                        heading_deg: float) -> float:
    """Project map velocity onto dsim's compass-style heading vector."""
    heading_rad = math.radians(heading_deg)
    fwd_x = math.sin(heading_rad)
    fwd_y = -math.cos(heading_rad)
    return vx_mps * fwd_x + vy_mps * fwd_y


def _nearest_range(items: list[ObstacleSectors], attr: str) -> float | None:
    values = [
        float(v) for v in (getattr(item, attr) for item in items)
        if v is not None and math.isfinite(float(v))
    ]
    if not values:
        return None
    return min(values)


def _try_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
