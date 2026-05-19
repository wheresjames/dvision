"""Optical-flow obstacle risk for daic.

The detector estimates image-space expansion between consecutive RGB frames.
When the drone moves toward a nearby wall, texture flows radially outward from
the image centre; this module converts that expansion into the same sector-risk
shape used by the SLAM obstacle detectors.
"""

from __future__ import annotations

import math
import time
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
_TTC_RANGE_GAIN = 8.0


class OpticalFlowAvoidance:
    """Dense optical-flow expansion detector."""

    def __init__(self) -> None:
        self._prev_gray: np.ndarray | None = None
        self._prev_t: float | None = None
        self._forward_speed_mps = 0.0
        self._persisted = _NULL_SECTORS
        self._status_text = "initialising"

    @property
    def status_text(self) -> str:
        return self._status_text

    def reset(self) -> None:
        self._prev_gray = None
        self._prev_t = None
        self._persisted = _NULL_SECTORS
        self._status_text = "reset"

    def set_motion_from_status(self, status: dict[str, Any]) -> None:
        """Update body-forward speed used to turn TTC into range."""
        vx = _try_float(status.get("drone.vx_mps"))
        vy = _try_float(status.get("drone.vy_mps"))
        yaw = _try_float(status.get("drone.heading_deg"))
        if None in (vx, vy, yaw):
            self._forward_speed_mps = 0.0
            return
        yaw_rad = math.radians(float(yaw))
        self._forward_speed_mps = (
            float(vx) * math.cos(yaw_rad) + float(vy) * math.sin(yaw_rad)
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
        sectors = _persist_sectors(self._persisted, instant)
        self._persisted = sectors
        self._status_text = (
            f"{sectors.method} f={sectors.front:.2f} "
            f"l={max(sectors.front_left, sectors.left):.2f} "
            f"r={max(sectors.front_right, sectors.right):.2f}"
        )
        return sectors


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
        front_range_m=previous.front_range_m,
        front_left_range_m=previous.front_left_range_m,
        front_right_range_m=previous.front_right_range_m,
        left_range_m=previous.left_range_m,
        right_range_m=previous.right_range_m,
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

    strong_ratio = float(np.percentile(vals, 45))
    if strong_ratio < _MIN_TTC_RATIO:
        return risk, None
    range_m = (range_scale_m / strong_ratio) * _TTC_RANGE_GAIN
    return risk, float(np.clip(range_m, _MIN_RANGE_M, _MAX_RANGE_M))


def _range_scale_m(forward_speed_mps: float, dt_s: float | None) -> float | None:
    if dt_s is None or dt_s <= 0.0:
        return None
    if forward_speed_mps < _MIN_RANGE_SPEED_MPS:
        return None
    return forward_speed_mps * dt_s


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
