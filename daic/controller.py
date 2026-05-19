"""Proportional visual-servo velocity controller.

Takes a Detection and drone telemetry and produces a velocity command dict
ready to pass to encode_command().  All outputs are clamped to safe limits.

Coordinate convention (body frame, matches dsim velocity command):
  forward_mps  +  = nose direction
  right_mps    +  = starboard
  up_mps       +  = climb, - = descend
  yaw_rate_dps +  = clockwise from above

Two-phase approach with distance estimation
-------------------------------------------
The camera is forward-facing with a 5° downward tilt.  A naïve servo fails
for a ground target because the camera geometry makes cy_err unreliable as
a distance signal, and unthrottled descent fires too early.

Distance estimation
  The target's physical radius is known (TARGET_RADIUS_M).  Combined with
  the detected radius in pixels and the camera focal length, we can estimate
  the horizontal distance to the target at any point in the approach.

Phase 1 — approach (d_horiz > _APPROACH_GATE_M)
  Target is far.  Fly forward at a speed that decelerates proportionally as
  the distance closes so the drone arrives at the gate with low momentum.
  Lateral correction (cx_err) active; descent suppressed.

Phase 2 — overhead descent (d_horiz ≤ _APPROACH_GATE_M, cy_err ≥ 0)
  Target is close and below camera centre.  Coupled descent with a
  trajectory-intercept forward correction:
    forward = descent_rate × (d_horiz / altitude)
  This steers the drone along the straight line from its current position
  to the target, exactly compensating for the remaining horizontal offset.
  Without it the drone descends straight down and lands short.

The cy_err ≥ 0 guard keeps phase 1 active when the drone is at ground level
with the target still ahead (target appears above camera centre due to
shallow angle), preventing the drone from stalling before it reaches the
target.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .detector import Detection


# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

# Pixel-error gains (pre-scaled; dsim multiplies forward/right by 0.1).
_K_LATERAL = 0.047
_K_YAW     = 0.06   # suppressed when nearly centred

# Phase-1 maximum approach speed (pre-scaled).  dsim × 0.1 → actual m/s.
_APPROACH_FWD_SPEED = 8.0

# Distance (m) at which phase 1 → phase 2 transition occurs.
# Phase 2 also requires cy_err ≥ 0 (target below camera centre).
_APPROACH_GATE_M = 2.0

# Start decelerating in phase 1 when within this distance.
_DECEL_START_M = 4.0

# Minimum phase-1 forward speed (pre-scaled) — keeps the drone moving.
_APPROACH_MIN_SPEED = 1.5

# Phase-1 target position at which we start trading forward speed for descent.
_BOTTOM_SLOW_START = 0.35
_BOTTOM_SLOW_FULL  = 0.85
_BOTTOM_DESCENT_RATE = -0.6

# Descent rates (m/s, NOT pre-scaled — dsim uses cmd_up directly).
_DESCENT_RATE_FAST = -4.0   # when target is small within phase 2
_DESCENT_RATE_SLOW = -1.8   # when target fills >25% of frame height

# Fade radius for coupled descent (normalised centre distance).
_DESCENT_FADE_RADIUS = 0.55

# Altitude at which horizontal lateral gain is at full strength.
_REF_ALT_M = 2.5

# Floor for altitude-based lateral gain reduction.
_MIN_ALT_SCALE = 0.25

# Search / transit speed (pre-scaled).
_SEARCH_SPEED = 5.0

# Hard command limits (pre-scaled unless noted).
_MAX_FORWARD =  12.0
_MAX_BACK    = -12.0
_MAX_RIGHT   =  10.0
_MAX_LEFT    = -10.0
_MAX_UP      =   6.0
_MAX_DOWN    =  -8.0
_MAX_YAW     =  45.0

# Minimum detector confidence before servo activates.
_MIN_CONFIDENCE = 0.35

# Physical radius of the landing target marker (metres).
_TARGET_RADIUS_M = 0.36

# Horizontal camera field of view (degrees) — matches dsim CAM_FOV.
_FOV_H_DEG = 70.0


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _focal_length_px(img_w: int) -> float:
    """Pinhole focal length in pixels for the configured horizontal FOV."""
    return img_w / (2.0 * math.tan(math.radians(_FOV_H_DEG / 2.0)))


# Approximate vertical half-FOV in degrees (70° horizontal, 4:3 frame).
_FOV_V_HALF_DEG = 26.5

# Downward camera pitch in degrees (matches dsim CAM_PITCH).
_CAM_PITCH_DEG = 5.0


def estimate_horiz_dist(radius_px: float, altitude_m: float,
                        img_w: int) -> float:
    """Estimate horizontal distance to target from apparent radius and altitude.

    Uses the pinhole model: d_3d = R_m × f_px / r_px, then subtracts altitude
    to get the horizontal component.

    Returns 0.0 when the target is too close/large for the estimate to be
    valid (d_3d < altitude).  Returns a large sentinel when radius is tiny.
    """
    if radius_px < 1.0:
        return 999.0
    d_3d = _TARGET_RADIUS_M * _focal_length_px(img_w) / radius_px
    return math.sqrt(max(0.0, d_3d * d_3d - altitude_m * altitude_m))


@dataclass
class ControlOutput:
    forward_mps:   float = 0.0
    right_mps:     float = 0.0
    up_mps:        float = 0.0
    yaw_rate_dps:  float = 0.0
    descending:    bool  = False
    horiz_dist_m:  float = 0.0   # estimated horizontal distance to target (informational)

    def as_command_fields(self) -> dict:
        return {
            "forward_mps":  self.forward_mps,
            "right_mps":    self.right_mps,
            "up_mps":       self.up_mps,
            "yaw_rate_dps": self.yaw_rate_dps,
        }


def hover() -> ControlOutput:
    return ControlOutput()


def servo(detection: Detection,
          img_w: int, img_h: int,
          altitude_m: float,
          d_horiz_override: float | None = None) -> ControlOutput:
    """Visual-servo toward the detected target.

    Returns a hover command when confidence is too low.
    """
    if not detection.visible or detection.confidence < _MIN_CONFIDENCE:
        return hover()

    img_cx = img_w / 2.0
    img_cy = img_h / 2.0

    cx_err = detection.cx - img_cx   # + = target right of centre
    cy_err = detection.cy - img_cy   # + = target below centre

    cx_norm = cx_err / max(img_cx, 1.0)
    cy_norm = cy_err / max(img_cy, 1.0)

    size_frac = (detection.radius * 2.0) / max(img_h, 1.0)

    alt_scale = max(_MIN_ALT_SCALE, min(1.0, altitude_m / _REF_ALT_M))
    yaw = (_clamp(_K_YAW * cx_err, -_MAX_YAW, _MAX_YAW)
           if abs(cx_norm) > 0.12 else 0.0)

    # Estimate horizontal distance; use planner-supplied override when the
    # radius estimate saturates (target too close, d_3d < altitude).
    d_horiz = estimate_horiz_dist(detection.radius, altitude_m, img_w)
    if d_horiz == 0.0 and d_horiz_override is not None:
        d_horiz = d_horiz_override

    # ── Phase selection ───────────────────────────────────────────────
    # Phase 2 requires the target to be close (d_horiz ≤ gate) AND appearing
    # below camera centre (cy_err ≥ 0).  The cy_err guard keeps the drone
    # flying forward when at ground level with the target still ahead — at
    # that point the target appears above the downward-aimed camera centre.
    in_phase2 = d_horiz <= _APPROACH_GATE_M and cy_err >= 0

    # ── Phase 1: approach with distance-proportional deceleration ─────
    if not in_phase2:
        # Decelerate linearly from full speed at _DECEL_START_M to minimum
        # at the gate, so the drone arrives with low residual momentum.
        if d_horiz < _DECEL_START_M:
            t = max(0.0, (d_horiz - _APPROACH_GATE_M) / (_DECEL_START_M - _APPROACH_GATE_M))
            fwd_speed = _APPROACH_MIN_SPEED + t * (_APPROACH_FWD_SPEED - _APPROACH_MIN_SPEED)
        else:
            fwd_speed = _APPROACH_FWD_SPEED
        bottom_pressure = _clamp(
            (cy_norm - _BOTTOM_SLOW_START) / (_BOTTOM_SLOW_FULL - _BOTTOM_SLOW_START),
            0.0,
            1.0,
        )
        if bottom_pressure > 0.0:
            fwd_speed *= 1.0 - 0.35 * bottom_pressure
        right = _clamp(_K_LATERAL * cx_err * alt_scale, _MAX_LEFT, _MAX_RIGHT)
        up = (_BOTTOM_DESCENT_RATE * bottom_pressure
              if altitude_m > 1.2 and abs(cx_norm) < 0.30 else 0.0)
        return ControlOutput(forward_mps=fwd_speed, right_mps=right,
                             up_mps=up, yaw_rate_dps=yaw,
                             descending=False, horiz_dist_m=d_horiz)

    # ── Phase 2: trajectory-intercept descent ─────────────────────────
    size_scale  = max(0.25, 1.0 - size_frac * 1.8)
    horiz_scale = alt_scale * size_scale
    horiz_limit = max(6.0, _MAX_FORWARD * horiz_scale)

    right = _clamp(_K_LATERAL * cx_err * horiz_scale, -horiz_limit, horiz_limit)

    # Coupled descent.
    # For a forward-facing camera, vertical image error mostly encodes target
    # range.  Do not block descent just because the ground target is low in the
    # frame; gate descent on lateral centring instead.
    centre_factor = max(0.0, 1.0 - abs(cx_norm) / _DESCENT_FADE_RADIUS)
    base_rate     = _DESCENT_RATE_SLOW if size_frac > 0.25 else _DESCENT_RATE_FAST
    up = base_rate * centre_factor if altitude_m > 0.3 else 0.0
    up = _clamp(up, _MAX_DOWN, _MAX_UP)

    # Do not descend faster than the available forward speed can intercept.
    # Forward/right commands are scaled by dsim (x0.1), while cmd_up is not.
    if up < 0.0 and altitude_m > 0.1 and d_horiz > 0.0:
        max_forward_actual = max(0.0, horiz_limit - abs(right)) * 0.1
        max_descent_for_intercept = max_forward_actual * altitude_m / d_horiz
        up = -min(abs(up), max_descent_for_intercept)

    # Trajectory-intercept forward correction.
    # The drone needs to cover d_horiz horizontal while descending altitude_m
    # vertical.  Adding forward = |descent| × (d_horiz / altitude) steers the
    # drone along the straight line to the target rather than straight down,
    # eliminating the "lands short" offset.
    # NOTE: cmd_up is actual m/s; forward is pre-scaled (dsim × 0.1 → actual).
    if altitude_m > 0.1 and d_horiz > 0.0:
        actual_descent = abs(up)   # m/s actual (cmd_up is not scaled)
        intercept_actual = actual_descent * d_horiz / altitude_m
        # Convert actual → pre-scaled (dsim applies × 0.1 to forward/right).
        forward = _clamp(intercept_actual / 0.1, 0.0, horiz_limit)
    else:
        forward = 0.0

    return ControlOutput(forward_mps=forward, right_mps=right,
                         up_mps=up, yaw_rate_dps=yaw,
                         descending=up < -0.1, horiz_dist_m=d_horiz)


# GPS navigation constants (pre-scaled where noted; dsim × 0.1 for forward/right)
_GPS_YAW_GAIN     = 0.8    # deg/s per degree of yaw error
_GPS_MAX_YAW_DPS  = 35.0
_GPS_ALIGN_DEG    = 25.0   # start forward motion below this yaw error
_GPS_NAV_SPEED    = 6.0    # pre-scaled full-speed
_GPS_NAV_MIN      = 2.0    # pre-scaled minimum speed (close to target)
_GPS_DECEL_DIST_M = 12.0   # distance at which deceleration begins


def navigate_to_bearing(yaw_error_deg: float, dist_m: float) -> ControlOutput:
    """Fly toward a GPS bearing.

    yaw_error_deg: target_bearing minus drone_compass_heading, normalized to
                   [-180, 180]. Positive = target is clockwise (right turn needed).
    dist_m:        metres to target.
    """
    yaw_error_deg = (yaw_error_deg + 180.0) % 360.0 - 180.0
    yaw = _clamp(yaw_error_deg * _GPS_YAW_GAIN, -_GPS_MAX_YAW_DPS, _GPS_MAX_YAW_DPS)
    if abs(yaw_error_deg) < _GPS_ALIGN_DEG:
        t = _clamp(dist_m / _GPS_DECEL_DIST_M, 0.0, 1.0)
        fwd = _GPS_NAV_MIN + t * (_GPS_NAV_SPEED - _GPS_NAV_MIN)
    else:
        fwd = 0.0
    return ControlOutput(forward_mps=fwd, yaw_rate_dps=yaw)


def search_step(heading_deg: float, speed: float = _SEARCH_SPEED) -> ControlOutput:
    """Move forward at search speed (caller handles heading via yaw commands)."""
    return ControlOutput(forward_mps=_clamp(speed, 0.0, _MAX_FORWARD))


def turn(yaw_rate_dps: float) -> ControlOutput:
    return ControlOutput(yaw_rate_dps=_clamp(yaw_rate_dps, -_MAX_YAW, _MAX_YAW))
