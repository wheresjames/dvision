"""Tests for daic.controller — pure logic, no I/O."""

import math

import pytest

from daic.controller import (
    ControlOutput, hover, servo, search_step, turn,
    estimate_horiz_dist,
    _MAX_FORWARD, _MAX_BACK, _MAX_RIGHT, _MAX_LEFT,
    _MAX_UP, _MAX_DOWN, _MAX_YAW, _MIN_CONFIDENCE,
    _DESCENT_RATE_FAST, _DESCENT_RATE_SLOW,
    _APPROACH_FWD_SPEED, _APPROACH_MIN_SPEED,
    _APPROACH_GATE_M, _DECEL_START_M,
    _BOTTOM_DESCENT_RATE,
    _TARGET_RADIUS_M, _FOV_H_DEG,
)
from daic.detector import Detection


IMG_W, IMG_H = 640, 480


# ---------------------------------------------------------------------------
# Helper: build detections at a known physical distance
# ---------------------------------------------------------------------------

def _focal_px(w=IMG_W):
    return w / (2.0 * math.tan(math.radians(_FOV_H_DEG / 2.0)))


def _radius_for_dist(d_3d: float, img_w: int = IMG_W) -> float:
    """Pixel radius that corresponds to a given 3D distance."""
    return _TARGET_RADIUS_M * _focal_px(img_w) / d_3d


def _det_at_dist(d_horiz: float, alt: float, cx=320.0, cy=300.0, conf=0.85):
    """Build a Detection whose radius encodes the given horizontal distance."""
    d_3d = math.hypot(d_horiz, alt)
    r = _radius_for_dist(d_3d)
    return Detection(visible=True, cx=cx, cy=cy, radius=r, confidence=conf)


def _det(visible=True, cx=320.0, cy=300.0, radius=40.0, confidence=0.8):
    """Quick detection helper; cy=300 (below centre) so the descent can fire."""
    return Detection(visible=visible, cx=cx, cy=cy,
                     radius=radius, confidence=confidence)


# ---------------------------------------------------------------------------
# Distance estimation
# ---------------------------------------------------------------------------

def test_estimate_horiz_dist_close():
    """At 1 m horizontal, 1 m altitude → ~1 m horiz."""
    r = _radius_for_dist(math.hypot(1.0, 1.0))
    d = estimate_horiz_dist(r, 1.0, IMG_W)
    assert abs(d - 1.0) < 0.2


def test_estimate_horiz_dist_far():
    """At 5 m horizontal, 3 m altitude → ~5 m horiz."""
    r = _radius_for_dist(math.hypot(5.0, 3.0))
    d = estimate_horiz_dist(r, 3.0, IMG_W)
    assert abs(d - 5.0) < 0.5


def test_estimate_horiz_dist_tiny_radius():
    """Near-zero radius returns a large distance, not a crash."""
    d = estimate_horiz_dist(0.0, 2.0, IMG_W)
    assert d > 100.0


# ---------------------------------------------------------------------------
# hover()
# ---------------------------------------------------------------------------

def test_hover_is_zero():
    out = hover()
    assert out.forward_mps  == 0.0
    assert out.right_mps    == 0.0
    assert out.up_mps       == 0.0
    assert out.yaw_rate_dps == 0.0


# ---------------------------------------------------------------------------
# Approach
# ---------------------------------------------------------------------------

def test_approach_far_target_full_speed():
    """Far target → full approach speed, no descent."""
    det = _det_at_dist(d_horiz=8.0, alt=3.0)
    out = servo(det, IMG_W, IMG_H, altitude_m=3.0)
    assert out.forward_mps == pytest.approx(_APPROACH_FWD_SPEED)
    assert out.up_mps == 0.0
    assert not out.descending


def test_approach_decelerates_as_target_approaches():
    """Speed should drop as horizontal distance shrinks toward gate."""
    det_far  = _det_at_dist(d_horiz=_DECEL_START_M + 1.0, alt=3.0)
    det_near = _det_at_dist(d_horiz=_APPROACH_GATE_M + 0.2, alt=3.0)
    out_far  = servo(det_far,  IMG_W, IMG_H, altitude_m=3.0)
    out_near = servo(det_near, IMG_W, IMG_H, altitude_m=3.0)
    assert out_far.forward_mps > out_near.forward_mps


def test_approach_minimum_speed_at_gate():
    """Just outside the gate the speed is clamped to the minimum."""
    det = _det_at_dist(d_horiz=_APPROACH_GATE_M + 0.05, alt=3.0)
    out = servo(det, IMG_W, IMG_H, altitude_m=3.0)
    assert out.forward_mps >= _APPROACH_MIN_SPEED - 0.1
    assert out.up_mps == 0.0


def test_approach_lateral_correction_right():
    """Target right of centre → move right during the approach."""
    det = _det_at_dist(d_horiz=5.0, alt=3.0, cx=420.0)
    out = servo(det, IMG_W, IMG_H, altitude_m=3.0)
    assert out.right_mps > 0.0
    assert out.yaw_rate_dps > 0.0


def test_approach_lateral_correction_left():
    det = _det_at_dist(d_horiz=5.0, alt=3.0, cx=200.0)
    out = servo(det, IMG_W, IMG_H, altitude_m=3.0)
    assert out.right_mps < 0.0
    assert out.yaw_rate_dps < 0.0


def test_approach_target_above_centre_keeps_flying():
    """The approach stays active when the target is above camera centre
    (cy_err < 0)."""
    det = _det_at_dist(d_horiz=_APPROACH_GATE_M - 0.1, alt=3.0, cy=100.0)
    out = servo(det, IMG_W, IMG_H, altitude_m=3.0)
    assert out.forward_mps >= _APPROACH_MIN_SPEED
    assert out.up_mps == 0.0


def test_approach_target_near_bottom_slows_and_descends():
    """A low-in-frame target should not be overflown out of view."""
    det = _det_at_dist(d_horiz=5.0, alt=3.0, cy=465.0)
    out = servo(det, IMG_W, IMG_H, altitude_m=3.0)
    assert 0.0 < out.forward_mps < _APPROACH_FWD_SPEED
    assert out.up_mps == pytest.approx(_BOTTOM_DESCENT_RATE)
    assert not out.descending


# ---------------------------------------------------------------------------
# Overhead descent
# ---------------------------------------------------------------------------

def _descent_det(alt=2.0, cx=320.0, cy=300.0):
    """Detection that puts the drone inside the descent gate."""
    return _det_at_dist(d_horiz=_APPROACH_GATE_M - 0.5, alt=alt, cx=cx, cy=cy)


def test_descent_centred_target_descends():
    out = servo(_descent_det(), IMG_W, IMG_H, altitude_m=2.0)
    assert out.up_mps < 0.0
    assert out.descending


def test_descent_above_centre_keeps_approaching():
    """Large close target but above camera centre → keep approaching."""
    det = _det_at_dist(d_horiz=_APPROACH_GATE_M - 0.5, alt=2.0, cy=100.0)
    out = servo(det, IMG_W, IMG_H, altitude_m=2.0)
    assert out.forward_mps >= _APPROACH_MIN_SPEED
    assert out.up_mps == 0.0


def test_descent_at_ground_no_descent():
    """Below altitude gate (alt ≤ 0.3 m) descent is suppressed."""
    out = servo(_descent_det(alt=0.2), IMG_W, IMG_H, altitude_m=0.2)
    assert out.up_mps == 0.0


def test_descent_trajectory_correction_forward():
    """The descent adds forward to intercept the target; > 0 with d_horiz > 0."""
    out = servo(_descent_det(alt=2.0), IMG_W, IMG_H, altitude_m=2.0)
    assert out.forward_mps > 0.0


def test_descent_respects_available_forward_speed():
    det = _det_at_dist(d_horiz=_APPROACH_GATE_M - 0.1, alt=1.0)
    out = servo(det, IMG_W, IMG_H, altitude_m=1.0)
    forward_actual = out.forward_mps
    assert abs(out.up_mps) <= forward_actual * 1.0 / out.horiz_dist_m + 1e-6


def test_descent_trajectory_closer_means_less_forward():
    """Closer target (smaller d_horiz) → less forward correction needed."""
    det_far   = _det_at_dist(d_horiz=1.5, alt=2.0)
    det_close = _det_at_dist(d_horiz=0.2, alt=2.0)
    out_far   = servo(det_far,   IMG_W, IMG_H, altitude_m=2.0)
    out_close = servo(det_close, IMG_W, IMG_H, altitude_m=2.0)
    assert out_far.forward_mps > out_close.forward_mps


def test_descent_off_centre_partial_descent():
    """Off-centre but within fade radius: partial descent."""
    out_centre = servo(_descent_det(cx=320), IMG_W, IMG_H, altitude_m=2.0)
    out_offset = servo(_descent_det(cx=390), IMG_W, IMG_H, altitude_m=2.0)
    assert out_offset.up_mps > out_centre.up_mps


def test_descent_low_in_frame_still_descends_when_laterally_centred():
    det = _det_at_dist(d_horiz=_APPROACH_GATE_M - 0.2, alt=2.0, cx=320.0, cy=460.0)
    out = servo(det, IMG_W, IMG_H, altitude_m=2.0)
    assert out.up_mps < 0.0
    assert out.descending


def test_descent_lateral_right():
    out = servo(_descent_det(cx=420), IMG_W, IMG_H, altitude_m=2.0)
    assert out.right_mps > 0.0


def test_descent_lateral_left():
    out = servo(_descent_det(cx=200), IMG_W, IMG_H, altitude_m=2.0)
    assert out.right_mps < 0.0


def test_descent_lateral_gain_reduced_at_low_altitude():
    """Lateral corrections gentler near the ground."""
    err_px = 80
    out_high = servo(_descent_det(alt=3.0, cx=320+err_px), IMG_W, IMG_H, altitude_m=3.0)
    out_low  = servo(_descent_det(alt=0.5, cx=320+err_px), IMG_W, IMG_H, altitude_m=0.5)
    assert abs(out_low.right_mps) < abs(out_high.right_mps)


# ---------------------------------------------------------------------------
# Low confidence / invisible
# ---------------------------------------------------------------------------

def test_servo_low_confidence_returns_hover():
    out = servo(_det(confidence=_MIN_CONFIDENCE - 0.1), IMG_W, IMG_H, altitude_m=2.0)
    assert out.forward_mps == 0.0
    assert out.right_mps   == 0.0
    assert out.up_mps      == 0.0


def test_servo_not_visible_returns_hover():
    out = servo(_det(visible=False), IMG_W, IMG_H, altitude_m=2.0)
    assert out.forward_mps == 0.0
    assert out.up_mps      == 0.0


# ---------------------------------------------------------------------------
# Hard limits
# ---------------------------------------------------------------------------

def test_servo_outputs_within_limits():
    for det in [_det(cx=0, cy=0), _descent_det(cx=0, cy=10)]:
        out = servo(det, IMG_W, IMG_H, altitude_m=3.0)
        assert _MAX_BACK  <= out.forward_mps <= _MAX_FORWARD
        assert _MAX_LEFT  <= out.right_mps   <= _MAX_RIGHT
        assert _MAX_DOWN  <= out.up_mps      <= _MAX_UP
        assert -_MAX_YAW  <= out.yaw_rate_dps <= _MAX_YAW


# ---------------------------------------------------------------------------
# search_step / turn
# ---------------------------------------------------------------------------

def test_search_step_moves_forward():
    out = search_step(270.0)
    assert out.forward_mps > 0.0
    assert out.right_mps   == 0.0


def test_turn_produces_yaw():
    assert turn(30.0).yaw_rate_dps == 30.0


def test_turn_clamped():
    assert turn(999.0).yaw_rate_dps  <=  _MAX_YAW
    assert turn(-999.0).yaw_rate_dps >= -_MAX_YAW


def test_control_output_as_command_fields():
    out = ControlOutput(forward_mps=1.0, right_mps=-0.5, up_mps=0.2, yaw_rate_dps=10.0)
    f = out.as_command_fields()
    assert f["forward_mps"]  == 1.0
    assert f["right_mps"]    == -0.5
    assert f["up_mps"]       == 0.2
    assert f["yaw_rate_dps"] == 10.0
