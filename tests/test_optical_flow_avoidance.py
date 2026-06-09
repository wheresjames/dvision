"""Tests for optical-flow obstacle risk extraction."""

import numpy as np

from daic.optical_flow_avoidance import (
    _body_forward_speed, _flow_to_sectors, _persist_sectors,
    fuse_obstacle_sectors,
)
from daic.orb_slam3_detector import ObstacleSectors, _NULL_SECTORS


def _radial_flow(scale: float, w: int = 160, h: int = 120) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    flow = np.zeros((h, w, 2), dtype=np.float32)
    flow[:, :, 0] = (xx - cx) * scale
    flow[:, :, 1] = (yy - cy) * scale
    return flow


def test_radial_expansion_without_translation_is_suppressed() -> None:
    """Radial-looking flow must not register as obstacle risk while stationary.

    Phase 7 root cause: yaw-scanning (the drone's dominant SEARCH-state motion,
    forward_speed_mps ~= 0) sweeps the off-axis-mounted camera through a small
    arc, and the nearby, steeply-foreshortened floor parallaxes hard against
    the distant scene -- producing flow with the *same* radially-symmetric,
    ~zero-median, depth-dependent signature as genuine obstacle expansion.
    Median-flow subtraction can't tell the two apart (both already have ~zero
    median); the only reliable discriminator is whether the camera is actually
    translating toward the scene, since optical "expansion toward an obstacle"
    is impossible without that translation. A live benchmark run confirmed the
    split cleanly: front risk reads 0.86 median (98% of ticks > 0.3) while
    forward_speed_mps < 0.06, vs. 0.14 median (15% > 0.3) once it isn't.
    """
    sectors = _flow_to_sectors(_radial_flow(0.035), forward_speed_mps=0.0)

    assert sectors.front == 0.0
    assert sectors.front_left == 0.0
    assert sectors.front_right == 0.0
    assert sectors.confidence == 0.0
    assert sectors.method == "flow:no_translation"


def test_radial_expansion_produces_front_risk_when_translating() -> None:
    sectors = _flow_to_sectors(_radial_flow(0.035), forward_speed_mps=0.5)

    assert sectors.confidence > 0.35
    assert sectors.front > 0.5
    assert sectors.method == "flow:expansion"


def test_radial_expansion_estimates_range_from_forward_motion() -> None:
    sectors = _flow_to_sectors(
        _radial_flow(0.04),
        forward_speed_mps=0.5,
        dt_s=0.05,
    )

    assert sectors.front > 0.0
    assert sectors.front_range_m is not None
    assert 4.5 <= sectors.front_range_m <= 5.5


def test_implausibly_close_ttc_range_is_rejected_not_clamped() -> None:
    """Phase 6.1 root cause: a TTC range this short during genuine forward
    translation is far more often the floor parallaxing close beneath the
    pitched-forward camera (the same near-field surface the ROI comment in
    `_flow_to_sectors` calls a "permanently close surface") than a real
    navigable wall. Clamping it up to `_MIN_RANGE_M` and trusting it let
    `local_map` quantize a phantom obstacle into the drone's own grid cell,
    so `front_occ_m` fired on ~90-98% of SEARCH/APPROACH ticks regardless of
    `target_dist_m`. The risk must still come through (it drives avoidance);
    only the implausible distance is dropped."""
    sectors = _flow_to_sectors(_radial_flow(0.35), forward_speed_mps=0.5, dt_s=0.05)

    assert sectors.front > 0.5
    assert sectors.front_range_m is None


def test_range_is_absent_just_above_the_translation_gate() -> None:
    """Just past the gate, range still needs `dt_s` and a clear ratio signal --
    confirming the gate isn't doing the range-gating work that `_range_scale_m`
    is responsible for."""
    sectors = _flow_to_sectors(
        _radial_flow(0.01),
        forward_speed_mps=0.06,
        dt_s=None,
    )

    assert sectors.front > 0.0
    assert sectors.front_range_m is None


def test_low_motion_flow_is_low_confidence() -> None:
    # forward_speed_mps must clear the translation gate so this exercises the
    # low-motion path specifically, rather than short-circuiting on it.
    sectors = _flow_to_sectors(np.zeros((120, 160, 2), dtype=np.float32),
                               forward_speed_mps=0.5)

    assert sectors.confidence == 0.0
    assert sectors.front == 0.0
    assert sectors.method == "flow:low_motion"


def test_zero_flow_without_translation_reports_no_translation() -> None:
    sectors = _flow_to_sectors(np.zeros((120, 160, 2), dtype=np.float32),
                               forward_speed_mps=0.0)

    assert sectors.confidence == 0.0
    assert sectors.front == 0.0
    assert sectors.method == "flow:no_translation"


def test_body_forward_speed_uses_simulator_compass_heading() -> None:
    assert abs(_body_forward_speed(0.0, -2.0, 0.0) - 2.0) < 1e-9
    assert abs(_body_forward_speed(2.0, 0.0, 90.0) - 2.0) < 1e-9
    assert abs(_body_forward_speed(0.0, 2.0, 180.0) - 2.0) < 1e-9
    assert abs(_body_forward_speed(-2.0, 0.0, 270.0) - 2.0) < 1e-9
    assert abs(_body_forward_speed(0.0, 2.0, 0.0) + 2.0) < 1e-9


def test_fuse_obstacle_sectors_takes_maximum_risk() -> None:
    slam = ObstacleSectors(0.1, 0.8, 0.0, 0.0, 0.0, 0.5, "slam")
    flow = ObstacleSectors(0.7, 0.0, 0.4, 0.0, 0.0, 0.9, "flow")

    fused = fuse_obstacle_sectors(slam, flow)

    assert fused.front == 0.7
    assert fused.front_left == 0.8
    assert fused.front_right == 0.4
    assert fused.confidence == 0.9
    assert fused.method == "slam+flow"


def test_fuse_obstacle_sectors_ignores_null_inputs() -> None:
    flow = ObstacleSectors(0.7, 0.0, 0.0, 0.0, 0.0, 0.8, "flow")

    fused = fuse_obstacle_sectors(_NULL_SECTORS, flow)

    assert fused == flow


def test_persist_sectors_keeps_recent_front_risk_on_dropout() -> None:
    previous = ObstacleSectors(0.8, 0.3, 0.0, 0.0, 0.0, 0.9, "flow")

    persisted = _persist_sectors(previous, _NULL_SECTORS)

    assert persisted.confidence > 0.35
    assert persisted.front > 0.6
    assert persisted.method == "flow:persist"


def test_persist_sectors_prefers_new_stronger_risk() -> None:
    previous = ObstacleSectors(0.4, 0.0, 0.0, 0.0, 0.0, 0.8, "flow")
    current = ObstacleSectors(0.9, 0.0, 0.5, 0.0, 0.0, 0.7, "flow:expansion")

    persisted = _persist_sectors(previous, current)

    assert persisted.front == 0.9
    assert persisted.front_right == 0.5
    assert persisted.confidence > current.confidence
    assert persisted.method == "flow:expansion+persist"


def test_persist_sectors_drops_stale_range_estimate() -> None:
    """Persisted (no fresh detection) sectors must not keep claiming a range.

    A persisted reading is a decayed memory of risk, not a live measurement —
    carrying its range estimate forward verbatim would let a single stale
    close-range reading "stick" to the drone indefinitely, since
    fuse_obstacle_sectors always prefers the nearest of the valid ranges. Only
    a fresh detection should report a distance.
    """
    previous = ObstacleSectors(
        0.8, 0.0, 0.0, 0.0, 0.0, 0.8, "flow",
        front_range_m=1.4,
    )

    persisted = _persist_sectors(previous, _NULL_SECTORS)

    assert persisted.front_range_m is None
