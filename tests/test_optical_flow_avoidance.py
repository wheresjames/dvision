"""Tests for optical-flow obstacle risk extraction."""

import numpy as np

from daic.optical_flow_avoidance import (
    _flow_to_sectors, _persist_sectors, fuse_obstacle_sectors,
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


def test_radial_expansion_produces_front_risk() -> None:
    sectors = _flow_to_sectors(_radial_flow(0.035))

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


def test_range_is_absent_without_forward_motion() -> None:
    sectors = _flow_to_sectors(
        _radial_flow(0.01),
        forward_speed_mps=0.0,
        dt_s=0.05,
    )

    assert sectors.front > 0.0
    assert sectors.front_range_m is None


def test_low_motion_flow_is_low_confidence() -> None:
    sectors = _flow_to_sectors(np.zeros((120, 160, 2), dtype=np.float32))

    assert sectors.confidence == 0.0
    assert sectors.front == 0.0
    assert sectors.method == "flow:low_motion"


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


def test_persist_sectors_keeps_range_estimate() -> None:
    previous = ObstacleSectors(
        0.8, 0.0, 0.0, 0.0, 0.0, 0.8, "flow",
        front_range_m=1.4,
    )

    persisted = _persist_sectors(previous, _NULL_SECTORS)

    assert persisted.front_range_m == 1.4
