"""Regression tests for DAIC SEARCH command arbitration."""

from __future__ import annotations

import inspect

from daic.daic import (
    DaicController, HeadlessAgent,
    _APPROACH_BLOCK_FRONT_OCC_M, _APPROACH_BLOCK_RISK,
    _SEARCH_HOLD_YAW_DPS,
    _approach_block_fields, _approach_gate_reason, _frontish_risk,
    _search_hold_scan_fields,
)
from daic.orb_slam3_detector import ObstacleSectors


def _sectors(front=0.0, front_left=0.0, front_right=0.0) -> ObstacleSectors:
    return ObstacleSectors(
        front=front,
        front_left=front_left,
        front_right=front_right,
        left=0.0,
        right=0.0,
        confidence=1.0,
        method="test",
    )


def test_search_local_route_is_not_gated_by_target_visibility() -> None:
    """SEARCH local routing must stay active during the target detection lock."""
    for method in (DaicController._run_ai, HeadlessAgent._tick):
        source = inspect.getsource(method)
        assert "not target_visible" not in source
        assert "target_visible" not in source


def test_search_hold_scan_is_stationary_except_yaw() -> None:
    fields = _search_hold_scan_fields()

    assert fields == {
        "forward_mps": 0.0,
        "right_mps": 0.0,
        "up_mps": 0.0,
        "yaw_rate_dps": _SEARCH_HOLD_YAW_DPS,
    }


def test_approach_gate_uses_frontish_risk() -> None:
    sectors = _sectors(front=_APPROACH_BLOCK_RISK)

    assert _frontish_risk(sectors) == _APPROACH_BLOCK_RISK
    assert _approach_gate_reason(sectors, None) == "front risk 0.25"


def test_approach_gate_uses_local_front_occupancy() -> None:
    reason = _approach_gate_reason(
        _sectors(),
        {"front_occ_m": _APPROACH_BLOCK_FRONT_OCC_M},
    )

    assert reason == "front occ 1.50m"


def test_approach_block_fields_preserve_yaw_only() -> None:
    fields = _approach_block_fields({
        "forward_mps": 7.0,
        "right_mps": -2.0,
        "up_mps": -1.0,
        "yaw_rate_dps": 12.0,
    })

    assert fields == {
        "forward_mps": 0.0,
        "right_mps": 0.0,
        "up_mps": 0.0,
        "yaw_rate_dps": 12.0,
    }
