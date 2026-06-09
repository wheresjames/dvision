"""Reactive obstacle avoidance for daic velocity commands."""

from __future__ import annotations

from typing import Any


_AVOID_MIN_CONFIDENCE = 0.35
_AVOID_START_RISK = 0.25
_AVOID_FULL_RISK = 0.75

# Phase 6.9 approach-speed envelope. When the local map reports a *mapped* wall
# close ahead (front_block_occ_m), slow the SEARCH approach to a steady,
# navigable cap that ramps down with distance instead of letting the reactive
# brake clamp forward to ~zero. The reactive zero-clamp strands the drone in a
# creep-stall (it stops, stops mapping, and never completes the turn) and then
# lurches forward when risk momentarily dips; a stable distance-based cap keeps
# the drone moving slowly enough for A* to commit to and finish a detour, with
# no lurch.
_BRAKE_START_M = 4.0
_BRAKE_FULL_M = 1.5
_BRAKE_MIN_MPS = 1.2
_NAV_CRUISE_MPS = 4.5


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def approach_speed_cap(front_block_occ_m: float | None) -> float | None:
    """Return the forward-speed cap for a mapped wall at this distance.

    Returns None when there is no close mapped wall (no cap; the caller should
    fall back to the reactive brake). Otherwise ramps from cruise speed at
    _BRAKE_START_M down to a navigable floor _BRAKE_MIN_MPS at _BRAKE_FULL_M.
    """
    if front_block_occ_m is None or front_block_occ_m >= _BRAKE_START_M:
        return None
    if front_block_occ_m <= _BRAKE_FULL_M:
        return _BRAKE_MIN_MPS
    t = (front_block_occ_m - _BRAKE_FULL_M) / (_BRAKE_START_M - _BRAKE_FULL_M)
    return _BRAKE_MIN_MPS + t * (_NAV_CRUISE_MPS - _BRAKE_MIN_MPS)


def apply_search_approach_brake(command_fields: dict,
                                sectors: Any,
                                front_block_occ_m: float | None,
                                ) -> tuple[dict, bool]:
    """Slow the SEARCH approach when a mapped wall is close ahead (Phase 6.9).

    When the local map reports a close front-blocking wall, cap forward speed to
    a steady, navigable envelope (never zero) so A* keeps moving toward and
    around the wall. With no close mapped wall, defer to the reactive brake.
    """
    cap = approach_speed_cap(front_block_occ_m)
    if cap is None:
        return apply_obstacle_avoidance(command_fields, sectors)

    fields = dict(command_fields)
    forward = float(fields.get("forward_mps", 0.0))
    if forward > cap:
        fields["forward_mps"] = cap
        return fields, True
    return fields, False


def _risk(sectors: Any, name: str) -> float:
    return _clamp(float(getattr(sectors, name, 0.0)), 0.0, 1.0)


def apply_obstacle_avoidance(command_fields: dict,
                             sectors: Any) -> tuple[dict, bool]:
    """Return velocity fields adjusted away from detected obstacle sectors.

    The planner remains responsible for steering. This last-ditch safety layer
    only trims forward speed when an obstacle is close ahead; injecting lateral
    or yaw corrections here fights the route follower and causes oscillation.
    """
    confidence = _clamp(float(getattr(sectors, "confidence", 0.0)), 0.0, 1.0)
    if confidence < _AVOID_MIN_CONFIDENCE:
        return dict(command_fields), False

    front = _risk(sectors, "front")
    forward_risk = max(
        front,
        _risk(sectors, "front_left") * 0.7,
        _risk(sectors, "front_right") * 0.7,
    )

    if forward_risk < _AVOID_START_RISK:
        return dict(command_fields), False

    fields = dict(command_fields)
    active = False

    forward = float(fields.get("forward_mps", 0.0))
    if forward > 0.0 and forward_risk >= _AVOID_START_RISK:
        t = _clamp(
            (forward_risk - _AVOID_START_RISK) / (_AVOID_FULL_RISK - _AVOID_START_RISK),
            0.0,
            1.0,
        )
        fields["forward_mps"] = forward * (1.0 - t)
        active = t > 0.0

    return fields, active
