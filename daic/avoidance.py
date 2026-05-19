"""Reactive obstacle avoidance for daic velocity commands."""

from __future__ import annotations

from typing import Any


_AVOID_MIN_CONFIDENCE = 0.35
_AVOID_START_RISK = 0.55
_AVOID_FULL_RISK = 0.90
def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


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
