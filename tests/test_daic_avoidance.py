"""Tests for daic reactive obstacle avoidance."""

from daic.avoidance import apply_obstacle_avoidance
from daic.orb_slam3_detector import ObstacleSectors


def _sectors(front=0.0, front_left=0.0, front_right=0.0,
             left=0.0, right=0.0, confidence=1.0) -> ObstacleSectors:
    return ObstacleSectors(
        front=front,
        front_left=front_left,
        front_right=front_right,
        left=left,
        right=right,
        confidence=confidence,
        method="test",
    )


def test_low_confidence_obstacles_do_not_change_command() -> None:
    fields = {"forward_mps": 6.0, "right_mps": 0.0, "yaw_rate_dps": 0.0}

    out, active = apply_obstacle_avoidance(
        fields,
        _sectors(front=1.0, confidence=0.1),
    )

    assert out == fields
    assert not active


def test_front_obstacle_slows_forward_motion_without_steering() -> None:
    out, active = apply_obstacle_avoidance(
        {"forward_mps": 6.0, "right_mps": 1.0, "yaw_rate_dps": -5.0},
        _sectors(front=0.75),
    )

    assert active
    assert 0.0 < out["forward_mps"] < 6.0
    assert out["right_mps"] == 1.0
    assert out["yaw_rate_dps"] == -5.0


def test_side_only_obstacle_does_not_override_route_follower() -> None:
    fields = {"forward_mps": 4.0, "right_mps": 0.0, "yaw_rate_dps": 0.0}

    out, active = apply_obstacle_avoidance(
        fields,
        _sectors(left=0.9, right=0.1),
    )

    assert out == fields
    assert not active


def test_strong_front_obstacle_stops_forward_motion() -> None:
    out, active = apply_obstacle_avoidance(
        {"forward_mps": 4.0, "right_mps": 0.0, "yaw_rate_dps": -35.0},
        _sectors(front=0.95),
    )

    assert active
    assert out["forward_mps"] == 0.0
    assert out["right_mps"] == 0.0
    assert out["yaw_rate_dps"] == -35.0
