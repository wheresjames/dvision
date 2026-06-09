"""Tests for daic reactive obstacle avoidance."""

from daic.avoidance import (
    apply_obstacle_avoidance,
    apply_search_approach_brake,
    approach_speed_cap,
)
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
        _sectors(front=0.50),
    )

    assert active
    assert 0.0 < out["forward_mps"] < 6.0
    assert out["right_mps"] == 1.0
    assert out["yaw_rate_dps"] == -5.0


def test_moderate_front_obstacle_brakes_early() -> None:
    out, active = apply_obstacle_avoidance(
        {"forward_mps": 6.0, "right_mps": 0.0, "yaw_rate_dps": 0.0},
        _sectors(front=0.40),
    )

    assert active
    assert 0.0 < out["forward_mps"] < 6.0


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
        _sectors(front=0.75),
    )

    assert active
    assert out["forward_mps"] == 0.0
    assert out["right_mps"] == 0.0
    assert out["yaw_rate_dps"] == -35.0


def test_approach_speed_cap_ramps_with_distance() -> None:
    # No close mapped wall -> no cap (caller falls back to reactive brake).
    assert approach_speed_cap(None) is None
    assert approach_speed_cap(6.0) is None
    # Closer wall -> lower cap, never below the navigable floor.
    near = approach_speed_cap(1.5)
    far = approach_speed_cap(3.5)
    assert near < far
    assert near >= 1.2
    assert approach_speed_cap(0.5) == approach_speed_cap(1.5)  # floored


def test_search_brake_caps_forward_near_mapped_wall_without_stalling() -> None:
    # A close mapped wall ahead (front_block_occ_m=1.5) caps forward to the
    # navigable floor instead of zeroing it (Phase 6.9): the drone keeps moving
    # slowly so A* can carry out the detour.
    out, active = apply_search_approach_brake(
        {"forward_mps": 4.5, "right_mps": 0.0, "yaw_rate_dps": 0.0},
        _sectors(front=1.0),
        front_block_occ_m=1.5,
    )

    assert active
    assert 1.0 < out["forward_mps"] < 4.5  # slowed, but not stalled to zero


def test_search_brake_falls_back_to_reactive_when_no_mapped_wall() -> None:
    # With no close mapped wall, strong front sector risk still stops the drone
    # via the reactive brake (unchanged safety behavior).
    out, active = apply_search_approach_brake(
        {"forward_mps": 4.0, "right_mps": 0.0, "yaw_rate_dps": 0.0},
        _sectors(front=0.75),
        front_block_occ_m=None,
    )

    assert active
    assert out["forward_mps"] == 0.0


def test_search_brake_does_not_force_forward_during_a_turn() -> None:
    # Route follower commands zero forward (turning in place); the cap must not
    # push the drone forward into the wall.
    out, active = apply_search_approach_brake(
        {"forward_mps": 0.0, "right_mps": 0.0, "yaw_rate_dps": 12.0},
        _sectors(front=1.0),
        front_block_occ_m=1.5,
    )

    assert out["forward_mps"] == 0.0
    assert not active
