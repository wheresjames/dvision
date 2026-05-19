"""Tests for daic local occupancy mapping and path planning."""

from daic.local_map import (
    LocalOccupancyMap, Pose2, pose_from_status, target_xy_from_status,
)
from daic.orb_slam3_detector import ObstacleSectors


def _sectors(front=0.0, front_left=0.0, front_right=0.0,
             left=0.0, right=0.0, confidence=1.0,
             front_range_m=None) -> ObstacleSectors:
    return ObstacleSectors(
        front=front,
        front_left=front_left,
        front_right=front_right,
        left=left,
        right=right,
        confidence=confidence,
        method="test",
        front_range_m=front_range_m,
    )


def test_pose_from_status_reads_sim_pose() -> None:
    pose = pose_from_status({
        "drone.x_m": "3.5",
        "drone.y_m": "4.5",
        "drone.heading_deg": "270",
    })

    assert pose == Pose2(3.5, 4.5, 270.0)


def test_target_xy_from_status_converts_gps_offset_to_map_axes() -> None:
    status = {
        "drone.x_m": "10.0",
        "drone.y_m": "10.0",
        "drone.lat_deg": "52.0",
        "drone.lon_deg": "13.0",
        "target.lat_deg": str(52.0 + 10.0 / 111_320.0),
        "target.lon_deg": "13.0",
    }

    target = target_xy_from_status(status)

    assert target is not None
    assert abs(target[0] - 10.0) < 0.05
    assert abs(target[1] - 0.0) < 0.05


def test_front_obstacle_marks_cells_and_routes_around_them() -> None:
    local_map = LocalOccupancyMap(cell_m=0.5, half_width_m=8.0)
    pose = Pose2(0.0, 0.0, 0.0)

    for _ in range(3):
        local_map.update(pose, _sectors(front=1.0))

    planned = local_map.plan_to_target(pose, (6.0, 0.0))

    assert planned is not None
    assert planned.path
    # A straight path would keep y near 0. The obstacle at x ~= 3 should force
    # the A* route to pick a side.
    assert max(abs(y) for _, y in planned.path) >= 0.5


def test_front_obstacle_uses_measured_range() -> None:
    local_map = LocalOccupancyMap(cell_m=0.5, half_width_m=8.0)
    pose = Pose2(0.0, 0.0, 0.0)

    for _ in range(3):
        local_map.update(pose, _sectors(front=1.0, front_range_m=1.2))

    diag = local_map.diagnostics(pose)

    assert diag["front_occ_m"] is not None
    assert diag["front_occ_m"] < 2.0


def test_route_command_turns_toward_next_waypoint() -> None:
    local_map = LocalOccupancyMap(cell_m=0.5, half_width_m=8.0)
    pose = Pose2(0.0, 0.0, 270.0)

    planned = local_map.plan_to_target(pose, (0.0, -6.0))

    assert planned is not None
    assert planned.fields["forward_mps"] > 0.0
    assert abs(planned.fields["yaw_rate_dps"]) < 1.0


def test_route_command_allows_slow_motion_while_turning() -> None:
    local_map = LocalOccupancyMap(cell_m=0.5, half_width_m=8.0)
    pose = Pose2(0.0, 0.0, 0.0)

    planned = local_map.plan_to_target(pose, (0.0, -6.0))

    assert planned is not None
    assert planned.fields["forward_mps"] > 0.0
    assert planned.fields["yaw_rate_dps"] < 0.0
    assert abs(planned.fields["yaw_rate_dps"]) <= 18.0
