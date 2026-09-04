"""Tests for daic local occupancy mapping and path planning."""

import pytest

from daic.local_map import (
    _CAMERA_HALF_FOV_DEG, _SECTOR_BANDS,
    LocalOccupancyMap, Pose2, pose_from_status, target_xy_from_status,
)
from daic.orb_slam3_detector import ObstacleSectors


def _sectors(front=0.0, front_left=0.0, front_right=0.0,
             left=0.0, right=0.0, confidence=1.0,
             front_range_m=None, front_left_range_m=None,
             front_right_range_m=None, left_range_m=None, right_range_m=None,
             method="test") -> ObstacleSectors:
    return ObstacleSectors(
        front=front,
        front_left=front_left,
        front_right=front_right,
        left=left,
        right=right,
        confidence=confidence,
        method=method,
        front_range_m=front_range_m,
        front_left_range_m=front_left_range_m,
        front_right_range_m=front_right_range_m,
        left_range_m=left_range_m,
        right_range_m=right_range_m,
    )


def test_pose_from_status_reads_sim_pose() -> None:
    pose = pose_from_status({
        "drone.x_m": "3.5",
        "drone.y_m": "4.5",
        "drone.heading_deg": "270",
    })

    assert pose == Pose2(3.5, 4.5, 180.0)


def test_compass_heading_north_routes_forward_to_north_target() -> None:
    local_map = LocalOccupancyMap(cell_m=0.5, half_width_m=8.0)
    pose = pose_from_status({
        "drone.x_m": "17.5",
        "drone.y_m": "16.5",
        "drone.heading_deg": "0",
    })

    assert pose is not None
    planned = local_map.plan_to_target(pose, (17.5, 7.5))

    assert planned is not None
    assert planned.fields["forward_mps"] > 0.0
    assert abs(planned.fields["yaw_rate_dps"]) < 1.0


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


def test_front_occ_diagnostics_include_provenance() -> None:
    local_map = LocalOccupancyMap(cell_m=0.5, half_width_m=8.0)
    pose = Pose2(0.0, 0.0, 0.0)

    for _ in range(3):
        local_map.update(
            pose,
            _sectors(front=1.0, front_range_m=1.2, method="mini_slam:ok"),
        )

    diag = local_map.diagnostics(pose)

    source = diag["front_occ_source"]
    assert source["source"] == "mini_slam:ok"
    assert source["sector"] == "front"
    assert source["ranged"] is True
    assert source["age_ticks"] == 0
    assert source["hit_count"] >= 1
    assert source["value"] > 0.0
    assert "mini_slam:ok|front|ranged" in diag["occupied_by_source"]


def test_provenance_age_increases_without_new_hit() -> None:
    local_map = LocalOccupancyMap(cell_m=0.5, half_width_m=8.0)
    pose = Pose2(0.0, 0.0, 0.0)

    local_map.update(
        pose,
        _sectors(front=1.0, front_range_m=1.2, method="mini_slam:ok"),
    )
    local_map.update(pose, _sectors())

    diag = local_map.diagnostics(pose)

    assert diag["front_occ_source"]["age_ticks"] == 1


def test_snapshot_keeps_cells_and_adds_provenance() -> None:
    local_map = LocalOccupancyMap(cell_m=0.5, half_width_m=8.0)
    pose = Pose2(0.0, 0.0, 0.0)

    local_map.update(
        pose,
        _sectors(front=1.0, front_range_m=1.2, method="mini_slam:ok"),
    )

    snap = local_map.snapshot()

    assert snap["cells"]
    assert snap["provenance"]
    assert all(isinstance(cell, tuple) for cell in snap["cells"])


def test_side_sector_cells_do_not_count_as_front_blocking() -> None:
    local_map = LocalOccupancyMap(cell_m=0.5, half_width_m=8.0)
    # A left-sector mark at yaw 0 lands in the broad front corridor once the
    # drone later points north, but it is not direct front evidence.
    mark_pose = Pose2(0.0, 0.0, 0.0)
    diag_pose = Pose2(0.0, 0.0, 270.0)

    for _ in range(3):
        local_map.update(
            mark_pose,
            _sectors(left=1.0, left_range_m=1.2, method="mini_slam:ok"),
        )

    diag = local_map.diagnostics(diag_pose)

    assert diag["front_occ_m"] is not None
    assert diag["front_occ_source"]["sector"] == "left"
    assert diag["front_block_occ_m"] is None


def test_front_sector_cells_count_as_front_blocking() -> None:
    local_map = LocalOccupancyMap(cell_m=0.5, half_width_m=8.0)
    pose = Pose2(0.0, 0.0, 0.0)

    for _ in range(3):
        local_map.update(
            pose,
            _sectors(front=1.0, front_range_m=1.2, method="mini_slam:ok"),
        )

    diag = local_map.diagnostics(pose)

    assert diag["front_block_occ_m"] is not None
    assert diag["front_block_occ_source"]["sector"] == "front"


def test_single_flow_ranged_risk_does_not_seed_local_map() -> None:
    local_map = LocalOccupancyMap(cell_m=0.5, half_width_m=8.0)
    pose = Pose2(0.0, 0.0, 0.0)

    local_map.update(
        pose,
        _sectors(front=1.0, front_range_m=1.2, method="flow:expansion"),
    )

    diag = local_map.diagnostics(pose)

    assert diag["occupied_cells"] == 0
    assert diag["front_occ_m"] is None


def test_confirmed_front_flow_ranged_risk_seeds_soft_local_map() -> None:
    local_map = LocalOccupancyMap(cell_m=0.5, half_width_m=8.0)
    pose = Pose2(0.0, 0.0, 0.0)

    for _ in range(3):
        local_map.update(
            pose,
            _sectors(front=1.0, front_range_m=1.2, method="flow:expansion"),
        )

    diag = local_map.diagnostics(pose)

    assert diag["occupied_cells"] > 0
    assert diag["front_occ_m"] is not None
    source = diag["front_occ_source"]
    assert source["source"] == "confirmed_flow:flow:expansion"
    assert source["sector"] == "front"
    assert source["ranged"] is True
    assert source["value"] < 1.6
    assert "confirmed_flow:flow:expansion|front|ranged" in diag["occupied_by_source"]


def test_confirmed_front_flow_soft_occupancy_decays_quickly() -> None:
    local_map = LocalOccupancyMap(cell_m=0.5, half_width_m=8.0)
    pose = Pose2(0.0, 0.0, 0.0)

    for _ in range(3):
        local_map.update(
            pose,
            _sectors(front=1.0, front_range_m=1.2, method="flow:expansion"),
        )
    before = local_map.diagnostics(pose)["front_occ_source"]["value"]

    local_map.update(pose, _sectors())
    after = local_map.diagnostics(pose)["front_occ_source"]["value"]

    assert after < before * 0.95


def test_confirmed_flow_does_not_downgrade_existing_hard_non_flow_cell() -> None:
    local_map = LocalOccupancyMap(cell_m=0.5, half_width_m=8.0)
    pose = Pose2(0.0, 0.0, 0.0)

    for _ in range(3):
        local_map.update(
            pose,
            _sectors(front=1.0, front_range_m=1.2, method="mini_slam:ok"),
        )
    before = local_map.diagnostics(pose)["front_occ_source"]

    for _ in range(3):
        local_map.update(
            pose,
            _sectors(front=1.0, front_range_m=1.2, method="flow:expansion"),
        )
    after = local_map.diagnostics(pose)["front_occ_source"]

    assert before["source"] == "mini_slam:ok"
    assert after["source"] == "mini_slam:ok"
    assert after["value"] >= 1.6


def test_repeated_front_flow_promotes_to_hard_occupancy() -> None:
    local_map = LocalOccupancyMap(cell_m=0.5, half_width_m=8.0)
    pose = Pose2(0.0, 0.0, 0.0)

    # Three hits make a soft cell; sustained same-cell hits beyond the hard
    # threshold promote it to an impassable A* obstacle (Phase 6.7).
    for _ in range(8):
        local_map.update(
            pose,
            _sectors(front=1.0, front_range_m=1.2, method="flow:expansion"),
        )

    diag = local_map.diagnostics(pose)
    source = diag["front_occ_source"]
    assert source["source"] == "confirmed_flow_hard:flow:expansion"
    assert source["value"] >= 1.6
    assert "confirmed_flow_hard:flow:expansion|front|ranged" in (
        diag["occupied_by_source"]
    )


def test_hard_confirmed_flow_decays_at_normal_rate() -> None:
    local_map = LocalOccupancyMap(cell_m=0.5, half_width_m=8.0)
    pose = Pose2(0.0, 0.0, 0.0)

    # Range placed beyond the free-fan band (dist <= 2.0 m) so the decay rate is
    # isolated from the per-tick free-space marking directly ahead. Front risk is
    # kept below the Phase 6.8 sustained-front threshold (0.6) so the nearer
    # sustained barrier does not form and this isolates confirmed-flow decay.
    for _ in range(8):
        local_map.update(
            pose,
            _sectors(front=0.5, front_range_m=3.0, method="flow:expansion"),
        )
    before = local_map.diagnostics(pose)["front_occ_source"]["value"]

    local_map.update(pose, _sectors())
    after = local_map.diagnostics(pose)["front_occ_source"]["value"]

    # Normal _DECAY (~0.995), not the fast soft-flow rate (_FLOW_SOFT_DECAY=0.92).
    assert after > before * 0.97


def test_sustained_rangeless_front_risk_with_steady_heading_maps_hard() -> None:
    local_map = LocalOccupancyMap(cell_m=0.5, half_width_m=8.0)
    # Drone holding a steady heading, confronting a rangeless wall ahead (the
    # creep-stall case: braked to a crawl but still facing the wall).
    sectors = _sectors(front=1.0, method="mini_slam:ok+flow:persist")

    for i in range(8):
        local_map.update(Pose2(float(i) * 0.02, 0.0, 0.0), sectors)

    pose = Pose2(8 * 0.02, 0.0, 0.0)
    diag = local_map.diagnostics(pose)
    assert diag["occupied_cells"] > 0
    source = diag["front_occ_source"]
    assert source["source"] == "sustained_front:mini_slam:ok+flow:persist"
    assert source["value"] >= 1.6  # hard / impassable in A*


def test_sustained_front_risk_while_yaw_scanning_stays_reactive_only() -> None:
    local_map = LocalOccupancyMap(cell_m=0.5, half_width_m=8.0)
    # Same strong rangeless front risk, but the drone is yaw-scanning (heading
    # changing every tick), so it must not paint a phantom 2 m halo as obstacles
    # sweep through the front sector (the Phase 6.2 failure mode).
    sectors = _sectors(front=1.0, method="mini_slam:ok+flow:persist")
    for i in range(12):
        local_map.update(Pose2(0.0, 0.0, float(i) * 5.0), sectors)

    diag = local_map.diagnostics(Pose2(0.0, 0.0, 60.0))
    assert all(
        "sustained_front" not in k for k in diag["occupied_by_source"]
    )


def test_confirmed_side_flow_ranged_risk_stays_reactive_only() -> None:
    local_map = LocalOccupancyMap(cell_m=0.5, half_width_m=8.0)
    pose = Pose2(0.0, 0.0, 0.0)

    for _ in range(8):
        local_map.update(
            pose,
            _sectors(
                left=1.0,
                left_range_m=1.2,
                method="flow:expansion",
            ),
        )

    diag = local_map.diagnostics(pose)

    assert diag["occupied_cells"] == 0
    assert diag["front_occ_m"] is None


def test_confirmed_fused_flow_ranged_risk_is_treated_as_flow() -> None:
    local_map = LocalOccupancyMap(cell_m=0.5, half_width_m=8.0)
    pose = Pose2(0.0, 0.0, 0.0)

    for _ in range(3):
        local_map.update(
            pose,
            _sectors(
                front=1.0,
                front_range_m=1.2,
                method="mini_slam:ok+flow:expansion+persist",
            ),
        )

    diag = local_map.diagnostics(pose)

    assert diag["occupied_cells"] > 0
    assert diag["front_occ_source"]["source"] == (
        "confirmed_flow:mini_slam:ok+flow:expansion+persist"
    )
    assert diag["front_occ_source"]["value"] < 1.6


def test_rangeless_mini_slam_risk_does_not_seed_local_map() -> None:
    local_map = LocalOccupancyMap(cell_m=0.5, half_width_m=8.0)
    pose = Pose2(0.0, 0.0, 0.0)

    # Transient rangeless mini-SLAM risk (fewer ticks than the Phase 6.8
    # sustained-front confirmation) stays reactive-only — the normal admission
    # path still bars rangeless evidence. (Sustained steady-heading rangeless
    # front risk is the Phase 6.8 exception, covered by
    # test_sustained_rangeless_front_risk_with_steady_heading_maps_hard.)
    for _ in range(4):
        local_map.update(pose, _sectors(front=1.0, method="mini_slam:ok"))

    diag = local_map.diagnostics(pose)

    assert diag["occupied_cells"] == 0
    assert diag["front_occ_m"] is None


def test_ranged_mini_slam_risk_can_seed_local_map() -> None:
    local_map = LocalOccupancyMap(cell_m=0.5, half_width_m=8.0)
    pose = Pose2(0.0, 0.0, 0.0)

    for _ in range(3):
        local_map.update(
            pose,
            _sectors(front=1.0, front_range_m=1.2, method="mini_slam:ok"),
        )

    diag = local_map.diagnostics(pose)

    assert diag["occupied_cells"] > 0
    assert diag["front_occ_m"] is not None
    assert diag["front_occ_m"] < 2.0


def test_repeated_weak_rangeless_mini_slam_risk_is_not_promoted() -> None:
    local_map = LocalOccupancyMap(cell_m=0.5, half_width_m=8.0)
    pose = Pose2(0.0, 0.0, 0.0)

    for _ in range(20):
        local_map.update(pose, _sectors(front=0.4, method="mini_slam:ok"))

    diag = local_map.diagnostics(pose)

    assert diag["occupied_cells"] == 0


def test_low_risk_obstacle_marks_cells_earlier() -> None:
    local_map = LocalOccupancyMap(cell_m=0.5, half_width_m=8.0)
    pose = Pose2(0.0, 0.0, 0.0)

    for _ in range(2):
        local_map.update(pose, _sectors(front=0.13))

    diag = local_map.diagnostics(pose)

    assert diag["occupied_cells"] > 0


def test_too_low_risk_obstacle_is_ignored() -> None:
    local_map = LocalOccupancyMap(cell_m=0.5, half_width_m=8.0)
    pose = Pose2(0.0, 0.0, 0.0)

    for _ in range(3):
        local_map.update(pose, _sectors(front=0.11))

    diag = local_map.diagnostics(pose)

    assert diag["occupied_cells"] == 0


def test_route_command_turns_toward_next_waypoint() -> None:
    local_map = LocalOccupancyMap(cell_m=0.5, half_width_m=8.0)
    pose = Pose2(0.0, 0.0, 270.0)

    planned = local_map.plan_to_target(pose, (0.0, -6.0))

    assert planned is not None
    assert planned.fields["forward_mps"] > 0.0
    assert abs(planned.fields["yaw_rate_dps"]) < 1.0


def test_route_command_turns_in_place_for_large_yaw_error() -> None:
    local_map = LocalOccupancyMap(cell_m=0.5, half_width_m=8.0)
    pose = Pose2(0.0, 0.0, 0.0)

    planned = local_map.plan_to_target(pose, (0.0, -6.0))

    assert planned is not None
    assert planned.fields["forward_mps"] == 0.0
    # A negative bearing error is a public left/negative-yaw command.
    assert planned.fields["yaw_rate_dps"] < 0.0
    assert abs(planned.fields["yaw_rate_dps"]) <= 18.0


def test_route_command_moves_forward_when_nearly_aligned() -> None:
    local_map = LocalOccupancyMap(cell_m=0.5, half_width_m=8.0)
    pose = Pose2(0.0, 0.0, 250.0)

    planned = local_map.plan_to_target(pose, (0.0, -6.0))

    assert planned is not None
    assert planned.fields["forward_mps"] > 0.0
    # A positive bearing error is a public right/positive-yaw command.
    assert planned.fields["yaw_rate_dps"] > 0.0


# --- sector bearings match the camera ----------------------------------

# The camera's actual half-angle, as a reviewed literal. Deliberately *not*
# local_map's own _CAMERA_HALF_FOV_DEG: a test that bounds a constant by itself
# passes no matter what that constant becomes, which is how the original
# +/-70 degree bug would slip straight back in.
_REVIEWED_CAMERA_HALF_FOV_DEG = 35.0


def test_local_map_agrees_with_the_simulator_about_the_camera() -> None:
    """The map's idea of the FOV must track the camera that produces the frames."""
    from dsim.dsim import Panda3DRenderer

    assert _CAMERA_HALF_FOV_DEG == pytest.approx(_REVIEWED_CAMERA_HALF_FOV_DEG)
    assert _CAMERA_HALF_FOV_DEG == pytest.approx(Panda3DRenderer.CAM_FOV_H / 2.0), (
        f"local_map assumes a +/-{_CAMERA_HALF_FOV_DEG:.1f} deg camera but dsim "
        f"renders {Panda3DRenderer.CAM_FOV_H:.1f} deg horizontally; observations "
        "would be planted outside the view that produced them")


def test_every_sector_is_planted_inside_the_camera_field_of_view() -> None:
    """The map planted left/right obstacles at +/-70 deg on a +/-35 deg camera.

    An observation can only come from a bearing the camera can actually see, so
    planting one outside the FOV smears a wall into an arc of phantom cells.
    """
    limit = _REVIEWED_CAMERA_HALF_FOV_DEG
    for name, (bearing, half_width) in _SECTOR_BANDS.items():
        assert abs(bearing) <= limit, name
        assert abs(bearing) + half_width <= limit + 1e-6, name


def test_sector_bearings_are_ordered_left_to_right() -> None:
    order = ["left", "front_left", "front", "front_right", "right"]
    bearings = [_SECTOR_BANDS[n][0] for n in order]

    assert bearings == sorted(bearings)
    assert _SECTOR_BANDS["front"][0] == 0.0
