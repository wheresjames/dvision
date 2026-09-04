from types import SimpleNamespace

import pytest

import numpy as np

import dsim.range as range_module
from dsim.dsim import (DroneSimulator, DroneState, Panda3DRenderer,
                       parse_args)
from dvision2_common import (
    BERLIN_CENTER_ALT_M,
    BERLIN_CENTER_LAT_DEG,
    BERLIN_CENTER_LON_DEG,
)


def _sim() -> DroneSimulator:
    sim = DroneSimulator.__new__(DroneSimulator)
    sim.args = SimpleNamespace(
        id="test",
        map="test.map",
        origin_lat=0.0,
        origin_lon=0.0,
        origin_alt=0.0,
        width=640,
        height=480,
        fps=30,
    )
    sim.map = SimpleNamespace(
        path="test.map",
        width=5,
        height=5,
        objects=[SimpleNamespace(kind="wall", x=1.6, y=1.0)],
    )
    sim.start_x = 1.0
    sim.start_y = 1.0
    sim.start_alt = 1.5
    sim.start_yaw = 270.0
    sim.target_x = None
    sim.target_y = None
    sim.started = 0.0
    sim.report_root = "reports/test"
    sim.crash_pos = None
    sim.status = None
    sim.state = DroneState(sim.start_x, sim.start_y, sim.start_alt)
    return sim


def test_scene_preset_cli_defaults_to_representative_and_accepts_legacy():
    assert parse_args(["--id", "test"]).scene_preset == "representative"
    assert parse_args([
        "--id", "test", "--scene-preset", "legacy",
    ]).scene_preset == "legacy"


def test_live_renderer_receives_the_selected_scene_preset(monkeypatch):
    import dsim.dsim as module

    received = {}

    def renderer(sim_map, width, height, *, scene_preset):
        received.update(map=sim_map, width=width, height=height,
                        scene_preset=scene_preset)
        return object()

    sim = _sim()
    sim.args.scene_preset = "representative"
    sim.args.verbose = False
    monkeypatch.setattr(module, "Panda3DRenderer", renderer)

    sim._init_renderer()

    assert received == {
        "map": sim.map,
        "width": 640,
        "height": 480,
        "scene_preset": "representative",
    }


def test_wall_collision_latches_crashed_state():
    sim = _sim()
    sim.state.armed = True
    sim.state.mode = "GUIDED"
    sim.state.yaw_deg = 180.0
    sim.state.cmd_forward = 10.0
    sim.integrate(0.1)

    assert sim.state.crashed
    assert sim.state.mode == "CRASHED"
    assert sim.state.status_message == "crashed"
    assert not sim.state.armed
    assert sim.state.x == sim.start_x
    assert sim.state.y == sim.start_y


def test_adjacent_wall_cells_have_no_gap_between_them():
    sim = _sim()
    sim.map.objects = [
        SimpleNamespace(kind="wall", x=1.5, y=1.5),
        SimpleNamespace(kind="wall", x=2.5, y=1.5),
    ]

    assert sim.is_blocked(2.0, 1.5)


def test_swept_collision_catches_fast_motion_through_wall():
    sim = _sim()
    sim.map.objects = [SimpleNamespace(kind="wall", x=2.5, y=1.5)]

    assert sim.path_blocked(1.0, 1.5, 4.0, 1.5)


def test_crashed_state_ignores_normal_commands():
    sim = _sim()
    sim.crash()
    sim.apply_command({"type": "velocity", "forward_mps": 10.0})

    assert sim.state.crashed
    assert sim.state.mode == "CRASHED"
    assert sim.state.cmd_forward == 0.0


def test_reset_restores_start_pose_from_crashed_state():
    sim = _sim()
    sim.state.x = 2.0
    sim.state.y = 3.0
    sim.state.z = 0.2
    sim.crash()
    sim.reset_drone()

    assert not sim.state.crashed
    assert sim.state.mode == "DISARMED"
    assert sim.state.status_message == "reset"
    assert sim.state.x == sim.start_x
    assert sim.state.y == sim.start_y
    assert sim.state.z == sim.start_alt


def test_status_reports_crashed_flag():
    sim = _sim()

    class Status:
        values = None

        def setAll(self, values):
            self.values = values

    sim.status = Status()
    sim.crash()
    sim.publish_status()

    assert sim.status.values["drone.mode"] == "CRASHED"
    assert sim.status.values["drone.crashed"] == "1"
    assert sim.status.values["status.message"] == "crashed"


def test_map_center_defaults_to_berlin_gps():
    sim = _sim()
    sim.args.origin_lat = BERLIN_CENTER_LAT_DEG
    sim.args.origin_lon = BERLIN_CENTER_LON_DEG
    sim.args.origin_alt = BERLIN_CENTER_ALT_M

    lat, lon, alt = sim.map_to_gps(sim.map.width / 2.0, sim.map.height / 2.0, 2.0)

    assert lat == pytest.approx(BERLIN_CENTER_LAT_DEG)
    assert lon == pytest.approx(BERLIN_CENTER_LON_DEG)
    assert alt == pytest.approx(BERLIN_CENTER_ALT_M + 2.0)


def test_status_reports_target_gps_when_target_exists():
    sim = _sim()
    sim.args.origin_lat = BERLIN_CENTER_LAT_DEG
    sim.args.origin_lon = BERLIN_CENTER_LON_DEG
    sim.args.origin_alt = BERLIN_CENTER_ALT_M
    sim.target_x = sim.map.width / 2.0
    sim.target_y = sim.map.height / 2.0

    class Status:
        values = None

        def setAll(self, values):
            self.values = values

    sim.status = Status()
    sim.publish_status()

    assert sim.status.values["target.lat_deg"] == f"{BERLIN_CENTER_LAT_DEG:.7f}"
    assert sim.status.values["target.lon_deg"] == f"{BERLIN_CENTER_LON_DEG:.7f}"
    assert sim.status.values["target.alt_m"] == f"{BERLIN_CENTER_ALT_M:.3f}"


def test_obstacles_are_boxes_not_infinite_columns():
    """Collision uses the height the object is drawn and ray-cast at."""
    sim = _sim()
    sim.map.objects = [
        SimpleNamespace(kind="wall", x=2.5, y=1.5),
        SimpleNamespace(kind="tree", x=4.5, y=1.5),
    ]

    assert sim.is_blocked(2.5, 1.5, 0.0)
    assert sim.is_blocked(2.5, 1.5, Panda3DRenderer.WALL_H - 0.01)
    assert not sim.is_blocked(2.5, 1.5, Panda3DRenderer.WALL_H + 0.01)
    # A tree stands taller than a wall, so the clear height differs per kind.
    assert sim.is_blocked(4.5, 1.5, Panda3DRenderer.WALL_H + 0.5)
    assert not sim.is_blocked(4.5, 1.5, Panda3DRenderer.TREE_MODEL_H + 0.01)


def test_flying_over_a_wall_does_not_crash():
    sim = _sim()
    sim.map.objects = [SimpleNamespace(kind="wall", x=2.5, y=1.5)]

    assert sim.path_blocked(1.0, 1.5, 4.0, 1.5, 1.0)
    assert not sim.path_blocked(1.0, 1.5, 4.0, 1.5,
                                Panda3DRenderer.WALL_H + 0.5)


def test_collision_height_matches_what_the_range_sensor_casts_against():
    """The sensor and the physics must agree on how tall an obstacle is.

    dsim.range clips a wall hit at WALL_H; a collision test that ignored z
    crashed the vehicle into geometry the range sensor reported as clear air.
    """
    sim = _sim()
    sim.map.objects = [SimpleNamespace(kind="wall", x=2.5, y=1.5)]
    above = Panda3DRenderer.WALL_H + 1.0
    pose = range_module.Pose(1.0, 1.5, above, 90.0)
    intrinsics = range_module.Intrinsics(32, 24, 20.0, 20.0, 16.0, 12.0)

    ranges, _ = range_module.raycast_map(sim.map, pose, intrinsics, stride=4)

    assert not np.isfinite(ranges).any()
    assert not sim.is_blocked(2.5, 1.5, above)
