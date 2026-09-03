"""End-to-end tour flights against the simulator, and their safe outcomes.

The rig lives in ``dtest.dway_rig``: a real ``DroneSimulator`` driven through
``DsimLink`` over an in-process transport and a controlled clock, shared with
the environment matrix and the repeatability harness so every dway flight in
the suite is flown exactly one way.
"""

import json

import pytest

from dtest.dway_rig import (
    FIXED_DT, FORWARD_TOUR, MAZE_012, ROOT, Rig,
)
from dway.mission import MissionState
from dway.tour import load_tour

# ---------------------------------------------------------------------------
# Flying the committed tour
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("strategy,expected", [("auto", "position"),
                                               ("velocity", "velocity")])
def test_forward_tour_flies_end_to_end(monkeypatch, tmp_path, strategy, expected) -> None:
    rig = Rig(monkeypatch, tmp_path, strategy=strategy)
    assert rig.fly() is MissionState.COMPLETE, rig.mission.reason
    assert rig.mission.strategy.name == expected

    summary = rig.summary()
    assert summary["outcome"] == "complete"
    assert summary["partial"] is False
    assert summary["strategy"] == expected
    assert summary["waypoints_reached"] == summary["waypoint_count"] == 2
    assert summary["failsafes"] == []
    # The last waypoint was actually reached, not merely sequenced past.
    last = rig.tour.waypoints[-1]
    assert rig.sim.state.x == pytest.approx(last.x, abs=0.1)
    assert rig.sim.state.y == pytest.approx(last.y, abs=0.1)
    # Landing is the default finish action.
    assert not rig.sim.state.armed


def test_summary_carries_every_required_field(monkeypatch, tmp_path) -> None:
    rig = Rig(monkeypatch, tmp_path)
    rig.fly()
    summary = rig.summary()
    for key in ("schema_version", "tour_id", "outcome", "reason", "started_at",
                "duration_s", "strategy", "coordinate_frame", "waypoint_count",
                "waypoints", "path_length_m", "max_cross_track_error_m",
                "failsafes", "partial"):
        assert key in summary, key
    assert summary["coordinate_frame"] == "map"
    assert summary["path_length_m"] > 8.0
    for index, entry in enumerate(summary["waypoints"]):
        assert entry["index"] == index
        assert entry["target"]["frame"] == "map"
        assert entry["first_target_s"] is not None
        assert entry["arrival_s"] is not None
        assert entry["arrival_s"] >= entry["first_target_s"]
        assert entry["dwell_s"] >= rig.tour.waypoints[index].dwell_s
        assert entry["overshoot_m"] >= 0.0
        assert entry["max_cross_track_error_m"] >= 0.0
    assert (rig.report_dir / "track.png").exists()
    events = [json.loads(line) for line
              in (rig.report_dir / "flight.jsonl").read_text().splitlines()]
    kinds = {event["event"] for event in events}
    assert {"acquire_control", "preflight", "arm", "setpoint", "arrived"} <= kinds
    assert all("request_id" in event for event in events
               if event["event"] in ("setpoint", "arm", "acquire_control"))


def test_a_local_ned_tour_flies_the_same_ground_as_its_map_twin(
        monkeypatch, tmp_path) -> None:
    """The same two points, authored in the frame a real autopilot uses."""
    map_tour = load_tour(FORWARD_TOUR)
    # maze_012 is 39 x 29, so map (x, y) becomes (height/2 - y, x - width/2).
    ned = tmp_path / "forward_ned.json"
    ned.write_text(json.dumps({
        "schema_version": 1, "tour_id": "maze_012.forward.ned",
        "status": "applicable", "coordinate_frame": "local_ned",
        "default_speed_mps": 1.0, "waypoint_tolerance_m": 0.05,
        "waypoints": [
            {"north_m": 14.5 - point.y, "east_m": point.x - 19.5,
             "down_m": -point.z, "heading_deg": point.heading_deg,
             "dwell_s": point.dwell_s}
            for point in map_tour.waypoints
        ],
    }))
    rig = Rig(monkeypatch, tmp_path, tour_path=ned, finish_action="hold")
    assert rig.fly() is MissionState.COMPLETE, rig.mission.reason
    # The tour names no map; the vehicle's own map is what relates its
    # published pose to the local-NED frame the waypoints are written in.
    assert rig.tour.map_path is None and rig.tour.map_sha is None
    assert rig.mission.sim_map is not None
    assert rig.mission.sim_map.path == MAZE_012
    last = map_tour.waypoints[-1]
    assert rig.sim.state.x == pytest.approx(last.x, abs=0.1)
    assert rig.sim.state.y == pytest.approx(last.y, abs=0.1)
    summary = rig.summary()
    assert summary["coordinate_frame"] == "local_ned"
    assert summary["waypoints_reached"] == 2
    assert summary["waypoints"][0]["target"]["frame"] == "local_ned"
    assert (rig.report_dir / "track.png").exists()


def test_a_vehicle_with_no_map_cannot_fly_a_local_ned_tour(
        monkeypatch, tmp_path) -> None:
    ned = tmp_path / "unanchored.json"
    ned.write_text(json.dumps({
        "schema_version": 1, "tour_id": "unanchored",
        "coordinate_frame": "local_ned",
        "waypoints": [{"north_m": 1.0, "east_m": 1.0, "down_m": -1.5}],
    }))
    rig = Rig(monkeypatch, tmp_path, tour_path=ned)
    monkeypatch.setattr(rig.link, "map_path", lambda: "")
    assert rig.fly(limit_s=5.0) is MissionState.FAILED
    assert "no map is available" in rig.mission.reason
    assert rig.sim.state.control_owner == ""


def test_finish_action_hold_leaves_the_vehicle_flying(monkeypatch, tmp_path) -> None:
    rig = Rig(monkeypatch, tmp_path, finish_action="hold")
    assert rig.fly() is MissionState.COMPLETE, rig.mission.reason
    assert rig.sim.state.armed
    assert rig.sim.state.mode == "HOLD"
    assert rig.sim.state.z == pytest.approx(1.5, abs=0.1)


def test_pause_stops_the_vehicle_and_resume_restreams(monkeypatch, tmp_path) -> None:
    rig = Rig(monkeypatch, tmp_path)
    rig.fly(until=lambda m: m.state is MissionState.FLYING)
    rig.fly(limit_s=1.0, until=lambda m: False)
    assert rig.mission.pause()
    rig.step()
    assert rig.sim.state.mode == "HOLD"
    paused_at = rig.mission.elapsed()
    for _ in range(20):
        rig.step()
    assert rig.mission.elapsed() == pytest.approx(paused_at, abs=1e-6)
    assert rig.mission.resume()
    assert rig.fly() is MissionState.COMPLETE, rig.mission.reason


# ---------------------------------------------------------------------------
# Safe outcomes
# ---------------------------------------------------------------------------

def test_a_wrong_map_hash_fails_preflight_before_control_is_taken(
        monkeypatch, tmp_path) -> None:
    payload = json.loads(FORWARD_TOUR.read_text())
    payload["map_sha"] = "0" * 64
    wrong = tmp_path / "wrong_map.json"
    wrong.write_text(json.dumps(payload))
    rig = Rig(monkeypatch, tmp_path, tour_path=wrong)
    assert rig.fly(limit_s=5.0) is MissionState.FAILED
    assert "map hash mismatch" in rig.mission.reason
    assert rig.sim.state.control_owner == ""
    assert not rig.sim.state.armed
    assert rig.summary()["outcome"] == "failed"


def test_an_obstructed_first_leg_is_refused(monkeypatch, tmp_path) -> None:
    rig = Rig(monkeypatch, tmp_path, start=(1.5, 1.5))
    assert rig.fly(limit_s=5.0) is MissionState.FAILED
    assert "leg 0 passes through map geometry" in rig.mission.reason
    assert not rig.sim.state.armed


def test_stale_state_holds_the_vehicle_and_fails_with_the_reason(
        monkeypatch, tmp_path) -> None:
    rig = Rig(monkeypatch, tmp_path)
    rig.fly(until=lambda m: m.state is MissionState.FLYING)
    rig.fly(limit_s=1.0, until=lambda m: False)
    # The simulator stops publishing: the pose the follower holds goes older
    # than the tour allows, and flying on it is exactly what must not happen.
    rig.transport.frozen = True
    rig.clock.advance(1.0)
    assert rig.fly(limit_s=5.0) is MissionState.FAILED
    assert "vehicle state is" in rig.mission.reason and "old" in rig.mission.reason
    assert rig.sim.state.mode == "HOLD"
    summary = rig.summary()
    assert summary["outcome"] == "failed"
    assert summary["partial"] is True


def test_a_rejected_setpoint_fails_without_retrying(monkeypatch, tmp_path) -> None:
    rig = Rig(monkeypatch, tmp_path)
    rig.fly(until=lambda m: m.state is MissionState.FLYING)
    # The vehicle drops this client's lease; the next setpoint is refused.
    rig.sim._clear_lease()
    assert rig.fly(limit_s=5.0) is MissionState.FAILED
    assert "control lease required" in rig.mission.reason
    assert rig.summary()["outcome"] == "failed"


def test_losing_the_lease_to_another_client_ends_the_flight(
        monkeypatch, tmp_path) -> None:
    rig = Rig(monkeypatch, tmp_path, lease_timeout=1.0)
    rig.fly(until=lambda m: m.state is MissionState.FLYING)
    # A stalled follower stops renewing, the vehicle fails safe, and another
    # client takes control.
    rig.clock.advance(2.0)
    rig.sim.integrate(FIXED_DT)
    rig.sim.apply_command({"type": "acquire_control", "source_id": "intruder",
                           "lease_id": "other", "request_id": "steal"})
    assert rig.fly(limit_s=5.0) is MissionState.FAILED
    assert "lease" in rig.mission.reason
    assert rig.sim.state.mode == "HOLD"


def test_a_leg_that_runs_out_of_time_fails_and_holds(monkeypatch, tmp_path) -> None:
    payload = json.loads(FORWARD_TOUR.read_text())
    payload["leg_timeout_s"] = 1.0
    slow = tmp_path / "impatient.json"
    slow.write_text(json.dumps(payload))
    rig = Rig(monkeypatch, tmp_path, tour_path=slow)
    assert rig.fly(limit_s=20.0) is MissionState.FAILED
    assert "leg timeout" in rig.mission.reason
    assert rig.sim.state.mode == "HOLD"
    summary = rig.summary()
    assert summary["partial"] is True
    assert summary["waypoints_reached"] == 0


def test_shutdown_holds_releases_control_and_leaves_a_partial_report(
        monkeypatch, tmp_path) -> None:
    rig = Rig(monkeypatch, tmp_path)
    rig.fly(until=lambda m: m.state is MissionState.FLYING)
    rig.fly(limit_s=2.0, until=lambda m: False)
    assert rig.sim.state.armed
    rig.mission.abort("shutdown")
    summary = rig.summary()
    assert summary["outcome"] == "aborted"
    assert summary["reason"] == "shutdown"
    assert summary["partial"] is True
    assert rig.sim.state.mode == "HOLD"
    # An airborne vehicle is held, never disarmed, and control is handed back.
    assert rig.sim.state.armed
    assert rig.sim.state.control_owner == ""


def test_a_too_slow_setpoint_stream_is_refused_at_preflight(
        monkeypatch, tmp_path) -> None:
    rig = Rig(monkeypatch, tmp_path)
    rig.mission.config.stream_hz = 0.5
    assert rig.fly(limit_s=5.0) is MissionState.FAILED
    assert "too slow" in rig.mission.reason
    assert rig.sim.state.control_owner == ""
