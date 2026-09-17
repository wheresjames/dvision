"""Vehicle protocol, coordinate, and simulator behaviour contracts."""

from types import SimpleNamespace

import pytest

import dsim.dsim as dsim_module
from dsim.dsim import DroneSimulator, DroneState
from dsim.realism import Realism
from dvision2_common import (COMMAND_RESULT_HISTORY, local_ned_to_map,
                             map_to_local_ned, parse_command_results)


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def advance(self, seconds: float) -> None:
        self.now += seconds


def make_sim(monkeypatch, *, armed: bool = False, setpoint_timeout: float = 0.0,
             lease_timeout: float = 100.0) -> tuple[DroneSimulator, Clock]:
    clock = Clock()
    monkeypatch.setattr(dsim_module.time, "monotonic", lambda: clock.now)
    sim = DroneSimulator.__new__(DroneSimulator)
    sim.args = SimpleNamespace(
        id="vehicle-contract", origin_lat=52.52, origin_lon=13.405, origin_alt=34.0,
        width=640, height=480, fps=30, setpoint_timeout=setpoint_timeout,
        control_lease_timeout=lease_timeout, max_speed_mps=5.0,
        max_accel_mps2=4.0,
    )
    sim.map = SimpleNamespace(path="contract.map", width=40.0, height=30.0,
                              objects=[])
    sim.start_x = 20.0
    sim.start_y = 15.0
    sim.start_alt = 1.5
    sim.start_yaw = dsim_module.compass_heading_to_sim_yaw(0.0)
    sim.target_x = sim.target_y = None
    sim.started = clock.now
    sim.report_root = "reports/contract"
    sim.crash_pos = None
    sim.status = None
    sim.state = DroneState(sim.start_x, sim.start_y, sim.start_alt,
                           yaw_deg=sim.start_yaw)
    acquire(sim)
    if armed:
        command(sim, "arm", armed=True)
    return sim, clock


def acquire(sim: DroneSimulator, source: str = "test", lease: str = "lease") -> None:
    sim.apply_command({"type": "acquire_control", "source_id": source,
                       "lease_id": lease, "request_id": "acquire"})


def command(sim: DroneSimulator, typ: str, **fields) -> None:
    sim.apply_command({"type": typ, "source_id": "test", "lease_id": "lease",
                       "request_id": f"request-{typ}", **fields})


def advance(sim: DroneSimulator, clock: Clock, seconds: float, dt: float = 0.05) -> None:
    for _ in range(round(seconds / dt)):
        clock.advance(dt)
        sim.integrate(dt)


def test_map_local_ned_round_trip_and_cardinal_signs() -> None:
    assert map_to_local_ned(20, 15, 2, 40, 30) == (0, 0, -2)
    assert map_to_local_ned(21, 14, 2, 40, 30) == (1, 1, -2)
    ned = map_to_local_ned(7.25, 22.5, 3.0, 40, 30)
    assert local_ned_to_map(*ned, 40, 30) == pytest.approx((7.25, 22.5, 3.0))


@pytest.mark.parametrize("heading,dx,dy", [
    (0.0, 0.0, -3.0), (90.0, 3.0, 0.0),
    (180.0, 0.0, 3.0), (270.0, -3.0, 0.0),
])
def test_map_position_targets_converge_from_cardinal_headings(
    monkeypatch, heading: float, dx: float, dy: float,
) -> None:
    sim, clock = make_sim(monkeypatch, armed=True)
    sim.state.yaw_deg = dsim_module.compass_heading_to_sim_yaw(heading)
    command(sim, "position_target", frame="map", x=20 + dx, y=15 + dy, z=2.0,
            heading_deg=heading, max_speed_mps=1.5)
    assert sim.state.result_accepted
    advance(sim, clock, 8.0)
    assert (sim.state.x, sim.state.y, sim.state.z) == pytest.approx(
        (20 + dx, 15 + dy, 2.0), abs=0.03)


def test_local_ned_target_uses_same_controller(monkeypatch) -> None:
    sim, clock = make_sim(monkeypatch, armed=True)
    command(sim, "position_target", frame="local_ned", north_m=2.0,
            east_m=3.0, down_m=-2.5, heading_deg=90.0, max_speed_mps=2.0)
    advance(sim, clock, 8.0)
    assert (sim.state.x, sim.state.y, sim.state.z) == pytest.approx(
        (23.0, 13.0, 2.5), abs=0.04)
    assert dsim_module.sim_yaw_to_compass_heading(sim.state.yaw_deg) == pytest.approx(
        90.0, abs=0.1)


def test_target_arbitration_and_rejections(monkeypatch) -> None:
    sim, _ = make_sim(monkeypatch, armed=True)
    command(sim, "position_target", frame="map", x=22, y=15, z=1.5,
            heading_deg=0, max_speed_mps=1)
    command(sim, "velocity", forward_mps=0.5)
    assert sim.state.result_accepted and sim.state.target_x is None
    command(sim, "position_target", frame="map", x=21, y=15, z=1.5,
            heading_deg=0, max_speed_mps=1)
    assert sim.state.result_accepted and sim.state.cmd_forward == 0.5
    sim.integrate(0.05)
    assert sim.state.cmd_forward != pytest.approx(0.5)

    sim.apply_command({"type": "velocity", "source_id": "intruder",
                       "lease_id": "wrong", "request_id": "bad",
                       "forward_mps": 1.0})
    assert not sim.state.result_accepted
    assert sim.state.result_request_id == "bad"
    command(sim, "position_target", frame="global", lat_deg=1, lon_deg=2,
            alt_m=3)
    assert not sim.state.result_accepted


def test_unarmed_target_is_rejected(monkeypatch) -> None:
    sim, _ = make_sim(monkeypatch)
    command(sim, "position_target", frame="map", x=21, y=15, z=1.5)
    assert not sim.state.result_accepted
    assert sim.state.result_reason == "vehicle is not armed"


def test_heartbeat_does_not_preserve_stale_setpoint(monkeypatch) -> None:
    """Two seconds of *flight* without a setpoint, not two seconds in the room.

    ``advance`` moves the simulated clock, which is the one the failsafe reads.
    The wall clock stays frozen by ``make_sim``, so a timer that crept back
    onto it would never fire and this would fail.
    """
    sim, clock = make_sim(monkeypatch, armed=True, setpoint_timeout=2.0)
    command(sim, "velocity", forward_mps=1.0)
    advance(sim, clock, 1.5)
    command(sim, "heartbeat")
    advance(sim, clock, 0.6)
    assert sim.state.mode == "HOLD"
    assert sim.state.failsafe_reason == "setpoint_timeout"
    assert sim.state.control_owner == "test"


def test_lease_expiry_holds_and_clears_owner(monkeypatch) -> None:
    sim, clock = make_sim(monkeypatch, armed=True, lease_timeout=1.0)
    command(sim, "velocity", forward_mps=1.0)
    advance(sim, clock, 1.1)
    assert sim.state.mode == "HOLD"
    assert sim.state.failsafe_reason == "control_lease_expired"
    assert sim.state.control_owner == ""


def test_origin_is_disarmed_only_and_result_is_correlated(monkeypatch) -> None:
    sim, _ = make_sim(monkeypatch)
    command(sim, "set_origin", lat_deg=1.0, lon_deg=2.0, alt_m=3.0)
    assert sim.state.result_request_id == "request-set_origin"
    assert sim.state.result_accepted
    assert (sim.args.origin_lat, sim.args.origin_lon, sim.args.origin_alt) == (1, 2, 3)
    command(sim, "arm", armed=True)
    command(sim, "set_origin", lat_deg=4.0, lon_deg=5.0, alt_m=6.0)
    assert not sim.state.result_accepted
    assert sim.state.result_reason == "set_origin requires disarmed"


def test_rtl_returns_home_and_lands(monkeypatch) -> None:
    sim, clock = make_sim(monkeypatch, armed=True, lease_timeout=1000.0)
    sim.state.x, sim.state.y = 25.0, 19.0
    command(sim, "rtl")
    advance(sim, clock, 30.0)
    assert not sim.state.armed
    assert sim.state.mode == "DISARMED"
    assert (sim.state.x, sim.state.y) == pytest.approx((20.0, 15.0), abs=0.12)


def test_status_advertises_the_vehicle_contract(monkeypatch) -> None:
    sim, _ = make_sim(monkeypatch, armed=True, setpoint_timeout=2.0)
    values = sim.status_fields()
    assert values["vehicle.frames"] == "map,local_ned"
    assert values["vehicle.accepts_position"] == "1"
    assert values["vehicle.setpoint_timeout_s"] == "2.000"
    assert values["control.owner"] == "test"
    assert values["home.lat_deg"]


# ---------------------------------------------------------------------------
# Command acknowledgement under more than one client
# ---------------------------------------------------------------------------

def test_a_second_client_in_the_same_tick_does_not_erase_the_first_result(monkeypatch):
    """A whole command queue is drained per tick, but status is published once.

    The single ``command.result.*`` slot only ever remembers the last command
    of that tick, which is how a client whose command had been accepted sat
    waiting for an acknowledgement that had already been overwritten.
    """
    sim, _ = make_sim(monkeypatch)
    command(sim, "heartbeat")                       # request-heartbeat, accepted
    sim.apply_command({"type": "velocity", "source_id": "other",
                       "lease_id": "no-lease", "request_id": "intruder",
                       "forward_mps": 1.0})         # rejected, and later
    values = sim.status_fields()

    assert values["command.result.request_id"] == "intruder"   # the slot moved on
    results = parse_command_results(values["command.results"])
    assert results["request-heartbeat"] == (True, "")
    assert results["intruder"] == (False, "control lease required")


def test_result_history_is_bounded_and_keeps_the_newest(monkeypatch):
    sim, _ = make_sim(monkeypatch)
    for n in range(COMMAND_RESULT_HISTORY + 4):
        sim.apply_command({"type": "heartbeat", "source_id": "test",
                           "lease_id": "lease", "request_id": f"r{n}"})
    results = parse_command_results(sim.status_fields()["command.results"])

    assert len(results) == COMMAND_RESULT_HISTORY
    assert "r0" not in results
    assert f"r{COMMAND_RESULT_HISTORY + 3}" in results


def test_reset_keeps_the_acknowledgements_other_clients_are_waiting_for(monkeypatch):
    """``reset`` replaces the whole vehicle state, which the history outlives."""
    sim, _ = make_sim(monkeypatch)
    command(sim, "heartbeat")
    sim.apply_command({"type": "reset", "source_id": "test",
                       "lease_id": "lease", "request_id": "reset-1"})
    results = parse_command_results(sim.status_fields()["command.results"])

    assert "request-heartbeat" in results
    assert results["reset-1"] == (True, "")
    assert sim.state.status_message == "reset"      # unchanged by the recording


def test_reset_requires_the_control_lease(monkeypatch):
    """A reset moves the vehicle further than any motion command does."""
    sim, _ = make_sim(monkeypatch, armed=True)
    command(sim, "velocity", forward_mps=1.0)
    sim.state.x = 5.0

    sim.apply_command({"type": "reset", "source_id": "intruder",
                       "lease_id": "other", "request_id": "reset-2"})

    assert sim.state.result_accepted is False
    assert sim.state.result_reason == "control lease required"
    assert sim.state.x == 5.0
    assert sim.state.control_owner == "test"


def test_reset_recovers_a_crashed_vehicle_for_the_lease_holder(monkeypatch):
    """The lease check sits above the crashed gate, so recovery still works."""
    sim, _ = make_sim(monkeypatch, armed=True)
    sim.crash()

    command(sim, "reset")

    assert sim.state.crashed is False
    assert sim.state.x == sim.start_x
    assert sim.crash_pos is None


def test_results_without_a_request_id_are_not_published_as_one(monkeypatch):
    """dfgb-era clients send no request id; their results address nobody."""
    sim, _ = make_sim(monkeypatch)
    before = sim.status_fields()["command.results"]
    sim.apply_command({"type": "heartbeat", "source_id": "test", "lease_id": "lease"})

    assert sim.status_fields()["command.results"] == before


def test_gps_altitude_carries_the_fix_error_and_not_the_barometer(monkeypatch):
    """``drone.alt_m`` is the GNSS triple's altitude, so it has one noise model.

    It used to be run through the barometer as well as the GPS vertical error,
    which put two independent draws on one reading -- so drone.alt_m minus
    drone.z_m did not come to origin.alt_m for any client that checked, and the
    two disagreed by more under a noisier profile.
    """
    sim, _ = make_sim(monkeypatch)
    sim.realism = Realism.from_settings({"gps": "off", "sensor_noise": "none",
                                         "realism_seed": 7})
    sim.state.z = 5.0

    values = sim.status_fields()

    # GPS off means no fix error, and no sensor noise means no barometer draw,
    # so the two altitudes differ by exactly the origin.
    assert (float(values["drone.alt_m"]) - float(values["drone.z_m"])
            == pytest.approx(float(values["origin.alt_m"]), abs=1e-6))


def test_gps_altitude_is_not_perturbed_twice_under_sensor_noise(monkeypatch):
    sim, _ = make_sim(monkeypatch)
    sim.realism = Realism.from_settings({"gps": "off", "sensor_noise": "heavy",
                                         "realism_seed": 7})
    sim.state.z = 5.0
    for _ in range(50):
        sim.realism.update(0.05)

    values = sim.status_fields()

    # z_m carries the barometer; alt_m carries the (here disabled) fix error.
    # Only one of the two is perturbed, so their difference is the origin plus
    # exactly one barometer draw rather than two.
    baro = float(values["drone.z_m"]) - sim.state.z
    assert baro != 0.0
    assert (float(values["drone.alt_m"]) - float(values["drone.z_m"])
            == pytest.approx(float(values["origin.alt_m"]) - baro, abs=1e-6))
