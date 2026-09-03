"""Nothing the vehicle does may depend on how busy the machine is.

The simulator advances on a timestep and the tests drive it on a fixed one, so
any behaviour measured against ``time.monotonic()`` instead is a function of
load rather than of flight. That is not a flaky test -- it is a wrong answer
that happens to be right when the machine is quiet.

Two bugs of exactly this shape lived here: the guided setpoint failsafe fired
after two seconds *in the room* rather than two seconds of flight, and the
optical-flow detector turned expansion into a distance using the wall time
between two frames rather than the distance the camera actually travelled.
"""

from __future__ import annotations

import numpy as np
import pytest

import dsim.dsim as dsim_module
from daic.local_map import LocalOccupancyMap, pose_from_status, _OCCUPIED
from daic.optical_flow_avoidance import OpticalFlowAvoidance
from dtest.calibration_scene import CHAIN_FRONT_MAP
from dtest.deterministic import DeterministicSim


class FrozenClock:
    """A wall clock that can be stopped, or made to lurch."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def read(self) -> float:
        return self.now

    def jump(self, seconds: float) -> None:
        self.now += seconds


# ---------------------------------------------------------------------------
# The vehicle's clock
# ---------------------------------------------------------------------------

def test_the_published_clock_counts_flight_not_wall_time(monkeypatch) -> None:
    clock = FrozenClock()
    monkeypatch.setattr(dsim_module.time, "monotonic", clock.read)
    sim = DeterministicSim(map_path=CHAIN_FRONT_MAP)

    for tick in range(1, 4):
        sim.step(0.1)
        clock.jump(7.0)      # the machine was busy; the vehicle was not
        assert float(sim.read_telemetry()["sim.time_s"]) == pytest.approx(0.1 * tick)


def test_the_guided_failsafe_counts_seconds_of_flight(monkeypatch) -> None:
    """A stalled loop must not look like a client that stopped sending."""
    from tests.test_dway_vehicle_contract import advance, command, make_sim

    sim, clock = make_sim(monkeypatch, armed=True, setpoint_timeout=2.0)
    command(sim, "velocity", forward_mps=1.0)

    # Wall time lurches far past the timeout while the vehicle flies for 1 s.
    clock.advance(30.0)
    advance(sim, clock, 1.0)
    assert sim.state.mode == "GUIDED", "the failsafe fired on wall time"

    advance(sim, clock, 1.2)
    assert sim.state.mode == "HOLD"
    assert sim.state.failsafe_reason == "setpoint_timeout"


# ---------------------------------------------------------------------------
# The perception chain
# ---------------------------------------------------------------------------

def _fly(monkeypatch, *, wall_jump_s: float):
    """Fly the front-wall fixture with the wall clock lurching each tick."""
    import daic.optical_flow_avoidance as ofa

    clock = FrozenClock()
    monkeypatch.setattr(ofa.time, "monotonic", clock.read)
    sim = DeterministicSim(map_path=CHAIN_FRONT_MAP, heading_deg=0.0)
    detector, local_map = OpticalFlowAvoidance(), LocalOccupancyMap()
    sim.send_body_velocity(0.8, 0.0, 0.0, 0.0)

    pose = None
    for _ in range(30):
        sim.step(0.1)
        telemetry = sim.read_telemetry()
        detector.set_motion_from_status(telemetry)
        clock.jump(wall_jump_s)
        reading = detector.detect_obstacles(sim.render().copy())
        pose = pose_from_status(telemetry)
        local_map.update(pose, reading)
    occupied = sorted(cell for cell, value in local_map._cells.items()
                      if value >= _OCCUPIED)
    return occupied, pose


@pytest.mark.parametrize("wall_jump_s", (0.001, 0.1, 2.0))
def test_the_chain_learns_the_same_map_however_slow_the_loop_was(
        monkeypatch, wall_jump_s: float) -> None:
    """Same frames and same flight, so the same cells -- at any loop speed."""
    baseline, _ = _fly(monkeypatch, wall_jump_s=0.05)
    occupied, pose = _fly(monkeypatch, wall_jump_s=wall_jump_s)

    assert occupied, "the chain learned nothing"
    assert occupied == baseline, (
        f"a loop taking {wall_jump_s}s per tick learned "
        f"{len(occupied)} cells against the baseline's {len(baseline)}; "
        "range is being inferred from wall time")


def test_the_detector_takes_its_interval_from_the_vehicle(monkeypatch) -> None:
    """The unit underneath: same frames, same vehicle clock, same answer."""
    import daic.optical_flow_avoidance as ofa

    rng = np.random.default_rng(7)
    frames = [rng.integers(0, 255, (120, 160, 3), dtype=np.uint8) for _ in range(3)]
    status = [{"drone.vx_mps": "0.0", "drone.vy_mps": "-0.8",
               "drone.heading_deg": "0.0", "sim.time_s": f"{0.1 * i:.3f}"}
              for i in range(3)]

    def run(wall_jump_s: float):
        clock = FrozenClock()
        monkeypatch.setattr(ofa.time, "monotonic", clock.read)
        detector = OpticalFlowAvoidance()
        out = []
        for frame, telemetry in zip(frames, status):
            detector.set_motion_from_status(telemetry)
            clock.jump(wall_jump_s)
            out.append(detector.detect_obstacles(frame.copy()))
        return [(s.front, s.left, s.right, s.front_range_m) for s in out]

    assert run(0.01) == run(3.0)


def test_an_explicit_interval_still_wins(monkeypatch) -> None:
    """A caller that knows the frame interval exactly can say so."""
    import daic.optical_flow_avoidance as ofa

    seen: list[float | None] = []
    original = ofa._flow_to_sectors
    monkeypatch.setattr(ofa, "_flow_to_sectors",
                        lambda flow, speed=0.0, dt_s=None: (
                            seen.append(dt_s) or original(flow, speed, dt_s)))

    rng = np.random.default_rng(3)
    frame = rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)
    detector = OpticalFlowAvoidance()
    detector.set_motion_from_status(
        {"drone.vx_mps": "0.0", "drone.vy_mps": "-0.8",
         "drone.heading_deg": "0.0", "sim.time_s": "0.000"})
    detector.detect_obstacles(frame.copy())
    detector.set_motion_from_status(
        {"drone.vx_mps": "0.0", "drone.vy_mps": "-0.8",
         "drone.heading_deg": "0.0", "sim.time_s": "0.100"})
    detector.detect_obstacles(frame.copy(), dt_s=0.25)

    assert seen == [0.25]


def test_a_vehicle_that_publishes_no_clock_falls_back_to_the_wall(
        monkeypatch) -> None:
    """A real link reporting no time leaves the wall clock as the only estimate."""
    import daic.optical_flow_avoidance as ofa

    seen: list[float | None] = []
    original = ofa._flow_to_sectors
    monkeypatch.setattr(ofa, "_flow_to_sectors",
                        lambda flow, speed=0.0, dt_s=None: (
                            seen.append(dt_s) or original(flow, speed, dt_s)))
    clock = FrozenClock()
    monkeypatch.setattr(ofa.time, "monotonic", clock.read)

    rng = np.random.default_rng(11)
    frame = rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)
    detector = OpticalFlowAvoidance()
    no_clock = {"drone.vx_mps": "0.0", "drone.vy_mps": "-0.8",
                "drone.heading_deg": "0.0"}
    detector.set_motion_from_status(no_clock)
    detector.detect_obstacles(frame.copy())
    clock.jump(0.4)
    detector.set_motion_from_status(no_clock)
    detector.detect_obstacles(frame.copy())

    assert seen == [pytest.approx(0.4)]
