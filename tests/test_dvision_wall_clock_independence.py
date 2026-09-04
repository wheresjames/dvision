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

import pathlib

import numpy as np
import pytest

import dsim.dsim as dsim_module
from daic.local_map import LocalOccupancyMap, pose_from_status, _OCCUPIED
from daic.optical_flow_avoidance import OpticalFlowAvoidance
from dtest.calibration_scene import CHAIN_FRONT_MAP
from dtest.deterministic import DeterministicSim
from dvision2_common import load_map

ROOT = pathlib.Path(__file__).resolve().parents[1]


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


# ---------------------------------------------------------------------------
# Simulated-time speed control
# ---------------------------------------------------------------------------

class _Ticker:
    """A wall clock the test advances by hand, and a sleep that obeys it."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.slept: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += max(0.0, seconds)

    def work(self, seconds: float) -> None:
        """Time the loop spends rendering and publishing, per tick."""
        self.now += seconds


def _speed_sim(monkeypatch, ticker, *extra_args, work_s: float = 0.0):
    """A simulator whose loop runs against ``ticker`` and renders nothing."""
    sim = make_sim(monkeypatch, ["--frames", "10", "--no-ui", *extra_args])
    monkeypatch.setattr(dsim_module.time, "monotonic", ticker.monotonic)
    monkeypatch.setattr(dsim_module.time, "sleep", ticker.sleep)
    monkeypatch.setattr(sim, "_init_renderer", lambda: None)
    monkeypatch.setattr(sim, "open_ipc", lambda: None)
    monkeypatch.setattr(sim, "close", lambda: None)
    published: list[float] = []
    monkeypatch.setattr(sim, "publish_frame",
                        lambda now: (published.append(sim.sim_time_s),
                                     ticker.work(work_s)))
    monkeypatch.setattr(sim, "publish_status", lambda **k: None)
    return sim, published


def make_sim(monkeypatch, argv):
    from dsim.dsim import DroneSimulator, DroneState, parse_args
    sim = DroneSimulator.__new__(DroneSimulator)
    sim.args = parse_args(["--id", "speed-test", *argv])
    sim.map = load_map(ROOT / "assets/maps/maze_001.txt")
    sim.start_x, sim.start_y, sim.start_alt = sim.map.start_x, sim.map.start_y, 1.5
    sim.start_yaw = 270.0
    sim.target_x = sim.target_y = None
    sim.started = 0.0
    sim.report_root = "/tmp/speed-test"
    sim.crash_pos = None
    sim.status = sim.command = sim.video = sim.module_bus = sim.ui = None
    sim.p3d = None
    sim.running = True
    sim.sim_time_s = 0.0
    sim.flight_positions = []
    sim.state = DroneState(sim.start_x, sim.start_y, 1.5)
    return sim


def test_real_time_keeps_the_measured_timestep(monkeypatch):
    """Absent the option, the loop is exactly what it always was.

    Real time must track reality: when the host stalls, the vehicle is owed
    the truth about how long the step actually took. Ten ticks costing 50 ms
    each therefore advance about half a second of flight, not the third of a
    second a fixed 30 Hz step would have given.
    """
    ticker = _Ticker()
    start = ticker.now
    sim, _ = _speed_sim(monkeypatch, ticker, work_s=0.05)   # 50 ms of work
    sim.run()

    assert sim.args.sim_speed is None
    # Simulated time tracked wall time, which is what real time means.
    assert sim.sim_time_s == pytest.approx(ticker.now - start, abs=0.06)
    # And it is emphatically not the fixed-step figure.
    assert sim.sim_time_s > 0.4


def test_a_scaled_run_uses_a_fixed_timestep(monkeypatch):
    """A scaled clock is not a measurement of the room, so it is not measured."""
    ticker = _Ticker()
    sim, _ = _speed_sim(monkeypatch, ticker, "--sim-speed", "4", work_s=0.05)
    sim.run()

    # 10 ticks at 30 Hz, whatever the host was doing.
    assert sim.sim_time_s == pytest.approx(10.0 / 30.0, abs=1e-6)


@pytest.mark.parametrize("speed", [0.5, 2.0, 4.0])
def test_simulated_time_advances_at_the_configured_multiple(monkeypatch, speed):
    """Each tick advances a fixed step and waits that step divided by speed."""
    ticker = _Ticker()
    sim, _ = _speed_sim(monkeypatch, ticker, "--sim-speed", str(speed))
    sim.run()

    frame_period = 1.0 / 30.0
    assert sim.sim_time_s == pytest.approx(10 * frame_period, abs=1e-9)
    assert ticker.slept, "a paced run has to wait"
    for waited in ticker.slept:
        assert waited == pytest.approx(frame_period / speed, abs=1e-9)


def test_max_does_not_sleep_at_all(monkeypatch):
    ticker = _Ticker()
    sim, _ = _speed_sim(monkeypatch, ticker, "--sim-speed", "max")
    sim.run()

    assert ticker.slept == []
    assert sim.sim_time_s == pytest.approx(10.0 / 30.0, abs=1e-6)


def test_video_hz_preserves_the_simulated_interval_between_frames(monkeypatch):
    """Publishing less often must not change what a consumer sees, only when."""
    ticker = _Ticker()
    every, published = _speed_sim(monkeypatch, ticker, "--sim-speed", "max",
                                  "--video-hz", "6")
    every.run()

    # 30 Hz physics, 6 Hz video: one frame every fifth tick, so the simulated
    # gap between frames is 1/6 s -- exactly what --video-hz asked for.
    assert every._video_tick_divisor() == 5
    assert len(published) == 2
    gaps = [b - a for a, b in zip(published, published[1:])]
    assert all(g == pytest.approx(1.0 / 6.0, abs=1e-6) for g in gaps)


def test_video_hz_is_rejected_above_the_physics_rate():
    from dsim.dsim import parse_args
    with pytest.raises(SystemExit):
        parse_args(["--id", "x", "--fps", "30", "--video-hz", "60"])


def test_sim_speed_parsing():
    from dsim.dsim import UNPACED, parse_args
    assert parse_args(["--id", "x"]).sim_speed is None
    assert parse_args(["--id", "x", "--sim-speed", "3.5"]).sim_speed == 3.5
    assert parse_args(["--id", "x", "--sim-speed", "max"]).sim_speed == UNPACED
    assert parse_args(["--id", "x", "--sim-speed", "0"]).sim_speed == UNPACED
    for bad in ("-1", "abc", "nan", "-0.5"):
        with pytest.raises(SystemExit):
            parse_args(["--id", "x", "--sim-speed", bad])


def test_the_published_speed_tells_a_client_what_it_is_attached_to(monkeypatch):
    ticker = _Ticker()
    for argv, expected in (([], "1.0000"),
                           (["--sim-speed", "2.5"], "2.5000"),
                           (["--sim-speed", "max"], "0.0000")):
        sim, _ = _speed_sim(monkeypatch, ticker, *argv)
        assert sim.status_fields()["sim.speed"] == expected


def test_the_speed_can_be_changed_while_the_simulation_runs(monkeypatch):
    """The loop reads the speed each tick, so a change lands on the next one.

    Nothing downstream reads the rate -- clients ask whether enough simulated
    time has passed, which has the same answer however fast the clock turns --
    so changing it mid-flight is sound, and the monitor offers it.
    """
    ticker = _Ticker()
    sim, _ = _speed_sim(monkeypatch, ticker, "--sim-speed", "2")
    frame_period = 1.0 / 30.0

    # Halfway through, ask for twice the speed.
    original_step = sim.step
    ticks = {"n": 0}

    def counting_step(dt, **kwargs):
        ticks["n"] += 1
        if ticks["n"] == 5:
            sim.set_sim_speed(4.0)
        return original_step(dt, **kwargs)

    monkeypatch.setattr(sim, "step", counting_step)
    sim.run()

    waits = ticker.slept
    assert waits[0] == pytest.approx(frame_period / 2.0, abs=1e-9)
    assert waits[-1] == pytest.approx(frame_period / 4.0, abs=1e-9)
    assert sim.status_fields()["sim.speed"] == "4.0000"


def test_switching_to_real_time_mid_run_restores_the_measured_step(monkeypatch):
    """Real time has to measure again the moment it is selected."""
    ticker = _Ticker()
    sim, _ = _speed_sim(monkeypatch, ticker, "--sim-speed", "max", work_s=0.05)
    steps: list[float] = []
    original_step = sim.step

    def recording_step(dt, **kwargs):
        steps.append(dt)
        if len(steps) == 4:
            sim.set_sim_speed(None)
        return original_step(dt, **kwargs)

    monkeypatch.setattr(sim, "step", recording_step)
    sim.run()

    assert all(d == pytest.approx(1.0 / 30.0) for d in steps[:4])   # fixed
    assert steps[-1] == pytest.approx(0.05, abs=1e-6)               # measured
    assert sim.status_fields()["sim.speed"] == "1.0000"


def test_set_sim_speed_refuses_a_value_it_cannot_run(monkeypatch):
    ticker = _Ticker()
    sim, _ = _speed_sim(monkeypatch, ticker, "--sim-speed", "2")
    for bad in (-1.0, float("nan"), float("-inf")):
        with pytest.raises(ValueError):
            sim.set_sim_speed(bad)
    # A rejected change leaves the simulation running exactly as it was.
    assert sim.args.sim_speed == 2.0


def test_the_speed_menu_shows_what_the_simulator_is_actually_running(monkeypatch):
    """Including a command-line speed that is not one of the presets."""
    from dsim.dsim import TopDownUi

    ticker = _Ticker()
    for argv, expected in (([], "real time"),
                           (["--sim-speed", "4"], "4x"),
                           (["--sim-speed", "max"], "max"),
                           (["--sim-speed", "3.5"], "3.5x")):
        sim, _ = _speed_sim(monkeypatch, ticker, *argv)
        assert TopDownUi._speed_label(sim) == expected


def test_every_speed_menu_entry_is_one_the_simulator_accepts(monkeypatch):
    ticker = _Ticker()
    sim, _ = _speed_sim(monkeypatch, ticker)
    for name, value in sim.SPEED_CHOICES:
        sim.set_sim_speed(value)          # must not raise
        assert sim.args.sim_speed == value, name
