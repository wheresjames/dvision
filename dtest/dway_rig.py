"""A deterministic dway flight against a real simulator, without shared memory.

The vehicle is a real ``DroneSimulator`` -- the same physics, command handling
and status keys the process runs -- driven through ``DsimLink`` over an
in-process transport and a controlled clock. That keeps flights repeatable
while still exercising the wire format, the acknowledgement correlation and
the control lease.

It lives here rather than beside one test file because both the flight
outcomes and the environment/fault matrix fly the same way, and because the
repeatability harness flies it too.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import dsim.dsim as dsim_module
from dsim.dsim import DroneSimulator, DroneState, compass_heading_to_sim_yaw
from dsim.realism import REALISM_DEFAULTS
from dvision2_common import decode_command, load_map
from dway.link import DsimLink
from dway.mission import (
    TERMINAL_STATES, Mission, MissionConfig, MissionState, mission_report_dir,
)
from dway.report import FlightRecorder
from dway.tour import load_tour

ROOT = Path(__file__).resolve().parents[1]
FORWARD_TOUR = ROOT / "assets/tours/maze_012.forward.v1.json"
MAZE_012 = ROOT / "assets/maps/maze_012.txt"

# A pose on the tour's own corridor. The map's drone start is walled off from
# the first waypoint, and nothing in dway avoids obstacles, so a flight starts
# where its first leg is actually flyable.
CLEAR_START = (34.5, 3.5)
FIXED_DT = 0.05
SETPOINT_TIMEOUT_S = 2.0


class Clock:
    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def read(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += float(seconds)


class LoopbackTransport:
    """The simulator's own command and status handling, minus shared memory.

    Status changes bump an epoch exactly as the key/value store does, so the
    link's freshness tracking is the real one; ``freeze`` stops publication the
    way a stalled or dead simulator would.
    """

    def __init__(self, sim: DroneSimulator) -> None:
        self.sim = sim
        self.connected = True
        self.frozen = False
        self.rejected_writes = False
        self._epoch = 0
        self._values: dict[str, str] = {}

    def connect(self) -> bool:
        return self.connected

    def write(self, payload: str) -> bool:
        if self.rejected_writes or not self.connected:
            return False
        command = decode_command(payload)
        if command is None:
            return False
        self.sim.apply_command(command)
        return True

    def status(self) -> dict[str, str]:
        if not self.frozen and self.connected:
            # Not forced: telemetry latency is part of what a flight has
            # to cope with, and None means nothing is due yet.
            values = self.sim.published_fields()
            if values is not None and values != self._values:
                self._values = values
                self._epoch += 1
        return dict(self._values)

    def epoch(self) -> int:
        return self._epoch

    def close(self) -> None:
        self.connected = False


def build_sim(clock: Clock, *, start=CLEAR_START, heading_deg: float = 0.0,
              map_path: Path = MAZE_012, report_root: Path,
              setpoint_timeout: float = SETPOINT_TIMEOUT_S,
              lease_timeout: float = 3.0,
              realism: dict | None = None) -> DroneSimulator:
    """A simulator built field by field, so no test depends on UI construction.

    ``realism`` names the knobs from ``REALISM_DEFAULTS``; anything omitted
    keeps its shipped default, which is the realism-off baseline.
    """
    sim = DroneSimulator.__new__(DroneSimulator)
    settings = dict(REALISM_DEFAULTS)
    settings.update(realism or {})
    sim.args = SimpleNamespace(
        id="dway-test", origin_lat=52.52, origin_lon=13.405, origin_alt=34.0,
        width=640, height=480, fps=30, setpoint_timeout=setpoint_timeout,
        control_lease_timeout=lease_timeout, max_speed_mps=5.0,
        max_accel_mps2=4.0, **settings)
    sim.map = load_map(map_path)
    sim.start_x, sim.start_y = start
    sim.start_alt = 1.5
    sim.start_yaw = compass_heading_to_sim_yaw(heading_deg)
    sim.target_x = sim.target_y = None
    sim.started = clock.now
    sim.report_root = report_root
    sim.crash_pos = None
    sim.status = None
    sim.command = None
    sim.state = DroneState(start[0], start[1], sim.start_alt,
                           yaw_deg=sim.start_yaw)
    return sim


class Rig:
    """One simulator, one link, one mission, and a clock that drives them."""

    def __init__(self, monkeypatch, tmp_path: Path, *, tour_path: Path = FORWARD_TOUR,
                 strategy: str = "auto", start=CLEAR_START,
                 finish_action: str = "land", setpoint_timeout: float = SETPOINT_TIMEOUT_S,
                 lease_timeout: float = 3.0, autostart: bool = True,
                 realism: dict | None = None) -> None:
        self.clock = Clock()
        monkeypatch.setattr(dsim_module.time, "monotonic", self.clock.read)
        self.report_root = tmp_path / "run"
        self.sim = build_sim(self.clock, start=start, report_root=self.report_root,
                             setpoint_timeout=setpoint_timeout,
                             lease_timeout=lease_timeout, realism=realism)
        self.transport = LoopbackTransport(self.sim)
        self.link = DsimLink("dway-test", client_id="dway-test",
                             transport=self.transport, clock=self.clock.read,
                             sleep=self._sleep)
        self.tour = load_tour(tour_path)
        self.report_dir = mission_report_dir(self.link, "dway-test")
        self.mission = Mission(
            self.link, self.tour, root=ROOT,
            # In the rig the virtual clock is the only clock there is: it is
            # patched over dsim's monotonic, so it stands in for both roles.
            clock=self.clock.read, wall=self.clock.read,
            recorder=FlightRecorder(self.report_dir),
            config=MissionConfig(strategy=strategy, finish_action=finish_action,
                                 autostart=autostart))

    def _sleep(self, seconds: float) -> None:
        """Time spent waiting for an acknowledgement is time the vehicle flies.

        The link sleeps while a result is outstanding. Advancing only the clock
        would let mission time run ahead of the physics, which turns delayed
        telemetry into an apparent leg timeout that no real process would see.
        """
        self.clock.advance(seconds)
        self.sim.integrate(seconds)

    def step(self, dt: float = FIXED_DT) -> MissionState:
        state = self.mission.step(self.clock.now)
        self.clock.advance(dt)
        self.sim.integrate(dt)
        return state

    def fly(self, *, limit_s: float = 180.0, dt: float = FIXED_DT,
            until=None) -> MissionState:
        for _ in range(int(limit_s / dt)):
            if until is not None and until(self.mission):
                return self.mission.state
            if self.mission.state in TERMINAL_STATES:
                return self.mission.state
            self.step(dt)
        return self.mission.state

    def summary(self) -> dict:
        self.mission.write_report(self.report_dir)
        self.mission.recorder.close()
        return json.loads((self.report_dir / "summary.json").read_text())


def write_corridor_tour(path: Path, *, tour_id: str = "maze_012.corridor") -> Path:
    """A baseline tour flyable straight out of ``maze_012``'s own start pose.

    The committed maze_012 tours begin on the far side of a wall from the
    drone start and dway does not avoid obstacles, so anything that launches a
    real simulator flies this instead.
    """
    from dway.tour import map_content_sha

    path.write_text(json.dumps({
        "schema_version": 1,
        "tour_id": tour_id,
        "status": "applicable",
        "coordinate_frame": "map",
        "map": "assets/maps/maze_012.txt",
        "map_sha": map_content_sha(MAZE_012),
        "default_speed_mps": 1.0,
        "waypoint_tolerance_m": 0.15,
        "min_clearance_m": 0.4,
        "waypoints": [
            {"x": 8.5, "y": 1.5, "z": 1.5, "heading_deg": 90.0, "dwell_s": 0.2},
            {"x": 14.5, "y": 1.5, "z": 1.5, "heading_deg": 90.0, "dwell_s": 0.2},
        ],
    }), encoding="utf-8")
    return path
