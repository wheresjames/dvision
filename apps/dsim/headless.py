"""Deterministic, production-neutral in-process simulator driver."""

from __future__ import annotations

import hashlib
import uuid
from pathlib import Path

import numpy as np

from dsim.dsim import (
    DroneSimulator, DroneState, Panda3DRenderer,
    compass_heading_to_sim_yaw, parse_args, sim_yaw_to_compass_heading,
)
from dvision2_common import load_map

FIXED_DT = 0.05
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480
_RENDERER: tuple[tuple[str, int, int, str], Panda3DRenderer] | None = None

#: Longer than any harness run, so the single acquire at construction holds.
_NO_LEASE_TIMEOUT_S = 1e9


def map_content_sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def shared_renderer(sim_map, width: int = DEFAULT_WIDTH,
                    height: int = DEFAULT_HEIGHT, *, scene_preset: str = "legacy") -> Panda3DRenderer:
    global _RENDERER
    key = (map_content_sha(sim_map.path), int(width), int(height), scene_preset)
    if _RENDERER is not None:
        if _RENDERER[0] == key:
            return _RENDERER[1]
        _RENDERER[1].close()
        _RENDERER = None
    renderer = Panda3DRenderer(sim_map, width, height, scene_preset=scene_preset)
    _RENDERER = (key, renderer)
    return renderer


class HeadlessSimulator:
    """A fixed-timestep ``DroneSimulator`` without IPC, UI, or wall clock."""

    def __init__(self, *, map_path: Path, heading_deg: float = 0.0,
                 altitude_m: float = 1.5, armed: bool = True,
                 width: int = DEFAULT_WIDTH, height: int = DEFAULT_HEIGHT,
                 instance_id: str = "dsim-headless",
                 scene_preset: str = "legacy") -> None:
        self.width, self.height = int(width), int(height)
        self.scene_preset = scene_preset
        sim = DroneSimulator.__new__(DroneSimulator)
        sim.map = load_map(Path(map_path))
        sim.start_x, sim.start_y = sim.map.start_x, sim.map.start_y
        sim.start_alt = altitude_m
        sim.start_yaw = compass_heading_to_sim_yaw(heading_deg)
        sim.state = DroneState(
            sim.start_x, sim.start_y, altitude_m, yaw_deg=sim.start_yaw,
            armed=armed, mode="GUIDED" if armed else "DISARMED",
        )
        sim.crash_pos = None
        sim.started = 0.0
        sim.command = None
        # No streaming client and no lease heartbeat: this driver sends a
        # setpoint and steps physics. The guided failsafe and the control lease
        # exist to notice a client that has stopped talking, which is a
        # protocol behaviour covered by the vehicle-contract and process tests;
        # leaving them armed here would stop the vehicle mid-measurement and
        # report it as a physics or perception result.
        sim.args = parse_args([
            "--id", instance_id, "--width", str(self.width),
            "--height", str(self.height),
            "--setpoint-timeout", "0",
            "--control-lease-timeout", str(_NO_LEASE_TIMEOUT_S),
        ])
        sim.status = None
        sim.report_root = Path("/nonexistent/dsim-headless")
        self._source_id = instance_id
        self._lease_id = uuid.uuid4().hex
        sim.apply_command({
            "type": "acquire_control", "source_id": self._source_id,
            "lease_id": self._lease_id, "request_id": uuid.uuid4().hex,
        })
        self.sim = sim
        self._renderer: Panda3DRenderer | None = None

    @property
    def state(self) -> DroneState:
        return self.sim.state

    @property
    def heading_deg(self) -> float:
        return sim_yaw_to_compass_heading(self.sim.state.yaw_deg)

    @property
    def position(self) -> tuple[float, float]:
        return self.sim.state.x, self.sim.state.y

    def set_pose(self, x: float, y: float, z: float,
                 heading_deg: float) -> None:
        values = (x, y, z, heading_deg)
        if not all(np.isfinite(value) for value in values):
            raise ValueError("pose values must be finite")
        if not (0.0 <= x <= self.sim.map.width
                and 0.0 <= y <= self.sim.map.height and z >= 0.0):
            raise ValueError("pose is outside map bounds")
        if self.sim.is_blocked(x, y, z):
            raise ValueError("pose camera is inside collision geometry")
        state = self.sim.state
        state.x, state.y, state.z = float(x), float(y), float(z)
        state.yaw_deg = compass_heading_to_sim_yaw(heading_deg)
        state.roll_deg = state.pitch_deg = 0.0
        self.sim.zero_motion()

    def arm(self) -> None:
        self._command("arm", armed=True)

    def disarm(self) -> None:
        self._command("arm", armed=False)

    def zero(self) -> None:
        self._command("zero")

    def send_body_velocity(self, forward_mps: float, right_mps: float,
                           up_mps: float, yaw_rate_dps: float) -> None:
        self._command(
            "velocity", forward_mps=forward_mps,
            right_mps=right_mps, up_mps=up_mps,
            yaw_rate_dps=yaw_rate_dps,
        )

    def _command(self, command_type: str, **fields) -> None:
        self.sim.apply_command({
            "type": command_type, "source_id": self._source_id,
            "lease_id": self._lease_id, "request_id": uuid.uuid4().hex,
            **fields,
        })

    def read_telemetry(self) -> dict[str, str]:
        return self.sim.status_fields()

    def read_frame(self, *, newer_than: int | None = None,
                   timeout: float = 5.0) -> tuple[int, np.ndarray]:
        del newer_than, timeout
        return 0, self.render()

    def step(self, seconds: float, *, dt: float = FIXED_DT) -> None:
        for _ in range(max(1, int(round(seconds / dt)))):
            self.sim.step(dt)

    def impulse(self, seconds: float = 1.0, *, settle: float = 0.0,
                **command: float) -> tuple[float, float]:
        if settle > 0.0:
            self.zero()
            self.step(settle)
        x0, y0 = self.position
        self.send_body_velocity(
            command.get("forward_mps", 0.0), command.get("right_mps", 0.0),
            command.get("up_mps", 0.0), command.get("yaw_rate_dps", 0.0),
        )
        self.step(seconds)
        self.zero()
        x1, y1 = self.position
        return x1 - x0, y1 - y0

    def render(self, out: np.ndarray | None = None) -> np.ndarray:
        self._renderer = shared_renderer(self.sim.map, self.width, self.height,
                                         scene_preset=self.scene_preset)
        frame = out if out is not None else np.zeros(
            (self.height, self.width, 3), dtype=np.uint8
        )
        if frame.shape != (self.height, self.width, 3):
            raise ValueError("output frame has the wrong shape")
        self._renderer.render(self.sim.state, frame)
        return frame

    def render_range(self) -> np.ndarray:
        self._renderer = shared_renderer(self.sim.map, self.width, self.height,
                                         scene_preset=self.scene_preset)
        return self._renderer.render_range(self.sim.state)

    def close(self) -> None:
        self._renderer = None

    def __enter__(self) -> "HeadlessSimulator":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
