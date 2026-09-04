"""Transport-neutral vehicle contract used by dway and future real links."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Callable, Literal, Protocol

from dvision2_common import (controlled_command_with_id, load_pymembus,
                             parse_command_results, shared_names)

Frame = Literal["map", "local_ned", "global"]


@dataclass(frozen=True)
class VehicleCapabilities:
    vehicle: str
    frames: tuple[Frame, ...]
    accepts_position_target: bool
    accepts_velocity_target: bool
    accepts_attitude_target: bool
    supports_missions: bool
    setpoint_timeout_s: float | None
    max_speed_mps: float
    max_accel_mps2: float


@dataclass(frozen=True)
class PositionTarget:
    frame: Frame
    x: float | None = None
    y: float | None = None
    z: float | None = None
    north_m: float | None = None
    east_m: float | None = None
    down_m: float | None = None
    lat_deg: float | None = None
    lon_deg: float | None = None
    alt_m: float | None = None
    heading_deg: float = 0.0
    max_speed_mps: float = 1.0


@dataclass(frozen=True)
class VelocityTarget:
    forward_mps: float = 0.0
    right_mps: float = 0.0
    up_mps: float = 0.0
    yaw_rate_dps: float = 0.0


@dataclass(frozen=True)
class VehicleState:
    sample_monotonic_s: float
    link_connected: bool
    armed: bool
    mode: str
    position: PositionTarget
    heading_deg: float
    vx_mps: float
    vy_mps: float
    vz_mps: float
    attitude_valid: bool
    local_position_valid: bool
    global_position_valid: bool
    velocity_valid: bool
    failsafe_reason: str | None
    last_setpoint_age_s: float | None


@dataclass(frozen=True)
class CommandResult:
    request_id: str
    accepted: bool
    reason: str = ""


class VehicleLink(Protocol):
    def capabilities(self) -> VehicleCapabilities: ...
    def state(self) -> VehicleState: ...
    def acquire_control(self, client_id: str) -> CommandResult: ...
    def release_control(self) -> CommandResult: ...
    def arm(self, armed: bool) -> CommandResult: ...
    def takeoff(self, alt_m: float) -> CommandResult: ...
    def land(self) -> CommandResult: ...
    def rtl(self) -> CommandResult: ...
    def send_position_target(self, target: PositionTarget) -> CommandResult: ...
    def send_velocity_target(self, target: VelocityTarget) -> CommandResult: ...
    def hold(self) -> CommandResult: ...


# ---------------------------------------------------------------------------
# dsim implementation
# ---------------------------------------------------------------------------

#: The published facts a vehicle page shows. They are health and environment,
#: never capabilities: what the vehicle accepts is static, what it is doing
#: right now is not, and the two must not be read from the same place.
DIAGNOSTIC_KEYS = (
    "drone.mode", "drone.armed", "drone.battery_pct",
    "gps.fix_type", "gps.satellites", "gps.hdop", "gps.vdop",
    "est.attitude_valid", "est.local_position_valid",
    "est.global_position_valid", "est.velocity_valid",
    "wind.speed_mps", "wind.dir_deg", "wind.gust_mps",
    "geofence.box", "geofence.action",
    "realism.telemetry_latency_ms", "realism.telemetry_jitter_ms",
    "realism.sensor_noise", "realism.battery_failsafe_pct",
    "realism.battery_drain_pct_s", "realism.seed",
    "control.owner", "control.lease_age_s", "control.lease_timeout_s",
    "setpoint.age_s", "failsafe.reason",
    "command.result.request_id", "command.result.accepted",
    "command.result.reason",
    "origin.lat_deg", "origin.lon_deg", "origin.alt_m",
    "home.lat_deg", "home.lon_deg", "home.alt_m",
    "drone.lat_deg", "drone.lon_deg", "drone.alt_m",
    "sim.map", "sim.report_dir",
)

#: How long a link waits for the acknowledgement of one command. Queue
#: admission is not acceptance -- a lease or mode check may still refuse it --
#: so every command, streamed setpoints included, waits for its result.
DEFAULT_ACK_TIMEOUT_S = 3.0


class StatusTransport(Protocol):
    """The two directions a vehicle link needs, and nothing else.

    Splitting the wire out of :class:`DsimLink` is what lets the parsing,
    correlation and lease behaviour be tested against a real simulator without
    shared memory in the way; the production transport below is the only piece
    that a test does not exercise.
    """

    @property
    def connected(self) -> bool: ...
    def connect(self) -> bool: ...
    def write(self, payload: str) -> bool: ...
    def status(self) -> dict[str, str]: ...
    def epoch(self) -> int: ...
    def close(self) -> None: ...


class SharedMemoryTransport:
    """pymembus command queue plus status key/value store, by instance id."""

    def __init__(self, instance_id: str, *, cmd_size: int = 65536) -> None:
        self.names = shared_names(instance_id)
        self.cmd_size = int(cmd_size)
        self._pm = load_pymembus()
        self._command = None
        self._status = None

    @property
    def connected(self) -> bool:
        return self._command is not None and self._status is not None

    def connect(self) -> bool:
        if self._command is None:
            handle = self._pm.memcmd()
            if handle.open(self.names["command"], self.cmd_size):
                self._command = handle
        if self._status is None:
            handle = self._pm.memkv()
            if handle.open(self.names["status"]):
                self._status = handle
        return self.connected

    def write(self, payload: str) -> bool:
        return bool(self._command is not None and self._command.write(payload))

    def status(self) -> dict[str, str]:
        if self._status is None:
            return {}
        return dict(self._status.getAll())

    def epoch(self) -> int:
        return 0 if self._status is None else int(self._status.getEpoch())

    def close(self) -> None:
        for name in ("_status", "_command"):
            handle = getattr(self, name)
            if handle is not None:
                try:
                    handle.close()
                except Exception:
                    pass
                setattr(self, name, None)


def _float(values: dict[str, str], key: str, default: float | None = None) -> float | None:
    raw = values.get(key, "")
    if raw == "":
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _flag(values: dict[str, str], key: str, default: bool) -> bool:
    raw = values.get(key, "")
    return default if raw == "" else raw == "1"


class DsimLink:
    """``VehicleLink`` over the dsim JSON command and status protocol.

    Every motion, mode and arming command carries this client's source id,
    lease id and a unique request id, and waits for the matching
    ``command.result.*`` before reporting acceptance.
    """

    def __init__(self, instance_id: str, *, client_id: str | None = None,
                 transport: StatusTransport | None = None,
                 ack_timeout_s: float = DEFAULT_ACK_TIMEOUT_S,
                 cmd_size: int = 65536,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.instance_id = instance_id
        self.client_id = client_id or f"dway-{instance_id}"
        self.lease_id = uuid.uuid4().hex
        self.ack_timeout_s = float(ack_timeout_s)
        self._transport = transport if transport is not None else \
            SharedMemoryTransport(instance_id, cmd_size=cmd_size)
        self._clock = clock
        self._sleep = sleep
        self._epoch = -1
        self._sample_monotonic = clock()
        self._values: dict[str, str] = {}

    # -- connection ----------------------------------------------------

    def connect(self) -> bool:
        return self._transport.connect()

    @property
    def connected(self) -> bool:
        return self._transport.connected

    def close(self) -> None:
        self._transport.close()

    def report_dir(self) -> str:
        return self._read().get("sim.report_dir", "")

    def sim_time_s(self) -> float:
        try:
            return float(self._read().get("sim.time_s", "0") or 0.0)
        except ValueError:
            return 0.0

    def map_path(self) -> str:
        return self._read().get("sim.map", "")

    # -- reading -------------------------------------------------------

    def _read(self) -> dict[str, str]:
        values = self._transport.status()
        epoch = self._transport.epoch()
        if values and (epoch != self._epoch or values != self._values):
            self._epoch = epoch
            self._values = values
            self._sample_monotonic = self._clock()
        return self._values

    def capabilities(self) -> VehicleCapabilities:
        values = self._read()
        frames = tuple(f for f in values.get("vehicle.frames", "").split(",") if f)
        return VehicleCapabilities(
            vehicle=values.get("vehicle.type", ""),
            frames=frames,  # type: ignore[arg-type]
            accepts_position_target=_flag(values, "vehicle.accepts_position", False),
            accepts_velocity_target=_flag(values, "vehicle.accepts_velocity", False),
            accepts_attitude_target=_flag(values, "vehicle.accepts_attitude", False),
            supports_missions=_flag(values, "vehicle.supports_missions", False),
            setpoint_timeout_s=_float(values, "vehicle.setpoint_timeout_s", None),
            max_speed_mps=_float(values, "vehicle.max_speed_mps", 0.0) or 0.0,
            max_accel_mps2=_float(values, "vehicle.max_accel_mps2", 0.0) or 0.0,
        )

    def state(self) -> VehicleState:
        values = self._read()
        position = PositionTarget(
            frame="map",
            x=_float(values, "drone.x_m", 0.0), y=_float(values, "drone.y_m", 0.0),
            z=_float(values, "drone.z_m", 0.0),
            heading_deg=_float(values, "drone.heading_deg", 0.0) or 0.0,
        )
        mode = values.get("drone.mode", "")
        # dsim publishes no estimator flags before the sensor-realism work
        # lands; a link reports what the vehicle says and assumes validity only
        # where the vehicle has no way to say otherwise.
        return VehicleState(
            sample_monotonic_s=self._sample_monotonic,
            link_connected=bool(values) and self._transport.connected,
            armed=values.get("drone.armed") == "1",
            mode=mode,
            position=position,
            heading_deg=position.heading_deg,
            vx_mps=_float(values, "drone.vx_mps", 0.0) or 0.0,
            vy_mps=_float(values, "drone.vy_mps", 0.0) or 0.0,
            vz_mps=_float(values, "drone.vz_mps", 0.0) or 0.0,
            attitude_valid=_flag(values, "est.attitude_valid", True),
            local_position_valid=_flag(values, "est.local_position_valid", True),
            global_position_valid=_flag(values, "est.global_position_valid",
                                        bool(values.get("origin.lat_deg"))),
            velocity_valid=_flag(values, "est.velocity_valid", True),
            failsafe_reason=values.get("failsafe.reason") or None,
            last_setpoint_age_s=_float(values, "setpoint.age_s", None),
        )

    def diagnostics(self) -> dict[str, str]:
        """Published health and environment, exactly as the vehicle states it."""
        values = self._read()
        return {key: values.get(key, "") for key in DIAGNOSTIC_KEYS}

    def set_gps(self, mode: str, noise_m: float | None = None) -> CommandResult:
        fields: dict[str, object] = {"mode": mode}
        if noise_m is not None:
            fields["noise_m"] = float(noise_m)
        return self._send("set_gps", **fields)

    def set_estimator(self, **flags: bool) -> CommandResult:
        return self._send("set_estimator", **{k: bool(v) for k, v in flags.items()})

    def control_owner(self) -> str:
        return self._read().get("control.owner", "")

    def owns_control(self) -> bool:
        return self.control_owner() == self.client_id

    # -- commanding ----------------------------------------------------

    def _send(self, command_type: str, **fields) -> CommandResult:
        request_id, payload = controlled_command_with_id(
            command_type, self.client_id, self.lease_id, **fields)
        if not self._transport.write(payload):
            return CommandResult(request_id, False, "command queue unavailable")
        return self._await_result(request_id)

    def _result_for(self, values: dict[str, str],
                    request_id: str) -> CommandResult | None:
        """This client's result, from the latest slot or the published history.

        The latest slot is preferred because it carries the untruncated reason,
        but it holds one result for the whole vehicle: any other client whose
        command lands in the same tick replaces it before it is published. The
        history is what makes the correlation survive a second commanding
        client; a vehicle that publishes none behaves exactly as it used to.
        """
        if values.get("command.result.request_id") == request_id:
            return CommandResult(
                request_id, values.get("command.result.accepted") == "1",
                values.get("command.result.reason", ""))
        recorded = parse_command_results(values.get("command.results", ""))
        if request_id in recorded:
            accepted, reason = recorded[request_id]
            return CommandResult(request_id, accepted, reason)
        return None

    def _await_result(self, request_id: str) -> CommandResult:
        deadline = self._clock() + self.ack_timeout_s
        while True:
            result = self._result_for(self._read(), request_id)
            if result is not None:
                return result
            if self._clock() >= deadline:
                return CommandResult(request_id, False, "acknowledgement timeout")
            self._sleep(0.005)

    def acquire_control(self, client_id: str | None = None) -> CommandResult:
        if client_id:
            self.client_id = client_id
        self.lease_id = uuid.uuid4().hex
        return self._send("acquire_control")

    def release_control(self) -> CommandResult:
        return self._send("release_control")

    def heartbeat(self) -> CommandResult:
        return self._send("heartbeat")

    def arm(self, armed: bool) -> CommandResult:
        return self._send("arm", armed=bool(armed))

    def takeoff(self, alt_m: float) -> CommandResult:
        return self._send("takeoff", alt_m=float(alt_m))

    def land(self) -> CommandResult:
        return self._send("land")

    def rtl(self) -> CommandResult:
        return self._send("rtl")

    def hold(self) -> CommandResult:
        return self._send("hold")

    def send_position_target(self, target: PositionTarget) -> CommandResult:
        fields: dict[str, float] = {}
        if target.frame == "map":
            fields = {"x": target.x, "y": target.y, "z": target.z}
        elif target.frame == "local_ned":
            fields = {"north_m": target.north_m, "east_m": target.east_m,
                      "down_m": target.down_m}
        else:
            return CommandResult("", False,
                                 f"dsim does not accept the {target.frame} frame")
        if any(value is None for value in fields.values()):
            return CommandResult("", False,
                                 f"incomplete {target.frame} position target")
        return self._send("position_target", frame=target.frame,
                          heading_deg=float(target.heading_deg),
                          max_speed_mps=float(target.max_speed_mps),
                          **{k: float(v) for k, v in fields.items()})

    def send_velocity_target(self, target: VelocityTarget) -> CommandResult:
        return self._send(
            "velocity", forward_mps=float(target.forward_mps),
            right_mps=float(target.right_mps), up_mps=float(target.up_mps),
            yaw_rate_dps=float(target.yaw_rate_dps))
