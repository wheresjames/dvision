"""The flight lifecycle: one explicit state machine for headless and UI runs.

``dway.py`` owns a window and a command line; this module owns what actually
happens to the vehicle, so a scripted flight and a clicked one cannot drift
apart. It is importable without a display, which is how ``dalg`` flies without
duplicating a control loop.

Reading of the contract worth stating once: a health fault -- stale state, an
invalid estimator, a lost link -- commands HOLD immediately and then enters
FAILED with the exact reason. HOLD is the vehicle-side reaction and FAILED is
the mission-side one; neither retries on its own.
"""

from __future__ import annotations

import datetime
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from dvision2_common import (
    DEFAULT_REPORT_ID, SimMap, load_map, new_run_id, report_root,
)
from dcmn.pacing import PeriodicDeadline
from dway.follower import (
    AIRBORNE_MIN_M, Follower, FollowerEvent, MIN_TAKEOFF_ALT_M, Sample,
    StrategyError, build_legs, select_strategy,
)
from dway.link import CommandResult, VehicleCapabilities, VehicleState
from dway.report import (
    SUMMARY_SCHEMA_VERSION, FlightRecorder, write_summary, write_track,
)
from dway.tour import (
    FrameContext, Tour, TourError, leg_clearances, load_tour_map, resolve_map,
)

#: The repository root: assets and reports live beside ``apps/``, not in it.
ROOT = Path(__file__).resolve().parents[2]

#: Bounded waits for transitions the vehicle acknowledges with motion rather
#: than with a command result.
ARMING_TIMEOUT_S = 5.0
TAKEOFF_TIMEOUT_S = 60.0
LANDING_TIMEOUT_S = 120.0
CONNECT_TIMEOUT_S = 15.0
TAKEOFF_SETTLE_M = 0.15


class MissionState(str, Enum):
    DISCONNECTED = "DISCONNECTED"
    PREFLIGHT = "PREFLIGHT"
    READY = "READY"
    ARMING = "ARMING"
    TAKING_OFF = "TAKING_OFF"
    FLYING = "FLYING"
    PAUSED = "PAUSED"
    COMPLETING = "COMPLETING"
    RTL = "RTL"
    LANDING = "LANDING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


TERMINAL_STATES = (MissionState.COMPLETE, MissionState.FAILED)
FINISH_ACTIONS = ("hold", "land", "rtl")


@dataclass
class MissionConfig:
    strategy: str = "auto"
    speed_mps: float | None = None
    stream_hz: float = 10.0
    heartbeat_hz: float = 1.0
    finish_action: str = "land"
    autostart: bool = True

    def validate(self) -> None:
        if self.finish_action not in FINISH_ACTIONS:
            raise ValueError(f"unknown finish action {self.finish_action!r}")
        if self.stream_hz <= 0.0 or self.heartbeat_hz <= 0.0:
            raise ValueError("stream and heartbeat rates must be positive")
        if self.speed_mps is not None and self.speed_mps <= 0.0:
            raise ValueError("speed must be positive")


@dataclass
class Mission:
    """Fly one tour on one vehicle link, and account for what happened."""

    link: Any
    tour: Tour
    config: MissionConfig = field(default_factory=MissionConfig)
    root: Path = ROOT
    sim_map: SimMap | None = None
    recorder: FlightRecorder | None = None
    #: Simulated time: everything about the flight is measured against the
    #: clock the vehicle keeps, so a busy host cannot shorten a dwell or expire
    #: a leg. In production this is the link's ``sim_time_s``.
    clock: Callable[[], float] = time.monotonic
    #: Wall time, and only for liveness: how old the state we received is.
    #: A dead simulator stops simulated time, so a staleness check measured on
    #: it could never fire. It must match the clock ``VehicleState`` was
    #: stamped with -- see ``DsimLink.__init__``.
    wall: Callable[[], float] = time.monotonic

    def __post_init__(self) -> None:
        self.config.validate()
        self.state = MissionState.DISCONNECTED
        self.reason = ""
        self.outcome = ""
        self.strategy = None
        self.context = FrameContext(0.0, 0.0, self.tour.geo_anchor)
        self.follower: Follower | None = None
        self.capabilities: VehicleCapabilities | None = None
        self.speed_mps = self.config.speed_mps or self.tour.default_speed_mps
        self.warnings: list[str] = []
        self.failsafes: list[dict[str, Any]] = []
        self.track: list[tuple[float, float]] = []
        self.planned: list[tuple[float, float]] = []
        self.started_at = datetime.datetime.now().astimezone().isoformat(
            timespec="seconds")
        self.last_state: VehicleState | None = None
        self._t0 = self.clock()
        self._paused_s = 0.0
        self._paused_at: float | None = None
        self._deadline: float | None = None
        self._stream_cadence = PeriodicDeadline(self.config.stream_hz)
        self._last_heartbeat_s = 0.0
        self._armed_sent = False
        self._takeoff_alt_m: float | None = None
        self._finished_s: float | None = None
        self._current_index = -1
        self._conditions: dict[str, Any] = {}
        #: Setpoints actually put on the wire, for the keeping-up report.
        self.setpoints_sent = 0

    # ------------------------------------------------------------------
    # Clocks and logging
    # ------------------------------------------------------------------

    def elapsed(self, now: float | None = None) -> float:
        """Mission time, which does not run while the tour is paused."""
        now = self.clock() if now is None else now
        paused = self._paused_s
        if self._paused_at is not None:
            paused += now - self._paused_at
        return now - self._t0 - paused

    def _log(self, kind: str, **fields) -> None:
        if self.recorder is not None:
            self.recorder.event(self.elapsed(), kind, **fields)

    def _transition(self, state: MissionState, reason: str = "") -> MissionState:
        if state is not self.state:
            self._log("mission_state", state=self.last_state,
                      mission_state=state.value, reason=reason or None)
        self.state = state
        if reason:
            self.reason = reason
        return state

    def _fail(self, reason: str) -> MissionState:
        """Stop the vehicle, then record the exact reason and stop retrying."""
        if self.state not in TERMINAL_STATES:
            try:
                self.link.hold()
            except Exception:
                pass
        self.outcome = "failed"
        return self._transition(MissionState.FAILED, reason)

    # ------------------------------------------------------------------
    # Operator commands
    # ------------------------------------------------------------------

    def start(self) -> bool:
        if self.state is not MissionState.READY:
            return False
        self._deadline = self.clock() + ARMING_TIMEOUT_S
        self._armed_sent = False
        self._transition(MissionState.ARMING)
        return True

    def pause(self) -> bool:
        if self.state not in (MissionState.FLYING, MissionState.TAKING_OFF):
            return False
        result = self.link.hold()
        self._log("hold", request_id=result.request_id, accepted=result.accepted,
                  reason=result.reason)
        self._paused_at = self.clock()
        self._transition(MissionState.PAUSED, "paused by operator")
        return True

    def resume(self) -> bool:
        if self.state is not MissionState.PAUSED:
            return False
        state = self.link.state()
        fault = self._health(state, self.clock())
        if fault:
            self._fail(fault)
            return False
        if not self._reacquire_if_needed():
            return False
        if self._paused_at is not None:
            self._paused_s += self.clock() - self._paused_at
            self._paused_at = None
        self.reason = ""
        self._stream_cadence.reset(self.clock())
        if self.follower is not None:
            self.follower.begin_leg(self._sample(state, self.clock()))
        self._transition(MissionState.FLYING)
        return True

    def land(self) -> CommandResult:
        result = self.link.land()
        self._log("land", request_id=result.request_id, accepted=result.accepted,
                  reason=result.reason)
        if result.accepted:
            self._deadline = self.clock() + LANDING_TIMEOUT_S
            self._transition(MissionState.LANDING)
        return result

    def rtl(self) -> CommandResult:
        result = self.link.rtl()
        self._log("rtl", request_id=result.request_id, accepted=result.accepted,
                  reason=result.reason)
        if result.accepted:
            self._deadline = self.clock() + LANDING_TIMEOUT_S
            self._transition(MissionState.RTL)
        return result

    def hold(self) -> CommandResult:
        result = self.link.hold()
        self._log("hold", request_id=result.request_id, accepted=result.accepted,
                  reason=result.reason)
        if result.accepted and self.state is MissionState.FLYING:
            self._paused_at = self.clock()
            self._transition(MissionState.PAUSED, "held by operator")
        return result

    def abort(self, reason: str = "shutdown") -> None:
        """Stop the vehicle and give up control without disarming it."""
        if self.state not in TERMINAL_STATES:
            try:
                self.link.hold()
            except Exception:
                pass
            self.outcome = "aborted"
            self._transition(MissionState.FAILED, reason)
        self.release()

    def release(self) -> None:
        try:
            self.link.release_control()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Stepping
    # ------------------------------------------------------------------

    def step(self, now: float | None = None) -> MissionState:
        now = self.clock() if now is None else now
        handler = {
            MissionState.DISCONNECTED: self._step_disconnected,
            MissionState.PREFLIGHT: self._step_preflight,
            MissionState.READY: self._step_ready,
            MissionState.ARMING: self._step_arming,
            MissionState.TAKING_OFF: self._step_taking_off,
            MissionState.FLYING: self._step_flying,
            MissionState.PAUSED: self._step_idle,
            MissionState.COMPLETING: self._step_completing,
            MissionState.RTL: self._step_descending,
            MissionState.LANDING: self._step_descending,
        }.get(self.state)
        if handler is None:
            return self.state
        return handler(now)

    def _step_idle(self, now: float) -> MissionState:
        self._observe(now)
        self._heartbeat(now)
        return self.state

    def _step_disconnected(self, now: float) -> MissionState:
        if self._deadline is None:
            self._deadline = now + CONNECT_TIMEOUT_S
        if self.link.connect() and self.link.capabilities().vehicle:
            self._deadline = None
            # Anchor mission time on the first reading we can actually take.
            # Constructing the mission reads a clock that has no vehicle behind
            # it yet, so a simulator that had been up for a while would make
            # the flight appear to have started before it was launched.
            self._t0 = self.clock()
            self._log("connected")
            return self._transition(MissionState.PREFLIGHT)
        if now > self._deadline:
            return self._fail("no vehicle answered on this instance id")
        return self.state

    # -- preflight ------------------------------------------------------

    def _step_preflight(self, now: float) -> MissionState:
        try:
            self._preflight(now)
        except (TourError, StrategyError, ValueError) as exc:
            return self._fail(str(exc))
        if self.state is MissionState.FAILED:
            return self.state
        self._transition(MissionState.READY)
        if self.config.autostart:
            self.start()
        return self.state

    def _preflight(self, now: float) -> None:
        capabilities = self.link.capabilities()
        self.capabilities = capabilities
        tour_map = load_tour_map(self.tour, self.root)
        if self.sim_map is None:
            # A tour that names no map still needs the vehicle's, because the
            # local-NED origin is the centre of the map the vehicle is flying
            # and nothing can relate the two frames without its size.
            self.sim_map = tour_map or self._vehicle_map()
        self.context = FrameContext.for_map(self.sim_map, self.tour.geo_anchor)

        self.strategy = select_strategy(
            capabilities, self.tour.coordinate_frame,
            speed_mps=self.speed_mps, forced=self.config.strategy)
        timeout = capabilities.setpoint_timeout_s
        if timeout and self.config.stream_hz <= 2.0 / timeout:
            raise ValueError(
                f"setpoint stream of {self.config.stream_hz:g} Hz is too slow for "
                f"a {timeout:g}s vehicle setpoint timeout")
        if capabilities.max_speed_mps and self.speed_mps > capabilities.max_speed_mps:
            raise ValueError(
                f"tour speed {self.speed_mps:g} m/s exceeds the vehicle limit of "
                f"{capabilities.max_speed_mps:g} m/s")

        state = self.link.state()
        self.last_state = state
        if (state.position.frame != self.tour.coordinate_frame
                and self.context.width_m <= 0.0):
            raise ValueError(
                f"vehicle publishes {state.position.frame}-frame positions and "
                f"no map is available to convert them to "
                f"{self.tour.coordinate_frame}")
        fault = self._health(state, now)
        if fault:
            raise ValueError(f"preflight state check failed: {fault}")
        if self.tour.coordinate_frame == "global" and not state.global_position_valid:
            raise ValueError("a global tour needs a valid global position estimate")

        legs = build_legs(self.tour, self.context,
                          frame=getattr(self.strategy, "frame",
                                        self.tour.coordinate_frame),
                          speed_mps=self.speed_mps)
        self.follower = Follower(self.tour, legs, speed_mps=self.speed_mps)
        self.planned = ([self.context.ned_to_map(leg.north_m, leg.east_m,
                                                 leg.down_m)[:2] for leg in legs]
                        if self.sim_map is not None else [])
        self._check_clearance(state)

        result = self.link.acquire_control()
        self._log("acquire_control", request_id=result.request_id,
                  accepted=result.accepted, reason=result.reason)
        if not result.accepted:
            raise ValueError(f"control not acquired: {result.reason}")
        self._conditions = self._read_conditions()
        self._log("preflight", strategy=self.strategy.name,
                  strategy_detail=self.strategy.describe(),
                  setpoint_timeout_s=capabilities.setpoint_timeout_s,
                  warnings=self.warnings or None)

    def _vehicle_map(self) -> SimMap | None:
        """The map the vehicle says it is flying, when it publishes one."""
        published = getattr(self.link, "map_path", lambda: "")()
        if not published:
            return None
        path = Path(published)
        if not path.is_absolute():
            path = self.root / path
        if not path.exists():
            return None
        try:
            return load_map(path)
        except ValueError:
            return None

    _CONDITION_KEYS = ("gps.fix_type", "gps.satellites", "gps.hdop",
                       "wind.speed_mps", "wind.dir_deg", "wind.gust_mps",
                       "geofence.box", "geofence.action",
                       "realism.telemetry_latency_ms",
                       "realism.telemetry_jitter_ms", "realism.sensor_noise",
                       "realism.battery_failsafe_pct",
                       "realism.battery_drain_pct_s", "realism.seed",
                       "est.local_position_valid", "est.global_position_valid")

    def _read_conditions(self) -> dict[str, Any]:
        diagnostics = getattr(self.link, "diagnostics", None)
        if diagnostics is None:
            return {}
        published = diagnostics()
        return {key: published[key] for key in self._CONDITION_KEYS
                if published.get(key, "") != ""}

    def _check_clearance(self, state: VehicleState) -> None:
        """Measure every leg, leg zero from the current pose included.

        A leg that intersects an obstacle is refused: the vehicle would fly
        into it, and nothing in ``dway`` avoids obstacles. A leg that merely
        passes closer than the tour's stated ``min_clearance_m`` is reported as
        a warning, because the committed tours fly legitimately tight lines and
        silently refusing them would hide the measurement rather than show it.
        """
        if self.sim_map is None:
            return
        pose_map = self.context.ned_to_map(*self.context.position_ned(state.position))
        clearances = leg_clearances(self.tour, self.sim_map, pose_map[:2])
        obstructed = [leg for leg in clearances if leg.obstructed]
        if obstructed:
            names = ", ".join(str(leg.index) for leg in obstructed)
            raise ValueError(
                f"leg {names} passes through map geometry; the tour cannot be "
                f"flown from this pose")
        for leg in clearances:
            if leg.clearance_m < self.tour.min_clearance_m:
                self.warnings.append(
                    f"leg {leg.index} clears obstacles by "
                    f"{leg.clearance_m:.2f} m, under the tour's "
                    f"{self.tour.min_clearance_m:.2f} m")

    # -- flight ---------------------------------------------------------

    def _step_ready(self, now: float) -> MissionState:
        self._observe(now)
        self._heartbeat(now)
        return self.state

    def _step_arming(self, now: float) -> MissionState:
        state = self._observe(now)
        if state is None:
            return self.state
        self._heartbeat(now)
        if not self._armed_sent:
            result = self.link.arm(True)
            self._log("arm", state=state, request_id=result.request_id,
                      accepted=result.accepted, reason=result.reason)
            self._armed_sent = True
            if not result.accepted:
                return self._fail(f"arm rejected: {result.reason}")
        if state.armed:
            return self._begin_takeoff(now, state)
        if self._deadline is not None and now > self._deadline:
            return self._fail("vehicle did not report itself armed")
        return self.state

    def _begin_takeoff(self, now: float, state: VehicleState) -> MissionState:
        assert self.follower is not None
        leg = self.follower.legs[0]
        altitude = max(-leg.down_m, MIN_TAKEOFF_ALT_M)
        height = -self.context.position_ned(state.position)[2]
        if height >= AIRBORNE_MIN_M:
            # Already airborne: hold this altitude and let the first leg fly
            # the vehicle to the waypoint's height.
            self._takeoff_alt_m = None
            return self._enter_flying(now, state)
        result = self.link.takeoff(altitude)
        self._log("takeoff", state=state, request_id=result.request_id,
                  accepted=result.accepted, reason=result.reason,
                  alt_m=altitude)
        if not result.accepted:
            return self._fail(f"takeoff rejected: {result.reason}")
        self._takeoff_alt_m = altitude
        self._deadline = now + TAKEOFF_TIMEOUT_S
        return self._transition(MissionState.TAKING_OFF)

    def _step_taking_off(self, now: float) -> MissionState:
        state = self._observe(now)
        if state is None:
            return self.state
        self._heartbeat(now)
        height = -self.context.position_ned(state.position)[2]
        settled = (self._takeoff_alt_m is None
                   or abs(height - self._takeoff_alt_m) <= TAKEOFF_SETTLE_M)
        if state.mode == "GUIDED" and settled:
            return self._enter_flying(now, state)
        if self._deadline is not None and now > self._deadline:
            return self._fail("takeoff did not settle")
        return self.state

    def _enter_flying(self, now: float, state: VehicleState) -> MissionState:
        assert self.follower is not None
        self.follower.begin_leg(self._sample(state, now))
        self._stream_cadence.reset(now)
        self._transition(MissionState.FLYING)
        return self._step_flying(now)

    def _step_flying(self, now: float) -> MissionState:
        assert self.follower is not None
        self._stream_cadence.set_rate(self.config.stream_hz)
        state = self._observe(now)
        if state is None:
            return self.state
        self._heartbeat(now)

        sample = self._sample(state, now)
        reached = self.follower.index
        event = self.follower.update(sample)
        if event in (FollowerEvent.ARRIVED, FollowerEvent.COMPLETE):
            entry = self.follower.progress[reached]
            self._log("arrived", state=state, waypoint=entry.index,
                      arrival_s=round(self.elapsed(now), 3),
                      dwell_s=round(entry.dwell_s, 3),
                      overshoot_m=round(entry.overshoot_m, 4),
                      max_cross_track_error_m=round(entry.max_cross_track_error_m, 4))
        if event is FollowerEvent.COMPLETE:
            return self._begin_completion(now)
        if event is FollowerEvent.LEG_TIMEOUT:
            return self._fail(
                f"waypoint {self.follower.index} not reached within its leg timeout")

        due = (event is FollowerEvent.ARRIVED
               or self._stream_cadence.due(now))
        if due:
            return self._stream(now, state, sample)
        return self.state

    def _stream(self, now: float, state: VehicleState,
                sample: Sample) -> MissionState:
        assert self.follower is not None and self.strategy is not None
        leg = self.follower.leg
        if leg is None:
            return self._begin_completion(now)
        result = self.strategy.send(self.link, leg, sample, self.speed_mps)
        self.setpoints_sent += 1
        # Advance an absolute simulated-time grid. Anchoring this to ``now``
        # turns every late poll into permanent drift (10 Hz became ~7.5 Hz on
        # a 30 Hz vehicle clock). Do not burst to catch up: skip deadlines
        # already behind us and retain the original phase.
        self._stream_cadence.advance(now)
        self._log("setpoint", state=state, request_id=result.request_id,
                  accepted=result.accepted, reason=result.reason,
                  waypoint=leg.index, strategy=self.strategy.name)
        self._current_index = leg.index
        if not result.accepted:
            return self._fail(f"setpoint rejected: {result.reason}")
        return self.state

    def _begin_completion(self, now: float) -> MissionState:
        result = self.link.hold()
        self._log("tour_complete", request_id=result.request_id,
                  accepted=result.accepted, reason=result.reason)
        self._finished_s = self.elapsed(now)
        self._transition(MissionState.COMPLETING)
        return self._step_completing(now)

    def _step_completing(self, now: float) -> MissionState:
        action = self.config.finish_action
        if action == "hold":
            self.outcome = "complete"
            return self._transition(MissionState.COMPLETE, "tour complete")
        result = self.land() if action == "land" else self.rtl()
        if not result.accepted:
            return self._fail(f"{action} rejected: {result.reason}")
        return self.state

    def _step_descending(self, now: float) -> MissionState:
        state = self._observe(now, gate_health=False)
        if state is None:
            return self.state
        self._heartbeat(now)
        if not state.armed and state.mode in ("DISARMED", "LAND"):
            # A land or RTL ordered after a failure ends the flight safely but
            # does not turn the failure into a success.
            if self.outcome in ("failed", "aborted"):
                return self._transition(MissionState.FAILED, self.reason)
            self.outcome = "complete"
            return self._transition(MissionState.COMPLETE, "landed")
        if state.mode == "CRASHED":
            return self._fail("vehicle crashed")
        if self._deadline is not None and now > self._deadline:
            return self._fail(f"{self.state.value.lower()} did not finish")
        return self.state

    # -- shared ---------------------------------------------------------

    def _sample(self, state: VehicleState, now: float) -> Sample:
        """The follower works in mission time, which does not run while paused.

        Every clock the follower keeps is a difference of this one -- a dwell,
        a leg deadline, a trim step -- and none of them should advance while
        the tour is held. It is also what makes the waypoint timestamps in
        :meth:`summary` the same quantity as ``duration_s`` and as the
        ``arrival_s`` written into the event log.
        """
        return Sample.from_state(state, self.context, self.elapsed(now))

    def _observe(self, now: float, *, gate_health: bool = True) -> VehicleState | None:
        state = self.link.state()
        self.last_state = state
        # The plotted track is the pose as the vehicle publishes it. Map-frame
        # states plot directly; a vehicle that publishes another frame leaves
        # the plot empty rather than a track drawn through a guessed origin.
        if state.link_connected and state.position.frame == "map" \
                and state.position.x is not None:
            self.track.append((float(state.position.x), float(state.position.y)))
        if state.failsafe_reason and (
                not self.failsafes
                or self.failsafes[-1]["reason"] != state.failsafe_reason):
            self.failsafes.append({"t_s": round(self.elapsed(now), 3),
                                   "reason": state.failsafe_reason})
        if gate_health:
            fault = self._health(state, now)
            if fault:
                self._fail(fault)
                return None
        return state

    def _health(self, state: VehicleState, now: float) -> str:
        """Everything that must hold before another setpoint may be sent."""
        if not state.link_connected:
            return "vehicle link disconnected"
        # Wall against wall: state.sample_wall_s was stamped by the link on the
        # wall clock, and `now` here is simulated time.
        age = self.wall() - state.sample_wall_s
        if age > self.tour.max_state_age_s:
            return f"vehicle state is {age:.2f}s old"
        if state.mode == "CRASHED":
            return "vehicle crashed"
        if state.failsafe_reason:
            return f"vehicle failsafe: {state.failsafe_reason}"
        if not state.local_position_valid:
            return "local position estimate is invalid"
        if not state.velocity_valid:
            return "velocity estimate is invalid"
        return ""

    def _heartbeat(self, now: float) -> None:
        """Renew the control lease. This never refreshes the setpoint timer."""
        if now - self._last_heartbeat_s < 1.0 / self.config.heartbeat_hz:
            return
        self._last_heartbeat_s = now
        heartbeat = getattr(self.link, "heartbeat", None)
        if heartbeat is not None:
            heartbeat()

    def _reacquire_if_needed(self) -> bool:
        if getattr(self.link, "owns_control", lambda: True)():
            return True
        result = self.link.acquire_control()
        self._log("acquire_control", request_id=result.request_id,
                  accepted=result.accepted, reason=result.reason)
        if not result.accepted:
            self._fail(f"control not reacquired: {result.reason}")
            return False
        return True

    def blocking_fact(self) -> str:
        """The one fact preventing flight right now, or an empty string.

        This is the question a vehicle page exists to answer, so it is
        computed here rather than assembled in the window: capability, health,
        ownership and failsafe are checked in the order in which each would
        stop a flight, and the first that does is named with the published
        value that says so.
        """
        if self.state is MissionState.FAILED:
            return self.reason
        state = self.last_state
        if state is None or not state.link_connected:
            return "no vehicle is publishing on this instance id"
        capabilities = self.capabilities
        if capabilities is not None:
            if not (capabilities.accepts_position_target
                    or capabilities.accepts_velocity_target):
                return ("vehicle accepts neither position nor velocity targets "
                        "(vehicle.accepts_position=0, vehicle.accepts_velocity=0)")
            if (self.tour.coordinate_frame == "global"
                    and not state.global_position_valid):
                return ("this tour is global and the global position estimate "
                        "is invalid (est.global_position_valid=0)")
        if state.mode == "CRASHED":
            return "vehicle has crashed and needs a reset"
        if state.failsafe_reason:
            return f"vehicle failsafe: {state.failsafe_reason}"
        if not state.local_position_valid:
            return "local position estimate is invalid (est.local_position_valid=0)"
        if not state.velocity_valid:
            return "velocity estimate is invalid (est.velocity_valid=0)"
        if not state.attitude_valid:
            return "attitude estimate is invalid (est.attitude_valid=0)"
        age = self.wall() - state.sample_wall_s
        if age > self.tour.max_state_age_s:
            return (f"vehicle state is {age:.2f}s old, older than the tour's "
                    f"{self.tour.max_state_age_s:.2f}s limit")
        owner = getattr(self.link, "control_owner", lambda: "")()
        client = getattr(self.link, "client_id", "")
        if owner and client and owner != client:
            return f"control is held by {owner}"
        if self.state in (MissionState.READY, MissionState.PREFLIGHT):
            return "" if self.state is MissionState.PREFLIGHT else "waiting for Start"
        return ""

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        outcome = self.outcome or (
            "complete" if self.state is MissionState.COMPLETE else "failed")
        waypoints: list[dict[str, Any]] = []
        follower = self.follower
        if follower is not None:
            for leg, entry in zip(follower.legs, follower.progress):
                waypoints.append({
                    "index": entry.index,
                    "target": leg.waypoint.describe(),
                    "first_target_s": _round_opt(entry.first_target_s),
                    "arrival_s": _round_opt(entry.arrival_s),
                    "dwell_s": round(entry.dwell_s, 3),
                    "overshoot_m": round(entry.overshoot_m, 4),
                    "max_cross_track_error_m": round(entry.max_cross_track_error_m, 4),
                })
        arrived = sum(1 for entry in waypoints if entry["arrival_s"] is not None)
        return {
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "tour_id": self.tour.tour_id,
            "outcome": outcome,
            "reason": self.reason,
            "started_at": self.started_at,
            "duration_s": round(self._finished_s if self._finished_s is not None
                                else self.elapsed(), 3),
            "strategy": self.strategy.name if self.strategy else "",
            "strategy_detail": self.strategy.describe() if self.strategy else "",
            "coordinate_frame": self.tour.coordinate_frame,
            "waypoint_count": len(self.tour.waypoints),
            "waypoints_reached": arrived,
            "waypoints": waypoints,
            "path_length_m": round(follower.path_length_m, 4) if follower else 0.0,
            "max_cross_track_error_m": (
                round(follower.max_cross_track_error_m, 4) if follower else 0.0),
            "failsafes": list(self.failsafes),
            "partial": outcome != "complete" or arrived < len(self.tour.waypoints),
            "warnings": list(self.warnings),
            "vehicle": self.capabilities.vehicle if self.capabilities else "",
            "map": str(resolve_map(self.tour, self.root) or ""),
            "conditions": self.conditions,
        }

    @property
    def conditions(self) -> dict[str, Any]:
        """The environment the flight actually happened in.

        A run that was flown in wind, with a degraded fix or through delayed
        telemetry is a different measurement from one that was not, so the
        report records what the vehicle published rather than what the
        operator believes was configured.
        """
        return dict(self._conditions)

    def write_report(self, report_dir: str | Path) -> Path:
        report_dir = Path(report_dir)
        summary = self.summary()
        write_summary(report_dir, summary)
        write_track(report_dir, planned=self.planned, flown=self.track,
                    sim_map=self.sim_map,
                    title=f"{self.tour.tour_id}  |  {summary['outcome']}"
                          f"  |  {summary['strategy']}")
        return report_dir


def _round_opt(stamp: float | None) -> float | None:
    """A follower timestamp, already in mission time, ready for the report."""
    return None if stamp is None else round(stamp, 3)


def mission_report_dir(link, instance_id: str | None) -> Path:
    """``<sim.report_dir>/dway/``, from the vehicle's own published root.

    A module never builds this path out of its own idea of where reports live;
    when the vehicle publishes nothing, one run directory is minted with the
    shared helper so a real flight and a simulated one still match in shape.
    """
    published = ""
    try:
        published = link.report_dir()
    except Exception:
        published = ""
    if published:
        return Path(published) / "dway"
    return report_root(instance_id or DEFAULT_REPORT_ID, new_run_id(),
                       root=ROOT / "reports") / "dway"
