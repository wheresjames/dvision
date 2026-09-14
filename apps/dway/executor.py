"""Fly dnav's permitted routes on the existing vehicle link.

dnav plans and permits; this executor flies and stops. The scope is deliberately
narrow: one planner and one executor per vehicle, fixed altitude, position
targets at a fixed activation heading, a stop at every corner and every
permission endpoint, and route changes only from a confirmed stop. The vehicle
must already be airborne in HOLD at the profile altitude: there is no takeoff,
landing or RTL here, on success, failure or shutdown.

Stopping is HOLD requested *before* a target, never the vehicle converging on
it. Calibration measured position-target overshoot of up to 0.62 m against a
HOLD stopping distance of 0.025 m, so converging on a permission endpoint could
carry the vehicle past it.

Every target is re-checked against the snapshot consumed in the same step:
context, goal, stop generation, permitted interval, deadline, tracking and the
stopping margin. A HOLD acknowledgement is not a stop: the stop is confirmed
only after the observed speed stays under ``hold_speed_mps`` for ``hold_dwell_s``.
Operator controls are local (the window and the headless runner call the same
methods) and are handled at the start of the next step, Pause/Cancel first.

``auto=True`` is target mode: the executor itself acquires an available lease,
arms, climbs to the profile altitude, confirms HOLD, and then presses Start and
Resume whenever an admitted route allows -- the same controls an operator would
use, recorded with origin ``auto``. Pause and Cancel still belong to the
operator and suppress it. It never lands.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict
import math
from pathlib import Path
import threading
import time

from dcmn.archive import Recorder, next_archive_dir
from dcmn.context import Context, validate_pose
from dcmn.navigation import (Snapshot, STATUS_SCHEMA, distance_along, new_session, permitted_points,
                             point_at, project_progress)
from dcmn.pacing import PeriodicDeadline
from dway.dynamic import DynamicRouteSource
from dway.frames import ProviderFrame
from dway.link import PositionTarget

MOVING = ('EXECUTING', 'BRAKING')
TERMINAL = ('COMPLETE', 'CANCELLED', 'FAILED')
#: Stops that continue on their own once the next step is validated, under the same Start.
AUTO_CONTINUE = ('corner', 'replacement')
#: Stops that need a new explicit Start.
START_REQUIRED = ('restart', 'goal')
#: Every other stop reason needs an explicit Resume with fresh validation.
HISTORY_S = 60.0
NOTABLE = ('execution.shutdown_requested', 'execution.control', 'execution.transition', 'execution.hold', 'execution.lease',
           'execution.health', 'execution.route', 'execution.shutdown', 'execution.launch', 'execution.readiness',
           'execution.coverage', 'execution.turn')


def course_deg(a, b):
    """Compass heading from map point ``a`` to ``b`` (x east, y south; 0 north, 90 east)."""
    return (math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])) + 90.0) % 360.0


def heading_error_deg(target, current):
    return (target - current + 180.0) % 360.0 - 180.0


def _goal_key(goal):
    if not isinstance(goal, dict): return None
    return (goal.get('authority'), goal.get('authority_epoch'), goal.get('revision'),
            tuple(goal.get('position') or ()))


class DynamicExecutor:
    def __init__(self, instance, link, profile, *, planner='dnav', clock=None, wall=time.monotonic,
                 sleep=time.sleep, context=None, navigation=None, output=None, maps=None,
                 record=True, recording_queue_bytes=32 << 20, recording_disk_bytes=1 << 30,
                 report=True, input_stall_s=1.0, overrun_s=None, auto=False, allow_uncalibrated=False,
                 wait_for=(), bus=None, takeoff_timeout_s=30.0, bus_enabled=True):
        if not allow_uncalibrated: profile.require_flight()
        self.auto, self.wait_for, self.bus = auto, tuple(wait_for), bus
        self.takeoff_timeout_s = takeoff_timeout_s
        self.launched = not auto
        self.launch_phase = None
        self.launch_note = ''
        self.ready_roles = set()
        self.shutdown_requested = False
        self._launch_sent = False
        self._launch_deadline = None
        self._launch_retry = float('-inf')
        self._launch_below = None
        self._auto_next = float('-inf')
        self._last_prepare = float('-inf')
        #: Tests and fixtures without a shared bus pass ``bus_enabled=False``.
        self.bus_enabled = bus_enabled
        self.shutdown_reason = ''
        self._hello_sent = False
        self._last_bus_heartbeat = None
        self.bus_error = ''
        self.coverage_request = None
        self.id, self.link, self.profile = instance, link, profile
        self.clock = clock or link.sim_time_s
        self.wall, self.sleep = wall, sleep
        self.source = DynamicRouteSource(instance, planner, profile, stall_s=input_stall_s)
        self.context = context if context is not None else Context(instance)
        self.navigation = navigation if navigation is not None else Snapshot(instance, planner=planner)
        self.output = output if output is not None else Snapshot(instance, 'execution')
        try:
            self.output.start()  # a second live executor for this vehicle fails here, before any command
        except ValueError as exc:
            raise ValueError(f'{exc}: another dway (dry run, dynamic or target) is already running for vehicle '
                             f'{instance!r}; stop it first (for example: pkill -f "dway.py --id {instance}")') from None
        self.maps = maps
        self.session = new_session()
        self.record_enabled, self.report_enabled = record, report
        self.recording_queue_bytes, self.recording_disk_bytes = recording_queue_bytes, recording_disk_bytes
        self.input_stall_s = input_stall_s
        self.overrun_s = overrun_s if overrun_s is not None else max(0.5, 3.0 / profile.stream_hz)
        self.state, self.reason = 'WAITING', 'waiting for a complete navigation snapshot'
        self.hold_kind = None
        self.owns_control = False
        self.lease_state = 'not held'
        self.authorized_goal = None
        self.heading_deg = None
        self.turning = False
        self.turn_error_deg = None
        self.active = None
        self.segment = 0
        self.progress = None
        self.pose = None
        self.speed = None
        self.cross_track = self.remaining = self.margin = self.validity = None
        self.violated_margin_m = None
        self.target = None
        self.vehicle = None
        self.vehicle_fault = ''
        self.snapshot = {}
        self.missions = 0
        self.targets_sent = 0
        self.sent = deque(maxlen=50000)
        self.pending = deque()
        self.control_results = {}
        self.events = deque(maxlen=300)
        self.history = deque(maxlen=int(HISTORY_S * profile.stream_hz) + 10)
        self.track = []
        self.sequence = 0
        self.status = {}
        self.closed = False
        self.recorder = None
        self.report_dir = None
        self.report_session = None
        self.report_error = ''
        self.reports = []
        self.finish_timings_s = {}
        self._report_threads = []
        self._stream = PeriodicDeadline(profile.stream_hz)
        self._samples = PeriodicDeadline(profile.stream_hz)
        self._brake_started = None
        self._below_since = None
        self._last_heartbeat = None
        self._last_wall = None
        self._pose_mark = None
        self._pose_wall = None
        self._route_mark = None
        self._track_mark = None
        self._view = {}

    # -- operator controls (same methods for window and headless) ----------

    def request(self, action, origin='ui'):
        if action not in ('start', 'pause', 'resume', 'cancel'): raise ValueError(f'unknown control {action!r}')
        self.pending.append((action, origin))
        self.control_results[action] = dict(state='pending', origin=origin)

    def _control(self, action, origin, now):
        try:
            accepted, reason = getattr(self, f'_do_{action}')(now)
        except Exception as exc:  # a control must never take the loop down
            accepted, reason = False, f'{type(exc).__name__}: {exc}'
        self.control_results[action] = dict(state='accepted' if accepted else 'refused', reason=reason,
                                            origin=origin, t_s=now)
        self._record('execution.control', dict(action=action, origin=origin, accepted=accepted, reason=reason))
        return accepted, reason

    # -- the loop ----------------------------------------------------------

    def step(self):
        if self.closed: return self.status
        # A live link must be opened (the tour mission does the same); until it
        # is, there is no vehicle state and no simulated clock -- sim time reads 0.
        try:
            if not self.link.connected: self.link.connect()
        except Exception as exc:
            self.vehicle_fault = f'vehicle link connect failed: {exc}'
        now, wall = self.clock(), self.wall()
        gap = None if self._last_wall is None else wall - self._last_wall
        self._last_wall = wall
        queued = [self.pending.popleft() for _ in range(len(self.pending))]
        for action, origin in queued:  # Pause and Cancel before any ordinary work
            if action in ('pause', 'cancel'): self._control(action, origin, now)
        self._read_inputs(now, wall)
        self._observe_vehicle(now, wall)
        if gap is not None and gap > self.overrun_s and self.state == 'EXECUTING':
            self._brake(now, 'overrun', f'control loop stalled {gap:.2f} s (limit {self.overrun_s:.2f} s); '
                                        'no obsolete target is sent')
        for action, origin in queued:
            if action in ('start', 'resume'): self._control(action, origin, now)
        self._coordinate(now, wall)
        if self.auto: self._auto_step(now)
        if self.owns_control: self._supervise(now, wall)
        {'WAITING': self._step_idle, 'READY': self._step_idle, 'TAKING_OFF': self._step_launch,
         'EXECUTING': self._step_executing,
         'BRAKING': self._step_braking, 'HOLDING': self._step_holding, 'COMPLETE': self._step_complete,
         'CANCELLED': self._step_terminal, 'FAILED': self._step_terminal}[self.state](now)
        self._publish(now)
        return self.status

    def _read_inputs(self, now, wall):
        try: self.snapshot = self.context.read() or {}
        except (ValueError, KeyError): self.snapshot = {}
        self._adopt_recording()
        if self.maps is not None:
            try: self.maps.poll()
            except Exception: pass
        mark = (self.snapshot.get('provider_id'), self.snapshot.get('pose_sequence'))
        if mark != self._pose_mark: self._pose_mark, self._pose_wall = mark, wall
        value = {}
        try:
            value = self.navigation.read()
            if value:
                self.source.consume(value, self.snapshot, wall=wall)
            elif self.source.active:
                self.source.state, self.source.reason = 'STOP_REQUIRED', 'navigation publisher unavailable'
            else:
                self.source.state, self.source.reason = 'WAITING', 'navigation publisher unavailable'
        except (ValueError, RuntimeError) as exc:
            self.source.state = 'STOP_REQUIRED' if self.source.active else 'REJECTED'
            self.source.reason = str(exc)
        if self._pose_wall is not None and wall - self._pose_wall > self.input_stall_s:
            self.source.state = 'STOP_REQUIRED' if self.source.active else 'REJECTED'
            self.source.reason = 'provider pose stream stalled'
        route_mark = (self.source.session, self.source.revision, self.source.state, self.source.reason,
                      self.source.stop_generation, (self.source.value.get('clearance') or {}).get('end'),
                      (self.source.disposition or {}).get('value'))
        if route_mark != self._route_mark:
            self._route_mark = route_mark
            grids = {}
            if self.maps is not None:
                grids = {sid: st.grid for sid, st in self.maps.states.items() if st.grid is not None}
            self._record('execution.route', dict(
                snapshot=self.source.value, source_state=self.source.state, source_reason=self.source.reason,
                disposition=self.source.disposition, handled_stop=self.source.handled_stop,
                route_age_s=None if not self.source.value or 'time_s' not in self.snapshot
                else self.snapshot['time_s'] - self.source.value['time_s']), grids)

    def _observe_vehicle(self, now, wall):
        try: state = self.link.state()
        except Exception as exc:
            self.vehicle, self.pose, self.speed = None, None, None
            self.vehicle_fault = f'vehicle link read failed: {exc}'
            return
        self.vehicle = state
        self.pose = self._map_pose(state)
        self.speed = math.hypot(state.vx_mps, state.vy_mps) if state.link_connected else None
        identity = (self.snapshot.get('provider_id'), self.snapshot.get('localization_epoch'),
                    self.snapshot.get('clock_epoch'))
        if self.pose is None or identity != self._track_mark:
            if self.track and self.track[-1] is not None: self.track.append(None)  # a gap, never joined
            self._track_mark = identity
        if self.pose is not None and (not self.track or self.track[-1] != tuple(self.pose[:2])):
            self.track.append(tuple(self.pose[:2]))
            if len(self.track) > 6000: del self.track[:len(self.track) - 6000]

    def _map_pose(self, state):
        if not state.link_connected: return None
        p = state.position
        if p.frame == 'map' and None not in (p.x, p.y, p.z):
            return [float(p.x), float(p.y), float(p.z)]
        if p.frame == 'local_ned' and None not in (p.north_m, p.east_m, p.down_m):
            try: return list(ProviderFrame.from_context(self.snapshot).ned_to_map(p.north_m, p.east_m, p.down_m))
            except ValueError: return None
        return None

    # -- supervision, health and lease ------------------------------------

    def _supervise(self, now, wall):
        if self._last_heartbeat is None or now - self._last_heartbeat >= 1.0 or now < self._last_heartbeat:
            self._last_heartbeat = now
            try: self.link.heartbeat()
            except Exception: pass
        try: owns = self.link.owns_control()
        except Exception: owns = False
        if not owns:
            owner = ''
            try: owner = self.link.control_owner()
            except Exception: pass
            self.owns_control = False
            self.lease_state = f'lost (owner {owner or "none"})'
            self._record('execution.lease', dict(event='lost', owner=owner))
            if self.state not in TERMINAL:
                return self._fail(now, f'control lease lost (now held by {owner or "nobody"}); '
                                       'no reacquisition or resumption', send_hold=False)
            self.vehicle_fault = f'control lease lost after {self.state.lower()}'
            return
        if self.state == 'TAKING_OFF': return  # climbing is outside the slab and not GUIDED/HOLD yet
        fault = self._health(self.vehicle, wall)
        if fault and fault != self.vehicle_fault:
            self._record('execution.health', dict(fault=fault, state=self.state))
        if fault and self.state in ('COMPLETE', 'CANCELLED'):
            self.vehicle_fault = fault  # mission outcome and vehicle state stay separate
        elif fault and self.state != 'FAILED':
            self._fail(now, fault)
        elif not fault:
            self.vehicle_fault = ''

    def _health(self, state, wall):
        p = self.profile
        if state is None or not state.link_connected: return 'vehicle link disconnected'
        age = wall - state.sample_wall_s
        if age > p.max_state_age_s: return f'vehicle state is {age:.2f} s old (limit {p.max_state_age_s:g} s)'
        if not state.mode: return 'vehicle status unavailable (no mode published; simulator stopped?)'
        if state.mode == 'CRASHED': return 'vehicle crashed'
        if state.failsafe_reason: return f'vehicle failsafe: {state.failsafe_reason}'
        if not state.local_position_valid: return 'local position estimate is invalid'
        if not state.velocity_valid: return 'velocity estimate is invalid'
        if state.mode not in ('GUIDED', 'HOLD'):
            return f'vehicle left GUIDED/HOLD (mode {state.mode}): manual takeover or vehicle failsafe'
        if self.pose is None: return 'vehicle position unavailable in the provider frame'
        if abs(self.pose[2] - p.altitude_m) > p.half_height_m:
            return (f'altitude {self.pose[2]:.2f} m outside the {p.altitude_m-p.half_height_m:.2f}–'
                    f'{p.altitude_m+p.half_height_m:.2f} m slab')
        return ''

    def _vehicle_ready(self, state, wall):
        p = self.profile
        if state is None or not state.link_connected: return 'vehicle link disconnected'
        age = wall - state.sample_wall_s
        if age > p.max_state_age_s: return f'vehicle state is {age:.2f} s old'
        if not state.armed: return 'vehicle is not armed (airborne HOLD required; no takeoff here)'
        if state.mode != 'HOLD': return f'vehicle mode is {state.mode}; confirmed HOLD required'
        if state.failsafe_reason: return f'vehicle failsafe: {state.failsafe_reason}'
        if not (state.local_position_valid and state.velocity_valid): return 'position/velocity estimate invalid'
        pose = self._map_pose(state)
        if pose is None: return 'vehicle position unavailable in the provider frame'
        if abs(pose[2] - p.altitude_m) > p.half_height_m:
            return f'altitude {pose[2]:.2f} m is not the profile altitude {p.altitude_m:g}±{p.half_height_m:g} m'
        speed = math.hypot(state.vx_mps, state.vy_mps)
        if speed > p.hold_speed_mps: return f'vehicle moving at {speed:.2f} m/s; not stopped'
        return ''

    def _environment_problem(self):
        p = self.profile
        try: diagnostics = self.link.diagnostics()
        except Exception: return ''
        def value(key):
            try: return float(diagnostics.get(key) or 0.)
            except ValueError: return 0.
        wind = max(value('wind.speed_mps'), value('wind.gust_mps'))
        if wind > p.max_wind_mps + 1e-9: return f'wind {wind:g} m/s outside the calibrated {p.max_wind_mps:g} m/s'
        latency = value('realism.telemetry_latency_ms') + value('realism.telemetry_jitter_ms')
        if latency > p.max_telemetry_latency_ms + 1e-9:
            return f'telemetry latency {latency:g} ms outside the calibrated {p.max_telemetry_latency_ms:g} ms'
        return ''

    # -- transitions ------------------------------------------------------

    def _transition(self, state, reason, kind=None):
        if state != self.state or reason != self.reason or kind != self.hold_kind:
            self._record('execution.transition', dict(previous=self.state, state=state, reason=reason, hold_kind=kind))
        self.state, self.reason, self.hold_kind = state, reason, kind

    def _fail(self, now, reason, *, send_hold=True):
        if self.state == 'FAILED': return
        self.target = None
        if send_hold and self.owns_control:
            try:
                result = self.link.hold()
                self._record('execution.hold', dict(phase='requested', kind='failure', reason=reason,
                                                    accepted=result.accepted, result=result.reason))
            except Exception as exc:
                self._record('execution.hold', dict(phase='rejected', kind='failure', reason=str(exc)))
        if self.source.active: self.source.release('rejected')
        self._transition('FAILED', reason, self.hold_kind)

    def _brake(self, now, kind, reason, violated=None):
        self.target = None
        self._brake_started, self._below_since = now, None
        self.violated_margin_m = violated
        try: result = self.link.hold()
        except Exception as exc:
            return self._fail(now, f'HOLD could not be sent ({exc}) while stopping: {reason}', send_hold=False)
        self._record('execution.hold', dict(phase='requested', kind=kind, reason=reason, accepted=result.accepted,
                                            result=result.reason, request_id=result.request_id, pose=self.pose,
                                            progress=self.progress, violated_margin_m=violated))
        self._transition('BRAKING', reason, kind)
        if not result.accepted:
            self._fail(now, f'HOLD rejected ({result.reason}) while stopping: {reason}; '
                            'targets ceased, stop not confirmed', send_hold=False)

    # -- controls ----------------------------------------------------------

    def _do_start(self, now):
        src, wall = self.source, self.wall()
        if self.state == 'TAKING_OFF': return False, f'taking off: {self.reason}'
        if self.state in MOVING: return False, 'already executing'
        if self.state in ('FAILED', 'CANCELLED'):
            return False, f'mission {self.state.lower()}: {self.reason}; start dway again for a new mission'
        if self.state == 'HOLDING' and self.hold_kind not in START_REQUIRED and not src.start_required:
            return False, 'holding an authorized route: use Resume'
        if src.state != 'READY': return False, f'no admitted route: {src.reason}'
        goal = self.snapshot.get('goal')
        if self.state == 'COMPLETE' and _goal_key(goal) == self.authorized_goal:
            return False, 'this goal is already complete; set a new goal first'
        try: capabilities = self.link.capabilities()
        except Exception as exc: return False, f'vehicle capabilities unavailable: {exc}'
        if not capabilities.accepts_position_target or not {'map', 'local_ned'} & set(capabilities.frames):
            return False, 'vehicle does not accept map/local-NED position targets (dynamic mode is position-only)'
        problem = self._environment_problem() or self._vehicle_ready(self.vehicle, wall)
        if problem: return False, problem
        acquired = False
        if not self.owns_control:
            owner = self.link.control_owner()
            if owner and owner != self.link.client_id:
                return False, f'control is held by {owner}; an available lease is required (never taken over)'
            result = self.link.acquire_control()
            self._record('execution.lease', dict(event='acquire', accepted=result.accepted, reason=result.reason,
                                                 request_id=result.request_id))
            if not result.accepted: return False, f'control lease not acquired: {result.reason}'
            self.owns_control, acquired, self.lease_state = True, True, 'held'
        # Recheck everything with the lease held, before any target.
        self._read_inputs(now, self.wall())
        self._observe_vehicle(now, self.wall())
        problem = self._vehicle_ready(self.vehicle, self.wall())
        if not problem and src.state != 'READY': problem = f'route changed during Start: {src.reason}'
        if not problem and _goal_key(self.snapshot.get('goal')) != _goal_key(goal): problem = 'goal changed during Start'
        if problem:
            if acquired: self._release_lease('start recheck failed')
            return False, f'recheck with lease held failed: {problem}'
        src.start()
        if src.active: src.release('stopped')
        src.consume(src.value, self.snapshot, wall=self.wall())
        if src.state != 'READY':
            if acquired: self._release_lease('start recheck failed')
            return False, f'route not admissible after restart: {src.reason}'
        src.activate()
        self.authorized_goal = _goal_key(self.snapshot.get('goal'))
        self.heading_deg = float(self.vehicle.heading_deg)
        self.missions += 1
        self._record('execution.conditions', dict(diagnostics=self._diagnostics(), capabilities=asdict(capabilities),
                                                  profile=asdict(self.profile), heading_deg=self.heading_deg))
        self._activate(now, src.value)
        self._transition('EXECUTING', 'started by operator')
        return True, 'started'

    def _do_pause(self, now):
        if self.state in ('EXECUTING', 'TAKING_OFF'):
            self.launched = True  # an operator pause ends the automatic launch
            self._brake(now, 'pause', 'paused by operator')
            return True, 'stopping'
        if self.state in ('BRAKING', 'HOLDING') and self.hold_kind in AUTO_CONTINUE + (None,):
            self.hold_kind, self.reason = 'pause', 'paused by operator'
            return True, 'held; automatic continuation suppressed'
        if self.state in ('BRAKING', 'HOLDING'): return True, 'already stopping or held'
        return False, f'not executing ({self.state})'

    def _do_resume(self, now):
        src, wall = self.source, self.wall()
        if self.state != 'HOLDING': return False, f'Resume applies to a held route, not {self.state}'
        if self.hold_kind in START_REQUIRED or src.start_required:
            return False, f'Start required: {self.reason}'
        if self.hold_kind in AUTO_CONTINUE: return False, 'this stop continues on its own once validated'
        if not self.owns_control: return False, 'control lease not held'
        problem = self._environment_problem() or self._vehicle_ready(self.vehicle, wall)
        if problem: return False, problem
        if src.state != 'READY': return False, f'no fresh permission: {src.reason}'
        if _goal_key(self.snapshot.get('goal')) != self.authorized_goal:
            return False, 'goal changed; Start required'
        if not src.active:
            src.activate()
            self._activate(now, src.value)
        else:
            value = src.value
            if value['geometry_revision'] != self.active['geometry_revision']:
                return False, 'active geometry changed while held; waiting for the stop to be processed'
            self.active = value
            target, final, _ = self._target(value, self.pose, self.progress or value['clearance']['start'])
            if final and math.dist(self.pose, target) <= self.profile.stopping_m and not self._reaches_goal(value):
                return False, 'permitted interval not extended beyond the stopping distance; still waiting for clearance'
        self._stream.reset(now)
        self._transition('EXECUTING', 'resumed by operator')
        return True, 'resumed'

    def _do_cancel(self, now):
        if self.state in TERMINAL: return False, f'mission already {self.state.lower()}'
        if self.state in ('EXECUTING', 'TAKING_OFF'):
            self.launched = True
            self._brake(now, 'cancel', 'cancelled by operator')
            return True, 'stopping'
        if self.state == 'BRAKING':
            self.hold_kind, self.reason = 'cancel', 'cancelled by operator'
            return True, 'stopping'
        if self.source.active: self.source.release('stopped')
        self._transition('CANCELLED', 'cancelled by operator')
        return True, 'cancelled'

    # -- target mode: launch and automatic Start/Resume ---------------------

    def _coordinate(self, now, wall=None):
        """The module bus: instance shutdown, presence, and the ``wait_for`` readiness handshake.

        Every mode listens, so dsim's "Kill all" (``system.shutdown``) reaches a
        dynamic or target executor whether or not it waits for other roles.
        """
        if not self.bus_enabled: return
        wall = self.wall() if wall is None else wall
        try:
            if self.bus is None:
                from dcmn.module_bus import PymembusModuleBus
                self.bus = PymembusModuleBus(self.id, 'navigator', 'dway', sim_time=self.clock)
            if not self.bus.connect(): return
            from dcmn.module_bus import requests_shutdown
            capabilities = dict(mode='target' if self.auto else 'dynamic', execution_endpoint=self.output.name,
                                holds_control_lease=self.owns_control)
            if not self._hello_sent:
                self.bus.publish('module.hello', run_id=self.session, payload=dict(state=self.state,
                                                                                  capabilities=capabilities))
                self._hello_sent = True
            if self._last_bus_heartbeat is None or wall - self._last_bus_heartbeat >= 1.0:
                self._last_bus_heartbeat = wall
                self.bus.publish('module.heartbeat', run_id=self.session, payload=dict(
                    state=self.state, reason=self.reason, ready=self.state not in ('WAITING', 'TAKING_OFF'),
                    capabilities=capabilities))
            for event in self.bus.receive():
                if requests_shutdown(event) and not self.shutdown_requested:
                    self.shutdown_requested = True
                    self.shutdown_reason = str(event.payload.get('reason') or 'instance shutdown requested')
                    self._record('execution.shutdown_requested', dict(reason=self.shutdown_reason,
                                                                      source=event.role, process=event.process_id))
                if event.type != 'run.ready' or event.run_id != self.session: continue
                for requirement in self.wait_for:
                    role, _, selector = requirement.partition(':')
                    if event.role == role and (not selector or selector in (
                            event.payload.get('profile'), event.payload.get('algorithm'), event.process_id)
                            or selector in event.payload.get('capabilities', {}).get('algorithms', ())):
                        if requirement not in self.ready_roles:
                            self._record('execution.readiness', dict(role=requirement, process=event.process_id))
                        self.ready_roles.add(requirement)
            if self.wait_for and (now - self._last_prepare >= 1.0 or now < self._last_prepare):
                self._last_prepare = now
                self.bus.publish('run.prepare', run_id=self.session,
                                 payload=dict(mode='target', required_roles=list(self.wait_for)))
            self.bus_error = ''
        except Exception as exc:  # the bus is coordination, never control
            self.bus_error = f'module bus error: {type(exc).__name__}: {exc}'

    def missing_roles(self):
        return [r for r in self.wait_for if r not in self.ready_roles]

    def _request_coverage(self):
        """Target mode owns the mission goal: ask dalg to map a region that contains it."""
        goal = (self.snapshot.get('goal') or {})
        position, revision = goal.get('position'), goal.get('revision')
        if not position or self.maps is None or revision == (self.coverage_request or {}).get('goal_revision'):
            return
        grid = next((st.grid for st in list(self.maps.states.values()) if st.grid is not None), None)
        if grid is None: return
        g, margin = grid.geometry, 5.0
        x0, y0 = g.origin_x_m, g.origin_y_m
        x1, y1 = x0 + g.width * g.cell_m, y0 + g.height * g.cell_m
        gx, gy = position[0], position[1]
        if x0 <= gx <= x1 and y0 <= gy <= y1:
            self.coverage_request = dict(goal_revision=revision, requested=False)
            return
        bounds = [min(x0, gx - margin), min(y0, gy - margin), max(x1, gx + margin), max(y1, gy + margin)]
        try:
            reset = self.context.request_reset(bounds, by='dway-target')
            self.coverage_request = dict(goal_revision=revision, requested=True, bounds=bounds, reset_revision=reset)
        except (ValueError, OSError) as exc:
            self.coverage_request = dict(goal_revision=revision, requested=False, error=str(exc))
        self._record('execution.coverage', dict(goal=position, coverage=[x0, y0, x1, y1], **self.coverage_request))

    def _auto_step(self, now):
        self._coordinate(now)
        self._request_coverage()
        if self.state in ('CANCELLED', 'FAILED'): return
        if not self.launched:
            if self.state not in ('WAITING', 'READY'): return
            missing = self.missing_roles()
            if missing:
                self.launch_note = (f'waiting for {", ".join(missing)} to report ready'
                                    + (f' ({self.bus_error})' if self.bus_error else ''))
                return
            self.launch_phase, self._launch_sent = 'lease', False
            self._record('execution.launch', dict(phase='begin', altitude_m=self.profile.altitude_m))
            self._transition('TAKING_OFF', 'acquiring an available control lease')
            return
        src = self.source
        if now < self._auto_next and now >= self._auto_next - 5.0: return
        if src.state != 'READY': return
        if self.state in ('WAITING', 'READY'):
            self._auto_next = now + 1.0
            self._control('start', 'auto', now)
        elif self.state == 'HOLDING' and self.hold_kind not in ('pause', 'cancel') + AUTO_CONTINUE:
            self._auto_next = now + 1.0
            restart = self.hold_kind in START_REQUIRED or src.start_required
            self._control('start' if restart else 'resume', 'auto', now)

    def _launch_fail(self, now, reason):
        self.launched = True
        self._fail(now, f'target mode launch failed: {reason}')

    def _step_launch(self, now):
        p, v, link = self.profile, self.vehicle, self.link
        if v is None or not v.link_connected:
            return self._transition('TAKING_OFF', 'waiting for the vehicle link')
        phase = self.launch_phase
        if phase == 'lease':
            if not self.owns_control:
                if now < self._launch_retry and now >= self._launch_retry - 5.0: return
                self._launch_retry = now + 1.0
                owner = link.control_owner()
                if owner and owner != link.client_id:
                    return self._transition('TAKING_OFF', f'waiting: control lease held by {owner} (never taken over)')
                result = link.acquire_control()
                self._record('execution.lease', dict(event='acquire', accepted=result.accepted, reason=result.reason,
                                                     request_id=result.request_id, origin='auto'))
                if not result.accepted:
                    return self._transition('TAKING_OFF', f'control lease not acquired: {result.reason}; retrying')
                self.owns_control, self.lease_state = True, 'held'
            phase, self._launch_sent = 'arm', False
        if phase == 'arm':
            if not v.armed:
                if not self._launch_sent:
                    result = link.arm(True)
                    self._record('execution.launch', dict(phase='arm', accepted=result.accepted, reason=result.reason))
                    self._launch_sent, self._launch_deadline = True, now + 5.0
                    if not result.accepted: return self._launch_fail(now, f'arm rejected: {result.reason}')
                if now > self._launch_deadline: return self._launch_fail(now, 'vehicle did not report itself armed')
                self.launch_phase = phase
                return self._transition('TAKING_OFF', 'arming')
            phase, self._launch_sent = 'climb', False
        if phase == 'climb':
            if self.pose is None:
                self.launch_phase = phase
                return self._transition('TAKING_OFF', 'waiting for a vehicle position in the provider frame')
            height = self.pose[2]
            if abs(height - p.altitude_m) > p.half_height_m or v.mode not in ('GUIDED', 'HOLD'):
                if not self._launch_sent:
                    result = link.takeoff(p.altitude_m)
                    self._record('execution.launch', dict(phase='takeoff', accepted=result.accepted,
                                                          reason=result.reason, from_m=height))
                    self._launch_sent, self._launch_deadline = True, now + self.takeoff_timeout_s
                    if not result.accepted: return self._launch_fail(now, f'takeoff rejected: {result.reason}')
                if now > self._launch_deadline:
                    return self._launch_fail(now, f'did not reach {p.altitude_m:g} m within {self.takeoff_timeout_s:g} s')
                self.launch_phase = phase
                return self._transition('TAKING_OFF', f'climbing to {p.altitude_m:g} m')
            phase, self._launch_sent = 'hold', False
        if phase == 'hold':
            if not self._launch_sent:
                result = link.hold()
                self._record('execution.launch', dict(phase='hold', accepted=result.accepted, reason=result.reason))
                self._launch_sent, self._launch_deadline, self._launch_below = True, now + p.hold_timeout_s, None
                if not result.accepted: return self._launch_fail(now, f'HOLD rejected: {result.reason}')
            if v.mode == 'HOLD' and self.speed is not None and self.speed <= p.hold_speed_mps:
                self._launch_below = now if self._launch_below is None else self._launch_below
                if now - self._launch_below >= p.hold_dwell_s:
                    self.launch_phase, self.launched = 'airborne', True
                    self._record('execution.launch', dict(phase='airborne', pose=self.pose))
                    return self._transition('WAITING', 'airborne in HOLD; waiting for an admitted route')
            else:
                self._launch_below = None
            if now > self._launch_deadline: return self._launch_fail(now, 'HOLD not confirmed after takeoff')
            self.launch_phase = phase
            return self._transition('TAKING_OFF', 'confirming HOLD at altitude')

    # -- per-state steps --------------------------------------------------

    def _step_idle(self, now):
        src = self.source
        if not self.launched:
            self._transition('WAITING', self.launch_note or 'target mode: preparing to take off')
        elif src.state == 'READY':
            self._transition('READY', 'route admitted; waiting for Start' +
                             (' (planner restarted: new Start required)' if src.start_required else ''))
        else:
            self._transition('WAITING', src.reason)

    def _activate(self, now, value):
        self.active = value
        start = value['clearance']['start'] if len(value['points']) > 1 else [0, 0.]
        self.progress = list(start)
        self.segment = start[0]
        self.violated_margin_m = None
        self._stream.reset(now)

    def _target(self, value, pose, progress):
        points, end = value['points'], value['clearance']['end']
        if len(points) == 1: return points[0], True, (0, 0.)
        segment = max(self.segment, progress[0])
        if (segment, 1.0) >= tuple(end): return point_at(points, end), True, tuple(end)
        return points[segment + 1], False, (segment, 1.0)

    def _reaches_goal(self, value):
        points, clearance = value['points'], value['clearance']
        if not clearance.get('reaches_goal'): return False
        # An arrival-only route is a single point: the goal is where we are.
        if len(points) == 1: return self._goal_distance(points[0]) <= self.profile.arrival_m
        if tuple(clearance['end']) != (len(points) - 2, 1.0): return False
        return self._goal_distance(points[-1]) <= self.profile.arrival_m

    def _goal_distance(self, point):
        goal = (self.snapshot.get('goal') or {}).get('position')
        if not goal or point is None: return math.inf
        return math.dist(point[:2], goal[:2])

    def _kind_for(self, reason):
        if 'goal' in reason: return 'goal'
        if 'stalled' in reason: return 'stalled'
        if 'expired' in reason or 'too little time' in reason: return 'expired'
        if 'context mismatch' in reason or 'restarted' in reason: return 'restart'
        return 'unavailable'

    def _step_executing(self, now):
        src, p = self.source, self.profile
        if src.state == 'STOP_REQUIRED':
            eligible = bool((src.value.get('clearance') or {}).get('eligible'))
            # A frame/clock/provider reset is a restart even when the record it
            # invalidated was still eligible: it needs a new Start, never an
            # automatic continuation.
            restart = src.restart_stop or 'context mismatch' in src.reason
            kind = ('restart' if restart else 'goal' if 'goal' in src.reason
                    else 'replacement' if eligible else self._kind_for(src.reason))
            return self._brake(now, kind, src.reason)
        if src.state != 'READY': return self._brake(now, self._kind_for(src.reason), src.reason)
        value = src.value
        if value['geometry_revision'] != self.active['geometry_revision']:
            return self._brake(now, 'replacement', 'route geometry changed while executing')
        self.active = value
        if _goal_key(self.snapshot.get('goal')) != self.authorized_goal:
            return self._brake(now, 'goal', 'goal changed or cleared; the new goal needs Start')
        if self.pose is None or self.speed is None: return self._brake(now, 'unavailable', 'vehicle pose unavailable')
        points, clearance = value['points'], value['clearance']
        progress, cross = project_progress(points, self.pose, self.progress)
        self.progress, self.cross_track = progress, cross
        end = clearance['end']
        self.remaining = distance_along(points, progress, end) if len(points) > 1 else math.dist(self.pose, points[0])
        required = p.stopping_m if self.speed > p.hold_speed_mps else 0.
        self.margin = self.remaining - required
        self.validity = clearance['valid_until_s'] - self.snapshot.get('time_s', now)
        tolerance = self.speed / p.stream_hz + 0.01
        if cross > p.tracking_m:
            return self._brake(now, 'tracking', f'cross-track {cross:.2f} m exceeds the {p.tracking_m:g} m allowance')
        if self.validity <= p.stopping_s + p.reaction_s:
            return self._brake(now, 'expired', f'permission deadline in {self.validity:.2f} s leaves too little '
                                               f'time to stop ({p.stopping_s + p.reaction_s:g} s)')
        if len(points) > 1 and tuple(progress) >= tuple(end):
            return self._brake(now, 'prefix', 'vehicle is at or beyond the permitted endpoint; stopping margin violated',
                               violated=self.margin)
        target, final, _ = self._target(value, self.pose, progress)
        if math.dist(self.pose, target) <= p.stopping_m:
            violated = self.margin if self.margin < -tolerance else None
            if not final: return self._brake(now, 'corner', f'stopping at corner {self.segment + 1}')
            if self._reaches_goal(value): return self._brake(now, 'arrival', 'approaching the goal', violated)
            return self._brake(now, 'prefix', 'permitted interval ends before the goal; waiting for clearance', violated)
        if self.margin < -tolerance:
            return self._brake(now, 'prefix', f'permitted interval shrank inside the stopping distance '
                                              f'(margin {self.margin:.2f} m)', violated=self.margin)
        if p.heading == 'travel' and self._face_segment(now, value, progress, target): return
        if not self._stream.due(now): return
        self._send(now, value, target)

    def _face_segment(self, now, value, progress, target):
        """Point along the segment; turn in place on the route before flying it. True while turning."""
        p, v = self.profile, self.vehicle
        points = value['points']
        anchor = point_at(points, progress) if len(points) > 1 else self.pose
        if math.dist(anchor[:2], target[:2]) < 0.05: return False  # at the target: keep the heading
        course = course_deg(anchor, target)
        error = heading_error_deg(course, v.heading_deg)
        self.heading_deg = course
        if abs(error) <= p.turn_tolerance_deg or self.speed > p.hold_speed_mps * 4:
            self.turning = False
            return False
        if not self.turning:
            self.turning = True
            self._record('execution.turn', dict(segment=self.segment, from_deg=v.heading_deg, to_deg=course,
                                                error_deg=error, pose=self.pose))
        self.turn_error_deg = error
        self.reason = f'turning to face segment {self.segment}'
        if self._stream.due(now):
            self._send(now, value, [anchor[0], anchor[1], anchor[2]])  # hold on the route while turning
        return True

    def _send(self, now, value, target):
        order = self._position_target(target)
        try: result = self.link.send_position_target(order)
        except Exception as exc: return self._fail(now, f'position target failed: {exc}')
        skipped = self._stream.advance(now)  # never burst to catch up
        self.targets_sent += 1
        self.target = list(target)
        entry = dict(t_s=now, target=list(target), accepted=result.accepted, reason=result.reason,
                     request_id=result.request_id, planner_session=value['session'], sequence=value['sequence'],
                     geometry_revision=value['geometry_revision'], end=list(value['clearance']['end']),
                     permitted=permitted_points(value), segment=self.segment, progress=list(self.progress),
                     skipped_slots=skipped)
        self.sent.append(entry)
        self._record('execution.target', entry)
        if not result.accepted: self._fail(now, f'position target rejected: {result.reason}')

    def _position_target(self, target):
        p = self.profile
        frames = ()
        try: frames = self.link.capabilities().frames
        except Exception: pass
        x, y, z = target
        if 'map' in frames or not frames:
            return PositionTarget(frame='map', x=x, y=y, z=z, heading_deg=self.heading_deg, max_speed_mps=p.speed_mps)
        north, east, down = ProviderFrame.from_context(self.snapshot).map_to_ned(x, y, z)
        return PositionTarget(frame='local_ned', north_m=north, east_m=east, down_m=down,
                              heading_deg=self.heading_deg, max_speed_mps=p.speed_mps)

    def _step_braking(self, now):
        p, state = self.profile, self.vehicle
        stopped = state is not None and state.mode == 'HOLD' and self.speed is not None and self.speed <= p.hold_speed_mps
        if stopped:
            self._below_since = now if self._below_since is None else self._below_since
            if now - self._below_since >= p.hold_dwell_s: return self._confirm_hold(now)
        else:
            self._below_since = None
        if now - self._brake_started > p.hold_timeout_s:
            self._fail(now, f'HOLD not confirmed within {p.hold_timeout_s:g} s (speed '
                            f'{"n/a" if self.speed is None else f"{self.speed:.2f} m/s"}, mode '
                            f'{state.mode if state else "n/a"}) while stopping: {self.reason}', send_hold=False)

    def _confirm_hold(self, now):
        src, kind = self.source, self.hold_kind
        generation = None
        if src.active and (src.restart_stop or src.stop_generation > src.handled_stop):
            restart = src.restart_stop
            src.confirm_stopped(src.stop_generation)
            generation = src.stop_generation
            if restart: kind = 'restart'
            elif kind == 'corner':
                # The stop that handled a withdrawal released the route: continuing
                # "along the same route" is no longer possible.
                kind = 'replacement' if (src.value.get('clearance') or {}).get('eligible') else 'unavailable'
        self._record('execution.hold', dict(phase='confirmed', kind=kind, reason=self.reason, pose=self.pose,
                                            progress=self.progress, speed_mps=self.speed, stop_generation=generation,
                                            violated_margin_m=self.violated_margin_m,
                                            confirm_s=now - self._brake_started))
        if kind == 'arrival':
            if self._goal_distance(self.pose) <= self.profile.arrival_m:
                if src.active: src.release('completed')
                return self._transition('COMPLETE', 'goal reached; HOLD confirmed', kind)
            kind = 'arrival-unconfirmed'
            return self._transition('HOLDING', f'stopped {self._goal_distance(self.pose):.2f} m from the goal, outside '
                                               f'the {self.profile.arrival_m:g} m arrival gate; Resume after validation', kind)
        if kind == 'cancel':
            if src.active: src.release('stopped')
            return self._transition('CANCELLED', 'cancelled by operator; HOLD confirmed', kind)
        if kind in ('replacement', 'goal', 'restart', 'tracking') and src.active: src.release('stopped')
        if kind == 'restart': src.start_required = True
        return self._transition('HOLDING', self._holding_reason(kind), kind)

    def _holding_reason(self, kind):
        return {'corner': 'stopped at a corner; continuing when still permitted',
                'replacement': 'stopped for a route change; continuing when a route from this pose is validated',
                'goal': 'goal changed or cleared; Start required for the new goal',
                'restart': 'planner, provider or frame restarted; Start required',
                'pause': 'paused by operator; Resume when ready',
                'prefix': 'permitted interval exhausted before the goal; Resume once clearance extends'
                }.get(kind, f'held ({kind}): {self.reason}; Resume after fresh validation')

    def _step_holding(self, now):
        src, p = self.source, self.profile
        if src.state == 'STOP_REQUIRED' and src.active and self.speed is not None and self.speed <= p.hold_speed_mps:
            restart = src.restart_stop
            src.confirm_stopped(src.stop_generation)
            self._record('execution.hold', dict(phase='confirmed', kind=self.hold_kind, reason='stop generation while held',
                                                pose=self.pose, progress=self.progress,
                                                stop_generation=src.stop_generation))
            kind = self.hold_kind
            if restart: kind = 'restart'; src.start_required = True
            elif kind == 'corner': kind = 'replacement' if (src.value.get('clearance') or {}).get('eligible') else 'unavailable'
            self._transition('HOLDING', self._holding_reason(kind), kind)
        if self.hold_kind not in START_REQUIRED and _goal_key(self.snapshot.get('goal')) != self.authorized_goal:
            if src.active: src.release('stopped')
            return self._transition('HOLDING', self._holding_reason('goal'), 'goal')
        if self.hold_kind == 'corner' and not src.active:
            # The route was released while stopped here; a validated route from
            # this pose continues exactly like a planned replacement.
            self._transition('HOLDING', self._holding_reason('replacement'), 'replacement')
        if self.hold_kind == 'corner':
            if not (src.state == 'READY' and src.active
                    and src.value['geometry_revision'] == self.active['geometry_revision']):
                return
            corner = self.active['points'][self.segment + 1]
            if math.dist(self.pose, corner) > p.join_m:
                # Too far from the corner to continue segment by segment: the
                # route cannot be resumed as it is, so release it and wait for
                # the route dnav plans from this stopped pose.
                if src.active: src.release('stopped')
                return self._transition('HOLDING', f'stopped {math.dist(self.pose, corner):.2f} m from the corner, '
                                                   f'beyond the {p.join_m:g} m join allowance; Resume after validation',
                                        'tracking')
            self.segment += 1
            self.active = src.value
            self._stream.reset(now)
            return self._transition('EXECUTING', f'continuing along segment {self.segment}')
        if self.hold_kind == 'replacement' and src.state == 'READY' and not src.active and not src.start_required:
            src.activate()
            self._activate(now, src.value)
            return self._transition('EXECUTING', 'route validated from the stopped pose; continuing under the same Start')

    def _step_complete(self, now):
        if self.source.state == 'READY' and _goal_key(self.snapshot.get('goal')) not in (None, self.authorized_goal):
            self._transition('READY', 'new goal admitted; Start required')

    def _step_terminal(self, now):
        pass

    # -- status, history and recording ------------------------------------

    def _diagnostics(self):
        try: return self.link.diagnostics()
        except Exception: return {}

    def _publish(self, now):
        src, p = self.source, self.profile
        active = self.active if self.state in MOVING + ('HOLDING',) else None
        unavailable = {}
        if self.speed is None: unavailable['speed_mps'] = 'vehicle state unavailable'
        if self.target is None: unavailable['commanded_target'] = f'no target while {self.state}'
        if active is None:
            for key in ('progress', 'tracking_error_m', 'remaining_permitted_m', 'stopping_margin_m'):
                unavailable[key] = 'no active route'
        pose = None
        try: pose = validate_pose(self.snapshot.get('pose'), self.snapshot)
        except (ValueError, KeyError, TypeError): pose = None
        self.sequence += 1
        self.status = dict(
            schema=STATUS_SCHEMA, vehicle_id=self.id, session=self.session, sequence=self.sequence,
            time_s=float(self.snapshot.get('time_s', now)), stop_generation=src.handled_stop,
            planner_session=src.session, planner=src.planner, state=self.state, reason=self.reason or self.state,
            hold_kind=self.hold_kind, dry_run=False, owns_control=self.owns_control, lease_state=self.lease_state,
            goal=self.snapshot.get('goal'), authorized_goal=None if self.authorized_goal is None else list(self.authorized_goal[:3]),
            start_required=bool(src.start_required or self.hold_kind in START_REQUIRED),
            disposition=src.disposition,
            geometry_revision=active['geometry_revision'] if active else src.revision,
            navigation_sequence=src.sequence, progress=list(self.progress) if active and self.progress else None,
            pose=pose, vehicle_pose=self.pose, speed_mps=self.speed, speed_cap_mps=p.speed_mps,
            commanded_target=self.target,
            tracking_error_m=self.cross_track if active else None, tracking_allowance_m=p.tracking_m,
            remaining_permitted_m=self.remaining if active else None,
            stopping_margin_m=self.margin if active else None,
            validity_remaining_s=self.validity if active else None,
            violated_margin_m=self.violated_margin_m, targets_sent=self.targets_sent, missions=self.missions,
            heading_deg=None if self.vehicle is None else self.vehicle.heading_deg,
            commanded_heading_deg=self.heading_deg, heading_mode=p.heading, turning=self.turning,
            turn_error_deg=self.turn_error_deg if self.turning else None,
            controls=dict(self.control_results), unavailable=unavailable, auto=self.auto,
            launch_phase=self.launch_phase, waiting_for=self.missing_roles(), permission=p.permission,
            bus_error=self.bus_error, coverage_request=self.coverage_request, profile=p.digest, profile_calibrated=p.calibrated,
            recording_error=self.report_error, vehicle_fault=self.vehicle_fault)
        try: self.output.write(self.status)
        except (ValueError, RuntimeError) as exc: self.report_error = f'status not published: {exc}'
        if self._samples.due(now):
            self._samples.advance(now)
            # The display timeline shares the sample cadence, so its length is
            # bounded by HISTORY_S * stream_hz however fast the loop polls.
            t = self.status['time_s']
            self.history.append((t, self.speed, p.speed_mps, self.cross_track if active else None, p.tracking_m,
                                 self.margin if active else None, self.state))
            while self.history and (t - self.history[0][0] > HISTORY_S or t < self.history[0][0]):
                self.history.popleft()
            self._record('execution.sample', dict(
                state=self.state, hold_kind=self.hold_kind, pose=self.pose, speed_mps=self.speed,
                speed_cap_mps=p.speed_mps, cross_track_m=self.cross_track if active else None,
                tracking_allowance_m=p.tracking_m, remaining_permitted_m=self.remaining if active else None,
                stopping_margin_m=self.margin if active else None,
                validity_remaining_s=self.validity if active else None,
                geometry_revision=active['geometry_revision'] if active else None,
                planner_session=src.session, segment=self.segment if active else None,
                progress=list(self.progress) if active and self.progress else None, target=self.target,
                mode=self.vehicle.mode if self.vehicle else None, owns_control=self.owns_control,
                heading_deg=None if self.vehicle is None else self.vehicle.heading_deg,
                commanded_heading_deg=self.heading_deg, turning=self.turning,
                epoch=[self.snapshot.get('provider_id'), self.snapshot.get('localization_epoch'),
                       self.snapshot.get('clock_epoch')]))
        self._view = self._make_view()

    def _make_view(self):
        value = self.source.value
        active = self.active if self.state in MOVING + ('HOLDING',) else None
        highlight = ()
        if active and len(active['points']) > 1 and self.segment + 1 < len(active['points']):
            highlight = (active['points'][self.segment][:2], active['points'][self.segment + 1][:2])
        grid = None
        if self.maps is not None:
            grid = next((st.grid for st in list(self.maps.states.values()) if st.grid is not None), None)
        return dict(status=dict(self.status), history=list(self.history), events=list(self.events), grid=grid,
                    track=[pt for pt in self.track[-3000:]], active=None if active is None else active['points'],
                    proposal=value.get('points', []), permitted=permitted_points(active or (
                        value if self.source.state == 'READY' else {})),
                    highlight=highlight, target=self.target, goal=(self.snapshot.get('goal') or {}).get('position'),
                    vehicle=None if self.pose is None or self.vehicle is None else
                    (self.pose[0], self.pose[1], self.vehicle.heading_deg))

    def view(self):
        """An immutable-by-convention snapshot for display threads; never mutated after publication."""
        return self._view

    def _record(self, kind, data, grids=None):
        data = dict(data, session=self.session, t_s=self.snapshot.get('time_s'), executor_state=self.state,
                    route=dict(planner_session=self.source.session, sequence=self.source.sequence,
                               geometry_revision=self.source.revision, stop_generation=self.source.stop_generation))
        if kind in NOTABLE:
            self.events.append(dict(kind=kind, t_s=data['t_s'], data={k: v for k, v in data.items() if k != 'snapshot'},
                                    points=(data.get('snapshot') or {}).get('points')))
        if self.recorder is not None:
            # State transitions, holds and the shutdown record are what the
            # report reconstructs decisions from; they are admitted even when
            # the recording queue is starved, while samples may drop.
            vital = kind in ('execution.transition', 'execution.hold', 'execution.shutdown')
            if self.recorder.record(kind, data, grids, vital=vital) is None and not self.report_error:
                self.report_error = 'recording dropped events (see archive gap markers)'

    def _adopt_recording(self):
        sid = self.snapshot.get('session_id')
        if not self.record_enabled or not sid or sid == self.report_session: return
        self._finish_recording(background=True)
        self.report_session = sid
        try:
            directory = next_archive_dir(Path(self.snapshot['report_root']) / 'dway')
            self.recorder = Recorder(directory, dict(mode='dynamic-flight', session=self.session, instance=self.id,
                                                     planner=self.source.planner, profile=asdict(self.profile),
                                                     profile_digest=self.profile.digest,
                                                     client_id=getattr(self.link, 'client_id', '')),
                                     module='dway', queue_bytes=self.recording_queue_bytes,
                                     disk_bytes=self.recording_disk_bytes)
            self.report_dir = directory
            self._record('retained_state', dict(
                note='segment baseline after start or rollover; not a flight authorization',
                state=self.state, reason=self.reason, hold_kind=self.hold_kind, authorized_goal=self.authorized_goal,
                active=self.active, context=self.snapshot, owns_control=self.owns_control, missions=self.missions))
        except (OSError, ValueError) as exc:
            self.recorder, self.report_dir, self.report_error = None, None, f'recording unavailable: {exc}'

    def _finish_recording(self, *, background):
        recorder, self.recorder = self.recorder, None
        if recorder is None: return
        def finish():
            started = time.perf_counter()
            try:
                recorder.close()
            except Exception as exc:  # e.g. the run directory was removed under a live recording
                self.report_error = f'recording could not be closed: {type(exc).__name__}: {exc}'
                return
            self.finish_timings_s = dict(recorder_close=round(time.perf_counter() - started, 3))
            if not self.report_enabled: return
            try:
                from dway.flightlog import build_report
                started = time.perf_counter()
                self.reports.append(build_report(recorder.directory))
                self.finish_timings_s['report'] = round(time.perf_counter() - started, 3)
            except Exception as exc:  # reports never change flight behaviour
                self.report_error = f'report failed: {type(exc).__name__}: {exc}'
        if background:
            thread = threading.Thread(target=finish, name='dway-report', daemon=True)
            thread.start(); self._report_threads.append(thread)
        else:
            finish()

    # -- shutdown ------------------------------------------------------------

    def _release_lease(self, why):
        try: result = self.link.release_control()
        except Exception as exc:
            self._record('execution.lease', dict(event='release', accepted=False, reason=str(exc), why=why))
            return False
        self._record('execution.lease', dict(event='release', accepted=result.accepted, reason=result.reason, why=why))
        self.owns_control = False
        self.lease_state = 'released' if result.accepted else f'release refused: {result.reason}'
        return result.accepted

    def shutdown(self, reason='shutdown', timeout_s=None):
        """Request HOLD, wait a bounded time for a measured stop, record it, release control. No landing."""
        if self.closed: return self.status
        p = self.profile
        timeout = p.hold_timeout_s if timeout_s is None else timeout_s
        result = dict(reason=reason, owned=self.owns_control, hold=None, confirmed=None, released=None,
                      state=self.state, timings_s={})
        phase_wall = [time.perf_counter()]

        def phase(name):
            now_wall = time.perf_counter()
            result['timings_s'][name] = round(now_wall - phase_wall[0], 3)
            phase_wall[0] = now_wall
        # Shutdown is bounded: a vehicle that has gone away (dsim "Kill all" stops
        # the simulator together with every module) answers nothing, so no
        # acknowledgement is waited for longer than a second.
        ack_timeout = getattr(self.link, 'ack_timeout_s', None)
        if ack_timeout is not None: self.link.ack_timeout_s = min(ack_timeout, 1.0)

        def stale(state):
            # The link keeps serving the last status after the vehicle exits, so
            # liveness is whether that status is still changing on the wall clock.
            return (state is None or not state.link_connected or not state.mode
                    or self.wall() - state.sample_wall_s > p.max_state_age_s)

        try: current = self.link.state()
        except Exception: current = None
        if self.owns_control and stale(current):
            result.update(hold=dict(accepted=False, reason='vehicle status stale; not sent'), confirmed=False,
                          released=False, hold_unconfirmable='vehicle status stale or unavailable; the vehicle '
                                                             'watchdog applies')
            self.owns_control, self.lease_state = False, 'abandoned: vehicle status stale at shutdown'
            phase('hold_wait')
        if self.owns_control:
            try:
                hold = self.link.hold()
                result['hold'] = dict(accepted=hold.accepted, reason=hold.reason)
            except Exception as exc:
                result['hold'] = dict(accepted=False, reason=str(exc))
            started_wall, below = self.wall(), None
            confirmed = False
            while self.wall() - started_wall < timeout:
                state = self.link.state()
                if stale(state):
                    # Nothing will ever confirm the stop: do not wait it out.
                    result['hold_unconfirmable'] = 'vehicle status stale or unavailable'
                    break
                speed = math.hypot(state.vx_mps, state.vy_mps)
                now = self.clock()
                if state.link_connected and state.mode == 'HOLD' and speed <= p.hold_speed_mps:
                    below = now if below is None else below
                    if now - below >= p.hold_dwell_s: confirmed = True; break
                else:
                    below = None
                try: self.link.heartbeat()
                except Exception: pass
                self.sleep(0.05)
            result['confirmed'] = confirmed
            phase('hold_wait')
            if result.get('hold_unconfirmable'):
                result['released'] = False
                self.owns_control, self.lease_state = False, 'abandoned: vehicle status stale at shutdown'
            else:
                result['released'] = self._release_lease('shutdown')
            phase('release')
        if ack_timeout is not None: self.link.ack_timeout_s = ack_timeout
        if self.state in MOVING + ('HOLDING', 'READY', 'WAITING'):
            self._transition('FAILED' if self.state in MOVING else self.state,
                             f'{reason}: mission not complete' if self.state in MOVING else self.reason, self.hold_kind)
        self._record('execution.shutdown', result)
        self.closed = True
        try:
            self.output.write(dict(self.status, sequence=self.sequence + 1, state='CLOSED', owns_control=False,
                                   commanded_target=None, reason=f'{reason}: HOLD '
                                   f'{"confirmed" if result["confirmed"] else "not confirmed" if result["owned"] else "not owned"}; '
                                   f'control {"released" if result["released"] else "not held"}'))
        except (ValueError, RuntimeError):
            pass
        self.shutdown_result = result
        phase('status')
        self._finish_recording(background=False)
        for thread in self._report_threads: thread.join(timeout=30)
        phase('recording_and_report')
        if self.bus is not None:
            try:
                self.bus.publish('module.goodbye', run_id=self.session, payload=dict(state=self.state, reason=reason))
            except Exception: pass
            try: self.bus.close()
            except Exception: pass
        for handle in (self.output, self.navigation):
            try: handle.close()
            except Exception: pass
        if self.maps is not None:
            try: self.maps.close()
            except Exception: pass
        phase('close')
        return result
