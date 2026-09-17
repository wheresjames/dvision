"""An in-memory stand-in for everything dmsn reads and writes.

`FakePorts` has the surface of :class:`dmsn.ports.LivePorts` with no pymembus:
a context snapshot that enforces the same goal-authority rule as
:class:`dcmn.context.Context`, execution and navigation snapshots a test sets
directly, a coverage extent, a provider status plane, and lists of what dmsn
wrote -- every status and intent validated against its schema, so a test that
passes here wrote records the real planes would accept.

`FakeExecutor` is the smallest dway a mission needs: it launches on a launch
intent, flies a goal to COMPLETE after a few steps, and honours hold and
resume. It is a script, not a vehicle.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

from dcmn.context import FRAME_ID, pose_context
from dcmn.mission import validate_intent, validate_status


class FakePorts:
    def __init__(self, report_root: str | Path | None = None, *, instance: str = 'fake', pose=(0., 0., 1.5, 90.),
                 coverage=(-50., -50., 50., 50.)) -> None:
        self.instance = instance
        self.errors: dict[str, str] = {}
        self.time_s = 0.0
        self.context: dict[str, Any] = dict(
            schema='dvision2.context.v1', session_id='fake-session', report_root=str(report_root or '/tmp/fake'),
            frame_id=FRAME_ID, localization_epoch=0, clock_domain_id=instance, clock_epoch=0, time_s=0.0,
            provider_kind='fixture', authority=None, authority_epoch=0, goal=None, goal_revision=0,
            reset_revision=0, mapping_request=None, history=[])
        self.pose = pose
        self.execution: dict[str, Any] = {}
        self.navigation: dict[str, Any] = {}
        self.coverage_bounds = coverage
        self.status: dict[str, str] | None = None
        self.written: list[dict[str, Any]] = []
        self.intents: list[dict[str, Any]] = []
        self.resets: list[dict[str, Any]] = []
        self.published: list[tuple[str, dict]] = []
        self.inbox: list = []
        self.claimed = False
        self.closed = False
        self._grids: dict[str, Any] = {}
        #: The detections plane a test publishes (``dcmn.detections`` shape), or ``{}`` for none.
        self.detections: dict[str, Any] = {}
        self.detections_reads = 0
        self.detections_plane = f'/dvision2.{instance}.detections.dalg'

    # -- the test's levers ---------------------------------------------------------

    def advance(self, seconds: float) -> None:
        self.time_s += seconds

    def take_goal(self, writer: str, position, role: str = 'ui') -> None:
        """Another authority takes the goal with an explicit handoff."""
        self.set_goal(writer, position, handoff=True, role=role)

    # -- LivePorts surface -----------------------------------------------------------

    def claim(self) -> None:
        self.claimed = True

    def read_context(self) -> dict[str, Any]:
        value = copy.deepcopy(self.context)
        value['time_s'] = self.time_s
        if self.pose is not None:
            x, y, z, heading = self.pose
            value['pose'] = dict(x_m=x, y_m=y, z_m=z, heading_deg=heading, roll_deg=0., pitch_deg=0.,
                                 **pose_context(self.instance, 0, 0, self.time_s))
        return value

    def read_execution(self) -> dict[str, Any]:
        return copy.deepcopy(self.execution)

    def read_navigation(self) -> dict[str, Any]:
        return copy.deepcopy(self.navigation)

    def read_detections(self) -> dict[str, Any]:
        self.detections_reads += 1
        return copy.deepcopy(self.detections)

    def poll_maps(self) -> None:
        pass

    def grids(self) -> dict[str, Any]:
        return dict(self._grids)

    def coverage(self):
        return self.coverage_bounds

    def provider_status(self):
        return None if self.status is None else dict(self.status)

    def set_goal(self, writer, position, *, handoff=False, role='mission'):
        authority = self.context['authority']
        if authority and authority['id'] != writer and not handoff:
            raise ValueError(f"goal authority belongs to {authority['role']} {authority['id'][:12]}; "
                             'explicit handoff required')
        if not authority or authority['id'] != writer: self.context['authority_epoch'] += 1
        self.context['authority'] = dict(id=writer, role=role)
        self.context['goal_revision'] += 1
        self.context['goal'] = None if position is None else dict(
            position=[float(v) for v in position], frame_id=FRAME_ID, localization_epoch=0, clock_epoch=0,
            revision=self.context['goal_revision'], authority_epoch=self.context['authority_epoch'],
            authority=writer, role=role)
        return self.context['goal']

    def release_authority(self, writer) -> bool:
        if (self.context['authority'] or {}).get('id') != writer: return False
        self.context['authority'] = None
        return True

    def request_reset(self, bounds, *, by='dmsn') -> int:
        self.context['reset_revision'] += 1
        self.resets.append(dict(bounds=list(bounds), by=by, revision=self.context['reset_revision']))
        return self.context['reset_revision']

    def write_status(self, value) -> None:
        import json
        validate_status(value)
        if len(json.dumps(value).encode()) > 65536: raise ValueError('mission snapshot exceeds 64 KiB')
        self.written.append(value)

    def write_intent(self, value) -> None:
        validate_intent(value)
        self.intents.append(value)

    def bus_publish(self, event_type, *, run_id='', payload=None) -> None:
        self.published.append((event_type, dict(payload or {})))

    def bus_receive(self) -> list:
        events, self.inbox = self.inbox, []
        return events

    def close(self) -> None:
        self.closed = True


class FakeExecutor:
    """A scripted target-mode dway on a `FakePorts`: launch, fly, complete, hold, turn, land.

    ``land_result`` scripts a landing's verdict: ``completed`` (on the ground
    after ``launch_steps``), ``failed`` (the timeout) or ``refused``.
    """

    def __init__(self, ports: FakePorts, *, wait_for_mission: bool = True, flight_steps: int = 5,
                 launch_steps: int = 3) -> None:
        self.ports = ports
        self.wait_for_mission = wait_for_mission
        self.flight_steps, self.launch_steps = flight_steps, launch_steps
        self.land_result = 'completed'
        self.turns: list[float] = []
        self.landings = 0
        self.state, self.reason, self.hold_kind = 'WAITING', 'waiting for a mission launch intent', None
        self.launch_phase = None
        self.launch_authorized = not wait_for_mission
        self.mission_hold = False
        self.authorized_goal = None
        self.ack = None
        self._handled = None
        self._countdown = 0
        self._goal_key = None
        self.session = 'fake-dway'
        self.publish()

    def publish(self) -> None:
        self.ports.execution = dict(
            schema='dvision2.execution.v1', state=self.state, reason=self.reason, hold_kind=self.hold_kind,
            launch_phase=self.launch_phase, auto=True, wait_for_mission=self.wait_for_mission,
            mission_hold=self.mission_hold, authorized_goal=self.authorized_goal,
            mission_intent=None if self.ack is None else dict(self.ack), owns_control=self.launch_phase == 'airborne',
            session=self.session)

    def step(self) -> None:
        intents = self.ports.intents
        if intents and (intents[-1]['session'], intents[-1]['revision']) != self._handled:
            value = intents[-1]
            self._handled = (value['session'], value['revision'])
            self.ack = dict(session=value['session'], revision=value['revision'], kind=value['kind'],
                            state='accepted', reason='')
            if value['kind'] == 'launch':
                self.launch_authorized = True
            elif value['kind'] == 'hold':
                self.mission_hold = True
                if self.state == 'EXECUTING': self.state, self.hold_kind, self.reason = 'HOLDING', 'pause', 'paused'
            elif value['kind'] == 'resume':
                self.mission_hold = False
                if self.state == 'HOLDING': self.state, self.hold_kind = 'WAITING', None
            elif value['kind'] == 'yaw_to':
                if self.launch_phase != 'airborne' or self.state not in ('HOLDING', 'COMPLETE'):
                    self.ack.update(state='refused', reason=f'yaw_to applies to a confirmed HOLD, not {self.state}')
                else:
                    heading = float(value['args']['heading_deg'])
                    self.ports.pose = (*self.ports.pose[:3], heading)
                    self.turns.append(heading)
                    self.ack.update(state='completed', reason=f'heading {heading:g}° reached; HOLD confirmed')
            elif value['kind'] == 'land':
                if self.launch_phase != 'airborne' or self.land_result == 'refused':
                    self.ack.update(state='refused', reason='not airborne')
                else:
                    self.state, self.reason, self._countdown = 'LANDING', 'descending', 0
        if self.state == 'LANDING':
            self._countdown += 1
            if self._countdown >= self.launch_steps:
                self._countdown = 0
                if self.land_result == 'failed':
                    self.state, self.reason = 'FAILED', 'landing not confirmed within 60 s'
                    self.ack.update(state='failed', reason='landing not confirmed within 60 s')
                else:
                    self.landings += 1
                    self.state, self.reason = 'WAITING', 'landed; waiting for a mission launch intent'
                    self.launch_phase, self.launch_authorized = None, False
                    self.ports.pose = (self.ports.pose[0], self.ports.pose[1], 0.0, self.ports.pose[3])
                    self.ack.update(state='completed', reason='landed; vehicle disarmed on the ground')
            return self.publish()
        if self.launch_phase != 'airborne':
            if self.launch_authorized:
                self.state, self.reason = 'TAKING_OFF', 'climbing'
                self._countdown += 1
                if self._countdown >= self.launch_steps:
                    self.state, self.reason, self.launch_phase = 'WAITING', 'airborne in HOLD', 'airborne'
                    if self.ack and self.ack['kind'] == 'launch': self.ack['state'] = 'completed'
                    self._countdown = 0
            return self.publish()
        goal = self.ports.context['goal']
        key = None if goal is None else (goal['authority'], goal['authority_epoch'], goal['revision'])
        if key != self._goal_key:
            self._goal_key, self._countdown = key, 0
            if goal is not None and not self.mission_hold and self.state != 'HOLDING':
                self.state, self.reason = 'EXECUTING', 'flying'
        if self.state == 'EXECUTING' and not self.mission_hold:
            self._countdown += 1
            if self._countdown >= self.flight_steps and goal is not None:
                self.ports.pose = (goal['position'][0], goal['position'][1], 1.5, 90.)
                self.state, self.reason, self.authorized_goal = 'COMPLETE', 'arrived', list(key)
        elif self.state in ('WAITING', 'COMPLETE') and goal is not None and key and list(key) != self.authorized_goal \
                and not self.mission_hold:
            self.state, self.reason, self._countdown = 'EXECUTING', 'flying', 0
        self.publish()


def drive(run, executor=None, *, ports: FakePorts, steps: int = 50, dt: float = 0.2, until=None) -> bool:
    """Step a runner (and a fake executor) in fake time until ``until(run)`` or the step limit."""
    for _ in range(steps):
        if until is not None and until(run): return True
        if executor is not None: executor.step()
        ports.advance(dt)
        run._pace.reset()
        run.step()
    return bool(until(run)) if until is not None else False


def distance(a, b) -> float:
    return math.dist(a, b)
