"""The planning loop: discover evidence, derive cost, plan, publish, record.

Shared by the window and the headless run, so what an archive contains and what
a window shows come from one object rather than two that agree. dnav holds no
control lease and sends no command: it is a route producer. A route through
never-observed cells is a proposal, not verified clearance; executing one needs
an executor with a verified horizon, which is outside this module.

Inputs are provider-neutral:

* **session context** (:mod:`dcmn.context`) -- data clock, frame, localization
  and clock epochs, the labelled pose, the goal and its authority, report root;
* **evidence plane** (:mod:`dcmn.maps`) -- one coherent generation of grids,
  never mixed with another.

A plan needs a goal, a valid pose no older than ``pose_max_age`` in the data
clock, and evidence whose frame and epochs match the pose and the goal. Every
change of evidence generation invalidates cached costs and the current route.
Each planning attempt -- including those that produce no route -- is recorded
with the exact pose, goal, policy, planner and evidence revisions it used.

Two facts about evidence are kept apart, because conflating them is how an
operator ends up debugging the wrong thing: the **data fact** (the newest record
is older than the horizon) and the **liveness fact** (the producer's heartbeat is
gone). Both are ``stale_map``; the reason says which.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Sequence

from dcmn.archive import Recorder, next_archive_dir
from dcmn.context import Context, validate_goal, validate_pose
from dcmn.health import IntakeMeter, grade, worst
from dcmn.maps import MapSession
from dcmn.module_bus import PipelineView, PymembusModuleBus, requests_shutdown
from dcmn.pacing import MAP_HZ, Paced
from dnav import route as R
from dnav.planners import DEFAULT_PLANNER, PLANNERS, build
from dnav.policy import CostPolicy, build_cost_map, load_policy

#: The bus role an evidence producer registers under.
PRODUCER_ROLE = 'algorithm'
#: The role dnav itself registers under. Not `controller`: nothing here holds a lease.
PLANNER_ROLE = 'planner'
#: What a route event carries; the full path stays in the archive.
MAX_PUBLISHED_WAYPOINTS = 64
#: Version of the attempt record's ``data`` layout.
ATTEMPT_SCHEMA = 'dvision2.planning-attempt.v1'


def parse_goal(text: str) -> tuple[float, float, float | None]:
    """``x,y`` or ``x,y,z`` in local-frame metres, with z left open when absent."""
    parts = [piece.strip() for piece in str(text).split(',')]
    if len(parts) not in (2, 3):
        raise ValueError('a goal is x,y or x,y,z in local-frame metres')
    values = [float(piece) for piece in parts]
    if not all(math.isfinite(v) for v in values): raise ValueError('goal coordinates must be finite')
    return (values[0], values[1], values[2] if len(values) == 3 else None)


def policy_from_record(value: dict[str, Any]) -> CostPolicy:
    """The exact policy an attempt used, from its recorded fields."""
    return CostPolicy(name=value['name'], occupied_threshold=value['occupied_threshold'],
                      inflation_m=value['inflation_m'], combine=value['combine'],
                      body_radius_m=value.get('body_radius_m', 0.0),
                      clearance_preference_m=value.get('clearance_preference_m', 0.5),
                      clearance_cost=value.get('clearance_cost', 1.0),
                      escape_margin=value.get('escape_margin', True),
                      schema_version=value['schema_version'], digest=value['digest'])


def replay_attempt(attempt: dict[str, Any]) -> R.Route:
    """Plan again from a reconstructed attempt (``ArchiveReader.reconstruct_attempt``).

    Uses only what the archive holds: grids, pose, goal, policy, planner and the
    stale set. A route identical to the recorded one is the proof the record
    is sufficient.
    """
    policy = policy_from_record(attempt['policy'])
    inputs = attempt['inputs']
    cost_map = build_cost_map(attempt['grids'], policy, stale=frozenset(inputs.get('stale', ())),
                              sim_time_s=inputs['time_s'])
    pose, goal = attempt['pose'], attempt['goal']['position']
    start = (pose['x_m'], pose['y_m'], pose['z_m'])
    target = (goal[0], goal[1], pose['z_m'] if len(goal) < 3 else goal[2])
    return build(attempt['planner']).plan(cost_map, start, target, policy)


class NavRun:
    """One planning session against one instance."""

    def __init__(self, instance_id: str, root: Path, *, policy: CostPolicy,
                 planner: str = DEFAULT_PLANNER,
                 goal: tuple[float, float, float | None] | None = None,
                 source: str | None = None, hz: float = MAP_HZ,
                 pose_max_age: float = .5, goal_role: str = 'ui',
                 recording_queue: int = 64 << 20, recording_disk: int = 4 << 30,
                 execution_profile=None, navigation_name="dnav",
                 profile_overrides=None, goal_from_provider: bool = False) -> None:
        if pose_max_age <= 0: raise ValueError('pose_max_age must be positive')
        self.id, self.root = instance_id, Path(root)
        self.policy = policy
        self.policy_name = policy.name
        self.source_filter = source
        self.pose_max_age = float(pose_max_age)
        self.goal_role = goal_role
        self.recording_queue, self.recording_disk = recording_queue, recording_disk
        self.context = Context(instance_id)
        self.snapshot: dict[str, Any] = {}
        self.session = MapSession(instance_id)
        self.bus = PymembusModuleBus(instance_id, PLANNER_ROLE, 'dnav', sim_time=self.sim_time_s)
        self.writer_id = f'dnav:{self.bus.process_id}'
        self.pipeline = PipelineView()
        self.planner_name = planner
        self.planner = build(planner)
        # Always built, never selected away from: an evidence-derived diagnostic
        # every route is quoted against. Not an oracle and not accuracy.
        self.control = build('control')
        self.goal: tuple[float, float, float] | None = None
        self.goal_error = ''
        self.pose_error = 'no session context yet'
        self._pending_goal: tuple | None = None
        self._pending_handoff = False
        #: Without a goal of its own, adopt the target the provider publishes
        #: (dsim's map ``*``), once per session and never over another goal.
        self.goal_from_provider = bool(goal_from_provider)
        self.provider_goal: dict[str, Any] | None = None
        self._provider_goal_session = None
        self._provider_goal_checked = float('-inf')
        if goal is not None: self.set_goal(*goal)
        self.route = R.failed(R.NO_GOAL, 'no goal set')
        self.control_route: R.Route | None = None
        self.cost_map = None
        self.history = R.RouteHistory()
        self.intake = IntakeMeter(0.0, basis='sim')
        self.plans = 0
        self.attempts = 0
        self.report_dir: Path | None = None
        self.run_id = ''
        self.recorder: Recorder | None = None
        self.shutdown_requested = False
        self.reason = ''
        self.partial = False
        self.closed = False
        self.snapshot_provider = None
        self.background_provider = None
        self._displayed_background = None
        self.transitions = 0
        self._time_s = 0.
        self._pace = Paced(hz)
        self._events: list[dict[str, Any]] = []
        self._reference_display: dict[str, Any] | None = None
        self._last_plan_key: tuple | None = None
        self._substitute: R.Route | None = None
        self._substitute_key: tuple | None = None
        self._last_heartbeat = -1e9
        self._hello_sent = False
        self._session_id: str | None = None
        self._generation: tuple | None = None
        self._arrivals: dict[str, tuple[int, float, int]] = {}
        self._arrival_order = 0
        self._trigger = 'start'
        self._algorithm_profiles: dict[str, str] = {}
        from dnav.clearance import ExecutionProfile
        from dnav.execution import RoutePublisher
        self.navigation = RoutePublisher(instance_id, ExecutionProfile.load(execution_profile, profile_overrides),
                                         navigation_name)
        self.policy = self.policy.for_execution(self.navigation.profile)

    # -- connections -------------------------------------------------------

    def sim_time_s(self) -> float:
        return self._time_s

    def connect(self) -> bool:
        try: self.snapshot = self.context.read()
        except ValueError as exc:
            self.snapshot = {}; self.pose_error = str(exc)
        if self.snapshot:
            self._time_s = float(self.snapshot.get('time_s', self._time_s))
            self._submit_pending_goal()
            self._adopt_provider_goal()
        self.bus.connect()
        self._adopt_session()
        return bool(self.snapshot)

    def _adopt_provider_goal(self) -> None:
        sid = self.snapshot.get('session_id')
        if (not self.goal_from_provider or not sid or sid == self._provider_goal_session
                or self._pending_goal is not None):
            return
        now = time.monotonic()
        if now - self._provider_goal_checked < 1.0: return
        self._provider_goal_checked = now
        if self.snapshot.get('goal') is not None:
            self._provider_goal_session = sid  # someone already set one: never override it
            self.provider_goal = dict(adopted=False, reason='a goal was already set')
            return
        from dcmn.provider_target import map_target, read_status
        position, reason = map_target(read_status(self.id), self.snapshot)
        if position is None:
            self.provider_goal = dict(adopted=False, reason=reason)
            if reason != 'provider status plane unavailable':
                self._provider_goal_session = sid  # a definite answer for this session
                self.note('goal.provider_target_unavailable', reason=reason)
            return
        self._provider_goal_session = sid
        self.provider_goal = dict(adopted=True, position=position, source='provider status target.*')
        self.note('goal.from_provider', position=position)
        self.set_goal(*position)

    def _adopt_session(self) -> None:
        """Follow the session the provider publishes; never construct a report root."""
        sid = self.snapshot.get('session_id')
        if not sid or sid == self._session_id: return
        previous = self.recorder
        if previous is not None:
            self.note('session.rollover', new_session=sid)
            # The picture first, then the background it used, then the archive
            # sealed: the report and its record can never disagree about which
            # reference revision was displayed.
            image = self.snapshot_image()
            self._archive_displayed_background()
            previous.close()
            self._write_report(image=image)
        self._session_id = sid
        self.run_id = sid
        self.report_dir = Path(self.snapshot['report_root']) / 'dnav'
        metadata = dict(module='dnav', session_id=sid, instance=self.id,
                        provider_kind=self.snapshot.get('provider_kind'),
                        frame=self.snapshot.get('frame'), policy=self._policy_record(),
                        planner=self.planner_name, pose_max_age_s=self.pose_max_age,
                        attempt_schema=ATTEMPT_SCHEMA)
        try:
            self.recorder = Recorder(next_archive_dir(self.report_dir), metadata, module='dnav',
                                     queue_bytes=self.recording_queue, disk_bytes=self.recording_disk)
        except OSError as exc:
            self.recorder = None
            self.note('recording.unavailable', reason=str(exc))
            return
        grids = self._grids()
        if previous is not None and grids:
            # Retained state, so this archive has no hidden dependency on the last.
            self.recorder.record('retained_state', dict(context=dict(self.session.context),
                                 route=self.route.as_dict()), grids)

    def vehicle(self) -> tuple[float, float, float, float] | None:
        """The validated pose as ``(x, y, z, heading)``, or None with ``pose_error`` set."""
        pose = self._pose()
        if pose is None: return None
        return (pose['x_m'], pose['y_m'], pose['z_m'], pose['heading_deg'])

    def _pose(self) -> dict[str, Any] | None:
        if not self.snapshot:
            self.pose_error = 'session provider unavailable'; return None
        try:
            pose = validate_pose(self.snapshot.get('pose'), self.snapshot, max_age=self.pose_max_age)
        except (ValueError, KeyError, TypeError) as exc:
            self.pose_error = str(exc); return None
        self.pose_error = ''
        return pose

    def poll_delay(self) -> float:
        return .02

    # -- goals ---------------------------------------------------------------

    @property
    def goal_descriptor(self) -> dict[str, Any] | None:
        return self.snapshot.get('goal') if self.snapshot else None

    @property
    def goal_xy(self) -> tuple[float, float] | None:
        goal = self.goal_descriptor
        return None if goal is None else (goal['position'][0], goal['position'][1])

    @property
    def goal_z(self) -> float | None:
        goal = self.goal_descriptor
        return None if goal is None or len(goal['position']) < 3 else goal['position'][2]

    @property
    def authority(self) -> dict[str, Any] | None:
        return self.snapshot.get('authority') if self.snapshot else None

    def set_goal(self, x_m: float, y_m: float, z_m: float | None = None, *, handoff: bool = False) -> bool:
        """Submit a goal as this dnav's authority; queued until a provider exists."""
        position = (float(x_m), float(y_m)) + (() if z_m is None else (float(z_m),))
        self._pending_goal, self._pending_handoff = position, handoff
        if self.snapshot: return self._submit_pending_goal()
        self.goal_error = 'goal queued until the session provider appears'
        return False

    def clear_goal(self, *, handoff: bool = False) -> bool:
        self._pending_goal, self._pending_handoff = ('clear',), handoff
        return self._submit_pending_goal() if self.snapshot else False

    def _submit_pending_goal(self) -> bool:
        if self._pending_goal is None: return False
        position = None if self._pending_goal == ('clear',) else list(self._pending_goal)
        try:
            self.context.set_goal(self.writer_id, position, role=self.goal_role,
                                  handoff=self._pending_handoff)
        except ValueError as exc:
            # Never overwrite another authority silently; say why instead.
            self.goal_error = str(exc)
            self.note('goal.rejected', reason=str(exc), position=position)
            self._pending_goal = None
            return False
        self.note('goal.submitted', position=position, handoff=self._pending_handoff)
        self._pending_goal = None; self.goal_error = ''
        self.snapshot = self.context.read()
        self._trigger = 'goal'
        return True

    def set_planner(self, name: str) -> None:
        if name not in PLANNERS: raise KeyError(name)
        self.planner_name = name
        self.planner = build(name)
        self._trigger = 'planner'

    def reload_policy(self) -> CostPolicy:
        """Re-read the policy file; the file stays the source of truth."""
        if self.policy.path is None: return self.policy
        self.policy = load_policy(str(self.policy.path), self.root).for_execution(self.navigation.profile)
        self._last_plan_key = None
        self._trigger = 'policy'
        self.note('policy.reloaded', digest=self.policy.digest[:12], **self.policy.values())
        return self.policy

    # -- the loop ----------------------------------------------------------

    def note(self, kind: str, **fields: Any) -> None:
        """One event for the summary log and the archive; never raises."""
        self._events.append(dict(time_s=round(self.sim_time_s(), 4), type=kind, **fields))
        del self._events[:-4096]
        if self.recorder is not None:
            self.recorder.record(kind, dict(time_s=self.sim_time_s(), **fields))

    def note_display(self, mode: str, **data: Any) -> None:
        """Record an operator display decision in this session's provenance.

        A reference background is optional decoration, but *showing* one is a
        fact about how the session was conducted: an operator who planned over
        a truth map was assisted, and a report that could not say so would
        make assisted and unassisted sessions comparable.
        Archived as an event and kept in the summary, where the evaluator
        reads it.
        """
        self._reference_display = dict(mode=mode, **data)
        self.note('display.reference', mode=mode, **data)

    def step(self) -> str:
        self.connect()
        wall = time.monotonic()
        self._presence(wall)
        for event in self.bus.receive():
            self.pipeline.observe(event, wall)
            if event.role == PRODUCER_ROLE:
                if event.type == 'module.goodbye':
                    self._algorithm_profiles.pop(event.process_id, None)
                elif event.type in ('module.hello', 'module.heartbeat') and event.payload.get('profile'):
                    self._algorithm_profiles[event.process_id] = str(event.payload['profile'])
            if requests_shutdown(event):
                self.shutdown_requested = True
                self.reason = str(event.payload.get('reason', 'instance shutdown requested'))
                self.note('shutdown.requested', reason=self.reason)
        self.session.poll()
        self._follow_generation()
        self._note_arrivals()
        if self._pace.due():
            self.replan()
            self.navigation.publish(self)
        return self.route.status

    def _follow_generation(self) -> None:
        """Invalidate cached cost and route when the evidence generation changes."""
        identity = self.session.identity or self.session.last_seen_identity
        generation = None if identity is None else (identity, tuple(sorted(self.session.context.items())))
        if generation == self._generation: return
        if self._generation is not None:
            self.transitions += 1
            self.cost_map = None; self.control_route = None; self._last_plan_key = None
            self._arrivals.clear()
            self.route = R.failed(R.STALE_MAP, 'evidence generation changed; cached costs and '
                                  'the previous route are invalid', sim_time_s=self.sim_time_s())
            self.note('mapping.transition', old=_describe(self._generation), new=_describe(generation))
            self.bus.publish('route.invalidated', run_id=self.run_id,
                             payload=dict(reason='evidence generation changed'))
        self._generation = generation
        self._trigger = 'generation'

    def _note_arrivals(self) -> None:
        """Arrival order of evidence revisions: what an attempt could have seen."""
        for sid, state in self.session.states.items():
            mark = (state.revision, state.sim_time_s)
            if state.grid is None or self._arrivals.get(sid, (None, None, None))[:2] == mark: continue
            self._arrival_order += 1
            self._arrivals[sid] = (*mark, self._arrival_order)

    def producer_alive(self) -> bool:
        """Whether a module claiming the ``maps`` capability is still announcing itself."""
        for member in self.pipeline.matching(PRODUCER_ROLE):
            capabilities = member.capabilities
            if isinstance(capabilities, dict) and capabilities.get('maps'):
                return True
        return False

    def sources(self) -> list[str]:
        known = sorted(self.session.states)
        if self.source_filter is None: return known
        return [sid for sid in known if sid == self.source_filter]

    def _grids(self) -> dict[str, Any]:
        grids = {}
        for sid in self.sources():
            grid = self.session.latest(sid)
            if grid is not None: grids[sid] = grid
        return grids

    def replan(self, *, force: bool = False) -> R.Route:
        """Produce the current answer, whatever it is."""
        sim_now = self.sim_time_s()
        grids = self._grids()
        stale = frozenset(sid for sid in self.sources() if self.session.stale(sid, sim_now))
        try:
            self.cost_map = (build_cost_map(grids, self.policy, stale=stale, sim_time_s=sim_now)
                             if grids else None)
        except ValueError as exc:
            # Mixed geometries mean mixed generations: never plan across them.
            self.cost_map = None
            self.note('evidence.rejected', reason=str(exc))
        pose = self._pose()
        goal = self.goal_descriptor
        key = (None if goal is None else goal.get('revision'), self.planner_name, self.policy.digest,
               None if self.cost_map is None else tuple(sorted(self.cost_map.revisions.items())),
               self._generation, bool(stale), self.producer_alive(), self.pose_error,
               None if pose is None else tuple(round(pose[k], 3) for k in ('x_m', 'y_m', 'z_m')))
        if not force and key == self._last_plan_key: return self.route
        self._last_plan_key = key
        trigger = 'forced' if force else self._trigger
        self._trigger = 'evidence_or_pose'
        started_wall = time.time()
        route = self._compute(sim_now, grids, stale, pose, goal, trigger)
        self._record(route, trigger, pose, goal, grids, stale, sim_now, started_wall)
        self._publish(route)
        return self.route

    def _compute(self, sim_now: float, grids: dict[str, Any], stale: frozenset[str],
                 pose: dict[str, Any] | None, goal: dict[str, Any] | None,
                 trigger: str) -> R.Route:
        common = dict(planner=self.planner_name, policy_digest=self.policy.digest, sim_time_s=sim_now)
        self.control_route = None
        self.goal = None
        if goal is None:
            reason = self.goal_error or 'set a goal to plan a route'
            history = [h for h in self.snapshot.get('history', ()) if h.get('kind') == 'goal.invalidated']
            if history and not self.goal_error: reason = f"goal withdrawn: {history[-1]['reason']}; reissue it"
            return R.failed(R.NO_GOAL, reason, **common)
        if pose is None:
            return R.failed(R.STALE_POSE, f'vehicle pose unusable: {self.pose_error}', **common)
        try: validate_goal(goal, self.snapshot)
        except ValueError as exc:
            return R.failed(R.FRAME_MISMATCH, f'{exc}; the goal authority must reissue it', **common)
        start = (pose['x_m'], pose['y_m'], pose['z_m'])
        position = goal['position']
        self.goal = (position[0], position[1], pose['z_m'] if len(position) < 3 else position[2])
        if not grids or self.cost_map is None:
            reason = self._absence_reason()
            if trigger == 'generation':
                # The transition withdrew the previous generation's grids, so
                # the absence is its consequence, not a producer that never
                # published. Said alone, the generic reason buries the cause.
                reason = f'evidence generation changed; {reason}'
            return R.failed(R.STALE_MAP, reason, goal=self.goal, start=start, **common)
        context = self.session.context
        for key in ('frame_id', 'localization_epoch', 'clock_epoch'):
            if context.get(key) != pose.get(key):
                return R.failed(R.FRAME_MISMATCH,
                                f'evidence {key} {context.get(key)!r} does not match the pose\'s '
                                f'{pose.get(key)!r}; waiting for evidence in the current epoch',
                                goal=self.goal, start=start, **common)
        missing = set(self.sources()) - grids.keys()
        if missing:
            return R.failed(R.STALE_MAP, 'waiting for evidence from: '+', '.join(sorted(missing)),
                            goal=self.goal, start=start, **common)
        if stale:
            names = ', '.join(sorted(stale))
            if not self.producer_alive():
                reason = f'{names}: the producer\'s heartbeat is gone from the module bus'
            else:
                age = max((self.session.age_s(sid, sim_now) or 0.0) for sid in stale)
                reason = (f'{names}: the newest evidence is {age:.1f} data-clock seconds old, past '
                          f'the {self.session.staleness_horizon_s:g} s horizon')
            return R.failed(R.STALE_MAP, reason, map_revision=self.cost_map.revision,
                            goal=self.goal, start=start, **common)
        geometry = self.cost_map.geometry
        for name, point in (('goal', self.goal), ('vehicle', start)):
            if geometry.to_cell(point[0], point[1]) is None:
                x0, y0, x1, y1 = geometry.bounds_m()
                return R.failed(R.OUTSIDE_COVERAGE,
                                f'{name} at {point[0]:.2f}, {point[1]:.2f} m is outside the evidence '
                                f'coverage [{x0:g}, {y0:g}] - [{x1:g}, {y1:g}] m; this does not mean no '
                                'route exists. Request a mapping reset with larger bounds',
                                map_revision=self.cost_map.revision, goal=self.goal, start=start,
                                diagnostics=dict(coverage=list(geometry.bounds_m())), **common)
        route = self.planner.plan(self.cost_map, start, self.goal, self.policy)
        self.control_route = self.control.plan(self.cost_map, start, self.goal, self.policy)
        self.plans += 1
        self.intake.record()
        return route

    def substitute_route(self) -> R.Route | None:
        """A route to the reachable point nearest a goal the planner cannot reach.

        None unless the current plan failed as ``goal_unreachable`` or
        ``no_route`` and the planner and profile support substitutes. Computed
        at most once per plan key; the publisher decides whether to use it.
        """
        radius = self.navigation.profile.goal_substitute_m
        if (radius <= 0 or self.route.status not in (R.GOAL_UNREACHABLE, R.NO_ROUTE) or self.cost_map is None
                or self.goal is None or not hasattr(self.planner, 'plan_near')):
            return None
        if self._substitute_key is None or self._substitute_key != self._last_plan_key:
            pose = self._pose()
            if pose is None: return None
            start = (pose['x_m'], pose['y_m'], pose['z_m'])
            self._substitute = self.planner.plan_near(self.cost_map, start, self.goal, self.policy, radius)
            self._substitute_key = self._last_plan_key
        return self._substitute

    def _absence_reason(self) -> str:
        if not self.session.sources:
            running = sorted({self._algorithm_profiles[member.process_id]
                              for member in self.pipeline.matching(PRODUCER_ROLE)
                              if member.process_id in self._algorithm_profiles})
            if running:
                return (f'dalg is running {", ".join(running)} and has not published an evidence '
                        'grid yet; it publishes once it has a session, a valid pose and a sensor')
            return f'no evidence producer has committed a manifest on {self.id}'
        if not self.producer_alive():
            return 'the evidence producer\'s heartbeat is gone from the module bus'
        return 'the evidence producer has published no grid yet'

    def _policy_record(self) -> dict[str, Any]:
        return dict(self.policy.as_dict(), digest=self.policy.digest,
                    path=None if self.policy.path is None else str(self.policy.path))

    def _record(self, route, trigger, pose, goal, grids, stale, sim_now, started_wall) -> None:
        """One planning attempt, with exactly the inputs it used."""
        self.attempts += 1
        if self.recorder is None: return
        used = grids if self.cost_map is not None else {}
        inputs = dict(time_s=sim_now, context=dict(self.session.context),
                      evidence_identity=list(self.session.identity or ()),
                      sources=sorted(used), stale=sorted(stale),
                      revisions={sid: g.revision for sid, g in used.items()},
                      record_times={sid: g.sim_time_s for sid, g in used.items()},
                      arrival_order={sid: self._arrivals.get(sid, (None, None, None))[2] for sid in used},
                      staleness_horizon_s=self.session.staleness_horizon_s,
                      producer_alive=self.producer_alive())
        control = self.control_route
        data = dict(schema=ATTEMPT_SCHEMA, attempt=self.attempts, trigger=trigger,
                    started_wall=started_wall, finished_wall=time.time(),
                    pose=pose, pose_error=self.pose_error, goal=goal,
                    authority=self.authority, policy=self._policy_record(), planner=self.planner_name,
                    route=route.as_dict(), control=None if control is None else control.as_dict(),
                    inputs=inputs)
        self.recorder.record('planning.attempt', data, used)

    def _publish(self, route: R.Route) -> None:
        self.route = route
        if self.history.observe(route):
            self.note('route.status', status=route.status, reason=route.reason,
                      planner=route.planner, map_revision=route.map_revision)
        payload = route.as_dict()
        if len(payload['waypoints']) > MAX_PUBLISHED_WAYPOINTS:
            payload['waypoints'] = payload['waypoints'][:MAX_PUBLISHED_WAYPOINTS]
            payload['waypoints_truncated'] = True
        payload['attempt'] = self.attempts
        payload['evidence_context'] = dict(self.session.context)
        goal = self.goal_descriptor
        payload['goal_revision'] = None if goal is None else goal.get('revision')
        if self.control_route is not None:
            payload['control'] = dict(
                kind='straight-line diagnostic on the same evidence-derived cost',
                cost=(None if self.control_route.cost == float('inf')
                      else round(self.control_route.cost, 6)),
                length_m=round(self.control_route.length_m, 4))
        self.bus.publish('route.planned', run_id=self.run_id, payload=payload)

    # -- presence ----------------------------------------------------------

    def summary_detour(self) -> dict[str, Any] | None:
        """How the current plan compares with the straight-line diagnostic, or None."""
        return _ratio(self.route, self.control_route)

    def source_rows(self, sim_now: float) -> Sequence[dict[str, Any]]:
        return describe_sources(self, sim_now)

    def capabilities(self) -> dict[str, Any]:
        return {'planners': sorted(PLANNERS), 'planner': self.planner_name,
                'navigation_endpoint': self.navigation.output.name,
                'policy': self.policy_name, 'policy_digest': self.policy.digest,
                'producer': self.session.manifest.get('producer', ''),
                'sources': self.sources(), 'holds_control_lease': False}

    def health(self) -> str:
        grades = [grade(self.intake.achieved_hz, self.intake.wanted_hz)]
        if self.route.status in (R.STALE_MAP, R.STALE_POSE, R.FRAME_MISMATCH): grades.append('bad')
        elif self.route.status in (R.NO_ROUTE, R.START_BLOCKED, R.GOAL_UNREACHABLE, R.OUTSIDE_COVERAGE):
            grades.append('warn')
        return worst(grades)

    def _presence(self, wall_now: float) -> None:
        state = self.route.status.upper()
        if not self._hello_sent:
            self.bus.publish('module.hello', payload={'state': state, 'capabilities': self.capabilities()})
            self._hello_sent = True
        if wall_now - self._last_heartbeat < 1.0: return
        self._last_heartbeat = wall_now
        cadence = self.session.cadence_s
        self.intake.set_wanted(1.0 / cadence if self.goal_descriptor is not None and cadence > 0 else 0.0)
        self.bus.publish('module.heartbeat', run_id=self.run_id, payload={
            'state': state, 'ready': True, 'reason': self.route.reason,
            'intake': self.intake.report(self.sim_time_s(), overruns=self.bus.overruns),
            'capabilities': self.capabilities(),
            'recording': None if self.recorder is None else self.recorder.report()})

    # -- shutdown ----------------------------------------------------------

    def _archive_displayed_background(self) -> None:
        """File the exact reference revision the operator's picture showed (§7).

        Called after the report image is rendered and before the recorder
        seals, so the archived revision is the one in the picture by
        construction, not by coincidence. Nothing here may raise into the
        shutdown path: a decoration that cannot be filed is a warning, never
        a lost report.
        """
        background = None
        if self.background_provider is not None:
            try: background = self.background_provider()
            except Exception: background = None    # noqa: BLE001 - decoration never kills the report
        self._displayed_background = background
        if background is None or self.recorder is None: return
        image = background.image
        try:
            self.recorder.record('report.background', dict(
                time_s=self.sim_time_s(), image_id=image.image_id, revision=image.revision,
                checksum=image.checksum, opacity=background.opacity,
                label=background.label(), source_category=image.source_category),
                images={'reference': image})
        except ValueError:
            self.note('report.background.unverified',
                      reason='image bytes do not match their declared checksum')

    def _reference_image_record(self) -> dict[str, Any] | None:
        background = self._displayed_background
        if background is None: return None
        image = background.image
        return dict(image_id=image.image_id, revision=image.revision, checksum=image.checksum,
                    opacity=background.opacity, label=background.label(),
                    source_category=image.source_category)

    def snapshot_image(self):
        """The route over the evidence-derived cost, as the operator showed it.

        The window's pane renders this with the reference background when the
        operator enabled one -- an assisted session, recorded in the summary's
        ``reference_display`` -- and the archived report must carry exactly
        that picture. The headless fallback, and every report written without
        the window, draws the evidence-only picture: no world.
        """
        if self.snapshot_provider is not None:
            try: return self.snapshot_provider()
            except Exception: return None
        from dcmn.map_pane import Overlay, snapshot_image
        grids = self._grids()
        if not grids or self.cost_map is None: return None
        source = sorted(grids)[0]
        vehicle = self.vehicle()
        return snapshot_image(
            grids[source], mode='cost', cost=self.cost_map.cost, never_observed=self.cost_map.never_observed,
            sim_now_s=self.sim_time_s(), horizon_s=self.session.staleness_horizon_s,
            overlay=Overlay(route=tuple(self.route.points),
                            vehicle=None if vehicle is None else (vehicle[0], vehicle[1], vehicle[3]),
                            goal=None if self.goal is None else self.goal[:2],
                            start=None if self.route.start is None else self.route.start[:2],
                            inflation_m=self.policy.inflation_m))

    def summary(self) -> dict[str, Any]:
        sim_now = self.sim_time_s()
        control = self.control_route
        return {
            'schema_version': 2, 'module': 'dnav', 'session_id': self.run_id,
            'instance': self.id, 'partial': self.partial, 'reason': self.reason,
            'time_s': round(sim_now, 4), 'pose_provider': self.snapshot.get('provider_kind', ''),
            'planner': self.planner_name, 'planners': sorted(PLANNERS),
            'plans': self.plans, 'attempts': self.attempts,
            'reference_display': self._reference_display,
            'reference_image': self._reference_image_record(),
            'policy': self._policy_record(),
            'goal': self.goal_descriptor, 'authority': self.authority,
            'route': self.route.as_dict(),
            'navigation': self.navigation.last,
            'navigation_error': self.navigation.error,
            'control_route': None if control is None else control.as_dict(),
            'control_note': 'straight-line diagnostic on the same evidence-derived cost; not an oracle',
            'detour': _ratio(self.route, control),
            'statuses': self.history.entries, 'status_counts': self.history.counts(),
            'generation_transitions': self.transitions,
            'map': dict(self.session.report(sim_now),
                        rejections=[reason for _, reason in self.session.rejections[-16:]]),
            'health': self.health(),
            'recording': None if self.recorder is None else self.recorder.report(),
            'archive': None if self.recorder is None else self.recorder.directory.name,
        }

    def _write_report(self, image=None) -> Path | None:
        if self.report_dir is None: return None
        from dnav.report import write_report
        return write_report(self.report_dir, summary=self.summary(), events=self._events,
                            image=self.snapshot_image() if image is None else image)

    def close(self, *, partial: bool = False) -> Path | None:
        if self.closed: return self.report_dir
        self.closed = True
        self.partial = self.partial or partial
        written = None
        try:
            self.connect()
            self.note('module.closed', partial=self.partial, reason=self.reason)
            # The picture first, then the background it used, then the archive
            # sealed -- the same order as a session rollover, so a report and
            # its record can never disagree about which reference revision was
            # displayed.
            image = self.snapshot_image()
            self._archive_displayed_background()
            if self.recorder is not None: self.recorder.close()
            written = self._write_report(image=image)
        except Exception as exc:                     # noqa: BLE001 - never raise out of reporting
            import sys
            print(f'dnav report: {exc}', file=sys.stderr)
        finally:
            self.bus.publish('module.goodbye', run_id=self.run_id,
                             payload={'state': self.route.status.upper(), 'reason': self.reason})
            self.bus.close()
            self.session.close()
            self.navigation.close()
        return written


def _describe(generation):
    if generation is None: return None
    identity, context = generation
    return dict(identity=list(identity), context=dict(context))


def _ratio(route: R.Route, control: R.Route | None) -> dict[str, Any] | None:
    """How the plan compares with the straight line it is quoted against."""
    if control is None or not route.ok or not control.ok: return None
    return {
        'control_cost': None if math.isinf(control.cost) else round(control.cost, 6),
        'control_length_m': round(control.length_m, 4),
        'cost_ratio': (None if math.isinf(control.cost) or control.cost <= 0
                       else round(route.cost / control.cost, 4)),
        'length_ratio': (None if control.length_m <= 0 else round(route.length_m / control.length_m, 4)),
        'control_blocked': math.isinf(control.cost),
    }


def describe_sources(run: NavRun, sim_now: float) -> Sequence[dict[str, Any]]:
    """One row per followed source, for the Map source panel and the status bar."""
    report = run.session.report(sim_now)
    rows = []
    for sid in run.sources():
        entry = report['sources'].get(sid, {})
        rows.append(dict(source=sid, revision=entry.get('revision', 0),
                         age_s=entry.get('age_s'), stale=entry.get('stale', True),
                         rate_hz=entry.get('rate_hz'),
                         sensor=entry.get('sensor', ''),
                         algorithm=entry.get('algorithm', '')))
    return rows
