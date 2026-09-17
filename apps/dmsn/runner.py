"""`MissionRun`: one mission against one vehicle, shared by the window and the headless loop.

The runner owns everything a mission body should not have to think about, in
fixed phases (:data:`PHASE_ORDER`):

``idle``
    a profile is loaded and editable; nothing is sent.
``starting``
    the arena is checked, coverage is requested while still on the ground
    (B1, Q24), the takeoff point is kept as home, and ``launch`` is asked of
    dway; the phase ends when dway reports the vehicle airborne.
``running``
    the mission body steps on each paced observation. Its goals go through the
    debounce (B7); its waits become the blocking fact; its ``Done`` ends it.
``finishing``
    the finish action: hold where it is, fly home and hold there, or land.
``done``
    an outcome and a reason, recorded and reported.

``paused`` sits beside the active phases. The operator pauses; so does losing
the goal authority or dway failing -- and dmsn never takes the authority or the
lease back (Q9). Resume asks for an explicit handoff.

dmsn holds no control lease. It moves the vehicle only by setting the session
goal as the ``mission`` authority and by publishing intents, which dway may
refuse. Every observation, event and decision is archived, so
:func:`replay` can run a recorded mission body again and compare decisions.
"""
from __future__ import annotations

import math
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from dcmn.archive import ArchiveReader, Recorder, next_archive_dir
from dcmn.context import validate_pose
from dcmn.maps import NEVER_OBSERVED, dequantize
from dcmn.mission import MAX_VERTICES, MAX_WAYPOINTS, STATUS_SCHEMA, clip
from dcmn.pacing import Paced
from dmsn import geometry as G
from dmsn import profiles as P
from dmsn.inputs import (FRONTIER_CAP, FRONTIER_EXCLUDE_M, WALL_CAP, WALL_RADIUS_M, Observation,
                         detections_view, execution_view, navigation_view)
from dmsn.intents import Action, Done, Goal, IntentWriter, Wait, to_dict
from dmsn.missions import MissionEvent, build, target_of
from dmsn.ports import IntentOutput

PHASE_ORDER = ('idle', 'starting', 'running', 'finishing', 'done')
ACTIVE = ('starting', 'running', 'finishing')
#: How often a mission body is stepped and the status plane written.
STEP_HZ = 10.0
#: The free/occupied threshold dalg paints with, reused when the runner
#: distils the evidence plane into frontier points.
EVIDENCE_FREE_THRESHOLD = 0.35
EVIDENCE_OCCUPIED_THRESHOLD = 0.65
#: Provider status (map name, battery, target) is polled this often, in wall seconds.
PROVIDER_POLL_S = 1.0
#: Events kept in memory for the summary log and the Status tab.
EVENT_MEMORY = 4096
TIMELINE_MEMORY = 512
#: Track points closer than this are not added to the flown path.
TRACK_STEP_M = 0.05
EXIT_CODES = {'complete': 0, 'failed': 1, 'aborted': 1, 'budget': 1}


class StartRefused(ValueError):
    """Start was refused for a reason an operator must fix (exit 2 headless)."""


class MissionRun:
    def __init__(self, instance: str, profile: dict[str, Any] | None = None, *, ports=None,
                 profile_path: Path | None = None, overrides=(), headless: bool = False,
                 allow_arena_mismatch: bool = False, record: bool = True, hz: float = STEP_HZ,
                 wall: Callable[[], float] = time.monotonic, planner: str = 'dnav',
                 recording_queue: int = 16 << 20, recording_disk: int = 1 << 30) -> None:
        self.id = instance
        self.session = uuid.uuid4().hex
        self.writer_id = f'dmsn:{self.session[:16]}'
        self.headless, self.allow_arena_mismatch = headless, allow_arena_mismatch
        self.record_enabled = record
        self.recording_queue, self.recording_disk = recording_queue, recording_disk
        self.wall = wall
        if ports is None:
            from dmsn.ports import LivePorts
            ports = LivePorts(instance, planner=planner, sim_time=self.sim_time_s)
        self.ports = ports
        self.ports.claim()  # a second dmsn for this vehicle fails here, before anything is sent (Q11)
        self.previous_status = getattr(ports, 'previous_status', None) or {}
        if profile is None and self.previous_status.get('configuration'):
            previous = dict(self.previous_status['configuration'], area=self.previous_status.get('area', []),
                            waypoints=[p['xy'] for p in self.previous_status.get('waypoints', [])])
            try: profile = P.validate(previous)
            except ValueError: pass
        self.document = P.ProfileDocument(profile, profile_path)
        self.overrides = list(overrides)
        self.intents = IntentWriter(IntentOutput(ports), instance, self.session)
        self._pace = Paced(hz, clock=wall)
        self.phase, self.outcome, self.reason = 'idle', '', 'profile loaded; press Start'
        self.blocking = ''
        self.mission = None
        self.active_profile: dict[str, Any] | None = None
        self.observation = Observation(writer=self.writer_id)
        self.snapshot: dict[str, Any] = {}
        self.execution: dict[str, Any] = {}
        self.navigation: dict[str, Any] = {}
        self._time_s = 0.0
        self.shutdown_requested = False
        self.partial = False
        self.closed = False
        from dcmn.module_bus import PipelineView
        from dcmn.series import EnvelopeSeries
        self.pipeline = PipelineView()
        #: Battery and flown path against sim time, for the Status tab's curves.
        self.battery_series = EnvelopeSeries(600)
        self.path_series = EnvelopeSeries(600)
        self._execution_mark = None
        # mission state
        self.started_s: float | None = None
        self.home: tuple[float, float] | None = None
        self.track: list[tuple[float, float]] = []
        self.path_m = 0.0
        self.goal_sent: dict[str, Any] | None = None
        self.goals: list[dict[str, Any]] = []
        self.suppressed = 0
        self._suppressed_xy = None
        self.handoff_needed = False
        self.paused_from: str | None = None
        self.pause_reason = ''
        self._hold_deferred = False
        self.expand_offer: dict[str, Any] | None = None
        self.coverage_requests: list[dict[str, Any]] = []
        self.final: tuple[str, str] | None = None
        self.finish_action = 'hold'
        self._finishing_action = 'hold'
        self._land_revision = None
        self.strategy: Any = None
        self._area_mask_key: tuple | None = None
        self._area_mask = None
        self._evidence_error = ''
        self._blacklist_seen = 0
        self._home_revision = None
        self.pending_events: list[MissionEvent] = []
        self.phases: list[dict[str, Any]] = []
        self.timeline: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self.acks: list[dict[str, Any]] = []
        self._ack_mark = None
        self.launch_sent = False
        self._coverage_pending = None
        self._coverage_error = ''
        self._had_control = False
        self._released_hold = False
        self.arena = dict(profile='', provider='', mismatch=False, allowed=False)
        self.provider: dict[str, str] | None = None
        self._provider_checked = -math.inf
        self.provider_target_used = False
        self.reference_display: dict[str, Any] | None = None
        self.decisions = 0
        # recording
        self.recorder: Recorder | None = None
        self.report_dir: Path | None = None
        self.run_id = ''
        self._session_id = None
        self.status_sequence = 0
        self.status_error = ''
        self._last_status = -math.inf
        self._hello_sent = False
        self._last_heartbeat = -math.inf
        self.snapshot_provider = None
        self.background_provider = None
        self._displayed_background = None
        self._enter('idle', 'profile loaded; press Start')

    # -- clocks ------------------------------------------------------------------

    def sim_time_s(self) -> float:
        return self._time_s

    def poll_delay(self) -> float:
        return .02

    # -- notes and events ----------------------------------------------------------

    def note(self, event_type: str, *, vital: bool = False, **fields: Any) -> None:
        """One event for the summary log and the archive; never raises."""
        entry = dict(time_s=round(self.sim_time_s(), 4), type=event_type, **fields)
        self.events.append(entry)
        del self.events[:-EVENT_MEMORY]
        if self.recorder is not None:
            self.recorder.record(event_type, dict(time_s=self.sim_time_s(), **fields), vital=vital)

    def archive(self, kind: str, **fields: Any) -> None:
        """Archive-only: per-step records a replay needs and a person reading events.jsonl does not."""
        if self.recorder is not None:
            self.recorder.record(kind, dict(time_s=self.sim_time_s(), **fields))

    def mark(self, kind: str, text: str) -> None:
        """A line on the Status tab's timeline."""
        self.timeline.append(dict(time_s=round(self.sim_time_s(), 3), kind=kind, text=clip(text, 200)))
        del self.timeline[:-TIMELINE_MEMORY]

    def note_display(self, mode: str, **data: Any) -> None:
        """An operator display decision (the reference background), kept as provenance."""
        self.reference_display = dict(mode=mode, **data)
        self.note('display.reference', mode=mode, **data)

    def _enter(self, phase: str, reason: str) -> None:
        now = self.sim_time_s()
        if self.phases and self.phases[-1]['end_s'] is None: self.phases[-1]['end_s'] = round(now, 4)
        previous = self.phase if self.phases else None
        self.phase, self.reason = phase, reason
        self.phases.append(dict(phase=phase, start_s=round(now, 4), end_s=None, reason=clip(reason, 200)))
        if previous is not None:
            self.note('mission.phase', vital=True, previous=previous, phase=phase, reason=reason)
            self.mark('phase', f'{phase}: {reason}')
            self.ports.bus_publish('mission.phase', run_id=self.run_id,
                                   payload=dict(phase=phase, previous=previous, reason=clip(reason)))

    # -- connection ------------------------------------------------------------------

    def _connect(self) -> None:
        self.snapshot = self.ports.read_context()
        if self.snapshot:
            self._time_s = float(self.snapshot.get('time_s', self._time_s))
            self._adopt_session()
        self.execution = self.ports.read_execution()
        self.navigation = self.ports.read_navigation()
        self.ports.poll_maps()
        wall = self.wall()
        if wall - self._provider_checked >= PROVIDER_POLL_S:
            self._provider_checked = wall
            self.provider = self.ports.provider_status()
        for event in self.ports.bus_receive():
            try: self.pipeline.observe(event, wall)
            except Exception: pass
            if (self._coverage_pending is not None and getattr(event, 'role', '') == 'algorithm'
                    and getattr(event, 'type', '') == 'module.heartbeat'
                    and event.payload.get('reset_revision') == (self.coverage_requests[-1].get('revision')
                                                               if self.coverage_requests else None)):
                reason = str(event.payload.get('reason', ''))
                if 'rejected' in reason or event.payload.get('state') in ('ALLOCATION_FAILED', 'CONFIGURATION_ERROR'):
                    self._coverage_error = reason
            if getattr(event, 'type', '') == 'system.shutdown':
                self.shutdown_requested = True
                self.reason = str(event.payload.get('reason', 'instance shutdown requested'))
                self.note('shutdown.requested', reason=self.reason)

    def _adopt_session(self) -> None:
        sid = self.snapshot.get('session_id')
        if not sid or sid == self._session_id: return
        previous = self.recorder
        if previous is not None:
            self.note('session.rollover', new_session=sid)
            previous.close()
            self._write_report()
        self._session_id = self.run_id = sid
        self.report_dir = Path(self.snapshot['report_root']) / 'dmsn'
        if not self.record_enabled: return
        try:
            self.recorder = Recorder(next_archive_dir(self.report_dir), dict(
                module='dmsn', session_id=sid, instance=self.id, dmsn_session=self.session,
                writer=self.writer_id, provider_kind=self.snapshot.get('provider_kind')),
                module='dmsn', queue_bytes=self.recording_queue, disk_bytes=self.recording_disk)
        except OSError as exc:
            self.recorder = None
            self.note('recording.unavailable', reason=str(exc))
            return
        self.recorder.record('mission.loaded', dict(time_s=self.sim_time_s(), profile=self.document.value,
                                                    digest=self.document.digest, phase=self.phase))

    # -- observation -------------------------------------------------------------------

    def _evidence_view(self, area, pose, exclude=()) -> tuple[tuple[tuple[float, float], ...], float | None,
                                                             tuple[tuple[float, float], ...]]:
        """The bounded evidence view of an area: frontiers, coverage and walls (Q28).

        Distils the evidence plane into what an observation may carry -- at
        most :data:`~dmsn.inputs.FRONTIER_CAP` frontier cells nearest the
        vehicle, the observed fraction of the area's cells, and at most
        :data:`~dmsn.inputs.WALL_CAP` occupied cells near the vehicle -- so an
        area mission decides without a grid and a recorded run replays decision
        for decision. A frontier is an observed free cell touching a
        never-observed one: the edge of what the vehicle knows. Frontiers near
        a point in ``exclude`` (the mission's blacklist) are dropped before the
        cap, so dead ends never crowd out the frontiers still worth flying to.
        """
        import numpy as np
        grids = self.ports.grids() if hasattr(self.ports, 'grids') else {}
        if not area or not grids: return (), None, ()
        grid = grids[sorted(grids)[0]]  # one evidence source: the same one the window shows
        geometry = grid.geometry
        occupancy = grid.layer(0)[0]
        never = occupancy == NEVER_OBSERVED
        probability = dequantize(occupancy)
        free = (~never) & (probability <= EVIDENCE_FREE_THRESHOLD)
        unknown = np.pad(never, 1, constant_values=False)  # never-observed, one ring out
        frontier = free & (unknown[:-2, 1:-1] | unknown[2:, 1:-1]
                           | unknown[1:-1, :-2] | unknown[1:-1, 2:])
        key = (geometry.origin_x_m, geometry.origin_y_m, geometry.cell_m, geometry.width,
               geometry.height, tuple((round(float(x), 6), round(float(y), 6)) for x, y in area))
        if self._area_mask_key != key:  # the area is fixed during a mission; the grid is not
            self._area_mask_key = key
            self._area_mask = np.asarray(G.cell_mask(list(area), geometry), bool)
        mask = self._area_mask
        here = None if pose is None else np.asarray(pose[:2], float)

        def centres(rows, cols):
            return np.stack((geometry.origin_x_m + (cols + .5) * geometry.cell_m,
                             geometry.origin_y_m + (rows + .5) * geometry.cell_m), axis=1)

        walls = ()
        if here is not None:
            rows, cols = np.nonzero((~never) & (probability >= EVIDENCE_OCCUPIED_THRESHOLD))
            if len(rows):
                points = centres(rows, cols)
                near = np.sum((points - here) ** 2, axis=1)
                keep = near <= WALL_RADIUS_M ** 2
                points, near = points[keep], near[keep]
                points = points[np.argsort(near, kind='stable')][:WALL_CAP]
                walls = tuple((float(x), float(y)) for x, y in points)
        if not mask.any(): return (), None, walls
        coverage = float((mask & ~never).sum()) / float(mask.sum())
        rows, cols = np.nonzero(frontier & mask)
        if not len(rows): return (), coverage, walls
        points = centres(rows, cols)
        for bx, by in exclude or ():
            points = points[np.sum((points - (float(bx), float(by))) ** 2, axis=1) > FRONTIER_EXCLUDE_M ** 2]
        if here is not None and len(points):
            near = np.sum((points - here) ** 2, axis=1)
            points = points[np.argsort(near, kind='stable')]
        return tuple((float(x), float(y)) for x, y in points[:FRONTIER_CAP]), coverage, walls

    def _observe(self) -> Observation:
        snapshot = self.snapshot
        pose, pose_error = None, ''
        if not snapshot:
            pose_error = 'session provider unavailable'
        else:
            try:
                p = validate_pose(snapshot.get('pose'), snapshot)
                pose = (p['x_m'], p['y_m'], p['z_m'], p['heading_deg'])
            except (ValueError, KeyError, TypeError) as exc:
                pose_error = str(exc)
        status = self.provider or {}
        battery = None
        try:
            if status.get('drone.battery_pct') not in (None, ''): battery = float(status['drone.battery_pct'])
        except ValueError:
            battery = None
        arena = Path(status['sim.map']).stem if status.get('sim.map') else ''
        target, target_reason = None, ''
        if status:
            from dcmn.provider_target import map_target
            position, target_reason = map_target(status, snapshot)
            target = None if position is None else (position[0], position[1])
        else:
            target_reason = 'provider status plane unavailable'
        goal = snapshot.get('goal') if snapshot else None
        detections, estimates, detections_time, detections_reason = (), (), None, ''
        target = target_of(self.params) if self.mission is not None else None
        if target is not None and self.phase in ACTIVE + ('paused',):
            detections, estimates, detections_time, detections_reason = detections_view(
                self.ports.read_detections(), target)
        frontiers, area_coverage, walls = (), None, ()
        if (self.mission is not None and self.phase in ACTIVE
                and 'area' in type(self.mission).editors):
            try:
                frontiers, area_coverage, walls = self._evidence_view(
                    (self.active_profile or self.document.value)['area'], pose,
                    exclude=tuple(getattr(self.mission, 'blacklist', ()) or ()))
                self._evidence_error = ''
            except Exception as exc:  # noqa: BLE001 - a bad grid must not stop the loop
                if str(exc) != self._evidence_error:
                    self._evidence_error = str(exc)
                    self.note('mission.evidence_error', error=str(exc))
        return Observation(
            time_s=self.sim_time_s(), session_id=str(snapshot.get('session_id', '')), pose=pose,
            pose_error=pose_error,
            goal=None if goal is None else dict(position=list(goal.get('position') or []),
                                                revision=goal.get('revision'), authority=goal.get('authority'),
                                                role=goal.get('role')),
            authority=snapshot.get('authority') if snapshot else None, writer=self.writer_id,
            execution=execution_view(self.execution), navigation=navigation_view(self.navigation),
            coverage=self.ports.coverage(), battery_pct=battery, arena=arena,
            provider_target=target, provider_target_reason=target_reason,
            frontiers=frontiers, area_coverage=area_coverage, walls=walls, detections=detections,
            estimates=estimates, detections_time_s=detections_time, detections_reason=detections_reason)

    # -- operator commands (the window and the CLI call the same methods) ------------------

    def click(self, x_m: float, y_m: float) -> str:
        """A map click: sets the profile's goal while idle, retargets while active."""
        xy = (float(x_m), float(y_m))
        if self.phase in ('idle', 'done'):
            schema = self.mission_schema()
            if 'goal' not in schema: return 'this mission takes no goal'
            self.document.set('params.goal', list(xy))
            return f'goal set to {xy[0]:.2f}, {xy[1]:.2f} m' if not self.document.error else self.document.error
        if self.mission is None or 'goal' not in type(self.mission).editors: return 'this mission takes no goal'
        self.pending_events.append(MissionEvent('retarget' if self.phase != 'starting' else 'click', xy))
        self.note('mission.operator_goal', xy=list(xy), phase=self.phase)
        return f'retarget to {xy[0]:.2f}, {xy[1]:.2f} m'

    def mission_schema(self):
        from dmsn.missions import schema_for
        return schema_for(self.document.value['mission'])

    def check_arena(self) -> str:
        """``''``, or the mismatch between the profile's arena and the running map (Q4)."""
        expected = self.document.value.get('arena') or ''
        provider = Path((self.provider or {}).get('sim.map', '')).stem
        self.arena.update(profile=expected, provider=provider)
        if not expected or not provider or expected == provider:
            self.arena['mismatch'] = False
            return ''
        self.arena['mismatch'] = True
        return f'profile arena {expected!r} differs from the running map {provider!r}'

    def start(self) -> str:
        """Begin the loaded profile. Raises `StartRefused` for a fixable refusal."""
        if self.phase in ACTIVE or self.phase == 'paused':
            raise StartRefused(f'a mission is already {self.phase}')
        try:
            profile = P.validate(self.document.value)
            mission = build(profile['mission'])
            params = mission.schema.validate(profile['params'])
        except ValueError as exc:
            raise StartRefused(str(exc)) from None
        if self.provider is None:
            self.provider = self.ports.provider_status()
            self._provider_checked = self.wall()
        self._check_target(params)
        mismatch = self.check_arena()
        if mismatch:
            self.arena['allowed'] = self.allow_arena_mismatch or not self.headless
            self.note('mission.arena_mismatch', vital=True, **self.arena)
            if not self.arena['allowed']:
                raise StartRefused(f'{mismatch}; pass --allow-arena-mismatch to fly it anyway')
        self._reset_mission_state()
        # Start is the operator's explicit request for the goal (§5.1): the
        # first goal takes the authority from whoever holds it, and says so.
        self.handoff_needed = True
        self.active_profile, self.mission, self.params = profile, mission, params
        self.strategy = None
        if profile['strategy']:
            from dmsn.strategies import build as build_strategy
            self.strategy = build_strategy(profile['strategy'], profile['strategy_params'])
        self.finish_action = profile['finish_action']
        obs = self._observe()
        self.observation = obs
        self.started_s = obs.time_s
        self.home = obs.xy
        self.note('mission.start', vital=True, profile=profile, digest=P.digest(profile), params=params,
                  overrides=self.overrides, obs=obs.as_dict())
        self.provider_target_used = params.get('goal_source') == 'provider_target'
        self._request_coverage(profile, obs)
        self._enter('starting', 'waiting for dway to report the vehicle airborne')
        self.ports.bus_publish('mission.started', run_id=self.run_id,
                               payload=dict(mission=profile['mission'], profile=profile['name']))
        return 'started'

    def _check_target(self, params) -> None:
        """A mission with a target starts only when some detector publishes its label (§13.6)."""
        target = target_of(params)
        if target is None: return
        label, detector = target['label'], target['detector']
        plane = getattr(self.ports, 'detections_plane', 'the detections plane')
        value = self.ports.read_detections()
        detectors = (value or {}).get('detectors', [])
        found = [d for d in detectors if label in d.get('labels', ()) and (not detector or d.get('id') == detector)]
        self.note('mission.target_check', vital=True, target=target, plane=plane,
                  detectors=[dict(id=d.get('id'), labels=d.get('labels'), state=d.get('state')) for d in detectors],
                  found=[d.get('id') for d in found])
        if not value:
            raise StartRefused(f'no detector publishing `{label}`: no detections plane at {plane} '
                               '(run dalg with a detector profile, e.g. --profiles lidar-baseline red-disc)')
        if not found:
            which = f' as detector {detector}' if detector else ''
            published = ', '.join(sorted({x for d in detectors for x in d.get('labels', ())})) or 'no labels'
            raise StartRefused(f'no detector publishing `{label}`{which} (the plane publishes {published})')

    def _reset_mission_state(self) -> None:
        self.outcome, self.final = '', None
        self.goal_sent, self.goals, self.suppressed, self._suppressed_xy = None, [], 0, None
        self.track, self.path_m, self.pending_events = [], 0.0, []
        self.handoff_needed, self.paused_from, self.pause_reason = False, None, ''
        self.expand_offer, self.coverage_requests, self._home_revision = None, [], None
        self.launch_sent, self._hold_deferred, self.decisions = False, False, 0
        self._released_hold = False
        self._coverage_pending, self._coverage_error, self._had_control = None, '', False
        self._finishing_action, self._land_revision = 'hold', None
        self._area_mask_key, self._area_mask, self._evidence_error = None, None, ''
        self._blacklist_seen = 0

    def _request_coverage(self, profile, obs) -> None:
        """Map the mission's area or workspace before takeoff (B1); nothing to ask for otherwise (Q24)."""
        area = profile.get('area') or []
        bounds = G.bbox(area) if area else profile.get('workspace')
        if bounds is None: return
        wanted = G.expand(bounds, profile['coverage_margin_m'])
        if obs.coverage is not None and G.contains(obs.coverage, wanted[:2]) and G.contains(obs.coverage, wanted[2:]):
            self.note('mission.coverage', requested=False, reason='already covered', bounds=list(wanted))
            return
        if obs.airborne:
            self.blocking = ('area outside evidence coverage; a mapping reset is only requested on the ground '
                             'before growable grids exist')
            self.note('mission.coverage', requested=False, reason='airborne', bounds=list(wanted))
            self._coverage_pending = wanted
            self._coverage_error = self.blocking
            return
        self._reset_coverage(wanted, cause='mission area' if area else 'profile workspace')

    def _reset_coverage(self, bounds, *, cause: str) -> None:
        self._coverage_pending, self._coverage_error = tuple(bounds), ''
        try:
            revision = self.ports.request_reset(list(bounds), by='dmsn')
            entry = dict(bounds=[round(v, 3) for v in bounds], cause=cause, revision=revision)
        except (ValueError, RuntimeError) as exc:
            entry = dict(bounds=[round(v, 3) for v in bounds], cause=cause, error=str(exc))
            self._coverage_error = str(exc)
        self.coverage_requests.append(entry)
        self.note('mission.coverage', vital=True, requested='error' not in entry, **entry)
        self.mark('coverage', f'coverage requested ({cause})')

    def expand_coverage(self) -> str:
        """The operator's explicit choice to reset the map so an outside goal is covered."""
        offer = self.expand_offer
        if offer is None: return 'no goal is waiting for coverage'
        self._reset_coverage(offer['bounds'], cause='operator expanded coverage (map reset)')
        self.expand_offer = None
        return 'coverage requested; the map is reset'

    def pause(self, reason: str = 'paused by operator', *, hold: bool = True) -> str:
        """Stop stepping the mission body; with ``hold``, also ask dway to hold.

        An operator's Pause holds the vehicle. Losing the goal authority or
        dway failing does not: whoever took the goal is now flying it, and a
        coordinator that held the vehicle underneath them would be taking
        control back by another name.
        """
        if self.phase not in ACTIVE: return f'nothing to pause ({self.phase})'
        self.paused_from, self.pause_reason = self.phase, reason
        self._enter('paused', reason)
        if not hold:
            pass
        elif self.execution.get('state') == 'TAKING_OFF':
            self._hold_deferred = True  # a hold mid-climb would strand the vehicle below its slab
        else:
            self._intent('hold', reason)
        self.pending_events.append(MissionEvent('paused', data=dict(reason=reason)))
        return 'paused'

    def resume(self) -> str:
        if self.phase != 'paused': return f'nothing to resume ({self.phase})'
        self.handoff_needed = True
        self._enter(self.paused_from or 'running', 'resumed by operator')
        self._hold_deferred = False
        self.pending_events.append(MissionEvent('resumed'))
        if self.goal_sent is not None:
            self._send_goal(Goal(tuple(self.goal_sent['xy']), self.goal_sent['tag'], True), self.observation,
                            force=True)
        if self.execution.get('mission_hold'): self._intent('resume', 'resumed by operator')
        return 'resumed'

    def abort(self, reason: str = 'aborted by operator') -> str:
        if self.phase not in ACTIVE and self.phase != 'paused': return f'nothing to abort ({self.phase})'
        self._intent('hold', reason)
        self.ports.release_authority(self.writer_id)
        self.final = ('aborted', reason)
        self._done()
        return 'aborted'

    def go_home(self) -> str:
        if self.phase not in ('running', 'paused', 'starting'): return f'cannot go home while {self.phase}'
        if self.home is None: return 'no home recorded (no valid pose at Start)'
        if self.phase == 'paused': self._enter(self.paused_from or 'running', 'going home')
        self._finish('aborted', 'operator sent the vehicle home', action='home')
        return 'going home'

    def land(self) -> str:
        """The Land button (B13): land the vehicle where it is and end the mission."""
        if self.phase not in ACTIVE and self.phase != 'paused': return f'nothing to land ({self.phase})'
        if not self.observation.airborne: return 'the vehicle is not airborne'
        if self.phase == 'paused': self._enter(self.paused_from or 'running', 'landing')
        self._finish('aborted', 'operator landed the vehicle', action='land')
        return 'landing'

    # -- the loop ---------------------------------------------------------------------------

    def step(self) -> str:
        if self.closed: return self.phase
        self._connect()
        if not self._pace.due():
            return self.phase
        obs = self._observe()
        self.observation = obs
        self._track(obs)
        if obs.battery_pct is not None: self.battery_series.add(obs.time_s, obs.battery_pct)
        if self.phase in ACTIVE: self.path_series.add(obs.time_s, self.path_m)
        self._watch_acks(obs)
        self._watch_execution(obs)
        if self.phase in ACTIVE or self.phase == 'paused': self._supervise(obs)
        if self.phase == 'starting': self._step_starting(obs)
        elif self.phase == 'running': self._step_running(obs)
        elif self.phase == 'finishing': self._step_finishing(obs)
        elif self.phase == 'paused': self._step_paused(obs)
        self._publish_status(obs)
        self._presence()
        return self.phase

    def _track(self, obs: Observation) -> None:
        if obs.xy is None or self.phase not in ACTIVE: return
        if self.track and math.dist(self.track[-1], obs.xy) < TRACK_STEP_M: return
        if self.track: self.path_m += math.dist(self.track[-1], obs.xy)
        self.track.append(obs.xy)
        if len(self.track) > 20000: del self.track[:len(self.track) - 20000]

    def _watch_acks(self, obs: Observation) -> None:
        ack = obs.execution.get('mission_intent')
        if not ack or ack.get('session') != self.session: return
        mark = (ack.get('revision'), ack.get('state'))
        if mark == self._ack_mark: return
        self._ack_mark = mark
        self.acks.append(dict(ack, time_s=round(obs.time_s, 4)))
        del self.acks[:-256]
        self.note('mission.ack', ack=ack)
        if ack.get('state') in ('refused', 'failed'):
            self.mark('intent', f'dway {ack["state"]} {ack.get("kind")}: {ack.get("reason")}')

    def _watch_execution(self, obs: Observation) -> None:
        """Executor state changes and stops, for the timeline."""
        ex = obs.execution
        mark = (ex.get('state'), ex.get('hold_kind'))
        if mark == self._execution_mark: return
        self._execution_mark = mark
        if not ex: return
        self.note('mission.execution', state=ex.get('state'), hold_kind=ex.get('hold_kind'), reason=ex.get('reason'))
        kind = 'stop' if ex.get('state') in ('BRAKING', 'HOLDING') else 'dway'
        self.mark(kind, f'dway {ex.get("state")}{"/" + ex["hold_kind"] if ex.get("hold_kind") else ""}: {ex.get("reason")}')

    def _intent(self, kind: str, reason: str, **args) -> bool:
        if not self.execution:
            self.note('mission.intent_unsent', kind=kind, reason='no dway execution status')
        try:
            value = self.intents.send(kind, self.sim_time_s(), reason, **args)
        except (ValueError, RuntimeError) as exc:
            self.note('mission.intent_failed', vital=True, kind=kind, reason=str(exc))
            return False
        self.note('mission.intent', vital=True, intent=value)
        self.mark('intent', f'{kind}: {reason}')
        return True

    def _supervise(self, obs: Observation) -> None:
        ex = obs.execution
        state = ex.get('state')
        if ex.get('owns_control'):
            self._had_control = True
        elif (self._had_control and self.phase != 'paused' and ex.get('owns_control') is False
              and not (self.phase == 'finishing' and self._finishing_action == 'land')):
            # A landing ends with dway giving the lease back on the ground; its
            # ack, not the lease, is the verdict there.
            self.pause(f'paused: control lease lost: {ex.get("vehicle_fault") or ex.get("reason")}', hold=False)
            return
        # Authority: another writer holds the goal we set. Pause, say so, never take it back.
        if (self.phase != 'paused' and self.goal_sent is not None and obs.authority
                and obs.authority.get('id') != self.writer_id):
            holder = obs.authority
            self.handoff_needed = True
            self.pending_events.append(MissionEvent('authority_lost', data=dict(holder=holder)))
            self.note('mission.authority_lost', vital=True, holder=holder)
            self.pause(f'paused: goal authority taken by {holder.get("role")} {str(holder.get("id"))[:18]}',
                       hold=False)
            return
        if self.phase != 'paused' and state in ('FAILED', 'CANCELLED'):
            if self.phase == 'starting':
                self.final = ('failed', f'dway {state.lower()} before the vehicle was airborne: {ex.get("reason")}')
                self._done()
            elif self.phase == 'finishing' and self._finishing_action == 'land':
                pass  # the landing's verdict is dway's ack, watched in _step_finishing
            else:
                self.pause(f'paused: dway {state.lower()}: {ex.get("reason")}', hold=False)
            return
        if self.phase in ('starting', 'running'): self._check_budgets(obs)

    def _check_budgets(self, obs: Observation) -> None:
        budgets = self.active_profile['budgets']
        elapsed = 0.0 if self.started_s is None else obs.time_s - self.started_s
        for limit, used, action, what in (
                (budgets['time_s'], elapsed, budgets['time_action'], 'time'),
                (budgets['path_m'], self.path_m, budgets['path_action'], 'path length')):
            if limit is not None and used >= limit:
                return self._finish('budget', f'{what} budget spent: {used:.1f} of {limit:g}', action=action)
        reserve = budgets['battery_reserve_pct']
        if reserve is not None and obs.battery_pct is not None and obs.battery_pct < reserve:
            return self._finish('budget', f'battery {obs.battery_pct:.1f}% below the {reserve:g}% reserve',
                                action=budgets['battery_action'])

    def _step_starting(self, obs: Observation) -> None:
        ex = obs.execution
        if self.home is None and obs.xy is not None: self.home = obs.xy
        wanted = self._coverage_pending
        if wanted is not None:
            if self._coverage_error:
                self.blocking = f'mapping coverage refused: {self._coverage_error}'
                return
            if (obs.coverage is None or not G.contains(obs.coverage, wanted[:2])
                    or not G.contains(obs.coverage, wanted[2:])):
                self.blocking = 'waiting for dalg to publish the requested mission coverage'
                return
            self._coverage_pending = None
        if not ex:
            self.blocking = f'waiting: no dway execution status on /dvision2.{self.id}.execution'
            return
        if ex.get('mission_hold') and not self._released_hold:
            self._intent('resume', 'a new mission starts')
            self._released_hold = True
        if not obs.airborne:
            if ex.get('wait_for_mission'):
                if not self.launch_sent:
                    self.launch_sent = self._intent('launch', 'mission start')
                self.blocking = f'launching: {ex.get("reason")}'
            else:
                self.blocking = f'waiting: dway launches on its own (no --wait-for-mission): {ex.get("reason")}'
            return
        area = [tuple(p) for p in self.active_profile['area']]
        waypoints = [tuple(p) for p in self.active_profile['waypoints']]
        # The mission sees exactly what the archive keeps (rounded), so a replay
        # meets the same threshold comparisons the live run did.
        self.mission.start(self.params, Observation.from_dict(obs.as_dict()), area=area, waypoints=waypoints,
                           strategy=self.strategy)
        self.note('mission.body_start', vital=True, mission=self.active_profile['mission'], params=self.params,
                  area=[list(p) for p in area], waypoints=[list(p) for p in waypoints],
                  strategy=None if self.strategy is None else self.strategy.name,
                  strategy_params=dict(getattr(self.strategy, 'params', None) or {}),
                  obs=obs.as_dict())
        self._enter('running', 'the vehicle is airborne; the mission body is stepping')
        self.blocking = ''
        self._step_running(obs)

    def _step_running(self, obs: Observation) -> None:
        for event in self.pending_events:
            self.mission.on_event(event)
            self.note('mission.event', event=event.as_dict())
        self.pending_events = []
        recorded = obs.as_dict()
        decision = self.mission.step(Observation.from_dict(recorded))  # the archived form: replay sees the same
        self.decisions += 1
        self.archive('mission.step', obs=recorded, decision=to_dict(decision))
        self._note_blacklist()
        if isinstance(decision, Goal):
            self.blocking = self._send_goal(decision, obs) or self._route_fact(obs)
        elif isinstance(decision, Action):
            self._intent(decision.kind, 'requested by the mission', **decision.args)
        elif isinstance(decision, Wait):
            self.blocking = f'waiting: {decision.reason}'
        elif isinstance(decision, Done):
            self._finish(decision.outcome, decision.reason, action=decision.finish or None)

    def _note_blacklist(self) -> None:
        """Archive each point an area mission gives up on, with its reason, as it happens."""
        points = getattr(self.mission, 'blacklist', None)
        if not points or len(points) <= self._blacklist_seen: return
        reasons = getattr(self.mission, 'blacklist_reasons', None) or []
        for index in range(self._blacklist_seen, len(points)):
            self.note('mission.blacklist', point=list(points[index]),
                      reason=reasons[index] if index < len(reasons) else '', count=index + 1)
            self.mark('blacklist', f'gave up on {points[index][0]:.1f}, {points[index][1]:.1f} m'
                                   + (f': {reasons[index]}' if index < len(reasons) else ''))
        self._blacklist_seen = len(points)

    def _step_paused(self, obs: Observation) -> None:
        if self._hold_deferred and obs.execution.get('state') != 'TAKING_OFF':
            self._hold_deferred = False
            self._intent('hold', self.pause_reason)
        self.blocking = self.pause_reason

    def _route_fact(self, obs: Observation) -> str:
        """What stands between the goal and the vehicle moving, in one line."""
        ex, nav = obs.execution, obs.navigation
        if not ex: return 'waiting: dway is not publishing execution status'
        if ex.get('mission_hold'): return 'held: dway is holding for this mission'
        if not nav: return 'waiting: dnav is not publishing a route'
        if nav.get('goal_revision') != (self.goal_sent or {}).get('revision'):
            return 'waiting: dnav has not planned for the new goal yet'
        status = nav.get('planning_status')
        if status not in (None, 'ok') and nav.get('route_mode', 'planned') == 'planned':
            return f'waiting: dnav reports {status}: {nav.get("reason")}'
        if ex.get('state') in ('WAITING', 'READY', 'HOLDING', 'BRAKING'):
            return f'dway {ex["state"].lower()}{"/" + ex["hold_kind"] if ex.get("hold_kind") else ""}: {ex.get("reason")}'
        return ''

    def _send_goal(self, goal: Goal, obs: Observation, *, force: bool = False) -> str:
        """Set the session goal, through the coverage check and the debounce. Returns a blocking fact or ''."""
        sent = self.goal_sent
        if (not force and sent is not None and math.dist(sent['xy'], goal.xy) < 1e-6
                and obs.goal_at(goal.xy)):
            return ''
        if obs.coverage is not None and not G.contains(obs.coverage, goal.xy):
            bounds = G.expand((min(obs.coverage[0], goal.xy[0]), min(obs.coverage[1], goal.xy[1]),
                               max(obs.coverage[2], goal.xy[0]), max(obs.coverage[3], goal.xy[1])),
                              self.active_profile['coverage_margin_m'] if self.active_profile else 10.)
            offer = dict(xy=list(goal.xy), bounds=list(bounds), operator=goal.operator)
            if self.expand_offer != offer:
                self.expand_offer = offer
                self.note('mission.goal_outside_coverage', vital=True, goal=list(goal.xy),
                          coverage=list(obs.coverage), operator=goal.operator)
            x0, y0, x1, y1 = obs.coverage
            return (f'goal {goal.xy[0]:.1f}, {goal.xy[1]:.1f} m is outside evidence coverage '
                    f'[{x0:g}, {y0:g}]-[{x1:g}, {y1:g}] m; Expand coverage resets the map')
        self.expand_offer = None
        if not force and not goal.operator and sent is not None:
            debounce = self.active_profile['debounce']
            elapsed = obs.time_s - sent['time_s']
            shift = math.dist(sent['xy'], goal.xy)
            if elapsed < debounce['min_commit_s'] or shift < debounce['min_shift_m']:
                if self._suppressed_xy != goal.xy:
                    self._suppressed_xy = goal.xy
                    self.suppressed += 1
                    self.note('mission.goal_suppressed', goal=list(goal.xy), elapsed_s=round(elapsed, 3),
                              shift_m=round(shift, 3), count=self.suppressed)
                return ''
        holder = obs.authority if obs.authority and obs.authority.get('id') != self.writer_id else None
        try:
            descriptor = self.ports.set_goal(self.writer_id, list(goal.xy), handoff=self.handoff_needed)
        except (ValueError, RuntimeError) as exc:
            self.note('mission.goal_rejected', vital=True, goal=list(goal.xy), reason=str(exc))
            if 'authority' in str(exc):
                self.handoff_needed = True
                self.pause(f'paused: {exc}', hold=False)
            return f'goal not set: {exc}'
        if self.handoff_needed and holder is not None:
            self.note('mission.handoff', vital=True, previous=holder)
            self.mark('authority', f'goal authority taken from {holder.get("role")} {str(holder.get("id"))[:18]}')
        self.handoff_needed = False
        if sent is not None and self.goals and self.goals[-1]['outcome'] == 'active':
            self.goals[-1]['outcome'] = 'replaced'
        revision = None if descriptor is None else descriptor.get('revision')
        self.goal_sent = dict(xy=goal.xy, tag=goal.tag, revision=revision, time_s=obs.time_s, operator=goal.operator)
        self.goals.append(dict(xy=[round(goal.xy[0], 4), round(goal.xy[1], 4)], tag=goal.tag, revision=revision,
                               set_s=round(obs.time_s, 4), operator=goal.operator, outcome='active'))
        del self.goals[:-MAX_WAYPOINTS]
        self._suppressed_xy = None
        self.note('mission.goal', vital=True, goal=list(goal.xy), tag=goal.tag, revision=revision,
                  operator=goal.operator)
        self.mark('goal', f'{goal.tag or "goal"} {goal.xy[0]:.1f}, {goal.xy[1]:.1f} m')
        self.ports.bus_publish('mission.goal', run_id=self.run_id,
                               payload=dict(xy=list(goal.xy), tag=goal.tag, revision=revision))
        if obs.execution.get('mission_hold') and self.phase in ('running', 'finishing'):
            self._intent('resume', 'new goal')
        return ''

    def _finish(self, outcome: str, reason: str, *, action: str | None = None) -> None:
        action = action or self.finish_action
        if outcome == 'budget' and self.mission is not None and hasattr(self.mission, 'budget_spent'):
            # A mission that owes a specific ending (search: no confirmed target) names it.
            outcome, reason = self.mission.budget_spent(reason)
        self.final = (outcome, reason)
        self.note('mission.finish', vital=True, outcome=outcome, reason=reason, action=action)
        if self.goals and self.goals[-1]['outcome'] == 'active':
            self.goals[-1]['outcome'] = 'reached' if outcome == 'complete' else 'abandoned'
        if action == 'home' and self.home is not None and self.observation.airborne:
            self._finishing_action = 'home'
            self._enter('finishing', f'{reason}; returning home')
            self._home_revision = None
            self._send_goal(Goal(self.home, 'home', True), self.observation, force=True)
            self._home_revision = (self.goal_sent or {}).get('revision')
            return
        if action == 'land':
            if self.observation.airborne:
                self._finishing_action = 'land'
                self._enter('finishing', f'{reason}; landing')
                self._land_revision = None
                self._send_land(reason)
                return
            self.note('mission.land_skipped', reason='the vehicle is not airborne')
        if (self.execution.get('state') in ('EXECUTING', 'BRAKING', 'TAKING_OFF')
                or (self.execution.get('wait_for_mission') and self.execution.get('state') != 'COMPLETE')):
            self._intent('hold', reason)
        self._done()

    def _send_land(self, reason: str) -> bool:
        """Ask dway to land; keep the intent's revision so its ack can be watched."""
        if self._intent('land', reason):
            self._land_revision = (self.intents.last or {}).get('revision')
            return True
        return False

    def _step_finishing(self, obs: Observation) -> None:
        if self._finishing_action == 'land':
            return self._step_finishing_land(obs)
        if obs.arrived(self.home):
            if self.goals and self.goals[-1]['outcome'] == 'active': self.goals[-1]['outcome'] = 'reached'
            return self._done()
        if not obs.goal_at(self.home):
            problem = self._send_goal(Goal(self.home, 'home', True), obs)
            if problem:
                self.blocking = problem
                return
        self.blocking = 'returning home' + (f'; {self._route_fact(obs)}' if self._route_fact(obs) else '')

    def _step_finishing_land(self, obs: Observation) -> None:
        """Watch dway land the vehicle: its ack for the land intent is the verdict."""
        if self._land_revision is None and self._send_land(self.final[1] if self.final else 'landing'):
            return
        ack = obs.execution.get('mission_intent')
        if (self._land_revision is not None and ack and ack.get('session') == self.session
                and ack.get('revision') == self._land_revision and ack.get('kind') == 'land'):
            state = ack.get('state')
            if state == 'completed':
                self.ports.release_authority(self.writer_id)  # on the ground: no goal to hold
                return self._done()
            if state in ('refused', 'failed'):
                self.final = ('failed', f'landing failed: {ack.get("reason")}')
                return self._done()
        ex = obs.execution
        if self._land_revision is not None and ex.get('state') == 'FAILED':
            self.final = ('failed', f'landing failed: dway failed: {ex.get("reason")}')
            return self._done()
        self.blocking = 'landing' + (f'; dway {ex.get("state", "").lower()}: {ex.get("reason")}'
                                    if ex.get('state') else '')

    def _done(self) -> None:
        outcome, reason = self.final or ('failed', 'ended without a reason')
        self.outcome = outcome
        self.blocking = ''
        self._enter('done', reason)
        self.note('mission.done', vital=True, outcome=outcome, reason=reason,
                  summary=None if self.mission is None else self.mission.summary())
        self.ports.bus_publish('mission.done', run_id=self.run_id, payload=dict(outcome=outcome, reason=clip(reason)))

    # -- publication ------------------------------------------------------------------------

    def status_value(self, obs: Observation) -> dict[str, Any]:
        profile = self.active_profile or self.document.value
        budgets = profile['budgets']
        elapsed = 0.0 if self.started_s is None else max(0.0, obs.time_s - self.started_s)
        progress = dict(elapsed_s=round(elapsed, 3), path_m=round(self.path_m, 3), goals=len(self.goals),
                        suppressed_goals=self.suppressed, decisions=self.decisions)
        if self.mission is not None: progress.update(self.mission.progress())
        states = progress.get('waypoint_states') or []
        self.status_sequence += 1
        return dict(
            schema=STATUS_SCHEMA, vehicle_id=self.id, session=self.session, sequence=self.status_sequence,
            time_s=float(obs.time_s), mission=profile['mission'], profile=clip(profile['name']),
            profile_digest=P.digest(profile)[:64], phase=self.phase, outcome=self.outcome,
            configuration={k: v for k, v in profile.items() if k not in ('area', 'waypoints')},
            reason=clip(self.reason), blocking=clip(self.blocking), writer=self.writer_id,
            goal=None if self.goal_sent is None else dict(xy=list(self.goal_sent['xy']), tag=clip(self.goal_sent['tag']),
                                                           revision=self.goal_sent['revision']),
            area=[list(p) for p in profile['area'][:MAX_VERTICES]],
            waypoints=[dict(xy=list(p), state=states[i] if i < len(states) else 'pending')
                       for i, p in enumerate(profile['waypoints'][:MAX_WAYPOINTS])],
            home=None if self.home is None else list(self.home), progress=progress,
            budgets=dict(budgets, elapsed_s=round(elapsed, 3), path_m=round(self.path_m, 3),
                         battery_pct=obs.battery_pct),
            expand_offer=self.expand_offer is not None)

    def _publish_status(self, obs: Observation) -> None:
        try:
            self.ports.write_status(self.status_value(obs))
            self.status_error = ''
        except (ValueError, RuntimeError) as exc:
            self.status_error = f'mission status not published: {exc}'

    def capabilities(self) -> dict[str, Any]:
        from dmsn.missions import MISSIONS
        return dict(missions=sorted(MISSIONS), mission_endpoint=f'/dvision2.{self.id}.mission',
                    intent_endpoint=f'/dvision2.{self.id}.mission_intent', holds_control_lease=False)

    def _presence(self) -> None:
        wall = self.wall()
        if not self._hello_sent:
            self.ports.bus_publish('module.hello', payload=dict(state=self.phase.upper(),
                                                                capabilities=self.capabilities()))
            self._hello_sent = True
        if wall - self._last_heartbeat < 1.0: return
        self._last_heartbeat = wall
        self.ports.bus_publish('module.heartbeat', run_id=self.run_id, payload=dict(
            state=self.phase.upper(), ready=True, reason=clip(self.blocking or self.reason),
            capabilities=self.capabilities(),
            recording=None if self.recorder is None else self.recorder.report()))

    # -- reports --------------------------------------------------------------------------------

    def exit_code(self) -> int:
        if self.phase != 'done': return 1
        return EXIT_CODES.get(self.outcome, 1)

    def summary(self) -> dict[str, Any]:
        profile = self.active_profile or self.document.value
        now = self.sim_time_s()
        phases = [dict(p, duration_s=round(max(0.0, (p['end_s'] if p['end_s'] is not None else now) - p['start_s']), 4))
                  for p in self.phases]
        return {
            'schema': 'dvision2.dmsn-report.v1', 'module': 'dmsn', 'session_id': self.run_id,
            'dmsn_session': self.session, 'instance': self.id, 'writer': self.writer_id,
            'partial': self.partial, 'time_s': round(now, 4),
            'mission': profile['mission'], 'profile': profile, 'profile_digest': P.digest(profile),
            'profile_path': None if self.document.path is None else str(self.document.path),
            'overrides': self.overrides,
            'phase': self.phase, 'outcome': self.outcome, 'reason': self.reason, 'blocking': self.blocking,
            'phases': phases, 'goals': self.goals, 'suppressed_goals': self.suppressed,
            'home': None if self.home is None else list(self.home),
            'track_m': round(self.path_m, 3), 'track_points': len(self.track),
            'elapsed_s': None if self.started_s is None else round(now - self.started_s, 4),
            'budgets': profile['budgets'], 'finish_action': self.finish_action,
            'coverage_requests': self.coverage_requests,
            'intents': None if self.intents.last is None else self.intents.revision,
            'last_intent': self.intents.last, 'acks': self.acks[-32:],
            'arena': self.arena,
            'assisted': dict(reference_display=self.reference_display,
                             provider_target_used=self.provider_target_used,
                             tour_import=bool(profile['assisted'].get('tour_import'))),
            'reference_image': self._reference_image_record(),
            'mission_summary': None if self.mission is None else self.mission.summary(),
            'decisions': self.decisions, 'timeline': self.timeline[-128:],
            'errors': dict(self.ports.errors, status=self.status_error) if self.status_error else dict(self.ports.errors),
            'recording': None if self.recorder is None else self.recorder.report(),
            'archive': None if self.recorder is None else self.recorder.directory.name,
        }

    def overlay_shapes(self) -> list:
        """The mission's own marks plus home, for the Mission tab and the report image."""
        from dcmn import theme
        from dcmn.mapview import Shape
        profile = self.active_profile or self.document.value
        shapes = []
        if profile['area']:
            shapes.append(Shape.polygon(profile['area'], outline=theme.WARN, width=2.))
        for index, (x, y) in enumerate(profile['waypoints']):
            shapes.append(Shape.label(x, y, str(index + 1), outline=theme.TEXT))
        if self.mission is not None: shapes.extend(self.mission.overlay())
        if self.home is not None:
            x, y = self.home
            shapes.append(Shape('rect', x - .3, y - .3, x + .3, y + .3, outline=theme.START, width=2.))
            shapes.append(Shape.label(x, y - .7, 'home', outline=theme.START))
        goal = profile['params'].get('goal') if self.phase in ('idle', 'done') else None
        if goal is not None:
            r = .35
            shapes.append(Shape('oval', goal[0] - r, goal[1] - r, goal[0] + r, goal[1] + r,
                                outline=theme.GOAL, width=1., dash=(3, 3)))
        return shapes

    def overlay(self):
        from dcmn.map_pane import Overlay
        from dcmn.navigation import permitted_points
        obs = self.observation
        points = [p[:2] for p in (self.navigation.get('points') or [])]
        goal = obs.goal.get('position')[:2] if obs.goal and obs.goal.get('position') else None
        return Overlay(route=tuple(tuple(p) for p in points), proposal=True,
                       permitted=tuple(tuple(p[:2]) for p in permitted_points(self.navigation)),
                       track=tuple(self.track[-3000:]),
                       vehicle=None if obs.pose is None else (obs.pose[0], obs.pose[1], obs.pose[3]),
                       goal=None if goal is None else tuple(goal), shapes=tuple(self.overlay_shapes()))

    def snapshot_image(self):
        if self.snapshot_provider is not None:
            try: return self.snapshot_provider()
            except Exception: return None
        from dcmn.map_pane import choose_fallback_extent, fallback_grid, snapshot_image
        grids = self.ports.grids() if hasattr(self.ports, 'grids') else {}
        grid = grids[sorted(grids)[0]] if grids else None
        if grid is None:
            profile = self.active_profile or self.document.value
            bounds, _ = choose_fallback_extent(area_bounds=G.bbox(profile['area']), workspace=profile['workspace'],
                                               pose=self.observation.xy)
            if bounds is None: return None
            grid = fallback_grid(bounds)
        return snapshot_image(grid, sim_now_s=self.sim_time_s(), overlay=self.overlay())

    def _reference_image_record(self) -> dict[str, Any] | None:
        background = self._displayed_background
        if background is None: return None
        image = background.image
        return dict(image_id=image.image_id, revision=image.revision, checksum=image.checksum,
                    opacity=background.opacity, label=background.label(), source_category=image.source_category)

    def _archive_displayed_background(self) -> None:
        background = None
        if self.background_provider is not None:
            try: background = self.background_provider()
            except Exception: background = None
        self._displayed_background = background
        if background is None or self.recorder is None: return
        image = background.image
        try:
            self.recorder.record('report.background', dict(
                time_s=self.sim_time_s(), image_id=image.image_id, revision=image.revision,
                checksum=image.checksum, opacity=background.opacity, label=background.label(),
                source_category=image.source_category), images={'reference': image})
        except ValueError:
            self.note('report.background.unverified', reason='image bytes do not match their declared checksum')

    def _write_report(self, image=None) -> Path | None:
        if self.report_dir is None: return None
        from dmsn.report import write_report
        return write_report(self.report_dir, summary=self.summary(), events=self.events,
                            image=self.snapshot_image() if image is None else image)

    def close(self, *, partial: bool = False) -> Path | None:
        if self.closed: return self.report_dir
        self.closed = True
        self.partial = self.partial or partial or self.phase != 'done'
        written = None
        try:
            try: self._connect()
            except Exception: pass
            if self.phase in ACTIVE or self.phase == 'paused':
                self.note('mission.unfinished', vital=True, phase=self.phase, reason=self.reason)
            self.note('module.closed', partial=self.partial, reason=self.reason)
            image = self.snapshot_image()
            self._archive_displayed_background()
            if self.recorder is not None: self.recorder.close()
            written = self._write_report(image=image)
        except Exception as exc:  # noqa: BLE001 - never raise out of reporting
            import sys
            print(f'dmsn report: {type(exc).__name__}: {exc}', file=sys.stderr)
        finally:
            self.ports.bus_publish('module.goodbye', run_id=self.run_id,
                                   payload=dict(state=self.phase.upper(), reason=clip(self.reason)))
            self.ports.close()
        return written


def replay(archive_dir: str | Path) -> dict[str, Any]:
    """Run a recorded mission body again from its archive and compare every decision (Q29).

    Uses only what the archive holds: the start record's mission and
    parameters, each delivered event, and each step's observation. A replay
    whose decisions all match is the proof that the record is sufficient and
    the mission is deterministic.
    """
    from dmsn.intents import to_dict as decision_dict
    reader = ArchiveReader(archive_dir)
    mission, steps, mismatches = None, 0, []
    for event in reader.events():
        kind, data = event['type'], event.get('data') or {}
        if kind == 'mission.body_start':
            mission = build(data['mission'])
            strategy = None
            if data.get('strategy'):
                from dmsn.strategies import build as build_strategy
                strategy = build_strategy(data['strategy'], data.get('strategy_params') or {})
            mission.start(data['params'], Observation.from_dict(data['obs']),
                          area=[tuple(p) for p in data.get('area') or []],
                          waypoints=[tuple(p) for p in data.get('waypoints') or []],
                          strategy=strategy)
        elif mission is None:
            continue
        elif kind == 'mission.event':
            mission.on_event(MissionEvent.from_dict(data['event']))
        elif kind == 'mission.step':
            obs = Observation.from_dict(data['obs'])
            steps += 1
            decision = decision_dict(mission.step(obs))
            if decision != data['decision']:
                mismatches.append(dict(sequence=event.get('sequence'), recorded=data['decision'], replayed=decision))
    return dict(steps=steps, mismatches=mismatches)
