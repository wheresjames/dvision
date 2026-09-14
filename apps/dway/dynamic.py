"""Dynamic route admission and observation. Intentionally no VehicleLink.

DryRun is a read-only vehicle client: it only owns its execution-status endpoint.
Start/arm/mode/setpoint APIs do not exist here, including on shutdown and errors.
"""
from __future__ import annotations

from dataclasses import asdict
import csv
import html
import io
import json
import math
from pathlib import Path
import time

from dcmn.archive import Recorder, next_archive_dir
from dcmn.context import Context, validate_goal, validate_pose
from dcmn.maps import MapSession
from dcmn.module_bus import PymembusModuleBus, requests_shutdown
from dcmn.navigation import (ExecutionProfile, Snapshot, STATUS_SCHEMA, context_identity, distance_along,
                             new_session, permitted_points, point_at, project_progress, validate)
from dway.frames import ProviderFrame

STALL_S = 3.0


class DynamicRouteSource:
    """Admission state independent of UI, transport and flight lifecycle.

    The stop-generation latch binds only an *active* route: an executor with
    nothing active has nothing to stop, so it adopts each generation as handled.
    Once active, any newer generation -- including one whose unavailable record
    was never seen -- requires a confirmed stop before admission resumes. A
    dry-run never activates, so it never claims to have stopped a vehicle.
    """
    kind = 'dynamic'

    def __init__(self, vehicle, planner, profile, *, stall_s=STALL_S):
        self.vehicle, self.planner, self.profile = vehicle, planner, profile
        self.stall_s = stall_s
        self.session = None
        self.sequence = -1
        self.stop_generation = 0
        self.handled_stop = 0
        self.revision = -1
        self.geometry = None
        self.value = {}
        self.active = False
        self.restart_stop = False    # planner session changed under an active route
        self.start_required = False  # a new Start is needed before any activation
        self.disposition = None
        self.progress = None
        self.state, self.reason = 'WAITING', 'waiting for complete navigation snapshot'
        self.changed_wall = time.monotonic()

    def describe(self):
        return f'dynamic routes from {self.planner}'

    def _dispose(self, value, revision):
        self.disposition = dict(value=value, geometry_revision=revision, navigation_sequence=self.sequence)

    def consume(self, value, context, *, wall=None):
        wall = time.monotonic() if wall is None else wall
        try:
            validate(value)
            if value['vehicle_id'] != self.vehicle or value['planner'] != self.planner:
                raise ValueError('wrong vehicle or selected planner')
            if value['session'] != self.session:
                if self.session is not None and self.active: self.restart_stop = True
                if self.session is not None: self.start_required = True
                self.session, self.sequence, self.revision, self.geometry = value['session'], -1, -1, None
                self.stop_generation = self.handled_stop = value['stop_generation']
                self.value = {}
            if value['sequence'] < self.sequence: raise ValueError('out-of-order navigation snapshot')
            if value['sequence'] == self.sequence and value != self.value:
                raise ValueError('publication sequence reused with different content')
            if value['sequence'] != self.sequence: self.changed_wall = wall
            if value['stop_generation'] < self.stop_generation: raise ValueError('stop generation moved backwards')
            if value['geometry_revision'] == self.revision and value['points'] != self.geometry:
                raise ValueError('geometry revision reused with different points')
            if value['geometry_revision'] < self.revision: raise ValueError('geometry revision moved backwards')
            self.sequence, self.stop_generation = value['sequence'], value['stop_generation']
            if value['geometry_revision'] != self.revision: self.progress = None
            self.revision, self.geometry, self.value = value['geometry_revision'], value['points'], value
            if not self.active: self.handled_stop = self.stop_generation
            if wall-self.changed_wall > self.stall_s: raise ValueError('planner snapshot stream stalled')
            if not context: raise ValueError('session provider unavailable')
            if any(value['context'].get(k) != v for k, v in context_identity(context).items()):
                raise ValueError('provider/frame/clock context mismatch')
            pose = validate_pose(context.get('pose'), context)
            ProviderFrame.from_context(context)
            goal = validate_goal(context.get('goal'), context)
            if value.get('goal') != goal: raise ValueError('goal authority/revision mismatch')
            if value.get('profile') != self.profile.digest:
                raise ValueError(f"execution profile mismatch (planner {str(value.get('profile'))[:10]}, executor "
                                 f"{self.profile.digest[:10]}): load the same profile and --profile-set in dnav and dway")
            if value['time_s'] > context['time_s']+1e-6: raise ValueError('navigation snapshot from future')
            clearance = value['clearance']
            if not clearance['eligible']: raise ValueError(clearance['reason'])
            if context['time_s'] + self.profile.stopping_s >= clearance['valid_until_s']:
                raise ValueError('clearance expired or too little time to stop')
            if not self.active:
                start = permitted_points(value)[0]
                if math.dist([pose[k] for k in ('x_m', 'y_m', 'z_m')], start) > self.profile.join_m:
                    raise ValueError('pose moved beyond route join allowance')
            if self._stop_needed():
                raise ValueError('latched stop')
            self.state = 'READY'
            if self.active:
                self.reason = 'active route still permitted'
                self._dispose('active', self.revision)
            else:
                self.reason = 'admitted for observation only; uncalibrated profile, no motion'
                self._dispose('accepted', self.revision)
        except (ValueError, KeyError, TypeError) as exc:
            if self._stop_needed() or self.active:
                self.state = 'STOP_REQUIRED'
                self.reason = ('planner restarted under an active route; stop and new Start required'
                               if self.restart_stop else f'stop generation {self.stop_generation} unhandled'
                               if self.stop_generation > self.handled_stop else f'stop required: {exc}')
            else:
                self.state, self.reason = 'REJECTED', str(exc)
                self._dispose('rejected', value.get('geometry_revision') if isinstance(value, dict)
                              and type(value.get('geometry_revision')) is int else None)
        return self.state

    def _stop_needed(self):
        return self.active and (self.restart_stop or self.stop_generation > self.handled_stop)

    def activate(self):
        """Mark the admitted route active. The dry-run never calls this; only a
        started executor or a deterministic fixture may."""
        if self.state != 'READY' or self.active: raise ValueError(f'cannot activate from {self.state}')
        if self.start_required: raise ValueError('planner session changed; new Start required')
        self.active = True
        self._dispose('active', self.revision)

    def confirm_stopped(self, generation):
        """Called only by a future measured executor or deterministic fixture."""
        if not self.active: raise ValueError('no active route to stop')
        if generation != self.stop_generation: raise ValueError('wrong stop generation')
        self.handled_stop, self.active, self.restart_stop = generation, False, False
        self.progress = None
        self._dispose('stopped', self.revision)

    def start(self):
        """An explicit operator Start clears the restart requirement (never automatic)."""
        self.start_required = False

    def release(self, disposition):
        """End the active route without claiming a stop generation (completion, cancel, stop)."""
        self.active, self.restart_stop = False, False
        self.handled_stop = self.stop_generation
        self.progress = None
        self._dispose(disposition, self.revision)

    def observe(self, pose, time_s):
        """Where the observed pose sits against the admitted interval. Nothing is commanded."""
        unavailable = {}
        result = dict(progress=None, tracking_error_m=None, remaining_permitted_m=None,
                      stopping_margin_m=None, validity_remaining_s=None)
        value = self.value if self.state == 'READY' else {}
        if not value:
            unavailable['route'] = 'no admitted route'
        elif pose is None:
            unavailable['pose'] = 'pose unavailable'
        else:
            clearance, points = value['clearance'], value['points']
            xyz = [pose[k] for k in ('x_m', 'y_m', 'z_m')]
            previous = self.progress or clearance['start']
            progress, error = project_progress(points, xyz, previous)
            self.progress = progress
            end = clearance['end']
            remaining = distance_along(points, progress, end)
            result.update(progress=progress, tracking_error_m=error, remaining_permitted_m=remaining,
                          stopping_margin_m=remaining - self.profile.stopping_m,
                          validity_remaining_s=clearance['valid_until_s'] - time_s,
                          endpoint=point_at(points, end), reaches_goal=bool(clearance.get('reaches_goal')))
        unavailable.update(speed_mps='not measured in dry-run', commanded_target='dry-run sends no targets')
        result['unavailable'] = unavailable
        return result


class DryRun:
    def __init__(self, instance, *, planner='dnav', profile=None, overrides=None):
        self.id = instance
        self.profile = ExecutionProfile.load(profile, overrides)
        self.source = DynamicRouteSource(instance, planner, self.profile)
        self.context = Context(instance)
        self.input = Snapshot(instance, planner=planner)
        self.output = Snapshot(instance, 'execution')
        self.output.start()  # fail before running if another executor owns this endpoint
        self.maps = MapSession(instance)
        self.bus = PymembusModuleBus(instance, 'executor', 'dway-dry-run')
        self.session = new_session()
        self.sequence = 0
        self.snapshot = {}
        self.status = {}
        self.observation = {}
        self.report_session = None
        self.recorder = None
        self.report_dir = None
        self.report_error = ''
        self.events = []
        self._last_record = None
        self.track = []
        self._pose_identity = None
        self._pose_seq = None
        self._provider_wall = time.monotonic()
        self._heartbeat = 0.
        self.closed = False
        self.shutdown_requested = False

    def _record(self, kind, data, grids=None):
        event = dict(kind=kind, data=data)
        self.events.append(event); del self.events[:-256]
        if self.recorder: self.recorder.record(kind, data, grids)

    def _adopt_report(self):
        sid = self.snapshot.get('session_id')
        if not sid or sid == self.report_session: return
        self._finish_report()
        self.report_session = sid
        try:
            # Derived files stay alongside this append-once archive, not a shared
            # flight.jsonl that a restarted process could truncate.
            directory = next_archive_dir(Path(self.snapshot['report_root'])/'dway')
            self.recorder = Recorder(directory, dict(mode='dynamic-dry-run', session=self.session,
                                     profile=asdict(self.profile)), module='dway')
            self.report_dir = directory
            self.events = []
            self._record('retained_state', dict(context=self.snapshot, navigation=self.source.value,
                                               execution=self.status))
        except (OSError, ValueError) as exc:
            self.recorder = None; self.report_dir = None; self.report_error = str(exc)

    def step(self):
        if self.closed: return self.status
        now = time.monotonic()
        try: self.snapshot = self.context.read()
        except (ValueError, KeyError): self.snapshot = {}
        self._adopt_report()
        self.maps.poll()
        self.bus.connect()
        for event in self.bus.receive():
            if requests_shutdown(event): self.shutdown_requested = True
        if now-self._heartbeat >= 1:
            self.bus.publish('module.heartbeat', payload=dict(state=self.source.state,
                capabilities=dict(dry_run=True, holds_control_lease=False,
                                  execution_endpoint=self.output.name)))
            self._heartbeat = now
        value = {}
        rejected_raw = None
        try:
            value = self.input.read()
            if value:
                self.source.consume(value, self.snapshot, wall=now)
            else:
                self.source.state, self.source.reason = 'WAITING', 'navigation publisher unavailable'
        except (ValueError, RuntimeError) as exc:
            rejected_raw = self.input.last_raw
            self.source.state, self.source.reason = 'REJECTED', str(exc)
        pose_mark = (self.snapshot.get('provider_id'), self.snapshot.get('pose_sequence'))
        if pose_mark != self._pose_seq:
            self._pose_seq, self._provider_wall = pose_mark, now
        if now-self._provider_wall > STALL_S:
            self.source.state, self.source.reason = 'REJECTED', 'provider pose stream stalled'
        pose = self.snapshot.get('pose')
        try: validate_pose(pose, self.snapshot)
        except (ValueError, KeyError, TypeError): pose = None
        identity = context_identity(self.snapshot)
        if identity != self._pose_identity or pose is None:
            self.track.clear(); self._pose_identity = identity  # never join across gaps or resets
        if pose:
            pt = (pose['x_m'], pose['y_m'])
            if not self.track or self.track[-1] != pt: self.track.append(pt)
            del self.track[:-3000]
        time_s = self.snapshot.get('time_s', 0.)
        self.observation = self.source.observe(pose, time_s)
        self.sequence += 1
        self.status = dict(schema=STATUS_SCHEMA, vehicle_id=self.id, session=self.session,
            sequence=self.sequence, time_s=time_s, stop_generation=self.source.handled_stop,
            planner_session=self.source.session, planner=self.source.planner,
            state=self.source.state, reason=self.source.reason, dry_run=True, owns_control=False,
            lease_state='not requested (dry-run)', goal=self.snapshot.get('goal'),
            disposition=self.source.disposition, geometry_revision=self.source.revision,
            navigation_sequence=self.source.sequence, pose=pose, commanded_target=None,
            progress=self.observation['progress'], tracking_error_m=self.observation['tracking_error_m'],
            remaining_permitted_m=self.observation['remaining_permitted_m'],
            stopping_margin_m=self.observation['stopping_margin_m'],
            validity_remaining_s=self.observation['validity_remaining_s'], speed_mps=None,
            unavailable=self.observation['unavailable'], profile=self.profile.digest,
            profile_calibrated=self.profile.calibrated, recording_error=self.report_error)
        self.output.write(self.status)
        grids = {sid: state.grid for sid, state in self.maps.states.items() if state.grid is not None}
        mark = (self.source.session, self.source.sequence, self.source.state, self.source.reason,
                self.report_session)
        if mark != self._last_record:
            self._record('navigation.observed', dict(navigation=value, rejected_raw=rejected_raw,
                                                    execution=self.status, context=self.snapshot), grids)
            self._last_record = mark
        return self.status

    def overlay(self):
        from dcmn.map_pane import Overlay
        value = self.source.value
        pose = self.status.get('pose')
        # Rejected/expired geometry stays a proposal, never a permission.
        permitted = permitted_points(value) if self.source.state == 'READY' else []
        return Overlay(route=[p[:2] for p in value.get('points', ())], proposal=True,
            permitted=[p[:2] for p in permitted], track=self.track,
            vehicle=None if not pose else (pose['x_m'], pose['y_m'], pose['heading_deg']),
            goal=(value['goal']['position'][:2] if value.get('goal') else None))

    def readouts(self):
        """Operator-facing lines; unavailable values say why instead of showing zero."""
        s, o, p = self.status, self.observation, self.profile
        def fmt(key, unit, digits=2):
            v = s.get(key)
            return f'{v:.{digits}f}{unit}' if isinstance(v, (int, float)) else 'n/a'
        grids = [st.grid for st in self.maps.states.values() if st.grid is not None]
        now = self.snapshot.get('time_s')
        ages = ', '.join(f'{g.source} {now-g.sim_time_s:.1f}s' for g in grids) if grids and now is not None else 'none'
        pose = s.get('pose')
        altitude = 'n/a' if not pose else f'{pose["z_m"]:.2f} m'
        disposition = s.get('disposition') or {}
        return [
            f'{s.get("state", "WAITING")}: {s.get("reason", "")}',
            f'disposition {disposition.get("value", "none")} (revision {disposition.get("geometry_revision")})  ·  '
            f'route revision {s.get("geometry_revision")}  ·  publication {s.get("navigation_sequence")}  ·  '
            f'stop generation {s.get("stop_generation")}',
            f'remaining permitted {fmt("remaining_permitted_m", " m")}  ·  stopping margin '
            f'{fmt("stopping_margin_m", " m")} (stop {p.stopping_m:g} m / {p.stopping_s:g} s)  ·  '
            f'validity {fmt("validity_remaining_s", " s", 1)}  ·  cross-track {fmt("tracking_error_m", " m")} '
            f'/ {p.tracking_m:g} m',
            f'altitude {altitude} (slab {p.altitude_m-p.half_height_m:.2f}–{p.altitude_m+p.half_height_m:.2f} m, '
            f'{p.slab_assumption})  ·  speed cap {p.speed_mps:g} m/s  ·  evidence age {ages}',
            f'lease: {s.get("lease_state", "n/a")}  ·  commanded target: none  ·  profile '
            f'{"calibrated" if p.calibrated else "SYNTHETIC"} {p.digest[:10]}  ·  recording '
            f'{self.report_error or self.report_dir or "waiting for provider"}',
            *(f'unavailable {k}: {v}' for k, v in (o.get('unavailable') or {}).items()),
        ]

    def _finish_report(self):
        if self.recorder is None: return
        recorder, self.recorder = self.recorder, None
        recorder.close()
        try:
            summary = dict(schema_version=1, mode='dynamic-dry-run', execution=self.status,
                           recording=recorder.report(), profile=asdict(self.profile),
                           note='observation only: no lease, arming, mode change or targets')
            directory = recorder.directory
            temp = directory/'summary.json.tmp'
            temp.write_text(json.dumps(summary, indent=2)+'\n'); temp.replace(directory/'summary.json')
            (directory/'events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in self.events))
            table = io.StringIO()
            writer = csv.writer(table)
            writer.writerow(('time_s', 'kind', 'state', 'reason', 'disposition', 'geometry_revision',
                             'navigation_sequence', 'stop_generation', 'remaining_permitted_m'))
            for e in self.events:
                x = e['data'].get('execution') or {}
                writer.writerow((x.get('time_s'), e['kind'], x.get('state'), x.get('reason'),
                                 (x.get('disposition') or {}).get('value'), x.get('geometry_revision'),
                                 x.get('navigation_sequence'), x.get('stop_generation'),
                                 x.get('remaining_permitted_m')))
            (directory/'events.csv').write_text(table.getvalue())
            from dcmn import report_html as page
            rows = [[html.escape(str(c)) for c in row] for row in list(csv.reader(io.StringIO(table.getvalue())))[1:]]
            (directory/'report.html').write_text(page.document('dway dynamic dry-run — no motion', blocks=[
                '<style>pre{white-space:pre-wrap;word-break:break-word;font-size:12px;margin:0}</style>',
                page.section('Summary', '<pre>' + html.escape(json.dumps(summary, indent=2)) + '</pre>'),
                page.section('Route', '<p class="muted">Dashed line: proposal. Thick line with square endpoint: '
                             'permitted interval. Thin trail: observed pose. No active flight, no commanded '
                             'target.</p><figure><img src="route.png" alt="Final observed route"></figure>'),
                page.section('Recent observation timeline', page.table(
                    ('time s', 'event', 'state', 'reason', 'disposition', 'revision', 'publication', 'stop gen',
                     'remaining m'), rows),
                    '<p class="muted">Complete committed observations and grids are in index.jsonl and chunks/; '
                    'the table (and events.csv) holds the most recent 256 events of this segment.</p>')]))
            grids = [s.grid for s in self.maps.states.values() if s.grid is not None]
            if grids:
                from dcmn.map_pane import snapshot_image
                snapshot_image(grids[0], sim_now_s=self.snapshot.get('time_s', 0.),
                               overlay=self.overlay()).save(directory/'route.png')
        except Exception as exc:  # report rendering must never break observation
            self.report_error = str(exc)

    def close(self):
        if self.closed: return
        self.closed = True
        self._record('dry_run.closed', dict(execution=self.status))
        self._finish_report()
        self.output.close(); self.input.close(); self.maps.close(); self.bus.close()


class DryWindow:
    def __init__(self, run, root=None):
        import tkinter as tk
        from tkinter import ttk
        from dcmn.map_pane import MapPane
        from dcmn.pacing import MAP_HZ, Paced, TEXT_HZ
        from dcmn.tktheme import apply_theme
        self.run = run
        self.root = root if root is not None else tk.Tk(); self.root.title('dway — dynamic dry-run (NO MOTION)')
        apply_theme(self.root)
        self.text = tk.StringVar()
        ttk.Label(self.root, text='DRY RUN — observation only; no lease, arming or vehicle commands').pack(fill='x')
        ttk.Label(self.root, textvariable=self.text, wraplength=900, justify='left').pack(fill='x')
        ttk.Label(self.root, text='Dashed: proposal (not permission)  |  Thick + square: permitted interval '
                                  'and stop point  |  Thin trail: observed pose  |  No target symbol: nothing '
                                  'commanded').pack(fill='x')
        self.pane = MapPane(self.root, width=750, height=550, title='Route admission')
        self.pane.widget.pack(fill='both', expand=True)
        self._map, self._text = Paced(MAP_HZ), Paced(TEXT_HZ)
        self.root.protocol('WM_DELETE_WINDOW', self.close)
        self.root.after(0, self.tick)

    def tick(self):
        if self.run.closed: return
        self.run.step()
        if self._text.due(): self.text.set('\n'.join(self.run.readouts()))
        if self._map.due():
            grids = [s.grid for s in self.run.maps.states.values() if s.grid is not None]
            self.pane.set_grid(grids[0] if grids else None, sim_now_s=self.run.snapshot.get('time_s', 0.))
            self.pane.state.overlay = self.run.overlay(); self.pane.refresh()
        if self.run.shutdown_requested: self.close()
        else: self.root.after(100, self.tick)

    def close(self):
        self.run.close(); self.root.destroy()


def run_cli(args):
    import signal
    run = DryRun(args.id, planner=args.planner, profile=args.execution_profile,
                 overrides=getattr(args, 'profile_set', None))
    def stop(*_): run.shutdown_requested = True
    signal.signal(signal.SIGINT, stop); signal.signal(signal.SIGTERM, stop)
    deadline = time.monotonic()+args.timeout if args.timeout else float('inf')
    try:
        if args.no_ui:
            last = None
            while not run.shutdown_requested and time.monotonic() < deadline:
                status = run.step()
                mark = status['state'], status['reason']
                if mark != last: print(f'dway dry-run: {mark[0]}: {mark[1]}', flush=True); last = mark
                time.sleep(.1)
        else:
            window = DryWindow(run)
            if args.timeout: window.root.after(int(args.timeout*1000), window.close)
            window.root.mainloop()
    finally: run.close()
    return 0
