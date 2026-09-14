"""One complete executable proposal; observation events may remain abbreviated."""
from dataclasses import asdict
import time

from dcmn.navigation import (SCHEMA, Snapshot, context_identity, new_session, validate)
from dnav.clearance import Clearance

# An executor whose status sequence has not advanced for this long (monotonic
# wall time) is not live, whatever its retained bytes or data clock say.
EXECUTOR_LIVENESS_S = 1.0


class RoutePublisher:
    def __init__(self, vehicle, profile, planner='dnav', *, liveness_s=EXECUTOR_LIVENESS_S):
        self.vehicle, self.planner, self.profile = vehicle, planner, profile
        self.liveness_s = liveness_s
        self.session = new_session()
        self.output = Snapshot(vehicle, planner=planner)
        self.output.start()
        self.feedback = Snapshot(vehicle, 'execution')
        self.validator = Clearance(profile)
        self.sequence = self.revision = self.stop_generation = 0
        self.last = {}
        self.geometry = None
        self.executor = {}
        self.error = ''
        self.binding = None
        self.pending_stop = None
        #: Why the latest stop generation was advanced; kept visible until the stop is confirmed.
        self.stop_cause = ''
        self._executor_mark = None
        self._executor_wall = float('-inf')

    def _read_executor(self, wall):
        try: feedback = self.feedback.read()
        except ValueError: feedback = {}
        mark = (feedback.get('session'), feedback.get('sequence'))
        if mark != self._executor_mark:
            self._executor_mark, self._executor_wall = mark, wall
        live = bool(feedback) and wall - self._executor_wall <= self.liveness_s
        self.executor = feedback
        # Only a live, moving-capable executor bound to this planner session counts.
        return feedback if (live and feedback.get('planner_session') == self.session
                            and feedback.get('vehicle_id') == self.vehicle
                            and feedback.get('dry_run') is False) else {}

    def publish(self, run, *, wall=None):
        wall = time.monotonic() if wall is None else wall
        points = [[w.x_m, w.y_m, w.z_m] for w in run.route.waypoints]
        self.sequence += 1
        executor = self._read_executor(wall)
        state = executor.get('state')
        engaged = state in ('EXECUTING', 'BRAKING', 'HOLDING')
        moving = state in ('EXECUTING', 'BRAKING')
        same_context = (self.last.get('goal') == run.goal_descriptor and all(
            self.last.get('context', {}).get(k) == v for k, v in context_identity(run.snapshot).items()))
        progress = (0, 0.)
        pinned = False
        # Keep the executing geometry, and pin a validated replacement while the
        # executor holds, so optional re-planning cannot churn it. A held
        # executor that has confirmed its stop and released the route is
        # waiting for the route dnav plans from its stopped pose: the executed
        # geometry is no longer pinned, but everything the executor still
        # owns (pause, expiry, a prefix waiting to extend) is.
        stopped = (state == 'HOLDING' and (executor.get('disposition') or {}).get('value') == 'stopped')
        if engaged and self.last and self.last['points'] and same_context and not stopped and (
                moving or self.last['clearance']['eligible']):
            points = self.last['points']
            pinned = True
            if executor.get('geometry_revision') == self.last['geometry_revision'] and executor.get('progress'):
                progress = tuple(executor['progress'])
            elif moving:
                progress = (-1, 0.)  # executing something we cannot locate: invalid, never assumed
        binding = repr((context_identity(run.snapshot), run.goal_descriptor,
                        run.session.identity, run.session.context, sorted(run._grids()), self.profile.digest))
        if self.binding is not None and binding != self.binding:
            self.stop_generation += 1
            self.stop_cause = 'evidence sources, mapping, goal or context changed'
            if moving: self.pending_stop = self.stop_generation
        self.binding = binding
        key = (points, run.goal_descriptor, context_identity(run.snapshot), self.profile.digest)
        if key != self.geometry:
            self.revision += 1
            self.geometry = key
        pose_record = run._pose()
        pose = None if pose_record is None else [pose_record[k] for k in ('x_m', 'y_m', 'z_m')]
        identity = repr((run.session.identity, run.session.context))
        clearance = self.validator.check(points, pose, run._grids(), run.sim_time_s(), identity, start=progress)
        if not run.route.ok and not pinned:
            # A pinned route is judged by its own clearance above; a replan that
            # failed from the moving pose (for example from inside a margin the
            # route is already leaving) does not withdraw the route being flown.
            clearance.update(eligible=False, reason=run.route.reason or run.route.status)
        if self.last and self.last['clearance']['eligible'] and not clearance['eligible']:
            self.stop_generation += 1
            self.stop_cause = f"permission withdrawn: {clearance['reason']}"
            if moving: self.pending_stop = self.stop_generation
        if self.pending_stop is not None:
            if state == 'HOLDING' and executor.get('stop_generation') == self.pending_stop:
                self.pending_stop = None
            else:
                clearance.update(eligible=False, reason=f'waiting for executor to confirm stop '
                                                        f'(generation {self.pending_stop}: {self.stop_cause})'[:500],
                                 withdrawn_reason=self.stop_cause)
        value = dict(schema=SCHEMA, vehicle_id=self.vehicle, planner=self.planner,
                     session=self.session, sequence=self.sequence, time_s=run.sim_time_s(),
                     stop_generation=self.stop_generation, geometry_revision=self.revision,
                     context=dict(context_identity(run.snapshot), evidence=dict(run.session.context)),
                     goal=run.goal_descriptor, planning_status=run.route.status,
                     points=points, start_pose=pose_record, attempt=run.attempts,
                     profile=self.profile.digest, clearance=clearance,
                     executor_reference=None if not executor else {
                         k: executor.get(k) for k in ('session', 'sequence', 'state', 'geometry_revision',
                                                      'progress', 'stop_generation', 'pose')},
                     evidence={sid: dict(revision=g.revision, time_s=g.sim_time_s)
                               for sid, g in run._grids().items()})
        try: validate(value)
        except ValueError as exc:
            # Never leave the previous eligible route retained after oversize failure.
            self.stop_generation += 1
            self.revision += 1
            if moving: self.pending_stop = self.stop_generation
            value.update(points=[], geometry_revision=self.revision, planning_status='unavailable', evidence={},
                         stop_generation=self.stop_generation, executor_reference=None,
                         clearance=dict(eligible=False, reason=f'unpublishable route: {exc}'))
            self.geometry = None
            validate(value)
        self.last = value
        try:
            self.output.write(value); self.error = ''
        except (ValueError, RuntimeError) as exc:
            self.error = str(exc)
        if run.recorder is not None:
            # Each validation references its exact grids, even when the planner
            # skipped an unchanged input key. Persistent veto state is archived too.
            from dcmn.maps import EvidenceGrid
            import numpy as np
            grids = dict(run._grids())
            template = next(iter(grids.values()), None)
            for sid, mask in self.validator.veto.items():
                if template is None or mask.shape != template.geometry.shape[1:]: continue
                occupancy = np.where(mask[None], 254, 0).astype(np.uint8)
                observed = np.full(occupancy.shape, max(1, round(run.sim_time_s()*1000)), np.uint32)
                grids['veto:'+sid] = EvidenceGrid(template.geometry, occupancy, observed,
                    'veto:'+sid, self.sequence, run.sim_time_s())
            run.recorder.record('navigation.snapshot', dict(snapshot=value, transport_error=self.error,
                                executor=self.executor, profile=asdict(self.profile)), grids)
        return value

    def close(self):
        self.output.close(); self.feedback.close()
