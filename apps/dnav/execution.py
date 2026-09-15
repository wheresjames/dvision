"""One complete executable proposal; observation events may remain abbreviated."""
from dataclasses import asdict
import math
import time

from dcmn.navigation import (SCHEMA, Snapshot, context_identity, distance_along, new_session, point_at,
                             validate)
from dnav.clearance import Clearance

# An executor whose status sequence has not advanced for this long (monotonic
# wall time) is not live, whatever its retained bytes or data clock say.
EXECUTOR_LIVENESS_S = 1.0
#: Planning failures a fallback route may stand in for. The others (stale
#: evidence or pose, frame changes, coverage) say nothing may be flown.
FALLBACK_STATUSES = ('goal_unreachable', 'no_route')


def _turn_deg(a, b, c, d):
    """Angle between directions ``a``->``b`` and ``c``->``d``; 0 when either is degenerate."""
    if math.dist(a[:2], b[:2]) < 1e-6 or math.dist(c[:2], d[:2]) < 1e-6: return 0.
    first, second = math.atan2(b[1]-a[1], b[0]-a[0]), math.atan2(d[1]-c[1], d[0]-c[0])
    return abs((math.degrees(second - first) + 180.) % 360. - 180.)


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
        #: The last route planned to the goal under the current binding, kept
        #: for when replanning fails.
        self.retained = None
        #: What the last published route was, and where a substitute ends.
        self.mode, self.substitute = 'planned', None
        #: A replacement offered to the moving executor, repeated until it reports
        #: the new revision: ``from_revision``, ``start`` and its ``revision``.
        self.handoff = None
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

    @staticmethod
    def _locate(points, executor, revision, pose):
        """Where a route adopted from a stop begins.

        The executor's own progress when it is on this geometry, else the
        nearest point on any segment; the join allowance then bounds how far
        from the route the vehicle may be.
        """
        if revision is not None and executor.get('geometry_revision') == revision and executor.get('progress'):
            return tuple(executor['progress'])
        if pose is None or len(points) < 2: return (0, 0.)
        best = None
        for i, (a, b) in enumerate(zip(points, points[1:])):
            ab = [q-p for p, q in zip(a, b)]
            span = sum(c*c for c in ab)
            t = 0. if span == 0 else max(0., min(1., sum((x-p)*c for x, p, c in zip(pose, a, ab))/span))
            error = math.dist(pose, point_at(points, (i, t)))
            if best is None or error < best[0] - 1e-9: best = (error, (i, t))
        return best[1]

    def _handoff(self, candidate, current, progress, remaining, judge, pose):
        """``(start, clearance)`` for taking up ``candidate`` while moving, or None.

        Offered only for a replacement the vehicle is already on (within
        ``handoff_m``), heading the same way (within ``turn_tolerance_deg``), and
        that permits more than is left of the route being flown.
        """
        p = self.profile
        if pose is None or len(candidate) < 2 or not 0 <= progress[0] < len(current) - 1: return None
        start = self._locate(candidate, {}, None, pose)
        if math.dist(pose[:2], point_at(candidate, start)[:2]) > p.handoff_m: return None
        if _turn_deg(current[progress[0]], current[progress[0]+1],
                     candidate[start[0]], candidate[start[0]+1]) > p.turn_tolerance_deg:
            return None
        verdict = judge(candidate, start, True)
        if not verdict['eligible'] or verdict['distance_m'] <= p.stopping_m: return None
        if not verdict['reaches_goal'] and verdict['distance_m'] <= remaining + 1e-6: return None
        return start, verdict

    def _substitute(self, run, judge, pose):
        """``(points, clearance, route)`` for an admissible route near an unreachable goal, or Nones."""
        plan = getattr(run, 'substitute_route', None)
        route = plan() if plan is not None else None
        if route is None or not route.ok or pose is None: return None, None, None
        candidate = [[w.x_m, w.y_m, w.z_m] for w in route.waypoints]
        # Already there: holding is the answer, not another route to the same point.
        if math.dist(pose[:2], candidate[-1][:2]) <= max(self.profile.arrival_m, self.profile.stopping_m):
            return None, None, None
        verdict = judge(candidate, (0, 0.), True)
        return (candidate, verdict, route) if verdict['eligible'] else (None, None, None)

    def publish(self, run, *, wall=None):
        wall = time.monotonic() if wall is None else wall
        planned = [[w.x_m, w.y_m, w.z_m] for w in run.route.waypoints]
        points, mode, substitute = planned, 'planned', None
        # A planner that cannot reach the goal (an errant obstacle on it, or one
        # closing the way far ahead) leaves the vehicle a fallback: the last
        # planned route as far as it is still clear, then a route to a point
        # near the goal. The goal itself never changes, and every replan still
        # aims at it.
        fallback = not run.route.ok and run.route.status in FALLBACK_STATUSES
        self.sequence += 1
        executor = self._read_executor(wall)
        state = executor.get('state')
        engaged = state in ('EXECUTING', 'BRAKING', 'HOLDING')
        moving = state in ('EXECUTING', 'BRAKING')
        same_context = (self.last.get('goal') == run.goal_descriptor and all(
            self.last.get('context', {}).get(k) == v for k, v in context_identity(run.snapshot).items()))
        binding = repr((context_identity(run.snapshot), run.goal_descriptor,
                        run.session.identity, run.session.context, sorted(run._grids()), self.profile.digest))
        if self.binding is not None and binding != self.binding:
            self.stop_generation += 1
            self.stop_cause = 'evidence sources, mapping, goal or context changed'
            if moving: self.pending_stop = self.stop_generation
            self.retained = None
        self.binding = binding
        if run.route.ok and planned: self.retained = planned
        pose_record = run._pose()
        pose = None if pose_record is None else [pose_record[k] for k in ('x_m', 'y_m', 'z_m')]
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
            points, mode, substitute = self.last['points'], self.mode, self.substitute
            pinned = True
            if executor.get('geometry_revision') == self.last['geometry_revision'] and executor.get('progress'):
                progress = tuple(executor['progress'])
            elif moving and self.handoff and executor.get('geometry_revision') == self.handoff['from_revision']:
                # Offered this route, still flying the one before it: locate it here.
                progress = self._locate(points, {}, None, pose)
            elif moving:
                progress = (-1, 0.)  # executing something we cannot locate: invalid, never assumed
        identity = repr((run.session.identity, run.session.context))
        grids, now = run._grids(), run.sim_time_s()
        def judge(candidate, location, prefix):
            return self.validator.check(candidate, pose, grids, now, identity, start=location, prefix=prefix)
        def revision_of(candidate):
            key = (candidate, run.goal_descriptor, context_identity(run.snapshot), self.profile.digest)
            return self.revision if key == self.geometry else None
        # With in-motion handoff enabled, a route blocked ahead is shortened to
        # the obstacle rather than withdrawn: the vehicle keeps flying while a
        # replacement is found, and stops by itself only at the shortened end.
        shorten = self.profile.handoff_m > 0
        switch, handoff, clearance = '', None, None
        if pinned:
            clearance = judge(points, progress, fallback or shorten)
            if fallback and mode == 'planned': mode = 'retained'
            if (shorten and state == 'EXECUTING' and run.route.ok and planned and planned != points
                    and executor.get('geometry_revision') == self.last['geometry_revision']
                    and (mode != 'planned' or not clearance['eligible'] or not clearance['reaches_goal'])):
                remaining = (distance_along(points, progress, clearance['end'])
                             if clearance['eligible'] and len(points) > 1 else 0.)
                taken = self._handoff(planned, points, progress, remaining, judge, pose)
                if taken is not None:
                    handoff = dict(from_revision=self.last['geometry_revision'], start=list(taken[0]))
                    points, mode, substitute, progress, clearance = planned, 'planned', None, taken[0], taken[1]
                    pinned = False
            if pinned and mode != 'planned' and run.route.ok and planned and planned != points:
                # A fallback only ever stands in for a plan to the goal.
                points, mode, substitute, pinned, progress = planned, 'planned', None, False, (0, 0.)
                clearance, switch = None, 'planner found a route to the goal'
        elif fallback and self.retained:
            points, mode = self.retained, 'retained'
            progress = self._locate(points, executor, revision_of(points), pose)
            clearance = judge(points, progress, True)
        if fallback and (clearance is None or not clearance['eligible']):
            candidate, verdict, route = self._substitute(run, judge, pose)
            if verdict is not None:
                if engaged and not stopped and self.last and candidate != self.last['points']:
                    switch = 'retained route used up; substitute route to a point near the goal'
                points, mode, clearance, progress = candidate, 'substitute', verdict, (0, 0.)
                substitute = candidate[-1]
        if clearance is None:
            clearance = judge(points, progress, False)
        if switch and engaged and not stopped:
            # A different geometry is only ever taken up from a confirmed stop.
            self.stop_generation += 1
            self.stop_cause = switch
            if moving: self.pending_stop = self.stop_generation
        key = (points, run.goal_descriptor, context_identity(run.snapshot), self.profile.digest)
        if key != self.geometry:
            self.revision += 1
            self.geometry = key
        if handoff is not None:
            self.handoff = dict(handoff, revision=self.revision)
        elif self.handoff and (self.handoff['revision'] != self.revision or not moving
                               or executor.get('geometry_revision') == self.revision):
            self.handoff = None
        if (pinned and state == 'HOLDING' and executor.get('hold_kind') == 'corner'
                and clearance['eligible'] and pose is not None):
            segment = executor.get('segment')
            if type(segment) is not int or not 0 <= segment < len(points)-2:
                clearance.update(eligible=False, reason='invalid corner segment in executor feedback')
            else:
                # This is the shortcut actually flown after stopping short of
                # a vertex. Require a complete permission for it before dway
                # may advance; otherwise the normal stop/replan handshake runs.
                # A fallback route may continue to where its permission ends.
                departure = judge([pose, points[segment+2]], (0, 0.), fallback)
                eligible = departure['eligible'] and (departure['reaches_goal'] or fallback)
                clearance['corner_departure'] = dict(eligible=eligible, pose=pose, segment=segment+1,
                                                     executor_session=executor['session'])
                clearance['valid_until_s'] = min(clearance['valid_until_s'], departure['valid_until_s'])
                if not eligible:
                    clearance.update(eligible=False, reason=f"corner departure blocked: {departure['reason']}")
        if mode == 'planned' and not run.route.ok and not pinned:
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
                     goal=run.goal_descriptor, planning_status=run.route.status, route_mode=mode,
                     points=points, start_pose=pose_record, attempt=run.attempts,
                     profile=self.profile.digest, clearance=clearance,
                     executor_reference=None if not executor else {
                         k: executor.get(k) for k in ('session', 'sequence', 'state', 'geometry_revision',
                                                      'progress', 'stop_generation', 'pose')},
                     evidence={sid: dict(revision=g.revision, time_s=g.sim_time_s)
                               for sid, g in run._grids().items()})
        if mode != 'planned':
            value['fallback_reason'] = (run.route.reason or run.route.status)[:500]
        if substitute is not None: value['substitute_goal'] = list(substitute)
        if self.handoff is not None:
            value['handoff'] = dict(from_revision=self.handoff['from_revision'], start=self.handoff['start'])
        try: validate(value)
        except ValueError as exc:
            # Never leave the previous eligible route retained after oversize failure.
            self.stop_generation += 1
            self.revision += 1
            if moving: self.pending_stop = self.stop_generation
            value.update(points=[], geometry_revision=self.revision, planning_status='unavailable', evidence={},
                         stop_generation=self.stop_generation, executor_reference=None, route_mode='planned',
                         clearance=dict(eligible=False, reason=f'unpublishable route: {exc}'))
            value.pop('fallback_reason', None); value.pop('substitute_goal', None); value.pop('handoff', None)
            self.handoff = None
            self.geometry = None
            validate(value)
        self.last = value
        published = value['route_mode']
        if published != self.mode and getattr(run, 'note', None) is not None:
            run.note('navigation.mode', mode=published, previous=self.mode,
                     reason=value.get('fallback_reason', ''), end=value.get('substitute_goal'))
        self.mode = published
        self.substitute = value.get('substitute_goal')
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
