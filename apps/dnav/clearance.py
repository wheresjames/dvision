"""Permission to execute a planned route, separate from unknown-space planning.

Two permissions, selected by the execution profile's ``permission``:

``evidence``
    Conservative observed-space admission. Small swept rectangles deliberately
    over-cover each subsegment, so they cannot miss a touched cell or cut a
    diagonal corner; every swept cell needs fresh free evidence.
``plan``
    Trust the planner's route through unknown space (a simulation research
    assumption). The route is withdrawn only where currently observed occupied
    evidence lies within ``plan_clearance_m`` of the remaining route, so the executor
    stops and dnav replans from the stop. The ``evidence`` verdict is still
    computed and published alongside as ``evidence_check``, so the two
    policies can be compared on the same flight.
"""
import math

import numpy as np

from dcmn.navigation import ExecutionProfile  # noqa: F401  (shared with dway)


class Clearance:
    def __init__(self, profile):
        self.profile = profile
        self.identity = None
        self.veto = {}

    def check(self, points, pose, grids, now, identity=None, start=(0, 0.), permission=None):
        p = self.profile
        permission = permission or p.permission
        if permission == 'plan':
            return self._plan(points, pose, grids, now, identity, start)
        result = dict(eligible=False, reason='no route', start=[0, 0.], end=[0, 0.],
                      distance_m=0., reaches_goal=False, valid_until_s=now, speed_mps=p.speed_mps, tracking_m=p.tracking_m,
                      calibrated=p.calibrated, altitude_m=p.altitude_m, stopping_m=p.stopping_m)
        if not points or pose is None: return result
        if (len(start) != 2 or type(start[0]) is not int or not 0 <= start[0] < max(1,len(points)-1)
                or not math.isfinite(start[1]) or not 0 <= start[1] <= 1):
            result['reason'] = 'invalid executor progress'; return result
        offset, fraction = start
        if len(points) > 1:
            a,b = points[offset:offset+2]
            points = [[x+(y-x)*fraction for x,y in zip(a,b)]] + points[offset+1:]
        result['start'] = list(start); result['end'] = list(start)
        if p.slab_assumption == 'none':
            result['reason'] = 'no declared full-height slab assumption'; return result
        if not grids or set(p.required_sources) - grids.keys():
            result['reason'] = 'required evidence unavailable'; return result
        if len(points) > 256:
            result['reason'] = 'route exceeds 256 points'; return result
        first = next(iter(grids.values()))
        geom = first.geometry
        if geom.layers != 1:
            result['reason'] = 'only a single evidence slab is supported'; return result
        # Source membership is not part of the reset: a source that drops out has
        # not supplied fresh free evidence, so its vetoes keep blocking. The
        # publisher separately invalidates permission when membership changes.
        generation = (identity, geom)
        if generation != self.identity:
            self.veto.clear(); self.identity = generation
        if any(g.geometry != geom for g in grids.values()):
            result['reason'] = 'mixed evidence geometry'; return result
        if any(abs(pt[2] - p.altitude_m) > 1e-6 for pt in points) or abs(pose[2]-p.altitude_m) > 1e-6:
            result['reason'] = 'route/pose outside fixed altitude'; return result
        if not geom.z0_m <= p.altitude_m-p.half_height_m or p.altitude_m+p.half_height_m >= geom.z0_m+geom.dz_m:
            result['reason'] = 'evidence slab does not cover body'; return result
        if math.dist(pose, points[0]) > p.join_m:
            result['reason'] = 'start pose moved beyond join allowance'; return result
        free = np.zeros((geom.height, geom.width), bool)
        expiry = np.full(free.shape, -np.inf)
        blocked = np.zeros(free.shape, bool)
        for sid, grid in grids.items():
            occ, observed = grid.layer(0)
            age = now - observed.astype(float)/1000.
            fresh = (observed != 0) & (age >= -0.0011) & (age <= p.max_age_s)
            qualifies = (occ <= p.free_threshold*254) & fresh
            veto = self.veto.setdefault(sid, np.zeros(free.shape, bool))
            veto[(occ != 255) & (occ >= p.occupied_threshold*254)] = True
            veto[qualifies] = False
            free |= qualifies
            expiry = np.maximum(expiry, np.where(qualifies, observed/1000.+p.max_age_s, -np.inf))
        for veto in self.veto.values(): blocked |= veto
        usable = free & ~blocked
        # Cover body, tracking and a conservative omnidirectional stopping area.
        radius = p.body_radius_m + p.tracking_m + p.stopping_m
        deadline = float('inf')
        def admit(a, b):
            nonlocal deadline
            x0, y0 = min(a[0], b[0])-radius, min(a[1], b[1])-radius
            x1, y1 = max(a[0], b[0])+radius, max(a[1], b[1])+radius
            # Include cells touched exactly on a boundary, on both sides.
            c0 = math.floor((x0-geom.origin_x_m)/geom.cell_m-1e-9)
            r0 = math.floor((y0-geom.origin_y_m)/geom.cell_m-1e-9)
            c1 = math.floor((x1-geom.origin_x_m)/geom.cell_m)
            r1 = math.floor((y1-geom.origin_y_m)/geom.cell_m)
            if c0 < 0 or r0 < 0 or c1 >= geom.width or r1 >= geom.height: return 'outside observed coverage'
            area = np.s_[r0:r1+1, c0:c1+1]
            if not usable[area].all(): return 'unknown, stale, ambiguous or occupied swept cells'
            until = float(expiry[area].min())
            if until-now <= p.stopping_s: return 'insufficient observation lifetime to stop'
            deadline = min(deadline, until)
            return ''
        reason = admit(pose, points[0])
        if reason: result['reason'] = reason; return result
        distance = 0.
        result['eligible'] = True
        work = 0
        for i, (a, b) in enumerate(zip(points, points[1:])):
            length = math.dist(a, b)
            steps = max(1, math.ceil(length / (geom.cell_m/2)))
            work += steps
            if work > 10000:
                result['reason'] = 'clearance work budget exceeded'; result['eligible'] = False; return result
            prev = a
            for j in range(1, steps+1):
                pt = [x+(y-x)*j/steps for x,y in zip(a,b)]
                reason = admit(prev, pt)
                if reason: break
                distance += math.dist(prev, pt)
                # Clamp the fraction: ``f + (1-f)*j/steps`` can land a hair above
                # 1.0 in floating point and an interval that leaves the route is
                # rejected by the snapshot schema.
                result['end'] = [offset+i, min(1., fraction+(1-fraction)*j/steps if i == 0 else j/steps)]
                prev = pt
            if reason: break
        result.update(distance_m=distance, valid_until_s=deadline,
                      reaches_goal=not bool(reason), reason=reason or (
                          'observed interval (calibrated profile)' if p.calibrated
                          else 'observed interval (synthetic dry-run profile)'))
        # A zero-distance prefix is useful only for an arrival-only goal.
        if distance == 0 and len(points) > 1: result['eligible'] = False
        return result

    def _plan(self, points, pose, grids, now, identity, start):
        """Trust the planned route; withdraw it only where observed obstacles block it."""
        p = self.profile
        evidence = self.check(points, pose, grids, now, identity, start, permission='evidence')
        result = dict(eligible=False, reason='no route', start=[0, 0.], end=[0, 0.], distance_m=0.,
                      reaches_goal=False, valid_until_s=now, speed_mps=p.speed_mps, tracking_m=p.tracking_m,
                      calibrated=p.calibrated, altitude_m=p.altitude_m, stopping_m=p.stopping_m,
                      permission='plan', evidence_check=dict(
                          eligible=evidence['eligible'], reaches_goal=evidence['reaches_goal'],
                          reason=evidence['reason'], distance_m=evidence['distance_m'], end=evidence['end']))
        if not points or pose is None: return result
        if (len(start) != 2 or type(start[0]) is not int or not 0 <= start[0] < max(1, len(points)-1)
                or not math.isfinite(start[1]) or not 0 <= start[1] <= 1):
            result['reason'] = 'invalid executor progress'; return result
        if len(points) > 256:
            result['reason'] = 'route exceeds 256 points'; return result
        count, (offset, fraction) = len(points), start
        remaining = points
        if count > 1:
            a, b = points[offset:offset+2]
            remaining = [[x+(y-x)*fraction for x, y in zip(a, b)]] + points[offset+1:]
        result['start'] = list(start)
        end = [count-2, 1.] if count > 1 else [0, 0.]
        if count > 1 and tuple(start) >= tuple(end):
            result.update(end=list(start), reason='executor is at the end of the planned route'); return result
        # Only what is observed occupied now -- the same evidence the planner
        # prices. The evidence check's persistent vetoes are deliberately left
        # out: seen live, a remembered optical-flow veto the planner no longer
        # saw sat on every fresh route, so plan permission withdrew each one and
        # the vehicle held forever.
        blocked, geom = None, None
        if grids:
            geom = next(iter(grids.values())).geometry
            if all(g.geometry == geom for g in grids.values()) and geom.layers == 1:
                blocked = np.zeros((geom.height, geom.width), bool)
                for grid in grids.values():
                    occ, _ = grid.layer(0)
                    blocked |= (occ != 255) & (occ >= p.occupied_threshold*254)
        radius = max(p.body_radius_m, p.plan_clearance_m)
        if blocked is not None:
            # Obstacles already around the vehicle cannot be avoided by stopping
            # where it stands; only what the route runs into ahead withdraws it.
            # Without this a vehicle that stopped near a newly seen wall could
            # never be given the route that leads it away.
            near = radius + geom.cell_m
            c0 = max(0, math.floor((pose[0]-near-geom.origin_x_m)/geom.cell_m))
            r0 = max(0, math.floor((pose[1]-near-geom.origin_y_m)/geom.cell_m))
            c1 = min(geom.width-1, math.floor((pose[0]+near-geom.origin_x_m)/geom.cell_m))
            r1 = min(geom.height-1, math.floor((pose[1]+near-geom.origin_y_m)/geom.cell_m))
            if c1 >= c0 and r1 >= r0:
                blocked = blocked.copy(); blocked[r0:r1+1, c0:c1+1] = False
        def hit(a, b):
            # A blocked cell counts when its centre lies within ``radius`` of the
            # segment -- the same centre-to-centre rule the planner's inflation
            # uses, so a route the planner kept clear is not withdrawn by
            # bounding-box over-coverage.
            if blocked is None: return False
            c0 = max(0, math.floor((min(a[0], b[0])-radius-geom.origin_x_m)/geom.cell_m-1e-9))
            r0 = max(0, math.floor((min(a[1], b[1])-radius-geom.origin_y_m)/geom.cell_m-1e-9))
            c1 = min(geom.width-1, math.floor((max(a[0], b[0])+radius-geom.origin_x_m)/geom.cell_m))
            r1 = min(geom.height-1, math.floor((max(a[1], b[1])+radius-geom.origin_y_m)/geom.cell_m))
            if c1 < c0 or r1 < r0: return False  # outside the evidence: unknown, trusted
            rows, cols = np.nonzero(blocked[r0:r1+1, c0:c1+1])
            if not len(rows): return False
            cx = geom.origin_x_m + (cols + c0 + .5) * geom.cell_m
            cy = geom.origin_y_m + (rows + r0 + .5) * geom.cell_m
            dx, dy = b[0]-a[0], b[1]-a[1]
            span = dx*dx + dy*dy
            t = np.zeros(len(cx)) if span == 0 else np.clip(((cx-a[0])*dx + (cy-a[1])*dy) / span, 0., 1.)
            return bool((np.hypot(cx-(a[0]+t*dx), cy-(a[1]+t*dy)) <= radius).any())
        distance, travelled, work = 0., 0., 0
        for a, b in zip(remaining, remaining[1:]):
            length = math.dist(a, b)
            steps = max(1, math.ceil(length / ((geom.cell_m if geom else 0.5)/2)))
            work += steps
            if work > 10000:
                result['reason'] = 'clearance work budget exceeded'; return result
            prev = a
            for j in range(1, steps+1):
                pt = [x+(y-x)*j/steps for x, y in zip(a, b)]
                if hit(prev, pt):
                    result['reason'] = (f'planned route blocked by an observed obstacle {travelled:.1f} m ahead; '
                                        'stop and replan')
                    result['blocked_at_m'] = travelled
                    return result
                travelled += math.dist(prev, pt)
                prev = pt
            distance += length
        result.update(eligible=True, end=end, distance_m=distance, reaches_goal=True, valid_until_s=now+p.max_age_s,
                      reason='planned route trusted (plan permission: unknown space allowed, observed obstacles '
                             'withdraw it)' + ('' if grids else '; no evidence available'))
        return result
