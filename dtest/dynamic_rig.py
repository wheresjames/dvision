"""Deterministic dynamic flights: dsim physics, real transports, a grid planner, dnav's publisher.

The vehicle is a real ``DroneSimulator`` behind ``DsimLink`` on the loopback
transport (as in ``dtest.dway_rig``). Everything between planner and executor is
the production path: the neutral context registry, the evidence map plane, dnav's
``RoutePublisher`` and ``Clearance``, the retained navigation/execution
snapshots, and ``DynamicExecutor``. One virtual clock drives data time, wall
time and physics, so runs are repeatable.

Two things are fixture-only and are declared as such. Evidence is generated from
the fixture's own map (this rig is the provider, which may hold truth), observed
within ``sensor_range_m`` of the vehicle with no occlusion; obstacles added with
:meth:`DynamicRig.block` exist only in the evidence. Planning is a small A* over
that evidence standing in for dnav's planner, so route shapes are predictable.
"""
from __future__ import annotations

import heapq
import math
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import uuid

import numpy as np

from dcmn.context import Context, validate_pose
from dcmn.maps import GridGeometry, MapPublisher, MapSession
from dcmn.navigation import ExecutionProfile, Snapshot
from dnav.execution import RoutePublisher
from dtest.dway_rig import Clock, LoopbackTransport, build_sim
from dtest.flight_calibration import open_map
from dway.link import CommandResult, DsimLink

ROOT = Path(__file__).resolve().parents[1]
FLIGHT_PROFILE = ROOT / 'assets/execution_profiles/dsim-position-v1.json'
DT_S = 0.05
CELL_M = 0.5


class GridPlanner:
    """8-connected A* with inflation, unknown space traversable, corners by line of sight."""

    def __init__(self, radius_m):
        self.radius_m = radius_m

    def plan(self, grid, start, goal, altitude):
        if math.dist(start[:2], goal[:2]) < 1e-6:
            return [[start[0], start[1], altitude]], 'ok'
        g = grid.geometry
        occupancy, _ = grid.layer(0)
        blocked = (occupancy != 255) & (occupancy >= 128)
        r = int(math.ceil(self.radius_m / g.cell_m))
        inflated = blocked.copy()
        for y, x in zip(*np.nonzero(blocked)):
            inflated[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1] = True

        def cell(point):
            return int((point[1] - g.origin_y_m) // g.cell_m), int((point[0] - g.origin_x_m) // g.cell_m)

        def centre(c):
            return (g.origin_x_m + (c[1] + .5) * g.cell_m, g.origin_y_m + (c[0] + .5) * g.cell_m)

        s, t = cell(start), cell(goal)
        inside = lambda c: 0 <= c[0] < g.height and 0 <= c[1] < g.width
        if not inside(s) or not inside(t): return None, 'outside_coverage'
        if inflated[t]: return None, 'goal_unreachable'
        free = lambda c: inside(c) and (not inflated[c] or c == s)
        frontier, came, cost = [(0., s)], {s: None}, {s: 0.}
        while frontier:
            _, c = heapq.heappop(frontier)
            if c == t: break
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if not dy and not dx: continue
                    n = (c[0] + dy, c[1] + dx)
                    if not free(n) or (dy and dx and not (free((c[0] + dy, c[1])) and free((c[0], c[1] + dx)))):
                        continue
                    step = cost[c] + math.hypot(dy, dx)
                    if step < cost.get(n, math.inf):
                        cost[n], came[n] = step, c
                        heapq.heappush(frontier, (step + math.dist(n, t), n))
        if t not in came: return None, 'no_route'
        cells = [t]
        while came[cells[-1]] is not None: cells.append(came[cells[-1]])
        cells.reverse()
        points = [tuple(start[:2])] + [centre(c) for c in cells[1:-1]] + [tuple(goal[:2])]

        def visible(a, b):
            steps = max(1, int(math.dist(a, b) / (g.cell_m / 4)))
            for i in range(steps + 1):
                q = (a[0] + (b[0] - a[0]) * i / steps, a[1] + (b[1] - a[1]) * i / steps)
                c = cell(q)
                if not inside(c) or (inflated[c] and c != s): return False
            return True

        corners, anchor = [points[0]], 0
        while anchor < len(points) - 1:
            nxt = len(points) - 1
            while nxt > anchor + 1 and not visible(points[anchor], points[nxt]): nxt -= 1
            corners.append(points[nxt]); anchor = nxt
        return [[x, y, altitude] for x, y in corners], 'ok'


class PlannerRun:
    """The NavRun surface ``RoutePublisher`` reads, backed by the rig's context and evidence."""

    def __init__(self, rig):
        self.rig = rig
        self.snapshot = {}
        self.goal_descriptor = None
        self.session = SimpleNamespace(identity='dynamic-rig-map', context={'mapping_epoch': 0})
        self.attempts = 0
        self.recorder = None
        self.route = SimpleNamespace(ok=False, status='no_goal', reason='no goal', waypoints=[])
        self._points = None

    def sim_time_s(self): return float(self.snapshot.get('time_s', 0.))
    def _grids(self): return dict(self.rig.grids)

    def _pose(self):
        try: return validate_pose(self.snapshot.get('pose'), self.snapshot)
        except (ValueError, KeyError, TypeError): return None

    def refresh(self):
        self.snapshot = self.rig.context.read() or {}
        self.goal_descriptor = self.snapshot.get('goal')
        pose = self._pose()
        if self.goal_descriptor is None or pose is None:
            self.route = SimpleNamespace(ok=False, status='no_goal' if pose else 'stale_pose',
                                         reason='no goal' if pose else 'pose unavailable', waypoints=[])
            return
        self.attempts += 1
        grid = self.rig.grids[self.rig.primary_source]
        goal = self.goal_descriptor['position']
        start = [pose['x_m'], pose['y_m'], pose['z_m']]
        points, status = self.rig.planner.plan(grid, start, goal, self.rig.profile.altitude_m)
        if self.rig.fixed_route is not None:
            points, status = [list(p) for p in self.rig.fixed_route], 'ok'
        self.route = SimpleNamespace(ok=status == 'ok', status=status, reason='' if status == 'ok' else status,
                                     waypoints=[SimpleNamespace(x_m=p[0], y_m=p[1], z_m=p[2]) for p in points or []])


class DynamicRig:
    def __init__(self, tmp_path, *, width=30, height=12, walls=(), start=(3.5, 6.0), goal=(12.5, 6.0),
                 heading_deg=90.0, sensor_range_m=None, profile=FLIGHT_PROFILE, realism=None,
                 nav_archive=False, sources=('scan',), fixed_route=None, executor_options=None):
        self.tmp = Path(tmp_path)
        self.tmp.mkdir(parents=True, exist_ok=True)
        self.id = 'fly-' + uuid.uuid4().hex[:8]
        self.clock = Clock()
        import dsim.dsim as dsim_module
        self._patch = mock.patch.object(dsim_module.time, 'monotonic', self.clock.read)
        self._patch.start()
        self.profile = ExecutionProfile.load(profile)
        self.map_path = open_map(self.tmp / 'open.txt', width=width, height=height,
                                 start=(int(start[0]), int(start[1])), walls=walls)
        self.sim = build_sim(self.clock, start=start, heading_deg=heading_deg, map_path=self.map_path,
                             report_root=self.tmp / 'run', realism=realism, setpoint_timeout=2.0)
        self.sim.state.armed, self.sim.state.mode = True, 'HOLD'
        self.sim.state.home_x, self.sim.state.home_y, self.sim.state.home_z = start[0], start[1], 0.
        self.transport = LoopbackTransport(self.sim)
        self.link = DsimLink(self.id, client_id=f'dway-{self.id}', transport=self.transport,
                             clock=self.clock.read, sleep=self.sleep)
        self.report_root = self.tmp / 'run'
        self.context = Context(self.id)
        self.context.start(self.report_root, local_ned_origin=(width / 2, height / 2, 0.))
        self.geometry = GridGeometry.from_extent(width, height, CELL_M)
        self.sources = tuple(sources)
        self.primary_source = self.sources[0]
        self.sensor_range_m = sensor_range_m
        self.extra = set()
        #: Cells never observed: occupancy stays the unknown sentinel and no
        #: observation stamp is written, so clearance must end before them
        #: while the planner (unknown space traversable) routes through.
        self.unobserved = set()
        self.occupancy = {s: np.full(self.geometry.shape, 255, np.uint8) for s in self.sources}
        self.observed = {s: np.zeros(self.geometry.shape, np.uint32) for s in self.sources}
        self.map_publisher = MapPublisher(self.id, self.geometry,
                                          [dict(id=s, sensor='scan', algorithm='fixture') for s in self.sources],
                                          context=dict(frame_id='local', localization_epoch=0,
                                                       clock_domain_id=self.id, clock_epoch=0))
        self.active_sources = set(self.sources[:1])
        self.grids = {}
        self.planner = GridPlanner(self.profile.body_radius_m + self.profile.tracking_m + self.profile.stopping_m)
        self.fixed_route = fixed_route
        self.run = PlannerRun(self)
        self.nav_recorder = None
        if nav_archive:
            from dcmn.archive import Recorder
            self.nav_recorder = Recorder(self.report_root / 'dnav' / 'archive', dict(fixture='dynamic_rig'), module='dnav')
            self.run.recorder = self.nav_recorder
        self.publisher = RoutePublisher(self.id, self.profile)
        self.planner_alive = True
        self.provider_frozen = False
        #: Advance to announce a localization reset (a frame discontinuity).
        self.localization_epoch = 0
        self._publish_pose()
        self.context.set_goal('fixture', [goal[0], goal[1], self.profile.altitude_m], role='mission')
        self._publish_evidence()
        self._next_evidence = self.sim.sim_time_s + 1.0
        self._next_plan = self.sim.sim_time_s
        self.maps = MapSession(self.id)
        from dway.executor import DynamicExecutor
        options = dict(clock=self.link.sim_time_s, wall=self.clock.read, sleep=self.sleep, maps=self.maps)
        options.update(executor_options or {})
        self.executor = DynamicExecutor(self.id, self.link, self.profile, **options)
        self.closed = False
        self.step_hooks = []

    # -- driving ---------------------------------------------------------

    def sleep(self, seconds):
        self.clock.advance(seconds)
        self.sim.integrate(seconds)

    def _publish_pose(self):
        from dsim.dsim import sim_yaw_to_compass_heading
        st = self.sim.state
        self.context.publish_pose(dict(x_m=st.x, y_m=st.y, z_m=st.z, heading_deg=sim_yaw_to_compass_heading(st.yaw_deg),
                                       roll_deg=0., pitch_deg=0.), self.sim.sim_time_s,
                                  localization_epoch=self.localization_epoch)

    def _truth_blocked(self, row, col):
        g = self.geometry
        x, y = g.origin_x_m + (col + .5) * g.cell_m, g.origin_y_m + (row + .5) * g.cell_m
        if (row, col) in self.extra: return True
        return any(self.sim.is_blocked(x + dx, y + dy, self.profile.altitude_m)
                   for dx in (-.24, 0., .24) for dy in (-.24, 0., .24))

    def _publish_evidence(self):
        now = self.sim.sim_time_s
        stamp = max(1, int(round(now * 1000)))
        x, y = self.sim.state.x, self.sim.state.y
        g = self.geometry
        for source in self.active_sources:
            occupancy, observed = self.occupancy[source], self.observed[source]
            for row in range(g.height):
                for col in range(g.width):
                    if (row, col) in self.unobserved:
                        occupancy[0, row, col] = 255  # never observed: unknown
                        continue
                    cx, cy = g.origin_x_m + (col + .5) * g.cell_m, g.origin_y_m + (row + .5) * g.cell_m
                    if self.sensor_range_m is not None and math.hypot(cx - x, cy - y) > self.sensor_range_m: continue
                    occupancy[0, row, col] = 254 if self._truth_blocked(row, col) else 0
                    observed[0, row, col] = stamp
            self.map_publisher.publish(source, occupancy, observed, now)
            from dcmn.maps import EvidenceGrid
            self.grids[source] = EvidenceGrid(g, occupancy.copy(), observed.copy(), source,
                                              self.grids[source].revision + 1 if source in self.grids else 1, now)
        for source in list(self.grids):
            if source not in self.active_sources: del self.grids[source]

    def step(self):
        now = self.sim.sim_time_s
        if not self.provider_frozen: self._publish_pose()
        if now + 1e-9 >= self._next_evidence:
            self._publish_evidence()
            self._next_evidence += 1.0
        if self.planner_alive and now + 1e-9 >= self._next_plan:
            self.run.refresh()
            self.publisher.publish(self.run, wall=self.clock.read())
            self._next_plan += 0.1
        for hook in list(self.step_hooks): hook(self)
        self.executor.step()
        self.clock.advance(DT_S)
        self.sim.integrate(DT_S)

    def fly(self, *, limit_s=120.0, until=None):
        steps = int(limit_s / DT_S)
        for _ in range(steps):
            if until is not None and until(self): return True
            if until is None and self.executor.state in ('COMPLETE', 'CANCELLED', 'FAILED'): return True
            self.step()
        return until(self) if until is not None else False

    def fly_to_state(self, *states, limit_s=120.0):
        return self.fly(limit_s=limit_s, until=lambda rig: rig.executor.state in states)

    def start(self, origin='fixture'):
        self.executor.request('start', origin)

    # -- scenario levers ----------------------------------------------------

    def set_goal(self, x, y):
        self.context.set_goal('fixture', [x, y, self.profile.altitude_m], role='mission')

    def block(self, x, y, radius_m=0.5):
        g = self.geometry
        for row in range(g.height):
            for col in range(g.width):
                cx, cy = g.origin_x_m + (col + .5) * g.cell_m, g.origin_y_m + (row + .5) * g.cell_m
                if math.hypot(cx - x, cy - y) <= radius_m: self.extra.add((row, col))
        self._publish_evidence()
        self._next_evidence = self.sim.sim_time_s + 1.0

    def never_observe(self, cells):
        """Mark cells permanently unknown: no observation stamp is written,
        so they can neither support nor veto permission."""
        g = self.geometry
        for row, col in cells:
            if not (0 <= row < g.height and 0 <= col < g.width):
                raise ValueError(f'cell outside the evidence grid: {(row, col)}')
            self.unobserved.add((row, col))
            for source in self.active_sources:
                self.occupancy[source][0, row, col] = 255
                self.observed[source][0, row, col] = 0
        self._publish_evidence()
        self._next_evidence = self.sim.sim_time_s + 1.0

    def add_source(self, source):
        self.active_sources.add(source)
        self._publish_evidence()

    def restart_planner(self):
        self.publisher.close()
        self.publisher = RoutePublisher(self.id, self.profile)

    def takeover(self, client='operator'):
        """Another client takes control the way a manual operator would once the lease lapses."""
        self.sim._clear_lease()
        operator = DsimLink(self.id, client_id=client, transport=self.transport, clock=self.clock.read, sleep=self.sleep)
        result = operator.acquire_control()
        operator.hold()
        return operator, result

    def reject_holds(self):
        self.executor.link.hold = lambda: CommandResult('injected', False, 'injected HOLD rejection')

    # -- checks ----------------------------------------------------------

    def targets_outside_permission(self):
        """Every sent target must lie on the permitted interval it was checked against."""
        bad = []
        for entry in self.executor.sent:
            permitted = entry['permitted']
            target = entry['target']
            if len(permitted) == 1:
                distance = math.dist(target, permitted[0])
            else:
                distance = min(_segment_distance(target, a, b) for a, b in zip(permitted, permitted[1:]))
            if distance > 1e-6: bad.append((entry, distance))
        return bad

    def status(self):
        return Snapshot(self.id, 'execution').read()

    def close(self):
        if self.closed: return
        self.closed = True
        try:
            if not self.executor.closed: self.executor.shutdown('rig close')
        finally:
            self.publisher.close()
            if self.nav_recorder is not None: self.nav_recorder.close()
            self.map_publisher.close()
            self.context.close()
            self._patch.stop()


def _segment_distance(p, a, b):
    ab = [y - x for x, y in zip(a, b)]
    span = sum(c * c for c in ab)
    t = 0. if span == 0 else max(0., min(1., sum((q - x) * c for q, x, c in zip(p, a, ab)) / span))
    return math.dist(p, [x + c * t for x, c in zip(a, ab)])
