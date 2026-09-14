"""The evidence-plane test fixture: a known room, published on the real plane.

A fixture, kept in test tooling rather than in dalg (DV-MAPPING §3). It exists
so that everything downstream of perception -- the
transport, the map pane, the cost policy, the planner, the whole `dnav` window
-- can be built and reviewed before any camera algorithm is known to be
correct. It publishes a *fixed known grid* through the same
:class:`~dcmn.maps.MapPublisher`, at the same simulated-time cadence, with the
same manifest and the same record format a real source uses, so a consumer
cannot tell it apart from one.

**This is a fixture, not a feature.** It has no settings of its own and must
never grow any: the room is hard-coded, and the only things it takes from
outside are the map geometry it is handed and the publish cadence (the
plane's), plus the session context's clock and epochs so a consumer can check
them exactly as it checks a real producer's. Anything that made it configurable would make it look
like an algorithm, and the point of it is to be the one thing in the pipeline
that is known to be right.

What varies is not the world but what has been *looked at*: an observer stands
at a fixed point in the room and sweeps one wedge per publish, so

* never-observed cells are visible, and stay visible where a wall hides them --
  the sentinel a cost map would have blurred away;
* the age plane carries a real gradient rather than one flat number;
* the revision advances on every publish while the sweep keeps discovering or
  re-dating cells, and the simulated timestamp advances whether it does or not.
"""

from __future__ import annotations

if __name__ == '__main__':
    import sys as _sys
    from pathlib import Path as _Path
    for _path in (str(_Path(__file__).resolve().parents[1]),
                  str(_Path(__file__).resolve().parents[1] / 'apps')):
        if _path not in _sys.path: _sys.path.insert(0, _path)

import math
import time

import numpy as np

from dcmn.maps import (DEFAULT_CADENCE_S, DEFAULT_STALENESS_HORIZON_S,
                       EvidenceGrid, GridGeometry, MapPublisher, quantize, stamp_ms)
from dcmn.module_bus import PymembusModuleBus, requests_shutdown
from dcmn.pacing import PeriodicDeadline

#: The one source this fixture publishes. Named for what it is, so nothing
#: reading a manifest can mistake it for a sensor's belief.
SOURCE_ID = 'synthetic'

#: The room, in fractions of the extent, so it scales with whatever geometry a
#: profile declares and still describes the same shape: an outer wall, a
#: dividing wall down the middle, and one doorway through it. A planner that
#: cannot find the doorway has a bug; a planner that goes through it does not.
WALL_M = 0.5
DIVIDER_X = 0.5
DOORWAY_Y = 0.5
DOORWAY_M = 2.0
OBSERVER_XY = (0.25, 0.5)

#: Occupancy the fixture publishes. Not 0 and 1: evidence is a belief, and a
#: consumer that only works against certainty is a consumer with a bug waiting
#: for the first real sensor.
FREE_P = 0.05
OCCUPIED_P = 0.95

#: One publish sweeps this much of the turn, so a full revolution takes twelve
#: publishes -- twelve simulated seconds at the default cadence.
SWEEP_DEG = 30.0
RAY_STEP_DEG = 0.5
MAX_RANGE_M = 60.0


def room_occupancy(geometry: GridGeometry) -> np.ndarray:
    """The fixed room, as a boolean ``(height, width)`` obstacle mask."""
    width_m, height_m = geometry.extent_m
    xs = (np.arange(geometry.width) + .5) * geometry.cell_m
    ys = (np.arange(geometry.height) + .5) * geometry.cell_m
    x, y = np.meshgrid(xs, ys)
    border = ((x < WALL_M) | (x > width_m - WALL_M) |
              (y < WALL_M) | (y > height_m - WALL_M))
    divider_x = width_m * DIVIDER_X
    doorway_y = height_m * DOORWAY_Y
    divider = (np.abs(x - divider_x) < WALL_M / 2 + geometry.cell_m / 2)
    doorway = np.abs(y - doorway_y) < DOORWAY_M / 2
    return border | (divider & ~doorway)


def observer_m(geometry: GridGeometry) -> tuple[float, float]:
    """Where the fixture's observer stands, in map metres."""
    width_m, height_m = geometry.extent_m
    return (geometry.origin_x_m + width_m * OBSERVER_XY[0],
            geometry.origin_y_m + height_m * OBSERVER_XY[1])


class SyntheticRoom:
    """The fixed room plus which of it has been observed, and when.

    Separate from the publisher and from any bus so a test can step it without
    shared memory: this is the part that has to be *known* correct.
    """

    def __init__(self, geometry: GridGeometry) -> None:
        self.geometry = geometry
        self.obstacles = room_occupancy(geometry)
        self.occupancy = np.full(geometry.shape, 255, np.uint8)
        self.observed_ms = np.zeros(geometry.shape, np.uint32)
        self.bearing_deg = 0.0
        self.sweeps = 0

    def sweep(self, sim_time_s: float) -> None:
        """Observe one wedge from the fixed point, and date what it reaches.

        A ray marches out in half-cell steps and stops at the first obstacle it
        meets, marking that obstacle occupied: an observer sees a wall, and sees
        nothing behind it. That shadow is what keeps never-observed cells on the
        far side of the divider never-observed, which is the property the map
        pane and the cost policy both have to handle.
        """
        geometry = self.geometry
        ox, oy = observer_m(geometry)
        stamp = int(stamp_ms(sim_time_s))
        step = geometry.cell_m / 2.
        steps = int(min(MAX_RANGE_M, math.hypot(*geometry.extent_m)) / step) + 1
        bearings = np.radians(np.arange(self.bearing_deg,
                                        self.bearing_deg + SWEEP_DEG, RAY_STEP_DEG))
        distances = np.arange(1, steps) * step
        xs = ox + np.cos(bearings)[:, None] * distances[None, :]
        ys = oy + np.sin(bearings)[:, None] * distances[None, :]
        cols = np.floor((xs - geometry.origin_x_m) / geometry.cell_m).astype(int)
        rows = np.floor((ys - geometry.origin_y_m) / geometry.cell_m).astype(int)
        inside = ((cols >= 0) & (cols < geometry.width) &
                  (rows >= 0) & (rows < geometry.height))
        blocked = np.zeros(len(bearings), bool)
        for index in range(distances.size):
            live = inside[:, index] & ~blocked
            if not live.any():
                if blocked.all(): break
                continue
            col, row = cols[live, index], rows[live, index]
            hit = self.obstacles[row, col]
            self.occupancy[0, row, col] = quantize(np.where(hit, OCCUPIED_P, FREE_P))
            self.observed_ms[0, row, col] = stamp
            blocked[np.flatnonzero(live)[hit]] = True
        self.bearing_deg = (self.bearing_deg + SWEEP_DEG) % 360.
        self.sweeps += 1

    def grid(self) -> EvidenceGrid:
        return EvidenceGrid(self.geometry, self.occupancy, self.observed_ms, SOURCE_ID)


class SyntheticProducer:
    """The fixture as a process: own the maps plane, publish on the context's clock.

    It holds no control lease, answers no run lifecycle and writes no report.
    It is a producer and nothing else. It labels its generation with the
    session context's frame and epochs, and starts a new generation when they
    change, exactly as dalg does.
    """

    def __init__(self, instance: str, geometry: GridGeometry, *,
                 cadence_s: float = DEFAULT_CADENCE_S,
                 staleness_horizon_s: float = DEFAULT_STALENESS_HORIZON_S,
                 profile_name: str = 'synthetic-fixture', profile_digest: str = '') -> None:
        from dcmn.context import Context
        self.id = instance
        self.geometry = geometry
        self.cadence_s = float(cadence_s)
        self.staleness_horizon_s = float(staleness_horizon_s)
        self.profile_name, self.profile_digest = profile_name, profile_digest
        self.room = SyntheticRoom(geometry)
        self.context = Context(instance)
        self.snapshot = {}
        self.publisher = None
        self.epoch = None
        self.generation = 0
        self.published = 0
        self.revision = 0
        self.shutdown_requested = False
        self.waiting_reported = False
        self.bus = PymembusModuleBus(instance, 'algorithm', 'dalg-synthetic',
                                     sim_time=self.sim_time_s)
        self._cadence = PeriodicDeadline(1. / self.cadence_s)
        self._last_heartbeat = -1e9
        self._hello_sent = False

    def sim_time_s(self) -> float:
        return float(self.snapshot.get('time_s', 0.))

    def connected(self) -> bool:
        """Whether a session context -- and so a clock and frame -- is available."""
        try: self.snapshot = self.context.read()
        except ValueError: self.snapshot = {}
        return bool(self.snapshot)

    def poll_delay(self) -> float:
        return .02

    def _generation(self) -> None:
        epoch = tuple(self.snapshot.get(k) for k in
                      ('frame_id', 'localization_epoch', 'clock_domain_id', 'clock_epoch'))
        if epoch == self.epoch and self.publisher is not None: return
        if self.publisher is not None: self.publisher.close()
        self.generation += 1
        self.epoch = epoch
        self.room = SyntheticRoom(self.geometry)
        self.publisher = MapPublisher(
            self.id, self.geometry,
            [dict(id=SOURCE_ID, sensor=SOURCE_ID, sensor_type='fixture', algorithm='synthetic')],
            producer='dalg-synthetic', profile_name=self.profile_name,
            profile_digest=self.profile_digest, cadence_s=self.cadence_s,
            staleness_horizon_s=self.staleness_horizon_s, generation=self.generation,
            context=dict(frame_id=epoch[0], localization_epoch=epoch[1],
                         clock_domain_id=epoch[2], clock_epoch=epoch[3]))
        self._cadence = PeriodicDeadline(1. / self.cadence_s)

    def _presence(self, wall_now: float) -> None:
        payload = dict(state='PUBLISHING' if self.published else 'WAITING',
                       ready=True, capabilities={'maps': [SOURCE_ID],
                                                 'algorithms': [], 'sensors': []})
        if not self._hello_sent:
            self.bus.publish('module.hello', payload=payload)
            self._hello_sent = True
        if wall_now - self._last_heartbeat < 1.: return
        self._last_heartbeat = wall_now
        self.bus.publish('module.heartbeat', payload=dict(
            payload, revision=self.revision, records=self.published,
            sim_time_s=self.sim_time_s()))

    def step(self) -> bool:
        """One iteration. Returns False once the process should stop."""
        self.bus.connect()
        self._presence(time.monotonic())
        for event in self.bus.receive():
            if requests_shutdown(event):
                self.shutdown_requested = True
                return False
        if not self.connected():
            if not self.waiting_reported:
                self.waiting_reported = True
                print(f'synthetic: waiting for the session context of {self.id}; nothing is '
                      'published without a clock', flush=True)
            return True
        self._generation()
        now = self.sim_time_s()
        if not self._cadence.due(now): return True
        self._cadence.advance(now)
        self.room.sweep(now)
        self.revision = self.publisher.publish(
            SOURCE_ID, self.room.occupancy, self.room.observed_ms, now)
        self.published += 1
        return True

    def close(self) -> None:
        self.bus.publish('module.goodbye', payload={'state': 'STOPPED'})
        self.bus.close()
        if self.publisher is not None: self.publisher.close()


def main(argv=None) -> int:
    """Publish the fixture room on an instance until shutdown or timeout."""
    import argparse
    import sys
    parser = argparse.ArgumentParser(description='synthetic evidence-plane fixture')
    parser.add_argument('--id', required=True)
    parser.add_argument('--bounds', default='0,0,20,20', help='xmin,ymin,xmax,ymax in local metres')
    parser.add_argument('--cell-m', type=float, default=.5)
    parser.add_argument('--timeout', type=float, default=180.)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    from dcmn.context import parse_bounds
    x0, y0, x1, y1 = parse_bounds(args.bounds)
    producer = SyntheticProducer(args.id, GridGeometry.from_extent(
        x1-x0, y1-y0, args.cell_m, origin_x_m=x0, origin_y_m=y0))
    deadline = time.monotonic() + args.timeout
    try:
        while producer.step() and time.monotonic() < deadline:
            time.sleep(producer.poll_delay())
    except KeyboardInterrupt:
        pass
    finally:
        producer.close()
    print(f'synthetic: published {producer.published} records, revision {producer.revision}',
          file=sys.stderr)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
