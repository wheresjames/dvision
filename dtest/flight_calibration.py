"""Measure HOLD stopping on the deterministic dsim rig, for the flight profile.

This is an offline calibration tool, not an operational module: it flies the
real ``DroneSimulator`` physics and command handling through ``DsimLink`` over
the in-process loopback transport, under a controlled clock, and compares what
the vehicle *published* when the stop was decided with where it truly came to
rest. The checked-in dynamic flight profile must cover every number measured
here plus its declared margins; ``tests/test_flight_profile.py`` re-measures
and fails when it does not.

Distances are measured from the *observed* position at the moment the stop was
decided (what an executor actually knows, telemetry latency included) to the
true position once true ground speed stays below the hold threshold for the
dwell. The acknowledgement wait is flown, not skipped: the rig integrates
physics while the link waits for the HOLD result.
"""
from __future__ import annotations

import json
import math
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]

#: The conditions the checked-in profile claims. Nothing outside them is supported.
SPEEDS_MPS = (0.25, 0.5)
LATENCIES_MS = (0.0, 50.0)
WINDS_MPS = (0.0,)
SEEDS = (1234, 4321)
DT_S = 0.05
STREAM_HZ = 10.0
HOLD_SPEED_MPS = 0.05
HOLD_DWELL_S = 0.5


def open_map(path: Path, *, width: int = 30, height: int = 12, start=(3, 6),
             walls=()) -> Path:
    """A walled rectangle with an optional list of interior wall cells."""
    rows = [[' '] * width for _ in range(height)]
    for x in range(width):
        rows[0][x] = rows[height - 1][x] = '0'
    for y in range(height):
        rows[y][0] = rows[y][width - 1] = '0'
    for x, y in walls:
        rows[y][x] = '0'
    sx, sy = start
    rows[sy][sx] = '+'
    path.write_text('--- DATA\ndrone-height=1.5\n\n--- VARS\n+=drone\n*=target\n0=wall\n1=tree\n\n'
                    '--- MAP\n' + '\n'.join(''.join(r) for r in rows) + '\n', encoding='utf-8')
    return path


@contextmanager
def airborne_rig(map_path: Path, report_root: Path, *, start=(3.5, 6.0), heading_deg: float = 90.0,
                 realism: dict | None = None, lease_timeout: float = 3.0):
    """A simulator hovering at 1.5 m in HOLD, a link, and a clock that flies both."""
    import dsim.dsim as dsim_module
    from dtest.dway_rig import Clock, LoopbackTransport, build_sim
    from dway.link import DsimLink

    clock = Clock()
    with mock.patch.object(dsim_module.time, 'monotonic', clock.read):
        sim = build_sim(clock, start=start, heading_deg=heading_deg, map_path=map_path,
                        report_root=report_root, realism=realism, lease_timeout=lease_timeout,
                        setpoint_timeout=2.0)
        sim.state.armed = True
        sim.state.mode = 'HOLD'
        sim.state.home_x, sim.state.home_y, sim.state.home_z = start[0], start[1], 0.0
        transport = LoopbackTransport(sim)

        def sleep(seconds):
            clock.advance(seconds)
            sim.integrate(seconds)

        link = DsimLink('calibration', client_id='calibration', transport=transport,
                        clock=clock.read, sleep=sleep)
        rig = dict(sim=sim, clock=clock, link=link, transport=transport, sleep=sleep)
        yield rig


def _truth_speed(sim):
    from dsim.dsim import DroneState  # noqa: F401  (documents the source of truth)
    wind_x, wind_y = sim.realism.wind_vector()
    st = sim.state
    return math.hypot(st.vx + wind_x, st.vy + wind_y)


def measure_stop(speed_mps: float, *, latency_ms: float = 0.0, wind_mps: float = 0.0,
                 seed: int = 1234, cruise_s: float = 6.0, work_dir: Path) -> dict:
    """Cruise east at ``speed_mps`` with position targets, then HOLD once and measure."""
    work_dir.mkdir(parents=True, exist_ok=True)
    map_path = open_map(work_dir / 'calibration-open.txt', width=40)
    realism = dict(telemetry_latency_ms=latency_ms, wind_mps=wind_mps, realism_seed=seed)
    from dway.link import PositionTarget
    with airborne_rig(map_path, work_dir / 'run', realism=realism) as rig:
        sim, clock, link, sleep = rig['sim'], rig['clock'], rig['link'], rig['sleep']
        assert link.acquire_control().accepted
        y0 = sim.state.y
        target = PositionTarget(frame='map', x=36.0, y=y0, z=1.5, heading_deg=90.0, max_speed_mps=speed_mps)
        next_send = clock.now
        cruise_until = clock.now + cruise_s
        max_cross_track = 0.
        observed_speeds = []
        while clock.now < cruise_until:
            if clock.now + 1e-9 >= next_send:
                result = link.send_position_target(target)
                if not result.accepted: raise RuntimeError(result.reason)
                next_send += 1.0 / STREAM_HZ
                link.heartbeat()
            state = link.state()
            observed_speeds.append(math.hypot(state.vx_mps, state.vy_mps))
            max_cross_track = max(max_cross_track, abs(sim.state.y - y0))
            sleep(DT_S)
        decided = link.state()
        observed_x = float(decided.position.x)
        truth_x_at_decision = sim.state.x
        decided_at = clock.now
        result = link.hold()
        if not result.accepted: raise RuntimeError(result.reason)
        below_since = None
        max_lateral = 0.
        stop_time = None
        while clock.now - decided_at < 10.0:
            max_lateral = max(max_lateral, abs(sim.state.y - y0))
            if _truth_speed(sim) <= HOLD_SPEED_MPS:
                below_since = clock.now if below_since is None else below_since
                if clock.now - below_since >= HOLD_DWELL_S:
                    stop_time = below_since - decided_at
                    break
            else:
                below_since = None
            sleep(DT_S)
        if stop_time is None: raise RuntimeError('vehicle never settled under HOLD')
        settled_x = sim.state.x
        # Drift while held: how far the confirmed stop wanders over a further 5 s.
        drift_from = (sim.state.x, sim.state.y)
        for _ in range(int(5.0 / DT_S)):
            link.heartbeat()
            sleep(DT_S)
        drift = math.dist(drift_from, (sim.state.x, sim.state.y))
        observed_cruise = observed_speeds[len(observed_speeds) // 2:]
        return dict(speed_mps=speed_mps, latency_ms=latency_ms, wind_mps=wind_mps, seed=seed,
                    cruise_speed_mps=round(sum(observed_cruise) / len(observed_cruise), 4),
                    stop_distance_m=round(settled_x - observed_x, 4),
                    stop_distance_from_truth_m=round(settled_x - truth_x_at_decision, 4),
                    observation_lag_m=round(truth_x_at_decision - observed_x, 4),
                    stop_time_s=round(stop_time, 3), lateral_m=round(max_lateral, 4),
                    cruise_cross_track_m=round(max_cross_track, 4),
                    hold_drift_5s_m=round(drift, 4))


def measure_arrival(speed_mps: float, *, latency_ms: float = 0.0, seed: int = 1234, work_dir: Path,
                    distance_m: float = 4.0) -> dict:
    """Command one position target and measure overshoot and time to settle at it."""
    work_dir.mkdir(parents=True, exist_ok=True)
    map_path = open_map(work_dir / 'calibration-open.txt', width=40)
    from dway.link import PositionTarget
    with airborne_rig(map_path, work_dir / 'run', realism=dict(telemetry_latency_ms=latency_ms,
                                                              realism_seed=seed)) as rig:
        sim, clock, link, sleep = rig['sim'], rig['clock'], rig['link'], rig['sleep']
        assert link.acquire_control().accepted
        goal_x = sim.state.x + distance_m
        target = PositionTarget(frame='map', x=goal_x, y=sim.state.y, z=1.5, heading_deg=90.0,
                                max_speed_mps=speed_mps)
        start = clock.now
        next_send = clock.now
        overshoot = 0.
        settled = None
        while clock.now - start < 60.0:
            if clock.now + 1e-9 >= next_send:
                link.send_position_target(target)
                next_send += 1.0 / STREAM_HZ
                link.heartbeat()
            overshoot = max(overshoot, sim.state.x - goal_x)
            if abs(sim.state.x - goal_x) <= 0.05 and _truth_speed(sim) <= HOLD_SPEED_MPS:
                settled = clock.now - start
                break
            sleep(DT_S)
        return dict(speed_mps=speed_mps, latency_ms=latency_ms, seed=seed, distance_m=distance_m,
                    overshoot_m=round(max(0., overshoot), 4),
                    settle_time_s=None if settled is None else round(settled, 3))


def measure_all(work_dir: Path) -> dict:
    stops = [measure_stop(v, latency_ms=lat, wind_mps=w, seed=s, work_dir=work_dir / f'stop-{i}')
             for i, (v, lat, w, s) in enumerate((v, lat, w, s) for v in SPEEDS_MPS for lat in LATENCIES_MS
                                                for w in WINDS_MPS for s in SEEDS)]
    arrivals = [measure_arrival(v, latency_ms=lat, work_dir=work_dir / f'arrive-{i}')
                for i, (v, lat) in enumerate((v, lat) for v in SPEEDS_MPS for lat in LATENCIES_MS)]
    return dict(stops=stops, arrivals=arrivals)


def envelope(measurements: dict) -> dict:
    """The worst case over every measured condition; the profile adds margins to this."""
    stops, arrivals = measurements['stops'], measurements['arrivals']
    return dict(
        stop_distance_m=max(m['stop_distance_m'] for m in stops),
        stop_time_s=max(m['stop_time_s'] for m in stops),
        lateral_m=max(m['lateral_m'] for m in stops),
        cruise_cross_track_m=max(m['cruise_cross_track_m'] for m in stops),
        hold_drift_5s_m=max(m['hold_drift_5s_m'] for m in stops),
        overshoot_m=max(m['overshoot_m'] for m in arrivals),
        settle_time_s=max(m['settle_time_s'] or math.inf for m in arrivals))


if __name__ == '__main__':  # pragma: no cover - manual recalibration aid
    import sys
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        result = measure_all(Path(tmp))
    result['envelope'] = envelope(result)
    json.dump(result, sys.stdout, indent=2)
    print()
