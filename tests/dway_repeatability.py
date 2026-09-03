#!/usr/bin/env python3
"""Fly one baseline tour N times and publish the spread.

`DV-DWAY.md` D7 asks how repeatable closed-loop following actually is, because
`dalg`'s premise -- that a tour is a predictable stimulus -- depends on the
answer. This is the measurement: the same tour, the same map, realism off,
flown end to end through real `dsim` and `dway` processes, aggregated into one
`repeatability.json`.

Real processes, not the deterministic rig: a fixed-timestep in-process flight
would answer a question nobody asked, since it is repeatable by construction.
What varies here is scheduling, timing jitter and transport latency, which is
what a run on this machine actually experiences.

Usage
-----
    python3 tests/dway_repeatability.py                 # 5 runs, default tour
    python3 tests/dway_repeatability.py --runs 10
    python3 tests/dway_repeatability.py --tour assets/tours/maze_012.forward.v1.json \
        --map assets/maps/maze_012.txt

Output
------
    reports/dway-repeatability/<name>/run1 .. runN/dway/summary.json
    reports/dway-repeatability/<name>/repeatability.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dtest.dway_rig import MAZE_012, write_corridor_tour  # noqa: E402
from dway.report import write_repeatability  # noqa: E402

SETPOINT_TIMEOUT_S = 2.0


def fly_once(instance_id: str, tour: Path, map_path: Path, run_dir: Path,
             *, flight_timeout_s: float) -> dict | None:
    """One dsim plus one dway, started and stopped, realism left at defaults."""
    run_dir.mkdir(parents=True, exist_ok=True)
    logs = (run_dir / "dsim.log").open("w", encoding="utf-8")
    dway_logs = (run_dir / "dway.log").open("w", encoding="utf-8")
    sim = dway = None
    try:
        sim = subprocess.Popen(
            [sys.executable, str(ROOT / "dsim/dsim.py"), "--id", instance_id,
             "--map", str(map_path), "--no-ui",
             "--report-dir", str(run_dir),
             "--setpoint-timeout", str(SETPOINT_TIMEOUT_S)],
            cwd=str(ROOT), stdout=logs, stderr=subprocess.STDOUT,
            start_new_session=True)
        # The simulator has to own its buffers before a client can find them.
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline and not (run_dir / "dsim").exists():
            if sim.poll() is not None:
                return None
            time.sleep(0.2)

        dway = subprocess.Popen(
            [sys.executable, str(ROOT / "dway/dway.py"), "--id", instance_id,
             "--tour", str(tour), "--no-ui", "--finish-action", "land",
             "--timeout", str(flight_timeout_s)],
            cwd=str(ROOT), stdout=dway_logs, stderr=subprocess.STDOUT,
            start_new_session=True)
        dway.wait(timeout=flight_timeout_s + 60.0)
    except subprocess.TimeoutExpired:
        return None
    finally:
        for process in (dway, sim):
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10.0)
                except subprocess.TimeoutExpired:
                    process.kill()
        logs.close()
        dway_logs.close()

    summary = run_dir / "dway" / "summary.json"
    return json.loads(summary.read_text()) if summary.exists() else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--name", default="baseline")
    parser.add_argument("--tour", default=None,
                        help="tour to fly (default: the maze_012 corridor tour)")
    parser.add_argument("--map", default=str(MAZE_012))
    parser.add_argument("--flight-timeout", type=float, default=90.0)
    args = parser.parse_args(argv)
    if args.runs < 2:
        parser.error("repeatability needs at least two runs")

    out = ROOT / "reports" / "dway-repeatability" / args.name
    out.mkdir(parents=True, exist_ok=True)
    tour = (Path(args.tour) if args.tour
            else write_corridor_tour(out / "baseline.tour.json",
                                     tour_id=f"repeatability.{args.name}"))

    summaries: list[dict] = []
    for index in range(1, args.runs + 1):
        instance_id = f"dwayrep{os.getpid()}x{index}"
        print(f"run {index}/{args.runs} ({instance_id}) ...", flush=True)
        summary = fly_once(instance_id, tour, Path(args.map),
                           out / f"run{index}",
                           flight_timeout_s=args.flight_timeout)
        if summary is None:
            print(f"  run {index} produced no summary", file=sys.stderr)
            continue
        summaries.append(summary)
        arrivals = [entry["arrival_s"] for entry in summary["waypoints"]]
        print(f"  {summary['outcome']}  path {summary['path_length_m']:.3f} m"
              f"  arrivals {arrivals}", flush=True)

    if not summaries:
        print("no run produced a summary", file=sys.stderr)
        return 1
    path = write_repeatability(out, summaries)
    aggregate = json.loads(path.read_text())
    print(f"\n{path}")
    print(f"  runs {aggregate['runs']}, complete {aggregate['complete_runs']}")
    print(f"  path length mean {aggregate['path_length_mean_m']:.3f} m, "
          f"variance {aggregate['path_length_variance_m2']:.6f} m^2")
    for entry in aggregate["arrival"]:
        print(f"  waypoint {entry['index']}: arrival mean "
              f"{entry['mean_s']:.3f} s, variance {entry['variance_s2']:.6f} s^2")
    return 0 if aggregate["complete_runs"] == aggregate["runs"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
