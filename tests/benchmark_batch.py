#!/usr/bin/env python3
"""Run N independent benchmark flights in parallel and aggregate them.

A single flight (or even N=4) isn't enough to tell "this change helped" apart
from "this run landed in the lucky tail of the distribution" — `maze_001` @
90 s alone has produced everything from a clean 17 s landing to FAILSAFEs 6 m
short on identical code. Comparing distributions
means running each configuration N times, which is the slow part: each flight
burns its own ~90 s of sim time plus startup/shutdown overhead. But each
dsim+daic pair is CPU-light (one camera-render loop and one perception/control
loop), so on a multi-core box several can fly at once without skewing timings
enough to matter for these pass/fail-style outcomes.

This script launches N flights of one configuration under
`reports/benchmarks/<name>/run1 .. runN/` (keeping all of a group's runs and
its combined verdict in one place rather than scattered across the top-level
`reports/benchmarks/` directory), runs them with bounded parallelism, and then
calls `benchmark_aggregate` on the resulting set, writing `aggregate.txt`
alongside the runs.

Each parallel flight needs its own dsim/daic shared-memory namespace (`--id`)
or they collide — see `tests/flight_test.py --id`.

Usage
-----
    # 5 runs of maze_001 @ 90s, up to 4 flying at once:
    python3 tests/benchmark_batch.py --name maze001-baseline \\
        --map assets/maps/maze_001.txt --duration 90 --runs 5 --parallel 4

Output
------
    reports/benchmarks/<name>/run1/ .. runN/   -- independent flight_test.py reports
    reports/benchmarks/<name>/aggregate.txt    -- combined verdict (benchmark_aggregate)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.benchmark_aggregate import aggregate, print_aggregate


def _run_one(run_dir: Path, *, map_file: str, duration: int, fps: int,
             instance_id: str) -> tuple[str, int]:
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(ROOT / "tests" / "flight_test.py"),
        "--map", map_file,
        "--duration", str(duration),
        "--fps", str(fps),
        "--report-dir", str(run_dir),
        "--id", instance_id,
    ]
    log_path = run_dir / "batch_run.log"
    with open(log_path, "w", encoding="utf-8") as fh:
        proc = subprocess.run(cmd, cwd=str(ROOT), stdout=fh,
                              stderr=subprocess.STDOUT, text=True, check=False)
    return run_dir.name, proc.returncode


def run_batch(*, name: str, map_file: str, duration: int, fps: int,
              runs: int, parallel: int) -> Path:
    group_dir = ROOT / "reports" / "benchmarks" / name
    run_dirs = [group_dir / f"run{i}" for i in range(1, runs + 1)]

    print(f"Launching {runs} runs of {map_file} @ {duration}s "
          f"(up to {parallel} concurrent) -> {group_dir}", file=sys.stderr)
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=max(1, parallel)) as pool:
        futures = {
            pool.submit(
                _run_one, run_dir,
                map_file=map_file, duration=duration, fps=fps,
                instance_id=f"bench-{name}-{run_dir.name}",
            ): run_dir
            for run_dir in run_dirs
        }
        for fut in as_completed(futures):
            run_dir = futures[fut]
            run_name, returncode = fut.result()
            status = "ok" if returncode == 0 else f"exit={returncode}"
            elapsed = time.monotonic() - t0
            print(f"  [{run_name}] finished ({status}) at {elapsed:.0f}s", file=sys.stderr)

    print(f"All runs finished in {time.monotonic() - t0:.0f}s total", file=sys.stderr)

    agg = aggregate(run_dirs)
    print_aggregate(agg)
    out_path = group_dir / "aggregate.txt"
    with open(out_path, "w", encoding="utf-8") as fh:
        print_aggregate(agg, file=fh)
    print(f"\nWrote aggregate to {out_path}", file=sys.stderr)
    return out_path


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True,
                        help="benchmark group name; runs are written to "
                             "reports/benchmarks/<name>/run1 .. runN")
    parser.add_argument("--map", default="assets/maps/maze_001.txt",
                        help="map file (relative to project root)")
    parser.add_argument("--duration", type=int, default=90,
                        help="maximum flight time in seconds, per run")
    parser.add_argument("--fps", type=int, default=30,
                        help="simulation fps (affects --frames budget)")
    parser.add_argument("--runs", type=int, default=5,
                        help="number of independent runs (3-5 is usually enough "
                             "to see the shape of the distribution)")
    parser.add_argument("--parallel", type=int, default=4,
                        help="max number of flights to run at once -- each "
                             "dsim+daic pair is CPU-light, so this can usually "
                             "be pushed higher on a multi-core box")
    args = parser.parse_args(argv[1:])

    map_path = ROOT / args.map
    if not map_path.exists():
        print(f"error: map not found: {map_path}", file=sys.stderr)
        return 1

    out_path = run_batch(name=args.name, map_file=args.map, duration=args.duration,
                         fps=args.fps, runs=args.runs, parallel=args.parallel)
    return 0 if out_path.exists() else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
