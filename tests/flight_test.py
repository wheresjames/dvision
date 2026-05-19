#!/usr/bin/env python3
"""Automated flight test runner for daic.

Launches dsim (headless) and daic (headless, AI enabled), waits for the
mission to complete or time out, then prints a structured report.

Usage
-----
    python3 tests/flight_test.py [options]

    # Quick run with defaults (maze_001, 90 s budget):
    python3 tests/flight_test.py

    # Custom map and duration:
    python3 tests/flight_test.py --map dsim/assets/maps/maze_002.txt --duration 120

    # Keep the log file for later inspection:
    python3 tests/flight_test.py --log /tmp/flight.jsonl

Exit codes
----------
    0  drone reached COMPLETE (landed on target)
    1  mission timed out, failed, or errored
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from daic.flight_log import analyze_log, print_report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wait_for_file(path: Path, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists() and path.stat().st_size > 0:
            return True
        time.sleep(0.2)
    return False


def run_test(map_file: str, duration_s: int, log_path: Path,
             fps: int, verbose: bool) -> dict:
    frames = duration_s * fps
    dsim_cmd = [
        sys.executable, str(ROOT / "dsim" / "dsim.py"),
        "--id",    "flighttest",
        "--map",   map_file,
        "--no-ui",
        "--fps",   str(fps),
        "--frames", str(frames),
    ]
    daic_cmd = [
        sys.executable, str(ROOT / "daic" / "daic.py"),
        "--id",        "flighttest",
        "--no-ui",
        "--enable-ai",
        "--log-file",  str(log_path),
        "--fps",       str(fps),
    ]
    if verbose:
        daic_cmd.append("--verbose")

    if verbose:
        print(f"dsim: {' '.join(dsim_cmd)}", file=sys.stderr)
        print(f"daic: {' '.join(daic_cmd)}", file=sys.stderr)

    # Launch dsim first so shared memory exists before daic connects.
    dsim_proc = subprocess.Popen(
        dsim_cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE if not verbose else None,
        cwd=str(ROOT),
    )

    # Give dsim a moment to create the shared-memory buffers.
    time.sleep(1.5)

    daic_proc = subprocess.Popen(
        daic_cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE if not verbose else None,
        cwd=str(ROOT),
    )

    # Wait for the log file to appear (daic connected and started logging).
    if not _wait_for_file(log_path, timeout=15.0):
        print("WARNING: log file did not appear within 15 s", file=sys.stderr)

    # Wait for dsim to finish (it exits after --frames).
    deadline = duration_s + 30
    try:
        dsim_proc.wait(timeout=deadline)
    except subprocess.TimeoutExpired:
        print("WARNING: dsim did not exit within budget, killing", file=sys.stderr)
        dsim_proc.kill()
        dsim_proc.wait()

    # Give daic a moment to write the final log record, then stop it.
    time.sleep(2.0)
    daic_proc.terminate()
    try:
        daic_proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        daic_proc.kill()
        daic_proc.wait()

    return {}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="daic automated flight test")
    parser.add_argument("--map",      default="dsim/assets/maps/maze_001.txt",
                        help="map file (relative to project root)")
    parser.add_argument("--duration", type=int, default=90,
                        help="maximum flight time in seconds")
    parser.add_argument("--fps",      type=int, default=30,
                        help="simulation fps (affects --frames budget)")
    parser.add_argument("--log",      default=None,
                        help="path for the JSONL log (default: /tmp/daic_flight_<ts>.jsonl)")
    parser.add_argument("--verbose",  action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    map_path = ROOT / args.map
    if not map_path.exists():
        print(f"error: map not found: {map_path}", file=sys.stderr)
        return 1

    ts = int(time.time())
    log_path = Path(args.log) if args.log else Path(f"/tmp/daic_flight_{ts}.jsonl")

    print(f"Map:      {args.map}", file=sys.stderr)
    print(f"Budget:   {args.duration} s  ({args.duration * args.fps} frames @ {args.fps} fps)", file=sys.stderr)
    print(f"Log:      {log_path}", file=sys.stderr)
    print(file=sys.stderr)

    run_test(str(map_path), args.duration, log_path, args.fps, args.verbose)

    if not log_path.exists():
        print("error: no log file produced", file=sys.stderr)
        return 1

    summary = analyze_log(log_path, map_path=str(map_path))
    print_report(summary)

    # Also emit the raw summary as JSON for programmatic consumption.
    import json
    print()
    print("JSON summary:")
    print(json.dumps(summary, indent=2))

    return 0 if summary.get("landed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
