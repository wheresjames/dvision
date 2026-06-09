#!/usr/bin/env python3
"""Diagnose why a DAIC benchmark run stayed in SEARCH or failed (Phase 5.5).

Summarizes route/control causes from a flight log — and, when available, the
matching route_log.jsonl and DAIC run-reporter summary.json — so a run can be
classified as a perception miss, map noise/trap, planning miss, control stall,
or target-reacquisition miss without re-tuning blindly.

Usage
-----
    # Point at a benchmark report directory (looks under <dir>/daic/ first):
    python3 tests/benchmark_diagnosis.py reports/benchmarks/phase5-maze002-approach-gate

    # Or point directly at a flight log (route_log.jsonl / summary.json are
    # picked up from the same directory if present):
    python3 tests/benchmark_diagnosis.py /tmp/flight.jsonl
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from daic.flight_log import diagnose_log, print_diagnosis


def _resolve_paths(arg: str) -> tuple[Path, Path | None, Path | None]:
    p = Path(arg)
    daic_dir = p / "daic" if p.is_dir() and (p / "daic" / "flight.jsonl").exists() else \
               (p if p.is_dir() else p.parent)
    log_path = daic_dir / "flight.jsonl" if p.is_dir() else p
    route_log = daic_dir / "route_log.jsonl"
    daic_summary = daic_dir / "summary.json"
    return (log_path,
            route_log if route_log.exists() else None,
            daic_summary if daic_summary.exists() else None)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python3 tests/benchmark_diagnosis.py <report_dir-or-flight.jsonl>",
              file=sys.stderr)
        return 2

    log_path, route_log_path, daic_summary_path = _resolve_paths(argv[1])
    if not log_path.exists():
        print(f"error: flight log not found: {log_path}", file=sys.stderr)
        return 1

    diagnosis = diagnose_log(log_path,
                             route_log_path=route_log_path,
                             daic_summary_path=daic_summary_path)
    print_diagnosis(diagnosis)
    return 0 if "error" not in diagnosis else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
