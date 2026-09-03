#!/usr/bin/env python3
"""Aggregate N independent benchmark runs into one comparable verdict.

Repeated runs of the *same* code against the *same* map can land anywhere from
a clean COMPLETE in 17s (closest pass 0.61m) to a FAILSAFE 5.85m short — so a
single run cannot tell "this change helped" apart from "this run landed in the
lucky tail of the distribution". This script takes a set of independent
benchmark report directories (each produced by tests/flight_test.py) and
reports the median and range across the set, so before/after comparisons are
made between distributions rather than between individual samples.

It deliberately does *not* run the benchmarks itself or merge their per-run
artifacts (flight.jsonl, occupancy galleries, ...) into one report — each run
stays a fully independent, inspectable directory. This is just a thin
aggregation layer on top.

Usage
-----
    # Point at a set of report directories produced with the same config:
    python3 tests/benchmark_aggregate.py reports/benchmarks/maze001-run*

    # Write the same summary to a file as well:
    python3 tests/benchmark_aggregate.py --out summary.txt reports/benchmarks/maze001-run*
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, TextIO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_run(report_dir: Path) -> dict[str, Any] | None:
    summary_path = report_dir / "summary.json"
    metadata_path = report_dir / "metadata.json"
    if not summary_path.exists():
        return None
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    metadata: dict[str, Any] = {}
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            metadata = {}

    diagnosis = summary.get("diagnosis") or {}
    hints = [h.get("label") for h in diagnosis.get("classification_hints", [])
             if isinstance(h, dict) and h.get("label")]
    return {
        "report_dir": str(report_dir),
        "map": metadata.get("map"),
        "duration_s": metadata.get("duration_s"),
        "git_commit": metadata.get("git_commit"),
        "landed": bool(summary.get("landed")),
        "final_state": summary.get("final_state"),
        "landing_error_m": _try_float(summary.get("landing_error_m")),
        "min_dist_to_target_m": _try_float(summary.get("min_dist_to_target_m")),
        "detection_rate": _try_float(summary.get("detection_rate")),
        "effective_fps": _try_float(summary.get("effective_fps")),
        "target_fps": _try_float(summary.get("target_fps")),
        "frame_drop_ratio": _try_float(summary.get("frame_drop_ratio")),
        "frame_drop_suspected": bool(summary.get("frame_drop_suspected")),
        "hints": hints,
    }


def _try_float(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def aggregate(report_dirs: list[Path]) -> dict[str, Any]:
    runs = []
    missing = []
    for d in report_dirs:
        run = _load_run(d)
        if run is None:
            missing.append(str(d))
        else:
            runs.append(run)

    if not runs:
        return {"runs": [], "missing": missing, "n": 0}

    landed_count = sum(1 for r in runs if r["landed"])
    final_states: dict[str, int] = {}
    for r in runs:
        final_states[r["final_state"]] = final_states.get(r["final_state"], 0) + 1

    hint_counts: dict[str, int] = {}
    for r in runs:
        for label in set(r["hints"]):
            hint_counts[label] = hint_counts.get(label, 0) + 1

    frame_drop_runs = [r for r in runs if r["frame_drop_suspected"]]

    return {
        "n": len(runs),
        "missing": missing,
        "maps": sorted({r["map"] for r in runs if r["map"]}),
        "git_commits": sorted({r["git_commit"] for r in runs if r["git_commit"]}),
        "landed_count": landed_count,
        "landed_rate": landed_count / len(runs),
        "final_states": final_states,
        "hint_counts": hint_counts,
        "landing_error_m": _stats([r["landing_error_m"] for r in runs]),
        "min_dist_to_target_m": _stats([r["min_dist_to_target_m"] for r in runs]),
        "detection_rate": _stats([r["detection_rate"] for r in runs]),
        "effective_fps": _stats([r["effective_fps"] for r in runs]),
        "frame_drop_count": len(frame_drop_runs),
        "frame_drop_runs": [r["report_dir"] for r in frame_drop_runs],
        "runs": runs,
    }


def _stats(values: list[float | None]) -> dict[str, float | None]:
    clean = [v for v in values if v is not None]
    if not clean:
        return {"median": None, "min": None, "max": None, "n": 0}
    return {
        "median": round(statistics.median(clean), 3),
        "min": round(min(clean), 3),
        "max": round(max(clean), 3),
        "n": len(clean),
    }


def print_aggregate(agg: dict[str, Any], file: TextIO | None = None) -> None:
    out = file or sys.stdout
    line = "=" * 54
    print(line, file=out)
    print("  Multi-Run Benchmark Aggregate", file=out)
    print(line, file=out)

    if agg["n"] == 0:
        print("  No runs with a summary.json were found.", file=out)
        for m in agg["missing"]:
            print(f"    missing: {m}", file=out)
        print(line, file=out)
        return

    if agg["frame_drop_count"]:
        print(f"  *** WARNING: {agg['frame_drop_count']}/{agg['n']} run(s) show suspected "
              f"dropped frames (effective fps well below the configured rate -- the "
              f"host was likely overloaded, e.g. by running too many benchmarks in "
              f"parallel). Their results may be unreliable: ***", file=out)
        for d in agg["frame_drop_runs"]:
            print(f"      - {d}", file=out)
        print(file=out)

    print(f"  Runs aggregated:  {agg['n']}", file=out)
    if agg["missing"]:
        print(f"  Skipped (no summary.json): {len(agg['missing'])}", file=out)
        for m in agg["missing"]:
            print(f"    - {m}", file=out)
    print(f"  Map(s):           {', '.join(agg['maps']) or '?'}", file=out)
    print(f"  Git commit(s):    {', '.join(agg['git_commits']) or '?'}", file=out)
    print(file=out)

    print(f"  Landed:           {agg['landed_count']}/{agg['n']} "
          f"({agg['landed_rate'] * 100:.0f}%)", file=out)
    states = ", ".join(f"{k}={v}" for k, v in sorted(agg["final_states"].items()))
    print(f"  Final states:     {states}", file=out)
    print(file=out)

    for key, label in (
        ("landing_error_m", "Landing error (m)"),
        ("min_dist_to_target_m", "Closest pass (m)"),
        ("detection_rate", "Detection rate"),
        ("effective_fps", "Effective fps"),
    ):
        s = agg[key]
        if s["n"] == 0:
            print(f"  {label:<20} n/a", file=out)
            continue
        print(f"  {label:<20} median={s['median']:<8} "
              f"range=[{s['min']}, {s['max']}]  (n={s['n']})", file=out)
    print(file=out)

    if agg["hint_counts"]:
        print("  Classification hints (how many of the N runs raised each):", file=out)
        for label, count in sorted(agg["hint_counts"].items(), key=lambda kv: -kv[1]):
            print(f"    - {label}: {count}/{agg['n']}", file=out)
    else:
        print("  Classification hints: none raised in any run.", file=out)

    print(file=out)
    print("  Per-run detail:", file=out)
    for r in agg["runs"]:
        landed = "LANDED " if r["landed"] else "FAILSAFE"
        flag = "  [DROPPED FRAMES]" if r["frame_drop_suspected"] else ""
        print(f"    [{landed}] {r['final_state']:<10} "
              f"err={r['landing_error_m']:<6} closest={r['min_dist_to_target_m']:<6} "
              f"det={r['detection_rate']:<5} -- {r['report_dir']}{flag}", file=out)
    print(line, file=out)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("report_dirs", nargs="+",
                        help="benchmark report directories to aggregate "
                             "(each must contain a summary.json)")
    parser.add_argument("--out", type=str, default=None,
                        help="also write the summary to this file")
    args = parser.parse_args(argv[1:])

    dirs = [Path(d) for d in args.report_dirs]
    agg = aggregate(dirs)
    print_aggregate(agg)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            print_aggregate(agg, file=fh)
        print(f"\nWrote summary to {args.out}", file=sys.stderr)

    return 0 if agg["n"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
