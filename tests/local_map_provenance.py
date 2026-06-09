#!/usr/bin/env python3
"""Summarize LocalOccupancyMap provenance from benchmark flight logs.

Phase 6.3 tool: explains which source/sector/range class produced close
`front_occ_m` cells. It reads DAIC logs only; no simulator map is required.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any


_FRONT_OCC_CLOSE_M = 1.5


def _load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if isinstance(rec, dict) and "t" in rec:
            records.append(rec)
    return records


def _resolve_log(arg: str) -> Path:
    p = Path(arg)
    if p.is_dir():
        log = p / "flight.jsonl"
        if not log.exists():
            log = p / "daic" / "flight.jsonl"
    else:
        log = p
    if not log.exists():
        raise FileNotFoundError(f"flight log not found: {log}")
    return log


def _source_key(source: dict[str, Any]) -> str:
    if not source:
        return "unknown"
    ranged = "ranged" if source.get("ranged") else "default"
    return f"{source.get('source')}|{source.get('sector')}|{ranged}"


def _median(values: list[float]) -> float | None:
    return round(statistics.median(values), 2) if values else None


def analyze(log_path: str | Path) -> dict[str, Any]:
    records = _load_records(Path(log_path))
    close_ticks = 0
    block_close_ticks = 0
    no_provenance = 0
    by_source: dict[str, int] = {}
    by_state: dict[str, int] = {}
    age_by_source: dict[str, list[float]] = {}
    value_by_source: dict[str, list[float]] = {}
    occupied_by_source_total: dict[str, int] = {}

    for rec in records:
        lm = rec.get("vision", {}).get("local_map", {}) or {}
        for key, count in (lm.get("occupied_by_source") or {}).items():
            occupied_by_source_total[key] = occupied_by_source_total.get(key, 0) + int(count)

        front_occ_m = lm.get("front_occ_m")
        front_block_occ_m = lm.get("front_block_occ_m")
        if front_block_occ_m is not None and float(front_block_occ_m) <= _FRONT_OCC_CLOSE_M:
            block_close_ticks += 1
        if front_occ_m is None or float(front_occ_m) > _FRONT_OCC_CLOSE_M:
            continue
        close_ticks += 1
        state = str(rec.get("state") or "unknown")
        by_state[state] = by_state.get(state, 0) + 1

        source = lm.get("front_occ_source") or {}
        key = _source_key(source)
        if key == "unknown":
            no_provenance += 1
        by_source[key] = by_source.get(key, 0) + 1
        age = source.get("age_ticks")
        if isinstance(age, (int, float)):
            age_by_source.setdefault(key, []).append(float(age))
        value = source.get("value")
        if isinstance(value, (int, float)):
            value_by_source.setdefault(key, []).append(float(value))

    return {
        "log_path": str(Path(log_path)),
        "tick_count": len(records),
        "front_occ_close_threshold_m": _FRONT_OCC_CLOSE_M,
        "front_occ_close_ticks": close_ticks,
        "front_occ_close_rate": round(close_ticks / len(records), 3) if records else 0.0,
        "front_block_occ_close_ticks": block_close_ticks,
        "front_block_occ_close_rate": (
            round(block_close_ticks / len(records), 3) if records else 0.0
        ),
        "front_occ_without_provenance_ticks": no_provenance,
        "front_occ_close_by_source": dict(sorted(by_source.items())),
        "front_occ_close_by_state": dict(sorted(by_state.items())),
        "front_occ_close_source_age": {
            key: {"median": _median(vals), "max": max(vals)}
            for key, vals in sorted(age_by_source.items())
        },
        "front_occ_close_source_value": {
            key: {"median": _median(vals), "max": round(max(vals), 3)}
            for key, vals in sorted(value_by_source.items())
        },
        "occupied_by_source_total": dict(sorted(occupied_by_source_total.items())),
    }


def print_report(result: dict[str, Any], file=None) -> None:
    out = file or sys.stdout
    print("Local-Map Provenance (Phase 6.3)", file=out)
    print(f"  Log:                  {result['log_path']}", file=out)
    print(f"  Ticks:                {result['tick_count']}", file=out)
    print(f"  front_occ_m <= {result['front_occ_close_threshold_m']} m: "
          f"{result['front_occ_close_ticks']} "
          f"({result['front_occ_close_rate']:.1%})", file=out)
    print(f"  front_block_occ_m <= {result['front_occ_close_threshold_m']} m: "
          f"{result['front_block_occ_close_ticks']} "
          f"({result['front_block_occ_close_rate']:.1%})", file=out)
    print(f"  Close ticks without provenance: "
          f"{result['front_occ_without_provenance_ticks']}", file=out)
    print("  Close-front by source:", file=out)
    ages = result["front_occ_close_source_age"]
    values = result["front_occ_close_source_value"]
    for key, count in result["front_occ_close_by_source"].items():
        age = ages.get(key) or {}
        value = values.get(key) or {}
        print(f"    {key}: {count} "
              f"age_med={age.get('median')} age_max={age.get('max')} "
              f"value_med={value.get('median')} value_max={value.get('max')}",
              file=out)
    print("  Close-front by state:", file=out)
    for key, count in result["front_occ_close_by_state"].items():
        print(f"    {key}: {count}", file=out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report_or_log", help="benchmark report directory or flight.jsonl")
    parser.add_argument("--json", action="store_true", help="print JSON instead of text")
    args = parser.parse_args(argv)
    result = analyze(_resolve_log(args.report_or_log))
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print_report(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
