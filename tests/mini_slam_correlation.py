#!/usr/bin/env python3
"""Offline mini-SLAM risk/map correlation analysis for Phase 6.2.

This script may load simulator maps because it is an offline benchmark/report
tool. DAIC runtime navigation must still not read maps.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dvision2_common import SimMap, load_map


_RISK_THRESHOLD = 0.30
_AHEAD_NEAR_M = 3.0
_AHEAD_FAR_M = 5.0
_AHEAD_LATERAL_M = 1.5


def _load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if isinstance(rec, dict) and "t" in rec:
            records.append(rec)
    return records


def _try_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _resolve_paths(arg: str, map_arg: str | None = None) -> tuple[Path, Path]:
    p = Path(arg)
    if p.is_dir():
        log_path = p / "flight.jsonl"
        if not log_path.exists():
            log_path = p / "daic" / "flight.jsonl"
        metadata_path = p / "metadata.json"
    else:
        log_path = p
        metadata_path = p.parent / "metadata.json"

    if not log_path.exists():
        raise FileNotFoundError(f"flight log not found: {log_path}")

    map_path: Path | None = Path(map_arg) if map_arg else None
    if map_path is None and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        raw_map = metadata.get("map")
        if raw_map:
            candidate = Path(raw_map)
            map_path = candidate if candidate.is_absolute() else ROOT / candidate

    if map_path is None:
        raise FileNotFoundError("map path not found; pass --map or use a report dir with metadata.json")
    if not map_path.exists():
        raise FileNotFoundError(f"map not found: {map_path}")
    return log_path, map_path


def _nearest_obstacle_ahead(sim_map: SimMap, x: float, y: float,
                            yaw_deg: float,
                            lateral_m: float = _AHEAD_LATERAL_M) -> float | None:
    yaw = math.radians(yaw_deg)
    fwd_x, fwd_y = math.cos(yaw), math.sin(yaw)
    nearest: float | None = None
    for obj in sim_map.objects:
        if obj.kind not in ("wall", "tree"):
            continue
        dx = obj.x - x
        dy = obj.y - y
        along = dx * fwd_x + dy * fwd_y
        if along <= 0.0:
            continue
        lateral = abs(-dx * fwd_y + dy * fwd_x)
        # Wall cells are 1 m boxes and trees have non-zero crown radius, so
        # allow roughly half a cell beyond the local-map front path width.
        if lateral <= lateral_m + 0.5:
            nearest = along if nearest is None else min(nearest, along)
    return nearest


def _mini_slam_frontish(slam: dict[str, Any]) -> float:
    return max(
        float(slam.get("front") or 0.0),
        float(slam.get("front_left") or 0.0) * 0.7,
        float(slam.get("front_right") or 0.0) * 0.7,
    )


def analyze(log_path: str | Path, map_path: str | Path) -> dict[str, Any]:
    sim_map = load_map(Path(map_path))
    records = _load_records(Path(log_path))

    total = 0
    mini_ticks = 0
    risk_ticks = 0
    risk_with_near = 0
    risk_with_far = 0
    no_risk_with_near = 0
    rangeless_risk = 0
    front_occ_close = 0
    nearest_when_risk: list[float] = []
    nearest_when_no_risk: list[float] = []
    by_state: dict[str, dict[str, int]] = {}

    for rec in records:
        telem = rec.get("telem") or {}
        vision = rec.get("vision") or {}
        slam = vision.get("slam") or {}
        method = str(slam.get("method") or "")
        if not method.startswith("mini_slam:"):
            continue

        x = _try_float(telem.get("drone.x_m"))
        y = _try_float(telem.get("drone.y_m"))
        yaw = _try_float(telem.get("drone.heading_deg"))
        if None in (x, y, yaw):
            continue

        total += 1
        mini_ticks += 1
        state = str(rec.get("state") or "unknown")
        bucket = by_state.setdefault(state, {
            "ticks": 0,
            "risk_ticks": 0,
            "risk_with_near_obstacle": 0,
            "risk_without_near_obstacle": 0,
        })
        bucket["ticks"] += 1

        nearest = _nearest_obstacle_ahead(sim_map, float(x), float(y), float(yaw))
        risk = _mini_slam_frontish(slam)
        has_risk = risk >= _RISK_THRESHOLD
        has_near = nearest is not None and nearest <= _AHEAD_NEAR_M
        has_far = nearest is not None and nearest <= _AHEAD_FAR_M
        local_map = vision.get("local_map") or {}
        front_occ = _try_float(local_map.get("front_occ_m"))
        if front_occ is not None and front_occ <= 1.5:
            front_occ_close += 1

        if has_risk:
            risk_ticks += 1
            bucket["risk_ticks"] += 1
            if has_near:
                risk_with_near += 1
                bucket["risk_with_near_obstacle"] += 1
            else:
                bucket["risk_without_near_obstacle"] += 1
            if has_far:
                risk_with_far += 1
            if all(slam.get(name) is None for name in (
                    "front_range_m", "front_left_range_m", "front_right_range_m")):
                rangeless_risk += 1
            if nearest is not None:
                nearest_when_risk.append(nearest)
        else:
            if has_near:
                no_risk_with_near += 1
            if nearest is not None:
                nearest_when_no_risk.append(nearest)

    def rate(num: int, den: int) -> float:
        return round(num / den, 3) if den else 0.0

    def median(vals: list[float]) -> float | None:
        if not vals:
            return None
        vals = sorted(vals)
        mid = len(vals) // 2
        if len(vals) % 2:
            return round(vals[mid], 3)
        return round((vals[mid - 1] + vals[mid]) / 2.0, 3)

    return {
        "log_path": str(Path(log_path)),
        "map_path": str(Path(map_path)),
        "mini_slam_ticks": mini_ticks,
        "risk_threshold": _RISK_THRESHOLD,
        "near_obstacle_ahead_m": _AHEAD_NEAR_M,
        "far_obstacle_ahead_m": _AHEAD_FAR_M,
        "risk_ticks": risk_ticks,
        "risk_rate": rate(risk_ticks, total),
        "risk_with_near_obstacle_ticks": risk_with_near,
        "risk_with_near_obstacle_rate": rate(risk_with_near, risk_ticks),
        "risk_with_far_obstacle_ticks": risk_with_far,
        "risk_with_far_obstacle_rate": rate(risk_with_far, risk_ticks),
        "no_risk_with_near_obstacle_ticks": no_risk_with_near,
        "rangeless_risk_ticks": rangeless_risk,
        "rangeless_risk_rate": rate(rangeless_risk, risk_ticks),
        "front_occ_close_ticks": front_occ_close,
        "front_occ_close_rate": rate(front_occ_close, total),
        "nearest_obstacle_ahead_median_when_risk": median(nearest_when_risk),
        "nearest_obstacle_ahead_median_when_no_risk": median(nearest_when_no_risk),
        "by_state": by_state,
    }


def print_report(result: dict[str, Any], file=None) -> None:
    out = file or sys.stdout
    print("Mini-SLAM Correlation (Phase 6.2)", file=out)
    print(f"  Log:              {result['log_path']}", file=out)
    print(f"  Map:              {result['map_path']}", file=out)
    print(f"  Mini-SLAM ticks:  {result['mini_slam_ticks']}", file=out)
    print(f"  Risk ticks:       {result['risk_ticks']} ({result['risk_rate']:.1%})", file=out)
    print(f"  Rangeless risk:   {result['rangeless_risk_ticks']} "
          f"({result['rangeless_risk_rate']:.1%} of risk ticks)", file=out)
    print(f"  Risk with obstacle <= {result['near_obstacle_ahead_m']} m ahead: "
          f"{result['risk_with_near_obstacle_ticks']} "
          f"({result['risk_with_near_obstacle_rate']:.1%})", file=out)
    print(f"  Risk with obstacle <= {result['far_obstacle_ahead_m']} m ahead:  "
          f"{result['risk_with_far_obstacle_ticks']} "
          f"({result['risk_with_far_obstacle_rate']:.1%})", file=out)
    print(f"  front_occ_m <= 1.5 m: {result['front_occ_close_ticks']} "
          f"({result['front_occ_close_rate']:.1%})", file=out)
    print("  By state:", file=out)
    for state, row in sorted(result["by_state"].items()):
        risk_ticks = row["risk_ticks"]
        print(f"    {state}: ticks={row['ticks']} risk={risk_ticks} "
              f"risk_with_near={row['risk_with_near_obstacle']} "
              f"risk_without_near={row['risk_without_near_obstacle']}", file=out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report_or_log", help="benchmark report directory or flight.jsonl")
    parser.add_argument("--map", dest="map_path", help="simulator map path")
    parser.add_argument("--json", action="store_true", help="print JSON instead of text")
    args = parser.parse_args(argv)

    log_path, map_path = _resolve_paths(args.report_or_log, args.map_path)
    result = analyze(log_path, map_path)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print_report(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
