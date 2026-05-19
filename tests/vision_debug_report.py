#!/usr/bin/env python3
"""Summarize daic vision/local-map diagnostics from a JSONL flight log."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _f(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python3 tests/vision_debug_report.py /tmp/flight.jsonl",
              file=sys.stderr)
        return 2

    path = Path(argv[1])
    ticks = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "vision" in rec:
                ticks.append(rec)

    if not ticks:
        print("no vision diagnostics found")
        return 1

    front_events = []
    occupied_samples = []
    for rec in ticks:
        v = rec["vision"]
        fused = v.get("fused", {})
        lm = v.get("local_map", {})
        if _f(fused.get("front")) >= 0.2 or _f(fused.get("front_left")) >= 0.2 or _f(fused.get("front_right")) >= 0.2:
            front_events.append(rec)
        if lm.get("occupied_cells", 0):
            occupied_samples.append(rec)

    print(f"ticks with diagnostics: {len(ticks)}")
    print(f"ticks with front-ish risk >= 0.2: {len(front_events)}")
    print(f"ticks with occupied map cells: {len(occupied_samples)}")

    if front_events:
        print("\nfirst front-risk ticks:")
        for rec in front_events[:8]:
            fused = rec["vision"]["fused"]
            lm = rec["vision"]["local_map"]
            telem = rec.get("telem", {})
            cmd = rec.get("cmd", {})
            print(
                f"t={rec['t']:>6.3f} pos=({_f(telem.get('drone.x_m')):5.2f},"
                f"{_f(telem.get('drone.y_m')):5.2f}) "
                f"risk F/FL/FR={_f(fused.get('front')):.2f}/"
                f"{_f(fused.get('front_left')):.2f}/"
                f"{_f(fused.get('front_right')):.2f} "
                f"range F/FL/FR={fused.get('front_range_m')}/"
                f"{fused.get('front_left_range_m')}/"
                f"{fused.get('front_right_range_m')} "
                f"occ nearest/front={lm.get('nearest_occ_m')}/"
                f"{lm.get('front_occ_m')} "
                f"cmd f/y={_f(cmd.get('forward_mps')):.2f}/"
                f"{_f(cmd.get('yaw_rate_dps')):.2f}"
            )

    if occupied_samples:
        print("\nlast occupied-map ticks:")
        for rec in occupied_samples[-8:]:
            lm = rec["vision"]["local_map"]
            telem = rec.get("telem", {})
            print(
                f"t={rec['t']:>6.3f} pos=({_f(telem.get('drone.x_m')):5.2f},"
                f"{_f(telem.get('drone.y_m')):5.2f}) "
                f"cells occ/free/path={lm.get('occupied_cells')}/"
                f"{lm.get('free_cells')}/{lm.get('path_len')} "
                f"nearest/front={lm.get('nearest_occ_m')}/{lm.get('front_occ_m')} "
                f"default_projection={lm.get('default_obstacle_projection_m')}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
