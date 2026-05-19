"""Structured JSONL flight logger and log analyzer for daic.

Each line is one JSON object.  Special records use an "event" key; per-tick
records use a "t" (monotonic seconds) key plus "state", "det", "cmd", "telem".
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

class FlightLogger:
    def __init__(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self._fh = p.open("w", encoding="utf-8")
        self._t0 = time.monotonic()
        self._write({"event": "start", "wall_time": time.time()})

    def log_tick(self, now: float, state_name: str, detection,
                 command_type: str, command_fields: dict,
                 telemetry: dict, status_text: str = "",
                 vision: dict | None = None) -> None:
        det: dict[str, Any]
        if detection.visible:
            det = {
                "visible": True,
                "cx":   round(detection.cx,         1),
                "cy":   round(detection.cy,         1),
                "r":    round(detection.radius,     1),
                "conf": round(detection.confidence, 3),
            }
        else:
            det = {"visible": False}

        # Keep telemetry compact — drop sim.* metadata keys.
        telem = {k: v for k, v in telemetry.items()
                 if not k.startswith("sim.")}

        fields: dict[str, Any] = {}
        for k, v in command_fields.items():
            try:
                fields[k] = round(float(v), 4)
            except (TypeError, ValueError):
                fields[k] = v

        record = {
            "t":      round(now - self._t0, 3),
            "state":  state_name,
            "status": status_text,
            "det":    det,
            "cmd":    {"type": command_type, **fields},
            "telem":  telem,
        }
        if vision is not None:
            record["vision"] = vision
        self._write(record)

    def log_event(self, event: str, **data: Any) -> None:
        self._write({"t": round(time.monotonic() - self._t0, 3),
                     "event": event, **data})

    def close(self) -> None:
        self._write({"event": "stop",
                     "wall_time": time.time(),
                     "elapsed_s": round(time.monotonic() - self._t0, 3)})
        self._fh.close()

    def _write(self, record: dict) -> None:
        self._fh.write(json.dumps(record, separators=(",", ":")) + "\n")
        self._fh.flush()


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

def _f(d: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(d.get(key, default))
    except (TypeError, ValueError):
        return default


def analyze_log(log_path: str | Path,
                map_path: str | Path | None = None) -> dict:
    """Parse a JSONL flight log and return a summary dict.

    If *map_path* is provided, the target position is loaded from the map so
    landing accuracy can be calculated.
    """
    records: list[dict] = []
    with open(log_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    ticks = [r for r in records if "state" in r and "telem" in r]
    events = [r for r in records if "event" in r]

    if not ticks:
        return {"error": "no tick data in log"}

    # ── Target position from map ──────────────────────────────────────
    target_pos: tuple[float, float] | None = None
    if map_path is not None:
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
            from dvision2_common import load_map
            sim_map = load_map(Path(map_path))
            targets = [o for o in sim_map.objects if o.kind == "target"]
            if targets:
                target_pos = (targets[0].x, targets[0].y)
        except Exception as exc:
            pass  # non-fatal

    # ── Drone trajectory ──────────────────────────────────────────────
    positions: list[tuple[float, float, float, float, str]] = []  # t,x,y,z,state
    for tick in ticks:
        telem = tick["telem"]
        positions.append((
            tick["t"],
            _f(telem, "drone.x_m"),
            _f(telem, "drone.y_m"),
            _f(telem, "drone.z_m"),
            tick["state"],
        ))

    # ── State dwell times ─────────────────────────────────────────────
    state_times: dict[str, float] = {}
    for i in range(1, len(ticks)):
        state = ticks[i - 1]["state"]
        dt = ticks[i]["t"] - ticks[i - 1]["t"]
        state_times[state] = state_times.get(state, 0.0) + dt

    # ── Detection statistics ──────────────────────────────────────────
    det_frames = sum(1 for t in ticks if t["det"].get("visible"))
    det_rate = det_frames / len(ticks) if ticks else 0.0

    # ── Outcome ───────────────────────────────────────────────────────
    final_state = ticks[-1]["state"]
    landed = final_state == "COMPLETE"

    # ── Final position and landing error ─────────────────────────────
    final_pos = (positions[-1][1], positions[-1][2], positions[-1][3])
    landing_error: float | None = None
    if target_pos is not None:
        dx = final_pos[0] - target_pos[0]
        dy = final_pos[1] - target_pos[1]
        landing_error = round(math.hypot(dx, dy), 3)

    # Also treat "on the ground near the target" as landed — covers the case
    # where dsim disarmed before the planner saw COMPLETE, or the planner
    # issued land but the state didn't flush before the log closed.
    if not landed:
        last_telem = ticks[-1]["telem"]
        on_ground = _f(last_telem, "drone.z_m") < 0.4
        if on_ground and last_telem.get("drone.armed") == "0":
            landed = True
        if on_ground and landing_error is not None and landing_error < 0.5:
            landed = True

    # Closest the drone ever got to the target (horizontally).
    min_dist: float | None = None
    if target_pos is not None:
        for _, x, y, _, _ in positions:
            d = math.hypot(x - target_pos[0], y - target_pos[1])
            if min_dist is None or d < min_dist:
                min_dist = d
        if min_dist is not None:
            min_dist = round(min_dist, 3)

    # ── Command analysis ──────────────────────────────────────────────
    # Peak forward speed commanded during APPROACH/LANDING.
    peak_fwd = 0.0
    for tick in ticks:
        if tick["state"] in ("APPROACH", "LANDING"):
            fwd = abs(_f(tick["cmd"], "forward_mps"))
            if fwd > peak_fwd:
                peak_fwd = fwd

    duration_s = positions[-1][0] - positions[0][0] if len(positions) > 1 else 0.0

    return {
        "landed":           landed,
        "final_state":      final_state,
        "duration_s":       round(duration_s, 2),
        "final_position":   {"x": round(final_pos[0], 3),
                             "y": round(final_pos[1], 3),
                             "z": round(final_pos[2], 3)},
        "target_position":  {"x": target_pos[0], "y": target_pos[1]}
                             if target_pos else None,
        "landing_error_m":  landing_error,
        "min_dist_to_target_m": min_dist,
        "state_times_s":    {k: round(v, 2) for k, v in state_times.items()},
        "detection_rate":   round(det_rate, 3),
        "peak_approach_fwd_mps": round(peak_fwd, 3),
        "tick_count":       len(ticks),
    }


def print_report(summary: dict, file=None) -> None:
    """Print a human-readable report from analyze_log() output."""
    out = file or sys.stdout
    print("=" * 54, file=out)
    print("  DAIC Flight Test Report", file=out)
    print("=" * 54, file=out)

    if "error" in summary:
        print(f"  ERROR: {summary['error']}", file=out)
        return

    result = "LANDED ✓" if summary["landed"] else "DID NOT LAND ✗"
    print(f"  Result:        {result}", file=out)
    print(f"  Final state:   {summary['final_state']}", file=out)
    print(f"  Duration:      {summary['duration_s']:.1f} s", file=out)
    print(f"  Ticks logged:  {summary['tick_count']}", file=out)
    print(f"  Detection rate:{summary['detection_rate']:.1%}", file=out)
    print(file=out)

    pos = summary["final_position"]
    print(f"  Final pos:     x={pos['x']:.2f}  y={pos['y']:.2f}  z={pos['z']:.2f}", file=out)

    tgt = summary.get("target_position")
    if tgt:
        print(f"  Target pos:    x={tgt['x']:.2f}  y={tgt['y']:.2f}", file=out)
    if summary.get("landing_error_m") is not None:
        print(f"  Landing error: {summary['landing_error_m']:.3f} m", file=out)
    if summary.get("min_dist_to_target_m") is not None:
        print(f"  Closest pass:  {summary['min_dist_to_target_m']:.3f} m", file=out)

    print(file=out)
    print(f"  Peak approach fwd: {summary['peak_approach_fwd_mps']:.2f} m/s (pre-scaled)", file=out)
    print(file=out)
    print("  State dwell times:", file=out)
    for state, t in sorted(summary["state_times_s"].items()):
        print(f"    {state:<12} {t:6.1f} s", file=out)
    print("=" * 54, file=out)
