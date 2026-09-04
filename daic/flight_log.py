"""Structured JSONL flight logger and log analyzer for daic.

Each line is one JSON object.  Special records use an "event" key; per-tick
records use a "t" (monotonic seconds) key plus "state", "det", "cmd", "telem".
"""

from __future__ import annotations

import json
import math
import re
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


#: Below this fraction of the configured tick rate, a run is flagged as
#: having likely dropped frames (e.g. CPU oversubscription from running too
#: many benchmarks in parallel). A corrupted batch measured ~50-57% of its
#: target fps, while clean runs measured ~98-99%, so 90% comfortably separates
#: the two.
FRAME_DROP_WARN_RATIO = 0.9


def analyze_log(log_path: str | Path,
                map_path: str | Path | None = None,
                target_fps: float | None = None) -> dict:
    """Parse a JSONL flight log and return a summary dict.

    If *map_path* is provided, the target position is loaded from the map so
    landing accuracy can be calculated.

    If *target_fps* is provided (the configured tick rate), the summary also
    reports whether the run's effective fps fell suspiciously short of it --
    a sign that the test host was overloaded and dropped frames, which can
    silently corrupt timing-sensitive results.
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
    # log_event() records -- start, stop, and whatever the run reported -- were
    # collected here and then dropped, so nothing a run announced ever reached
    # the summary. They are few and small, so they are carried through whole.
    events = [r for r in records if "event" in r]

    if not ticks:
        return {"error": "no tick data in log"}

    # ── Target position from map ──────────────────────────────────────
    target_pos: tuple[float, float] | None = None
    target_error: str | None = None
    if map_path is not None:
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
            from dvision2_common import load_map
            sim_map = load_map(Path(map_path))
            targets = [o for o in sim_map.objects if o.kind == "target"]
            if targets:
                target_pos = (targets[0].x, targets[0].y)
        except Exception as exc:
            # Non-fatal, but not silent: without the map there is no target, so
            # landing_error_m comes back None and the reason belongs in the
            # summary rather than only in the mind of whoever wrote this.
            target_error = f"{type(exc).__name__}: {exc}"

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

    # ── Effective frame rate / dropped-frame detection ────────────────
    # tick_count / wall-clock duration -- a host that's maxed out on CPU
    # (e.g. too many parallel benchmark runs) ticks slower than configured,
    # silently corrupting timing-sensitive comparisons.
    effective_fps: float | None = None
    if duration_s > 0:
        effective_fps = round(len(ticks) / duration_s, 2)

    frame_drop_ratio: float | None = None
    frame_drop_suspected = False
    if target_fps and effective_fps is not None:
        frame_drop_ratio = round(effective_fps / target_fps, 3)
        frame_drop_suspected = frame_drop_ratio < FRAME_DROP_WARN_RATIO

    return {
        "landed":           landed,
        "final_state":      final_state,
        "duration_s":       round(duration_s, 2),
        "final_position":   {"x": round(final_pos[0], 3),
                             "y": round(final_pos[1], 3),
                             "z": round(final_pos[2], 3)},
        "target_position":  {"x": target_pos[0], "y": target_pos[1]}
                             if target_pos else None,
        "target_position_error": target_error,
        "events":           events,
        "landing_error_m":  landing_error,
        "min_dist_to_target_m": min_dist,
        "state_times_s":    {k: round(v, 2) for k, v in state_times.items()},
        "detection_rate":   round(det_rate, 3),
        "peak_approach_fwd_mps": round(peak_fwd, 3),
        "tick_count":       len(ticks),
        "effective_fps":    effective_fps,
        "target_fps":       target_fps,
        "frame_drop_ratio": frame_drop_ratio,
        "frame_drop_suspected": frame_drop_suspected,
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

    if summary.get("frame_drop_suspected"):
        print(f"  *** WARNING: effective fps ({summary['effective_fps']:.1f}) is only "
              f"{summary['frame_drop_ratio']:.0%} of the configured "
              f"{summary['target_fps']:.1f} fps -- this run likely dropped frames "
              f"(host overloaded?) and its results may be unreliable. ***", file=out)
        print(file=out)

    result = "LANDED ✓" if summary["landed"] else "DID NOT LAND ✗"
    print(f"  Result:        {result}", file=out)
    print(f"  Final state:   {summary['final_state']}", file=out)
    print(f"  Duration:      {summary['duration_s']:.1f} s", file=out)
    print(f"  Ticks logged:  {summary['tick_count']}", file=out)
    if summary.get("effective_fps") is not None:
        line = f"  Effective fps: {summary['effective_fps']:.1f}"
        if summary.get("target_fps"):
            line += f" (target {summary['target_fps']:.1f}, " \
                    f"{summary['frame_drop_ratio']:.0%} of target)"
        if summary.get("frame_drop_suspected"):
            line += "  *** DROPPED FRAMES SUSPECTED -- host overloaded? ***"
        print(line, file=out)
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
    print(f"  Peak approach fwd: {summary['peak_approach_fwd_mps']:.2f} m/s", file=out)
    print(file=out)
    print("  State dwell times:", file=out)
    for state, t in sorted(summary["state_times_s"].items()):
        print(f"    {state:<12} {t:6.1f} s", file=out)
    print("=" * 54, file=out)


# ---------------------------------------------------------------------------
# Benchmark diagnosis — why does the drone stay in SEARCH?
# ---------------------------------------------------------------------------

# Matches the live status text emitted for an active local-route command, e.g.
# "local route 12 cells" (optionally suffixed with "; avoid <method>").
_ROUTE_ACTIVE_RE = re.compile(r"^local route \d+ cells")

# Matches the APPROACH-gate's front_occ_m cutoff (daic/daic.py _APPROACH_BLOCK_FRONT_OCC_M)
# so "close occupancy" in the diagnosis lines up with the gate's own definition.
_FRONT_OCC_CLOSE_M = 1.5

# Commands at or below this are treated as "commanded zero forward".
_FORWARD_ZERO_EPS = 0.05


def _load_records(path: str | Path) -> list[dict]:
    records: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    vals = sorted(values)
    mid = len(vals) // 2
    if len(vals) % 2:
        return round(vals[mid], 1)
    return round((vals[mid - 1] + vals[mid]) / 2.0, 1)


def _bearing_deviation(rec: dict) -> dict:
    """Pair a route_log waypoint bearing with the direct bearing to the target."""
    bearing = rec.get("bearing")
    x, y = rec.get("x"), rec.get("y")
    tx, ty = rec.get("target_x"), rec.get("target_y")
    direct = deviation = None
    if None not in (bearing, x, y, tx, ty):
        direct = math.degrees(math.atan2(ty - y, tx - x)) % 360.0
        deviation = abs((bearing - direct + 180.0) % 360.0 - 180.0)
    return {
        "t":                 rec.get("t"),
        "bearing_deg":       bearing,
        "direct_bearing_deg": round(direct, 1) if direct is not None else None,
        "deviation_deg":     round(deviation, 1) if deviation is not None else None,
        "target_dist_m":     rec.get("target_dist_m"),
    }


def _classify_hints(stats: dict) -> list[dict]:
    """Heuristic pointers toward the likely failure category.

    These are evidence-based suggestions, not a verdict — confirm the call
    against the occ_*.png occupancy snapshot gallery.
    """
    total       = max(1, stats["tick_count"])
    wall_detect = stats["wall_detect_ticks"] or 0
    detour      = stats["detour_path_ticks"] or 0
    straight    = stats["straight_path_ticks"] or 0
    occ_peak    = stats["occ_peak_cells"] or 0
    hints: list[dict] = []

    if wall_detect == 0 and stats["front_occ_close_ticks"] < total * 0.02:
        hints.append({
            "label": "perception miss",
            "evidence": (
                f"wall_detect_ticks={wall_detect}, front_occ_m<="
                f"{_FRONT_OCC_CLOSE_M:.1f}m on {stats['front_occ_close_ticks']}/{total} "
                "ticks — sector risk and close front occupancy almost never "
                "appear. Check the sector timeline / occupancy gallery for whether "
                "risk should have risen earlier."
            ),
        })

    if (stats["front_occ_close_ticks"] > total * 0.2
            and stats["longest_front_occ_close_run_s"] > 5.0):
        hints.append({
            "label": "map noise / trap",
            "evidence": (
                f"front_occ_m<={_FRONT_OCC_CLOSE_M:.1f}m on "
                f"{stats['front_occ_close_ticks']}/{total} ticks with a "
                f"{stats['longest_front_occ_close_run_s']:.1f}s continuous stretch "
                f"(peak occupied cells={occ_peak}). Inspect occ_*.png for cells "
                "persisting close to the drone in open space."
            ),
        })

    if wall_detect > total * 0.1 and detour < straight * 0.5:
        hints.append({
            "label": "planning miss",
            "evidence": (
                f"wall_detect_ticks={wall_detect} but detour_path_ticks={detour} "
                f"vs straight_path_ticks={straight}. Compare the blue A* path "
                "against red occupied cells in the occupancy gallery — walls "
                "may be marked but not routed around."
            ),
        })

    route_active = stats["route_active_ticks"]
    if route_active > total * 0.1:
        ratio = stats["route_stalled_ticks"] / route_active
        if ratio > 0.5:
            hints.append({
                "label": "control stall",
                "evidence": (
                    f"{stats['route_stalled_ticks']} of {route_active} active "
                    f"local-route ticks ({ratio:.0%}) commanded ~zero forward — "
                    f"{stats['forward_zero_alignment_ticks']} while still aligning to "
                    f"a waypoint, {stats['avoidance_clamped_ticks']} avoidance-clamped "
                    "(these can overlap). The route may be plausible but forward "
                    "motion is repeatedly zeroed or clamped."
                ),
            })

    if stats["target_visible_rate"] < 0.02 and total > 100:
        hints.append({
            "label": "target reacquisition miss",
            "evidence": (
                f"target visible on only {stats['target_visible_rate']:.1%} of "
                f"{total} ticks. APPROACH may never be reached because the "
                "detector rarely or never reconfirms the target."
            ),
        })

    if not hints:
        hints.append({
            "label": "inconclusive",
            "evidence": (
                "No single pattern crossed the heuristic thresholds — read "
                "the occupancy gallery and sector timeline by hand to classify "
                "this run."
            ),
        })
    return hints


def diagnose_log(log_path: str | Path,
                 route_log_path: str | Path | None = None,
                 daic_summary_path: str | Path | None = None) -> dict:
    """Summarize route/control causes for why a run stayed in SEARCH or failed.

    Combines per-tick evidence from *log_path* (flight.jsonl) with waypoint
    bearing/deviation samples from *route_log_path* (route_log.jsonl) and the
    aggregate route/control counters the DAIC run reporter already writes to
    *daic_summary_path* (<report_dir>/daic/summary.json), so route-change and
    straight-vs-detour counts aren't recomputed from scratch.
    """
    ticks = [r for r in _load_records(log_path) if "state" in r and "telem" in r]
    if not ticks:
        return {"error": "no tick data in log"}

    total = len(ticks)
    target_visible      = 0
    approach_gated      = 0
    yaw_scan            = 0
    route_active        = 0
    forward_zero_align  = 0
    avoidance_clamped   = 0
    route_stalled       = 0
    front_occ_close     = 0
    front_block_occ_close = 0
    front_occ_sources: dict[str, int] = {}
    front_occ_source_age: dict[str, list[float]] = {}
    close_run_start: float | None = None
    longest_close_run   = 0.0

    for rec in ticks:
        status = rec.get("status") or ""
        det    = rec.get("det", {})
        cmd    = rec.get("cmd", {})
        lm     = rec.get("vision", {}).get("local_map", {})
        t      = _f(rec, "t")

        if det.get("visible"):
            target_visible += 1
        if "approach gated" in status:
            approach_gated += 1
        if "local route unavailable, yaw scan" in status:
            yaw_scan += 1

        fwd_zero = abs(_f(cmd, "forward_mps")) < _FORWARD_ZERO_EPS
        clamped  = "; avoid " in status
        if clamped:
            avoidance_clamped += 1

        if _ROUTE_ACTIVE_RE.match(status):
            route_active += 1
            if fwd_zero:
                forward_zero_align += 1
            if fwd_zero or clamped:
                route_stalled += 1

        front_occ_m = lm.get("front_occ_m")
        if front_occ_m is not None and front_occ_m <= _FRONT_OCC_CLOSE_M:
            front_occ_close += 1
            source = lm.get("front_occ_source") or {}
            if source:
                ranged = "ranged" if source.get("ranged") else "default"
                key = f"{source.get('source')}|{source.get('sector')}|{ranged}"
                front_occ_sources[key] = front_occ_sources.get(key, 0) + 1
                age = source.get("age_ticks")
                if isinstance(age, (int, float)):
                    front_occ_source_age.setdefault(key, []).append(float(age))
            if close_run_start is None:
                close_run_start = t
            longest_close_run = max(longest_close_run, t - close_run_start)
        else:
            close_run_start = None
        front_block_occ_m = lm.get("front_block_occ_m")
        if front_block_occ_m is not None and front_block_occ_m <= _FRONT_OCC_CLOSE_M:
            front_block_occ_close += 1

    route_events: list[dict] = []
    if route_log_path is not None and Path(route_log_path).exists():
        route_events = [_bearing_deviation(r) for r in _load_records(route_log_path)]
    deviations = [e["deviation_deg"] for e in route_events if e["deviation_deg"] is not None]

    daic_summary: dict[str, Any] = {}
    if daic_summary_path is not None and Path(daic_summary_path).exists():
        try:
            daic_summary = json.loads(Path(daic_summary_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            daic_summary = {}

    stats: dict[str, Any] = {
        "tick_count":                    total,
        "target_visible_ticks":          target_visible,
        "target_visible_rate":           round(target_visible / total, 3),
        "approach_gated_ticks":          approach_gated,
        "yaw_scan_ticks":                yaw_scan,
        "route_active_ticks":            route_active,
        "forward_zero_alignment_ticks":  forward_zero_align,
        "avoidance_clamped_ticks":       avoidance_clamped,
        "route_stalled_ticks":           route_stalled,
        "front_occ_close_ticks":         front_occ_close,
        "front_block_occ_close_ticks":   front_block_occ_close,
        "front_occ_close_threshold_m":   _FRONT_OCC_CLOSE_M,
        "longest_front_occ_close_run_s": round(longest_close_run, 2),
        "front_occ_close_sources":       dict(sorted(front_occ_sources.items())),
        "front_occ_close_source_age":    {
            key: {
                "median": _median(vals),
                "max": round(max(vals), 1),
            }
            for key, vals in sorted(front_occ_source_age.items())
            if vals
        },
        "route_changes":                 daic_summary.get("route_changes"),
        "straight_path_ticks":           daic_summary.get("straight_path_ticks"),
        "detour_path_ticks":             daic_summary.get("detour_path_ticks"),
        "wall_detect_ticks":             daic_summary.get("wall_detect_ticks"),
        "avoidance_ticks":               daic_summary.get("avoidance_ticks"),
        "occ_peak_cells":                daic_summary.get("occ_peak_cells"),
    }
    stats["waypoint_bearing"] = {
        "samples":                len(route_events),
        "mean_abs_deviation_deg": (round(sum(deviations) / len(deviations), 1)
                                   if deviations else None),
        "max_abs_deviation_deg":  round(max(deviations), 1) if deviations else None,
        "first_samples":          route_events[:5],
        "last_samples":           route_events[-5:],
    }
    stats["classification_hints"] = _classify_hints(stats)
    return stats


def print_diagnosis(diagnosis: dict, file=None) -> None:
    """Print a human-readable report from diagnose_log() output."""
    out = file or sys.stdout
    print("=" * 54, file=out)
    print("  DAIC Benchmark Diagnosis", file=out)
    print("=" * 54, file=out)

    if "error" in diagnosis:
        print(f"  ERROR: {diagnosis['error']}", file=out)
        return

    d = diagnosis
    print(f"  Ticks logged:                 {d['tick_count']}", file=out)
    print(f"  Target-visible ticks:         {d['target_visible_ticks']} "
          f"({d['target_visible_rate']:.1%})", file=out)
    print(f"  Approach-gated ticks:         {d['approach_gated_ticks']}", file=out)
    print(f"  Yaw-scan ticks:               {d['yaw_scan_ticks']}", file=out)
    print(f"  Local-route active ticks:     {d['route_active_ticks']}", file=out)
    print(f"  Forward-zero (alignment):     {d['forward_zero_alignment_ticks']}", file=out)
    print(f"  Avoidance-clamped ticks:      {d['avoidance_clamped_ticks']}", file=out)
    print(f"  Route-active stalled ticks:   {d['route_stalled_ticks']} "
          f"(forward~zero or avoidance-clamped, union)", file=out)
    print(f"  front_occ_m <= {d['front_occ_close_threshold_m']:.1f} m ticks:    "
          f"{d['front_occ_close_ticks']}", file=out)
    if "front_block_occ_close_ticks" in d:
        print(f"  front_block_occ_m <= {d['front_occ_close_threshold_m']:.1f} m ticks: "
              f"{d['front_block_occ_close_ticks']}", file=out)
    print(f"  Longest close-occupancy run:  {d['longest_front_occ_close_run_s']:.1f} s", file=out)
    if d.get("front_occ_close_sources"):
        print("  Close-front occupancy sources:", file=out)
        ages = d.get("front_occ_close_source_age", {})
        for key, count in d["front_occ_close_sources"].items():
            age = ages.get(key) or {}
            suffix = ""
            if age:
                suffix = f" (age median={age.get('median')} max={age.get('max')} ticks)"
            print(f"    {key}: {count}{suffix}", file=out)
    print(file=out)
    print("  From DAIC run reporter (daic/summary.json):", file=out)
    print(f"    A* route changes:           {d['route_changes']}", file=out)
    print(f"    Straight-path ticks:        {d['straight_path_ticks']}", file=out)
    print(f"    Detour-path ticks:          {d['detour_path_ticks']}", file=out)
    print(f"    Wall-detect ticks:          {d['wall_detect_ticks']}", file=out)
    print(f"    Avoidance ticks:            {d['avoidance_ticks']}", file=out)
    print(f"    Peak occupied cells:        {d['occ_peak_cells']}", file=out)
    print(file=out)

    bw = d["waypoint_bearing"]
    print("  Waypoint bearing / deviation (route_log.jsonl):", file=out)
    print(f"    Samples:                    {bw['samples']}", file=out)
    print(f"    Mean |deviation|:           {bw['mean_abs_deviation_deg']} deg", file=out)
    print(f"    Max |deviation|:            {bw['max_abs_deviation_deg']} deg", file=out)
    for tag, samples in (("first", bw["first_samples"]), ("last", bw["last_samples"])):
        for e in samples:
            print(f"      [{tag:>5}] t={e['t']!s:>7} bearing={e['bearing_deg']} "
                  f"direct={e['direct_bearing_deg']} dev={e['deviation_deg']} "
                  f"dist={e['target_dist_m']}", file=out)
    print(file=out)

    print("  Classification hints (confirm against occ_*.png gallery):", file=out)
    for hint in d["classification_hints"]:
        print(f"    - {hint['label']}: {hint['evidence']}", file=out)
    print("=" * 54, file=out)
