"""Flight artefacts: the run summary, the event log, and the track plot.

Everything lands in ``<sim.report_dir>/dway/``. The root is published by the
vehicle, never constructed here, so a real flight and a simulated one are read
out of the same place.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any, Iterable, Sequence

from dcmn import theme
from dcmn.mapview import draw_map_axes

#: Bumped when a field changes meaning. New fields may be added without one.
SUMMARY_SCHEMA_VERSION = 1

OUTCOMES = ("complete", "failed", "aborted")


def repeatability_summary(summaries: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate repeated baseline flights into path and arrival variance."""
    complete = [item for item in summaries if item.get("outcome") == "complete"]
    paths = [float(item["path_length_m"]) for item in complete]
    waypoint_count = min((len(item.get("waypoints", ())) for item in complete),
                         default=0)
    arrivals: list[dict[str, Any]] = []
    for index in range(waypoint_count):
        values = [item["waypoints"][index].get("arrival_s") for item in complete]
        numeric = [float(value) for value in values if value is not None]
        arrivals.append({
            "index": index, "samples": len(numeric),
            "mean_s": round(statistics.fmean(numeric), 6) if numeric else None,
            "variance_s2": round(statistics.pvariance(numeric), 8)
            if numeric else None,
        })
    return {
        "schema_version": 1,
        "runs": len(summaries),
        "complete_runs": len(complete),
        "path_length_mean_m": round(statistics.fmean(paths), 6) if paths else None,
        "path_length_variance_m2": round(statistics.pvariance(paths), 8)
        if paths else None,
        "arrival": arrivals,
    }


def write_repeatability(report_dir: str | Path,
                        summaries: Sequence[dict[str, Any]]) -> Path:
    """Publish the aggregate used by the five-run baseline check."""
    path = Path(report_dir) / "repeatability.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(repeatability_summary(summaries), indent=2,
                               sort_keys=True), encoding="utf-8")
    return path


class FlightRecorder:
    """Append-only event log for one flight.

    Written as it happens rather than at the end, so a run that is killed still
    leaves the evidence of what it was doing when it stopped.
    """

    def __init__(self, report_dir: str | Path) -> None:
        self.dir = Path(report_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._handle = (self.dir / "flight.jsonl").open("w", encoding="utf-8")

    def event(self, t_s: float, kind: str, *, state: Any = None,
              request_id: str | None = None, accepted: bool | None = None,
              reason: str | None = None, **extra: Any) -> None:
        record: dict[str, Any] = {"t_s": round(float(t_s), 4), "event": kind}
        if request_id is not None:
            record["request_id"] = request_id
        if accepted is not None:
            record["accepted"] = bool(accepted)
        if reason:
            record["reason"] = reason
        if state is not None:
            record["state"] = state_snapshot(state)
        record.update(extra)
        self._handle.write(json.dumps(record, sort_keys=True,
                                      separators=(",", ":")) + "\n")
        self._handle.flush()

    def close(self) -> None:
        try:
            self._handle.close()
        except Exception:
            pass


def state_snapshot(state) -> dict[str, Any]:
    """The parts of a vehicle state worth keeping in a log line."""
    position = state.position
    snapshot = {
        "mode": state.mode, "armed": state.armed,
        "heading_deg": round(state.heading_deg, 3),
        "frame": position.frame,
        "vx_mps": round(state.vx_mps, 4), "vy_mps": round(state.vy_mps, 4),
        "vz_mps": round(state.vz_mps, 4),
        "local_position_valid": state.local_position_valid,
        "velocity_valid": state.velocity_valid,
    }
    for name in ("x", "y", "z", "north_m", "east_m", "down_m",
                 "lat_deg", "lon_deg", "alt_m"):
        value = getattr(position, name)
        if value is not None:
            snapshot[name] = round(float(value), 6)
    if state.failsafe_reason:
        snapshot["failsafe_reason"] = state.failsafe_reason
    if state.last_setpoint_age_s is not None:
        snapshot["last_setpoint_age_s"] = round(state.last_setpoint_age_s, 3)
    return snapshot


def write_summary(report_dir: str | Path, summary: dict[str, Any]) -> Path:
    path = Path(report_dir) / "summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True),
                    encoding="utf-8")
    return path


def write_track(report_dir: str | Path, *, planned: Sequence[tuple[float, float]],
                flown: Sequence[tuple[float, float]], sim_map=None,
                title: str = "") -> Path | None:
    """Planned versus flown, in map coordinates. Returns None without matplotlib."""
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
    except ImportError:
        return None
    if not planned and not flown:
        return None

    width = float(getattr(sim_map, "width", 0.0)) or _extent(planned, flown, axis=0)
    height = float(getattr(sim_map, "height", 0.0)) or _extent(planned, flown, axis=1)
    figure = Figure(figsize=(10, max(4.0, 10.0 * height / max(width, 1.0))),
                    facecolor=theme.BG)
    canvas = FigureCanvasAgg(figure)
    axes = figure.add_subplot(111, facecolor=theme.CANVAS)
    axes.set_xlim(0, width)
    axes.set_ylim(height, 0)
    axes.set_aspect("equal")
    axes.tick_params(colors=theme.DIM)
    for spine in axes.spines.values():
        spine.set_color(theme.GRID)
    axes.set_xlabel("X (m)", color=theme.DIM)
    axes.set_ylabel("Y (m)", color=theme.DIM)

    # The same map every window draws, so a report and a live view show the
    # same world rather than two drawings of it.
    if sim_map is not None:
        draw_map_axes(axes, sim_map)

    if planned:
        axes.plot([p[0] for p in planned], [p[1] for p in planned], "--o",
                  color=theme.WARN, markersize=5, linewidth=1.4, zorder=4,
                  label="planned")
    if flown:
        axes.plot([p[0] for p in flown], [p[1] for p in flown], "-",
                  color=theme.ACCENT, linewidth=1.8, zorder=5, label="flown")
        axes.plot(flown[0][0], flown[0][1], "o", color=theme.OK, markersize=8,
                  zorder=6, label="start")
        axes.plot(flown[-1][0], flown[-1][1], "s", color=theme.DANGER,
                  markersize=8, zorder=6, label="end")
    # Lower right: the legend sat over the middle of the map, which is exactly
    # where a track usually is.
    axes.legend(loc="lower right", facecolor=theme.BUTTON, edgecolor=theme.GRID,
                labelcolor=theme.TEXT, fontsize=8, framealpha=0.92)
    if title:
        axes.set_title(title, color=theme.TEXT, fontsize=9, pad=8)

    path = Path(report_dir) / "track.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.print_figure(str(path), dpi=150, bbox_inches="tight",
                        facecolor=theme.BG)
    return path


def _extent(*series: Iterable[tuple[float, float]], axis: int = 0) -> float:
    values = [point[axis] for group in series for point in group]
    return max(values) + 1.0 if values else 1.0
