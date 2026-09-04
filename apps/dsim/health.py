"""How well the simulation and everything attached to it are keeping up.

The simulator knows two things nobody else does: how fast simulated time is
actually advancing against the wall clock, and how long a frame costs. Together
those answer the question an operator asks first -- *I asked for ten times real
time and did not get it, where did it go?* -- because a shortfall in the first
with a large second is the renderer, and a shortfall with a small second is
something else.

Every attached module already announces itself once a second. Those heartbeats
now carry what the module meant to sample and what it managed, so the same
projection that draws the pipeline list also says who is falling behind.

**Nothing here gates anything.** It is measurement, published one way, and the
simulator never waits on any of it.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from dcmn.health import (BAD, OK, UNKNOWN, SteadyGrade, describe, grade,
                         worst)
from dcmn import report_html, theme
from dcmn.series import EnvelopeSeries

#: Wall seconds between samples. Matches the heartbeat cadence, so every sample
#: sees fresh module reports, and keeps the log readable for a long run.
SAMPLE_PERIOD_S = 1.0

#: Buckets kept per tracked quantity. At one sample a second this is a little
#: over half an hour before the first compression, and bounded thereafter.
SERIES_CAPACITY = 2000


class SimulationHealth:
    """Samples the simulator and its pipeline, and writes the record as it goes."""

    def __init__(self, report_dir: Path | None = None, *,
                 capacity: int = SERIES_CAPACITY) -> None:
        self.speed = EnvelopeSeries(capacity)
        self.speed_ratio = EnvelopeSeries(capacity)
        self.render_ms = EnvelopeSeries(capacity)
        self.tick_hz = EnvelopeSeries(capacity)
        #: One rate series per process. Two instances of the same module must
        #: not be folded into one jagged, misleading line.
        self.modules: dict[str, EnvelopeSeries] = {}
        self.module_facts: dict[str, dict[str, Any]] = {}
        self.grade = SteadyGrade(runs=3)
        self.samples = 0
        self.achieved: float | None = None
        self.requested: float = 1.0

        self._ticks = 0
        self._render_total_s = 0.0
        self._render_count = 0
        self._last_wall: float | None = None
        self._last_sim: float | None = None
        self._next_sample_wall = -1e9
        self._started_wall: float | None = None
        self._handle = None
        if report_dir is not None:
            # Appended as it happens, not written at the end: a run somebody
            # had to interrupt is exactly the run whose health is worth
            # reading, and it is the same reason dway streams its flight log.
            path = Path(report_dir)
            path.mkdir(parents=True, exist_ok=True)
            self._handle = (path / "health.jsonl").open("w", encoding="utf-8")

    # -- collection -----------------------------------------------------

    def note_tick(self, render_s: float | None = None) -> None:
        """One physics tick, and what its frame cost if it published one."""
        self._ticks += 1
        if render_s is not None:
            self._render_total_s += render_s
            self._render_count += 1

    def due(self, wall_now: float) -> bool:
        return wall_now - self._next_sample_wall >= SAMPLE_PERIOD_S

    def sample(self, *, wall_now: float, sim_now: float,
               requested: float, members=(),
               member_expiry_s: float = 3.0) -> dict[str, Any] | None:
        """Close a window and record it. Returns the sample, or None if early."""
        if not self.due(wall_now):
            return None
        self._next_sample_wall = wall_now
        if self._started_wall is None:
            self._started_wall = wall_now
        previous_wall, self._last_wall = self._last_wall, wall_now
        previous_sim, self._last_sim = self._last_sim, sim_now
        self.requested = float(requested)

        wall_elapsed = None if previous_wall is None else wall_now - previous_wall
        sim_elapsed = None if previous_sim is None else sim_now - previous_sim
        # Never divide by an interval without checking it is positive: the
        # first sample has no window, and a restarted simulator runs backwards.
        if wall_elapsed and wall_elapsed > 0.0 and sim_elapsed is not None \
                and sim_elapsed >= 0.0:
            self.achieved = sim_elapsed / wall_elapsed
            self.speed.add(sim_now, self.achieved)
            self.speed_ratio.add(sim_now, 1.0 if self.requested <= 0.0 else
                                 self.achieved / self.requested)
            self.tick_hz.add(sim_now, self._ticks / wall_elapsed)
        if self._render_count:
            self.render_ms.add(sim_now,
                               1000.0 * self._render_total_s / self._render_count)
        self._ticks = 0
        self._render_total_s = 0.0
        self._render_count = 0

        modules = self._read_modules(sim_now, members, member_expiry_s)
        observed = worst([self._speed_grade(), *(m["grade"] for m in modules)])
        overall = self.grade.update(observed)
        self.samples += 1

        record = {
            "t_sim_s": round(sim_now, 3),
            "t_wall_s": round(wall_now - self._started_wall, 3),
            "speed": {"requested": self.requested,
                      "achieved": None if self.achieved is None
                      else round(self.achieved, 4),
                      "grade": self._speed_grade()},
            "render_ms": self.render_ms.latest(),
            "tick_hz": self.tick_hz.latest(),
            "modules": modules,
            "grade": overall,
        }
        self._write(record)
        return record

    def _speed_grade(self) -> str:
        """How the simulator itself is doing against the pace it was asked for.

        ``max`` has no target -- "as fast as you can" is met by definition --
        so it is graded ``ok`` rather than measured against a number nobody
        chose. Real time's target is 1.0, and it falls short exactly when a
        tick overruns the timestep clamp.
        """
        if self.achieved is None:
            return UNKNOWN
        if self.requested <= 0.0:
            return OK
        return grade(self.achieved, self.requested)

    def _read_modules(self, sim_now: float, members,
                      expiry_s: float) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for member, age in members:
            key = member.process_id
            intake = describe(getattr(member, "intake", None))
            expired = age > expiry_s
            record = {
                "role": member.role,
                "implementation": member.implementation,
                "process_id": member.process_id,
                "state": member.state,
                "age_s": round(age, 3),
                **intake,
            }
            if expired:
                record["state"] = "expired"
                record["grade"] = BAD
            wanted, achieved = intake["wanted_hz"], intake["achieved_hz"]
            if wanted and achieved is not None:
                self.modules.setdefault(key, EnvelopeSeries(
                    self.speed.capacity)).add(sim_now, achieved / wanted)
            self.module_facts[key] = record
            out.append(record)
        return out

    # -- output ---------------------------------------------------------

    def _write(self, record: dict[str, Any]) -> None:
        if self._handle is None:
            return
        try:
            self._handle.write(json.dumps(record, separators=(",", ":")) + "\n")
            self._handle.flush()
        except Exception:
            # A run must not be lost because its diagnostics could not be.
            self._handle = None

    def summary(self) -> dict[str, Any]:
        """What the run looked like overall, for the report and summary.json."""
        speed_extremes = self.speed.extremes()
        render_extremes = self.render_ms.extremes()
        modules = []
        for key, record in sorted(self.module_facts.items()):
            series = self.modules.get(key)
            worst_ratio = None if series is None else series.extremes()[0]
            modules.append({
                **record,
                "worst_ratio": None if worst_ratio is None else round(worst_ratio, 4),
            })
        return {
            "samples": self.samples,
            "requested_speed": self.requested,
            "achieved_speed": {
                "last": None if self.achieved is None else round(self.achieved, 4),
                "mean": _series_mean(self.speed),
                "min": None if speed_extremes is None else round(speed_extremes[0], 4),
                "max": None if speed_extremes is None else round(speed_extremes[1], 4),
            },
            "render_ms": {
                "mean": _series_mean(self.render_ms),
                "min": None if render_extremes is None else round(render_extremes[0], 3),
                "max": None if render_extremes is None else round(render_extremes[1], 3),
            },
            "grade": self.grade.value,
            "modules": modules,
        }

    def close(self) -> None:
        if self._handle is not None:
            try:
                self._handle.close()
            except Exception:
                pass
            self._handle = None

    def write_report(self, report_dir: Path) -> Path:
        """Write the compact chart and HTML view; JSON remains authoritative."""
        report_dir = Path(report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        chart = self._write_chart(report_dir / "health.png")
        summary = self.summary()

        speed = summary["achieved_speed"]
        module_rows = []
        for module in summary["modules"]:
            achieved, wanted = module.get("achieved_hz"), module.get("wanted_hz")
            basis = "wall" if module.get("basis") == "wall" else "sim"
            rate = "-" if achieved is None or not wanted else \
                f"{achieved:.2f} / {wanted:.2f} Hz ({basis})"
            module_rows.append((
                report_html.esc(module["role"]),
                report_html.esc(module["implementation"]),
                report_html.esc(module["state"]),
                report_html.graded(rate, module["grade"]),
                report_html.esc(module.get("skipped", 0)),
                report_html.esc(module.get("overruns", 0)),
            ))
        blocks = [report_html.section("Run health", report_html.facts((
            ("Overall", report_html.graded(summary["grade"], summary["grade"])),
            ("Requested pace", f'{summary["requested_speed"]:.2f}x'),
            ("Mean achieved", "-" if speed["mean"] is None else
             f'{speed["mean"]:.2f}x'),
            ("Minimum achieved", "-" if speed["min"] is None else
             f'{speed["min"]:.2f}x'),
            ("Samples", summary["samples"]),
        )))]
        if chart is not None:
            blocks.append(report_html.section(
                "Timeline", report_html.figure(chart,
                "Achieved simulator pace and module intake ratios; the shaded envelope preserves short spikes after compression.")))
        blocks.append(report_html.section("Modules", report_html.table(
            ("Role", "Implementation", "State", "Achieved / wanted",
             "Skipped", "Overruns"), module_rows, numeric=(4, 5))))
        output = report_dir / "report.html"
        output.write_text(report_html.document(
            "Simulation health", subtitle="Keeping-up diagnostics; reporting never gates the run.",
            blocks=blocks), encoding="utf-8")
        return output

    def _write_chart(self, path: Path) -> Path | None:
        try:
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            from matplotlib.figure import Figure
        except ImportError:
            return None
        if not len(self.speed) and not self.modules:
            return None
        figure = Figure(figsize=(10, 4.8), facecolor=theme.BG)
        canvas = FigureCanvasAgg(figure)
        axes = figure.add_subplot(111, facecolor=theme.CANVAS)
        self._plot_ratio(axes, self.speed_ratio, 1.0, "simulator")
        for process_id, series in sorted(self.modules.items()):
            facts = self.module_facts.get(process_id, {})
            label = f'{facts.get("role", "module")}:{facts.get("implementation", process_id[:8])}'
            self._plot_ratio(axes, series, 1.0, label)
        axes.axhline(1.0, color=theme.OK, linewidth=1, linestyle="--")
        axes.axhline(0.9, color=theme.WARN, linewidth=1, linestyle=":")
        axes.set_xlabel("simulated time (s)", color=theme.DIM)
        axes.set_ylabel("achieved / wanted", color=theme.DIM)
        axes.tick_params(colors=theme.DIM)
        for spine in axes.spines.values():
            spine.set_color(theme.GRID)
        axes.legend(facecolor=theme.BUTTON, edgecolor=theme.GRID,
                    labelcolor=theme.TEXT, fontsize=8)
        canvas.print_figure(str(path), dpi=130, bbox_inches="tight",
                            facecolor=theme.BG)
        return path

    @staticmethod
    def _plot_ratio(axes, series: EnvelopeSeries, divisor: float,
                    label: str) -> None:
        x, mean, minimum, maximum = series.plot_arrays()
        if not x:
            return
        divisor = divisor if divisor > 0.0 else 1.0
        mean = [value / divisor for value in mean]
        minimum = [value / divisor for value in minimum]
        maximum = [value / divisor for value in maximum]
        line, = axes.plot(x, mean, linewidth=1.5, label=label)
        axes.fill_between(x, minimum, maximum, color=line.get_color(), alpha=.16)


def _series_mean(series: EnvelopeSeries) -> float | None:
    spans = series.spans()
    if not spans:
        return None
    total = sum(s.mean * s.count for s in spans)
    count = sum(s.count for s in spans)
    return round(total / count, 4)


def now_wall() -> float:
    """The clock health sampling runs on. Liveness is a wall-clock question."""
    return time.monotonic()


__all__ = ["SimulationHealth", "SAMPLE_PERIOD_S", "SERIES_CAPACITY",
           "now_wall", "BAD", "OK"]
