"""The Status tab: a sim-time strip of phases, goals and executor stops, and module health.

The strip reads left to right from Start: one coloured band per mission phase,
a tick per goal set, a tick per dway stop and per intent, so "why did it take
so long" is a picture before it is a log. Below it, the timeline as a table,
the budget curves' current values, and `PipelineView`'s members -- which
modules are present, their state, and how long since each was heard.
"""
from __future__ import annotations

from dcmn import theme

PHASE_COLOURS = {'idle': theme.GRID, 'starting': theme.CAUTION, 'running': theme.ACCENT, 'paused': theme.WARN,
                 'finishing': theme.GOAL, 'done': theme.OK}
MARK_COLOURS = {'goal': theme.GOAL, 'stop': theme.DANGER, 'intent': theme.WARN, 'coverage': theme.START,
                'phase': theme.DIM, 'dway': theme.DIM}
STRIP_HEIGHT = 64
CURVE_HEIGHT = 90
TABLE_ROWS = 200


class StatusTab:
    def __init__(self, parent, run, tk, ttk) -> None:
        self.run, self.tk, self.ttk = run, tk, ttk
        page = ttk.Frame(parent)
        page.grid(row=0, column=0, sticky='nsew')
        parent.rowconfigure(0, weight=1); parent.columnconfigure(0, weight=1)
        page.columnconfigure(0, weight=1); page.rowconfigure(2, weight=1)
        strip = ttk.LabelFrame(page, text='Timeline (sim time)', padding=6)
        strip.grid(row=0, column=0, sticky='ew', pady=(0, 6))
        strip.columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(strip, height=STRIP_HEIGHT, background=theme.CANVAS, highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky='ew')
        self.curves = tk.Canvas(strip, height=CURVE_HEIGHT, background=theme.CANVAS, highlightthickness=0)
        self.curves.grid(row=1, column=0, sticky='ew', pady=(4, 0))
        self.readout = tk.StringVar(value='--')
        ttk.Label(page, textvariable=self.readout, style='Dim.TLabel').grid(row=1, column=0, sticky='w', pady=(0, 6))
        lower = ttk.Frame(page)
        lower.grid(row=2, column=0, sticky='nsew')
        lower.columnconfigure(0, weight=3); lower.columnconfigure(1, weight=2); lower.rowconfigure(0, weight=1)
        self.table = ttk.Treeview(lower, columns=('time', 'kind', 'what'), show='headings', height=14)
        for column, width in (('time', 80), ('kind', 80), ('what', 520)):
            self.table.heading(column, text=column)
            self.table.column(column, width=width, anchor='w', stretch=column == 'what')
        self.table.grid(row=0, column=0, sticky='nsew', padx=(0, 6))
        self.modules = ttk.Treeview(lower, columns=('role', 'module', 'state', 'age'), show='headings', height=14)
        for column, width in (('role', 80), ('module', 80), ('state', 120), ('age', 60)):
            self.modules.heading(column, text=column)
            self.modules.column(column, width=width, anchor='w')
        self.modules.grid(row=0, column=1, sticky='nsew')
        self._shown_marks = 0

    def paint(self) -> None:
        run = self.run
        self._paint_strip()
        self._paint_curves()
        marks = run.timeline
        if len(marks) != self._shown_marks:
            self.table.delete(*self.table.get_children())
            for mark in marks[-TABLE_ROWS:]:
                self.table.insert('', 0, values=(f"{mark['time_s']:.2f}", mark['kind'], mark['text']))
            self._shown_marks = len(marks)
        self.modules.delete(*self.modules.get_children())
        for member, age in run.pipeline.members():
            self.modules.insert('', 'end', values=(member.role, member.implementation, member.state, f'{age:.1f}s'))
        elapsed = 0.0 if run.started_s is None else max(0.0, run.sim_time_s() - run.started_s)
        battery = run.observation.battery_pct
        self.readout.set(f'phase {run.phase} · elapsed {elapsed:.1f} s · flown {run.path_m:.1f} m · battery '
                         f"{'--' if battery is None else format(battery, '.1f') + '%'} · goals {len(run.goals)} · "
                         f'intents {run.intents.revision}')

    def _paint_curves(self) -> None:
        """Battery (0-100 %) and flown path (0 to its maximum) on the strip's time axis."""
        run, canvas = self.run, self.curves
        canvas.delete('all')
        if not run.phases: return
        width = max(canvas.winfo_width(), 200)
        start = run.phases[0]['start_s']
        end = max(run.sim_time_s(), start + 1e-6)
        x = lambda t: 4 + (t - start) * (width - 8) / (end - start)
        top, bottom = 6, CURVE_HEIGHT - 16
        for series, colour, label, scale in ((run.battery_series, theme.OK, 'battery %', 100.),
                                             (run.path_series, theme.ACCENT, 'flown m', None)):
            spans = series.spans()
            if not spans: continue
            high = scale or max(max(s.maximum for s in spans), 1e-6)
            y = lambda v, high=high: bottom - (bottom - top) * max(0., min(1., v / high))
            points = [c for s in spans for c in (x(s.x), y(s.mean))]
            if len(points) >= 4: canvas.create_line(*points, fill=colour, width=2)
            canvas.create_text(width - 4, top if scale else top + 14, anchor='ne', fill=colour,
                               text=f'{label} {spans[-1].mean:.1f}' + ('' if scale else f' (max {high:.1f})'))

    def _paint_strip(self) -> None:
        run, canvas = self.run, self.canvas
        canvas.delete('all')
        width = max(canvas.winfo_width(), 200)
        phases = run.phases
        if not phases: return
        start = phases[0]['start_s']
        end = max(run.sim_time_s(), start + 1e-6)
        scale = (width - 8) / (end - start)
        x = lambda t: 4 + (t - start) * scale
        for phase in phases:
            x0, x1 = x(phase['start_s']), x(phase['end_s'] if phase['end_s'] is not None else end)
            canvas.create_rectangle(x0, 6, max(x1, x0 + 1), 26, fill=PHASE_COLOURS.get(phase['phase'], theme.GRID),
                                    outline='')
            if x1 - x0 > 40: canvas.create_text(x0 + 3, 16, text=phase['phase'], anchor='w', fill=theme.CANVAS)
        for mark in run.timeline:
            if mark['kind'] == 'phase': continue
            px = x(mark['time_s'])
            canvas.create_line(px, 30, px, 58, fill=MARK_COLOURS.get(mark['kind'], theme.DIM), width=2)
        canvas.create_text(4, STRIP_HEIGHT - 2, text=f'{start:.0f} s', anchor='sw', fill=theme.DIM)
        canvas.create_text(width - 4, STRIP_HEIGHT - 2, text=f'{end:.0f} s', anchor='se', fill=theme.DIM)
