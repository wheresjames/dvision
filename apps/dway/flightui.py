"""The live window and headless runner for dynamic dway flight.

Everything here is for looking at and operating a flight that
:class:`~dway.executor.DynamicExecutor` is already flying. The window never
touches the vehicle: it calls the same ``request`` methods the headless
runner calls, and paints the immutable view the executor publishes each
step. Painting is paced independently of the control loop, and a repaint
or a report can never block a step -- the executor owns the loop, the window
only schedules the next one.

    ./apps/dway/dway.py --id area1 --mode dynamic --execution-profile <profile>
    ./apps/dway/dway.py --id area1 --mode dynamic --no-ui --start ...

Start/Pause/Resume/Cancel are local to this process; a rejected HOLD, a lost
lease or an expired permission stops the vehicle and says why, and the window
closing requests HOLD, waits a bounded time for the measured stop, releases
control and finishes the recording. It never lands, RTLs or disarms.
"""
from __future__ import annotations

import math
import signal
import threading
import time

from dcmn.navigation import ExecutionProfile
from dcmn.pacing import MAP_HZ, TEXT_HZ
from dway.link import DsimLink

#: How much of the recent past the timeline shows; older history is in the archive.
TIMELINE_S = 60.0
#: The timeline's fixed plot height; the right column scrolls rather than shrink it.
TIMELINE_HEIGHT = 170

#: One line under the map: what each symbol means.
LEGEND = ('Dashed: proposed route   Solid: active route   Thick: permitted interval   '
          'Square: stop point   Cross: commanded target   Trail: flown track')

#: The sidebar cards: (title, ((field, label), ...)).
CARDS = (
    ('Flight', (('mission', 'Mission'), ('hold', 'Hold'), ('launch', 'Launch'), ('targets', 'Targets sent'))),
    ('Motion', (('speed', 'Speed'), ('altitude', 'Altitude'), ('heading', 'Heading'), ('tracking', 'Cross-track'))),
    ('Route', (('admission', 'Admission'), ('revision', 'Revision'), ('remaining', 'Remaining'),
               ('margin', 'Stop margin'), ('validity', 'Permission'))),
    ('System', (('lease', 'Control'), ('profile', 'Profile'), ('recording', 'Recording'), ('vehicle', 'Vehicle'))),
)


def _fmt(value, unit='', digits=2):
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        return 'n/a'
    return f'{value:.{digits}f}{unit}'


def state_colors(theme):
    """Badge and timeline colour for each executor state."""
    return {'EXECUTING': theme.OK, 'COMPLETE': theme.OK, 'BRAKING': theme.WARN, 'HOLDING': theme.CAUTION,
            'TAKING_OFF': theme.ACCENT, 'FAILED': theme.DANGER, 'CANCELLED': theme.DIM,
            'READY': theme.ACCENT_BUTTON, 'WAITING': theme.BUTTON_ACTIVE}


def describe_event(event):
    """(title, detail, tag) for one recent executor event."""
    kind, d = event['kind'], event['data']
    if kind == 'execution.control':
        return (f"{d.get('action', '').capitalize()} {'accepted' if d.get('accepted') else 'refused'}",
                f"{d.get('origin', '')}: {d.get('reason', '')}", '' if d.get('accepted') else 'warn')
    if kind == 'execution.transition':
        state = d.get('state', '')
        return state, d.get('reason', ''), 'bad' if state == 'FAILED' else 'ok' if state == 'COMPLETE' else ''
    if kind == 'execution.hold':
        return (f"Hold {d.get('phase', '')}", f"{d.get('kind') or ''}: {d.get('reason') or d.get('result') or ''}",
                'bad' if d.get('phase') == 'rejected' else 'warn')
    if kind == 'execution.lease':
        return f"Lease {d.get('event', '')}", d.get('reason') or d.get('owner') or '', 'bad' if d.get('event') == 'lost' else ''
    if kind == 'execution.health':
        return 'Health', d.get('fault', ''), 'bad'
    if kind == 'execution.route':
        return 'Route', f"{d.get('source_state', '')}: {d.get('source_reason', '')}", ''
    if kind == 'execution.launch':
        return f"Launch {d.get('phase', '')}", d.get('reason') or '', ''
    if kind == 'execution.turn':
        return 'Turn', f"{_fmt(d.get('from_deg'), '°', 0)} → {_fmt(d.get('to_deg'), '°', 0)}", ''
    if kind == 'execution.readiness':
        return 'Ready', d.get('role', ''), ''
    if kind == 'execution.coverage':
        return 'Coverage', 'mapping reset requested' if d.get('requested') else 'goal inside coverage', ''
    if kind in ('execution.shutdown', 'execution.shutdown_requested'):
        return 'Shutdown', d.get('reason', ''), 'warn'
    return kind.replace('execution.', '').capitalize(), '', ''


class FlightWindow:
    """Header with state and controls, the live map, grouped readouts, recent events and a timeline.

    Everything is drawn from the immutable view the executor publishes; the
    buttons only queue the same requests the headless runner uses.
    """

    def __init__(self, executor, root=None, *, control_thread=True):
        import tkinter as tk
        from tkinter import ttk
        from dcmn import theme
        from dcmn.map_pane import MapPane
        from dcmn.pacing import Paced
        from dcmn.tktheme import apply_theme
        from dcmn.window import restore_window_pos
        self.executor, self.tk, self.theme = executor, tk, theme
        self.closed = False
        self.selected_event = None
        self.colors = state_colors(theme)
        self.root = root if root is not None else tk.Tk()
        mode = 'target' if executor.auto else 'dynamic'
        self.root.title(f'dway {mode} flight — {executor.id}')
        style = apply_theme(self.root)
        style.configure('Card.TFrame', background=theme.PANEL)
        style.configure('CardTitle.TLabel', background=theme.PANEL, foreground=theme.DIM,
                        font=('TkDefaultFont', 9, 'bold'))
        # Tight rows in the readout cards: no label padding, the font sets the line height.
        style.configure('CardKey.TLabel', background=theme.PANEL, foreground=theme.DIM, padding=0,
                        font=('TkDefaultFont', 9))
        style.configure('CardValue.TLabel', background=theme.PANEL, foreground=theme.TEXT, padding=0,
                        font=('TkDefaultFont', 9))
        style.configure('Status.TLabel', background=theme.PANEL, foreground=theme.DIM, padding=(10, 3))
        self.root.protocol('WM_DELETE_WINDOW', self.close)
        self.root.minsize(1100, 680)
        restore_window_pos(self.root, f'dway.dynamic.{executor.id}')
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        # -- header: identity, state, controls ------------------------------
        header = ttk.Frame(self.root, style='Header.TFrame', padding=(14, 10))
        header.grid(row=0, column=0, sticky='ew')
        header.columnconfigure(2, weight=1)
        ttk.Label(header, text=f'dway  ·  {mode} flight  ·  {executor.id}', style='Brand.TLabel').grid(
            row=0, column=0, sticky='w', padx=(0, 16))
        self.state_badge = tk.Label(header, text='WAITING', font=('TkDefaultFont', 10, 'bold'), padx=12, pady=3,
                                    bg=theme.BUTTON_ACTIVE, fg=theme.TEXT, borderwidth=0)
        self.state_badge.grid(row=0, column=1, sticky='w')
        self.state_var = tk.StringVar(value='')
        ttk.Label(header, textvariable=self.state_var, style='HeaderDim.TLabel').grid(
            row=1, column=0, columnspan=3, sticky='w', pady=(6, 0))
        controls = ttk.Frame(header, style='Header.TFrame')
        controls.grid(row=0, column=3, rowspan=2, sticky='e')
        self.buttons = {}
        for column, (action, style_name) in enumerate((('start', 'Accent.TButton'), ('pause', 'TButton'),
                                                       ('resume', 'TButton'), ('cancel', 'Danger.TButton'))):
            button = ttk.Button(controls, text=action.capitalize(), style=style_name, width=8,
                                command=lambda a=action: executor.request(a, 'window'))
            button.grid(row=0, column=column, padx=(6, 0))
            self.buttons[action] = button
        self.controls_var = tk.StringVar(value='No operator actions yet')
        ttk.Label(controls, textvariable=self.controls_var, style='HeaderDim.TLabel').grid(
            row=1, column=0, columnspan=4, sticky='e', pady=(6, 0))

        # -- body: left column (map, events) | right column (cards, timeline) --
        body = ttk.Frame(self.root, padding=(12, 12, 12, 8))
        body.grid(row=1, column=0, sticky='nsew')
        body.columnconfigure(0, weight=1)
        body.rowconfigure(0, weight=1)
        left = ttk.Frame(body)
        left.grid(row=0, column=0, sticky='nsew')
        left.columnconfigure(0, weight=1)
        left.rowconfigure(0, weight=3)
        left.rowconfigure(2, weight=1)
        self.pane = MapPane(left, width=720, height=430, title='Live map')
        self.pane.widget.grid(row=0, column=0, sticky='nsew')
        ttk.Label(left, text=LEGEND, style='Dim.TLabel').grid(row=1, column=0, sticky='w', pady=(6, 8))

        # The right column scrolls when the window is too short, so no card is
        # squashed: a canvas holds the cards, the scrollbar shows only when needed.
        column = ttk.Frame(body)
        column.grid(row=0, column=1, sticky='nsew', padx=(12, 0))
        column.rowconfigure(0, weight=1)
        self.side_canvas = tk.Canvas(column, background=theme.BG, highlightthickness=0, borderwidth=0)
        self.side_canvas.grid(row=0, column=0, sticky='nsew')
        self.side_scroll = ttk.Scrollbar(column, orient='vertical', command=self.side_canvas.yview)
        self.side_canvas.configure(yscrollcommand=self._side_scrolled)
        side = ttk.Frame(self.side_canvas)
        side.columnconfigure(0, weight=1)
        self.side = side
        self._side_window = self.side_canvas.create_window(0, 0, window=side, anchor='nw')
        side.bind('<Configure>', self._side_resized)
        self.side_canvas.bind('<Configure>', lambda e: self.side_canvas.itemconfigure(self._side_window,
                                                                                      width=e.width))
        for widget in (self.side_canvas, side):
            widget.bind('<Enter>', lambda _e: self._bind_wheel(True))
            widget.bind('<Leave>', lambda _e: self._bind_wheel(False))
        self.fields = {}
        for row, (title, rows) in enumerate(CARDS):
            card = ttk.Frame(side, style='Card.TFrame', padding=(12, 5))
            card.grid(row=row, column=0, sticky='ew', pady=(0, 5))
            card.columnconfigure(1, weight=1)
            ttk.Label(card, text=title.upper(), style='CardTitle.TLabel').grid(
                row=0, column=0, columnspan=2, sticky='w', pady=(0, 1))
            for index, (name, label) in enumerate(rows, start=1):
                ttk.Label(card, text=label, style='CardKey.TLabel', width=12).grid(row=index, column=0, sticky='nw')
                var = tk.StringVar(value='—')
                ttk.Label(card, textvariable=var, style='CardValue.TLabel', wraplength=270,
                          justify='left').grid(row=index, column=1, sticky='w', pady=0)
                self.fields[name] = (label, var)

        events = ttk.Frame(left, style='Card.TFrame', padding=(12, 8))
        events.grid(row=2, column=0, sticky='nsew')
        events.columnconfigure(0, weight=1)
        events.rowconfigure(1, weight=1)
        ttk.Label(events, text='RECENT EVENTS', style='CardTitle.TLabel').grid(row=0, column=0, sticky='w', pady=(0, 4))
        self.events = ttk.Treeview(events, columns=('time', 'event', 'detail'), show='headings', height=5,
                                   selectmode='browse')
        for column, heading, width, stretch in (('time', 'Time', 70, False), ('event', 'Event', 130, False),
                                                ('detail', 'Detail', 420, True)):
            self.events.heading(column, text=heading, anchor='w')
            self.events.column(column, width=width, stretch=stretch, anchor='w')
        self.events.tag_configure('bad', foreground=theme.DANGER)
        self.events.tag_configure('warn', foreground=theme.WARN)
        self.events.tag_configure('ok', foreground=theme.OK)
        self.events.grid(row=1, column=0, sticky='nsew')
        scroll = ttk.Scrollbar(events, orient='vertical', command=self.events.yview)
        scroll.grid(row=1, column=1, sticky='ns')
        self.events.configure(yscrollcommand=scroll.set)
        self.events.bind('<<TreeviewSelect>>', self._select_event)
        self.event_var = tk.StringVar(value='Select an event to see its details and route')
        ttk.Label(events, textvariable=self.event_var, style='CardKey.TLabel', wraplength=700,
                  justify='left').grid(row=2, column=0, columnspan=2, sticky='ew', pady=(6, 0))
        self._event_rows = {}

        # -- timeline: the last card in the right column, taking what height is left --
        timeline = ttk.Frame(side, style='Card.TFrame', padding=(12, 5))
        timeline.grid(row=len(CARDS), column=0, sticky='ew')
        timeline.columnconfigure(0, weight=1)
        ttk.Label(timeline, text=f'TIMELINE  ·  LAST {TIMELINE_S:.0f} S', style='CardTitle.TLabel').grid(
            row=0, column=0, sticky='w')
        self.timeline = tk.Canvas(timeline, width=360, height=TIMELINE_HEIGHT, background=theme.CANVAS,
                                  highlightthickness=0, borderwidth=0)
        self.timeline.grid(row=1, column=0, sticky='ew', pady=(4, 0))
        self.drawing_var = tk.StringVar(value='view: waiting')
        ttk.Label(self.root, textvariable=self.drawing_var, style='Status.TLabel', anchor='w').grid(
            row=2, column=0, sticky='ew')

        self._text = Paced(TEXT_HZ)
        self._events_seen = None
        self._painted_sequence = None
        self.views_painted = self.views_coalesced = 0
        self.control_error = ''
        # The control loop runs on its own thread: a slow repaint, an event
        # list rebuild or a window drag never delays a step. Tk stays on this
        # thread and only reads the immutable view the executor publishes.
        # ``control_thread=False`` lets a deterministic fixture drive the steps.
        self._stop = threading.Event()
        self._thread = None
        if control_thread:
            self._thread = threading.Thread(target=self._control_loop, name='dway-control', daemon=True)
            self._thread.start()
        self.root.after(50, self.tick)

    # -- the scrolling right column --------------------------------------

    def _side_resized(self, _event=None):
        canvas = self.side_canvas
        canvas.configure(scrollregion=canvas.bbox('all'), width=self.side.winfo_reqwidth())

    def _side_scrolled(self, first, last):
        self.side_scroll.set(first, last)
        # Show the scrollbar only when the cards do not fit.
        if float(first) <= 0. and float(last) >= 1.:
            self.side_scroll.grid_remove()
        else:
            self.side_scroll.grid(row=0, column=1, sticky='ns', padx=(4, 0))

    def _bind_wheel(self, active):
        canvas = self.side_canvas
        if not active:
            for sequence in ('<MouseWheel>', '<Button-4>', '<Button-5>'):
                canvas.unbind_all(sequence)
            return
        def wheel(event):
            if canvas.yview() == (0.0, 1.0): return
            delta = -1 if getattr(event, 'num', 0) == 4 or getattr(event, 'delta', 0) > 0 else 1
            canvas.yview_scroll(delta, 'units')
        for sequence in ('<MouseWheel>', '<Button-4>', '<Button-5>'):
            canvas.bind_all(sequence, wheel)

    # -- painting -------------------------------------------------------

    def _overlay(self, view):
        from dcmn.map_pane import Overlay
        return Overlay(
            route=[p[:2] for p in view['proposal']], proposal=True,
            candidate=[p[:2] for p in (self.selected_event or {}).get('points') or ()],
            permitted=[p[:2] for p in view['permitted']],
            track=[p for p in view['track'] if p is not None and len(p) == 2],
            highlight=[p[:2] for p in view['highlight']],
            target=None if view['target'] is None else tuple(view['target'][:2]),
            vehicle=view['vehicle'], goal=None if not view['goal'] else tuple(view['goal'][:2]),
            inflation_m=view['status'].get('tracking_allowance_m', 0.))

    def _values(self, view):
        s, p = view['status'], self.executor.profile
        disposition = s.get('disposition') or {}
        pose = s.get('pose') or {}
        launch = s.get('launch_phase')
        heading = _fmt(s.get('heading_deg'), '°', 0)
        if s.get('commanded_heading_deg') is not None:
            heading += f"  →  {_fmt(s.get('commanded_heading_deg'), '°', 0)}"
        if s.get('turning'):
            heading += '  (turning)'
        return dict(
            mission=f"{s.get('state', 'WAITING')}  ·  {s.get('missions', 0)} started",
            hold=s.get('hold_kind') or '—',
            launch=launch or ('not used' if not s.get('auto') else 'pending'),
            targets=str(s.get('targets_sent', 0)),
            speed=f"{_fmt(s.get('speed_mps'), ' m/s')} of {_fmt(s.get('speed_cap_mps'), ' m/s')}",
            altitude=f"{_fmt(pose.get('z_m'), ' m')}  (slab {p.altitude_m - p.half_height_m:.2f}–"
                     f"{p.altitude_m + p.half_height_m:.2f} m)",
            heading=heading,
            tracking=f"{_fmt(s.get('tracking_error_m'), ' m')} of {p.tracking_m:g} m",
            admission=f"{disposition.get('value', 'none')}  ·  {self.executor.source.state.lower()}",
            revision=f"{s.get('geometry_revision')}  ·  pub {s.get('navigation_sequence')}  ·  "
                     f"stop gen {s.get('stop_generation')}",
            remaining=_fmt(s.get('remaining_permitted_m'), ' m'),
            margin=f"{_fmt(s.get('stopping_margin_m'), ' m')}  (stop {p.stopping_m:g} m / {p.stopping_s:g} s)",
            validity=f"valid {_fmt(s.get('validity_remaining_s'), ' s', 1)}  ·  {p.permission}",
            lease=f"{s.get('lease_state', 'n/a')}",
            profile=f"{'calibrated' if s.get('profile_calibrated') else 'research'}  ·  {p.permission}  ·  "
                    f"heading {p.heading}",
            recording=s.get('recording_error') or 'healthy',
            vehicle=s.get('vehicle_fault') or s.get('bus_error') or 'ok')

    def readout_text(self):
        """The sidebar as plain text: state, reason and every labelled value."""
        lines = [self.state_badge.cget('text'), self.state_var.get()]
        lines += [f'{label}: {var.get()}' for label, var in self.fields.values()]
        return '\n'.join(lines)

    def _paint_header(self, view):
        s = view['status']
        state = s.get('state', 'WAITING')
        color = self.colors.get(state, self.theme.BUTTON_ACTIVE)
        dark_text = state in ('EXECUTING', 'COMPLETE', 'BRAKING', 'HOLDING', 'TAKING_OFF')
        self.state_badge.configure(text=state.replace('_', ' '), bg=color,
                                   fg=self.theme.CANVAS if dark_text else self.theme.ON_EMPHASIS)
        reason = s.get('reason', '')
        if state in ('COMPLETE', 'CANCELLED', 'FAILED'):
            reason += '   —   controls remain for supervision; close the window to release the vehicle'
        self.state_var.set(reason)
        results = s.get('controls') or {}
        if results:
            action, outcome = max(results.items(), key=lambda kv: kv[1].get('t_s') or 0)
            self.controls_var.set(f"{action.capitalize()} {outcome.get('state')}: {outcome.get('reason') or ''}"[:90])

    def _paint_timeline(self, view):
        theme, canvas = self.theme, self.timeline
        canvas.delete('all')
        width, height = max(1, canvas.winfo_width()), max(1, canvas.winfo_height())
        history = view['history']
        if len(history) < 2:
            canvas.create_text(12, height // 2, anchor='w', fill=theme.DIM, text='Waiting for samples…')
            return
        label_w, strip_h, gap = 92, 8, 6
        now = history[-1][0]
        x0 = now - TIMELINE_S
        plot_w = max(1, width - label_w - 8)
        def sx(t): return label_w + (t - x0) / TIMELINE_S * plot_w
        bands = (('Speed', 'm/s', 1, 2), ('Cross-track', 'm', 3, 4), ('Stop margin', 'm', 5, None))
        band_h = (height - strip_h - gap - 4) / len(bands)
        for index, (label, unit, value_i, limit_i) in enumerate(bands):
            top = index * band_h
            bottom = top + band_h - 4
            pairs = [(h[0], h[value_i]) for h in history if h[0] >= x0]
            values = [v for _, v in pairs if isinstance(v, (int, float))]
            limit = history[-1][limit_i] if limit_i is not None else 0.
            lo = min([0.] + values + ([limit] if isinstance(limit, (int, float)) else []))
            hi = max([1e-3] + values + ([limit] if isinstance(limit, (int, float)) else []))
            span = (hi - lo) * 1.15 or 1.
            def sy(v, _top=top, _bottom=bottom, _lo=lo, _span=span):
                return _bottom - (v - _lo) / _span * (_bottom - _top)
            canvas.create_rectangle(label_w, top, width - 8, bottom, outline=theme.PLOT_GRID, fill=theme.CANVAS)
            latest = values[-1] if values else None
            canvas.create_text(8, top + 2, anchor='nw', fill=theme.TEXT, text=label, font=('TkDefaultFont', 8))
            canvas.create_text(8, top + 15, anchor='nw', fill=theme.DIM, font=('TkDefaultFont', 7),
                               text=_fmt(latest, ' ' + unit))
            if limit_i:
                canvas.create_text(8, top + 26, anchor='nw', fill=theme.DIM, font=('TkDefaultFont', 7),
                                   text=f"limit {_fmt(limit, '')}")
            if isinstance(limit, (int, float)):
                y = sy(limit)
                canvas.create_line(label_w, y, width - 8, y, fill=theme.DANGER if limit_i else theme.GRID, dash=(4, 3))
            previous = None
            for t, v in pairs:
                if not isinstance(v, (int, float)):
                    previous = None
                    continue
                point = (sx(t), sy(v))
                if previous is not None:
                    canvas.create_line(*previous, *point, fill=theme.ACCENT, width=1.5)
                previous = point
        strip_top = height - strip_h - 2
        canvas.create_text(8, strip_top - 2, anchor='nw', fill=theme.DIM, text='State', font=('TkDefaultFont', 7))
        run_start, run_state = None, None
        for t, state in [(h[0], h[6]) for h in history if h[0] >= x0] + [(now, None)]:
            if state != run_state:
                if run_state is not None:
                    canvas.create_rectangle(sx(run_start), strip_top, sx(t), strip_top + strip_h, width=0,
                                            fill=self.colors.get(run_state, theme.BUTTON_ACTIVE))
                run_start, run_state = t, state

    def _paint_fields(self, view):
        for name, value in self._values(view).items():
            self.fields[name][1].set(value)

    def _paint_events(self, view):
        events = view['events']
        mark = (len(events), events[-1]['t_s'] if events else None)
        if mark == self._events_seen:
            return
        self._events_seen = mark
        selected = self.selected_event
        self.events.delete(*self.events.get_children())
        self._event_rows = {}
        for event in reversed(events[-80:]):
            title, detail, tag = describe_event(event)
            item = self.events.insert('', 'end', values=(_fmt(event['t_s'], ' s', 1), title, detail),
                                      tags=(tag,) if tag else ())
            self._event_rows[item] = event
            if event is selected:
                self.events.selection_set(item)

    def _select_event(self, _event):
        selection = self.events.selection()
        if not selection or selection[0] not in self._event_rows:
            return
        event = self._event_rows[selection[0]]
        self.selected_event = event
        title, detail, _ = describe_event(event)
        route = event['data'].get('route') or {}
        self.event_var.set(f"{title} at {_fmt(event['t_s'], ' s', 1)} — {detail}\n"
                           f"route revision {route.get('geometry_revision')}  ·  stop generation "
                           f"{route.get('stop_generation')}  (its route: dim dashed line on the map)")

    # -- the loop ---------------------------------------------------------

    def _control_loop(self):
        period = 1. / self.executor.profile.stream_hz
        while not self._stop.is_set() and not self.executor.closed:
            try:
                self.executor.step()
            except Exception as exc:  # a failed step must stop the flight, visibly
                self.control_error = f'control step error: {type(exc).__name__}: {exc}'
                try: self.executor._fail(self.executor.clock(), self.control_error)
                except Exception: pass
            # Poll at twice the stream rate; the executor's own deadline on
            # provider time decides whether a target is due.
            self._stop.wait(period / 2)

    def tick(self):
        if self.closed:
            return
        if self.executor.shutdown_requested:
            # dsim "Kill all": HOLD, bounded wait, release control, finish the recording, close.
            self.close(f'instance shutdown: {self.executor.shutdown_reason}')
            return
        view = self.executor.view()
        if not view or self.root.state() == 'iconic':
            # Nothing published yet, or the window is minimized: skip the repaint.
            self.root.after(round(1000. / MAP_HZ), self.tick)
            return
        sequence = view['status'].get('sequence')
        if sequence != self._painted_sequence:
            if self._painted_sequence is not None and sequence is not None:
                self.views_coalesced += max(0, sequence - self._painted_sequence - 1)
            self.views_painted += 1
            self._painted_sequence = sequence
        if self._text.due():
            self.drawing_var.set(
                f'View painted {self.views_painted}  ·  coalesced {self.views_coalesced}  ·  '
                f'map ≤ {MAP_HZ:g} Hz, text ≤ {TEXT_HZ:g} Hz  ·  control runs on its own thread'
                + (f'   ·   {self.control_error}' if self.control_error else ''))
            self._paint_header(view)
            self._paint_fields(view)
            self._paint_events(view)
            self._paint_timeline(view)
        # Handing the pane a new view is cheap; refresh() spends its own repaint budget.
        self.pane.set_grid(view.get('grid'), sim_now_s=view['status'].get('time_s'))
        self.pane.state.overlay = self._overlay(view)
        self.pane.refresh()
        self.root.after(round(1000. / MAP_HZ), self.tick)

    def run(self):
        self.root.mainloop()
        return 0

    def close(self, reason='window closed'):
        if self.closed:
            return
        self.closed = True
        from dcmn.window import save_window_pos
        save_window_pos(self.root, f'dway.dynamic.{self.executor.id}')
        self._stop.set()
        if self._thread is not None: self._thread.join(timeout=5.)
        self.executor.shutdown(reason)
        self.root.destroy()


def json_target(target):
    if target is None:
        return 'none'
    return f'({_fmt(target[0], "", 1)}, {_fmt(target[1], "", 1)}, {_fmt(target[2], " m", 1)})'


#: The profile target mode loads when none is given.
TARGET_PROFILE = 'sim-target'


def load_flight_profile(args):
    """Dynamic mode needs a calibrated profile; target mode is research flight and accepts any."""
    target = args.mode == 'target'
    profile = ExecutionProfile.load(args.execution_profile or (TARGET_PROFILE if target else None),
                                    getattr(args, 'profile_set', None))
    if not target: profile.require_flight()  # a synthetic profile cannot authorize dynamic flight
    return profile


def executor_options(args):
    target = args.mode == 'target'
    return dict(auto=target, allow_uncalibrated=target, wait_for=tuple(args.wait_for) if target else ())


def run_flight(args):
    """Headless dynamic or target flight. The same executor methods the window calls."""
    profile = load_flight_profile(args)
    link = DsimLink(args.id, client_id=args.client_id or f'dway-{args.id}',
                    ack_timeout_s=args.ack_timeout)
    from dcmn.maps import MapSession
    from dway.executor import DynamicExecutor
    executor = DynamicExecutor(args.id, link, profile, planner=args.planner, maps=MapSession(args.id),
                               **executor_options(args))
    print(f'dway {args.mode}: profile {profile.digest[:10]} permission={profile.permission} '
          f'speed={profile.speed_mps:g} m/s altitude={profile.altitude_m:g} m'
          f'{"" if profile.calibrated else " (uncalibrated research profile)"}', flush=True)

    def shutdown(*_):
        # A signal shuts the mission down (bounded HOLD wait, lease release,
        # finished recording); it never cancels or lands the vehicle.
        executor.pending.clear()  # a queued control is not sent during shutdown
        result = executor.shutdown('signal')
        executor.shutdown_printed = True
        timings = dict(result.get('timings_s') or {}, **getattr(executor, 'finish_timings_s', {}))
        print(f'dway: shutdown hold {"confirmed" if result.get("confirmed") else "not confirmed"}, '
              f'lease {"released" if result.get("released") else "not held"}; took '
              + ', '.join(f'{k} {v:.2f} s' for k, v in timings.items()), flush=True)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    started = not args.start
    deadline = time.monotonic() + args.timeout if args.timeout else float('inf')
    last, next_line = None, 0.
    try:
        while not executor.closed and time.monotonic() < deadline and not executor.shutdown_requested:
            try:
                status = executor.step()
            except Exception as exc:  # a failed step stops the flight visibly; supervision continues
                import traceback
                traceback.print_exc()
                try: executor._fail(executor.clock(), f'control step error: {type(exc).__name__}: {exc}')
                except Exception: pass
                status = executor.status
            mark = (executor.state, executor.reason, executor.hold_kind)
            if mark != last:
                last = mark
                pose = executor.pose
                print(f'[sim {status.get("time_s", 0.):8.2f}] {executor.state}'
                      f'{"/" + executor.hold_kind if executor.hold_kind else ""}: {executor.reason}'
                      + ('' if pose is None else f'  pose=({pose[0]:.2f},{pose[1]:.2f},{pose[2]:.2f})'), flush=True)
            if time.monotonic() >= next_line:
                next_line = time.monotonic() + 5.
                pose, s = executor.pose, status
                print(f'    [{executor.state}] route {executor.source.state}: {executor.source.reason} | '
                      f'speed {_fmt(s.get("speed_mps"), " m/s")} remaining {_fmt(s.get("remaining_permitted_m"), " m")} '
                      f'targets {s.get("targets_sent", 0)}'
                      + ('' if pose is None else f' pose=({pose[0]:.1f},{pose[1]:.1f},{pose[2]:.2f})')
                      + (f' | {executor.bus_error}' if executor.bus_error else ''), flush=True)
            if not started and executor.source.state == 'READY' and executor.state in ('WAITING', 'READY'):
                executor.request('start', 'cli')
                started = True
            if args.exit_on_finish and executor.state in ('COMPLETE', 'CANCELLED', 'FAILED'):
                break
            time.sleep(max(0.005, 1. / profile.stream_hz - 0.002))
        result = executor.shutdown(f'instance shutdown: {executor.shutdown_reason}' if executor.shutdown_requested
                                   else 'headless run finished')
        if executor.shutdown_requested:
            print(f'dway: stopped by instance shutdown ({executor.shutdown_reason}); HOLD '
                  f'{"confirmed" if result.get("confirmed") else "not confirmed" if result.get("owned") else "not owned"}, '
                  f'control {"released" if result.get("released") else "not held"}', flush=True)
        print(f'dway: {executor.state}: {executor.reason}')
        if not getattr(executor, 'shutdown_printed', False):
            timings = dict(result.get('timings_s') or {}, **getattr(executor, 'finish_timings_s', {}))
            print('dway: shutdown took ' + ', '.join(f'{k} {v:.2f} s' for k, v in timings.items()), flush=True)
        return {'COMPLETE': 0, 'CANCELLED': 1}.get(executor.state, 2)
    finally:
        link.close()
