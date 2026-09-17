"""The Mission tab: the map, the controls, and what the mission is doing and why.

The map is dcmn's `MapPane` over the evidence dalg publishes, with dnav's route,
the vehicle's track, the goal, home and the mission's own marks drawn over it.
Before any evidence exists the pane falls back to the mission's area or
workspace, the reference image or a square around the vehicle, so a goal can
be clicked before dalg has published (B2). The reference background is opt-in
per mission (Q10) and every change is recorded.

The pane is zoomable: the wheel zooms about the pointer, the middle button or
control-drag pans, and a translucent Reset view badge appears in the image's
corner while the view is not the fitted one. None of that reaches the mission
or the report -- the snapshot renders the whole grid either way.

The editors shown come from the mission's ``editors``; goto asks only for
``goal``: a click sets the profile's goal while idle and retargets while
flying. An area or waypoint mission instead takes the click for its geometry
(:mod:`dmsn.ui.map_editors`): click to append, drag to move, right-click to
remove, Hull and Clear and Undo buttons, and a reorderable waypoint list.
Editing happens between missions; one line -- the blocking fact -- always
says what the mission is waiting on.
"""
from __future__ import annotations

from dataclasses import replace

from dmsn import geometry as G
from dmsn.missions import MISSIONS


def _value(value):
    """A progress value in words: coordinates to centimetres, nothing as a dash."""
    if value is None: return '--'
    if isinstance(value, (list, tuple)): return ', '.join(_value(v) for v in value)
    if isinstance(value, float): return f'{value:.2f}'
    return str(value)


def _text(value, digits=1, unit=''):
    if value is None: return '--'
    return f'{value:.{digits}f}{unit}'


class MissionTab:
    def __init__(self, parent, run, tk, ttk, *, show_reference=None) -> None:
        from dcmn.map_pane import MapPane, ReferenceBackgroundHost
        from dcmn.scroll import Scrollable
        self.run, self.tk, self.ttk = run, tk, ttk
        page = ttk.Frame(parent)
        page.grid(row=0, column=0, sticky='nsew')
        parent.rowconfigure(0, weight=1); parent.columnconfigure(0, weight=1)
        page.columnconfigure(0, weight=1); page.rowconfigure(1, weight=1)
        mission = MISSIONS.get(run.document.value['mission'])
        preference = run.document.value['assisted'].get('reference_background')
        enabled = (show_reference if show_reference is not None else
                   preference if preference is not None else bool(mission and mission.reference_default))
        self.reference = ReferenceBackgroundHost(page, run.id, enabled=enabled, on_change=self._reference_changed)
        self.reference.widget.grid(row=0, column=0, columnspan=2, sticky='ew')
        if enabled:
            run.note_display('reference-on', source='--show-reference' if show_reference else 'mission default')
        self.pane = MapPane(page, title='mission', on_click=self._clicked, width=480, height=480,
                            zoom=True,
                            on_press=self._pressed, on_drag=self._dragged, on_release=self._released,
                            on_context=self._context_menu, on_key=self._key)
        self.pane.widget.grid(row=1, column=0, sticky='nsew', padx=(0, 6))
        self.scroll = Scrollable(page)
        self.scroll.outer.grid(row=1, column=1, sticky='ns')
        side = ttk.Frame(self.scroll.inner)
        side.grid(row=0, column=0, sticky='ns')

        controls = ttk.LabelFrame(side, text='Controls', padding=8)
        controls.grid(row=0, column=0, sticky='ew', pady=(0, 6))
        self.buttons = {}
        for index, (label, command, style) in enumerate((
                ('Start', self.start, 'Accent.TButton'), ('Pause', run.pause, None), ('Resume', run.resume, None),
                ('Abort', run.abort, 'Danger.TButton'), ('Go home', run.go_home, None),
                ('Land', run.land, None))):
            options = dict(text=label, command=(lambda c=command: self._command(c)) if command else None)
            if style: options['style'] = style
            button = ttk.Button(controls, **options)
            button.grid(row=index // 2, column=index % 2, sticky='ew', padx=(0, 4), pady=(0, 4))
            self.buttons[label] = button
        self.buttons['Land'].state(['disabled'])
        self.expand = ttk.Button(controls, text='Expand coverage (resets the map)',
                                 command=lambda: self._command(run.expand_coverage))
        self.expand.grid(row=3, column=0, columnspan=2, sticky='ew', pady=(4, 0))
        previous = run.previous_status
        self.notice = tk.StringVar(value=(f'Previous mission: {previous.get("phase")}: '
                                         f'{previous.get("reason", "")}. Press Start to start again.'
                                         if previous else ''))
        ttk.Label(controls, textvariable=self.notice, style='Dim.TLabel', wraplength=230, justify='left').grid(
            row=4, column=0, columnspan=2, sticky='w', pady=(4, 0))
        controls.columnconfigure(0, weight=1); controls.columnconfigure(1, weight=1)

        blocking = ttk.LabelFrame(side, text='Blocking fact', padding=8)
        blocking.grid(row=2, column=0, sticky='ew', pady=(0, 6))
        self.blocking = tk.StringVar(value='--')
        ttk.Label(blocking, textvariable=self.blocking, wraplength=230, justify='left').grid(row=0, column=0, sticky='w')

        self.rows = self._readout(side, 3, 'Mission', ('phase', 'outcome', 'mission', 'goal', 'dnav route',
                                                       'dway', 'intent', 'progress', 'arena'))
        budgets = ttk.LabelFrame(side, text='Budgets', padding=8)
        budgets.grid(row=4, column=0, sticky='ew', pady=(0, 6))
        budgets.columnconfigure(1, weight=1)
        self.bars = {}
        for index, name in enumerate(('time', 'path', 'battery')):
            ttk.Label(budgets, text=name, style='Dim.TLabel').grid(row=index * 2, column=0, sticky='w', padx=(0, 8))
            bar = ttk.Progressbar(budgets, maximum=100., length=150)
            bar.grid(row=index * 2, column=1, sticky='ew')
            var = tk.StringVar(value='--')
            ttk.Label(budgets, textvariable=var, style='Dim.TLabel').grid(row=index * 2 + 1, column=1, sticky='w')
            self.bars[name] = (bar, var)
        self._shown = None
        from dmsn.ui.map_editors import MapEditors
        self.editors = MapEditors(side, run, self.pane, lambda text: self.notice.set(text), row=1)
        self.editors.sync()

    def _readout(self, side, row, title, names):
        frame = self.ttk.LabelFrame(side, text=title, padding=8)
        frame.grid(row=row, column=0, sticky='ew', pady=(0, 6))
        frame.columnconfigure(1, weight=1)
        rows = {}
        for index, name in enumerate(names):
            self.ttk.Label(frame, text=name, style='Dim.TLabel').grid(row=index, column=0, sticky='nw', padx=(0, 8))
            var = self.tk.StringVar(value='--')
            self.ttk.Label(frame, textvariable=var, wraplength=190, justify='left').grid(row=index, column=1, sticky='ew')
            rows[name] = var
        return rows

    # -- input ---------------------------------------------------------------------------

    def _command(self, command) -> None:
        try:
            self.notice.set(str(command()))
        except ValueError as exc:
            self.notice.set(f'refused: {exc}')

    def start(self) -> str:
        return self.run.start()

    def _clicked(self, x_m, y_m) -> None:
        if self.editors.consume_click(x_m, y_m): return
        mission = MISSIONS.get(self.run.document.value['mission'])
        if 'goal' in getattr(mission, 'editors', ()):  # only a goal mission turns a click into a goal
            self.notice.set(self.run.click(x_m, y_m))

    def _pressed(self, x_m, y_m, modifiers) -> None:
        self.editors.press(x_m, y_m, modifiers)

    def _dragged(self, x_m, y_m, modifiers) -> None:
        self.editors.drag(x_m, y_m, modifiers)

    def _released(self, x_m, y_m, modifiers) -> None:
        self.editors.release(x_m, y_m, modifiers)

    def _context_menu(self, x_m, y_m, modifiers) -> None:
        self.editors.context(x_m, y_m, modifiers)

    def _key(self, keysym, pointer) -> None:
        self.editors.key(keysym, pointer)

    def _reference_changed(self, enabled, opacity) -> None:
        self.run.note_display('reference-on' if enabled else 'reference-off', opacity=round(opacity, 3))

    # -- painting --------------------------------------------------------------------------

    def sync(self) -> None:
        run = self.run
        grids = run.ports.grids()
        grid = grids[sorted(grids)[0]] if grids else None
        self.pane.set_grid(grid, sim_now_s=run.sim_time_s())
        background = self.reference.background()
        self.pane.set_background(background, refresh=False)
        if grid is None:
            from dcmn.map_pane import choose_fallback_extent, reference_footprint
            profile = run.active_profile or run.document.value
            bounds, label = choose_fallback_extent(
                area_bounds=G.bbox(profile['area']), workspace=profile['workspace'],
                reference_bounds=None if background is None else reference_footprint(background.image),
                pose=run.observation.xy)
            self.pane.set_fallback_extent(bounds, label)
            self.pane.set_header('mission — no evidence yet' + (f' · extent: {label}' if bounds else ''))
        else:
            self.pane.set_fallback_extent(None)
            self.pane.set_header(f'mission — evidence {grid.source} rev {grid.revision}')
        self.editors.sync()
        overlay = run.overlay()
        self.pane.state.overlay = replace(overlay, shapes=tuple(overlay.shapes) + tuple(self.editors.shapes()))

    def paint_map(self) -> None:
        self.sync()
        self.pane.refresh(force=True)

    def snapshot(self):
        self.sync()
        return self.pane.snapshot()

    def displayed_background(self):
        return self.pane.state.background

    def paint_text(self) -> None:
        run, obs = self.run, self.run.observation
        profile = run.active_profile or run.document.value
        self.blocking.set(run.blocking or ('--' if run.phase != 'idle' else run.reason))
        self.rows['phase'].set(run.phase + (f' (from {run.paused_from})' if run.phase == 'paused' else ''))
        self.rows['outcome'].set(f'{run.outcome}: {run.reason}' if run.outcome else '--')
        self.rows['mission'].set(f"{profile['mission']} · {profile['name']}")
        goal = run.goal_sent
        if goal is not None:
            self.rows['goal'].set(f"{goal['xy'][0]:.2f}, {goal['xy'][1]:.2f} m · {goal['tag']} · rev {goal['revision']}"
                                  + (' · operator' if goal['operator'] else ''))
        else:
            pending = profile['params'].get('goal')
            self.rows['goal'].set('--' if pending is None else f'{pending[0]:.2f}, {pending[1]:.2f} m (profile)')
        nav = obs.navigation
        self.rows['dnav route'].set('--' if not nav else f"{nav.get('planning_status')}"
                                    f"{' · permitted' if nav.get('eligible') else ''}: {nav.get('reason', '')}")
        ex = obs.execution
        self.rows['dway'].set('--' if not ex else f"{ex.get('state')}{'/' + ex['hold_kind'] if ex.get('hold_kind') else ''}"
                              f"{' · held for mission' if ex.get('mission_hold') else ''}: {ex.get('reason')}")
        ack = ex.get('mission_intent') if ex else None
        last = run.intents.last
        self.rows['intent'].set('--' if last is None else f"{last['kind']} rev {last['revision']}: "
                                + ('not yet acknowledged' if not ack or ack.get('revision') != last['revision']
                                   else f"{ack.get('state')} {ack.get('reason', '')}"))
        extra = dict(run.mission.progress()) if run.mission is not None else {}
        states = extra.pop('waypoint_states', None)
        if states is not None:
            index = extra.pop('waypoint_index', None)
            extra = dict(waypoint=f'{len(states) if index is None else index + 1}/{len(states)}', **extra)
        extra.pop('blacklist', None)  # the count is shown; the points are on the map and in the report
        estimates = extra.pop('estimates', None)
        if estimates is not None:  # search: counts here, the estimates themselves are on the map
            extra['estimates'] = (f"{sum(1 for e in estimates if e.get('confirmed'))} confirmed"
                                  f" / {len(estimates)}")
        details = ', '.join(f'{k.replace("_", " ")} {_value(v)}' for k, v in extra.items())
        self.rows['progress'].set(f'{_text(run.path_m)} m flown · {len(run.goals)} goals · '
                                  f'{run.suppressed} suppressed' + (f' · {details}' if details else ''))
        arena = run.arena
        warning = run.check_arena() if run.phase == 'idle' else ''
        self.rows['arena'].set(warning or f"{arena.get('profile') or '--'} / running {arena.get('provider') or '--'}")
        budgets = profile['budgets']
        elapsed = 0.0 if run.started_s is None else max(0.0, obs.time_s - run.started_s)
        for name, used, limit, unit in (('time', elapsed, budgets['time_s'], 's'),
                                        ('path', run.path_m, budgets['path_m'], 'm')):
            bar, var = self.bars[name]
            bar['value'] = 0. if not limit else min(100., 100. * used / limit)
            var.set(f'{used:.1f} {unit}' + ('' if limit is None else f' of {limit:g} {unit}'))
        bar, var = self.bars['battery']
        bar['value'] = 0. if obs.battery_pct is None else obs.battery_pct
        reserve = budgets['battery_reserve_pct']
        var.set(_text(obs.battery_pct, 1, '%') + ('' if reserve is None else f' (reserve {reserve:g}%)'))
        active = run.phase in ('starting', 'running', 'finishing')
        for label, enabled in (('Start', run.phase in ('idle', 'done')), ('Pause', active),
                               ('Resume', run.phase == 'paused'), ('Abort', active or run.phase == 'paused'),
                               ('Go home', run.phase in ('starting', 'running', 'paused') and run.home is not None),
                               ('Land', active and obs.airborne)):
            self.buttons[label].state(['!disabled'] if enabled else ['disabled'])
        self.expand.state(['!disabled'] if run.expand_offer else ['disabled'])
