"""The one profile editor: algorithm sources and their settings, nothing else.

A profile is a name and a list of sources, each a sensor (or the
``primary_camera`` selector) paired with an evidence algorithm and its
settings. Tours are mission tooling; map extent, resolution and slab are
runtime mapping configuration (``dalg --bounds/--cell-m``) and are shown here
read-only, as dalg resolved them, never saved into the file.

Rows are validated with the same rules a run applies at readiness, so the editor
and the runtime cannot disagree about what a profile means.
"""
from dataclasses import fields
from pathlib import Path

from dalg.profiles import (PRIMARY_CAMERA, Source, profile_dir, save_sources_profile,
                          source_configs, source_errors)


def algorithms_for(sensor_type=None):
    """Evidence algorithms a sensor of this type can run; all of them when unknown."""
    return tuple(name for name, (kind, _) in source_configs().items()
                 if sensor_type is None or kind == sensor_type)


def coerce(kind, text):
    """Parse a form field back to its configured type.

    bool is the trap: bool("False") is True, so every unticked setting would
    silently save as enabled.
    """
    if kind is bool:
        lowered = str(text).strip().lower()
        if lowered in ('1', 'true', 'yes', 'on'): return True
        if lowered in ('0', 'false', 'no', 'off'): return False
        raise ValueError(f'expected true or false, got {text!r}')
    return kind(text)


def row_errors(sources, manifest=None):
    """Row-addressed errors, as a run would report them at readiness.

    The ``primary_camera`` selector is checked against the manifest's declared
    primary camera, never against whichever camera happens to be first.
    """
    resolved = []
    primary = (manifest or {}).get('primary_camera')
    for source in sources:
        if source.sensor == PRIMARY_CAMERA and manifest is not None and primary:
            source = Source(primary, source.algorithm, source.settings)
        resolved.append(source)
    errors = source_errors(resolved, manifest if manifest is not None else None)
    if manifest is not None and not primary:
        for index, source in enumerate(sources):
            if source.sensor == PRIMARY_CAMERA:
                errors[index] = 'the sensor manifest declares no primary camera'
    return errors


class SourceEditor:
    def __init__(self, page, profile, tk, ttk, *, root, on_open=None):
        from dcmn.scroll import Scrollable

        self.page, self.tk, self.ttk, self.root = page, tk, ttk, Path(root)
        # In a viewport, so a short window scrolls the form rather than
        # crushing its lists to a sliver.
        self._scroll = Scrollable(page)
        self._scroll.outer.grid(row=0, column=0, sticky='nsew')
        page.rowconfigure(0, weight=1); page.columnconfigure(0, weight=1)
        page = self._scroll.inner
        self.profile = profile
        self.manifest = None
        self.path = profile.path
        self.sources = list(profile.sources)
        self._draft_errors = {}
        self._drafts = {}
        self.name = tk.StringVar(value=profile.name)
        self.name.trace_add('write', lambda *_: self.validate())
        self.notice = tk.StringVar(value='Offline: sensor IDs are checked against the provider at readiness')
        self.runtime = tk.StringVar(value='Runtime geometry: not resolved here. dalg resolves it from '
                                           '--bounds/--cell-m or the first valid pose and optional goal.')
        top = ttk.Frame(page)
        top.grid(row=0, column=0, columnspan=2, sticky='ew')
        ttk.Label(top, text='Name').grid(row=0, column=0, sticky='w')
        ttk.Entry(top, textvariable=self.name).grid(row=0, column=1, sticky='ew')
        top.columnconfigure(1, weight=1)
        left = ttk.LabelFrame(page, text='Discovered sensors', padding=6)
        left.grid(row=1, column=0, sticky='nsew', padx=(0, 6), pady=6)
        right = ttk.LabelFrame(page, text='Profile sources', padding=6)
        right.grid(row=1, column=1, sticky='nsew', pady=6)
        self.sensors = ttk.Treeview(left, columns=('type', 'rate'), show='tree headings', height=5)
        self.sensors.heading('#0', text='Sensor'); self.sensors.column('#0', width=90)
        for key, width in (('type', 130), ('rate', 65)):
            self.sensors.heading(key, text=key); self.sensors.column(key, width=width)
        self.sensors.pack(fill='both', expand=True)
        self.sensors.bind('<<TreeviewSelect>>', self._sensor_selected)
        self.sensor = tk.StringVar(value=PRIMARY_CAMERA)
        self.algorithm = tk.StringVar(value='ground_plane')
        ttk.Entry(left, textvariable=self.sensor).pack(fill='x', pady=2)
        self.algorithm_box = ttk.Combobox(left, textvariable=self.algorithm,
                                          values=algorithms_for(), state='readonly')
        self.algorithm_box.pack(fill='x')
        self._scroll.claim_wheel(self.algorithm_box)
        ttk.Button(left, text='Add source', command=self.add_selected).pack(anchor='e', pady=4)
        self.rows = ttk.Treeview(right, columns=('algorithm', 'validation'), show='tree headings', height=5)
        self.rows.heading('#0', text='Sensor'); self.rows.column('#0', width=110)
        for key, width in (('algorithm', 150), ('validation', 250)):
            self.rows.heading(key, text=key); self.rows.column(key, width=width)
        self.rows.pack(fill='both', expand=True)
        self.rows.bind('<<TreeviewSelect>>', self._select)
        controls = ttk.Frame(right); controls.pack(fill='x')
        for title, callback in (('Remove', self.remove_selected),
                                ('Up', lambda: self.move_selected(-1)),
                                ('Down', lambda: self.move_selected(1))):
            ttk.Button(controls, text=title, command=callback).pack(side='left')
        self.settings_frame = ttk.LabelFrame(page, text='Selected source settings', padding=6)
        self.settings_frame.grid(row=2, column=0, columnspan=2, sticky='ew')
        geometry = ttk.LabelFrame(page, text='Runtime mapping (read-only; not part of the profile)', padding=6)
        geometry.grid(row=3, column=0, columnspan=2, sticky='ew', pady=6)
        ttk.Label(geometry, textvariable=self.runtime, wraplength=900, justify='left').grid(
            row=0, column=0, sticky='w')
        actions = ttk.Frame(page); actions.grid(row=4, column=0, columnspan=2, sticky='ew')
        self.save_button = ttk.Button(actions, text='Save', command=self.save_current)
        self.save_button.pack(side='right')
        self.save_as_button = ttk.Button(actions, text='Save As…', command=self.save_as)
        self.save_as_button.pack(side='right', padx=6)
        # Only where the host can act on it: the offline editor opens another
        # profile in place; a running dalg's tab describes its launch profile.
        self.open_button = None
        if on_open is not None:
            self.open_button = ttk.Button(actions, text='Open…', command=on_open)
            self.open_button.pack(side='right')
        ttk.Label(page, textvariable=self.notice, wraplength=900).grid(row=5, column=0, columnspan=2, sticky='w')
        page.rowconfigure(1, weight=1)
        page.columnconfigure(0, weight=1); page.columnconfigure(1, weight=2)
        self.setting_vars = {}
        self._refresh_rows()
        if self.sources:
            self.rows.selection_set('0'); self._select()
        self._saved_signature = self.state_signature()

    # -- state ---------------------------------------------------------------------

    def state_signature(self):
        """Everything a save would write, drafts included, to tell edited from untouched."""
        return (self.name.get(),
                tuple((source.sensor, source.algorithm, tuple(sorted(source.settings.items())))
                      for source in self.sources),
                tuple(sorted((key, tuple(sorted(draft.items())))
                             for key, draft in self._drafts.items())))

    @property
    def dirty(self):
        return self.state_signature() != self._saved_signature

    def set_manifest(self, manifest):
        if not manifest or manifest == self.manifest: return
        self.manifest = manifest
        self.sensors.delete(*self.sensors.get_children())
        for sid, entry in manifest.get('sensors', {}).items():
            label = sid + (' (primary)' if sid == manifest.get('primary_camera') else '')
            self.sensors.insert('', 'end', iid=sid, text=label,
                                values=(entry['type'], f"{entry['rate_hz']:g} Hz"))
        self.validate()

    def set_runtime(self, *, geometry, basis='', state='', reason=''):
        """Show what the running dalg resolved; never saved."""
        if geometry is None:
            text = f'Runtime geometry: unresolved ({state}{": " + reason if reason else ""})'
        else:
            x0, y0, x1, y1 = geometry.bounds_m()
            text = (f'Runtime geometry ({basis}): [{x0:g}, {y0:g}] - [{x1:g}, {y1:g}] m, '
                    f'{geometry.width} x {geometry.height} cells of {geometry.cell_m:g} m, '
                    f'slab {geometry.z0_m:g}-{geometry.z0_m + geometry.dz_m:g} m; {state}')
        if self.runtime.get() != text: self.runtime.set(text)

    # -- rows ------------------------------------------------------------------------

    def _sensor_type(self, sensor):
        manifest = self.manifest or {}
        if sensor == PRIMARY_CAMERA: return 'camera.rgb'
        entry = manifest.get('sensors', {}).get(sensor)
        return None if entry is None else entry.get('type')

    def _sensor_selected(self, _event=None):
        selected = self.sensors.selection()
        if not selected: return
        sid = selected[0]; self.sensor.set(sid)
        choices = list(algorithms_for(self.manifest['sensors'][sid]['type']))
        self.algorithm_box.configure(values=choices)
        self.algorithm.set(choices[0] if choices else '')

    def add_source(self, sensor, algorithm, settings=None):
        self.sources.append(Source(sensor, algorithm, dict(settings or {})))
        self._refresh_rows()
        self.rows.selection_set(str(len(self.sources)-1)); self._select()

    def add_selected(self):
        self.add_source(self.sensor.get().strip(), self.algorithm.get())

    def selected_index(self):
        selected = self.rows.selection()
        return int(selected[0]) if selected else None

    def remove_selected(self):
        index = self.selected_index()
        if index is not None:
            source = self.sources.pop(index)
            self._draft_errors.pop(source.id, None); self._drafts.pop(source.id, None)
            self._refresh_rows(); self._select()

    def move_selected(self, delta):
        index = self.selected_index()
        if index is None or not 0 <= index+delta < len(self.sources): return
        self.sources[index], self.sources[index+delta] = self.sources[index+delta], self.sources[index]
        self._refresh_rows(); self.rows.selection_set(str(index+delta)); self._select()

    def _refresh_rows(self):
        self.rows.delete(*self.rows.get_children())
        for index, source in enumerate(self.sources):
            self.rows.insert('', 'end', iid=str(index), text=source.sensor, values=(source.algorithm, ''))
        self.validate()

    def _select(self, _event=None):
        for child in self.settings_frame.winfo_children(): child.destroy()
        self.setting_vars = {}
        index = self.selected_index()
        if index is None: return
        source = self.sources[index]
        frame, ttk = self.settings_frame, self.ttk
        ttk.Label(frame, text='algorithm').grid(row=0, column=0, sticky='w')
        choice = self.tk.StringVar(value=source.algorithm)
        box = ttk.Combobox(frame, textvariable=choice, state='readonly',
                           values=algorithms_for(self._sensor_type(source.sensor)))
        box.grid(row=0, column=1, sticky='ew')
        box.bind('<<ComboboxSelected>>', lambda _e: self._change_algorithm(index, choice.get()))
        self._scroll.claim_wheel(box)
        self.row_algorithm = box
        row = 1
        config = source_configs().get(source.algorithm, (None, None))[1]
        draft = self._drafts.get(source.id, source.settings)
        if config is not None:
            defaults = config()
            for field in fields(defaults):
                row = self._setting_row(row, field.name, getattr(defaults, field.name), draft)
        frame.columnconfigure(1, weight=1)

    def _setting_row(self, row, name, default, draft):
        variable = self.tk.StringVar(value=str(draft.get(name, default)))
        self.setting_vars[name] = (variable, type(default))
        self.ttk.Label(self.settings_frame, text=name).grid(row=row, column=0, sticky='w')
        self.ttk.Entry(self.settings_frame, textvariable=variable).grid(row=row, column=1, sticky='ew')
        variable.trace_add('write', self._settings_changed)
        return row + 1

    def _change_algorithm(self, index, algorithm):
        """Swap a row's algorithm from that algorithm's defaults; settings never carry over."""
        source = self.sources[index]
        if algorithm == source.algorithm: return
        self._drafts.pop(source.id, None); self._draft_errors.pop(source.id, None)
        self.sources[index] = Source(source.sensor, algorithm, {})
        self._refresh_rows(); self.rows.selection_set(str(index)); self._select()

    def _settings_changed(self, *_):
        index = self.selected_index()
        if index is None: return
        source = self.sources[index]
        self._drafts[source.id] = {key: variable.get() for key, (variable, _) in self.setting_vars.items()}
        try:
            config = source_configs()[source.algorithm][1]
            defaults = config()
            values = {}
            for key, (variable, kind) in self.setting_vars.items():
                value = coerce(kind, variable.get())
                # Only what differs from the algorithm's defaults is written: a
                # profile states choices, not a copy of today's defaults.
                if value != getattr(defaults, key) or key in source.settings: values[key] = value
            self.sources[index] = Source(source.sensor, source.algorithm, values)
            self._draft_errors.pop(source.id, None)
        except ValueError as exc:
            self._draft_errors[source.id] = str(exc)
        self.validate()

    # -- validation and saving ------------------------------------------------------------

    def validate(self):
        errors = row_errors(self.sources, self.manifest)
        for index, source in enumerate(self.sources):
            error = self._draft_errors.get(source.id) or errors.get(index, '')
            if error: errors[index] = error
            fine = 'valid' if self.manifest else 'sensor not checked (offline)'
            self.rows.item(str(index), text=source.sensor, values=(source.algorithm, error or fine))
        valid = bool(self.sources) and not errors and bool(self.name.get().strip())
        for button in (self.save_button, self.save_as_button):
            button.configure(state='normal' if valid else 'disabled')
        self.notice.set('Add at least one source and fix the rows before saving' if not valid else
            ('Validated against the sensor manifest. Saved changes apply on the next launch.'
             if self.manifest else 'Offline: sensor IDs are checked against the provider at readiness.'))
        return valid

    def save(self, path):
        if not self.validate(): raise ValueError('profile has validation errors')
        name = self.name.get().strip()
        save_sources_profile(Path(path), name=name, sources=self.sources)
        self.path = Path(path)
        self._drafts.clear()
        self._saved_signature = self.state_signature()
        self.notice.set(f'Saved {path}. Launch this profile to use the changes.')

    def save_current(self):
        if self.path is None: self.save_as(); return
        try: self.save(self.path)
        except (ValueError, OSError) as exc: self.notice.set(f'Not saved: {exc}')

    def save_as(self):
        from tkinter import filedialog
        path = filedialog.asksaveasfilename(parent=self.page.winfo_toplevel(),
            initialdir=profile_dir(self.root), initialfile=self.name.get()+'.json',
            defaultextension='.json', filetypes=(('Profile JSON', '*.json'),))
        if path:
            try: self.save(path)
            except (ValueError, OSError) as exc: self.notice.set(f'Not saved: {exc}')
