"""The Setup tab: pick a mission, fill in its generated form, and manage the profile file.

Every form row is generated from a `ParamSchema` -- the mission's own, then the
profile's general settings, budgets, debounce and assisted flags -- so a new
mission type brings its form with it and this tab never changes. An edit is
checked on Return or when focus leaves the field; a refused value is put back
and the reason shown, so the form never holds something the profile does not.

The file is the source of truth: New, Load, Save, Save as and Revert act on
the profile document, the digest is always visible, and a dirty marker says
when what is on screen is not what is on disk. Edits made while a mission is
flying apply to the next Start.

Import tour runs the converter as a child process (B9): the window only reads
the profile the child wrote, so a malformed tour can never touch it.
"""
from __future__ import annotations

from pathlib import Path

from dmsn import profiles as P
from dmsn.missions import MISSIONS


def sections(document):
    """``(title, key prefix, schema)`` for every form section of a profile."""
    from dmsn.missions import schema_for
    from dmsn.strategies import STRATEGIES
    value = document.value
    result = [('Mission parameters', 'params.', schema_for(value['mission'])),
              ('General', '', P.GENERAL)]
    strategy = STRATEGIES.get(value.get('strategy') or '')
    if strategy is not None and tuple(strategy.schema):
        result.append((f'Strategy parameters: {strategy.name}', 'strategy_params.', strategy.schema))
    result += [('Budgets', 'budgets.', P.BUDGETS),
               ('Goal debounce', 'debounce.', P.DEBOUNCE),
               ('Assisted', 'assisted.', P.ASSISTED)]
    return tuple(result)


def choices(key, field):
    """The values a combobox offers for a field, or None for a free-text entry."""
    if key == 'strategy':
        from dmsn.strategies import STRATEGIES
        return ('', *sorted(STRATEGIES))
    if field.kind == 'choice': return field.choices
    if field.kind == 'bool': return ('', 'true', 'false') if field.optional else ('true', 'false')
    return None


def current(document, prefix, name):
    value = document.value
    if prefix == 'params.': return value['params'].get(name)
    if prefix: return value[prefix[:-1]].get(name)
    return value.get(name)


class SetupTab:
    def __init__(self, parent, run, tk, ttk) -> None:
        from dcmn.scroll import Scrollable
        self.run, self.tk, self.ttk = run, tk, ttk
        self.document = run.document
        page = ttk.Frame(parent)
        page.grid(row=0, column=0, sticky='nsew')
        parent.rowconfigure(0, weight=1); parent.columnconfigure(0, weight=1)
        page.columnconfigure(0, weight=1); page.rowconfigure(2, weight=1)

        header = ttk.LabelFrame(page, text='Profile', padding=8)
        header.grid(row=0, column=0, sticky='ew', pady=(0, 6))
        ttk.Label(header, text='mission', style='Dim.TLabel').grid(row=0, column=0, sticky='w', padx=(0, 8))
        self.mission_var = tk.StringVar(value=self.document.value['mission'])
        self.mission_combo = ttk.Combobox(header, textvariable=self.mission_var, values=sorted(MISSIONS),
                                          state='readonly', width=14)
        self.mission_combo.grid(row=0, column=1, sticky='w')
        self.mission_combo.bind('<<ComboboxSelected>>', self._mission_changed)
        self.mission_label = tk.StringVar(value='')
        ttk.Label(header, textvariable=self.mission_label, style='Dim.TLabel').grid(row=0, column=2, sticky='w', padx=8)
        ttk.Label(header, text='name', style='Dim.TLabel').grid(row=1, column=0, sticky='w', padx=(0, 8), pady=(4, 0))
        self.name_var = tk.StringVar(value=self.document.value['name'])
        name = ttk.Entry(header, textvariable=self.name_var, width=28)
        name.grid(row=1, column=1, columnspan=2, sticky='w', pady=(4, 0))
        name.bind('<Return>', lambda _e: self._commit('name', self.name_var))
        name.bind('<FocusOut>', lambda _e: self._commit('name', self.name_var))
        buttons = ttk.Frame(header)
        buttons.grid(row=2, column=0, columnspan=4, sticky='w', pady=(8, 4))
        for index, (label, command) in enumerate((('New', self.new), ('Load…', self.load), ('Save', self.save),
                                                  ('Save as…', self.save_as), ('Revert', self.revert),
                                                  ('Import tour…', self.import_tour))):
            ttk.Button(buttons, text=label, command=command).grid(row=0, column=index, padx=(0, 4))
        ttk.Label(header, text='open', style='Dim.TLabel').grid(row=3, column=0, sticky='w', padx=(0, 8))
        self.available_var = tk.StringVar(value='')
        self.available = ttk.Combobox(header, textvariable=self.available_var, state='readonly', width=48)
        self.available.grid(row=3, column=1, columnspan=3, sticky='w')
        self.available.bind('<<ComboboxSelected>>', self._open_listed)
        self.identity = tk.StringVar(value='')
        ttk.Label(header, textvariable=self.identity, style='Dim.TLabel').grid(
            row=4, column=0, columnspan=4, sticky='w', pady=(6, 0))
        self.notice = tk.StringVar(value='the file is the source of truth')
        ttk.Label(header, textvariable=self.notice, wraplength=640, justify='left').grid(
            row=5, column=0, columnspan=4, sticky='w')
        header.columnconfigure(3, weight=1)

        self.scroll = Scrollable(page)
        self.scroll.outer.grid(row=2, column=0, sticky='nsew')
        self.form = ttk.Frame(self.scroll.inner)
        self.form.grid(row=0, column=0, sticky='nsew')
        self.rows: dict[str, tuple] = {}
        self._built_for = None
        self._shown_revision = None
        self._listed = []
        self.refresh_available()
        self.build()

    # -- the form ---------------------------------------------------------------------

    def build(self) -> None:
        """Generate one row per field of every section, for the current mission."""
        tk, ttk = self.tk, self.ttk
        for child in self.form.winfo_children(): child.destroy()
        self.rows = {}
        row = 0
        for title, prefix, schema in sections(self.document):
            frame = ttk.LabelFrame(self.form, text=title, padding=8)
            frame.grid(row=row, column=0, sticky='ew', pady=(0, 6)); row += 1
            frame.columnconfigure(1, weight=1)
            for index, field in enumerate(schema):
                key = prefix + field.name
                label = field.title + (f' ({field.unit})' if field.unit else '')
                ttk.Label(frame, text=label, style='Dim.TLabel').grid(row=index * 2, column=0, sticky='w', padx=(0, 8))
                var = tk.StringVar(value=field.format(current(self.document, prefix, field.name)))
                values = choices(key, field)
                if values is not None:
                    widget = ttk.Combobox(frame, textvariable=var, values=values, state='readonly', width=24)
                    widget.bind('<<ComboboxSelected>>', lambda _e, k=key, v=var: self._commit(k, v))
                    self.scroll.claim_wheel(widget)
                else:
                    widget = ttk.Entry(frame, textvariable=var, width=40)
                    widget.bind('<Return>', lambda _e, k=key, v=var: self._commit(k, v))
                    widget.bind('<FocusOut>', lambda _e, k=key, v=var: self._commit(k, v))
                widget.grid(row=index * 2, column=1, sticky='w')
                if field.help:
                    ttk.Label(frame, text=field.help, style='Dim.TLabel', wraplength=560, justify='left').grid(
                        row=index * 2 + 1, column=1, sticky='w', pady=(0, 4))
                self.rows[key] = (field, var, widget, prefix)
        self._built_for = (self.document.value['mission'], self.document.value['strategy'])
        self._shown_revision = self.document.revision
        self._paint_identity()

    def _values_to_form(self) -> None:
        if (self.document.value['mission'], self.document.value['strategy']) != self._built_for: return self.build()
        for field, var, _widget, prefix in self.rows.values():
            text = field.format(current(self.document, prefix, field.name))
            if var.get() != text: var.set(text)
        self.mission_var.set(self.document.value['mission'])
        self.name_var.set(self.document.value['name'])
        self._shown_revision = self.document.revision
        self._paint_identity()

    def _paint_identity(self) -> None:
        doc = self.document
        mission = MISSIONS.get(doc.value['mission'])
        self.mission_label.set('' if mission is None else mission.label)
        where = '(unsaved)' if doc.path is None else str(doc.path)
        self.identity.set(f"{'● modified' if doc.dirty else 'saved'}  ·  digest {doc.digest[:12]}  ·  {where}")

    def refresh(self) -> None:
        """Pick up changes made elsewhere (a map click sets the goal) without clobbering typing."""
        if self.document.revision != self._shown_revision: self._values_to_form()
        else: self._paint_identity()

    def _commit(self, key: str, var) -> bool:
        field = self.rows.get(key, (None,))[0]
        text = var.get()
        before = self.document.value
        ok = self.document.set(key, text)
        if not ok:
            self.notice.set(f'not changed: {self.document.error}')
            restored = self.document.value['name'] if key == 'name' else (
                '' if field is None else field.format(current(self.document, self.rows[key][3], field.name)))
            var.set(restored)
            return False
        if self.document.value != before:
            self.notice.set(f'{key} changed' + ('; applies at the next Start' if self.run.phase not in ('idle', 'done')
                                                 else ''))
        self._values_to_form()
        return True

    def set_field(self, key: str, text: str) -> bool:
        """Type into one form field and commit it, as an operator would."""
        field, var, _widget, _prefix = self.rows[key]
        var.set(text)
        return self._commit(key, var)

    # -- profile file -------------------------------------------------------------------

    def refresh_available(self) -> None:
        self._listed = P.available()
        self.available.configure(values=[f'{label}: {path.stem}' for label, path in self._listed])

    def _mission_changed(self, _event=None) -> None:
        if self.mission_var.get() == self.document.value['mission']: return
        if not self.document.set('mission', self.mission_var.get()):
            self.notice.set(f'not changed: {self.document.error}')
        self.build()

    def new(self) -> None:
        self.document.new(self.mission_var.get() or 'goto')
        self.notice.set('new profile')
        self.build()

    def open_path(self, path) -> bool:
        try:
            self.document.open(path)
        except ValueError as exc:
            self.notice.set(f'not loaded: {exc}')
            return False
        self.notice.set(f'loaded {self.document.path}')
        self.build()
        return True

    def _open_listed(self, _event=None) -> None:
        index = self.available.current()
        if 0 <= index < len(self._listed): self.open_path(self._listed[index][1])

    def load(self) -> None:
        from tkinter import filedialog
        path = filedialog.askopenfilename(title='Load mission profile', initialdir=str(P.user_dir()),
                                          filetypes=(('mission profile', '*.json'),))
        if path: self.open_path(path)

    def save(self, path=None) -> Path | None:
        try:
            saved = self.document.save(path)
        except (OSError, ValueError) as exc:
            self.notice.set(f'not saved: {exc}')
            return None
        self.notice.set(f'saved {saved}')
        self.refresh_available()
        self._values_to_form()
        return saved

    def save_as(self) -> None:
        from tkinter import filedialog
        P.user_dir().mkdir(parents=True, exist_ok=True)
        path = filedialog.asksaveasfilename(title='Save mission profile as', initialdir=str(P.user_dir()),
                                            initialfile=f"{self.document.value['name']}.json",
                                            defaultextension='.json', filetypes=(('mission profile', '*.json'),))
        if path: self.save(path)

    def revert(self) -> None:
        try:
            self.document.revert()
        except ValueError as exc:
            self.notice.set(f'not reverted: {exc}')
            return
        self.notice.set('reverted to the file')
        self.build()

    # -- tour import (B9) -----------------------------------------------------------------

    def pick_tour(self) -> str:
        """The tour to convert, chosen by the operator."""
        from tkinter import filedialog
        return filedialog.askopenfilename(title='Import tour as a waypoints mission',
                                          initialdir=str(P.ROOT / 'assets' / 'tours'),
                                          filetypes=(('tour', '*.json'),))

    def import_tour(self) -> None:
        """Convert a tour through a child dmsn, then open what it wrote.

        The converter runs as its own process so a bad tour is refused there,
        with nothing half-converted left in this one; all this method does with
        a success is open the profile file the child saved.
        """
        from dmsn.tour_import import run_converter
        tour = self.pick_tour()
        if not tour: return
        out = P.user_dir() / f'{Path(tour).stem}.json'
        converted, message = run_converter(tour, out)
        if not converted:
            self.notice.set(f'tour not imported: {message}')
            return
        if self.open_path(out): self.notice.set(f'tour imported: {out.name}')
        self.refresh_available()
