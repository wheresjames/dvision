"""Bounded, passive event-bus inspection, reusable by Tk applications."""
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import time
import tkinter as tk
from tkinter import ttk

from dcmn.module_bus import PymembusModuleBus
from dcmn import theme


@dataclass
class EventRow:
    number: int
    arrival: str
    event: object
    detail: str
    size: int

    @property
    def summary(self):
        return json.dumps(self.event.payload, ensure_ascii=False, separators=(',', ':'))[:240]


class EventHistory:
    """Arrival order and local eviction accounting, independent of display filters."""
    def __init__(self, max_rows=1000, max_bytes=4 * 1024 * 1024):
        self.max_rows, self.max_bytes = max_rows, max_bytes
        self.rows = deque()
        self.bytes = self.received = self.discarded = 0

    def append(self, event):
        self.received += 1
        detail = json.dumps(asdict(event), indent=2, ensure_ascii=False)
        size = len(detail.encode('utf-8'))
        if size > self.max_bytes:
            self.discarded += 1
            return
        row = EventRow(self.received, datetime.now().astimezone().isoformat(timespec='milliseconds'),
                       event, detail, size)
        self.rows.append(row)
        self.bytes += size
        while len(self.rows) > self.max_rows or self.bytes > self.max_bytes:
            self.bytes -= self.rows.popleft().size
            self.discarded += 1

    def matching(self, source='', kind='', run='', hide_heartbeats=True,
                 hide_health=True):
        for row in self.rows:
            event = row.event
            if hide_heartbeats and event.type == 'module.heartbeat':
                continue
            if hide_health and event.type == 'module.sensor_health':
                continue
            if source.casefold() not in f'{event.implementation} {event.role} {event.process_id}'.casefold():
                continue
            if kind.casefold() not in event.type.casefold() or run.casefold() not in event.run_id.casefold():
                continue
            yield row

    def clear(self):
        self.rows.clear()
        self.bytes = 0


class EventViewer:
    """Independent reader; pause freezes painting while collection continues."""
    def __init__(self, parent, instance_id, *, reader=None, history=None):
        self.reader = reader or PymembusModuleBus(instance_id, 'observer', 'event-viewer', read_only=True)
        self.history = history or EventHistory()
        self.page = ttk.Frame(parent, padding=10)
        self.page.columnconfigure(0, weight=1)
        self.page.rowconfigure(1, weight=1)
        self.paused = tk.BooleanVar(master=self.page, value=False)
        self.follow = tk.BooleanVar(master=self.page, value=True)
        self.hide_heartbeats = tk.BooleanVar(master=self.page, value=True)
        self.hide_health = tk.BooleanVar(master=self.page, value=True)
        self.source = tk.StringVar(master=self.page)
        self.kind = tk.StringVar(master=self.page)
        self.run = tk.StringVar(master=self.page)
        self.status = tk.StringVar(master=self.page)
        self.visible = {}
        self._last_paint = -1e9
        self._closed = False
        self._error = ''
        bar = ttk.Frame(self.page)
        bar.grid(row=0, column=0, sticky='ew')
        for index, (label, variable) in enumerate((('Source', self.source), ('Event type', self.kind), ('Run ID', self.run))):
            ttk.Label(bar, text=label).grid(row=0, column=index * 2, padx=(0, 5))
            ttk.Entry(bar, textvariable=variable, width=18).grid(row=0, column=index * 2 + 1, sticky='ew', padx=(0, 10))
            bar.columnconfigure(index * 2 + 1, weight=1)
            variable.trace_add('write', lambda *_: self.render(force=True))
        for index, (label, variable) in enumerate((('Pause display', self.paused), ('Auto-follow', self.follow),
                                                  ('Hide heartbeats', self.hide_heartbeats),
                                                  ('Hide sensor health', self.hide_health))):
            ttk.Checkbutton(bar, text=label, variable=variable, command=lambda: self.render(force=True)).grid(
                row=1, column=index * 2, columnspan=2, sticky='w', pady=6)
        ttk.Button(bar, text='Clear', command=self.clear).grid(row=1, column=8)
        panes = ttk.Panedwindow(self.page, orient='vertical')
        panes.grid(row=1, column=0, sticky='nsew')
        table = ttk.Frame(panes)
        table.rowconfigure(0, weight=1)
        table.columnconfigure(0, weight=1)
        panes.add(table, weight=3)
        columns = ('arrival', 'source', 'type', 'run', 'summary')
        self.table = ttk.Treeview(table, columns=columns, show='headings', selectmode='browse', height=12)
        for name, title, width in zip(columns, ('Arrival (local)', 'Source', 'Event type', 'Run ID', 'Summary'),
                                       (205, 120, 170, 130, 360)):
            self.table.heading(name, text=title)
            self.table.column(name, width=width, minwidth=60)
        self.table.grid(row=0, column=0, sticky='nsew')
        scroll = ttk.Scrollbar(table, orient='vertical', command=self._scroll)
        scroll.grid(row=0, column=1, sticky='ns')
        self.table.configure(yscrollcommand=scroll.set)
        horizontal = ttk.Scrollbar(table, orient='horizontal', command=self.table.xview)
        horizontal.grid(row=1, column=0, sticky='ew')
        self.table.configure(xscrollcommand=horizontal.set)
        self.table.bind('<<TreeviewSelect>>', self._select)
        # Any direct navigation of the table cancels auto-follow, or the next
        # paint's see() keeps snapping the view back to the newest row.
        for event in ('<MouseWheel>', '<Button-4>', '<Button-1>', '<KeyPress-Up>',
                      '<KeyPress-Down>', '<KeyPress-Prior>', '<KeyPress-Home>'):
            self.table.bind(event, self._stop_follow, add=True)
        details = ttk.Frame(panes)
        details.columnconfigure(0, weight=1)
        details.rowconfigure(1, weight=1)
        panes.add(details, weight=2)
        ttk.Label(details, text='Selected event — decoded envelope and payload').grid(row=0, column=0, sticky='w')
        self.detail = tk.Text(details, height=9, wrap='word', state='disabled',
                              background=theme.CANVAS, foreground=theme.TEXT)
        self.detail.grid(row=1, column=0, sticky='nsew')
        detail_scroll = ttk.Scrollbar(details, orient='vertical', command=self.detail.yview)
        detail_scroll.grid(row=1, column=1, sticky='ns')
        self.detail.configure(yscrollcommand=detail_scroll.set)
        ttk.Label(self.page, textvariable=self.status).grid(row=2, column=0, sticky='w', pady=(6, 0))
        self.render(force=True)

    def poll(self):
        if self._closed:
            return
        try:
            for event in self.reader.receive(limit=128):
                self.history.append(event)
            self._error = ''
        except (RuntimeError, ValueError) as exc:
            self._error = str(exc)
            self.reader.close()
        self.render()

    def render(self, *, force=False):
        now = time.monotonic()
        if not force and now - self._last_paint < .2:
            return
        self._last_paint = now
        if not self.paused.get():
            rows = list(self.history.matching(self.source.get(), self.kind.get(), self.run.get(),
                                              self.hide_heartbeats.get(), self.hide_health.get()))
            wanted = {str(row.number): row for row in rows}
            for iid in self.visible.keys() - wanted.keys():
                self.table.delete(iid)
            for iid, row in wanted.items():
                if iid not in self.visible:
                    self.table.insert('', 'end', iid=iid, values=(row.arrival, row.event.implementation,
                                      row.event.type, row.event.run_id, row.summary))
            self.visible = wanted
            # Filters can reintroduce earlier rows: restore arrival ordering.
            # The live case already has it, so pay for the moves only on change.
            if tuple(wanted) != self.table.get_children(''):
                for index, iid in enumerate(wanted):
                    self.table.move(iid, '', index)
            if self.follow.get() and wanted:
                self.table.see(next(reversed(wanted)))
            self._select()
        connection = self._error or ('connected' if self.reader.session_id is not None else 'waiting for event bus')
        self.status.set(f'{connection} | {"paused" if self.paused.get() else "live"} | '
                        f'{len(self.visible)} shown / {len(self.history.rows)} retained | '
                        f'{self.history.received} received | {self.reader.overruns} reader overruns | '
                        f'{self.history.discarded} locally discarded')

    def _select(self, _event=None):
        selected = self.table.selection()
        row = self.visible.get(selected[0]) if selected else None
        text = '' if row is None else f'Arrival: {row.arrival}\n\n{row.detail}'
        if self.detail.get('1.0', 'end-1c') != text:
            self.detail.configure(state='normal')
            self.detail.delete('1.0', 'end')
            self.detail.insert('1.0', text)
            self.detail.configure(state='disabled')

    def _stop_follow(self, event):
        if getattr(event, 'delta', 0) < 0:  # downward mouse-wheel movement
            return
        self.follow.set(False)

    def _scroll(self, *args):
        self.follow.set(False)
        self.table.yview(*args)

    def clear(self):
        self.history.clear()
        self.table.delete(*self.table.get_children())
        self.visible = {}
        self._select()
        self.render(force=True)

    def close(self):
        self._closed = True
        self.reader.close()
