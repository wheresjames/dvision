"""The dmsn window: Mission, Setup, Status, Events, and a status bar."""
from __future__ import annotations

from dcmn.pacing import MAP_HZ, Paced, TEXT_HZ


class Window:
    def __init__(self, run, *, show_reference=None) -> None:
        import tkinter as tk
        from tkinter import ttk
        from dcmn.event_viewer import EventViewer
        from dcmn.tktheme import apply_theme
        from dcmn.window import restore_window_geometry, save_window_geometry
        from dmsn.ui.mission_tab import MissionTab
        from dmsn.ui.setup_tab import SetupTab
        from dmsn.ui.status_tab import StatusTab

        self.tk, self.ttk, self.run = tk, ttk, run
        self._save_geometry = save_window_geometry
        self.root = tk.Tk()
        apply_theme(self.root)
        self.root.title(f'dmsn mission coordinator — {run.id}')
        self.root.geometry('1180x820')
        self.root.minsize(760, 520)
        self.root.protocol('WM_DELETE_WINDOW', self.close)
        self.running = True
        notebook = ttk.Notebook(self.root)
        notebook.grid(row=0, column=0, sticky='nsew')
        pages = {}
        for name in ('Mission', 'Setup', 'Status', 'Events'):
            pages[name] = ttk.Frame(notebook, padding=8 if name != 'Events' else 0)
            notebook.add(pages[name], text=name)
        self.notebook = notebook
        self.mission = MissionTab(pages['Mission'], run, tk, ttk, show_reference=show_reference)
        self.setup = SetupTab(pages['Setup'], run, tk, ttk)
        self.status_tab = StatusTab(pages['Status'], run, tk, ttk)
        self.events = EventViewer(pages['Events'], run.id)
        self.events.page.grid(row=0, column=0, sticky='nsew')
        pages['Events'].rowconfigure(0, weight=1); pages['Events'].columnconfigure(0, weight=1)
        self.pages = pages
        self.status = tk.StringVar(value='connecting')
        ttk.Label(self.root, textvariable=self.status, style='Dim.TLabel', anchor='w').grid(
            row=1, column=0, sticky='ew', padx=8, pady=(2, 6))
        self.root.columnconfigure(0, weight=1); self.root.rowconfigure(0, weight=1)
        self._paint_map = Paced(MAP_HZ)
        self._paint_text = Paced(TEXT_HZ)
        run.snapshot_provider = self.mission.snapshot
        run.background_provider = self.mission.displayed_background
        restore_window_geometry(self.root, f'dmsn.{run.id}')

    def save_geometry(self) -> None:
        self._save_geometry(self.root, f'dmsn.{self.run.id}')

    def close(self) -> None:
        self.save_geometry()
        self.events.close()
        self.mission.reference.close()
        self.running = False

    def selected(self) -> str:
        return self.notebook.tab(self.notebook.select(), 'text')

    def update(self) -> None:
        self.root.update_idletasks(); self.root.update()
        self.events.poll()
        tab = self.selected()
        if self._paint_map.due():
            if tab == 'Mission': self.mission.paint_map()
            elif tab == 'Status': self.status_tab.paint()
        if self._paint_text.due():
            self.mission.paint_text()
            self.setup.refresh()
            self._paint_status()

    def _paint_status(self) -> None:
        run = self.run
        errors = '; '.join(f'{k}: {v}' for k, v in run.ports.errors.items())
        self.status.set(
            f'{run.phase}{": " + run.outcome if run.outcome else ""}  ·  t {run.sim_time_s():.2f}s  ·  '
            f"session {run.run_id[:10] or '--'}  ·  report {run.report_dir or '--'}"
            + (f'  ·  {run.status_error}' if run.status_error else '')
            + (f'  ·  {errors}' if errors else ''))
