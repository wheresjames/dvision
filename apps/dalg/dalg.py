#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# The repository root, two levels up now that the applications live under
# ``apps/``. ``apps`` itself is a source root rather than a package -- like a
# ``src/`` directory -- so sibling applications keep importing each other as
# ``dsim.dsim`` and ``dcmn.window`` with no ``apps.`` prefix anywhere.
ROOT = Path(__file__).resolve().parents[2]
APPS = ROOT / "apps"
for _path in (str(ROOT), str(APPS)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from dvision2_common import validate_id
from dcmn import theme
from dcmn.tktheme import apply_theme
from dcmn.window import (disable_input_method, restore_window_geometry,
                          save_window_geometry)
from dcmn.mapview import contained_size
from dcmn.pacing import Paced, TEXT_HZ, VIDEO_HZ
from dalg.overlay import prediction_image
from dalg.profiles import Profile, load_profile, load_profiles, preflight, profile_dir
from dalg.run import DalgRun


def join_negative_values(argv, options):
    """``--bounds -10,-5,30,25`` as ``--bounds=-10,-5,30,25``.

    argparse reads a value starting with ``-`` as another option, and negative
    local coordinates are ordinary, not an edge case.
    """
    argv, out = list(argv), []
    for token in argv:
        if out and out[-1] in options and token.startswith('-') and ',' in token:
            out[-1] = f'{out[-1]}={token}'
        else: out.append(token)
    return out


def parse_args(argv):
    argv = join_negative_values(argv, ('--bounds',))
    parser = argparse.ArgumentParser(
        description="dvision2 evidence producer: sensor samples in, evidence grids out")
    parser.add_argument("--edit", action="store_true",
                        help="edit a profile without connecting to a provider")
    parser.add_argument("--id")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--profile", default=None, help="one algorithm profile")
    selection.add_argument("--profiles", nargs='+', help="compose algorithm profiles")
    mapping = parser.add_argument_group("mapping (runtime geometry; never from a profile)")
    mapping.add_argument("--bounds", default=None,
                         help="xmin,ymin,xmax,ymax in local metres; overrides automatic sizing")
    mapping.add_argument("--cell-m", type=float, default=0.5, help="grid resolution in metres")
    mapping.add_argument("--z0-m", type=float, default=0.0, help="slab floor in the local frame")
    mapping.add_argument("--dz-m", type=float, default=3.0, help="slab thickness")
    mapping.add_argument("--mapping-budget-mib", type=float, default=512.0,
                         help="estimated mapping memory budget (not a process limit)")
    mapping.add_argument("--min-side-m", type=float, default=40.0)
    mapping.add_argument("--size-multiplier", type=float, default=2.5,
                         help="side length = max(min side, k * |start-goal|, |start-goal| + 2*margin)")
    mapping.add_argument("--margin-m", type=float, default=10.0)
    parser.add_argument("--camera-hz", type=float, default=5.0,
                        help="camera admission rate per source, data-clock Hz")
    parser.add_argument("--pose-max-age", type=float, default=0.5,
                        help="pose freshness limit in data-clock seconds")
    parser.add_argument("--recording-queue-mib", type=float, default=64.0)
    parser.add_argument("--recording-disk-mib", type=float, default=4096.0)
    parser.add_argument("--show-reference", action="store_true",
                        help="open with the reference-image background enabled "
                             "(a debug display; recorded in session provenance)")
    parser.add_argument("--no-ui", action="store_true")
    parser.add_argument("--timeout", type=float, default=0.0,
                        help="stop after this many wall seconds; 0 runs until shut down")
    args = parser.parse_args(argv)
    if args.profiles and len(args.profiles) > 1 and args.edit:
        parser.error("--edit accepts only one profile")
    if args.profiles and len(args.profiles) == 1:
        args.profile = args.profiles[0]
    if args.edit:
        if args.id: parser.error("--edit does not use --id")
        if args.no_ui: parser.error("--edit cannot be combined with --no-ui")
    else:
        if not args.id: parser.error("--id is required unless --edit is used")
        validate_id(args.id)
        if not args.profile and not args.profiles:
            parser.error("choose --profile NAME or --profiles NAME...; e.g. --profile lidar-baseline")
    if args.timeout < 0: parser.error("--timeout may not be negative")
    for name in ("camera_hz", "pose_max_age", "recording_queue_mib", "recording_disk_mib",
                 "mapping_budget_mib"):
        if getattr(args, name) <= 0: parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def mapping_config(args):
    """The runtime mapping configuration the command line asks for."""
    from dcmn.context import parse_bounds
    from dcmn.mapping import MappingConfig
    return MappingConfig(bounds=None if args.bounds is None else parse_bounds(args.bounds),
                         cell_m=args.cell_m, z0_m=args.z0_m, dz_m=args.dz_m,
                         budget_bytes=int(args.mapping_budget_mib * 2**20),
                         min_side_m=args.min_side_m, multiplier=args.size_multiplier,
                         margin_m=args.margin_m)


class Window:
    def __init__(self, run, *, show_reference=False):
        import tkinter as tk
        from tkinter import ttk
        from PIL import Image, ImageTk
        del Image
        self.tk, self.ImageTk, self.run = tk, ImageTk, run
        self.root = tk.Tk()
        apply_theme(self.root)
        self.root.title("dalg algorithm demonstrator")
        self.root.geometry("1120x780")
        self.root.minsize(720, 480)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.running = True
        notebook = ttk.Notebook(self.root)
        notebook.grid(row=0, column=0, sticky="nsew")
        live = ttk.Frame(notebook)
        notebook.add(live, text="Live")
        self.notebook, self.live = notebook, live
        from dcmn.map_pane import MapPane, ReferenceBackgroundHost
        from dcmn.event_viewer import EventViewer
        grids = ttk.Frame(notebook)
        notebook.add(grids, text="Grids")
        self.grids = grids
        # The reference background is opt-in and its every change recorded:
        # a truth map behind live evidence marks the session as assisted
        # (DV-MAPPING §7), and the provenance entry is how that marking
        # survives past the window.
        self.reference = ReferenceBackgroundHost(
            grids, run.id, enabled=show_reference, on_change=self._reference_changed)
        self.reference.widget.grid(row=0, column=0, columnspan=2, sticky="ew")
        self.map_pane = MapPane(grids, title="Camera evidence")
        self.map_pane.widget.grid(row=1, column=0, sticky="nsew")
        grids.rowconfigure(1, weight=1); grids.columnconfigure(0, weight=1)
        self.map_panes = {}
        # What the run files at close: the pane's own rendering, background
        # included, so the report shows exactly what the operator saw and the
        # archive names the exact reference revision behind it.
        run.background_provider = self._displayed_background
        if show_reference:
            run.note_display('reference-on', source='--show-reference')
        self.events = EventViewer(notebook, run.id)
        profiles = ttk.Frame(notebook, padding=12)
        notebook.add(profiles, text="Profile")
        notebook.add(self.events.page, text="Events")
        source = ttk.Combobox(live, state="readonly", values=(run.profile.algorithm,))
        self.source = source
        source.set(run.profile.algorithm)
        source.grid(row=0, column=0, sticky="w")
        self.status = tk.StringVar(value="connecting")
        ttk.Label(self.root, textvariable=self.status, anchor="w",
                  style="Dim.TLabel").grid(
            row=1, column=0, sticky="ew")
        self.canvases = []
        self.lidar_views = {}
        # What the sensor cell currently shows: 'camera' or a lidar source
        # id. Geometry is touched only when this changes -- repacking the cell
        # every paint unmaps and remaps it at video rate, which flickers the
        # picture and closes the renderer's dropdowns as fast as they open.
        self._sensor_shown = None
        self._source_values = ()
        # The sensor sample and the algorithm's own belief. No truth pane: the
        # world is the evaluator's business, never the producer's.
        for col, title in enumerate(("sensor", "prediction")):
            frame = ttk.Frame(live, style="Panel.TFrame")
            frame.grid(row=1, column=col, sticky="nsew", padx=3, pady=3)
            ttk.Label(frame, text=title, style="Brand.TLabel").pack(
                fill="x", padx=6, pady=(4, 3))
            canvas = tk.Canvas(frame, width=320, height=240, bg=theme.CANVAS,
                               highlightbackground=theme.GRID,
                               highlightcolor=theme.ACCENT,
                               highlightthickness=1)
            canvas.pack(fill="both", expand=True)
            self.canvases.append(canvas)
            live.columnconfigure(col, weight=1)
        live.rowconfigure(1, weight=1)
        self.root.columnconfigure(0, weight=1); self.root.rowconfigure(0, weight=1)
        self._paint_video = Paced(VIDEO_HZ)
        self._paint_text = Paced(TEXT_HZ)
        self.profile_editor = build_profile_editor(profiles, run.profile, tk, ttk)
        restore_window_geometry(self.root, f"dalg.{run.id}")

    def save_geometry(self):
        save_window_geometry(self.root, f"dalg.{self.run.id}")

    def close(self):
        self.save_geometry()
        self.running = False
        self.events.close()
        self.reference.close()

    def _displayed_background(self):
        """The background the Grids panes last rendered -- the exact revision a report files."""
        return self.map_pane.state.background

    def _reference_changed(self, enabled, opacity):
        """Record an operator's display decision: provenance, not a secret."""
        self.run.note_display('reference-on' if enabled else 'reference-off',
                              opacity=round(opacity, 3))

    def update(self):
        self.root.update_idletasks(); self.root.update()
        self.events.poll()
        # The loop spins at 50 Hz so the bus stays drained and shutdown stays
        # heard; the panes below do not need to be repainted that often.
        if self._paint_text.due():
            run = self.run
            geometry = run.geometry
            extent = ("--" if geometry is None else
                      f"[{geometry.origin_x_m:g},{geometry.origin_y_m:g}] "
                      f"{geometry.extent_m[0]:g}x{geometry.extent_m[1]:g} m @ {geometry.cell_m:g} m "
                      f"({run.geometry_basis})")
            text = (f"{run.state} ({run.admission})  session={run.run_id[:10]}  "
                    f"frames={run.frames}  t={run.sim_time_s():.2f}s  coverage={extent}  "
                    f"intake={run.sensor_intake.grade()}  report={run.report_dir or '--'}")
            if run.reason: text += f"  -- {run.reason}"
            if run.unavailable_sources:
                text += "  omitted: " + '; '.join(
                    f"{sid}: {reason}" for sid, reason in run.unavailable_sources.items())
            self.status.set(text)
            self.profile_editor.set_manifest(run.sensor_session.manifest)
            self.profile_editor.set_runtime(geometry=geometry, basis=run.geometry_basis,
                                            state=run.state, reason=run.reason)
        sources = self.run.sources
        available = list(sources.states) if sources is not None else []
        if available and tuple(available) != self._source_values:
            # Reconfigured only when the source list changes: a values update
            # every loop turn closes the selector's dropdown as it opens.
            self._source_values = tuple(available)
            self.source.configure(values=available)
        if available and self.source.get() not in available: self.source.set(available[0])
        if self.notebook.select() == str(self.grids):
            from dcmn.map_pane import MapPane
            background = self.reference.background()
            producers = ({sid: state.evidence for sid, state in sources.states.items()}
                         if sources is not None else {})
            for index, (sid, producer) in enumerate(producers.items()):
                if sid not in self.map_panes:
                    pane = self.map_pane if not self.map_panes else MapPane(self.grids, title=sid)
                    pane.widget.grid(row=index//2 + 1, column=index%2, sticky="nsew")
                    self.grids.rowconfigure(index//2 + 1, weight=1)
                    self.grids.columnconfigure(index%2, weight=1)
                    self.map_panes[sid] = pane
                pane = self.map_panes[sid]
                grid = producer.latest
                pane.set_grid(grid, sim_now_s=self.run.sim_time_s())
                pane.set_background(background, refresh=False)
                sensor = producer.publisher.sources[sid]['sensor']
                health = self.run.source_health.get(sensor, {}).get('state', 'starting')
                error = sources.states[sid].error if sources is not None else ''
                pane.set_header(f"{sid}\nrev={grid.revision if grid else 0}  "
                    f"publish={producer.rate_hz:.2f} Hz  "
                    f"sim={grid.sim_time_s if grid else 0:.2f}s  intake={health} {error}")
                pane.refresh()
            if not producers:
                self.map_pane.set_header(f"No evidence yet: {self.run.state} {self.run.reason}")
                self.map_pane.refresh()
        if self.notebook.select() != str(self.live): return
        if not self._paint_video.due():
            return
        from PIL import Image
        selected = sources.states.get(self.source.get()) if sources is not None else None
        lidar = selected is not None and selected.config.algorithm == "lidar_inverse"
        wanted = selected.config.id if lidar else 'camera'
        if wanted != self._sensor_shown:
            for renderer in self.lidar_views.values(): renderer.widget.pack_forget()
            self.canvases[0].pack_forget()
            if lidar:
                from dcmn.device_view import PolarRenderer
                if selected.config.id not in self.lidar_views:
                    self.lidar_views[selected.config.id] = PolarRenderer(
                        self.canvases[0].master, selected.stream.entry)
                self.lidar_views[selected.config.id].widget.pack(fill="both", expand=True)
            else:
                self.canvases[0].pack(fill="both", expand=True)
            self._sensor_shown = wanted
        if lidar and selected.stream.latest is not None:
            self.lidar_views[selected.config.id].draw(selected.stream.latest)
        result = selected.preview if selected is not None else self.run.preview_result
        frame = selected.last_image if selected is not None else self.run.last_frame
        images = [None if frame is None else Image.fromarray(frame)]
        images.append(None if result is None else prediction_image(result.grid, scale=1))
        for index, (canvas, image) in enumerate(zip(self.canvases, images)):
            if image is None:
                canvas.delete("preview")
                canvas.create_text(max(1, canvas.winfo_width())//2, max(1, canvas.winfo_height())//2,
                    text="No camera image for this source" if index == 0 else "Waiting for samples",
                    fill=theme.TEXT, tags="preview")
                continue
            width = max(1, canvas.winfo_width())
            height = max(1, canvas.winfo_height())
            fitted = contained_size(image.width, image.height, width, height)
            if image.size != fitted:
                from PIL import Image
                resampling = (Image.Resampling.LANCZOS if index == 0
                              else Image.Resampling.NEAREST)
                image = image.resize(fitted, resampling)
            photo = self.ImageTk.PhotoImage(image)
            canvas.delete("preview")
            canvas.create_image(width // 2, height // 2, image=photo,
                                anchor="center", tags="preview")
            canvas.image = photo


def build_profile_editor(page, profile, tk, ttk, *, on_open=None):
    """The profile editor, the same in the Profile tab and under --edit.

    There is one, and it edits sources and their settings only: tours belong
    to mission tooling and map extents to the runtime (--bounds/--cell-m).
    ``on_open`` adds an Open button for a host that can swap the profile.
    """
    from dalg.source_editor import SourceEditor
    return SourceEditor(page, profile, tk, ttk, root=ROOT, on_open=on_open)


class EditorWindow:
    def __init__(self, profile, *, root=None):
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
        self.tk, self.ttk = tk, ttk
        # ``root`` lets a test hand in a hidden one; run from the command line
        # this is the application's own window.
        self.root = root if root is not None else tk.Tk()
        apply_theme(self.root)
        self.root.title("dalg profile editor")
        self.root.minsize(560, 360)
        self.page = ttk.Frame(self.root, padding=12)
        self.page.grid(row=0, column=0, sticky="nsew")
        self.root.rowconfigure(0, weight=1); self.root.columnconfigure(0, weight=1)
        # The two dialogs, held as attributes so a test can answer them.
        self.choose_file = filedialog.askopenfilename
        self.confirm = messagebox.askyesno
        self.editor = self._build(profile)
        self.root.geometry("1080x740")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        restore_window_geometry(self.root, "dalg.editor")

    def _build(self, profile):
        return build_profile_editor(self.page, profile, self.tk, self.ttk,
                                    on_open=self.open_profile)

    def open_profile(self, path=None):
        """Replace the profile being edited with one read from disk.

        The editor is rebuilt rather than refilled: it holds drafts, per-row
        errors, a range sensor and a settings form generated for one
        algorithm, and a fresh one cannot carry any of that across by mistake.
        A file that will not load leaves the current profile exactly where it
        was and says why. Returns whether a profile was opened.
        """
        editor = self.editor
        if editor.dirty and not self.confirm(
                "Discard changes?",
                f"Discard the unsaved changes to {editor.name.get() or 'this profile'}?",
                parent=self.root):
            return False
        if path is None:
            path = self.choose_file(
                parent=self.root, title="Open profile", initialdir=profile_dir(ROOT),
                filetypes=(("Profile JSON", "*.json"), ("All files", "*")))
        if not path: return False
        try:
            profile = load_profile(str(path), ROOT)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            editor.notice.set(f"Not opened: {Path(path).name}: {exc}")
            return False
        for child in self.page.winfo_children(): child.destroy()
        self.editor = self._build(profile)
        # Prefixed rather than replaced: a profile that opens with problems
        # has already said so in this line, and that must stay visible.
        self.editor.notice.set(f"Opened {Path(path).name}. {self.editor.notice.get()}")
        return True

    def run(self):
        self.root.mainloop()
        return 0

    def close(self):
        save_window_geometry(self.root, "dalg.editor")
        self.root.destroy()


def main(argv=None):
    disable_input_method()
    args = parse_args(sys.argv[1:] if argv is None else argv)
    profile = None
    if args.profile is not None or args.profiles:
        try:
            profile = load_profiles(args.profiles or [args.profile], ROOT)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"dalg: cannot load profiles {args.profiles or [args.profile]!r}: {exc}", file=sys.stderr)
            return 1
    if args.edit:
        # A bare --edit starts a new, empty profile rather than any committed one.
        profile = profile or Profile('new-profile', (), '')
        try:
            import tkinter as tk
            return EditorWindow(profile).run()
        except ImportError as exc:
            print(f"dalg: tkinter is unavailable: {exc}", file=sys.stderr)
            return 1
        except tk.TclError as exc:
            print(f"dalg: cannot open profile editor: {exc}", file=sys.stderr)
            return 1
    errors = preflight(profile, ROOT)
    if errors:
        for error in errors: print(f"dalg: {error}", file=sys.stderr)
        return 1
    try:
        mapping = mapping_config(args)
    except ValueError as exc:
        print(f"dalg: {exc}", file=sys.stderr)
        return 1
    run = DalgRun(args.id, profile, ROOT, mapping=mapping, camera_hz=args.camera_hz,
                  pose_max_age=args.pose_max_age,
                  recording_queue=int(args.recording_queue_mib * 2**20),
                  recording_disk=int(args.recording_disk_mib * 2**20))
    window = None if args.no_ui else Window(run, show_reference=args.show_reference)
    deadline = None if args.timeout <= 0 else time.monotonic() + args.timeout
    try:
        while ((window is None and not run.done) or
               (window is not None and window.running)):
            # Stepped even after stopping: it is what drains the bus and keeps
            # presence alive while a window is still open.
            run.step()
            if window is not None:
                window.update()
                if run.shutdown_requested: window.close()
            if not run.done and deadline is not None and time.monotonic() >= deadline:
                run.finish(partial=True)
                if window is None: break
            time.sleep(run.poll_delay())
    except KeyboardInterrupt:
        run.finish(partial=True)
    finally:
        run.close()
        if window is not None:
            try:
                window.save_geometry()
                window.root.destroy()
            except Exception: pass
    if run.report_dir:
        print(f"dalg: report directory -> {run.report_dir}", file=sys.stderr)
    # A clean stop is success whether or not a mission ran; only a
    # configuration or allocation failure at exit is not.
    return 1 if run.state in ("CONFIGURATION_ERROR", "ALLOCATION_FAILED") else 0


if __name__ == "__main__":
    raise SystemExit(main())
