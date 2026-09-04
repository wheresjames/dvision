#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from dataclasses import fields

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dvision2_common import validate_id
from dcmn import theme
from dcmn.tktheme import apply_theme
from dcmn.window import (disable_input_method, restore_window_geometry,
                          save_window_geometry)
from dcmn.mapview import contained_size
from dalg.overlay import overlay_image, prediction_image
from dalg.profiles import load_profile, save_profile
from dalg.algo import ALGORITHMS, CONFIGS
from dalg.run import DalgRun


def parse_args(argv):
    parser = argparse.ArgumentParser(description="dvision2 algorithm demonstrator")
    parser.add_argument("--edit", action="store_true",
                        help="edit a profile without connecting to a simulator")
    parser.add_argument("--id")
    parser.add_argument("--profile", default=None)
    parser.add_argument("--no-ui", action="store_true")
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args(argv)
    if args.edit:
        if args.id: parser.error("--edit does not use --id")
        if args.no_ui: parser.error("--edit cannot be combined with --no-ui")
        args.profile = args.profile or "sgbm-manual"
    else:
        if not args.id: parser.error("--id is required unless --edit is used")
        validate_id(args.id)
        args.profile = args.profile or "sgbm-default"
    if args.timeout <= 0: parser.error("--timeout must be positive")
    return args


class ProfileEditor:
    """Profile form shared by connected and offline windows."""
    def __init__(self, page, profile, tk, ttk):
        self.profile, self.tk, self.ttk = profile, tk, ttk
        self.notice = tk.StringVar(value="Edit settings, then save a named profile")
        self.profile_name = tk.StringVar(value=profile.name)
        self.algorithm_name = tk.StringVar(value=profile.algorithm)
        tour = "" if profile.tour is None else str(profile.tour.relative_to(ROOT))
        sensors = profile.sensor_config or list(profile.sensors)
        self.tour = tk.StringVar(value=tour)
        self.sensors = tk.StringVar(value=json.dumps(sensors, sort_keys=True))
        labels = (("name", self.profile_name), ("tour (optional)", self.tour),
                  ("sensors (JSON)", self.sensors))
        for row, (label, variable) in enumerate(labels):
            ttk.Label(page, text=label, style="Dim.TLabel").grid(
                row=row, column=0, sticky="w", padx=(0, 12), pady=2)
            ttk.Entry(page, textvariable=variable).grid(row=row, column=1,
                                                        sticky="ew", pady=2)
        ttk.Button(page, text="Browse…", command=self._choose_tour).grid(
            row=1, column=2, sticky="ew", padx=(8, 0), pady=2)
        ttk.Label(page, text="algorithm", style="Dim.TLabel").grid(
            row=3, column=0, sticky="w", padx=(0, 12), pady=2)
        combo = ttk.Combobox(page, textvariable=self.algorithm_name,
                             values=tuple(ALGORITHMS), state="readonly")
        combo.grid(row=3, column=1, sticky="ew", pady=2)
        combo.bind("<<ComboboxSelected>>", self._settings_form)
        self.settings_frame = ttk.Frame(page)
        self.settings_frame.grid(row=4, column=0, columnspan=2, sticky="nsew", pady=8)
        ttk.Button(page, text="Save profile", command=self._save).grid(
            row=5, column=1, sticky="e")
        ttk.Label(page, textvariable=self.notice, style="Dim.TLabel").grid(
            row=6, column=0, columnspan=2, sticky="w", pady=(8, 0))
        page.columnconfigure(1, weight=1)
        self.setting_vars = {}; self._settings_form()

    def _choose_tour(self):
        from tkinter import filedialog
        selected = filedialog.askopenfilename(
            parent=self.settings_frame.winfo_toplevel(),
            title="Select tour",
            initialdir=ROOT / "assets/tours",
            filetypes=(("Tour JSON", "*.json"), ("All files", "*")))
        if not selected: return
        path = Path(selected).resolve()
        try:
            value = str(path.relative_to(ROOT))
        except ValueError:
            value = str(path)
        self.tour.set(value)

    def _settings_form(self, _event=None):
        for child in self.settings_frame.winfo_children(): child.destroy()
        self.setting_vars = {}
        config = CONFIGS.get(self.algorithm_name.get())
        if config is None: return
        defaults = config()
        for row, field in enumerate(fields(config)):
            self.ttk.Label(self.settings_frame, text=field.name,
                           style="Dim.TLabel").grid(
                row=row, column=0, sticky="w", padx=(0, 12), pady=2)
            value = self.profile.settings.get(field.name, getattr(defaults, field.name))
            var = self.tk.StringVar(value=str(value))
            self.setting_vars[field.name] = (var, type(getattr(defaults, field.name)))
            self.ttk.Entry(self.settings_frame, textvariable=var).grid(
                row=row, column=1, sticky="ew", pady=2)
        self.settings_frame.columnconfigure(1, weight=1)

    @staticmethod
    def _coerce(kind, text):
        """Parse a form field back to its configured type.

        bool is the trap: bool("False") is True, so every unticked setting
        would silently save as enabled.
        """
        if kind is bool:
            lowered = text.strip().lower()
            if lowered in ("1", "true", "yes", "on"): return True
            if lowered in ("0", "false", "no", "off"): return False
            raise ValueError(f"expected true or false, got {text!r}")
        return kind(text)

    def _save(self):
        try:
            settings = {name: self._coerce(kind, var.get())
                        for name, (var, kind) in self.setting_vars.items()}
            name = self.profile_name.get().strip()
            if not name or any(c not in
                    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
                    for c in name):
                raise ValueError("name may contain letters, digits, - and _")
            sensors = json.loads(self.sensors.get())
            if not isinstance(sensors, (list, dict)):
                raise ValueError("sensors must be a JSON list or object")
            tour = self.tour.get().strip() or None
            if tour is not None and not (ROOT / tour).is_file():
                raise ValueError(f"tour does not exist: {tour}")
            save_profile(ROOT / "dalg/profiles" / f"{name}.json", name=name,
                         algorithm=self.algorithm_name.get(), tour=tour,
                         sensors=sensors, settings=settings)
            self.notice.set(f"saved {name}")
        except Exception as exc:
            self.notice.set(f"not saved: {exc}")


class Window:
    def __init__(self, run):
        import tkinter as tk
        from tkinter import ttk
        from PIL import Image, ImageTk
        del Image
        self.tk, self.ImageTk, self.run = tk, ImageTk, run
        self.root = tk.Tk()
        apply_theme(self.root)
        self.root.title("dalg algorithm demonstrator")
        self.root.geometry("1000x460")
        self.root.minsize(480, 220)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.running = True
        notebook = ttk.Notebook(self.root)
        notebook.grid(row=0, column=0, sticky="nsew")
        live = ttk.Frame(notebook)
        notebook.add(live, text="Live")
        profiles = ttk.Frame(notebook, padding=12)
        notebook.add(profiles, text="Profiles")
        self.status = tk.StringVar(value="connecting")
        ttk.Label(live, textvariable=self.status, anchor="w",
                  style="Dim.TLabel").grid(
            row=0, column=0, columnspan=3, sticky="ew")
        self.canvases = []
        for col, title in enumerate(("camera", "prediction", "truth / prediction")):
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
        self.profile_editor = ProfileEditor(profiles, run.profile, tk, ttk)
        restore_window_geometry(self.root, f"dalg.{run.id}")

    def save_geometry(self):
        save_window_geometry(self.root, f"dalg.{self.run.id}")

    def close(self):
        self.save_geometry()
        self.running = False

    def update(self):
        self.root.update_idletasks(); self.root.update()
        self.status.set(f"{self.run.state}  run={self.run.run_id[:10]}  frames={self.run.frames}")
        images = []
        if self.run.last_frame is not None:
            from PIL import Image
            images.append(Image.fromarray(self.run.last_frame))
        else: images.append(None)
        result = self.run.preview_result
        if result is not None:
            # The same renderer the report writes, so what is on screen and
            # what is filed afterwards cannot drift apart. Scale 1: the canvas
            # fit below enlarges it to whatever the pane is.
            images.append(prediction_image(result.grid, scale=1))
            images.append(overlay_image(self.run.truth, result.grid))
        else: images.extend((None, None))
        for index, (canvas, image) in enumerate(zip(self.canvases, images)):
            if image is None: continue
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


class EditorWindow:
    def __init__(self, profile):
        import tkinter as tk
        from tkinter import ttk
        self.root = tk.Tk()
        apply_theme(self.root)
        self.root.title("dalg profile editor")
        self.root.geometry("620x620")
        self.root.minsize(440, 360)
        page = ttk.Frame(self.root, padding=12)
        page.grid(row=0, column=0, sticky="nsew")
        self.root.rowconfigure(0, weight=1); self.root.columnconfigure(0, weight=1)
        self.editor = ProfileEditor(page, profile, tk, ttk)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        restore_window_geometry(self.root, "dalg.editor")

    def run(self):
        self.root.mainloop()
        return 0

    def close(self):
        save_window_geometry(self.root, "dalg.editor")
        self.root.destroy()


def main(argv=None):
    disable_input_method()
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        profile = load_profile(args.profile, ROOT)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"dalg: cannot load profile {args.profile!r}: {exc}", file=sys.stderr)
        return 1
    if args.edit:
        try:
            import tkinter as tk
            return EditorWindow(profile).run()
        except ImportError as exc:
            print(f"dalg: tkinter is unavailable: {exc}", file=sys.stderr)
            return 1
        except tk.TclError as exc:
            print(f"dalg: cannot open profile editor: {exc}", file=sys.stderr)
            return 1
    run = DalgRun(args.id, profile, ROOT)
    window = None if args.no_ui else Window(run)
    deadline = time.monotonic() + args.timeout
    try:
        while ((window is None and not run.done) or
               (window is not None and window.running)):
            # Stepped even after the run has finished. step() already refuses
            # to start or advance a measurement once done, and it is the only
            # thing that drains the bus and publishes presence -- so skipping
            # it left a window open that had stopped listening for
            # system.shutdown and had aged out of every pipeline view.
            run.step()
            if window is not None:
                window.update()
                if run.shutdown_requested: window.close()
            if not run.done and time.monotonic() >= deadline:
                run.reason = "dalg timeout"
                if run.active: run.finish(partial=True)
                else: break
            time.sleep(.02)
    finally:
        run.close()
        if window is not None:
            try:
                window.save_geometry()
                window.root.destroy()
            except Exception: pass
    if run.report_dir:
        print(f"dalg: report directory -> {run.report_dir}", file=sys.stderr)
        html = run.report_dir / "report.html"
        if html.is_file():
            print(f"dalg: report -> {html}", file=sys.stderr)
    outcome = run.provenance.get(
        "coordinator_outcome", run.provenance.get("navigator_outcome"))
    return 0 if run.done and outcome == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
