#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk

_MODULE_STARTED = time.perf_counter()

import numpy as np
from PIL import Image, ImageTk

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dcmn import theme
from dcmn.tktheme import apply_theme
from dvision2_common import (
    STATUS_KEYS, controlled_command, load_pymembus, new_control_identity,
    restore_window_pos, save_window_pos, shared_names, validate_id,
)

# ---------------------------------------------------------------------------
# Dark theme palette
# ---------------------------------------------------------------------------
# One palette, in dcmn.theme, so every window's map is the same colour.
_BG        = theme.BG          # window / frame background
_BG_PANEL  = theme.PANEL       # panel surface
_BG_ENTRY  = theme.ENTRY       # entry / spinbox field
_FG        = theme.TEXT        # primary text
_FG_DIM    = theme.DIM         # secondary / key labels
_ACCENT    = theme.ACCENT      # focus / primary actions
_ACCENT_OK = theme.OK
_ACCENT_BAD = theme.DANGER
_BORDER    = theme.GRID        # widget borders
_BTN_BG    = theme.BUTTON      # button resting
_BTN_ACT   = theme.BUTTON_ACTIVE   # button hover
_VIDEO_BG  = theme.CANVAS      # video viewport background

_MANUAL_YAW_RATE_DPS = 45.0


# ---------------------------------------------------------------------------
# Joystick
# ---------------------------------------------------------------------------

class JoystickManager:
    """Polls a gamepad via pygame's joystick subsystem (no pygame display needed).

    Xbox controller axis layout (Linux xpad / BT hid-xpad):
      0  left-stick X      right = +1   → strafe right
      1  left-stick Y      down  = +1   → negate for forward
      2  left trigger      rest  = -1   (unused)
      3  right-stick X     right = +1   → yaw right command
      4  right-stick Y     down  = +1   → negate for ascend
      5  right trigger     rest  = -1   (unused)

    Button layout:
      0 A     – hover/zero     1 B  – land
      2 X     – arm toggle     3 Y  – takeoff
      4 LB    – (unused)       5 RB – (unused)
      6 Back  – disarm         7 Start – arm
    """

    DEADZONE = 0.12

    def __init__(self) -> None:
        self._pygame = None
        self._enabled = False
        self._joystick = None
        self.name = ""
        self.error = ""
        self._axes: list[float] = []
        self._buttons: dict[int, bool] = {}
        self._prev_buttons: dict[int, bool] = {}
        self._rising: set[int] = set()

        try:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
            os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
            import pygame
            import pygame.display
            import pygame.joystick
            pygame.display.init()   # required for event pump; dummy driver = no window
            pygame.joystick.init()
            self._pygame = pygame
            self._enabled = True
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"

    @property
    def available(self) -> bool:
        return self._joystick is not None

    def poll(self) -> None:
        if not self._enabled:
            return
        pg = self._pygame
        try:
            pg.event.pump()
        except Exception:
            return

        count = pg.joystick.get_count()
        if count == 0:
            if self._joystick is not None:
                try:
                    self._joystick.quit()
                except Exception:
                    pass
                self._joystick = None
                self.name = ""
            self._axes = []
            self._prev_buttons = dict(self._buttons)
            self._buttons = {}
            self._rising = set()
            return

        if self._joystick is None:
            try:
                joy = pg.joystick.Joystick(0)
                joy.init()
                self._joystick = joy
                self.name = joy.get_name()
            except Exception:
                return

        joy = self._joystick
        try:
            self._axes = [joy.get_axis(i) for i in range(joy.get_numaxes())]
            self._prev_buttons = dict(self._buttons)
            self._buttons = {i: bool(joy.get_button(i)) for i in range(joy.get_numbuttons())}
            self._rising = {i for i, v in self._buttons.items() if v and not self._prev_buttons.get(i)}
        except Exception:
            self._axes = []
            self._rising = set()

    def _axis(self, n: int) -> float:
        if n >= len(self._axes):
            return 0.0
        v = self._axes[n]
        return v if abs(v) >= self.DEADZONE else 0.0

    def button_pressed(self, n: int) -> bool:
        return n in self._rising

    @property
    def forward(self) -> float:
        return -self._axis(1)

    @property
    def right(self) -> float:
        return self._axis(0)

    @property
    def up(self) -> float:
        return -self._axis(4)

    @property
    def yaw(self) -> float:
        return self._axis(3)

    def status_str(self) -> str:
        if not self._enabled:
            return "pygame unavailable"
        if self._joystick is None:
            return "none"
        return self.name[:28]

    def close(self) -> None:
        if self._joystick is not None:
            try:
                self._joystick.quit()
            except Exception:
                pass
            self._joystick = None
        if self._enabled and self._pygame is not None:
            try:
                self._pygame.joystick.quit()
            except Exception:
                pass


def _clamp(v: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


# ---------------------------------------------------------------------------
# Controller UI
# ---------------------------------------------------------------------------

class DroneController:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        # Potentially slow optional-interface discovery is deferred until the
        # window has painted. Shared-memory *connection* never gates window
        # creation; open_missing() keeps retrying it from the Tk loop.
        self.pymembus = None
        self.names = shared_names(args.id)
        self.control_source, self.control_lease = new_control_identity(f"dctl-{args.id}")
        self.video = None
        self.command = None
        self.status = None
        self.status_epoch = 0
        self.last_video_seq = -1
        self.last_status_update = 0.0
        self.held: set[str] = set()
        self.running = True
        self._joy_legend_shown = None  # None = not yet determined
        self._held_velocity_active = False
        self._last_heartbeat = 0.0
        self._last_open_attempt = 0.0

        self.joy = _DisabledJoystick()
        self._interfaces_initializing = True
        self._ui_ready = False
        self._dashboard_scheduled = False

        stage_started = time.perf_counter()
        self.root = tk.Tk()
        self._trace(f"Tk created in {time.perf_counter() - stage_started:.3f}s")
        self.root.title(f"dctl {args.id}")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.bind("<Map>", self._on_window_mapped, add="+")
        self._window_mapped = False
        self.photo = None
        self.status_vars: dict[str, tk.StringVar] = {}
        self.log_var = tk.StringVar(value="starting")
        self.conn_var = tk.StringVar(
            value="window ready; initializing interfaces, waiting for dsim")
        self.armed_var = tk.StringVar(value="unknown")
        self.takeoff_alt = tk.DoubleVar(value=3.0)
        self._joy_name_var = tk.StringVar(value="")
        self._build_startup_window()
        restore_window_pos(self.root, f"dctl.{args.id}")

    def _build_startup_window(self) -> None:
        """A cheap first frame that does not depend on dashboard layout."""
        self._apply_dark_theme()
        self.root.configure(background=_BG)
        # Keep the top-level at its dashboard size from its first map. Resetting
        # a mapped 640x180 splash to the widgets' requested size caused a second
        # multi-second geometry transaction on some X11 window managers.
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        window_w = max(320, min(1280, screen_w - 80))
        window_h = max(320, min(800, screen_h - 80))
        self.root.geometry(f"{window_w}x{window_h}")
        self.root.minsize(min(900, window_w), min(600, window_h))
        frame = ttk.Frame(self.root, padding=24)
        self._startup_frame = frame
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text=f"dctl  {self.args.id}", style="Brand.TLabel").pack(
            anchor="w")
        ttk.Label(frame, textvariable=self.conn_var, style="HeaderDim.TLabel").pack(
            anchor="w", pady=(12, 0))
        ttk.Label(frame, text="Connecting video; controls are loading…",
                  style="Dim.TLabel").pack(anchor="w", pady=(8, 8))
        self.video_label = ttk.Label(
            frame, text="waiting for dsim video", anchor="center",
            style="Video.TLabel")
        self.video_label.pack(fill="both", expand=True)
        ttk.Button(frame, text="Load controls now",
                   command=self._schedule_dashboard).pack(anchor="e", pady=(8, 0))

    def _on_window_mapped(self, event) -> None:
        """Begin external probing only after the actual toplevel is visible."""
        if event.widget is not self.root or self._window_mapped:
            return
        self._window_mapped = True
        self._trace(
            f"window mapped after {time.perf_counter() - _MODULE_STARTED:.3f}s")
        # Transport/video comes first. The expensive dashboard realization is
        # deliberately later so it cannot hide the first available frame.
        self.root.after(25, self._initialize_interfaces)

    def _schedule_dashboard(self, delay_ms: int = 0) -> None:
        if self._dashboard_scheduled or self._ui_ready:
            return
        self._dashboard_scheduled = True
        self.root.after(delay_ms, self._finish_window)

    def _finish_window(self) -> None:
        if not self.running:
            return
        stage_started = time.perf_counter()
        width = max(1, self.root.winfo_width())
        height = max(1, self.root.winfo_height())
        dashboard = ttk.Frame(self.root)
        self._dashboard_host = dashboard
        # Realize the large widget tree outside the visible top-level while the
        # startup viewport remains mapped with its most recent video frame.
        dashboard.place(x=-width * 2, y=0, width=width, height=height)
        self.build_ui(dashboard)
        dashboard.update_idletasks()
        if self.photo is not None:
            self.video_label.configure(image=self.photo, text="")
        dashboard.place_configure(x=0, y=0, relwidth=1, relheight=1,
                                  width=0, height=0)
        dashboard.lower(self._startup_frame)
        self._trace(
            f"dashboard staged in {time.perf_counter() - stage_started:.3f}s")
        # Mapping the large tree is asynchronous and is the slow operation on
        # affected X11 setups. Keep the video shell stacked above it until Tk
        # returns to this timer, which happens after that realization settles.
        self.root.after(100, self._reveal_dashboard)

    def _reveal_dashboard(self) -> None:
        if not self.running:
            return
        self._dashboard_host.lift()
        self._startup_frame.destroy()
        self._ui_ready = True
        self._trace(
            f"dashboard visible after {time.perf_counter() - _MODULE_STARTED:.3f}s")

    def _initialize_interfaces(self) -> None:
        """Run imports/probes only after Tk has had a chance to paint."""
        if not self.running:
            return
        stage_started = time.perf_counter()
        try:
            self.pymembus = load_pymembus()
        except Exception as exc:
            self._interfaces_initializing = False
            self.log(f"shared memory unavailable: {exc}")
            self.update_connection_text()
            return
        self._trace(f"pymembus loaded in {time.perf_counter() - stage_started:.3f}s")
        if not self.args.no_joystick:
            self.conn_var.set(
                "window ready; shared memory loaded; probing joystick")
            self.root.update_idletasks()
            stage_started = time.perf_counter()
            self.joy = JoystickManager()
            self._trace(
                f"joystick probe finished in "
                f"{time.perf_counter() - stage_started:.3f}s")
        self._interfaces_initializing = False
        if not self.args.no_joystick and getattr(self.joy, "error", ""):
            message = f"joystick disabled: {self.joy.error}"
            self.log(message)
            print(f"dctl: {message}", file=sys.stderr)
        self.update_connection_text()

    # ------------------------------------------------------------------
    # Theme
    # ------------------------------------------------------------------

    def _apply_dark_theme(self) -> None:
        # The shared base, then the widgets only the manual pilot uses: the
        # video viewport and the joystick legend.
        s = apply_theme(self.root)
        s.configure("Video.TLabel", background=_VIDEO_BG, foreground=_FG_DIM)

        s.configure("TLabelframe",
            background=_BG_PANEL, bordercolor=_BORDER,
            lightcolor=_BORDER, darkcolor=_BORDER)
        s.configure("TLabelframe.Label",
            background=_BG_PANEL, foreground=_ACCENT)

        # Joystick legend uses a subtler accent colour
        s.configure("Joy.TLabelframe",
            background=_BG_PANEL, bordercolor=_BORDER,
            lightcolor=_BORDER, darkcolor=_BORDER)
        s.configure("Joy.TLabelframe.Label",
            background=_BG_PANEL, foreground=_FG_DIM)
        s.configure("Joy.TFrame", background=_BG_PANEL)
        s.configure("JoyKey.TLabel",
            background=_BG_PANEL, foreground=_ACCENT,
            font=("TkFixedFont", 9, "bold"))
        s.configure("JoyVal.TLabel",
            background=_BG_PANEL, foreground=_FG,
            font=("TkFixedFont", 9))
        s.configure("JoySec.TLabel",
            background=_BG_PANEL, foreground=_FG_DIM,
            font=("TkDefaultFont", 8))

        s.configure("TSpinbox",
            background=_BG_ENTRY, foreground=_FG,
            fieldbackground=_BG_ENTRY, bordercolor=_BORDER,
            arrowcolor=_FG, insertcolor=_FG,
            lightcolor=_BORDER, darkcolor=_BORDER)
        s.map("TSpinbox",
            fieldbackground=[("readonly", _BG_ENTRY)],
            bordercolor=[("focus", _ACCENT)])

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def build_ui(self, parent=None) -> None:
        self._apply_dark_theme()
        parent = self.root if parent is None else parent

        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)

        top = ttk.Frame(parent, padding=(12, 10), style="Header.TFrame")
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(1, weight=1)
        ttk.Label(top, text=f"dctl  {self.args.id}", style="Brand.TLabel").grid(
            row=0, column=0, sticky="w", padx=(0, 12))
        ttk.Label(top, textvariable=self.conn_var, style="HeaderDim.TLabel").grid(
            row=0, column=1, sticky="w")

        body = ttk.Frame(parent, padding=(12, 12, 12, 12))
        body.grid(row=1, column=0, sticky="nsew")
        body.columnconfigure(0, weight=1)
        body.rowconfigure(0, weight=1)

        # video_area holds the video and the legend below it so the legend
        # is always bounded by the video column and never bleeds into the side panel.
        video_area = ttk.Frame(body)
        video_area.grid(row=0, column=0, sticky="nsew")
        video_area.columnconfigure(0, weight=1)
        video_area.rowconfigure(0, weight=1)

        self.video_label = ttk.Label(
            video_area, text="waiting for video", anchor="center",
            style="Video.TLabel")
        self.video_label.grid(row=0, column=0, sticky="nsew")

        side = ttk.Frame(body, padding=(12, 0, 0, 0))
        side.grid(row=0, column=1, sticky="ns")

        # ── Controls ──────────────────────────────────────────────────
        controls = ttk.LabelFrame(side, text="Controls", padding=8)
        controls.grid(row=0, column=0, sticky="ew")
        ttk.Button(controls, text="Arm", style="Accent.TButton",
                   command=lambda: self.send_command("arm", armed=True)
                   ).grid(row=0, column=0, sticky="ew", padx=(0, 6), pady=(0, 6))
        ttk.Button(controls, text="Disarm", style="Danger.TButton",
                   command=lambda: self.send_command("arm", armed=False)
                   ).grid(row=0, column=1, sticky="ew", pady=(0, 6))
        ttk.Button(controls, text="Takeoff", style="Accent.TButton",
                   command=self.takeoff).grid(row=1, column=0, sticky="ew", padx=(0, 6), pady=(0, 6))
        ttk.Spinbox(controls, from_=0.5, to=20.0, increment=0.5,
                    textvariable=self.takeoff_alt, width=6
                    ).grid(row=1, column=1, sticky="ew", pady=(0, 6))
        ttk.Button(controls, text="Land", style="Danger.TButton",
                   command=lambda: self.send_command("land")
                   ).grid(row=2, column=0, sticky="ew", padx=(0, 6))
        ttk.Button(controls, text="Stop", style="Danger.TButton",
                   command=lambda: self.send_command("zero")
                   ).grid(row=2, column=1, sticky="ew")
        ttk.Button(controls, text="Take Control", command=self.take_control
                   ).grid(row=3, column=0, sticky="ew", padx=(0, 6), pady=(6, 0))
        ttk.Button(controls, text="Release Control", command=self.release_control
                   ).grid(row=3, column=1, sticky="ew", pady=(6, 0))

        # ── Movement ──────────────────────────────────────────────────
        move = ttk.LabelFrame(side, text="Movement", padding=8)
        move.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        ttk.Button(move, text="Yaw Left",
                   command=lambda: self.send_velocity(yaw_rate_dps=_manual_yaw_rate(-1.0))
                   ).grid(row=0, column=0, sticky="ew")
        ttk.Button(move, text="Forward",
                   command=lambda: self.send_velocity(forward_mps=self.args.speed)
                   ).grid(row=0, column=1, sticky="ew")
        ttk.Button(move, text="Yaw Right",
                   command=lambda: self.send_velocity(yaw_rate_dps=_manual_yaw_rate(1.0))
                   ).grid(row=0, column=2, sticky="ew")
        ttk.Button(move, text="Left",
                   command=lambda: self.send_velocity(right_mps=-self.args.speed)
                   ).grid(row=1, column=0, sticky="ew")
        ttk.Button(move, text="Hover",
                   command=lambda: self.send_command("zero")
                   ).grid(row=1, column=1, sticky="ew")
        ttk.Button(move, text="Right",
                   command=lambda: self.send_velocity(right_mps=self.args.speed)
                   ).grid(row=1, column=2, sticky="ew")
        ttk.Button(move, text="Up",
                   command=lambda: self.send_velocity(up_mps=self.args.vertical_speed)
                   ).grid(row=2, column=0, sticky="ew")
        ttk.Button(move, text="Back",
                   command=lambda: self.send_velocity(forward_mps=-self.args.speed)
                   ).grid(row=2, column=1, sticky="ew")
        ttk.Button(move, text="Down",
                   command=lambda: self.send_velocity(up_mps=-self.args.vertical_speed)
                   ).grid(row=2, column=2, sticky="ew")
        for col in range(3):
            move.columnconfigure(col, weight=1)

        # ── Telemetry ─────────────────────────────────────────────────
        telemetry = ttk.LabelFrame(side, text="Telemetry", padding=8)
        telemetry.grid(row=2, column=0, sticky="new", pady=(10, 0))
        telem_rows = [
            ("armed",    "drone.armed"),
            ("mode",     "drone.mode"),
            ("heading",  "drone.heading_deg"),
            ("gps lat",  "drone.lat_deg"),
            ("gps lon",  "drone.lon_deg"),
            ("gps alt",  "drone.alt_m"),
            ("x",        "drone.x_m"),
            ("y",        "drone.y_m"),
            ("z",        "drone.z_m"),
            ("speed",    "drone.speed_mps"),
            ("vz",       "drone.vz_mps"),
            ("roll",     "drone.roll_deg"),
            ("pitch",    "drone.pitch_deg"),
            ("battery",  "drone.battery_pct"),
            ("control",  "control.owner"),
            ("last cmd", "link.last_command_type"),
            ("accepted", "command.result.accepted"),
            ("reason",   "command.result.reason"),
            ("status",   "status.message"),
        ]
        half = (len(telem_rows) + 1) // 2
        for i, (label, key) in enumerate(telem_rows):
            grid_row = i if i < half else i - half
            col_base = 0 if i < half else 3
            ttk.Label(telemetry, text=label, style="Dim.TLabel").grid(
                row=grid_row, column=col_base, sticky="w",
                padx=(16 if col_base else 0, 8))
            var = tk.StringVar(value="-")
            self.status_vars[key] = var
            ttk.Label(telemetry, textvariable=var, width=14).grid(
                row=grid_row, column=col_base + 1, sticky="w")

        # ── Log ───────────────────────────────────────────────────────
        # A Configure handler used to assign the label's current width back to
        # wraplength on every geometry event. That changes the requested size,
        # causes another geometry event, and made Tk spend several seconds in
        # a first-map feedback loop on some window managers.
        _log_lbl = ttk.Label(side, textvariable=self.log_var,
                             style="Dim.TLabel", wraplength=420)
        _log_lbl.grid(row=3, column=0, sticky="ew", pady=(10, 0))

        # ── Legends (joystick or keyboard) under the video ────────────
        below_video = ttk.Frame(video_area, padding=(0, 8, 0, 0))
        below_video.grid(row=1, column=0, sticky="ew")
        below_video.columnconfigure(0, weight=1)
        self._build_joy_legend(below_video)
        self._build_kb_legend(below_video)

        # Key bindings
        for key in (
            "w", "a", "s", "d", "q", "e", "r", "f",
            "Up", "Down", "Left", "Right", "Prior", "Next",
            "Page_Up", "Page_Down", "Home", "End",
        ):
            self.root.bind(f"<KeyPress-{key}>",  self.key_down)
            self.root.bind(f"<KeyRelease-{key}>", self.key_up)
        self.root.bind("<space>",       lambda _e: self.send_command("zero"))
        self.root.bind("<KeyPress-t>",  lambda _e: self.takeoff())
        self.root.bind("<KeyPress-l>",  lambda _e: self.send_command("land"))
        self.root.bind("<KeyPress-m>",  lambda _e: self.toggle_arm())

    def _build_joy_legend(self, parent: ttk.Frame) -> None:
        """Build the gamepad legend panel (initially not gridded)."""
        frame = ttk.LabelFrame(parent, text="Gamepad", padding=(8, 4, 8, 8),
                               style="Joy.TLabelframe")
        self._joy_legend = frame

        inner = ttk.Frame(frame, style="Joy.TFrame")
        inner.grid(row=0, column=0, sticky="ew")

        # Controller name spans both columns
        ttk.Label(inner, textvariable=self._joy_name_var,
                  style="JoySec.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 6))

        # ── Left: Sticks ──────────────────────────────────────────────
        sticks_frame = ttk.Frame(inner, style="Joy.TFrame")
        sticks_frame.grid(row=1, column=0, sticky="nw", padx=(0, 24))

        ttk.Label(sticks_frame, text="Sticks", style="JoySec.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 4))

        stick_rows = [
            ("L ↕ ↔", "fwd / back / strafe"),
            ("R ↔",    "yaw"),
            ("R ↕",    "up / down"),
        ]
        for i, (key, val) in enumerate(stick_rows):
            ttk.Label(sticks_frame, text=f"  {key}", style="JoyKey.TLabel").grid(
                row=1 + i, column=0, sticky="w")
            ttk.Label(sticks_frame, text=val, style="JoyVal.TLabel").grid(
                row=1 + i, column=1, sticky="w", padx=(6, 0))

        # ── Right: Buttons ────────────────────────────────────────────
        btns_frame = ttk.Frame(inner, style="Joy.TFrame")
        btns_frame.grid(row=1, column=1, sticky="nw")

        ttk.Label(btns_frame, text="Buttons", style="JoySec.TLabel").grid(
            row=0, column=0, columnspan=4, sticky="w", pady=(0, 4))

        btn_rows = [
            ("A",     "hover",       "B",    "land"),
            ("Y",     "takeoff",     "X",    "arm toggle"),
            ("Start", "arm",         "Back", "disarm"),
        ]
        for i, (k1, v1, k2, v2) in enumerate(btn_rows):
            ttk.Label(btns_frame, text=f"  {k1}", style="JoyKey.TLabel").grid(
                row=1 + i, column=0, sticky="w")
            ttk.Label(btns_frame, text=v1, style="JoyVal.TLabel").grid(
                row=1 + i, column=1, sticky="w", padx=(4, 12))
            ttk.Label(btns_frame, text=k2, style="JoyKey.TLabel").grid(
                row=1 + i, column=2, sticky="w")
            ttk.Label(btns_frame, text=v2, style="JoyVal.TLabel").grid(
                row=1 + i, column=3, sticky="w", padx=(4, 0))

    def _build_kb_legend(self, parent: ttk.Frame) -> None:
        """Build the keyboard shortcut legend panel (shown when no joystick)."""
        frame = ttk.LabelFrame(parent, text="Keyboard", padding=(8, 4, 8, 8),
                               style="Joy.TLabelframe")
        self._kb_legend = frame

        inner = ttk.Frame(frame, style="Joy.TFrame")
        inner.grid(row=0, column=0, sticky="ew")

        # ── Left: Movement ────────────────────────────────────────────
        move_frame = ttk.Frame(inner, style="Joy.TFrame")
        move_frame.grid(row=0, column=0, sticky="nw", padx=(0, 24))

        ttk.Label(move_frame, text="Movement", style="JoySec.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 4))

        move_rows = [
            ("W / ↑",    "forward"),
            ("S / ↓",    "back"),
            ("A / ←",    "strafe left"),
            ("D / →",    "strafe right"),
            ("Q / Home", "yaw left"),
            ("E / End",  "yaw right"),
            ("R / PgUp", "up"),
            ("F / PgDn", "down"),
        ]
        for i, (key, val) in enumerate(move_rows):
            ttk.Label(move_frame, text=f"  {key}", style="JoyKey.TLabel").grid(
                row=1 + i, column=0, sticky="w")
            ttk.Label(move_frame, text=val, style="JoyVal.TLabel").grid(
                row=1 + i, column=1, sticky="w", padx=(6, 0))

        # ── Right: Commands ───────────────────────────────────────────
        cmd_frame = ttk.Frame(inner, style="Joy.TFrame")
        cmd_frame.grid(row=0, column=1, sticky="nw")

        ttk.Label(cmd_frame, text="Commands", style="JoySec.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 4))

        cmd_rows = [
            ("Space", "hover / stop"),
            ("T",     "takeoff"),
            ("L",     "land"),
            ("M",     "arm toggle"),
        ]
        for i, (key, val) in enumerate(cmd_rows):
            ttk.Label(cmd_frame, text=f"  {key}", style="JoyKey.TLabel").grid(
                row=1 + i, column=0, sticky="w")
            ttk.Label(cmd_frame, text=val, style="JoyVal.TLabel").grid(
                row=1 + i, column=1, sticky="w", padx=(6, 0))

    def _update_joy_legend(self) -> None:
        connected = self.joy.available
        if connected == self._joy_legend_shown:
            return
        self._joy_legend_shown = connected
        if connected:
            self._joy_name_var.set(self.joy.name or "Controller")
            self._kb_legend.grid_remove()
            self._joy_legend.grid(row=0, column=0, sticky="ew")
        else:
            self._joy_legend.grid_remove()
            self._kb_legend.grid(row=0, column=0, sticky="ew")

    # ------------------------------------------------------------------
    # Runtime
    # ------------------------------------------------------------------

    def open_missing(self) -> None:
        pm = self.pymembus
        if pm is None:
            return
        now = time.monotonic()
        if now - self._last_open_attempt < 0.5:
            return
        self._last_open_attempt = now
        if self.video is None:
            vid = pm.memvid()
            if vid.open_existing(self.names["video"]):
                self.video = vid
                self.log("video connected")
        if self.command is None:
            cmd = pm.memcmd()
            if cmd.open(self.names["command"], self.args.cmd_size):
                self.command = cmd
                self.log("command connected")
        if self.status is None:
            kv = pm.memkv()
            if kv.open(self.names["status"]):
                self.status = kv
                self.status_epoch = kv.getEpoch()
                self.last_status_update = time.monotonic()
                self.log("status connected")

    def run(self) -> None:
        self.root.after(50, self.tick)
        self.root.mainloop()

    def tick(self) -> None:
        if not self.running:
            return
        self.open_missing()
        self._maintain_control()
        self.update_connection_text()
        self.update_video()
        self.update_status()
        if self._ui_ready:
            self.joy.poll()
            self._handle_joy_buttons()
            self._update_joy_legend()
            self.send_held_velocity()
        self.root.after(max(10, int(1000 / max(1, self.args.fps))), self.tick)

    def _maintain_control(self) -> None:
        if self.command is None or self.status is None:
            return
        now = time.monotonic()
        owner = self.status.getAll().get("control.owner", "")
        if owner == self.control_source and now - self._last_heartbeat >= 1.0:
            self.send_command("heartbeat", quiet=True)
            self._last_heartbeat = now

    def take_control(self) -> None:
        """Acquire an unowned vehicle without contending with another client."""
        owner = "" if self.status is None else \
            self.status.getAll().get("control.owner", "")
        if owner and owner != self.control_source:
            self.log(f"control held by {owner}")
            return
        self.send_command("acquire_control")
        self._last_heartbeat = time.monotonic()

    def release_control(self) -> None:
        owner = "" if self.status is None else \
            self.status.getAll().get("control.owner", "")
        if owner != self.control_source:
            self.log("this dctl does not own control")
            return
        self.send_command("release_control")

    def _handle_joy_buttons(self) -> None:
        joy = self.joy
        if joy.button_pressed(0):   # A – hover
            self.send_command("zero")
        if joy.button_pressed(1):   # B – land
            self.send_command("land")
        if joy.button_pressed(3):   # Y – takeoff
            self.takeoff()
        if joy.button_pressed(2):   # X – arm toggle
            self.toggle_arm()
        if joy.button_pressed(7):   # Start – arm
            self.send_command("arm", armed=True)
        if joy.button_pressed(6):   # Back – disarm
            self.send_command("arm", armed=False)

    def update_connection_text(self) -> None:
        if self._interfaces_initializing:
            self.conn_var.set(
                "window ready; initializing interfaces, waiting for dsim")
            return
        parts = [
            f"video={'ok' if self.video else 'wait'}",
            f"command={'ok' if self.command else 'wait'}",
            f"status={'ok' if self.status else 'wait'}",
            f"joy={self.joy.status_str()}",
        ]
        stale = self.status is not None and time.monotonic() - self.last_status_update > 2.0
        if stale:
            parts.append("status=stale")
        self.conn_var.set("  ".join(parts))

    def update_video(self) -> None:
        if self.video is None:
            return
        seq = self.video.getSeq()
        if seq == self.last_video_seq or seq <= 0:
            return
        self.last_video_seq = seq
        slot  = self.video.getPtr(-1)
        frame = _client_rgb_frame(np.array(self.video[slot], copy=False))
        image = Image.fromarray(frame, "RGB")
        max_w = self.args.width or self.video.getWidth()
        max_h = self.args.height or self.video.getHeight()
        image.thumbnail((max_w, max_h))
        self.photo = ImageTk.PhotoImage(image)
        self.video_label.configure(image=self.photo, text="")
        if not self._ui_ready:
            # Preserve the first frame long enough to be visibly presented;
            # only then realize the larger controls/telemetry dashboard.
            self._schedule_dashboard(500)

    def update_status(self) -> None:
        if self.status is None:
            return
        changed, epoch = self.status.getChanged(self.status_epoch)
        if changed:
            self.status_epoch = epoch
            self.last_status_update = time.monotonic()
        values = self.status.getAll()
        for key, var in self.status_vars.items():
            value = values.get(key, "-")
            if key == "drone.armed":
                value = "armed" if value == "1" else "disarmed"
            var.set(format_status_value(key, value))

    def key_down(self, event) -> None:
        self.held.add(_control_key(event.keysym))

    def key_up(self, event) -> None:
        self.held.discard(_control_key(event.keysym))
        self.send_held_velocity(force=True)

    def send_held_velocity(self, force: bool = False) -> None:
        kb_forward, kb_right, kb_up, kb_yaw = _held_axes(self.held)

        joy = self.joy
        forward  = _clamp(kb_forward + joy.forward)
        right    = _clamp(kb_right   + joy.right)
        up       = _clamp(kb_up      + joy.up)
        yaw_norm = _clamp(kb_yaw     + joy.yaw)

        active = any((forward, right, up, yaw_norm))
        if active or force or self._held_velocity_active:
            self.send_command(
                "velocity",
                forward_mps=forward  * self.args.speed,
                right_mps=right      * self.args.speed,
                up_mps=up            * self.args.vertical_speed,
                yaw_rate_dps=_manual_yaw_rate(yaw_norm),
                quiet=True,
            )
            self._held_velocity_active = active

    def send_velocity(self, forward_mps: float = 0.0, right_mps: float = 0.0,
                      up_mps: float = 0.0, yaw_rate_dps: float = 0.0) -> None:
        self.send_command("velocity",
                          forward_mps=forward_mps, right_mps=right_mps,
                          up_mps=up_mps, yaw_rate_dps=yaw_rate_dps)

    def takeoff(self) -> None:
        self.send_command("takeoff", alt_m=float(self.takeoff_alt.get()))

    def toggle_arm(self) -> None:
        value = self.status_vars.get("drone.armed", tk.StringVar(value="-")).get()
        self.send_command("arm", armed=value != "armed")

    def send_command(self, typ: str, quiet: bool = False, **fields) -> None:
        if self.command is None:
            if not quiet:
                self.log("command buffer not connected")
            return
        payload = controlled_command(
            typ, self.control_source, self.control_lease, **fields)
        if self.command.write(payload):
            if typ in ("zero", "land", "takeoff"):
                self._held_velocity_active = False
            if not quiet:
                self.log(f"sent {typ}")
        elif not quiet:
            self.log("failed to send command")

    def log(self, message: str) -> None:
        self.log_var.set(message)
        if self.args.verbose:
            print(message)

    def _trace(self, message: str) -> None:
        if self.args.verbose:
            print(f"dctl startup: {message}", file=sys.stderr, flush=True)

    def close(self) -> None:
        self.running = False
        try:
            owner = "" if self.status is None else \
                self.status.getAll().get("control.owner", "")
            if owner == self.control_source:
                self.send_command("zero", quiet=True)
                self.send_command("release_control", quiet=True)
        finally:
            self.joy.close()
            for handle in (self.status, self.command, self.video):
                if handle is not None:
                    handle.close()
            save_window_pos(self.root, f"dctl.{self.args.id}")
            self.root.destroy()


# ---------------------------------------------------------------------------
# Disabled joystick stub
# ---------------------------------------------------------------------------

class _DisabledJoystick:
    """Drop-in replacement used when --no-joystick is passed."""

    available = False
    name = ""
    forward = right = up = yaw = 0.0

    def poll(self) -> None:
        pass

    def button_pressed(self, _n: int) -> bool:
        return False

    def status_str(self) -> str:
        return "disabled"

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_KEY_ALIASES = {
    "up": "w",
    "down": "s",
    "left": "a",
    "right": "d",
    "prior": "r",      # PageUp
    "page_up": "r",
    "next": "f",       # PageDown
    "page_down": "f",
    "home": "q",
    "end": "e",
}


def _control_key(keysym: str) -> str:
    key = keysym.lower()
    return _KEY_ALIASES.get(key, key)


def _held_axes(held: set[str]) -> tuple[float, float, float, float]:
    """Normalized (forward, right, up, yaw-right) from the held control keys.

    Kept out of the widget so the documented key-to-action mapping can be
    tested without a display server.
    """
    def axis(positive: str, negative: str) -> float:
        return (1.0 if positive in held else 0.0) + (-1.0 if negative in held else 0.0)

    return axis("w", "s"), axis("d", "a"), axis("r", "f"), axis("e", "q")


def _client_rgb_frame(frame: np.ndarray) -> np.ndarray:
    """Normalize a shared RGB frame without changing pixel orientation."""
    return np.ascontiguousarray(frame)


def _manual_yaw_rate(yaw_right_norm: float) -> float:
    """Convert human-facing yaw-right input to the simulator command sign."""
    return _clamp(yaw_right_norm) * _MANUAL_YAW_RATE_DPS


def format_status_value(key: str, value: str) -> str:
    try:
        if key in ("drone.heading_deg", "drone.roll_deg", "drone.pitch_deg"):
            return f"{float(value):.1f} deg"
        if key in ("drone.lat_deg", "drone.lon_deg"):
            return f"{float(value):.6f}"
        if key.endswith("_m") or key.endswith("_mps"):
            return f"{float(value):.2f}"
        if key == "drone.battery_pct":
            return f"{float(value):.0f}%"
    except ValueError:
        pass
    return value if value != "" else "-"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="dvision2 drone controller")
    parser.add_argument("--id",           required=True)
    parser.add_argument("--width",        type=int,   default=960)
    parser.add_argument("--height",       type=int,   default=720)
    parser.add_argument("--fps",          type=int,   default=30)
    parser.add_argument("--cmd-size",     type=int,   default=65536)
    parser.add_argument("--speed",        type=float, default=1.5)
    parser.add_argument("--vertical-speed", type=float, default=1.0)
    parser.add_argument("--no-joystick",  action="store_true",
                        help="disable joystick/gamepad support")
    parser.add_argument("--verbose",      action="store_true")
    args = parser.parse_args(argv)
    validate_id(args.id)
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.verbose:
        print(f"dctl startup: imports completed in "
              f"{time.perf_counter() - _MODULE_STARTED:.3f}s",
              file=sys.stderr, flush=True)
    controller = DroneController(args)

    def stop(_signum, _frame):
        controller.close()

    signal.signal(signal.SIGINT,  stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        controller.run()
    except Exception as exc:
        print(f"dctl: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
