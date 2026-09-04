#!/usr/bin/env python3
"""dfgb — dvision2 FlightGear bridge.

Drop-in replacement for dsim that uses FlightGear for physics and rendering
while exposing the identical shared-memory interface, so dctl requires no
changes.

  /dvision2.<id>.video    — RGB24 frames captured from FG via ffmpeg
  /dvision2.<id>.control  — JSON commands from dctl (unchanged protocol)
  /dvision2.<id>.status   — key-value telemetry sourced from FG property tree

FlightGear is launched under a dedicated Xvfb virtual display.  ffmpeg
captures that display and pipes raw RGB24 frames to this process.  A UDP
generic protocol carries control inputs to FG and state datagrams back.

Usage:
  python3 dfgb/dfgb.py --id area1
  python3 dfgb/dfgb.py --install          # install dependencies first
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import tkinter as tk
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dvision2_common import (
    STATUS_KEYS,
    clamp,
    decode_command,
    load_pymembus,
    shared_names,
    validate_id,
)
from dcmn.module_bus import PymembusModuleBus, requests_shutdown
from dcmn.window import (disable_input_method, restore_window_pos,
                          save_window_pos)

PROTOCOLS_DIR = Path(__file__).parent / "protocols"
SHARED_MODELS_URL = "https://us1mirror.flightgear.org/terrasync/SharedModels.txz"

# UDP ports — FG listens on CTRL, bridge listens on STATE
FG_CTRL_PORT  = 5500
FG_STATE_PORT = 5501

# dvision2 commands are drone-style body-frame setpoints.  FlightGear's UFO
# flight controls are useful for manual aircraft-like flying, but they pitch
# and roll the vehicle instead of translating cleanly like a multirotor.  dfgb
# therefore drives the UFO kinematically by publishing position/heading every
# frame while keeping flight surfaces centered.
_UFO_MAX_SPEED_MPS = 20.0
_HORIZONTAL_ACCEL_TAU = 0.30
_HORIZONTAL_DECEL_TAU = 0.85
_VISUAL_TILT_GAIN = 1.2   # degrees of visual lean per m/s command
_MAX_VISUAL_TILT = 18.0
_VISUAL_TILT_TAU = 0.12

FPS_TO_MPS = 0.3048   # feet-per-second → metres-per-second
METERS_PER_DEG_LAT = 111_320.0


# ---------------------------------------------------------------------------
# FG state (populated from UDP datagrams)
# ---------------------------------------------------------------------------

@dataclass
class FGState:
    lat_deg:         float = 0.0
    lon_deg:         float = 0.0
    alt_ft:          float = 0.0
    roll_deg:        float = 0.0
    pitch_deg:       float = 0.0
    heading_deg:     float = 0.0
    speed_north_fps: float = 0.0   # world-frame north velocity
    speed_east_fps:  float = 0.0   # world-frame east velocity
    speed_down_fps:  float = 0.0   # world-frame down velocity (positive = descending)
    groundspeed_kt:  float = 0.0
    u_fps:           float = 0.0   # body-frame forward velocity
    v_fps:           float = 0.0   # body-frame right velocity
    updated:         float = field(default_factory=lambda: 0.0)

    @property
    def alt_m(self) -> float:
        return self.alt_ft * FPS_TO_MPS

    @property
    def vx_mps(self) -> float:
        return self.speed_east_fps * FPS_TO_MPS

    @property
    def vy_mps(self) -> float:
        return self.speed_north_fps * FPS_TO_MPS

    @property
    def vz_mps(self) -> float:
        return -self.speed_down_fps * FPS_TO_MPS

    @property
    def speed_mps(self) -> float:
        return self.groundspeed_kt * 0.5144


@dataclass(frozen=True)
class FGControl:
    throttle: float
    aileron: float
    elevator: float
    rudder: float
    lat_deg: float | None = None
    lon_deg: float | None = None
    altitude_ft: float | None = None
    heading_deg: float | None = None
    roll_deg: float | None = None
    pitch_deg: float | None = None

    def csv_line(self) -> str:
        line = (
            f"{self.throttle:.4f},{self.aileron:.4f},"
            f"{self.elevator:.4f},{self.rudder:.4f}"
        )
        if (
            self.lat_deg is not None
            and self.lon_deg is not None
            and self.altitude_ft is not None
            and self.heading_deg is not None
            and self.roll_deg is not None
            and self.pitch_deg is not None
        ):
            line += (
                f",{self.lat_deg:.9f},{self.lon_deg:.9f},"
                f"{self.altitude_ft:.4f},{self.heading_deg:.4f},"
                f"{self.roll_deg:.4f},{self.pitch_deg:.4f}"
            )
        return line + "\n"


def integrate_drone_pose(
    *,
    cmd_forward: float,
    cmd_right: float,
    cmd_up: float,
    cmd_yaw_rate: float,
    lat_deg: float,
    lon_deg: float,
    alt_m: float,
    heading_deg: float,
    dt: float,
) -> tuple[float, float, float, float]:
    """Integrate drone body-frame commands into global FG pose."""
    heading_rad = math.radians(heading_deg)
    north_mps = (
        cmd_forward * math.cos(heading_rad)
        - cmd_right * math.sin(heading_rad)
    )
    east_mps = (
        cmd_forward * math.sin(heading_rad)
        + cmd_right * math.cos(heading_rad)
    )

    lat = lat_deg + north_mps * dt / METERS_PER_DEG_LAT
    lon_scale = METERS_PER_DEG_LAT * max(0.01, math.cos(math.radians(lat_deg)))
    lon = lon_deg + east_mps * dt / lon_scale
    alt = max(0.0, alt_m + cmd_up * dt)
    heading = (heading_deg + cmd_yaw_rate * dt) % 360.0
    return lat, lon, alt, heading


def integrate_axis_velocity(
    *,
    current_mps: float,
    target_mps: float,
    dt: float,
) -> float:
    """First-order velocity response with slower braking than acceleration."""
    current_abs = abs(current_mps)
    target_abs = abs(target_mps)
    braking = target_abs < current_abs or current_mps * target_mps < 0.0
    tau = _HORIZONTAL_DECEL_TAU if braking else _HORIZONTAL_ACCEL_TAU
    alpha = 1.0 if dt <= 0.0 else 1.0 - math.exp(-dt / tau)
    value = current_mps + (target_mps - current_mps) * alpha
    if abs(value) < 0.001 and abs(target_mps) < 0.001:
        return 0.0
    return value


def integrate_visual_attitude(
    *,
    cmd_forward: float,
    cmd_right: float,
    roll_deg: float,
    pitch_deg: float,
    dt: float,
) -> tuple[float, float]:
    """Low-pass a drone-style visual lean from body-frame velocity commands."""
    target_roll = clamp(cmd_right * _VISUAL_TILT_GAIN,
                        -_MAX_VISUAL_TILT, _MAX_VISUAL_TILT)
    target_pitch = clamp(-cmd_forward * _VISUAL_TILT_GAIN,
                         -_MAX_VISUAL_TILT, _MAX_VISUAL_TILT)
    alpha = 1.0 if dt <= 0.0 else 1.0 - math.exp(-dt / _VISUAL_TILT_TAU)
    roll = roll_deg + (target_roll - roll_deg) * alpha
    pitch = pitch_deg + (target_pitch - pitch_deg) * alpha
    return roll, pitch


def compute_drone_control(
    *,
    lat_deg: float | None = None,
    lon_deg: float | None = None,
    altitude_ft: float | None = None,
    heading_deg: float | None = None,
    roll_deg: float | None = None,
    pitch_deg: float | None = None,
) -> FGControl:
    """Build a FlightGear control packet for kinematic UFO control."""
    return FGControl(
        throttle=0.0,
        aileron=0.0,
        elevator=0.0,
        rudder=0.0,
        lat_deg=lat_deg,
        lon_deg=lon_deg,
        altitude_ft=altitude_ft,
        heading_deg=heading_deg,
        roll_deg=roll_deg,
        pitch_deg=pitch_deg,
    )


# ---------------------------------------------------------------------------
# Status UI
# ---------------------------------------------------------------------------

class StatusUi:
    """Tkinter window showing live bridge and FlightGear status."""

    _BG      = "#1a1e24"
    _CARD    = "#21262d"
    _FG      = "#c9d1d9"
    _DIM     = "#8b949e"
    _GREEN   = "#3fb950"
    _RED     = "#f85149"
    _YELLOW  = "#d29922"
    _FONT    = ("Monospace", 10)
    _BOLD    = ("Monospace", 9, "bold")

    def __init__(self, bridge: FGBridge) -> None:
        self.bridge = bridge
        self.closed = False

        self.root = tk.Tk()
        self.root.title(f"dfgb  {bridge.args.id}")
        self.root.configure(bg=self._BG)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.resizable(False, False)

        self._build()
        restore_window_pos(self.root, f"dfgb.{bridge.args.id}")

    # ------------------------------------------------------------------ build

    def _card(self, title: str, row: int) -> tk.Frame:
        outer = tk.Frame(self.root, bg=self._BG)
        outer.grid(row=row, column=0, sticky="ew", padx=10, pady=(8, 0))
        tk.Label(outer, text=title, bg=self._BG, fg=self._DIM,
                 font=self._BOLD).pack(anchor="w")
        inner = tk.Frame(outer, bg=self._CARD, padx=10, pady=6)
        inner.pack(fill="x")
        return inner

    def _row(self, parent: tk.Frame, label: str,
             col: str | None = None) -> tk.Label:
        """Add a label + value row; return the value Label widget."""
        f = tk.Frame(parent, bg=self._CARD)
        f.pack(fill="x", pady=1)
        tk.Label(f, text=f"{label:<15}", bg=self._CARD, fg=self._DIM,
                 font=self._FONT, anchor="w").pack(side="left")
        val = tk.Label(f, text="—", bg=self._CARD,
                       fg=col or self._FG, font=self._FONT, anchor="w")
        val.pack(side="left")
        return val

    def _build(self) -> None:
        self.root.columnconfigure(0, weight=1)

        # Processes
        pc = self._card("PROCESSES", 0)
        self._lbl_xvfb   = self._row(pc, "Xvfb")
        self._lbl_fg     = self._row(pc, "FlightGear")
        self._lbl_ffmpeg = self._row(pc, "ffmpeg")

        # Connections
        cc = self._card("CONNECTIONS", 1)
        self._lbl_state_link = self._row(cc, "State UDP")
        self._lbl_video_link = self._row(cc, "Video pipe")

        # Drone
        dc = self._card("DRONE", 2)
        self._lbl_armed = self._row(dc, "Armed / mode")
        self._lbl_alt   = self._row(dc, "Altitude")
        self._lbl_hdg   = self._row(dc, "Heading")
        self._lbl_att   = self._row(dc, "Roll / pitch")
        self._lbl_speed = self._row(dc, "Speed")
        self._lbl_pos   = self._row(dc, "Lat / lon")
        self._lbl_vel   = self._row(dc, "Vx / Vy / Vz")

        # Commands
        qc = self._card("COMMANDS", 3)
        self._lbl_cmd_count  = self._row(qc, "Count")
        self._lbl_cmd_last   = self._row(qc, "Last")
        self._lbl_cmd_status = self._row(qc, "Status")

        tk.Frame(self.root, bg=self._BG, height=8).grid(row=4, column=0)

    # ----------------------------------------------------------------- update

    def _proc_fmt(self, proc: subprocess.Popen | None) -> tuple[str, str]:
        if proc is None:
            return "not started", self._DIM
        rc = proc.poll()
        if rc is None:
            return f"● running   pid {proc.pid}", self._GREEN
        return f"✗ exited ({rc})", self._RED

    def _age(self, t: float) -> tuple[str, str]:
        """(text, colour) for time since a monotonic timestamp."""
        if t == 0.0:
            return "no data", self._RED
        age = time.monotonic() - t
        if age < 2.0:
            text  = f"● {age * 1000:.0f} ms ago"
            color = self._GREEN
        else:
            text  = f"✗ {age:.1f} s ago"
            color = self._RED
        return text, color

    def update(self) -> None:
        if self.closed:
            return
        b = self.bridge

        # Processes
        for lbl, proc in [
            (self._lbl_xvfb,   b._proc_xvfb),
            (self._lbl_fg,     b._proc_fg),
            (self._lbl_ffmpeg, b._proc_ffmpeg),
        ]:
            txt, col = self._proc_fmt(proc)
            lbl.config(text=txt, fg=col)

        # Connections
        with b._state_lock:
            state_updated = b.fg_state.updated
        txt, col = self._age(state_updated)
        self._lbl_state_link.config(text=txt, fg=col)

        txt, col = self._age(b._last_frame_time)
        self._lbl_video_link.config(text=txt, fg=col)

        # Drone state
        with b._state_lock:
            st = b.fg_state
            alt, hdg        = st.alt_m, st.heading_deg
            roll, pitch     = st.roll_deg, st.pitch_deg
            spd             = st.speed_mps
            lat, lon        = st.lat_deg, st.lon_deg
            vx, vy, vz      = st.vx_mps, st.vy_mps, st.vz_mps

        self._lbl_armed.config(
            text=f"{'YES' if b.armed else 'no'}   {b.mode}",
            fg=self._GREEN if b.armed else self._DIM,
        )
        self._lbl_alt.config(  text=f"{alt:.1f} m")
        self._lbl_hdg.config(  text=f"{hdg:.1f}°")
        self._lbl_att.config(  text=f"{roll:+.1f}°  /  {pitch:+.1f}°")
        self._lbl_speed.config(text=f"{spd:.2f} m/s")
        self._lbl_pos.config(  text=f"{lat:.7f}   {lon:.7f}")
        self._lbl_vel.config(  text=f"{vx:+.2f}  {vy:+.2f}  {vz:+.2f} m/s")

        # Commands
        self._lbl_cmd_count.config(text=str(b.command_count))
        if b.last_cmd_monotonic:
            age = time.monotonic() - b.last_cmd_monotonic
            self._lbl_cmd_last.config(
                text=f"{b.last_cmd_type}  ({age:.1f}s ago)")
        else:
            self._lbl_cmd_last.config(text="none")
        ok = b.status_message in ("ok", "ready")
        self._lbl_cmd_status.config(
            text=b.status_message,
            fg=self._FG if ok else self._YELLOW,
        )

        self.root.update_idletasks()
        self.root.update()

    # ------------------------------------------------------------------ close

    def _on_close(self) -> None:
        self.closed = True
        save_window_pos(self.root, f"dfgb.{self.bridge.args.id}")
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    def close(self) -> None:
        if not self.closed:
            self._on_close()


# ---------------------------------------------------------------------------
# Bridge
# ---------------------------------------------------------------------------

class FGBridge:

    def __init__(self, args: argparse.Namespace) -> None:
        self.args      = args
        self.pymembus  = load_pymembus()
        self.names     = shared_names(args.id)
        self.running   = True
        self.started   = time.monotonic()

        # FG state (written by _recv_state on main thread, protected by lock
        # because the UI also reads it)
        self.fg_state    = FGState()
        self._state_lock = threading.Lock()

        # Command state (mirrors DroneSimulator fields; main-thread only)
        self.armed                   = False
        self.mode                    = "DISARMED"
        self.cmd_forward             = 0.0
        self.cmd_right               = 0.0
        self.cmd_up                  = 0.0
        self.cmd_yaw_rate            = 0.0
        self.target_alt: float|None  = None
        self.origin_alt_m: float|None = None
        self._desired_lat_deg: float|None = None
        self._desired_lon_deg: float|None = None
        self._desired_alt_m: float|None = None
        self._desired_heading_deg: float|None = None
        self._body_forward_mps = 0.0
        self._body_right_mps = 0.0
        self._visual_roll_deg = 0.0
        self._visual_pitch_deg = 0.0
        self._last_control_time       = time.monotonic()
        self.battery_pct             = 100.0
        self.last_cmd_monotonic: float|None = None
        self.last_cmd_type           = ""
        self.command_count           = 0
        self.status_message          = "ready"

        # Child processes
        self._proc_xvfb:   subprocess.Popen|None = None
        self._proc_fg:     subprocess.Popen|None = None
        self._proc_ffmpeg: subprocess.Popen|None = None
        self._fg_log                              = None

        # Sockets
        self._ctrl_sock:  socket.socket|None = None
        self._state_sock: socket.socket|None = None

        # pymembus handles
        self.video   = None
        self.command = None
        self.status  = None
        self.module_bus = None

        # Latest raw RGB24 frame from the video reader thread
        self._frame_bytes: bytes|None = None
        self._frame_lock  = threading.Lock()
        self._last_frame_time: float = 0.0   # monotonic; written on main thread

        # Resolved virtual display string (e.g. ":99"), set by _start_xvfb
        self._display: str = f":{args.display}"

        # Status UI (created in run(), before FG wait)
        self.ui: StatusUi|None = None

    # ------------------------------------------------------------------
    # Process management
    # ------------------------------------------------------------------

    def _start_xvfb(self) -> None:
        n = self._find_free_display(self.args.display)
        self._display = f":{n}"
        cmd = [
            "Xvfb", self._display,
            "-screen", "0", f"{self.args.width}x{self.args.height}x24",
        ]
        self._proc_xvfb = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(1.0)
        if self._proc_xvfb.poll() is not None:
            raise RuntimeError(
                f"Xvfb failed to start on display {self._display}"
            )
        if self.args.verbose:
            print(f"dfgb: Xvfb started on display {self._display} "
                  f"(pid {self._proc_xvfb.pid})")

    @staticmethod
    def _find_free_display(start: int) -> int:
        """Return the first free X display number at or after *start*.

        A display is considered free when it has no lock file, or when its
        lock file refers to a dead process (stale lock cleaned up on the spot).
        """
        for n in range(start, start + 50):
            lock = Path(f"/tmp/.X{n}-lock")
            if not lock.exists():
                return n
            try:
                pid = int(lock.read_text().strip())
                os.kill(pid, 0)   # raises ProcessLookupError if dead
                # Process is alive — display is in use, try the next one.
            except ProcessLookupError:
                # Stale lock: owner is gone, clean up and claim this slot.
                lock.unlink(missing_ok=True)
                Path(f"/tmp/.X11-unix/X{n}").unlink(missing_ok=True)
                return n
            except (ValueError, PermissionError):
                pass   # unreadable lock — skip and try next
        raise RuntimeError(
            f"Could not find a free X display in :{start}–:{start + 49}"
        )

    def _fg_root(self) -> str|None:
        if self.args.fg_root:
            return self.args.fg_root
        for p in [
            "/usr/share/games/flightgear",
            "/usr/share/flightgear",
            "/usr/local/share/flightgear",
            "/snap/flightgear/current/usr/share/flightgear",
        ]:
            if Path(p).is_dir():
                return p
        return None

    def _shared_models_dir(self) -> Path:
        raw = getattr(self.args, "shared_models_dir", None)
        if raw:
            return Path(raw).expanduser()
        return Path.home() / ".fgfs" / "TerraSync"

    _EMPTY_XML_MODEL = '<?xml version="1.0"?><PropertyList/>\n'
    _EMPTY_AC3D_MODEL = (
        "AC3Db\n"
        'MATERIAL "stub" rgb 0 0 0 amb 0 0 0 emis 0 0 0 '
        "spec 0 0 0 shi 0 trans 0\n"
        "OBJECT world\n"
        "kids 0\n"
    )

    # Paths that FG checks to decide whether to show the
    # "shared scenery models not installed" dialog.  FG 2024.1 checks the
    # first four; the remaining paths are checked by older startup code.
    _SHARED_MODEL_SENTINELS = {
        "Models/Airport/marker.ac": _EMPTY_AC3D_MODEL,
        "Models/Airport/beacon.xml": _EMPTY_XML_MODEL,
        "Models/Airport/localizer.xml": _EMPTY_XML_MODEL,
        "Models/Misc/trigpoint.ac": _EMPTY_AC3D_MODEL,
        "Models/Airport/windsock_lit.xml": _EMPTY_XML_MODEL,
        "Models/Industrial/generic_chimney_01.xml": _EMPTY_XML_MODEL,
        "Models/Airport/Vehicle/Cobus_3000.xml": _EMPTY_XML_MODEL,
        "Models/Industrial/GenericStorageTank40m.ac": _EMPTY_AC3D_MODEL,
        "Models/Boundaries/Fence_50m.ac": _EMPTY_AC3D_MODEL,
        "Models/Residential/french_house_s.xml": _EMPTY_XML_MODEL,
        "Models/Power/generic_pylon_50m.ac": _EMPTY_AC3D_MODEL,
    }

    def _ensure_shared_models_sentinel(self) -> None:
        """Stub out every sentinel file FG checks before showing the
        'shared scenery models not installed' startup dialog.

        Writes to ~/.fgfs/TerraSync, which is user-owned and is part of FG's
        scenery-search path even when automatic TerraSync downloads are off.
        """
        base = Path.home() / ".fgfs" / "TerraSync"
        for rel, stub in self._SHARED_MODEL_SENTINELS.items():
            path = base / rel
            if path.exists():
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(stub)
            if self.args.verbose:
                print(f"dfgb: created shared-models stub at {path}")

    def _fg_scenery(self) -> str | None:
        """Return an explicit scenery path when shared models live outside
        the default TerraSync path."""
        shared = self._shared_models_dir()
        terrasync = Path.home() / ".fgfs" / "TerraSync"
        if shared.resolve() == terrasync.resolve():
            return None
        paths = [shared, terrasync]
        return ":".join(str(p) for p in paths if p.is_dir())

    def _start_flightgear(self) -> None:
        self._ensure_shared_models_sentinel()
        self._start_dialog_dismisser()
        fg_root = self._fg_root()

        cmd = [
            "fgfs",
            f"--aircraft={self.args.aircraft}",
        ]
        if self.args.airport:
            cmd.append(f"--airport={self.args.airport}")
            if self.args.runway:
                cmd.append(f"--runway={self.args.runway}")
        else:
            cmd.extend([
                f"--lat={self.args.lat}",
                f"--lon={self.args.lon}",
                f"--altitude={self.args.alt}",
                "--heading=0",
            ])

        cmd.extend([
            "--vc=0",
            "--timeofday=noon",
            "--disable-sound",
            "--disable-ai-traffic",
            "--disable-real-weather-fetch",
            "--disable-fullscreen",
            f"--prop:/engines/engine/speed-max-mps={_UFO_MAX_SPEED_MPS}",
            # dfgb does not use procedural scenery objects for its control
            # loop, so keep them disabled even when shared models are present.
            "--prop:/sim/rendering/random-objects=false",
            "--prop:/sim/rendering/random-vegetation=false",
            "--prop:/sim/rendering/random-buildings=false",
            f"--geometry={self.args.width}x{self.args.height}",
            f"--generic=socket,in,{self.args.fps},,{FG_CTRL_PORT},udp,dvision2-ctrl",
            f"--generic=socket,out,{self.args.fps},localhost,{FG_STATE_PORT},udp,dvision2-state",
        ])
        if self.args.disable_terrasync:
            cmd.extend([
                "--disable-terrasync",
                "--prop:/sim/terrasync/enabled=false",
            ])
        else:
            cmd.append("--enable-terrasync")
        if fg_root:
            cmd.append(f"--fg-root={fg_root}")
        fg_scenery = self._fg_scenery()
        if fg_scenery:
            cmd.append(f"--fg-scenery={fg_scenery}")
        if self.args.fg_aircraft:
            cmd.append(f"--fg-aircraft={self.args.fg_aircraft}")

        env = os.environ.copy()
        env["DISPLAY"] = self._display

        log_path     = Path(f"dfgb-{self.args.id}.log")
        self._fg_log = open(log_path, "w")

        self._proc_fg = subprocess.Popen(
            cmd, env=env,
            stdin=subprocess.DEVNULL,
            stdout=self._fg_log,
            stderr=self._fg_log,
        )
        if self.args.verbose:
            print(f"dfgb: FlightGear started (pid {self._proc_fg.pid}), "
                  f"log → {log_path}")

    def _wait_for_fg(self, timeout: float = 90.0) -> None:
        """Block until FG sends its first state datagram, pumping the UI meanwhile."""
        deadline = time.monotonic() + timeout
        if self.args.verbose:
            print(f"dfgb: waiting for FlightGear (up to {int(timeout)}s)…")
        while time.monotonic() < deadline:
            if self._proc_fg and self._proc_fg.poll() is not None:
                raise RuntimeError(
                    "FlightGear exited during startup — check "
                    f"dfgb-{self.args.id}.log for details"
                )
            try:
                data, _ = self._state_sock.recvfrom(4096)
                if data:
                    if self.args.verbose:
                        print("dfgb: FlightGear ready")
                    return
            except BlockingIOError:
                pass
            self._publish_placeholder_frame()
            self._publish_status()
            if self.ui is not None:
                self.ui.update()
                if self.ui.closed:
                    self.running = False
                    raise RuntimeError("UI closed during FlightGear startup")
            time.sleep(0.1)
        raise RuntimeError(
            f"FlightGear did not send state within {int(timeout)}s"
        )

    def _start_dialog_dismisser(self) -> None:
        """Spawn a persistent daemon thread that dismisses FG popups for the
        entire session.  The dialog can appear during scenery loading which
        is after the first UDP state packet, so dismissal must continue past
        the FG-ready point and into the main loop."""
        threading.Thread(
            target=self._dialog_dismisser_loop,
            daemon=True,
            name="dfgb-dismiss",
        ).start()

    def _dialog_dismisser_loop(self) -> None:
        while self.running:
            self._dismiss_fg_dialogs()
            time.sleep(3.0)

    def _dismiss_fg_dialogs(self) -> None:
        """Click the OK button in any blocking FG dialog."""
        if not shutil.which("xdotool"):
            return
        env = os.environ.copy()
        env["DISPLAY"] = self._display
        try:
            # Search by class name (reliable) as well as title substring.
            result = subprocess.run(
                ["xdotool", "search", "--any",
                 "--class", "fgfs",
                 "--name", "FlightGear"],
                env=env, capture_output=True, text=True, timeout=3,
            )
            wids = result.stdout.strip().split()
            if not wids:
                return
            wid = wids[0]

            # Activate so key events land in the window.
            subprocess.run(
                ["xdotool", "windowactivate", "--sync", wid],
                env=env, timeout=3,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            # Return key activates the focused button (OK).
            subprocess.run(
                ["xdotool", "key", "--window", wid, "Return"],
                env=env, timeout=3,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            # Also click at the OK button position.  The shared-models dialog
            # is centred; OK sits at roughly 66 % across, 72 % down.
            ok_x = int(self.args.width  * 0.66)
            ok_y = int(self.args.height * 0.72)
            subprocess.run(
                ["xdotool", "mousemove", "--window", wid,
                 str(ok_x), str(ok_y), "click", "1"],
                env=env, timeout=3,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass

    def _start_ffmpeg(self) -> None:
        cmd = [
            "ffmpeg",
            "-f", "x11grab",
            "-r", str(self.args.fps),
            "-video_size", f"{self.args.width}x{self.args.height}",
            "-i", self._display,
            "-pix_fmt", "rgb24",
            "-f", "rawvideo",
            "pipe:1",
            "-loglevel", "error",
        ]
        self._proc_ffmpeg = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )
        if self.args.verbose:
            print(f"dfgb: ffmpeg capture started (pid {self._proc_ffmpeg.pid})")

        frame_bytes = self.args.width * self.args.height * 3
        threading.Thread(
            target=self._video_reader,
            args=(frame_bytes,),
            daemon=True,
            name="dfgb-video",
        ).start()

    def _video_reader(self, frame_bytes: int) -> None:
        pipe = self._proc_ffmpeg.stdout
        while self.running:
            buf = bytearray()
            while len(buf) < frame_bytes:
                chunk = pipe.read(frame_bytes - len(buf))
                if not chunk:
                    return
                buf += chunk
            with self._frame_lock:
                self._frame_bytes = bytes(buf)

    # ------------------------------------------------------------------
    # IPC
    # ------------------------------------------------------------------

    def _open_ipc(self) -> None:
        pm = self.pymembus
        pm.memvid.remove(self.names["video"])
        pm.memcmd.remove(self.names["command"])
        pm.memkv.remove(self.names["status"])
        pm.memmsg.remove(self.names["events"])

        self.video = pm.memvid()
        fmt = getattr(pm.video_format, "rgb24", 24)
        if not self.video.open(self.names["video"], True, self.args.width,
                               self.args.height, fmt, self.args.fps, self.args.bufs):
            raise RuntimeError(
                f"failed to create video segment {self.names['video']}: "
                f"{pm.last_error_message()}"
            )

        self.command = pm.memcmd()
        if not self.command.open(self.names["command"], self.args.cmd_size,
                                  True, True):
            raise RuntimeError(
                f"failed to create command segment {self.names['command']}: "
                f"{pm.last_error_message()}"
            )

        self.status = pm.memkv()
        max_name = max(len(k) for k in STATUS_KEYS) + 1
        if not self.status.create(self.names["status"], len(STATUS_KEYS),
                                   max_name, 128, True):
            raise RuntimeError(
                f"failed to create status segment {self.names['status']}: "
                f"{pm.last_error_message()}"
            )
        for idx, key in enumerate(STATUS_KEYS):
            if not self.status.setName(idx, key):
                raise RuntimeError(f"failed to set status key {key!r}")
        self.module_bus = PymembusModuleBus(
            self.args.id, "simulator", "dfgb", create=True,
            sim_time=lambda: time.monotonic() - self.started)
        if not self.module_bus.connect():
            raise RuntimeError(
                f"failed to create event buffer {self.names['events']}")
        self.module_bus.publish("module.hello", payload={
            "state": "ready", "capabilities": ["vehicle", "video", "status"]})

    def _open_sockets(self) -> None:
        self._ctrl_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._state_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._state_sock.bind(("127.0.0.1", FG_STATE_PORT))
        self._state_sock.setblocking(False)

    # ------------------------------------------------------------------
    # Per-frame work
    # ------------------------------------------------------------------

    def _drain_commands(self) -> None:
        if self.command is None:
            return
        while self.command.poll():
            raw, overrun = self.command.read_with_overrun(0)
            if overrun:
                self.status_message = "command overrun"
                continue
            payload = decode_command(raw)
            if payload is None:
                self.status_message = "ignored unsupported command payload"
                continue
            self._apply_command(payload)

    def _apply_command(self, payload: dict) -> None:
        typ = payload["type"]
        self.last_cmd_monotonic = time.monotonic()
        self.last_cmd_type      = typ
        self.command_count     += 1
        self.status_message     = "ok"

        if typ == "heartbeat":
            return
        if typ == "arm":
            self.armed = bool(payload.get("armed", True))
            self.mode  = "GUIDED" if self.armed else "DISARMED"
            if not self.armed:
                self._zero_cmds()
                self._reset_inertia()
            return
        if typ == "takeoff":
            if self.armed:
                self.target_alt = max(float(payload.get("alt_m", 3.0)), 0.5)
                self.mode       = "TAKEOFF"
            return
        if typ == "land":
            self.target_alt  = 0.0
            self.mode        = "LAND"
            self.cmd_forward = self.cmd_right = 0.0
            return
        if typ == "zero":
            self._zero_cmds()
            if self.armed:
                self.mode = "HOLD"
            return
        if typ == "velocity":
            if not self.armed:
                self._zero_cmds()
                return
            self.cmd_forward  = float(payload.get("forward_mps",  0.0))
            self.cmd_right    = float(payload.get("right_mps",    0.0))
            self.cmd_up       = float(payload.get("up_mps",       0.0))
            self.cmd_yaw_rate = float(payload.get("yaw_rate_dps", 0.0))
            self.mode         = "GUIDED"
            self.target_alt   = None

    def _zero_cmds(self) -> None:
        self.cmd_forward = self.cmd_right = self.cmd_up = self.cmd_yaw_rate = 0.0

    def _reset_inertia(self) -> None:
        self._body_forward_mps = 0.0
        self._body_right_mps = 0.0

    def _recv_state(self) -> None:
        if self._state_sock is None:
            return
        latest: bytes|None = None
        try:
            while True:
                data, _ = self._state_sock.recvfrom(4096)
                if data:
                    latest = data
        except BlockingIOError:
            pass
        if latest is None:
            return
        try:
            parts = latest.decode().strip().split(",")
            if len(parts) < 12:
                return
            with self._state_lock:
                st                 = self.fg_state
                st.lat_deg         = float(parts[0])
                st.lon_deg         = float(parts[1])
                st.alt_ft          = float(parts[2])
                st.roll_deg        = float(parts[3])
                st.pitch_deg       = float(parts[4])
                st.heading_deg     = float(parts[5])
                st.speed_north_fps = float(parts[6])
                st.speed_east_fps  = float(parts[7])
                st.speed_down_fps  = float(parts[8])
                st.groundspeed_kt  = float(parts[9])
                st.u_fps           = float(parts[10])
                st.v_fps           = float(parts[11])
                st.updated         = time.monotonic()
                if self.origin_alt_m is None:
                    self.origin_alt_m = st.alt_m
                if self._desired_lat_deg is None:
                    self._desired_lat_deg = st.lat_deg
                if self._desired_lon_deg is None:
                    self._desired_lon_deg = st.lon_deg
                if self._desired_alt_m is None:
                    self._desired_alt_m = st.alt_m
                if self._desired_heading_deg is None:
                    self._desired_heading_deg = st.heading_deg
                    self._visual_roll_deg = st.roll_deg
                    self._visual_pitch_deg = st.pitch_deg
        except (ValueError, IndexError):
            pass

    def _send_control(self) -> None:
        if self._ctrl_sock is None:
            return
        now = time.monotonic()
        dt = clamp(now - self._last_control_time, 0.0, 0.25)
        self._last_control_time = now

        if not self.armed:
            self._send_fg_control(FGControl(0.0, 0.0, 0.0, 0.0))
            return

        with self._state_lock:
            vz_mps =  self.fg_state.vz_mps
            lat_deg = self.fg_state.lat_deg
            lon_deg = self.fg_state.lon_deg
            alt_m  =  self.fg_state.alt_m
            heading_deg = self.fg_state.heading_deg
            origin_alt_m = self.origin_alt_m

        desired_lat_deg = self._desired_lat_deg
        if desired_lat_deg is None:
            desired_lat_deg = lat_deg
        desired_lon_deg = self._desired_lon_deg
        if desired_lon_deg is None:
            desired_lon_deg = lon_deg
        desired_alt_m = self._desired_alt_m
        if desired_alt_m is None:
            desired_alt_m = alt_m
        desired_heading_deg = self._desired_heading_deg
        if desired_heading_deg is None:
            desired_heading_deg = heading_deg

        if self.target_alt is not None:
            rel_alt_m = alt_m - (origin_alt_m if origin_alt_m is not None else alt_m)
            dz = self.target_alt - rel_alt_m
            desired_alt_m = (origin_alt_m if origin_alt_m is not None else alt_m) + self.target_alt
            if abs(dz) < 0.1 and abs(vz_mps) < 0.1:
                self.target_alt = None
                if self.mode == "TAKEOFF":
                    self.mode = "GUIDED"
                elif self.mode == "LAND":
                    self.armed = False
                    self.mode  = "DISARMED"
                    self._zero_cmds()
                    self._reset_inertia()
        else:
            self._body_forward_mps = integrate_axis_velocity(
                current_mps=self._body_forward_mps,
                target_mps=self.cmd_forward,
                dt=dt,
            )
            self._body_right_mps = integrate_axis_velocity(
                current_mps=self._body_right_mps,
                target_mps=self.cmd_right,
                dt=dt,
            )
            desired_lat_deg, desired_lon_deg, desired_alt_m, desired_heading_deg = (
                integrate_drone_pose(
                    cmd_forward=self._body_forward_mps,
                    cmd_right=self._body_right_mps,
                    cmd_up=self.cmd_up,
                    cmd_yaw_rate=self.cmd_yaw_rate,
                    lat_deg=desired_lat_deg,
                    lon_deg=desired_lon_deg,
                    alt_m=desired_alt_m,
                    heading_deg=desired_heading_deg,
                    dt=dt,
                )
            )

        self._desired_lat_deg = desired_lat_deg
        self._desired_lon_deg = desired_lon_deg
        self._desired_alt_m = desired_alt_m
        self._desired_heading_deg = desired_heading_deg
        self._visual_roll_deg, self._visual_pitch_deg = integrate_visual_attitude(
            cmd_forward=self._body_forward_mps,
            cmd_right=self._body_right_mps,
            roll_deg=self._visual_roll_deg,
            pitch_deg=self._visual_pitch_deg,
            dt=dt,
        )

        if not self.armed:
            self._send_fg_control(FGControl(0.0, 0.0, 0.0, 0.0))
            return

        control = compute_drone_control(
            lat_deg=desired_lat_deg,
            lon_deg=desired_lon_deg,
            altitude_ft=desired_alt_m / FPS_TO_MPS,
            heading_deg=desired_heading_deg,
            roll_deg=self._visual_roll_deg,
            pitch_deg=self._visual_pitch_deg,
        )
        self._send_fg_control(control)

    def _send_fg_control(self, control: FGControl) -> None:
        if self._ctrl_sock is None:
            return
        try:
            self._ctrl_sock.sendto(control.csv_line().encode(),
                                   ("127.0.0.1", FG_CTRL_PORT))
        except OSError:
            pass

    def _publish_placeholder_frame(self) -> None:
        """Write a dark frame during FG startup so dctl shows a live feed immediately."""
        if self.video is None:
            return
        slot  = self.video.getPtr(0)
        frame = np.array(self.video[slot], copy=False)
        frame[:] = 0   # black
        pts = int(time.monotonic() * 1_000_000)
        self.video.setVpts(slot, pts)
        self.video.setApts(slot, pts)
        self.video.next(1)

    def _publish_frame(self) -> None:
        if self.video is None:
            return
        with self._frame_lock:
            raw = self._frame_bytes
            self._frame_bytes = None
        if raw is None:
            return

        slot  = self.video.getPtr(0)
        frame = np.array(self.video[slot], copy=False)
        # ffmpeg outputs top-to-bottom; video segment stores bottom-to-top
        # (OpenGL convention) so dctl's [::-1] display flip is correct.
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(
            self.args.height, self.args.width, 3)
        frame[:] = arr[::-1]
        pts = int(time.monotonic() * 1_000_000)
        self.video.setVpts(slot, pts)
        self.video.setApts(slot, pts)
        self.video.next(1)
        self._last_frame_time = time.monotonic()

    def _publish_status(self) -> None:
        if self.status is None:
            return

        with self._state_lock:
            st    = self.fg_state
            lat   = st.lat_deg
            lon   = st.lon_deg
            alt   = st.alt_m
            roll  = st.roll_deg
            pitch = st.pitch_deg
            hdg   = st.heading_deg
            vx    = st.vx_mps
            vy    = st.vy_mps
            vz    = st.vz_mps
            spd   = st.speed_mps
            origin_alt_m = self.origin_alt_m
        rel_alt = alt - (origin_alt_m if origin_alt_m is not None else alt)

        last_cmd = (-1.0 if self.last_cmd_monotonic is None
                    else time.monotonic() - self.last_cmd_monotonic)

        values = {
            "sim.id":                 self.args.id,
            "sim.map":                "flightgear",
            "sim.time_s":             f"{time.monotonic() - self.started:.3f}",
            "drone.armed":            "1" if self.armed else "0",
            "drone.mode":             self.mode,
            "drone.x_m":             "0.000",
            "drone.y_m":             "0.000",
            "drone.z_m":              f"{rel_alt:.3f}",
            "drone.lat_deg":          f"{lat:.7f}",
            "drone.lon_deg":          f"{lon:.7f}",
            "drone.alt_m":            f"{alt:.3f}",
            "target.lat_deg":         "",
            "target.lon_deg":         "",
            "target.alt_m":           "",
            "drone.roll_deg":         f"{roll:.2f}",
            "drone.pitch_deg":        f"{pitch:.2f}",
            "drone.heading_deg":      f"{hdg:.2f}",
            "drone.vx_mps":           f"{vx:.3f}",
            "drone.vy_mps":           f"{vy:.3f}",
            "drone.vz_mps":           f"{vz:.3f}",
            "drone.speed_mps":        f"{spd:.3f}",
            "drone.battery_pct":      f"{self.battery_pct:.1f}",
            "drone.crashed":          "0",
            "drone.last_command_s":   f"{last_cmd:.3f}",
            "link.command_count":     str(self.command_count),
            "link.last_command_type": self.last_cmd_type,
            "status.message":         self.status_message,
        }
        self.status.setAll(values)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def run(self) -> None:
        try:
            self._open_sockets()
            self._open_ipc()
            # Publish an initial status immediately so dctl can show something
            # while FlightGear is still starting up.
            self.status_message = "starting FlightGear"
            self._publish_status()

            self._start_xvfb()
            self._start_flightgear()

            if not self.args.no_ui:
                try:
                    self.ui = StatusUi(self)
                except tk.TclError as exc:
                    print(f"dfgb: UI unavailable ({exc}), running headless",
                          file=sys.stderr)

            self._wait_for_fg()
            self._start_ffmpeg()

            frame_period = 1.0 / max(1, self.args.fps)
            while self.running:
                t0 = time.monotonic()
                if (self.module_bus is not None and
                        any(requests_shutdown(event)
                            for event in self.module_bus.receive())):
                    self.running = False
                    break
                self._drain_commands()
                self._recv_state()
                self._send_control()
                self._publish_frame()
                self._publish_status()
                if self.ui is not None:
                    self.ui.update()
                    if self.ui.closed:
                        self.running = False
                elapsed = time.monotonic() - t0
                time.sleep(max(0.0, frame_period - elapsed))
        finally:
            self._stop()

    def _stop(self) -> None:
        self.running = False
        if getattr(self, "module_bus", None) is not None:
            self.module_bus.publish("module.goodbye", payload={"state": "stopped"})
            self.module_bus.close()
        if self.ui is not None:
            self.ui.close()
        for proc in (self._proc_ffmpeg, self._proc_fg, self._proc_xvfb):
            if proc and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()   # reap zombie after kill
        for sock in (self._ctrl_sock, self._state_sock):
            if sock:
                try:
                    sock.close()
                except OSError:
                    pass
        for handle in (self.status, self.command, self.video):
            if handle is not None:
                try:
                    handle.close()
                except Exception:
                    pass
        if self._fg_log:
            self._fg_log.close()


# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------

def _run_cmd(cmd: list[str]) -> bool:
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd).returncode == 0


def _apt_install(packages: list[str]) -> None:
    print(f"\nInstalling via apt: {', '.join(packages)}")
    if not _run_cmd(["sudo", "apt-get", "install", "-y"] + packages):
        raise SystemExit("apt-get install failed")


def _dpkg_installed(package: str) -> bool:
    if not shutil.which("dpkg-query"):
        return True
    return subprocess.run(
        ["dpkg-query", "-W", "-f=${Status}", package],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    ).stdout.strip() == "install ok installed"


def _find_fg_protocol_dir() -> Path | None:
    for root in [
        "/usr/share/games/flightgear",
        "/usr/share/flightgear",
        "/usr/local/share/flightgear",
        "/snap/flightgear/current/usr/share/flightgear",
    ]:
        p = Path(root) / "Protocol"
        if p.is_dir():
            return p
    return None


def _is_dfgb_shared_model_stub(path: Path) -> bool:
    try:
        data = path.read_text()
    except (OSError, UnicodeDecodeError):
        return False
    return data in {
        FGBridge._EMPTY_XML_MODEL,
        FGBridge._EMPTY_AC3D_MODEL,
    }


def _shared_models_complete(path: Path) -> bool:
    for rel in FGBridge._SHARED_MODEL_SENTINELS:
        candidate = path / rel
        if not candidate.is_file():
            return False
        if _is_dfgb_shared_model_stub(candidate):
            return False
    return True


def _safe_extract_tar(archive: Path, dest: Path) -> None:
    dest_real = dest.resolve()
    with tarfile.open(archive) as tf:
        for member in tf.getmembers():
            target = (dest / member.name).resolve()
            if target != dest_real and dest_real not in target.parents:
                raise RuntimeError(f"unsafe path in archive: {member.name}")
        tf.extractall(dest)


def _find_extracted_scenery_root(path: Path) -> Path:
    if (path / "Models").is_dir():
        return path
    children = [p for p in path.iterdir() if p.is_dir()]
    for child in children:
        if (child / "Models").is_dir():
            return child
    raise RuntimeError("SharedModels archive did not contain a Models directory")


def _download_file(url: str, dst: Path) -> None:
    tmp = dst.with_suffix(dst.suffix + ".partial")
    tmp.unlink(missing_ok=True)
    print(f"Downloading {url}")
    with urllib.request.urlopen(url) as response, open(tmp, "wb") as out:
        shutil.copyfileobj(response, out)
    tmp.replace(dst)


def _install_shared_models(args: argparse.Namespace) -> None:
    target = Path(args.shared_models_dir).expanduser()
    url = args.shared_models_url
    if _shared_models_complete(target):
        print(f"Shared scenery models already installed at {target}.")
        return

    archive = target.parent / "SharedModels.txz"
    target.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nInstalling shared scenery models into {target}")
    _download_file(url, archive)
    with tempfile.TemporaryDirectory(prefix="dfgb-shared-models-") as td:
        extract_dir = Path(td)
        print(f"Extracting {archive}")
        _safe_extract_tar(archive, extract_dir)
        root = _find_extracted_scenery_root(extract_dir)
        target.mkdir(parents=True, exist_ok=True)
        for item in root.iterdir():
            dst = target / item.name
            if item.is_dir():
                shutil.copytree(item, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dst)

    if not _shared_models_complete(target):
        raise SystemExit(
            "Shared models were extracted, but required model files are still "
            f"missing under {target}"
        )
    print("Shared scenery models installed.")


def do_install(args: argparse.Namespace) -> None:
    print("=== dfgb install ===\n")

    missing = []
    if not shutil.which("fgfs"):
        missing.append("flightgear")
    if not shutil.which("Xvfb"):
        missing.append("xvfb")
    if not shutil.which("ffmpeg"):
        missing.append("ffmpeg")
    if not shutil.which("xdotool"):
        missing.append("xdotool")
    if not _dpkg_installed("flightgear-data-all"):
        missing.append("flightgear-data-all")

    if missing:
        _apt_install(missing)
    else:
        print("System packages already present "
              "(fgfs, FlightGear data, Xvfb, ffmpeg, xdotool).")

    _install_shared_models(args)

    proto_dir = _find_fg_protocol_dir()
    if proto_dir is None:
        print(
            "\nWARNING: could not locate FlightGear Protocol directory.\n"
            "Protocol files NOT installed — copy them manually:\n"
            f"  {PROTOCOLS_DIR}/dvision2-ctrl.xml\n"
            f"  {PROTOCOLS_DIR}/dvision2-state.xml\n"
        )
    else:
        for name in ("dvision2-ctrl.xml", "dvision2-state.xml"):
            src = PROTOCOLS_DIR / name
            dst = proto_dir / name
            if dst.exists() and dst.read_bytes() == src.read_bytes():
                print(f"Protocol {name} already up to date.")
                continue
            print(f"Installing {name} → {dst}")
            if os.access(proto_dir, os.W_OK):
                shutil.copy2(src, dst)
            else:
                if not _run_cmd(["sudo", "cp", str(src), str(dst)]):
                    raise SystemExit(f"Failed to install {name}")
        print("Protocol files installed.")

    bridge = FGBridge.__new__(FGBridge)
    bridge.args = args
    fg_root = bridge._fg_root()
    if fg_root:
        ufo_dir = Path(fg_root) / "Aircraft" / "ufo"
        if ufo_dir.is_dir():
            print(f"Aircraft 'ufo' found at {ufo_dir}.")
        else:
            print(
                f"\nWARNING: 'ufo' aircraft not found in {fg_root}/Aircraft.\n"
                "Install flightgear-data-all, or pass --aircraft to dfgb.py."
            )

    print("\n=== Installation complete ===")
    print("Run:  python3 dfgb/dfgb.py --id area1 --aircraft ufo\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="dvision2 FlightGear bridge — replaces dsim",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--install",     action="store_true",
                   help="install FlightGear dependencies, shared models, and FG protocol files")
    p.add_argument("--id",          help="instance id (required unless --install)")
    p.add_argument("--aircraft",    default="ufo",
                   help="FlightGear aircraft model to use")
    p.add_argument("--airport",     default="KSFO",
                   help="spawn airport ICAO; use empty string to spawn by lat/lon")
    p.add_argument("--runway",      default="28R",
                   help="spawn runway; only used with --airport")
    p.add_argument("--lat",         type=float, default=37.6213,
                   help="spawn latitude when --airport is empty")
    p.add_argument("--lon",         type=float, default=-122.3790,
                   help="spawn longitude when --airport is empty")
    p.add_argument("--alt",         type=float, default=50.0,
                   help="spawn altitude in feet when --airport is empty")
    p.add_argument("--display",     type=int,   default=99,
                   help="Xvfb display number")
    p.add_argument("--width",       type=int,   default=640)
    p.add_argument("--height",      type=int,   default=480)
    p.add_argument("--fps",         type=int,   default=30)
    p.add_argument("--bufs",        type=int,   default=4)
    p.add_argument("--cmd-size",    type=int,   default=65536)
    p.add_argument("--fg-root",     default=None,
                   help="override FlightGear data root directory")
    p.add_argument("--fg-aircraft", default=None,
                   help="extra --fg-aircraft path passed to fgfs")
    p.add_argument("--shared-models-dir",
                   default=str(Path.home() / ".fgfs" / "TerraSync"),
                   help="directory containing FlightGear shared scenery models")
    p.add_argument("--shared-models-url", default=SHARED_MODELS_URL,
                   help="SharedModels.txz URL used by --install")
    p.add_argument("--disable-terrasync", action="store_true",
                   help="disable automatic scenery downloads; may show ocean if local scenery is missing")
    p.add_argument("--no-ui",       action="store_true",
                   help="disable the status window (for background / headless use)")
    p.add_argument("--verbose",     action="store_true")
    args = p.parse_args(argv)

    if not args.install:
        if not args.id:
            p.error("--id is required unless --install is specified")
        validate_id(args.id)

    return args


def main(argv: list[str] | None = None) -> int:
    disable_input_method()
    args = parse_args(sys.argv[1:] if argv is None else argv)

    if args.install:
        do_install(args)
        return 0

    bridge = FGBridge(args)

    def _stop(_sig, _frame):
        bridge.running = False

    signal.signal(signal.SIGINT,  _stop)
    signal.signal(signal.SIGTERM, _stop)

    try:
        bridge.run()
    except Exception as exc:
        print(f"dfgb: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
