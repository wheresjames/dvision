#!/usr/bin/env python3
"""daic – AI drone controller for dvision2.

Connects to the same shared-memory buffers as dctl but drives the drone
autonomously.  A manual toggle in the UI enables or disables AI control.
"""

from __future__ import annotations

import argparse
import math
import signal
import sys
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import numpy as np
from PIL import Image, ImageTk

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dvision2_common import (
    STATUS_KEYS, encode_command, load_pymembus,
    restore_window_pos, save_window_pos, shared_names, validate_id,
)
from daic.detector import detect, Detection
from daic.planner import Planner, State
from daic.flight_log import FlightLogger
from daic.avoidance import apply_obstacle_avoidance, apply_search_approach_brake
from daic.optical_flow_avoidance import (
    OpticalFlowAvoidance, fuse_obstacle_sectors,
)
from daic.local_map import (
    LocalOccupancyMap, pose_from_status, target_xy_from_status,
)
from daic.orb_slam3_detector import (
    ORBSLAM3Detector, ObstacleSectors, _NULL_SECTORS,
    _S_OK, _S_RECENTLY_LOST, _S_LOST,
)
try:
    from daic.mini_slam_detector import MiniSLAMDetector
except ImportError:
    MiniSLAMDetector = None

# ---------------------------------------------------------------------------
# SLAM installation helpers
# ---------------------------------------------------------------------------

_VOCAB_URL = (
    "https://github.com/UZ-SLAMLab/ORB_SLAM3"
    "/raw/HEAD/Vocabulary/ORBvoc.txt.tar.gz"
)
_VOCAB_DEFAULT_PATH = Path.home() / ".local" / "share" / "daic" / "ORBvoc.txt"


def _download_vocab(dest: Path, verbose: bool = False) -> None:
    """Download and extract ORBvoc.txt.tar.gz to dest, with a progress bar."""
    import tarfile
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    tgz = dest.parent / "ORBvoc.txt.tar.gz"

    _last_pct: list[int] = [-1]

    def _progress(count: int, block: int, total: int) -> None:
        if total > 0:
            pct = min(100, count * block * 100 // total)
            if pct != _last_pct[0]:
                _last_pct[0] = pct
                mb = count * block / 1_000_000
                bar = "#" * (pct // 5) + "-" * (20 - pct // 5)
                print(f"\r    [{bar}] {pct:3d}%  {mb:.0f} MB",
                      end="", flush=True)

    print(f"  downloading from GitHub...", flush=True)
    try:
        urllib.request.urlretrieve(_VOCAB_URL, tgz,
                                   None if verbose else _progress)
    finally:
        print()   # end the progress line

    print("  extracting...", end=" ", flush=True)
    with tarfile.open(tgz) as tf:
        # Only extract the vocabulary file itself; ignore any directory traversal.
        for member in tf.getmembers():
            if member.name.endswith("ORBvoc.txt") and "/" not in member.name.lstrip("/"):
                member.name = "ORBvoc.txt"
                tf.extract(member, dest.parent, filter="data")
                break
        else:
            # Fallback: extract everything and let the file land wherever.
            tf.extractall(dest.parent, filter="data")
    tgz.unlink(missing_ok=True)
    print("ok")

    if not dest.exists():
        raise RuntimeError(f"extraction succeeded but {dest} not found")


def _cmd_install(verbose: bool = False) -> int:
    """Check and install SLAM dependencies. Prints a report; returns exit code."""
    import importlib
    import subprocess

    all_ok = True
    W = 26   # column width for alignment

    print("daic: vision dependency check")
    print("=" * 56)

    # ── Python packages ───────────────────────────────────────────────
    print("\nPython packages")
    for mod, pkg in [("numpy", "numpy"), ("cv2", "opencv-python"), ("PIL", "Pillow")]:
        try:
            m   = importlib.import_module(mod)
            ver = getattr(m, "__version__", "?")
            print(f"  {pkg:<{W}} ok  ({ver})")
        except ImportError:
            print(f"  {pkg:<{W}} MISSING – installing…", flush=True)
            r = subprocess.run(
                [sys.executable, "-m", "pip", "install", "--quiet", pkg],
                capture_output=not verbose,
            )
            if r.returncode == 0:
                print(f"  {pkg:<{W}} installed")
            else:
                print(f"  {pkg:<{W}} FAILED  →  pip install {pkg}")
                all_ok = False

    # ── OpenCV vision modules ────────────────────────────────────────
    print("\nOpenCV vision avoidance")
    try:
        import cv2
        if hasattr(cv2, "calcOpticalFlowFarneback"):
            print(f"  {'optical_flow':<{W}} ok  (Farneback)")
        else:
            print(f"  {'optical_flow':<{W}} FAILED  → opencv-python missing Farneback")
            all_ok = False
    except Exception as exc:
        print(f"  {'optical_flow':<{W}} FAILED – {exc}")
        all_ok = False

    # ── Mini SLAM ─────────────────────────────────────────────────────
    print("\nMini SLAM  (pure-OpenCV, optional supporting signal)")
    try:
        mini_mod = importlib.import_module("daic.mini_slam_detector")
        mini_mod.MiniSLAMDetector()
        print(f"  {'mini_slam_detector':<{W}} ok  (active by default)")
    except Exception as exc:
        print(f"  {'mini_slam_detector':<{W}} FAILED – {exc}")
        all_ok = False

    # ── ORB_SLAM3 Python bindings ─────────────────────────────────────
    print("\nFull ORB_SLAM3 Python bindings  (optional)")
    try:
        importlib.import_module("orbslam3")
        print(f"  {'orbslam3':<{W}} installed  ✓")
    except ImportError:
        print(f"  {'orbslam3':<{W}} not installed  (mini SLAM is used instead)")

    # ── Vocabulary file ───────────────────────────────────────────────
    vocab = _VOCAB_DEFAULT_PATH
    print(f"\nVocabulary file  →  {vocab}")
    if vocab.exists():
        mb = vocab.stat().st_size / 1_000_000
        print(f"  {'ORBvoc.txt':<{W}} exists  ({mb:.0f} MB)")
    else:
        print(f"  {'ORBvoc.txt':<{W}} not found – downloading (~74 MB)…")
        try:
            _download_vocab(vocab, verbose)
            mb = vocab.stat().st_size / 1_000_000
            print(f"  {'ORBvoc.txt':<{W}} ok  ({mb:.0f} MB)")
        except Exception as exc:
            print(f"  {'ORBvoc.txt':<{W}} download failed: {exc}")
            print(f"  Manual:  wget -O {vocab}.tar.gz '{_VOCAB_URL}'")
            print(f"           tar -xf {vocab}.tar.gz -C {vocab.parent}")

    # ── ORB_SLAM3 build instructions ──────────────────────────────────
    print("\n" + "─" * 56)
    print("To install full ORB_SLAM3 (optional — mini SLAM works now):")
    print()
    print("  # 1. Build dependencies")
    print("  sudo apt install -y build-essential cmake git \\")
    print("    libopencv-dev libeigen3-dev libssl-dev \\")
    print("    libglew-dev libpython3-dev pybind11-dev")
    print()
    print("  # 2. Clone and build ORB_SLAM3")
    print("  git clone https://github.com/UZ-SLAMLab/ORB_SLAM3 /opt/ORB_SLAM3")
    print("  cd /opt/ORB_SLAM3 && chmod +x build.sh && ./build.sh")
    print()
    print("  # 3. Build Python bindings (pybind11 community fork)")
    print("  git clone https://github.com/niconielsen32/ORB-SLAM3-python")
    print("  cd ORB-SLAM3-python && pip install .")
    print()
    print(f"  # 4. Run daic with full ORB_SLAM3")
    print(f"  ./daic/daic.py --id area1 --enable-ai \\")
    print(f"    --slam-vocab {vocab}")
    print()
    print("=" * 56)
    print("Mini SLAM is running and needs no additional setup.")
    return 0 if all_ok else 1


# ---------------------------------------------------------------------------
# Dark theme (matches dctl palette)
# ---------------------------------------------------------------------------
_BG        = "#0d1117"
_BG_PANEL  = "#161b22"
_BG_ENTRY  = "#21262d"
_FG        = "#e6edf3"
_FG_DIM    = "#8b949e"
_ACCENT    = "#58a6ff"
_ACCENT2   = "#57ab5a"   # green for healthy / enabled
_ACCENT3   = "#e5534b"   # red for error / disabled
_BORDER    = "#30363d"
_BTN_BG    = "#21262d"
_BTN_ACT   = "#30363d"
_VIDEO_BG  = "#010409"
_SEARCH_HOLD_YAW_DPS = 18.0
_APPROACH_BLOCK_RISK = 0.25
_APPROACH_BLOCK_FRONT_OCC_M = 1.5


def _search_hold_scan_fields() -> dict:
    return {
        "forward_mps": 0.0,
        "right_mps": 0.0,
        "up_mps": 0.0,
        "yaw_rate_dps": _SEARCH_HOLD_YAW_DPS,
    }


def _frontish_risk(sectors) -> float:
    def risk(name: str) -> float:
        try:
            return max(0.0, min(1.0, float(getattr(sectors, name, 0.0))))
        except (TypeError, ValueError):
            return 0.0

    return max(
        risk("front"),
        risk("front_left") * 0.7,
        risk("front_right") * 0.7,
    )


def _approach_gate_reason(sectors, local_map_diag: dict | None) -> str | None:
    front_risk = _frontish_risk(sectors)
    if front_risk >= _APPROACH_BLOCK_RISK:
        return f"front risk {front_risk:.2f}"

    if local_map_diag is not None:
        try:
            front_occ_m = local_map_diag.get("front_block_occ_m")
            if front_occ_m is None:
                front_occ_m = local_map_diag.get("front_occ_m")
            if front_occ_m is not None and float(front_occ_m) <= _APPROACH_BLOCK_FRONT_OCC_M:
                return f"front occ {float(front_occ_m):.2f}m"
        except (TypeError, ValueError):
            pass

    return None


def _approach_block_fields(fields: dict) -> dict:
    blocked = dict(fields)
    blocked["forward_mps"] = 0.0
    blocked["right_mps"] = 0.0
    blocked["up_mps"] = 0.0
    return blocked


# ---------------------------------------------------------------------------
# Component health tracker
# ---------------------------------------------------------------------------

class Health:
    """Tracks per-component status for the health panel."""

    COMPONENTS = ["video", "command", "status", "detector", "planner", "slam"]

    def __init__(self) -> None:
        self._states: dict[str, str] = {c: "wait" for c in self.COMPONENTS}
        self._detail: dict[str, str] = {c: "" for c in self.COMPONENTS}

    def ok(self, name: str, detail: str = "") -> None:
        self._states[name] = "ok"
        self._detail[name] = detail

    def warn(self, name: str, detail: str = "") -> None:
        self._states[name] = "warn"
        self._detail[name] = detail

    def err(self, name: str, detail: str = "") -> None:
        self._states[name] = "err"
        self._detail[name] = detail

    def wait(self, name: str, detail: str = "") -> None:
        self._states[name] = "wait"
        self._detail[name] = detail

    def state(self, name: str) -> str:
        return self._states.get(name, "wait")

    def detail(self, name: str) -> str:
        return self._detail.get(name, "")


# ---------------------------------------------------------------------------
# Annotate frame with detection overlay
# ---------------------------------------------------------------------------

def _annotate(frame_rgb: np.ndarray, detection: Detection) -> np.ndarray:
    """Draw detection circle on a copy of frame_rgb (local annotation only)."""
    try:
        import cv2
    except ImportError:
        return frame_rgb
    bgr = np.ascontiguousarray(frame_rgb[:, :, ::-1])  # RGB → BGR, contiguous
    if detection.visible:
        cx, cy, r = int(detection.cx), int(detection.cy), int(detection.radius)
        cv2.circle(bgr, (cx, cy), r, (50, 220, 50), 2)
        cv2.circle(bgr, (cx, cy), 4, (50, 220, 50), -1)
        label = f"{detection.confidence:.2f}"
        cv2.putText(bgr, label, (cx + r + 4, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50, 220, 50), 1)
    return bgr[:, :, ::-1].copy()  # BGR → RGB, contiguous


def _client_rgb_frame(frame: np.ndarray) -> np.ndarray:
    """Convert shared-memory renderer output to daic's interpreted RGB frame."""
    return np.ascontiguousarray(frame)


def _display_rgb_frame(frame: np.ndarray) -> np.ndarray:
    """Convert shared-memory renderer output to daic's displayed RGB frame."""
    return np.ascontiguousarray(frame)


def _display_detection(detection: Detection, width: int) -> Detection:
    """Convert an interpreted-frame detection into display coordinates."""
    return detection


def _display_sectors(sectors: ObstacleSectors) -> ObstacleSectors:
    """Convert interpreted-frame sector risks into display HUD coordinates."""
    return sectors


def _annotate_slam(frame_rgb: np.ndarray,
                   sectors: ObstacleSectors,
                   tracking_state: int,
                   n_points: int,
                   scale: float | None) -> np.ndarray:
    """Draw SLAM status text and five-sector risk bar HUD onto frame_rgb."""
    try:
        import cv2
    except ImportError:
        return frame_rgb

    bgr = np.ascontiguousarray(frame_rgb[:, :, ::-1])
    h, w = bgr.shape[:2]

    # ── Status line ──────────────────────────────────────────────────
    _state_labels = {
        -1: "NOT_READY", 0: "NO_IMGS", 1: "INIT",
         2: "OK",        3: "LOST+",   4: "LOST",
    }
    tag      = _state_labels.get(tracking_state, f"S{tracking_state}")
    sc_str   = f"  {scale:.3f}m/u" if scale else "  unscaled"
    txt      = f"SLAM:{tag}  pts:{n_points}{sc_str}"
    font     = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(txt, font, 0.38, 1)
    cv2.rectangle(bgr, (6, 4), (tw + 14, th + 11), (10, 14, 20), -1)
    state_col = ({_S_OK: (80, 200, 80), _S_RECENTLY_LOST: (50, 160, 230)}
                 .get(tracking_state, (110, 110, 110)))
    cv2.putText(bgr, txt, (9, th + 7),  font, 0.38, (0, 0, 0),    2, cv2.LINE_AA)
    cv2.putText(bgr, txt, (8, th + 6),  font, 0.38, state_col,     1, cv2.LINE_AA)

    # ── Sector bars ───────────────────────────────────────────────────
    BAR_H   = 44
    LABEL_H = 15
    FILL_H  = BAR_H - LABEL_H
    bar_y   = h - BAR_H
    bar_w   = w // 5
    conf    = sectors.confidence

    for i, (label, risk) in enumerate([
        ("L",  sectors.left),
        ("FL", sectors.front_left),
        ("F",  sectors.front),
        ("FR", sectors.front_right),
        ("R",  sectors.right),
    ]):
        x0, x1 = i * bar_w, (i + 1) * bar_w
        cv2.rectangle(bgr, (x0, bar_y), (x1, h), (18, 22, 28), -1)

        if conf > 0.0 and risk > 0.01:
            fill_px = max(1, int(risk * FILL_H))
            fy0 = bar_y + FILL_H - fill_px
            cv2.rectangle(bgr, (x0 + 1, fy0), (x1 - 1, bar_y + FILL_H),
                          _risk_color_bgr(risk), -1)

        lx  = x0 + (bar_w - len(label) * 6) // 2
        col = (85, 95, 105) if conf == 0.0 else (175, 180, 188)
        cv2.putText(bgr, label, (lx, h - 3), font, 0.30, col, 1, cv2.LINE_AA)
        if i > 0:
            cv2.line(bgr, (x0, bar_y), (x0, h), (35, 40, 50), 1)

    cv2.line(bgr, (0, bar_y), (w, bar_y), (35, 40, 50), 1)
    return bgr[:, :, ::-1].copy()


def _risk_color_bgr(risk: float) -> tuple[int, int, int]:
    """BGR: green (low risk) → orange → red (high risk)."""
    if risk < 0.35:
        return (50, 190, 60)
    if risk < 0.65:
        return (40, 140, 220)
    return (50, 60, 230)


def _sectors_debug(sectors: ObstacleSectors) -> dict:
    return {
        "method": sectors.method,
        "confidence": round(sectors.confidence, 3),
        "front": round(sectors.front, 3),
        "front_left": round(sectors.front_left, 3),
        "front_right": round(sectors.front_right, 3),
        "left": round(sectors.left, 3),
        "right": round(sectors.right, 3),
        "front_range_m": _round_opt(sectors.front_range_m),
        "front_left_range_m": _round_opt(sectors.front_left_range_m),
        "front_right_range_m": _round_opt(sectors.front_right_range_m),
        "left_range_m": _round_opt(sectors.left_range_m),
        "right_range_m": _round_opt(sectors.right_range_m),
    }


def _round_opt(value: float | None) -> float | None:
    return round(value, 2) if value is not None else None


def _nice_step(target: float) -> float:
    """Round target up to the nearest 1/2/5 × 10^n."""
    if target <= 0:
        return 1.0
    exp  = math.floor(math.log10(target))
    base = 10.0 ** exp
    for m in (1.0, 2.0, 5.0, 10.0):
        if m * base >= target:
            return m * base
    return 10.0 * base


# ---------------------------------------------------------------------------
# Main controller / UI
# ---------------------------------------------------------------------------

class DaicController:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.pymembus = load_pymembus()
        self.names = shared_names(args.id)

        self.video   = None
        self.command = None
        self.status  = None
        self.status_epoch = 0
        self.last_status_time = 0.0
        self.last_video_seq  = -1
        self.running = True

        self.health   = Health()
        self.planner  = Planner(img_w=args.video_w, img_h=args.video_h)

        vocab = getattr(args, "slam_vocab", None)
        if vocab:
            self.slam_detector: ORBSLAM3Detector | MiniSLAMDetector = ORBSLAM3Detector(
                vocab_path=vocab,
                settings_path=getattr(args, "slam_settings", None),
            )
        else:
            if MiniSLAMDetector is None:
                raise RuntimeError("opencv-python is required; run ./daic/daic.py --install")
            self.slam_detector = MiniSLAMDetector()
        self.flow_detector = OpticalFlowAvoidance()
        self.local_map = LocalOccupancyMap()
        self._last_slam_sectors = _NULL_SECTORS
        self._last_flow_sectors = _NULL_SECTORS
        self._last_sectors  = _NULL_SECTORS
        self._slam_started  = False
        self._slam_map_canvas: tk.Canvas | None = None
        self._local_map_canvas: tk.Canvas | None = None
        self._slam_canvas_sz = 220
        self._last_route_status: dict = {}
        self.last_detection = Detection(False, 0, 0, 0, 0)
        self.ai_enabled = False
        self._last_heartbeat = 0.0
        self._pending_enable_ai = args.enable_ai
        self._auto_log = not bool(args.log_file)
        self.logger: FlightLogger | None = (
            FlightLogger(args.log_file) if args.log_file else None
        )
        self._enable_reporting = args.enable_ai
        self.reporter = None   # RunReporter | None — created from sim.report_dir
        self._last_planner_state_name = "IDLE"
        self._last_target_dist_m: float | None = None
        self._last_annotated_frame: np.ndarray | None = None

        # Tk UI — all tk.*Var objects must come after tk.Tk()
        self.root = tk.Tk()
        self.root.title(f"daic {args.id}")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.photo = None
        self._status_vars: dict[str, tk.StringVar] = {}
        self._health_vars: dict[str, tk.StringVar]  = {}
        self._health_color_vars: dict[str, str]      = {}
        self._health_labels: dict[str, ttk.Label]    = {}
        self._state_var   = tk.StringVar(value="IDLE")
        self._ai_var      = tk.StringVar(value="AI: OFF")
        self._alt_lock    = tk.BooleanVar(value=True)
        self._planner_var = tk.StringVar(value="—")
        self._detect_var  = tk.StringVar(value="no target")
        self._conn_var    = tk.StringVar(value="connecting…")
        self._toggle_btn: ttk.Button | None = None
        self._build_ui()
        restore_window_pos(self.root, f"daic.{args.id}")

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _apply_theme(self) -> None:
        self.root.configure(bg=_BG)
        s = ttk.Style(self.root)
        s.theme_use("clam")
        s.configure(".", background=_BG, foreground=_FG,
                    bordercolor=_BORDER, darkcolor=_BG_PANEL,
                    lightcolor=_BG_PANEL, fieldbackground=_BG_ENTRY,
                    troughcolor=_BG_PANEL, selectbackground=_ACCENT,
                    selectforeground=_FG)
        s.configure("TFrame",     background=_BG)
        s.configure("Header.TFrame", background=_BG_PANEL)
        s.configure("TLabel",     background=_BG, foreground=_FG)
        s.configure("Dim.TLabel", background=_BG, foreground=_FG_DIM)
        s.configure("Brand.TLabel", background=_BG_PANEL, foreground=_FG,
                    font=("TkDefaultFont", 11, "bold"))
        s.configure("HeaderDim.TLabel", background=_BG_PANEL, foreground=_FG_DIM)
        s.configure("Video.TLabel", background=_VIDEO_BG, foreground=_FG_DIM)
        s.configure("TLabelframe",
                    background=_BG_PANEL, bordercolor=_BORDER,
                    lightcolor=_BORDER,   darkcolor=_BORDER)
        s.configure("TLabelframe.Label",
                    background=_BG_PANEL, foreground=_ACCENT)
        s.configure("TButton",
                    background=_BTN_BG, foreground=_FG,
                    bordercolor=_BORDER, lightcolor=_BORDER, darkcolor=_BORDER,
                    focuscolor=_ACCENT, padding=(10, 6), relief="flat")
        s.map("TButton",
              background=[("pressed", _BG_PANEL), ("active", _BTN_ACT)],
              bordercolor=[("active", _ACCENT)])
        s.configure("Accent.TButton", background="#1f6feb", foreground="#ffffff",
                    bordercolor="#388bfd", lightcolor="#388bfd", darkcolor="#1f6feb",
                    padding=(10, 6), relief="flat")
        s.map("Accent.TButton",
              background=[("pressed", "#1158c7"), ("active", "#388bfd")])
        s.configure("Danger.TButton", background="#da3633", foreground="#ffffff",
                    bordercolor="#f85149", lightcolor="#f85149", darkcolor="#da3633",
                    padding=(10, 6), relief="flat")
        s.map("Danger.TButton",
              background=[("pressed", "#b62324"), ("active", "#f85149")])
        # Health label styles
        for tag, color in (("Ok", _ACCENT2), ("Warn", "#e09440"), ("Err", _ACCENT3), ("Wait", _FG_DIM)):
            s.configure(f"Health{tag}.TLabel",
                        background=_BG_PANEL, foreground=color,
                        font=("TkFixedFont", 9, "bold"))
        s.configure("HealthKey.TLabel",
                    background=_BG_PANEL, foreground=_FG_DIM,
                    font=("TkFixedFont", 9))
        s.configure("HealthDetail.TLabel",
                    background=_BG_PANEL, foreground=_FG_DIM,
                    font=("TkDefaultFont", 8))

    def _build_ui(self) -> None:
        self._apply_theme()
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        # ── Top bar ──────────────────────────────────────────────────
        top = ttk.Frame(self.root, padding=(12, 10), style="Header.TFrame")
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(1, weight=1)
        ttk.Label(top, text=f"daic  {self.args.id}",
                  style="Brand.TLabel").grid(row=0, column=0, sticky="w", padx=(0, 12))
        ttk.Label(top, textvariable=self._conn_var,
                  style="HeaderDim.TLabel", width=58).grid(row=0, column=1, sticky="w")

        # ── Body ─────────────────────────────────────────────────────
        body = ttk.Frame(self.root, padding=(12, 12, 12, 12))
        body.grid(row=1, column=0, sticky="nsew")
        body.columnconfigure(0, weight=1)
        body.rowconfigure(0, weight=1)
        body.rowconfigure(1, weight=0)

        self.video_label = ttk.Label(
            body, text="waiting for video", anchor="center",
            style="Video.TLabel")
        self.video_label.grid(row=0, column=0, sticky="nsew")

        side = ttk.Frame(body, padding=(12, 0, 0, 0))
        side.grid(row=0, column=1, rowspan=2, sticky="ns")
        side.configure(width=300)
        side.grid_propagate(False)

        # ── AI Control ───────────────────────────────────────────────
        ai_frame = ttk.LabelFrame(side, text="AI Control", padding=8)
        ai_frame.grid(row=0, column=0, sticky="ew")

        ttk.Label(ai_frame, textvariable=self._ai_var,
                  font=("TkDefaultFont", 11, "bold")).grid(
            row=0, column=0, columnspan=2, pady=(0, 6))

        self._toggle_btn = ttk.Button(ai_frame, text="Enable AI", style="Accent.TButton",
                                      command=self._toggle_ai)
        self._toggle_btn.grid(row=1, column=0, sticky="ew", padx=(0, 6))
        ttk.Button(ai_frame, text="Emergency Stop", style="Danger.TButton",
                   command=self._emergency_stop).grid(row=1, column=1, sticky="ew")

        ttk.Label(ai_frame, text="state", style="Dim.TLabel").grid(
            row=2, column=0, sticky="w", padx=(0, 6), pady=(6, 0))
        ttk.Label(ai_frame, textvariable=self._state_var).grid(
            row=2, column=1, sticky="w", pady=(6, 0))

        ttk.Label(ai_frame, text="planner", style="Dim.TLabel").grid(
            row=3, column=0, sticky="w", padx=(0, 6))
        ttk.Label(ai_frame, textvariable=self._planner_var,
                  width=26).grid(row=3, column=1, sticky="w")

        ttk.Label(ai_frame, text="target", style="Dim.TLabel").grid(
            row=4, column=0, sticky="w", padx=(0, 6))
        ttk.Label(ai_frame, textvariable=self._detect_var, width=26).grid(
            row=4, column=1, sticky="w")

        ttk.Checkbutton(ai_frame, text="Altitude lock  (descend on target only)",
                        variable=self._alt_lock).grid(
            row=5, column=0, columnspan=2, sticky="w", pady=(8, 0))

        ai_frame.columnconfigure(1, weight=1)

        # ── Component Health ─────────────────────────────────────────
        health_frame = ttk.LabelFrame(side, text="Components", padding=8)
        health_frame.grid(row=1, column=0, sticky="ew", pady=(10, 0))

        for row, name in enumerate(Health.COMPONENTS):
            ttk.Label(health_frame, text=name,
                      style="HealthKey.TLabel").grid(
                row=row, column=0, sticky="w", padx=(0, 6))
            var = tk.StringVar(value="wait")
            lbl = ttk.Label(health_frame, textvariable=var,
                            style="HealthWait.TLabel", width=5)
            lbl.grid(row=row, column=1, sticky="w")
            det = tk.StringVar(value="")
            ttk.Label(health_frame, textvariable=det,
                      style="HealthDetail.TLabel", width=24).grid(
                row=row, column=2, sticky="w", padx=(4, 0))
            self._health_vars[name] = var
            self._health_color_vars[name] = det
            self._health_labels[name] = lbl

        # ── Telemetry ────────────────────────────────────────────────
        telem = ttk.LabelFrame(side, text="Telemetry", padding=8)
        telem.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        telem_rows = [
            ("armed",    "drone.armed"),
            ("mode",     "drone.mode"),
            ("x",        "drone.x_m"),
            ("y",        "drone.y_m"),
            ("z",        "drone.z_m"),
            ("heading",  "drone.heading_deg"),
            ("speed",    "drone.speed_mps"),
            ("battery",  "drone.battery_pct"),
            ("status",   "status.message"),
        ]
        for row, (label, key) in enumerate(telem_rows):
            ttk.Label(telem, text=label,
                      style="Dim.TLabel").grid(row=row, column=0, sticky="w", padx=(0, 8))
            var = tk.StringVar(value="-")
            self._status_vars[key] = var
            ttk.Label(telem, textvariable=var, width=16).grid(row=row, column=1, sticky="w")

        # ── Map strip under video ────────────────────────────────────
        map_strip = ttk.Frame(body)
        map_strip.grid(row=1, column=0, columnspan=2, sticky="w", pady=(10, 0))

        slam_frame = ttk.LabelFrame(map_strip, text="SLAM Map", padding=4)
        slam_frame.grid(row=0, column=0, sticky="w")
        sz = self._slam_canvas_sz
        self._slam_map_canvas = tk.Canvas(
            slam_frame, width=sz, height=sz,
            bg=_VIDEO_BG, highlightthickness=0,
        )
        self._slam_map_canvas.pack()

        local_frame = ttk.LabelFrame(map_strip, text="Local Route", padding=4)
        local_frame.grid(row=0, column=1, sticky="w", padx=(10, 0))
        self._local_map_canvas = tk.Canvas(
            local_frame, width=sz, height=sz,
            bg=_VIDEO_BG, highlightthickness=0,
        )
        self._local_map_canvas.pack()

        side.lift()

    # ------------------------------------------------------------------
    # Runtime
    # ------------------------------------------------------------------

    def open_missing(self) -> None:
        pm = self.pymembus
        if self.video is None:
            vid = pm.memvid()
            if vid.open_existing(self.names["video"]):
                self.video = vid
                self.health.ok("video", "connected")
            else:
                self.health.wait("video", "waiting")
        if self.command is None:
            cmd = pm.memcmd()
            if cmd.open(self.names["command"], self.args.cmd_size):
                self.command = cmd
                self.health.ok("command", "connected")
            else:
                self.health.wait("command", "waiting")
        if self.status is None:
            kv = pm.memkv()
            if kv.open(self.names["status"]):
                self.status = kv
                self.status_epoch = kv.getEpoch()
                self.last_status_time = time.monotonic()
                self.health.ok("status", "connected")
                if self._auto_log and self.logger is None:
                    self._init_logger_from_status()
                self._init_reporter_from_status()
            else:
                self.health.wait("status", "waiting")

    def _init_logger_from_status(self) -> None:
        if self.status is None or self.logger is not None:
            return
        report_dir = self.status.getAll().get("sim.report_dir", "")
        if report_dir:
            log_path = Path(report_dir) / "daic" / "flight.jsonl"
            self.logger = FlightLogger(log_path)

    def _init_reporter_from_status(self) -> None:
        if self.reporter is not None or not self._enable_reporting:
            return
        if self.status is None:
            return
        report_dir = self.status.getAll().get("sim.report_dir", "")
        if report_dir:
            from daic.run_reporter import RunReporter
            self.reporter = RunReporter(Path(report_dir) / "daic")

    def run(self) -> None:
        self.root.after(50, self.tick)
        self.root.mainloop()

    def tick(self) -> None:
        if not self.running:
            return
        now = time.monotonic()

        self.open_missing()
        if self._pending_enable_ai and self.command is not None:
            self._pending_enable_ai = False
            self._toggle_ai()
        self._try_start_slam()
        self._update_video()
        self._update_status(now)
        self._update_conn_text(now)
        self._run_ai(now)
        self._update_health_ui()
        self._draw_slam_map()
        self._draw_local_map()
        self.root.after(max(10, int(1000 / max(1, self.args.fps))), self.tick)

    def _update_video(self) -> None:
        if self.video is None:
            return
        seq = self.video.getSeq()
        if seq == self.last_video_seq or seq <= 0:
            return
        self.last_video_seq = seq
        slot  = self.video.getPtr(-1)
        frame = np.array(self.video[slot], copy=False)
        rgb   = _client_rgb_frame(frame)
        display_rgb = _display_rgb_frame(frame)

        # Run detector on every new frame.
        try:
            self.last_detection = detect(rgb)
            if self.last_detection.visible:
                self.health.ok("detector",
                               f"conf={self.last_detection.confidence:.2f}")
            else:
                self.health.ok("detector", "no target")
        except Exception as exc:
            self.health.err("detector", str(exc)[:30])

        # Run SLAM obstacle detection on the same raw frame.
        sd = self.slam_detector
        slam_sectors = self._last_slam_sectors
        if sd._available:
            try:
                slam_sectors = sd.detect_obstacles(rgb)
                self._last_slam_sectors = slam_sectors
                ts = sd.tracking_state
                n  = sd.n_map_points
                if ts == _S_OK:
                    self.health.ok("slam", f"ok pts={n}")
                elif ts == _S_RECENTLY_LOST:
                    self.health.warn("slam", f"lost+ pts={n}")
                elif ts == _S_LOST:
                    self.health.err("slam", "lost")
                else:
                    self.health.wait("slam", sd.status_text[:28])
            except Exception as exc:
                self.health.err("slam", str(exc)[:28])
        else:
            self.health.wait("slam", sd.status_text[:28])

        try:
            if self.status is not None:
                self.flow_detector.set_motion_from_status(self.status.getAll())
            self._last_flow_sectors = self.flow_detector.detect_obstacles(rgb)
            self._last_sectors = fuse_obstacle_sectors(
                slam_sectors, self._last_flow_sectors,
            )
        except Exception as exc:
            self.health.warn("slam", f"flow error: {str(exc)[:16]}")
            self._last_sectors = slam_sectors

        # Annotate and display.
        display_detection = _display_detection(
            self.last_detection, display_rgb.shape[1],
        )
        annotated = _annotate(display_rgb, display_detection)
        annotated = _annotate_slam(
            annotated, _display_sectors(self._last_sectors),
            sd.tracking_state, sd.n_map_points, sd.scale,
        )
        self._last_annotated_frame = annotated
        image = Image.fromarray(annotated, "RGB")
        max_w = self.args.display_w or self.video.getWidth()
        max_h = self.args.display_h or self.video.getHeight()
        image.thumbnail((max_w, max_h))
        self.photo = ImageTk.PhotoImage(image)
        self.video_label.configure(image=self.photo, text="")

    def _update_status(self, now: float) -> None:
        if self.status is None:
            return
        changed, epoch = self.status.getChanged(self.status_epoch)
        if changed:
            self.status_epoch = epoch
            self.last_status_time = now
        values = self.status.getAll()
        for key, var in self._status_vars.items():
            raw = values.get(key, "-")
            if key == "drone.armed":
                raw = "armed" if raw == "1" else "disarmed"
            var.set(raw)

    def _update_conn_text(self, now: float) -> None:
        stale = (self.status is not None and
                 now - self.last_status_time > 2.0)
        parts = [
            f"video={'ok' if self.video else 'wait'}",
            f"cmd={'ok' if self.command else 'wait'}",
            f"status={'stale' if stale else ('ok' if self.status else 'wait')}",
            f"AI={'ON' if self.ai_enabled else 'off'}",
        ]
        self._conn_var.set("  ".join(parts))

        if stale:
            self.health.warn("status", "stale > 2 s")
        elif self.status is not None:
            self.health.ok("status", "live")

    def _run_ai(self, now: float) -> None:
        if not self.ai_enabled:
            return
        if self.command is None:
            return

        # Build a status snapshot for the planner.
        status_snap: dict = {}
        if self.status is not None:
            status_snap = self.status.getAll()

        # Inject wall-clock staleness info.
        stale = now - self.last_status_time > 2.0
        if stale and self.status is not None:
            # Planner will failsafe; pass a marker so it can detect it.
            status_snap["_stale"] = "1"

        pose = pose_from_status(status_snap)
        if pose is not None:
            self.local_map.update(pose, self._last_sectors)

        try:
            out = self.planner.tick(self.last_detection, status_snap)
            self.health.ok("planner", out.status_text[:40])
        except Exception as exc:
            self.health.err("planner", str(exc)[:40])
            self._send("zero")
            return

        self._state_var.set(out.state.name)
        self._planner_var.set(out.status_text)
        det = self.last_detection
        if det.visible:
            self._detect_var.set(f"cx={det.cx:.0f} cy={det.cy:.0f} "
                                  f"r={det.radius:.0f} c={det.confidence:.2f}")
        else:
            self._detect_var.set("no target")

        target_xy = target_xy_from_status(status_snap)
        avoiding  = False
        effective_status = out.status_text
        if out.send_command:
            fields = dict(out.command_fields)
            if self._alt_lock.get() and out.command_type == "velocity":
                if out.state == State.LANDING:
                    fields["up_mps"] = min(0.0, fields.get("up_mps", 0.0))
                else:
                    fields["up_mps"] = 0.0
            if out.command_type == "velocity" and out.state == State.SEARCH:
                planned = None
                if pose is not None and target_xy is not None:
                    planned = self._local_route_command(status_snap)
                if planned is not None:
                    fields = planned.fields
                    self._last_route_status = {
                        "pose": pose,
                        "target_xy": target_xy,
                    }
                    effective_status = planned.status
                    self.health.ok("planner", planned.status[:40])
                elif pose is not None and target_xy is not None:
                    fields = _search_hold_scan_fields()
                    effective_status = "local route unavailable, yaw scan"
                    self.health.warn("planner", "local route unavailable")
            if out.command_type == "velocity" and out.state == State.SEARCH:
                front_block_m = None
                if pose is not None:
                    front_block_m = self.local_map.diagnostics(
                        pose, target_xy).get("front_block_occ_m")
                fields, avoiding = apply_search_approach_brake(
                    fields, self._last_sectors, front_block_m)
                if avoiding:
                    self.health.warn("planner", f"avoid {self._last_sectors.method}")
                    effective_status = f"{effective_status}; avoid {self._last_sectors.method}"
            if out.command_type == "velocity" and out.state == State.APPROACH:
                local_diag = (
                    self.local_map.diagnostics(pose, target_xy)
                    if pose is not None else None
                )
                gate_reason = _approach_gate_reason(self._last_sectors, local_diag)
                if gate_reason is not None:
                    planned = None
                    if pose is not None and target_xy is not None:
                        planned = self._local_route_command(status_snap)
                    if planned is not None:
                        fields = planned.fields
                        effective_status = f"approach gated ({gate_reason}); {planned.status}"
                    else:
                        fields = _approach_block_fields(fields)
                        effective_status = f"approach gated ({gate_reason}); yaw hold"
                    self.health.warn("planner", effective_status[:40])
            self._planner_var.set(effective_status)
            self._send(out.command_type, **fields)

        self._last_planner_state_name = out.state.name
        if pose is not None and target_xy is not None:
            self._last_target_dist_m = math.hypot(
                target_xy[0] - pose.x, target_xy[1] - pose.y)

        if self.logger is not None:
            self.logger.log_tick(now, out.state.name, self.last_detection,
                                 out.command_type, fields if out.send_command else {},
                                 status_snap, effective_status,
                                 self._vision_debug(status_snap))

        if self.reporter is not None:
            try:
                slam_snap = self.slam_detector.get_map_snapshot()
            except Exception:
                slam_snap = (None, None, -1)
            self.reporter.tick(
                pose=pose,
                target_xy=target_xy,
                sectors=self._last_sectors,
                local_map_snap=self.local_map.snapshot(),
                slam_snapshot=slam_snap,
                annotated_frame=self._last_annotated_frame,
                avoiding=avoiding,
            )

        # Heartbeat roughly every second regardless.
        if now - self._last_heartbeat >= 1.0:
            self._send("heartbeat", quiet=True)
            self._last_heartbeat = now

    def _try_start_slam(self) -> None:
        """Start the SLAM detector once the dsim status buffer is available."""
        if self._slam_started or self.slam_detector is None or self.status is None:
            return
        self._slam_started = True
        status_snap = self.status.getAll()
        ok = self.slam_detector.start(status_snap)
        if ok:
            self.health.ok("slam", "initialising")
        else:
            self.health.warn("slam", self.slam_detector.status_text[:28])

    def _local_route_command(self, status_snap: dict):
        pose = pose_from_status(status_snap)
        target_xy = target_xy_from_status(status_snap)
        if pose is None or target_xy is None:
            return None
        return self.local_map.plan_to_target(pose, target_xy)

    def _vision_debug(self, status_snap: dict) -> dict:
        pose = pose_from_status(status_snap)
        target_xy = target_xy_from_status(status_snap) if pose is not None else None
        return {
            "fused": _sectors_debug(self._last_sectors),
            "flow": _sectors_debug(self._last_flow_sectors),
            "slam": _sectors_debug(self._last_slam_sectors),
            "local_map": (
                self.local_map.diagnostics(pose, target_xy)
                if pose is not None else {}
            ),
        }

    def _draw_slam_map(self) -> None:
        """Render the top-down SLAM map onto the canvas each tick."""
        canvas = self._slam_map_canvas
        if canvas is None:
            return

        sz = self._slam_canvas_sz
        canvas.delete("all")
        canvas.create_rectangle(0, 0, sz, sz, fill=_VIDEO_BG, outline="")

        sd = self.slam_detector
        pose, pts, state = sd.get_map_snapshot()
        _labels = {-1: "NOT READY", 0: "NO IMAGES", 1: "INIT",
                    2: "OK", 3: "LOST+", 4: "LOST"}
        label = _labels.get(state, str(state))

        if pts is None or len(pts) == 0 or pose is None:
            canvas.create_text(sz // 2, sz // 2,
                               text=f"SLAM: {label}\nno map data",
                               fill=_FG_DIM, font=("TkDefaultFont", 8), justify="center")
            return

        # Project map points into the CURRENT camera frame so the drone is
        # always centred and obstacles appear relative to it.
        # p_cam = R * p_world + t  (Tcw = [R|t])
        R = pose[:3, :3]
        t = pose[:3, 3]
        disp = pts[:2000] if len(pts) > 2000 else pts
        pts_cam = disp @ R.T + t      # Nx3 in camera frame
        # Keep only forward-hemisphere points (z > 0)
        fwd = pts_cam[:, 2] > 0.05
        pts_cam = pts_cam[fwd]

        # Canvas: X left = camera X (the frame fed to SLAM is horizontally flipped
        # by _client_rgb_frame, so camera X+ is the left of the displayed image).
        # Z forward = camera Z (north = up in canvas).
        mx, mz = pts_cam[:, 0], pts_cam[:, 2]
        drone_xw, drone_zw = 0.0, 0.0   # drone at origin
        heading = 0.0                     # always faces up (north) in canvas

        # Auto-fit view: square window centred on all points + drone.
        all_x  = np.append(mx, drone_xw)
        all_z  = np.append(mz, drone_zw)
        span   = max(float(all_x.max() - all_x.min()), float(all_z.max() - all_z.min()), 1.0)
        pad    = span * 0.18
        cx_w   = float((all_x.min() + all_x.max()) / 2.0)
        cz_w   = float((all_z.min() + all_z.max()) / 2.0)
        half   = span / 2.0 + pad
        x_min, x_max = cx_w - half, cx_w + half
        z_min, z_max = cz_w - half, cz_w + half

        M = 6   # canvas margin pixels
        use = sz - 2 * M

        def w2c(wx: float, wz: float) -> tuple[int, int]:
            px = M + int((1.0 - (wx - x_min) / (x_max - x_min)) * use)
            py = M + int((1.0 - (wz - z_min) / (z_max - z_min)) * use)
            return px, py

        # Grid lines.
        step = _nice_step((x_max - x_min) / 3.5)
        gx = math.ceil(x_min / step) * step
        while gx <= x_max:
            cx_, _ = w2c(gx, z_min);  canvas.create_line(cx_, M, cx_, sz - M, fill="#1c2128")
            gx += step
        gz = math.ceil(z_min / step) * step
        while gz <= z_max:
            _, cy_ = w2c(x_min, gz);  canvas.create_line(M, cy_, sz - M, cy_, fill="#1c2128")
            gz += step

        # Map points coloured by depth (z in camera frame = forward distance).
        d2 = mx ** 2 + mz ** 2
        d90 = float(np.percentile(d2, 90)) if len(d2) > 0 else 1.0
        for i in range(len(mx)):
            px, py = w2c(float(mx[i]), float(mz[i]))
            if not (0 <= px < sz and 0 <= py < sz):
                continue
            norm = min(1.0, d2[i] / max(d90, 1e-9))
            col  = "#f85149" if norm < 0.25 else ("#e09440" if norm < 0.6 else "#388bfd")
            canvas.create_rectangle(px, py, px + 1, py + 1, fill=col, outline="")

        # FOV cone (half-angle 35°).
        dc_x, dc_y = w2c(drone_xw, drone_zw)
        fov_r = math.radians(35.0)
        x_scale = (x_max - x_min) / use
        z_scale = (z_max - z_min) / use
        avg_s   = (x_scale + z_scale) / 2.0
        cone_w  = use * 0.28 * avg_s
        for side in (-1.0, 1.0):
            ang  = heading + fov_r * side
            ex_w = drone_xw + math.sin(ang) * cone_w
            ez_w = drone_zw + math.cos(ang) * cone_w
            ex_c, ey_c = w2c(ex_w, ez_w)
            canvas.create_line(dc_x, dc_y, ex_c, ey_c, fill="#2d3340", width=1)

        # Sector risk arcs (thin wedges just outside the FOV cone).
        sectors = self._last_sectors
        _arc_sectors = [
            (-90.0, -30.0, sectors.left),
            (-30.0, -10.0, sectors.front_left),
            (-10.0,  10.0, sectors.front),
            ( 10.0,  30.0, sectors.front_right),
            ( 30.0,  90.0, sectors.right),
        ]
        arc_r = use * 0.38 * avg_s
        for lo_deg, hi_deg, risk in _arc_sectors:
            if risk < 0.05 or sectors.confidence == 0.0:
                continue
            col = ("#3fb950" if risk < 0.35 else ("#e09440" if risk < 0.65 else "#f85149"))
            n_seg = max(2, int((hi_deg - lo_deg) / 5))
            for seg in range(n_seg):
                t0 = (lo_deg + (hi_deg - lo_deg) * seg / n_seg)
                t1 = (lo_deg + (hi_deg - lo_deg) * (seg + 1) / n_seg)
                a0 = heading + math.radians(t0)
                a1 = heading + math.radians(t1)
                ax0, ay0 = w2c(drone_xw + math.sin(a0) * arc_r,
                               drone_zw + math.cos(a0) * arc_r)
                ax1, ay1 = w2c(drone_xw + math.sin(a1) * arc_r,
                               drone_zw + math.cos(a1) * arc_r)
                canvas.create_line(ax0, ay0, ax1, ay1, fill=col, width=2)

        # Drone triangle.  X offsets are negated to match the flipped X axis in w2c.
        SZ = 7
        tip = (dc_x - int(math.sin(heading) * SZ),
               dc_y - int(math.cos(heading) * SZ))
        lft = (dc_x - int(math.sin(heading - 2.5) * int(SZ * 0.55)),
               dc_y - int(math.cos(heading - 2.5) * int(SZ * 0.55)))
        rgt = (dc_x - int(math.sin(heading + 2.5) * int(SZ * 0.55)),
               dc_y - int(math.cos(heading + 2.5) * int(SZ * 0.55)))
        canvas.create_polygon(*tip, *lft, *rgt, fill=_ACCENT, outline="#c9d1d9", width=1)

        # Labels.
        state_col = (_ACCENT2 if state == 2 else (_FG_DIM if state == 4 else _ACCENT))
        canvas.create_text(4, 3, anchor="nw",
                           text=f"{label}  {sd.n_map_points}pts",
                           fill=state_col, font=("TkFixedFont", 7))
        sc_str = f"{sd.scale:.3f}m/u" if sd.scale else "unscaled"
        canvas.create_text(sz - 3, sz - 3, anchor="se",
                           text=sc_str, fill=_FG_DIM, font=("TkFixedFont", 7))

    def _draw_local_map(self) -> None:
        """Render the local occupancy grid, planned path, target, and drone."""
        canvas = self._local_map_canvas
        if canvas is None:
            return

        sz = self._slam_canvas_sz
        canvas.delete("all")
        canvas.create_rectangle(0, 0, sz, sz, fill=_VIDEO_BG, outline="")

        pose = self._last_route_status.get("pose")
        if pose is None and self.status is not None:
            pose = pose_from_status(self.status.getAll())
        if pose is None:
            canvas.create_text(sz // 2, sz // 2, text="no pose",
                               fill=_FG_DIM, font=("TkDefaultFont", 8))
            return

        target_xy = self._last_route_status.get("target_xy")
        if target_xy is None and self.status is not None:
            target_xy = target_xy_from_status(self.status.getAll())

        snap = self.local_map.snapshot()
        cells = snap["cells"]
        path = snap["path"]
        half = float(snap["half_width_m"])
        cell_m = float(snap["cell_m"])
        M = 6
        use = sz - 2 * M

        def w2c(wx: float, wy: float) -> tuple[int, int]:
            px = M + int((wx - (pose.x - half)) / (2.0 * half) * use)
            py = M + int((wy - (pose.y - half)) / (2.0 * half) * use)
            return px, py

        for off in range(-int(half), int(half) + 1, 2):
            x0, y0 = w2c(pose.x - half, pose.y + off)
            x1, y1 = w2c(pose.x + half, pose.y + off)
            canvas.create_line(x0, y0, x1, y1, fill="#1c2128")
            x0, y0 = w2c(pose.x + off, pose.y - half)
            x1, y1 = w2c(pose.x + off, pose.y + half)
            canvas.create_line(x0, y0, x1, y1, fill="#1c2128")

        cell_px = max(2, int(cell_m / (2.0 * half) * use))
        for (cx, cy), value in cells.items():
            wx = cx * cell_m
            wy = cy * cell_m
            if abs(wx - pose.x) > half or abs(wy - pose.y) > half:
                continue
            px, py = w2c(wx, wy)
            if value > 0.2:
                col = "#f85149" if value >= 1.6 else "#e09440"
            elif value < -0.2:
                col = "#238636"
            else:
                continue
            canvas.create_rectangle(px - cell_px, py - cell_px,
                                    px + cell_px, py + cell_px,
                                    fill=col, outline="")

        if len(path) >= 2:
            pts = []
            for wx, wy in path:
                px, py = w2c(wx, wy)
                pts.extend([px, py])
            canvas.create_line(*pts, fill="#58a6ff", width=2)

        if target_xy is not None:
            tx, ty = w2c(target_xy[0], target_xy[1])
            canvas.create_oval(tx - 4, ty - 4, tx + 4, ty + 4,
                               outline="#ffdf5d", width=2)

        dx, dy = w2c(pose.x, pose.y)
        yaw = math.radians(pose.yaw_deg)
        tip = (dx + int(math.cos(yaw) * 9), dy + int(math.sin(yaw) * 9))
        lft = (dx + int(math.cos(yaw + 2.5) * 5), dy + int(math.sin(yaw + 2.5) * 5))
        rgt = (dx + int(math.cos(yaw - 2.5) * 5), dy + int(math.sin(yaw - 2.5) * 5))
        canvas.create_polygon(*tip, *lft, *rgt, fill=_ACCENT, outline="#c9d1d9")
        canvas.create_text(4, 3, anchor="nw",
                           text=f"path {len(path)}  cells {len(cells)}",
                           fill=_FG_DIM, font=("TkFixedFont", 7))

    def _update_health_ui(self) -> None:
        style_map = {"ok": "HealthOk", "warn": "HealthWarn",
                     "err": "HealthErr", "wait": "HealthWait"}
        label_map = {"ok": "ok  ", "warn": "warn", "err": "ERR ", "wait": "wait"}
        for name, var in self._health_vars.items():
            st = self.health.state(name)
            var.set(label_map.get(st, "wait"))
            self._health_labels[name].configure(
                style=f"{style_map.get(st, 'HealthWait')}.TLabel")
            self._health_color_vars[name].set(self.health.detail(name)[:35])

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    def _send(self, typ: str, quiet: bool = False, **fields) -> None:
        if self.command is None:
            return
        payload = encode_command(typ, **fields)
        ok = self.command.write(payload)
        if not ok and not quiet and self.args.verbose:
            print(f"daic: failed to write command {typ}", file=sys.stderr)

    # ------------------------------------------------------------------
    # UI callbacks
    # ------------------------------------------------------------------

    def _toggle_ai(self) -> None:
        self.ai_enabled = not self.ai_enabled
        if self.ai_enabled:
            self._ai_var.set("AI: ON")
            if self._toggle_btn:
                self._toggle_btn.configure(text="Disable AI")
            status_snap = self.status.getAll() if self.status else {}
            self.planner.enable(status_snap)
        else:
            self._ai_var.set("AI: OFF")
            if self._toggle_btn:
                self._toggle_btn.configure(text="Enable AI")
            out = self.planner.disable()
            self._send(out.command_type)
            self._state_var.set("IDLE")
            self._planner_var.set("—")

    def _emergency_stop(self) -> None:
        self.ai_enabled = False
        self._ai_var.set("AI: OFF")
        if self._toggle_btn:
            self._toggle_btn.configure(text="Enable AI")
        self.planner.disable()
        self._send("zero")
        self._state_var.set("IDLE")
        self._planner_var.set("emergency stop")
        self.health.warn("planner", "emergency stop")

    def close(self) -> None:
        self.running = False
        try:
            self._send("zero", quiet=True)
        finally:
            if self.slam_detector is not None:
                self.slam_detector.stop()
            if self.logger is not None:
                self.logger.close()  # flush flight.jsonl before reporter reads it
            if self.reporter is not None:
                try:
                    crashed = (self.status.getAll().get("drone.crashed", "0") == "1"
                               if self.status else False)
                    self.reporter.close(self._last_planner_state_name,
                                        crashed, self._last_target_dist_m)
                except Exception as exc:
                    print(f"daic: reporter close: {exc}", file=sys.stderr)
            for handle in (self.status, self.command, self.video):
                if handle is not None:
                    handle.close()
            save_window_pos(self.root, f"daic.{self.args.id}")
            try:
                self.root.destroy()
            except tk.TclError:
                pass


# ---------------------------------------------------------------------------
# Headless agent (no Tk — suitable for automated testing)
# ---------------------------------------------------------------------------

class HeadlessAgent:
    """Runs the AI loop without any UI.  Use --no-ui to select this path."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.pymembus = load_pymembus()
        self.names = shared_names(args.id)
        self.video   = None
        self.command = None
        self.status  = None
        self.status_epoch    = 0
        self.last_status_time = 0.0
        self.last_video_seq  = -1
        self.running = True
        self.last_detection  = Detection(False, 0, 0, 0, 0)
        self.planner = Planner(img_w=args.video_w, img_h=args.video_h)
        self._alt_lock = True
        self._last_heartbeat = 0.0
        self._pending_enable_ai = args.enable_ai
        self._auto_log = not bool(args.log_file)
        self.logger: FlightLogger | None = (
            FlightLogger(args.log_file) if args.log_file else None
        )
        self._enable_reporting = args.enable_ai
        self.reporter = None   # RunReporter | None
        self._last_planner_state_name = "IDLE"
        self._last_target_dist_m: float | None = None
        vocab = getattr(args, "slam_vocab", None)
        if vocab:
            self.slam_detector: ORBSLAM3Detector | MiniSLAMDetector = ORBSLAM3Detector(
                vocab_path=vocab,
                settings_path=getattr(args, "slam_settings", None),
            )
        else:
            if MiniSLAMDetector is None:
                raise RuntimeError("opencv-python is required; run ./daic/daic.py --install")
            self.slam_detector = MiniSLAMDetector()
        self.flow_detector = OpticalFlowAvoidance()
        self.local_map = LocalOccupancyMap()
        self._last_slam_sectors = _NULL_SECTORS
        self._last_flow_sectors = _NULL_SECTORS
        self._last_sectors = _NULL_SECTORS
        self._slam_started = False

    def open_missing(self) -> None:
        pm = self.pymembus
        if self.video is None:
            vid = pm.memvid()
            if vid.open_existing(self.names["video"]):
                self.video = vid
        if self.command is None:
            cmd = pm.memcmd()
            if cmd.open(self.names["command"], self.args.cmd_size):
                self.command = cmd
        if self.status is None:
            kv = pm.memkv()
            if kv.open(self.names["status"]):
                self.status = kv
                self.status_epoch = kv.getEpoch()
                self.last_status_time = time.monotonic()
                if self._auto_log and self.logger is None:
                    self._init_logger_from_status()
                self._init_reporter_from_status()

    def _init_logger_from_status(self) -> None:
        if self.status is None or self.logger is not None:
            return
        report_dir = self.status.getAll().get("sim.report_dir", "")
        if report_dir:
            log_path = Path(report_dir) / "daic" / "flight.jsonl"
            self.logger = FlightLogger(log_path)

    def _init_reporter_from_status(self) -> None:
        if self.reporter is not None or not self._enable_reporting:
            return
        if self.status is None:
            return
        report_dir = self.status.getAll().get("sim.report_dir", "")
        if report_dir:
            from daic.run_reporter import RunReporter
            self.reporter = RunReporter(Path(report_dir) / "daic")

    def run(self) -> None:
        frame_period = 1.0 / max(1, self.args.fps)
        if self.args.verbose:
            print(f"daic headless: id={self.args.id}", file=sys.stderr)
        while self.running:
            tick_start = time.monotonic()
            self._tick(tick_start)
            elapsed = time.monotonic() - tick_start
            time.sleep(max(0.0, frame_period - elapsed))

    def _tick(self, now: float) -> None:
        self.open_missing()

        if self._pending_enable_ai and self.command is not None:
            self._pending_enable_ai = False
            status_snap = self.status.getAll() if self.status else {}
            self.planner.enable(status_snap)
            if self.args.verbose:
                print("daic: AI enabled", file=sys.stderr)

        # Read latest video frame and run detector + SLAM.
        if self.video is not None:
            seq = self.video.getSeq()
            if seq != self.last_video_seq and seq > 0:
                self.last_video_seq = seq
                slot  = self.video.getPtr(-1)
                frame = np.array(self.video[slot], copy=False)
                rgb   = _client_rgb_frame(frame)
                try:
                    self.last_detection = detect(rgb)
                except Exception:
                    pass
                if self.slam_detector is not None and self.slam_detector._available:
                    try:
                        self._last_slam_sectors = self.slam_detector.detect_obstacles(rgb)
                    except Exception:
                        pass
                try:
                    if self.status is not None:
                        self.flow_detector.set_motion_from_status(self.status.getAll())
                    self._last_flow_sectors = self.flow_detector.detect_obstacles(rgb)
                    self._last_sectors = fuse_obstacle_sectors(
                        self._last_slam_sectors, self._last_flow_sectors,
                    )
                except Exception:
                    self._last_sectors = self._last_slam_sectors

        # Start SLAM once the status buffer is available.
        if (not self._slam_started
                and self.slam_detector is not None
                and self.status is not None):
            self._slam_started = True
            status_snap = self.status.getAll()
            self.slam_detector.start(status_snap)
            if self.args.verbose:
                print(f"daic: SLAM start: {self.slam_detector.status_text}",
                      file=sys.stderr)

        # Read status.
        status_snap: dict = {}
        if self.status is not None:
            changed, epoch = self.status.getChanged(self.status_epoch)
            if changed:
                self.status_epoch = epoch
                self.last_status_time = now
            status_snap = self.status.getAll()
            if now - self.last_status_time > 2.0:
                status_snap["_stale"] = "1"

        pose = pose_from_status(status_snap)
        if pose is not None:
            self.local_map.update(pose, self._last_sectors)

        # Check if dsim has exited (video buffer gone).
        if self.video is not None:
            try:
                self.video.getSeq()
            except Exception:
                if self.args.verbose:
                    print("daic: video buffer lost, stopping", file=sys.stderr)
                self.running = False
                return

        if self.command is None:
            return

        try:
            out = self.planner.tick(self.last_detection, status_snap)
        except Exception as exc:
            print(f"daic: planner error: {exc}", file=sys.stderr)
            self._send("zero")
            return

        if self.args.verbose and out.status_text:
            print(f"  [{out.state.name}] {out.status_text}", file=sys.stderr)

        target_xy = target_xy_from_status(status_snap)
        avoiding  = False
        effective_status = out.status_text
        if out.send_command:
            fields = dict(out.command_fields)
            if self._alt_lock and out.command_type == "velocity":
                if out.state == State.LANDING:
                    fields["up_mps"] = min(0.0, fields.get("up_mps", 0.0))
                else:
                    fields["up_mps"] = 0.0
            if out.command_type == "velocity" and out.state == State.SEARCH:
                planned = None
                if pose is not None and target_xy is not None:
                    planned = self._local_route_command(status_snap)
                if planned is not None:
                    fields = planned.fields
                    effective_status = planned.status
                elif pose is not None and target_xy is not None:
                    fields = _search_hold_scan_fields()
                    effective_status = "local route unavailable, yaw scan"
            if out.command_type == "velocity" and out.state == State.SEARCH:
                front_block_m = None
                if pose is not None:
                    front_block_m = self.local_map.diagnostics(
                        pose, target_xy).get("front_block_occ_m")
                fields, avoiding = apply_search_approach_brake(
                    fields, self._last_sectors, front_block_m)
                if avoiding:
                    effective_status = f"{effective_status}; avoid {self._last_sectors.method}"
            if out.command_type == "velocity" and out.state == State.APPROACH:
                local_diag = (
                    self.local_map.diagnostics(pose, target_xy)
                    if pose is not None else None
                )
                gate_reason = _approach_gate_reason(self._last_sectors, local_diag)
                if gate_reason is not None:
                    planned = None
                    if pose is not None and target_xy is not None:
                        planned = self._local_route_command(status_snap)
                    if planned is not None:
                        fields = planned.fields
                        effective_status = f"approach gated ({gate_reason}); {planned.status}"
                    else:
                        fields = _approach_block_fields(fields)
                        effective_status = f"approach gated ({gate_reason}); yaw hold"
            self._send(out.command_type, **fields)

        self._last_planner_state_name = out.state.name
        if pose is not None and target_xy is not None:
            self._last_target_dist_m = math.hypot(
                target_xy[0] - pose.x, target_xy[1] - pose.y)

        if self.logger is not None:
            self.logger.log_tick(now, out.state.name, self.last_detection,
                                 out.command_type,
                                 fields if out.send_command else {},
                                 status_snap, effective_status,
                                 self._vision_debug(status_snap))

        if self.reporter is not None:
            try:
                slam_snap = self.slam_detector.get_map_snapshot()
            except Exception:
                slam_snap = (None, None, -1)
            self.reporter.tick(
                pose=pose,
                target_xy=target_xy,
                sectors=self._last_sectors,
                local_map_snap=self.local_map.snapshot(),
                slam_snapshot=slam_snap,
                annotated_frame=None,
                avoiding=avoiding,
            )

        # Heartbeat once per second.
        if now - self._last_heartbeat >= 1.0:
            self._send("heartbeat", quiet=True)
            self._last_heartbeat = now

        # Stop when mission is complete.
        if out.state.name in ("COMPLETE", "FAILSAFE"):
            time.sleep(1.0)
            self.running = False

    def _send(self, typ: str, quiet: bool = False, **fields) -> None:
        if self.command is None:
            return
        payload = encode_command(typ, **fields)
        ok = self.command.write(payload)
        if not ok and not quiet and self.args.verbose:
            print(f"daic: failed to write {typ}", file=sys.stderr)

    def _local_route_command(self, status_snap: dict):
        pose = pose_from_status(status_snap)
        target_xy = target_xy_from_status(status_snap)
        if pose is None or target_xy is None:
            return None
        return self.local_map.plan_to_target(pose, target_xy)

    def _vision_debug(self, status_snap: dict) -> dict:
        pose = pose_from_status(status_snap)
        target_xy = target_xy_from_status(status_snap) if pose is not None else None
        return {
            "fused": _sectors_debug(self._last_sectors),
            "flow": _sectors_debug(self._last_flow_sectors),
            "slam": _sectors_debug(self._last_slam_sectors),
            "local_map": (
                self.local_map.diagnostics(pose, target_xy)
                if pose is not None else {}
            ),
        }

    def close(self) -> None:
        self.running = False
        try:
            self._send("zero", quiet=True)
        finally:
            if self.slam_detector is not None:
                self.slam_detector.stop()
            if self.logger is not None:
                self.logger.close()  # flush flight.jsonl before reporter reads it
            if self.reporter is not None:
                try:
                    crashed = (self.status.getAll().get("drone.crashed", "0") == "1"
                               if self.status else False)
                    self.reporter.close(self._last_planner_state_name,
                                        crashed, self._last_target_dist_m)
                except Exception as exc:
                    print(f"daic: reporter close: {exc}", file=sys.stderr)
            for handle in (self.status, self.command, self.video):
                if handle is not None:
                    try:
                        handle.close()
                    except Exception:
                        pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="dvision2 AI drone controller")
    parser.add_argument("--install",     action="store_true",
                        help="check and install SLAM dependencies, then exit")
    parser.add_argument("--id",          default=None,
                        help="instance id (required unless --install is used)")
    parser.add_argument("--display-w",   type=int, default=960,
                        help="max display width for video (0 = native)")
    parser.add_argument("--display-h",   type=int, default=720,
                        help="max display height for video (0 = native)")
    parser.add_argument("--video-w",     type=int, default=640,
                        help="expected video frame width (for detector/servo)")
    parser.add_argument("--video-h",     type=int, default=480,
                        help="expected video frame height (for detector/servo)")
    parser.add_argument("--fps",         type=int, default=30)
    parser.add_argument("--cmd-size",    type=int, default=65536)
    parser.add_argument("--enable-ai",   action="store_true",
                        help="enable AI control immediately on startup")
    parser.add_argument("--no-ui",       action="store_true",
                        help="headless mode — no Tk window, suitable for automated tests")
    parser.add_argument("--log-file",      default=None,
                        help="write structured JSONL flight log to this path")
    parser.add_argument("--slam-vocab",   default=None,
                        help="path to ORBvoc.txt; enables ORB_SLAM3 obstacle detection")
    parser.add_argument("--slam-settings", default=None,
                        help="path to a pre-written ORB_SLAM3 YAML; generated from "
                             "dsim camera params when omitted")
    parser.add_argument("--verbose",       action="store_true")
    args = parser.parse_args(argv)
    if not args.install:
        if not args.id:
            parser.error("--id is required (or use --install)")
        validate_id(args.id)
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    if args.install:
        return _cmd_install(verbose=args.verbose)

    if args.no_ui:
        agent: HeadlessAgent | DaicController = HeadlessAgent(args)
    else:
        agent = DaicController(args)

    def stop(_sig, _frame):
        agent.close()

    signal.signal(signal.SIGINT,  stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        agent.run()
    except Exception as exc:
        print(f"daic: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
