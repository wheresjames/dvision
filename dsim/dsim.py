#!/usr/bin/env python3

from __future__ import annotations

import argparse
import datetime
import math
import random
import signal
import sys
import time
import tkinter as tk
import uuid
from dataclasses import dataclass
from pathlib import Path
from tkinter import ttk

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dvision2_common import (
    BERLIN_CENTER_ALT_M,
    BERLIN_CENTER_LAT_DEG,
    BERLIN_CENTER_LON_DEG,
    STATUS_KEYS,
    SimMap,
    clamp,
    decode_command,
    load_map,
    load_pymembus,
    memkv_aligned_name_len,
    local_to_gps,
    restore_window_pos,
    save_window_pos,
    shared_names,
    validate_id,
)


# ---------------------------------------------------------------------------
# UI theme
# ---------------------------------------------------------------------------

_UI_BG = "#0d1117"
_UI_PANEL = "#161b22"
_UI_CANVAS = "#010409"
_UI_CELL = "#161b22"
_UI_GRID = "#30363d"
_UI_TEXT = "#e6edf3"
_UI_DIM = "#8b949e"
_UI_ACCENT = "#58a6ff"
_UI_DANGER = "#f85149"
_UI_BUTTON = "#21262d"
_UI_BUTTON_ACTIVE = "#30363d"


# ---------------------------------------------------------------------------
# Panda3D renderer
# ---------------------------------------------------------------------------

class Panda3DRenderer:
    """Offscreen drone-camera renderer using Panda3D.

    Coordinate mapping (map → Panda3D world):
      map X  →  world X   (east)
      map Y  →  world Y   (south, row-positive)
      map Z  →  world Z   (up)

    Heading conversion:
      panda_H = (90 - sim_yaw_deg) % 360
        sim yaw 0°   (east  +X) → H=90
        sim yaw 90°  (south +Y) → H=0
        sim yaw 270° (north -Y) → H=180
    """

    WALL_H       = 2.5
    TREE_TRUNK_H = 1.6
    TREE_TRUNK_W = 0.22
    TREE_CROWN_R = 0.60
    TREE_MODEL_H = 4.5
    TARGET_R = 0.36
    CAM_Z_OFFSET  = 0.1
    CAM_PITCH     = -5.0   # slight forward-down tilt
    CAM_FOV_H     = 70.0   # horizontal field of view, degrees
    GROUND_TEXTURE_M = 2.1

    def __init__(self, sim_map: SimMap, width: int, height: int) -> None:
        from panda3d.core import loadPrcFileData
        loadPrcFileData("", "\n".join([
            "window-type offscreen",
            "audio-library-name null",
            f"win-size {width} {height}",
            "sync-video false",
            "framebuffer-hardware true",
            "show-frame-rate-meter false",
        ]))

        from direct.showbase.ShowBase import ShowBase
        from panda3d.core import (
            AmbientLight, DirectionalLight, Fog,
            GraphicsOutput, LColor, Texture,
        )

        self._width = width
        self._height = height
        self._ground_node = None

        self._base = ShowBase()
        self._base.disableMouse()
        self._base.setBackgroundColor(0.42, 0.62, 0.80, 1)

        lens = self._base.camLens
        lens.setFov(self.CAM_FOV_H)
        lens.setNear(0.15)
        lens.setFar(150.0)
        self._far_m = 150.0

        # Pixel readback texture
        self._tex = Texture()
        self._base.win.addRenderTexture(self._tex, GraphicsOutput.RTMCopyRam)

        # Lights
        amb = AmbientLight("amb")
        amb.setColor(LColor(0.38, 0.38, 0.42, 1))
        self._base.render.setLight(self._base.render.attachNewNode(amb))

        sun = DirectionalLight("sun")
        sun.setColor(LColor(1.0, 0.94, 0.82, 1))
        sun_np = self._base.render.attachNewNode(sun)
        sun_np.setHpr(50, -55, 0)
        self._base.render.setLight(sun_np)

        # Distance fog (matches sky colour)
        fog = Fog("atmos")
        fog.setColor(0.42, 0.62, 0.80)
        fog.setLinearRange(25, 80)
        self._base.render.setFog(fog)

        # Pre-load prototype models for instancing
        self._proto_box    = self._base.loader.loadModel("models/box")
        self._proto_sphere = self._base.loader.loadModel("models/misc/sphere")
        self._wall_texture = self._load_texture(
            ROOT / "dsim/assets/textures/Bricks042/Bricks042_1K-JPG_Color.jpg"
        )
        self._ground_texture = self._load_texture(
            ROOT / "dsim/assets/textures/Ground037/Ground037_1K-JPG_Color.jpg"
        )
        self._tree_models = self._load_tree_models(
            ROOT / "dsim/assets/models/trees"
        )

        self._build_scene(sim_map)

    # ------------------------------------------------------------------
    # Scene helpers
    # ------------------------------------------------------------------

    def _box(self, cx: float, cy: float, bz: float,
             w: float, d: float, h: float, color: tuple,
             texture=None) -> None:
        node = self._proto_box.copyTo(self._base.render)
        node.setScale(w, d, h)
        node.setPos(cx - w * 0.5, cy - d * 0.5, bz)
        node.setColor(*color, 1)
        if texture is not None:
            node.setTexture(texture, 1)

    def _load_texture(self, path: Path):
        if not path.exists():
            return None
        tex = self._base.loader.loadTexture(str(path))
        tex.setWrapU(tex.WM_repeat)
        tex.setWrapV(tex.WM_repeat)
        return tex

    def _sphere(self, cx: float, cy: float, cz: float,
                r: float, color: tuple) -> None:
        node = self._proto_sphere.copyTo(self._base.render)
        node.setScale(r)
        node.setPos(cx, cy, cz)
        node.setColor(*color, 1)

    def _ground(self, cx: float, cy: float, size: float) -> None:
        from panda3d.core import CardMaker, TextureStage

        card = CardMaker("ground")
        half = size * 0.5
        card.setFrame(-half, half, -half, half)
        node = self._base.render.attachNewNode(card.generate())
        node.setPos(cx, cy, -0.01)
        node.setP(-90.0)
        node.setColor(0.26, 0.37, 0.20, 1)
        node.setTwoSided(True)
        node.setFogOff()
        if self._ground_texture is not None:
            stage = TextureStage.getDefault()
            node.setTexture(stage, self._ground_texture, 1)
            repeats = size / self.GROUND_TEXTURE_M
            node.setTexScale(stage, repeats, repeats)
        self._ground_node = node

    def _target_marker(self, cx: float, cy: float) -> None:
        from panda3d.core import (
            Geom, GeomNode, GeomTriangles, GeomVertexData, GeomVertexFormat,
            GeomVertexWriter,
        )

        def make_disc(name: str, r: float, z: float, color: tuple[float, float, float]) -> None:
            segments = 48
            data = GeomVertexData(name, GeomVertexFormat.getV3(), Geom.UHStatic)
            vertices = GeomVertexWriter(data, "vertex")
            vertices.addData3(0.0, 0.0, 0.0)
            for i in range(segments):
                angle = math.tau * i / segments
                vertices.addData3(math.cos(angle) * r, math.sin(angle) * r, 0.0)

            tris = GeomTriangles(Geom.UHStatic)
            for i in range(segments):
                tris.addVertices(0, i + 1, 1 + ((i + 1) % segments))
            tris.closePrimitive()

            geom = Geom(data)
            geom.addPrimitive(tris)
            geom_node = GeomNode(name)
            geom_node.addGeom(geom)
            node = self._base.render.attachNewNode(geom_node)
            node.setPos(cx, cy, z)
            node.setColor(*color, 1)

        make_disc("target-red", self.TARGET_R, 0.002, (0.88, 0.14, 0.10))
        make_disc("target-centre", self.TARGET_R * 0.22, 0.004, (1.0, 1.0, 1.0))

    def _load_tree_models(self, path: Path) -> list:
        if not path.exists():
            return []
        models = []
        for model_path in sorted(path.iterdir()):
            if model_path.suffix.lower() not in (".bam", ".egg", ".glb", ".gltf", ".obj"):
                continue
            try:
                model = self._base.loader.loadModel(str(model_path))
                model.setTwoSided(True)
                model.clearModelNodes()
                model.flattenLight()
                models.append(model)
            except Exception as exc:
                print(f"dsim: failed to load tree model {model_path}: {exc}",
                      file=sys.stderr)
        return models

    def _tree_model_transform(self, model) -> tuple[float, float, float, float, float]:
        bounds = model.getTightBounds()
        if bounds is None:
            return 1.0, 0.0, 0.0, 0.0, 0.0
        lo, hi = bounds
        sx = max(hi.x - lo.x, 0.001)
        sy = max(hi.y - lo.y, 0.001)
        sz = max(hi.z - lo.z, 0.001)
        if sy > sx and sy > sz:
            scale = self.TREE_MODEL_H / sy
            return (
                scale,
                -(lo.x + hi.x) * 0.5 * scale,
                (lo.z + hi.z) * 0.5 * scale,
                -lo.y * scale,
                90.0,
            )
        scale = self.TREE_MODEL_H / sz
        return (
            scale,
            -(lo.x + hi.x) * 0.5 * scale,
            -(lo.y + hi.y) * 0.5 * scale,
            -lo.z * scale,
            0.0,
        )

    def _model_tree(self, cx: float, cy: float, model, heading_deg: float) -> None:
        scale, offset_x, offset_y, offset_z, pitch_deg = self._tree_model_transform(model)
        parent = self._base.render.attachNewNode("tree")
        parent.setPos(cx, cy, 0)
        parent.setH(heading_deg)
        node = model.copyTo(parent)
        node.setScale(scale)
        node.setP(pitch_deg)
        node.setPos(offset_x, offset_y, offset_z)

    def _fallback_tree(self, cx: float, cy: float) -> None:
        self._box(cx, cy, 0,
                  self.TREE_TRUNK_W, self.TREE_TRUNK_W, self.TREE_TRUNK_H,
                  (0.35, 0.21, 0.09))
        self._sphere(cx, cy,
                     self.TREE_TRUNK_H + self.TREE_CROWN_R * 0.6,
                     self.TREE_CROWN_R, (0.16, 0.54, 0.20))

    def _build_scene(self, sim_map: SimMap) -> None:
        ground_size = max(sim_map.width, sim_map.height) + self._far_m * 2.0
        self._ground(sim_map.width / 2, sim_map.height / 2, ground_size)

        for obj in sim_map.objects:
            x, y = obj.x, obj.y
            if obj.kind == "wall":
                self._box(x, y, 0.0, 1.0, 1.0, self.WALL_H,
                          (1.0, 1.0, 1.0), self._wall_texture)
            elif obj.kind == "tree":
                if self._tree_models:
                    seed = f"{sim_map.path}:{x:.3f}:{y:.3f}"
                    rng = random.Random(seed)
                    model = rng.choice(self._tree_models)
                    self._model_tree(x, y, model, rng.uniform(0.0, 360.0))
                else:
                    self._fallback_tree(x, y)
            elif obj.kind == "target":
                self._target_marker(x, y)

    # ------------------------------------------------------------------
    # Per-frame render
    # ------------------------------------------------------------------

    def render(self, state: DroneState, out_frame: np.ndarray) -> None:
        panda_h = (90.0 - state.yaw_deg) % 360.0
        cam = self._base.camera
        cam.setPos(state.x, state.y, state.z + self.CAM_Z_OFFSET)
        cam.setHpr(panda_h, self.CAM_PITCH - state.pitch_deg, state.roll_deg)

        self._base.graphicsEngine.renderFrame()

        if self._tex.hasRamImage():
            data = self._tex.getRamImageAs("RGB")
            arr  = np.frombuffer(bytes(data), dtype=np.uint8)
            out_frame[:] = arr.reshape((self._height, self._width, 3))[::-1, ::-1]

    def close(self) -> None:
        try:
            self._base.destroy()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Physics tuning constants
# ---------------------------------------------------------------------------

# First-order time constants (seconds).  Larger = more sluggish.
_TAU_H   = 0.30   # horizontal velocity response
_TAU_V   = 0.35   # vertical velocity response
_TAU_YAW = 0.10   # yaw-rate response
_TAU_ATT = 0.14   # visual attitude (roll/pitch tilt) lag

# Camera tilt: degrees of lean per m/s of body-frame velocity
_TILT_GAIN = 9.0
_MAX_TILT  = 28.0   # degrees

# Velocity scale applied to incoming commands (reduce to slow the simulated drone)
_SPEED_SCALE = 0.1

# Obstacles occupy their full map cell.  Map object centres are at x/y + 0.5,
# so a half extent of 0.5 makes adjacent wall cells touch with no seam.
_OBSTACLE_HALF_EXTENT_M = 0.5
_COLLISION_SWEEP_STEP_M = 0.1


# ---------------------------------------------------------------------------
# Drone state
# ---------------------------------------------------------------------------

@dataclass
class DroneState:
    x: float
    y: float
    z: float
    roll_deg: float = 0.0
    pitch_deg: float = 0.0
    yaw_deg: float = 270.0
    # Actual world-frame velocities (physics output)
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    yaw_rate: float = 0.0
    # Pilot setpoints in body frame (physics input, set by commands)
    cmd_forward: float = 0.0
    cmd_right: float = 0.0
    cmd_up: float = 0.0
    cmd_yaw_rate: float = 0.0
    target_alt: float | None = None
    armed: bool = False
    mode: str = "DISARMED"
    battery_pct: float = 100.0
    crashed: bool = False
    last_command_monotonic: float | None = None
    last_command_type: str = ""
    command_count: int = 0
    status_message: str = "ready"


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

class DroneSimulator:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.pymembus = load_pymembus()
        self.names = shared_names(args.id)
        self.map = load_map(Path(args.map))
        self.target_x, self.target_y = self._find_target_position()
        start_alt = args.start_alt
        if args.start_alt is None:
            start_alt = float(self.map.data.get("drone-height", 1.5))
        self.start_x = self.map.start_x
        self.start_y = self.map.start_y
        self.start_alt = start_alt
        self.start_yaw = 270.0
        self.state = DroneState(self.map.start_x, self.map.start_y, start_alt)
        self.running = True
        self.started = time.monotonic()
        self.video = None
        self.command = None
        self.status = None
        self.ui = None
        self.p3d: Panda3DRenderer | None = None

        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_id = f"{ts}-{uuid.uuid4().hex[:8]}"
        if getattr(args, "report_dir", None):
            report_root = Path(args.report_dir)
            if not report_root.is_absolute():
                report_root = ROOT / report_root
            self.report_root = report_root
            self.run_id = report_root.name
        else:
            self.report_root = ROOT / "reports" / self.run_id
        self.dsim_report_dir = self.report_root / "dsim"
        self.dsim_report_dir.mkdir(parents=True, exist_ok=True)

        self.flight_positions: list[tuple[float, float, float]] = []  # (x, y, elapsed_s)
        self.crash_pos: tuple[float, float] | None = None

    def _find_target_position(self) -> tuple[float | None, float | None]:
        for obj in self.map.objects:
            if obj.kind == "target":
                return obj.x, obj.y
        return None, None

    def map_to_gps(self, x_m: float, y_m: float, z_m: float) -> tuple[float, float, float]:
        """Convert map coordinates to GPS with the map centered on the origin.

        Map X is east-positive. Map Y is row-positive/south-positive, so it is
        inverted before passing north/east offsets into local_to_gps().
        """
        center_x = float(getattr(self.map, "width", 0.0)) / 2.0
        center_y = float(getattr(self.map, "height", 0.0)) / 2.0
        east_m = x_m - center_x
        north_m = center_y - y_m
        return local_to_gps(
            east_m,
            north_m,
            z_m,
            self.args.origin_lat,
            self.args.origin_lon,
            self.args.origin_alt,
        )

    def _init_renderer(self) -> None:
        self.p3d = Panda3DRenderer(self.map, self.args.width, self.args.height)
        if self.args.verbose:
            print("dsim: using Panda3D renderer")

    def open_ipc(self) -> None:
        pm = self.pymembus
        pm.memvid.remove(self.names["video"])
        pm.memcmd.remove(self.names["command"])
        pm.memkv.remove(self.names["status"])

        self.video = pm.memvid()
        fmt = getattr(pm.video_format, "rgb24", 24)
        if not self.video.open(self.names["video"], True, self.args.width,
                               self.args.height, fmt, self.args.fps, self.args.bufs):
            raise RuntimeError(
                f"failed to create video buffer {self.names['video']}: "
                f"{pm.last_error_message()}"
            )

        self.command = pm.memcmd()
        if not self.command.open(self.names["command"], self.args.cmd_size, True, True):
            raise RuntimeError(
                f"failed to create command buffer {self.names['command']}: "
                f"{pm.last_error_message()}"
            )

        self.status = pm.memkv()
        max_value_len = 512
        max_name_len = memkv_aligned_name_len(
            max(len(k) for k in STATUS_KEYS) + 1, max_value_len
        )
        if not self.status.create(self.names["status"], len(STATUS_KEYS),
                                  max_name_len, max_value_len, True):
            raise RuntimeError(
                f"failed to create status buffer {self.names['status']}: "
                f"{pm.last_error_message()}"
            )
        for idx, key in enumerate(STATUS_KEYS):
            if not self.status.setName(idx, key):
                raise RuntimeError(
                    f"failed to define status key {key}: {pm.last_error_message()}"
                )
        self.publish_status()

    def close(self) -> None:
        if self.flight_positions:
            try:
                self._save_flight_image(self.dsim_report_dir / "flight_path.png")
            except Exception as exc:
                print(f"dsim: flight image error: {exc}", file=sys.stderr)
        if self.ui is not None:
            self.ui.close()
        if self.p3d is not None:
            self.p3d.close()
        for handle in (self.status, self.command, self.video):
            if handle is not None:
                handle.close()

    def run(self) -> None:
        self._init_renderer()
        self.open_ipc()
        if not self.args.no_ui:
            try:
                self.ui = TopDownUi(self)
            except tk.TclError as exc:
                raise RuntimeError(
                    f"failed to open dsim UI; use --no-ui for headless mode: {exc}"
                ) from exc
        if self.args.verbose:
            print(f"dsim {self.args.id}: map={self.map.path} "
                  f"size={self.map.width}x{self.map.height}")
            print(f"dsim {self.args.id}: video={self.names['video']} "
                  f"command={self.names['command']} status={self.names['status']}")

        frame_period = 1.0 / max(1, self.args.fps)
        last = time.monotonic()
        frames_left = self.args.frames
        try:
            while self.running:
                now = time.monotonic()
                dt  = clamp(now - last, 0.001, 0.1)
                last = now
                self.drain_commands()
                self.integrate(dt)
                self.flight_positions.append(
                    (self.state.x, self.state.y, now - self.started))
                self.publish_frame(now)
                self.publish_status()
                if self.ui is not None:
                    self.ui.update()
                    if self.ui.closed:
                        self.running = False
                if frames_left is not None:
                    frames_left -= 1
                    if frames_left <= 0:
                        break
                elapsed = time.monotonic() - now
                time.sleep(max(0.0, frame_period - elapsed))
        finally:
            self.close()

    def drain_commands(self) -> None:
        if self.command is None:
            return
        while self.command.poll():
            raw, overrun = self.command.read_with_overrun(0)
            if overrun:
                self.state.status_message = "command overrun"
                continue
            payload = decode_command(raw)
            if payload is None:
                self.state.status_message = "ignored unsupported command payload"
                continue
            self.apply_command(payload)

    def apply_command(self, payload: dict) -> None:
        typ = payload["type"]
        self.state.last_command_monotonic = time.monotonic()
        self.state.last_command_type = typ
        self.state.command_count += 1

        if typ == "reset":
            self.reset_drone()
            return
        if self.state.crashed:
            self.state.status_message = "crashed"
            return
        self.state.status_message = "ok"

        if typ == "heartbeat":
            return
        if typ == "arm":
            self.state.armed = bool(payload.get("armed", True))
            self.state.mode = "GUIDED" if self.state.armed else "DISARMED"
            if not self.state.armed:
                self.zero_motion()
            return
        if typ == "takeoff":
            if self.state.armed:
                self.state.target_alt = max(float(payload.get("alt_m", 3.0)), 0.5)
                self.state.mode = "TAKEOFF"
            return
        if typ == "land":
            self.state.target_alt = 0.0
            self.state.mode = "LAND"
            # Clear horizontal setpoints so the drone descends in place
            self.state.cmd_forward = 0.0
            self.state.cmd_right   = 0.0
            return
        if typ == "zero":
            self.state.cmd_forward   = 0.0
            self.state.cmd_right     = 0.0
            self.state.cmd_up        = 0.0
            self.state.cmd_yaw_rate  = 0.0
            self.state.mode = "HOLD" if self.state.armed else self.state.mode
            return
        if typ == "velocity":
            if not self.state.armed:
                self.state.cmd_forward = self.state.cmd_right = 0.0
                self.state.cmd_up = self.state.cmd_yaw_rate = 0.0
                return
            self.state.cmd_forward  = float(payload.get("forward_mps", 0.0)) * _SPEED_SCALE
            self.state.cmd_right    = float(payload.get("right_mps",   0.0)) * _SPEED_SCALE
            self.state.cmd_up       = float(payload.get("up_mps",      0.0))
            self.state.cmd_yaw_rate = float(payload.get("yaw_rate_dps", 0.0))
            self.state.mode         = "GUIDED"
            self.state.target_alt   = None

    def reset_drone(self) -> None:
        self.state = DroneState(self.start_x, self.start_y, self.start_alt,
                                yaw_deg=self.start_yaw,
                                status_message="reset")

    def crash(self) -> None:
        st = self.state
        if self.crash_pos is None:
            self.crash_pos = (st.x, st.y)
        self.zero_motion()
        st.target_alt = None
        st.armed = False
        st.mode = "CRASHED"
        st.crashed = True
        st.status_message = "crashed"

    def zero_motion(self) -> None:
        st = self.state
        st.vx = st.vy = st.vz = st.yaw_rate = 0.0
        st.cmd_forward = st.cmd_right = st.cmd_up = st.cmd_yaw_rate = 0.0

    def integrate(self, dt: float) -> None:
        st = self.state
        if st.crashed:
            self.zero_motion()
            st.mode = "CRASHED"
            st.status_message = "crashed"
            return

        # ── Altitude target (takeoff / land) ──────────────────────────────
        # Drives cmd_up toward a velocity that smoothly approaches target_alt.
        if st.target_alt is not None:
            dz = st.target_alt - st.z
            if abs(dz) < 0.04 and abs(st.vz) < 0.06:
                # Settled at target
                st.z = st.target_alt
                st.vz = 0.0
                st.cmd_up = 0.0
                st.target_alt = None
                if st.mode == "TAKEOFF":
                    st.mode = "GUIDED"
                elif st.mode == "LAND":
                    st.armed = False
                    st.mode  = "DISARMED"
                    self.zero_motion()
                    return
            else:
                # Proportional approach speed, clamped so we don't overshoot
                st.cmd_up = clamp(dz * 2.5, -2.0, 2.0)

        # ── Yaw ───────────────────────────────────────────────────────────
        alpha_yaw  = 1.0 - math.exp(-dt / _TAU_YAW)
        st.yaw_rate += (st.cmd_yaw_rate - st.yaw_rate) * alpha_yaw
        st.yaw_deg  = (st.yaw_deg + st.yaw_rate * dt) % 360.0

        yaw_rad = math.radians(st.yaw_deg)
        cos_y   = math.sin(yaw_rad)
        sin_y   = math.cos(yaw_rad)

        # ── Horizontal velocity (first-order lag in body frame) ───────────
        # Decompose world velocity into body-frame components.
        # v_fwd   =  st.vx * cos_y + st.vy * sin_y
        # v_right = -st.vx * sin_y + st.vy * cos_y
        v_fwd   = -st.vx * sin_y + st.vy * cos_y
        v_right =  st.vx * cos_y + st.vy * sin_y

        # Drive each toward its setpoint with a lag filter.
        alpha_h  = 1.0 - math.exp(-dt / _TAU_H)
        v_fwd   += (st.cmd_forward - v_fwd)   * alpha_h
        v_right += (-st.cmd_right   - v_right) * alpha_h

        # Rotate back to world frame using the *updated* yaw.
        # st.vx = v_fwd * cos_y - v_right * sin_y
        # st.vy = v_fwd * sin_y + v_right * cos_y
        st.vx = v_right * cos_y - v_fwd * sin_y
        st.vy = v_right * sin_y + v_fwd * cos_y

        # ── Vertical velocity ─────────────────────────────────────────────
        alpha_v = 1.0 - math.exp(-dt / _TAU_V)
        st.vz  += (st.cmd_up - st.vz) * alpha_v

        # ── Visual attitude (roll / pitch for renderer) ───────────────────
        # Target tilt is proportional to actual body-frame velocity so the
        # camera leans into the motion naturally and levels out on hover.
        target_pitch = clamp(v_fwd    * _TILT_GAIN, -_MAX_TILT, _MAX_TILT)
        target_roll  = clamp(v_right  * _TILT_GAIN, -_MAX_TILT, _MAX_TILT)
        alpha_att    = 1.0 - math.exp(-dt / _TAU_ATT)
        st.pitch_deg += (target_pitch - st.pitch_deg) * alpha_att
        st.roll_deg  += (target_roll  - st.roll_deg)  * alpha_att

        # ── Position integration ──────────────────────────────────────────
        next_x = st.x + st.vx * dt
        next_y = st.y + st.vy * dt
        if self.path_blocked(st.x, st.y, next_x, next_y):
            self.crash()
            return
        st.x = clamp(next_x, 0.1, self.map.width  - 0.1)
        st.y = clamp(next_y, 0.1, self.map.height - 0.1)

        st.z = max(0.0, st.z + st.vz * dt)
        if st.z <= 0.0:
            st.vz = max(0.0, st.vz)   # no downward push through the ground
            if st.mode == "LAND":
                st.armed = False
                st.mode  = "DISARMED"
                st.target_alt = None
                self.zero_motion()
                return

        if st.armed:
            st.battery_pct = max(0.0, st.battery_pct - dt * 0.01)

    def is_blocked(self, x: float, y: float) -> bool:
        for obj in self.map.objects:
            if (obj.kind in ("wall", "tree")
                    and abs(obj.x - x) <= _OBSTACLE_HALF_EXTENT_M
                    and abs(obj.y - y) <= _OBSTACLE_HALF_EXTENT_M):
                return True
        return False

    def path_blocked(self, x0: float, y0: float, x1: float, y1: float) -> bool:
        dist = math.hypot(x1 - x0, y1 - y0)
        steps = max(1, math.ceil(dist / _COLLISION_SWEEP_STEP_M))
        for i in range(1, steps + 1):
            t = i / steps
            x = x0 + (x1 - x0) * t
            y = y0 + (y1 - y0) * t
            if self.is_blocked(x, y):
                return True
        return False

    def publish_status(self) -> None:
        if self.status is None:
            return
        st = self.state
        lat, lon, alt = self.map_to_gps(st.x, st.y, st.z)
        target_x = getattr(self, "target_x", None)
        target_y = getattr(self, "target_y", None)
        if target_x is not None and target_y is not None:
            target_lat, target_lon, target_alt = self.map_to_gps(target_x, target_y, 0.0)
            target_lat_s = f"{target_lat:.7f}"
            target_lon_s = f"{target_lon:.7f}"
            target_alt_s = f"{target_alt:.3f}"
        else:
            target_lat_s = target_lon_s = target_alt_s = ""
        speed    = math.sqrt(st.vx ** 2 + st.vy ** 2 + st.vz ** 2)
        last_cmd = (-1.0 if st.last_command_monotonic is None
                    else time.monotonic() - st.last_command_monotonic)
        cam_fov_h = Panda3DRenderer.CAM_FOV_H
        cam_w     = self.args.width
        cam_h     = self.args.height
        _half_tan = math.tan(math.radians(cam_fov_h / 2.0))
        cam_fx    = cam_w / (2.0 * _half_tan)
        cam_fov_v = math.degrees(2.0 * math.atan(_half_tan * cam_h / cam_w))
        values = {
            "sim.id":                self.args.id,
            "sim.map":               str(self.map.path),
            "sim.time_s":            f"{time.monotonic() - self.started:.3f}",
            "sim.report_dir":        str(self.report_root),
            "drone.armed":           "1" if st.armed else "0",
            "drone.mode":            st.mode,
            "drone.x_m":             f"{st.x:.3f}",
            "drone.y_m":             f"{st.y:.3f}",
            "drone.z_m":             f"{st.z:.3f}",
            "drone.lat_deg":         f"{lat:.7f}",
            "drone.lon_deg":         f"{lon:.7f}",
            "drone.alt_m":           f"{alt:.3f}",
            "target.lat_deg":        target_lat_s,
            "target.lon_deg":        target_lon_s,
            "target.alt_m":          target_alt_s,
            "drone.roll_deg":        f"{st.roll_deg:.2f}",
            "drone.pitch_deg":       f"{st.pitch_deg:.2f}",
            "drone.heading_deg":     f"{(270.0 - st.yaw_deg) % 360.0:.2f}",
            "drone.compass_deg":     f"{(270.0 - st.yaw_deg) % 360.0:.2f}",
            "drone.vx_mps":          f"{st.vx:.3f}",
            "drone.vy_mps":          f"{st.vy:.3f}",
            "drone.vz_mps":          f"{st.vz:.3f}",
            "drone.speed_mps":       f"{speed:.3f}",
            "drone.battery_pct":     f"{st.battery_pct:.1f}",
            "drone.crashed":         "1" if st.crashed else "0",
            "drone.last_command_s":  f"{last_cmd:.3f}",
            "link.command_count":    str(st.command_count),
            "link.last_command_type": st.last_command_type,
            "status.message":        st.status_message,
            "camera.fov_h_deg":      f"{cam_fov_h:.4f}",
            "camera.fov_v_deg":      f"{cam_fov_v:.4f}",
            "camera.pitch_deg":      f"{abs(Panda3DRenderer.CAM_PITCH):.4f}",
            "camera.fx_px":          f"{cam_fx:.4f}",
            "camera.fy_px":          f"{cam_fx:.4f}",
            "camera.cx_px":          f"{cam_w / 2.0:.4f}",
            "camera.cy_px":          f"{cam_h / 2.0:.4f}",
            "camera.width_px":       str(cam_w),
            "camera.height_px":      str(cam_h),
            "camera.fps":            str(self.args.fps),
        }
        self.status.setAll(values)

    def publish_frame(self, now: float) -> None:
        if self.video is None:
            return
        slot  = self.video.getPtr(0)
        frame = np.array(self.video[slot], copy=False)
        if self.p3d is None:
            raise RuntimeError("Panda3D renderer is not initialized")
        self.p3d.render(self.state, frame)
        pts = int(now * 1_000_000)
        self.video.setVpts(slot, pts)
        self.video.setApts(slot, pts)
        self.video.next(1)

    def save_snapshot(self) -> None:
        ts = datetime.datetime.now().strftime("%H%M%S")
        path = self.dsim_report_dir / f"snapshot_{ts}.png"
        try:
            self._save_flight_image(path)
        except Exception as exc:
            print(f"dsim: snapshot error: {exc}", file=sys.stderr)

    def _save_flight_image(self, path: Path) -> None:
        try:
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            from matplotlib.collections import LineCollection
            from matplotlib.patches import Circle, Rectangle
        except ImportError:
            print("dsim: matplotlib not installed — skipping flight path image",
                  file=sys.stderr)
            return

        sim_map = self.map
        aspect = sim_map.height / max(sim_map.width, 1)
        fig = Figure(figsize=(10, max(4.0, 10.0 * aspect)), facecolor="#0d1117")
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot(111, facecolor="#161b22")
        ax.set_xlim(0, sim_map.width)
        ax.set_ylim(sim_map.height, 0)  # row 0 at top, matches the UI
        ax.set_aspect("equal")
        ax.tick_params(colors="#8b949e")
        for sp in ax.spines.values():
            sp.set_color("#30363d")
        ax.set_xlabel("X (m)", color="#8b949e")
        ax.set_ylabel("Y (m)", color="#8b949e")

        # Grid
        for x in range(sim_map.width + 1):
            ax.axvline(x, color="#21262d", linewidth=0.4, zorder=0)
        for y in range(sim_map.height + 1):
            ax.axhline(y, color="#21262d", linewidth=0.4, zorder=0)

        # Map objects
        for obj in sim_map.objects:
            if obj.kind == "wall":
                rect = Rectangle((obj.x - 0.5, obj.y - 0.5), 1.0, 1.0,
                                  facecolor="#6e7681", edgecolor="#8b949e",
                                  linewidth=0.5, zorder=1)
                ax.add_patch(rect)
            elif obj.kind == "tree":
                circ = Circle((obj.x, obj.y), 0.36,
                               facecolor="#238636", edgecolor="#3fb950",
                               linewidth=0.5, zorder=1)
                ax.add_patch(circ)
            elif obj.kind == "target":
                circ = Circle((obj.x, obj.y), 0.34,
                               facecolor="#da3633", edgecolor="#f85149",
                               linewidth=1.0, zorder=2)
                ax.add_patch(circ)
                r = 0.34
                ax.plot([obj.x - r, obj.x + r], [obj.y, obj.y],
                        "w-", linewidth=1.0, zorder=3)
                ax.plot([obj.x, obj.x], [obj.y - r, obj.y + r],
                        "w-", linewidth=1.0, zorder=3)

        # Flight path
        positions = list(self.flight_positions)
        if len(positions) >= 2:
            # Collect 10-second tick marks from full position list before subsampling.
            tick_interval = 10.0
            next_tick = tick_interval
            tick_marks: list[tuple[int, float, float, float]] = []  # (idx, x, y, t)
            for i, (px, py, pt) in enumerate(positions):
                if pt >= next_tick:
                    tick_marks.append((i, px, py, pt))
                    next_tick += tick_interval

            # Subsample for line rendering.
            if len(positions) > 4000:
                step = len(positions) // 4000
                positions = positions[::step]
            xs = np.array([p[0] for p in positions])
            ys = np.array([p[1] for p in positions])
            pts = np.c_[xs, ys].reshape(-1, 1, 2)
            segments = np.concatenate([pts[:-1], pts[1:]], axis=1)
            t = np.linspace(0.0, 1.0, len(segments))
            lc = LineCollection(segments, cmap="cool", linewidth=2.0,
                                alpha=0.9, zorder=4)
            lc.set_array(t)
            ax.add_collection(lc)

            ax.plot(xs[0], ys[0], "o", color="#3fb950", markersize=9,
                    markeredgecolor="#e6edf3", markeredgewidth=1.5,
                    zorder=6, label="Start")
            end_col = "#f85149" if self.state.crashed else "#f2cc60"
            ax.plot(xs[-1], ys[-1], "s", color=end_col, markersize=9,
                    markeredgecolor="#e6edf3", markeredgewidth=1.5,
                    zorder=6, label="End")

            # Draw 10-second tick marks perpendicular to the path.
            tick_len = max(sim_map.width, sim_map.height) * 0.018
            full = self.flight_positions
            for idx, tx, ty, tt in tick_marks:
                # Local direction from the position before to the one after.
                i0 = max(0, idx - 3)
                i1 = min(len(full) - 1, idx + 3)
                dx = full[i1][0] - full[i0][0]
                dy = full[i1][1] - full[i0][1]
                mag = math.hypot(dx, dy)
                if mag < 1e-6:
                    continue
                # Perpendicular unit vector (rotate 90° CCW).
                px2, py2 = -dy / mag, dx / mag
                ax.plot(
                    [tx - px2 * tick_len, tx + px2 * tick_len],
                    [ty - py2 * tick_len, ty + py2 * tick_len],
                    "-", color="#8b949e", linewidth=1.2, zorder=5,
                )
                ax.annotate(
                    f"{int(tt)}s",
                    (tx + px2 * tick_len * 1.5, ty + py2 * tick_len * 1.5),
                    ha="center", va="center",
                    color="#8b949e", fontsize=6, zorder=5,
                )

        # Crash marker
        if self.crash_pos is not None:
            ax.plot(self.crash_pos[0], self.crash_pos[1], "x",
                    color="#f85149", markersize=18, markeredgewidth=4,
                    zorder=7, label="Crash")

        elapsed = time.monotonic() - self.started
        crashed_str = "  [CRASHED]" if self.state.crashed else ""
        ax.set_title(
            f"dsim {self.args.id}  |  {Path(self.args.map).name}"
            f"  |  {elapsed:.1f} s{crashed_str}  |  run {self.run_id}",
            color="#e6edf3", fontsize=8, pad=8,
        )
        if positions:
            ax.legend(facecolor="#21262d", edgecolor="#30363d",
                      labelcolor="#e6edf3", fontsize=8)

        path.parent.mkdir(parents=True, exist_ok=True)
        canvas.print_figure(str(path), dpi=150, bbox_inches="tight",
                            facecolor="#0d1117")
        print(f"dsim: flight path → {path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Top-down monitor UI
# ---------------------------------------------------------------------------

class TopDownUi:
    def __init__(self, sim: DroneSimulator):
        self.sim    = sim
        self.closed = False
        self.root   = tk.Tk()
        self.root.title(f"dsim {sim.args.id}")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self._apply_theme()

        map_w = sim.map.width
        map_h = sim.map.height
        self.cell   = max(18, min(42, int(720 / max(map_w, map_h, 1))))
        self.margin = 18
        canvas_w    = map_w * self.cell + self.margin * 2
        canvas_h    = map_h * self.cell + self.margin * 2

        self.status_var = tk.StringVar(value="")
        self.command_var = tk.StringVar(value="")
        header = ttk.Frame(self.root, padding=(12, 10), style="Header.TFrame")
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(1, weight=1)
        ttk.Label(header, text=f"dsim  {sim.args.id}", style="Brand.TLabel").grid(
            row=0, column=0, sticky="w", padx=(0, 12))
        ttk.Label(header, text=str(sim.map.path), style="HeaderDim.TLabel").grid(
            row=0, column=1, sticky="w")

        body = ttk.Frame(self.root, padding=(12, 12, 12, 0))
        body.grid(row=1, column=0, sticky="nsew")
        body.columnconfigure(0, weight=1)
        body.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(body, width=canvas_w, height=canvas_h,
                                bg=_UI_CANVAS, highlightthickness=1,
                                highlightbackground=_UI_GRID)
        self.canvas.grid(row=0, column=0, sticky="nsew")

        footer = ttk.Frame(self.root, padding=(12, 8, 12, 12), style="Header.TFrame")
        footer.grid(row=2, column=0, sticky="ew")
        footer.columnconfigure(0, weight=1)
        ttk.Label(footer, textvariable=self.status_var, anchor="w",
                  style="HeaderDim.TLabel").grid(row=0, column=0, sticky="ew")
        ttk.Label(footer, textvariable=self.command_var, anchor="w",
                  style="HeaderDim.TLabel").grid(row=1, column=0, sticky="ew", pady=(4, 0))
        ttk.Button(footer, text="Save Snapshot", command=self.sim.save_snapshot,
                   style="TButton").grid(row=0, column=1, rowspan=2,
                                         sticky="e", padx=(12, 0))
        ttk.Button(footer, text="Reset drone", command=self.sim.reset_drone,
                   style="Accent.TButton").grid(row=0, column=2, rowspan=2,
                                                sticky="e", padx=(8, 0))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        self.draw_static_map()
        self.drone_items: list[int] = []

        restore_window_pos(self.root, f"dsim.{sim.args.id}")

    def _apply_theme(self) -> None:
        self.root.configure(bg=_UI_BG)
        s = ttk.Style(self.root)
        s.theme_use("clam")
        s.configure(".", background=_UI_BG, foreground=_UI_TEXT,
                    bordercolor=_UI_GRID, darkcolor=_UI_PANEL, lightcolor=_UI_PANEL,
                    fieldbackground=_UI_BUTTON, troughcolor=_UI_PANEL,
                    selectbackground=_UI_ACCENT, selectforeground=_UI_TEXT)
        s.configure("TFrame", background=_UI_BG)
        s.configure("Header.TFrame", background=_UI_PANEL)
        s.configure("TLabel", background=_UI_BG, foreground=_UI_TEXT)
        s.configure("Brand.TLabel", background=_UI_PANEL, foreground=_UI_TEXT,
                    font=("TkDefaultFont", 11, "bold"))
        s.configure("HeaderDim.TLabel", background=_UI_PANEL, foreground=_UI_DIM)
        s.configure("TButton", background=_UI_BUTTON, foreground=_UI_TEXT,
                    bordercolor=_UI_GRID, lightcolor=_UI_GRID, darkcolor=_UI_GRID,
                    focuscolor=_UI_ACCENT, padding=(10, 6), relief="flat")
        s.map("TButton", background=[("pressed", _UI_PANEL), ("active", _UI_BUTTON_ACTIVE)],
              bordercolor=[("active", _UI_ACCENT)])
        s.configure("Accent.TButton", background="#1f6feb", foreground="#ffffff",
                    bordercolor="#388bfd", lightcolor="#388bfd", darkcolor="#1f6feb",
                    padding=(12, 7), relief="flat")
        s.map("Accent.TButton", background=[("pressed", "#1158c7"), ("active", "#388bfd")])

    def close(self) -> None:
        self.closed = True
        try:
            save_window_pos(self.root, f"dsim.{self.sim.args.id}")
            self.root.destroy()
        except tk.TclError:
            pass

    def update(self) -> None:
        if self.closed:
            return
        self.draw_drone()
        self.root.update_idletasks()
        self.root.update()

    def xy(self, x: float, y: float) -> tuple[float, float]:
        return self.margin + x * self.cell, self.margin + y * self.cell

    def draw_static_map(self) -> None:
        sim_map = self.sim.map
        for y in range(sim_map.height):
            for x in range(sim_map.width):
                x0, y0 = self.xy(x, y)
                x1, y1 = self.xy(x + 1, y + 1)
                self.canvas.create_rectangle(x0, y0, x1, y1,
                                             fill=_UI_CELL, outline=_UI_GRID)
        for obj in sim_map.objects:
            x0, y0 = self.xy(obj.x - 0.5, obj.y - 0.5)
            x1, y1 = self.xy(obj.x + 0.5, obj.y + 0.5)
            if obj.kind == "wall":
                self.canvas.create_rectangle(x0, y0, x1, y1,
                                             fill="#6e7681", outline="#8b949e")
            elif obj.kind == "tree":
                cx, cy = self.xy(obj.x, obj.y)
                r = self.cell * 0.36
                self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                                        fill="#238636", outline="#3fb950")
            elif obj.kind == "target":
                cx, cy = self.xy(obj.x, obj.y)
                r = self.cell * 0.34
                self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                                        fill="#da3633", outline="#f85149")
                self.canvas.create_line(cx - r, cy, cx + r, cy,
                                        fill="#ffffff", width=2)
                self.canvas.create_line(cx, cy - r, cx, cy + r,
                                        fill="#ffffff", width=2)
            else:
                self.canvas.create_rectangle(x0, y0, x1, y1,
                                             fill="#30363d", outline=_UI_GRID)

    def draw_drone(self) -> None:
        for item in self.drone_items:
            self.canvas.delete(item)
        self.drone_items.clear()

        st  = self.sim.state
        cx, cy = self.xy(st.x, st.y)
        yaw  = math.radians((180.0 - st.yaw_deg) % 360.0)
        size = self.cell * 0.34
        nose  = (cx + math.cos(yaw) * size, cy + math.sin(yaw) * size)
        left  = (cx + math.cos(yaw + 2.45) * size, cy + math.sin(yaw + 2.45) * size)
        right = (cx + math.cos(yaw - 2.45) * size, cy + math.sin(yaw - 2.45) * size)
        vlen  = self.cell * 1.8
        vl = (cx + math.cos(yaw - 0.45) * vlen, cy + math.sin(yaw - 0.45) * vlen)
        vr = (cx + math.cos(yaw + 0.45) * vlen, cy + math.sin(yaw + 0.45) * vlen)

        self.drone_items.append(self.canvas.create_polygon(
            cx, cy, vl[0], vl[1], vr[0], vr[1],
            fill="#f2cc60", outline="", stipple="gray50"))
        self.drone_items.append(self.canvas.create_polygon(
            nose[0], nose[1], left[0], left[1], right[0], right[1],
            fill=_UI_ACCENT if not st.crashed else _UI_DANGER,
            outline="#c9d1d9", width=2))
        self.drone_items.append(self.canvas.create_oval(
            cx - 3, cy - 3, cx + 3, cy + 3, fill="#010409", outline=""))

        self.status_var.set(
            f"x={st.x:.2f} y={st.y:.2f} z={st.z:.2f}m  "
            f"heading={st.yaw_deg:.1f}°  mode={st.mode}  "
            f"armed={'yes' if st.armed else 'no'}  "
            f"status={st.status_message}"
        )
        self.command_var.set(f"command: {self.command_text(st)}")

    @staticmethod
    def command_text(st: DroneState) -> str:
        eps = 0.05
        parts: list[str] = []
        if st.cmd_forward > eps:
            parts.append("move forward")
        elif st.cmd_forward < -eps:
            parts.append("move back")
        if st.cmd_right > eps:
            parts.append("move right")
        elif st.cmd_right < -eps:
            parts.append("move left")
        if st.cmd_up > eps:
            parts.append("move up")
        elif st.cmd_up < -eps:
            parts.append("move down")
        if st.cmd_yaw_rate > eps:
            parts.append("yaw right")
        elif st.cmd_yaw_rate < -eps:
            parts.append("yaw left")
        if not parts:
            return "hover"
        return ", ".join(parts)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="dvision2 drone simulator")
    parser.add_argument("--id",       required=True)
    parser.add_argument("--map",      default="dsim/assets/maps/maze_001.txt")
    parser.add_argument("--width",    type=int,   default=640)
    parser.add_argument("--height",   type=int,   default=480)
    parser.add_argument("--fps",      type=int,   default=30)
    parser.add_argument("--bufs",     type=int,   default=4)
    parser.add_argument("--cmd-size", type=int,   default=65536)
    parser.add_argument("--start-alt", type=float,
                        help="initial altitude; default uses map drone-height or 1.5")
    parser.add_argument("--origin-lat", type=float, default=BERLIN_CENTER_LAT_DEG,
                        help="GPS latitude for the center of the map")
    parser.add_argument("--origin-lon", type=float, default=BERLIN_CENTER_LON_DEG,
                        help="GPS longitude for the center of the map")
    parser.add_argument("--origin-alt", type=float, default=BERLIN_CENTER_ALT_M,
                        help="ground altitude in meters for the center of the map")
    parser.add_argument("--frames",   type=int,
                        help="run for a fixed number of frames, for smoke tests")
    parser.add_argument("--report-dir", default=None,
                        help="write run artifacts to this exact directory")
    parser.add_argument("--no-ui",    action="store_true",
                        help="disable the top-down simulator UI")
    parser.add_argument("--verbose",  action="store_true")
    args = parser.parse_args(argv)
    validate_id(args.id)
    if args.width <= 0 or args.height <= 0 or args.fps <= 0 or args.bufs <= 0:
        raise SystemExit("width, height, fps, and bufs must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    sim  = DroneSimulator(args)

    def stop(_signum, _frame):
        sim.running = False

    signal.signal(signal.SIGINT,  stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        sim.run()
    except Exception as exc:
        print(f"dsim: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
