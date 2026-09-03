#!/usr/bin/env python3

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import random
import signal
import sys
import time
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import ttk

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dcmn import theme
from dcmn.mapview import MapView, draw_map_axes
from dcmn.tktheme import apply_theme
from dsim.realism import GEOFENCE_ACTIONS, GPS_MODES, REALISM_DEFAULTS, Realism, SENSOR_NOISE_PROFILES
from dsim.realism_panel import RealismPanel
from dsim.scene import SCENE_PRESETS
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
    local_ned_to_map,
    local_to_gps,
    new_run_id,
    report_root,
    restore_window_pos,
    save_window_pos,
    shared_names,
    validate_id,
)


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

    KNOWN INTERNAL INCONSISTENCY.  The text map is south-positive in Y while
    Panda's Y is north-positive, so feeding map Y straight in mirrors the whole
    scene; ``render`` then mirrors the readback back again.  The delivered frame
    obeys the public image contract (verified by
    ``tests/test_dvision_calibration_render.py``), but the Panda-space scene is
    chirality-flipped: model and texture handedness are mirrored, and the camera
    roll sign has to be negated below to compensate.  Un-mirroring the scene
    produces a geometrically identical camera image but changes ground/brick
    texture handedness enough to move DAIC's SLAM obstacle estimates, so it is
    left alone here rather than changed as a side effect.  Anything that reasons
    about Panda world coordinates directly must account for the mirror.
    """

    WALL_H       = 2.5
    TREE_TRUNK_H = 1.6
    TREE_TRUNK_W = 0.22
    TREE_CROWN_R = 0.60
    TREE_MODEL_H = 4.5
    TARGET_R = 0.36

    # Flat, unlit diagnostic marker colours.  Red is deliberately desaturated
    # (HSV S below daic.detector's 160 gate) so a calibration panel can never
    # be mistaken for the saturated red landing target, while still passing an
    # independent r>150 / g<100 / b<100 threshold.
    MARKER_COLORS = {
        "red":    (0.70, 0.34, 0.32),
        "blue":   (0.00, 0.00, 1.00),
        "yellow": (1.00, 1.00, 0.00),
        "green":  (0.00, 1.00, 0.00),
        "white":  (1.00, 1.00, 1.00),
    }
    CAM_Z_OFFSET  = 0.1
    CAM_PITCH     = -5.0   # slight forward-down tilt
    CAM_FOV_H     = 70.0   # horizontal field of view, degrees
    GROUND_TEXTURE_M = 2.1
    SHADOW_MAP_PX = 2048   # representative preset only

    def __init__(self, sim_map: SimMap, width: int, height: int,
                 *, scene_preset: str = "legacy") -> None:
        if scene_preset not in SCENE_PRESETS:
            raise ValueError(f"unknown scene preset: {scene_preset}")
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
        self.scene_preset = scene_preset

        self._base = ShowBase()
        self._base.disableMouse()
        self._base.setBackgroundColor(0.42, 0.62, 0.80, 1)

        lens = self._base.camLens
        lens.setFov(self.CAM_FOV_H)
        lens.setNear(0.15)
        lens.setFar(150.0)
        self._near_m = 0.15
        self._far_m = 150.0

        # Pixel readback texture
        self._tex = Texture()
        self._base.win.addRenderTexture(self._tex, GraphicsOutput.RTMCopyRam)
        self._depth_tex = Texture()
        self._base.win.addRenderTexture(
            self._depth_tex, GraphicsOutput.RTMCopyRam, GraphicsOutput.RTPDepth
        )

        # Lights
        amb = AmbientLight("amb")
        amb.setColor(LColor(*self._material_color((0.38, 0.38, 0.42)), 1))
        self._base.render.setLight(self._base.render.attachNewNode(amb))

        sun = DirectionalLight("sun")
        sun.setColor(LColor(*self._material_color((1.0, 0.94, 0.82)), 1))
        sun_np = self._base.render.attachNewNode(sun)
        sun_np.setHpr(50, -55, 0)
        if scene_preset == "representative":
            # A directional shadow caster renders the scene from the light,
            # through an orthographic lens that has to be told how much world
            # to cover. Left at its defaults the frustum is a few units across
            # and sits at the origin, so nothing outside a corner of the map
            # casts anything -- which is how "shadows on" can change under one
            # per cent of the pixels and still look enabled.
            sun_np.setPos(sim_map.width / 2.0, sim_map.height / 2.0, 60.0)
            span = max(sim_map.width, sim_map.height) * 1.5
            shadow_lens = sun.getLens()
            shadow_lens.setFilmSize(span, span)
            shadow_lens.setNearFar(1.0, 200.0)
            sun.setShadowCaster(True, self.SHADOW_MAP_PX, self.SHADOW_MAP_PX)
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
            ROOT / "assets/textures/Bricks042/Bricks042_1K-JPG_Color.jpg"
        )
        self._ground_texture = self._load_texture(
            ROOT / "assets/textures/Ground037/Ground037_1K-JPG_Color.jpg"
        )
        self._tree_models = self._load_tree_models(
            ROOT / "assets/models/trees"
        )

        self._build_scene(sim_map)
        self._pbr = None
        if scene_preset == "representative":
            self._pbr = self._enable_pbr_shadows()

    def _enable_pbr_shadows(self):
        """Turn on the shading pipeline that actually renders cast shadows.

        Panda's fixed-function path ignores a light's shadow map, and its
        generated auto-shader did not produce one in this offscreen context
        either: enabling ``setShadowCaster`` alone changed no pixel of the
        ground under a tree. ``panda3d-simplepbr`` is the pipeline this project
        already pins for the purpose, so the representative preset renders
        through it. Its ``ImportError`` is deliberately allowed to propagate:
        a preset that quietly fell back to the legacy look while still
        publishing a new ``scene_version`` would put unchanged pixels behind a
        changed version number, and every comparison drawn against it would be
        measuring nothing.
        """
        import simplepbr

        pipeline = simplepbr.init(render_node=self._base.render,
                                  window=self._base.win,
                                  camera_node=self._base.cam,
                                  use_normal_maps=False,
                                  enable_shadows=True,
                                  msaa_samples=0)
        # simplepbr finishes compiling its shader inside a task, and this
        # ShowBase never runs a task loop of its own; without one step the very
        # first readback raises "Shader input camera_world_position is not
        # present" instead of returning a frame.
        self._base.taskMgr.step()
        return pipeline

    # ------------------------------------------------------------------
    # Scene helpers
    # ------------------------------------------------------------------

    def _material_color(self, color: tuple) -> tuple:
        """Convert a scene colour into the space the active pipeline expects.

        The fixed-function path writes ``setColor`` straight to the framebuffer,
        so the literals in this file are sRGB. ``simplepbr`` treats the same
        value as a *linear* base colour and applies the sRGB transfer curve on
        output, which lifts every mid-tone: the red calibration panel came out
        pink and stopped being classifiable as red at all. Linearising here
        keeps the delivered pixels comparable between presets, so the
        representative scene differs by shadows and ground periodicity -- the
        things it is supposed to change -- and not by a global colour shift
        that would invalidate the calibration oracle for no benefit.
        """
        if self.scene_preset != "representative":
            return color
        return tuple(channel ** 2.2 for channel in color)

    def _box(self, cx: float, cy: float, bz: float,
             w: float, d: float, h: float, color: tuple,
             texture=None, *, unlit: bool = False) -> None:
        node = self._proto_box.copyTo(self._base.render)
        node.setScale(w, d, h)
        node.setPos(cx - w * 0.5, cy - d * 0.5, bz)
        node.setColor(*self._material_color(color), 1)
        if texture is not None:
            node.setTexture(texture, 1)
        if unlit:
            # The prototype box carries a texture that would otherwise
            # modulate setColor; calibration markers must be flat and
            # fully saturated so colour thresholds are unambiguous.
            node.setTextureOff(1)
            node.setLightOff()
            node.setFogOff()
            # ``setLightOff`` is a fixed-function state, and a shading pipeline
            # is free to ignore it -- simplepbr does, and shaded the flat red
            # calibration panel into a pink one that the colour oracle could no
            # longer classify. Taking the node out of the shader entirely is
            # what makes "unlit" mean the same thing under both presets, so the
            # calibration fixture measures geometry rather than lighting.
            node.setShaderOff(1)

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
        node.setColor(*self._material_color(color), 1)

    def _ground(self, cx: float, cy: float, size: float) -> None:
        from panda3d.core import (
            Geom, GeomNode, GeomTriangles, GeomVertexData, GeomVertexFormat,
            GeomVertexWriter,
        )

        # Panda's fixed-function fog is vertex based. Subdivide one continuous
        # mesh so fog interpolates locally without tile cracks or UV resets.
        spacing = 5.0
        count = max(1, math.ceil(size / spacing))
        extent = count * spacing
        start_x, start_y = cx - extent * 0.5, cy - extent * 0.5
        data = GeomVertexData("ground", GeomVertexFormat.getV3n3t2(), Geom.UHStatic)
        vertices = GeomVertexWriter(data, "vertex")
        normals = GeomVertexWriter(data, "normal")
        texcoords = GeomVertexWriter(data, "texcoord")
        for iy in range(count + 1):
            y = start_y + iy * spacing
            for ix in range(count + 1):
                x = start_x + ix * spacing
                vertices.addData3(x, y, -0.01)
                normals.addData3(0.0, 0.0, 1.0)
                if self.scene_preset == "representative":
                    # Deterministic low-frequency warp breaks the conspicuous
                    # periodic lattice without adding another asset or random
                    # state. Geometry remains unchanged.
                    u = x / self.GROUND_TEXTURE_M + 0.19 * math.sin(y * 0.173)
                    v = y / self.GROUND_TEXTURE_M + 0.17 * math.sin(x * 0.137)
                else:
                    u, v = x / self.GROUND_TEXTURE_M, y / self.GROUND_TEXTURE_M
                texcoords.addData2(u, v)
        triangles = GeomTriangles(Geom.UHStatic)
        row = count + 1
        for iy in range(count):
            for ix in range(count):
                a = iy * row + ix
                b, c, d = a + 1, a + row, a + row + 1
                triangles.addVertices(a, b, d)
                triangles.addVertices(a, d, c)
        triangles.closePrimitive()
        geom = Geom(data)
        geom.addPrimitive(triangles)
        geom_node = GeomNode("ground")
        geom_node.addGeom(geom)
        node = self._base.render.attachNewNode(geom_node)
        node.setColor(*self._material_color((0.26, 0.37, 0.20)), 1)
        node.setTwoSided(True)
        if self._ground_texture is not None:
            node.setTexture(self._ground_texture, 1)
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
            node.setColor(*self._material_color(color), 1)

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
                    seed = scene_object_seed(sim_map.path, x, y)
                    rng = random.Random(seed)
                    model = rng.choice(self._tree_models)
                    self._model_tree(x, y, model, rng.uniform(0.0, 360.0))
                else:
                    self._fallback_tree(x, y)
            elif obj.kind == "target":
                self._target_marker(x, y)
            elif obj.kind.startswith("marker_"):
                marker = obj.kind.removeprefix("marker_")
                colors = self.MARKER_COLORS
                if marker in colors:
                    # Vertical panels provide an unambiguous, thresholdable
                    # world-space calibration target. Yellow is deliberately
                    # elevated and green deliberately low to test image Y.
                    bz = 2.2 if marker == "yellow" else 0.0
                    h = 0.8 if marker == "green" else 1.6
                    # Square in plan so the panel presents the same face from
                    # any bearing. A thin panel is edge-on from 90 degrees away,
                    # which leaves a sliver of a mask and makes a rotationally
                    # symmetric calibration fixture measure differently at each
                    # heading for reasons that have nothing to do with
                    # orientation correctness.
                    self._box(x, y, bz, 0.9, 0.90, h, colors[marker], unlit=True)

    # ------------------------------------------------------------------
    # Per-frame render
    # ------------------------------------------------------------------

    def render(self, state: DroneState, out_frame: np.ndarray) -> None:
        self._position_camera(state)
        self._base.graphicsEngine.renderFrame()

        if self._tex.hasRamImage():
            data = self._tex.getRamImageAs("RGB")
            arr  = np.frombuffer(bytes(data), dtype=np.uint8)
            # Row reversal turns Panda's bottom-up framebuffer into the
            # top-left-origin DVision image; the column reversal undoes the
            # mirrored scene described in the class docstring.
            out_frame[:] = arr.reshape((self._height, self._width, 3))[::-1, ::-1]

    def _position_camera(self, state: DroneState) -> None:
        panda_h = (90.0 - state.yaw_deg) % 360.0
        cam = self._base.camera
        cam.setPos(state.x, state.y, state.z + self.CAM_Z_OFFSET)
        # Panda P is nose-up, matching the published pitch convention.  Positive
        # DVision roll is a right-wing-down bank, which must lift the right end
        # of the horizon in the delivered image; the Panda scene is mirrored
        # (see the class docstring), so the roll sign is inverted here.
        cam.setHpr(panda_h, self.CAM_PITCH + state.pitch_deg, -state.roll_deg)

    def render_range(self, state: DroneState) -> np.ndarray:
        """Return Euclidean first-surface range in metres, NaN at the far plane."""
        self._position_camera(state)
        self._base.graphicsEngine.renderFrame()
        if not self._depth_tex.hasRamImage():
            raise RuntimeError("depth buffer readback is unavailable")
        raw = np.frombuffer(bytes(self._depth_tex.getRamImage()), dtype=np.float32)
        depth = raw.reshape((self._height, self._width))[::-1, ::-1].copy()
        z_ndc = depth * 2.0 - 1.0
        axial = (2.0 * self._near_m * self._far_m /
                 (self._far_m + self._near_m
                  - z_ndc * (self._far_m - self._near_m)))
        lens = self._base.camLens
        fov_h = math.radians(lens.getFov()[0])
        fx = self._width / (2.0 * math.tan(fov_h / 2.0))
        yy, xx = np.indices(depth.shape, dtype=np.float32)
        ray_scale = np.sqrt(1.0 + ((xx - self._width / 2.0) / fx) ** 2
                            + ((yy - self._height / 2.0) / fx) ** 2)
        ranges = axial * ray_scale
        ranges[(depth >= 1.0 - 1e-7) | (depth <= 0.0)] = np.nan
        if not np.isfinite(ranges).any():
            raise RuntimeError("depth buffer readback contains only far-plane samples")
        return ranges.astype(np.float32)

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

# Position controller trim: how fast the controller learns a steady
# disturbance, and the most velocity it will ever trim out. The cap is well
# under the wind speeds worth simulating and keeps a blocked vehicle from
# winding up into a lurch when whatever held it is removed.
_POSITION_TRIM_GAIN = 0.5      # (m/s) of trim per metre of error per second
_MAX_POSITION_TRIM_MPS = 2.0
_POSITION_TRIM_BAND_M = 1.0    # only trim once the approach is essentially done
_POSITION_TRIM_SPEED_MPS = 0.5 # ...and the vehicle is hovering, not travelling

# Obstacles occupy their full map cell.  Map object centres are at x/y + 0.5,
# so a half extent of 0.5 makes adjacent wall cells touch with no seam.
_OBSTACLE_HALF_EXTENT_M = 0.5
_COLLISION_SWEEP_STEP_M = 0.1


def scene_object_seed(map_path: Path, x: float, y: float) -> str:
    """Stable scene seed independent of path spelling and checkout location."""
    map_sha = hashlib.sha256(Path(map_path).read_bytes()).hexdigest()
    return f"{map_sha}:{x:.3f}:{y:.3f}"


def sim_yaw_to_compass_heading(yaw_deg: float) -> float:
    """Convert the renderer's internal yaw to public compass heading."""
    return (270.0 - float(yaw_deg)) % 360.0


def compass_heading_to_sim_yaw(heading_deg: float) -> float:
    """Convert public compass heading to the renderer's internal yaw."""
    return (270.0 - float(heading_deg)) % 360.0


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
    target_x: float | None = None
    target_y: float | None = None
    target_z: float | None = None
    target_heading_deg: float | None = None
    target_max_speed_mps: float = 1.0
    # Trim accumulated by the position controller against a steady
    # disturbance, in metres per second of commanded map velocity.
    target_trim_x: float = 0.0
    target_trim_y: float = 0.0
    last_setpoint_monotonic: float | None = None
    guided_entered_monotonic: float | None = None
    control_owner: str = ""
    control_lease_id: str = ""
    lease_acquired_monotonic: float | None = None
    lease_heartbeat_monotonic: float | None = None
    failsafe_reason: str = ""
    result_request_id: str = ""
    result_accepted: bool = False
    result_reason: str = ""
    home_x: float | None = None
    home_y: float | None = None
    home_z: float | None = None
    rtl_stage: str = ""


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

class DroneSimulator:
    #: Simulated time, in seconds since the run began. A class default so a
    #: simulator built with ``__new__`` by a test rig still has a clock.
    sim_time_s = 0.0

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
        start_heading = float(getattr(args, "start_heading", 0.0))
        self.start_yaw = compass_heading_to_sim_yaw(start_heading)
        self.state = DroneState(
            self.map.start_x, self.map.start_y, start_alt,
            yaw_deg=self.start_yaw,
        )
        self.realism = Realism.from_settings(vars(args))
        self.running = True
        self.started = time.monotonic()
        self.sim_time_s = 0.0
        self.video = None
        self.command = None
        self.status = None
        self.ui = None
        self.p3d: Panda3DRenderer | None = None

        self.run_id = new_run_id()
        if getattr(args, "report_dir", None):
            # An explicit directory wins outright: a caller that named a path
            # wants its artifacts there, not somewhere derived from it.
            chosen = Path(args.report_dir)
            if not chosen.is_absolute():
                chosen = ROOT / chosen
            self.report_root = chosen
            self.run_id = chosen.name
        else:
            self.report_root = report_root(getattr(args, "id", None),
                                           self.run_id, root=ROOT / "reports")
        self.dsim_report_dir = self.report_root / "dsim"
        self.dsim_report_dir.mkdir(parents=True, exist_ok=True)
        print(f"dsim: report directory → {self.report_root}", file=sys.stderr)

        self.flight_positions: list[tuple[float, float, float]] = []  # (x, y, elapsed_s)
        self.crash_pos: tuple[float, float] | None = None

    def __getattr__(self, name: str):
        """Build the realism model on demand for simulators made by ``__new__``.

        The deterministic harnesses and several tests construct a simulator
        without running ``__init__`` so they can pin its map, pose and clock.
        Rather than make every one of them remember a new attribute, an
        environment is derived from whatever settings their arguments carry --
        which, for a namespace that names none of them, is the same
        realism-off configuration those tests already assume.
        """
        if name == "realism":
            realism = Realism.from_settings(vars(self.args))
            self.realism = realism
            return realism
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}")

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
        self.p3d = Panda3DRenderer(
            self.map,
            self.args.width,
            self.args.height,
            scene_preset=self.args.scene_preset,
        )
        if self.args.verbose:
            print(f"dsim: using Panda3D renderer ({self.args.scene_preset})")

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
        self.publish_status(force=True)

    def close(self) -> None:
        if self.flight_positions:
            try:
                self._save_flight_image(self.dsim_report_dir / "flight_path.png")
            except Exception as exc:
                print(f"dsim: flight image error: {exc}", file=sys.stderr)
        try:
            elapsed = time.monotonic() - self.started
            summary = {
                "duration_s": round(elapsed, 3),
                "crashed": bool(self.state.crashed),
                "mode": self.state.mode,
                "status_message": self.state.status_message,
                "x_m": round(self.state.x, 4),
                "y_m": round(self.state.y, 4),
                "z_m": round(self.state.z, 4),
                "speed_mps": round(math.hypot(self.state.vx, self.state.vy), 4),
                "crash_position": self.crash_pos,
            }
            (self.dsim_report_dir / "summary.json").write_text(
                json.dumps(summary, indent=2), encoding="utf-8")
        except Exception as exc:
            print(f"dsim: summary error: {exc}", file=sys.stderr)
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
                self.step(dt, drain_commands=True)
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

    def step(self, dt: float, *, drain_commands: bool = False) -> None:
        """Advance command processing and physics by an explicit timestep.

        Tests pass a fixed ``dt``; the real-time loop passes its measured
        timestep. Rendering and publication remain separately callable so
        deterministic tests can select exactly which boundaries they exercise.
        """
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        if drain_commands:
            self.drain_commands()
        self.integrate(dt)

    def apply_command(self, payload: dict) -> None:
        typ = payload["type"]
        self.state.last_command_monotonic = self.clock()
        self.state.last_command_type = typ
        self.state.command_count += 1

        request_id = str(payload.get("request_id", ""))
        if typ == "acquire_control":
            source = str(payload.get("source_id", ""))
            lease = str(payload.get("lease_id", ""))
            if not source or not lease:
                self._command_result(request_id, False, "source_id and lease_id required")
            elif self._lease_active() and self.state.control_owner != source:
                self._command_result(request_id, False,
                                     f"controlled by {self.state.control_owner}")
            else:
                now = self.clock()
                self.state.control_owner = source
                self.state.control_lease_id = lease
                self.state.lease_acquired_monotonic = now
                self.state.lease_heartbeat_monotonic = now
                self._command_result(request_id, True)
            return
        if typ == "release_control":
            if not self._owns_control(payload):
                self._command_result(request_id, False, "control lease required")
            else:
                self._clear_lease()
                self._command_result(request_id, True)
            return
        if typ == "reset":
            self.reset_drone()
            self.state.result_request_id = request_id
            self.state.result_accepted = True
            self.state.result_reason = ""
            return
        if self.state.crashed:
            self.state.status_message = "crashed"
            self._command_result(request_id, False, "crashed")
            return
        self.state.status_message = "ok"

        if typ == "heartbeat":
            if self._owns_control(payload):
                self.state.lease_heartbeat_monotonic = self.clock()
            self._command_result(request_id, True)
            return
        if typ != "land" and not self._owns_control(payload):
            self._command_result(request_id, False, "control lease required")
            return
        if typ == "arm":
            was_armed = self.state.armed
            self.state.armed = bool(payload.get("armed", True))
            self.state.mode = "GUIDED" if self.state.armed else "DISARMED"
            if self.state.armed and not was_armed:
                self.state.home_x = self.state.x
                self.state.home_y = self.state.y
                self.state.home_z = self.state.z
                self.state.failsafe_reason = ""
                self.state.last_setpoint_monotonic = None
                self.state.guided_entered_monotonic = self.clock()
            if not self.state.armed:
                self._clear_targets()
            self._command_result(request_id, True)
            return
        if typ == "takeoff":
            if self.state.armed:
                try:
                    altitude = float(payload.get("alt_m", 3.0))
                    if not math.isfinite(altitude) or altitude < 0.5:
                        raise ValueError("invalid takeoff altitude")
                except (TypeError, ValueError) as exc:
                    self._command_result(request_id, False, str(exc))
                    return
                self.state.target_alt = altitude
                self.state.mode = "TAKEOFF"
                self._clear_position_target()
                self._command_result(request_id, True)
            else:
                self._command_result(request_id, False, "vehicle is not armed")
            return
        if typ == "land":
            self.state.target_alt = 0.0
            self.state.mode = "LAND"
            self._clear_position_target()
            self.state.cmd_forward = self.state.cmd_right = 0.0
            self.state.vx = self.state.vy = 0.0
            self._command_result(request_id, True)
            return
        if typ in ("zero", "hold"):
            self._clear_targets()
            self.state.mode = "HOLD" if self.state.armed else self.state.mode
            self._command_result(request_id, True)
            return
        if typ == "rtl":
            if not self.state.armed or self.state.home_x is None:
                self._command_result(request_id, False, "home unavailable")
            else:
                self._clear_targets()
                self.state.mode = "RTL"
                self.state.rtl_stage = "climb"
                self._command_result(request_id, True)
            return
        if typ == "set_gps":
            try:
                self.realism.set_gps(
                    str(payload.get("mode", self.realism.gps_mode)),
                    None if payload.get("noise_m") is None
                    else float(payload["noise_m"]))
            except (TypeError, ValueError) as exc:
                self._command_result(request_id, False, str(exc))
            else:
                self._command_result(request_id, True)
            return
        if typ == "set_estimator":
            flags = {name: bool(payload[name])
                     for name in ("attitude", "local", "global", "velocity")
                     if name in payload}
            if not flags:
                self._command_result(request_id, False, "no estimator named")
                return
            try:
                self.realism.set_estimator(**flags)
            except ValueError as exc:
                self._command_result(request_id, False, str(exc))
            else:
                self._command_result(request_id, True)
            return
        if typ == "set_origin":
            if self.state.armed:
                self._command_result(request_id, False, "set_origin requires disarmed")
            else:
                try:
                    self.args.origin_lat = float(payload["lat_deg"])
                    self.args.origin_lon = float(payload["lon_deg"])
                    self.args.origin_alt = float(payload["alt_m"])
                except (KeyError, TypeError, ValueError):
                    self._command_result(request_id, False, "invalid origin")
                else:
                    self._command_result(request_id, True)
            return
        if typ == "velocity":
            if not self.state.armed:
                self.state.cmd_forward = self.state.cmd_right = 0.0
                self.state.cmd_up = self.state.cmd_yaw_rate = 0.0
                self._command_result(request_id, False, "vehicle is not armed")
                return
            if self.state.mode not in ("GUIDED", "HOLD"):
                self._command_result(request_id, False, "GUIDED or HOLD mode required")
                return
            try:
                forward = float(payload.get("forward_mps", 0.0))
                right = float(payload.get("right_mps", 0.0))
                up = float(payload.get("up_mps", 0.0))
                yaw_rate = float(payload.get("yaw_rate_dps", 0.0))
                if not all(math.isfinite(v) for v in (forward, right, up, yaw_rate)):
                    raise ValueError("non-finite velocity")
                if math.hypot(forward, right) > self._max_speed_mps() or abs(up) > self._max_speed_mps():
                    raise ValueError("velocity outside limits")
            except (TypeError, ValueError) as exc:
                self._command_result(request_id, False, str(exc))
                return
            self._clear_position_target()
            self.state.cmd_forward = forward
            self.state.cmd_right = right
            self.state.cmd_up = up
            self.state.cmd_yaw_rate = yaw_rate
            self.state.mode         = "GUIDED"
            self.state.guided_entered_monotonic = self.clock()
            self.state.target_alt   = None
            self.state.last_setpoint_monotonic = self.clock()
            self.state.failsafe_reason = ""
            self._command_result(request_id, True)
            return
        if typ == "position_target":
            if not self.state.armed:
                self._command_result(request_id, False, "vehicle is not armed")
                return
            if self.state.mode not in ("GUIDED", "HOLD"):
                self._command_result(request_id, False, "GUIDED or HOLD mode required")
                return
            if not self.realism.estimators()["local"]:
                # A position setpoint against an invalid position estimate is
                # a request to fly to a place the vehicle cannot locate.
                self._command_result(request_id, False,
                                     "local position estimate is invalid")
                return
            try:
                frame = payload["frame"]
                if frame == "map":
                    x, y, z = (float(payload[k]) for k in ("x", "y", "z"))
                elif frame == "local_ned":
                    x, y, z = local_ned_to_map(
                        float(payload["north_m"]), float(payload["east_m"]),
                        float(payload["down_m"]), self.map.width, self.map.height)
                else:
                    raise ValueError("unsupported frame")
                heading = float(payload.get("heading_deg", 0.0)) % 360.0
                max_speed = float(payload.get("max_speed_mps", 1.0))
                if not all(math.isfinite(v) for v in (x, y, z, heading, max_speed)):
                    raise ValueError("non-finite target")
                if not (0.1 <= x <= self.map.width - 0.1
                        and 0.1 <= y <= self.map.height - 0.1
                        and z >= 0.0 and 0.0 < max_speed <= self._max_speed_mps()):
                    raise ValueError("target outside limits")
            except (KeyError, TypeError, ValueError) as exc:
                self._command_result(request_id, False, str(exc))
                return
            if self.realism.outside_geofence(x, y, z):
                self._command_result(request_id, False, "target outside geofence")
                return
            self.state.target_alt = None
            self.state.target_x, self.state.target_y, self.state.target_z = x, y, z
            self.state.target_heading_deg = heading
            self.state.target_max_speed_mps = max_speed
            self.state.mode = "GUIDED"
            self.state.guided_entered_monotonic = self.clock()
            self.state.last_setpoint_monotonic = self.clock()
            self.state.failsafe_reason = ""
            self._command_result(request_id, True)
            return
        self._command_result(request_id, False, f"unsupported command: {typ}")

    def _command_result(self, request_id: str, accepted: bool, reason: str = "") -> None:
        self.state.result_request_id = request_id
        self.state.result_accepted = accepted
        self.state.result_reason = reason
        self.state.status_message = "ok" if accepted else reason

    def _lease_active(self) -> bool:
        stamp = self.state.lease_heartbeat_monotonic
        return bool(self.state.control_lease_id and stamp is not None
                    and self.clock() - stamp <= self._lease_timeout_s())

    def _owns_control(self, payload: dict) -> bool:
        return (self._lease_active()
                and payload.get("source_id") == self.state.control_owner
                and payload.get("lease_id") == self.state.control_lease_id)

    def clock(self) -> float:
        """The vehicle's clock: simulated seconds since the run began.

        Every timer that gates flight -- the guided setpoint failsafe, the
        control lease, how long the vehicle has been in GUIDED -- reads this
        rather than the wall clock. In the live loop ``dt`` comes from the wall
        clock so the two track each other; under a fixed-timestep harness they
        do not, and a failsafe that fires because the machine was busy rather
        than because the vehicle flew for two seconds is measuring the wrong
        thing.
        """
        return self.sim_time_s

    def _lease_timeout_s(self) -> float:
        return float(getattr(self.args, "control_lease_timeout", 3.0))

    def _max_speed_mps(self) -> float:
        return float(getattr(self.args, "max_speed_mps", 5.0))

    def _clear_lease(self) -> None:
        self.state.control_owner = self.state.control_lease_id = ""
        self.state.lease_acquired_monotonic = None
        self.state.lease_heartbeat_monotonic = None

    def _clear_position_target(self) -> None:
        st = self.state
        st.target_x = st.target_y = st.target_z = st.target_heading_deg = None
        st.target_trim_x = st.target_trim_y = 0.0

    def _clear_targets(self) -> None:
        self._clear_position_target()
        self.state.target_alt = None
        self.zero_motion()

    def reset_drone(self) -> None:
        owner = self.state.control_owner
        lease_id = self.state.control_lease_id
        acquired = self.state.lease_acquired_monotonic
        heartbeat = self.state.lease_heartbeat_monotonic
        self.state = DroneState(self.start_x, self.start_y, self.start_alt,
                                yaw_deg=self.start_yaw,
                                status_message="reset")
        self.state.control_owner = owner
        self.state.control_lease_id = lease_id
        self.state.lease_acquired_monotonic = acquired
        self.state.lease_heartbeat_monotonic = heartbeat

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
        # Time the vehicle experienced, not time that passed in the room. In
        # the live loop dt comes from the wall clock so the two track each
        # other; under a fixed-timestep harness they do not, and a client that
        # infers a distance from "how long since the last frame" needs the one
        # the physics actually advanced by.
        self.sim_time_s += dt
        if st.crashed:
            self.zero_motion()
            st.mode = "CRASHED"
            st.status_message = "crashed"
            return

        now = self.clock()
        self.realism.update(dt)
        if st.control_lease_id and not self._lease_active():
            if st.armed:
                self._enter_failsafe_hold("control_lease_expired")
            else:
                self._clear_lease()
        timeout = float(getattr(self.args, "setpoint_timeout", 0.0))
        setpoint_clock = (st.last_setpoint_monotonic
                          if st.last_setpoint_monotonic is not None
                          else st.guided_entered_monotonic)
        if (timeout > 0.0 and st.armed and st.mode == "GUIDED"
                and setpoint_clock is not None
                and now - setpoint_clock > timeout):
            self._enter_failsafe_hold("setpoint_timeout")

        self._check_safety_limits()
        if st.mode == "RTL":
            self._update_rtl()
        if st.target_x is not None:
            self._update_position_commands(dt)

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
                    st.last_setpoint_monotonic = None
                    st.guided_entered_monotonic = self.clock()
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
        # Internal renderer yaw runs opposite the public compass convention.
        # Positive public yaw is right/clockwise and increases compass heading.
        st.yaw_deg  = (st.yaw_deg - st.yaw_rate * dt) % 360.0

        # Body unit vectors in map coordinates (X east, Y south) derived from
        # the *updated* yaw.  At compass heading 0 (yaw 270) this gives
        # forward = (0, -1) north and right = (1, 0) east, matching the
        # published coordinate contract.
        yaw_rad = math.radians(st.yaw_deg)
        fwd_x,   fwd_y   = -math.cos(yaw_rad),  math.sin(yaw_rad)
        right_x, right_y = -math.sin(yaw_rad), -math.cos(yaw_rad)

        # ── Horizontal velocity (first-order lag in body frame) ───────────
        v_fwd   = st.vx * fwd_x   + st.vy * fwd_y
        v_right = st.vx * right_x + st.vy * right_y

        # Drive each toward its setpoint with a lag filter, then enforce the
        # advertised physical acceleration limit on the resulting vector.
        old_vx, old_vy = st.vx, st.vy
        alpha_h  = 1.0 - math.exp(-dt / _TAU_H)
        v_fwd   += (st.cmd_forward - v_fwd)   * alpha_h
        v_right += (st.cmd_right   - v_right) * alpha_h

        # Rotate back to the world frame.
        st.vx = v_fwd * fwd_x + v_right * right_x
        st.vy = v_fwd * fwd_y + v_right * right_y
        # Hand-built deterministic simulators predating this setting omit it;
        # keeping their unbounded step is useful for collision-boundary tests.
        max_accel = getattr(self.args, "max_accel_mps2", None)
        max_dv = math.inf if max_accel is None else float(max_accel) * dt
        dvx, dvy = st.vx - old_vx, st.vy - old_vy
        dv = math.hypot(dvx, dvy)
        if dv > max_dv:
            scale = max_dv / dv
            st.vx, st.vy = old_vx + dvx * scale, old_vy + dvy * scale

        # ── Vertical velocity ─────────────────────────────────────────────
        alpha_v = 1.0 - math.exp(-dt / _TAU_V)
        desired_vz = st.vz + (st.cmd_up - st.vz) * alpha_v
        st.vz += clamp(desired_vz - st.vz, -max_dv, max_dv)

        # ── Visual attitude (roll / pitch for renderer) ───────────────────
        # Target tilt is proportional to actual body-frame velocity so the
        # camera leans into the motion naturally and levels out on hover.
        # Signs follow the aviation convention shared with dfgb: positive roll
        # is a right-wing-down bank and positive pitch is nose-up, so flying
        # forward pitches nose-down and a right strafe banks right.
        target_pitch = clamp(-v_fwd   * _TILT_GAIN, -_MAX_TILT, _MAX_TILT)
        target_roll  = clamp(v_right  * _TILT_GAIN, -_MAX_TILT, _MAX_TILT)
        alpha_att    = 1.0 - math.exp(-dt / _TAU_ATT)
        st.pitch_deg += (target_pitch - st.pitch_deg) * alpha_att
        st.roll_deg  += (target_roll  - st.roll_deg)  * alpha_att

        # ── Position integration ──────────────────────────────────────────
        # Wind moves the airframe over the ground whether or not it is
        # commanded to, so it enters here rather than in the velocity filter:
        # the vehicle's own velocity is through the air, the sum is over the
        # ground, and the difference is what a position controller has to
        # notice and correct.
        #
        # A disarmed vehicle is parked, not flying: it does not get carried
        # away while a client is still connecting. Without this the simulator
        # blows its own start pose into a wall before anything has armed.
        wind_x, wind_y = self.realism.wind_vector() if st.armed else (0.0, 0.0)
        next_x = st.x + (st.vx + wind_x) * dt
        next_y = st.y + (st.vy + wind_y) * dt
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
            st.battery_pct = max(
                0.0, st.battery_pct - dt * self.realism.battery_drain_pct_s)

    def _check_safety_limits(self) -> None:
        """Geofence and battery, the two limits that end a mission by themselves.

        Both are ignored once the vehicle is already on its way down: a
        failsafe that restarts itself during its own recovery never lands.
        """
        st = self.state
        if not st.armed or st.mode in ("LAND", "DISARMED", "CRASHED"):
            return
        if self.realism.battery_exhausted(st.battery_pct) and st.mode != "RTL":
            self._begin_failsafe_return("battery_low")
            return
        if self.realism.outside_geofence(st.x, st.y, st.z):
            if self.realism.geofence_action == "rtl":
                if st.mode != "RTL":
                    self._begin_failsafe_return("geofence")
            elif st.mode != "HOLD" or st.failsafe_reason != "geofence":
                self._enter_failsafe_hold("geofence")

    def _begin_failsafe_return(self, reason: str) -> None:
        """Return home and land, keeping the reason that started it."""
        if self.state.home_x is None:
            self._enter_failsafe_hold(reason)
            return
        self._clear_targets()
        self.state.mode = "RTL"
        self.state.rtl_stage = "climb"
        self.state.failsafe_reason = reason
        self.state.status_message = reason

    def _enter_failsafe_hold(self, reason: str) -> None:
        self._clear_targets()
        self.state.mode = "HOLD"
        self.state.failsafe_reason = reason
        self.state.status_message = reason
        if reason == "control_lease_expired":
            self._clear_lease()

    def _update_position_commands(self, dt: float) -> None:
        st = self.state
        dx, dy, dz = st.target_x - st.x, st.target_y - st.y, st.target_z - st.z
        # Proportional alone parks the vehicle at an offset of exactly the
        # disturbance divided by the gain, so in any wind at all it settles
        # downwind of its setpoint and never arrives. The trim term is what a
        # real position controller uses to cancel a steady disturbance.
        #
        # It accumulates only once the vehicle is near the target and has
        # slowed to a hover. Integrating during the approach is windup: the
        # error there is distance still to travel, not disturbance, and
        # trimming it out throws the vehicle past the waypoint. What is left
        # at a hover is the disturbance, and that is what gets cancelled. The
        # clamp is the second guard, for a vehicle held against an obstacle.
        wind_x, wind_y = self.realism.wind_vector()
        ground_speed = math.hypot(st.vx + wind_x, st.vy + wind_y)
        if (math.hypot(dx, dy) <= _POSITION_TRIM_BAND_M
                and ground_speed <= _POSITION_TRIM_SPEED_MPS):
            st.target_trim_x = clamp(st.target_trim_x + dx * _POSITION_TRIM_GAIN * dt,
                                     -_MAX_POSITION_TRIM_MPS, _MAX_POSITION_TRIM_MPS)
            st.target_trim_y = clamp(st.target_trim_y + dy * _POSITION_TRIM_GAIN * dt,
                                     -_MAX_POSITION_TRIM_MPS, _MAX_POSITION_TRIM_MPS)
        vx = clamp(dx + st.target_trim_x,
                   -st.target_max_speed_mps, st.target_max_speed_mps)
        vy = clamp(dy + st.target_trim_y,
                   -st.target_max_speed_mps, st.target_max_speed_mps)
        mag = math.hypot(vx, vy)
        if mag > st.target_max_speed_mps:
            vx *= st.target_max_speed_mps / mag
            vy *= st.target_max_speed_mps / mag
        yaw_rad = math.radians(st.yaw_deg)
        fwd_x, fwd_y = -math.cos(yaw_rad), math.sin(yaw_rad)
        right_x, right_y = -math.sin(yaw_rad), -math.cos(yaw_rad)
        st.cmd_forward = vx * fwd_x + vy * fwd_y
        st.cmd_right = vx * right_x + vy * right_y
        st.cmd_up = clamp(dz, -2.0, 2.0)
        current = sim_yaw_to_compass_heading(st.yaw_deg)
        error = (st.target_heading_deg - current + 180.0) % 360.0 - 180.0
        st.cmd_yaw_rate = clamp(error * 2.0, -90.0, 90.0)

    def _update_rtl(self) -> None:
        st = self.state
        safe_z = max(st.z, (st.home_z or 0.0) + 2.0)
        if st.rtl_stage == "climb":
            st.target_x, st.target_y, st.target_z = st.x, st.y, safe_z
            st.target_heading_deg = sim_yaw_to_compass_heading(st.yaw_deg)
            if abs(st.z - safe_z) < 0.08:
                st.rtl_stage = "return"
        elif st.rtl_stage == "return":
            st.target_x, st.target_y, st.target_z = st.home_x, st.home_y, safe_z
            if math.hypot(st.x - st.home_x, st.y - st.home_y) < 0.10:
                self._clear_position_target()
                st.vx = st.vy = st.cmd_forward = st.cmd_right = 0.0
                st.target_alt = 0.0
                st.mode = "LAND"

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

    def publish_status(self, *, force: bool = False) -> None:
        if self.status is None:
            return
        values = self.published_fields(force=force)
        if values is not None:
            self.status.setAll(values)

    def published_fields(self, *, force: bool = False) -> dict[str, str] | None:
        """What a client should be able to read right now.

        Split from :meth:`status_fields` because telemetry latency is a
        property of the link, not of the vehicle: the fields are true when
        built and merely arrive late. ``None`` means nothing has come out of
        the delay ring yet, and the previous values stand.
        """
        values = self.status_fields()
        delay = self.realism.telemetry
        if force or delay is None or not delay.enabled:
            return values
        # The vehicle's clock here too: a delay measured in wall time is not
        # reproducible under a fixed-timestep harness, and in the live loop the
        # two are the same thing.
        now = self.clock()
        delay.push(now, values)
        return delay.release(now)

    def status_fields(self) -> dict[str, str]:
        """Build the published telemetry key/value map.

        Split from :meth:`publish_status` so the deterministic test harness can
        assert against the *same* dict the real process publishes instead of
        rebuilding one beside it -- a duplicate builder cannot catch a sign
        error in this one.
        """
        st = self.state
        # The published pose is what the sensors say, not what is true: GPS
        # wanders by its own error, the barometer drifts, the compass is
        # noisy. Physics above this line never sees any of it.
        # Published velocity is velocity over the ground, which is what a GPS
        # or a visual estimator measures. The physics velocity is through the
        # air, so in wind the two differ by exactly the wind, and a client that
        # gates arrival on speed sees the difference.
        wind_x, wind_y = self.realism.wind_vector()
        north_err, east_err, up_err = self.realism.gps_offset_m()
        lat, lon, alt = self.map_to_gps(st.x + east_err, st.y - north_err,
                                        st.z + up_err)
        heading = self.realism.noisy_heading_deg(
            sim_yaw_to_compass_heading(st.yaw_deg))
        altitude_m = self.realism.noisy_altitude_m(st.z)
        vx = self.realism.noisy_velocity_mps(st.vx + wind_x)
        vy = self.realism.noisy_velocity_mps(st.vy + wind_y)
        vz = self.realism.noisy_velocity_mps(st.vz)
        target_x = getattr(self, "target_x", None)
        target_y = getattr(self, "target_y", None)
        if target_x is not None and target_y is not None:
            target_lat, target_lon, target_alt = self.map_to_gps(target_x, target_y, 0.0)
            target_lat_s = f"{target_lat:.7f}"
            target_lon_s = f"{target_lon:.7f}"
            target_alt_s = f"{target_alt:.3f}"
        else:
            target_lat_s = target_lon_s = target_alt_s = ""
        speed    = math.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
        last_cmd = (-1.0 if st.last_command_monotonic is None
                    else self.clock() - st.last_command_monotonic)
        cam_fov_h = Panda3DRenderer.CAM_FOV_H
        cam_w     = self.args.width
        cam_h     = self.args.height
        _half_tan = math.tan(math.radians(cam_fov_h / 2.0))
        cam_fx    = cam_w / (2.0 * _half_tan)
        cam_fov_v = math.degrees(2.0 * math.atan(_half_tan * cam_h / cam_w))
        values = {
            "sim.id":                self.args.id,
            "sim.map":               str(self.map.path),
            "sim.time_s":            f"{self.sim_time_s:.3f}",
            "sim.report_dir":        str(self.report_root),
            "sim.camera_in_geometry": "1" if self.is_blocked(st.x, st.y) else "0",
            "vehicle.type":          "dsim",
            "vehicle.frames":        "map,local_ned",
            "vehicle.accepts_position": "1",
            "vehicle.accepts_velocity": "1",
            "vehicle.accepts_attitude": "0",
            "vehicle.supports_missions": "0",
            "vehicle.setpoint_timeout_s": ("" if float(getattr(self.args, "setpoint_timeout", 0.0)) <= 0 else f"{float(self.args.setpoint_timeout):.3f}"),
            "vehicle.max_speed_mps": f"{self._max_speed_mps():.3f}",
            "vehicle.max_accel_mps2": f"{float(getattr(self.args, 'max_accel_mps2', 4.0)):.3f}",
            "origin.lat_deg":        f"{self.args.origin_lat:.7f}",
            "origin.lon_deg":        f"{self.args.origin_lon:.7f}",
            "origin.alt_m":          f"{self.args.origin_alt:.3f}",
            "home.lat_deg":          "",
            "home.lon_deg":          "",
            "home.alt_m":            "",
            "control.owner":         st.control_owner,
            "control.lease_age_s":   ("" if st.lease_heartbeat_monotonic is None else f"{self.clock() - st.lease_heartbeat_monotonic:.3f}"),
            "control.lease_timeout_s": f"{self._lease_timeout_s():.3f}",
            "setpoint.age_s":        ("" if st.last_setpoint_monotonic is None else f"{self.clock() - st.last_setpoint_monotonic:.3f}"),
            "failsafe.reason":       st.failsafe_reason,
            "command.result.request_id": st.result_request_id,
            "command.result.accepted": "1" if st.result_accepted else "0",
            "command.result.reason": st.result_reason,
            "drone.armed":           "1" if st.armed else "0",
            "drone.mode":            st.mode,
            "drone.x_m":             f"{st.x:.3f}",
            "drone.y_m":             f"{st.y:.3f}",
            "drone.z_m":             f"{altitude_m:.3f}",
            "drone.lat_deg":         f"{lat:.7f}",
            "drone.lon_deg":         f"{lon:.7f}",
            "drone.alt_m":           f"{self.realism.noisy_altitude_m(alt):.3f}",
            "target.lat_deg":        target_lat_s,
            "target.lon_deg":        target_lon_s,
            "target.alt_m":          target_alt_s,
            "drone.roll_deg":        f"{st.roll_deg:.2f}",
            "drone.pitch_deg":       f"{st.pitch_deg:.2f}",
            "drone.heading_deg":     f"{heading:.2f}",
            "drone.compass_deg":     f"{heading:.2f}",
            "drone.vx_mps":          f"{vx:.3f}",
            "drone.vy_mps":          f"{vy:.3f}",
            "drone.vz_mps":          f"{vz:.3f}",
            "drone.speed_mps":       f"{speed:.3f}",
            "drone.battery_pct":     f"{st.battery_pct:.1f}",
            "drone.crashed":         "1" if st.crashed else "0",
            "drone.last_command_s":  f"{last_cmd:.3f}",
            "link.command_count":    str(st.command_count),
            "link.last_command_type": st.last_command_type,
            "status.message":        st.status_message,
            "camera.fov_h_deg":      f"{cam_fov_h:.4f}",
            "camera.fov_v_deg":      f"{cam_fov_v:.4f}",
            "camera.tx_m":           "0.0000",
            "camera.ty_m":           "0.0000",
            "camera.tz_m":           f"{Panda3DRenderer.CAM_Z_OFFSET:.4f}",
            "camera.roll_deg":       f"{st.roll_deg:.4f}",
            "camera.pitch_deg":      f"{Panda3DRenderer.CAM_PITCH + st.pitch_deg:.4f}",
            "camera.yaw_deg":        "0.0000",
            "camera.fx_px":          f"{cam_fx:.4f}",
            "camera.fy_px":          f"{cam_fx:.4f}",
            "camera.cx_px":          f"{cam_w / 2.0:.4f}",
            "camera.cy_px":          f"{cam_h / 2.0:.4f}",
            "camera.width_px":       str(cam_w),
            "camera.height_px":      str(cam_h),
            "camera.fps":            str(self.args.fps),
        }
        values.update(self.realism.status_fields())
        if st.home_x is not None:
            hlat, hlon, halt = self.map_to_gps(st.home_x, st.home_y, st.home_z)
            values["home.lat_deg"] = f"{hlat:.7f}"
            values["home.lon_deg"] = f"{hlon:.7f}"
            values["home.alt_m"] = f"{halt:.3f}"
        return values

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
        except ImportError:
            print("dsim: matplotlib not installed — skipping flight path image",
                  file=sys.stderr)
            return

        sim_map = self.map
        aspect = sim_map.height / max(sim_map.width, 1)
        fig = Figure(figsize=(10, max(4.0, 10.0 * aspect)), facecolor=theme.BG)
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot(111, facecolor=theme.CANVAS)
        ax.set_xlim(0, sim_map.width)
        ax.set_ylim(sim_map.height, 0)  # row 0 at top, matches the UI
        ax.set_aspect("equal")
        ax.tick_params(colors=theme.DIM)
        for sp in ax.spines.values():
            sp.set_color(theme.GRID)
        ax.set_xlabel("X (m)", color=theme.DIM)
        ax.set_ylabel("Y (m)", color=theme.DIM)

        # The same map the monitor and dway's report draw.
        draw_map_axes(ax, sim_map)

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

            ax.plot(xs[0], ys[0], "o", color=theme.OK, markersize=9,
                    markeredgecolor=theme.TEXT, markeredgewidth=1.5,
                    zorder=6, label="Start")
            end_col = theme.DANGER if self.state.crashed else theme.WARN
            ax.plot(xs[-1], ys[-1], "s", color=end_col, markersize=9,
                    markeredgecolor=theme.TEXT, markeredgewidth=1.5,
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
                    "-", color=theme.DIM, linewidth=1.2, zorder=5,
                )
                ax.annotate(
                    f"{int(tt)}s",
                    (tx + px2 * tick_len * 1.5, ty + py2 * tick_len * 1.5),
                    ha="center", va="center",
                    color=theme.DIM, fontsize=6, zorder=5,
                )

        # Crash marker
        if self.crash_pos is not None:
            ax.plot(self.crash_pos[0], self.crash_pos[1], "x",
                    color=theme.DANGER, markersize=18, markeredgewidth=4,
                    zorder=7, label="Crash")

        elapsed = time.monotonic() - self.started
        crashed_str = "  [CRASHED]" if self.state.crashed else ""
        ax.set_title(
            f"dsim {self.args.id}  |  {Path(self.args.map).name}"
            f"  |  {elapsed:.1f} s{crashed_str}  |  run {self.run_id}",
            color=theme.TEXT, fontsize=8, pad=8,
        )
        if positions:
            # Lower right, matching dway's track plot: the default put it over
            # the middle of the map, which is where a flight path usually is.
            ax.legend(loc="lower right", facecolor=theme.BUTTON,
                      edgecolor=theme.GRID, labelcolor=theme.TEXT,
                      fontsize=8, framealpha=0.92)

        path.parent.mkdir(parents=True, exist_ok=True)
        canvas.print_figure(str(path), dpi=150, bbox_inches="tight",
                            facecolor=theme.BG)
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
        apply_theme(self.root)

        self.view = MapView.fitted(sim.map)
        canvas_w, canvas_h = self.view.canvas_size(sim.map)

        self.status_var = tk.StringVar(value="")
        self.command_var = tk.StringVar(value="")
        header = ttk.Frame(self.root, padding=(12, 10), style="Header.TFrame")
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(1, weight=1)
        ttk.Label(header, text=f"dsim  {sim.args.id}", style="Brand.TLabel").grid(
            row=0, column=0, sticky="w", padx=(0, 12))
        ttk.Label(header, text=str(sim.map.path), style="HeaderDim.TLabel").grid(
            row=0, column=1, sticky="w")

        # Two tabs. The header and footer stay outside them: the status line
        # and Reset are about the vehicle whichever page you are reading.
        notebook = ttk.Notebook(self.root)
        notebook.grid(row=1, column=0, sticky="nsew", padx=12, pady=(12, 0))

        world = ttk.Frame(notebook, padding=(0, 8, 0, 0))
        notebook.add(world, text="Map")
        world.columnconfigure(0, weight=1)
        world.rowconfigure(0, weight=1)
        self.canvas = tk.Canvas(world, width=canvas_w, height=canvas_h,
                                bg=theme.CANVAS, highlightthickness=1,
                                highlightbackground=theme.GRID)
        self.canvas.grid(row=0, column=0, sticky="nsew")

        # The realism form is taller than the map; it asks for the map's height
        # and scrolls, so the notebook stays the size the monitor wants.
        self.realism_panel = RealismPanel(notebook, sim, height=canvas_h)
        notebook.add(self.realism_panel.page, text="Realism")

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
        self.realism_panel.refresh()
        self.root.update_idletasks()
        self.root.update()

    def xy(self, x: float, y: float) -> tuple[float, float]:
        return self.view.xy(x, y)

    def draw_static_map(self) -> None:
        self.view.draw_map(self.canvas, self.sim.map)

    def draw_drone(self) -> None:
        for item in self.drone_items:
            self.canvas.delete(item)
        self.drone_items.clear()

        st = self.sim.state
        self.drone_items.extend(self.view.draw_drone(
            self.canvas, st.x, st.y, sim_yaw_to_compass_heading(st.yaw_deg),
            crashed=st.crashed))

        self.status_var.set(self.status_text(st))
        self.command_var.set(f"command: {self.command_text(st)}")

    @staticmethod
    def status_text(st: DroneState) -> str:
        """Displayed status line.

        The displayed heading is the public compass heading, never the
        internal renderer yaw.
        """
        return (
            f"x={st.x:.2f} y={st.y:.2f} z={st.z:.2f}m  "
            f"heading={sim_yaw_to_compass_heading(st.yaw_deg):.1f}°  mode={st.mode}  "
            f"armed={'yes' if st.armed else 'no'}  "
            f"status={st.status_message}"
        )

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

def _resolve_realism(args: argparse.Namespace) -> None:
    """Merge realism settings: an explicit flag beats a profile beats a default.

    The flags outnumber the threshold at which a profile file earns its keep,
    but the flags stay because the command line that produced a report is the
    most reliable record of how it was produced. The file is for the cases
    where a whole environment is reused.
    """
    profile: dict[str, object] = {}
    path = getattr(args, "vehicle_profile", None)
    if path:
        try:
            loaded = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"vehicle profile {path}: {exc}") from exc
        if not isinstance(loaded, dict):
            raise SystemExit(f"vehicle profile {path}: expected a JSON object")
        unknown = sorted(set(loaded) - set(REALISM_DEFAULTS))
        if unknown:
            raise SystemExit(
                f"vehicle profile {path}: unknown settings {', '.join(unknown)}")
        profile = loaded
    for key, fallback in REALISM_DEFAULTS.items():
        if hasattr(args, key):
            continue          # given explicitly on the command line
        setattr(args, key, profile.get(key, fallback))
    try:
        Realism.from_settings(vars(args))
    except ValueError as exc:
        raise SystemExit(f"realism settings: {exc}") from exc


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="dvision2 drone simulator")
    parser.add_argument("--id",       required=True)
    parser.add_argument("--map",      default="assets/maps/maze_001.txt")
    parser.add_argument("--width",    type=int,   default=640)
    parser.add_argument("--height",   type=int,   default=480)
    parser.add_argument("--fps",      type=int,   default=30)
    parser.add_argument("--bufs",     type=int,   default=4)
    parser.add_argument(
        "--scene-preset",
        choices=tuple(SCENE_PRESETS),
        default="representative",
        help="renderer appearance preset (default: representative)",
    )
    parser.add_argument("--cmd-size", type=int,   default=65536)
    parser.add_argument("--start-alt", type=float,
                        help="initial altitude; default uses map drone-height or 1.5")
    parser.add_argument("--start-heading", type=float, default=0.0,
                        help="initial compass heading in degrees (0 north, 90 east)")
    parser.add_argument("--origin-lat", type=float, default=BERLIN_CENTER_LAT_DEG,
                        help="GPS latitude for the center of the map")
    parser.add_argument("--origin-lon", type=float, default=BERLIN_CENTER_LON_DEG,
                        help="GPS longitude for the center of the map")
    parser.add_argument("--origin-alt", type=float, default=BERLIN_CENTER_ALT_M,
                        help="ground altitude in meters for the center of the map")
    parser.add_argument("--setpoint-timeout", type=float, default=2.0,
                        help="GUIDED setpoint failsafe in seconds (default: 2)")
    parser.add_argument("--control-lease-timeout", type=float, default=3.0)
    parser.add_argument("--max-speed-mps", type=float, default=5.0)
    parser.add_argument("--max-accel-mps2", type=float, default=4.0)
    # Realism knobs. Their defaults are deliberately absent here so that a
    # vehicle profile can supply them and an explicit flag can still win; see
    # _resolve_realism below.
    realism = parser.add_argument_group("realism")
    realism.add_argument("--vehicle-profile", default=None,
                         help="JSON file of realism settings; explicit flags win")
    realism.add_argument("--gps", choices=tuple(GPS_MODES),
                         default=argparse.SUPPRESS,
                         help="GPS quality (default: good)")
    realism.add_argument("--gps-noise-m", type=float, default=argparse.SUPPRESS,
                         help="override the GPS mode's position error in metres")
    realism.add_argument("--local-estimator", choices=("on", "off"),
                         default=argparse.SUPPRESS,
                         help="local (VIO/flow) position estimate validity")
    realism.add_argument("--wind-mps", type=float, default=argparse.SUPPRESS,
                         help="steady wind speed in m/s")
    realism.add_argument("--wind-dir-deg", type=float, default=argparse.SUPPRESS,
                         help="compass direction the wind blows from")
    realism.add_argument("--wind-gust-mps", type=float, default=argparse.SUPPRESS,
                         help="gust magnitude added to the steady wind")
    realism.add_argument("--telemetry-latency-ms", type=float,
                         default=argparse.SUPPRESS,
                         help="delay before published telemetry is readable")
    realism.add_argument("--telemetry-jitter-ms", type=float,
                         default=argparse.SUPPRESS,
                         help="random variation added to the telemetry delay")
    realism.add_argument("--sensor-noise", choices=tuple(SENSOR_NOISE_PROFILES),
                         default=argparse.SUPPRESS,
                         help="published heading/altitude/velocity noise profile")
    realism.add_argument("--battery-failsafe-pct", type=float,
                         default=argparse.SUPPRESS,
                         help="battery percentage that triggers RTL then LAND")
    realism.add_argument("--battery-drain-pct-s", type=float,
                         default=argparse.SUPPRESS,
                         help="battery drain per armed second")
    realism.add_argument("--geofence", default=argparse.SUPPRESS,
                         help="boundary box x0,y0,x1,y1[,max_alt_m] in map metres")
    realism.add_argument("--geofence-action", choices=GEOFENCE_ACTIONS,
                         default=argparse.SUPPRESS,
                         help="what a fence breach does (default: hold)")
    realism.add_argument("--realism-seed", type=int, default=argparse.SUPPRESS,
                         help="seed for wind, GPS and sensor noise")
    parser.add_argument("--frames",   type=int,
                        help="run for a fixed number of frames, for smoke tests")
    parser.add_argument("--report-dir", default=None,
                        help="write run artifacts to this exact directory")
    parser.add_argument("--no-ui",    action="store_true",
                        help="disable the top-down simulator UI")
    parser.add_argument("--verbose",  action="store_true")
    args = parser.parse_args(argv)
    validate_id(args.id)
    _resolve_realism(args)
    if args.width <= 0 or args.height <= 0 or args.fps <= 0 or args.bufs <= 0:
        raise SystemExit("width, height, fps, and bufs must be positive")
    if args.setpoint_timeout < 0:
        raise SystemExit("setpoint timeout must be non-negative")
    if min(args.control_lease_timeout, args.max_speed_mps,
           args.max_accel_mps2) <= 0:
        raise SystemExit("lease timeout and motion limits must be positive")
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
