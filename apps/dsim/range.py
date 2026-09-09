"""Pose-complete ray geometry shared by every range sensor, and the DALG oracle.

One intersection routine serves the scalar range sensors, both LiDAR outputs
and the exact camera-range oracle, so a correction to the geometry cannot
reach one consumer and miss another. Obstacle footprints and heights are read
from the constants the collision test uses, which is what keeps "the sensor
reports clear" and "the physics reports solid" the same statement.

Nothing here consults the renderer's appearance: a lighting or texture preset
cannot move a measured range.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from dsim.dsim import (Panda3DRenderer, _OBSTACLE_HALF_EXTENT_M,
                       _obstacle_height_m)
from dsim.transforms import pinhole_rays
from dsim.sensor_models import directions

#: Ray-by-obstacle intersection runs in blocks of about this many (ray, box)
#: pairs. Sized to stay inside cache: the arithmetic is memory bound, and a
#: block large enough to spill costs more than the extra loop iterations.
_BLOCK_ELEMENTS = 1 << 14


@dataclass(frozen=True)
class Scene:
    """The solid boxes a ray may hit: one per wall or tree cell.

    ``centres`` are cell centres in map metres, ``tops`` each box's height
    above the ground, and ``half`` the shared half extent. The ground is not a
    box; it is the infinite ``z = 0`` plane, tested separately.
    """

    centres: np.ndarray
    tops: np.ndarray
    half: float = _OBSTACLE_HALF_EXTENT_M

    def __len__(self) -> int:
        return len(self.tops)


#: Identity cache: object lists are rebuilt rarely and several sensors cast
#: against the same map every simulated tick. The list itself is retained so a
#: freed list's id cannot be reused underneath a stale entry.
_SCENE_CACHE: dict[int, tuple[list, Scene]] = {}


def cast(scene, pose_world, kind: str, model) -> np.ndarray:
    """True first-surface range along every ray of one capture.

    ``inf`` where the beam leaves the world without hitting anything. The
    sensor's rays are generated in its own axes and rotated by the composed
    pose, so a mount chain, a body attitude and a fixed misalignment all reach
    the geometry through the same path.

    This lives beside the ray service rather than in ``sensor_models`` because
    it is the one part of a range capture that is about *this* world. The
    model around it -- ray directions, noise, the reducer -- is backend-neutral
    and stays there, so a second provider reuses the model and answers this
    from its own geometry. See ``dcmn.sensor_backend.SensorBackend``.
    """
    rays = directions(kind, model) @ np.asarray(pose_world)[:3, :3].T
    return cast_rays(scene, np.asarray(pose_world)[:3, 3], rays,
                     min_range_m=model['min_range_m'], max_range_m=model['max_range_m'])


def scene_geometry(sim_map) -> Scene:
    """A map's solid geometry, in the form the ray service intersects."""
    objects = sim_map.objects
    cached = _SCENE_CACHE.get(id(objects))
    if cached is not None and cached[0] is objects:
        return cached[1]
    solid = [obj for obj in objects if obj.kind in ("wall", "tree")]
    scene = Scene(
        np.array([(obj.x, obj.y) for obj in solid], np.float64).reshape(-1, 2),
        np.array([_obstacle_height_m(obj.kind) for obj in solid], np.float64),
    )
    if len(_SCENE_CACHE) > 8:
        _SCENE_CACHE.clear()
    _SCENE_CACHE[id(objects)] = (objects, scene)
    return scene


def cast_rays(scene: Scene, origin, directions, *, min_range_m: float = 0.0,
              max_range_m: float = math.inf, ground: bool = True,
              tops: bool = True) -> np.ndarray:
    """Distance to the first surface along each ray; ``inf`` where there is none.

    ``origin`` is a map-frame point and ``directions`` an ``(..., 3)`` array of
    map-frame unit vectors, so each caller composes its own extrinsics through
    :mod:`dsim.transforms` first and this routine never assumes a mount. The
    result has the shape of ``directions`` without its last axis.

    ``tops`` selects whether a box's horizontal top face is a surface. It is on
    for every simulated sensor -- a downward rangefinder above a wall must read
    the wall rather than the ground through it -- and off only for the camera
    range oracle, whose side-faces-only behaviour DALG's scores are built on.
    """
    rays = np.asarray(directions, np.float64)
    flat = np.ascontiguousarray(rays.reshape(-1, 3))
    origin = np.asarray(origin, np.float64)
    best = np.full(len(flat), np.inf)
    if ground:
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(flat[:, 2] < 0.0, -origin[2] / flat[:, 2], np.inf)
        best = np.where(np.isfinite(t) & (t >= min_range_m), t, best)
    centres, tops_m = scene.centres, scene.tops
    if len(scene) and math.isfinite(max_range_m):
        # Only a box whose bounding circle is reachable can contribute, so a
        # 7 m sonar in a large maze touches a handful of cells, not every wall.
        near = np.hypot(*(centres - origin[:2]).T) <= max_range_m + scene.half * math.sqrt(2.0)
        centres, tops_m = centres[near], tops_m[near]
    if len(tops_m):
        block = max(1, _BLOCK_ELEMENTS // len(tops_m))
        for start in range(0, len(flat), block):
            stop = start + block
            np.minimum(best[start:stop],
                       _cast_block(flat[start:stop], origin, centres, tops_m,
                                   scene.half, min_range_m, tops),
                       out=best[start:stop])
    best[best > max_range_m] = np.inf
    return best.reshape(rays.shape[:-1])


def _slab(lo, hi, origin, direction):
    """Entry and exit parameters of one axis-aligned slab, per (ray, box).

    ``direction`` is already free of exact zeros, so a ray parallel to the slab
    lands far outside it in both directions rather than dividing by zero.
    """
    t1 = (lo - origin) / direction
    t2 = (hi - origin) / direction
    return np.minimum(t1, t2), np.maximum(t1, t2)


#: Smallest ray component treated as motion along an axis. Substituting it for
#: an exact zero keeps the slab arithmetic finite without a branch.
_MIN_COMPONENT = 1e-12


def _cast_block(rays, origin, centres, tops_m, half, min_range_m, tops):
    dx, dy, dz = (np.where(np.abs(c) < _MIN_COMPONENT, _MIN_COMPONENT, c)
                  for c in (rays[:, 0:1], rays[:, 1:2], rays[:, 2:3]))
    x_lo, x_hi = _slab(centres[None, :, 0] - half, centres[None, :, 0] + half,
                       origin[0], dx)
    y_lo, y_hi = _slab(centres[None, :, 1] - half, centres[None, :, 1] + half,
                       origin[1], dy)
    enter = np.maximum(x_lo, y_lo)
    leave = np.minimum(x_hi, y_hi)
    if tops:
        z_lo, z_hi = _slab(0.0, tops_m[None, :], origin[2], dz)
        enter = np.maximum(enter, z_lo)
        leave = np.minimum(leave, z_hi)
        candidates = (enter,)
    else:
        # Side faces only: the entry face, or the exit face when the ray enters
        # over the top or under the base of the box.
        candidates = (enter, leave)
    inside = enter <= leave
    best = np.full(len(rays), np.inf)
    for t in candidates:
        usable = inside & (t >= min_range_m) & np.isfinite(t)
        if not tops:
            z = origin[2] + t * dz
            usable &= (z >= 0.0) & (z <= tops_m[None, :])
        np.minimum(best, np.where(usable, t, np.inf).min(axis=1), out=best)
    return best


# ---------------------------------------------------------------------------
# Exact camera range oracle
#
# DALG runs this in its own process as a configured algorithm input. It is not
# a published sensor and it is deliberately frozen: ground and box tops stay
# out of it, and its noise pattern repeats per call, because every recorded
# DALG score was measured against exactly this behaviour. New work belongs on
# the sensor models in :mod:`dsim.sensor_models`, which use the shared
# per-capture noise scheme instead.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Pose:
    """Where a sensor is and which way it points, in map coordinates."""

    x_m: float
    y_m: float
    z_m: float
    heading_deg: float
    roll_deg: float = 0.0
    pitch_deg: float = 0.0


@dataclass(frozen=True)
class Intrinsics:
    """A pinhole camera, in pixels."""

    width_px: int
    height_px: int
    fx_px: float
    fy_px: float
    cx_px: float
    cy_px: float


@dataclass(frozen=True)
class RangeConfig:
    name: str = "exact"
    min_range_m: float = 0.15
    max_range_m: float = 150.0
    fov_h_deg: float = Panda3DRenderer.CAM_FOV_H
    noise_std_m: float = 0.0
    dropout_probability: float = 0.0
    quantization_m: float = 0.0
    confidence_model: str = "exact"
    sensor_frame: str = "camera"
    extrinsics_from_frame: str = "camera"
    extrinsics_to_frame: str = "body"
    extrinsics_tx_m: float = 0.0
    extrinsics_ty_m: float = 0.0
    extrinsics_tz_m: float = 0.1
    extrinsics_roll_deg: float = 0.0
    extrinsics_pitch_deg: float = -5.0
    extrinsics_yaw_deg: float = 0.0
    timestamp_offset_s: float = 0.0
    seed: int = 1


RANGE_CONFIGS = {
    "exact": RangeConfig(),
    # Material-independent first models. Their limitations are explicit in the
    # configuration name and report; reflectance models can be versioned later.
    "lidar_flash_short": RangeConfig(
        name="lidar_flash_short", max_range_m=5.0, fov_h_deg=70.0,
        noise_std_m=0.01, dropout_probability=0.02, quantization_m=0.005,
        confidence_model="range_linear", sensor_frame="lidar_flash",
        extrinsics_from_frame="lidar_flash"),
    "lidar_tof_wide": RangeConfig(
        name="lidar_tof_wide", max_range_m=20.0, fov_h_deg=90.0,
        noise_std_m=0.03, dropout_probability=0.05, quantization_m=0.01,
        confidence_model="range_linear", sensor_frame="lidar_tof",
        extrinsics_from_frame="lidar_tof"),
}


def range_config(name: str) -> RangeConfig:
    try:
        return RANGE_CONFIGS[name]
    except KeyError as error:
        raise ValueError(f"unknown range configuration: {name}") from error


def _oracle_rays(pose, intrinsics, rows, columns):
    """The oracle's own spherical ray parameterisation about the fixed mount."""
    pitch = (math.radians(pose.pitch_deg + Panda3DRenderer.CAM_PITCH)
             - np.arctan((rows - intrinsics.cy_px) / intrinsics.fy_px))
    yaw = (math.radians(pose.heading_deg)
           + np.arctan((columns - intrinsics.cx_px) / intrinsics.fx_px))
    cos_pitch = np.cos(pitch)[:, None]
    return np.stack((np.sin(yaw)[None, :] * cos_pitch,
                     -np.cos(yaw)[None, :] * cos_pitch,
                     np.broadcast_to(np.sin(pitch)[:, None], (len(rows), len(columns)))),
                    axis=-1)


def raycast_map(sim_map, pose, intrinsics, *, config: RangeConfig = RangeConfig(),
                stride: int = 1, camera_pose_world=None) -> tuple[np.ndarray, np.ndarray]:
    """Exact per-pixel range for DALG's configured range oracle.

    Ranges are metres, NaN where nothing was hit; confidence is 0-255. When
    ``camera_pose_world`` is supplied the rays are the calibrated optical rays
    of that composed pose, and otherwise the legacy fixed-mount
    heading/pitch parameterisation about ``pose``.
    """
    height, width = intrinsics.height_px, intrinsics.width_px
    ranges = np.full((height, width), np.nan, np.float32)
    confidence = np.zeros((height, width), np.uint8)
    scene = scene_geometry(sim_map)
    rows = np.arange(0, height, stride)
    columns = np.arange(0, width, stride)
    if len(scene) and len(rows) and len(columns):
        if camera_pose_world is not None:
            origin = np.asarray(camera_pose_world)[:3, 3]
            rays = (pinhole_rays(dict(fx_px=intrinsics.fx_px, fy_px=intrinsics.fy_px,
                                      cx_px=intrinsics.cx_px, cy_px=intrinsics.cy_px),
                                 rows, columns)
                    @ np.asarray(camera_pose_world)[:3, :3].T)
        else:
            origin = (pose.x_m, pose.y_m, pose.z_m + Panda3DRenderer.CAM_Z_OFFSET)
            rays = _oracle_rays(pose, intrinsics, rows, columns)
        best = cast_rays(scene, origin, rays, min_range_m=config.min_range_m,
                         max_range_m=config.max_range_m, ground=False, tops=False)
        found = np.isfinite(best)
        grid = np.ix_(rows, columns)
        ranges[grid] = np.where(found, best, np.nan).astype(np.float32)
        confidence[grid] = np.where(found, 255, 0)
    if config.name != "exact":
        rng = np.random.default_rng(config.seed)
        valid = np.isfinite(ranges)
        ranges[valid] += rng.normal(0.0, config.noise_std_m,
                                    int(valid.sum())).astype(np.float32)
        if config.quantization_m > 0:
            ranges[valid] = (np.round(ranges[valid] / config.quantization_m)
                             * config.quantization_m)
        dropped = valid & (rng.random(ranges.shape) < config.dropout_probability)
        ranges[dropped] = np.nan
        if config.confidence_model == "range_linear":
            scaled = 255.0 * (1.0 - np.clip(ranges / config.max_range_m, 0, 1))
            confidence[valid & ~dropped] = scaled[valid & ~dropped].astype(np.uint8)
        confidence[dropped] = 0
    return ranges, confidence
