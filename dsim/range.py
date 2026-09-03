"""Exact renderer-aligned range utilities and sensor configuration types."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from dsim.dsim import Panda3DRenderer

EXACT_BACKEND = "raycast"
EXACT_BACKEND_REASON = (
    "Panda3D headless depth readback returned only far-plane samples, so it "
    "failed the availability gate; the exact ray-cast fallback is selected."
)


@dataclass(frozen=True)
class Pose:
    """Where a sensor is and which way it points, in map coordinates.

    Defined here because :func:`raycast_map` is what consumes it: the geometry
    types belong with the geometry, not with whichever client happens to be
    asking this week.
    """

    x_m: float
    y_m: float
    z_m: float
    heading_deg: float
    roll_deg: float = 0.0
    pitch_deg: float = 0.0


@dataclass(frozen=True)
class Intrinsics:
    """A pinhole camera, in pixels. Matches the ``camera.*`` status keys."""

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


def raycast_map(sim_map, pose, intrinsics, *, config: RangeConfig = RangeConfig(),
                stride: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Ray-cast walls and collision tree cells; deterministic exact fallback."""
    height, width = intrinsics.height_px, intrinsics.width_px
    ranges = np.full((height, width), np.nan, np.float32)
    confidence = np.zeros((height, width), np.uint8)
    heading = math.radians(pose.heading_deg)
    pitch0 = math.radians(pose.pitch_deg + Panda3DRenderer.CAM_PITCH)
    obstacles = [obj for obj in sim_map.objects if obj.kind in ("wall", "tree")]
    for py in range(0, height, stride):
        v = (py - intrinsics.cy_px) / intrinsics.fy_px
        pitch = pitch0 - math.atan(v)
        cp = math.cos(pitch)
        for px in range(0, width, stride):
            u = (px - intrinsics.cx_px) / intrinsics.fx_px
            yaw = heading + math.atan(u)
            dx, dy, dz = math.sin(yaw) * cp, -math.cos(yaw) * cp, math.sin(pitch)
            best = config.max_range_m
            for obj in obstacles:
                zmax = (Panda3DRenderer.WALL_H if obj.kind == "wall"
                        else Panda3DRenderer.TREE_MODEL_H)
                for axis_origin, axis_dir, lo, hi, other_origin, other_dir, olo, ohi in (
                    (pose.x_m, dx, obj.x - .5, obj.x + .5,
                     pose.y_m, dy, obj.y - .5, obj.y + .5),
                    (pose.y_m, dy, obj.y - .5, obj.y + .5,
                     pose.x_m, dx, obj.x - .5, obj.x + .5),
                ):
                    if abs(axis_dir) < 1e-9:
                        continue
                    for boundary in (lo, hi):
                        t = (boundary - axis_origin) / axis_dir
                        if not (config.min_range_m <= t < best):
                            continue
                        other = other_origin + t * other_dir
                        z = pose.z_m + Panda3DRenderer.CAM_Z_OFFSET + t * dz
                        if olo <= other <= ohi and 0.0 <= z <= zmax:
                            best = t
            if best < config.max_range_m:
                ranges[py, px] = best
                confidence[py, px] = 255
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
