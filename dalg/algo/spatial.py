"""Small 2-D geometry and occupancy fusion shared by monocular baselines.

Map coordinates are x east, y south -- rows increase downward, matching the
occupancy raster -- and z up. Level forward at heading ``h`` is therefore
``(sin h, -cos h, 0)`` and a positive pitch tilts the camera up.
"""
from __future__ import annotations

import math
import numpy as np


def bearing(pose, pixel_x: float, intrinsics) -> float:
    return math.radians(pose.heading_deg) + math.atan(
        (pixel_x-intrinsics.cx_px)/intrinsics.fx_px)


def camera_axes(pose):
    """Right, down and forward unit vectors of a camera at ``pose``."""
    h, p = math.radians(pose.heading_deg), math.radians(pose.pitch_deg)
    sh, ch, sp, cp = math.sin(h), math.cos(h), math.sin(p), math.cos(p)
    return ((ch, sh, 0.0), (sh*sp, -ch*sp, -cp), (sh*cp, -ch*cp, sp))


def project_pixels(pose, pixel_x, pixel_y, depth_z, intrinsics):
    """World points for pinhole pixels at perpendicular depth ``depth_z``.

    Stereo disparity yields depth along the optical axis, not distance along
    the viewing ray. Treating the two as interchangeable pulls everything at
    the edge of frame toward the camera -- by 13% at the corner of a 70-degree
    lens -- so the full 3-D reconstruction is done here instead.

    Accepts scalars or arrays; returns ``(x, y, z)`` matching the input shape.
    """
    u = (np.asarray(pixel_x, np.float64)-intrinsics.cx_px)/intrinsics.fx_px
    v = (np.asarray(pixel_y, np.float64)-intrinsics.cy_px)/intrinsics.fy_px
    depth_z = np.asarray(depth_z, np.float64)
    right, down, forward = camera_axes(pose)
    return tuple(origin+depth_z*(u*r+v*d+f) for origin, r, d, f in
                 zip((pose.x_m, pose.y_m, pose.z_m), right, down, forward))


def project_ranges(pose, pixel_x, pixel_y, slant_m, intrinsics):
    """World points for range samples, inverting :func:`dsim.range.raycast_map`.

    The range sensor separates yaw from pitch rather than tracing a pinhole
    ray, so the inverse has to separate them too. It also reports slant range:
    the horizontal component is ``slant * cos(pitch)``, and using the raw
    reading instead over-projects the bottom of the frame by 13%.
    """
    yaw = math.radians(pose.heading_deg)+np.arctan(
        (np.asarray(pixel_x, np.float64)-intrinsics.cx_px)/intrinsics.fx_px)
    pitch = math.radians(pose.pitch_deg)-np.arctan(
        (np.asarray(pixel_y, np.float64)-intrinsics.cy_px)/intrinsics.fy_px)
    slant_m = np.asarray(slant_m, np.float64)
    horizontal = slant_m*np.cos(pitch)
    return (pose.x_m+np.sin(yaw)*horizontal,
            pose.y_m-np.cos(yaw)*horizontal,
            pose.z_m+slant_m*np.sin(pitch))


def obstacle_band(z, xs, ys, pose, *, min_height_m, max_height_m,
                  min_range_m, max_range_m):
    """Which samples are plausible obstacles rather than floor or sky.

    A 2-D occupancy grid has no way to express height, so without this every
    patch of textured floor in front of the vehicle is fused as a wall.
    """
    radial = np.hypot(np.asarray(xs)-pose.x_m, np.asarray(ys)-pose.y_m)
    return ((np.asarray(z) >= min_height_m) & (np.asarray(z) <= max_height_m)
            & (radial >= min_range_m) & (radial <= max_range_m))


def triangulate_xy(pose_a, pixel_a, pose_b, pixel_b, intrinsics,
                   *, min_angle_deg=1.5, max_range_m=25.0):
    """Intersect two horizontal image bearings, rejecting weak geometry."""
    angle_a, angle_b = bearing(pose_a, pixel_a, intrinsics), bearing(
        pose_b, pixel_b, intrinsics)
    da = np.array((math.sin(angle_a), -math.cos(angle_a)))
    db = np.array((math.sin(angle_b), -math.cos(angle_b)))
    cross = da[0]*db[1]-da[1]*db[0]
    separation = abs(math.degrees(math.asin(max(-1.0, min(1.0, cross)))))
    if separation < min_angle_deg or abs(cross) < 1e-6: return None
    delta = np.array((pose_b.x_m-pose_a.x_m, pose_b.y_m-pose_a.y_m))
    ta = (delta[0]*db[1]-delta[1]*db[0])/cross
    tb = (delta[0]*da[1]-delta[1]*da[0])/cross
    if not 0.15 < ta < max_range_m or not 0.15 < tb < max_range_m: return None
    point_a = np.array((pose_a.x_m, pose_a.y_m))+ta*da
    point_b = np.array((pose_b.x_m, pose_b.y_m))+tb*db
    if np.linalg.norm(point_a-point_b) > .5: return None
    return tuple((point_a+point_b)/2)


def ray_cells(x0, y0, x1, y1):
    """Integer cells from a sensor to, but excluding, its endpoint."""
    steps = max(abs(x1-x0), abs(y1-y0))
    if steps <= 1: return np.empty(0, int), np.empty(0, int)
    xs = np.rint(np.linspace(x0, x1, steps+1)[:-1]).astype(int)
    ys = np.rint(np.linspace(y0, y1, steps+1)[:-1]).astype(int)
    return xs, ys


# One clear line of sight should be enough to call a cell free. The scoring
# threshold sits at log-odds -0.619 (FREE_THRESHOLD 0.35), so a delta weaker
# than that leaves a singly-carved cell undecided however obvious it looked --
# it renders at grey 93, indistinguishable from decided free space, and counts
# for nothing. The previous -0.55 fell just short of the line and cost the
# three algorithms that take this default about a tenth of their free-space
# score; plane_sweep, which sets its own -0.7, was never affected.
def fuse_endpoint(grid, pose, point, *, free=-.7, occupied=2.5):
    (x0, x1), (y0, y1) = grid.cells((pose.x_m, point[0]), (pose.y_m, point[1]))
    xs, ys = ray_cells(int(x0), int(y0), int(x1), int(y1))
    grid.update(xs, ys, free)
    grid.update([x1], [y1], occupied)
