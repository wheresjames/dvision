"""Pixel to ground: where a detection lies, shared by every detector.

The ray through the detection's pixel is intersected with the ground plane
``z = z0_m``, using the frame's intrinsics and **the camera pose delivered with
the sample** (:attr:`dalg.model.Frame.camera_pose`) -- never the body pose plus
``camera.*`` status keys, which mixes an absolute pitch with a relative mount
offset and biases every range. The camera axes are
:func:`dalg.algo.spatial.camera_axes`, the ones every evidence algorithm uses:
map x east, y south, z up; a positive pitch tilts the camera up.

A detection is **not projectable** -- still published and recorded, with the
reason -- when its ray is at or above the horizon, when the ground point is
beyond ``max_range_m``, or when the frame carried no camera pose.

``sigma_m`` is a deliberately crude footprint uncertainty: the ground size of
the detection's extent at that slant range, combined with the along-range
smear of one pixel at that depression angle. It is a scale for gating and
display, not a covariance.

Every projection keeps its inputs (:func:`inputs`), so :func:`replay` can redo
it from an archived record and must get the same answer.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from dalg.algo.spatial import camera_axes
from dalg.model import Intrinsics, Pose

#: A ray this close to horizontal (its downward direction component) meets the ground nowhere useful.
HORIZON_EPSILON = 1e-6
#: The smallest sine of depression used for the along-range smear, so a ray
#: grazing the ground reports a large sigma instead of an infinite one.
MIN_DEPRESSION_SIN = 0.02


@dataclass(frozen=True)
class Projection:
    projectable: bool
    reason: str = ''
    xy_m: tuple[float, float] | None = None
    range_m: float | None = None
    bearing_deg: float | None = None
    sigma_m: float | None = None

    def as_dict(self) -> dict[str, Any]:
        value = dict(projectable=self.projectable, reason=self.reason)
        if self.projectable:
            value.update(xy_m=[round(self.xy_m[0], 6), round(self.xy_m[1], 6)], range_m=round(self.range_m, 6),
                         bearing_deg=round(self.bearing_deg, 6), sigma_m=round(self.sigma_m, 6))
        return value


def project(cx_px: float, cy_px: float, extent_px: float, camera: Pose | None, intrinsics: Intrinsics, *,
            z0_m: float = 0.0, max_range_m: float = 25.0) -> Projection:
    """Where on the ground plane pixel ``(cx_px, cy_px)`` looks, or why nowhere."""
    if camera is None: return Projection(False, 'no camera pose with the frame')
    u = (float(cx_px) - intrinsics.cx_px) / intrinsics.fx_px
    v = (float(cy_px) - intrinsics.cy_px) / intrinsics.fy_px
    right, down, forward = camera_axes(camera)
    direction = tuple(u * r + v * d + f for r, d, f in zip(right, down, forward))
    height = camera.z_m - float(z0_m)
    if direction[2] >= -HORIZON_EPSILON or height <= 0.0:
        return Projection(False, 'at or above the horizon')
    t = height / -direction[2]
    x = camera.x_m + t * direction[0]
    y = camera.y_m + t * direction[1]
    ground = math.hypot(x - camera.x_m, y - camera.y_m)
    if ground > max_range_m:
        return Projection(False, f'range {ground:.1f} m > {max_range_m:g} m')
    length = math.sqrt(sum(c * c for c in direction))
    slant = t * length
    depression = max(-direction[2] / length, MIN_DEPRESSION_SIN)
    size = max(float(extent_px), 1.0) * slant / intrinsics.fx_px
    smear = slant / (intrinsics.fy_px * depression)
    bearing = math.degrees(math.atan2(x - camera.x_m, -(y - camera.y_m))) % 360.0
    return Projection(True, '', (x, y), ground, bearing, math.hypot(size, smear))


def inputs(cx_px: float, cy_px: float, extent_px: float, camera: Pose | None, intrinsics: Intrinsics, *,
           z0_m: float, max_range_m: float) -> dict[str, Any]:
    """Everything :func:`project` used, in the form the archive keeps."""
    return dict(cx_px=float(cx_px), cy_px=float(cy_px), extent_px=float(extent_px),
                camera=None if camera is None else asdict(camera), intrinsics=asdict(intrinsics),
                z0_m=float(z0_m), max_range_m=float(max_range_m))


def replay(recorded: dict[str, Any]) -> Projection:
    """Project again from an archived ``inputs`` record."""
    camera = recorded.get('camera')
    return project(recorded['cx_px'], recorded['cy_px'], recorded['extent_px'],
                   None if camera is None else Pose(**camera), Intrinsics(**recorded['intrinsics']),
                   z0_m=recorded['z0_m'], max_range_m=recorded['max_range_m'])
