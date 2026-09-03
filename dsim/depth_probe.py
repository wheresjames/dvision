"""Measured selection of the exact-range backend.

The scored ``exact`` range core may read the renderer's depth buffer back, or it
may ray-cast the map explicitly. Which one production uses is decided by
measurement rather than assumption: this module runs the depth-buffer readback
against the ray-cast reference and reports whether it clears the accuracy and
headless-availability gates recorded in ``dsim/range_backend.v1.json``.

The ray-cast implementation in :mod:`dsim.range` is the correctness reference in
either case, so the probe is also the place that proves the Panda/map chirality
correction on an asymmetric fixture.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dsim.dsim import DroneState, Panda3DRenderer, compass_heading_to_sim_yaw
from dsim.range import RangeConfig, raycast_map
from dvision2_common import load_map

#: Distances, in metres, at which absolute range error is characterised.
GATE_DEPTHS_M = (1.0, 5.0, 10.0, 30.0, 100.0)

#: Readback is accepted only if error stays within this bound through 30 m.
GATE_MAX_ERROR_THROUGH_30M_M = 0.02

#: ...and if the p99 of the absolute error over all finite samples clears this.
GATE_P99_ERROR_M = 0.05

#: Beyond this range the gate is characterised but not enforced.
GATE_ENFORCED_THROUGH_M = 30.0

RECORD_PATH = Path(__file__).resolve().parent / "range_backend.v1.json"


@dataclass(frozen=True)
class BackendProbe:
    """Outcome of one depth-backend measurement."""

    backend: str
    reason: str
    available: bool
    max_error_through_30m_m: float | None
    p99_error_m: float | None
    per_depth_error_m: dict[str, float | None]

    def as_record(self, schema_version: int = 1) -> dict:
        return {
            "schema_version": schema_version,
            "backend": self.backend,
            "reason": self.reason,
            "gate": {
                "depths_m": list(GATE_DEPTHS_M),
                "max_error_through_30m_m": GATE_MAX_ERROR_THROUGH_30M_M,
                "p99_error_m": GATE_P99_ERROR_M,
            },
            "measured": {
                "available": self.available,
                "max_error_through_30m_m": self.max_error_through_30m_m,
                "p99_error_m": self.p99_error_m,
                "per_depth_error_m": self.per_depth_error_m,
            },
        }


class _Intrinsics:
    """Minimal pinhole intrinsics matching the renderer's 70 degree lens."""

    def __init__(self, width: int, height: int) -> None:
        self.width_px, self.height_px = width, height
        half_tan = math.tan(math.radians(Panda3DRenderer.CAM_FOV_H / 2.0))
        self.fx_px = self.fy_px = width / (2.0 * half_tan)
        self.cx_px, self.cy_px = width / 2.0, height / 2.0


class _Pose:
    def __init__(self, x: float, y: float, z: float, heading_deg: float) -> None:
        self.x_m, self.y_m, self.z_m = x, y, z
        self.heading_deg = heading_deg
        self.roll_deg = self.pitch_deg = 0.0


def read_record(path: Path = RECORD_PATH) -> dict:
    """Load the committed backend-selection record."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def selected_backend(path: Path = RECORD_PATH) -> str:
    return read_record(path)["backend"]


def probe_backend(map_path: Path, renderer: Panda3DRenderer, *,
                  width: int, height: int,
                  heading_deg: float = 90.0,
                  altitude_m: float = 1.5) -> BackendProbe:
    """Measure depth-buffer readback against the ray-cast reference.

    Returns the backend production should use. A readback that raises, or that
    misses the accuracy gate, selects ``raycast`` with the measured reason
    attached -- never a silent fallback.
    """
    sim_map = load_map(Path(map_path))
    intrinsics = _Intrinsics(width, height)
    pose = _Pose(sim_map.start_x, sim_map.start_y, altitude_m, heading_deg)
    state = DroneState(pose.x_m, pose.y_m, altitude_m,
                       yaw_deg=compass_heading_to_sim_yaw(heading_deg),
                       armed=True, mode="GUIDED")
    empty: dict[str, float | None] = {f"{d:g}": None for d in GATE_DEPTHS_M}
    try:
        measured = renderer.render_range(state)
    except Exception as exc:
        return BackendProbe(
            "raycast",
            f"depth-buffer readback is unavailable headless: {exc}",
            False, None, None, empty,
        )

    reference, _ = raycast_map(sim_map, pose, intrinsics,
                               config=RangeConfig(), stride=1)
    valid = np.isfinite(measured) & np.isfinite(reference)
    if not valid.any():
        return BackendProbe(
            "raycast",
            "depth-buffer readback produced no sample the reference also saw",
            False, None, None, empty,
        )

    error = np.abs(measured[valid] - reference[valid])
    distance = reference[valid]
    near = distance <= GATE_ENFORCED_THROUGH_M
    max_near = float(error[near].max()) if near.any() else None
    p99 = float(np.percentile(error, 99))

    per_depth: dict[str, float | None] = {}
    for depth in GATE_DEPTHS_M:
        band = np.abs(distance - depth) <= 0.25
        per_depth[f"{depth:g}"] = float(error[band].max()) if band.any() else None

    accepted = (max_near is not None
                and max_near <= GATE_MAX_ERROR_THROUGH_30M_M
                and p99 <= GATE_P99_ERROR_M)
    if accepted:
        reason = (f"depth-buffer readback met the gate: {max_near:.4f} m through "
                  f"{GATE_ENFORCED_THROUGH_M:g} m, p99 {p99:.4f} m")
        return BackendProbe("depth_buffer", reason, True, max_near, p99, per_depth)
    reason = (f"depth-buffer readback missed the gate: {max_near} m through "
              f"{GATE_ENFORCED_THROUGH_M:g} m, p99 {p99:.4f} m; the ray-cast "
              f"reference is used instead")
    return BackendProbe("raycast", reason, True, max_near, p99, per_depth)


def chirality_reference(map_path: Path, *, width: int = 64, height: int = 48,
                        altitude_m: float = 1.5) -> dict[str, float]:
    """Centre-pixel ray-cast range for each cardinal heading on a fixture.

    An asymmetric fixture makes every heading a different distance, so a mirrored
    or transposed map/render convention cannot pass by coincidence.
    """
    sim_map = load_map(Path(map_path))
    intrinsics = _Intrinsics(width, height)
    centre_y, centre_x = height // 2, width // 2
    result = {}
    for name, heading in (("north", 0.0), ("east", 90.0),
                          ("south", 180.0), ("west", 270.0)):
        pose = _Pose(sim_map.start_x, sim_map.start_y, altitude_m, heading)
        ranges, _ = raycast_map(sim_map, pose, intrinsics, stride=1)
        result[name] = float(ranges[centre_y, centre_x])
    return result
