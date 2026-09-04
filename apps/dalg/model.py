from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Pose:
    x_m: float
    y_m: float
    z_m: float
    heading_deg: float
    roll_deg: float = 0.0
    pitch_deg: float = 0.0


@dataclass(frozen=True)
class Intrinsics:
    width_px: int
    height_px: int
    fx_px: float
    fy_px: float
    cx_px: float
    cy_px: float


@dataclass(frozen=True)
class Frame:
    frame_id: int
    timestamp_s: float
    rgb: Any
    pose: Pose
    range_m: Any = None
    range_confidence: Any = None
    camera: Pose | None = None

    @property
    def camera_pose(self) -> Pose:
        """Where the lens is, which is not where the vehicle datum is.

        The simulator publishes the camera's mounting offset and its absolute
        attitude separately from the body pose. Geometry that reads the body
        pose instead silently drops the fixed downward tilt and the mast
        height, which biases every projected range.
        """
        return self.pose if self.camera is None else self.camera


@dataclass
class Result:
    grid: Any
    diagnostics: dict[str, Any] = field(default_factory=dict)
