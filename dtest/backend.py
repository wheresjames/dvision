"""Normalized vehicle-backend contract shared by DSIM and future adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class BackendCapabilities:
    body_velocity: bool = True
    compass_heading: bool = True
    rgb_video: bool = True
    deterministic: bool = False
    physical_vehicle: bool = False


@runtime_checkable
class VehicleBackend(Protocol):
    def capabilities(self) -> BackendCapabilities: ...
    def arm(self) -> None: ...
    def disarm(self) -> None: ...
    def zero(self) -> None: ...
    def send_body_velocity(self, forward_mps: float, right_mps: float,
                           up_mps: float, yaw_rate_dps: float) -> None: ...
    def read_telemetry(self) -> dict: ...
    def read_frame(self, *, newer_than: int | None = None,
                   timeout: float = 5.0) -> tuple[int, np.ndarray]: ...
