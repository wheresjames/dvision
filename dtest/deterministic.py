"""Test adapter over the production-neutral deterministic DSIM driver."""

from __future__ import annotations

from pathlib import Path

from dsim.headless import FIXED_DT, HeadlessSimulator, shared_renderer
from dtest.backend import BackendCapabilities
from dtest.calibration_scene import (
    CALIBRATION_MAP, FRAME_HEIGHT, FRAME_WIDTH, START_ALT_M,
)


class DeterministicSim(HeadlessSimulator):
    def __init__(self, *, heading_deg: float = 0.0,
                 map_path: Path = CALIBRATION_MAP,
                 altitude_m: float = START_ALT_M,
                 armed: bool = True,
                 scene_preset: str = "legacy") -> None:
        super().__init__(
            heading_deg=heading_deg, map_path=map_path,
            altitude_m=altitude_m, armed=armed,
            width=FRAME_WIDTH, height=FRAME_HEIGHT,
            instance_id="dtest-deterministic",
            scene_preset=scene_preset,
        )

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(deterministic=True, physical_vehicle=False)


__all__ = ["DeterministicSim", "FIXED_DT", "shared_renderer"]
