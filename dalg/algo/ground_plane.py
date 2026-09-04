"""Fast floor-boundary projection using known camera height and attitude."""
from __future__ import annotations
from dataclasses import dataclass
import math
import cv2
import numpy as np
from dalg.grid import LogOddsGrid
from dalg.model import Result
from dalg.algo.spatial import bearing, fuse_endpoint


@dataclass(frozen=True)
class GroundPlaneConfig:
    # Used only when the simulator has not published the camera's attitude and
    # mounting height; the published values are authoritative when present.
    camera_pitch_down_deg: float = 5.0
    camera_height_offset_m: float = .1
    column_stride: int = 4
    edge_threshold: float = 28.0
    min_range_m: float = .3
    max_range_m: float = 20.0


class GroundPlaneAlgorithm:
    name, sensors = "ground_plane", ("rgb",)

    def __init__(self, width_m, height_m, intrinsics, settings=None, **_):
        self.size, self.intrinsics = (width_m, height_m), intrinsics
        self.config = GroundPlaneConfig(**(settings or {})); self.start()

    def start(self): self.grid = LogOddsGrid(*self.size); self.frames = self.points = 0

    def _camera(self, frame):
        """Camera attitude and height, preferring what the simulator publishes."""
        if frame.camera is not None:
            return frame.camera, frame.camera.pitch_deg, frame.camera.z_m
        return (frame.pose,
                frame.pose.pitch_deg-self.config.camera_pitch_down_deg,
                frame.pose.z_m+self.config.camera_height_offset_m)

    def observe(self, frame):
        gray = cv2.cvtColor(np.asarray(frame.rgb, np.uint8), cv2.COLOR_RGB2GRAY)
        gradient = abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
        horizon = int(self.intrinsics.cy_px)
        pose, pitch_deg, camera_height = self._camera(frame)
        camera_height = max(.1, camera_height)
        for px in range(0, gray.shape[1], self.config.column_stride):
            column = gradient[horizon:, px]
            if not len(column): continue
            py = horizon+int(np.argmax(column))
            if gradient[py, px] < self.config.edge_threshold: continue
            down = -math.radians(pitch_deg)
            down += math.atan((py-self.intrinsics.cy_px)/self.intrinsics.fy_px)
            if down <= 0: continue
            distance = camera_height/math.tan(down)
            if not self.config.min_range_m <= distance <= self.config.max_range_m: continue
            angle = bearing(pose, px, self.intrinsics)
            point = (pose.x_m+math.sin(angle)*distance,
                     pose.y_m-math.cos(angle)*distance)
            fuse_endpoint(self.grid, pose, point); self.points += 1
        self.frames += 1

    def _result(self): return Result(self.grid.result(), {"frames": self.frames, "projected_boundaries": self.points})
    def preview(self): return self._result()
    def finish(self): return self._result()
