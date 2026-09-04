"""Lucas-Kanade tracks triangulated with known metric poses."""
from __future__ import annotations
from dataclasses import dataclass
import math
import cv2
import numpy as np
from dalg.grid import LogOddsGrid
from dalg.model import Result
from dalg.algo.spatial import fuse_endpoint, triangulate_xy


@dataclass(frozen=True)
class OpticalFlowConfig:
    max_corners: int = 800
    quality_level: float = .02
    min_distance_px: float = 7.0
    min_baseline_m: float = .3
    min_parallax_deg: float = 1.5
    max_range_m: float = 25.0


class OpticalFlowTriangulationAlgorithm:
    name, sensors = "optical_flow_triangulation", ("rgb",)
    def __init__(self, width_m, height_m, intrinsics, settings=None, **_):
        self.size, self.intrinsics = (width_m, height_m), intrinsics
        self.config = OpticalFlowConfig(**(settings or {})); self.start()
    def start(self):
        self.grid = LogOddsGrid(*self.size); self.previous = None
        self.tracks = self.points = self.frames = 0
    def observe(self, frame):
        gray = cv2.cvtColor(np.asarray(frame.rgb, np.uint8), cv2.COLOR_RGB2GRAY)
        if self.previous is not None:
            old_gray, old_pose = self.previous
            baseline = math.hypot(frame.camera_pose.x_m-old_pose.x_m, frame.camera_pose.y_m-old_pose.y_m)
            if baseline >= self.config.min_baseline_m:
                corners = cv2.goodFeaturesToTrack(old_gray, self.config.max_corners,
                    self.config.quality_level, self.config.min_distance_px)
                if corners is not None:
                    moved, status, _ = cv2.calcOpticalFlowPyrLK(old_gray, gray, corners, None)
                    for old, new, valid in zip(corners[:, 0], moved[:, 0], status[:, 0]):
                        if not valid: continue
                        self.tracks += 1
                        point = triangulate_xy(old_pose, old[0], frame.camera_pose, new[0],
                            self.intrinsics, min_angle_deg=self.config.min_parallax_deg,
                            max_range_m=self.config.max_range_m)
                        if point is not None:
                            fuse_endpoint(self.grid, frame.camera_pose, point); self.points += 1
                self.previous = (gray, frame.camera_pose)
        else: self.previous = (gray, frame.camera_pose)
        self.frames += 1
    def _result(self): return Result(self.grid.result(), {"frames": self.frames, "tracks": self.tracks, "triangulated_points": self.points})
    def preview(self): return self._result()
    def finish(self): return self._result()
