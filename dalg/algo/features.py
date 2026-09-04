"""Sparse ORB matching and known-pose bearing triangulation."""
from __future__ import annotations
from dataclasses import dataclass
import math
import cv2
import numpy as np
from dalg.grid import LogOddsGrid
from dalg.model import Result
from dalg.algo.spatial import fuse_endpoint, triangulate_xy


@dataclass(frozen=True)
class FeatureConfig:
    max_features: int = 1000
    min_baseline_m: float = .35
    max_baseline_m: float = 4.0
    max_heading_delta_deg: float = 35.0
    ratio_test: float = .72
    min_parallax_deg: float = 1.5
    max_range_m: float = 25.0
    keyframe_distance_m: float = .35


class FeatureTriangulationAlgorithm:
    name, sensors = "feature_triangulation", ("rgb",)
    def __init__(self, width_m, height_m, intrinsics, settings=None, **_):
        self.size, self.intrinsics = (width_m, height_m), intrinsics
        self.config = FeatureConfig(**(settings or {})); self.start()
    def start(self):
        self.grid = LogOddsGrid(*self.size); self.frames = []; self.matches = self.points = 0
        self.orb = cv2.ORB_create(nfeatures=self.config.max_features)
    @staticmethod
    def _delta(a, b): return (a-b+180)%360-180
    def observe(self, frame):
        gray = cv2.cvtColor(np.asarray(frame.rgb, np.uint8), cv2.COLOR_RGB2GRAY)
        if self.frames:
            old = self.frames[-1][2]
            if math.hypot(frame.camera_pose.x_m-old.x_m, frame.camera_pose.y_m-old.y_m) < self.config.keyframe_distance_m: return
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        if descriptors is None: return
        for old_points, old_descriptors, old_pose in reversed(self.frames[-8:]):
            baseline = math.hypot(frame.camera_pose.x_m-old_pose.x_m, frame.camera_pose.y_m-old_pose.y_m)
            if not self.config.min_baseline_m <= baseline <= self.config.max_baseline_m: continue
            if abs(self._delta(frame.camera_pose.heading_deg, old_pose.heading_deg)) > self.config.max_heading_delta_deg: continue
            pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(old_descriptors, descriptors, k=2)
            for pair in pairs:
                # knnMatch returns min(k, available) matches per row, so a
                # near-textureless frame with a single descriptor yields rows
                # of one and the ratio test has nothing to compare against.
                if len(pair) < 2: continue
                best, second = pair
                if best.distance >= self.config.ratio_test*second.distance: continue
                self.matches += 1
                point = triangulate_xy(old_pose, old_points[best.queryIdx].pt[0],
                                       frame.camera_pose, keypoints[best.trainIdx].pt[0],
                                       self.intrinsics,
                                       min_angle_deg=self.config.min_parallax_deg,
                                       max_range_m=self.config.max_range_m)
                if point is not None:
                    fuse_endpoint(self.grid, frame.camera_pose, point); self.points += 1
            break
        self.frames.append((keypoints, descriptors, frame.camera_pose))
    def _result(self): return Result(self.grid.result(), {"keyframes": len(self.frames), "matches": self.matches, "triangulated_points": self.points})
    def preview(self): return self._result()
    def finish(self): return self._result()
