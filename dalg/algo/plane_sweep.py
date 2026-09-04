"""Small incremental metric-pose plane-sweep baseline."""
from __future__ import annotations

from dataclasses import dataclass
import math
import cv2
import numpy as np

from dalg.grid import LogOddsGrid
from dalg.model import Result
from dalg.algo.spatial import ray_cells


@dataclass(frozen=True)
class PlaneSweepConfig:
    depth_min_m: float = .5
    depth_max_m: float = 15.0
    depth_steps: int = 16
    min_baseline_m: float = .5
    max_baseline_m: float = 2.5
    max_heading_delta_deg: float = 15.0
    min_keyframe_distance_m: float = .5
    max_keyframes: int = 256
    match_scale: float = .5
    window_px: int = 5
    min_texture_std: float = .025
    confidence_margin: float = .18
    max_cost: float = .15
    stride: int = 4
    free_log_odds: float = -.7
    occupied_log_odds: float = 4.0
    min_height_m: float = .25
    max_height_m: float = 4.5
    preview_batch: int = 2
    preview_interval_s: float = .75


class PlaneSweepAlgorithm:
    name = "plane_sweep"
    sensors = ("rgb",)

    def __init__(self, width_m, height_m, intrinsics, settings=None, **_):
        self.size, self.intrinsics = (width_m, height_m), intrinsics
        self.config = PlaneSweepConfig(**(settings or {}))
        if self.config.depth_steps < 4: raise ValueError("depth_steps must be at least 4")
        if self.config.depth_min_m <= 0 or self.config.depth_max_m <= self.config.depth_min_m:
            raise ValueError("invalid depth range")
        if not 0 < self.config.match_scale <= 1: raise ValueError("match_scale must be in (0, 1]")
        self.frames = []
        self.grid = LogOddsGrid(*self.size)
        self.processed: set[int] = set()
        self.accepted_points = 0

    def start(self):
        self.frames.clear(); self.processed.clear()
        self.grid = LogOddsGrid(*self.size); self.accepted_points = 0

    @staticmethod
    def _angle_delta(a, b): return (a-b+180) % 360 - 180

    def observe(self, frame):
        if len(self.frames) >= self.config.max_keyframes: return
        if self.frames:
            previous = self.frames[-1][1]
            distance = math.hypot(frame.camera_pose.x_m-previous.x_m,
                                  frame.camera_pose.y_m-previous.y_m)
            if distance < self.config.min_keyframe_distance_m: return
        gray = cv2.cvtColor(np.asarray(frame.rgb, np.uint8), cv2.COLOR_RGB2GRAY)
        if self.config.match_scale != 1:
            gray = cv2.resize(gray, None, fx=self.config.match_scale,
                              fy=self.config.match_scale, interpolation=cv2.INTER_AREA)
        self.frames.append((gray.astype(np.float32) / 255.0, frame.camera_pose))

    def _neighbor(self, index):
        pose = self.frames[index][1]
        choices = []
        for other_index, (_, other) in enumerate(self.frames):
            if other_index == index: continue
            baseline = math.hypot(other.x_m-pose.x_m, other.y_m-pose.y_m)
            heading_delta = abs(self._angle_delta(other.heading_deg, pose.heading_deg))
            if (self.config.min_baseline_m <= baseline <= self.config.max_baseline_m
                    and heading_delta <= self.config.max_heading_delta_deg):
                choices.append((baseline, other_index))
        return max(choices, default=(0, None))[1]

    def _depth(self, index, neighbor):
        ref, pose = self.frames[index]
        other, other_pose = self.frames[neighbor]
        h, w = ref.shape
        yy, xx = np.mgrid[0:h, 0:w]
        scale = self.config.match_scale
        fx, cx = self.intrinsics.fx_px*scale, self.intrinsics.cx_px*scale
        depths = 1.0 / np.linspace(1/self.config.depth_min_m,
                                  1/self.config.depth_max_m,
                                  self.config.depth_steps)
        volume = np.empty((len(depths), h, w), np.float32)
        heading = math.radians(pose.heading_deg)
        rays = np.arctan((xx-cx)/fx) + heading
        other_heading = math.radians(other_pose.heading_deg)
        for k, depth in enumerate(depths):
            wx = pose.x_m + np.sin(rays)*depth
            wy = pose.y_m - np.cos(rays)*depth
            dx, dy = wx-other_pose.x_m, wy-other_pose.y_m
            forward = dx*np.sin(other_heading) - dy*np.cos(other_heading)
            right = dx*np.cos(other_heading) + dy*np.sin(other_heading)
            px = cx + fx*right/np.maximum(forward, 1e-6)
            inside = (forward > 0) & (px >= 0) & (px < w-1)
            warped = cv2.remap(other, px.astype(np.float32), yy.astype(np.float32),
                               cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                               borderValue=1)
            volume[k] = cv2.boxFilter(abs(ref-warped), -1,
                                      (self.config.window_px, self.config.window_px))
            volume[k][~inside] = 1
        best = np.argmin(volume, axis=0)
        gy, gx = np.mgrid[0:h, 0:w]
        first = volume[best, gy, gx]
        runner = volume.copy()
        for offset in (-1, 0, 1):
            runner[np.clip(best+offset, 0, len(depths)-1), gy, gx] = np.inf
        second = runner.min(axis=0)
        mean = cv2.boxFilter(ref, -1, (self.config.window_px, self.config.window_px))
        variance = np.maximum(cv2.boxFilter(
            ref*ref, -1, (self.config.window_px, self.config.window_px))-mean*mean, 0)
        confident = ((second-first)/np.maximum(second, 1e-6) >=
                     self.config.confidence_margin)
        confident &= first <= self.config.max_cost
        confident &= np.sqrt(variance) >= self.config.min_texture_std
        return np.where(confident, depths[best], np.nan)

    def _fuse(self, depth, pose):
        """Carve free space to each accepted depth and mark its endpoint.

        The sweep hypothesises horizontal radial distance, so the row of a
        pixel only enters here as the ray's pitch -- which is what says whether
        the sample is a wall or the patch of floor a metre ahead.
        """
        scale = self.config.match_scale
        fx, cx = self.intrinsics.fx_px*scale, self.intrinsics.cx_px*scale
        fy, cy = self.intrinsics.fy_px*scale, self.intrinsics.cy_px*scale
        pitch = math.radians(pose.pitch_deg)
        (origin_x,), (origin_y,) = self.grid.cells([pose.x_m], [pose.y_m])
        for py in range(0, depth.shape[0], self.config.stride):
            ray_pitch = pitch - math.atan((py-cy)/fy)
            rise = math.tan(ray_pitch)
            for px in range(0, depth.shape[1], self.config.stride):
                value = float(depth[py, px])
                if not np.isfinite(value): continue
                height = pose.z_m + value*rise
                if not self.config.min_height_m <= height <= self.config.max_height_m:
                    continue
                yaw = math.radians(pose.heading_deg) + math.atan((px-cx)/fx)
                (end_x,), (end_y,) = self.grid.cells(
                    [pose.x_m+math.sin(yaw)*value], [pose.y_m-math.cos(yaw)*value])
                xs, ys = ray_cells(origin_x, origin_y, end_x, end_y)
                self.grid.update(xs, ys, self.config.free_log_odds)
                self.grid.update([end_x], [end_y], self.config.occupied_log_odds)
                self.accepted_points += 1

    def _process(self, limit=None):
        count = 0
        for index, (_, pose) in enumerate(self.frames):
            if index in self.processed: continue
            neighbor = self._neighbor(index)
            if neighbor is None: continue
            self._fuse(self._depth(index, neighbor), pose)
            self.processed.add(index); count += 1
            if limit is not None and count >= limit: break

    def _result(self):
        return Result(self.grid.result(), {"frames": len(self.frames),
                      "frames_with_depth": len(self.processed),
                      "depth_hypotheses": self.config.depth_steps,
                      "accepted_points": self.accepted_points})

    def preview(self):
        self._process(self.config.preview_batch)
        return self._result() if self.frames else None

    def finish(self):
        self._process()
        return self._result()
