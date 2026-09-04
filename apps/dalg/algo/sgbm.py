from __future__ import annotations

from dataclasses import dataclass
import math
import cv2
import numpy as np

from dalg.grid import LogOddsGrid
from dalg.model import Result
from dalg.algo.spatial import obstacle_band, project_pixels, project_ranges


@dataclass(frozen=True)
class SGBMConfig:
    min_baseline_m: float = .25
    max_baseline_m: float = 4.0
    max_heading_delta_deg: float = 2.0
    max_forward_fraction: float = .15
    num_disparities: int = 128
    block_size: int = 5
    uniqueness_ratio: int = 12
    stride: int = 8
    min_range_m: float = .15
    max_range_m: float = 25.0
    min_height_m: float = .25
    max_height_m: float = 4.5
    occupied_log_odds: float = 4.0
    preview_batch: int = 2
    preview_interval_s: float = .5


class SGBMAlgorithm:
    name = "sgbm"
    sensors = ("rgb",)

    def __init__(self, width_m, height_m, intrinsics, settings=None, **_):
        self.grid_size = (width_m, height_m)
        self.intrinsics = intrinsics
        self.config = SGBMConfig(**(settings or {}))
        if self.config.num_disparities <= 0 or self.config.num_disparities % 16:
            raise ValueError("num_disparities must be a positive multiple of 16")
        if self.config.block_size < 3 or self.config.block_size % 2 == 0:
            raise ValueError("block_size must be odd and at least 3")
        self._matcher = None
        self.start()

    def start(self):
        self.frames = []
        self.range_frames = []
        self._reset_fusion()

    def _reset_fusion(self):
        """Drop everything derived from the capture, keeping the capture."""
        self.grid = LogOddsGrid(*self.grid_size)
        self.processed: set[int] = set()
        self.ranges_fused = 0
        self.pairs = 0
        self.stereo_points = 0
        self.range_points = 0

    def observe(self, frame):
        gray = cv2.cvtColor(np.asarray(frame.rgb, np.uint8), cv2.COLOR_RGB2GRAY)
        self.frames.append((gray, frame.camera_pose))
        if frame.range_m is not None:
            self.range_frames.append((np.asarray(frame.range_m, np.float64),
                                      frame.camera_pose))

    def _matcher_for(self):
        """One matcher, reused. Rebuilding it per pair dominated the run."""
        if self._matcher is None:
            self._matcher = cv2.StereoSGBM_create(
                minDisparity=0, numDisparities=self.config.num_disparities,
                blockSize=self.config.block_size,
                P1=8*self.config.block_size**2, P2=32*self.config.block_size**2,
                uniquenessRatio=self.config.uniqueness_ratio,
                speckleWindowSize=50, speckleRange=2, disp12MaxDiff=1)
        return self._matcher

    def _pair(self, i, poses):
        """The widest baseline to the right of frame ``i``, or None.

        OpenCV expects the second image to be the right-hand camera, so only
        frames on that side qualify; a frame to the left picks this one up as
        its own partner when its turn comes, so no baseline is wasted.
        """
        pose = self.frames[i][1]
        h = math.radians(pose.heading_deg)
        right = np.array([math.cos(h), math.sin(h)])
        forward = np.array([math.sin(h), -math.cos(h)])
        delta = poses[:, :2] - np.array([pose.x_m, pose.y_m])
        baseline = np.hypot(delta[:, 0], delta[:, 1])
        with np.errstate(invalid="ignore", divide="ignore"):
            eligible = ((baseline >= self.config.min_baseline_m)
                        & (baseline <= self.config.max_baseline_m)
                        & (np.abs((poses[:, 2]-pose.heading_deg+180) % 360-180)
                           <= self.config.max_heading_delta_deg)
                        & (np.abs(delta @ forward)
                           <= self.config.max_forward_fraction*baseline)
                        & (delta @ right > 0))
        eligible[i] = False
        if not eligible.any(): return None
        candidates = np.flatnonzero(eligible)
        best = candidates[np.argmax(baseline[candidates])]
        return float(baseline[best]), int(best)

    def _fuse_stereo(self, index, other, baseline):
        reference, pose = self.frames[index]
        disparity = self._matcher_for().compute(
            reference, self.frames[other][0]).astype(np.float32)/16
        step = max(1, self.config.stride)
        sampled = disparity[::step, ::step]
        rows, columns = np.nonzero(sampled > 0)
        if not len(rows): return
        depth_z = self.intrinsics.fx_px*baseline/sampled[rows, columns]
        x, y, z = project_pixels(pose, columns*step, rows*step, depth_z,
                                 self.intrinsics)
        keep = obstacle_band(z, x, y, pose,
                             min_height_m=self.config.min_height_m,
                             max_height_m=self.config.max_height_m,
                             min_range_m=self.config.min_range_m,
                             max_range_m=self.config.max_range_m)
        if not keep.any(): return
        cells = self.grid.cells(x[keep], y[keep])
        self.grid.accumulate(*cells, self.config.occupied_log_odds)
        self.stereo_points += int(keep.sum())

    def _fuse_range(self, ranges, pose):
        """Fuse every sample the sensor actually returned.

        The ray cast leaves NaN wherever it did not sample, so the sampling
        stride never has to be agreed on separately -- and cannot drift out of
        step with the sensor's.
        """
        rows, columns = np.nonzero(np.isfinite(ranges))
        if not len(rows): return
        x, y, z = project_ranges(pose, columns, rows, ranges[rows, columns],
                                 self.intrinsics)
        keep = obstacle_band(z, x, y, pose,
                             min_height_m=self.config.min_height_m,
                             max_height_m=self.config.max_height_m,
                             min_range_m=self.config.min_range_m,
                             max_range_m=self.config.max_range_m)
        if not keep.any(): return
        cells = self.grid.cells(x[keep], y[keep])
        self.grid.accumulate(*cells, self.config.occupied_log_odds)
        self.range_points += int(keep.sum())

    def _process(self, limit=None):
        """Fuse each frame exactly once, in capture order, ``limit`` per call.

        Rebuilding the whole history on every preview made the cost quadratic
        in captured frames; a frame's disparity does not change once it has a
        partner, so it is computed exactly once.
        """
        count = 0
        while self.ranges_fused < len(self.range_frames):
            self._fuse_range(*self.range_frames[self.ranges_fused])
            self.ranges_fused += 1
            count += 1
            if limit is not None and count >= limit: return
        if len(self.frames) < 2: return
        poses = np.array([(p.x_m, p.y_m, p.heading_deg) for _, p in self.frames])
        for index in range(len(self.frames)):
            if index in self.processed: continue
            pair = self._pair(index, poses)
            if pair is None: continue
            self._fuse_stereo(index, pair[1], pair[0])
            self.processed.add(index)
            self.pairs += 1
            count += 1
            if limit is not None and count >= limit: return

    def _result(self):
        return Result(self.grid.result(), {
            "frames": len(self.frames), "stereo_pairs": self.pairs,
            "stereo_points": self.stereo_points,
            "range_frames": len(self.range_frames),
            "range_points": self.range_points})

    def preview(self):
        if not self.frames: return None
        self._process(self.config.preview_batch)
        return self._result()

    def finish(self):
        # Rebuild from the complete capture rather than keeping whatever the
        # previews fused. A preview pairs each frame with the first partner to
        # clear min_baseline_m, because the wider ones have not been captured
        # yet, and _process never revisits a frame it has already paired -- so
        # the reported grid used to depend on how often a front end asked for a
        # preview, which is not a property of the algorithm. One extra pass at
        # the end is linear in captured frames; it was rebuilding on *every*
        # preview that made the cost quadratic.
        self._reset_fusion()
        self._process()
        return self._result()
