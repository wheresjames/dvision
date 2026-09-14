"""Inverse model for calibrated scan2d returns, entirely in the public frame."""
from dataclasses import dataclass
import math

import numpy as np

from dalg.grid import ObservedGrid
from dalg.model import Result


@dataclass(frozen=True)
class LidarConfig:
    free_log_odds: float = -0.7
    occupied_log_odds: float = 2.5
    min_confidence: int = 1

    def __post_init__(self):
        if not math.isfinite(self.free_log_odds) or self.free_log_odds >= 0:
            raise ValueError('free_log_odds must be finite and negative')
        if not math.isfinite(self.occupied_log_odds) or self.occupied_log_odds <= 0:
            raise ValueError('occupied_log_odds must be finite and positive')
        if not isinstance(self.min_confidence, int) or not 1 <= self.min_confidence <= 255:
            raise ValueError('min_confidence must be an integer in [1, 255]')


def traversed_cells(geometry, start, end):
    """Cells crossed by a segment clipped to the grid's extent and z slab.

    Partition at cell boundaries; interval midpoints visit diagonal rays
    without missing narrow intersections or drawing outside the declared map.
    """
    low = np.array([geometry.origin_x_m, geometry.origin_y_m, geometry.z0_m])
    high = low + [*geometry.extent_m, geometry.dz_m]
    delta = end-start
    enter, leave = 0., 1.
    for axis in range(3):
        if abs(delta[axis]) < 1e-12:
            if not low[axis] <= start[axis] < high[axis]: return set()
        else:
            a, b = sorted(((low[axis]-start[axis])/delta[axis],
                           (high[axis]-start[axis])/delta[axis]))
            enter, leave = max(enter, a), min(leave, b)
    if enter >= leave: return set()
    cuts = [enter, leave]
    for axis, count in ((0, geometry.width), (1, geometry.height)):
        if abs(delta[axis]) < 1e-12: continue
        boundaries = low[axis]+np.arange(1, count)*geometry.cell_m
        ts = (boundaries-start[axis])/delta[axis]
        cuts.extend(ts[(ts > enter) & (ts < leave)])
    cuts = np.unique(cuts)
    points = start[None]+((cuts[:-1]+cuts[1:])/2)[:, None]*delta
    cols = np.floor((points[:, 0]-low[0])/geometry.cell_m).astype(int)
    rows = np.floor((points[:, 1]-low[1])/geometry.cell_m).astype(int)
    return {(int(x), int(y)) for x, y in zip(cols, rows)
            if 0 <= x < geometry.width and 0 <= y < geometry.height}


class LidarInverseModel:
    name, sensors = 'lidar_inverse', ('lidar.scan2d',)

    def __init__(self, geometry, settings=None):
        self.geometry = geometry
        self.config = LidarConfig(**(settings or {}))
        self.grid = ObservedGrid(geometry)
        self.frames = self.rays = 0

    def observe(self, sample):
        if not sample.status or sample.fields is None: return
        calibration = sample.payload['calibration']
        ranges = np.asarray(sample.fields['range_m'], dtype=float)
        confidence = np.asarray(sample.fields['confidence'])
        count = calibration['samples']
        if ranges.shape != (count,) or confidence.shape != (count,):
            raise ValueError('lidar arrays disagree with scan calibration')
        pose = np.asarray(sample.payload['pose_world'], dtype=float)
        angles = np.radians(calibration['angle_min_deg'] +
                            np.arange(count)*calibration['angle_increment_deg'])
        elevation = math.radians(calibration['elevation_deg'])
        if (pose.shape != (4, 4) or not np.isfinite(pose).all()
                or not np.isfinite(angles).all() or not math.isfinite(elevation)):
            raise ValueError('lidar pose and calibration must be finite')
        model = sample.entry['model']
        valid = (np.isfinite(ranges) & (ranges >= model['min_range_m'])
                 & (ranges <= model['max_range_m'])
                 & (confidence >= self.config.min_confidence))
        # NaN/zero-confidence is ambiguous (dropout or no return), never a
        # license to clear a max-range ray. Use only measured valid returns.
        directions = np.column_stack((np.cos(angles)*math.cos(elevation),
            np.sin(angles)*math.cos(elevation), np.full(count, math.sin(elevation))))
        directions = directions @ pose[:3, :3].T
        origin = pose[:3, 3]
        free, occupied = set(), set()
        for index in np.flatnonzero(valid):
            end = origin + ranges[index]*directions[index]
            start = origin + model['min_range_m']*directions[index]
            free.update(traversed_cells(self.geometry, start, end))
            cell = self.geometry.to_cell(*end[:2])
            if cell is not None and self.geometry.z0_m <= end[2] < self.geometry.z0_m+self.geometry.dz_m:
                occupied.add(cell)
        free -= occupied
        self.grid.timestamp_s = sample.sim_time_s
        for cells, delta in ((free, self.config.free_log_odds),
                             (occupied, self.config.occupied_log_odds)):
            if cells:
                xs, ys = zip(*cells)
                self.grid.update(xs, ys, delta)
        self.frames += 1
        self.rays += int(valid.sum())

    def preview(self):
        return Result(self.grid.result(), {'scans': self.frames, 'valid_rays': self.rays})

    finish = preview
