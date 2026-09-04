"""Which cells a flight could actually have seen.

Scoring a perception run over the whole map charges every algorithm for the
rooms the vehicle never flew past, which flatters the controls and buries the
difference between the real algorithms. The mask built here is deliberately
generous -- it asks only whether a cell was ever within the camera's horizontal
field of view, in range, and not behind a wall -- so it removes the unanswerable
cells without pretending to model what the algorithm should have resolved.
"""
from __future__ import annotations

import math
import numpy as np

MAX_POSES = 400


def observable_mask(truth, poses, *, fov_h_deg: float = 70.0,
                    max_range_m: float = 25.0, rays: int = 181) -> np.ndarray:
    """Cells within line of sight of any pose, as a boolean array."""
    occupied = truth.occupied
    mask = np.zeros(occupied.shape, bool)
    poses = list(poses)
    if not poses or rays < 2: return mask
    if len(poses) > MAX_POSES:
        keep = np.linspace(0, len(poses)-1, MAX_POSES).round().astype(int)
        poses = [poses[index] for index in dict.fromkeys(keep.tolist())]
    height, width = occupied.shape
    # Half a cell per step: fine enough that a ray cannot tunnel through a wall.
    step_m = truth.cell_m*.5
    distance = np.arange(1, int(max_range_m/step_m)+1)*step_m
    offsets = np.radians(np.linspace(-fov_h_deg/2, fov_h_deg/2, rays))
    for pose in poses:
        yaw = (math.radians(pose.heading_deg)+offsets)[:, None]
        xs = np.floor((pose.x_m+np.sin(yaw)*distance)/truth.cell_m).astype(int)
        ys = np.floor((pose.y_m-np.cos(yaw)*distance)/truth.cell_m).astype(int)
        inside = (xs >= 0) & (ys >= 0) & (xs < width) & (ys < height)
        # A ray that leaves the map does not come back, so truncate at the
        # first outside sample rather than masking sample by sample.
        carried = np.cumprod(inside, axis=1).astype(bool)
        blocked = carried & occupied[np.clip(ys, 0, height-1),
                                     np.clip(xs, 0, width-1)]
        # cumsum - itself is 0 up to and including the first blocking cell,
        # which stays visible: you can see the wall that stops you.
        visible = carried & (np.cumsum(blocked, axis=1)-blocked == 0)
        mask[ys[visible], xs[visible]] = True
    return mask
