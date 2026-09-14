"""Offline evaluation helpers: truth rasters, scores, visibility and oracle controls.

Evaluator-only. These read simulator world files, which is exactly what the
operational modules must never do: dalg and dnav run from sensor samples, the
labelled ideal pose and the session context, and a truth-independence test
checks that none of this module is reachable from them. The helpers live here so
the oracle checks that used to run inside dalg keep running -- against recorded
or in-process outputs -- without a secret parallel truth path in the runner.
"""
from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
from PIL import Image

from dalg.grid import OccupancyGrid
from dalg.model import Result

# -- truth -------------------------------------------------------------------


def ground_truth(sim_map, cell_m: float = .25) -> OccupancyGrid:
    """Rasterise a world's obstacle cells at the occupancy grid's resolution."""
    grid = OccupancyGrid.unknown(sim_map.width, sim_map.height, cell_m)
    grid.observed[:] = True
    grid.probabilities[:] = .05
    height, width = grid.probabilities.shape
    for obj in sim_map.objects:
        if obj.kind not in ("wall", "tree"):
            continue
        # Unit cells centred on their coordinate; flooring keeps a negative
        # coordinate from wrapping to the far edge.
        x0, y0 = math.floor((obj.x-.5)/cell_m), math.floor((obj.y-.5)/cell_m)
        x1, y1 = math.ceil((obj.x+.5)/cell_m), math.ceil((obj.y+.5)/cell_m)
        grid.probabilities[max(0, y0):min(height, y1), max(0, x0):min(width, x1)] = .95
    return grid


def ground_truth_on(sim_map, geometry) -> OccupancyGrid:
    """Truth on an evidence grid's own cells: its origin, extent and resolution.

    Cells beyond the world are left unobserved, so they count for nothing.
    """
    base = ground_truth(sim_map, geometry.cell_m)
    base_height, base_width = base.probabilities.shape
    columns = np.floor((geometry.origin_x_m + (np.arange(geometry.width) + .5)
                        * geometry.cell_m) / geometry.cell_m).astype(int)
    rows = np.floor((geometry.origin_y_m + (np.arange(geometry.height) + .5)
                     * geometry.cell_m) / geometry.cell_m).astype(int)
    inside = (((rows >= 0) & (rows < base_height))[:, None]
              & ((columns >= 0) & (columns < base_width))[None, :])
    picked = np.ix_(np.clip(rows, 0, base_height - 1), np.clip(columns, 0, base_width - 1))
    probabilities = np.where(inside, base.probabilities[picked], .5).astype(np.float32)
    return OccupancyGrid(probabilities, inside & base.observed[picked], geometry.cell_m)


# -- scores ------------------------------------------------------------------


def _ratio(a, b):
    return None if b == 0 else float(a / b)


def score_occupancy(predicted: OccupancyGrid, truth: OccupancyGrid, observable=None):
    if predicted.probabilities.shape != truth.probabilities.shape:
        raise ValueError("predicted and truth grids differ")
    mask = truth.observed.copy()
    if observable is not None:
        mask &= np.asarray(observable, bool)
    gt_occ, gt_free = truth.occupied & mask, truth.free & mask
    pr_occ, pr_free = predicted.occupied & mask, predicted.free & mask
    occupied_intersection = np.count_nonzero(gt_occ & pr_occ)
    free_intersection = np.count_nonzero(gt_free & pr_free)
    decided = np.count_nonzero((pr_occ | pr_free) & mask)
    brier_mask = predicted.observed & mask
    brier = None if not brier_mask.any() else float(np.mean(
        (predicted.probabilities[brier_mask] - truth.probabilities[brier_mask]) ** 2))
    return {
        "occupied_iou": _ratio(occupied_intersection, np.count_nonzero(gt_occ | pr_occ)),
        "occupied_precision": _ratio(occupied_intersection, np.count_nonzero(pr_occ)),
        "occupied_recall": _ratio(occupied_intersection, np.count_nonzero(gt_occ)),
        "free_iou": _ratio(free_intersection, np.count_nonzero(gt_free | pr_free)),
        "coverage": _ratio(decided, np.count_nonzero(mask)),
        "brier": brier,
        "hallucination_rate": _ratio(np.count_nonzero(pr_occ & gt_free), np.count_nonzero(gt_free)),
    }


def evidence_occupancy(grid, layer: int = 0) -> OccupancyGrid:
    """A published evidence grid as the scorer reads it: never-observed is unobserved."""
    occupancy, _ = grid.layer(layer)
    never = occupancy == 255
    probabilities = np.where(never, .5, occupancy / 254.0).astype(np.float32)
    return OccupancyGrid(probabilities, ~never, grid.geometry.cell_m)


def score_evidence(grid, sim_map, camera_poses, *, fov_h_deg: float = 70.0) -> dict:
    """A published grid scored against truth on its own cells, over what was visible."""
    geometry = grid.geometry
    truth = ground_truth_on(sim_map, geometry)
    local = [replace(pose, x_m=pose.x_m - geometry.origin_x_m,
                     y_m=pose.y_m - geometry.origin_y_m) for pose in camera_poses]
    observable = observable_mask(truth, local, fov_h_deg=fov_h_deg)
    return score_occupancy(evidence_occupancy(grid), truth, observable)


# -- visibility --------------------------------------------------------------

MAX_POSES = 400


def observable_mask(truth, poses, *, fov_h_deg: float = 70.0,
                    max_range_m: float = 25.0, rays: int = 181) -> np.ndarray:
    """Cells within line of sight of any pose: the only cells worth scoring."""
    occupied = truth.occupied
    mask = np.zeros(occupied.shape, bool)
    poses = list(poses)
    if not poses or rays < 2: return mask
    if len(poses) > MAX_POSES:
        keep = np.linspace(0, len(poses)-1, MAX_POSES).round().astype(int)
        poses = [poses[index] for index in dict.fromkeys(keep.tolist())]
    height, width = occupied.shape
    step_m = truth.cell_m*.5
    distance = np.arange(1, int(max_range_m/step_m)+1)*step_m
    offsets = np.radians(np.linspace(-fov_h_deg/2, fov_h_deg/2, rays))
    for pose in poses:
        yaw = (math.radians(pose.heading_deg)+offsets)[:, None]
        xs = np.floor((pose.x_m+np.sin(yaw)*distance)/truth.cell_m).astype(int)
        ys = np.floor((pose.y_m-np.cos(yaw)*distance)/truth.cell_m).astype(int)
        inside = (xs >= 0) & (ys >= 0) & (xs < width) & (ys < height)
        carried = np.cumprod(inside, axis=1).astype(bool)
        blocked = carried & occupied[np.clip(ys, 0, height-1), np.clip(xs, 0, width-1)]
        visible = carried & (np.cumsum(blocked, axis=1)-blocked == 0)
        mask[ys[visible], xs[visible]] = True
    return mask


# -- verdict images ----------------------------------------------------------

BACKGROUND = (32, 39, 48)
TRUE_POSITIVE = (76, 195, 138)
FALSE_POSITIVE = (255, 107, 107)
FALSE_NEGATIVE = (94, 150, 255)


def verdict_raster(truth, predicted, observable=None) -> np.ndarray:
    seen = np.ones_like(truth.observed) if observable is None else np.asarray(observable, bool)
    rgb = np.full((*truth.observed.shape, 3), BACKGROUND, np.uint8)
    gt, pred = truth.occupied, predicted.occupied
    rgb[gt & pred & seen] = TRUE_POSITIVE
    rgb[~gt & pred & seen] = FALSE_POSITIVE
    rgb[gt & ~pred & seen] = FALSE_NEGATIVE
    return rgb


def overlay_image(truth, predicted, scale: int = 4, observable=None) -> Image.Image:
    image = Image.fromarray(verdict_raster(truth, predicted, observable))
    return image.resize((image.width * scale, image.height * scale), Image.Resampling.NEAREST)


# -- oracle controls ---------------------------------------------------------


class ConstantAlgorithm:
    """Carries no information: every cell 0.2. The floor any algorithm must beat."""
    name = "constant"

    def __init__(self, width_m, height_m, *_, **__):
        self.width_m, self.height_m = width_m, height_m
    def start(self): pass
    def observe(self, frame): del frame
    def preview(self): return self.finish()
    def finish(self):
        grid = OccupancyGrid.unknown(self.width_m, self.height_m)
        grid.observed[:] = True
        grid.probabilities[:] = .2
        return Result(grid)


class ExactRangeAlgorithm:
    """An oracle ceiling built from truth. Evaluation only; never a source."""
    name = "exact_range"

    def __init__(self, truth, **_): self.truth = truth
    def start(self): pass
    def observe(self, frame): del frame
    def preview(self): return self.finish()
    def finish(self):
        return Result(OccupancyGrid(self.truth.probabilities.copy(),
                                    self.truth.observed.copy(), self.truth.cell_m),
                      {"control": "ground_truth_ceiling"})
