from __future__ import annotations

import numpy as np
from dalg.grid import OccupancyGrid


def _ratio(a: int, b: int) -> float | None:
    return None if b == 0 else float(a / b)


def score_occupancy(predicted: OccupancyGrid, truth: OccupancyGrid,
                    observable=None) -> dict[str, float | None]:
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
        "occupied_iou": _ratio(occupied_intersection,
                               np.count_nonzero(gt_occ | pr_occ)),
        "occupied_precision": _ratio(occupied_intersection,
                                     np.count_nonzero(pr_occ)),
        "occupied_recall": _ratio(occupied_intersection,
                                  np.count_nonzero(gt_occ)),
        "free_iou": _ratio(free_intersection, np.count_nonzero(gt_free | pr_free)),
        "coverage": _ratio(decided, np.count_nonzero(mask)),
        "brier": brier,
        "hallucination_rate": _ratio(np.count_nonzero(pr_occ & gt_free),
                                     np.count_nonzero(gt_free)),
    }
