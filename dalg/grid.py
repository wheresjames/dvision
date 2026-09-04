from __future__ import annotations

from dataclasses import dataclass
import numpy as np

FREE_THRESHOLD = 0.35
OCCUPIED_THRESHOLD = 0.65
LOG_ODDS_LIMIT = 8.0


@dataclass
class OccupancyGrid:
    probabilities: np.ndarray
    observed: np.ndarray
    cell_m: float = 0.25

    def __post_init__(self) -> None:
        self.probabilities = np.asarray(self.probabilities, np.float32)
        self.observed = np.asarray(self.observed, bool)
        if self.probabilities.shape != self.observed.shape:
            raise ValueError("grid arrays must have equal shapes")

    @classmethod
    def unknown(cls, width_m: float, height_m: float,
                cell_m: float = 0.25) -> "OccupancyGrid":
        shape = (int(np.ceil(height_m / cell_m)),
                 int(np.ceil(width_m / cell_m)))
        return cls(np.full(shape, .5, np.float32), np.zeros(shape, bool), cell_m)

    @property
    def occupied(self) -> np.ndarray:
        return self.observed & (self.probabilities >= OCCUPIED_THRESHOLD)

    @property
    def free(self) -> np.ndarray:
        return self.observed & (self.probabilities <= FREE_THRESHOLD)


class LogOddsGrid:
    def __init__(self, width_m: float, height_m: float,
                 cell_m: float = 0.25) -> None:
        grid = OccupancyGrid.unknown(width_m, height_m, cell_m)
        self.log_odds = np.zeros(grid.probabilities.shape, np.float32)
        self.observed = grid.observed
        self.cell_m = cell_m

    def cells(self, xs, ys) -> tuple[np.ndarray, np.ndarray]:
        """Cell indices for world metres.

        Flooring rather than truncating matters: ``int()`` rounds toward zero,
        so every point in the half-cell just outside the west or north edge
        would land on row or column 0 and be fused as real occupancy instead of
        being discarded by the bounds check below.
        """
        return (np.floor(np.asarray(xs, np.float64) / self.cell_m).astype(int),
                np.floor(np.asarray(ys, np.float64) / self.cell_m).astype(int))

    def _inside(self, xs, ys) -> tuple[np.ndarray, np.ndarray]:
        xs, ys = np.asarray(xs, int), np.asarray(ys, int)
        valid = ((xs >= 0) & (ys >= 0) & (xs < self.log_odds.shape[1])
                 & (ys < self.log_odds.shape[0]))
        return xs[valid], ys[valid]

    def update(self, xs, ys, delta: float) -> None:
        """Apply ``delta`` at most once per cell, however often it is listed.

        Ray casts revisit a cell whenever the line is close to diagonal, and a
        single sweep of free space should not count twice for that.
        """
        xs, ys = self._inside(xs, ys)
        self.log_odds[ys, xs] = np.clip(self.log_odds[ys, xs] + delta,
                                        -LOG_ODDS_LIMIT, LOG_ODDS_LIMIT)
        self.observed[ys, xs] = True

    def accumulate(self, xs, ys, delta: float) -> None:
        """Apply ``delta`` once per listing, so repeated hits reinforce.

        The counterpart to :meth:`update`, for batches of independent
        measurements that happen to agree on a cell.
        """
        xs, ys = self._inside(xs, ys)
        np.add.at(self.log_odds, (ys, xs), delta)
        self.log_odds[ys, xs] = np.clip(self.log_odds[ys, xs],
                                        -LOG_ODDS_LIMIT, LOG_ODDS_LIMIT)
        self.observed[ys, xs] = True

    def result(self) -> OccupancyGrid:
        probabilities = 1.0 / (1.0 + np.exp(-self.log_odds))
        return OccupancyGrid(probabilities, self.observed.copy(), self.cell_m)
