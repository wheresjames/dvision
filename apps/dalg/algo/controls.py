from __future__ import annotations
from dalg.grid import OccupancyGrid
from dalg.model import Result


class ConstantAlgorithm:
    name = "constant"
    sensors = ("rgb",)
    # Every algorithm is constructed through one call site in dalg.run, which
    # passes intrinsics positionally and settings by keyword. The controls
    # ignore both, but they still have to accept them.
    def __init__(self, width_m, height_m, intrinsics=None, settings=None, **_):
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
    name = "exact_range"
    # This is an oracle ceiling built from DALG's truth grid, not a simulated
    # range sensor. RGB keeps its run barrier compatible with ordinary tours.
    sensors = ("rgb",)
    def __init__(self, truth, **_): self.truth = truth
    def start(self): pass
    def observe(self, frame): del frame
    def preview(self): return self.finish()
    def finish(self):
        return Result(OccupancyGrid(self.truth.probabilities.copy(),
                                    self.truth.observed.copy(), self.truth.cell_m),
                      {"control": "ground_truth_ceiling"})
