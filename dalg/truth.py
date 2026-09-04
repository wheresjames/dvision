from __future__ import annotations

import math

from dalg.grid import OccupancyGrid


def ground_truth(sim_map, cell_m: float = .25) -> OccupancyGrid:
    """Rasterise the map's obstacle cells at the occupancy grid's resolution."""
    grid = OccupancyGrid.unknown(sim_map.width, sim_map.height, cell_m)
    grid.observed[:] = True
    grid.probabilities[:] = .05
    height, width = grid.probabilities.shape
    for obj in sim_map.objects:
        if obj.kind not in ("wall", "tree"):
            continue
        # Objects are unit cells centred on their coordinate. Working in metres
        # rather than whole-cell multiples keeps this right for a cell size that
        # does not divide a metre, and flooring keeps a negative coordinate from
        # wrapping around to the far edge of the grid.
        x0, y0 = math.floor((obj.x-.5)/cell_m), math.floor((obj.y-.5)/cell_m)
        x1, y1 = math.ceil((obj.x+.5)/cell_m), math.ceil((obj.y+.5)/cell_m)
        grid.probabilities[max(0, y0):min(height, y1),
                           max(0, x0):min(width, x1)] = .95
    return grid
