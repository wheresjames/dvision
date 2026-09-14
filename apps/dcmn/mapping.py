"""Runtime mapping context: bounded geometry and allocation admission.

dalg alone resolves the evidence geometry, once, on the first valid pose:

* explicit ``bounds`` win outright (snapped outward to whole cells);
* otherwise a square centred between the start S and a compatible goal G, of side
  ``max(min_side_m, k * d, d + 2 * margin_m)`` with ``d = |SG|`` horizontally --
  the multiplier applies to the *side length*, not area or radius;
* otherwise a ``min_side_m`` square around the first valid pose.

The side is a resource allocation, not the arena and not permission to fly. It
cannot guarantee room for a detour. Growing it later, rolling windows and tiles
are deferred work; a goal outside coverage is reported, never clipped.

Admission is an *estimate* of mapping memory -- every configured source's
algorithm arrays, transport ring slots and publication/consumer copies, plus
the planner's working arrays -- checked against a budget before anything is
allocated. It is not an operating-system memory guarantee, and it excludes model
weights, camera transport and image caches, which are reported separately.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from dcmn.maps import DEFAULT_RING_SLOTS, MAX_GRID_CELLS, GridGeometry

#: Per evidence cell and per source, in bytes: log-odds, probability and
#: timestamp planes inside the algorithm, the quantized publication copy and its
#: encoded payload, ``DEFAULT_RING_SLOTS`` transport slots of five bytes, the
#: consumer's decoded copy, the recorder's snapshot, and the per-source cost
#: layer (occupancy, never-observed, obstacle, inflation and float cost planes).
#: Measured (tracemalloc, 2026-09-11) at about 40 B/cell/source for lidar and
#: ground-plane sources; kept conservative. tests/test_dcmn_mapping.py holds the
#: estimate above the measured peak of representative configurations.
SOURCE_CELL_BYTES = 48 + 5 * DEFAULT_RING_SLOTS + 64
#: Per evidence cell, once: the combined cost surface and A* working arrays
#: (float64 costs, best-so-far, predecessor, closed set) plus heap entries.
#: Measured at about 50 B/cell on an open grid searched corner to corner.
PLANNER_CELL_BYTES = 256
#: Fixed per-source overhead: algorithm objects and their per-frame working
#: buffers at typical camera resolutions, manifests and headers. Measured at
#: about 1.5-2 MiB per source; a 64 KiB allowance underestimated small grids.
SOURCE_FIXED_BYTES = 2 << 20


class AllocationError(ValueError):
    """Requested coverage does not fit the cell cap or the mapping budget."""


def cells_for(bounds, cell_m):
    x0, y0, x1, y1 = bounds
    return math.ceil((x1-x0)/cell_m - 1e-9) * math.ceil((y1-y0)/cell_m - 1e-9)


@dataclass(frozen=True)
class MappingConfig:
    bounds: tuple | None = None
    cell_m: float = .5
    z0_m: float = 0.
    dz_m: float = 3.
    budget_bytes: int = 512 * 1024 * 1024
    min_side_m: float = 40.
    multiplier: float = 2.5
    margin_m: float = 10.

    def __post_init__(self):
        for name in ('cell_m', 'dz_m', 'min_side_m', 'multiplier'):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0: raise ValueError(f'{name} must be positive and finite')
        if not math.isfinite(float(self.z0_m)): raise ValueError('z0_m must be finite')
        if not math.isfinite(float(self.margin_m)) or self.margin_m < 0:
            raise ValueError('margin_m must be finite and nonnegative')
        if self.bounds is not None:
            bounds = tuple(float(v) for v in self.bounds)
            if len(bounds) != 4 or not all(math.isfinite(v) for v in bounds):
                raise ValueError('bounds require four finite numbers')
            if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
                raise ValueError('bounds must have positive extent')
            object.__setattr__(self, 'bounds', bounds)

    def as_dict(self):
        value = asdict(self)
        value['bounds'] = None if self.bounds is None else list(self.bounds)
        return value

    def requested_bounds(self, start, goal):
        """The square (or explicit bounds) before snapping, and how it was chosen."""
        if self.bounds is not None: return self.bounds, 'explicit'
        end = start if goal is None else goal
        distance = math.hypot(end[0]-start[0], end[1]-start[1])
        side = max(self.min_side_m, self.multiplier*distance, distance+2*self.margin_m)
        cx, cy = (start[0]+end[0])/2, (start[1]+end[1])/2
        return (cx-side/2, cy-side/2, cx+side/2, cy+side/2), ('goal' if goal is not None else 'pose')

    def resolve(self, start, goal=None):
        """The snapped geometry. Raises with requested and allowed sizes on failure."""
        bounds, _ = self.requested_bounds(start, goal)
        x0, y0 = (math.floor(v/self.cell_m)*self.cell_m for v in bounds[:2])
        x1, y1 = (math.ceil(v/self.cell_m)*self.cell_m for v in bounds[2:])
        cells = cells_for((x0, y0, x1, y1), self.cell_m)
        if cells > MAX_GRID_CELLS:
            raise AllocationError(f'coverage {x1-x0:g} x {y1-y0:g} m at {self.cell_m:g} m needs {cells:,} cells '
                             f'per grid; the limit is {MAX_GRID_CELLS:,}. Supply smaller --bounds or a '
                             'coarser --cell-m explicitly; nothing is clipped or coarsened automatically')
        return GridGeometry.from_extent(x1-x0, y1-y0, self.cell_m,
            origin_x_m=x0, origin_y_m=y0, z0_m=self.z0_m, dz_m=self.dz_m)

    @staticmethod
    def estimate(geometry, sources):
        return (geometry.cells * (SOURCE_CELL_BYTES*int(sources) + PLANNER_CELL_BYTES)
                + int(sources)*SOURCE_FIXED_BYTES)

    def admit(self, geometry, sources, *, retained_bytes=0):
        """The estimate, if it fits beside ``retained_bytes`` still in use; else raise."""
        estimate = self.estimate(geometry, sources)
        if self.budget_bytes <= 0 or estimate+retained_bytes > self.budget_bytes:
            raise AllocationError(f'mapping allocation needs ~{(estimate+retained_bytes)/2**20:.1f} MiB '
                             f'({geometry.width}x{geometry.height} cells x {sources} source(s)'
                             f'{"" if not retained_bytes else f" plus {retained_bytes/2**20:.1f} MiB retained"}); '
                             f'budget {self.budget_bytes/2**20:.1f} MiB')
        return estimate


def contains(geometry, point):
    """Whether ``(x, y)`` lies inside the geometry's coverage."""
    return geometry is not None and geometry.to_cell(point[0], point[1]) is not None
