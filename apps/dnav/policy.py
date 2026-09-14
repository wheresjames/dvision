"""The cost policy: how a belief about the world becomes a number to plan on.

This is dnav's, not dalg's, and that split is the architectural content of the
whole design. `dalg` publishes *evidence* -- how likely a cell is occupied and
when anyone last looked -- and says nothing about what that should cost. dnav
applies a policy to turn evidence into cost, and because the policy is a
versioned, digest-carrying file rather than a constant in a planner, two
sensors disagreeing becomes a visible parameter instead of hidden fusion.

Version 1 is deliberately three rules:

* occupancy at or above ``occupied_threshold`` marks an obstacle;
* obstacles inflate by ``inflation_m`` -- the vehicle's footprint plus its
  margin -- which is what turns "an obstacle is in this cell" into "stay this
  far away from it";
* layers combine by **max**, because a cell is as bad as its worst witness.
  Never by sum: summing two sensors' opinions of the same wall counts that wall
  twice, which is the exact mistake publishing evidence instead of cost exists
  to avoid.

Per-source thresholds, per-source weights and a tunable cost for
never-observed cells are later knobs on this same artifact. They are not new
machinery, and they are deliberately not here yet.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from dcmn.maps import EvidenceGrid, GridGeometry

SCHEMA = 'dvision2.cost-policy.v1'

#: Where committed policies live, beside the tours and the profiles, because
#: that is what a policy is: fixture data a run is configured with, owned by no
#: single consumer, and edited far more often than the code that reads it.
POLICIES_SUBDIR = Path('assets') / 'cost_policies'

#: What a traversable cell costs. Free and never-observed are both passable --
#: a planner that refused to enter the unknown could never leave the room it
#: started in -- but they are not equally attractive, and the gap is what makes
#: a route prefer ground somebody has actually looked at. These are constants
#: rather than policy fields because §4.3 names the never-observed cost as a
#: later knob; when it becomes one, it becomes a field here and nothing else
#: changes.
FREE_COST = 1.0
UNOBSERVED_COST = 1.5

#: A cell no route may enter. Infinite rather than merely large: "expensive"
#: and "forbidden" are different answers, and a planner that can be bribed
#: through a wall by a long enough detour is not one.
BLOCKED = math.inf

DEFAULT_OCCUPIED_THRESHOLD = 0.5
DEFAULT_INFLATION_M = 0.6
COMBINERS = ('max',)


@dataclass(frozen=True)
class CostPolicy:
    """A versioned rule for turning evidence into cost, and its digest."""

    name: str = 'default'
    occupied_threshold: float = DEFAULT_OCCUPIED_THRESHOLD
    inflation_m: float = DEFAULT_INFLATION_M
    combine: str = 'max'
    #: A vehicle that stopped inside an obstacle's margin (never inside the
    #: obstacle) may be routed out of it, through that margin only, at a high
    #: cost. Off, such a start is ``start_blocked`` and nothing is planned.
    escape_margin: bool = True
    schema_version: int = 1
    digest: str = ''
    path: Path | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, 'occupied_threshold', float(self.occupied_threshold))
        object.__setattr__(self, 'inflation_m', float(self.inflation_m))
        if not 0.0 < self.occupied_threshold <= 1.0:
            raise ValueError('occupied_threshold must be within (0, 1]')
        if self.inflation_m < 0.0 or not math.isfinite(self.inflation_m):
            raise ValueError('inflation_m must be a finite, nonnegative distance')
        if self.combine not in COMBINERS:
            raise ValueError(f'unknown layer combiner {self.combine!r}; '
                             f'this version implements {", ".join(COMBINERS)}')

    def inflation_radius_cells(self, cell_m: float) -> float:
        """The margin as a radius in cells, measured centre to centre.

        Deliberately not rounded up to a whole cell. A cell is inside the
        margin when its centre is within ``inflation_m`` of an obstacle cell's
        centre, which is a rule that can be stated exactly; rounding the radius
        up to the next cell instead turns a 0.6 m margin into a 1.0 m one on a
        0.5 m grid, and 0.4 m of invented clearance on each side is enough to
        close a two-metre doorway that a drone of the declared size fits
        through comfortably. Quantisation still costs something -- an obstacle
        cell is half a cell wide, so the clearance from the obstacle's face is
        a little more than the radius -- and that error is in the safe
        direction without being large enough to seal a gap.
        """
        return self.inflation_m / float(cell_m)

    def values(self) -> dict[str, Any]:
        """What the Cost tab lists beside the layers it is drawing."""
        return {'occupied_threshold': self.occupied_threshold,
                'inflation_m': self.inflation_m, 'combine': self.combine,
                'escape_margin': self.escape_margin,
                'free_cost': FREE_COST, 'unobserved_cost': UNOBSERVED_COST}

    def as_dict(self) -> dict[str, Any]:
        return {'schema': SCHEMA, 'name': self.name,
                'schema_version': self.schema_version,
                'occupied_threshold': self.occupied_threshold,
                'inflation_m': self.inflation_m, 'combine': self.combine,
                'escape_margin': self.escape_margin}


def policy_dir(root: Path) -> Path:
    return Path(root) / POLICIES_SUBDIR


def parse_policy(raw: bytes, *, path: Path | None = None) -> CostPolicy:
    """One policy file, with its digest taken over the bytes as they were read.

    Over the bytes rather than over the parsed fields, so the digest names the
    file that was actually loaded -- the same discipline a tour and a profile
    already follow.
    """
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError('cost policy must be an object')
    if value.get('schema') != SCHEMA:
        raise ValueError(f'cost policy schema must be {SCHEMA}')
    version = value.get('schema_version', 1)
    if version != 1:
        raise ValueError(f'cost policy schema_version {version} is not supported')
    return CostPolicy(
        name=str(value.get('name', 'unnamed')),
        occupied_threshold=value.get('occupied_threshold', DEFAULT_OCCUPIED_THRESHOLD),
        inflation_m=value.get('inflation_m', DEFAULT_INFLATION_M),
        combine=str(value.get('combine', 'max')), escape_margin=bool(value.get('escape_margin', True)),
        schema_version=version,
        digest=hashlib.sha256(raw).hexdigest(), path=path)


def load_policy(name_or_path: str, root: Path) -> CostPolicy:
    """A committed policy by name, or any file by path."""
    path = Path(name_or_path)
    if not path.suffix: path = policy_dir(root) / f'{name_or_path}.json'
    elif not path.is_absolute(): path = Path(root) / path
    return parse_policy(path.read_bytes(), path=path)


# -- inflation ---------------------------------------------------------------

def disc_offsets(radius_cells: float) -> list[tuple[int, int]]:
    """Every cell offset whose centre is within ``radius_cells`` of the origin.

    A disc rather than the square a naive dilation produces: a square margin
    keeps 1.4 times the asked-for distance diagonally, which quietly closes
    gaps a vehicle of the declared size fits through -- the doorway case,
    exactly.
    """
    radius = float(radius_cells)
    if radius <= 0.0: return [(0, 0)]
    span = range(-int(math.floor(radius)), int(math.floor(radius)) + 1)
    return [(dx, dy) for dy in span for dx in span
            if dx * dx + dy * dy <= radius * radius + 1e-9]


def inflate(mask: np.ndarray, radius_cells: float) -> np.ndarray:
    """Grow a boolean obstacle mask by a disc of ``radius_cells``.

    Shifts and ORs rather than a convolution, because the radius is a couple of
    cells and this needs no dependency beyond numpy.
    """
    mask = np.asarray(mask, bool)
    if radius_cells <= 0.0 or not mask.any(): return mask.copy()
    height, width = mask.shape
    grown = np.zeros_like(mask)
    for dx, dy in disc_offsets(radius_cells):
        source_rows = slice(max(0, -dy), height - max(0, dy))
        source_cols = slice(max(0, -dx), width - max(0, dx))
        target_rows = slice(max(0, dy), height - max(0, -dy))
        target_cols = slice(max(0, dx), width - max(0, -dx))
        grown[target_rows, target_cols] |= mask[source_rows, source_cols]
    return grown


# -- the cost map ------------------------------------------------------------

@dataclass(frozen=True)
class CostLayer:
    """One source's evidence, and the cost this policy derives from it."""

    source: str
    obstacles: np.ndarray        # bool: at or over the threshold
    inflated: np.ndarray         # bool: obstacles plus their margin
    never_observed: np.ndarray   # bool
    cost: np.ndarray             # float32, BLOCKED where inflated
    revision: int = 0
    sim_time_s: float = 0.0
    stale: bool = False

    @property
    def margin(self) -> np.ndarray:
        """The cells the inflation added, which is what the Cost tab draws."""
        return self.inflated & ~self.obstacles


@dataclass(frozen=True)
class CostMap:
    """Every layer, and the single combined surface a planner searches."""

    geometry: GridGeometry
    cost: np.ndarray                    # float32 (height, width)
    layers: tuple[CostLayer, ...]
    policy: CostPolicy
    sim_time_s: float = 0.0

    @property
    def never_observed(self):
        """Unknown only where no source has observed the cell."""
        return np.logical_and.reduce([layer.never_observed for layer in self.layers])

    @property
    def blocked(self) -> np.ndarray: return ~np.isfinite(self.cost)

    @property
    def revisions(self) -> dict[str, int]:
        return {layer.source: layer.revision for layer in self.layers}

    @property
    def revision(self) -> int:
        """One number for "has the evidence moved", across every layer."""
        return sum(layer.revision for layer in self.layers)

    @property
    def stale(self) -> bool: return any(layer.stale for layer in self.layers)

    def layer(self, source: str) -> CostLayer | None:
        for candidate in self.layers:
            if candidate.source == source: return candidate
        return None

    def cell_of(self, x_m: float, y_m: float) -> tuple[int, int] | None:
        return self.geometry.to_cell(x_m, y_m)

    def is_blocked(self, x_m: float, y_m: float) -> bool | None:
        """Whether a point is unusable, or None if it is off the grid."""
        cell = self.cell_of(x_m, y_m)
        if cell is None: return None
        return not math.isfinite(float(self.cost[cell[1], cell[0]]))


def layer_cost(grid: EvidenceGrid, policy: CostPolicy, *, layer: int = 0) -> CostLayer:
    """Turn one source's evidence into one cost layer."""
    occupancy, _ = grid.layer(layer)
    never = occupancy == 255
    probability = occupancy.astype(np.float32) / 254.0
    obstacles = ~never & (probability >= policy.occupied_threshold)
    inflated = inflate(obstacles, policy.inflation_radius_cells(grid.geometry.cell_m))
    cost = np.where(never, np.float32(UNOBSERVED_COST), np.float32(FREE_COST))
    cost[inflated] = np.float32(BLOCKED)
    return CostLayer(grid.source, obstacles, inflated, never, cost,
                     grid.revision, grid.sim_time_s)


def build_cost_map(grids: Mapping[str, EvidenceGrid], policy: CostPolicy, *,
                   geometry: GridGeometry | None = None, layer: int = 0,
                   stale: frozenset[str] = frozenset(),
                   sim_time_s: float = 0.0) -> CostMap | None:
    """Every source's cost layer, combined under the policy.

    Returns None when there is nothing to plan on. Every grid must share the
    geometry -- one ``cell_m`` per profile is what makes stacking layers an
    overlay rather than a resample in the middle of every plan.
    """
    layers: list[CostLayer] = []
    for source in sorted(grids):
        grid = grids[source]
        if geometry is None: geometry = grid.geometry
        if grid.geometry != geometry:
            raise ValueError(f'{source}: grid geometry differs from the cost map; '
                             'every source of one profile shares its cell_m')
        derived = layer_cost(grid, policy, layer=layer)
        layers.append(CostLayer(derived.source or source, derived.obstacles,
                                derived.inflated, derived.never_observed,
                                derived.cost, derived.revision, derived.sim_time_s,
                                source in stale))
    if not layers or geometry is None: return None
    # Max, never sum. A cell is as bad as its worst witness, and one wall seen
    # by two sensors is still one wall.
    combined = layers[0].cost.copy()
    for extra in layers[1:]:
        combined = np.maximum(combined, extra.cost)
    return CostMap(geometry, combined, tuple(layers), policy, sim_time_s)
