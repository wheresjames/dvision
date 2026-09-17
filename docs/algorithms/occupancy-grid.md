# The occupancy grid and the fusion helpers

Every evidence algorithm in dalg ends in the same place: a log-odds occupancy
grid, written through a handful of shared helpers. This document is that shared
half, so the per-algorithm documents can talk about geometry instead.

Source: [`apps/dalg/grid.py`](../../apps/dalg/grid.py),
[`apps/dalg/algo/spatial.py`](../../apps/dalg/algo/spatial.py),
[`apps/dalg/evidence.py`](../../apps/dalg/evidence.py).

## Log odds, and why

A cell holds `L = log(p / (1 − p))`. Evidence adds:

```
  L += delta            delta < 0  "I saw through this cell"
                        delta > 0  "I saw something stop here"

  L is clipped to ±LOG_ODDS_LIMIT (8.0), so no amount of agreement makes a
  cell unrevisable: about 12 free sweeps at −0.7, or two to four occupied hits
  depending on the algorithm's delta, saturate it.
```

Addition in log odds is multiplication of odds, which is what independent
measurements do to a belief, and it is symmetric: a wall that is removed can
be un-believed by the same number of observations that built it.

`result()` maps back with a sigmoid and hands out an `OccupancyGrid`:

```
  p = 1 / (1 + exp(−L))

   L:  −8      −2      −0.62     0     +0.62    +2      +8
   p: 0.0003  0.12     0.35     0.5    0.65    0.88   0.9997
                        │                │
                        │ FREE           │ OCCUPIED
                        │ THRESHOLD      │ THRESHOLD
                        └── free ────────┴── occupied ──>
                              undecided in between
```

`FREE_THRESHOLD = 0.35` and `OCCUPIED_THRESHOLD = 0.65` are the two numbers
everything downstream reads: dnav's clearance, dmsn's frontier distillation,
and the scoring in `dtest/evaluation.py`. A cell that was never written stays
`observed = False` and is neither.

That threshold is why the default free delta is **−0.7** and not −0.55. One
clear line of sight should be enough to call a cell free, and the free
threshold sits at log-odds −0.619; a weaker delta leaves a singly-carved cell
*undecided* however obvious it looked — it renders at grey 93,
indistinguishable from decided free space, and counts for nothing. The
previous −0.55 cost the three algorithms that take the default about a tenth
of their free-space score.

## `update` versus `accumulate`

Two ways to write, and picking the wrong one is a real error:

| Method | Rule | For |
|---|---|---|
| `update(xs, ys, delta)` | `delta` applied **at most once per cell**, however often the cell is listed | ray casts — a near-diagonal line revisits cells, and one sweep of free space must not count twice |
| `accumulate(xs, ys, delta)` | `delta` applied **once per listing**, via `np.add.at` | batches of independent measurements that happen to agree on a cell, so agreement reinforces |

Cell indices come from `cells()`, which **floors** rather than truncates.
`int()` rounds toward zero, so every point in the half-cell just outside the
west or north edge would land on row or column 0 and be fused as real
occupancy instead of being discarded by the bounds check.

## `ObservedGrid`: the stamps the wire requires

`ObservedGrid` is `LogOddsGrid` plus `observed_ms`, the capture time of each
actual cell update. Whoever feeds the grid sets `timestamp_s` first — per frame
for an algorithm that fuses as it observes, per fused frame for one that fuses
later — and every write stamps. A cell marked observed with no time on it is a
record the evidence format forbids, and the consumer rejects the whole record
it arrives in.

Publication (`GridEvidence.publish`) quantises to the wire:

```
  probability ──> uint8 0..254   (PROBABILITY_SCALE = 254)
  never observed ──> 255         (NEVER_OBSERVED sentinel)
  observed_ms ──> uint32 ms, clamped up off zero (0 is the sentinel there)
```

Publication alone never refreshes a stamp: only ray updates do, so old cells
never look newly observed just because the grid was republished.

`CameraEvidence.observe` translates the pose into grid-local metres before the
algorithm sees it, because an algorithm's cell indexing starts at the grid's
own origin, which may be negative. **Only x and y translate** — camera height
and attitude keep their physical meaning above the ground.

## The fusion helpers

### `ray_cells`

*(`ray_cells(x0, y0, x1, y1)`)*

Integer cells from the sensor to, but **excluding**, the endpoint. The endpoint
is the measurement; everything before it is the line of sight.

### `fuse_endpoint`

*(`fuse_endpoint(grid, pose, point, free=−0.7, occupied=2.5)`)*

One sample: carve the ray free, mark the endpoint occupied. No occlusion test.

```
  camera                                   endpoint
    C . . . . . . . . . . . . . . . . . . . . X
    └──────────── update(free) ─────────────┘ └ update(occupied)
```

### `fuse_endpoints`

*(`fuse_endpoints(grid, pose, samples, ...)`)*

The same, for a batch, **nearest first, dropping occluded samples**. This is
the floor-reasoning baselines' fusion (`ground_plane`, `plane_sweep`), and the
rule exists because those algorithms observe the edges of things standing *on*
the floor. A point above the floor projects **past** the obstacle it belongs
to: the floor intersection the projection assumes lies beyond it.

```
  Before:  the wall's own base is fused first (it is nearer)

      C . . . . . . . . . ▓▓            ▓▓ = occupied cell, log odds >= +0.62
                          wall base

  Then:    a sample from a point higher up the same wall projects to here
                                     ↓
      C . . . . . . . . . ▓▓ . . . . . o     ray crosses the wall first
                          └── blocked ──┘    → dropped, not fused

  Without the rule this sample would carve free space *through* the real wall
  and paint an obstacle in open floor.
```

The rule cannot tell a spurious *near* mark from a true one: a floor-tile
boundary projects exactly like an obstacle's base, fuses first by being
nearer, and the true obstacle behind it then reads as occluded. What stays
behind such a mark is **unknown**, though — never carved free through a real
thing — which is the safe side of the trade.

### `clear_rays`

*(`clear_rays(grid, pose, xs, ys, free=−0.7)`)*

The dense algorithms' counterpart (`sgbm`, `monocular_depth` in evidence
mode): a depth map sees along thousands of rays a frame, and each one proves
the space it crossed is clear. Endpoints are **merged by cell first**, so the
cost is one line of sight per distinct endpoint cell rather than per pixel.

A ray proves its column free at the ray's own height, not all the way down, so
a low obstacle close to the camera can sit under a ray to the floor.
`ground_plane` has always made the same approximation.

### `obstacle_band`

*(`obstacle_band(z, xs, ys, pose, ...)`)*

Which samples are plausible obstacles rather than floor or sky. A 2-D
occupancy grid has no way to express height, so without this **every patch of
textured floor in front of the vehicle is fused as a wall**.

```
  side view                        max_height_m (4.5)
        ─────────────────────────────────────────
                    ░░░ sky sample: dropped
        C ─────────────────▓▓▓  wall sample: kept
                    ░░░ floor sample: dropped
        ───────────────────────────────────────── min_height_m (0.25)
          |<-- min_range_m          max_range_m -->|
```

`free_space_samples` is its complement for ray carving: everything *below* the
top of the band — floor included — ended a line of sight worth clearing. Sky is
left out, because a ray to it leaves the flight band part-way and clearing its
whole length would clear space the ray merely passed over.

### `triangulate_xy`

*(`triangulate_xy(pose_a, px_a, pose_b, px_b, ...)`)*

Two horizontal image bearings intersected, with weak geometry rejected. See
[`feature-triangulation.md`](feature-triangulation.md), which explains the
rejection tests and their geometry.

## Where it lives

| Thing | File |
|---|---|
| `OccupancyGrid`, `LogOddsGrid`, `ObservedGrid`, thresholds | [`apps/dalg/grid.py`](../../apps/dalg/grid.py) |
| Frames, projection, bands, fusion helpers | [`apps/dalg/algo/spatial.py`](../../apps/dalg/algo/spatial.py) |
| Publication and the pose translation | [`apps/dalg/evidence.py`](../../apps/dalg/evidence.py) |
| Wire quantisation and stamps | [`apps/dcmn/maps.py`](../../apps/dcmn/maps.py) |
| Tests | `tests/dalg/test_dalg_core.py`, `tests/dalg/test_dalg_evidence_algorithms.py`, `tests/dalg/test_dalg_synthetic.py` |
