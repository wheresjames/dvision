# Clearance: which moves are geometrically allowed

The cost map says which *cells* a route may occupy. Clearance says whether the
**segment between two points** keeps far enough from every occupied cell — and
it is the same code in the planner and in admission, on purpose.

Source: [`apps/dcmn/clearance.py`](../../apps/dcmn/clearance.py) (shared),
[`apps/dnav/clearance.py`](../../apps/dnav/clearance.py) (route permission).
The admission side is documented operationally in
[`docs/navigation.md`](../navigation.md#clearance-and-frame-details); this
document is the geometry.

## Why a cell test is not enough

```
  an eight-connected diagonal step between two FREE cells can still
  clip the corner of an obstacle:

        ┌─────┬─────┐
        │  ·  │▓▓▓▓▓│        · free cell
        │  a  │▓▓▓▓▓│        ▓ occupied cell
        ├─────┼─────┤
        │▓▓▓▓▓│  ·  │        the segment a→b passes through the shared corner
        │▓▓▓▓▓│  b  │        of two obstacles. Both endpoint cells are free.
        └─────┴─────┘

  ObstacleClearance.allows(a, b) measures the SEGMENT against the full
  occupied SQUARES, so this move is refused.
```

Planner inflation, search edges, route shortening and plan admission all
account for **full occupied cells**, not cell centres. Measuring to a centre is
half a cell short on the side a wall faces, which is the difference between a
margin that holds and one that does not.

## `solid_cells`: unknown beside known

```python
solid = occupied | (dilate(occupied, one cell) & never_observed)
```

Plan permission trusts unknown space — that is the whole reason
`UNOBSERVED_COST` is finite — but an unseen cell **touching** an observed
obstacle is as likely its continuation as free space.

```
  ▓ observed occupied      ? never observed      · observed free

   · · · · · · · · ·              · · · · · · · · ·
   · ▓ ▓ ▓ ▓ ▓ ? ? ·      ──>     · ▓ ▓ ▓ ▓ ▓ █ ? ·      █ = treated as solid
   · · · · · · · · ·              · · · · · · · · ·
              ↑
    the wall's real end is here, but nobody has seen it yet
```

This came from an archived crash (`area1`, 15 Sep): a wall's end cells were
unobserved, and a route that cleared the *observed* end by 0.35 m flew through
the real one. The fixture is kept as
`tests/assets/corner_clearance/area1.json`.

The rule applies in the planner *and* in admission. A thin margin otherwise
admits a route through a wall whose end the lidar has not reached.

## `ObstacleClearance.distances(a, b)`

Returns three arrays over the occupied cells near the segment: the distance
from `a`, the distance from `b`, and the **minimum distance along the whole
segment**. Exact, not sampled.

```
  1. gather candidate cells: the segment's bounding box, grown by `radius`
     and clipped to the grid

  2. each candidate is an axis-aligned SQUARE [lower, upper]

  3. point-to-square distance, for both endpoints:
         max(lower − p, p − upper, 0)  componentwise, then its norm
         (zero inside the square)

  4. segment-to-square minimum — for disjoint pairs it always occurs at an
     endpoint or a corner, so the four corners are projected onto the segment:

              corner ●
                      ╲  perpendicular from the corner to the segment,
                       ╲ clamped to t in [0, 1]
              a ────────●──────── b

  5. and the slab test, which catches a segment crossing a cell INTERIOR
     (where no corner is the closest point at all):

         enter = max over axes of the near slab crossing
         leave = min over axes of the far slab crossing
         enter <= leave  ──>  the segment enters the square: distance 0
```

Steps 4 and 5 together are why this is exact rather than conservative: corner
projection handles the disjoint case, and the slab intersection handles the
penetrating one.

`lateral_m(a, b)` is the same minimum, capped at `radius` — the number dnav
publishes per segment as `clearance.lateral_m`, which dway narrows its
cross-track allowance against. It is rounded **down** so the published value
never overstates the clearance.

## `allows(a, b)` and the escape branch

```python
clear = minimum > radius + 1e−9                     the ordinary rule
if escape:
    clear |= (start_distance <= radius              already inside the margin
              & minimum      >  body_radius         the body stays clear
              & minimum      >= start_distance      never gets closer
              & end_distance >= start_distance)     and ends no closer
```

Every candidate cell must satisfy one of the two branches; `all()` over the
candidates is the verdict.

The escape branch is what lets a vehicle parked inside an inflated margin be
routed out of it (see [`astar.md`](astar.md#the-margin-escape-rule)). Its
correctness rests on one geometric fact:

```
  distance to a convex square is CONVEX along a segment

     distance
        │  ●                                 ●
        │    ●                            ●
        │      ●                       ●
        │         ●                 ●
        │            ● ● ● ● ● ●
        └──────────────────────────────────> t along the segment
        a                                  b

  so if the minimum over the segment equals the distance at `a`, separation is
  NONDECREASING for the whole move — checking the endpoints and the minimum is
  enough, with no sampling
```

An occupied start is never escapable: the branch requires the body radius to
stay clear, and `_prepare` in the planner refuses outright when the start cell
is observed occupied or never observed.

## Planning clearance versus execution admission

Two different questions, same geometry, deliberately separate:

```
  PLANNER (dnav)                        ADMISSION (dnav, for dway)
  ──────────────                        ──────────────────────────
  "is this move geometrically           "may this route be FLOWN, given what
   clear of the cells I believe          is observed right now?"
   are solid?"
                                        `evidence` permission:
  ObstacleClearance.allows on              conservative swept rectangles per
  the cost map's solid cells               half-cell subsegment, expanded by
                                           body radius, tracking allowance and
  trusts unknown space                     a stopping radius; EVERY swept cell
  (UNOBSERVED_COST is finite)              needs fresh FREE evidence

                                        `plan` permission:
                                           trust the planner through unknown
                                           space; withdraw only where observed
                                           OCCUPIED evidence intersects the
                                           route's clearance envelope
```

The `evidence` verdict is computed and published as `evidence_check` even under
`plan` permission, so the two policies can be compared on the same flight. The
swept rectangle deliberately over-covers — it may reject a narrow route a
precise capsule check would admit — but it **cannot** miss a touched cell or
cut a diagonal corner, and that trade is the right way round.

Validation is bounded to 10,000 subsegments and one evidence slab.

## Two rules that catch people out

- **A route may pass through a clearance band, but may not end in a margin.**
  The planner's final check is `corridor.allows(end, end)` — a point test on
  the endpoint.
- **A source that disappears keeps its vetoes.** Absence is not free evidence.
  Unknown, ambiguous and stale-only evidence is not clearance either, and
  republishing an old grid does not refresh its cells.

## Where it lives

| Thing | File |
|---|---|
| `solid_cells`, `ObstacleClearance` | [`apps/dcmn/clearance.py`](../../apps/dcmn/clearance.py) |
| Route permission, swept rectangles, corners | [`apps/dnav/clearance.py`](../../apps/dnav/clearance.py) |
| The envelope's radii | `ExecutionProfile` in [`apps/dcmn/navigation.py`](../../apps/dcmn/navigation.py) |
| Operational contract | [`docs/navigation.md`](../navigation.md#clearance-and-frame-details) |
| Tests | `tests/navigation/test_corner_clearance.py`, `tests/dnav/test_dnav_planning.py` |
