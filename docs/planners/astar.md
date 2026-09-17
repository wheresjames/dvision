# `astar`: A\* over the combined cost map

The default planner. Eight-connected search over the cost map, charging each
step the length of the move times the cost of the cell it enters, with an
infinite cost for a cell no route may occupy. It returns the cheapest route on
the grid it is given, or the reason there is none.

It was extracted from `daic`'s prototype rather than inherited from it. What is
kept is the shape that was right: eight-connected moves, cost charged per cell
entered, infinity for forbidden cells. What is fixed is the part that was
subtly wrong there — **the heuristic has to be measured in the same units the
cost function charges**, or A\* stops being admissible and settles for the first
route it finds instead of the cheapest one.

Source: [`apps/dnav/planners/astar.py`](../../apps/dnav/planners/astar.py).

## The whole of `plan()`

```
  start, goal in map metres
       │
   1.  ├─ to_cell(start) is None ────────────> start_blocked  "outside the mapped area"
       ├─ to_cell(goal)  is None ────────────> goal_unreachable "outside the mapped area"
       │
   2.  ├─ _prepare(): clearance rule, and margin escape if the start is inside one
       │                └─ refused ─────────> start_blocked  (with which refusal)
       │
   3.  ├─ cost[goal] is infinite ───────────> goal_unreachable
       │                                       "an obstacle or inside its N m margin"
   4.  ├─ _search(): cheapest eight-connected cell path
       │                └─ none found ──────> no_route  (coverage-exhausted or enclosed)
       │
   5.  ├─ simplify(): drop cells that only continue a straight run
   6.  ├─ shorten():  pull the string taut, cost-checked and clearance-checked
   7.  ├─ revalidate every consecutive pair, and the endpoint
       │                └─ any fails ───────> no_route  "no route with the required
       │                                                  obstacle clearance"
   8.  └─ price the published polyline, count unknown cells, return ok
```

Steps 5–8 are what makes the published route *followable* rather than merely
optimal on a lattice, and every one of them is cost-checked so it can only
improve the route.

## 1. The route runs between the given points

```
  the vehicle is HERE, not in the middle of its own cell:

   ┌─────┬─────┬─────┐              points = [ exact start ]
   │     │     │     │                     + [ cell centres of the simplified
   │  ◉──┼──·──┼──·──┼── ...                   interior cells ]
   │     │     │     │                     + [ exact goal ]
   └─────┴─────┴─────┘
      ↑      ↑
   exact   centres
```

A vehicle already standing somewhere does not fly to the middle of its own cell
first. The exact start and goal are also what every clearance check is run
against — substituting them *after* simplification would change the geometry
that was validated, which is why `position()` maps the start and goal cells
back to their true coordinates during the search as well.

## 2. `_prepare`: clearance, and escaping a margin

Two things come out of this: the `allows(a, b)` predicate every search edge and
every shortcut is tested with, and a cost map in which a start inside an
obstacle *margin* has been made passable.

```python
solid    = solid_cells(any layer's obstacles, never_observed)
corridor = ObstacleClearance(geometry, solid, policy.inflation_m, policy.body_radius_m)
allows   = lambda a, b: corridor.allows(a, b, escape=policy.escape_margin)
```

`solid_cells` adds never-observed cells that **touch** an observed obstacle: an
unseen cell beside a known wall is as likely its continuation as free space.
That rule came from a real crash — a wall's end cells were unobserved, and a
route that cleared the observed end by 0.35 m flew through the real one. See
[`clearance.md`](clearance.md).

### The margin-escape rule

A vehicle can end up parked inside an inflated margin: the margin grew around
it as evidence arrived, or it was placed there. Refusing to plan would strand
it.

```
  ▓▓▓▓▓ obstacle        ░░░░░ its inflation margin (BLOCKED)        · free

        ░░░░░░░░░░░░░░░░░
        ░▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░
        ░▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░
        ░░░░░░░◉░░░░░░░░░   ◉ the vehicle, inside the margin
        · · · ·│· · · · ·
               ▼            the connected margin region around it is re-priced
                            at ESCAPE_COST = 10 x max(FREE_COST, UNOBSERVED_COST)
                            = 15.0 per cell: allowed, but never preferred over
                            open or unknown space
```

An **occupied** start is never assumed to be a sensor error. The escape is
refused, with the reason named, when:

| Refusal | Condition |
|---|---|
| `no obstacle layers to separate obstacle from margin` | there is nothing to distinguish the two with |
| `its cell is observed occupied` | the start is in the obstacle itself, not its margin |
| `no source has observed its cell` | the start cell is never-observed |
| `margin escape is disabled by the cost policy` | `escape_margin: false` |

The region is found by a flood fill over cells that are blocked **but not
obstacles**, so the escape can only pass through margin. And every escape edge
must keep the body clear and **never decrease** the distance to a nearby
obstacle — the `escape=True` branch of `ObstacleClearance.allows`, which works
because distance to a convex square is convex along a segment.

`escape_cells` is reported in the diagnostics and in the route's reason, so a
route that started by backing out of a margin says so.

## 4. `_search`: the search itself

```
  moves: eight-connected              step lengths charged
                                      STRAIGHT = 1.0
     ↖  ↑  ↗                          DIAGONAL = sqrt(2)
     ←  ·  →
     ↙  ↓  ↘                          g(neighbour) = g(here) + step x cost[neighbour]
```

A cell with infinite cost is skipped before anything else; `allows` is checked
only for edges that would actually improve a cell, because the clearance test
is the expensive one.

### The heuristic, and why it is admissible

```python
dx, dy    = abs(ncol − goal_col), abs(nrow − goal_row)
heuristic = FREE_COST * (max(dx, dy) + (DIAGONAL − 1.0) * min(dx, dy))
            └─ 1.0 ─┘   └──────── octile distance in cells ────────┘
```

The octile distance is the exact number of straight and diagonal steps needed
on an empty grid. Multiplying by `FREE_COST` — the cheapest any cell can cost,
since `UNOBSERVED_COST` is 1.5 and everything else is higher — makes the
estimate a lower bound on any real remaining cost. So it can never
overestimate, the first route to the goal is the cheapest one, and the control
route can be used as a hard floor in a test.

Using the octile distance *in cells* against a cost function charged *in
cost units* is exactly the mismatch that made the prototype inadmissible.

### Implementation: flat arrays, not dicts of tuples

```python
flat = cost.ravel()                 index = row * width + col
best = np.full(flat.size, inf)      came = np.full(flat.size, −1)
closed = np.zeros(flat.size, bool)  heap = [(0.0, start_index)]
```

This runs once per evidence revision on a grid that may be forty thousand
cells, and the difference between flat arrays and dictionaries of tuples is the
whole plan budget. `expanded` counts cells popped and closed, which is the
number the report quotes and the honest measure of the search's cost.

### Why a failure distinguishes two cases

```python
boundary = any closed cell lies on the grid's outer edge
```

```
  coverage_exhausted = True                coverage_exhausted = False

   ┌───────────────┐                        ┌───────────────┐
   │ ░░░░░░░░░░░░░░│ <- the search reached  │   ▓▓▓▓▓▓▓▓▓   │
   │ ░░░░░░░░░░░░░░│    the edge of the     │   ▓ ░░░░░ ▓   │
   │ ░░░░◉░░░░░░░░░│    mapped area         │   ▓ ░◉░░░ ▓   │
   │ ░░░░░░░░░░░░░░│                        │   ▓▓▓▓▓▓▓▓▓   │
   └───────────────┘                        └───────────────┘
   "a detour outside coverage                "observed obstacles enclose the
    is not excluded"                          vehicle or the goal"
```

A bounded search failing inside fixed coverage says nothing about whether a
physical route exists beyond it. Reporting the two the same way would make
"request more coverage" and "the room is genuinely sealed" look identical.

## 5–7. Making the path followable

| Step | What it does | Why it is safe |
|---|---|---|
| `simplify` | drops interior cells that only continue a straight run | re-walking the result reproduces the identical cell path, so the cost is unchanged |
| `shorten` | joins waypoints whose direct segment costs **no more** than the path it replaces, and passes `allows` | the route can only get cheaper; the planner stays no worse than the control |
| revalidate | re-checks every consecutive pair with `allows`, plus `corridor.allows(end, end)` | a shortcut that lost clearance fails the whole plan rather than being published |

`shorten` exists because an eight-connected search can only turn in 45° steps,
so on open ground its route is a staircase several percent longer than the
straight line it is imitating — which makes "how much longer than the control"
read as a detour when it is only quantisation.

```
  before shorten                     after shorten

   ◉                                  ◉
   │  ╲                               │  ╲
   │   ╲__                            │    ╲
   │      ╲__                         │      ╲
   │         ╲__ ✚                    │        ╲___ ✚
   staircase: 45° steps               one segment, if it costs no more
```

The endpoint check (`allows(end, end)`) is a *point* test: a route may pass
through a clearance band, but it may not **end** inside a margin.

## 8. What the route says

The cost quoted is `route_cost(cost_map, points)` — the price of the polyline
the route actually carries, not the price of the cell path the search walked.
The number in the report is therefore the number a follower would pay, and the
comparison with the control route is like for like. Note that when the start
escaped a margin, pricing uses the escape-modified cost map, so the escape's
high cost appears in the total.

Diagnostics:

| Field | Meaning |
|---|---|
| `unknown_cells`, `unknown_corridor` | how much of the cell path crosses never-observed cells. A route through unknown space is a **hypothesis, not a verified corridor** |
| `path_cells` | the length of the searched cell path |
| `coverage_exhausted` | the search touched the coverage boundary |
| `escaped_margin`, `escape_cells` | the start was inside a margin, and how many cells it was re-priced through |
| `substitute` | present only on a `plan_near` result: the real goal and the distance to the point actually routed to |

## `plan_near`: a substitute goal

For a goal the planner cannot reach — typically one an errant obstacle cell
sits on. Not part of the `Planner` protocol; the planning loop discovers it by
`hasattr`.

```
  1. flood fill from the start over the finite-cost mask ──> reachable cells
  2. scan the cells around the goal for ones that are reachable, lie within
     radius_m of it, and pass the point clearance test
  3. sort by (distance to goal, distance from start)
  4. nothing strictly closer than where the vehicle already stands?
     ──> goal_unreachable "the vehicle already holds the reachable point
         nearest the goal (N m away)"
  5. plan() to each of the nearest 5 in turn; return the first that succeeds,
     tagged with diagnostics.substitute = {goal, distance_m}
  6. none works ──> goal_unreachable "no reachable point within N m of the goal"
```

The nearest *few* rather than the nearest one, so a candidate whose route fails
its final clearance checks does not end the search. The goal itself is
unchanged: every later replan still aims at the real goal, and the substitute
is offered again only while the real one stays unreachable. The radius is the
execution profile's `goal_substitute_m` (default 2 m; 0 disables), and it is
consulted only when the current plan failed `goal_unreachable` or `no_route`.

Step 4 is a fixed point, and it is load-bearing for the consumer. Without it
the answer oscillates: standing on the nearest reachable cell, the
next-nearest is the best candidate, and from there the first one is again — an
eligible substitute for ever, which a mission that defers to one while it is
being flown would defer to for ever too.

### The reachability pass

Step 1 is a whole-array flood fill over `np.isfinite(cost)`, propagating a row
and then a column at a time, so it costs a handful of passes rather than one
per cell of the longest path.

It does **not** re-test clearance per edge, because the cost map already
carries it per cell: `layer_cost` blocks every cell within `inflation_m` of a
solid one, so a finite cell is one whose centre the vehicle may occupy. For an
orthogonal step between two adjacent centres that is not an approximation but
an equivalence — the segment's closest approach to a grid-aligned obstacle
square is at one of its endpoints, and both are finite by construction. Only
diagonals differ, and there the fill is deliberately stricter: four-connected
propagation reaches everything an eight-connected walk reaches *except* a
diagonal that clips an obstacle's corner, since a diagonal that does not clip
one has both shared orthogonal neighbours free and is therefore a two-step
orthogonal path to the same cell.

So the reachable set is conservative — measured against the exact per-edge
walk on this repository's archived grids it never adds a cell and drops about
one in a thousand — which is the safe direction, because this only filters the
candidates step 5 plans to and `plan()` remains the authority on the route
actually returned. The one exception runs the other way: on a start inside an
obstacle margin, `_prepare` re-prices that margin to `ESCAPE_COST`, which is
finite, so the fill crosses it freely where the exact walk would require
non-decreasing clearance. That costs a candidate slot and nothing else.

Why it is written this way is worth recording, because the previous version
cost a whole run. It walked the grid cell by cell and called
`ObstacleClearance.allows` on every one of the eight neighbours of every cell
— order 10^5 NumPy calls — which took **2.8–3.2 s on a 200 × 200 grid**, and it
spent that time inside `NavigationPublisher.publish`. dnav therefore stopped
publishing for three seconds after every refused goal, and every route it then
offered arrived older than dway's `max_age_s` (3 s) and was rejected. In the
17 Sep area1 search run that was **105.7 s of a 201.7 s run** and 49 substitute
routes offered, 49 rejected. The same grids now measure 10–78 ms, and a
serpentine maze — the worst shape, where a fill that grows one cell per pass
would crawl — measures 51 ms.

## Cost and limits

Planning time is the real budget. Measured on an 80 × 100 grid: **0.15 s for
863 expanded cells and 0.75 s for 5,715**. A search that exhausts a
40,000-cell grid — which is what "goal unreachable" does, since it expands
everything reachable before giving up — could take several seconds. That is
accepted for v1, and it is why the mapping extent is not simply set huge
(`--max-side-m`, Q16).

The mitigations already in place: the plan key, so an unchanged situation is
not re-searched; and `expanded` in every report, so a slow plan is visible
rather than inferred.

`plan_near` is priced separately and has a harder deadline, because it runs
inside the publisher: whatever it costs is added to the age of every route
dnav offers, against the executor's `max_age_s`. Its reachability pass is tens
of milliseconds (above); the five candidate searches are ordinary A\* and carry
A\*'s cost, so on a wide-open 100 m arena the whole call can still reach a
second. Every attempt records `substitute_s`, and `dnav/summary.json` carries
`substitute_worst_s`, so this is visible in a report rather than needing an
archive investigation.

## What it gets wrong

- **Optimal on the grid, not in the world.** A 0.5 m lattice with 45° turns is
  not the space the vehicle flies in. `shorten` recovers most of the
  quantisation, but a 22° opening is not representable.
- **No kinematics.** No turning radius, no speed, no heading cost. A route may
  zig-zag in a way that costs a `travel`-heading executor a turn in place at
  every vertex; `corner_blend_deg` and the corner-departure check are dway's
  and dnav's answer to that, not the planner's.
- **Unknown space is cheap.** `UNOBSERVED_COST` is only 1.5 against free 1.0,
  so a route will happily cross a large unknown region rather than take a long
  observed detour. That is a deliberate choice — a planner that refused to
  enter the unknown could never leave the room it started in — and it is why
  `unknown_cells` is published with every route.
- **Two dimensions.** One layer index, one altitude slab. There is no vertical
  planning at all.
- **`plan_near`'s reachable set is approximate on the diagonals**, and on an
  escape start it can be optimistic (above). Both only cost candidate slots.
- **The margin is binary.** Inside the hard margin is infinite; outside it, the
  clearance band is the only preference. A corridor exactly as wide as the
  margin is not "tight", it is impassable.

## Where it lives

| Thing | File |
|---|---|
| The planner | [`apps/dnav/planners/astar.py`](../../apps/dnav/planners/astar.py) |
| Cost map and policy | [`apps/dnav/policy.py`](../../apps/dnav/policy.py) → [`cost-policy.md`](cost-policy.md) |
| Clearance | [`apps/dcmn/clearance.py`](../../apps/dcmn/clearance.py) → [`clearance.md`](clearance.md) |
| Route, costing, `simplify`, `shorten` | [`apps/dnav/route.py`](../../apps/dnav/route.py) → [`routes.md`](routes.md) |
| The planning loop | [`apps/dnav/plan.py`](../../apps/dnav/plan.py) |
| Tests | `tests/dnav/test_dnav_planning.py`, `tests/navigation/test_corner_clearance.py`, `tests/dnav/test_dnav_run.py` |
