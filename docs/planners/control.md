# `control`: the straight-line baseline

A planner that ignores cost and returns the segment from start to goal, priced
against the same cost map as the real plan. It is listed in `PLANNERS`
alongside A\* on purpose.

Every measured thing in this repository is bracketed by a control — dalg has
`constant` and `exact_range` for exactly this reason — and a route is no
different. The straight line is **the floor**: the shortest path that could
possibly exist between two points. A route from A\* that costs more than this
one has a bug in its search; a route that costs a great deal less has found a
real detour, and the ratio is the number that says whether the plan was worth
computing.

Source: [`apps/dnav/planners/controls.py`](../../apps/dnav/planners/controls.py).

## The whole planner

```
  to_cell(start) is None ──> start_blocked      "outside the mapped area"
  to_cell(goal)  is None ──> goal_unreachable   "outside the mapped area"

  waypoints = [start, goal]                     two points, no interior
  cells     = line_cells(geometry, start, goal) Bresenham, eight-connected
  cost      = path_cost(cost.cells)             infinite if any cell is blocked
  length_m  = |goal − start|
  expanded  = len(cells)
```

```
        ◉ start
         ╲
          ╲___              the Bresenham cell path priced by path_cost:
              ╲___          ┌───┬───┬───┬───┬───┐
                  ╲___      │ ◉ │ · │   │   │   │   one cell per step, and a
                      ╲     ├───┼───┼───┼───┼───┤   step that satisfies both
                       ✚    │   │ · │ · │   │   │   error tests is the DIAGONAL
                      goal  ├───┼───┼───┼───┼───┤   one — the same move set A*
                            │   │   │ · │ · │ ✚ │   searches over
                            └───┴───┴───┴───┴───┘
```

## Why the cell walk matters

`line_cells` is eight-connected and one cell per step, which is **exactly the
move set A\* searches over**. That is what makes the two numbers comparable: a
straight line measured this way is a path A\* could have returned, so A\*
costing more than it is a bug in the search rather than a difference of units.

Cells outside the grid are **dropped, not clamped**. Clamping would fold a
route that leaves the map onto its edge and price it as though it had not.

## Why it does not refuse a blocked line

It ignores cost on purpose, which is why it does not report `start_blocked` or
`goal_unreachable` for a cell the policy calls blocked — only for a point off
the grid entirely. A control that declined to answer whenever the answer was
bad would stop being a baseline exactly when one was needed.

Instead it prices its line at **infinity**, and infinity is a perfectly good
thing to be better than. Downstream, `_ratio` reports `control_blocked: true`
and leaves `cost_ratio` as `None`, so the readout says "the straight line was
blocked" rather than dividing by infinity.

## How the comparison is used

The control is planned on **every** attempt, right after the real planner, and
the pair goes into the route event, the status readout and the report:

```python
cost_ratio   = route.cost / control.cost          # < 1 is impossible
length_ratio = route.length_m / control.length_m
```

Two cases the readout has to survive, and does:

- **A blocked control** has no finite cost — common, since any goal behind a
  wall blocks the straight line.
- **A vehicle already standing at its goal** has a control of zero length,
  which is exactly where every tour ends, so the readout meets it on every run.

`describe_detour` in [`apps/dnav/dnav.py`](../../apps/dnav/dnav.py) turns the
pair into words without ever crashing on either.

## Running it as the planner

```sh
python3 ./apps/dnav/dnav.py --id area1 --goal 52.4,2.4 --planner control
```

Legitimate for exactly two things: seeing what the floor looks like on a given
map, and confirming that a route failure is the planner's rather than the cost
map's. It is not a flyable planner — it will happily publish a segment through
a wall at infinite cost, and admission is what stops that going anywhere.

## Where it lives

| Thing | File |
|---|---|
| The planner | [`apps/dnav/planners/controls.py`](../../apps/dnav/planners/controls.py) |
| `line_cells`, `path_cost` | [`apps/dnav/route.py`](../../apps/dnav/route.py) → [`routes.md`](routes.md) |
| The ratio | `_ratio` in [`apps/dnav/plan.py`](../../apps/dnav/plan.py) |
| Tests | `tests/dnav/test_dnav_planning.py` (the admissibility gate is written against this planner) |
