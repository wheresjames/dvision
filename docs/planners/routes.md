# Routes: what a plan is, and what one costs

The answer object every planner returns, the arithmetic that prices it, and the
two transformations that turn a lattice path into something a vehicle can
follow. Every route carries the **map revision** and the **cost-policy digest**
it was planned against; a route without them is a route nobody can reproduce.

Source: [`apps/dnav/route.py`](../../apps/dnav/route.py).

## `Route`

```python
@dataclass(frozen=True)
class Route:
    status: str                    # one of STATUSES — see planners/README.md
    reason: str = ''
    waypoints: tuple[Waypoint, ...] = ()
    cost: float = math.inf
    length_m: float = 0.0
    planner: str = ''
    map_revision: int = 0          # summed across evidence layers
    policy_digest: str = ''        # the EFFECTIVE policy, not the file
    plan_time_s: float = 0.0
    expanded: int = 0              # cells the search closed
    sim_time_s: float = 0.0
    goal: tuple | None = None
    start: tuple | None = None
    diagnostics: dict = {}         # compare=False: never part of equality
```

Two invariants enforced at construction:

```
  status not in STATUSES         ──> ValueError
  status != ok and waypoints     ──> ValueError  "a <status> route may not carry waypoints"
  status != ok and not reason    ──> ValueError  "a <status> route must say why"
```

A failure is a message, never silence. There is no way to build a silent
failure, which is why every dnav status readout has text in it.

`diagnostics` is excluded from comparison so that replaying an attempt can
compare the route that matters — status, geometry, cost — against the recorded
one without timing noise breaking equality.

### `Waypoint`

```python
Waypoint(x_m, y_m, z_m, speed_mps=None)
```

Every viewer in dnav ignores the speed. It is carried because a **tour's**
waypoint list has the same shape, so an executor that later follows a route
needs no translation — and adding the field afterwards would have meant
changing a published format.

## Costing

One accounting rule, in one function, so two planners' numbers can be compared
at all.

```
  STRAIGHT = 1.0        DIAGONAL = sqrt(2)      step lengths, in cells

  path_cost(cost, cells):
      total = 0
      for each step from cell i to cell i+1:
          step = DIAGONAL if both indices changed else STRAIGHT
          total += step * cost[cell i+1]           the cell ENTERED
      the FIRST cell is free — the vehicle is already standing in it
      any infinite cell ──> the whole path is infinite
```

This is the same `g` that A\* accumulates, which is what makes the search's
answer and the published price the same number.

```
  line_cells(geometry, a, b)      Bresenham in map metres, eight-connected,
                                  one cell per step. A step that satisfies both
                                  error tests is the DIAGONAL one, which keeps
                                  this on A*'s move set instead of taking two
                                  orthogonal steps around a corner A* would cut.
                                  Cells off the grid are DROPPED, not clamped:
                                  clamping would fold a route that leaves the
                                  map onto its edge and price it as though it
                                  had not.

  polyline_cells(geometry, pts)   one continuous cell path through a polyline,
                                  without doubling its joins

  route_cost(cost_map, pts)       price an arbitrary polyline: what the report
                                  quotes, and what a follower would pay
```

## `simplify`: drop the collinear cells

```
  a cell-by-cell path                     simplified

   ◉─·─·─·─·─·─·─┐                         ◉─────────────┐
                 ·                                       │
                 ·                                       │
                 └─·─·─·─· ✚                             └─────── ✚

   100 waypoints where 3 would do —        every dropped cell only CONTINUED a
   and every one is a point a              straight run, so re-walking the
   follower would slow down for            result reproduces the identical cell
                                           path: the cost is unchanged, and can
                                           be quoted against either one
```

## `shorten`: pull the string taut

An eight-connected search can only turn in 45° steps, so on open ground its
route is a staircase several percent longer than the straight line it is
imitating — which makes "how much longer than the control route" read as a
detour when it is only quantisation.

```
  for each anchor, try the FURTHEST reachable waypoint first and walk back:

      direct  = path_cost(line_cells(anchor, candidate))
      through = path_cost(polyline_cells(anchor .. candidate))

      take the join only when   direct <= through + 1e−9
                          and   allows(anchor, candidate)     when a clearance
                                                              rule was supplied
```

The acceptance test is what keeps it honest: a join is taken only when the
direct segment costs **no more** than the corner it replaces, so the route can
only get cheaper and the planner stays no worse than the control it is measured
against. A test states exactly that — pulling the string taut can only make a
route cheaper.

Note the two independent gates. Cost alone is not enough: a shortcut can cost
the same and still clip a corner, so `allows` is checked as well, and A\*
re-validates the whole polyline afterwards.

## `RouteHistory`

```python
history.observe(route)   # returns whether the status or reason CHANGED
```

Appends an entry only on a change, with the sim time, status, reason, planner
and map revision. A report that only holds the last route says nothing about a
run that spent most of it stale and recovered at the end; `counts()` turns the
entries into a status tally for the summary.

## What a status means for the caller

Recapping [`planners/README.md`](README.md#every-planner-answers-always), from
the point of view of what to *do*:

| Status | Do |
|---|---|
| `ok` | fly it — subject to admission, which is a separate verdict |
| `goal_unreachable` | pick another goal (or accept a `plan_near` substitute) |
| `no_route` | the map has to change; check `coverage_exhausted` before assuming the world is sealed |
| `start_blocked` | move the vehicle, or allow `escape_margin` |
| `stale_map` | look at the producer: is it the data fact or the liveness fact |
| `stale_pose` | localisation |
| `frame_mismatch` | the goal's authority must reissue the goal |
| `outside_coverage` | request a mapping reset with larger bounds. This says nothing about the world |
| `no_goal` | set a goal |

## Where it lives

| Thing | File |
|---|---|
| `Route`, `Waypoint`, statuses, costing, `simplify`, `shorten`, `RouteHistory` | [`apps/dnav/route.py`](../../apps/dnav/route.py) |
| Who produces them | [`astar.md`](astar.md), [`control.md`](control.md) |
| Publication and the retained plane | [`apps/dnav/execution.py`](../../apps/dnav/execution.py), [`docs/navigation.md`](../navigation.md#retained-protocol) |
| Report and replay | [`apps/dnav/report.py`](../../apps/dnav/report.py), [`docs/reports.md`](../reports.md) |
| Tests | `tests/dnav/test_dnav_planning.py`, `tests/dnav/test_dnav_report.py`, `tests/navigation/test_route_handoff.py` |
