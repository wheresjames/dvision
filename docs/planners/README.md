# Planner documents: dnav

This directory holds one focused document per **planner** in dnav, plus the
shared machinery every planner is defined against: the cost policy that turns
evidence into numbers, the clearance envelope that decides which moves are
allowed, and the route object that carries the answer.

What these documents do **not** repeat: the dnav↔dway protocol, execution
modes, admission, the retained plane, the UI or the report layout. That is
[`docs/navigation.md`](../navigation.md). The evidence these planners consume is
produced by [`docs/algorithms/`](../algorithms/README.md); the goals they are
given usually come from [`docs/missions/`](../missions/README.md).

| Planner | What it does | Document |
|---|---|---|
| `astar` | A\* over the combined cost map: the cheapest route on the grid it is given. The default | [`astar.md`](astar.md) |
| `control` | the straight line, priced against the same cost map: the floor a plan has to beat | [`control.md`](control.md) |

| Shared machinery | Document |
|---|---|
| evidence → cost: thresholds, inflation, the clearance band, layer combination | [`cost-policy.md`](cost-policy.md) |
| which moves are geometrically allowed, and the margin-escape rule | [`clearance.md`](clearance.md) |
| what a route is, how one is priced, simplified and pulled taut | [`routes.md`](routes.md) |

## The seam

The architectural content of the whole design is **where the numbers come
from**:

```
  dalg                          dnav                              dway
  ────                          ────                              ────

  evidence plane                cost policy (a versioned FILE)
  occupancy + age    ──────>    ├─ occupied_threshold             route
  "how likely is this           ├─ inflation_m                    ──────>
   cell occupied, and           ├─ clearance band                 admission,
   when did anyone look"        └─ combine: max                   following,
                                        │                         the lease
  says NOTHING about                    ▼
  what that should cost           cost map (per source + combined)
                                        │
                                        ▼
                                   planner.plan(cost_map, start, goal, policy)
                                        │
                                        ▼
                                   Route: waypoints, cost, status, reason
                                          + the map revision and policy digest
                                            it was planned against
```

Because the policy is a digest-carrying artifact rather than a constant inside
a planner, two sensors disagreeing about a wall becomes a **visible parameter**
instead of hidden fusion — and every route can be replanned later from the
exact inputs that produced it.

dnav holds no control lease and sends no command. It is a route producer.

## What a planner is

```python
@runtime_checkable
class Planner(Protocol):
    name: str
    def plan(self, cost_map, start, goal, policy) -> Route: ...
```

That is all of it. A new planner — hybrid A\*, RRT\*, field D\* — is a module in
[`apps/dnav/planners/`](../../apps/dnav/planners/) and a line in `PLANNERS`,
and nothing above it changes. `build(name)` **rejects an unknown planner rather
than defaulting**, so a typo in `--planner` is an error, not a silent fallback
to A\*.

One optional extra, discovered by `hasattr` rather than declared in the
protocol: `plan_near(cost_map, start, goal, policy, radius_m)`, used when the
goal itself cannot be reached — a route to the nearest reachable point within
`radius_m`, offered only while that point is closer than where the vehicle
already stands. `astar` has it; `control` does not, and the planning loop
simply offers no substitute when it is absent.

## Every planner answers, always

A failure is a message, never silence. dnav publishes and displays the reason
rather than showing an empty map and letting the operator guess:

| Status | Meaning | Whose problem |
|---|---|---|
| `ok` | a route was found | — |
| `no_route` | start and goal are both usable, but nothing connects them | the map, or the world |
| `goal_unreachable` | the goal cannot be occupied: an obstacle, inside its margin, or off the grid | the goal |
| `start_blocked` | the vehicle is standing somewhere the policy calls blocked | the vehicle's position |
| `stale_map` | the evidence is too old to plan on, or its producer is gone | the producer |
| `stale_pose` | the pose is missing, invalid, or older than `--pose-max-age` | localisation |
| `frame_mismatch` | goal, pose and evidence disagree about frame or epoch | the goal's authority must reissue |
| `outside_coverage` | the goal or the vehicle is outside the evidence grid's fixed extent | coverage allocation |
| `no_goal` | nothing has been asked for. Not a failure — an invitation | the operator |

The distinctions are deliberate, because the fix differs: `goal_unreachable`
means *pick another goal*, `no_route` means *the map has to change*, and
`outside_coverage` proves nothing at all about the world — coverage is a
resource allocation, and the reason says so.

A non-`ok` route may never carry waypoints and must always carry a reason;
`Route.__post_init__` enforces both.

## What happens before a planner is called

Most statuses above never reach a planner. `NavRun._compute`
([`apps/dnav/plan.py`](../../apps/dnav/plan.py)) is the gate:

```
  goal set?                          no ──> no_goal
  pose valid and fresh?              no ──> stale_pose
  goal's frame/epoch matches?        no ──> frame_mismatch
  any evidence at all?               no ──> stale_map  (with why: never published,
                                                        generation changed, ...)
  evidence frame/epochs == pose's?   no ──> frame_mismatch
  every followed source present?     no ──> stale_map  ("waiting for evidence from: ...")
  any source stale?                  yes ─> stale_map  (data fact vs liveness fact —
                                                        the reason says WHICH)
  goal and vehicle inside coverage?  no ──> outside_coverage
        │
        ▼
  planner.plan(...)      and then, always, control.plan(...) for the ratio
```

Two facts about evidence are kept apart on purpose, because conflating them is
how an operator ends up debugging the wrong thing: the **data fact** (the newest
record is older than the staleness horizon) and the **liveness fact** (the
producer's heartbeat is gone from the module bus). Both are `stale_map`.

Planning is **keyed, not periodic**: `replan` builds a key from the goal
revision, planner name, policy digest, per-source evidence revisions, the
generation, staleness, producer liveness and the pose rounded to a millimetre.
An unchanged key returns the cached route without re-searching.

## Every plan is bracketed by a control

The straight line is planned on **every** attempt, beside the real route, and
the pair is published and reported as a ratio:

```
  cost_ratio   = route.cost / control.cost
  length_ratio = route.length_m / control.length_m
```

`cost_ratio` below 1 is impossible and is a bug in the search — there is a test
that says exactly that. Well above 1 means the planner found a real detour, and
the number is what says whether the plan was worth computing. A blocked control
has infinite cost, and infinity is a perfectly good thing to be better than.

## The review gate

Two properties, held by `tests/dnav/test_dnav_planning.py`, are what these documents
are ultimately describing:

1. **A\* never costs more than the straight line** it is measured against
   (admissibility), across a spread of goals on the committed synthetic room.
2. **A\* never routes through an inflated obstacle**: every cell of the
   published polyline has finite cost.

Plus: a route costs what it says it costs (`route_cost(published points)` equals
the quoted `cost`), and pulling the string taut can only make a route cheaper.

## Running one

```sh
python3 ./apps/dnav/dnav.py --id area1 --goal 52.4,2.4 --policy default --planner astar
python3 ./apps/dnav/dnav.py --id area1 --goal 52.4,2.4 --planner control    # the baseline alone
```

| Flag | Default | Effect |
|---|---|---|
| `--planner` | `astar` | one of `PLANNERS`; an unknown name is refused |
| `--policy` | `default` | committed policy name under `assets/cost_policies/`, or any path |
| `--pose-max-age` | 0.5 | pose freshness limit, in data-clock seconds |
| `--source` | every source | plan on one evidence source instead of all of them |
| `--execution-profile` | — | the profile whose envelope tightens the effective policy; dway must load the same one |
