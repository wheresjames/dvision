# `explore`: cover an area

Fly to the goals a strategy proposes, until there are none left or the coverage
target is met. The area is the profile's polygon. This mission owns everything
the strategy does not: **commitment** (one goal at a time), the **blacklist**
(points it has given up on, each with its reason), and **every way the mission
can end**.

Profile: `maze_020-explore`. Editor: **area**. Default strategy: `frontier`.
The reference background opens disabled (Q10) — this is not an operator
mission, and a reference image would leak the answer.

```sh
python3 ./apps/dmsn/dmsn.py --id area1 --profile maze_020-explore
```

## Deciding without a grid

The mission never sees the occupancy grid. The runner distils the evidence
plane into three bounded fields of the observation (Q28):

```
  evidence plane (occupancy + age)
        │
        │  _evidence_view(area, pose, exclude=blacklist)
        ▼
  frontiers      observed-free cells 4-adjacent to a never-observed cell,
                 masked to the area polygon, points within 2 m of a
                 blacklisted point dropped BEFORE the cap, nearest first,
                 at most 64
  area_coverage  observed fraction of the polygon's cells
  walls          occupied cells within 8 m of the vehicle, nearest first,
                 at most 64
```

That bound is what keeps a replay honest: the mission's input is small, fully
archived, and reproduces its decisions exactly.

## The state machine

```
                   ┌─────────────────────────────┐
                   │  no committed goal          │
                   └─────────────┬───────────────┘
                                 │ _propose()
                                 ▼
            strategy.propose(obs, area, blacklist)
                                 │
          ┌──────────────────────┼─────────────────────────┐
          │                      │                         │
     candidate is           candidate is              no candidate
     outside evidence       inside coverage                 │
     coverage                    │                          ▼
          │                      ▼                  no frontiers left?
     blacklist it,          COMMIT to it,           ──> Done('complete',
     try again (up to       Goal(xy, reason)            'no frontier left to
     8 per step)                 │                        explore (N blacklisted)')
          │                      │                  otherwise
          ▼                      │                  ──> Done('complete',
     8 tried in one step         │                     'no reachable goal left:
     ──> Wait(...)               │                      <strategy> proposes none')
                                 ▼
                   ┌─────────────────────────────┐
                   │  committed                  │
                   └─────────────┬───────────────┘
                                 │
        ┌────────────────────────┼─────────────────────────┐
        │                        │                         │
   arrived(committed)     dnav refusing for          no progress of 0.25 m
        │                 unreachable_s                for stuck_s
        ▼                        │                         │
   drop it, propose        blacklist it                blacklist it
                           'dnav: <status>'           'stuck for N s'
                                 │                         │
                                 └───────> propose <───────┘
```

Checked before any of that, on every step:

```
  no area in the profile                 ──> Done('failed', 'no area in the profile')
  coverage_pct >= coverage_target_pct    ──> Done('complete', 'area coverage N%
                                               reached the M% target')
```

### Commitment

One goal at a time, held until it arrives or is given up. This is not a
detail: dway stops the vehicle to re-plan, so a mission that re-proposed every
step would ratchet forward in stutters (B7). The runner's debounce is a second
guard on the same problem, and `operator=False` on every strategy goal is what
subjects them to it.

### The blacklist

A point is given up on — with its reason recorded — when:

| Reason | Condition |
|---|---|
| `outside coverage` | the proposal lies outside the evidence coverage extent. A strategy's goal never resets the map (B1), so the proposal is discarded, not obeyed |
| `dnav: goal_unreachable` / `dnav: outside_coverage` | a verdict about the point itself — an obstacle, its margin, or ground off the map. Given up as soon as it is **confirmed** (`GOAL_REFUSAL_S`, 0.5 s), not waited out: nothing about standing still can make an obstacle's margin passable |
| `dnav: no_route` / `dnav: start_blocked` | a refusal that may be transient — a plan made for the previous revision, a vehicle still braking — held for `unreachable_s` before it counts |
| `dnav: <status>; flew to the nearest reachable point instead` / `; stopped at the last clear point short of it` | dnav offered a fallback, the vehicle flew it, and dway is holding at its end. That is as close as permission allows, so the frontier is given up having been observed from there |
| `dnav: <status>; the route toward it stopped closing on it` | a fallback was on offer but the vehicle stopped getting closer for `unreachable_s` |
| `stuck for N s` | the distance to it stopped improving by `PROGRESS_M` (0.25 m) for `stuck_s` |

The four `dnav:` rows are one decision, `refusal_verdict` in
[`apps/dmsn/inputs.py`](../../apps/dmsn/inputs.py), shared with the search
mission. Its shape comes from a measured failure: in the 17 Sep area1 search
run nine of sixteen frontier proposals were refused as "an obstacle or inside
its 0.35 m margin" and each one cost five seconds of waiting out a fact that
was never going to change — on top of three seconds of dnav computing a
substitute route that nobody ever flew. A frontier sits at the edge of
observed space by construction, so this is the normal case, not the odd one.

**Why a fallback is deferred to rather than counted.** A planning status of
`goal_unreachable` does not mean the vehicle has nowhere to go. dnav answers
with the last good route as far as it is still clear (`retained`), or with a
route to the nearest reachable point within `goal_substitute_m` (`substitute`).
Flying either is the useful thing to do: it puts the vehicle close enough to
*observe* the frontier, which is the whole point of going there, and observing
it is also what can clear the refusal — a never-observed cell beside a wall
counts as solid until somebody looks at it. In the 17 Sep 14:27 area1 run a
goal was refused for two seconds and planned again afterwards, with the vehicle
flying a permitted retained route throughout.

So while an eligible fallback is on offer the refusal does not count, as long
as the vehicle is still closing on the goal. The clock is what bounds that:
suppression with no bound would pin the mission to an unreachable goal for
ever.

Blacklisted points feed back into the frontier distillation: frontiers within
`FRONTIER_EXCLUDE_M` (2 m) of one are dropped **before** the 64-point cap, so
dead ends can never crowd out the frontiers still worth flying to. The
strategies apply the same exclusion to their own proposals.

`PROPOSALS_PER_STEP` is 8: the mission will discard up to eight
out-of-coverage proposals in one step, then `Wait` and ask again on the next
observation rather than spin.

## Parameters

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `coverage_target_pct` | float 0..100, optional | unset | finish once this much of the area is observed. **Unset means finish when the strategy runs out of goals** |
| `stuck_s` | float, s | 20.0 | how long without getting closer to the goal counts as stuck |
| `unreachable_s` | float, s | 5.0 | how long dnav may refuse a goal before it is blacklisted |

The strategy and its own parameters come from the profile's `strategy` and
`strategy_params`; see [`strategies.md`](strategies.md).

## How it ends

| Outcome | Reason |
|---|---|
| `complete` | `area coverage N% reached the M% target` |
| `complete` | `no frontier left to explore (N blacklisted)` |
| `complete` | `no reachable goal left: <strategy> proposes none (N blacklisted)` |
| `failed` | `no area in the profile`, or `no goal strategy in the profile` |
| `budget` | the runner's time, path or battery budget |
| `aborted` | the operator aborted |

Running out of goals is `complete`, not `failed`: the mission did what it could
with the evidence it had, and the blacklist count in the reason says how much
it could not reach. An explore that covers the area badly is acceptable **as
long as it shows what it is doing and ends with a reason** — that is the
standard these missions are held to, not coverage quality.

## The overlay

Everything the last observation told the mission, remembered in `step` so the
picture is derived from observations only and a replay draws the same thing:

```
  ·  frontier dots (accent, small circles)
  ✕  blacklisted points, crossed out (danger)
  ◯  the committed goal, ringed (ok, heavy)
     "N% covered" labelled at the area's centre (dim), or "unmeasured"
     + whatever the strategy draws (wall_follow draws the wall it is following)
```

## Where it lives

| Thing | File |
|---|---|
| The mission | [`apps/dmsn/missions/explore.py`](../../apps/dmsn/missions/explore.py) |
| Frontier distillation | `_evidence_view` in [`apps/dmsn/runner.py`](../../apps/dmsn/runner.py) |
| Polygon geometry | [`apps/dmsn/geometry.py`](../../apps/dmsn/geometry.py) |
| Strategies | [`apps/dmsn/strategies/`](../../apps/dmsn/strategies/) |
| Tests | `tests/dmsn/test_dmsn_missions.py`, `tests/dmsn/test_dmsn_flight.py`, `tests/dmsn/test_dmsn_geometry.py` |
