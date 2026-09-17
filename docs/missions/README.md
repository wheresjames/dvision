# Mission documents: dmsn

This directory holds one focused document per **mission type** in dmsn, plus
one for the goal strategies the area missions plug in. Each document records
what the mission decides, the state machine behind it, every parameter with its
default, how it ends, and what it draws on the map.

What these documents do **not** repeat: how the chain is launched, the profile
file format, the window's tabs, the report layout. That is
[`docs/missions.md`](../missions.md),
[`docs/reports.md`](../reports.md) and the README. The perception side — the
evidence and detections these missions consume — is in
[`docs/algorithms/`](../algorithms/README.md); the planners that fly a mission's
goals are in [`docs/planners/`](../planners/README.md).

| Mission | One line | Editors | Document |
|---|---|---|---|
| `goto` | fly to one point | goal | [`goto.md`](goto.md) |
| `waypoints` | fly a list of points in order | waypoints | [`waypoints.md`](waypoints.md) |
| `explore` | cover an area with the goals a strategy proposes | area | [`explore.md`](explore.md) |
| `search` | explore until a target estimate is confirmed, then land on it | area | [`search.md`](search.md) |
| — | the goal strategies: `frontier`, `wall_follow` | — | [`strategies.md`](strategies.md) |

## What a mission is

A mission is **pure logic over an observation**. It never opens a plane, reads
a clock, or asks a module anything:

```
     ┌──────────────────────────────────────────────────────────┐
     │ MissionRun (the runner)                                  │
     │                                                          │
     │  planes, clock, ports ──> Observation ──┐                │
     │                                          │               │
     │                                          ▼               │
     │                                   mission.step(obs)      │
     │                                          │               │
     │                                          ▼               │
     │              Goal / Action / Wait / Done ──> debounce,   │
     │                                              intents,    │
     │                                              phases      │
     │                                                          │
     │  both halves go into the archive ──> replay compares     │
     └──────────────────────────────────────────────────────────┘
```

That is what lets a recorded run be replayed through the same mission body and
checked **decision for decision**, and what lets a mission be tested with no
processes at all. The runner even hands the mission
`Observation.from_dict(obs.as_dict())` — the *archived*, rounded form — so a
replay meets the same threshold comparisons the live run did.

Take-off, going home and the finish action belong to the runner. A mission
implements only its body:

```python
class Mission(Protocol):
    name: str; label: str; schema: ParamSchema
    editors: tuple[str, ...]        # 'goal' | 'area' | 'waypoints'
    reference_default: bool         # does the reference background open enabled (Q10)

    def start(self, params, obs, *, area=(), waypoints=(), strategy=None) -> None: ...
    def step(self, obs) -> Decision: ...
    def on_event(self, event) -> None: ...
    def overlay(self) -> list[Shape]: ...
    def progress(self) -> dict: ...
    def summary(self) -> dict: ...
```

A new mission is a module in [`apps/dmsn/missions/`](../../apps/dmsn/missions/)
and a line in `MISSIONS`; nothing above it changes. `editors` is what keeps the
window honest — the Mission tab shows the tools a mission asks for without
knowing which mission it is.

## The four decisions

| Decision | Meaning |
|---|---|
| `Goal(xy, tag, operator)` | set or keep this goal; the runner debounces unchanged ones. `operator=True` (a click, a retarget, going home) skips the debounce |
| `Action(kind, args)` | an executor action by name: `launch`, `hold`, `resume`, `yaw_to`, … published as a retained intent dway may refuse |
| `Wait(reason)` | nothing to do; the reason becomes the run's **blocking fact** |
| `Done(outcome, reason, finish)` | over: `complete`, `failed`, `aborted`, `budget`. `finish` may name the finish action (`land`) instead of the profile's |

Decisions are plain values with a dictionary form, so the archive holds them
exactly and a replay compares them exactly.

## What every mission may know

`Observation` ([`apps/dmsn/inputs.py`](../../apps/dmsn/inputs.py)) is
immutable, bounded, and carries only what a mission legitimately has: pose,
goal and authority, dway's execution state, dnav's route status, the evidence
coverage extent, battery, map name, the provider's published target. **No
grids, no world.**

Three bounded exceptions, each capped the same way every other field is:

- **Area missions** get the evidence plane distilled into `frontiers` (at most
  64, nearest first), `walls` (at most 64 occupied cells within 8 m) and
  `area_coverage` (the observed fraction of the mission polygon), so an
  explore mission decides without a grid and a replay still reproduces it.
- **Missions that declare a target** (`target_label` in their schema) get
  `detections` and `estimates`, filtered to that label. Every other mission
  never opens the detections plane.
- Nothing else. A mission that wants more has to declare it.

### The questions a mission asks, answered once

These live on `Observation` so that every mission answers them the same way:

| Question | Definition |
|---|---|
| `arrived(xy)` | dway reports `COMPLETE` for **this coordinator's** goal at `xy`, for the goal revision now in the session. The one definition of arrival |
| — | how a refused goal is judged from those answers is one function, `refusal_verdict`, shared by every mission that can give a goal up |
| `route_refusal(xy)` | dnav's planning status for this goal — `goal_unreachable`, `no_route`, `outside_coverage`, `start_blocked` — and only for a plan made for the *current* goal revision, so a refusal of the previous goal is never blamed on this one |
| `fallback_offered(xy)` | dnav is offering an eligible route toward this goal although it cannot plan one to it — the last good route as far as it is clear (`retained`), or a route to a reachable point near it (`substitute`). Either way there is somewhere to fly, and flying it is what can clear the refusal |
| `fallback_reached(xy)` | the vehicle flew one of those and dway is holding at its end. dway's arrival gate is measured to the goal itself, so a fallback ends in a `prefix` hold and never in `COMPLETE` — which is why this is a question of its own rather than a case of `arrived` |
| `goal_is_ours` / `goal_at(xy)` | is the session goal ours, and is it this point |
| `airborne` | target mode says so; dynamic mode requires it |

## The runner's phases

```
  idle ──> starting ──> running ──> finishing ──> done
             │            │            │
             │            │            └─ hold where it is / fly home and hold / land
             │            │
             │            └─ the mission body steps on each paced observation (10 Hz)
             │
             └─ check the arena, request coverage while still on the ground (B1, Q24),
                keep the takeoff point as home, ask dway to launch

              paused  ── sits beside the active phases
                         operator pause, lost goal authority, or dway failing.
                         dmsn NEVER takes the authority or the lease back (Q9);
                         resume asks for an explicit handoff.
```

dmsn holds no control lease. It moves the vehicle only by setting the session
goal as the `mission` authority and by publishing intents, which dway may
refuse.

### Budgets

Checked in `starting` and `running`, each with its own action:

| Budget | Spent when | Typical action |
|---|---|---|
| `time_s` | elapsed sim time since the body started | `land` |
| `path_m` | flown path length | `hold` |
| `battery_reserve_pct` | battery below the reserve | `home` |

A spent budget ends the run with outcome `budget` — unless the mission owes a
more specific ending, which it declares by implementing `budget_spent()`.
`search` is the one that does: without a landing it ends `failed: no confirmed
target`, because "the time ran out" is not the answer to "did you find it?"

### The goal debounce (B7)

A goal change costs a full stop in dway, so a strategy that retargets often
would crawl. Non-operator goals are suppressed unless they clear both:

```
  min_commit_s (3.0 s)   since the last goal was sent
  min_shift_m  (1.0 m)   from the last goal's position
```

Operator goals — clicks, retargets, going home — skip it entirely.
Suppressions are counted and noted, not hidden.

A goal outside the evidence coverage is **not** sent at all: the runner reports
the blocking fact and offers `Expand coverage`, which resets the map (B1 — a
strategy's goal never resets the map behind the operator's back).

## Profiles

One JSON file per mission profile in
[`assets/mission_profiles/`](../../assets/mission_profiles/):

```json
{
  "schema": "dvision2.mission-profile.v1",
  "name": "maze_020-search",
  "mission": "search",
  "params": { "...": "validated against the mission's ParamSchema" },
  "finish_action": "land",
  "coverage_margin_m": 10.0,
  "arena": "maze_020",
  "strategy": "frontier",
  "strategy_params": {},
  "budgets": {"time_s": 900.0, "time_action": "land",
              "battery_reserve_pct": null, "battery_action": "home",
              "path_m": null, "path_action": "hold"},
  "debounce": {"min_commit_s": 3.0, "min_shift_m": 1.0},
  "area": [[2, 2], [65, 2], [65, 33], [2, 33]],
  "waypoints": []
}
```

Parameters are declared by the mission as a `ParamSchema` of typed `Field`s
with units, ranges and help text, which is what both the profile validator and
the Mission tab's editor read — so a parameter exists in exactly one place.

## Reading a mission's document

Each one is laid out the same way:

1. **What it decides** — and the smallest thing that makes it interesting.
2. **The state machine** — a diagram, then each transition.
3. **Parameters** — every field, with its default.
4. **How it ends** — every outcome and the reason text it produces.
5. **The overlay** — what it draws, so a screenshot is readable.
6. **Where it lives** — source, tests.
