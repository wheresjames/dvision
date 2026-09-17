# Goal strategies: `frontier` and `wall_follow`

A strategy answers one question — **where next?** — and nothing else. It only
proposes; the mission owns commitment, blacklisting and termination. That
division is what lets a second, structurally different explorer exist without
touching the mission, and proving the seam is real is half the reason the
second one is here.

```python
class GoalStrategy(Protocol):
    name: str; schema: ParamSchema
    def propose(self, obs, area, blacklist) -> Candidate | None: ...
    def overlay(self) -> list[Shape]: ...
```

`Candidate(xy, reason)` — the reason becomes the goal's tag and is kept in the
archive, so every goal in a report says *why* it was chosen. Returning `None`
means "nothing to propose", and what that means is the mission's decision.

A new strategy is a module in [`apps/dmsn/strategies/`](../../apps/dmsn/strategies/)
and a line in `STRATEGIES`. Used by [`explore`](explore.md) and, through it,
[`search`](search.md); selected by the profile's `strategy` and configured by
`strategy_params`.

---

## `frontier`: the nearest frontier

The baseline. A **frontier** is an observed free cell touching a never-observed
one — the edge of what the vehicle knows — and the strategy flies to the
closest one.

```
  the evidence, as the runner distils it

      ·  never observed        ▓  occupied        ░  observed free
      F  frontier: observed free, 4-adjacent to never-observed

      ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
      ▓░░░░░░░░░░░F·········▓·        the vehicle has seen the corridor it
      ▓░░░░░░░░░░░F·········▓·        flew down; the openings at its far end
      ▓░░░░◉░░░░░░F·········▓·        and the side passage it passed are
      ▓░░░░░░░░░░░F·········▓·        frontiers
      ▓▓▓▓FF▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓·
      ▓···░░···················        ◉ the vehicle
      ▓························

  propose() = the frontier minimising (distance, x, y) — here the side
  passage below, not the far end of the corridor
```

The whole strategy:

```python
points = obs.frontiers                              # already nearest-first, capped at 64
points = [p for p in points if point_in_polygon(p, area)]
points = [p for p in points if all(dist(p, b) > 2.0 for b in blacklist)]
if not points: return None
return Candidate(min(points, key=lambda p: (dist(p, obs.xy), p[0], p[1])), ...)
```

No map, no grid, no geometry beyond a point-in-polygon test: the strategy only
ever sees what the observation carries (Q28). The tie-break on `x` then `y`
after distance is there so two equidistant frontiers are resolved
deterministically, which a replay depends on. With no pose yet it takes the
first point, which is already the nearest the runner found.

`AVOID_BLACKLIST_M` is `FRONTIER_EXCLUDE_M` (2 m) — the same radius the runner
uses when it drops frontiers near blacklisted points, so the strategy and the
distillation agree about what counts as "the same dead end".

**Parameters:** none.

**Behaviour to expect.** Greedy nearest-frontier reads, in a maze, as a
depth-first plunge down whichever opening is closest, with backtracking when a
branch dead-ends. It is not a lawnmower, a spiral or a boustrophedon: there is
no precomputed pattern at all, and the path is a consequence of what the
evidence revealed and in what order. Two runs over the same map with slightly
different evidence take different paths, legitimately.

It is also why exploration finishes: every frontier either gets visited (and
stops being a frontier) or gets blacklisted, and when the list empties the
mission is done.

---

## `wall_follow`: the right-hand rule, by goal stepping

The second strategy, and deliberately a *reactive* one — with one twist that
makes it fit the pipeline: it **steps goals, it never steers the vehicle**. A
truly reactive loop would need a velocity-command path through dway, which is
out of scope; being able to add this one without framework changes is itself
the extensibility test.

It is expected to explore **poorly**. That is fine and was known when it
landed.

```
  keep the nearest occupied evidence on one hand and step along it

                      ░░░░░░░░░░░░░░░░░░░░░░
     travel ──────>   ░ standoff_m (1.5 m) ░
                      ░░░░░░░░░░░░░░░░░░░░░░
       ◉ ─────────────────────────────> ✚          ✚ = proposed goal,
       │              step_m (4 m)                      step_m ahead
       │ normal (toward the nearest wall)
       ▼
     ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓            the followed wall,
                                                    kept on the RIGHT hand
```

One proposal, step by step:

```
  1. walls = obs.walls within reach_m (6 m)           the observation's occupied cells
     none? ──> carry straight on in the current direction, looking for one

  2. wall = the nearest of them
     normal = unit vector from the vehicle TOWARD it
     direction = rotate(normal, −90°)                 wall on the right means
                                                      travelling a quarter turn
                                                      LEFT of the normal
  3. pull = clamp(distance − standoff_m, ±step_m/2)   too far from the wall: drift in
     offset = normal · pull                           too close: drift out

  4. target = vehicle + direction·step_m + offset
```

And the recovery, which is where the right-hand rule actually shows:

```
  a target that leaves the area, lands on a wall, or lands on the blacklist is
  turned AWAY from the wall hand, 45° at a time — "blocked ahead, turn left"

        turn 0      turn 1      turn 2      ...  up to TURNS = 8
         ✚           ✚           ✚
         │            ╲           ╲                all eight rejected?
         ◉             ◉──✚        ◉                ──> None: boxed in on
         │                                              every side, and the
      ▓▓▓▓▓        ▓▓▓▓▓       ▓▓▓▓▓                    mission decides what
                                                        that means
```

Rejection tests applied to every candidate target, in order:

| Test | Threshold |
|---|---|
| inside the area polygon | the profile's polygon |
| not near a blacklisted point | `AVOID_BLACKLIST_M` = 1.0 m |
| clear of every known wall | `min(standoff_m, step_m) · 0.5` |

The `offset` nudge is applied **only to the unturned candidate** (turn 0): once
the strategy is turning away from an obstacle, drifting toward the wall it was
following is the wrong correction.

The direction is seeded from the vehicle's heading on the first proposal
(`obs.pose[3]`) and carried between proposals, so the follower has a sense of
which way along the wall it was going. With no pose or no wall in reach it
keeps going straight.

**Parameters** (`strategy_params` in the profile):

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `step_m` | float 0.5..50, m | 4.0 | how far along the wall each proposed goal is (B7's default) |
| `hand` | choice | `right` | which side of the vehicle the followed wall stays on |
| `standoff_m` | float 0.25..10, m | 1.5 | distance kept from the followed wall |
| `reach_m` | float 0.5..8, m | 6.0 | a wall further than this is not followed; carry straight on |

**Overlay:** the wall point being followed (warn, box), a dashed line to the
proposed goal, and an `R`/`L` label for the hand — so a screenshot shows not
just where it is going but what it thinks it is following.

**Behaviour to expect.** With `step_m` at 4 m the mission retargets often,
which is exactly the case B7's debounce exists for; `min_shift_m` (1 m) is
comfortably cleared by a 4 m step, but `min_commit_s` (3 s) will pace it. It
ends only when boxed in on all eight headings or when a budget ends it — unlike
`frontier`, it has no notion of having seen everything, so it will happily
circle a room it has already mapped.

---

## Deliberately not here

`Trémaux` — a coarse cell graph from the evidence, where the next goal is the
least-visited neighbouring junction — is designed and deferred past v1. It
would be a third module and a third registry line, and the fact that it needs
nothing else changed is the point of this seam.

## Where it lives

| Thing | File |
|---|---|
| Protocol, `Candidate`, registry | [`apps/dmsn/strategies/__init__.py`](../../apps/dmsn/strategies/__init__.py) |
| `frontier` | [`apps/dmsn/strategies/frontier.py`](../../apps/dmsn/strategies/frontier.py) |
| `wall_follow` | [`apps/dmsn/strategies/wall_follow.py`](../../apps/dmsn/strategies/wall_follow.py) |
| Frontier and wall distillation | `_evidence_view` in [`apps/dmsn/runner.py`](../../apps/dmsn/runner.py) |
| Tests | `tests/dmsn/test_dmsn_missions.py` |
