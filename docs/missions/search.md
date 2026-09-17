# `search`: find a target and land on it

Explore an area until a detector's target estimate is confirmed, look again
from a standoff, and land on it. The mission **runs no detector**: it declares
the label it needs and reads dalg's detections plane (§13.6). The runner
refuses to start it unless some dalg detector publishes that label.

Profile: `maze_020-search`. Editor: **area**. Default strategy: `frontier`.
The reference background opens disabled (Q10) — it would leak the answer.

```sh
python3 ./apps/dalg/dalg.py --id area1 --profiles lidar-baseline red-disc &
python3 ./apps/dmsn/dmsn.py --id area1 --profile maze_020-search
```

Without a detector publishing `landing_pad`, start is refused. That is a
feature: a search that silently never sees anything is indistinguishable from a
search of an empty room.

## The strategy, in one paragraph

Search has **no search geometry of its own**. Area coverage is delegated
wholesale to an embedded [`ExploreMission`](explore.md) with its
[strategy](strategies.md) — `frontier` by default, so the pattern is greedy
nearest-frontier, not a lawnmower or a spiral. What search adds is everything
that happens *once a target estimate appears*: back off to a standoff, turn to
face it, require it to still be there, then fly to it and land. Every one of
those decisions is made on observations alone, and each is logged with the
estimate's ID and revision so the summary links back to dalg's archive.

## The state machine

```
                      ┌──────────────────────────────────────────────┐
                      │ explore                                      │
                      │   delegate every step to ExploreMission      │
                      │   check each observation for a usable        │
                      │   estimate                                   │
                      └──────────┬────────────────────┬──────────────┘
                                 │                    │
                 usable estimate │                    │ explore ends
                                 ▼                    ▼
                      ┌──────────────────┐   Done('failed', 'no confirmed
                      │ standoff         │     target: <explore's reason>;
                      │   Goal: a point  │     gave up N estimate(s)')
                      │   standoff_m     │
                      │   short of the   │
                      │   estimate, on   │
                      │   the line from  │
                      │   the vehicle    │
                      └────────┬─────────┘
                       arrived │
                               ▼
                      ┌──────────────────┐
                      │ face             │   Action('yaw_to', heading)
                      │   wait for dway's│   a refused or failed turn is noted
                      │   ack, or 60 s   │   and the mission looks anyway
                      └────────┬─────────┘
                               ▼
                      ┌──────────────────┐
                      │ observe          │   Wait observe_s, then require the
                      │                  │   estimate to be confirmed, fresh,
                      │                  │   AND seen again since observing began
                      └───┬──────────┬───┘
                      yes │          │ no
                          │          ▼
                          │   attempts < max_standoff_attempts?
                          │      yes ──> back to `face` ("look again")
                          │      no  ──> give up this estimate, back to `explore`
                          ▼
                      ┌──────────────────┐
                      │ approach         │   Goal: the estimate itself
                      └────────┬─────────┘
                       arrived │
                               ▼
                   Done('complete', 'landed on estimate <id>
                        (spread N m, M observations)', finish='land')
```

At any point in `standoff`, `face` or `observe`, an estimate that stops being
published or goes stale sends the mission back to `explore` — but **not** in
`approach`. Once it has committed to flying to the estimate it finishes the
approach on the last known position, because a pad that stops being visible as
the camera pitches past it is the expected case, not a failure.

## Why the standoff, the turn and the second look exist

All three exist for one reason (B6): **the camera cannot see the pad it is
landing on.** `camera-lidar`'s front camera is pitched −5° at 1.5 m, and sees
the ground only from about 2.3 m ahead.

```
  side view, at the estimate                 side view, from the standoff

     ◉ camera, −5°                              ◉ ── − − − − − ─┐
     │ ╲                                        │  ╲            │ inside the
     │  ╲__ ground footprint starts             │   ╲___________ ▼ footprint
     │     ╲    ~2.3 m ahead                    │        ╲      ⊚
   ──┴──────╲──────────────────                ─┴─────────╲─────⊚───────
     ⊚ the pad, directly below:                            standoff_m (4 m)
       INVISIBLE                                           the pad is VISIBLE
```

So: stop `standoff_m` short, on the line from the vehicle to the estimate;
`yaw_to` the estimate's bearing so it is in frame; watch for `observe_s` and
require a *fresh* observation in that window. Then fly to the estimate's
coordinates and land.

**The landing is on the estimate, never a visual servo.** Precision landing
waits for a downward camera in a new drone profile (Q7). Landing on the pad is
not a v1 requirement; ending with a reason is.

## What makes an estimate usable

```python
def _usable(self, estimate):
    return (estimate['confirmed'] and not estimate['stale']
            and estimate['confidence'] >= self.params['min_confidence']
            and estimate['id'] not in self.given_up)
```

`confirmed` is dalg's — `confirm_n` (3) observations — and `stale` is dalg's
too (`expire_s`, 30 s unseen). Search adds its own confidence floor and its own
memory of what it has already given up on.

### Which candidate it picks

```python
min(usable, key=lambda e: (−e['count'], dist(here, e['xy_m']), e['id']))
```

**Evidence weight first, distance second.** The most-observed estimate wins;
distance only breaks ties between equally observed ones, and the ID breaks the
rest deterministically. A pad seen thirty times across the room beats a
two-sighting flicker at arm's length, which is the right priority for a
detector whose false positives are exactly those flickers.

### The second look

In `observe`, being confirmed is not enough:

```python
seen = estimate['last_s'] >= self.observe_s      # seen since observing BEGAN
if self._usable(estimate) and seen: ...approach
```

A confirmed estimate is a statement about the past. Requiring a **new**
observation from the standoff, facing it, is what distinguishes "there is a pad
there" from "there was something red there a while ago". When it fails, the
reason is recorded as either `not seen again from the standoff` or `no longer
confirmed or confident enough`, and the mission looks again — up to
`max_standoff_attempts` (2) times — before giving the estimate up and resuming
exploration (Q26).

### Giving up, and remembering

A refusal is judged by `refusal_verdict`, the same rules
[`explore`](explore.md#the-blacklist) applies to a frontier: a standoff or an
approach point is computed geometry too, and one that lands inside an obstacle
margin is no more reachable for being waited on. A substitute route dnav
offers is flown before the estimate is given up, since getting closer is also
what may re-observe it.

`given_up` maps estimate ID to reason, and a given-up estimate can never
become usable again. The reasons are every way a target can fail:

| Reason | Phase | Comes back? |
|---|---|---|
| `estimate went stale` / `estimate no longer published` | standoff, face, observe | yes |
| `standoff unreachable: dnav <status> after N standoff(s)` | standoff | yes |
| `stuck short of the standoff for N s after N standoff(s)` | standoff | yes |
| `not seen again from the standoff after N look(s)` | observe | no |
| `no longer confirmed or confident enough after N look(s)` | observe | no |
| `estimate unreachable: dnav <status>` | approach | yes |
| `stuck short of the estimate for N s` | approach | yes |

**A point that could not be flown to is not a bad target.** A standoff is
computed geometry — `standoff_m` short of the estimate, on the bearing back to
the vehicle — so one that lands inside an obstacle's margin says nothing about
the pad. Two things follow, and both were learned from the 17 Sep 14:27 area1
run, which discarded a pad it went on to observe 96 more times:

- **A refused standoff costs an attempt, not the estimate.** The mission moves
  the standoff `STANDOFF_TURN_DEG` (60°) round the estimate and tries again —
  same distance, so the camera footprint is unchanged, different route — up to
  `max_standoff_attempts`, which it shares with the re-observation looks.
- **A give-up for geometry is reconsidered.** The estimate is admitted again
  once the detector has seen it `RETRY_OBSERVATIONS` (20) more times, and its
  attempts reset with it. The evidence that blocked the route changes as the
  vehicle explores, so a target the detector keeps confirming deserves another
  approach. Give-ups about the *target* — not seen again, no longer confident —
  stay permanent, since looking again is exactly what already failed.

## Parameters

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `target_label` | text | `landing_pad` | the label a dalg detector must publish; search refuses to start without one |
| `target_detector` | text | `""` | only this detector ID (e.g. `front-red_disc`); empty accepts any detector of the label |
| `min_confidence` | float 0..1 | 0.3 | detections and estimates below this confidence are ignored |
| `standoff_m` | float 0.5..20, m | 4.0 | how far short of the estimate the vehicle stops to look again |
| `observe_s` | float 0.5..60, s | 3.0 | how long it looks from the standoff before deciding |
| `max_standoff_attempts` | int 1..10 | 2 | looks at one estimate before it is given up |
| `stuck_s` | float, s | 20.0 | how long without getting closer to a goal counts as stuck |
| `unreachable_s` | float, s | 5.0 | how long dnav may refuse a goal before it is given up |

Note what search does **not** expose: `coverage_target_pct`. It starts its
embedded explore with only `stuck_s` and `unreachable_s`, so exploration never
stops at a coverage percentage — it runs until there is no frontier or no
proposal left, or a budget ends the run. `maze_020-search` sets `time_s` to
900 s with `time_action: land`.

## How it ends

| Outcome | Reason |
|---|---|
| `complete` | `landed on estimate <id> (spread N m, M observations)`, with `finish='land'` |
| `failed` | `no confirmed target: <explore's reason>; gave up N estimate(s)` |
| `failed` (budget) | `no confirmed target: <budget>` — or `no confirmed target landed on: <budget> while at <phase> for <id>` |

That last row is `budget_spent()`, the hook the runner calls instead of ending
with a bare `budget` outcome. A search whose clock ran out did not "spend its
budget"; it **failed to find the target**, and if it was mid-approach when the
clock ran out, the reason says which estimate and which phase. "Did you find
it?" has one answer per run, and it is in the outcome.

## The estimate log

Every decision that acted on an estimate is appended to `log` with the
estimate's ID, revision, position and observation count:

```
  go to standoff · face estimate · observe · look again · approach estimate
  · land on estimate · give up
```

The last `LOG_CAP` (256) entries ride in the summary; the archive keeps every
one. That is the chain a report reader follows from "landed here" back to the
individual detections in dalg's archive that put it there.

## The overlay

```
  ·  projected detections (warn, small ovals) — only projectable ones
  ◯  each estimate, radius = its spread (min 0.3 m)
       filled when confirmed and fresh; dim when stale or given up
       ringed in highlight when it is the current target
       labelled with its ID
  □  the standoff point, labelled "standoff"
  □  the approach point, labelled "land"
  +  everything the embedded explore draws: frontiers, blacklist, coverage
```

## Where it lives

| Thing | File |
|---|---|
| The mission | [`apps/dmsn/missions/search.py`](../../apps/dmsn/missions/search.py) |
| The embedded explorer | [`apps/dmsn/missions/explore.py`](../../apps/dmsn/missions/explore.py) |
| Target declaration | `target_of` in [`apps/dmsn/missions/__init__.py`](../../apps/dmsn/missions/__init__.py) |
| Start refusal, detections view | `_check_target`, `detections_view` in [`runner.py`](../../apps/dmsn/runner.py) / [`inputs.py`](../../apps/dmsn/inputs.py) |
| The perception half | [`docs/algorithms/red-disc.md`](../algorithms/red-disc.md), [`detection-pipeline.md`](../algorithms/detection-pipeline.md) |
| Tests | `tests/dmsn/test_dmsn_search.py`, `tests/dmsn/test_dmsn_nightly.py` |
