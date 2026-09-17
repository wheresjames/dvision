# `waypoints`: fly a list of points in order

The points come from the profile's `waypoints` list — the Mission tab edits
them, or a tour import writes them. The mission flies them in order and has
exactly one policy knob: **what to do when a point never arrives**.

Profile: `maze_020-waypoints`. Editor: **waypoints**. The reference background
opens disabled.

## What it decides

```
  step(obs)
     │
     ├─ no waypoints in the profile ──> Done('failed', 'no waypoints in the profile')
     │
     ├─ arrived(waypoints[i]) ──> mark visited ──> advance
     │
     ├─ dnav refusing for unreachable_s ──> give up (policy)
     ├─ no progress for stuck_s          ──> give up (policy)
     │
     └─ Goal(waypoints[i], "waypoint i+1")
```

```
  waypoint states, one per point, reported and drawn

    pending ──> visiting ──┬──> visited
                           ├──> skipped   (policy = skip)
                           └──> failed    (policy = abort)
```

## When a point "never arrives"

Two independent tests, because they fail differently:

**dnav refuses to plan to it** — `route_refusal` returns one of
`goal_unreachable`, `no_route`, `outside_coverage`, `start_blocked`, and has
done so continuously for `unreachable_s` (5 s). Only a plan made for the
*current* goal revision counts, so a refusal of the previous waypoint is never
blamed on this one.

**The vehicle is stuck** — and "stuck" is measured as **progress, not time**:

```
  distance to the waypoint
     │
   8 ┤●   anchor ─┐
     │ ●●         │ closes PROGRESS_M from the ANCHOR ──> reset, new anchor
   6 ┤   ●●●      │
     │      ●●  ──┘
   4 ┤        ●●●●●●●●●●●●●●    no improvement of PROGRESS_M (0.25 m)
     │        └──────────────┘   for stuck_s (20 s) ──> stuck
   2 ┤
     └──────────────────────────> sim time
```

`best` is the distance at the **last reset** and stays there until real
progress moves it — not the distance at the previous observation. That
distinction is the whole rule: measured per step, at 10 Hz, closing 0.25 m
demands 2.5 m/s from a vehicle whose profile allows 1.0, so nothing would ever
count as progress and every leg longer than `stuck_s` would be called stuck
while it was still flying. It used to be measured that way; the 17 Sep area1
search run gave up a goal the vehicle had closed 14 m on.

A long legitimate flight across an empty arena therefore never counts as
stuck, however long it takes — which a plain timeout would get wrong.

**Not applied here:** the refusal handling the area missions gained (defer to
a substitute route, give up a verdict about the point at once). An operator
who typed a waypoint asked for that point, not one within 2 m of it — and dway
cannot report arrival at a substitute's end anyway, so deferring would only
delay the `failure_policy` decision. With `abort` as the default, giving up on
a single observation would also fail a whole mission on one frame of evidence.
Waypoints keep the timer.

## The policy

```
  give up on waypoint i
     │
     ├─ abort (default) ──> states[i] = 'failed'
     │                      Done('failed', 'waypoint N (x, y m) is unreachable: <reason>')
     │
     ├─ retry, attempts < retries ──> re-send the SAME point, reset the stuck clock,
     │                                tag it "retry K"
     │
     └─ skip (or retries exhausted) ──> states[i] = 'skipped', advance
```

`abort` is the careful default: a waypoint list is usually a plan, and silently
dropping part of a plan is worse than stopping. `skip` suits a survey where
coverage matters more than completeness; `retry` suits a point that is blocked
by something transient.

## Parameters

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `failure_policy` | choice | `abort` | `abort`, `skip` or `retry` |
| `retries` | int 0..10 | 0 | with `retry`: attempts per waypoint before it is given up |
| `stuck_s` | float, s | 20.0 | how long without getting closer counts as stuck |
| `unreachable_s` | float, s | 5.0 | how long dnav may refuse before the point counts as unreachable |

The list itself is not a parameter: it is the profile's `waypoints` array, at
most `MAX_WAYPOINTS` (256) points, and it is edited between missions, not
during one — `on_event` deliberately does nothing.

## How it ends

| Outcome | Reason |
|---|---|
| `complete` | `visited N waypoints` — with `, skipped M` appended when any were |
| `failed` | `no waypoints in the profile`, or `waypoint N (x, y m) is unreachable: <reason>` under `abort` |
| `budget` | the runner's budget, with the profile's action |
| `aborted` | the operator aborted |

Note that `complete` is reported even when every point was skipped. The count
is in the reason, the states are in the summary, and "completed having skipped
all 6 waypoints" is a truthful and useful outcome.

## The overlay

Each point as a circle coloured by state — pending (accent), visiting (ok,
heavy), visited (dim), skipped (warn), failed (danger) — with a dim leg drawn
from the previous point, so the intended route and the progress along it are
one picture.

## Where it lives

| Thing | File |
|---|---|
| The mission | [`apps/dmsn/missions/waypoints.py`](../../apps/dmsn/missions/waypoints.py) |
| Tour import (writes the list) | [`apps/dmsn/tour_import.py`](../../apps/dmsn/tour_import.py) |
| Tests | `tests/dmsn/test_dmsn_missions.py`, `tests/dmsn/test_dmsn_flight.py` |
