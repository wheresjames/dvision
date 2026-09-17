# `goto`: fly to one point

The smallest real mission, and the one every other is measured against: set a
goal, wait for dway to report it complete, finish. How the vehicle gets there —
the route, the stops, the permission — is dnav's and dway's business, and the
mission only reports what they say.

It is also the mission that proves the seams work. If `goto` cannot fly to a
clicked point, nothing above it will.

Profiles: `maze_020-goto`, `maze_020-goto-click`,
`maze_020-goto-provider-target`, and the small `maze_001`/`maze_012` variants.
Editor: **goal**. The reference background opens **enabled** (Q10), because
this is an operator mission.

## What it decides

```
  step(obs)
     │
     ├─ no target yet?
     │     goal_source = 'xy'              ──> Wait("goal_source is xy but no goal is set")
     │     goal_source = 'click'           ──> Wait("click the map to set a goal")
     │     goal_source = 'provider_target' ──> obs.provider_target, if published,
     │                                         else Wait("no provider target: <reason>")
     │
     ├─ obs.arrived(target)  ──> Done('complete', "arrived at X, Y m")
     │
     └─ Goal(target, tag, operator)
```

Where the target comes from, and whether it counts as **operator intent** (and
so skips the debounce), is the whole of this mission's configuration:

| `goal_source` | The point comes from | `operator` |
|---|---|---|
| `click` | a map click, at any time — the profile's `goal` seeds it if set | yes |
| `xy` | the profile's `goal` field | yes |
| `provider_target` | the target the simulator publishes; assisted, and recorded as such | no, until the operator retargets |

## Retargeting

A `click` or `retarget` event with a point replaces the goal **at once**, in
any goal source, while flying:

```python
def on_event(self, event):
    if event.kind in ('click', 'retarget') and event.xy is not None:
        if self.target is not None: self.retargets += 1
        self._set(event.xy, operator=True)
```

The new goal is operator intent, so the debounce does not delay it: an operator
who clicks gets a new goal on the next step, and dway stops and re-plans. The
count is reported, and once the operator has retargeted a `provider_target`
run, the tag stops claiming the point came from the provider.

## Parameters

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `goal_source` | choice | `click` | `click`, `xy` or `provider_target` |
| `goal` | point | — | `x,y` in local map metres; a map click fills it in |

## How it ends

| Outcome | Reason |
|---|---|
| `complete` | `arrived at X, Y m` |
| `budget` | the runner's time, path or battery budget, with the profile's action |
| `aborted` | the operator aborted |
| `failed` | only from the runner: dway failing before the vehicle was airborne, for instance |

Note what is **not** here: `goto` never gives up on an unreachable point. A goal
dnav refuses shows as the run's blocking fact and the mission keeps asking,
because an operator who clicked a wall should see *why* nothing is happening
rather than have the mission quietly end. Giving up is
[`waypoints`](waypoints.md)' and [`explore`](explore.md)' business, where there
is something else to try.

## The overlay

Previous targets, dimmed — so a retarget-heavy run reads as a trail of
where the operator sent the vehicle before. The current goal is drawn by the
runner, not the mission.

## Where it lives

| Thing | File |
|---|---|
| The mission | [`apps/dmsn/missions/goto.py`](../../apps/dmsn/missions/goto.py) |
| Tests | `tests/dmsn/test_dmsn_missions.py`, `tests/dmsn/test_dmsn_flight.py` |
