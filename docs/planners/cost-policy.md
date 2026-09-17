# The cost policy: evidence into numbers

How a belief about the world becomes a number to plan on. This is **dnav's**,
not dalg's, and that split is the architectural content of the whole design:
dalg publishes evidence — how likely a cell is occupied, and when anyone last
looked — and says nothing about what that should cost.

Because the policy is a versioned, digest-carrying **file** rather than a
constant in a planner, two sensors disagreeing about a wall becomes a visible
parameter instead of hidden fusion, and any route can be replanned later from
the exact inputs that produced it.

Source: [`apps/dnav/policy.py`](../../apps/dnav/policy.py). Committed policies:
[`assets/cost_policies/`](../../assets/cost_policies/).

## Version 1 is three rules

1. Occupancy at or above `occupied_threshold` marks an **obstacle**.
2. Obstacles inflate by `inflation_m` — the vehicle's footprint plus its least
   margin — which is what turns "an obstacle is in this cell" into "never come
   closer than this".
3. A band `clearance_preference_m` wide outside that hard margin costs up to
   `clearance_cost` more, rising toward the wall, so routes keep their distance
   where there is room and centre themselves where there is not.

And one rule about combining: **layers combine by `max`, never by sum.**

## The three cost constants

```python
FREE_COST       = 1.0      # a cell somebody has observed to be free
UNOBSERVED_COST = 1.5      # a cell nobody has looked at
BLOCKED         = math.inf # a cell no route may enter
```

Free and never-observed are both **passable** — a planner that refused to enter
the unknown could never leave the room it started in — but they are not equally
attractive, and the 1.0 / 1.5 gap is what makes a route prefer ground somebody
has actually looked at.

`BLOCKED` is infinite rather than merely large because "expensive" and
"forbidden" are different answers, and a planner that can be bribed through a
wall by a long enough detour is not one.

These are constants rather than policy fields because the never-observed cost
is a named later knob; when it becomes one, it becomes a field here and nothing
else changes.

## One layer, end to end

`layer_cost(grid, policy)` turns one source's evidence into one `CostLayer`:

```
  occupancy uint8 (255 = never observed)
        │
        ├─ never       = occupancy == 255
        ├─ probability = occupancy / 254
        ├─ obstacles   = ~never & (probability >= occupied_threshold)
        │
        ▼
  solid_cells(obstacles, never)        obstacles PLUS never-observed cells
        │                              touching one — see clearance.md
        ▼
  obstacle_distance(solid, hard + band)    distance in cells to the nearest
        │                                  full obstacle SQUARE
        ├─ inflated  = distance <= hard              ──> BLOCKED
        ├─ preferred = ~inflated & distance < hard+band
        │
        ▼
  cost = FREE_COST, or UNOBSERVED_COST where never observed
       + clearance_cost * ((hard + band − distance) / band)**2   on `preferred`
       = BLOCKED                                                 on `inflated`
```

### The cost profile across a wall

```
  cost
    │
 4.0┤ ∞ ∞ ∞ │●                          hard   = inflation_m / cell_m
    │ ∞ ∞ ∞ │ ●                         band   = clearance_preference_m / cell_m
 3.0┤ ∞ ∞ ∞ │  ●                        peak   = clearance_cost above free
    │ ▓▓▓▓▓ │   ●●                       (quadratic falloff)
 2.0┤ ▓wall▓│     ●●●
    │ ▓▓▓▓▓ │        ●●●●
 1.0┤ ∞ ∞ ∞ │            ●●●●●●●●●●●●●●●●●●  FREE_COST
    │       │
    └───────┴────────────────────────────────> distance from the wall
      margin │<─ clearance band ─>│  open ground
      BLOCKED    (0.75 m default)

  the band is FINITE: it prefers space around walls without forbidding a
  corridor that meets the hard clearance, and because it rises toward the wall
  it also centres a route in a corridor too narrow to leave the band at all
```

`clearance_cost: 0` disables the band entirely, and then `band` is not even
computed.

### Why the margin is a disc, and measured to the full square

Two geometry choices that each fix a specific failure:

```
  disc, not square                      to the full cell, not its centre

   square dilation keeps 1.4x the        ┌─────┐
   asked-for distance diagonally:        │▓▓▓▓▓│  distance is measured to the
                                         │▓▓▓▓▓│  nearest point of the occupied
     ░░░░░      ░·░·░                    └─────┘  SQUARE, so a margin is not
     ░▓▓▓░  vs  ░▓▓▓░                       ↑     half a cell short on the
     ░░░░░      ░·░·░                    ·──┘     side a wall faces
      disc       square
                                         max(|dx|−0.5, 0), max(|dy|−0.5, 0)
   which quietly closes gaps a vehicle
   of the declared size fits through —
   the doorway case, exactly
```

`disc_offsets(radius_cells)` and `inflate(mask, radius_cells)` implement mask
growth by shifting and ORing (no convolution: the radius is a couple of cells
and this needs nothing beyond numpy). `obstacle_distance` applies the same
full-cell rule but **keeps the distance**, so the band outside the hard margin
can be graded instead of flat — which is what `layer_cost` actually uses.

## Combining layers

```
  source A cost   source B cost        combined
  ┌───┬───┬───┐   ┌───┬───┬───┐       ┌───┬───┬───┐
  │1.0│1.0│ ∞ │   │1.0│ ∞ │ ∞ │  max  │1.0│ ∞ │ ∞ │
  ├───┼───┼───┤   ├───┼───┼───┤ ────> ├───┼───┼───┤
  │1.5│1.0│1.0│   │1.0│1.0│1.0│       │1.5│1.0│1.0│
  └───┴───┴───┘   └───┴───┴───┘       └───┴───┴───┘
```

A cell is as bad as its **worst witness**. Never by sum: summing two sensors'
opinions of the same wall counts that wall twice, which is the exact mistake
that publishing evidence instead of cost exists to avoid.

`CostMap` keeps every layer beside the combined surface, so the Cost tab can
draw which source blocked what, and:

| Property | Definition |
|---|---|
| `never_observed` | the **AND** across layers — unknown only where *no* source has observed the cell |
| `blocked` | `~isfinite(cost)` |
| `revision` | the **sum** of layer revisions: one number for "has the evidence moved" |
| `stale` | any layer stale |
| `margin` (per layer) | `inflated & ~obstacles` — the cells the inflation added, which is what the UI draws |

Every grid must share the geometry. One `cell_m` per profile is what makes
stacking layers an overlay rather than a resample in the middle of every plan;
a mismatch raises, and the planning loop treats that as "never plan across
mixed generations".

## The fields

`CostPolicy`, validated in `__post_init__` — a bad value fails at load, not
mid-plan.

| Field | Default | Meaning |
|---|---|---|
| `name` | `default` | what the policy is called in the UI and the report |
| `occupied_threshold` | 0.5 | occupancy at or above this is an obstacle; must be in (0, 1] |
| `inflation_m` | 0.35 | hard margin: no route may come closer; finite and non-negative |
| `clearance_preference_m` | 0.75 | width of the finite-cost band outside the hard margin |
| `clearance_cost` | 3.0 | extra per-cell cost at the margin's edge, falling quadratically to zero at the band's outer edge. 0 disables the band |
| `body_radius_m` | 0.0 | the body half-width the escape rule must keep clear |
| `escape_margin` | `true` | may a vehicle stopped inside a margin (never inside the obstacle) be routed out through it at high cost |
| `combine` | `max` | the only combiner this version implements |
| `schema_version` | 1 | anything else is refused |

Deliberately **not** here yet: per-source thresholds, per-source weights, and a
tunable never-observed cost. They are later knobs on this same artifact, not
new machinery.

The two committed policies:

```json
{"schema": "dvision2.cost-policy.v1", "name": "default",
 "occupied_threshold": 0.5, "inflation_m": 0.35,
 "clearance_preference_m": 0.75, "clearance_cost": 3.0, "combine": "max"}

{"schema": "dvision2.cost-policy.v1", "name": "cautious",
 "occupied_threshold": 0.5, "inflation_m": 1.1, "combine": "max"}
```

`cautious` is the same rules with a metre of margin and no band — useful for
seeing which corridors a bigger vehicle loses.

## Two digests, and why

```
  parse_policy(raw bytes)      digest = sha256(the FILE's bytes as read)
        │                      names the file that was actually loaded, the
        │                      same discipline a tour and a profile follow
        ▼
  policy.for_execution(profile)
        │  inflation_m       = max(policy, profile.clearance_radius_m)
        │  occupied_threshold= min(policy, profile.occupied_threshold)
        │  body_radius_m     = max(policy, profile.body_radius_m)
        │
        └─ digest = sha256(canonical JSON of the RESULT)
           names the EFFECTIVE policy, which is what every route carries
```

The effective policy is never looser than either input: at least the execution
envelope's margin, and the stricter occupancy cutoff. That is what makes
planning, admission and replay agree — a plan made with a smaller margin than
the executor will admit is a plan that gets withdrawn at the worst moment.

For the `sim-target` profile the effective hard radius works out at **0.35 m**:

```
  clearance_radius_m = max(plan_clearance_m,
                           body_radius_m + mapping_margin_m
                             + min(min_tracking_m, tracking_m))
```

The rest of `tracking_m` is deliberately not reserved as a wall — an earlier
0.65 m hard radius closed a 3 m maze corridor that lidar mapped 2 m wide. What
replaced it is the per-segment `clearance.lateral_m` dway narrows its
cross-track allowance against; see
[`docs/navigation.md`](../navigation.md#clearance-and-frame-details).

Every route carries `map_revision` and `policy_digest`. A route without them is
a route nobody can reproduce, and reproducing a plan is the whole reason the
policy is an artifact rather than a constant.

## What it gets wrong

- **One threshold for every source.** A noisy camera and a lidar get the same
  0.5 cutoff, which is the first thing per-source thresholds will fix.
- **A binary hard margin.** A corridor exactly as wide as the margin is
  impassable, not tight; there is no "squeeze through slowly" cost.
- **The band is quadratic by fiat.** The shape was chosen because it centres a
  route in a narrow corridor and decays smoothly; nothing measured says it is
  the right curve.
- **Two dimensions, one layer index.** Everything here is a horizontal slab.
- **Unknown space is priced by a constant.** 1.5 against free 1.0 is a guess,
  and the one that most changes where routes go.

## Where it lives

| Thing | File |
|---|---|
| Policy, cost map, inflation | [`apps/dnav/policy.py`](../../apps/dnav/policy.py) |
| `solid_cells`, clearance geometry | [`apps/dcmn/clearance.py`](../../apps/dcmn/clearance.py) → [`clearance.md`](clearance.md) |
| Execution envelope | `ExecutionProfile` in [`apps/dcmn/navigation.py`](../../apps/dcmn/navigation.py) |
| Committed policies | [`assets/cost_policies/`](../../assets/cost_policies/) |
| Tests | `tests/dnav/test_dnav_planning.py` |
