# Dynamic routes: planning, observation and flight

dnav plans and permits; dway observes (`--dry-run`) or flies. One selected dnav and
one dway per vehicle, fixed altitude, position targets at a fixed heading, a stop at
every corner and permission endpoint, and route changes only from a confirmed stop.
Static tour commands are unchanged. There is no automatic takeoff, landing or RTL.

## Run

### Observation without vehicle commands

The same chain `tests/test_navigation_process.py` runs, against dsim:

```sh
python3 apps/dsim/dsim.py --id area1 --map assets/maps/maze_012.txt &
python3 apps/dalg/dalg.py --id area1 --profile ground-plane-baseline &
python3 apps/dnav/dnav.py --id area1 --goal 5.5,1.5 \
  --execution-profile assets/execution_profiles/dry-run.json &
python3 apps/dway/dway.py --id area1 --mode dynamic --dry-run \
  --execution-profile assets/execution_profiles/dry-run.json
```

or against the deterministic synthetic provider, which hovers at the profile altitude
and therefore shows admission (`READY`) as well as waits:

```sh
python3 dtest/provider.py --id area1 --report-dir reports/area1-dry --sensors scan \
  --path '2,0;2,2.5;2,2.5' --timeout 60 &
python3 apps/dalg/dalg.py --id area1 --profile lidar-baseline --no-ui &
python3 apps/dnav/dnav.py --id area1 --goal 8,2 --no-ui \
  --execution-profile assets/execution_profiles/dry-run.json &
python3 apps/dway/dway.py --id area1 --mode dynamic --dry-run --no-ui --timeout 30 \
  --execution-profile assets/execution_profiles/dry-run.json
```

dnav and dway accept `--no-ui --timeout N` for a bounded headless observation.
In the tested dsim chain, dway never reaches `READY`: it waits on `unknown, stale,
ambiguous or occupied swept cells`, because the ground-plane evidence does not
freshly cover the swept stopping area. That is the intended conservative wait, not
something to work around. Other visible waits include `route/pose outside fixed
altitude` when the vehicle is not at the profile altitude.
Start order does not matter. `--navigation-name experiment` on dnav and
`--planner experiment` on dway select an alternate named planner endpoint.
One dway status writer per vehicle is allowed; another live writer is refused.
The dry run creates no VehicleLink or Mission, even on shutdown or error.

### Flight

Flight needs a **calibrated** profile; the synthetic `dry-run.json` is refused at
startup (`calibrated: false ... cannot authorize dynamic flight`). dnav and dway
must load the same profile, and dway rejects a snapshot whose profile digest differs.

1. Establish the start state with existing controls: armed, airborne at the profile
   altitude (1.5 m for `dsim-position-v1`), in HOLD, then **release** their lease.
2. Start the planner with the same profile:

   ```sh
   python3 apps/dnav/dnav.py --id area1 --goal 12.5,6 \
     --execution-profile assets/execution_profiles/dsim-position-v1.json
   ```

3. Start the executor and press **Start** once the route is admitted:

   ```sh
   python3 apps/dway/dway.py --id area1 --mode dynamic \
     --execution-profile assets/execution_profiles/dsim-position-v1.json
   # headless: Start is requested once, when a route is first admitted
   python3 apps/dway/dway.py --id area1 --mode dynamic --no-ui --start --timeout 300 \
     --execution-profile assets/execution_profiles/dsim-position-v1.json
   ```

Receiving a route never acquires a lease, arms or starts. Start re-checks, with the
lease held, everything it checked before acquiring it. The headless runner exits 0
on COMPLETE, 1 on CANCELLED and 2 otherwise. SIGINT/SIGTERM in the window pauses;
headless, they shut down (bounded HOLD wait, lease release, recording finished).

The fully deterministic flights (`tests/test_dynamic_flight.py`) run the production
publisher, clearance, snapshots, route source and executor against a real
`DroneSimulator` on the loopback transport. Their evidence generator and small A*
planner are declared fixtures in `dtest/dynamic_rig.py`. A live multi-process flight
additionally needs evidence that freshly covers the swept stopping area at the flight
altitude; with ground-plane dalg evidence it waits, as described above.

### Target mode: take off and fly to the goal

Target mode is the research path: one command per module, and dway takes off and
flies dnav's route to dnav's goal without an operator pressing anything.

```sh
python3 ./apps/dsim/dsim.py --id area1 --map ./assets/maps/maze_020.txt --drone-profile camera-lidar --sim-speed 1.5 &
python3 ./apps/dalg/dalg.py --id area1 --profiles lidar-baseline.json optical-flow-baseline.json &
python3 ./apps/dnav/dnav.py --id area1 --goal 52.444,2.389 --execution-profile sim-target &
python3 ./apps/dway/dway.py --id area1 --mode target --wait-for algorithm
```

`--goal` is optional. Without it dnav adopts the map target the simulator publishes
(the `*` cell in the map, on dsim's status plane as `target.lat_deg`/`target.lon_deg`),
converted with the published datum and the context's local NED origin, so no world file
is read. It does this once per session, only when no goal is set, and never replaces
another authority's goal. If the provider publishes no target it does nothing and says
so. `--no-provider-goal` turns this off:

```sh
python3 ./apps/dsim/dsim.py --id area1 --map ./assets/maps/maze_002.txt --drone-profile camera-lidar --sim-speed 1.5 &
python3 ./apps/dalg/dalg.py --id area1 --profiles lidar-baseline.json &
python3 ./apps/dway/dway.py --id area1 --mode target --wait-for algorithm &
python3 ./apps/dnav/dnav.py --id area1 --execution-profile sim-target &
```

Add `--no-ui` to any of them for a headless run; `dway --no-ui --exit-on-finish
--timeout N` ends with exit code 0 on arrival. In target mode dway:

1. waits for every `--wait-for` role to answer `run.prepare` (as the tour does);
2. asks dalg for a mapping reset that includes the goal when the goal lies outside the
   evidence coverage (recorded as `execution.coverage`);
3. acquires an available control lease (never one another client holds), arms, climbs
   to the profile altitude unless it is already there, and confirms HOLD by measured
   low speed (`execution.launch` events: begin, arm, takeoff, hold, airborne);
4. presses Start, and Resume after any stop, whenever a route is admitted. These are
   the ordinary controls, recorded with origin `auto`. An operator Pause or Cancel
   suppresses them;
5. hovers in HOLD at the goal and keeps supervising. It never lands.

dway loads `sim-target` unless `--execution-profile` names another. dnav must load
the same profile with the same overrides; otherwise dway rejects every route with
`execution profile mismatch (planner <digest>, executor <digest>)`.

#### Dials

Every execution-profile field can be overridden on the command line, in both
processes, with the same values:

```sh
python3 ./apps/dnav/dnav.py --id area1 --goal 52.444,2.389 --execution-profile sim-target \
  --profile-set speed_mps=1.5 --profile-set mapping_margin_m=0.15 &
python3 ./apps/dway/dway.py --id area1 --mode target --wait-for algorithm \
  --profile-set speed_mps=1.5 --profile-set mapping_margin_m=0.15
```

| Dial | Where | Effect |
| --- | --- | --- |
| `permission` | profile | `plan` trusts dnav's route through unknown space; `evidence` flies only swept cells with fresh free evidence (the strict mode above) |
| `plan_clearance_m` | profile | floor on the shared hard clearance radius; cannot reduce the body, mapping and least tracking envelope. Distances are measured to full occupied cell squares |
| `mapping_margin_m` | profile | additional map/localization uncertainty allowance (default 0.1 m); choose from measured error for the deployment |
| `tracking_m`, `min_tracking_m` | profile | cross-track allowance with room to spare, and the least allowance a route may leave (default 0.1 m). In between, dway narrows the allowance to the segment's published obstacle clearance and slows down |
| `heading` | profile | `travel` (the `sim-target` default) turns in place on the route to face each segment, then flies it facing that way, so forward sensors such as the front camera look where the vehicle goes; `fixed` holds the heading from Start. The 2d lidar scans 360 degrees either way |
| `turn_tolerance_deg` | profile | with `travel`, the heading error at which a segment may begin |
| `handoff_m` | profile | largest offset from a replacement route at which dway takes it up while moving (`sim-target` 0.5 m); with it, plan permission shortens a route blocked ahead instead of withdrawing it. 0 keeps route changes stop-first |
| `corner_blend_deg` | profile | largest turn flown through without stopping, where dnav cleared the shortcut (`sim-target` 65); 0 stops at every corner |
| `goal_substitute_m` | profile | when the goal cannot be reached and the retained route is used up, route to the reachable point nearest the goal within this radius (default 2 m); 0 disables |
| `speed_mps`, `stopping_m`, `join_m`, `arrival_m`, `max_age_s`, `altitude_m`, `half_height_m`, `hold_*` | profile | following, stopping, arrival, validity window, flight slab and HOLD confirmation |
| `escape_margin` | cost policy JSON (`--policy`) | default `true`: a vehicle that stopped inside an obstacle's inflation margin is routed out through that margin at high cost. Every escape segment must preserve or increase distance to each nearby obstacle and keep the body clear. Occupied or unobserved start cells stay `start_blocked` |
| `inflation_m`, `occupied_threshold` | cost policy JSON | configured planner margin and cutoff; the effective policy uses at least the execution envelope and the stricter occupancy cutoff |
| `clearance_preference_m`, `clearance_cost` | cost policy JSON | finite-cost band outside the hard margin (default 0.75 m wide), costing up to `clearance_cost` (default 3) more per cell at the hard margin and falling quadratically to nothing at its outer edge. Routes keep their distance where there is room and centre themselves in a corridor too narrow to leave the band |

The shared hard radius is
`max(plan_clearance_m, body_radius_m + mapping_margin_m + min(min_tracking_m, tracking_m))`.
For the default `sim-target` profile this is **0.35 m**. The rest of `tracking_m` is not
reserved as a wall: a 3 m maze corridor that lidar maps 2 m wide, with a stray occupied
cell beside the goal, closed under the earlier 0.65 m hard radius
(`reports/area1/20260915-123000-1d464a83`). Instead, every eligible route publishes
`clearance.lateral_m`, one obstacle clearance per segment over its permitted part
(capped at `body_radius_m + mapping_margin_m + tracking_m`). dway brakes when cross-track
exceeds `min(tracking_m, lateral_m - body_radius_m - mapping_margin_m)` and scales its
speed cap by that allowance over `tracking_m`, down to a quarter of `speed_mps`. With
`evidence` permission, unknown and stale cells count against `lateral_m` as they do against admission.
With `plan` permission, never-observed cells touching an observed obstacle count as solid in the
planner and in admission: the thin margin otherwise admits a route through a wall whose end
the lidar has not seen yet (the archived Sep 15 wall-end crash, `tests/assets/corner_clearance/area1.json`).
Braking remains an along-route stopping-distance check in dway; it does not widen
the footprint against parallel walls. Planner inflation, search edges,
route shortening and plan admission account for full occupied cells. Evidence admission
uses a conservative swept rectangle with the same radius. The effective policy, including
its adjusted margin and digest, is recorded for replay. A route through an existing
margin must leave it before ending; nearby occupied cells are never erased to admit it.

At a confirmed corner stop, dnav checks the direct segment from the actual stopped pose
to the next target and publishes `clearance.corner_departure`. Dway waits for permission
bound to its session, segment and stopped pose before advancing. An unsafe departure
withdraws permission and triggers the normal stop/replan handshake. Restart both dnav and
dway after upgrading: the new profile field changes the profile digest.

#### When the goal is blocked

A lidar phantom on or near the goal, or one closing the way far ahead, makes every replan
`goal_unreachable` or `no_route`. That no longer strands the vehicle
(`reports/area1/20260915-131438-3a72d536` held 16 m short behind one stray cell). dnav keeps
publishing, in order:

1. `route_mode: retained`, the last route planned to the goal, permitted up to the
   last clear point before the first observed obstacle. Getting closer lets the sensors
   look again, and distant phantoms often clear. A permission ending within `stopping_m`
   of the vehicle counts as used up.
2. `route_mode: substitute`, a route to the reachable point nearest the goal within
   `goal_substitute_m`, with its end in `substitute_goal`. dnav does not re-issue one to
   a point the vehicle already holds at.

Both carry `fallback_reason` (the planner's). Every replan still aims at the real goal:
as soon as one succeeds, dnav publishes it as `planned` and advances the stop generation,
so the vehicle takes it up from a confirmed stop. dway ends a fallback in an ordinary
`prefix` hold and never reports COMPLETE unless it is within `arrival_m` of the real goal.
Its status and samples carry `route_mode` and `substitute_goal`; dnav records each change as
a `navigation.mode` event. Stale evidence or pose, frame changes and coverage gaps get no fallback.

`sim-target` is uncalibrated: its limits are chosen, not measured, and target mode
accepts that. Dynamic mode still refuses uncalibrated profiles.

#### What to watch

With plan permission dnav still computes the strict verdict for the same route and
publishes it beside the permission as `clearance.evidence_check`, so both policies can
be compared on one flight. `dnav --no-ui` prints each permission change with that
verdict. When dnav withdraws a route, the published reason names the cause
(`waiting for executor to confirm stop (generation N: permission withdrawn: ...)`).
Turns are recorded as `execution.turn` events, and heading, commanded heading and turning go into the status and `samples.csv`. `dway --no-ui` prints every state change with the pose, and a status line every five
seconds (route admission, speed, remaining distance, targets sent). The window and
reports show the same, plus the launch, readiness and coverage events.

#### Where it stands

Target mode is wired end to end and measured, not tuned. On the maze_020 command above
(2026-09-14), dway took off by itself, started automatically and flew about 24 m of the
maze in 16 stop, replan and resume cycles, then held about 11 m short of the goal.
The optical-flow source had marked phantom obstacles around the goal, in cells the
lidar had not observed yet, and dnav reported the goal unreachable. The same source had
earlier marked the drone's own cell occupied. Such starts are now rejected rather than
assumed to be false obstacles. These are evidence-quality findings; a larger margin
cannot correct occupied evidence at the vehicle's own position.
`tests/test_target_flight_process.py` (`DVISION_NIGHTLY=1`) runs the four processes and
checks the wiring: takeoff, automatic Start, targets, progress toward the goal, a
written report and no world reads. It records arrival without requiring it.

With `handoff_m` 0, route changes happen only from a confirmed stop: every wall the lidar
reveals across the trusted route costs a stop, a replan from that pose and an automatic
Resume. `reports/area1/20260915-140020-efd23612` stopped 18 times that way, for obstacles
8-19 m ahead, and 16 more times at corners of 18-52°. With `handoff_m` above 0:

- **Shortened, not withdrawn.** A route blocked ahead stays permitted up to the obstacle,
  so the vehicle keeps flying. It is withdrawn only when the block is within `stopping_m`
  or the departure from the pose is blocked.
- **Handoff while moving.** When the permitted part no longer reaches the goal and the
  planner has a replacement, dnav publishes it with `handoff: {from_revision, start}` and
  no stop generation. That happens only if all of these hold: the vehicle is within
  `handoff_m` of it, it heads within `turn_tolerance_deg` of the segment being flown, it
  passes the same clearance check from the pose, and it permits more than is left. dway
  repeats its own checks (offset within `handoff_m` and its tracking allowance, stopping
  distance left, the new route heading within `turn_tolerance_deg` of the segment being
  flown) and records `execution.handoff`. It compares route directions, as dnav does,
  not the vehicle's heading, which lags a corner just flown through. Otherwise
  it brakes as before. A route switch in the goal fallback stays stop-first.
- **Corners.** Every eligible route carries `clearance.corners`, one flag per interior
  vertex. A flag is true when the chord from `stopping_m` before the vertex (or from
  the start of a shorter leg) to the vertex after it, or to the permitted end where that
  comes first, is clear, with more than `stopping_m` permitted past the vertex. dway flies through a flagged turn up to
  `corner_blend_deg` instead of stopping, provided cutting it stays inside the next
  segment's tracking allowance. It slows to `cos(turn)` of its speed within 1 m of the
  vertex and records `execution.corner_blend`.
- **Moving off.** A stop can leave the vehicle further off the route than its tracking
  allowance, as a corner stop that overshoots does. Braking again before it moves would
  hold it there for good (`reports/area1/20260915-143229-fb015cb6` alternated corner and
  tracking stops for 16 s). dnav checked the departure from that pose, so dway lets the
  vehicle keep the offset it moved off with, plus 0.02 m, until it is back within the
  allowance.

## Execution profile and measured limits

`assets/execution_profiles/dsim-position-v1.json` is the only calibrated profile. It
was measured by `dtest/flight_calibration.py` and **re-measured** by
`tests/test_flight_profile.py` on every run, so a simulator or link change that
invalidates it fails there.

| Measured on dsim over 0.25/0.5 m/s, 0/50 ms latency, no wind, seeds 1234/4321 | Value |
| --- | --- |
| HOLD stop distance from the observed decision pose | 0.0248 m |
| HOLD stop time to 0.05 m/s held for 0.5 s | 0.055 s |
| Lateral excursion during stop / cruise cross-track / drift held 5 s | 0 m / 0 m / 0 m |
| Position-target overshoot (4 m move) | 0.62 m |
| Position-target settle time | 23.2 s |

| Limit | Value | Derivation |
| --- | --- | --- |
| `speed_mps` | 0.5 | largest measured speed |
| `stopping_m` | 0.2 | ≥ stop distance + one control period at cruise (0.05) + 0.1 margin |
| `stopping_s` | 0.5 | ≥ stop time + `reaction_s` 0.2 |
| `tracking_m` | 0.25 | ≥ lateral + cruise cross-track + 0.25 margin |
| `join_m`, `arrival_m` | 0.3 | must accept a vehicle stopped `stopping_m` short |
| `max_age_s` | 3 | evidence lifetime; dalg publishes near 1 Hz |
| `max_state_age_s` | 0.5 | vehicle state staleness (wall clock) |
| `max_wind_mps`, `max_telemetry_latency_ms` | 0, 50 | measured conditions only |
| altitude slab | 1.5 ± 0.15 m | `vertical-extrusion` scenario assumption |

The loader refuses extrapolation: speed, wind or latency above the measured
conditions, stop/tracking/join/arrival limits below the envelope plus margins, a
missing calibration record, or no declared vertical assumption. Start and Resume
also refuse when the vehicle's reported wind or latency exceeds the profile.

Because position-target overshoot (0.62 m) exceeds `stopping_m`, the executor never
stops by converging on a target: it requests HOLD when the vehicle is within
`stopping_m` of a corner or permission endpoint. dsim's HOLD zeroes velocity at once
and does not hold position against wind; real vehicles decelerate. **None of these
numbers transfer to hardware.**

## Execution lifecycle

Every target is checked in the same step against the snapshot consumed: context,
goal, stop generation, geometry revision, cross-track against `tracking_m`, remaining
permission against `stopping_m`, and the deadline against `stopping_s + reaction_s`.
A target is the next waypoint or the permission endpoint, whichever comes first; the
next unvalidated waypoint is never commanded. Targets stream at `stream_hz` on
provider time and never burst to catch up. A step whose wall-clock gap exceeds
`max(0.5 s, 3 periods)` brakes (`overrun`) instead of sending an obsolete target.

A stop is confirmed only when the vehicle is in HOLD with speed ≤ `hold_speed_mps` for
`hold_dwell_s`; an acknowledgement is not a stop. No confirmation within
`hold_timeout_s`, a rejected HOLD or a failed send is FAILED, and targets cease.

| Stop (`hold_kind`) | Continues by |
| --- | --- |
| `corner` | itself, when still permitted and within `join_m` of the corner |
| `replacement` (dnav advanced the stop generation with an eligible route) | itself, once a route validated from the stopped pose is admitted |
| `pause`, `prefix` (permission ends before the goal), `expired`, `unavailable`, `stalled`, `tracking`, `overrun`, `arrival-unconfirmed` | explicit **Resume** with fresh permission |
| `goal` (goal changed or cleared), `restart` (planner session, provider, frame or clock reset) | a new **Start** |
| `arrival` | COMPLETE once within `arrival_m` of the goal and HOLD is confirmed |
| `cancel` | CANCELLED once HOLD is confirmed |

Lease loss (for example a manual takeover) and vehicle health failures (link, stale
state, failsafe, invalid estimate, a mode other than GUIDED/HOLD, leaving the slab)
are FAILED; there is no reacquisition or automatic resumption. Start never takes a
lease another client holds. After COMPLETE or CANCELLED, dway keeps servicing the
lease and health in HOLD. Closing the window or shutting down headless requests HOLD,
waits up to `hold_timeout_s` for a measured stop, records the result and releases
control. An interrupted flight is recorded as FAILED, never as success.

## Retained protocol

Single-host pymembus endpoints:

| Name | Writer | Readers |
| --- | --- | --- |
| `/dvision2.<id>.navigation.<name>` | Named dnav (default `dnav`) | dway/viewers |
| `/dvision2.<id>.execution` | dway (dry run or executor) | dnav/viewers |

Each has one JSON `snapshot` value, committed atomically using `setAll`.
The maximum encoded record is 65,536 bytes; routes have at most 256 XYZ points.
Literal examples are in `tests/assets/navigation_examples/`: `ready`, `unavailable`,
`expired-invalid` (intentionally invalid), `stop-required`, `arrival-only` and the
executor status `execution-stopped`. Tests check each against the shipped profile. `dcmn.navigation.validate` and tests define the v1 schema.
The existing `route.planned` module event remains abbreviated diagnostic history;
never execute its possibly truncated waypoint list.

An exclusive lifetime file lock protects each writer. A conflicting live writer
cannot unlink an endpoint; a new owner may recover a stale area after the previous
process exits. Readers reopen each poll so they observe restarted endpoints.
The module heartbeat advertises endpoint names, but retained data is authoritative.
No command/urgent channel, distributed acknowledgement history or new registry exists.

Navigation records carry publisher session/sequence, complete geometry/revision,
provider/frame/clock identity, evidence generation, goal authority/revision, profile
identity, attempt/input references and clearance. Intervals use segment/fraction
pairs and may end before a waypoint. A monotonic stop generation persists across
publications and is scoped to the planner session.

The stop latch binds an **active** route. An executor with nothing active has
nothing to stop and adopts each generation as handled, so the dry run, which never
activates, returns to `READY` once permission is available again. Once a route is
active, any newer generation, including one whose unavailable record was never read,
yields `STOP_REQUIRED` until the executor confirms its stop. A planner session change
under an active route also requires the stop, and then an explicit new Start.

dnav treats an executor as live only while its status `sequence` advances within
1 s of monotonic wall time and it names this planner session with `dry_run: false`.
While such an executor reports `EXECUTING` or `BRAKING`, dnav keeps that geometry and
validates only the remaining interval from the reported progress. When permission is
lost, it advances the generation and publishes nothing eligible until the executor
reports `HOLDING` with that generation. A held executor that reports disposition
`stopped` has released the route: the replacement is planned from its stopped pose
and pinned while it holds. A dead executor can neither pin geometry nor confirm a stop.

A dry-run status always says `dry_run: true`, `owns_control: false` and
`commanded_target: null`, and its state is one of `WAITING`, `READY`, `REJECTED`,
`STOP_REQUIRED` or `CLOSED`; the validator rejects anything else. Executor status adds
`EXECUTING`, `BRAKING`, `HOLDING`, `COMPLETE`, `CANCELLED` and `FAILED`, the
`hold_kind`, `start_required`, lease state, commanded target, speed against the cap,
operator control results (`pending`/`accepted`/`refused` with reasons), the violated
stopping margin when one occurred, and any vehicle fault kept separate from the
mission outcome. Both carry the goal, latest `disposition`
(`accepted`/`active`/`rejected`/`stopped`/`completed` with its geometry revision), the
handled `stop_generation`, progress, cross-track error, remaining permitted distance,
stopping margin and validity remaining. Values that are not measured are `null`, with
the reason under `unavailable`.

Progress is projected only onto the current and next segment and never moves
backwards (`dcmn.navigation.project_progress`), so a route that crosses itself
cannot jump to its later pass. The consumer checks goal, context, complete geometry,
profile, remaining lifetime, join distance, duplicate/reordered records and stop
generation. Unchanged route or pose streams mark a stalled peer on the wall clock
(3 s in the dry run, 1 s for the executor), independently of data-clock expiry.
Repeated retained bytes cannot refresh liveness or clearance.

## Clearance and frame details

The validator reads original occupancy/observation planes, not only planner cost.
Every swept cell needs fresh free evidence; occupied evidence from any admitted
source vetoes it until that source reports fresh free evidence or the mapping
identity/grid geometry resets. A source that disappears keeps its vetoes, because
absence is not free evidence. Unknown, ambiguous and stale-only evidence is not
clearance. Source membership and mapping/context changes advance the stop
generation, invalidating previous permission.

A conservative rectangle covers each half-cell subsegment, expanded by body radius,
tracking allowance and a stopping-area radius. This may reject narrow routes that
a more precise capsule check would admit, but cannot skip diagonal/cell corners.
Validation is bounded to 10,000 subsegments and one evidence slab. Multi-layer
clearance, smoothing and exploration are deferred. Observation lifetime must leave
room for stopping. Republishing an old grid does not refresh its cells.

Neutral context optionally includes `vehicle_transform`:
`dvision2.local-ned-transform.v1`, source `frame_id`, `localization_epoch`, and local
XYZ origin of NED. For local east/south/up `(x,y,z)` and origin `(ox,oy,oz)`, NED is
`(oy-y, x-ox, oz-z)`. dsim publishes its map-center origin; the synthetic provider
publishes zero origin. Consumers never read world dimensions. A transform from an
old localization epoch is rejected; providers must reissue context/transform after
changing that origin. Providers without a transform still support dnav planning,
but dway admission waits with an explicit reason.

## UI

The dry-run window has no flight controls. It shows the complete proposal as a thin
dashed line, the permitted interval as a thick solid line with a square stop endpoint,
and the observed pose and trail. dnav's Plan and Cost views use the same dashed
proposal and interval overlay, and its status line shows the clearance reason.

The flight window (`dway.flightui.FlightWindow`) adds Start/Pause/Resume/Cancel with
their pending/accepted/refused results, the commanded target (a cross, distinct from
the vehicle symbol and the published-pose trail), the segment being flown, and the
goal. Its readouts give state and reason, disposition, revision, publication and stop
generation, remaining distance and stopping margin against the stopping distance and
time, validity, speed against the cap, cross-track against the allowance, altitude
against the slab, lease, profile identity and recording health, plus every
unavailable value with its reason. A 60 s timeline plots speed against its cap,
cross-track against its allowance and the stopping margin, with state changes. The
recent-event list explains a selected event and draws its route as a dim dashed line.

The control loop runs on its own thread; Tk only reads the view the executor publishes
each step, so painting cannot delay control. Map repaint is capped at `MAP_HZ` and text
at `TEXT_HZ` (`dcmn.pacing`), repaint is skipped while minimized, and the window shows
how many executor updates it coalesced. The track is broken at pose gaps and epoch
changes; no position is invented.

## Reports, replay and comparison

dway's route source is `dway.route_source`: `TourRouteSource` wraps the static tour
exactly as `Flight` used it, and `DynamicRouteSource` reads only the retained
snapshot. The dynamic path imports no tour, world or mission module.

dnav records each navigation validation with exact evidence and persistent-veto grids.
The dry run records consumed proposals (and rejected raw JSON), context and
dispositions under append-once `dway/archive`, `archive-2`, etc. with a segment
summary, bounded `events.jsonl`/`events.csv`, `report.html` and `route.png`.

The executor records under the same append-once directories: every consumed snapshot
with its evidence grids, operator controls and results, transitions, HOLD
requests/confirmations/rejections, lease and health events, the conditions at Start,
every target with the permitted interval it was checked against, 10 Hz vehicle samples
and the shutdown result. Transitions, holds and shutdown are admitted even when the
queue is full; samples may drop and leave gap markers. A report rollover finalizes the
old segment and records a baseline (not an authorization) in the new one. When a
segment closes, its report is written in the background to `<archive>/report/`:

| File | Content |
| --- | --- |
| `summary.json` | `dvision2.dway-dynamic-report.v1`: outcome, goal, final pose, state durations, interventions, per-revision attribution, speed/tracking/stopping/cadence metrics, route ages, gaps, recording completeness, profile, conditions, and the definition, units and weighting of every metric |
| `events.jsonl`, `samples.csv`, `routes.csv`, `holds.csv` | the committed events and tables |
| `map.png`, `timeline.png` | evidence, proposed/active/permitted routes, flown track, targets and stops; speed, cross-track and margin with state bands and gaps |
| `report.html` | all of the above, readable offline |
| `manifest.json` | inputs used (the dway archive and sibling dnav archives) and inputs missing, with reasons |

Measurements are attributed to the route revision active when taken. Unavailable
values are `null` with a reason; estimated track length uses published poses and is
not truth. A complete flight with an incomplete recording says both.

```sh
python3 apps/dway/flightlog.py report reports/<id>/<run>/dway/archive [--out DIR]
python3 apps/dway/flightlog.py compare RUN_REPORT_DIR... --out comparison/
python3 apps/dway/replay.py reports/<id>/<run>/dway/archive
python3 apps/dway/replay.py ARCHIVE --no-ui --export frames/ --at 12.5
python3 apps/dcmn/archive.py reports/<id>/<run>/dway/archive   # validate
```

`compare` writes `comparison.csv/json/html` with every run's outcome, reason, recording
completeness, stops, interventions, speed, tracking, margin violations, cadence and
profile digest; failures are included, and planner cost is not compared.

Replay (`dway.replay.ReplayModel`, Tk `ReplayWindow`) opens no vehicle or memkv
endpoint. It plays, pauses, seeks, steps between decision events, selects a route
revision, and shows the target, pose, state, reason and the evidence recorded with the
consumed route at or before that time. Values are the last record at or before the
selected time; gaps and epoch changes are marked and the track is broken there,
never interpolated. `--export` writes the frame as JSON, and as PNG when evidence exists.

To share a run, copy the whole run directory (`reports/<id>/<run>/`), which holds the
dway archive with its report and the sibling dnav archives. `tests/test_flight_reports.py`
opens such a copy elsewhere, rebuilds the report and replays from it.

## Verification

- `tests/test_navigation.py`: bounds, oversize refusal, fixtures, strict profiles,
  stale/conflicting evidence, joins/crossings/progress, the stop latch with missed
  withdrawals, planner restart, the dnav publisher against a fake executor, frames,
  tour/dynamic adapters, dry-run window and archive rollover.
- `tests/test_navigation_process.py`: isolated dalg/dnav/dway dry-run processes
  against a late synthetic provider and dsim; operational world/truth reads and
  mission imports are rejected.
- `tests/test_flight_profile.py`: re-measures the calibration and refuses extrapolation.
- `tests/test_dynamic_flight.py`: clear goal completion, explicit Start, airborne HOLD
  and foreign-planner refusal, unknown-space prefix stop and fresh-clearance resume,
  evidence expiry, newly blocked route with replanning from the stop, missed
  withdrawal, control-loop overrun, HOLD acknowledgement without a measured stop,
  rejected HOLD, self-crossing route, coincident goal, geometry replacement, goal
  change, pause/resume, cancel with supervision, planner restart, localization reset,
  manual takeover, no lease stealing, a second executor, provider stall, wind outside
  the profile (refused at Start; honest failure mid-flight), 50 ms telemetry latency
  with the calibration seeds, identical commands with recording/reports off, recording
  failure, bounded shutdown and the display view. Every flight asserts that no target
  lay outside the permitted interval it was checked against.
- `tests/test_flight_window.py`: painting never steps control, buttons reach the
  executor, readouts/overlays separate proposal, permission, command and motion, and
  closing holds and releases without landing.
- `tests/test_flight_reports.py`: report artifacts, null-with-reason values, replay
  lookup/gap/epoch rules, export, corrupted recording, copied run bundle, comparison
  with failures, and (`DVISION_NIGHTLY=1`) a thirty-minute recording.

The thirty-minute recording (five missions with supervised hovering between them) was
measured on 2026-09-14 on an Intel i9-12900H (20 threads), Python 3.13: 1818 s of data
clock in 114 s wall (16x real time), 18,185 samples and 18,843 events in one complete
archive, and 769 MiB peak RSS for the whole test process (simulator, rig and report
included). At the end the executor's live history held at most 610 samples, the track
6,000 points and the event list 300. Mid-flight seek, frame export and the full report
all worked from the archive.

Run GUI/process tests with a working display, for example `xvfb-run -a python3 -m
pytest tests/test_navigation.py tests/test_navigation_process.py tests/test_flight_window.py`.

## Supported and deferred

Supported: one planner and one executor per vehicle, fixed altitude, position targets
at the heading held at Start, stops at corners and permission ends, replacement only
from a confirmed stop, the dsim position-target vehicle inside its measured profile.

Deferred: smooth in-motion replacement, velocity-strategy fallback, yaw for visibility,
3D planning, automatic exploration/takeoff/landing, remote or multi-authority execution
control, calibrated wind and hardware flight, raw-sensor replay, browser replay and a
general comparison application.
