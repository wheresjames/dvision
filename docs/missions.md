# Missions: dmsn

`dmsn` is the mission coordinator. dnav plans, dway flies and holds the control
lease, and dmsn decides **where the vehicle goes next and when the job is
finished**. It holds no lease: it sets the session goal as the `mission`
authority, asks dway for a small set of vehicle actions through a retained
intent, and reports.

Missions available today:

| Mission | What it does | Editors |
|---|---|---|
| `goto` | fly to one point, chosen by a map click, typed coordinates or the provider's published target; retargetable while flying | goal |
| `waypoints` | fly a list of points in order; an unreachable point aborts the mission, is skipped, or is retried | waypoints |
| `explore` | cover an area polygon by flying to the goals a strategy proposes, until none is left or the coverage target is met | area |
| `search` | explore an area until a detector's target estimate is confirmed, look again from a standoff, then land on the estimate | area |

Goal strategies for `explore`:

| Strategy | What it proposes |
|---|---|
| `frontier` | the nearest frontier (an observed free cell next to never-observed space) inside the area |
| `wall_follow` | a point `step_m` (4 m) along the nearest wall, kept on the chosen `hand` at `standoff_m`; straight on when no wall is within `reach_m` |

Mission algorithms are deliberately baseline: a mission that
explores badly is acceptable as long as it shows what it is doing and ends with
a reason.

[`docs/missions/`](missions/README.md) documents each mission type in depth --
its state machine, every parameter, every way it ends and what it draws -- plus
one document for the two goal strategies. This document stays the operational
one: the chain, the profiles, the phases, the intents and the window.

## The chain

A mission chain waits to be told. Nothing moves until a mission starts:

```sh
python3 ./apps/dsim/dsim.py --id area1 --map ./assets/maps/maze_020.txt --drone-profile camera-lidar --sim-speed 1.5 &
python3 ./apps/dalg/dalg.py --id area1 --profile lidar-baseline &
python3 ./apps/dnav/dnav.py --id area1 --execution-profile sim-target &
python3 ./apps/dway/dway.py --id area1 --mode target --execution-profile sim-target --wait-for-mission &
python3 ./apps/dmsn/dmsn.py --id area1 &
```

A search needs a detector beside evidence in dalg:
`./apps/dalg/dalg.py --id area1 --profiles lidar-baseline red-disc` (see
[detection](detection.md)).

- **dnav** adopts no goal on its own: the provider's map target is opt-in
  (`--provider-goal`). `--no-provider-goal` is still accepted and changes nothing.
- **dalg** sizes its first grid as a 100 m square around the vehicle
  (`--min-side-m`), which covers every committed map from its start cell, so a
  mission's goals are inside coverage without a mapping reset.
- **dway** `--wait-for-mission` keeps target mode on the ground in `WAITING`
  ("waiting for a mission launch intent") until dmsn publishes `launch`. Without
  the flag target mode behaves as before and reads no intents.
- **dmsn** is the goal surface while a mission is loaded. A dnav Plan-tab click
  with "take goal authority" is an explicit override: the mission pauses and both
  windows say so.

`dsim`'s `sim.map` status key names the running map. dmsn compares its base
name with the profile's `arena`: the window warns, a headless `--start` refuses
(exit 2) unless `--allow-arena-mismatch`. dmsn never opens the map file.

## Command line

```sh
./apps/dmsn/dmsn.py --id area1                                    # pick in the window
./apps/dmsn/dmsn.py --id area1 --mission goto                     # click-to-go ready
./apps/dmsn/dmsn.py --id area1 --no-ui --mission goto --goal 52.4,2.4 --start
./apps/dmsn/dmsn.py --id area1 --no-ui --mission goto --set goal_source=provider_target --start
./apps/dmsn/dmsn.py --id area1 --no-ui --profile maze_020-goto --set budgets.time_s=600 --start --timeout 900
./apps/dmsn/dmsn.py --id area1 --no-ui --profile maze_020-goto --allow-arena-mismatch --start
./apps/dmsn/dmsn.py --id area1 --no-ui --profile maze_020-explore --set budgets.time_s=600 --start --timeout 900
./apps/dmsn/dmsn.py --id area1 --no-ui --profile maze_020-waypoints --start
./apps/dmsn/dmsn.py --id area1 --no-ui --profile maze_020-search --set budgets.time_s=900 --start
./apps/dmsn/dmsn.py --import-tour assets/tours/maze_020.default.v1.json \
    --out ~/.config/dvision2/missions/maze_020-tour.json             # converts and exits
./apps/dmsn/dmsn.py --validate assets/mission_profiles/maze_020-goto.json    # no instance
```

| Option | Meaning |
|---|---|
| `--profile NAME\|PATH` | a profile; a bare name is looked up in the user directory, then `assets/mission_profiles/` |
| `--mission NAME` | start from a new profile for this mission type |
| `--goal x,y` | shorthand for `--set goal=x,y` (and `goal_source=xy` when the source was `click`) |
| `--set KEY=VALUE` | override a field: `goal_source=xy`, `budgets.time_s=600`, `finish_action=home`, `debounce.min_commit_s=5`; repeatable, recorded in the report |
| `--start` | start once the session provider is up |
| `--allow-arena-mismatch` | headless: fly a profile made for a different map |
| `--show-reference`, `--hide-reference` | override the mission's default for the reference background |
| `--no-ui`, `--timeout S` | headless, printing one line per phase, goal or blocking-fact change |
| `--validate PROFILE` | check a profile and print its digest |
| `--import-tour TOUR --out PROFILE` | convert a tour into a `waypoints` profile and exit (no instance) |
| `--detector-wait-s S` | headless `--start` of a mission with a target: wait this long (30 s) for a detector publishing its label before refusing |

Exit codes: **0** the mission completed; **1** it failed, was aborted, spent a
budget, or did not finish before `--timeout`; **2** a usage or startup problem --
a bad profile, a refused arena mismatch, or a second dmsn for the same vehicle
(`live writer already owns /dvision2.<id>.mission`).

## Profiles

A profile (`dvision2.mission-profile.v1`) is everything an operator chose, in
one JSON file. The file is the source of truth; the Setup tab edits a copy and
saves it back, and a run records the digest of exactly what it flew (the name
is not part of the digest).

```json
{
  "schema": "dvision2.mission-profile.v1",
  "name": "maze_020-goto",
  "mission": "goto",
  "params": {"goal_source": "xy", "goal": [52.444, 2.389]},
  "finish_action": "hold",
  "coverage_margin_m": 10.0,
  "workspace": null,
  "arena": "maze_020",
  "strategy": "", "strategy_params": {},
  "budgets": {"time_s": null, "time_action": "hold", "battery_reserve_pct": null,
              "battery_action": "home", "path_m": null, "path_action": "hold"},
  "debounce": {"min_commit_s": 3.0, "min_shift_m": 1.0},
  "assisted": {"reference_background": null, "tour_import": false},
  "area": [], "waypoints": [], "imported_from": null
}
```

- **Where they live.** Committed examples in `assets/mission_profiles/`;
  operator saves default to `~/.config/dvision2/missions/`
  (`DVISION2_MISSION_DIR` overrides it), so flying never dirties the repository.
- **Parameters** come from each mission's `ParamSchema`, which also generates
  the Setup tab's form and parses `--set`. Kinds: `float`, `int`, `bool`,
  `choice`, `text`, `point`, `bounds`, `points`, `polygon`.
- **Bounds.** At most 256 area vertices and 256 waypoints; free text at most 120
  characters. The editors and `--validate` refuse more.
- **Coverage.** Only a mission with an `area`, or a profile `workspace`, asks
  dalg for coverage -- the bounding box plus `coverage_margin_m`, as a mapping
  reset, before takeoff. Launch waits for that coverage to be published;
  allocation refusals appear in the blocking fact. goto without a workspace
  asks for nothing and keeps dalg's own grid.
- **Budgets.** Sim time from Start, battery reserve (`drone.battery_pct`) and
  flown track length, each with an action: `hold` where it is, fly `home` (the
  takeoff point) and hold there, or `land` where it is. The mission then ends
  with outcome `budget`.
- **Finish action.** `hold` (default), `home` or `land`.
- **Strategy.** `strategy` and `strategy_params` choose and configure the goal
  strategy of an area mission (`--set strategy=wall_follow`,
  `--set strategy_params.hand=left`).
- **Tour imports.** `--import-tour` reads a tour from `assets/tours/`, keeps its
  x,y stops as waypoints, takes the tour's map name as `arena`, finishes with
  `land`, sets `assisted.tour_import` (the points were authored against the
  map) and records `imported_from` (tour id, digest, map and map hash). It is
  the only dmsn command that reads a map-bound file, so it converts and exits
  and never runs under the truth-isolation hook; the running mission only ever
  loads the converted profile.

## How a mission runs

```text
idle ─Start─▶ starting ─airborne─▶ running ─Done─▶ finishing ─▶ done
                 │                    │                ▲
                 └──────── paused ◀───┘ (Pause, authority taken, dway failed)
```

- **starting** checks the arena, requests coverage while on the ground, keeps
  the takeoff point as home, takes the goal authority by explicit handoff, and
  publishes `launch`. It ends when dway reports `launch_phase: airborne`.
- **running** steps the mission body at 10 Hz on one immutable observation:
  pose, goal and authority, dway's execution state, dnav's route status, the
  evidence coverage extent, battery, map name, the provider target. Missions
  read nothing else.
- **Goals.** A mission returns `Goal`, `Wait`, `Action` or `Done`. Operator goals
  (a click, a retarget, going home) are sent at once; a strategy's goal is kept
  at least `min_commit_s` and replaced only when it moves at least
  `min_shift_m`. Suppressed goals are counted and shown. A goal outside the
  evidence coverage is not sent: the blocking fact says so, and **Expand
  coverage** (an explicit operator choice) requests a mapping reset.
- **Arrival** is only ever dway's `COMPLETE` for the goal revision dmsn set.
- **Area missions** also observe a bounded distillation of the evidence plane:
  up to 64 frontier points inside the area (nearest first, leaving out those
  near points the mission gave up on), up to 64 occupied points within 8 m of
  the vehicle, and the observed fraction of the area's cells. The rest of the
  grid stays out of the observation, so a recorded run still replays.
- **Unreachable goals.** A goal is given up when dnav refuses to plan to it, or
  when the vehicle stops getting closer for `stuck_s` (20 s). How long a
  refusal must stand depends on what it says: `goal_unreachable` and
  `outside_coverage` are verdicts about the point — an obstacle, its margin,
  ground off the map — and are acted on as soon as they are confirmed across
  observations (0.5 s), because waiting cannot change them; `no_route` and
  `start_blocked` may be transient and are held for `unreachable_s` (5 s). The
  area missions additionally defer to a substitute route dnav offers, flying
  to the nearest reachable point so the goal can at least be observed from
  there; `waypoints` does not, since an operator asked for the point itself.
  "Stops getting closer" is measured from the distance at the last reset, so a
  long leg flown at 1 m/s never counts as stuck. `waypoints` then applies its `failure_policy`: `abort` fails
  the mission naming the point, `skip` marks it skipped and flies on, `retry`
  tries it `retries` more times before skipping. `explore` blacklists the point
  with its reason and asks the strategy again; a proposal outside the evidence
  coverage is blacklisted `outside coverage` at once, so a strategy never
  resets the map.
- **Search** (`search`) explores like `explore` until the detections plane
  holds an estimate of its `target_label` that is confirmed, not stale, not
  given up, and at least `min_confidence` (0.3). `target_label` is the label a
  dalg profile row publishes (its `label`, e.g. `landing_pad` in `red-disc`). Then it works through phases:
  1. **Standoff.** It flies to a standoff point `standoff_m` (4 m) short of the
     estimate, on the line from the vehicle.
  2. **Face.** It asks dway to `yaw_to` the estimate.
  3. **Observe.** It looks for `observe_s` (3 sim s). The estimate must stay
     confirmed and fresh and be seen again in that time. Otherwise it looks
     again, and after `max_standoff_attempts` (2) it gives the estimate up and
     resumes exploring.
  4. **Approach and land.** It flies to the estimate and ends `complete`,
     "landed on estimate ...", with the `land` finish action whatever the
     profile's is.

  An estimate that goes stale, or a standoff or estimate dnav keeps refusing,
  is given up the same way.
  - **Ending without a landing.** If exploration or a budget ends first, the
    mission ends `failed: no confirmed target: ...` (a budget ending is failed,
    not budget).
  - **Start check.** Start refuses, and headless exits 2, when no detector
    publishes the label ("no detector publishing `landing_pad`"). A named
    `target_detector` must also be the one publishing it.
  - **Observations.** Only a mission with a target reads the detections plane.
    Its observations carry the plane's detections (at or above
    `min_confidence`) and estimates of its label, capped as the plane caps
    them. goto, waypoints and explore never open it.
  - **Summary and overlay.** Every decision that acted on an estimate is kept
    in the summary's `estimate_log` with the estimate ID and revision. The
    overlay adds the detections, the estimates (solid when confirmed, grey when
    stale or given up), the standoff and the landing point.
- **How explore ends.** `area coverage N% reached the M% target`
  (`coverage_target_pct`), `no frontier left to explore`, `no reachable goal
  left` (the strategy proposes none), or a budget, Abort, Go home or Land.
  Every blacklisted point is kept with its reason.
- **Pause** holds dway and suppresses its automatic Start. **Resume** re-sends the
  goal (with a handoff if the authority was lost) and releases the hold.
- **Losing the goal authority** pauses the mission *without* holding the
  vehicle: whoever took the goal is flying it. dmsn never takes it back on its
  own. **dway failing or being cancelled** pauses it too.
- **Abort** holds dway, releases the goal authority and ends `aborted`. **Go
  home** ends the mission body and flies home. **Land** ends it `aborted` and
  lands where the vehicle is.
- **Landing** (the `land` finish or budget action, or Land) publishes `land`
  and waits in `finishing` for dway's verdict: `completed` once the vehicle is
  disarmed on the ground, when dmsn also gives the goal authority back, or
  `failed` with dway's reason (for example `landing not confirmed within 60 s`),
  which fails the mission.
- If dmsn exits mid-mission, dway finishes the current goal and holds there.
  After a crash, dmsn shows the retained status and restores its profile when no
  profile was explicitly selected. Press Start to start again; resuming
  mid-mission is not supported.

## Intents and acknowledgements

dmsn publishes one retained intent on `/dvision2.<id>.mission_intent`
(`dvision2.mission-intent.v1`): a `kind` -- `launch`, `hold`, `resume`,
`cancel`, `land` or `yaw_to` -- its `args`, a reason, and a new `revision` per
request. dway (only with
`--wait-for-mission`) acts on each revision once and publishes what it did in
the execution status it already writes:

```json
"mission_intent": {"session": "...", "revision": 3, "kind": "hold", "state": "accepted", "reason": "stopping"}
```

`state` is `pending`, `accepted`, `refused`, `completed` (a launch once airborne
in HOLD, a landing once disarmed, a turn once HOLD is confirmed at the heading)
or `failed` (with the reason). `launch` authorizes the automatic launch; `hold`
pauses a moving vehicle and suppresses the automatic Start until `resume`;
`resume` presses Resume or Start as the held route needs; `cancel` cancels
(terminal for that dway).

- `land` puts dway in `LANDING`: the route is given up, the link's LAND command
  is sent, and the altitude and flight-mode health checks are skipped while
  link, crash and failsafe checks stay. It completes on disarmed in
  `LAND`/`DISARMED`, gives the control lease back and returns to `WAITING`
  ("landed; waiting for a mission launch intent"); only a new `launch` flies
  again. If the ground is not confirmed within `dway --land-timeout-s` (60 sim
  seconds), the intent fails with `landing not confirmed` and dway faults; it
  never re-arms or retries on its own, and a landing cannot be cancelled.
- `yaw_to` (`args: {"heading_deg": 180}`) turns in place through the link's
  yaw-rate command from a confirmed HOLD (a held route or a reached goal),
  confirms HOLD at the heading, and returns to the state it started from. It
  fails after 45 s without reaching the heading.

dway may refuse any intent, with the reason.

dmsn's own status is retained on `/dvision2.<id>.mission` (`dvision2.mission.v1`):
mission, profile and digest, phase, outcome, reason, blocking fact, goal, area,
waypoints, home, progress and budgets, within the 64 KiB snapshot limit. Module
bus events: `mission.started`, `mission.phase`, `mission.goal`, `mission.done`.

## The window

- **Mission.** The evidence map (dcmn's `MapPane`) with dnav's proposed and
  permitted route, the vehicle and its track, the goal, home and the mission's
  own marks. Before any evidence exists the map falls back to the mission area,
  the profile workspace, the reference image (only while displayed) or a 100 m
  square around the vehicle, and says which, so a goal can be clicked before
  dalg publishes. Controls: Start, Pause, Resume, Abort, Go home, Land, Expand
  coverage. Readouts: phase, outcome, goal,
  dnav route, dway state and hold kind, the last intent and its acknowledgement,
  progress, arena, budget bars, and the **blocking fact** -- one line saying
  what the mission is waiting on. The reference background is on by default for
  goto and off for autonomous missions; every change is recorded.
  An area mission's overlay adds the frontiers, the blacklist crossed out, the
  committed goal and the coverage percentage; wall following marks the wall it
  follows and its hand; waypoints are numbered and coloured by state (pending,
  visiting, visited, skipped, failed).
- **The view.** The wheel zooms the map about the pointer; the middle button
  drags it, or control with the left button for a mouse without a middle one.
  Zoom stops where the evidence runs out of detail -- one cell at 32 screen
  pixels -- so the ceiling is a property of the grid and the canvas, not a
  constant: maze_020's 0.25 m cells in a 480 px pane reach about 17x. Only the
  cells the canvas can show are ever scaled and uploaded, so a repaint costs
  the same at 17x as at 1x. The same view controls are on dnav's Plan map and
  dway's live map; the README describes them once under "Zooming a pane".
  The image can never be dragged off its own canvas, so
  "anything to reset" and "not the fitted view" are the same question: while
  they are true, a translucent **Reset view** badge sits in the image's
  top-right corner, showing the current factor, and a press on it restores the
  fitted view. A press on the badge is not a press on the map -- it never adds
  a vertex or sets a goal. The zoom belongs to the pane, not to the mission:
  `mission.png` and every recorded snapshot render the whole grid at their own
  scale whatever the operator is looking at. Editor hit radii are in pixels,
  so zooming in makes a crowded vertex easier to grab.
- **Map editors**, shown when the mission asks for them and active between
  missions (a flying mission keeps the geometry it started with):
  - **area**: click to add a vertex (on an edge when the click is on one,
    otherwise into the edge it grows least without crossing), drag a vertex to
    move it, right-click to remove it, shift-drag for a rectangle, and Hull,
    Clear and Undo buttons. Clear, then click corner by corner, draws a new
    area. While a mission with a map editor is loaded, map clicks never set a
    goal;
  - **waypoints**: the same gestures on the list, shift-drag to select and
    Delete to remove the selection, and a list with Up, Down, Delete and Undo.

  An edit that would make the area's edges cross is refused with the reason, as
  is a 257th vertex or waypoint. A point outside the evidence coverage is kept
  but flagged. Every accepted edit changes the profile, so it marks it modified
  and is saved with it.
- **Setup.** Mission type, the generated form (mission parameters, general,
  budgets, debounce, assisted), New / Load… / Save / Save as… / Revert, a list of
  saved and example profiles, the digest and a modified marker. **Import
  tour…** runs `dmsn --import-tour` as a child process into the user directory
  and opens the profile it wrote; the window itself never opens the tour.
- **Status.** A sim-time strip of phases, goals, dway stops and intents; the
  timeline as a table; and the modules present on the bus.
- **Events.** dcmn's event viewer.

## Reports

`<report_root>/dmsn/`:

- `summary.json` (`dvision2.dmsn-report.v1`): mission, profile, digest, path and
  overrides; outcome, reason and phases with sim durations; goals with their
  outcomes (`active`, `replaced`, `reached`, `abandoned`) and suppressed-goal
  count; home, track length, budgets, coverage requests, intents and
  acknowledgements; arena check; assisted flags (reference display, provider
  target used, tour import); the mission's own summary (waypoint states, or the
  explore area, strategy, coverage and every blacklisted point with its
  reason); recording state.
- `events.jsonl`, `mission.png` (the Mission tab's rendering at the end),
  `report.html`.
- `archive/` (`dcmn.archive`): every observation, operator event and decision,
  goals, intents and acknowledgements. `dmsn.runner.replay(archive)` runs the
  recorded mission body again and compares every decision; a mission that
  decides differently is caught.

A later evaluator can score a run against truth offline from the summary's
goals, track and final state. Nothing in dmsn scores.

## Adding a mission

A mission is one module under `apps/dmsn/missions/` and one line in `MISSIONS`:

```python
class Mission(Protocol):
    name: ClassVar[str]                     # registry key
    label: ClassVar[str]                    # what the Setup tab shows
    schema: ClassVar[ParamSchema]           # form, profile validation and --set
    editors: ClassVar[tuple[str, ...]]      # "goal", "area", "waypoints"
    reference_default: ClassVar[bool]       # reference background on by default?
    def start(self, params, obs, *, area=(), waypoints=(), strategy=None) -> None: ...
    def step(self, obs) -> Decision: ...    # Goal | Action | Wait | Done(outcome, reason, finish='')
    def on_event(self, event) -> None: ...  # click, retarget, paused, resumed, authority_lost
    def overlay(self) -> list[Shape]: ...   # map metres
    def progress(self) -> dict: ...
    def summary(self) -> dict: ...
```

Missions are pure logic over observations: no planes, clocks or windows, so a
mission is tested with `dtest/mission_fixture.py`'s `FakePorts` and
`FakeExecutor`, and replayed from its archive. Take-off, going home, budgets and
the finish action belong to the runner. A `Done` may name the finish action
its ending needs (`finish='land'`), and a mission may define
`budget_spent(reason) -> (outcome, reason)` to say how a spent budget ends it.
A mission that needs a target declares `target_label` (and optionally
`target_detector`, `min_confidence`) in its schema. The runner then refuses to
start it without a publisher and hands it the detections.

A goal strategy is one module under `apps/dmsn/strategies/` and one line in
`STRATEGIES`. It only proposes; the mission owns commitment, blacklisting and
termination:

```python
class GoalStrategy(Protocol):
    name: ClassVar[str]
    schema: ClassVar[ParamSchema]           # strategy_params
    def propose(self, obs, area, blacklist) -> Candidate | None: ...  # Candidate(xy, reason)
    def overlay(self) -> list[Shape]: ...
```
