# dvision2

`dvision2` is a local drone simulation and autonomy development platform. It
contains:

- `dsim`: a Python/Panda3D drone simulator, with a simulated environment --
  GPS quality, estimator validity, wind, telemetry latency, sensor noise,
  battery and geofence -- that can be changed while it flies
- `dctl`: a manual keyboard/gamepad controller, and the device browser that
  shows what the vehicle's sensors are actually producing
- `daic`: a vision-driven autonomy controller
- `dway`: the autopilot client, which flies a waypoint tour and reports on it
- `dalg`: the algorithm demonstrator, which scores a mapping algorithm's
  occupancy grid against ground truth over a repeatable tour
- `dfgb`: a FlightGear bridge that stands in for `dsim` behind the same
  buffers -- a work in progress
- `dcmn`: what the windows share -- one palette, one drawing of a map, one
  device browser

The processes communicate through shared-memory video, command, and status
buffers. The current autonomy work is intentionally centered on what a real
camera client would have: the live video stream and the status/telemetry
buffer.

The project is meant for fast local iteration. The simulator is small enough
to read, the command protocol is JSON over `pymembus` shaped so every message
maps onto a MAVLink one, and the autonomy stack is split into detector,
planner, avoidance, local mapping, and control layers so each piece can be
tested or replaced independently.

![dctl device layout](images/dctl-002.png)

## Contents

- [Project Layout](#project-layout)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Simulator: dsim](#simulator-dsim)
- [Manual Controller: dctl](#manual-controller-dctl)
- [AI Controller: daic](#ai-controller-daic)
- [Waypoint Navigation: dway](#waypoint-navigation-dway)
- [Vision Navigation](#vision-navigation)
- [Algorithm Demonstrator: dalg](#algorithm-demonstrator-dalg)
- [Automated Testing and Diagnostics](#automated-testing-and-diagnostics)
- [Maps](#maps)
- [Shared Memory Protocol](#shared-memory-protocol)
- [Telemetry](#telemetry)
- [Rendering and Assets](#rendering-and-assets)
- [Development Notes](#development-notes)
- [Known Limitations](#known-limitations)
- [Comparison to Similar Projects](#comparison-to-similar-projects)

## Project Layout

```text
dvision2_common.py          Shared protocol, map loading, ids, status keys
compare.py                  Offline comparison of dalg summaries across runs
docs/
  clock.md                  Simulated vs wall time, module sync, and the failure modes
  modcom.md                 Module communication: the four shared-memory planes
  membus.md                 The shared-memory buffer inventory and payloads
  reports.md                Report layout: who owns what, and the rules
  recovery.md               How a module recovers from a reset, restart or profile change
  sensors-protocols.md      Sensor configuration and the protocols behind it
  dctl-devices.md           The dctl device browser: intake, layout, renderers
  sensor/                   One document per simulated sensor type, plus the
                            rules they share
  mavlink-slam-nav.md       The reference architecture the vehicle seam borrows from

scripts/
  build_sensor_pymembus.py  Build the byte-safe pymembus binding locally
  install_dalg_depth_model.py  Download, export and install dalg's depth ONNX model

apps/                       The six applications and the view layer they share.
                            A source root rather than a package, like src/:
                            they import each other as `dsim.dsim` and
                            `dcmn.window`, never `apps.dsim`
  dcmn/
    theme.py                  The one dvision2 colour palette
    tktheme.py                That palette applied to ttk, shared by every window
    mapview.py                Top-down map and vehicle drawing, shared by every view
    pacing.py                 Repaint caps, so a window never paces control
    window.py                 Window geometry persistence and the input-method opt-out
    sensors.py                Sensor discovery, record wire format, camera intake
    device_view.py            The Devices tab: tree, pane grid, and every renderer
    device_export.py          Pane snapshot PNG and JSON/CSV sample dumps
    layout.py                 The pane grid: placement, spans, repair, persistence
    series.py                 Min/max envelope series, so a 100 Hz sensor plots cheaply
    module_bus.py             Module presence and run coordination over pymembus
    event_viewer.py           The Events tab: passive, bounded event-bus inspection
    health.py                 The one health vocabulary: wanted against achieved, graded
    report_html.py            The page a report is written on, whatever it is about

  dsim/
    dsim.py                   Simulator: physics, rendering, IPC server, UI
    headless.py               Fixed-timestep in-process driver with set_pose()
    profiles.py               Drone hardware profiles: load, validate, resolve
    transforms.py             The parent-linked body/mount/sensor transform graph
    sensor_manager.py         Simulated-time sensor schedule and publication
    sensor_models.py          Range/LiDAR measurement models and capture noise
    sensors_panel.py          The Sensors tab: load, inspect, edit, apply
    add_menu.py               The Add dropdown: described items, drawn to the palette
    scroll.py                 Scrollable form viewport and popup, shared by the tabs
    range.py                  Shared ray geometry and the exact range oracle
    depth_probe.py            Measured selection of the exact-range backend
    state_sensors.py          GNSS, IMU, barometer, magnetometer, thermometer models
    realism.py                GPS, estimators, wind, latency, noise, battery, geofence
    realism_panel.py          The Realism tab: those settings, changeable in flight
    health.py                 Whether simulated time and every attached module keep up
    scene.py                  Renderer appearance presets
    range_backend.v1.json     The measured exact-range backend choice, committed

  dctl/
    dctl.py                   Manual controller UI

  daic/
    daic.py                   AI controller, UI and headless modes
    detector.py               OpenCV red target detector
    planner.py                Mission state machine
    controller.py             Target visual-servo controller
    avoidance.py              Forward-speed obstacle brake
    optical_flow_avoidance.py Dense optical-flow risk and range estimation
    local_map.py              Vision-built local occupancy map and A* route plan
    mini_slam_detector.py     Lightweight visual-motion obstacle detector
    orb_slam3_detector.py     Optional ORB_SLAM3 integration wrapper
    flight_log.py             JSONL flight logger and report analyzer
    run_reporter.py           Run summary, images and HTML for a flight

  dway/
    dway.py                   Tour follower client, window and headless modes
    link.py                   VehicleLink contract and the dsim implementation
    tour.py                   Tour load/save/validate, frames, clearance
    follower.py               Arrival rules, sequencing, control strategies
    mission.py                Flight lifecycle state machine
    frames.py                 map / local NED / global transforms
    editor.py                 Map and waypoint editor, with live geometry checks
    report.py                 Flight summary, event log, track plot, repeatability

  dalg/
    dalg.py                   Algorithm demonstrator: window, profiles, headless
    run.py                    Observer lifecycle and the run-coordination barrier
    profiles.py               Profile load/save; the set lives in assets/profiles/
    model.py                  Pose, Frame and Result: what an algorithm sees and returns
    algo/                     One module per algorithm, plus the two controls
    grid.py                   Occupancy and log-odds grids
    truth.py                  Ground truth rasterised from the map
    visibility.py             Which cells the flight could actually have seen
    score.py                  IoU, coverage, Brier and hallucination scoring
    overlay.py                Prediction and verdict rasters
    report.py, report_html.py  Report directory, summaries, overlays, HTML

  dfgb/
    dfgb.py                   FlightGear bridge, a work in progress
    protocols/                The FlightGear property-tree protocol XML it installs

assets/                       Shared fixture data (not owned by one consumer)
  maps/                       Text map files
  textures/                   CC0 ground/wall textures
  models/trees/               CC0 tree GLB models
  tours/                      Committed benchmark tours and their diagnostics
  profiles/                   Committed dalg algorithm profiles
  drone_profiles/             Committed vehicle hardware profiles, sensors and all
  planner_queries/            Committed planner start/goal sidecars

dtest/
  contract.py               Literal coordinate/sign expectations (the oracle)
  calibration_scene.py      Calibration fixture paths and image expectations
  deterministic.py          Fixed-timestep in-process dsim driver
  process_harness.py        Real-process dsim harness over the live transports
  dway_rig.py               Deterministic dway flight: real dsim, in-process transport
  conformance.py            Backend-neutral suite both harnesses must pass
  color_probe.py            Independent RGB landmark measurement
  faults.py                 Test-local fault injection for oracle self-checks
  assertions.py             High-level assertions with failure artifacts
  artifacts.py              Failure bundles (frames, timeline, path plot)
  backend.py                Normalized vehicle-backend protocol
  tkfixture.py              Withdrawn Tk roots, and the opt-in for mapped ones
  preflight.py              Dependency preflight for the test groups

tests/
  flight_test.py            End-to-end headless flight runner
  reversal_mutations.py     Audits the suite against sign/orientation reversals
  test_dvision_perception_chain.py  Render -> detector -> occupancy map, end to end
  vision_debug_report.py    Vision/map diagnostic summary from flight logs
  benchmark_diagnosis.py    Route/control cause summary + failure classification hints
  test_daic_*.py            DAIC detector, planner, map, avoidance tests
  test_dctl_controls.py     Manual control tests
  test_dsim_crash_reset.py  Simulator collision/reset tests
  test_dsim_realism_controls.py  Changing the environment while the sim runs
  test_dcmn_mapview.py      Shared map geometry, both drawing backends
  test_dcmn_theme.py        One palette, and no module keeping its own copy
  test_dcmn_device_view.py  Renderers, pane grid, freeze, export, pop-out
  test_dcmn_layout.py       Grid placement, spans and record repair
  test_dcmn_session.py      Sensor intake, caching and accounting
  test_dctl_devices.py      The Devices tab inside a real dctl window
  test_sensor_*.py          Sensor contract, geometry, state, stereo, health, release
  test_dalg_*.py            Algorithm core, scoring, profile editor, real-process run
  test_module_bus.py        Module presence, run coordination and shutdown
  test_event_viewer.py      Bounded event history, filters and eviction
  test_dvision_wall_clock_independence.py  Nothing depends on how busy the machine is
  test_dvision_sim_speed_conformance.py  The same tour flown at two speeds, and the same report
  test_dtest_harness.py     The suite's own invariants, including staying off screen
  test_dway_*.py            Vehicle contract, tours, flights, realism, editor, transports
  dway_repeatability.py     Repeated baseline flights, aggregated into variance
  benchmark_batch.py        N parallel flights of one configuration, aggregated
```

`assets/` holds fixture data owned by no single consumer: the maps every module
flies, the CC0 textures and tree models the renderer uses, and the committed
tours with their geometry diagnostics.

## Architecture

`dsim` owns the simulated world. It creates the shared-memory buffers, runs the
physics loop, renders the drone camera feed, checks collisions, and publishes
status telemetry.

`dctl`, `daic` and `dway` are clients. They connect to the same buffers,
display video and telemetry, and send commands. Clients retry missing buffers
continuously, so they can be started before or after `dsim`.

Control is leased: exactly one client owns motion, mode and arming at a time,
so three clients on one buffer cannot silently fight. A client acquires the
lease, renews it with a 1 Hz heartbeat, and gives it back on exit; the vehicle
refuses commands that do not carry it, and drops an armed vehicle into `HOLD`
when the lease expires. See [Control ownership](#control-ownership).

`dway` reaches the vehicle only through `dway.link.VehicleLink`, which is the
seam a real drone sits behind: `DsimLink` speaks the JSON protocol below, and a
`MavlinkLink` speaking pymavlink is what a real vehicle would add without
anything above the link changing. `dway` never asks whether it is talking to a
simulator -- it asks what the vehicle's published capabilities allow.

Two modules are shared rather than owned by any process. `dvision2_common.py`
is the protocol -- status keys, command encoding, map loading, report paths --
and stays free of any display dependency so headless code can import it.
`apps/dcmn/` is the layer above it, for what a *view* shares.

`dcmn.theme` is the palette. No window module spells a colour out by hand, and
a test enforces that. `dcmn.tktheme.apply_theme` is that palette applied to
ttk, and every window calls it: a hand-built style block per window agrees on
the colours and drifts on the details, which is how a disabled button ends up
legible in one window and not the next. A window that needs more configures its
own styles on top of the shared base.

`dcmn.mapview` is the top-down map. `dsim`'s monitor, `dway`'s Fly tab, the
tour editor, `dway`'s `track.png` and `dsim`'s `flight_path.png` all draw the
same world, so they draw it from one description: the geometry lives once, in
map metres, and the backends are thin adapters -- one paints it onto a Tk
canvas, one onto matplotlib axes. A private copy per view is how the same wall
becomes light grey in one and dark in another, and how a plot quietly stops
drawing the targets.

All processes share an instance id such as `area1`. Buffer names are derived
from that id:

```text
/dvision2.area1.sensors   Sensor registry dsim -> clients
/dvision2.area1.control   JSON commands clients -> dsim
/dvision2.area1.status    Telemetry k/v dsim -> clients
/dvision2.area1.events    Module bus    every module
```

The camera image is no longer a single `.video` area: `dsim` publishes a
sensor registry from which clients discover every enabled sensor, including
the primary camera's generation-qualified video ring and per-frame metadata.
A profile may also contain further RGB cameras (a synchronized stereo pair is
two of them), 2D and range-image LiDAR on their own array rings, infrared,
ultrasonic and laser rangefinders, and the state sensors a real airframe
carries -- GNSS, IMU, barometer, magnetometer and an ambient thermometer -- on
the shared sample ring. The default profile fits the state sensors and one
camera; `stereo-nav-and-proximity` is the reference vehicle with all twelve.
No client in this repository is required to consume the range, LiDAR or state
streams. See [docs/membus.md](docs/membus.md) for the complete shared-memory
buffer inventory, payloads and module communication guide,
[docs/modcom.md](docs/modcom.md) for the contract and
[docs/sensor/README.md](docs/sensor/README.md) for the per-sensor docs,
limitations and the measured reference budget.

Whether a receiver is *fitted* is the profile's decision, and it is not the
same as whether it has a fix: a profile with no `position.gnss` sensor reports
`gps.fix_type` 0 and an invalid global estimate, which is a different failure
from `--gps off` on a vehicle that carries one.

Multiple independent simulator/controller pairs can run at the same time with
different ids.

## Requirements

Required Python packages:

```text
numpy
Pillow
pymembus
panda3d
opencv-python
```

`matplotlib` is not required to fly, but without it the report images are
skipped: `dsim`'s `flight_path.png` and `dway`'s `track.png` are the two that
go missing, and both say so on stderr rather than failing the run.

The rendering and vision tests additionally need, and pin in
`requirements-visiontests.txt`, packages that can move a rendered pixel or a
measured number:

```text
panda3d-simplepbr  the shading pipeline the `representative` scene preset renders through
matplotlib         diagnostic figures
opencv-contrib     stereo and feature algorithms
```

These are pinned rather than floated because a version change here changes what
the renderer draws, and therefore what any vision algorithm is measured
against.

Optional:

```text
pygame             gamepad/joystick input for dctl
ORB_SLAM3 binding  optional full SLAM obstacle detection for daic
```

Install the common dependencies:

```sh
pip install numpy Pillow pymembus opencv-python panda3d pygame
```

Install and run with the same Python interpreter. In particular, joystick
support is unavailable when `pygame` is installed in a virtual environment but
`dctl` is launched with a different `python3`. Check the interpreter with:

```sh
python3 -c 'import sys, pygame; print(sys.executable, pygame.version.ver)'
```

`daic` also has an installer/check mode:

```sh
python3 apps/daic/daic.py --install
```

That mode checks the OpenCV features needed by the optical-flow and mini-SLAM
paths and reports optional ORB_SLAM3 setup status.

### X input methods

Every Tk front end here declines the X input method as it starts, by setting
`XMODIFIERS=@im=none` before it builds a window. The XIM handshake costs
several hundred milliseconds of a bare Tk start-up, and ibus additionally
leaks a pair of windows per client until the session ends, so a desk that
launches these apps all day watches every launch get slower. None of these
windows takes text an input method exists to compose.

Set `DVISION2_INPUT_METHOD=1` to keep the session's input method instead.

## Quick Start

Manual control:

```sh
# Terminal 1
python3 apps/dsim/dsim.py --id area1

# Terminal 2
python3 apps/dctl/dctl.py --id area1
```

Autonomous flight with DAIC:

```sh
# Terminal 1
python3 apps/dsim/dsim.py --id area1 --map assets/maps/maze_002.txt

# Terminal 2
python3 apps/daic/daic.py --id area1 --enable-ai
```

Fly a tour:

```sh
# Terminal 1
python3 apps/dsim/dsim.py --id area1 --map assets/maps/maze_012.txt

# Terminal 2
python3 apps/dway/dway.py --id area1 --tour assets/tours/maze_012.forward.v1.json
```

The committed `maze_012` tours start on the far side of a wall from that map's
own drone start, and `dway` avoids nothing, so flying one from the map start is
refused by preflight. Move the vehicle onto the corridor with `dctl` first, or
fly a tour authored from where the drone actually is.

Fly a tour headless, with a wind that the vehicle has to trim out:

```sh
python3 apps/dsim/dsim.py --id area1 --map assets/maps/maze_012.txt --no-ui --wind-mps 0.4 &
python3 apps/dway/dway.py --id area1 --tour assets/tours/maze_012.forward.v1.json --no-ui
```

Author a tour with no simulator running at all:

```sh
python3 apps/dway/dway.py --edit --map assets/maps/maze_012.txt
```

Headless automated run:

```sh
python3 tests/flight_test.py --map assets/maps/maze_002.txt --duration 20 --fps 20
```

Headless run with a vision diagnostic log:

```sh
python3 tests/flight_test.py --map assets/maps/maze_002.txt --duration 20 --fps 20 --log /tmp/maze002.jsonl
python3 tests/vision_debug_report.py /tmp/maze002.jsonl
```

Permanent benchmark run with route/control diagnosis (why is the drone still
in `SEARCH`? — perception miss, map noise/trap, planning miss, control stall,
or target-reacquisition miss):

```sh
python3 tests/flight_test.py --map assets/maps/maze_002.txt --duration 30 --fps 20 \
    --report-dir reports/benchmarks/<run-id>
python3 tests/benchmark_diagnosis.py reports/benchmarks/<run-id>
```

`--report-dir` writes `summary.json` (with a nested `diagnosis` block),
`diagnosis.txt`, `report.md`/`report.html`, and the occupancy snapshot gallery
together so a run can be classified without re-running anything. Confirm the
script's classification hints against the `occ_*.png` gallery — they are
heuristic pointers, not a verdict.

## Simulator: dsim

![dsim top-down monitor showing the maze map, drone position, and heading](images/dsim-001.png)

`dsim` creates the world, renders the camera image, accepts control commands,
and publishes telemetry after every tick.

```sh
python3 apps/dsim/dsim.py --id area1 \
  --map assets/maps/maze_001.txt \
  --drone-profile assets/drone_profiles/default.json
```

Options:

| Option | Description |
|---|---|
| `--id` | Required instance id |
| `--map` | Map file to load |
| `--drone-profile` | Built-in profile name or JSON path; omitted resolves the built-in default. The profile is the vehicle's simulated hardware: every sensor with its model, rate and mount pose, the mount/PTZ tree they hang from, and the 300 Hz physics cadence. `stereo-nav-and-proximity` is the committed reference with a PTZ-mounted stereo pair, both LiDAR outputs and three rangefinders. See [docs/sensor/README.md](docs/sensor/README.md) |
| `--sim-speed` | Advance simulated time at this multiple of real time, or `max` for no pacing. Omitted means real time, which is never made to wait. Changeable while running, from the monitor's header |
| `--cmd-size` | Command buffer size in bytes |
| `--start-alt` | Override initial altitude; otherwise map `drone-height` or `1.5` |
| `--origin-lat/lon/alt` | GPS coordinate for the map center |
| `--start-heading` | Initial compass heading in degrees |
| `--setpoint-timeout` | Guided setpoint failsafe in seconds, default 2; `0` disables it |
| `--control-lease-timeout` | Seconds a control lease survives without a heartbeat, default 3 |
| `--max-speed-mps`, `--max-accel-mps2` | Published motion limits |
| `--scene-preset` | Renderer appearance: `representative` (default) or `legacy` |
| `--report-dir` | Write this run's reports here instead of minting a run directory |
| `--frames` | Stop after N frames, useful for tests |
| `--no-ui` | Disable the top-down simulator monitor |
| `--verbose` | Print runtime diagnostics |

The environment flags are a section of their own, below.

### Running faster than real time

Omit `--sim-speed` and the simulator runs in real time, exactly as it always
has: it publishes and moves on, and a client that cannot keep up drops frames.
That is the contract a real vehicle needs, so nothing on the bus is ever
allowed to hold the simulation back.

`--sim-speed` scales the vehicle's clock. Every timer that gates flight reads
`sim.time_s`, so a run at four times real time is the same flight, four times
sooner -- the conformance suite flies one tour at both speeds and compares the
reports. Speed can also be changed mid-flight from the monitor's header.

The cost of a tick is almost entirely rendering: the physics is microseconds
and a frame is about 12 ms, so publishing video at the rate consumers actually
sample is what makes a scaled run fast. The camera profile rate does that
without changing the physics rate or the *simulated* interval between frames.

```sh
# An unattended measurement sweep, as fast as this machine manages.
python3 apps/dsim/dsim.py --id area1 --map assets/maps/maze_020.txt \
        --no-ui --sim-speed max --drone-profile fast-sweep &
simulator=$!
python3 apps/dalg/dalg.py --id area1 --no-ui \
        --profile assets/profiles/optical-flow-maze020.json &
algorithm=$!
python3 apps/dway/dway.py --id area1 --no-ui --exit-on-finish \
        --tour assets/tours/maze_020.default.v1.json \
        --wait-for algorithm:optical-flow-maze020 &
navigator=$!

# The flight and the measurement end themselves; the vehicle does not, because
# a vehicle has no idea it was only wanted for one tour.
wait $navigator $algorithm
kill $simulator
```

That block is measured, not illustrative: it completes a 131-second flight in
about 11 seconds of wall time, roughly 12x, and exits on its own. Swapping
`fast-sweep` for the default 30 Hz camera profile costs almost all of it -- a
scaled run with every physics tick still rendering tops out near 2.9x, because
a `representative` frame is about 12 ms and thirty of them a second is already
most of a second.

One caveat worth knowing before trusting a fast run: a consumer polls on its
own wall-clock loop, so the same frames arriving in a shorter wall interval
outrun it. `dalg` asking for 5 frames per simulated second captured 4.9 in real
time, 4.0 at `--sim-speed 4` and 2.3 at `max`. The *flight* is unaffected at
every speed -- arrival, path and duration all hold -- but a measurement taken
at `max` is built from fewer samples than it asked for. Pick a bounded speed
when the samples matter.

### Environment and Sensor Realism

Every knob below exists because a client has to behave differently when it is
switched on; a setting no client reacts to would be decoration. Their defaults
are a clean baseline -- a good 3-D fix, a valid local estimator, still air, no
delay, no sensor noise, no failsafe and no fence -- so a plain `dsim` is what
other runs are compared against. Every value is published in the status keys,
so a report can name the conditions it was flown in rather than trusting the
command line to have been remembered. `apps/dsim/realism.py` owns the model.

| Option | Description |
|---|---|
| `--gps off\|degraded\|good\|rtk` | Fix quality; sets `gps.fix_type`, satellites and dilution, and the position noise |
| `--gps-noise-m` | Override the mode's own position noise |
| `--local-estimator on\|off` | Whether a local (VIO/SLAM/flow) position estimate exists at all |
| `--wind-mps`, `--wind-dir-deg` | Steady wind speed and the compass direction it blows *from* |
| `--wind-gust-mps` | Gust magnitude, applied as a correlated process on top of the steady wind |
| `--telemetry-latency-ms`, `--telemetry-jitter-ms` | Delay published status through a ring, so clients see the pose late |
| `--sensor-noise none\|light\|heavy` | Compass, barometer (noise plus slow drift) and velocity noise on published state |
| `--ambient-temp-c` | Ground-level air temperature; the temperature sensor cools with height from it |
| `--battery-failsafe-pct` | Battery percentage that triggers RTL, then LAND |
| `--battery-drain-pct-s` | Drain rate while armed |
| `--geofence x0,y0,x1,y1[,max_alt_m]` | Boundary box in map metres, with an optional ceiling |
| `--geofence-action hold\|rtl` | What crossing it does |
| `--realism-seed` | Seed for every random process, so a run reproduces |
| `--vehicle-profile <file.json>` | Defaults for any of the above; an explicit flag still wins |

Two runtime commands change the environment mid-flight, so denial can be
tested on a vehicle that is already airborne: `set_gps` (`mode`, `noise_m`)
and `set_estimator` (`attitude`, `local`, `global`, `velocity`). The simulator's
own Realism tab reaches all of these knobs, not just those two.

`Realism.apply()` is what both paths go through. It validates the whole change
before touching anything, so a rejected value leaves the environment exactly as
it was, and it re-tunes the noise processes without re-rolling them: turning the
wind up must not also teleport the gust. Changing `realism_seed` is how you ask
for the dice back.

The noise processes are correlated in time rather than white -- a fix wanders,
a barometer drifts, gusts build and fade -- because uncorrelated noise on every
sample is trivial to filter and teaches a client nothing.

**What each one buys, and how a correct client reacts**

| Condition | Vehicle behaviour | What the client must do |
|---|---|---|
| `--gps off`, local estimator on | `est.global_position_valid=0`, local stays valid | Fly on: this is the GPS-denied case, and map/local-NED tours do not need a global fix |
| `--gps off`, `--local-estimator off` | Position setpoints are refused | Refuse to fly and name `est.local_position_valid=0`; `dway` fails preflight |
| Estimator faulted mid-flight | Same, while airborne | HOLD and fail with the reason; never fly on the last known pose |
| Wind | The airframe is carried over the ground; the position loop trims it out at hover | Allow settling time: see below |
| Telemetry latency | Status arrives late, in order | Gate on pose age (`max_state_age_s`), not on arrival order |
| `--sensor-noise light` | Published pose and heading wander slightly | Nothing: `light` stays inside the committed tours' arrival gates |
| `--sensor-noise heavy` | Wander exceeds those gates | Widen the tour's own gates; the follower never widens them silently |
| Battery below the failsafe | RTL, then LAND, keeping `failsafe.reason=battery_low` | Stop commanding, report the failsafe |
| Geofence, `hold` | HOLD with `failsafe.reason=geofence`; targets outside are refused outright | Fail with the reason |
| Geofence, `rtl` | RTL, then LAND, same reason | Fail with the reason |

**Wind and the position controller.** A purely proportional approach parks at
an offset of exactly the disturbance divided by its gain, so in any wind at all
it hovers downwind of the setpoint and never satisfies an arrival gate. Both
loops -- the one `dsim` runs onboard and the external one `dway`'s velocity
backend closes -- therefore carry a trim term, accumulated only within a metre
of the target and below hover speed, because during an approach the error is
distance still to travel rather than disturbance and integrating it overshoots
the waypoint.

The measured consequence on `maze_012.forward.v1`, whose legs are 8 m at 1 m/s
with a 0.05 m tolerance:

| Steady wind | Outcome |
|---|---|
| 0.4 m/s | Arrives within the default `max(10 s, 3 x distance / speed)` leg timeout |
| 0.8 m/s | Trims out and arrives, but after roughly 35 s; the default leg timeout expires first, so the tour must set `leg_timeout_s` |
| 1.2 m/s and above | Cross-track drift during travel exceeds this corridor's clearance and the vehicle hits a wall |

A tour therefore has a wind ceiling set by its own clearance, and a wind budget
set by its own `leg_timeout_s`. Neither is adjusted for it silently.

The simulator uses a simple first-order response model:

| Axis | Time constant |
|---|---|
| Horizontal forward/right | 0.30 s |
| Vertical up/down | 0.35 s |
| Yaw rate | 0.10 s |
| Visual roll/pitch | 0.14 s |

`forward_mps`, `right_mps`, and `up_mps` are actual SI velocity setpoints.
`yaw_rate_dps` is degrees per second. Positive yaw is a right/clockwise turn
and increases compass heading. Attitude follows the aviation convention shared
with the FlightGear bridge: `drone.roll_deg` is positive for a right-wing-down
bank (a right strafe banks right) and `drone.pitch_deg` is positive nose-up (so
forward flight pitches nose down).

Collision is box based: wall and tree objects occupy their full map cell and
stand as tall as they are drawn (2.5 m for a wall, 4.5 m for a tree), so flying
over one clears it -- the same heights `dsim.range` casts against, so the range
sensor and the physics agree. A crash puts the drone in `CRASHED` until reset.

### Monitor window

Four tabs. The status line, Save Snapshot and Reset drone sit outside them,
because they are about the vehicle whichever page you are reading. So does the
**speed** menu in the header: it is not a property of the environment the way
the Realism settings are, it changes how fast the whole simulation runs, and
it takes effect on the next tick. Nothing downstream has to be told -- clients
ask whether enough simulated time has passed, and that has the same answer
however fast the clock is turning.

**Map** is the top-down monitor: the map, the drone's position and heading, its
view cone, and the armed/mode state.

**Realism** is every environment knob from the section above, as a form, and it
takes effect on the running simulation. That is the point of it: a fault is far
more informative switched on against a vehicle that is already flying -- deny
GPS mid-leg, raise the wind while a controller is holding station, narrow the
fence under a drone -- and none of that is reachable from a flag you had to
choose before takeoff. Editing a field and pressing Enter applies it, as does
picking from a dropdown; a rejected value says why and changes nothing.

The Estimators row is the runtime fault switch, the same one the `set_estimator`
command reaches. Unticking one faults that estimator; what is actually valid
right now is in the readout underneath, which is a different question -- a 2-D
fix invalidates the global estimate without anybody having faulted it.

Nothing here is persisted. The command line stays the record of how a run
started, and **Reset to command line** puts it back, faults included.

The form is taller than the map, so it scrolls inside a viewport the height of
the map canvas: a notebook is as tall as its tallest page, and without that the
realism tab would set the height of the whole window and push the monitor down
the screen. The wheel scrolls the page even over a combobox, which ttk would
otherwise spin -- a scroll aimed at the page must not silently edit a setting.

**Sensors** also edits the profile name, physics cadence, transport retention
and memory ceiling, camera FOV, and PTZ axis limits. Drag a component onto a
mount or body to reparent it. Edits remain a draft until a disarmed Apply; camera
buffers and channels are prepared before the new generation is advertised.

**Sensors** is the vehicle's hardware: the drone profile, edited as the tree
its parent links imply. Add, duplicate, remove, reparent, enable and rename
components; edit the six-axis mount pose and the type-specific model fields;
make a camera primary, or drop in a synchronized stereo pair. Add opens the
full type list as a drawn popup: each item is a name with a one-line
description beneath it, highlighted as one thing, scrolling when the list
outgrows the screen. The readout
resolves what the draft actually means -- intrinsics and field of view, the
transform from the body, a pair's baseline, its disparity at 10 m and the
rotation between its two cameras, per channel and total shared memory -- and
a validation error appears beside the field that caused it. The settings
scroll when they outgrow the window, with the resolved readout pinned below,
and the wheel over a field scrolls the form rather than spinning a combobox.

Editing here changes nothing. **Apply** hands the simulator one complete,
validated, immutable profile, which it constructs as a new generation and
swaps in whole: a draft that does not resolve, or a generation whose channels
cannot be allocated, leaves the vehicle flying on exactly the profile it
already had. Apply is disabled while armed. Resetting the drone restarts
sample cadence and the noise streams but keeps the selected profile.

**Pipeline** is who is attached and whether anything is falling behind, as a
tree rather than a list. Each module is a row with its own loop rate, and its
sensor inputs hang under it: generation, observed against expected rate, age,
skipped sequences, overruns and sync state, one row per input. `dsim`'s own
production hangs under a **simulator** root beside them, one row per configured
sensor. The two are separate measurements on purpose -- a producer publishing
perfectly is exactly what a stalled reader looks like from the other side --
and the module's single grade is derived from the worst of its *required*
inputs plus its own processing and liveness, so a healthy stream can never
hide a failed one.

### Report Layout

`dsim` owns the report directory for a run and publishes it as the
`sim.report_dir` status key. Every other module reads that key and writes into
its own subdirectory, so a run's pieces stay together and no component has to
agree a name with any other:

```text
reports/<id>/<timestamp>-<random>/
  dsim/     simulator flight path, snapshots, summary.json,
            drone-profile.json, sensor-manifest.json, sensor-plan.json,
            health.jsonl and health report.html
  daic/     controller occupancy snapshots, route log, frames, summary.json
  dway/     flight summary.json, flight.jsonl, track.png
  dalg/     occupancy overlays, prediction grids, scores, summary.json, report.html
  <module>/ any other client, named after itself
```

The instance id comes first so two instances running side by side --
`--id area1` and `--id area2` -- produce two trees instead of one interleaved
list. The run name is a local timestamp for sorting plus eight random hex
characters, so two runs started in the same second cannot collide. A run
started without an id lands under `reports/default/`.

`--report-dir <path>` overrides the whole thing and writes directly to the
named directory, which is how the test harnesses pin a run to a known place.

`dvision2_common.report_root()` and `new_run_id()` define this layout; a new
module should call them rather than build a path of its own.
[`docs/reports.md`](docs/reports.md) is the full contract: ownership, what each
module writes, and the rules a new module follows.

## Manual Controller: dctl

![dctl showing the camera feed, controls, and keyboard legend](images/dctl-001.png)

`dctl` is the manual pilot. It displays the camera feed and sends velocity,
arm, takeoff, land and zero commands. It has three tabs: **Flight** is the
pilot, **Devices** browses every sensor the vehicle publishes, and **Events**
watches the module bus.

```sh
python3 apps/dctl/dctl.py --id area1 \
  --width 960 \
  --height 720 \
  --fps 30
```

Options:

| Option | Description |
|---|---|
| `--id` | Required instance id |
| `--width`, `--height` | Maximum displayed Flight video size |
| `--fps` | Control tick rate. Painting is capped separately: 30 Hz video, 4 Hz text |
| `--cmd-size` | Command buffer size |
| `--speed` | Horizontal speed sent to dsim in m/s |
| `--vertical-speed` | Vertical speed sent to dsim |
| `--no-joystick` | Disable gamepad polling |
| `--camera` | Flight video device; the manifest's primary by default |
| `--devices` | Comma-separated device ids to open in the Devices tab at start |
| `--layout` | Named device layout to restore; the profile name by default |
| `--sensor-cache-mb` | Sensor cache ceiling in MiB, default 64 |
| `--no-sensors` | Disable sensor discovery, video and the Devices tab |
| `--verbose` | Log commands to stdout |

Keyboard controls:

| Key | Action |
|---|---|
| `W` / Up | Move forward |
| `S` / Down | Move back |
| `A` / Left | Strafe left |
| `D` / Right | Strafe right |
| `R` / Page Up | Move up |
| `F` / Page Down | Move down |
| `Q` / Home | Yaw left |
| `E` / End | Yaw right |
| `Space` | Hover / zero velocity |
| `T` | Takeoff |
| `L` | Land |
| `M` | Arm/disarm toggle |

Gamepad controls use an Xbox-style layout:

| Control | Action |
|---|---|
| Left stick X/Y | Strafe / forward-back |
| Right stick X/Y | Yaw / up-down |
| A | Hover |
| B | Land |
| X | Arm toggle |
| Y | Takeoff |
| Back | Disarm |
| Start | Arm |

Manual yaw is intentionally normalized so joystick and keyboard yaw directions
match the UI labels and simulator heading behavior.

**Control ownership.** The vehicle takes commands from one client at a time, so
`dctl` claims the lease on connect -- but only if the vehicle is unowned; it
never contends with a `dway` tour or a flying `daic` -- and renews it about
once a second while it is running. Renewal is paced against the lease age the
vehicle publishes, so an accelerated `--sim-speed`, which retires a lease
sooner in wall time, does not cost the operator control mid-flight. The
Controls panel has **Take Control** and **Release Control** for the case that
matters: handing the vehicle to `dway` for a tour and taking it back
afterwards. A release is deliberate and latches -- `dctl` will not reclaim the
vehicle until you press **Take Control** again. The telemetry panel shows who
currently holds it, and a failed acquire names the holder rather than failing
silently. Without the lease only `land` is accepted: a `dctl` that shows an
empty `control.owner` has every other button refused.

Because the guided setpoint failsafe is on by default, a `dctl` that stops
sending velocity -- all keys released, no stick input -- lets the vehicle fall
into `HOLD` after `--setpoint-timeout` seconds. That is the intended behaviour:
a heartbeat deliberately keeps the lease alive without keeping a stale setpoint
alive. `dctl` holds its last velocity only while an input is actually held.

### Devices tab

The reference vehicle publishes twelve devices of eleven types through one
discovery registry, and the Flight tab looks at one of them. The **Devices**
tab is the window that answers "what is this vehicle actually seeing right
now" (pictured at the top of this file). It is driven entirely by the
published `sensors.manifest`, so a sensor type added to `dsim` appears here
without `dctl` being edited.

The tree on the left mirrors the profile's mount/PTZ component tree, with a
health dot per device. Drag a row onto a grid cell to open a pane in exactly
that cell, replacing whatever pane stood there; the check mark is an
indicator, not a control. Panes tile, span and resize: drag a pane's title
onto another to swap them, drag its bottom-right handle to extend its row and
column span, drag the sashes to change proportions, or tick **Arrange** to
edit the grid skeleton itself with **+ Row** / **+ Col** and their removals.

Each pane's header carries three small discs, sized and packed so a pane
squeezed narrow clips its own name rather than losing its controls:

| Disc | Action |
|---|---|
| **i** | Swap the graphic for the pane's text readout, and back |
| **❄** | Freeze this pane alone; intake keeps draining behind it |
| **×** | Close the pane, which unchecks the device |

The readout replaces the graphic rather than sharing the cell with it: a
health dot, achieved against configured rate in simulated Hz, sample time,
capture id, the sequence the pane attached at, gaps, association and late
drops, plus whatever the renderer has to say about the sample. The choice
persists with the pane.

What each device class draws:

- **Cameras** -- the frame, fit to `contain`, `cover` or `native` pixels, with
  resolution, calibrated fields of view and capture pose in the readout.
- **Stereo pairs** -- a `sync_group` of exactly two cameras appears as one
  `stereo:<group>` device, composed only from frames sharing a `capture_id`,
  as `side-by-side`, `anaglyph` or `difference`.
- **Scanning LiDAR** -- calibrated rays and range rings coloured by
  confidence, forward-up or north-up, automatic or manual range scale.
- **Range images** -- a fixed viridis ramp over the sensor's configured metre
  range, or the diverging confidence ramp; invalid cells grey, nearest-neighbour
  resizing, and hover reporting the original cell's range.
- **Rangefinders** -- metres, a gauge, and a 10/30/60-second strip chart;
  invalid samples say **no return** and disable the gauge.
- **IMU** -- angular rate and specific force as three-axis envelope charts
  sharing one time axis, so a 100 Hz sensor stays affordable at a 4 Hz paint.
- **GNSS** -- the fix state as one of three distinct failures, quality,
  position, NED velocities, and a north/east error scatter: a blue dot per
  envelope bucket at its mean error with a grey box over that bucket's
  observed min/max, so drift has a visible shape.
- **Barometer and thermometer** -- value, units and strip chart.
  **Magnetometer** -- a full compass rose, since a heading that wraps makes a
  line plot noise.

**Freeze all** holds every visible pane at one shared `capture_id` so they
show the same simulated instant; untick **Sync captures** to hold each pane at
its own newest record instead. Right-click a pane to pop it out into its own
window, or to export: **Snapshot PNG** rasterizes the displayed graphic with
its readout, **Dump JSON**/**Dump CSV** write the samples behind it into the
run's report directory. The bench -- cells, spans, sash weights and each
renderer's options -- persists per profile name in
`~/.config/dvision2/device_layouts.json`, and a popped-out window's geometry
in `window_pos.json` beside it.

Painting is budgeted, never gating: image panes share 30 Hz between them, text
panes repaint at 4 Hz, and a pane showing its text body drops out of the video
budget entirely. [`docs/dctl-devices.md`](docs/dctl-devices.md) is the full
contract -- intake and accounting, the renderer contract, stereo pairing,
freeze invalidation and the measured speed ceiling.

### Events tab

The **Events** tab is a passive reader of the instance's module bus
(`/dvision2.<id>.events`). It registers as an observer and never publishes, so
watching a run cannot change it.

Rows arrive in order with their local arrival time, source, event type, run id
and a payload summary; selecting one shows the decoded envelope and payload
beneath. The three entry fields filter by substring on source, type and run
id. **Hide heartbeats** and **Hide sensor health** are on by default, because
otherwise the once-a-second traffic buries everything else. **Pause display**
freezes painting while collection continues. **Auto-follow** sticks to the
newest row and switches itself off the moment you scroll or arrow away, so the
next paint does not snap the view back.

History is bounded -- 1000 rows or 4 MiB, whichever comes first, oldest
evicted -- and the status line reports exactly where the losses were: shown
against retained, total received, reader overruns on the bus, and rows
discarded locally. [`docs/membus.md`](docs/membus.md) and
[`docs/modcom.md`](docs/modcom.md) describe what the events themselves mean.

## AI Controller: daic

![daic showing the camera feed with obstacle overlays, SLAM map, and local route](images/daic-001.png)

`daic` is the autonomous client. It reads only the video and status buffers,
then sends commands through the command buffer. It can run with a Tk UI or in
headless mode for automated testing.

Like `dctl` it acquires the control lease and heartbeats while connected, and
re-acquires it if something else took the vehicle; it never sends motion
commands without holding it.

```sh
# UI, manual AI toggle
python3 apps/daic/daic.py --id area1

# UI, autonomy enabled immediately
python3 apps/daic/daic.py --id area1 --enable-ai

# Headless
python3 apps/daic/daic.py --id area1 --enable-ai --no-ui

# Log every control tick
python3 apps/daic/daic.py --id area1 --enable-ai --log-file /tmp/flight.jsonl
```

Options:

| Option | Description |
|---|---|
| `--install` | Check/install DAIC vision dependencies, then exit |
| `--id` | Instance id; required unless `--install` is used |
| `--display-w`, `--display-h` | Maximum displayed video size, `0` for native |
| `--video-w`, `--video-h` | Expected frame size for detector/servo gains |
| `--fps` | Control loop rate. Painting is capped separately: 30 Hz video, 10 Hz maps, 4 Hz text |
| `--cmd-size` | Command buffer size |
| `--enable-ai` | Enable AI immediately on startup |
| `--no-ui` | Headless mode |
| `--log-file` | Write structured JSONL flight log |
| `--slam-vocab` | Path to `ORBvoc.txt`; enables ORB_SLAM3 obstacle detection |
| `--slam-settings` | Optional ORB_SLAM3 YAML; generated from camera status when omitted |
| `--verbose` | Print planner state each tick |

### Controller window

The DAIC window is arranged around the live camera feed. The video is displayed
with target and obstacle overlays. The SLAM/map panels sit under the video so
the window stays reasonably wide instead of becoming excessively tall.

The control side panel contains:

- AI enable/disable state
- Emergency stop
- Current mission state
- Planner status text
- Target detection readout
- Altitude lock
- Component health for video, command, status, detector, SLAM/flow, and planner
- Telemetry such as armed state, mode, position, heading, speed, and battery

Altitude lock is enabled by default. Outside active landing it prevents DAIC
from sending vertical velocity commands, which keeps route-planning tests from
accidentally mixing horizontal navigation with altitude changes.

### Mission State Machine

DAIC's mission planner handles arming, search/transit, target approach, and
landing:

```text
IDLE -> ARMING -> SEARCH -> APPROACH -> LANDING -> COMPLETE
                  ^            |
                  |            |
                  +-- target lost

any state -> FAILSAFE on stale telemetry, low battery, timeout, or crash
```

During `SEARCH`, DAIC primarily navigates toward the target GPS position from
the status buffer. When local vision mapping has an available route, the GPS
transit command is replaced by the local route command.

During `APPROACH` and `LANDING`, the red target detector and visual-servo
controller take over. The target controller approaches before it descends: it
keeps forward motion while the target is small/far, then descends only once the
target is large enough and below the camera center.

## Waypoint Navigation: dway

`dway` is the autopilot client. It loads a tour, negotiates with the vehicle
how the tour can be flown, streams setpoints, advances on arrival, and writes a
flight report. `dctl` is the manual pilot and `daic` is the vision experiment;
`dway` is the one that flies a plan.

```sh
# Terminal 1
python3 apps/dsim/dsim.py --id area1 --map assets/maps/maze_012.txt

# Terminal 2
python3 apps/dway/dway.py --id area1 --tour assets/tours/maze_012.forward.v1.json
```

Options:

| Option | Description |
|---|---|
| `--id` | Required instance id, shared with the simulator |
| `--tour` | Tour JSON file to fly |
| `--strategy` | `auto` (default), `position`, or `velocity` to force a backend |
| `--speed` | Override the tour's `default_speed_mps` |
| `--stream-hz` | Setpoint stream rate, default 10 |
| `--finish-action` | `land` (default), `hold`, or `rtl` after the last waypoint |
| `--wait-for-start` | Stay in `READY` until Start is pressed. Disables the readiness barrier below |
| `--wait-for` | Require a module role to acknowledge this run before it starts, e.g. `algorithm:sgbm-maze020`. Repeatable |
| `--ready-timeout-s` | Simulated seconds to wait for those modules, default 15 |
| `--start-delay-s` | Simulated seconds between readiness and the start, default 3 |
| `--client-id` | Control-lease identity, default `dway-<id>` |
| `--ack-timeout` | Seconds to wait for a command acknowledgement, default 3 |
| `--timeout` | Abort the flight after this many wall-clock seconds |
| `--exit-on-finish` | Close the window when the flight ends |
| `--no-ui` | Run headless, for scripted flights |
| `--edit` | Open the tour editor alone, with no vehicle and no `--id` |
| `--map` | Map to open in `--edit` mode |

```sh
# Author a tour without a simulator running at all
python3 apps/dway/dway.py --edit --map assets/maps/maze_012.txt
```

### Waiting for other modules

A tour is a stimulus other modules measure, and a measurement that begins
after the flight does is a shorter measurement than the one that was asked
for. `--wait-for` makes `dway` the coordinator: it publishes `run.prepare` on
the instance's event bus, holds in `READY` until every named role has answered
`run.ready` for that exact run, then schedules a start at an absolute
simulated time so every participant begins together.

```sh
python3 apps/dsim/dsim.py --id area1 --map assets/maps/maze_020.txt &
python3 apps/dway/dway.py --id area1 \
        --tour assets/tours/maze_020.default.v1.json \
        --wait-for algorithm:optical-flow-maze020 &
python3 apps/dalg/dalg.py --id area1 \
        --profile assets/profiles/optical-flow-maze020.json &
```

The argument is `role[:selector]`, and both halves catch people out:

- **The role is the one the module registers on the bus with**, not its
  program name. `dalg` registers as `algorithm`, so `--wait-for dalg` never
  matches and the run aborts with `readiness timeout waiting for dalg`.
- **The selector is the profile's `name` field**, not the path passed to
  `--profile`. For the profile above that is `optical-flow-maze020`; the
  algorithm name `optical_flow_triangulation` and the module's process id also
  match. A path never matches.

Omitting the selector (`--wait-for algorithm`) accepts any one algorithm
module, and aborts with `ambiguous exclusive participant` if two answer.
Naming the profile is what pins a run to the module you meant.

Launch order does not matter: the preparation snapshot is repeated once a
second, so a module that starts after `dway` still joins the same run. What
does matter is that the participant agrees with the tour -- a `dalg` profile
naming a different tour rejects the run outright, and `dway` aborts
immediately with the reason rather than waiting out the timeout.

Without `--wait-for`, `dway` flies as soon as it is ready and other modules
observe opportunistically. That usually works, because `--start-delay-s`
leaves them time to attach; the barrier is what turns "usually" into a run
that either starts together or fails loudly.

`--wait-for` has no effect under `--wait-for-start`, where the flight begins
from the Start button instead.

### Follower window

Three tabs, and the window is optional -- `--no-ui` has been there from the
start so scripted flights need no display.

**Fly** draws the map with the tour on it, the vehicle and its heading, the
current target and the leg to it, progress through the waypoints with the
dwell countdown, live telemetry, and the strategy that was selected together
with the capability facts that chose it. Start, pause, resume, hold, RTL and
land are there; a control the mission or the vehicle refuses says so rather
than appearing to do nothing.

**Vehicle** is the page to open when the drone will not fly. It names the one
fact preventing flight -- a missing capability, an invalid estimator, a stale
pose, another client holding the lease, or an active failsafe -- checked in the
order in which each would stop a flight, and then shows the negotiated
capability profile, fix type and satellites, estimator validity, setpoint age,
control ownership, battery, wind and geofence. It answers "why not" without
reading a log.

**Tour editor** (`apps/dway/editor.py`) opens a map, places and drags waypoints,
rotates each one's heading by dragging its arrow, and edits the tour's speed,
tolerance and clearance. It shows live geometry -- path length, longest
straight run, per-leg clearance -- and measures **leg zero from the map's own
start pose**, because a tour whose first leg is unflyable otherwise looks fine
in an editor. Saving writes the file and immediately reads it back through the
loader: saving is not the same as being loadable. `dalg` imports this tab
rather than growing one of its own.

All three tabs draw the map through `dcmn.mapview`, so a wall, a tree, a target
and the vehicle look the same here as they do in `dsim`'s own monitor. What
`dway` adds on top -- the planned route, waypoint numbers, the current leg and
the flown track -- is its own.

### The vehicle seam

`dway` never asks whether it is talking to a simulator. Everything goes through
`dway.link.VehicleLink`: capabilities, state, control lease, arm, takeoff,
land, RTL, hold, and the two setpoint types. `DsimLink` speaks the JSON
protocol below; a `MavlinkLink` speaking pymavlink is what a real vehicle would
add, and nothing above the link would change.

Control is leased. `dway` acquires the lease before it arms anything, renews it
with a 1 Hz heartbeat, and gives it back on exit. A heartbeat renews the lease
but deliberately does **not** refresh the vehicle's setpoint timer, so a client
that stops flying but keeps saying hello still fails safe.

Every command carries a request id and waits for the matching result. Queue
admission is not acceptance: a lease check or a mode check can still refuse a
setpoint, and a refusal ends the flight with the reason rather than being
retried.

### Tours

A tour is a file in the repository -- there is no store and no database. The
loader accepts a single tour object with `status` absent or `applicable`, at
least one waypoint, and a supported `schema_version`; it rejects the aggregate
`diagnostics.v1.json` and every `not_applicable` tour with a reason.

| Field | Meaning |
|---|---|
| `coordinate_frame` | `map` (default), `local_ned`, or `global` |
| `waypoints` | `{x,y,z}`, `{north_m,east_m,down_m}`, or `{lat_deg,lon_deg,alt_m}` plus `heading_deg` and `dwell_s` |
| `map`, `map_sha` | Map the tour was authored against; the hash is checked before control is acquired |
| `waypoint_tolerance_m` | Arrival distance gate |
| `arrival_speed_mps` | Arrival speed gate, default 0.15 |
| `heading_tolerance_deg` | Arrival heading gate, default 5 |
| `max_state_age_s` | Oldest pose that may be flown on, default 0.5 |
| `leg_timeout_s` | Overrides the default `max(10 s, 3 x distance / speed)` |
| `min_clearance_m` | Clearance a leg is expected to keep from map geometry |
| `geo_anchor` | `origin_lat_deg`, `origin_lon_deg`, `origin_alt_m`, `rotation_deg` -- where a map-frame tour sits on the Earth |

Map X is east-positive, map Y is south-positive, map Z is metres above map
ground. The local-NED origin is the published geographic origin at the centre
of the map:

```text
east_m  = x - width/2
north_m = height/2 - y
down_m  = -z
```

Compass heading is unchanged between those two frames. `geo_anchor.rotation_deg`
is the clockwise angle from map north to true north, applied to the
`(east, north)` vector and to headings before projecting to WGS84.
`apps/dway/frames.py` owns both directions of every conversion.

Preflight measures the clearance of every leg **including leg zero**, the
movement from wherever the vehicle currently is to the first waypoint. A leg
that passes through map geometry is refused outright -- `dway` flies the tour it
was given and avoids nothing. A leg that merely passes closer than the tour's
`min_clearance_m` is reported as a warning in the log and the report, because
several committed tours legitimately fly lines that tight. Note that the
`maze_012` tours start on the far side of a wall from that map's own drone
start, so flying one from the map start is refused; move the vehicle onto the
corridor first.

### Following

Sequencing happens off-vehicle: publish the current target continuously,
advance only on arrival. A sample is inside the arrival gate when the 3-D
distance, the total speed and the wrapped heading error are all within their
tolerances, and all three must stay true continuously for the waypoint's
`dwell_s`. Leaving the gate resets the dwell clock, and a zero-dwell waypoint
advances on the first in-gate sample rather than being flown through.

The strategy is chosen from capabilities, never from the vehicle's identity:

```text
accepts_position_target      -> stream position targets
else accepts_velocity_target -> close the position loop here, send velocity
else                         -> refuse to fly, and say which capability is missing
```

The stream must be faster than twice the vehicle's advertised setpoint timeout
or preflight refuses the configuration.

### Flight lifecycle

```text
DISCONNECTED -> PREFLIGHT -> READY -> ARMING -> TAKING_OFF -> FLYING
                                      |                         |
                                      +-> FAILED <--------------+
FLYING <-> PAUSED -> COMPLETING -> LANDING -> COMPLETE
   |          |            |
   +----------+----------> RTL -> LANDING
```

Preflight validates the tour and the map hash, reads capabilities and a fresh
state, selects a strategy, checks clearance, and acquires control. Takeoff is
skipped when the vehicle is already airborne. Pause commands HOLD and stops the
mission clock; resume revalidates health, reacquires control if it was lost,
and restreams. A rejected command, a lost lease, a crash, stale state, an
invalid estimator or a leg timeout enters `FAILED` with the exact reason and
never retries. `SIGINT` or closing the window commands HOLD, writes a partial
report and releases control -- it does not disarm an airborne vehicle.

### Report

`<sim.report_dir>/dway/` holds `summary.json` (versioned; per-waypoint arrival
times, dwell, overshoot and cross-track error, path length, failsafes,
`partial`, and a `conditions` block naming the environment the run was flown
in), `flight.jsonl` (one line per command and event, with the request and
result ids and a state snapshot) and `track.png` (planned versus flown, over
the same map the live windows draw, with the legend in the lower right).

### Repeatability

`dalg`'s premise is that a tour is a predictable stimulus, so how repeatable
closed-loop following actually is had to be measured rather than assumed.
`tests/dway_repeatability.py` flies one baseline tour N times through real
`dsim` and `dway` processes with realism off and aggregates the runs:

```sh
python3 tests/dway_repeatability.py --runs 5
```

It writes `reports/dway-repeatability/<name>/repeatability.json` -- run count,
mean and variance of path length, and mean and variance of each waypoint's
arrival time.

Five runs of the two-waypoint `maze_012` corridor tour, 13 m of path at
1 m/s:

| Measure | Mean | Variance | Spread |
|---|---|---|---|
| Path length | 13.056 m | 2.2e-5 m² | ~5 mm, 0.04% of path |
| Waypoint 0 arrival | 8.919 s | 1.9e-3 s² | ~43 ms |
| Waypoint 1 arrival | 16.291 s | 2.6e-3 s² | ~51 ms |

Millimetres of path and tens of milliseconds of timing across whole flights, so
closed-loop following is repeatable enough to be a stimulus and the open-loop
tour-player fallback is not needed. Numbers are from one machine; rerun the
harness rather than trusting these on another.

## Vision Navigation

DAIC's obstacle/navigation stack is deliberately vision-first. It may use:

- RGB video frames from the video buffer
- telemetry/status values such as pose, velocity, heading, camera intrinsics,
  and target GPS

It must not read `dsim` map files or simulator object lists to decide where to
fly. Map files are for the simulator and tests only.

### Processing Pipeline

```text
RGB frame + status
      |
      +--> red target detector
      |
      +--> mini-SLAM / optical-flow obstacle detection
                 |
                 v
          fused obstacle sectors
                 |
                 v
        local occupancy map
                 |
                 v
          A* local route
                 |
                 v
       planner command + avoidance brake
```

### Obstacle Sectors

Obstacle detectors emit five sector risks:

```text
left, front_left, front, front_right, right
```

Each sector risk is `0.0` for clear and `1.0` for fully blocked. Sectors also
carry optional range estimates such as `front_range_m`. The fused sector result
takes the maximum risk from available detectors and the nearest range estimate
for each sector.

### Optical Flow and Range

`apps/daic/optical_flow_avoidance.py` computes dense Farneback optical flow between
successive video frames. Radial expansion from the image center indicates that
the drone is moving toward visible structure.

The detector now estimates range using time-to-contact:

```text
range ~= forward_speed * dt / radial_expansion_ratio
```

The implementation uses body-forward speed from `drone.vx_mps`,
`drone.vy_mps`, and `drone.heading_deg`, then applies conservative clipping
and a calibration gain for the dsim camera geometry. The bottom of the frame is
ignored for range estimation because the pitched camera sees the floor; using
that region directly makes every wall appear extremely close.

Range is not available when the drone has too little forward motion or when
the optical-flow field is too weak. In that case the local map falls back to a
default projection distance.

### Local Occupancy Map

`apps/daic/local_map.py` maintains a rolling occupancy grid around the drone. It:

- decays stale cells over time
- marks a short free-space fan in front of the drone
- projects obstacle sectors into world-frame grid cells
- uses per-sector range estimates when available
- falls back to a default projection distance when range is unavailable
- plans a local A* path to the status-derived target position

The route follower converts the next lookahead waypoint into forward velocity
and yaw-rate commands. It is intentionally damped so the drone can move while
turning moderately instead of spinning in place at every waypoint.

### Avoidance Brake

`apps/daic/avoidance.py` is the last safety layer before sending velocity commands.
It does not inject lateral movement or yaw. It only trims forward speed when
front-sector risk is high. That keeps steering under the route planner while
still reducing forward motion into detected obstacles.

## Algorithm Demonstrator: dalg

`dalg` answers a narrow question: given the same flight, how well does a
mapping algorithm reconstruct the world? It attaches to a running `dsim`,
observes the sensors its profile asks for while `dway` flies a tour, builds an
occupancy grid, and scores that grid against ground truth rasterised from the
map.

```sh
python3 apps/dsim/dsim.py --id area1 --map assets/maps/maze_020.txt &
python3 apps/dalg/dalg.py --id area1 \
        --profile assets/profiles/optical-flow-maze020.json &
python3 apps/dway/dway.py --id area1 --no-ui --exit-on-finish \
        --tour assets/tours/maze_020.default.v1.json \
        --wait-for algorithm:optical-flow-maze020
```

Options:

| Option | Description |
|---|---|
| `--id` | Instance id of the simulator to observe |
| `--profile` | Committed profile name or JSON path |
| `--no-ui` | Run headless; the report is still written |
| `--timeout` | Seconds to wait for the run before giving up, default 180 |
| `--edit` | Open the profile editor instead of running |

A **profile** is one flat, diffable JSON object: the algorithm to run, the
tour it expects, the sensors it wants, and the algorithm's own settings. It
carries a digest, so a report names the exact configuration it was produced
by. The committed set lives in `assets/profiles/` beside the maps and tours,
because that is what a profile is -- fixture data, owned by no consumer.

`dalg` registers on the module bus as `algorithm`, which is the role
`dway --wait-for` names; see
[Waiting for other modules](#waiting-for-other-modules) for why the barrier
matters and why `--wait-for dalg` never matches. A profile naming a different
tour than the one being flown rejects the run outright rather than quietly
measuring the wrong flight.

The algorithms live one to a module under `apps/dalg/algo/`:
`sgbm` and `plane_sweep` from a stereo pair, `feature_triangulation` and
`optical_flow_triangulation` from a moving monocular camera, `ground_plane`
from camera geometry, and `monocular_depth` from an ONNX metric-depth model
(installed by `scripts/install_dalg_depth_model.py`). Two of them are not
algorithms but controls: `constant` predicts one probability everywhere and is
the floor any real result must clear, and `exact_range` is an oracle built
from the truth grid -- the ceiling, not a simulated sensor. A result that
cannot beat `constant` has measured nothing.

Scoring covers only the cells the flight *could* have seen. `dalg.visibility`
builds a deliberately generous mask -- ever within the camera's horizontal
field of view, in range, and not behind a wall -- because charging an
algorithm for rooms the vehicle never flew past flatters the controls and
buries the difference between the real algorithms. Within that mask,
`summary.json` reports occupied and free IoU with occupied precision and
recall, coverage (how much of the region the algorithm committed to at all),
a Brier score over the probabilities, and a hallucination rate: free truth
cells predicted occupied.

The run writes into the shared report tree at `reports/<id>/<run>/dalg/` --
`summary.json`, per-algorithm overlay and raw prediction images, the scored
region, `events.jsonl` and an HTML report. `compare.py` reads those summaries
offline to put several runs beside each other. An aborted run still writes its
report, marked `partial` with the reason.

## Automated Testing and Diagnostics

Run all tests:

```sh
pytest -q
```

`pytest -q` never opens a window on your desktop: widget tests run on a
withdrawn root, and the few device-browser tests that genuinely need a
*mapped* toplevel — freeze, pop-out and keyboard-focus delivery — take their
headless halves instead. Set `DVISION2_GUI_TESTS=1` to exercise those widget
halves on a real display:

```sh
DVISION2_GUI_TESTS=1 pytest -q tests/test_dcmn_device_view.py tests/test_dctl_devices.py
```

Install and verify the pinned vision-test environment:

```sh
python3 -m pip install -r requirements-visiontests.txt
python3 -m dtest.preflight
```

`dtest.preflight` reports the `deterministic`, `rendering`, and `process`
dependency groups separately. If rendering or IPC support is missing, those two
test groups are skipped from collection and the run header says so; the
deterministic physics and coordinate contract tests always run.

Run only the deterministic coordinate/video contract:

```sh
pytest -q tests/test_dvision_coordinate_contract.py \
  tests/test_dvision_calibration_render.py
```

Run the real-process command/status/video and DAIC integration checks:

```sh
pytest -q tests/test_dvision_process_transport.py
```

The process harness allocates a unique IPC ID per test, waits on readiness and
status conditions rather than fixed startup sleeps, records failure frames and
telemetry, and removes shared-memory resources during cleanup. The DCTL GUI
smoke case uses `xvfb-run` when a virtual display is available.

`dtest/conformance.py` holds the backend-neutral suite. It runs against the
in-process deterministic simulator and the real DSIM process today, and is what
a future MAVLink backend must pass.

For CI artifact upload, point failure bundles at a persistent directory:

```sh
DVISION_TEST_ARTIFACTS=/path/to/ci-artifacts pytest -q
```

A failure bundle contains the initial and final raw frames, an annotated frame
showing observed centroids against expected regions, the commands sent, a
pose/velocity/heading/epoch/frame-sequence timeline, a top-down path plot when
the drone moved, the fixture and camera parameters, and a `result.json`
summary. These are diagnostics; the assertions themselves are the oracle.

Run the opt-in longer calibration stream (20 seconds by default):

```sh
DVISION_NIGHTLY=1 pytest -m nightly -q
```

Set `DVISION_NIGHTLY_SECONDS` to change its duration. The CI/scheduler should
retain `DVISION_TEST_ARTIFACTS` when a job fails.

Run the end-to-end perception chain (rendered pixels → detector → occupancy
map):

```sh
pytest -q tests/test_dvision_perception_chain.py
```

Each link in that chain is unit-tested in isolation, which is not the same as
testing the chain: a convention can be applied consistently *within* two stages
and still disagree *between* them. These tests fly the real detector over real
rendered frames and ask where the obstacle ended up in world coordinates. The
left/right fixtures are mirror images, so a handedness error has to change the
answer's sign rather than merely shift it.

### Reversal audit

Sign and orientation bugs are the failure mode this project hits most: the data
is well-formed and only its *interpretation* is mirrored, so a test that merely
asserts "something happened" passes straight through. `tests/reversal_mutations.py`
measures whether the suite actually notices. It injects one reversal at a time
into a production file, runs the suite, and reports whether anything failed:

```sh
python3 tests/reversal_mutations.py            # audit every catalogued reversal
python3 tests/reversal_mutations.py --list     # show the catalogue
python3 tests/reversal_mutations.py -k slam    # only matching mutations
```

A `MISSED` row is the useful output: it names a boundary where a mirrored axis
or an inverted sign would ship silently. The script exits non-zero if anything
is missed, so it can gate a merge. Target files are backed up to a temporary
directory and restored in a `finally` — including on Ctrl-C — and the restore
is verified by digest.

Add a case to the `MUTATIONS` catalogue whenever a new coordinate, sign, or
image-orientation boundary appears. An `ANCHOR` row means the catalogue snippet
no longer matches the code and needs updating.

Run a headless mission:

```sh
python3 tests/flight_test.py
```

Useful options:

| Option | Description |
|---|---|
| `--map` | Map file relative to the project root |
| `--duration` | Maximum flight time in seconds |
| `--fps` | Simulation FPS and frame-budget basis |
| `--log` | JSONL log path; defaults to `/tmp/daic_flight_<ts>.jsonl` |
| `--verbose` | Print subprocess diagnostics |

Example:

```sh
python3 tests/flight_test.py \
  --map assets/maps/maze_002.txt \
  --duration 20 \
  --fps 20 \
  --log /tmp/maze002.jsonl
```

The test runner:

1. launches `dsim` with `--no-ui --frames N`
2. waits for buffers to exist
3. launches `daic` with `--no-ui --enable-ai --log-file`
4. waits for the simulator budget to finish
5. terminates DAIC
6. analyzes the log
7. exits `0` on a successful landing and `1` otherwise

### Flight Logs

DAIC logs one JSON record per control tick. A tick contains mission state,
planner status, target detection, command fields, telemetry, and vision
diagnostics:

```json
{
  "t": 0.351,
  "state": "SEARCH",
  "status": "GPS nav 9 m to target",
  "det": {"visible": false},
  "cmd": {"type": "velocity", "forward_mps": 0.45, "yaw_rate_dps": 0.0},
  "telem": {"drone.x_m": "17.500", "drone.y_m": "16.490"},
  "vision": {
    "fused": {
      "method": "flow:expansion+persist",
      "front": 0.35,
      "front_range_m": 1.54
    },
    "local_map": {
      "occupied_cells": 11,
      "front_occ_m": 1.99,
      "default_obstacle_projection_m": 3.0
    }
  }
}
```

Use the diagnostic report to inspect whether perception and mapping agree:

```sh
python3 tests/vision_debug_report.py /tmp/maze002.jsonl
```

The report highlights:

- ticks with front/front-left/front-right obstacle risk
- fused sector risk values
- estimated sector ranges
- nearest and front occupied map cells
- occupied/free cell counts
- whether the map had to fall back to the default projection distance

This is the fastest way to debug cases where the drone appears to see a wall
but the planner behaves as if it is trapped or as if the wall is at the wrong
distance.

## Maps

Maps are plain text files with three sections:

```text
--- DATA
drone-height=1.5

--- VARS
+=drone
*=target
0=wall
1=tree

--- MAP
000000
0    0
0 +* 0
0    0
000000
```

`DATA` contains key/value settings. `drone-height` sets the starting altitude.

`VARS` maps characters to object kinds.

`MAP` is an ASCII grid. Each character is one simulated meter. Columns are
local `x`, rows are local `y`, and object centers are at cell centers. Exactly
one drone start cell (`+`) is required.

Built-in symbols:

| Symbol | Kind | Rendered as |
|---|---|---|
| `+` | drone | Start position |
| `*` | target | Red ground marker |
| `0` | wall | Brick-textured box, 2.5 m tall |
| `1` | tree | GLB tree model or primitive fallback |
| space | empty | Traversable floor |

Included maps:

| File | Description |
|---|---|
| `maze_001.txt` | Default challenge map |
| `maze_002.txt` | Corridor/interior-wall navigation case |
| `maze_003.txt` | Additional layout |
| `maze_012.txt`, `maze_013.txt`, `maze_014.txt` | The layouts the committed tours are authored against |
| `test_direct.txt` | Open field, no obstacles |

Test fixtures, which are maps but are not meant to be flown for fun:

| File | Purpose |
|---|---|
| `calibration_orientation.txt`, `calibration_orientation_ring.txt` | Coloured landmarks at known bearings, the render/orientation oracle |
| `chain_front_obstacle.txt`, `chain_left_obstacle.txt`, `chain_right_obstacle.txt` | One obstacle in one place, for the perception chain; left and right are mirror images so a handedness error changes the answer's sign |
| `range_chirality.txt` | Asymmetric geometry that catches a mirrored range backend |

### Committed tours

`assets/tours/` holds committed waypoint tours for `maze_001` and `maze_012` to
`maze_014`, in six archetypes: `forward`, `strafe`, `yaw_only`, `orbit`,
`boustrophedon` and `stop_and_stare`. `diagnostics.v1.json` is the aggregate
geometry record for the set; it is not itself flyable, and `dway`'s loader
rejects it with a reason, as it does any tour marked `not_applicable`.

The tour format, its coordinate frames and its arrival gates are described
under [Waypoint Navigation](#waypoint-navigation-dway).

## Shared Memory Protocol

All buffers are provided by `pymembus` and named as:

```text
/dvision2.<id>.<channel>
```

| Buffer | Channel | Direction | Type |
|---|---|---|---|
| Sensor registry | `.sensors` | dsim -> clients | `memkv` discovery manifest |
| Camera video | `...sensor.<id>.video` (generation-qualified) | dsim -> clients | `memvid` RGB24 ring buffer |
| Sensor samples | `...sensor.samples` (generation-qualified) | dsim -> clients | shared record ring: camera and LiDAR metadata, scalar range samples |
| LiDAR arrays | `...sensor.<id>.array` (generation-qualified) | dsim -> clients | dedicated record ring, one per array sensor |
| Command | `.control` | clients -> dsim | `memcmd` text queue |
| Status | `.status` | dsim -> clients | `memkv` key-value store |

### Commands

Commands are compact JSON objects:

```json
{"magic":"dvision2.command.v1","type":"velocity","source_id":"dway-area1","lease_id":"1f3c...","request_id":"9ab2...","forward_mps":0.8,"right_mps":0.0,"up_mps":0.0,"yaw_rate_dps":0.0}
```

The `magic` field is a version gate. Commands with the wrong magic or missing
type are ignored.

| Type | Fields | Effect |
|---|---|---|
| `acquire_control` | none | Claim the control lease; refused while another client holds it |
| `release_control` | none | Give the lease back |
| `heartbeat` | none | Renews the lease; does **not** refresh the setpoint timer |
| `arm` | `armed` | Arms/disarms; arming captures home, disarm zeroes motion |
| `takeoff` | `alt_m` | Climb to target altitude |
| `land` | none | Descend and disarm on touchdown; allowed without a lease |
| `rtl` | none | Climb to a safe height, return to home, then land |
| `zero`, `hold` | none | Clear every setpoint; enter HOLD if armed |
| `velocity` | `forward_mps`, `right_mps`, `up_mps`, `yaw_rate_dps` | Body-frame velocity setpoint |
| `position_target` | `frame` plus `x,y,z` or `north_m,east_m,down_m`, `heading_deg`, `max_speed_mps` | Position setpoint the simulator flies to |
| `set_origin` | `lat_deg`, `lon_deg`, `alt_m` | Move the geographic origin; disarmed only |
| `set_gps` | `mode`, `noise_m` | Deny or restore GPS mid-flight; simulation only |
| `set_estimator` | `attitude`, `local`, `global`, `velocity` | Fault or restore an estimator; simulation only |
| `reset` | none | Return the vehicle to its start pose |

### Control ownership

There is one active controller. `acquire_control` creates a lease; every
motion, mode and arming command must carry the current `source_id` and
`lease_id` or it is refused. Heartbeats from the owner renew the lease;
anything else does not. The lease expires after `--control-lease-timeout`
seconds (default 3), which puts an armed vehicle into `HOLD`. An emergency
`land` is accepted without a lease.

Every command carries a `request_id`, and its outcome is published in
`command.result.request_id` / `.accepted` / `.reason`. Queue admission is not
acceptance -- streamed setpoints wait for their result too.

Those three keys hold one latest value, and a whole command queue is drained
per frame against a single status publication: when two clients command in the
same frame, only the last outcome would survive there. `command.results`
carries the last 16 outcomes as `request_id accepted reason` lines (the reason
truncated to 96 characters), so a client can still find its own. A client
correlates against the slot first, for the untruncated reason, and falls back
to the history; a vehicle that publishes no history -- `dfgb` -- still works
for the single client commanding it.

`--setpoint-timeout` (default `2` seconds; `0` disables it) is the guided-mode
failsafe: an armed vehicle in `GUIDED` that stops receiving position or
velocity targets clears them, enters `HOLD`, and publishes
`failsafe.reason=setpoint_timeout`. `TAKEOFF`, `LAND`, `RTL`, `HOLD` and
`DISARMED` are not subject to it. A heartbeat renews the control lease and
deliberately does not refresh this timer, so a client that stops flying but
keeps saying hello still fails safe.

`dctl`, `daic` and `dway` all acquire a lease and send heartbeats while
connected.

### MAVLink mapping

The JSON stays honest by being mappable one-for-one onto MAVLink. Where a row
has no equivalent it is simulation-only, and a future bridge must not pretend
to translate it.

| dsim JSON | MAVLink equivalent | Notes |
|---|---|---|
| `arm` | `MAV_CMD_COMPONENT_ARM_DISARM` | |
| `takeoff` | `MAV_CMD_NAV_TAKEOFF` | altitude only |
| `land` | `MAV_CMD_NAV_LAND` | |
| `rtl` | `MAV_CMD_NAV_RETURN_TO_LAUNCH` | |
| `position_target` (`frame:"local_ned"`) | `SET_POSITION_TARGET_LOCAL_NED` | same axes, same units |
| `position_target` (`frame:"map"`) | -- | dvision2 convenience; converts to local NED |
| `velocity` | `SET_POSITION_TARGET_LOCAL_NED` with velocity mask, `MAV_FRAME_BODY_NED` | body frame |
| `heartbeat` | `HEARTBEAT` | |
| `set_origin` | `SET_GPS_GLOBAL_ORIGIN` | |
| `set_gps`, `set_estimator` | -- | simulation control, no vehicle equivalent |
| `acquire_control`, `release_control` | -- | dvision2 control lease |
| `reset` | -- | simulation control |
| status `drone.*` | `GLOBAL_POSITION_INT`, `LOCAL_POSITION_NED`, `ATTITUDE` | |
| status `gps.*` | `GPS_RAW_INT` | |
| status `est.*` | `ESTIMATOR_STATUS` | subset |
| status `vehicle.*` | `AUTOPILOT_VERSION` + `HEARTBEAT` capability flags | |
| status `wind.*`, `geofence.*`, `realism.*` | -- | simulation conditions, recorded in reports |

### Video

`dsim` writes RGB24 frames into a ring buffer. Clients read the newest slot by
sequence number. Client-side frame orientation is normalized before display
and vision processing so DAIC's video, target detector, and obstacle detector
operate on the same image orientation.

### Status

`dsim` writes telemetry to a key/value store after every physics tick. Clients
track status epochs and mark telemetry stale if no update arrives for more
than two seconds.

## Telemetry

Common status keys:

| Key | Description |
|---|---|
| `sim.id` | Instance id |
| `sim.map` | Loaded map path |
| `sim.time_s` | Simulated seconds since the run began -- time the vehicle experienced, not time that passed in the room |
| `sim.speed` | How simulated seconds map onto real ones: `1` in real time, the configured multiple under `--sim-speed`, `0` when unpaced |
| `sim.report_dir` | This run's report root; every module writes into its own subdirectory of it |
| `sim.camera_in_geometry` | `"1"` when the camera is inside a wall or tree, so a vision test can discard the frame |

Camera model, mount pose, and per-capture pose are no longer status keys. They
live in the sensor registry and the per-frame `camera.frame` record; see
[docs/modcom.md](docs/modcom.md) and
[docs/sensor/camera-rgb.md](docs/sensor/camera-rgb.md).
| `drone.armed` | `"1"` or `"0"` |
| `drone.mode` | `DISARMED`, `GUIDED`, `TAKEOFF`, `LAND`, `RTL`, `HOLD`, `CRASHED` |
| `drone.x_m`, `drone.y_m`, `drone.z_m` | Local position |
| `drone.lat_deg`, `drone.lon_deg`, `drone.alt_m` | GPS-equivalent position |
| `target.lat_deg`, `target.lon_deg`, `target.alt_m` | GPS-equivalent target |
| `drone.roll_deg`, `drone.pitch_deg`, `drone.heading_deg` | Attitude; roll positive right-wing-down, pitch positive nose-up |
| `drone.compass_deg` | Compass heading, 0 north and 90 east |
| `drone.vx_mps`, `drone.vy_mps`, `drone.vz_mps` | World-frame velocity |
| `drone.speed_mps` | Speed magnitude |
| `drone.battery_pct` | Simulated battery |
| `drone.crashed` | `"1"` after collision |
| `drone.last_command_s` | Seconds since last command |
| `link.command_count` | Commands received by simulator |
| `link.last_command_type` | Most recent command type |
| `status.message` | Human-readable simulator status |

Vehicle contract keys, which a client negotiates with rather than assumes:

| Key | Description |
|---|---|
| `vehicle.type` | `dsim`, or the autopilot behind a real link |
| `vehicle.frames` | Accepted position frames, comma separated |
| `vehicle.accepts_position`, `.accepts_velocity`, `.accepts_attitude` | Setpoint types accepted |
| `vehicle.supports_missions` | Onboard mission storage, `"0"` for `dsim` |
| `vehicle.setpoint_timeout_s` | Guided setpoint failsafe; empty when disabled |
| `vehicle.max_speed_mps`, `vehicle.max_accel_mps2` | Configured motion limits |
| `origin.lat_deg`, `origin.lon_deg`, `origin.alt_m` | Geographic origin at the map centre |
| `home.lat_deg`, `home.lon_deg`, `home.alt_m` | Pose captured at arming; RTL returns here |
| `control.owner` | `source_id` of the current lease holder, empty when free |
| `control.lease_age_s`, `control.lease_timeout_s` | Lease freshness and its limit |
| `setpoint.age_s` | Seconds since the last position or velocity target |
| `failsafe.reason` | `setpoint_timeout`, `control_lease_expired`, `geofence`, `battery_low`, or empty |
| `command.result.request_id`, `.accepted`, `.reason` | Outcome of the most recent command |
| `command.results` | The last 16 outcomes, newest last, as `request_id accepted reason` lines |

Capabilities are static interfaces and configured limits. Freshness, ownership
and failsafe state are live vehicle state, and the two are deliberately kept
apart.

Sensor health and environment keys, which say what a run was flown in:

| Key | Description |
|---|---|
| `gps.fix_type` | `0` none, `2` 2D, `3` 3D, `4` RTK, following MAVLink's `GPS_FIX_TYPE` |
| `gps.satellites`, `gps.hdop`, `gps.vdop` | Fix quality; the published lat/lon/alt carry the mode's noise |
| `est.attitude_valid`, `.local_position_valid`, `.global_position_valid`, `.velocity_valid` | Estimator validity, which arming alone never confers |
| `wind.speed_mps`, `wind.dir_deg`, `wind.gust_mps` | Steady wind, the direction it blows from, and gust magnitude |
| `geofence.box`, `geofence.action` | Configured boundary and what crossing it does |
| `realism.telemetry_latency_ms`, `.telemetry_jitter_ms` | Delay applied to published status |
| `realism.sensor_noise` | Noise profile name |
| `realism.battery_failsafe_pct`, `.battery_drain_pct_s` | Battery failsafe threshold and drain rate |
| `realism.seed` | Seed every random process was built from |

`dway` copies these into its report's `conditions` block at preflight, so a run
flown in wind or through a degraded fix is never mistaken for a clean one.

## Rendering and Assets

The renderer builds a simple 3D scene:

- tiled ground plane using the Ground037 texture
- brick-textured wall boxes 2.5 m tall, using the Bricks042 texture
- tree GLB models selected deterministically per map position
- primitive trunk/crown fallback trees
- flat red target marker at ground level
- forward-facing drone camera with 70 degree horizontal FOV, near 0.15 m, far
  150 m, and 5 degree downward pitch

`--scene-preset` picks the appearance:

| Preset | What it renders |
|---|---|
| `representative` (default) | The `panda3d-simplepbr` pipeline with shadow mapping |
| `legacy` | The original fixed-function lighting and fog |

A preset changes appearance only. The geometry is identical across presets,
which is what makes a lighting change safe to measure against unchanged truth:
the exact-range oracle casts through the same world either way. `apps/dsim/scene.py`
carries a version string per preset, and anything recording which scene a
result came from should record the *version* rather than the preset name, so a
renderer change shows up in the record instead of hiding behind a stable label.

Asset sources are documented in `SOURCE.md` files beside the imported assets.

## Development Notes

Physics are intentionally approximate. There is no full multirotor model,
gravity or propeller simulation. Horizontal velocity follows body-frame
forward/right setpoints through a first-order lag. Vertical velocity follows
`up_mps` directly.

Wind is simulated, but as an environment rather than as aerodynamics: it moves
the airframe over the ground without acting on attitude, so the vehicle's own
velocity is through the air and the published velocity is over the ground. That
difference is the whole point -- it is what a position controller has to notice
and correct -- and it is enough to tell a controller that works from one that
only looked like it worked. It is not a model of how a real airframe reacts to
a gust.

Velocity command values in logs are SI setpoints. Status velocity keys show the
actual simulated world-frame response after lag and collision handling.

**Simulated time is the vehicle's clock.** `sim.time_s` counts the seconds the
physics advanced, and every timer that gates flight reads it rather than the
wall clock: the guided setpoint failsafe, the control lease and telemetry
latency in `dsim`; the whole mission clock in `dway` -- dwell, arrival gates,
leg timeouts, the setpoint stream; the planner's state timers in `daic`; the
frame capture rate in `dalg`. A failsafe that fires because the machine was
busy rather than because the vehicle flew for two seconds is measuring the
wrong thing.

`dt` is fixed at `1/physics_hz` (300 Hz in the default drone profile) in
every mode. Real time paces those fixed steps against the wall clock and
never enlarges a step to catch up: a process that stalls falls behind and
reports the shortfall through `sim.speed_achieved`, because the physics did
not run. A scaled run paces the identical steps at `step / multiplier`, which
is what makes a scaled run repeatable.

Only two things may read the wall clock: liveness between processes
(heartbeats, expiry, acknowledgement deadlines) and how often a window
repaints. [`docs/clock.md`](docs/clock.md) is the full contract, including the
failure modes that follow from getting the split wrong.

A client that infers a distance from motion needs the same clock. `daic`'s
optical-flow detector turns expansion into a range using `speed x elapsed`, and
it takes both halves from the vehicle: taking the elapsed half from the wall
clock made every range estimate a function of how busy the machine was.

GPS values are derived from local XY using a flat-earth projection around the
configured map origin. This is good enough for short local maps and gives DAIC
a status-only target bearing without exposing simulator map geometry.

DAIC should be debugged in layers:

1. Check video orientation and target detection.
2. Check fused obstacle sector risks.
3. Check sector range estimates.
4. Check local map occupied/free cells.
5. Check the planned path and waypoint command.
6. Check the final avoidance brake and sent command.

The most useful command pair for obstacle-navigation debugging is:

```sh
python3 tests/flight_test.py --map assets/maps/maze_002.txt --duration 20 --fps 20 --log /tmp/maze002.jsonl
python3 tests/vision_debug_report.py /tmp/maze002.jsonl
```

## Comparison to Similar Projects

Several projects overlap with parts of `dvision2`, and none overlaps with all
of it: `dsim` is a simulator, `dctl` is an operator window, `dway` is an
autopilot client and `dalg` is a scoring harness, and the honest comparison is
different for each. The short version is that `dvision2` is not competing with
the flight-stack simulators. It is a fast, legible stand-in for a vehicle,
built for developing the *client* -- the perception or navigation code that
consumes video and telemetry -- and it gives up real flight dynamics and real
MAVLink to be that.

### PX4 and ArduPilot SITL

[PX4](https://github.com/PX4/PX4-Autopilot) and
[ArduPilot](https://github.com/ArduPilot/ardupilot) compile the actual
autopilot firmware for the host and fly a simulated airframe under it, paired
with Gazebo, jMAVSim or ArduPilot's own built-in physics, and driven by
QGroundControl or Mission Planner. This is the production stack: the same
firmware runs on the real vehicle.

**Key differences:**

- The autopilot is real. `dsim` is not one -- it accepts position and velocity
  setpoints and integrates them, with no attitude controller, no mixer and no
  estimator underneath. Anything you learn about control-law behaviour in
  `dsim` says nothing about the real thing.
- MAVLink is the protocol, and the entire ground-station, companion-computer
  and log ecosystem speaks it. `dvision2` speaks JSON over `pymembus`, shaped
  so every message maps onto a MAVLink one, but nothing here has been tested
  against a real autopilot.
- SITL has hardware-in-the-loop, mission upload and `AUTO`, failsafe suites,
  parameter systems, log analysis and a very large community. `dvision2` has
  none of that ecosystem.
- PX4's lockstep SITL steps the simulator and the autopilot together.
  `dvision2` reaches a similar place from the other direction: simulated time
  is the vehicle's clock everywhere, and a conformance test flies one tour at
  two speeds and compares the two reports.
- A SITL toolchain plus Gazebo is a substantially larger thing to install and
  keep working than five pip packages.

**Choose PX4 or ArduPilot SITL if** anything you are building will eventually
fly on a real vehicle, if you need MAVLink compatibility, or if the control
stack itself is what you are working on.

**Choose dvision2 if** you are developing a client rather than a vehicle, and
you want the vehicle to be a cheap, readable stand-in whose GPS quality,
estimator validity, wind, telemetry latency and sensor noise you can change
while it is flying.

### AirSim and Colosseum

[AirSim](https://github.com/microsoft/AirSim) was the sensor-rich,
Unreal-rendered drone and car simulator with an RPC client API. Microsoft
archived it in 2022; [Colosseum](https://github.com/CodexLabsLLC/Colosseum) is
the community fork that carried it to Unreal Engine 5.

**Key differences:**

- Photorealistic Unreal rendering against a deliberately plain Panda3D test
  scene. If an algorithm's performance depends on image realism, this is the
  difference that matters, and `dvision2` will mislead you.
- Full multirotor dynamics, with an optional PX4 hardware- or
  software-in-the-loop link.
- Sensors are reached through an RPC API rather than shared memory, and there
  is no live multi-pane device window -- inspection is whatever you write
  against the API yourself.
- Unreal and a capable GPU, against a laptop and `pip install`.
- Upstream is archived. Colosseum is active, but with a smaller community than
  AirSim had.

**Choose AirSim or Colosseum if** you need photorealistic imagery, a real
multirotor model, or environments authored in Unreal.

**Choose dvision2 if** iteration speed and legibility matter more than image
realism, and you want the whole vehicle to be Python you can read in an
afternoon.

### Flightmare

[Flightmare](https://github.com/uzh-rpg/flightmare) (UZH Robotics and
Perception Group) separates Unity rendering from a configurable dynamics layer
specifically so that vision and reinforcement-learning experiments can run far
faster than real time.

**Key differences:**

- Flightmare and `dvision2` start from the same observation: rendering
  dominates the cost of a tick, so it has to be separable from the physics.
  Flightmare makes rendering optional per experiment; `dvision2` drops the
  camera profile rate while the physics rate stays fixed, which is the
  difference between roughly 12x and 2.9x under `--sim-speed max`.
- Flightmare has real quadrotor dynamics and a PX4 path. `dvision2` has
  neither.
- Flightmare targets RL, with parallel environments and gym interfaces.
  `dvision2` has no RL surface at all.
- Unity and a catkin-style build, against pip.
- Development has been quiet for a while; check current activity before
  adopting it.

**Choose Flightmare if** you are doing RL or agile-flight research and need
speed together with dynamics you can trust.

**Choose dvision2 if** you want the same speed argument without Unity, and you
care more about a repeatable operator-facing scenario than about a training
loop.

### gym-pybullet-drones

[gym-pybullet-drones](https://github.com/utiasDSL/gym-pybullet-drones) (UTIAS
Dynamic Systems Lab) is a small, pure-Python multi-quadrotor simulator on
PyBullet, with Gymnasium interfaces and control baselines.

**Key differences:**

- This is the closest match anywhere to `dvision2`'s "small enough to read"
  premise, and it is the better-founded of the two: real quadrotor dynamics
  down to individual motor speeds, where `dvision2` has a first-order velocity
  lag and nothing underneath it.
- Its camera is a PyBullet viewport, not a sensor pipeline. There is no
  discovery registry, no per-sensor rings, and no LiDAR, rangefinder or GNSS
  models carrying their own rates, mounts and noise.
- No operator window, no environment realism knobs, no ground-truth scoring
  harness.
- Gymnasium interfaces out of the box, which `dvision2` does not have.

**Choose gym-pybullet-drones if** you are studying control or training a
policy, and the dynamics are the thing that has to be right.

**Choose dvision2 if** you are developing perception or navigation against a
many-sensor vehicle, and the controller is explicitly not what you are
studying.

### Webots

[Webots](https://github.com/cyberbotics/webots) (Cyberbotics, Apache 2.0) is a
batteries-included robot simulator with drone models, a large device API, a
scene editor and ROS/ROS 2 bridges.

**Key differences:**

- Webots' device API is the closest mainstream analogue to `dvision2`'s sensor
  manifest: a robot description declares its devices and code enumerates them.
  Webots is a general robot simulator first, though, and its drone and
  autopilot story is thinner than PX4's.
- A real physics engine, a scene editor and a large model library, none of
  which `dvision2` has or wants.
- Device inspection in Webots is per device, in the IDE. `dvision2`'s Devices
  tab is a tiled bench you arrange, freeze across every pane at one shared
  capture id, and export from.
- Much larger to install, and much more to learn.

**Choose Webots if** you want a general robot simulator with real physics, a
scene editor and ROS integration.

**Choose dvision2 if** you want a drone-shaped scenario with environment
realism knobs and no simulator to learn.

### Isaac Sim and the Pegasus Simulator

[Pegasus Simulator](https://github.com/PegasusSimulator/PegasusSimulator) is an
open-source extension adding multirotor vehicles and PX4 integration to
NVIDIA's Isaac Sim, with Isaac Lab covering reinforcement learning.

**Key differences:**

- Best-in-class rendering and sensor simulation -- ray-traced cameras, RTX
  LiDAR -- and massively parallel environments for training.
- Full PX4 integration, so it belongs with the flight-stack simulators above
  rather than beside `dvision2`.
- Requires an RTX GPU and a large NVIDIA runtime. Pegasus itself is open
  source; Isaac Sim's own licensing has changed over its life, so check the
  current terms.
- The setup-cost gap on this page is widest here.

**Choose Isaac Sim and Pegasus if** you have the hardware and need
photorealism, RTX sensor models, or parallel RL at scale.

**Choose dvision2 if** you want something that starts on any laptop, in a
terminal, in seconds.

### Rerun, PlotJuggler and Foxglove

[Rerun](https://github.com/rerun-io/rerun),
[PlotJuggler](https://github.com/facontidavide/PlotJuggler) and
[Foxglove](https://foxglove.dev/) are the inspection layer. They compare to
`dctl`'s Devices and Events tabs rather than to `dsim`, and they are
complements rather than alternatives -- you would use one *with* PX4 and
Gazebo.

**Key differences:**

- Rerun is the closest thing anywhere to the Devices tab: multi-pane
  per-stream viewers, images and scalars together on one timeline. It is also
  far better at time travel than `dvision2`'s freeze, which holds a capture
  rather than scrubbing a recording.
- All three are general viewers you log *into*. `dvision2`'s Devices tab is
  the inverse: it discovers what the vehicle publishes and picks a renderer
  per sensor type, with no logging calls anywhere in the producer.
- None of them is a simulator, and none will fly the vehicle in the same
  window.
- PlotJuggler is the strongest pure time-series plotter of the three. Foxglove
  is open core, and its desktop application's licence has changed over time --
  check the current terms before depending on it.

**Choose one of these if** you already have a stack producing data and want to
look at it properly, or you need recording and scrubbing rather than a live
hold.

**Choose dvision2's Devices tab if** you want zero-instrumentation inspection
of whatever a vehicle publishes, in the same window that flies it.

### Summary

| | dvision2 | PX4/ArduPilot SITL | AirSim/Colosseum | Flightmare | gym-pybullet-drones | Webots | Isaac + Pegasus |
|---|---|---|---|---|---|---|---|
| Flight dynamics | First-order velocity lag | Real autopilot | Full multirotor | Full quadrotor | Full quadrotor | Physics engine | Full multirotor |
| Rendering | Panda3D test scene | Via Gazebo | Unreal | Unity | PyBullet viewport | Built in | Omniverse RTX |
| MAVLink / real autopilot | No | Yes | Yes (PX4) | Yes (PX4) | No | Via PX4 bridge | Yes (PX4) |
| ROS / ROS 2 | No | Yes | Yes | Yes | Optional | Yes | Yes |
| Simulated time as the vehicle's clock | Enforced by tests | Lockstep available | Partial | Yes | Yes | Yes | Yes |
| Live multi-sensor operator window | Yes, manifest-driven | MAVLink inspector only | No | No | No | Per device, in the IDE | Viewport and sensor views |
| Ground-truth scoring harness | Yes (`dalg`) | No | No | No | RL rewards | No | Isaac Lab (RL) |
| Environment realism, changeable in flight | Yes | Partly, via parameters | Some | Limited | No | Some | Some |
| Install footprint | pip, no ROS | Toolchain + Gazebo | Unreal Engine | Unity + build | pip | One package | RTX GPU + Omniverse |
| Community | Small | Very large | Large, upstream archived | Small | Moderate | Large | Growing |

If the goal is a vehicle that will fly, start with PX4 or ArduPilot SITL and
treat this project as a curiosity. `dvision2` is worth a look when the vehicle
is not the subject: when you want a sensor-rich, scriptable stand-in that
starts instantly, whose environment you can degrade on purpose while it flies,
and whose entire surface you can read.
