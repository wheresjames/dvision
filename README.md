# dvision2

`dvision2` is a local drone simulation and autonomy development platform. It
contains:

- `dsim`: a Python/Panda3D drone simulator
- `dctl`: a manual keyboard/gamepad controller
- `daic`: a vision-driven autonomy controller

The processes communicate through shared-memory video, command, and status
buffers. The current autonomy work is intentionally centered on what a real
camera client would have: the live video stream and the status/telemetry
buffer. `daic` must not use simulator map data for navigation decisions.

The project is meant for fast local iteration. The simulator is small enough
to read, the command protocol is JSON over `pymembus`, and the autonomy stack
is split into detector, planner, avoidance, local mapping, and control layers
so each piece can be tested or replaced independently.

## Contents

- [Project Layout](#project-layout)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Simulator: dsim](#simulator-dsim)
- [Manual Controller: dctl](#manual-controller-dctl)
- [AI Controller: daic](#ai-controller-daic)
- [Vision Navigation](#vision-navigation)
- [Automated Testing and Diagnostics](#automated-testing-and-diagnostics)
- [Maps](#maps)
- [Shared Memory Protocol](#shared-memory-protocol)
- [Telemetry](#telemetry)
- [Rendering and Assets](#rendering-and-assets)
- [Development Notes](#development-notes)
- [Known Limitations](#known-limitations)
- [Work in Progress](#work-in-progress)

## Project Layout

```text
dvision2_common.py          Shared protocol, map loading, ids, status keys
OVERVIEW.md                 Architecture and API reference

dsim/
  dsim.py                   Simulator: physics, rendering, IPC server, UI
  assets/
    maps/                   Text map files
    textures/               CC0 ground/wall textures
    models/trees/           CC0 tree GLB models

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

tests/
  flight_test.py            End-to-end headless flight runner
  vision_debug_report.py    Vision/map diagnostic summary from flight logs
  test_daic_*.py            DAIC detector, planner, map, avoidance tests
  test_dctl_controls.py     Manual control tests
  test_dsim_crash_reset.py  Simulator collision/reset tests
```

## Architecture

`dsim` owns the simulated world. It creates the shared-memory buffers, runs the
physics loop, renders the drone camera feed, checks collisions, and publishes
status telemetry.

`dctl` and `daic` are clients. They connect to the same buffers, display video
and telemetry, and send commands. Clients retry missing buffers continuously,
so they can be started before or after `dsim`.

All processes share an instance id such as `area1`. Buffer names are derived
from that id:

```text
/dvision2.area1.video     RGB24 video   dsim -> clients
/dvision2.area1.control   JSON commands clients -> dsim
/dvision2.area1.status    Telemetry k/v dsim -> clients
```

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

Optional:

```text
pygame             gamepad/joystick input for dctl
ORB_SLAM3 binding  optional full SLAM obstacle detection for daic
```

Install the common dependencies:

```sh
pip install numpy Pillow pymembus opencv-python panda3d pygame
```

`daic` also has an installer/check mode:

```sh
python3 daic/daic.py --install
```

That mode checks the OpenCV features needed by the optical-flow and mini-SLAM
paths and reports optional ORB_SLAM3 setup status.

## Quick Start

Manual control:

```sh
# Terminal 1
python3 dsim/dsim.py --id area1

# Terminal 2
python3 dctl/dctl.py --id area1
```

Autonomous flight with DAIC:

```sh
# Terminal 1
python3 dsim/dsim.py --id area1 --map dsim/assets/maps/maze_002.txt

# Terminal 2
python3 daic/daic.py --id area1 --enable-ai
```

Headless automated run:

```sh
python3 tests/flight_test.py --map dsim/assets/maps/maze_002.txt --duration 20 --fps 20
```

Headless run with a vision diagnostic log:

```sh
python3 tests/flight_test.py --map dsim/assets/maps/maze_002.txt --duration 20 --fps 20 --log /tmp/maze002.jsonl
python3 tests/vision_debug_report.py /tmp/maze002.jsonl
```

## Simulator: dsim

![dsim top-down monitor showing the maze map, drone position, and heading](images/dsim-001.png)

`dsim` creates the world, renders the camera image, accepts control commands,
and publishes telemetry after every tick.

```sh
python3 dsim/dsim.py --id area1 \
  --map dsim/assets/maps/maze_001.txt \
  --width 640 \
  --height 480 \
  --fps 30
```

Options:

| Option | Description |
|---|---|
| `--id` | Required instance id |
| `--map` | Map file to load |
| `--width`, `--height` | Rendered video frame size |
| `--fps` | Simulation and video update rate |
| `--bufs` | Video ring-buffer slot count |
| `--cmd-size` | Command buffer size in bytes |
| `--start-alt` | Override initial altitude; otherwise map `drone-height` or `1.5` |
| `--origin-lat/lon/alt` | GPS coordinate for the map center |
| `--frames` | Stop after N frames, useful for tests |
| `--no-ui` | Disable the top-down simulator monitor |
| `--verbose` | Print runtime diagnostics |

The simulator uses a simple first-order response model:

| Axis | Time constant |
|---|---|
| Horizontal forward/right | 0.30 s |
| Vertical up/down | 0.35 s |
| Yaw rate | 0.10 s |
| Visual roll/pitch | 0.14 s |

`forward_mps` and `right_mps` commands are pre-scaled: `dsim` multiplies them
by `_SPEED_SCALE = 0.1`. A command of `forward_mps=10.0` produces roughly
`1.0 m/s` actual forward speed after the first-order response catches up.
`up_mps` and `yaw_rate_dps` are not scaled.

The simulator UI shows the map, drone position, heading, armed/mode state,
current command, and a reset button. Collision is cell based: wall and tree
objects occupy their full map cell, and a crash puts the drone in `CRASHED`
until reset.

## Manual Controller: dctl

![dctl showing the camera feed, controls, and keyboard legend](images/dctl-001.png)

`dctl` displays the camera feed and sends manual velocity, arm, takeoff, land,
and zero commands.

```sh
python3 dctl/dctl.py --id area1 \
  --width 960 \
  --height 720 \
  --fps 30
```

Options:

| Option | Description |
|---|---|
| `--id` | Required instance id |
| `--width`, `--height` | Maximum displayed video size |
| `--fps` | UI refresh rate |
| `--cmd-size` | Command buffer size |
| `--speed` | Horizontal speed sent to dsim, pre-scaled |
| `--vertical-speed` | Vertical speed sent to dsim |
| `--no-joystick` | Disable gamepad polling |
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

## AI Controller: daic

![daic showing the camera feed with obstacle overlays, SLAM map, and local route](images/daic-001.png)

`daic` is the autonomous client. It reads only the video and status buffers,
then sends commands through the command buffer. It can run with a Tk UI or in
headless mode for automated testing.

```sh
# UI, manual AI toggle
python3 daic/daic.py --id area1

# UI, autonomy enabled immediately
python3 daic/daic.py --id area1 --enable-ai

# Headless
python3 daic/daic.py --id area1 --enable-ai --no-ui

# Log every control tick
python3 daic/daic.py --id area1 --enable-ai --log-file /tmp/flight.jsonl
```

Options:

| Option | Description |
|---|---|
| `--install` | Check/install DAIC vision dependencies, then exit |
| `--id` | Instance id; required unless `--install` is used |
| `--display-w`, `--display-h` | Maximum displayed video size, `0` for native |
| `--video-w`, `--video-h` | Expected frame size for detector/servo gains |
| `--fps` | UI/control loop refresh rate |
| `--cmd-size` | Command buffer size |
| `--enable-ai` | Enable AI immediately on startup |
| `--no-ui` | Headless mode |
| `--log-file` | Write structured JSONL flight log |
| `--slam-vocab` | Path to `ORBvoc.txt`; enables ORB_SLAM3 obstacle detection |
| `--slam-settings` | Optional ORB_SLAM3 YAML; generated from camera status when omitted |
| `--verbose` | Print planner state each tick |

### UI

The DAIC UI is arranged around the live camera feed. The video is displayed
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
controller take over. The target controller uses a two-phase strategy: keep
forward motion while the target is small/far, then descend only once the target
is large enough and below the camera center.

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

`daic/optical_flow_avoidance.py` computes dense Farneback optical flow between
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

`daic/local_map.py` maintains a rolling occupancy grid around the drone. It:

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

`daic/avoidance.py` is the last safety layer before sending velocity commands.
It does not inject lateral movement or yaw. It only trims forward speed when
front-sector risk is high. That keeps steering under the route planner while
still reducing forward motion into detected obstacles.

## Automated Testing and Diagnostics

Run all tests:

```sh
pytest -q
```

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
  --map dsim/assets/maps/maze_002.txt \
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
  "cmd": {"type": "velocity", "forward_mps": 4.5, "yaw_rate_dps": 0.0},
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
| `maze_012.txt`, `maze_013.txt`, `maze_014.txt` | Additional layouts |
| `test_direct.txt` | Open field, no obstacles |

## Shared Memory Protocol

All buffers are provided by `pymembus` and named as:

```text
/dvision2.<id>.<channel>
```

| Buffer | Channel | Direction | Type |
|---|---|---|---|
| Video | `.video` | dsim -> clients | `memvid` RGB24 ring buffer |
| Command | `.control` | clients -> dsim | `memcmd` text queue |
| Status | `.status` | dsim -> clients | `memkv` key-value store |

### Commands

Commands are compact JSON objects:

```json
{"magic":"dvision2.command.v1","type":"velocity","forward_mps":8.0,"right_mps":0.0,"up_mps":0.0,"yaw_rate_dps":0.0}
```

The `magic` field is a version gate. Commands with the wrong magic or missing
type are ignored.

| Type | Fields | Effect |
|---|---|---|
| `heartbeat` | none | Keeps link alive |
| `arm` | `armed` | Arms/disarms; disarm zeroes motion |
| `takeoff` | `alt_m` | Climb to target altitude |
| `land` | none | Descend and disarm on touchdown |
| `zero` | none | Clear velocity setpoints; enter HOLD if armed |
| `velocity` | `forward_mps`, `right_mps`, `up_mps`, `yaw_rate_dps` | Body-frame velocity setpoint |

`dctl` and `daic` both send heartbeats while connected.

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
| `sim.time_s` | Elapsed simulation time |
| `camera.width_px`, `camera.height_px` | Video dimensions |
| `camera.fx_px`, `camera.fy_px` | Camera intrinsics |
| `camera.fov_h_deg`, `camera.fov_v_deg` | Camera FOV |
| `camera.pitch_deg` | Camera pitch |
| `camera.fps` | Camera FPS |
| `drone.armed` | `"1"` or `"0"` |
| `drone.mode` | `DISARMED`, `GUIDED`, `TAKEOFF`, `LAND`, `HOLD`, `CRASHED` |
| `drone.x_m`, `drone.y_m`, `drone.z_m` | Local position |
| `drone.lat_deg`, `drone.lon_deg`, `drone.alt_m` | GPS-equivalent position |
| `target.lat_deg`, `target.lon_deg`, `target.alt_m` | GPS-equivalent target |
| `drone.roll_deg`, `drone.pitch_deg`, `drone.heading_deg` | Attitude |
| `drone.compass_deg` | Compass heading, 0 north and 90 east |
| `drone.vx_mps`, `drone.vy_mps`, `drone.vz_mps` | World-frame velocity |
| `drone.speed_mps` | Speed magnitude |
| `drone.battery_pct` | Simulated battery |
| `drone.crashed` | `"1"` after collision |
| `drone.last_command_s` | Seconds since last command |
| `link.command_count` | Commands received by simulator |
| `link.last_command_type` | Most recent command type |
| `status.message` | Human-readable simulator status |

## Rendering and Assets

The renderer builds a simple 3D scene:

- tiled ground plane using the Ground037 texture
- brick-textured wall boxes using the Bricks042 texture
- tree GLB models selected deterministically per map position
- primitive trunk/crown fallback trees
- flat red target marker at ground level
- forward-facing drone camera with 70 degree horizontal FOV, near 0.15 m, far
  150 m, and 5 degree downward pitch

Asset sources are documented in `SOURCE.md` files beside the imported assets.

## Development Notes

Physics are intentionally approximate. There is no full multirotor model,
gravity, wind, or propeller simulation. Horizontal velocity follows body-frame
forward/right setpoints through a first-order lag. Vertical velocity follows
`up_mps` directly.

Velocity command values in logs are command units, not always actual m/s.
Remember that horizontal values are pre-scaled by `dsim`; status velocity keys
show actual simulated world-frame speed.

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
python3 tests/flight_test.py --map dsim/assets/maps/maze_002.txt --duration 20 --fps 20 --log /tmp/maze002.jsonl
python3 tests/vision_debug_report.py /tmp/maze002.jsonl
```

## Known Limitations

- DAIC navigation is still experimental. It can build a local map from vision,
  but route quality in mazes is still being tuned.
- Monocular optical-flow range is approximate. It depends on forward motion,
  texture, camera geometry, and filtering. It is useful for local planning but
  should not be treated as metric depth with sensor-grade accuracy.
- Mini-SLAM and optical flow can produce intermittent obstacle detections on
  low-texture surfaces or during rapid yaw.
- ORB_SLAM3 support is optional and depends on external native bindings and a
  vocabulary file.
- The command protocol is project JSON over `pymembus`, not MAVLink.
- Physics and collision are simplified.
- The simulator world is a local test scene, not a photorealistic environment.
- Asset licensing depends on files under `dsim/assets`; keep `SOURCE.md` notes
  beside imported assets.

## Work in Progress

`dfgb`, the FlightGear bridge, is present as a work in progress. It is not the
main path described above; treat `dsim`, `dctl`, and `daic` as the primary
development loop for now.
