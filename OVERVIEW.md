# dvision2 Overview

## Goal

dvision2 is a local drone simulation and control development platform. The aim is a tight feedback loop for experimenting with drone behavior, camera rendering, and control logic without real hardware. The simulator is small enough to read and modify, the wire protocol is simple enough to replace or extend, and both processes can be started independently.

## Architecture

Two independent processes communicate through shared memory:

```
  ┌────────────────────────────────────────────────────────────────┐
  │                         dsim                                   │
  │  loads map · runs physics · renders camera · publishes state   │
  └───────┬───────────────────────────────────────┬────────────────┘
          │  video (RGB24 frames)                 │  status (k/v telemetry)
          │  dsim → dctl                          │  dsim → dctl
          ▼                                       ▼
  ┌────────────────────────────────────────────────────────────────┐
  │                         dctl                                   │
  │  displays video · shows telemetry · sends commands             │
  └───────────────────────────────────┬────────────────────────────┘
                                      │  control (JSON commands)
                                      │  dctl → dsim
                                      ▼
                                    dsim
```

**dsim** is the owner. It creates all three shared-memory buffers on startup and tears them down on exit.

**dctl** is the client. It opens the existing buffers and retries in a polling loop, so it can be started before or after dsim.

Both processes are identified by a shared instance id (e.g. `area1`). All buffer names are derived from that id.

## Shared-Memory Buffers

Buffer names follow the pattern `/dvision2.<id>.<channel>`:

| Buffer | Name | Direction | Type |
|---|---|---|---|
| video | `/dvision2.<id>.video` | dsim → dctl | `memvid` RGB24 ring buffer |
| command | `/dvision2.<id>.control` | dctl → dsim | `memcmd` text queue |
| status | `/dvision2.<id>.status` | dsim → dctl | `memkv` key-value store |

All three transports are provided by the `pymembus` library.

### Video Buffer (`memvid`)

dsim writes RGB24 frames into a ring buffer at the configured fps and resolution (default 640×480 @ 30 fps). Each slot gets a monotonic microsecond timestamp. dctl polls for new sequence numbers and reads the latest slot. Frames are vertically flipped on the dctl side before display.

### Command Buffer (`memcmd`)

dctl writes JSON strings into a command queue. dsim drains the entire queue each physics tick. If a command overruns the buffer, dsim records the overrun in the status message and discards that command.

### Status Buffer (`memkv`)

dsim writes all telemetry to a named key-value store after every physics tick. dctl polls for epoch changes to detect updates. If no update arrives within two seconds, dctl marks status as stale in the connection header.

## Command Protocol

Every command is a JSON object with two required fields:

```json
{"magic": "dvision2.command.v1", "type": "<command_type>", ...}
```

The magic string `dvision2.command.v1` acts as a version gate. Commands with the wrong magic or a missing `type` are silently discarded.

### Command Types

| Type | Key Fields | Effect |
|---|---|---|
| `acquire_control` | — | claims the control lease; refused while another client holds it |
| `release_control` | — | hands the lease back |
| `heartbeat` | — | renews the lease; does **not** refresh the setpoint failsafe timer |
| `arm` | `armed: bool` | arms or disarms the drone; arming captures home, disarm zeroes all motion |
| `takeoff` | `alt_m: float` | sets a target altitude and transitions to TAKEOFF mode |
| `land` | — | targets altitude 0, clears setpoints, transitions to LAND |
| `rtl` | — | climbs to a safe height, returns to home, then lands |
| `zero`, `hold` | — | clears every setpoint; enters HOLD if armed |
| `velocity` | `forward_mps`, `right_mps`, `up_mps`, `yaw_rate_dps` | body-frame velocity setpoint |
| `position_target` | `frame` plus `x,y,z` (map) or `north_m,east_m,down_m` (local NED), `heading_deg`, `max_speed_mps` | position setpoint the vehicle flies to |
| `set_origin` | `lat_deg`, `lon_deg`, `alt_m` | moves the geographic origin; disarmed only |
| `set_gps` | `mode`, `noise_m` | denies or restores GPS; simulation only |
| `set_estimator` | `attitude`, `local`, `global`, `velocity` | faults or restores an estimator; simulation only |
| `reset` | — | returns the vehicle to its start pose |

A position and a velocity target replace one another, and either is accepted
only while armed and in `GUIDED` or `HOLD`; accepting one moves `HOLD` back to
`GUIDED`. `LAND`, `RTL` and `HOLD` clear whichever is current.

**Control ownership.** There is one active controller. Every motion, mode and
arming command carries `source_id`, `lease_id` and a unique `request_id`, and
is refused without the current lease; an emergency `land` is the exception.
The lease expires after `--control-lease-timeout` seconds without a heartbeat
from its owner, which puts an armed vehicle into `HOLD`. Each command's outcome
is published in `command.result.request_id` / `.accepted` / `.reason`, so queue
admission is never mistaken for acceptance.

**Setpoint failsafe.** An armed vehicle in `GUIDED` that stops receiving
targets for `--setpoint-timeout` seconds (default 2) clears them, enters
`HOLD` and publishes `failsafe.reason=setpoint_timeout`.

dctl acquires the lease on connect and sends a `heartbeat` roughly once per second. It sends `velocity` commands on every tick while any movement key or joystick axis is active, and sends a final `zero` when all inputs return to neutral. `dway` streams position or velocity targets at 10 Hz and heartbeats separately at 1 Hz.

### Encoding

Commands are produced by `encode_command()` in `dvision2_common.py`:

```python
json.dumps({"magic": COMMAND_MAGIC, "type": typ, **fields}, separators=(",", ":"), sort_keys=True)
```

The output is a compact sorted JSON string. dsim decodes with `decode_command()` which validates the magic and type fields before passing the payload to `apply_command()`.

## Telemetry Keys

dsim publishes the following keys to the status buffer after every tick:

| Key | Description |
|---|---|
| `sim.id` | Instance id |
| `sim.map` | Loaded map file path |
| `sim.time_s` | Elapsed simulation time |
| `drone.armed` | `"1"` / `"0"` |
| `drone.mode` | `DISARMED`, `GUIDED`, `TAKEOFF`, `LAND`, `RTL`, `HOLD`, `CRASHED` |
| `drone.x_m`, `drone.y_m`, `drone.z_m` | Local position in meters |
| `drone.lat_deg`, `drone.lon_deg`, `drone.alt_m` | Live GPS-equivalent position |
| `target.lat_deg`, `target.lon_deg`, `target.alt_m` | GPS-equivalent target position |
| `drone.roll_deg`, `drone.pitch_deg`, `drone.heading_deg` | Attitude |
| `drone.vx_mps`, `drone.vy_mps`, `drone.vz_mps`, `drone.speed_mps` | Velocity |
| `drone.battery_pct` | Simulated battery level |
| `drone.last_command_s` | Seconds since last command received |
| `link.command_count` | Total commands received |
| `link.last_command_type` | Most recently processed command type |
| `status.message` | Human-readable status (`"ok"`, `"command overrun"`, etc.) |
| `vehicle.*` | Negotiated capability profile: type, frames, accepted setpoint types, mission support, setpoint timeout, speed and acceleration limits |
| `origin.*`, `home.*` | Geographic origin at the map centre, and the pose captured at arming |
| `control.owner`, `control.lease_age_s`, `control.lease_timeout_s` | Who holds control and how fresh the lease is |
| `setpoint.age_s` | Seconds since the last position or velocity target |
| `failsafe.reason` | `setpoint_timeout`, `control_lease_expired`, `geofence`, `battery_low`, or empty |
| `command.result.*` | Request id, acceptance and reason for the most recent command |
| `gps.*`, `est.*` | Fix quality and estimator validity |
| `wind.*`, `geofence.*`, `realism.*` | The simulated conditions a run was flown in |

`dvision2_common.STATUS_KEYS` is the fixed schema; the README's Telemetry section describes each key.

GPS coordinates are derived from local XY position using a simple flat-earth projection. For `dsim`, the configured `--origin-lat/lon/alt` is the map center and defaults to Berlin, Germany.

## Physics Model

dsim runs a first-order lag model at the configured fps:

- Horizontal velocity in body frame tracks `cmd_forward` / `cmd_right` with time constant `τ = 0.30 s`.
- Vertical velocity tracks `cmd_up` with `τ = 0.35 s`.
- Yaw rate tracks `cmd_yaw_rate` with `τ = 0.10 s`.
- Visual roll/pitch are derived from body-frame velocity and lag with `τ = 0.14 s`.
- Takeoff and land use a proportional altitude controller that drives `cmd_up`.
- A position target is flown with a proportional approach clamped by `max_speed_mps`, plus a trim term that cancels a steady disturbance such as wind once the vehicle is near the target and at hover speed.
- Wind is added at position integration, so the vehicle's own velocity is through the air and the sum is over the ground.
- Collision with `wall` and `tree` cells absorbs the velocity component perpendicular to the wall.

## Drone Modes

| Mode | Meaning |
|---|---|
| `DISARMED` | On ground, ignores velocity commands |
| `GUIDED` | Armed and flying; accepts velocity commands |
| `TAKEOFF` | Climbing to target altitude; transitions to GUIDED on arrival |
| `LAND` | Descending to zero; transitions to DISARMED on touchdown |
| `RTL` | Climbing to a safe height, returning to home, then landing |
| `HOLD` | Armed, hovering at the current pose; set by `zero`/`hold` or by a failsafe |
| `CRASHED` | Collided; ignores commands until `reset` |

## Running Both Processes

```sh
# Terminal 1 – simulator
python3 dsim/dsim.py --id area1

# Terminal 2 – controller
python3 dctl/dctl.py --id area1
```

Multiple independent simulation instances can run simultaneously with different ids (e.g. `area1`, `area2`). Each pair of processes uses its own isolated set of buffers.
