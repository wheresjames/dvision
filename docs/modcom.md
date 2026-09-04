# Module communication

This document describes how dvision2 processes communicate through pymembus
today, which process owns each shared-memory area, and what crosses each
boundary. It ends with the proposed module-coordination channel discussed in
`DV-DWAY.md` and `DV-DALG.md`; that channel is not implemented yet.

## Instance naming and ownership

Every live vehicle instance has an `--id`. `dvision2_common.shared_names(id)`
currently expands it into three POSIX shared-memory names:

```text
/dvision2.<id>.video
/dvision2.<id>.control
/dvision2.<id>.status
```

For `--id area1`, for example, the names are
`/dvision2.area1.video`, `/dvision2.area1.control`, and
`/dvision2.area1.status`.

Exactly one **vehicle provider** owns an id. Normally that is `dsim`; `dfgb`
is an alternative provider backed by FlightGear. The provider removes stale
areas with those names, creates new ones, publishes video and status, and is
the sole reader of vehicle commands. Starting `dsim` and `dfgb` with the same
id is invalid because each will replace the other's shared memory.

Clients retry opening missing areas, which permits the provider and clients to
start in either order. Recreating an area starts a new pymembus session. Video
sequence/session information and the status epoch let clients recognize new
data or a restarted provider; they are transport facts, not experiment run
identity.

## Current topology

| Process | Role | `.video` | `.control` | `.status` |
|---|---|---|---|---|
| `dsim` | Primary simulated vehicle provider | creates, writes | creates, reads | creates, writes |
| `dfgb` | Alternative FlightGear vehicle provider | creates, writes | creates, reads | creates, writes |
| `dctl` | Manual controller and viewer | opens, reads | opens, writes | opens, reads |
| `daic` | Autonomous perception/controller | opens, reads | opens, writes | opens, reads |
| `dway` | Waypoint navigator | does not open | opens, writes | opens, reads |
| `dtest.process_harness` | Test-only vehicle client | may open/read | opens/writes | opens/reads |

`dalg` does not exist yet. Its design calls for reading video and status and
using the proposed module bus. It must not write vehicle commands.

The three existing areas have deliberately different semantics:

```text
                         latest RGB frames
 vehicle provider  ─────────────────────────▶ viewers/perception
        │                 memvid
        │
        ├──────── retained vehicle state ───▶ all clients
        │                 memkv
        │
        ◀──────── serialized commands ─────── controllers/navigator
                          memcmd
```

## Video: `.video`

The video area is a `pymembus.memvid` ring buffer. The vehicle provider creates
it from its configured width, height, frame rate and number of slots. Current
defaults for both providers are 640 × 480, 30 frames per second and four
buffers. The pixel format is RGB24.

`dsim` renders directly into the current writable slot, assigns video and audio
presentation timestamps, and advances the ring. Timestamps are integer
microseconds derived from the provider's local monotonic time. `dfgb` writes
RGB frames captured from FlightGear through its own capture path.

`dctl` and `daic` open the existing video area and read the newest completed
slot (`getPtr(-1)`). They use the sequence number to avoid processing the same
frame repeatedly. Readers do not remove frames and do not coordinate with one
another; a slow reader observes a newer frame rather than applying backpressure
to the renderer.

The video ring carries pixels and pymembus frame metadata only. Camera
intrinsics, extrinsics, pose and simulator time are published separately in
status. There is currently no atomic cross-area transaction tying one status
snapshot to one video slot. A consumer correlates them by observation time and
the available video timestamp. If exact frame/state pairing becomes necessary,
the protocol will need an explicit shared frame identifier or timestamp
contract.

## Commands: `.control`

The command area is a `pymembus.memcmd` text queue, normally 65,536 bytes. All
controllers may write; the one vehicle provider is the only reader. It drains
the queue with `read_with_overrun(0)`. An overrun is reported in vehicle status
and the affected command is not applied.

Messages are compact JSON. `dvision2_common.encode_command()` adds the version
gate:

```json
{"magic":"dvision2.command.v1","type":"hold"}
```

Malformed JSON, the wrong `magic`, or a missing string `type` is ignored.
Controlled commands normally also contain:

```json
{
  "source_id": "dway-area1",
  "lease_id": "opaque-lease-id",
  "request_id": "opaque-request-id"
}
```

`dsim` implements these command types:

| Type | Additional fields | Meaning |
|---|---|---|
| `acquire_control` | — | Acquire the single vehicle-control lease. |
| `release_control` | — | Release the caller's lease. |
| `heartbeat` | — | Refresh the owner's lease, but not the setpoint timer. |
| `arm` | `armed` | Arm or disarm. |
| `takeoff` | `alt_m` | Take off to an altitude above map ground. |
| `land` | — | Land; emergency land is accepted without a lease. |
| `rtl` | — | Return to captured home and land. |
| `zero`, `hold` | — | Clear targets and hold when armed. |
| `velocity` | `forward_mps`, `right_mps`, `up_mps`, `yaw_rate_dps` | Send a body-frame velocity target. |
| `position_target` | `frame`, coordinates, `heading_deg`, `max_speed_mps` | Send a map or local-NED position target. |
| `set_origin` | `lat_deg`, `lon_deg`, `alt_m` | Change the geographic origin while disarmed. |
| `set_gps` | `mode`, optional `noise_m` | Change the simulated GPS condition. |
| `set_estimator` | any of `attitude`, `local`, `global`, `velocity` | Change simulated estimator validity. |
| `reset` | — | Reset the simulated vehicle pose and motion. |

Except for emergency land, motion, mode and arming commands must carry the
active `source_id` and `lease_id`. `dsim` publishes the result of every command
through the three `command.result.*` status keys. `dway.DsimLink` waits for the
matching `request_id`; successfully writing to the queue is not acceptance.
The three keys are a single latest-value slot, which one command per frame
per vehicle would be enough for; `dsim` drains its whole queue before
publishing status once, so a second commanding client would otherwise replace
an outcome before its owner ever saw it. `command.results` publishes the last
`COMMAND_RESULT_HISTORY` outcomes beside the slot, and a client that cannot
find its `request_id` in the slot looks there before deciding it has been
ignored. The slot keeps the untruncated reason; the history truncates it to
`COMMAND_RESULT_REASON_MAX` characters.

`dctl`, `daic` and `dway` can acquire the control lease and send owner
heartbeats. `dctl` avoids contending when it observes another owner. The lease
and guided-setpoint timeouts are separate: a heartbeat keeps ownership alive,
while only a fresh velocity or position target keeps guided motion alive.

### FlightGear compatibility boundary

`dfgb` currently creates the same command area and accepts the older subset
`heartbeat`, `arm`, `takeoff`, `land`, `zero`, and `velocity`. It does not
currently implement DSIM's control leases, correlated command results,
position targets, RTL, origin changes, or realism fault commands. It also does
not publish the complete vehicle-capability contract described below.

Consequently, sharing the same area names does not yet make `dfgb` a complete
`DsimLink` replacement. A client must not infer protocol support from the
presence of `.control`; it must use advertised capabilities, and `dfgb` needs
the newer contract before `dway` can safely treat it as equivalent.

## Status: `.status`

The status area is a `pymembus.memkv` retained key/value store. The vehicle
provider creates its fixed schema and is its sole writer. Clients use
`getAll()` to read a coherent current snapshot and may use the epoch/change
mechanisms to detect updates. Values are strings; empty string means that a
defined fact is currently unavailable. Status is current state, not an event
log, and intermediate values may be overwritten before a slow reader sees
them.

The canonical schema is `dvision2_common.STATUS_KEYS`. It currently contains:

| Group | Keys |
|---|---|
| Simulator identity and time | `sim.id`, `sim.map`, `sim.time_s`, `sim.report_dir`, `sim.camera_in_geometry` |
| Vehicle capabilities | `vehicle.type`, `vehicle.frames`, `vehicle.accepts_position`, `vehicle.accepts_velocity`, `vehicle.accepts_attitude`, `vehicle.supports_missions`, `vehicle.setpoint_timeout_s`, `vehicle.max_speed_mps`, `vehicle.max_accel_mps2` |
| GPS quality | `gps.fix_type`, `gps.satellites`, `gps.hdop`, `gps.vdop` |
| Estimator validity | `est.attitude_valid`, `est.local_position_valid`, `est.global_position_valid`, `est.velocity_valid` |
| Environment | `wind.speed_mps`, `wind.dir_deg`, `wind.gust_mps`, `geofence.box`, `geofence.action` |
| Realism configuration | `realism.telemetry_latency_ms`, `realism.telemetry_jitter_ms`, `realism.sensor_noise`, `realism.battery_failsafe_pct`, `realism.battery_drain_pct_s`, `realism.seed` |
| Geographic reference | `origin.lat_deg`, `origin.lon_deg`, `origin.alt_m`, `home.lat_deg`, `home.lon_deg`, `home.alt_m` |
| Ownership and failsafe | `control.owner`, `control.lease_age_s`, `control.lease_timeout_s`, `setpoint.age_s`, `failsafe.reason` |
| Latest command result | `command.result.request_id`, `command.result.accepted`, `command.result.reason` |
| Recent command results | `command.results` |
| Vehicle mode and map pose | `drone.armed`, `drone.mode`, `drone.x_m`, `drone.y_m`, `drone.z_m` |
| Global position and target | `drone.lat_deg`, `drone.lon_deg`, `drone.alt_m`, `target.lat_deg`, `target.lon_deg`, `target.alt_m` |
| Attitude and velocity | `drone.roll_deg`, `drone.pitch_deg`, `drone.heading_deg`, `drone.compass_deg`, `drone.vx_mps`, `drone.vy_mps`, `drone.vz_mps`, `drone.speed_mps` |
| Vehicle health | `drone.battery_pct`, `drone.crashed`, `drone.last_command_s` |
| Link diagnostics | `link.command_count`, `link.last_command_type`, `status.message` |
| Camera model | `camera.fov_h_deg`, `camera.fov_v_deg`, `camera.tx_m`, `camera.ty_m`, `camera.tz_m`, `camera.roll_deg`, `camera.pitch_deg`, `camera.yaw_deg`, `camera.fx_px`, `camera.fy_px`, `camera.cx_px`, `camera.cy_px`, `camera.width_px`, `camera.height_px`, `camera.fps` |

`dsim` publishes the full contract. Some values deliberately describe noisy or
delayed telemetry rather than physics truth. In particular, position,
heading, altitude and velocity can reflect configured sensor errors, and
status publication can pass through the telemetry delay ring. `sim.time_s` is
the simulator's authoritative elapsed clock for coordinated experiments.

`dfgb` creates the canonical key names for compatibility but currently fills
only its available subset: simulator identity/time, basic vehicle pose,
attitude, velocity, battery/crash state, last-command diagnostics and a status
message. Consumers must treat empty or missing values as unavailable and check
capabilities rather than assuming DSIM behavior.

## What each client uses

### `dctl`

`dctl` opens all three areas. It displays the newest video frame and retained
vehicle status. Manual controls and buttons write commands. It creates a
control identity and lease id, observes `control.owner`, acquires only when it
will not contend with another controller, and sends heartbeats only while it
owns control.

### `daic`

Both UI and headless DAIC paths open all three areas. Perception consumes the
newest video frame; planning, optical flow compensation, logging and reporting
consume status. Its controller writes velocity, mode and lifecycle commands
and maintains a control lease. Files under the report directory are outputs,
not communication back to the vehicle provider.

### `dway`

`dway.DsimLink` opens only `.control` and `.status`. It does not consume video.
It normalizes retained status into `VehicleCapabilities` and `VehicleState`,
writes correlated commands, and waits for their matching latest result. The
mission streams position targets when supported and otherwise uses velocity
targets. Its `flight.jsonl` and `summary.json` are report artifacts, not live
IPC for DALG or another module.

### `dtest.process_harness`

The process harness is a test client, not a runtime module. It can open all
three areas, send commands, inspect status, and verify that the provider
removes its shared memory on shutdown.

## Module coordination: `.events`

The implementation adds a fourth per-instance name:

```text
/dvision2.<id>.events
```

It is a many-publisher, many-subscriber `ModuleBus` carrying JSON presence,
readiness and run-lifecycle events. pymembus is the first transport adapter;
the application-facing interface must also permit future MQTT or ROS 2
adapters. The vehicle provider would create and remove the pymembus area with
the other instance areas, but it would not own the information published on
it.

The event plane is separate for semantic and mechanical reasons:

| Plane | Current/proposed primitive | Authority | Delivery model |
|---|---|---|---|
| Video | `memvid` | vehicle provider | latest frames in a ring |
| Vehicle commands | `memcmd` | leased controller writes; vehicle reads | many writers to one reader |
| Vehicle status | `memkv` | vehicle provider | one writer, retained latest values |
| Module coordination | proposed `memmsg` | each module speaks for itself | broadcast events to every subscriber |

The envelope and vocabulary are normative in `DV-DWAY.md` §2.2.
Briefly, modules publish `module.hello`, heartbeats, readiness, and goodbye;
the navigator publishes `run.prepare` and an absolute simulator-time start;
required participants reply for that exact run; and all modules publish run
state and terminal outcomes using a shared `run_id`.

This design avoids a shared writable module `memkv`. Such a store would need a
fixed slot allocation or a registry owner, plus expiry and cleanup rules for
crashed writers. Periodic broadcast heartbeats allow every subscriber,
including a future DSIM pipeline panel, to maintain the same local expiring
view without making DSIM a broker.

The transport also carries `system.shutdown`. This is an instance-scoped,
orderly shutdown request rather than an operating-system kill: every connected
module stops its loop, releases control and closes its resources. DSIM's
**Kill all** button publishes the event before DSIM closes itself. The event's
payload contains `scope: "instance"` and a human-readable `reason`; receivers
must act on the event regardless of `run_id` or source role.

Before selecting `memmsg`, verify with a multi-process spike that it provides:

1. an independent cursor for every subscriber;
2. one writer's message delivered to every active subscriber;
3. safe concurrent publishers;
4. detectable ring overrun and recovery;
5. session/recreation detection when the provider restarts;
6. bounded non-blocking behavior when a subscriber is slow or dead.

If `memmsg` does not provide those semantics, retain the `ModuleBus` contract
and implement the smallest pymembus fan-out adapter that does. Do not weaken
coordination into a single-consumer queue, and do not expose pymembus calls to
mission or algorithm state machines.

## Rules for adding or replacing a module

- Use `shared_names(id)`; do not construct shared-memory names independently.
- Only the vehicle provider creates `.video`, `.control`, and `.status`.
- Only the vehicle provider writes `.status` and reads `.control`.
- Only a control-lease holder writes motion or mode commands, apart from the
  explicitly lease-free emergency land operation.
- Treat capabilities and live validity as separate facts.
- Treat report files as artifacts, never as live IPC.
- Match protocol roles and versions, not executable names. A future `dway2`
  replaces `dway` by implementing the navigator and vehicle-link contracts;
  existing observers should require no modification.
- Carry experiment identity explicitly as `run_id`; a pymembus session id,
  process id, report directory, and `--id` identify different things.
