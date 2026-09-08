# Module communication

This document describes how dvision2 processes communicate through pymembus
today, which process owns each shared-memory area, and what crosses each
boundary. The module-coordination event channel is implemented. For a complete
inventory of actual buffer names, payloads, readers/writers and lifecycle
behavior, see [the shared-memory guide](membus.md).

## Recovery contract for new interfaces

[Crash recovery and module rejoin](recovery.md) defines the target restart and
reconnection contract: stable vehicle identity, fresh process sessions,
recoverable discovery, clock continuity, readiness, and command reconciliation.
It applies to new interfaces and the sensor migration. Automatic mission
recovery is future module work; a publisher restart does not itself mean a
vehicle or mission restart. The current interfaces described below are not a
claim that all of those recovery behaviors are already implemented.

## Instance naming and ownership

Every live vehicle instance has an `--id`. `dvision2_common.shared_names(id)`
expands it into the stable per-instance POSIX shared-memory names:

```text
/dvision2.<id>.sensors   sensor registry (memkv)  dsim -> clients
/dvision2.<id>.control   JSON commands             clients -> dsim
/dvision2.<id>.status    Telemetry k/v             dsim -> clients
/dvision2.<id>.events    module coordination bus   every module
```

For `--id area1`, for example, the names are `/dvision2.area1.sensors`,
`/dvision2.area1.control`, `/dvision2.area1.status`, and
`/dvision2.area1.events`.

Beside the stable names, the sensor provider publishes **generation-qualified
data channels** derived from its session and the current sensor-manifest
generation (see "Sensors" below). Every shared-memory name in this repository,
stable or qualified, is at most 254 bytes — the measured pymembus/POSIX name
limit — and IDs are never truncated to fit it; a name that would exceed it is
rejected by validation before any area is created.

Exactly one **vehicle provider** owns an id. Normally that is `dsim`. The
provider removes stale areas with those names, creates new ones, publishes the
sensor registry and status, and is the sole reader of vehicle commands.
Starting two providers with the same id is invalid because each will replace
the other's shared memory.

Clients retry opening missing areas, which permits the provider and clients to
start in either order. Recreating an area starts a new pymembus session. The
registry's provider session id and generation let clients recognize new data
or a restarted provider; they are transport facts, not experiment run identity.

`dfgb` is currently **not** a sensor-contract provider: it still creates the
old `/dvision2.<id>.video` area that this migration removed, so it no longer
interoperates with the current clients. It is set aside (see `DV-SENSORS.md`)
and is not part of the acceptance gate for the sensor work.

## Current topology

| Process | Role | `.sensors` | `.control` | `.status` |
|---|---|---|---|---|
| `dsim` | Primary simulated vehicle provider | creates, writes | creates, reads | creates, writes |
| `dctl` | Manual controller and viewer | opens, reads | opens, writes | opens, reads |
| `daic` | Autonomous perception/controller | opens, reads | opens, writes | opens, reads |
| `dalg` | Algorithm measurement runner | opens, reads | opens, writes | opens, reads |
| `dway` | Waypoint navigator | does not open | opens, writes | opens, reads |
| `dtest.process_harness` | Test-only vehicle client | opens/reads | opens/writes | opens/reads |

The areas have deliberately different semantics:

```text
                    sensor registry + generation-qualified channels
 vehicle provider  ─────────────────────────────────────▶ viewers/perception
        │            memkv discovery + memvid/record rings
        │
        ├──────── retained vehicle state ───▶ all clients
        │                 memkv
        │
        ◀──────── serialized commands ─────── controllers/navigator
                          memcmd
```

## Sensors: registry, channels, and the sample record

This section is the **frozen v1 wire contract** for sensor discovery and
transport ([`DV-SENSORS.md`](../DV-SENSORS.md) is the design; this section is
what the code implements). Producers and consumers are implemented against
this text; changing any of it requires a schema-version increment and a
contract-test update.

### Primitives and measured limits

The sensor plane uses only pymembus primitives:

| Purpose | Primitive | Measured/selected limit |
|---|---|---|
| Discovery registry | `memkv` (fixed key names, one atomic `setAll` commit) | 9 keys, values up to 64 KiB |
| Camera pixels | `memvid` RGB24 ring | one per camera, dimensions from the profile model |
| Compact sample records (incl. camera and LiDAR metadata) | one shared `memmsg` broadcast ring via byte-safe APIs | records up to 8192 bytes, ring ≥ 64 KiB |
| Bulk array samples (LiDAR scans and range images) | one dedicated `memmsg` ring per array sensor | record sized from the declared array, ring ≥ 64 KiB |

The byte-safe `memmsg` APIs (`write_bytes`, `read_bytes_with_overrun`) are
provided by the local pymembus build `2.1.0+records` (see
`scripts/build_sensor_pymembus.py`, output under `.cache/sensor-pymembus`);
the upstream Python binding exposes only text conversion for the existing
binary-safe C++ ring. Measured primitive guarantees frozen by
`tests/test_sensor_contract.py`:

- `memkv.getAll()` returns one coherent committed snapshot of all keys.
- `memvid` returns the exact committed frame sequence of a slot before and
  after `next()`, which is what makes sequence-matched intake possible.
- Names are at most **254 bytes**; longer names fail creation, so the profile
  validator rejects them instead of truncating IDs.
- A canonical sensor manifest is at most **64 KiB**; the registry is created
  with sufficient value capacity for it and oversized manifests are rejected.
- A record payload plus header is at most **8192 bytes** (`MAX_RECORD`).
- Reader overrun is reported per `read_bytes_with_overrun` call and
  accumulates; it never blocks or corrupts the ring.

### Registry

The stable registry name is `/dvision2.<id>.sensors`, a `memkv` area created by
the provider with these nine fixed keys:

| Key | Value |
|---|---|
| `sensors.schema` | `dvision2.sensor-manifest.v1` |
| `sensors.profile_name` | resolved drone-profile name |
| `sensors.profile_digest` | SHA-256 of the canonical resolved profile JSON |
| `sensors.manifest` | compact JSON manifest (below) |
| `sensors.generation` | decimal integer, starts at 1 |
| `vehicle_id` | the instance id |
| `provider_session_id` | lowercase 32-hex UUID, fresh per provider process |
| `clock_domain_id` | the instance id (the provider owns simulated time) |
| `clock_epoch` | changes only when timestamp continuity is lost |

All nine keys are published together with one `memkv.setAll()`; readers use
`getAll()` and therefore observe a complete committed generation or the
previous one, never a mixture. The registry is the **only** discovery
mechanism: vehicle status carries no per-sensor keys.

### Generation-qualified channel names

Data areas embed session and generation so a restarted provider or a new
generation can never be confused with stale readable handles:

```text
/dvision2.<id>.s<session32hex>.g<generation>.sensor.<sensor-id>.video
/dvision2.<id>.s<session32hex>.g<generation>.sensor.<sensor-id>.array
/dvision2.<id>.s<session32hex>.g<generation>.sensor.samples
```

`<session32hex>` is the registry's `provider_session_id`; `<generation>` is the
registry's `sensors.generation`. The shared compact ring is one
generation-qualified `sensor.samples` area per provider generation; each RGB
camera additionally gets its own `sensor.<id>.video` `memvid`, and each
array-producing sensor its own `sensor.<id>.array` record ring. A sensor whose
samples are small — every scalar range type — has no channel of its own and
publishes on the shared ring.

### Manifest

The manifest is configuration, not live samples. Fields frozen in v1:

| Field | Meaning |
|---|---|
| `schema` | `dvision2.sensor-manifest.v1` |
| `provider_session_id`, `vehicle_id`, `generation` | transport identity |
| `clock_domain_id`, `clock_epoch` | timestamp authority |
| `primary_camera` | the semantic RGB camera single-camera consumers open |
| `profile` | the resolved drone profile (canonical form) |
| `profile_digest` | digest of the resolved profile |
| `memory_bytes` | pre-validated shared-memory budget for the profile |
| `sample_channel`, `sample_capacity` | compact record ring name and byte capacity |
| `sensors` | map of sensor id → the discovery entry below, for every **enabled** sensor. A disabled or removed sensor has no entry and no channel |

Each `sensors` entry carries `type`, `enabled`, `rate_hz`, `parent`,
`pose_parent`, the resolved `model`, `sync_group` (or `null`), and a
`transport` naming where its samples appear:

| `transport` | Additional fields |
|---|---|
| `video` | `channel` (its `memvid`), `pixel_format` (`RGB24`), `payload_type` 1, `payload_schema` `camera.frame.v1`, `calibration_revision` (the profile digest) |
| `array` | `channel` (its dedicated ring), `capacity`, `record_bytes`, `payload_type` 100, `layout` (the packed fields below), `calibration` (angular calibration), `metadata_type` 3, `metadata_schema` `lidar.frame.v1` |
| `compact` | `channel` (the shared ring), `payload_type` 2, `payload_schema` `range.sample.v1` |

`primary_camera` names one `video` entry; a single-camera consumer opens that
one. Nothing else distinguishes it, and a consumer may open any camera by id.

### Sample record envelope

Every record — camera and LiDAR metadata, every scalar sensor sample, and
every packed array — uses one fixed little-endian header
followed by a payload. The header is exactly **128 bytes** (`struct`
`'<4sHH16sQQQqQQ48sHHI'`), and its first eight bytes are the frozen fixture
`44 56 53 31 01 00 80 00` (`DVS1`, version 1, header size 128):

| Offset | Type | Field |
|---|---|---|
| 0 | `4s` | magic `DVS1` |
| 4 | `H` | envelope version `1` |
| 6 | `H` | header size `128` |
| 8 | `16s` | provider session UUID bytes |
| 24 | `Q` | generation |
| 32 | `Q` | per-sensor sequence (starts at 1; advances on attempted captures, including failures) |
| 40 | `Q` | `capture_id` (identity of the logical capture; identical for synchronized samples) |
| 48 | `q` | `sim_time_us` — integer simulated microseconds |
| 56 | `Q` | `reset_epoch` (increments on drone reset; time, session, generation, sequences unchanged) |
| 64 | `Q` | `clock_epoch` |
| 72 | `48s` | sensor id, ASCII, NUL-padded |
| 120 | `H` | payload type |
| 122 | `H` | status (`1` valid, `0` invalid; JSON uses `null`, packed arrays use NaN and zero confidence) |
| 124 | `I` | payload length in bytes |

Payload types below 100 are **strict compact UTF-8 JSON objects**:
`sort_keys`, no insignificant whitespace, `allow_nan=False`, and `NaN` /
`Infinity` / `-Infinity` are rejected on both encode and decode. Invalid
measurements are `null` with explicit status, never NaN or infinity. Payload
types 100 and above are packed little-endian arrays whose dtype and shape the
manifest declares. The assigned types are:

| Type | Schema | Ring |
|---|---|---|
| 1 | `camera.frame.v1` | shared compact |
| 2 | `range.sample.v1` | shared compact |
| 3 | `lidar.frame.v1` | shared compact |
| 4 | `gnss.sample.v1` | shared compact |
| 5 | `imu.sample.v1` | shared compact |
| 6 | `barometer.sample.v1` | shared compact |
| 7 | `magnetometer.sample.v1` | shared compact |
| 8 | `temperature.sample.v1` | shared compact |
| 100 | packed array | the sensor's dedicated array ring |

`sequence` is per sensor and starts at 1 in each generation. A LiDAR's
metadata record and its array record share one sequence value, which is what
associates them; a consumer that follows both streams therefore counts their
sequence gaps separately.

### Camera transport and frame association

An RGB camera is one `memvid` RGB24 ring whose width, height, rate, and slot
count come from the profile model (`ceil(rate_hz * transport.retention_s) + 2`
slots, with one simulated second of retention by default). The provider renders into the current
slot, then sets both `memvid` presentation timestamps to the **integer
simulated timestamp** (`sim_time_us`) — not wall/monotonic time — advances the
ring, and reads back the exact committed frame sequence from the write
operation.

Pixels alone carry no pose. Immediately after each frame commit the provider
writes one `camera.frame` record (payload type 1) to the shared compact ring:

```json
{
  "schema": "camera.frame.v1",
  "video_sequence": 214,
  "pose_world": [[...4x4 transform...]],
  "pose": {"x_m": 0.0, "y_m": 0.0, "z_m": 1.6, "heading_deg": 0.0,
            "roll_deg": 0.0, "pitch_deg": -5.0},
  "body": {"x_m": 0.0, "y_m": 0.0, "z_m": 1.5, "heading_deg": 0.0,
            "roll_deg": 0.0, "pitch_deg": 0.0,
            "vx_mps": 0.0, "vy_mps": 0.0, "vz_mps": 0.0},
  "calibration_revision": "<profile digest>"
}
```

- `video_sequence` is the `memvid` frame sequence of the committed slot; the
  association key is `(provider_session_id, generation, sensor_id,
  video_sequence)`.
- `pose_world` is the composed 4x4 sensor→world transform (row-major nested
  lists) at capture time; `pose` is the same pose as public
  heading/pitch/roll/position in map coordinates. Units are metres and
  degrees. `body` is the vehicle datum state frozen for the capture.
- `calibration_revision` is the profile digest the intrinsics belong to.

A consumer admits a frame **only** when its video sequence has matching
metadata. Unmatched images are copied into a bounded pending cache and
retried as metadata drains; the cache is bounded by the configured camera
rate times retention (rounded up), plus two entries. An evicted unmatched entry increments an
`association_drops` counter. No consumer waits indefinitely or processes an
unmatched image. Vehicle status is advisory retained telemetry; it is not
frame truth, and consumers derive per-frame calibration and pose from the
manifest and record, not from status keys.

### Scalar range transport

A scalar range sensor (`range.infrared`, `range.ultrasonic`, `range.laser`)
publishes one `range.sample.v1` record (payload type 2) per capture on the
shared compact ring:

```json
{
  "schema": "range.sample.v1",
  "type": "range.ultrasonic",
  "range_m": 1.83,
  "confidence": 0.74,
  "returns": 7,
  "samples": 9,
  "reducer": "nearest",
  "min_range_m": 0.2,
  "max_range_m": 7.0,
  "pose": {"x_m": 0.0, "y_m": 0.0, "z_m": 1.47, "heading_deg": 0.0,
            "roll_deg": 0.0, "pitch_deg": -90.0},
  "pose_world": [[...4x4 transform...]],
  "body": {"x_m": 0.0, "y_m": 0.0, "z_m": 1.5, "heading_deg": 0.0, "...": 0.0}
}
```

- `range_m` is metres along the reported return, `confidence` is 0.0–1.0.
- **No valid return is not maximum range.** The envelope status is `0`,
  `range_m` is `null`, `confidence` is `0.0` and `returns` is `0`.
- `returns` counts how many of the `samples` cone rays produced a valid
  return, and `reducer` names which of them `range_m` came from.
- `pose`/`pose_world`/`body` follow the `camera.frame.v1` definitions above.

### State-sensor transport

GNSS, IMU, barometer, magnetometer and temperature publish one compact record
per capture on the shared ring, payload types 4 to 8. Their fields are
documented per sensor under [`docs/sensor/`](sensor/README.md); what is frozen
here is the shape they all share.

A state-sensor record carries **the measurement and nothing else**: no `pose`,
no `pose_world`, no `body` datum. A GNSS record beside the true position, or
an altimeter record beside the true altitude, is not a measurement, and the
axes a consumer needs are static configuration it already has from the
manifest's `pose_parent`. This is the one place the record shape differs from
the camera and ranging sensors, which do carry the capture pose because a
range or a pixel is meaningless without one.

Invalid follows the same rule as everywhere else on the compact ring: the
envelope status is `0` and each unmeasured field is `null`. A GNSS record with
no fix still reports `fix_type`, `satellites`, `hdop` and `vdop`, because
"no fix, zero satellites" is itself the measurement; its position and velocity
are `null`.

Installation is the profile's decision, not the environment's. A vehicle whose
profile contains no enabled `position.gnss` sensor publishes no GNSS records
**and** reports `gps.fix_type` 0 with `est.global_position_valid` 0 in vehicle
status. "No receiver fitted", "receiver with no fix" and "fix rejected by the
estimator" are three distinct states with three distinct causes.

### Bulk array transport

A `lidar.scan2d` or `lidar.range_image` sensor publishes **two** records per
capture, sharing one `sequence` and one `capture_id`:

1. the packed array (payload type 100) on the sensor's own ring, and
2. one `lidar.frame.v1` record (payload type 3) on the shared compact ring,
   carrying `array_sequence`, `type`, `returns`, `samples`, `calibration`,
   and the same `pose`/`pose_world`/`body`/range-gate fields as a scalar
   range sample.

Pose and timing live in the compact record so the array stays a pure array,
which is what keeps the dedicated ring's records a fixed declared size. The
array payload is the manifest `layout` fields concatenated, little-endian and
row-major:

| Field | dtype | Shape | Invalid |
|---|---|---|---|
| `range_m` | `<f4` | `[samples]`, or `[height_px, width_px]` | `NaN` |
| `confidence` | `\|u1` | same | `0` |

`NaN` is the invalid marker **only** in packed arrays; compact JSON still
forbids it and uses `null` plus status. `confidence` is 0–255 in an array and
0.0–1.0 in JSON, because one is a byte per sample and the other is one number
per record.

Angular calibration is in the manifest entry's `calibration`, not repeated per
sample:

- `lidar.scan2d`: `angle_min_deg`, `angle_increment_deg`, `samples`,
  `elevation_deg`. Ray *i* points at `angle_min_deg + i · angle_increment_deg`
  in the sensor's own frame, where 0° is the sensor's +X and a **positive
  angle turns toward +Y (right)**, matching the body yaw convention. A full
  360° scan steps `fov / samples` so it does not repeat its first ray; a
  partial sector steps `fov / (samples − 1)` so it includes both endpoints.
- `lidar.range_image`: `width_px`, `height_px`, `fx_px`, `fy_px`, `cx_px`,
  `cy_px` — the same pinhole calibration a camera publishes, so a range image
  converts to points with the sensor pose and nothing else.

Bulk traffic never shares the compact ring, so a large or slow LiDAR consumer
cannot evict a GPS, barometer or scalar range record.

### Generation, restart, and reset lifecycle

- **Provider startup** removes a stale registry (crashed predecessor),
  creates a fresh one, and publishes generation 1 under a fresh session UUID.
- **Profile application** (`dsim` Sensors tab, disarmed only) creates all
  generation N channels, publishes the complete manifest and generation with
  one `setAll()`, then unlinks generation N−1's areas. Open handles drain
  naturally; readers reopen on identity change. Failure during construction
  retains the previous generation untouched.
- **Provider restart** is a fresh session: readers that lose liveness reopen
  the stable registry name even if old mappings remain readable, and treat
  the new session like a generation change (fresh `memvid` sequence space).
- **Drone reset** preserves session, generation, sequences, capture ids, and
  simulated time; it increments `reset_epoch` so consumers invalidate sensor
  history without cancelling the transport or the mission.
- A reader's logical sequence is continuous across generations: it offsets the
  new generation's 0-based `memvid` sequences by the highest sequence it
  already observed, so "same sequence as last time" remains a valid
  skip test after a reopen.

### Consumer intake accounting

A consumer reports two different numbers, because they answer two different
questions:

- **ring overruns** are what the transport noticed: the reader was lapped
  between two reads of one ring.
- **skipped** is what the consumer actually missed, counted from gaps in a
  sensor's own `sequence`. Sequences start at 1 within a generation, so this
  includes records published before the consumer attached to that generation;
  a generation change resets the count along with the rate windows.

Neither replaces the other: a ring can lap a reader without the reader ever
learning it from the transport, and a sensor can stop publishing without any
ring overrunning.

Single-camera consumers (`daic`, `dalg`, `dctl`, test harnesses) open the
registry, read `primary_camera`, and open that camera's advertised channel.
They do not fall back to any legacy `.video` name or `camera.*` status keys;
those interfaces no longer exist.

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

Camera model, extrinsics, capture pose, and frame-true time are **not** status
keys: they live in the sensor registry and per-capture `camera.frame` records
(see "Sensors" above). Consumers derive the values they need from those two
sources.

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

`dctl` opens the sensor registry (and the discovered primary camera's
channel) plus control and status. It displays the newest matched video frame
and retained vehicle status. Manual controls and buttons write commands. It
creates a control identity and lease id, observes `control.owner`, acquires
only when it will not contend with another controller, and sends heartbeats
only while it owns control.

### `daic`

Both UI and headless DAIC paths open the sensor registry (primary camera),
control, and status. Perception consumes the newest matched video frame;
planning, optical flow compensation, logging and reporting consume status.
Its controller writes velocity, mode and lifecycle commands and maintains a
control lease. Files under the report directory are outputs, not
communication back to the vehicle provider.

### `dalg`

`dalg` opens the sensor registry (primary camera) and status; it does not
write vehicle commands. It runs perception algorithms against matched frames
and the exact in-process range oracle; its provenance records the sensor
manifest of the run. Its run reports are artifacts, not live IPC.

### `dway`

`dway.DsimLink` opens only `.control` and `.status`. It does not consume
sensors. It normalizes retained status into `VehicleCapabilities` and
`VehicleState`, writes correlated commands, and waits for their matching
latest result. The mission streams position targets when supported and
otherwise uses velocity targets. Its `flight.jsonl` and `summary.json` are
report artifacts, not live IPC for DALG or another module.

### `dtest.process_harness`

The process harness is a test client, not a runtime module. It opens the
sensor registry and primary camera, sends commands, inspects status, and
verifies that the provider removes its shared memory on shutdown.

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

### `module.sensor_health`

Every module that consumes sensors publishes one `module.sensor_health` event
per wall-clock second, beside its heartbeat and not inside it: the heartbeat
is a liveness message and has to stay small, while this grows with the number
of subscriptions. The payload is the module's `state` plus `sensor_inputs`,
one record per sensor it says it is using:

```json
{
  "state": "ready",
  "sensor_inputs": {
    "nav_left": {
      "generation": 3, "required": true,
      "expected_hz": 30.0, "observed_hz": 29.8,
      "last_sequence": 1820, "age_s": 0.018,
      "skipped": 2, "overruns": 0, "drops": 0,
      "sync": "ok", "state": "ok"
    }
  }
}
```

- `expected_hz` is the sensor's configured rate from the manifest;
  `observed_hz` and `age_s` are in **simulated** time, because sample cadence
  is a property of the data. Module liveness is the heartbeat's wall-clock
  question and is answered separately.
- `skipped` counts gaps in that sensor's own sequence -- records this module
  never received. `overruns` is what the transport noticed. They are different
  numbers and neither replaces the other; see "Consumer intake accounting".
- `required` is the module's own declaration. An optional input a module can
  work without never degrades its summary grade.
- `sync` is `ok`, `diverged` or `alone`. A synchronized group whose members
  report different `capture_id`s is `diverged` and unhealthy however good both
  rates look, because the geometry that made them a group no longer holds.
- `state` is `starting`, `ok`, `warn` or `bad`. `starting` is deliberately not
  a grade and does not aggregate: a module that has just connected, or has
  just reopened on a new generation, is not failing.

Reports are **per module, not merged**. `dsim` retains them in its
`PipelineView` and shows them as child rows under each module, beside its own
per-sensor production counts under a root of its own. A module's single
top-level grade is *derived* here rather than asked for: the worst required
sensor input, combined with the module's own processing rate and its
wall-clock liveness. A report older than the pipeline expiry is dropped rather
than shown -- a module that has stopped reporting is not a module whose
cameras are still healthy, and its last good numbers read exactly like that if
they are left on the screen.

`dsim` publishes production health for every configured sensor and never waits
on any consumer's report. Each record carries the configured and observed rate,
the scheduled, generated, published and invalid counts, publisher drops with the fault
message of the last failure, the last sequence and simulated time, and the
wall-clock work seconds spent producing that sensor's own samples -- a shared
stereo render pass is split evenly between the cameras that took part, so the
per-sensor figures sum to the time the sensor plane actually spent. Correlating
the two sides is a display and reporting job; there is no acknowledgement and
no backpressure anywhere in it. Counts and their elapsed-time denominator both
survive drone reset; a generation change starts new counters. Rendering and
publication faults are recorded per sensor while other sensor types continue.

Consumer intake windows restart on provider session, generation, or reset-epoch
changes. Camera removal or registry disappearance clears stale frames and
handles. Each camera intake reports `cache_bytes` separately from shared memory:
copied RGB frames plus encoded pending metadata, excluding Python object overhead.

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
- Only the vehicle provider creates the sensor registry, the
  generation-qualified sensor channels, `.control`, and `.status`.
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
