# MAVLink Architecture for SLAM-Guided Drone Navigation

## Purpose

This document proposes an extensible architecture for guiding MAVLink-compatible drones through waypoints using an off-drone SLAM system.

The design aims to support a practical range of vehicles:

- drones with reliable GPS and conventional mission support;
- GPS-denied drones with onboard local navigation;
- drones that can fuse externally supplied SLAM odometry;
- drones that accept only velocity or attitude targets;
- ArduPilot, PX4, and conservative generic-MAVLink integrations.

The main design principle is to treat MAVLink as a **capability-negotiated control transport**. The system should discover what the connected vehicle supports, determine which navigation estimates are currently valid, and choose the highest-level safe guidance method available.

GPS presence alone should not select the strategy. The more useful questions are:

1. What position and velocity estimates does the autopilot currently consider valid?
2. What command and setpoint interfaces does the autopilot accept?
3. Can the autopilot fuse the external SLAM estimate?
4. Are the SLAM data timely and trustworthy enough for the proposed control mode?

---

## Recommended system architecture

```text
                     VIDEO / SENSOR DATA
                             │
                             ▼
                    ┌─────────────────┐
                    │   Off-drone     │
                    │      SLAM       │
                    │                 │
                    │ pose + velocity │
                    │ covariance      │
                    │ timestamps      │
                    └────────┬────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Coordinate Manager   │
                  │                      │
                  │ SLAM/map frame       │
                  │ MAVLink local frame  │
                  │ GPS/global frame     │
                  │ body frame           │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Navigation Manager   │
                  │                      │
                  │ waypoint sequencing │
                  │ trajectory logic    │
                  │ arrival detection   │
                  │ strategy selection  │
                  └──────────┬───────────┘
                             │
               ┌─────────────┼─────────────┐
               ▼             ▼             ▼
          GPS mission    Local position   Velocity /
          controller     controller       attitude ctl
               │             │             │
               └─────────────┼─────────────┘
                             ▼
                          MAVLink
                             │
                             ▼
                     Flight controller
```

The key abstraction should be a normalized vehicle profile:

```python
@dataclass
class VehicleCapabilities:
    vehicle_type: str
    autopilot_type: str
    firmware_version: str | None

    attitude_valid: bool
    global_position_valid: bool
    local_position_valid: bool
    local_velocity_valid: bool

    accepts_global_position_target: bool
    accepts_local_position_target: bool
    accepts_velocity_target: bool
    accepts_attitude_target: bool
    accepts_external_odometry: bool
    supports_missions: bool
```

The navigation manager selects a backend from this profile. Autopilot-specific adapters supply the details that cannot be discovered reliably through generic MAVLink alone.

---

## 1. Identify the connected vehicle

Listen first for `HEARTBEAT`. It identifies:

- the vehicle class through `MAV_TYPE`—multirotor, fixed wing, VTOL, rover, and so on;
- the autopilot implementation through `MAV_AUTOPILOT`—for example, ArduPilot or PX4;
- the current system and flight mode state.

Next, request `AUTOPILOT_VERSION`, normally using `MAV_CMD_REQUEST_MESSAGE`.

The `AUTOPILOT_VERSION.capabilities` field contains `MAV_PROTOCOL_CAPABILITY` bits such as:

- `MAV_PROTOCOL_CAPABILITY_MISSION_INT`;
- `MAV_PROTOCOL_CAPABILITY_COMMAND_INT`;
- `MAV_PROTOCOL_CAPABILITY_SET_POSITION_TARGET_LOCAL_NED`;
- `MAV_PROTOCOL_CAPABILITY_SET_POSITION_TARGET_GLOBAL_INT`;
- `MAV_PROTOCOL_CAPABILITY_SET_ATTITUDE_TARGET`;
- `MAV_PROTOCOL_CAPABILITY_MAVLINK2`.

These bits indicate that an interface exists. They do **not** guarantee that every field, coordinate frame, mode, or type-mask combination is implemented.

Capability discovery should therefore combine:

```text
generic MAVLink capability flags
            +
autopilot identity and firmware version
            +
known ArduPilot/PX4 behavior profiles
            +
observed telemetry and estimator state
```

The software should expose explicit adapters:

```text
MavlinkVehicle
    ├── ArduPilotVehicle
    ├── PX4Vehicle
    └── GenericMavlinkVehicle
```

The generic adapter should be conservative. Unsupported behavior should result in a safe refusal to navigate, not experimental movement commands.

---

## 2. Determine which navigation estimates are valid

Sensor presence is useful, but estimator validity is more important.

### `SYS_STATUS`

`SYS_STATUS` reports the presence, enablement, and health of systems such as:

- GPS;
- optical flow;
- vision position;
- laser position;
- external ground truth;
- altitude control;
- horizontal position control.

These fields are helpful hints about the hardware and configured control functions.

### `ESTIMATOR_STATUS`

When the autopilot publishes it, `ESTIMATOR_STATUS` more directly answers whether navigation outputs are usable. Relevant flags include:

- `ESTIMATOR_ATTITUDE`;
- `ESTIMATOR_VELOCITY_HORIZ`;
- `ESTIMATOR_VELOCITY_VERT`;
- `ESTIMATOR_POS_HORIZ_REL`;
- `ESTIMATOR_POS_HORIZ_ABS`;
- `ESTIMATOR_POS_VERT_ABS`;
- `ESTIMATOR_POS_VERT_AGL`.

This distinguishes two very different questions:

- **Sensor question:** Does this vehicle have a GPS receiver?
- **Navigation question:** Can this vehicle currently navigate in a stable local or global coordinate frame?

A vehicle with no GPS may still have valid local position and velocity from optical flow, VIO, lidar, UWB, motion capture, or another source.

### GPS-specific telemetry

Use `GPS_RAW_INT` to examine the physical GPS measurement, including:

- `fix_type`;
- satellite count;
- reported horizontal and vertical accuracy;
- speed and course accuracy where available.

Do not confuse `GPS_RAW_INT` with the fused vehicle state. `GLOBAL_POSITION_INT` reports an estimated global position that may combine GPS with IMU, barometer, vision, and other sensors.

A normalized state might look like:

```yaml
gps:
  present: false
  usable: false

estimator:
  attitude: valid
  horizontal_velocity: valid
  local_horizontal_position: valid
  global_horizontal_position: invalid
```

That vehicle may still be fully capable of autonomous local navigation.

---

## 3. Make the SLAM map the application’s canonical frame

Store application waypoints in the SLAM/map coordinate system, not directly as NED coordinates or latitude/longitude.

For example:

```python
@dataclass
class Waypoint:
    x: float
    y: float
    z: float
    acceptance_radius_m: float = 0.75
```

The coordinate manager should maintain explicit transforms:

```text
SLAM_MAP → MAV_LOCAL
SLAM_MAP → WGS84
SLAM_MAP → BODY
```

This is similar to a robotics transform tree. It prevents axis swaps and sign changes from leaking throughout the navigation code.

### Frame-convention warning

SLAM systems commonly use ENU/FLU-like conventions, while MAVLink flight stacks commonly use NED/FRD-like conventions.

Typical local NED axes are:

```text
X = North
Y = East
Z = Down
```

An application waypoint 5 metres above the local origin therefore commonly becomes `z = -5` in NED.

In a completely GPS-denied environment, “north” need not be geographic north. A local frame may define:

```text
X = direction established at initialization
Y = 90 degrees to the right
Z = down
```

What matters is that SLAM, waypoint storage, odometry injection, and setpoint generation use a consistent and continuously tracked transform.

### Initialization and alignment

At startup, record the transform between the SLAM map and the vehicle’s local frame. Alignment may use:

- the pose at arming or takeoff;
- a known fiducial or surveyed reference point;
- initial vehicle heading;
- overlapping GPS and SLAM data when GPS is available;
- a calibrated camera-to-body transform.

The transform must include translation and rotation. A translation-only offset is insufficient if the SLAM axes and autopilot axes are not aligned.

---

## 4. Guidance strategy selection

The system should choose the highest-level safe control interface available.

| Priority | Conditions | Navigation method | SLAM’s role |
|---|---|---|---|
| A | Reliable global position and mission support | MAVLink global mission | Monitor, map, cross-check, and enhance |
| B | External odometry fusion and local position control | SLAM `ODOMETRY` plus local position targets | Primary navigation sensor |
| C | Valid onboard local position and local position control | Local position targets using onboard estimate | Independent monitoring and waypoint transformation |
| D | Velocity control is viable but position control is not | External SLAM position loop producing velocity targets | Primary external position controller |
| E | Only attitude targets are viable | External position and velocity loops producing attitude/thrust | Primary external navigation controller |
| F | No safe compatible interface | No autonomous waypoint navigation | Telemetry and observation only |

A practical selection policy is:

```text
valid global position + mission support
    → GLOBAL MISSION

else if external odometry fusion + local position control
    → SLAM ODOMETRY + LOCAL POSITION TARGET

else if onboard local position valid + local position control
    → LOCAL POSITION TARGET

else if velocity control is viable
    → SLAM POSITION LOOP + VELOCITY TARGET

else if attitude control is viable and explicitly supported/configured
    → SLAM POSITION/VELOCITY LOOP + ATTITUDE TARGET

else
    → NO AUTONOMOUS WAYPOINT NAVIGATION
```

---

## Strategy A: use an onboard global mission

When the aircraft has a trustworthy global estimate and robust mission support, upload conventional waypoints using:

```text
MISSION_ITEM_INT
MAV_CMD_NAV_WAYPOINT
```

Then monitor mission acknowledgements, progress, position, and failsafe state.

This is normally the most resilient option because waypoint sequencing and fast control loops remain onboard. Temporary loss of the off-drone system or network need not immediately interrupt flight.

SLAM can continue to provide:

- obstacle and map information;
- localization cross-checking;
- mission supervision;
- local waypoint refinement;
- evidence that GPS and visual estimates are diverging.

Use this strategy only when global waypoints can be correctly registered to the SLAM map.

---

## Strategy B: inject SLAM odometry and command local positions

This is the preferred GPS-denied architecture when the vehicle supports it.

Send the external SLAM estimate to the autopilot using `ODOMETRY`, including as much reliable information as is available:

- position;
- orientation;
- linear velocity;
- angular velocity;
- pose covariance;
- velocity covariance;
- timestamp;
- estimator reset counter;
- quality.

The autopilot fuses that data into its estimator, then uses its normal position, velocity, attitude, and motor-control loops.

```text
SLAM
 │
 │ ODOMETRY
 ▼
Autopilot estimator
 │
 │ fused local state
 ▼
Autopilot position controller
 ▲
 │ SET_POSITION_TARGET_LOCAL_NED
 │
Waypoint manager
```

This creates a clean division of responsibility:

- the SLAM system provides localization;
- the waypoint manager provides goals;
- the flight controller provides stabilization and fast control.

For ArduPilot, external-navigation fusion requires appropriate EKF source configuration. PX4 likewise requires configuration selecting which external-vision position, velocity, height, and yaw measurements its estimator should fuse. Receiving an `ODOMETRY` packet does not itself prove that fusion is active.

### Local position targets

Use `SET_POSITION_TARGET_LOCAL_NED` for local position, velocity, acceleration, yaw, or yaw-rate targets. In a conventional NED frame:

```text
x = 10
y = 3
z = -5
```

means approximately 10 m along local X/north, 3 m along local Y/east, and 5 m above the origin.

ArduPilot Copter supports several useful frames in Guided mode, including:

- `MAV_FRAME_LOCAL_NED` — relative to a fixed local origin;
- `MAV_FRAME_LOCAL_OFFSET_NED` — offset from the current vehicle position;
- `MAV_FRAME_BODY_NED` — body-aligned velocity/acceleration behavior;
- `MAV_FRAME_BODY_OFFSET_NED` — offset from current position and heading.

PX4’s supported offboard frame combinations are more restrictive and should be handled through its adapter. When a relative frame is unavailable, calculate an absolute local target in the application:

```python
target_x = current_x + delta_x
target_y = current_y + delta_y
target_z = current_z + delta_z
```

For waypoint routes, prefer fixed coordinates in the canonical SLAM map over chained “move relative to the previous waypoint” instructions. Fixed targets keep the route definition stable even though the estimator itself can still drift.

---

## Strategy C: use the vehicle’s onboard local estimate

Some vehicles already maintain reliable local position without GPS using:

- optical flow and a rangefinder;
- onboard VIO;
- lidar;
- UWB;
- visual positioning;
- motion capture.

If the vehicle publishes a valid `LOCAL_POSITION_NED` and accepts local position targets, it may be safer and simpler to use its estimate rather than inject the off-drone SLAM output.

Establish the transform:

```text
T_slam_map_to_vehicle_local
```

Then transform each SLAM-map waypoint into the aircraft’s local coordinate frame before sending it.

SLAM remains useful for monitoring and mapping. Differences between SLAM and the onboard estimate should be tracked, but the two estimates should not be mixed informally. Either fuse them through a supported estimator interface or clearly select which estimate closes each control loop.

---

## Strategy D: use SLAM to generate velocity targets

When external odometry fusion is unavailable or inappropriate, but the autopilot provides a reliable velocity controller, the application can close the position loop externally.

```text
SLAM position
      │
      ▼
waypoint error
      │
      ▼
external position controller
      │
      ▼
desired vx, vy, vz
      │
SET_POSITION_TARGET_LOCAL_NED
      │
      ▼
aircraft velocity controller
```

A minimal conceptual controller is:

```python
error = waypoint - slam_position
velocity_command = clamp(Kp * error, max_speed)
```

A real implementation should also include:

- acceleration and jerk limits;
- independent horizontal and vertical gains;
- braking-distance logic;
- velocity-frame transformations;
- stale-pose detection;
- command saturation;
- integral-windup avoidance if integral control is used;
- arrival hysteresis and dwell time.

This approach is more latency-sensitive than Strategy B because the position loop runs off the vehicle, but the autopilot still handles velocity, attitude, and motor stabilization.

---

## Strategy E: use attitude targets only as an advanced fallback

`SET_ATTITUDE_TARGET` can be used when the vehicle exposes attitude/thrust control but no suitable position or velocity interface.

The external system then implements:

```text
SLAM position
    ↓
position controller
    ↓
desired velocity
    ↓
velocity controller
    ↓
desired roll/pitch/thrust
    ↓
SET_ATTITUDE_TARGET
```

This substantially increases system responsibility and risk. Network latency, packet loss, SLAM stalls, controller tuning, thrust normalization, mass changes, and frame errors become much more consequential.

Support this only as an explicitly configured advanced mode. Do not use direct actuator or motor control as a generic navigation fallback.

---

## 5. Local waypoint sequencing

Standard MAVLink missions are primarily global/geographic on ArduPilot and PX4. Portable local waypoint navigation should therefore be sequenced by the companion or offboard application.

A simple manager repeatedly publishes the current target and advances only after arrival criteria are satisfied:

```python
waypoints = [
    Waypoint(0.0, 0.0, 3.0),
    Waypoint(10.0, 3.0, 3.0),
    Waypoint(15.0, 8.0, 3.0),
    Waypoint(8.0, 12.0, 3.0),
]

current_index = 0

while navigation_active:
    pose = slam_pose_buffer.latest_valid()
    target = waypoints[current_index]

    backend.publish_target(target, pose)

    if has_arrived(pose, target):
        current_index += 1
        if current_index == len(waypoints):
            backend.finish_mission()  # hover, land, or another configured action
```

Arrival should normally consider more than distance:

- 3-D or horizontal/vertical position tolerance;
- maximum velocity;
- required dwell time;
- target visibility and SLAM health;
- the vehicle estimator’s position, where applicable;
- overshoot or corridor constraints.

---

## 6. Handle SLAM resets, loop closures, and uncertainty

SLAM estimates can jump after relocalization or loop closure:

```text
before loop closure: x = 24.7 m
after loop closure:  x = 23.2 m
```

That does not mean the drone physically teleported.

Use the reset-counter fields provided by `ODOMETRY` and related vision messages so the receiving estimator can distinguish a coordinate reset from normal motion.

The coordinate manager must also decide how a map correction affects active waypoints:

- waypoints fixed in the corrected map should remain fixed in map coordinates;
- transforms into the controller’s local frame may need to change;
- large discontinuities may require pausing guidance and re-establishing alignment;
- the vehicle should not chase a suddenly shifted target at full speed.

Publish realistic covariance. Floating-point output does not imply centimetre-level accuracy. Covariance and quality should degrade when there is:

- low texture or poor illumination;
- motion blur;
- rapid rotation;
- partial tracking loss;
- insufficient parallax;
- delayed video;
- map ambiguity;
- relocalization uncertainty.

---

## 7. Account for off-drone latency

The complete pose delay may include:

```text
camera exposure
      ↓
video encoding
      ↓
radio/network transport
      ↓
video decoding
      ↓
SLAM processing
      ↓
MAVLink command return path
```

Track at least:

```python
@dataclass
class PoseEstimate:
    pose: Pose3D
    velocity: Vector3
    capture_timestamp: float
    estimate_timestamp: float
    receive_timestamp: float
    covariance: Matrix
    quality: int
    reset_counter: int
```

The navigation system should calculate pose age and reject or degrade control when it becomes stale.

Suggested health states are:

```text
HEALTHY   → normal speed and control mode
DEGRADED  → reduced speed and tighter limits
STALE     → hold, brake, or transfer to onboard failsafe
LOST      → execute the preconfigured loss-of-localization action
```

Position targets closed onboard tolerate delay much better than external velocity or attitude loops. This is another reason to prefer `ODOMETRY` fusion plus local position targets.

---

## 8. Stream setpoints and supervise connection health

For portability, implement a continuous setpoint publisher rather than relying on one-shot targets.

PX4 Offboard mode requires a continuing proof-of-life/setpoint stream and exits Offboard when it is lost. A generic implementation might publish the active setpoint at roughly 5–20 Hz, subject to the selected autopilot’s requirements.

The publisher should be independent from waypoint sequencing:

```text
Waypoint manager changes current_target occasionally
                     │
                     ▼
Setpoint publisher emits current_target continuously
```

Monitor:

- MAVLink connection heartbeat;
- command acknowledgements;
- estimator validity;
- SLAM pose age and quality;
- current flight mode;
- arming state;
- local/global position telemetry;
- battery and system health;
- geofence and failsafe indications.

Never infer support by sending a movement command and watching what happens. Capability tests should be non-moving, configuration-aware, and performed before arming whenever possible.

---

## 9. GPS and SLAM transitions

Keep the SLAM world running even when GPS is good. This permits graceful comparison and transition:

```text
GPS good
   ↓
GPS + SLAM cross-checking
   ↓
GPS degrading
   ↓
SLAM / ExternalNav
   ↓
GPS returns and stabilizes
   ↓
GPS + SLAM cross-checking
```

Estimator-source switching is autopilot-specific. It belongs inside the ArduPilot or PX4 adapter rather than in generic navigation code.

Use hysteresis and sustained health evidence. Do not switch because of one missing GPS packet or one poor SLAM frame. Before switching frames or estimators, verify that their positions, headings, velocities, and timestamps are sufficiently aligned.

---

## 10. MAVLink, video, and cameras

MAVLink supports camera discovery and control, but normally does **not** transport the video frames themselves.

A MAVLink camera component can provide:

- `CAMERA_INFORMATION`;
- `VIDEO_STREAM_INFORMATION`;
- stream resolution and frame rate;
- bitrate and encoding information such as H.264 or H.265;
- a URI for an RTSP, RTP/UDP, TCP MPEG, MPEG-TS, WHEP/WebRTC, or similar stream;
- commands for capture, recording, zoom, focus, tracking, and stream management.

The normal architecture is:

```text
MAVLink
    ├── camera discovery and control
    ├── start/stop streaming
    └── video stream URI
                 │
                 ▼
        separate video transport
                 │
                 ▼
                SLAM
```

The video pipeline and MAVLink telemetry should share an explicit time model. Camera timestamps, clock synchronization, decoding delay, and frame drops affect SLAM quality and control latency.

---

## 11. MAVLink and lidar

MAVLink supports processed range and obstacle data, but it is not intended as a high-bandwidth raw point-cloud transport.

Relevant messages include:

- `DISTANCE_SENSOR` for individual range measurements;
- `OBSTACLE_DISTANCE` for arrays of obstacle ranges;
- `OBSTACLE_DISTANCE_3D` in the ArduPilot dialect for a 3-D obstacle vector.

A good lidar architecture is:

```text
raw lidar point cloud
        │
        ├────────► SLAM / mapping system
        │
        └────────► obstacle processor
                         │
                         ▼
              OBSTACLE_DISTANCE and/or
                  DISTANCE_SENSOR
                         │
                         ▼
                    autopilot
```

Transport raw lidar using a separate high-bandwidth protocol such as UDP, TCP, ROS/ROS 2, or a vendor-specific stream. Use MAVLink for navigation-relevant summaries and autopilot integration.

---

## 12. Example normalized vehicle profile

Capability discovery should produce a human-readable record that can be logged and inspected before flight:

```yaml
vehicle:
  type: multirotor
  autopilot: ardupilot
  firmware: 4.x

state:
  attitude: valid
  velocity_xy: valid
  local_xy: valid
  global_xy: invalid
  gps_fix: none

interfaces:
  external_odometry: yes
  local_position_target: yes
  global_position_target: yes
  velocity_target: yes
  attitude_target: yes
  mission_int: yes

selected:
  localization: external_slam
  guidance: local_position

health_limits:
  max_pose_age_ms: 250
  minimum_slam_quality: configured_per_system
  offboard_setpoint_rate_hz: 10
```

The exact thresholds must be established through simulation, controlled testing, and the latency characteristics of the real video link.

---

## 13. Recommended software modules

```text
VideoInput
    Supplies timestamped frames and stream health

SlamProvider
    Produces pose, velocity, covariance, reset counter, and quality

TransformManager
    Maintains SLAM ↔ local ↔ global ↔ body transforms

VehicleDiscovery
    Collects heartbeat, version, capability, and system metadata

VehicleStateMonitor
    Tracks estimator validity, sensor health, flight mode, and failsafes

AutopilotAdapter
    Implements ArduPilot, PX4, or conservative generic behavior

StrategySelector
    Chooses the highest-level safe guidance backend

NavigationBackend
    GlobalMissionBackend
    ExternalOdomLocalPositionBackend
    OnboardLocalPositionBackend
    VelocityBackend
    AttitudeBackend

WaypointManager
    Sequences waypoints and applies arrival rules

SetpointPublisher
    Continuously streams the current target

SafetySupervisor
    Enforces pose-age, quality, link, estimator, speed, and mode limits

FlightRecorder
    Logs source data, transforms, decisions, commands, and acknowledgements
```

Each strategy backend should declare both static requirements and live health requirements. A strategy is usable only when both are satisfied.

---

## 14. Safety and fallback policy

Failsafe behavior must exist independently of the off-drone Python process. Depending on the vehicle and environment, configure an appropriate onboard action for:

- loss of MAVLink control stream;
- loss or staleness of SLAM;
- estimator failure;
- GPS loss;
- low battery;
- geofence breach;
- radio or network loss;
- excessive disagreement between onboard and external estimates.

“Return to launch” is not automatically safe in a GPS-denied building. The correct action may instead be:

- brake and hover;
- land immediately;
- climb or descend to a known safe altitude;
- follow a prevalidated local contingency route;
- transfer control to a human pilot.

The action must be selected for the operating environment and validated on the actual autopilot.

Test in stages:

1. Software-in-the-loop simulation.
2. MAVLink command and frame validation with motors disabled.
3. Restrained or protected low-energy testing where appropriate.
4. Low-altitude hover and single-axis motion.
5. Small local waypoints at low speed.
6. Deliberate SLAM degradation, network delay, and packet-loss tests.
7. GPS/SLAM transition tests where supported.
8. Full route testing only after all fallback actions are confirmed.

---

## Recommended implementation order

1. Implement `HEARTBEAT` and `AUTOPILOT_VERSION` discovery.
2. Build normalized vehicle and estimator-health models.
3. Implement the SLAM-to-NED/FRD transform manager with unit tests.
4. Record timestamps, covariance, quality, and SLAM reset events.
5. Add ArduPilot and PX4 adapters with explicit supported-interface profiles.
6. Implement Strategy B: external `ODOMETRY` plus local position targets.
7. Implement Strategy C: onboard local position targets.
8. Implement Strategy A: global missions for GPS-capable vehicles.
9. Add the velocity-control fallback.
10. Add attitude control only if a real target vehicle requires it.
11. Build the safety supervisor and fault-injection test suite alongside every strategy.

The most important part to make correct first is the **capability, estimator-health, coordinate-transform, and strategy-selection layer**. Once those boundaries are stable, adding another supported drone becomes primarily an adapter task rather than a rewrite of the SLAM or waypoint system.

---

## Reference documentation

- [MAVLink Common Message Set](https://mavlink.io/en/messages/common.html)
- [MAVLink Offboard Control Interface](https://mavlink.io/en/services/offboard_control.html)
- [MAVLink Camera Protocol](https://mavlink.io/en/services/camera.html)
- [ArduPilot Copter Guided-Mode Commands](https://ardupilot.org/dev/docs/copter-commands-in-guided-mode.html)
- [ArduPilot Non-GPS Position Estimation](https://ardupilot.org/dev/docs/mavlink-nongps-position-estimation.html)
- [ArduPilot Non-GPS Navigation Overview](https://ardupilot.org/copter/docs/common-non-gps-navigation-landing-page.html)
- [PX4 Offboard Mode](https://docs.px4.io/main/en/flight_modes/offboard.html)
- [PX4 Visual-Inertial Odometry](https://docs.px4.io/main/en/computer_vision/visual_inertial_odometry.html)
- [PX4 EKF2 External Vision Configuration](https://docs.px4.io/main/en/advanced_config/tuning_the_ecl_ekf.html)

> **Safety note:** Autonomous flight software is safety-critical. MAVLink support, estimator configuration, coordinate frames, and failsafe behavior vary by vehicle and firmware version. Confirm the current documentation for each target platform and validate all behavior in simulation and controlled tests before flight.
