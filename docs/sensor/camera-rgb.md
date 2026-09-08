# `camera.rgb` — variable-FOV RGB camera

Simulated pinhole RGB camera rendered from the configured scene through
Panda3D. A profile may contain any number of them within its resource limits,
and one enabled instance must be the profile's `primary_camera` — the camera a
single-camera consumer opens. Cameras that share a `sync_group` are captured
together from one vehicle state, which is how a stereo rig is expressed.

## Profile fields

A camera is one entry of the profile's flat `sensors` array:

| Field | Default | Validation |
|---|---|---|
| `id` | required | unique across mounts and sensors, `[A-Za-z0-9_.-]+`, ≤ 48 bytes, not `body` |
| `type` | required | must be `camera.rgb` |
| `enabled` | `true` | boolean; the `primary_camera` must be enabled |
| `rate_hz` | `30.0` | positive, ≤ `physics_hz`, and an integer divisor of it |
| `parent` | required | `body` or a mount id; must exist, no cycles, one path to `body` |
| `pose_parent` | identity | six numbers, metres/degrees, parent-relative |
| `sync_group` | — | nonempty string; a group needs at least two members, and they must agree on `type`, `rate_hz`, `enabled` and the whole lens `model` |

`model` fields:

| Field | Default | Validation |
|---|---|---|
| `width_px`, `height_px` | `640`, `480` | positive integers |
| `fov_h_deg` **or** `fx_px` | `70.0` derived | **exactly one** of the two; providing both is rejected, providing neither is rejected |
| `fy_px` | = `fx_px` | positive |
| `cx_px`, `cy_px` | image centre | finite |
| `near_m`, `far_m` | `0.15`, `150.0` | positive, `near < far` |

The resolved profile always stores `fx_px` (and derived `fy/cx/cy`); `fov_h_deg`
is consumed at load time via `fx = width / (2·tan(fov/2))`. Because resolved
profiles retain `fx_px` rather than a redundant FOV, a saved profile re-loads
unchanged. Memory is pre-validated: each enabled camera reserves
`(ceil(rate·retention) + 2) · (width·height·3 + 128) + 4096` bytes plus the
registry and shared record ring. The default per-instance budget is 256 MiB.
The optional profile `transport` object sets positive `retention_s` (default 1)
and `memory_limit_mib` (default 256); active and staged generations must fit
the selected budget together. The same retention sizes compact, array, video,
and consumer pending caches. Both settings are editable in the Sensors tab.

The tab offers horizontal FOV even for a loaded profile expressed with `fx_px`.
Editing FOV switches the draft to that single lens parameter and updates the
resolved intrinsics immediately. Equal focal lengths stay equal; an explicitly
different vertical focal length is preserved. Camera buffers are allocated
before Apply commits discovery; allocation failure retains the previous views.
Profile name and physics cadence are also editable, and dragging components
onto body or a mount reparents the draft without allowing cycles.

Minimal example (the built-in default, resolved):

```json
{
  "schema": "dvision2.drone-profile.v1",
  "name": "default",
  "physics_hz": 300.0,
  "primary_camera": "front",
  "mounts": [],
  "sensors": [
    {
      "id": "front", "type": "camera.rgb", "enabled": true, "rate_hz": 30.0,
      "parent": "body",
      "pose_parent": {"x_m": 0.0, "y_m": 0.0, "z_m": 0.1,
                       "roll_deg": 0.0, "pitch_deg": -5.0, "yaw_deg": 0.0},
      "model": {"width_px": 640, "height_px": 480, "fov_h_deg": 70.0}
    }
  ]
}
```

The default reproduces today's historical effective camera — 640×480, 30 Hz,
70° horizontal FOV, 0.1 m up, 5° down-tilt — so default runs look unchanged.

## Stereo pairs

Stereo is not a pixel format or a sensor type: it is two ordinary cameras with
a shared `sync_group`. `dsim.profiles.stereo_pair()` (and the Sensors tab's
**Add stereo pair**) builds them, and stores the resulting **explicit left and
right poses** rather than a `baseline_m` scalar, so an asymmetric
manufacturing error has somewhere to live. The right member sits toward +Y in
body axes, so a point ahead lands further left in its image and

```text
disparity = x_left - x_right = fx · baseline / depth
```

is positive. Members are validated to share a rate, a resolution and a lens
model in v1, which is what makes that relation well-defined; only their poses
may differ. On a due tick both are rendered from one frozen snapshot and
published with the same `capture_id` and `sim_time_us`, and each still gets its
own `memvid` channel and its own `camera.frame` record. `dsim.headless`
exposes `render_group(name)` to render every member of a group in one call.

See `assets/drone_profiles/stereo-nav-and-proximity.json` for a pair mounted on
a PTZ head, which is also the profile that exercises the rest of the sensor
set.

## Frames, pose, and timestamps

- Body frame: +X forward, +Y right, +Z up; roll about +X, pitch about +Y,
  yaw about +Z. `pose_parent` is the transform **from the named parent to
  this sensor**, composed as `T_world_child = T_world_parent · T_parent_child`
  with `R = Rz(yaw)·Ry(pitch)·Rx(roll)`. See `DV-SENSORS.md` "Frames, pose,
  and misalignment" and `apps/dsim/transforms.py`.
- Camera optical axes (+X right, +Y down, +Z forward) map to sensor-body
  `(z, x, −y)` before the sensor pose is applied. The delivered image is
  row-major RGB24 with row 0 at the top and +X right, matching the published
  intrinsics directly; the mirrored Panda3D scene basis is an internal detail
  of the renderer adapter.
- The pose is pose-**sensitive**: a non-trivial `pose_parent` translation
  rotates with the vehicle (a stereo baseline must), and roll/pitch/yaw
  misalignment compose through the parent chain. `pose_parent` is the true
  mount; there is no separate reported-calibration error in v1.
- Timestamps: every capture carries `sim_time_us`, the integer simulated
  microseconds of the physics snapshot it was rendered from, and a
  `capture_id`. The `memvid` presentation timestamps (video and audio) are
  set to the **same simulated value**, not wall/monotonic time.

## Simulation algorithm

Each physics tick runs a fixed `1/physics_hz` step (`DroneSimulator.step`).
After the step and command processing, the tick's **immutable state snapshot**
(`dataclasses.replace(state)`) is taken. When the camera's deadline is due
(every `physics_hz / rate_hz` ticks — an integer by validation), the provider:

1. resolves the world→sensor transform from the snapshot through the parent
   chain (`transforms.resolve`), for every camera due on this tick;
2. renders the scene from each pose into that camera's current `memvid` slot.
   Scene geometry is shared and cached; each camera owns an offscreen buffer
   sized from its own model, and only the lens and camera matrix change
   between renders. Nothing advances the world between them, so a
   synchronized group observes exactly one state;
3. sets both PTS to the snapshot's `sim_time_us` and advances each ring;
4. reads back each committed `memvid` frame sequence and immediately writes
   one `camera.frame` metadata record keyed by that sequence.

Rendered pixels read scene geometry truth (map walls/trees/ground) only;
lighting/scene presets change appearance, never geometry. No noise, bias,
quantization, dropout, or confidence model is applied to RGB in v1 — pixels are
deterministic given pose and scene, so there is no per-frame randomness to
seed. (Deterministic measurement noise rules for later numeric sensors are
specified in `DV-SENSORS.md` "Internal dsim architecture".)

## Transport

Discovery, naming, the record envelope, and the association protocol are
frozen in [`docs/modcom.md`](../modcom.md) "Sensors". Summary for this
sensor:

- Pixels: one dedicated generation-qualified `memvid` RGB24 ring
  (`...sensor.<id>.video`), slot count `ceil(rate) + 2`.
- Metadata: the shared compact record ring (`...sensor.samples`), one
  `camera.frame.v1` payload-type-1 record per capture, association key
  `(session, generation, sensor_id, video_sequence)`. Consumers admit only
  sequence-matched frames and count bounded pending-cache evictions as
  association drops.
- Intrinsics come from the manifest (`sensors[id].model`,
  `calibration_revision` = profile digest), never from vehicle status.
- A `camera.frame` record also carries `sync_group`, so a consumer can tell
  which frames were captured together without consulting the profile.

## Scheduling, lifecycle, and health

- Cadence is deadline-accumulation on physics ticks; the simulated interval
  between frames is exactly `1/rate_hz` regardless of achieved wall speed.
  Under load, `sim.speed_achieved` drops; frames are never time-stretched.
- Generation: applying a profile while disarmed recreates the camera's ring
  (if dimensions changed) and publishes the next generation; consumers reopen.
  Drone reset increments `reset_epoch`, resets the sensor tick phase, and
  keeps sequences, capture ids, and simulated time.
- Provider-side health is reported through run artifacts and the run summary:
  the resolved profile, its manifest and its memory plan are written to the
  report directory (`drone-profile.json`, `sensor-manifest.json`,
  `sensor-plan.json`), and the summary records per-sensor scheduled,
  published, invalid and dropped counts with configured and observed rates.
- Consumer-side health is separate and never derived from it: each module that
  opens this camera declares it as a required subscription and publishes its
  own observed rate, age, skipped sequences and sync state once a second on
  `module.sensor_health` (see [`docs/modcom.md`](../modcom.md)). Both sides
  appear in the simulator's Pipeline view and in the run's health report.

## Limitations (v1)

- FOV/lens changes take effect at profile apply (disarmed); live optical zoom
  is deferred.
- Stereo members must share a lens model exactly. Heterogeneous pairs and
  rectified virtual outputs are deferred until raw stereo is proven correct.
- Cameras render sequentially. Several high-resolution cameras at a high rate
  cost proportionally more per tick and will reduce `sim.speed_achieved`
  before anything else gives way.
- No exposure, motion blur, rolling shutter, chromatic aberration, or
  sensor-noise model; no reported-vs-true calibration error.
- Panda3D renders with its internal mirrored basis; only the adapter
  (`render_pose`) knows that basis. Image-orientation correctness is guarded
  by the calibration tests below, not by the renderer's internals.

## Correctness tests

- `tests/test_sensor_contract.py` — envelope byte fixture, profile
  validation, transform goldens (translation rotates with heading; roll/pitch/
  yaw +90° goldens), registry/generation rollover, provider restart,
  reset-epoch, frame/metadata matching, pending-unmatched behaviour, record
  ring overrun.
- `tests/test_dvision_calibration_render.py` — landmark fixtures prove FOV/
  intrinsics agree with rendered landmarks at every cardinal heading, channel
  order is RGB24, attitude signs move the horizon correctly.
- `tests/test_dvision_perception_chain.py`, `tests/test_daic_frame_orientation.py`
  — delivered image orientation and the DAIC perception chain on the new
  transport.
- `tests/test_sensor_stereo.py` — disparity sign and magnitude against
  `fx · baseline / depth` at several baselines, depths and headings; both
  members rendered from one state; a per-camera misalignment moving only its
  own image.
- `tests/test_dvision_scene_presets.py` — appearance presets do not change
  geometric truth.