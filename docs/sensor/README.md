# Sensor documents

This directory holds one focused document per simulated sensor type. Each
document records the profile fields, physical model, transport, scheduling,
limitations, and correctness tests for its sensor. Shared contracts —
discovery, the sample-record envelope, timestamps, generations, and the
transform convention — live once in [`docs/modcom.md`](../modcom.md),
[`docs/clock.md`](../clock.md), and `DV-SENSORS.md` and are linked, not
copied. A sensor type is not complete until its document and tests land with
it.

Profile `type` values map to documents as follows:

| Profile `type` | Document | Status |
|---|---|---|
| `camera.rgb` | [`camera-rgb.md`](camera-rgb.md) | Implemented |
| `lidar.scan2d` | [`lidar-scan2d.md`](lidar-scan2d.md) | Implemented |
| `lidar.range_image` | [`lidar-range-image.md`](lidar-range-image.md) | Implemented |
| `range.infrared` | [`range-infrared.md`](range-infrared.md) | Implemented |
| `range.ultrasonic` | [`range-ultrasonic.md`](range-ultrasonic.md) | Implemented |
| `range.laser` | [`range-laser.md`](range-laser.md) | Implemented |
| `position.gnss` | [`gnss.md`](gnss.md) | Implemented |
| `motion.imu` | [`imu.md`](imu.md) | Implemented |
| `altimeter.barometric` | [`barometer.md`](barometer.md) | Implemented |
| `heading.magnetometer` | [`magnetometer.md`](magnetometer.md) | Implemented |
| `environment.temperature` | [`temperature.md`](temperature.md) | Implemented |
| `mount.fixed`, `rig.fixed`, `mount.ptz` | transform nodes, not sensors; see [`mounts.md`](mounts.md) | Implemented |

The profile loader rejects any other `type` with a field-specific error.
Optical flow, UWB, radar, airspeed, RTK correction streams, dual-GNSS heading
and visual odometry are deliberately schema extensions rather than v1 sensors;
`DV-SENSORS.md` records why.

The two groups behave differently and the difference is worth stating once.
The **ray sensors** -- both LiDAR outputs and the three `range.*` types -- read
the scene through one shared geometry service, and their rules are below. The
**state sensors** -- GNSS, IMU, barometer, magnetometer and temperature -- read
the vehicle and the environment instead: no ray casting, no renderer, and no
pose in their records, because a measurement published beside the truth it was
supposed to measure is not a measurement. Their shared rules are under
"What the state sensors share".

## What the ray sensors share

`lidar.scan2d`, `lidar.range_image` and the three `range.*` types are the same
machine with different rays and different reductions. Their common parts are
documented once here and referenced, not repeated, by each sensor:

- **Geometry.** [`apps/dsim/range.py`](../../apps/dsim/range.py) casts every
  ray. Obstacle footprints and heights come from the constants the collision
  test uses, so "the sensor reports clear" and "the physics reports solid"
  cannot disagree. Walls, trees and the ground plane are all surfaces, and a
  box has a top: a rangefinder above a wall reads the wall.
- **Rays.** Each sensor generates its rays in its own forward/right/up axes
  and [`transforms.resolve`](../../apps/dsim/transforms.py) rotates them by the
  composed pose, so mounts, body attitude and fixed misalignment all reach the
  geometry through one path.
- **Error model.** [`apps/dsim/sensor_models.py`](../../apps/dsim/sensor_models.py)
  applies, in this order and per ray: Gaussian noise of
  `noise_std_m · scale`, quantization to `quantization_m`, a dropout draw
  against `dropout_probability · scale`, then the `min_range_m`/`max_range_m`
  gate, where `scale = 1 + limit_degradation · (range / max_range_m)`. One
  normal and one uniform are drawn for **every** ray, whether or not it
  returned anything, so a miss cannot shift a later ray's error.
- **Confidence.** `range_linear` gives `255 · (1 − range / max_range_m)`;
  `exact` gives 255 for every valid return. Invalid is always 0.
- **Determinism.** The generator for one capture is
  `capture_rng(run_seed, sensor_id, reset_epoch, logical_index)`: SHA-256 over
  a versioned encoding of exactly those four values, used as the seed sequence
  of a NumPy PCG64. Two sensors cannot collide, a replayed capture cannot
  drift, and a skipped publication does not shift the next one. The scheme tag
  is `dvision2.sensor-noise.v1`; changing the encoding, the digest or the
  generator is a contract change with a frozen test vector.
- **Truth is geometry only.** No appearance preset, lighting change or texture
  reaches a range measurement, and LiDAR v1 is material-independent:
  reflectivity, multiple returns, motion distortion during a scan, weather
  attenuation and transparent surfaces are all deliberately absent.

## What the state sensors share

- **Two noise sources, kept apart.** Slowly-wandering environmental state -- a
  GNSS fix drifting, a barometer warming up -- lives in
  [`apps/dsim/realism.py`](../../apps/dsim/realism.py), evolves on every
  physics tick, and is *read* by the sensor rather than re-rolled, so one
  wander reaches vehicle status and a sensor record as the same number. Per
  sample white noise is drawn from the logical capture's own generator, the
  same scheme the ray sensors use. Where a sensor has both a hardware figure
  and a realism figure for the same quantity they are independent sources and
  combine in quadrature.
- **Hardware, conditions and estimate are three separate decisions.** The
  profile decides what is fitted, realism decides how well it works, and
  `est.*` decides whether the fused navigation estimate is trusted. GNSS is
  where this matters most: "no receiver", "no fix" and "fix rejected" are
  three different failures with three different causes.
- **Observation only.** Nothing a state sensor publishes feeds back into the
  physics or the flight estimator in v1. Raw samples supplement fused vehicle
  telemetry; they do not replace it, and closing the simulated estimator loop
  is a separate project.
- **Records carry the measurement and nothing else.** No pose, no vehicle
  datum. The axes a consumer needs are static configuration it already has
  from the manifest's `pose_parent`.

## Reference budget

`assets/drone_profiles/stereo-nav-and-proximity.json` is the reference
vehicle: a PTZ-mounted 640x480 stereo pair at 30 Hz, a 720-ray 2D LiDAR and a
64x48 range image at 10 Hz, three rangefinders, and all five state sensors
including a 100 Hz IMU. 12 sensors, 306 compact records a second, 59.6 MiB of
shared memory against a 256 MiB ceiling.

A 60-second run at real time on the reference machine below produced **60.00 s
of simulated time in 60.74 s of wall time**: achieved pace mean 1.000, minimum
0.983; every one of the 12 sensors published 100% of its configured samples;
zero publisher drops and zero compact-ring overruns. The gate is 0.90x pace
and 90% of samples. Rendering the stereo pair cost 15.3 ms mean per capture and
all ray and state sampling together cost 3.5 s of the 60.

```text
reference machine: 12th Gen Intel Core i9-12900H, 20 threads, 31 GiB
                   Mesa Intel Iris Xe Graphics (ADL GT2), OpenGL via GLX
                   Linux 7.1.5 x86_64, Python 3.13.14, NumPy 2.3.2
```

`tests/test_sensor_release.py` runs the same measurement two ways: a
reduced-resolution copy of the reference profile in the normal suite, and the
committed profile for the full 60 seconds under
`DVISION_NIGHTLY=1 pytest -m nightly`.
