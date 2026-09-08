# `motion.imu` -- inertial measurement unit

Angular rate and specific force in the sensor's own axes, derived from
successive physics snapshots. It is observational: nothing it publishes feeds
back into the flight estimator, which keeps its existing validity controls.

Shared scheduling, generation and determinism rules are in
[`README.md`](README.md); the record envelope and the shared ring are in
[`docs/modcom.md`](../modcom.md).

## Profile fields

`id`, `type`, `enabled`, `rate_hz`, `parent` and `pose_parent` follow the
common rules in [`camera-rgb.md`](camera-rgb.md). An IMU is fully pose-sensitive: its `pose_parent` rotation is what its axes *are*, so mounting one upside down inverts what it reads.

| `model` field | Default | Validation |
|---|---|---|
| `gyro_noise_std_dps` | `0.05` | >= 0; white noise per sample, degrees/s |
| `accel_noise_std_mps2` | `0.05` | >= 0; white noise per sample, m/s^2 |
| `gyro_bias_dps` | `0.0` | constant offset added to every axis |
| `accel_bias_mps2` | `0.0` | constant offset added to every axis |

Any other `model` key is rejected by name.

```json
{
  "id": "imu", "type": "motion.imu", "enabled": true,
  "rate_hz": 100.0, "parent": "body",
  "model": {"gyro_noise_std_dps": 0.05, "accel_noise_std_mps2": 0.05}
}
```

## Axes and sign conventions

Axes are the sensor's, which are the body's through its mount: **+X forward,
+Y right, +Z up**. The three angular-rate signs follow the public angle
conventions exactly:

| Reading | Positive means |
|---|---|
| `angular_rate_dps.x` | roll rate, right wing dropping |
| `angular_rate_dps.y` | pitch rate, nose rising |
| `angular_rate_dps.z` | yaw rate, turning right |

**Gravity.** `specific_force_mps2` is *proper acceleration* -- what an
accelerometer actually reads -- not the vehicle's acceleration. At rest and
level the Z axis reads **+9.80665**, not zero and not -g; in free fall all
three axes read zero. The published `gravity_mps2` field states the constant
used.

## Simulation

Angular rate is measured rather than differentiated from Euler angles: the
sensor's rotation matrix is resolved at both snapshots and the rotation
between them is read off directly. That is why it stays correct under
combined roll and pitch, where the three Euler rates do not equal the three
body rates. Specific force is the world-frame velocity difference over the
timestep, minus gravity, rotated into the sensor's axes.

The interval is one physics step, so an IMU slower than `physics_hz` reports
the instantaneous rate at its sample instant rather than an average over its
own period -- which is what a real instrument does. The very first sample
after a start, a reset or a new generation has no predecessor to difference
and reports zero rate and gravity alone.

## Transport

One `imu.sample.v1` record (payload type 5) per capture on the **shared**
compact ring -- no channel of its own. State-sensor records carry the
measurement and nothing else: no pose and no vehicle datum. The axes a consumer needs are static configuration, already in the manifest's `pose_parent`.

## Scheduling, lifecycle, and health

`rate_hz` must divide `physics_hz` exactly, so every sample lands on a physics
snapshot. Applying a profile while disarmed publishes a new generation; a
drone reset increments `reset_epoch`, restarts the cadence and the noise
stream, and leaves sequences and simulated time alone. Per-sensor production
counts -- scheduled, published, invalid, drops, configured and observed rate --
appear in the run summary and in the simulator's Pipeline view.

## Limitations

**The lever arm is not modelled**: an IMU mounted away from the centre
of rotation does not see the centripetal or tangential acceleration that
offset would produce. Bias is a constant, not a random walk, and there is no
warm-up, no scale-factor or cross-axis error, no temperature dependence, no
vibration or rotor harmonics, and no saturation. There is no magnetometer or
barometer inside it: those are separate sensors in this profile.

## Correctness tests

`tests/test_sensor_state.py` -- one g upward at rest, zero in free fall, forward acceleration on the
forward axis, each rate sign against its angle convention in both directions,
an inverted mount inverting the reading, and the first sample having nothing
to differentiate.
`tests/test_sensor_release.py` -- the sensor is present in the reference
profile and produces every configured sample inside the run budget.
