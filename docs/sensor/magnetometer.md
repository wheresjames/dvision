# `heading.magnetometer` -- magnetic compass

A measured compass heading with its validity. The minimum useful
magnetometer: what a corrected compass would report, not a raw field vector.

Shared scheduling, generation and determinism rules are in
[`README.md`](README.md); the record envelope and the shared ring are in
[`docs/modcom.md`](../modcom.md).

## Profile fields

`id`, `type`, `enabled`, `rate_hz`, `parent` and `pose_parent` follow the
common rules in [`camera-rgb.md`](camera-rgb.md). Not pose-sensitive in v1: the published heading is the vehicle's, so mounting the sensor at an angle does not rotate the reading. A field-vector output would be, and is deferred with it.

| `model` field | Default | Validation |
|---|---|---|
| `noise_std_deg` | `0.0` | >= 0; this instrument's own white noise, degrees |

Any other `model` key is rejected by name.

```json
{
  "id": "compass", "type": "heading.magnetometer", "enabled": true,
  "rate_hz": 25.0, "parent": "body",
  "model": {"noise_std_deg": 0.4}
}
```

## Quantity and convention

`heading_deg` is a compass heading in `[0, 360)`: **0 is north and it
increases clockwise**, the same convention as `drone.heading_deg` and as a
positive yaw rate. It is magnetic and true at once, because declination is not
modelled.

## Noise

Two independent sources combined in quadrature: the realism
`--sensor-noise` profile's heading figure, and this instrument's own
`noise_std_deg`. Both are white per sample; the compass has no modelled
drift.

## Transport

One `magnetometer.sample.v1` record (payload type 7) per capture on the **shared**
compact ring -- no channel of its own. State-sensor records carry the
measurement and nothing else: no pose and no vehicle datum. A compass record beside the true heading would not be a measurement.

## Scheduling, lifecycle, and health

`rate_hz` must divide `physics_hz` exactly, so every sample lands on a physics
snapshot. Applying a profile while disarmed publishes a new generation; a
drone reset increments `reset_epoch`, restarts the cadence and the noise
stream, and leaves sequences and simulated time alone. Per-sensor production
counts -- scheduled, published, invalid, drops, configured and observed rate --
appear in the run summary and in the simulator's Pipeline view.

## Limitations

No field vector, no declination or inclination, no hard-iron or
soft-iron distortion, no interference from motor current, and no calibration
state. Publishing the three-axis field, and with it the mount sensitivity a
field vector implies, is the natural next version of this sensor.

## Correctness tests

`tests/test_sensor_state.py` -- the heading convention, and independent noise sources combining in
quadrature.
`tests/test_sensor_release.py` -- the sensor is present in the reference
profile and produces every configured sample inside the run budget.
