# `position.gnss` -- GNSS receiver

A satellite navigation fix: position, ground velocity and fix quality.
Three separate things decide what a client sees, and the whole point of this
sensor is that they stay separate.

- **Installation** is the profile's. A vehicle with no `position.gnss` sensor
  publishes no records at all *and* reports no fix in vehicle status: the
  simulator tells the environment what hardware is fitted when it applies a
  profile.
- **Fix quality** is the environment's, through `--gps` and `--gps-noise-m`.
  A fitted receiver that cannot see the sky reports `fix_type` 0 with an
  invalid record.
- **Estimator validity** is `est.global_position_valid`, which a fault can
  clear while the receiver is still reporting a perfect fix.

"Not fitted", "fitted with no fix" and "fitted but rejected" are three
different failures, and a client that cannot tell them apart cannot be tested
against any of them.

Shared scheduling, generation and determinism rules are in
[`README.md`](README.md); the record envelope and the shared ring are in
[`docs/modcom.md`](../modcom.md).

## Profile fields

`id`, `type`, `enabled`, `rate_hz`, `parent` and `pose_parent` follow the
common rules in [`camera-rgb.md`](camera-rgb.md). A receiver has no pose sensitivity in v1: the antenna lever arm is not modelled, so `pose_parent` records where the antenna is without changing what it reports.

The model has no fields. Everything that varies is either the profile's
`rate_hz` or a realism setting, because a receiver's accuracy is a property of
the sky rather than of the part.

Any other `model` key is rejected by name.

```json
{
  "id": "gnss", "type": "position.gnss", "enabled": true,
  "rate_hz": 10.0, "parent": "body"
}
```

## Quantity, axes, and validity

`lat_deg`, `lon_deg` and `alt_m` are WGS-84 degrees and metres above mean sea
level, from the map origin the simulator was launched with. `error_north_m`,
`error_east_m` and `error_up_m` are the position error currently in the fix,
so a test can recover the truth without reading vehicle status.

Velocity is over the ground -- the vehicle's velocity through the air plus the
wind carrying it -- in the north/east/down convention every GNSS receiver
uses. Map Y is south-positive and map Z is up, so `vel_north_mps` is `-vy` and
`vel_down_mps` is `-vz`.

When there is no fix the envelope status is `0` and every measured field is
`null`: `lat_deg`, `lon_deg`, `alt_m`, all three velocities and all three
errors. `fix_type`, `satellites`, `hdop` and `vdop` still report, because "no
fix, zero satellites" is itself the measurement.

## Simulation

The fix quality comes from the realism GPS mode (`off`, `degraded`, `good`,
`rtk`), which sets `fix_type`, `satellites`, `hdop` and `vdop`. The position
error is the same slow correlated wander vehicle status publishes -- read, not
re-rolled -- so a client cannot get a better position by reading the sensor
instead of the fused telemetry. There is no additional per-sample white noise.

## Transport

One `gnss.sample.v1` record (payload type 4) per capture on the **shared**
compact ring -- no channel of its own. State-sensor records carry the
measurement and nothing else: no pose and no vehicle datum. A GNSS record beside the true position would not be a measurement.

## Scheduling, lifecycle, and health

`rate_hz` must divide `physics_hz` exactly, so every sample lands on a physics
snapshot. Applying a profile while disarmed publishes a new generation; a
drone reset increments `reset_epoch`, restarts the cadence and the noise
stream, and leaves sequences and simulated time alone. Per-sensor production
counts -- scheduled, published, invalid, drops, configured and observed rate --
appear in the run summary and in the simulator's Pipeline view.

## Limitations

No lever arm, no multipath, no satellite geometry, no ionospheric
model, and no RTCM or RTK correction stream: `rtk` is a noise figure, not a
simulated base station. Time is the simulator's, not GPS time, and there is no
week number or leap-second handling. Velocity comes from the physics rather
than from Doppler, so it does not degrade independently of position.

## Correctness tests

`tests/test_sensor_state.py` -- the fix carrying the same error vehicle status publishes, a missing
fix reporting `null` rather than a wrong position, ground velocity including
the wind, and installation, fix and estimator validity as three separate
answers.
`tests/test_sensor_release.py` -- the sensor is present in the reference
profile and produces every configured sample inside the run budget.
