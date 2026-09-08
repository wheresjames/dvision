# `altimeter.barometric` -- barometric altimeter

Pressure altitude and the pressure it was derived from. Cheap, always
available, and the altitude source that still works in a GPS-denied profile --
which is most of why it is in the first sensor set.

Shared scheduling, generation and determinism rules are in
[`README.md`](README.md); the record envelope and the shared ring are in
[`docs/modcom.md`](../modcom.md).

## Profile fields

`id`, `type`, `enabled`, `rate_hz`, `parent` and `pose_parent` follow the
common rules in [`camera-rgb.md`](camera-rgb.md). Not pose-sensitive: static port position and aerodynamic pressure error are not modelled, so `pose_parent` records where the sensor is without changing what it reads.

| `model` field | Default | Validation |
|---|---|---|
| `noise_std_m` | `0.0` | >= 0; this instrument's own white noise, metres |
| `quantization_m` | `0.0` | >= 0; reporting grid, `0` disables |
| `sea_level_pressure_pa` | `101325.0` | >= 0; the datum the pressure is computed against |

Any other `model` key is rejected by name.

```json
{
  "id": "baro", "type": "altimeter.barometric", "enabled": true,
  "rate_hz": 25.0, "parent": "body",
  "model": {"noise_std_m": 0.05, "quantization_m": 0.01}
}
```

## Quantity and the pressure/altitude relationship

`altitude_m` is height above the **map's ground**, the same datum every other
altitude in this project uses -- not above mean sea level and not above the
takeoff point. `pressure_pa` is the ISA troposphere pressure at that altitude
*above mean sea level*, using the map origin's elevation
(`--origin-alt`), so the two invert each other:

```text
h_amsl  = origin_alt_m + altitude_m
pressure = sea_level_pressure_pa * (1 - 0.0065 * h_amsl / 288.15) ^ 5.25588
```

## Noise, drift, and where each part comes from

Two independent sources, combined in quadrature:

- the **environment's** slow altitude drift, a correlated process that wanders
  and returns on a 30 s time constant and is scaled by the realism
  `--sensor-noise` profile. It is *read* rather than re-rolled, so the wander
  a client sees in `drone.z_m` and the wander in a barometer record are the
  same number.
- this **instrument's** `noise_std_m`, drawn per logical capture.

Quantization is applied last, so a coarsely quantized sensor reports on its
own grid rather than on a noisy value's.

## Transport

One `barometer.sample.v1` record (payload type 6) per capture on the **shared**
compact ring -- no channel of its own. State-sensor records carry the
measurement and nothing else: no pose and no vehicle datum. An altimeter record beside the true altitude would not be a measurement.

## Scheduling, lifecycle, and health

`rate_hz` must divide `physics_hz` exactly, so every sample lands on a physics
snapshot. Applying a profile while disarmed publishes a new generation; a
drone reset increments `reset_epoch`, restarts the cadence and the noise
stream, and leaves sequences and simulated time alone. Per-sensor production
counts -- scheduled, published, invalid, drops, configured and observed rate --
appear in the run summary and in the simulator's Pipeline view.

## Limitations

No temperature compensation, no dynamic or position error from
airflow over the port, no sensor hysteresis, and no weather: the sea-level
pressure is a configured constant, so the zero does not move during a flight
the way a real QNH does. The ISA model is troposphere-only.

## Correctness tests

`tests/test_sensor_state.py` -- pressure and altitude inverting each other, the environment drift
being read rather than re-rolled, and quantization reporting on its own grid.
`tests/test_sensor_release.py` -- the sensor is present in the reference
profile and produces every configured sample inside the run budget.
