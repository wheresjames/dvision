# `environment.temperature` -- ambient air temperature

Air temperature at the vehicle, in Celsius. The cheapest sensor in the
set, and it is here for a second reason: it proves the shared compact ring
carries a plain scalar without needing a channel, a status key or a schema of
its own.

Shared scheduling, generation and determinism rules are in
[`README.md`](README.md); the record envelope and the shared ring are in
[`docs/modcom.md`](../modcom.md).

## Profile fields

`id`, `type`, `enabled`, `rate_hz`, `parent` and `pose_parent` follow the
common rules in [`camera-rgb.md`](camera-rgb.md). Not pose-sensitive, apart from height: the sensor's altitude comes from the vehicle, and self-heating and airflow are not modelled.

| `model` field | Default | Validation |
|---|---|---|
| `noise_std_c` | `0.1` | >= 0; white noise per sample, Celsius |
| `quantization_c` | `0.01` | >= 0; reporting grid, `0` disables |
| `lapse_rate_c_per_m` | `0.0065` | >= 0; cooling with height, the ISA rate |

Any other `model` key is rejected by name.

```json
{
  "id": "ambient", "type": "environment.temperature", "enabled": true,
  "rate_hz": 1.0, "parent": "body",
  "model": {"noise_std_c": 0.1}
}
```

## Quantity and where the value comes from

The ground-level temperature is a **realism** setting, `--ambient-temp-c`
(default 20 C), because it is a condition rather than a property of the part.
The sensor cools with height at its configured lapse rate:

```text
temperature_c = ambient_temp_c - lapse_rate_c_per_m * altitude_m + noise
```

Over a twenty-metre flight that is about a tenth of a degree. It is included
because it is free and correct, not because it matters at this scale.

## Transport

One `temperature.sample.v1` record (payload type 8) per capture on the **shared**
compact ring -- no channel of its own. State-sensor records carry the
measurement and nothing else: no pose and no vehicle datum. There is no truth worth restating beside a one-number measurement.

## Scheduling, lifecycle, and health

`rate_hz` must divide `physics_hz` exactly, so every sample lands on a physics
snapshot. Applying a profile while disarmed publishes a new generation; a
drone reset increments `reset_epoch`, restarts the cadence and the noise
stream, and leaves sequences and simulated time alone. Per-sensor production
counts -- scheduled, published, invalid, drops, configured and observed rate --
appear in the run summary and in the simulator's Pipeline view.

## Limitations

No self-heating, no airflow or radiation error, no humidity, no
thermal lag, and no weather: the ground-level value is constant for a run.
The lapse rate is dry-adiabatic and applied to height above the map ground
rather than above sea level.

## Correctness tests

`tests/test_sensor_state.py` -- the value falling with height from the environment setting, and a
sample depending on its capture index rather than on call order.
`tests/test_sensor_release.py` -- the sensor is present in the reference
profile and produces every configured sample inside the run budget.
