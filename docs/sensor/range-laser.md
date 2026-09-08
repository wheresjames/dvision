# `range.laser` — laser / time-of-flight rangefinder

A narrow beam with almost no angular spread and the longest reach of
the three. Where the ultrasonic sensor finds whatever is broadly in front of it,
this one answers about a nearly specific direction.

Shared ray geometry, error model and determinism rules are in
[`README.md`](README.md); only what is specific to this sensor is below.

## Profile fields

`id`, `type`, `enabled`, `rate_hz`, `parent` and `pose_parent` follow the
common rules in [`camera-rgb.md`](camera-rgb.md). The three scalar range types
share one implementation and one set of `model` fields; they differ only in
their defaults, which is what makes them different instruments rather than
different code.

| `model` field | Default | Validation |
|---|---|---|
| `beam_fov_deg` | `1.0` | `0` – `179`; the full cone angle |
| `beam_samples` | `3` | integer ≥ 1; sample 0 is the beam axis |
| `reducer` | `nearest` | `nearest`, `farthest`, or `median` of the valid returns |
| `min_range_m` | `0.03` | ≥ 0, below `max_range_m` |
| `max_range_m` | `40.0` | positive |
| `noise_std_m` | `0.01` | ≥ 0 |
| `quantization_m` | `0.001` | ≥ 0; `0` disables |
| `dropout_probability` | `0.005` | `0` – `1` |
| `limit_degradation` | `0.2` | `0` – `10` |
| `confidence_model` | `range_linear` | `exact` or `range_linear` |

Any other `model` key is rejected by name.

```json
{
  "id": "front_tof", "type": "range.laser", "enabled": true,
  "rate_hz": 25.0, "parent": "nav_ptz",
  "pose_parent": {"x_m": 0.02, "y_m": 0.0, "z_m": 0.03,
                   "roll_deg": 0.0, "pitch_deg": 0.0, "yaw_deg": 0.0},
  "model": {"beam_fov_deg": 1.0, "beam_samples": 3, "max_range_m": 40.0}
}
```

With a 1 degree cone the three samples are within half a degree of
the axis, so the reducer rarely changes the answer; `beam_samples` is kept
above 1 so that a grazing edge still produces a return rather than a
coin-flip. `limit_degradation` of 0.2 is mild: a ToF unit holds its accuracy
much further into its range than an IR or acoustic one.

## Quantity, axes, and reduction

The published value is a **single slant range in metres from the sensor
origin**, along whichever cone ray the reducer selected. The cone is centred on
the sensor's +X: sample 0 is the axis, and the rest spiral outward on a
sunflower pattern — polar angle `half_angle · sqrt(i / (n − 1))`, turned a
further golden angle about the axis each time — so any prefix of the pattern
is still spread evenly and the pattern is fully determined by `beam_samples`
and `beam_fov_deg`.

`nearest` is the default and the physical answer for a time-of-flight beam:
the first echo wins. Both the reducer and the sample count are in the resolved
model and the manifest, so a reading can be reproduced exactly.

**No valid return is not maximum range.** When every cone ray misses, is
dropped, or falls outside the range gate, the record's status is invalid,
`range_m` is `null`, `confidence` is `0.0`, and `returns` is `0`.

The sensor is fully pose-sensitive: the cone rotates with its mount chain and
the vehicle, and each capture carries the composed pose.

## Simulation

On a due tick the manager freezes the vehicle state, resolves the world→sensor
transform, rotates the cone rays by it, and casts them against the shared
geometry. The returned truth goes through the common error model per ray, and
the surviving returns are reduced to one value.

## Transport

One `range.sample.v1` record (payload type 2) per capture on the **shared**
compact ring — no channel of its own. The schema is in
[`docs/modcom.md`](../modcom.md) "Scalar range transport": `range_m`,
`confidence` (0.0–1.0), `returns`, `samples`, `reducer`, the range gate, and
the composed `pose`/`pose_world` plus the `body` datum.

## Scheduling, lifecycle, and health

`rate_hz` must divide `physics_hz` exactly, so a reading always lands on a
physics snapshot. Applying a profile while disarmed publishes a new
generation; a drone reset increments `reset_epoch`, restarts cadence and the
noise stream, and leaves sequences and simulated time alone. Production counts
per sensor — scheduled, published, invalid, configured and observed rate —
appear in the run summary, and `invalid` is the useful one here: a beam that
never returns is a configuration or geometry answer, not a fault.

## Limitations

One return, no material response, no ambiguity interval and no
retro-reflector or specular behaviour. A beam aimed at a surface at a shallow
angle returns the geometric distance, where a real unit would often return
nothing.

## Correctness tests

`tests/test_sensor_geometry.py` — analytic distances downward, forward and
tilted at every heading, the box-top case, min/max gates, misses staying
invalid, cone spread and reducer selection, mount translation rotating with
heading, and the determinism rules.
`tests/test_sensor_contract.py` — the published schema, the invalid-reading
representation with no NaN in JSON, per-sensor cadence, and the shared-ring
behaviour under bulk LiDAR traffic.
