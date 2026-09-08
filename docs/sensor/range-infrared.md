# `range.infrared` — infrared proximity rangefinder

A narrow, very short-range optical beam that degrades noticeably as
it approaches its limit. The cheap proximity sensor of the set: reliable within
a metre or so, and honestly unreliable beyond it.

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
| `beam_fov_deg` | `8.0` | `0` – `179`; the full cone angle |
| `beam_samples` | `5` | integer ≥ 1; sample 0 is the beam axis |
| `reducer` | `nearest` | `nearest`, `farthest`, or `median` of the valid returns |
| `min_range_m` | `0.05` | ≥ 0, below `max_range_m` |
| `max_range_m` | `1.5` | positive |
| `noise_std_m` | `0.005` | ≥ 0 |
| `quantization_m` | `0.001` | ≥ 0; `0` disables |
| `dropout_probability` | `0.01` | `0` – `1` |
| `limit_degradation` | `1.0` | `0` – `10` |
| `confidence_model` | `range_linear` | `exact` or `range_linear` |

Any other `model` key is rejected by name.

```json
{
  "id": "front_ir", "type": "range.infrared", "enabled": true,
  "rate_hz": 20.0, "parent": "body",
  "pose_parent": {"x_m": 0.12, "y_m": 0.0, "z_m": 0.0,
                   "roll_deg": 0.0, "pitch_deg": 0.0, "yaw_deg": 0.0},
  "model": {"beam_fov_deg": 8.0, "beam_samples": 5, "max_range_m": 1.5}
}
```

`limit_degradation` of 1.0 is the highest of the three defaults: at
full scale both the noise and the dropout probability double. Combined with a
1.5 m `max_range_m`, an IR sensor on an open-air vehicle reports invalid most
of the time, which is the correct answer and not a fault — `invalid` in the
production counts is the metric to read, not a missing sample.

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

No ambient-light rejection, no surface albedo or colour response,
and no sunlight saturation, all of which dominate a real IR rangefinder
outdoors. The model is geometry plus a range-dependent error, and its
usefulness is bounded by that.

## Correctness tests

`tests/test_sensor_geometry.py` — analytic distances downward, forward and
tilted at every heading, the box-top case, min/max gates, misses staying
invalid, cone spread and reducer selection, mount translation rotating with
heading, and the determinism rules.
`tests/test_sensor_contract.py` — the published schema, the invalid-reading
representation with no NaN in JSON, per-sensor cadence, and the shared-ring
behaviour under bulk LiDAR traffic.
