# `range.ultrasonic` — ultrasonic rangefinder

A wide, short-range acoustic beam with a blind zone close in. The
widest cone of the three scalar range types, and the one most likely to find a
surface a single ray would miss — which is exactly what makes it useful looking
down at the ground.

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
| `beam_fov_deg` | `25.0` | `0` – `179`; the full cone angle |
| `beam_samples` | `9` | integer ≥ 1; sample 0 is the beam axis |
| `reducer` | `nearest` | `nearest`, `farthest`, or `median` of the valid returns |
| `min_range_m` | `0.2` | ≥ 0, below `max_range_m` |
| `max_range_m` | `7.0` | positive |
| `noise_std_m` | `0.01` | ≥ 0 |
| `quantization_m` | `0.005` | ≥ 0; `0` disables |
| `dropout_probability` | `0.02` | `0` – `1` |
| `limit_degradation` | `0.5` | `0` – `10` |
| `confidence_model` | `range_linear` | `exact` or `range_linear` |

Any other `model` key is rejected by name.

```json
{
  "id": "down_sonar", "type": "range.ultrasonic", "enabled": true,
  "rate_hz": 20.0, "parent": "body",
  "pose_parent": {"x_m": 0.0, "y_m": 0.0, "z_m": -0.03,
                   "roll_deg": 0.0, "pitch_deg": -90.0, "yaw_deg": 0.0},
  "model": {"beam_fov_deg": 25.0, "beam_samples": 9,
             "min_range_m": 0.20, "max_range_m": 7.0,
             "noise_std_m": 0.01}
}
```

The `min_range_m` default of 0.2 m is the blind zone: a real
transducer is still ringing that close and reports nothing, so a return inside
it is discarded rather than clamped. `limit_degradation` of 0.5 widens the
noise and the dropout probability by half at full scale, which is the
echo-strength falloff a cheap sonar shows near its limit.

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

The beam is a cone of independent rays, not a pressure field: there
is no specular dropout off a smooth angled surface, no cross-talk between
sensors, no temperature-dependent speed of sound, and no multi-path. Material
response is absent — a curtain and a wall read the same.

## Correctness tests

`tests/test_sensor_geometry.py` — analytic distances downward, forward and
tilted at every heading, the box-top case, min/max gates, misses staying
invalid, cone spread and reducer selection, mount translation rotating with
heading, and the determinism rules.
`tests/test_sensor_contract.py` — the published schema, the invalid-reading
representation with no NaN in JSON, per-sensor cadence, and the shared-ring
behaviour under bulk LiDAR traffic.
