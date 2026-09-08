# `lidar.scan2d` — scanning 2D LiDAR

An ordered polar sweep in the sensor's own plane: one slant range and one
confidence per ray, published as a packed array with the angular calibration
in the manifest. Covers both a full-circle spinning unit and a partial-sector
scanner.

Shared ray geometry, error model and determinism rules are in
[`README.md`](README.md); only what is specific to this sensor is below.

## Profile fields

`id`, `type`, `enabled`, `rate_hz`, `parent` and `pose_parent` follow the
common rules in [`camera-rgb.md`](camera-rgb.md).

| `model` field | Default | Validation |
|---|---|---|
| `fov_deg` | `360.0` | `0.01` – `360.0` |
| `samples` | `720` | integer ≥ 2 |
| `elevation_deg` | `0.0` | `-89` – `89`; tilts the whole sweep out of the sensor's XY plane |
| `min_range_m` | `0.15` | ≥ 0, below `max_range_m` |
| `max_range_m` | `25.0` | positive |
| `noise_std_m` | `0.02` | ≥ 0 |
| `quantization_m` | `0.005` | ≥ 0; `0` disables |
| `dropout_probability` | `0.01` | `0` – `1` |
| `limit_degradation` | `0.0` | `0` – `10` |
| `confidence_model` | `range_linear` | `exact` or `range_linear` |

Any other `model` key is rejected by name.

```json
{
  "id": "scan", "type": "lidar.scan2d", "enabled": true, "rate_hz": 10.0,
  "parent": "body",
  "pose_parent": {"x_m": 0.0, "y_m": 0.0, "z_m": 0.05,
                   "roll_deg": 0.0, "pitch_deg": 0.0, "yaw_deg": 0.0},
  "model": {"fov_deg": 360.0, "samples": 720,
             "min_range_m": 0.15, "max_range_m": 25.0,
             "noise_std_m": 0.02, "quantization_m": 0.005,
             "dropout_probability": 0.01}
}
```

## Quantity, axes, and angular calibration

Each sample is a **slant range in metres from the sensor origin**, not a
horizontal distance and not a coordinate. Ray *i* points at
`angle_min_deg + i · angle_increment_deg` in the sensor's own frame, where 0°
is the sensor's +X and positive turns toward +Y — the same sense as a positive
body yaw. `elevation_deg` lifts every ray toward +Z.

`angle_min_deg` is `-fov_deg / 2`. The increment is `fov_deg / samples` for a
full circle, so the sweep does not repeat its first ray, and
`fov_deg / (samples − 1)` for a partial sector, so it includes both endpoints.
Both values are published in the manifest entry's `calibration`; a consumer
never has to re-derive them.

The sensor is fully pose-sensitive: the whole sweep rotates with its mount
chain and the vehicle. Every capture carries the composed pose, so a scan is
convertible to map points with nothing but the record.

## Simulation

On a due tick the manager freezes the vehicle state, resolves the world→sensor
transform, generates the `samples` rays in the sensor frame, rotates them by
that transform, and casts them against the shared geometry. The returned
truth then goes through the common error model. A ray that hits nothing within
`max_range_m` is `NaN` with confidence 0 — **not** `max_range_m`.

Cost is proportional to configured output: a 720-ray scan against a large maze
takes a few milliseconds, and the ray service culls obstacles outside
`max_range_m` before intersecting anything.

## Transport

Two records per capture, sharing one `sequence` and `capture_id`
([`docs/modcom.md`](../modcom.md) "Bulk array transport"):

- the packed array on the sensor's own ring (`...sensor.<id>.array`),
  `range_m` as `<f4[samples]` followed by `confidence` as `|u1[samples]`;
- one `lidar.frame.v1` record on the shared compact ring carrying
  `array_sequence`, `returns`, `samples`, the angular `calibration`, and the
  composed pose, so pose and timing never travel inside the array.

The dedicated ring is sized from the profile: `ceil(rate) + 2` records of
`128 + 5 · samples` bytes, at least 64 KiB, counted against the 256 MiB
per-instance budget before anything is allocated.

## Scheduling, lifecycle, and health

`rate_hz` must divide `physics_hz` exactly, so a scan always lands on a
physics snapshot. Publication is latest-value: if a consumer falls behind, its
ring laps and the gap shows in the per-sensor `sequence`, never as a stalled
simulator. Applying a profile while disarmed publishes a new generation and a
new ring; a drone reset increments `reset_epoch`, restarts the cadence and the
noise stream, and leaves sequences and simulated time alone. Production counts
(scheduled, published, invalid, configured and observed rate) appear per sensor
in the run summary.

## Limitations

Material-independent, single-return, and instantaneous: the whole sweep is
sampled from one vehicle state, so there is no motion distortion across a
rotation. Reflectivity, multiple returns, beam divergence within a ray, and
weather attenuation are deliberately absent. A 2D scan is one plane — a
sloped or 3D unit is a `lidar.range_image` instead.

## Correctness tests

`tests/test_sensor_geometry.py` — full-circle and partial-sector angle
calibration, ray ordering around the vehicle, analytic distance to a known
wall, misses staying invalid, and the determinism rules.
`tests/test_sensor_contract.py` — the packed layout round-trips through the
manifest, metadata and array share a sequence, the dedicated ring absorbs bulk
traffic without evicting compact records, and a disabled scanner leaves no
manifest entry.
