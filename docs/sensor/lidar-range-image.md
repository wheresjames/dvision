# `lidar.range_image` — flash / solid-state range image

A rectangular grid of slant ranges and confidences, calibrated as a pinhole
camera. This is the shape a flash or solid-state LiDAR delivers, and it is
losslessly convertible to points given its calibration and the capture pose.

Shared ray geometry, error model and determinism rules are in
[`README.md`](README.md); only what is specific to this sensor is below.

## Profile fields

`id`, `type`, `enabled`, `rate_hz`, `parent` and `pose_parent` follow the
common rules in [`camera-rgb.md`](camera-rgb.md).

| `model` field | Default | Validation |
|---|---|---|
| `width_px`, `height_px` | `64`, `48` | positive integers |
| `fov_h_deg` **or** `fx_px` | `fov_h_deg` required if `fx_px` absent | **exactly one** of the two |
| `fy_px` | = `fx_px` | positive |
| `cx_px`, `cy_px` | image centre | finite |
| `min_range_m` | `0.15` | ≥ 0, below `max_range_m` |
| `max_range_m` | `20.0` | positive |
| `noise_std_m` | `0.03` | ≥ 0 |
| `quantization_m` | `0.01` | ≥ 0; `0` disables |
| `dropout_probability` | `0.02` | `0` – `1` |
| `limit_degradation` | `0.0` | `0` – `10` |
| `confidence_model` | `range_linear` | `exact` or `range_linear` |

The lens resolution rule is exactly the camera's: a resolved profile stores
`fx_px` and drops the redundant FOV, so saving and reloading is a fixed point.
Any other `model` key is rejected by name.

```json
{
  "id": "flash", "type": "lidar.range_image", "enabled": true, "rate_hz": 10.0,
  "parent": "body",
  "pose_parent": {"x_m": 0.05, "y_m": 0.0, "z_m": 0.02,
                   "roll_deg": 0.0, "pitch_deg": -5.0, "yaw_deg": 0.0},
  "model": {"width_px": 64, "height_px": 48, "fov_h_deg": 70.0,
             "min_range_m": 0.15, "max_range_m": 20.0,
             "noise_std_m": 0.03, "quantization_m": 0.01,
             "dropout_probability": 0.02}
}
```

## Quantity, axes, and angular calibration

Each element is a **slant range in metres from the sensor origin along that
element's ray** — not a depth along the optical axis, and not a coordinate.
Converting an element `(row, column)` to a point is:

```text
u = (column - cx_px) / fx_px
v = (row    - cy_px) / fy_px
direction_sensor = normalize((1, u, -v))          # forward, right, up
point_world = pose_world · (range_m · direction_sensor)
```

which is the camera optical convention (+X right, +Y down, +Z forward mapping
to sensor-body `(z, x, −y)`) written out. Row 0 is the top of the image and
column 0 the left, matching the RGB contract, so a range image and a camera
with the same calibration index the same directions.

The calibration travels in the manifest entry's `calibration`
(`width_px`, `height_px`, `fx_px`, `fy_px`, `cx_px`, `cy_px`); the composed
pose travels with each capture.

## Simulation

The rays are the pinhole rays of the declared calibration, rotated by the
composed pose and cast against the shared geometry, then passed through the
common error model. A ray that hits nothing within `max_range_m` is `NaN` with
confidence 0.

Cost scales with `width_px · height_px`. A 64×48 image against a large maze is
a few milliseconds per capture; a full camera-sized range image is not a
reasonable configuration at high rates, and the profile's memory budget will
usually stop it before the frame rate does.

## Transport

Two records per capture, sharing one `sequence` and `capture_id`
([`docs/modcom.md`](../modcom.md) "Bulk array transport"):

- the packed array on `...sensor.<id>.array`: `range_m` as
  `<f4[height_px, width_px]` (row-major) followed by `confidence` as
  `|u1[height_px, width_px]`;
- one `lidar.frame.v1` record on the shared compact ring with
  `array_sequence`, `returns`, `samples`, `calibration`, and the composed pose.

The dedicated ring holds `ceil(rate) + 2` records of
`128 + 5 · width_px · height_px` bytes, at least 64 KiB, validated against the
256 MiB per-instance budget before allocation.

## Scheduling, lifecycle, and health

Identical to [`lidar-scan2d.md`](lidar-scan2d.md): rates divide `physics_hz`,
publication is latest-value, generations recreate the ring, drone reset
restarts cadence and noise but not sequences, and per-sensor production counts
appear in the run summary.

## Limitations

Material-independent and single-return, with no beam divergence, no
per-element exposure, and no motion during the capture. The stored range is
the geometric distance to the first surface; there is no ambiguity interval,
multi-path, or retro-reflector behaviour. Point clouds are deliberately not a
transport in v1 — a range image plus its calibration is the same information
without a high-rate point ABI.

## Correctness tests

`tests/test_sensor_geometry.py` — row-major shape, the centre element
measuring the analytic distance to a known wall, invalid returns, and the
determinism rules.
`tests/test_sensor_contract.py` — layout round-trip, metadata/array sequence
pairing, and dedicated-ring behaviour under load.
