# `monocular_depth`: a metric-depth network

The only algorithm here that learns anything. It runs a user-supplied ONNX
**metric**-depth network on each frame, treats the output as distance along the
optical axis, and fuses the resulting point cloud exactly as `sgbm` fuses its
disparity. One frame is enough — no baseline, no pairing, no parallax — which
makes it the only camera algorithm that works while the vehicle is stationary
or flying straight forward.

Profile: `monocular-depth-baseline`, sensor selector `primary_camera`.

```sh
python3 ./scripts/install_dalg_depth_model.py       # once: download, export, install
python3 ./apps/dalg/dalg.py --id area1 --profile monocular-depth-baseline
```

The model is **not optional and not bundled**. Without a readable
`settings.model_path` the constructor raises, and dalg refuses to launch;
without `onnxruntime` it raises and points at the install script. That is
deliberate: an algorithm that silently publishes nothing is worse than one that
refuses to start.

## The pipeline

```
  RGB frame
     │
     ▼  measure()  — the expensive half, run ONCE per frame
  blobFromImage(1/255, input_width × input_height)
     │
     ▼
  onnxruntime session.run                       CPUExecutionProvider
     │
     ├─ output must reduce to H×W, else ValueError
     ▼
  resize to the camera's full resolution (INTER_LINEAR)
     │
     ▼
  depth · scale + offset_m                      the one calibration hook
     │
     ├─ sample every `stride` (8) pixels; keep finite and > 0
     ▼
  project_pixels(camera_pose, px, py, depth, intrinsics)
     │                       metric-depth heads report distance along the
     │                       optical axis — the convention project_pixels
     │                       inverts — so the ROW of a pixel gives the sample
     │                       a height, and floor and sky stop reading as walls
     ▼
  PointObservation(pose, xs, ys, zs)
     │
     ▼  fuse()  — the cheap half, run once PER GRID
  ┌──┴────────────────────────────────────┐
  │                                       │
  obstacle_band(...)              free_space_samples(...)   [evidence only]
  0.25..4.5 m high, 0.3..25 m out         │
  │                                       ▼
  ▼                               clear_rays(grid, pose, ...)
  grid.accumulate(cells, +2.5)
```

### Why `measure` and `fuse` are separate

`PointObservation` exists for one reason: a run can hold **two grids** — the
scored one and the published evidence one — and running a depth network once
per grid was what left this algorithm unable to keep up with the camera. The
measurement is done once, and each grid fuses the same points. `translated()`
lets a grid whose origin is not at the map origin reuse the identical
observation.

The `evidence` flag also controls free-space carving: the evidence copy marks
the space each depth ray crossed as free, the scored copy does not, so its
scores are exactly what they were before this algorithm could publish.

## Settings

`MonocularDepthConfig`, with the committed baseline's values in the third
column where they differ:

| Setting | Default | `monocular-depth-baseline` | Meaning |
|---|---|---|---|
| `model_path` | `""` | `.cache/models/metric-depth.onnx` | ONNX metric-depth model; **required**, must be a file |
| `input_width` | 256 | 280 | network input width |
| `input_height` | 192 | 210 | network input height |
| `scale` | 1.0 | 1.0 | multiplies the network's output |
| `offset_m` | 0.0 | 0.0 | added after the scale |
| `min_range_m` | 0.3 | 0.3 | obstacle band, radial |
| `max_range_m` | 25.0 | 20.0 | obstacle band, radial |
| `min_height_m` | 0.25 | — | obstacle band, vertical |
| `max_height_m` | 4.5 | — | obstacle band, vertical |
| `stride` | 8 | 8 | every Nth pixel of the depth map is fused |
| `occupied_log_odds` | 2.5 | — | per surviving sample, **accumulated** |

`scale` and `offset_m` are the whole calibration interface. A network trained
in a different metric convention, or one whose head reports something
proportional to depth, is corrected here and nowhere else — and if it is wrong,
every obstacle in the map is at the wrong range, confidently.

Diagnostics: `frames`, `depth_points`. `preview()` and `finish()` are
identical; fusion happens in `observe`.

## What it gets wrong

- **It is a prior, not a measurement.** A monocular network infers depth from
  what the scene looks like. On a scene unlike its training data it is not
  noisy — it is *wrong with confidence*, and there is no parallax test, no cost
  margin and no uniqueness ratio to catch that. Nothing in this algorithm can
  tell a good inference from a bad one.
- **Scale is the model's opinion.** Metric heads are only as metric as their
  training; `scale`/`offset_m` correct a systematic error but not a
  range-dependent one.
- **Edge bleed.** Resizing a low-resolution depth map to full camera
  resolution smears depth across object boundaries, which puts samples in the
  space beside an obstacle.
- **Cost.** It is the heaviest algorithm here and runs on CPU. `input_width`,
  `input_height` and `stride` are the dials; the measure/fuse split means the
  cost is per frame, not per grid.
- **Free space at the ray's height only**, as for every carving algorithm.

## Where it lives

| Thing | File |
|---|---|
| The algorithm | [`apps/dalg/algo/monocular_depth.py`](../../apps/dalg/algo/monocular_depth.py) |
| `PointObservation`, `project_pixels`, `clear_rays` | [`apps/dalg/algo/spatial.py`](../../apps/dalg/algo/spatial.py) |
| Model install | [`scripts/install_dalg_depth_model.py`](../../scripts/install_dalg_depth_model.py) |
| Tests | `tests/dalg/test_dalg_evidence_algorithms.py` |
