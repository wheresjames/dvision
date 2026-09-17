# `ground_plane`: floor-boundary projection

The cheapest camera algorithm in dalg. It assumes
**the world is a flat floor with things standing on it**, finds where each
image column stops being floor, and converts that row into a range using the
camera's known height and attitude. No pairing, no motion, no depth network:
one frame in, one fan of ranges out.

Profile: `ground-plane-baseline`, sensor selector `primary_camera`.

```sh
python3 ./apps/dalg/dalg.py --id area1 --profile ground-plane-baseline
```

## The idea in one picture

```
  side view
                       camera at height H, pitched down by θ
                    C
                    |\ θ                      ray through image row py
                    | \
                  H |  \  down = θ + atan((py − cy) / fy)
                    |   \
                    |    \
                    |     \                 ▓▓▓▓  the thing standing on it
    ────────────────┴──────●─────────────────▓▓───────────  floor, z = 0
                           ↑
                     distance = H / tan(down)

  the boundary row is where the floor stops:

  image column px            gradient |∂I/∂y|
   ┌───────────────┐          ┌───────────────┐
   │   wall        │          │      ░        │
   │   wall        │          │      ░        │
   ├───────────────┤ <─ py    │   ████████    │ <─ argmax below the horizon
   │   floor       │          │      ░        │
   │   floor       │          │      ░        │
   └───────────────┘          └───────────────┘
```

A strong horizontal edge below the horizon is where the floor meets something
vertical. Its row fixes the depression angle of the ray that lands there, and
one division by `tan` gives the range. Nothing else in dalg gets a metric range
out of a single frame this cheaply.

## The pipeline

```
  RGB frame
     │
     ├─> grayscale
     ├─> |Sobel(dy)|                        vertical gradient magnitude
     │
     └─> for every column, stride column_stride (4):
             py = horizon + argmax(gradient[horizon:, px])
             reject if gradient[py, px] < edge_threshold (28)
             down = −pitch_deg + atan((py − cy) / fy)
             reject if down <= 0                        (at or above horizon)
             distance = camera_height / tan(down)
             reject if outside [min_range_m, max_range_m]
             yaw = heading + atan((px − cx) / fx)
             point = (x + sin(yaw)·d, y − cos(yaw)·d)
     │
     ▼
  fuse_endpoints(grid, pose, points)        nearest first, occlusion-checked
```

`horizon` is taken as `int(cy_px)`: the search starts at the principal row and
looks **downward only**, so sky and ceiling edges are never candidates.

### Camera attitude and height

```python
if frame.camera is not None:
    return frame.camera, frame.camera.pitch_deg, frame.camera.z_m
return (frame.pose,
        frame.pose.pitch_deg - config.camera_pitch_down_deg,
        frame.pose.z_m + config.camera_height_offset_m)
```

**What the simulator publishes is authoritative.** The `camera_pitch_down_deg`
and `camera_height_offset_m` settings are the fallback for a source that
delivers no camera pose, and nothing else. This matters more here than
anywhere else in dalg: the range is `H / tan(down)`, so a wrong height scales
every range linearly and a wrong pitch bends the whole fan.

The height is floored at 0.1 m, because a camera at or below the floor plane
gives a meaningless or infinite range.

### Why fusion is occlusion-checked

Every sample this algorithm produces is a claimed **floor intersection**. When
the detected edge is not really a floor boundary — a poster, a shadow, the top
of a low box — the projected point lands *beyond* the real obstacle. Fusing it
blindly carves free space through a real wall. `fuse_endpoints` sorts nearest
first and drops any sample whose line of sight already crosses an occupied
cell, so the nearer, more believable marks decide what the far ones overshot
past. The full rule, with its known blind spot, is in
[`occupancy-grid.md`](occupancy-grid.md#fuse_endpoints).

## Settings

`GroundPlaneConfig`:

| Setting | Default | Meaning |
|---|---|---|
| `camera_pitch_down_deg` | 5.0 | fallback downward tilt, used only when no camera pose arrives |
| `camera_height_offset_m` | 0.1 | fallback mast height above the body datum, same condition |
| `column_stride` | 4 | every Nth image column is probed |
| `edge_threshold` | 28.0 | minimum Sobel magnitude for a row to count as a boundary |
| `min_range_m` | 0.3 | nearer than this is rejected |
| `max_range_m` | 20.0 | further than this is rejected |

Fusion uses `fuse_endpoints`' defaults: free −0.7 per crossed cell, occupied
+2.5 at the endpoint.

Diagnostics: `{'frames': n, 'projected_boundaries': points}`. `preview()` and
`finish()` are identical — fusion happens in `observe`, so there is nothing to
rebuild.

## What it gets wrong

These are properties of the assumption, not bugs:

- **A range error that grows as `1/tan`.** Near the horizon a one-pixel error
  in `py` is metres of range. At 20 m with a 5° tilt the fan is very soft;
  close in it is sharp.
- **Textured floors.** Any strong horizontal line on the floor — a tile seam,
  a paint stripe, a shadow edge — is a boundary as far as `argmax` is
  concerned, and becomes a phantom obstacle at that range. The occlusion rule
  then hides the real wall behind it, which is the safe failure but still a
  failure.
- **Non-flat ground.** A ramp, a step or a kerb breaks `z = 0` and biases
  every range in that column.
- **One boundary per column.** `argmax` returns the single strongest edge, so a
  doorway with a bright frame can mask the wall beside it.
- **Free space only at the ray's height.** Same approximation as `clear_rays`:
  a low obstacle can sit under a ray to the floor beyond it.

For a demonstration of how badly a camera-only map can behave — and why
`lidar-baseline` is the development recommendation — see "Where it stands" in
[`docs/navigation.md`](../navigation.md).

## Where it lives

| Thing | File |
|---|---|
| The algorithm | [`apps/dalg/algo/ground_plane.py`](../../apps/dalg/algo/ground_plane.py) |
| `bearing`, `fuse_endpoints` | [`apps/dalg/algo/spatial.py`](../../apps/dalg/algo/spatial.py) |
| Tests | `tests/dalg/test_dalg_evidence_algorithms.py`, `tests/dalg/test_dalg_synthetic.py` |
