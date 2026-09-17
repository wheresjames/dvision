# Algorithm documents: dalg

This directory holds one focused document per **algorithm** in dalg: the six
camera evidence algorithms, the LiDAR inverse model, and the detector stack.
Each document records what the algorithm computes, the geometry and the
arithmetic behind it, every setting with its default, the failure modes it is
known to have, and the tests that hold it in place.

What these documents deliberately do **not** repeat: how dalg is launched, how
profiles are composed, how sensors are admitted, or what the published planes
look like on the wire. That is
[`docs/camera-evidence.md`](../camera-evidence.md),
[`docs/multisensor-evidence.md`](../multisensor-evidence.md),
[`docs/detection.md`](../detection.md),
[`docs/modcom.md`](../modcom.md) and [`docs/membus.md`](../membus.md).
Missions, which *consume* both products, are in
[`docs/missions/`](../missions/README.md), and the planners that turn evidence
into routes are in [`docs/planners/`](../planners/README.md).

## The two products

dalg turns sensor samples into two beliefs, and nothing else:

```
  sensor samples                     dalg                        published planes
  ──────────────                ──────────────                   ────────────────

  camera.rgb frame ─┬──> evidence algorithm ──> log-odds grid ──> maps plane   (occupancy + age)
                    │      (one per source)
                    │
                    └──> detector ──> projection ──> estimates ──> detections plane
                           (pixels)     (ground)      (running)

  lidar.scan2d ─────────> lidar_inverse ─────────> log-odds grid ──> maps plane
```

Neither product ever reads a world file, `dsim/truth`, or the provider's
`target.*` keys. Every grid and every estimate is a **belief**, stamped with
the capture time of the sample that produced it.

## Evidence algorithms

| Algorithm | Sensor | Principle | Document |
|---|---|---|---|
| `lidar_inverse` | `lidar.scan2d` | inverse sensor model on calibrated returns | [`lidar-inverse.md`](lidar-inverse.md) |
| `ground_plane` | `camera.rgb` | floor-boundary row → range by camera height | [`ground-plane.md`](ground-plane.md) |
| `sgbm` | `camera.rgb` | stereo from motion: dense disparity across a sideways baseline | [`sgbm.md`](sgbm.md) |
| `plane_sweep` | `camera.rgb` | photometric cost over hypothesised depths | [`plane-sweep.md`](plane-sweep.md) |
| `feature_triangulation` | `camera.rgb` | ORB matches, two-bearing intersection | [`feature-triangulation.md`](feature-triangulation.md) |
| `optical_flow_triangulation` | `camera.rgb` | Lucas–Kanade tracks, two-bearing intersection | [`optical-flow-triangulation.md`](optical-flow-triangulation.md) |
| `monocular_depth` | `camera.rgb` | ONNX metric-depth network | [`monocular-depth.md`](monocular-depth.md) |

The camera list lives in one place, `CAMERA_EVIDENCE` in
[`apps/dalg/profiles.py`](../../apps/dalg/profiles.py); an algorithm joins it
once it meets the evidence contract, and joining it is the whole switch. The
`constant` and `exact_range` **controls** are not dalg algorithms: they are
built from truth or from nothing, and live with the other oracle checks in
[`dtest/evaluation.py`](../../dtest/evaluation.py).

Shared machinery that every one of them writes through — the log-odds grid,
the ray carving, the obstacle band, the frame conventions — is documented once
in [`occupancy-grid.md`](occupancy-grid.md) and linked, not copied.

## Detection

| Stage | Module | Document |
|---|---|---|
| pixels → detections | [`apps/dalg/detect/red_disc.py`](../../apps/dalg/detect/red_disc.py) | [`red-disc.md`](red-disc.md) |
| pixels → ground, estimates, plane | `detect/projection.py`, `detect/estimate.py`, `detect/host.py` | [`detection-pipeline.md`](detection-pipeline.md) |

A detector works in **image space only**. Projecting onto the ground, keeping
a running estimate, publishing and recording are the host's, shared by every
detector — so a new detector is one module and one line in `DETECTORS`.

## The frame every document assumes

Local map metres. **x east, y south, z up**, so rows of the occupancy raster
increase with y and no flip is needed anywhere. Heading is a compass bearing:
0 is north, and it increases **clockwise**.

```
                 north, heading 0
                        ^
                        |  −y
                        |
      heading 270 ------o------> +x   heading 90 (east)
              −x        |
                        |  +y
                        v
                 south, heading 180
```

Level forward at heading `h` is therefore `(sin h, −cos h, 0)`, right is
`(cos h, sin h, 0)`, and a **positive pitch tilts the camera up**. The three
axes come from one function, `camera_axes` in
[`apps/dalg/algo/spatial.py`](../../apps/dalg/algo/spatial.py), so every
algorithm and the detector projection agree by construction.

Two pose rules that have each cost a real bug:

- **Use the camera pose delivered with the sample** (`Frame.camera_pose`),
  never the body pose plus `camera.*` status keys. The simulator publishes the
  camera's mounting offset and its *absolute* attitude separately; geometry
  that reads the body pose silently drops the fixed downward tilt and the mast
  height, and biases every projected range.
- **Depth along the optical axis is not distance along the ray.** Treating
  them as interchangeable pulls everything at the edge of frame toward the
  camera — by 13% at the corner of a 70° lens. `project_pixels` does the full
  3-D reconstruction; `project_ranges` inverts the range sensor's separate
  yaw/pitch convention and its slant range instead.

## Reading an algorithm's document

Each one is laid out the same way:

1. **What it is** — one paragraph, and the profile that runs it.
2. **The pipeline** — a diagram, then each stage.
3. **Settings** — every field of its config dataclass, with the default.
4. **What it gets wrong** — the failure modes that are properties of the
   method, not bugs.
5. **Where it lives** — source, tests.
