# `lidar_inverse`: the inverse sensor model

The baseline every camera algorithm is measured against, and the one to fly
with while developing anything else. It takes calibrated `lidar.scan2d`
returns and does the textbook thing: **carve the ray free, mark the return
occupied**. No image processing, no pairing, no temporal state — one scan in,
one set of cell updates out.

Profile: `lidar-baseline` (`assets/algorithm_profiles/lidar-baseline.json`),
sensor selector `scan`, no settings.

```sh
python3 ./apps/dalg/dalg.py --id area1 --profile lidar-baseline
```

Unlike the camera algorithms, this one is constructed on the **runtime
geometry** directly (`LidarInverseModel(geometry, settings)`) and writes an
`ObservedGrid` itself, so it works entirely in the public map frame: the
sample carries a 4×4 world pose and the model never sees a grid-local
translation.

## The pipeline

```
  scan2d sample
      │
      ├─ status false or no fields ──────────────> ignored
      │
      ├─ calibration: samples, angle_min_deg, angle_increment_deg, elevation_deg
      ├─ fields:      range_m[N], confidence[N]
      ├─ payload:     pose_world (4x4)
      ├─ entry.model: min_range_m, max_range_m
      │
      ▼
  validity gate      finite range, within the sensor's model range,
      │              confidence >= min_confidence
      ▼
  ray directions     unit vector per beam, in the sensor frame,
      │              then rotated by pose_world[:3, :3]
      ▼
  traversed_cells()  every cell the segment [min_range, range] crosses,
      │              clipped to the grid extent and its z slab
      ▼
  free -= occupied   a cell holding a return is never also carved free
      ▼
  grid.update(free, free_log_odds); grid.update(occupied, occupied_log_odds)
```

## Beam geometry

Each beam `i` has azimuth `angle_min_deg + i * angle_increment_deg` and the
scan's single `elevation_deg`, giving a sensor-frame direction

```
  d = ( cos(az)·cos(el),  sin(az)·cos(el),  sin(el) )
```

rotated into the world by the sample's own pose. The ray is traced from
`min_range_m` along that direction — not from the sensor origin, since inside
the minimum range the sensor proves nothing — to the measured return:

```
  top view, one beam

     sensor            min_range_m                                return
       o───────────────────┼─────────────────────────────────────────X
                           └──── free: every cell crossed ──────────┘└ occupied
```

### Why the free set is a segment walk and not a Bresenham line

`traversed_cells` partitions the segment at cell boundaries and samples the
**midpoint of each interval**:

```
  cut the parameter t at every x- and y-boundary the segment crosses,
  then take the midpoint of each resulting interval as a probe point

      │     │     │     │          cuts at t = enter, boundaries..., leave
   ───┼─────┼─────┼─────┼───       probes at interval midpoints  •
      │  •  │  •  │  •  │
   ───┼─────┼─────┼─────┼───       a diagonal ray visits both cells of every
      │     │  •  │  •  │          corner it clips, and never a cell it only
   ───┴─────┴─────┴─────┴───       touches at a vertex
```

A midpoint is unambiguously inside exactly one cell, so a diagonal ray is
traced without missing narrow intersections and without drawing outside the
declared map. The segment is first clipped against the grid's extent *and* its
z slab on all three axes, which is also what keeps a beam that flies over the
slab from clearing anything.

A return is marked occupied only when its endpoint cell exists and its
**height falls inside the slab** (`z0_m <= z < z0_m + dz_m`); a return above or
below the mapped layer still contributes its free ray.

### NaN is not permission to clear

```python
valid = (isfinite(ranges) & (ranges >= model.min_range_m)
         & (ranges <= model.max_range_m) & (confidence >= min_confidence))
```

A NaN or zero-confidence sample is **ambiguous** — a dropout, or genuinely no
return — and never a licence to clear a max-range ray. Only measured, valid
returns write anything. The alternative (treat no-return as free to the
maximum range) is the classic way to carve corridors through glass, through
dark surfaces and through sky.

`free -= occupied` runs before either write, so a cell that holds a return in
this same scan is not also carved free by a neighbouring beam that passed
beside it.

## Settings

`LidarConfig`, validated in `__post_init__` — a bad value fails at construction,
not silently at runtime.

| Setting | Default | Meaning |
|---|---|---|
| `free_log_odds` | −0.7 | applied once per crossed cell; must be finite and negative |
| `occupied_log_odds` | +2.5 | applied to the return's cell; must be finite and positive |
| `min_confidence` | 1 | a beam below this confidence is ignored entirely; integer in 1..255 |

`min_range_m` and `max_range_m` are **not** settings: they come from the
sensor's own model in the sample entry, so the algorithm cannot disagree with
the hardware it is reading.

## Diagnostics

`preview()` and `finish()` are the same call — there is nothing deferred to
rebuild — and report `{'scans': frames, 'valid_rays': rays}`.

## What it gets wrong

- **It is 2-D.** A single elevation means a scan sees one horizontal slab.
  Anything above or below it is invisible, including the floor, so this
  algorithm never carves the free space a camera algorithm sees under a table.
- **Thin and specular surfaces.** Glass and polished metal return nothing; the
  NaN rule keeps the cell unknown rather than free, which is safe but means
  the map never fills in.
- **Motion during a scan** is not modelled: every beam uses the one
  `pose_world` delivered with the sample.

## Where it lives

| Thing | File |
|---|---|
| The model | [`apps/dalg/algo/lidar.py`](../../apps/dalg/algo/lidar.py) |
| Segment walk | `traversed_cells` in the same file |
| Publication adapter | `LidarEvidence` in [`apps/dalg/evidence.py`](../../apps/dalg/evidence.py) |
| Sensor contract | [`docs/sensor/lidar-scan2d.md`](../sensor/lidar-scan2d.md) |
| Tests | `tests/dalg/test_dalg_core.py`, `tests/dalg/test_dalg_evidence.py` |
