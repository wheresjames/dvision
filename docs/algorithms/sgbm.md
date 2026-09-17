# `sgbm`: stereo from motion

Dense stereo with **one camera**. There is no second lens, so the second view
comes from somewhere else the vehicle has been: two frames captured at poses
that happen to form a left/right stereo pair are handed to OpenCV's
semi-global block matcher, and the disparity is turned into metric depth by the
known baseline between them.

Profile: `sgbm-baseline`, sensor selector `primary_camera`.

```sh
python3 ./apps/dalg/dalg.py --id area1 --profile sgbm-baseline
```

**Know this before you pick it:** it pairs frames only when the camera has
moved *sideways*. On a tour that flies forward it finds no pairs and publishes
nothing. Fly it along a wall, heading held.

## What makes two frames a stereo pair

`StereoSGBM` assumes rectified, parallel cameras with a horizontal baseline.
Two frames from one moving camera satisfy that only when the motion between
them was **pure sideways translation with the heading held**. `_pair` enforces
exactly that, and nothing more:

```
  reference frame at pose P, heading h            right = (cos h, sin h)
                                                  forward = (sin h, −cos h)

                 forward (the way it is looking)
                        ▲
                        │
      ┌─────────────────┼─────────────────┐
      │  ✗ too far      │                 │   baseline > max_baseline_m (4 m)
      │    ┌────────────┼────────────┐    │
      │    │  ✗ too much forward     │    │   |delta·forward| >
      │    │ ~ ~ ~ ~ ~ ~│~ ~ ~ ~ ~ ~ │    │      0.15 · baseline
      │    │  ✗ left    │   ✓ RIGHT  │    │   delta·right must be > 0
      │    │            P ──────────>│────┼──> right
      │    │  ✗ left    │   ✓ RIGHT  │    │
      │    │ ~ ~ ~ ~ ~ ~│~ ~ ~ ~ ~ ~ │    │
      │    └────────────┼────────────┘    │
      │  ✗ too near     │                 │   baseline < min_baseline_m (0.25 m)
      └─────────────────┼─────────────────┘
                        │
        plus: |heading difference| <= max_heading_delta_deg (2°)
        among the eligible, the WIDEST baseline wins
```

Only frames to the **right** qualify, because OpenCV expects the second image
to be the right-hand camera. Nothing is wasted by that asymmetry: a frame to
the left picks this one up as its own partner when its turn comes.

The forward test is the one that stops forward flight from producing garbage.
Motion along the optical axis gives no disparity to speak of — the scene scales
instead of shifting — and `max_forward_fraction` caps the along-axis component
at 15% of the baseline.

## The two ways it runs

The live evidence copy and the offline scored copy differ, and every
difference sits behind the `evidence` flag.

```
  EVIDENCE (live, causal)                 SCORED (a captured tour, rebuilt)
  ─────────────────────────               ─────────────────────────────────
  observe(frame)                          observe(frame)
    keep newest EVIDENCE_WINDOW (64)        keep everything
    mark arrived                            finish() rebuilds from scratch
  fuse_available()                          _process() pairs every frame once,
    pair the NEWEST frame against             with partners on either side
    frames ALREADY CAPTURED only            no free-space carving
    widest baseline either way
    stamp cells with the newest
      frame's capture time
    also carve free space (clear_rays)
```

The causal restriction is the whole point: live evidence may not use a frame
from the future. `finish()` in the scored path deliberately **rebuilds from the
complete capture** rather than keeping what the previews fused, because a
preview pairs each frame with the first partner to clear `min_baseline_m` — the
wider ones have not been captured yet — and the reported grid would otherwise
depend on how often a front end asked for a preview, which is not a property of
the algorithm.

One thing worth knowing about the live path: dalg's run loop also calls
`preview()` once per admitted frame for the Grids readout, and in evidence mode
`preview()` advances `_process`, which pairs any frame `fuse_available` did not
record in `processed`. The same pair can therefore be fused twice — once
causally, once by the preview pass. Log odds clip at ±8, so the effect is that
cells saturate sooner, not that they take the wrong sign.

## From disparity to cells

```
  reference (left)   partner (right)
        │                  │
        └──── StereoSGBM ──┘      one matcher, created once and reused:
                 │                rebuilding it per pair dominated the run
                 ▼
          disparity / 16.0        OpenCV's fixed-point output
                 │
                 ├─ sample every `stride` (8) pixels, keep disparity > 0
                 │
                 ▼
          depth_z = fx · baseline / disparity          along the optical axis
                 │
                 ▼
          project_pixels(pose, px, py, depth_z, intrinsics)
                 │                full 3-D pinhole reconstruction: depth along
                 │                the axis, NOT distance along the ray
                 ▼
          ┌──────┴───────────────────────────────┐
          │                                      │
   obstacle_band(...)                    free_space_samples(...)   [evidence only]
   height 0.25..4.5 m                    everything below the band top
   range 0.15..25 m                              │
          │                                      ▼
          ▼                              clear_rays(grid, pose, ...)
   grid.accumulate(cells, +4.0)          update(free) per distinct endpoint cell
   (repeated hits reinforce)
```

`accumulate` rather than `update` here: each surviving pixel is an independent
measurement, and several agreeing on a cell should reinforce it. The free
carving uses `update`, so one sweep counts once.

### The range-sensor path

`sgbm` also has `_fuse_range`, which fuses a frame's `range_m` image directly
through `project_ranges`. It exists for the **scored** path, where a capture may
carry a range image beside the RGB; `CameraEvidence.observe` strips `range_m`,
so the live evidence copy never uses it. Every finite sample is fused — the ray
cast leaves NaN wherever it did not sample, so the sampling stride never has to
be agreed on separately and cannot drift out of step with the sensor's.

## Settings

`SGBMConfig`:

| Setting | Default | Meaning |
|---|---|---|
| `min_baseline_m` | 0.25 | narrower pairs are rejected: too little parallax |
| `max_baseline_m` | 4.0 | wider pairs are rejected: too little overlap, and disparity leaves the search range |
| `max_heading_delta_deg` | 2.0 | the pair must be near-parallel; SGBM does not rectify |
| `max_forward_fraction` | 0.15 | along-axis motion allowed, as a fraction of the baseline |
| `num_disparities` | 128 | SGBM search range; must be a positive multiple of 16 |
| `block_size` | 5 | SGBM window; must be odd and >= 3 |
| `uniqueness_ratio` | 12 | SGBM's own margin test |
| `stride` | 8 | every Nth pixel of the disparity map is fused |
| `min_range_m` / `max_range_m` | 0.15 / 25.0 | obstacle band, radial |
| `min_height_m` / `max_height_m` | 0.25 / 4.5 | obstacle band, vertical |
| `occupied_log_odds` | 4.0 | per surviving sample, accumulated |
| `preview_batch` | 2 | frames fused per `preview()` call |
| `preview_interval_s` | 0.5 | how often a front end should ask |

`num_disparities` and `block_size` are validated in the constructor; the rest
are trusted.

The interaction to remember: `num_disparities` caps the **nearest** thing that
can be matched, at `fx · baseline / num_disparities`. Widening
`max_baseline_m` without widening the search range pushes near geometry out of
range entirely.

Diagnostics: `frames`, `stereo_pairs`, `stereo_points`, `range_frames`,
`range_points`.

## What it gets wrong

- **Forward flight produces nothing.** By design, per the eligibility tests. An
  empty grid on a forward tour is the algorithm working correctly.
- **Textureless surfaces.** A blank wall has no disparity to match; SGBM's
  `uniquenessRatio` and speckle filters reject it, leaving unknown cells.
- **Repetitive texture** (brickwork, railings) is the classic SGBM false-match,
  and lands obstacles at the wrong depth.
- **Pose error is baseline error.** Depth scales linearly with the baseline
  taken from the two poses, so any drift in them scales the whole
  reconstruction.
- **Cost.** It is the heaviest camera algorithm here apart from the depth
  network; `stride` and `preview_batch` are the dials.

## Where it lives

| Thing | File |
|---|---|
| The algorithm | [`apps/dalg/algo/sgbm.py`](../../apps/dalg/algo/sgbm.py) |
| Projection, bands, carving | [`apps/dalg/algo/spatial.py`](../../apps/dalg/algo/spatial.py) |
| Tests | `tests/dalg/test_dalg_evidence_algorithms.py`, `tests/dalg/test_dalg_synthetic.py` |
