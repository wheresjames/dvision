# `plane_sweep`: photometric cost over hypothesised depths

Where [`sgbm`](sgbm.md) asks OpenCV to find the disparity, the plane sweep
**guesses the depth and checks**. For each of a fixed set of candidate
distances it warps an earlier keyframe into the current one as if the whole
scene sat at that distance, measures how badly the two disagree, and keeps the
distance that disagreed least — if it won clearly enough to be believed.

Profile: `plane-sweep-baseline`, sensor selector `primary_camera`.

```sh
python3 ./apps/dalg/dalg.py --id area1 --profile plane-sweep-baseline
```

It is far more permissive about motion than `sgbm` (15° of heading change
against 2°), because it never assumes rectified geometry — it re-projects
explicitly for every hypothesis.

## The sweep

```
  reference keyframe R at pose P          partner keyframe Q at pose P'

  for each hypothesised depth d (16 of them):

      every pixel column of R defines a world ray;
      place a point on it at distance d; project that point into Q;
      sample Q there; compare with R.

                      d1      d2      d3   ...        d16
                       ·       ·       ·                ·
      R ──────ray──────┼───────┼───────┼────────────────┼──────>
                        ╲       ╲       ╲                ╲
                         ╲       ╲       ╲                ╲       where each
                          ▼       ▼       ▼                ▼      hypothesis
      Q ────────────────[px1]  [px2]  [px3]  ...........[px16]    lands in Q

  cost volume:  |R − warp(Q, d)| box-filtered over a window_px window

      cost
       │   ●               ●
       │     ●           ●
       │       ●       ●
       │         ●   ●
       │           ●            <- argmin = the accepted depth
       └───────────────────────> d
             first  │
                    └ second: the best cost outside a ±1 neighbourhood
```

Depths are spaced **uniformly in inverse depth** (`1/linspace(1/dmin, 1/dmax)`),
which is where the information actually is: a pixel shift buys far more depth
resolution near the camera than far away, so uniform metric spacing wastes most
of the 16 hypotheses on distances the images cannot distinguish.

The projection into the partner is done in the map frame, not with an epipolar
shortcut:

```python
rays = arctan((xx − cx)/fx) + heading          # world bearing of each column
wx = pose.x_m + sin(rays)·depth                # the hypothesised point
wy = pose.y_m − cos(rays)·depth
forward = dx·sin(h') − dy·cos(h')              # into the partner's frame
right   = dx·cos(h') + dy·sin(h')
px = cx + fx·right/max(forward, 1e−6)          # its column in the partner
```

Pixels whose hypothesis falls behind the partner camera or outside its frame
get cost 1 (the maximum), so they can never win.

## Accepting or rejecting a depth

Three tests, all required, and all of them are about **whether the minimum is
trustworthy**, not about whether it is small:

```
  confident = (second − first) / second  >=  confidence_margin (0.18)   distinctness
            & first                      <=  max_cost          (0.15)   absolute fit
            & sqrt(local variance)       >=  min_texture_std   (0.025)  texture
```

The `second` best is taken **excluding the winner and its two neighbours**, so a
broad but genuine minimum is not rejected for having a close runner-up one step
away. The texture test is the one that matters most in practice: on a blank
wall every hypothesis costs about the same, the argmin is noise, and without
the test the grid fills with confident nonsense.

Rejected pixels become `NaN` and are simply not fused.

## From depth to cells

The sweep hypothesises **horizontal radial distance** — the ray it builds per
column is level — so the row of a pixel enters only as the ray's *pitch*, which
is what decides whether a sample is a wall or the patch of floor a metre ahead:

```
  for each strided pixel (px, py):

      ray_pitch = camera pitch − atan((py − cy)/fy)
      height    = camera z + depth · tan(ray_pitch)
      reject unless min_height_m <= height <= max_height_m

      yaw   = heading + atan((px − cx)/fx)
      point = (x + sin(yaw)·depth,  y − cos(yaw)·depth)

  fuse_endpoints(grid, pose, samples, free=−0.7, occupied=+4.0)
```

Fusion goes through `fuse_endpoints` — nearest first, occluded samples dropped
— for the same reason `ground_plane` does: no level-ray hypothesis fits a
vertical face, so pixels looking at one overshoot, matching metres beyond the
real thing. The rule and its blind spot are in
[`occupancy-grid.md`](occupancy-grid.md#fuse_endpoints).

## Keyframes, and the two ways it runs

A frame is kept only if it is at least `min_keyframe_distance_m` (0.5 m) from
the last one kept, downscaled by `match_scale` (0.5) and converted to float
grayscale in 0..1. The intrinsics are scaled to match.

```
  EVIDENCE (live, causal)                  SCORED (a captured tour)
  ─────────────────────────                ────────────────────────
  sliding window of 32 keyframes           hard stop at max_keyframes (256)
  fuse_available():                        _process():
    partner = widest baseline among          partner = widest baseline among
      keyframes ALREADY CAPTURED               ALL keyframes, either side
    cells stamped with the new
      keyframe's capture time
  a keyframe with no earlier partner
    is not fused; a later keyframe
    can still pair with it
```

Partner eligibility is just two tests — baseline in
`[min_baseline_m, max_baseline_m]` and heading within
`max_heading_delta_deg` — and the widest wins. As with `sgbm`, dalg's run loop
also calls `preview()` per admitted frame, which advances `_process` on the same
grid, so a keyframe can be fused both causally and by the preview pass.

## Settings

`PlaneSweepConfig`:

| Setting | Default | Meaning |
|---|---|---|
| `depth_min_m` / `depth_max_m` | 0.5 / 15.0 | the hypothesis range; validated as a proper interval |
| `depth_steps` | 16 | hypotheses, spaced in inverse depth; must be >= 4 |
| `min_baseline_m` / `max_baseline_m` | 0.5 / 2.5 | partner geometry |
| `max_heading_delta_deg` | 15.0 | partner attitude |
| `min_keyframe_distance_m` | 0.5 | how far the camera must move for a new keyframe |
| `max_keyframes` | 256 | scored path only; the evidence copy slides a 32-frame window instead |
| `match_scale` | 0.5 | image downscale before matching; must be in (0, 1] |
| `window_px` | 5 | box filter side for the cost and for the texture measure |
| `min_texture_std` | 0.025 | texture floor, in 0..1 intensity units |
| `confidence_margin` | 0.18 | how much better the winner must be than the runner-up |
| `max_cost` | 0.15 | absolute fit required of the winner |
| `stride` | 4 | every Nth pixel of the accepted depth map is fused |
| `free_log_odds` | −0.7 | per crossed cell |
| `occupied_log_odds` | 4.0 | at the endpoint |
| `min_height_m` / `max_height_m` | 0.25 / 4.5 | height gate on the sample |
| `preview_batch` | 2 | keyframes fused per `preview()` |
| `preview_interval_s` | 0.75 | how often a front end should ask |

Diagnostics: `frames`, `frames_with_depth`, `depth_hypotheses`,
`accepted_points`.

## What it gets wrong

- **Depth resolution is quantised to 16 steps.** Nothing between two
  hypotheses can be reported; a wall at 7 m lands on whichever hypothesis is
  nearest, and the inverse spacing makes those steps coarse far away.
- **Nothing beyond `depth_max_m` exists.** A surface at 30 m matches best at
  the 15 m hypothesis or fails the cost test; it never reports 30 m.
- **Level-ray hypotheses on vertical faces** overshoot, which is what the
  occlusion-checked fusion is there to contain.
- **Repetitive texture** passes the texture test and can pass the margin test
  too, which is the worst combination: a confident wrong depth.
- **The cost volume is the run's expense.** `depth_steps · height · width` per
  pair, at `match_scale`; halving `match_scale` quarters it.

## Where it lives

| Thing | File |
|---|---|
| The algorithm | [`apps/dalg/algo/plane_sweep.py`](../../apps/dalg/algo/plane_sweep.py) |
| `fuse_endpoints` | [`apps/dalg/algo/spatial.py`](../../apps/dalg/algo/spatial.py) |
| Tests | `tests/dalg/test_dalg_evidence_algorithms.py`, `tests/dalg/test_dalg_synthetic.py` |
