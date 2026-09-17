# `feature_triangulation`: ORB matches and two-bearing intersection

The sparse counterpart to the dense algorithms. It finds ORB keypoints in each
keyframe, matches them against a recent keyframe, and intersects the two
**horizontal bearings** of each match to get a map position. Known metric poses
do all the work a bundle adjustment would otherwise have to: there is no
structure-from-motion, no scale ambiguity and no optimiser.

Profile: `features-baseline`, sensor selector `primary_camera`.

```sh
python3 ./apps/dalg/dalg.py --id area1 --profile features-baseline
```

## The pipeline

```
  frame ──> grayscale
              │
              ├─ reject unless the camera has moved keyframe_distance_m (0.35) since
              │  the last keyframe kept
              ▼
        ORB detectAndCompute (max_features 1000)
              │
              ├─ no descriptors (a blank view) ──> dropped, not even kept as a keyframe
              ▼
        walk the kept keyframes NEWEST FIRST (a deque of MATCH_WINDOW = 8):
              │
              ├─ skip unless min_baseline_m <= baseline <= max_baseline_m (0.35..4.0)
              ├─ skip unless |heading difference| <= max_heading_delta_deg (35°)
              │
              ▼  the first keyframe that qualifies is the only one used
        BFMatcher(NORM_HAMMING).knnMatch(old, new, k=2)
              │
              ├─ Lowe ratio test: best.distance < 0.72 · second.distance
              │      (a row with fewer than 2 matches is skipped: a near-textureless
              │       frame yields rows of one and the ratio test has nothing to
              │       compare against)
              ▼
        triangulate_xy(old_pose, old_px, new_pose, new_px, ...)
              │
              ▼
        fuse_endpoint(grid, camera_pose, point)     free −0.7 along the ray, +2.5 at it
```

Only the newest eight keyframes are kept. Older ones are never looked at
again, so holding them was unbounded memory spent on nothing — but the
`keyframes` counter survives separately, because a capped list would quietly
cap the reported number too.

## `triangulate_xy`, in full

This is the shared helper (`apps/dalg/algo/spatial.py`) that this algorithm and
[`optical_flow_triangulation`](optical-flow-triangulation.md) both use. Only
the **column** of a pixel is used: each observation is a vertical plane through
the camera, reduced to a bearing on the map.

```
  bearing(pose, px) = heading + atan((px − cx) / fx)

                        A                             B
                        o───────── delta ─────────────o
                         ╲ da                     db ╱
                          ╲                         ╱
                           ╲        separation     ╱
                            ╲         angle       ╱
                             ╲         ⌄         ╱
                              ╲       ╱ ╲       ╱
                               ╲     ╱   ╲     ╱
                                ╲   ╱     ╲   ╱
                                 ╲ ╱       ╲ ╱
                                  X    the intersection
                            ta ───┘     └─── tb
```

```python
cross      = da.x·db.y − da.y·db.x
separation = |asin(cross)|                  degrees between the two bearings
ta = (delta.x·db.y − delta.y·db.x) / cross  distance along A's bearing
tb = (delta.x·da.y − delta.y·da.x) / cross  distance along B's bearing
```

Four rejections, and each one exists because the geometry is otherwise a lie:

| Test | Why |
|---|---|
| `separation >= min_parallax_deg` (1.5°) | near-parallel bearings put the intersection anywhere; range error explodes as `1/sin(separation)` |
| `abs(cross) >= 1e−6` | numerically parallel: no intersection at all |
| `0.15 < ta < max_range_m` and the same for `tb` | the intersection must be **in front of both** cameras and inside the useful range; a negative `t` is the back-projection, which is not an observation |
| `‖point_a − point_b‖ <= 0.5 m` | the two bearings are skew in 3-D; if their closest points disagree by more than half a metre the match is wrong, not noisy |

The result is the midpoint of the two.

## Settings

`FeatureConfig`:

| Setting | Default | Meaning |
|---|---|---|
| `max_features` | 1000 | ORB keypoint budget per frame |
| `min_baseline_m` | 0.35 | narrower pairs give too little parallax |
| `max_baseline_m` | 4.0 | wider pairs stop matching: ORB is not that viewpoint-invariant |
| `max_heading_delta_deg` | 35.0 | generous, because matching is descriptor-based, not rectified |
| `ratio_test` | 0.72 | Lowe's ratio; lower is stricter |
| `min_parallax_deg` | 1.5 | the triangulation gate above |
| `max_range_m` | 25.0 | both `ta` and `tb` must be inside this |
| `keyframe_distance_m` | 0.35 | how far the camera must move for a new keyframe |

There is no obstacle band and no height gate — see below. Fusion takes
`fuse_endpoint`'s defaults: free −0.7, occupied +2.5. `preview()` and
`finish()` are identical; everything is fused in `observe`.

Diagnostics: `keyframes`, `matches`, `triangulated_points`.

## What it gets wrong

- **It has no idea how high anything is.** Two horizontal bearings fix a map
  position and say nothing about z, so a matched feature on the **floor**
  becomes an obstacle at that floor position, and one on the ceiling does too.
  This is the single biggest difference from the dense algorithms, which gate
  on `obstacle_band`.
- **Sparse by construction.** It marks the cells it triangulates and carves the
  rays to them; the space between features stays unknown. The grid looks like a
  dotted outline, not a wall.
- **Wrong matches become confident obstacles.** The ratio test and the 0.5 m
  agreement test catch most, but repetitive texture defeats both.
- **One partner per keyframe.** The first eligible keyframe, newest first, is
  the only one matched against — cheap, and it means a keyframe that would have
  matched better against an older partner simply does not.

## Where it lives

| Thing | File |
|---|---|
| The algorithm | [`apps/dalg/algo/features.py`](../../apps/dalg/algo/features.py) |
| `triangulate_xy`, `bearing`, `fuse_endpoint` | [`apps/dalg/algo/spatial.py`](../../apps/dalg/algo/spatial.py) |
| Tests | `tests/dalg/test_dalg_evidence_algorithms.py`, `tests/dalg/test_dalg_synthetic.py` |
