# `optical_flow_triangulation`: Lucas–Kanade tracks, triangulated

The same arithmetic as [`feature_triangulation`](feature-triangulation.md) with
a different correspondence engine. Instead of detecting and describing
keypoints in both frames and matching descriptors, it detects corners in the
**older** frame and *tracks* them into the newer one with pyramidal
Lucas–Kanade. Each surviving track is a pair of pixel columns, and the two
bearings are intersected exactly as before.

Profile: `optical-flow-baseline`, sensor selector `primary_camera`.

```sh
python3 ./apps/dalg/dalg.py --id area1 --profile optical-flow-baseline
```

Tracking suits small inter-frame motion, which is what a camera at a few frames
a second on a slow vehicle actually delivers; descriptor matching suits large
viewpoint changes. That is the whole trade between these two algorithms.

## The pipeline

```
  frame ──> grayscale
              │
              ▼
        is there an anchor frame?
              │ no ──> become the anchor; nothing else happens
              │ yes
              ▼
        baseline = |camera position − anchor position|
              │
              ├─ baseline < min_baseline_m (0.3) ──> DO NOTHING, and keep the
              │                                      SAME anchor: the vehicle is
              │                                      still accumulating parallax
              ▼
        goodFeaturesToTrack(anchor)          max_corners 800,
              │                              quality_level 0.02,
              │                              min_distance_px 7
              ▼
        calcOpticalFlowPyrLK(anchor → new)   per corner: moved position + status
              │
              ├─ status == 0 ──> dropped
              ▼
        triangulate_xy(anchor_pose, old_px, new_pose, new_px, ...)
              │                    min_parallax_deg 1.5, max_range_m 25
              ▼
        fuse_endpoint(grid, camera_pose, point)
              │
              ▼
        the new frame becomes the anchor
```

The anchor discipline is worth reading twice: the anchor advances **only when a
pair was actually processed**. A vehicle creeping forward does not burn its
baseline one centimetre at a time; the anchor is held until the camera has
moved `min_baseline_m`, and then the widest available pair is the one used.

## Settings

`OpticalFlowConfig`:

| Setting | Default | Meaning |
|---|---|---|
| `max_corners` | 800 | corner budget in the anchor frame |
| `quality_level` | 0.02 | Shi–Tomasi threshold, as a fraction of the best corner's score |
| `min_distance_px` | 7.0 | minimum spacing between corners |
| `min_baseline_m` | 0.3 | the anchor is held until the camera has moved this far |
| `min_parallax_deg` | 1.5 | triangulation gate (see [`triangulate_xy`](feature-triangulation.md#triangulate_xy-in-full)) |
| `max_range_m` | 25.0 | both intersection distances must be inside this |

No heading test, no maximum baseline, no obstacle band. Fusion takes
`fuse_endpoint`'s defaults: free −0.7 along the ray, +2.5 at the point.
`preview()` and `finish()` are identical.

Diagnostics: `frames`, `tracks`, `triangulated_points`.

## What it gets wrong

Everything [`feature_triangulation`](feature-triangulation.md#what-it-gets-wrong)
gets wrong — no height information, sparse output, confident wrong points —
plus two of its own:

- **No maximum baseline and no heading gate.** Lucas–Kanade is asked to track
  across whatever motion happened since the anchor. Past a rotation of any
  size the tracks fail, and `status` is the only thing rejecting them.
- **Aperture-effect drift.** A track that slides along an edge instead of
  following a point produces a *plausible* pixel pair and therefore a
  plausible, wrong intersection. The 0.5 m bearing-agreement test inside
  `triangulate_xy` is the only defence, and it does not always fire.

## Where it lives

| Thing | File |
|---|---|
| The algorithm | [`apps/dalg/algo/optical_flow.py`](../../apps/dalg/algo/optical_flow.py) |
| `triangulate_xy`, `fuse_endpoint` | [`apps/dalg/algo/spatial.py`](../../apps/dalg/algo/spatial.py) |
| Tests | `tests/dalg/test_dalg_evidence_algorithms.py`, `tests/dalg/test_dalg_synthetic.py` |
