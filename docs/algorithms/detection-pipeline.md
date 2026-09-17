# The detection pipeline: scheduling, projection, estimates

Everything between a detector and a mission. A detector works in pixels; this
is what turns its output into a position on the map, a belief that survives
more than one frame, and a published plane — and it is shared by **every**
detector, so a new detector is one module and one line in `DETECTORS`.

Source: [`apps/dalg/detect/host.py`](../../apps/dalg/detect/host.py),
[`detect/projection.py`](../../apps/dalg/detect/projection.py),
[`detect/estimate.py`](../../apps/dalg/detect/estimate.py). Operational view:
[`docs/detection.md`](../detection.md).

```
  camera frame
      │
      ▼  host: rate, frame checks
  detector.detect(frame) ──> PixelDetection(cx_px, cy_px, extent_px, confidence)
      │                      + rejected Candidates, each with a reason
      ▼  projection
  project(...) ──> Projection(projectable, xy_m, range_m, bearing_deg, sigma_m)
      │
      ▼  estimates
  associate(...) ──> Estimate(xy_m, spread_m, count, confirmed, stale, ...)
      │
      ▼
  detections plane  +  archive events
      │
      ▼
  a mission that declared target_label reads it (never the detector, never truth)
```

## Scheduling: one rate per detector

Each detector keeps its own `PeriodicDeadline` against the **data clock**.

```
  hz = 5  →  one frame every 0.2 sim seconds

  frames arriving:   │ │ │ │ │ │ │ │ │ │ │ │
  processed:         ▲       ▲       ▲          the newest frame each time
  skipped:            └┴┘     └┴┘     └┴┘       counted, never queued
```

A detector that falls behind **skips frames and counts them**, so a slow
detector never holds evidence up, and the skip count is in the report rather
than hidden as latency. Each processed frame is the newest one its sensor
delivered, checked by the run exactly as an evidence frame is: capture pose
present, same frame and epochs, not from the future.

For every processed frame — **empty ones included**, since a miss rate and "it
stopped seeing the pad at 14.2 s" need them — the archive gets a
`detector.frame`; each detection a `detector.detection` with its pixel fields,
projection inputs and result, and the estimate it joined; rejected candidates a
`detector.candidates`; overruns and exceptions `detector.skipped` /
`detector.failed`; every estimate change an `estimate.*` event; and every
publication a `detections.published`.

Imagery rides as optional archive payloads: a crop around each detection
(`crop_px`, default 96), the full frame of each estimate's first observation,
and full frames at `record_frames_hz` when set.

## Projection: pixel to ground

The ray through the detection's pixel is intersected with the ground plane
`z = z0_m`.

```
  side view

     camera at (x, y, z), attitude from camera_axes()
        C
        │╲
        │ ╲  direction = u·right + v·down + forward
   h    │  ╲       u = (cx_px − cx)/fx
        │   ╲      v = (cy_px − cy)/fy
        │    ╲
        │     ╲                 t = h / (−direction.z)
   ─────┴──────●──────────      ground point = C + t·direction
   z = z0_m    ↑
          ground range = hypot(dx, dy)     must be <= max_range_m
```

It uses **the camera pose delivered with the frame** (`Frame.camera_pose`),
never the body pose plus `camera.*` status keys — which mixes an absolute pitch
with a relative mount offset and biases every range. The axes are
`camera_axes`, the same ones every evidence algorithm uses.

A detection is **not projectable** — still published and recorded, with the
reason — when:

| Reason | Condition |
|---|---|
| `no camera pose with the frame` | the frame carried none |
| `at or above the horizon` | `direction.z >= −1e−6`, or the camera is at/below `z0_m` |
| `range N m > M m` | the ground point is beyond `max_range_m` |

### `sigma_m` is a footprint, not a covariance

```
  size  = extent_px · slant / fx                  ground size of the blob
  smear = slant / (fy · sin(depression))          along-range smear of one pixel
  sigma = hypot(size, smear)
```

Deliberately crude: a scale for gating and display. The `sin(depression)` floor
(`MIN_DEPRESSION_SIN = 0.02`) is what makes a ray grazing the ground report a
large sigma instead of an infinite one — the geometry that makes an oblique
view's centre untrustworthy shows up here as a large number rather than a
crash.

Every projection keeps its inputs (`inputs()`), so `replay()` can redo it from
an archived record and **must get the same answer**. That is the test that keeps
this module honest.

## Estimates: association and a weighted mean

Deliberately simple, and explicitly *not* a filter. A projected detection joins
the nearest estimate of the same label **from the same detector** within
`gate_m`; otherwise it opens a new one.

```
  top view, gate_m = 1.5

              ╭ ─ ─ ─ ─ ─ ╮
            ╱   estimate    ╲         new detection at ✚:
           │    landing_pad-1│          distance 0.9 m <= 1.5 m  → joins it
           │        ⊗    ✚   │
            ╲   xy_m       ╱          new detection at ✱:
              ╰ ─ ─ ─ ─ ─ ╯             distance 4.2 m  > 1.5 m  → opens
                                        landing_pad-2
                              ✱
```

Each estimate keeps confidence-weighted sums, and the reported quantities are
properties derived from them:

```
  weight += w                     w = max(confidence, 1e−3)
  sum_x  += w·x     sum_y += w·y
  sum_r2 += w·(x² + y²)

  xy_m       = (sum_x/weight, sum_y/weight)                weighted mean
  spread_m   = sqrt(sum_r2/weight − |xy_m|²)               weighted RMS scatter
  confidence = weight / max(count, 1)                      mean detector confidence
```

`spread_m` is the number a mission reports when it lands: it says how much the
observations of this thing disagreed, which is the honest summary of a crude
estimator.

State transitions, each returning an event the host records:

```
              first observation
                    │
                    ▼
            ┌──────────────┐  count >= confirm_n (3)   ┌───────────────┐
            │  unconfirmed │ ────────────────────────> │   confirmed   │
            └──────────────┘                           └───────────────┘
                    │                                          │
       unseen for expire_s (30 s)                   unseen for expire_s
                    │                                          │
                    ▼                                          ▼
            ┌───────────────────────────────────────────────────────┐
            │  stale  — MARKED, never deleted; seen again → fresh   │
            └───────────────────────────────────────────────────────┘

  epoch change ──> clear(): every estimate forgotten, one `cleared` event each
```

Estimates are never deleted by ageing, only marked, because "there was
something here at 14.2 s and it is not there now" is information a mission and a
report both need. They *are* dropped wholesale when the mapping epoch changes,
since their coordinates no longer mean anything.

IDs are `landing_pad-1`, `landing_pad-2`, … prefixed with the detector ID and a
slash when more than one detector runs (`front-red_disc/landing_pad-1`; no
prefix at all for a single detector), and stable for a mapping epoch. `MAX_TRACKED` is 256; beyond it, a detection that would open
another is counted in `dropped` and discarded.

**Confirmation that needs the vehicle to move** — go closer, turn, look again —
is the mission's business, not this module's. See
[`docs/missions/search.md`](../missions/search.md).

## Publication

The detections plane (`dcmn.detections`) is published whenever a frame was
processed or an estimate changed, and **at least once a sim second** so a
consumer can judge its freshness by sim time. What it carries is ordered and
capped:

```
  estimates: confirmed first, then fresh, then most recently seen
             (sort key: not confirmed, stale, −last_s, id)
  caps:      MAX_DETECTORS 8, MAX_DETECTIONS 16, MAX_ESTIMATES 32
```

A mission never sees the plane raw: `detections_view` in
[`apps/dmsn/inputs.py`](../../apps/dmsn/inputs.py) filters it to the mission's
label, its named detector if it named one, and its `min_confidence` for
detections. Estimates carry their own confidence and the mission judges it
itself.

## Settings every detector takes

`DetectorConfig`, validated by `check()` — a typo in a profile is refused by
name, so it never silently runs defaults.

| Setting | Default | Meaning |
|---|---|---|
| `hz` | 5.0 | frames processed per sim second; overdue frames are skipped and counted |
| `max_range_m` | 25.0 | ground range beyond which a detection is not projectable |
| `gate_m` | 1.5 | a projected detection this close to an estimate of its label joins it |
| `confirm_n` | 3 | observations before an estimate is confirmed |
| `expire_s` | 30.0 | sim seconds unseen before an estimate is marked stale |
| `record_frames_hz` | 0.0 | full frames archived per sim second; 0 keeps none |
| `crop_px` | 96 | side of the square crop archived around each detection (16..512) |
| `candidates` | 8 | rejected candidates kept per frame for the overlay and archive (0..32) |

A detector with settings of its own names a dataclass as `settings_class`;
`split_settings` divides one profile `settings` object between the two and
refuses unknown keys by name.

## Where it lives

| Thing | File |
|---|---|
| Host: rates, archive, publication | [`apps/dalg/detect/host.py`](../../apps/dalg/detect/host.py) |
| Projection and its replay | [`apps/dalg/detect/projection.py`](../../apps/dalg/detect/projection.py) |
| Estimates | [`apps/dalg/detect/estimate.py`](../../apps/dalg/detect/estimate.py) |
| Plane schema and caps | [`apps/dcmn/detections.py`](../../apps/dcmn/detections.py) |
| The mission-side view | `detections_view` in [`apps/dmsn/inputs.py`](../../apps/dmsn/inputs.py) |
| Tests | `tests/dalg/test_dalg_detect.py`, `tests/dalg/test_dalg_detection_run.py`, `tests/dmsn/test_dmsn_search.py` |
