# `red_disc`: the red landing-pad detector

The first registered detector, and the one `search` missions use. It finds a
**reddish, roughly elliptical blob** in one frame and reports at most one
detection per frame: a chromatic gate picks red pixels, and a shape test then
decides whether the blob could be a circular pad seen from some viewpoint.

It works in **image space only**. Projecting the detection onto the ground,
keeping a running estimate and publishing are the host's job — see
[`detection-pipeline.md`](detection-pipeline.md).

Profile: `red-disc` (`assets/algorithm_profiles/red-disc.json`),
composed beside an evidence profile:

```sh
python3 ./apps/dalg/dalg.py --id area1 --profiles lidar-baseline red-disc
```

Label: `landing_pad`. Sensor type: `camera.rgb`. It has **no settings of its
own** — `settings_class` is `None` — on purpose: its thresholds are constants in
the source, each with the measurement that chose it.

## Why the gate is chromatic and not photometric

Hue and saturation are properties of the *surface*; value is a property of the
*illumination* on it. A red pad in open sun, in overcast, in a tree's shadow,
or under a camera that has just stopped down all produce the same hue and much
the same saturation, and wildly different values.

This is not a theoretical concern. An earlier version required `V >= 120`,
which is a statement that the pad is brightly lit. When the simulator gained a
physically-shaded scene the pad rendered at `V = 90` — hue and saturation
perfect, brightness merely lower — and the detector reported **nothing at all
on every frame of every approach**. A pad in shade is still a pad.

So value is used only as a noise floor: below `V = 30` a pixel is dark enough
that its hue is quantisation noise rather than colour.

```
  OpenCV hue, 0..180 (red wraps around 0)

   0    5        8      12   20                90                175   180
   │────│────────│──────│────│─────────────────│─────────────────│─────│
   ▓▓▓▓▓▓                                                        ▓▓▓▓▓▓▓
   └─ pad: H 0-2 ─┘                                              └ wrapped
        │      │
        │      └── tree canopy H ~8         brick H 12-20 ──┘
        │
        └── the band is ±5, tight enough to exclude both

  plus  S >= 175     (pad measures 227-250; background median ~80, p99 ~213)
  and   V >= 30      (noise floor, not an illumination gate)

  morphology: CLOSE 3x3 ellipse, twice.  CLOSE ONLY — an opening step would
  erase the pad at range, where it is a few pixels tall, which is exactly the
  distance the search phase needs it at.
```

## Why the shape test fits an ellipse

A circular pad is only circular from directly above. From a drone on approach
it is an ellipse whose aspect ratio is about `1/sin(elevation)`:

```
  elevation 90°        elevation 30°        elevation 6°
  (overhead)                                (on approach, 9 m out at 1 m up)

     ╭─────╮              ╭───────╮
    │       │            ╰─────────╯          ────────────
    │       │             aspect 2:1          aspect ~9:1
     ╰─────╯
     aspect 1:1
```

Testing *circularity* would therefore amount to testing that the drone is
nearly overhead — which is the one phase of flight where the pad is easiest to
find anyway. Instead the blob is fitted with an ellipse and judged on how well
it **fills** that ellipse, which is invariant to viewing angle, plus solidity
and extent, which reject the ragged, fragmented blobs that foliage and
brickwork produce.

## The reject cascade

Every blob is either chosen or rejected **with a reason**, and the reasons are
published and archived, so "why didn't it see the pad?" is answerable from the
overlay alone. At most `candidates` (8) rejections are kept per frame, largest
area first.

```
  connectedComponentsWithStats(mask, 8)
     │
     ├─ area < 25 px .................................. "area N < 25"
     ├─ extent = area / bbox area < 0.50 .............. "extent N < 0.5"
     │      an ellipse scores pi/4 = 0.785
     │
     ├─ clipped?  (blob touches any frame edge)
     │      → remember it; the fill test cannot apply to a partial shape.
     │        This happens on EVERY final descent, when the pad fills the
     │        bottom of the image.
     │
     ├─ contour, >= 5 points, hull area > 0:
     │      solidity = area / hull area < 0.70 ........ "solidity N < 0.7"
     │      pad's 1st percentile is 0.70; foliage fragments median 0.46
     │
     ├─ area >= 60 px: fitEllipse(contour)
     │      centre and radius are REPLACED by the ellipse's
     │      fill   = area / (pi/4 · major · minor)
     │      aspect = major / minor
     │
     ├─ aspect > 14 .................................... "aspect N > 14"
     │      14 = 1/sin(4°): beyond that the pad is a line and its centre is
     │      not usefully localised
     ├─ fill outside 0.82..1.60, unless clipped ........ "fill N outside .."
     │      a filled ellipse scores 1.0; observed pads 0.91+; ragged
     │      foliage fragments ~0.68
     │
     ▼
  score = area · min(fill, 1.0)
     │
     └─ the highest score wins; every other survivor is rejected with
        "score N below the chosen blob"
```

Below `area = 60` an ellipse fit is not meaningful, so `fill` and `aspect` stay
`None` and extent alone carries the shape test. That is the regime the search
phase depends on — a pad at 20 m is a handful of pixels — and it is also where
false alarms live, which is why the host requires several observations before a
mission acts (`confirm_n`).

## Confidence

```
  quality    = min(fill, 1.0)            or 1.0 when no ellipse was fitted
  confidence = quality / max(1, aspect/4)        clamped to 0..1
```

It blends how cleanly the blob fills its ellipse with **how oblique the view
is**: a pad seen nearly edge-on is a real detection whose centre is a poor
position estimate, and a consumer should weigh it accordingly. The estimate
layer does exactly that — confidence is the weight in its weighted mean.

## What a detection carries

`PixelDetection(label, cx_px, cy_px, extent_px, confidence, box_px,
diagnostics)`, where `extent_px` is half the ellipse's major axis (or half the
larger box side), and `diagnostics` holds `area_px`, `extent`, `clipped`,
`solidity`, `fill`, `aspect` and `score` — bounded, JSON-serialisable, and kept
in the archive.

## What it gets wrong

- **Anything red and roughly convex passes.** There is no texture, no size
  prior and no context. A red jacket, a red car roof or a stop sign lying flat
  is a landing pad.
- **Sensor noise and JPEG chroma artefacts** can fringe a canopy edge into the
  red band. Such blobs are ragged and flicker frame to frame, which is what
  `confirm_n` is for.
- **One detection per frame.** Two pads in view means the lower-scoring one is
  reported only as a rejected candidate.
- **The ellipse centre is not the pad centre at high aspect.** Hence the
  confidence penalty, and hence `search`'s standoff-and-look-again phase rather
  than a visual servo.
- **Thresholds are tuned to this simulator's scene.** They are constants, not
  settings, so a different scene means editing the file — and the comments
  record the measurements that justified each number so that edit is informed.

## Where it lives

| Thing | File |
|---|---|
| The detector | [`apps/dalg/detect/red_disc.py`](../../apps/dalg/detect/red_disc.py) |
| Registry, `PixelDetection`, `Candidate`, `DetectorConfig` | [`apps/dalg/detect/__init__.py`](../../apps/dalg/detect/__init__.py) |
| Operational doc | [`docs/detection.md`](../detection.md) |
| Tests | `tests/dalg/test_dalg_detect.py`, `tests/dalg/test_dalg_detection_run.py` |

Its thresholds came unchanged from `apps/daic/detector.py`. daic is legacy:
nothing in the new stack may import it, and the only change made in the copy
was that every rejected blob now says why.
