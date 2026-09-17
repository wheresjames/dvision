# Target detection: dalg detectors

Detection is perception, not mission logic, and not every run needs it. Mapping
needs none, search needs one, and later missions may want a different detector
or several side by side. So detectors live in **dalg**, as a second product
beside evidence. dalg already owns sensor intake, frame and pose matching,
profiles, readiness, health and recording. Missions only read the result (see
[missions](missions.md)).

Detections are **belief, not truth**. They are stamped with their capture time,
causal and bounded. No detector, projection or estimate reads the world,
`dsim/truth` or the provider's `target.*` keys.

For the detector's own thresholds and the arithmetic of projection and
estimation, see [`docs/algorithms/red-disc.md`](algorithms/red-disc.md) and
[`docs/algorithms/detection-pipeline.md`](algorithms/detection-pipeline.md).

## Running one

```sh
python3 ./apps/dalg/dalg.py --id area1 --profiles lidar-baseline red-disc &   # evidence + detection
python3 ./apps/dalg/dalg.py --id area1 --profile lidar-baseline &             # mapping only: no detector runs
```

A detector profile holds detectors only and is composed beside an evidence
profile. Detectors run in the one dalg beside evidence, so `--profile red-disc`
on its own is refused: "detectors run beside evidence". A detector-only dalg
process comes after v1, because two dalg processes on one vehicle ID collide.

```json
{
  "name": "red-disc",
  "sources": [],
  "detectors": [
    {"sensor": "primary_camera", "detector": "red_disc", "label": "landing_pad",
     "settings": {"hz": 5, "max_range_m": 25, "gate_m": 1.5, "confirm_n": 3,
                  "expire_s": 30, "record_frames_hz": 0}}
  ]
}
```

- **Labels.** `label` is what a row's detections and estimates are published
  as, so it is the name a mission's `target_label` must use. Without it, a row
  publishes the label its detector declares in code (`red_disc`:
  `landing_pad`). The committed `red-disc` profile states it anyway, so the
  file says what the run publishes. A label may rename only a detector that
  declares exactly one label (`"label": "red_pad"`). It must be 1-120
  characters of `[A-Za-z0-9_.-]`, and it is part of the row's settings digest.
- **IDs.** A detector's ID is `<sensor>-<detector>`, with the sensor selector
  resolved (`front-red_disc`), exactly like a source ID. Two of the same
  detector on one sensor, with different settings, are told apart by a
  `suffix` (`front-red_disc-slow`). Each publishes separately. When several
  detectors run, their estimate IDs carry the detector ID too.
- **Settings every detector takes:**

| Setting | Default | Meaning |
|---|---|---|
| `hz` | 5 | frames processed per sim second; overdue frames are skipped and counted |
| `max_range_m` | 25 | ground range beyond which a detection is not projectable |
| `gate_m` | 1.5 | a projected detection this close to an estimate of its label joins it |
| `confirm_n` | 3 | observations before an estimate is confirmed |
| `expire_s` | 30 | sim seconds unseen before an estimate is marked stale (never deleted) |
| `record_frames_hz` | 0 | full frames archived per sim second, for offline detector work |
| `crop_px` | 96 | side of the crop archived around each detection |
| `candidates` | 8 | rejected candidates kept per frame for the overlay and archive |

  A detector with settings of its own adds them to the same object. Unknown
  keys are refused by name.
- **Validation.** Validation, the profile digest, preflight and the Profile tab
  editor cover detectors the same way they cover sources. In the editor, a row
  whose algorithm is a detector saves under `detectors`.
- **Readiness.** The readiness selector `algorithm:<selector>` also matches
  detector names and labels: `algorithm:red_disc`, `algorithm:landing_pad`.
  The heartbeat's capabilities list each detector's ID, name and labels, and
  name the detections plane.

## The detector contract

```python
@dataclass(frozen=True)
class PixelDetection:
    label: str; cx_px: float; cy_px: float
    extent_px: float          # radius, or half the larger box side
    confidence: float         # 0..1, the detector's own
    box_px: tuple | None = None
    diagnostics: dict = {}    # bounded, JSON

@dataclass(frozen=True)
class Candidate:              # considered and rejected
    cx_px: float; cy_px: float; reason: str   # "aspect 16.2 > 14", "area 3 < 25"

class Detector(Protocol):
    name: ClassVar[str]                     # registry key
    labels: ClassVar[tuple[str, ...]]      # what it emits; a profile row's label may rename a single one
    sensor_type: ClassVar[str]              # "camera.rgb"
    settings_class: ClassVar[type | None]   # its own settings dataclass, or None
    def __init__(self, config, intrinsics): ...
    def detect(self, frame) -> DetectorResult: ...   # detections + candidates
```

A detector works in **image space only**. It never projects, fuses or
publishes, so it can be unit-tested on saved images. A detector that needs
history keeps it inside its own instance. A new detector is one module under
`apps/dalg/detect/` and one line in `DETECTORS`. Nothing above the registry
changes.

`red_disc` is the first detector. It is daic's red-target detector, copied
with its thresholds unchanged. It emits `landing_pad`: the best-scoring
reddish, roughly elliptical blob, chosen exactly as daic chose it. Every other
blob is listed with the reason it lost: area, extent, solidity, aspect, fill,
or a lower score than the chosen blob. daic's copy stays where it is until daic
is removed. Nothing in the new stack imports it.

## Projection and estimates

`dalg/detect/projection.py` intersects the ray through a detection's pixel with
the ground plane `z = --z0-m`. It uses the frame's intrinsics and **the camera
pose delivered with the sample**, and the same camera axes as every evidence
algorithm. Each projection yields:

- `xy_m`, `range_m` and `bearing_deg`;
- a crude footprint uncertainty `sigma_m`, from the detection's extent at that
  slant range and one pixel's along-range smear.

A detection is **not projectable**, but is still published and recorded with
the reason, in three cases:

- its ray is at or above the horizon;
- the ground point is beyond `max_range_m`;
- the frame carried no camera pose.

The ground plane is flat. A pad on a raised surface projects long, and the
spread and range fields make that visible.

`dalg/detect/estimate.py` keeps one crude running estimate per target:

- **Association.** A projected detection joins the nearest estimate with the
  same label and detector within `gate_m`. Otherwise it opens a new one.
- **Position.** Each estimate keeps a confidence-weighted mean position and its
  spread (the weighted RMS distance of its observations).
- **Bookkeeping.** It also keeps an observation count, first and last sighting
  in sim time, and a revision that moves with every change.
- **States.** An estimate is `confirmed` at `confirm_n` observations and
  `stale` once unseen for `expire_s`. Seeing it again makes it fresh.

A mapping reset or epoch change clears every estimate, and the archive says so.
Estimate IDs (`landing_pad-1`) are stable within a mapping epoch.

Confirmation that needs the vehicle to move (go closer, look again) belongs to
the mission, not here.

## The detections plane

`/dvision2.<id>.detections.dalg` (`dvision2.detections.v1`, `dcmn/detections.py`)
is a retained snapshot with a 64 KiB cap. Its producer owns it, and a second
producer is refused. It holds:

- **detectors** (at most 8): ID, name, sensor, labels, settings digest, state
  (`waiting`, `running`, `failed`) and reason, processed, skipped and failed
  frames, effective and configured Hz, latency p50/p95, and the last frame's
  capture time;
- **detections** (at most 16): the latest processed frame's detections per
  detector. Each carries pixel fields and confidence, the capture time and
  sequence, and either the projection (`xy_m`, `range_m`, `bearing_deg`,
  `sigma_m`) or why there is none, plus the estimate it joined;
- **estimates** (at most 32, confirmed first): ID, label, detector, `xy_m`,
  spread, count, revision, confirmed, stale, first and last seen, confidence.

`revision` counts publications. `time_s` is dalg's sim time, so a consumer
judges freshness by sim time. The plane is republished whenever a frame was
processed or an estimate changed, and at least once a sim second.

Module bus events: `detection.estimate_opened`, `detection.estimate_confirmed`,
`detection.estimate_stale`, `detection.detector_failed`.

**Missions declare needs, not detectors.** A mission with a `target_label`
parameter is started only when some detector on the plane publishes that label.
If a `target_detector` is named, it must be that detector. Only such a mission's
observations carry detections and estimates. goto, waypoints and explore never
open the plane.

## Watching it: the dalg window

- **Live.**
  - The selector lists detectors beside evidence sources.
  - With a detector selected, the sensor pane draws the frame the detector
    actually processed: detections circled with label and confidence,
    rejected candidates as numbered dim markers, and a corner stamp (capture
    time, frame age, detect latency). The next pane lists the candidates'
    reasons.
  - "Overlay detectors on camera sources" draws every detector's latest
    detections over an evidence camera's frames.
- **Detections.**
  - A map over the evidence grid, or over the detections alone when there is
    none. It shows projected detections fading with age, estimates with their
    spread circle and ID (hollow when unconfirmed, solid when confirmed, grey
    when stale) and each detector's camera footprint.
  - An estimates table; selecting a row highlights that estimate on the map.
  - Per-detector health rows: state, processed / skipped / failed frames,
    effective and configured Hz, latency p50/p95, detections per frame,
    not-projectable counts by reason, last error.
  - A sim-time strip per detector: green ticks for frames with detections, dim
    for empty frames, amber for skips, red for failures.
- **Status line.** Adds `detectors=<n> estimates=<confirmed>/<total>` and names
  a failed detector.

A detector that raises is counted and recorded, and marked `failed` on the
plane and in the window. Evidence keeps publishing, and so do other detectors.

## After the fact

The archive, reports and `apps/dcmn/archive.py` options (`--detections`,
`--estimate ID`, `--detection N --out PATH`) are described in
[reports](reports.md). In short:

- every processed frame, empty ones included, every detection with its
  projection inputs, and every estimate change are recorded;
- a crop around each detection is kept, plus the full frame for each
  estimate's first observation;
- `dalg.detect.projection.replay(inputs)` re-projects a recorded detection
  exactly.

## Testing detectors

`dtest/provider.py` takes `pads=[(x_m, y_m, radius_m)]` (`--pads x,y[,r];...`
as a process). These are flat saturated red discs painted into the fixture
camera's floor, 0.6 m radius by default. The fixture is the provider and may
hold that truth; dalg only ever sees the pads through the camera. Process
tests check plumbing, plus one loose projection guard: the estimate lies
within 2 m of the pad. That is not a detector-quality metric and is never
tightened. If `red_disc` misses a fixture pad, fix the fixture, not the
detector.
