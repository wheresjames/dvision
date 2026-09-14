# Run reports

Every dvision2 run writes its artifacts into one directory tree, owned by the
simulator and shared by every module attached to that instance. This document
is the contract: where the tree is, who creates it, what goes in it, and what a
new module must do to join in.

---

## 1. Layout

```text
reports/<id>/<run>/
  dsim/        the simulator's own artifacts
  daic/        the AI controller's artifacts
  dway/        the tour follower's artifacts
  dalg/        the evidence producer's summary, images and archive/
  dnav/        the route planner's summary, events, route image and archive/
  <module>/    any other attached client, named after itself
```

- `<id>` is the shared-memory instance id — the `--id` passed to every process
  in the run. Grouping by it first keeps concurrent instances apart at the top
  level: `--id area1` and `--id area2` running side by side produce two trees
  rather than one interleaved list that can only be untangled by opening files.
- `<run>` is `YYYYmmdd-HHMMSS-xxxxxxxx`: a local timestamp so runs sort and can
  be found by eye, plus eight random hex characters so two runs started in the
  same second cannot land in the same directory.
- Each module gets exactly one subdirectory, named after the module. A module
  writes there and nowhere else.

A run started without an id lands under `reports/default/`. In normal use an id
is always present — `dsim --id` is required — so this is a fallback for
programmatic construction, not a mode of operation.

### Example

```bash
./apps/dsim/dsim.py --id area1 --map ./assets/maps/maze_001.txt &
./apps/daic/daic.py --id area1 --enable-ai &
```

```text
reports/area1/20260902-173839-1e0a0954/
  dsim/
    flight_path.png
    snapshot_HHMMSS.png
    summary.json
  daic/
    flight.jsonl
    route_log.jsonl
    occ_000.png …
    slam_00000.npz …
    frames/
    sector_timeline.png
    summary.json
    report.html
```

---

## 2. Ownership

**`dsim` creates the root and publishes it.** No other module derives a path,
and no two modules have to agree on a name.

At startup `dsim` mints the run name and builds the root:

```python
self.run_id = new_run_id()
self.report_root = report_root(args.id, self.run_id, root=ROOT / "reports")
self.dsim_report_dir = self.report_root / "dsim"
```

It then publishes the root on the status buffer, alongside pose and telemetry:

```text
sim.report_dir = /abs/path/to/reports/area1/20260902-173839-1e0a0954
```

**Every other module reads that key** and appends its own name:

```python
report_dir = self.status.getAll().get("sim.report_dir", "")
if report_dir:
    self.reporter = RunReporter(Path(report_dir) / "daic")
```

This is why a client needs no `--report-dir` of its own, and why start order
does not matter: a client that connects before `dsim` exists simply retries
until the status buffer opens, then picks up the root.

### Two consequences worth knowing

1. **The pickup latches.** A client reads `sim.report_dir` once, when it first
   opens the status buffer. If `dsim` is restarted while the client keeps
   running, the simulator mints a *new* run directory but the client goes on
   writing into the old one. A `apps/dsim/` directory with no sibling is the symptom.
2. **An absent key is not an error.** A client that connects before the
   simulator has published sees an empty string; it must wait rather than
   inventing a path.

---

## 3. Helpers

Both live in `dvision2_common.py` and are the only supported way to build a
report path:

```python
new_run_id() -> str
    "20260902-173839-1e0a0954"

report_root(instance_id, run_id, *, root=None) -> Path
    reports/<id>/<run_id>/
```

`report_root` validates the id (`[A-Za-z0-9_.-]+`) and raises on a malformed
one rather than falling back silently — a mistyped `--id` has already put the
process on the wrong shared-memory namespace, and hiding it in the report path
would make that harder to see, not easier. An id of `None` or `""` yields
`DEFAULT_REPORT_ID` (`"default"`).

### Overriding the location

`dsim --report-dir <path>` replaces the whole scheme and writes directly to the
named directory, taking the run name from that directory's own name. This is
how the test harnesses pin a run to a known place:

```python
cmd = [..., "--report-dir", str(self.report_dir)]
```

The override is deliberate and total: a caller that named a path wants its
artifacts there, not somewhere derived from it. Everything downstream still
works, because `sim.report_dir` publishes whatever root was chosen.

---

## 4. What a module writes

### Required

**`summary.json`** — the end-of-run record, one JSON object, written when the
module shuts down. It is the file a later comparison tool will read, so it
holds numbers and outcomes rather than prose.

`dsim` writes:

```json
{"duration_s": 42.5, "crashed": false, "mode": "GUIDED",
 "status_message": "ok", "x_m": 12.5, "y_m": 8.25, "z_m": 1.5,
 "speed_mps": 0.0, "crash_position": null}
```

`daic` writes its own, describing what its controller did:

```json
{"duration_s": 42.5, "final_state": "COMPLETE", "crashed": false,
 "target_dist_final_m": 0.8, "total_ticks": 1275, "route_changes": 14,
 "straight_path_ticks": 900, "detour_path_ticks": 210,
 "wall_detect_ticks": 88, "avoidance_ticks": 31, "occ_peak_cells": 4200,
 "flow_conf_mean": 0.62, "occ_snapshots": 8, "slam_snapshots": 4,
 "frame_captures": 12}
```

`dway` writes a versioned one, because a later comparison between a simulated
flight and a real one has to read the same fields:

```json
{"schema_version": 1, "tour_id": "maze_012.forward.v1", "outcome": "complete",
 "reason": "landed", "started_at": "2026-09-02T18:40:11+02:00",
 "duration_s": 24.8, "strategy": "position", "coordinate_frame": "map",
 "waypoint_count": 2, "waypoints_reached": 2, "waypoints": [],
 "path_length_m": 10.6, "max_cross_track_error_m": 0.09, "failsafes": [],
 "partial": false}
```

Each `waypoints` entry carries its index, target, first-target time, arrival
time or `null`, dwell, overshoot and maximum cross-track error. A `conditions`
block records the environment the vehicle published at preflight -- fix
quality, estimator validity, wind, telemetry latency, sensor-noise profile,
battery and geofence -- so a run flown in wind or on a degraded fix is never
read as a clean one. Additive fields are allowed; changing what a field means
increments `schema_version`.

`dnav` writes a versioned one too, because a plan is only meaningful against
the evidence, the pose, the goal and the policy it was made from:

```json
{"schema_version": 2, "module": "dnav", "session_id": "…", "instance": "area1",
 "partial": false, "pose_provider": "ideal-simulation", "planner": "astar",
 "plans": 70, "attempts": 94, "health": "warn",
 "policy": {"name": "default", "inflation_m": 0.6, "occupied_threshold": 0.5,
            "combine": "max", "digest": "…"},
 "goal": {"position": [15.5, 10.0], "frame_id": "local", "localization_epoch": 0,
          "revision": 3, "authority_epoch": 1, "role": "ui"},
 "route": {"status": "no_route", "reason": "…", "waypoints": [], "cost": null,
           "diagnostics": {"coverage_exhausted": false}},
 "control_route": {"status": "ok", "cost": null, "length_m": 16.38},
 "control_note": "straight-line diagnostic on the same evidence-derived cost; not an oracle",
 "statuses": [], "status_counts": {"ok": 1, "no_route": 1},
 "reference_display": null, "reference_image": null,
 "generation_transitions": 0, "map": {}, "recording": {"complete": true},
 "archive": "archive"}
```

`statuses` is every status the run passed through with the data-clock time it
changed, because a report holding only the last route says nothing about a run
that spent most of it stale. `reference_display` records the operator's display
decision (`reference-on`/`reference-off`, with its source or opacity) and
`reference_image` the exact revision the final picture displayed; both are
`null` in a headless or unassisted run, which is what keeps an assisted
session visible rather than quietly comparable to an unassisted one.
`route.png` is the map pane's own rendering of
the final route over evidence-derived cost -- composed with the reference
background the operator displayed, when one was displayed, whose exact
revision the archive also holds; never a world file.
`dalg/summary.json` (schema 3) is truth-free: state, resolved geometry
and how it was chosen, omitted sources, counters, recording status and one
evidence image per source. Neither carries a score. Beyond these there is no shared schema, and deliberately
so: each module reports what it measured. What *is* fixed is the file name and the location, which is what lets
a tool find every module's numbers for a run without knowing what they mean.

### Conventional, where they apply

| File | Purpose |
|---|---|
| `flight.jsonl` | One JSON object per line, per tick or per event, flushed as it is written. Survives a crash, unlike anything buffered until exit. |
| `<name>_NNN.png` | Periodic image snapshots, zero-padded so they sort. |
| `frames/` | Individual captured frames, kept in their own subdirectory so the top level stays readable. |
| `report.html` | Optional human-readable rollup, generated at close from the files above. Never the authority — anything it shows must exist in `summary.json` or a log first. |

---

### Numeric archives (dalg, dnav)

Screenshots and summaries lose the numbers. dalg and dnav therefore also keep
an `archive/` (`apps/dcmn/archive.py`, schema `dvision2.archive.v2`) of what
they actually published or consumed:

```text
archive/
  archive.json   manifest: module, provenance metadata, state (recording |
                 finalized | unfinished), counters, drops and errors, complete
  chunks/        chunk-NNNNNN.npz: numeric arrays only, loaded with allow_pickle=False
  chunks.jsonl   one line per committed chunk: file, sha256, bytes, contents
  index.jsonl    one event per line, in recording-sequence order
```

- **dalg** records every evidence publication (including unchanged content,
  which is stored once and referenced by content digest), every admitted and
  rejected sample with its capture-associated pose, mapping initialisations
  with geometry, sizing basis, sources, model digests and allocation estimate,
  every reset with its cause, and mission/lifecycle bus events.
- **dnav** records every planning attempt, successful or not: trigger, exact
  pose, goal descriptor and authority, policy and planner, route, the
  straight-line diagnostic, and the exact evidence revisions used (with their
  arrival order and the stale set), with the grids themselves.
- A run that displayed a reference image also records a **`report.background`**
  event -- image id, revision, checksum, opacity, label and source category --
  with the exact PNG bytes stored in its chunk as a uint8 blob and verified
  against both its declared checksum and its content digest. It is filed after
  the report's picture is rendered and before the recorder seals, so the
  archived revision is the one in the picture by construction. Imagery is
  optional in the archive too: a missing or corrupt blob is a warning, never an
  incomplete event -- it costs the background, never the numbers. The newest
  `report.background` event wins, even when its bytes did not survive, because
  an older revision's picture would misregister.

Chunks commit every second or 32 MiB, whichever comes first; a chunk file is
fsynced and renamed before its `chunks.jsonl` line, and an index line is written
only after the payload it references. At most 64 MiB of copied payload waits in
memory (`--recording-queue-mib`); beyond that a record is dropped without
blocking the module and an `archive.gap` marker names the missing sequences. At
4 GiB per module per session (`--recording-disk-mib`) payloads stop, keeping a
1 MiB reserve for the final manifest. Clean shutdown drains within 5 s. An
archive with any drop, error or unclean end is never labelled complete.

`python3 apps/dcmn/archive.py DIR` validates an archive -- checksums, content
digests, gaps, truncation -- and recovers committed chunks from an interrupted
one. `--attempt N` resolves a dnav planning attempt to its grids, pose, goal
and policy; `dnav.plan.replay_attempt` plans it again, and the route must match.
`--render N` (with `--out PATH`) draws one attempt to a PNG from the archive
alone -- grids, route, pose and goal from the numbers, optionally over the
exact reference revision the report displayed, all through the same renderer
the live pane used, and with no dsim and no world file.
Output archives support evaluating what happened; they are not a raw sensor
recording and cannot rerun a perception algorithm.

A module that finds an `archive/` already present (a restart within one
session) writes `archive-2/` and so on; archives are append-once.

### Evaluator-only truth (dsim)

`dsim/truth/` holds `world.txt` (a copy of the world file), `world.json` (its
digest and frame) and `trajectory.jsonl` (the true pose and collision state
every tick). They exist for a later offline evaluator. No operational module
reads them; the dalg/dnav process tests forbid it with an audit hook.

---

## 5. Rules

**Write only inside your own subdirectory.** A module that writes into another
module's directory, or into the root, breaks the one property this layout
provides: that you can tell who produced a file by where it is.

**Flush logs as you write them.** A run that ends in a crash is the run whose
log you most want. `flight.jsonl` flushes per record for exactly this reason.

**Do image and figure work off the control thread.** `daic` saves every
snapshot in a daemon thread so reporting cannot stall the loop it is reporting
on. Writing a PNG in the middle of a control tick changes the thing being
measured.

**Never let reporting break the run.** Every write is wrapped, and a failure
prints to stderr and continues:

```python
try:
    self._save_summary(...)
except Exception as exc:
    print(f"daic reporter: summary: {exc}", file=sys.stderr)
```

A missing report is a nuisance; a crashed flight because a disk filled up is a
lost experiment.

**Do not put run identity in the file names.** The directory already carries
the id and the run. `summary.json`, not `area1-20260902-summary.json`.

**Reports are outputs, not state.** `reports/` is in `.gitignore`. Nothing in
the tree is an input to anything, and deleting it costs history, never
correctness.

---

## 6. Adding a new module

1. Take `--id` and open the shared buffers by `shared_names(id)`, retrying
   until they exist.
2. Read the session context (`dcmn.context.Context(id).read()`) and take its
   `report_root` and `session_id`. If there is no context yet, keep waiting; do
   not construct a path. (Status-plane clients such as `daic` and `dway` still
   read `sim.report_dir`, which dsim publishes with the same value.)
3. Create `<report_root>/<yourname>/` and write only there. When `session_id`
   changes (a rollover), start a new archive under the new root and write a
   full baseline of any state you retain.
4. Write `summary.json` on shutdown. Append to `flight.jsonl` as you go if the
   run has per-tick state worth keeping.
5. Wrap every write; never raise out of reporting code.

A module that needs to create a root of its own — because it is driving a
vehicle that is not `dsim` — calls `report_root()` and `new_run_id()` and then
publishes the result the same way, so its clients need no special case.

---

## 7. Reference

| Thing | Where |
|---|---|
| `new_run_id`, `report_root`, `DEFAULT_REPORT_ID` | `dvision2_common.py` |
| Root creation, session context and `sim.report_dir` publication | `apps/dsim/dsim.py` |
| The neutral session context (session, frame, epochs, pose, goal, resets) | `apps/dcmn/context.py` |
| A provider without dsim (tests, replay) | `dtest/provider.py` |
| Numeric archives: writer, reader, validator | `apps/dcmn/archive.py` |
| Optional reference imagery: publisher, session, display host | `apps/dcmn/imagery.py`, `apps/dcmn/map_pane.py` |
| Reference-image archiving and the standalone render | `apps/dcmn/archive.py` |
| A worked reporter (images, logs, summary, HTML) | `apps/daic/run_reporter.py` |
| A worked JSONL logger | `apps/daic/flight_log.py` |
| A versioned summary and event log | `apps/dway/report.py` |
| Pinning a run to a fixed directory | `dtest/process_harness.py` |

## Dynamic dway segments

Dynamic dway (dry run and flight) records under `dway/archive`, `archive-2`, etc., with
an append-once segment per restart/rollover; it never overwrites static-tour flight
logs. Dry-run segments contain exact consumed navigation records, context and evidence
references, plus a segment summary, recent event view, HTML and map PNG. Flight
segments add controls, transitions, holds, lease/health events, targets with the
permission they were checked against, 10 Hz vehicle samples and the shutdown result;
their report (`summary.json`, CSV tables, `map.png`, `timeline.png`, `report.html`,
`manifest.json`) is written under `<archive>/report/` by `apps/dway/flightlog.py`,
which may read sibling `dnav/` archives but writes only under dway. dnav adds
`navigation.snapshot` events with exact validation inputs and persistent veto state,
and writes a dark `dnav/report.html` built from its summary: route outcome, route image,
execution permission beside the strict evidence check, status history and evidence sources.
See [navigation reporting](navigation.md#reports-replay-and-comparison) for metrics,
replay, comparison and sharing a run.
