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
./dsim/dsim.py --id area1 --map ./assets/maps/maze_001.txt &
./daic/daic.py --id area1 --enable-ai &
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
   writing into the old one. A `dsim/` directory with no sibling is the symptom.
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

Beyond that one there is no shared schema, and deliberately so: each module
reports what it measured. What *is* fixed is the file name and the location, which is what lets
a tool find every module's numbers for a run without knowing what they mean.

### Conventional, where they apply

| File | Purpose |
|---|---|
| `flight.jsonl` | One JSON object per line, per tick or per event, flushed as it is written. Survives a crash, unlike anything buffered until exit. |
| `<name>_NNN.png` | Periodic image snapshots, zero-padded so they sort. |
| `frames/` | Individual captured frames, kept in their own subdirectory so the top level stays readable. |
| `report.html` | Optional human-readable rollup, generated at close from the files above. Never the authority — anything it shows must exist in `summary.json` or a log first. |

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
2. On first successful status read, take `sim.report_dir`. If it is empty, keep
   waiting; do not construct a path.
3. Create `<sim.report_dir>/<yourname>/` and write only there.
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
| Root creation and `sim.report_dir` publication | `dsim/dsim.py` |
| A worked reporter (images, logs, summary, HTML) | `daic/run_reporter.py` |
| A worked JSONL logger | `daic/flight_log.py` |
| A versioned summary and event log | `dway/report.py` |
| Pinning a run to a fixed directory | `dtest/process_harness.py` |
