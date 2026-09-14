# Camera and lidar evidence

Composing `ground-plane-baseline` with `lidar-baseline` runs `ground_plane` on
the drone's declared primary camera and `lidar_inverse` on the `scan` lidar.
Both publish through one dalg maps manifest, as separate source grids of one
generation: same runtime geometry, frame, localization/clock epoch, mapping
epoch and geometry revision. dnav derives each cost layer independently, then
takes the maximum.

Launch these in separate terminals from the repository root:

```sh
python3 apps/dsim/dsim.py --id multi-demo --map assets/maps/maze_020.txt \
  --drone-profile assets/drone_profiles/camera-lidar.json
```

```sh
python3 apps/dalg/dalg.py --id multi-demo --profiles ground-plane-baseline lidar-baseline
```

```sh
python3 apps/dnav/dnav.py --id multi-demo --goal 52.444,2.389
```

```sh
python3 apps/dway/dway.py --id multi-demo \
  --tour assets/tours/maze_020.default.v1.json \
  --wait-for algorithm:lidar_inverse --finish-action hold
```

The tour supplies motion; dalg neither reads nor checks it. dalg sizes its
coverage from the first valid pose and the goal dnav submitted (if it was
already set), so starting dnav with its goal before dalg gives a square that
contains the route; otherwise pass `--bounds` explicitly, or request a mapping
reset with `python3 apps/dcmn/context.py --id multi-demo reset --bounds ...`.

In dalg, **Grids** shows both sources side by side, each with occupancy, age,
and never-observed modes, hover readout, publication rate and sensor health.
Lidar observes around the vehicle, including cells behind the forward camera.
The **Live** selector switches between the camera image and the shared polar
lidar viewer, with that source's own prediction beside it. There is no truth
pane: the world belongs to the evaluator.

In dnav, **Cost** offers the two evidence layers, their two derived cost layers,
and the combined stack. A cell is unknown in the combined display only when
neither sensor observed it. Inflated obstacles remain visible even where the
other sensor has no observations. dnav waits for every source in the current
generation to publish its first record before planning, never mixes grids from
two generations, and does not control the vehicle.

## Build a profile

Open dalg's **Profile** tab, or edit offline:

```sh
python3 apps/dalg/dalg.py --edit --profile lidar-baseline
```

Select a discovered sensor on the left (or type `primary_camera` to bind the
declared primary camera), choose a compatible algorithm, and click **Add
source**. Select a source row to edit its generated settings; only values that
differ from the algorithm's defaults are saved. Remove and move rows with the
adjacent controls. Missing sensors, incompatible algorithms, duplicate sources
(including a selector and the camera it resolves to) and invalid settings
appear on their rows and disable saving.

A profile holds a name and sources, nothing else. The **Runtime mapping** box
shows the geometry a running dalg resolved, read-only; extent, resolution and
slab are `--bounds`, `--cell-m`, `--z0-m` and `--dz-m` on dalg's command line,
not profile fields. Saved changes take effect on the next launch.

Offline sensor IDs are marked unchecked. When the profile runs, dalg resolves
the selector against the provider's manifest and validates every source before
it publishes.

## Inverse model and reports

Lidar uses the scan's angular calibration, confidence and full capture-time
sensor transform, including mounting offsets and rotation. Valid rays clear
crossed cells up to their return and mark the endpoint occupied. Endpoints win
when multiple rays touch a cell in the same scan. NaN, zero confidence and
out-of-range returns provide no evidence: a missing return could be dropout,
so it cannot safely clear a maximum-range ray. Cells between sparse rays stay
unknown; the model does not invent observations by filling the entire sector.

The shared slab is `[0, 3)` metres above ground by default. This contains the
demo's 1.5 m flight altitude and lidar's 0.05 m mounting offset. Rays are
clipped to the slab and the resolved extent; returns outside it never become
occupied cells inside it. A single-height scan remains an approximation of free
space across this flight band. Sources retain their own capture timestamps and
beliefs; no sensor fusion or persistent mapping is performed.

Every sample is admitted only with a valid capture-associated pose in the
current frame and epochs; a sample whose pose is invalid, from another epoch or
from the future is rejected and recorded. Reports contain one
`evidence-<source>.png` per source and a truth-free `summary.json`; the numbers
themselves -- every published grid and the samples behind it -- are in
`dalg/archive/`.

Source validation, inverse-model geometry, profile round-trip, both UI panes
and layered-cost checks:

```sh
xvfb-run -a pytest -q tests/test_dalg_sources.py tests/test_dalg_runtime.py
```

## Composing algorithm profiles

Algorithm profiles live in `assets/algorithm_profiles/`: exactly one source-only
baseline per evidence algorithm. `--profile NAME` selects one file;
`--profiles NAME [NAME ...]` combines the sources of several into one
publisher. Names with or without `.json` search the profile directory;
explicit relative paths resolve from the repository root.

Composition concatenates the `sources` lists. Duplicate sensor/algorithm pairs
-- before or after the `primary_camera` selector is resolved -- are errors.
Reports and archives record the component names, paths and digests. `--edit`
accepts only one profile. Old profile files that carry `tour`, `map`,
`algorithm`, `sensors` or `settings` at the top level are refused with the
field named; there is no legacy loader and no alias for a retired name.

A source whose sensor is absent is omitted, with the reason in the UI, the
heartbeat, the summary and the archive. Only available sources appear in the
evidence manifest, so dnav can plan on their layers. With no configured sensor
available dalg stays up in `WAITING_SENSORS` and keeps looking. A sensor with
the wrong type is a configuration error, not an optional missing device.

Sample gaps recover without a rebuild. A sensor manifest change retires the
current generation and rebuilds the sources that were active and are still
present; a removed sensor disappears from the manifest, so dnav never waits on
it. A newly added sensor is reported as available and joins after an explicit
mapping reset (`apps/dcmn/context.py ... reset`) or a dalg restart.
