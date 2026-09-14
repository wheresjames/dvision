# Live camera evidence and planning

Every camera algorithm in dalg publishes evidence: `ground_plane`,
`optical_flow_triangulation`, `feature_triangulation`, `monocular_depth`,
`plane_sweep` and `sgbm`, each with a source-only baseline profile. The list
lives in one place, `CAMERA_EVIDENCE` in `apps/dalg/profiles.py`. The
`constant` and `exact_range` controls are not dalg algorithms any more; they
live with the other oracle checks in `dtest/evaluation.py`.

The evidence path never uses a world file, truth, or simulated range ray-cast
from the map. It sees camera frames, their capture-associated poses (labelled
*ideal* when they come from dsim), the frame and epochs in the session context,
and the runtime geometry dalg resolved. Every grid is a belief, not a collision
guarantee.

Each algorithm is its own instance on that geometry and only ever uses frames
already captured, so pairing algorithms (`sgbm`, `plane_sweep`) are causal and
their temporal state is discarded on any reset. `monocular_depth` and `sgbm`
mark free space along every depth ray. Clearing along a ray proves the column
free at the ray's height, not all the way down, so a low obstacle close to the
camera can sit under a ray to the floor -- the same approximation
`ground_plane` makes.

Two algorithms need knowing before you pick them. `sgbm` is stereo from
motion: it pairs frames only when the camera has moved *sideways*, so on a tour
that flies forward it finds no pairs and publishes nothing. Fly it along a wall,
heading held, with `max_baseline_m` small enough that the disparity fits its
search range. `monocular_depth` needs the ONNX model installed by
`scripts/install_dalg_depth_model.py`; dalg refuses to launch without it.

Run these commands in separate terminals from the repository root:

```sh
python3 apps/dsim/dsim.py --id camera-demo --map assets/maps/maze_020.txt
```

```sh
python3 apps/dnav/dnav.py --id camera-demo --goal 52.444,2.389
```

```sh
python3 apps/dalg/dalg.py --id camera-demo --profile ground-plane-baseline
```

```sh
python3 apps/dway/dway.py --id camera-demo \
  --tour assets/tours/maze_020.default.v1.json \
  --wait-for algorithm:ground-plane-baseline --finish-action hold
```

Any other baseline works the same way: `optical-flow-baseline`,
`features-baseline`, `monocular-depth-baseline`, `plane-sweep-baseline` or
`sgbm-baseline`. Starting dnav first means its goal is already in the context
when dalg sizes coverage; otherwise pass `--bounds` to dalg. Camera admission
defaults to 5 frames per data-clock second per source (`--camera-hz`), taking
the latest matched frame and counting the ones it skipped.

In dalg, select **Grids** and switch between occupancy, age, and never-observed.
Hover for cell coordinates, probability, and observation age. The header names
the camera source and shows revision, measured publication rate, intake health,
and last publication time. **Live** shows the camera and the algorithm's own
prediction; the status line shows the resolved coverage and why dalg is
waiting, if it is. **Events** shows the shared module bus. The **Profile** tab
edits sources; see the [camera and lidar demo](multisensor-evidence.md).

In dnav, move the goal by clicking the Plan pane. Routes update as evidence
changes. A camera boundary may block the start or goal; those failures remain
visible as route statuses, as does a goal outside the evidence coverage.
dnav holds no vehicle control lease.

Evidence publishes every data-clock second, including unchanged grids. Only ray
updates refresh observation timestamps; publication alone never makes old cells
look newly observed. When the tour ends, dalg keeps observing -- a mission
ending is only a recorded event. Closing dalg removes its maps plane; dnav marks
the map stale after its freshness horizon.

The session's report directory receives `dalg/summary.json` (truth-free),
one `evidence-<sensor>-<algorithm>.png` per source, and `dalg/archive/` with
every published grid, the samples admitted and the capture poses they were
admitted at. Scoring that against the world is an offline step: the tour-flight
test does it with `dtest/evaluation.py`, from the archive, after dalg has
exited.

For a short automated demonstration using a real simulator and dway corridor
tour, including dnav planning and the offline evaluation:

```sh
pytest -q tests/test_dalg_evidence.py -k tour_flown
```

For widget and snapshot parity checks on a virtual display:

```sh
xvfb-run -a pytest -q tests/test_dalg_evidence.py
```
