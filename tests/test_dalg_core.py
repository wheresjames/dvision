from __future__ import annotations

import math
import time
import numpy as np
import pytest
from pathlib import Path

from dalg.grid import LogOddsGrid, OccupancyGrid
from dalg.profiles import load_profile, profile_dir
from dalg.algo import ALGORITHMS
from dalg.algo.plane_sweep import PlaneSweepAlgorithm
from dalg.algo.features import FeatureTriangulationAlgorithm
from dalg.algo.ground_plane import GroundPlaneAlgorithm
from dalg.algo.monocular_depth import MonocularDepthAlgorithm
from dalg.algo.optical_flow import OpticalFlowTriangulationAlgorithm
from dalg.algo.spatial import triangulate_xy
from dalg.model import Frame, Intrinsics, Pose
from dcmn.module_bus import ModuleEvent, PipelineView
from dalg.run import DalgRun, matches_prepare
from dtest.evaluation import (FALSE_NEGATIVE, FALSE_POSITIVE, TRUE_POSITIVE,
                              observable_mask, score_occupancy, verdict_raster)


def test_the_registry_holds_evidence_algorithms_and_no_controls():
    """The oracle and constant controls are evaluation tooling, never sources."""
    assert set(ALGORITHMS) == {"sgbm", "plane_sweep", "feature_triangulation",
                               "optical_flow_triangulation", "ground_plane", "monocular_depth"}
    assert "exact_range" not in ALGORITHMS and "constant" not in ALGORITHMS


def test_readiness_selectors_match_profile_names_and_algorithms():
    root = Path(__file__).resolve().parents[1]
    profile = load_profile("sgbm-baseline", root)
    assert matches_prepare(profile, ["algorithm:sgbm-baseline"])
    assert matches_prepare(profile, ["algorithm:sgbm"])
    assert not matches_prepare(profile, ["algorithm:plane_sweep"])
    assert matches_prepare(profile, [])


def test_pipeline_members_expire_and_goodbye_removes():
    view = PipelineView(expiry_s=2)
    event = ModuleEvent("e", "demo", "algorithm", "dalg", "p", 1, 0,
                        "module.heartbeat", "run", {"state": "READY", "ready": True})
    view.observe(event, now=10)
    assert view.members(now=11)[0][0].ready
    assert not view.members(now=13)
    view.observe(ModuleEvent("bye", "demo", "algorithm", "dalg", "p", 2, 0,
                             "module.goodbye", "run", {}), now=13)
    assert not view.members(now=13, include_expired=True)


def test_scoring_and_verdict_colours():
    truth = OccupancyGrid(np.array([[.9, .1], [.1, .9]], np.float32),
                          np.ones((2, 2), bool), 1.0)
    predicted = OccupancyGrid(np.array([[.9, .9], [.1, .1]], np.float32),
                              np.ones((2, 2), bool), 1.0)
    score = score_occupancy(predicted, truth)
    assert score["occupied_iou"] == 1 / 3
    raster = verdict_raster(truth, predicted)
    assert tuple(raster[0, 0]) == TRUE_POSITIVE
    assert tuple(raster[0, 1]) == FALSE_POSITIVE
    assert tuple(raster[1, 1]) == FALSE_NEGATIVE


def test_plane_sweep_downsamples_and_keeps_spaced_keyframes():
    algorithm = PlaneSweepAlgorithm(10, 10, Intrinsics(64, 48, 45, 45, 32, 24))
    rgb = np.zeros((48, 64, 3), np.uint8)
    algorithm.observe(Frame(1, 0, rgb, Pose(5, 5, 1.5, 0)))
    algorithm.observe(Frame(2, .1, rgb, Pose(5, 4.8, 1.5, 0)))
    algorithm.observe(Frame(3, .2, rgb, Pose(5, 4.4, 1.5, 0)))
    assert len(algorithm.frames) == 2
    assert algorithm.frames[0][0].shape == (24, 32)


def test_plane_sweep_marks_ray_free_and_endpoint_occupied():
    algorithm = PlaneSweepAlgorithm(10, 10, Intrinsics(64, 48, 45, 45, 32, 24))
    depth = np.full((24, 32), np.nan, np.float32)
    depth[0, 16] = 2.0
    algorithm._fuse(depth, Pose(5, 5, 1.5, 0))
    # Heading north: the sensor is cell (20,20), endpoint is around (20,12).
    assert algorithm.grid.log_odds[16, 20] < 0
    assert algorithm.grid.log_odds[12, 20] > 0


def test_known_pose_bearings_triangulate_a_known_point():
    intrinsics = Intrinsics(64, 48, 45, 45, 32, 24)
    point = triangulate_xy(Pose(5, 5, 1.5, 0), 47,
                           Pose(4, 5, 1.5, 0), 62, intrinsics)
    assert point == pytest.approx((6, 2))


def test_additional_monocular_algorithms_accept_frames():
    intrinsics = Intrinsics(128, 96, 90, 90, 64, 48)
    rng = np.random.default_rng(4)
    rgb = rng.integers(0, 256, (96, 128, 3), dtype=np.uint8)
    shifted = np.roll(rgb, 3, axis=1)
    first = Frame(1, 0, rgb, Pose(5, 5, 1.5, 0))
    second = Frame(2, 1, shifted, Pose(4.5, 5, 1.5, 0))
    for algorithm_type in (FeatureTriangulationAlgorithm,
                           OpticalFlowTriangulationAlgorithm,
                           GroundPlaneAlgorithm):
        algorithm = algorithm_type(10, 10, intrinsics)
        algorithm.observe(first); algorithm.observe(second)
        result = algorithm.finish()
        assert result.grid.probabilities.shape == (40, 40)


def test_monocular_depth_requires_an_explicit_model_file():
    with pytest.raises(ValueError, match="ONNX metric-depth model"):
        MonocularDepthAlgorithm(10, 10,
                                Intrinsics(64, 48, 45, 45, 32, 24))


def test_every_camera_baseline_constructs_through_the_run_call_path():
    """A profile that only loads is not a profile that runs."""
    from dalg.profiles import camera_evidence_algorithms
    root = Path(__file__).resolve().parents[1]
    intrinsics = Intrinsics(640, 480, 554.3, 554.3, 320.0, 240.0)
    for path in sorted(profile_dir(root).glob("*.json")):
        source = load_profile(path.stem, root).sources[0]
        if source.algorithm not in camera_evidence_algorithms(): continue
        if source.algorithm == "monocular_depth":
            continue  # needs the installed ONNX model; covered by preflight tests
        ALGORITHMS[source.algorithm](40, 30, intrinsics, settings=source.settings, evidence=True)


def test_settings_an_algorithm_does_not_declare_are_refused():
    from dalg.profiles import Source, validate_sources
    with pytest.raises(ValueError, match="source 1"):
        validate_sources([Source("front", "sgbm", {"not_a_real_setting": 1})])
    with pytest.raises(ValueError, match="source 1"):
        validate_sources([Source("front", "sgbm", {"range_stride": 8})])


def test_cells_floor_so_points_outside_the_map_are_rejected():
    """int() truncates toward zero, so a point in the half-cell west of the
    map used to land on column 0 and be fused as real occupancy."""
    grid = LogOddsGrid(4, 4, .25)
    xs, ys = grid.cells([-0.1, -0.3, 0.1], [1.0, 1.0, 1.0])
    assert xs.tolist() == [-1, -2, 0]            # truncation would give 0, 0, 0
    grid.update(xs, ys, 4.0)
    row = grid.log_odds[4]
    assert row[0] == pytest.approx(4.0)          # only the point inside the map
    assert np.count_nonzero(row) == 1
    assert row[-1] == 0.0                        # nothing wrapped to the far edge


def test_accumulate_reinforces_while_update_applies_once():
    grid = LogOddsGrid(4, 4, .25)
    grid.update([2, 2, 2], [3, 3, 3], 1.0)
    assert grid.log_odds[3, 2] == pytest.approx(1.0)
    grid.accumulate([2, 2, 2], [3, 3, 3], 1.0)
    assert grid.log_odds[3, 2] == pytest.approx(4.0)


def test_range_projection_inverts_the_simulator_ray_cast():
    """The sensor reports slant range along a yaw/pitch ray. Treating it as a
    horizontal distance over-projected the bottom of the frame."""
    from dalg.algo.spatial import project_ranges
    intrinsics = Intrinsics(320, 240, 277.0, 277.0, 160.0, 120.0)
    camera = Pose(5.0, 6.0, 1.5, 37.0, 0.0, -5.0)
    row, column, slant = 220.0, 300.0, 7.0
    pitch = math.radians(camera.pitch_deg)-math.atan((row-intrinsics.cy_px)/intrinsics.fy_px)
    yaw = math.radians(camera.heading_deg)+math.atan((column-intrinsics.cx_px)/intrinsics.fx_px)
    expected = (camera.x_m+math.sin(yaw)*math.cos(pitch)*slant,
                camera.y_m-math.cos(yaw)*math.cos(pitch)*slant,
                camera.z_m+math.sin(pitch)*slant)
    assert project_ranges(camera, column, row, slant, intrinsics) == pytest.approx(expected)


def test_stereo_projection_uses_perpendicular_depth_not_radial_range():
    """fx*baseline/disparity is depth along the optical axis. Using it as a
    distance along the viewing ray pulled the frame edges toward the camera."""
    from dalg.algo.spatial import project_pixels
    intrinsics = Intrinsics(640, 480, 554.3, 554.3, 320.0, 240.0)
    camera = Pose(0.0, 0.0, 1.5, 0.0)
    x, y, z = project_pixels(camera, 630.0, 240.0, 6.0, intrinsics)
    assert float(y) == pytest.approx(-6.0)               # forward stays at Z
    assert float(z) == pytest.approx(1.5)                # centre row, no rise
    radial = math.hypot(float(x), float(y))
    assert radial > 6.0                                  # the ray is longer than Z
    assert radial == pytest.approx(6.0*math.hypot(1, (630-320)/554.3))


def test_visibility_mask_stops_at_the_first_wall():
    truth = OccupancyGrid(np.full((5, 5), .05, np.float32), np.ones((5, 5), bool), 1.0)
    truth.probabilities[2, 2] = .95
    mask = observable_mask(truth, [Pose(2.5, 4.5, 1.0, 0.0)],
                           fov_h_deg=20, max_range_m=10, rays=9)
    assert mask[2, 2] and mask[3, 2] and mask[4, 2]      # up to and including it
    assert not mask[1, 2] and not mask[0, 2]             # nothing behind it


def test_report_lands_in_the_module_directory_with_numbers_and_no_scores(tmp_path):
    """``<report_root>/dalg/``: a summary, one evidence image per source, no truth."""
    import json
    from dalg.report import write_report
    from dcmn.maps import EvidenceGrid, GridGeometry

    geometry = GridGeometry.from_extent(6, 6, 1., origin_x_m=-3, origin_y_m=-3)
    grid = EvidenceGrid.blank(geometry, "scan-lidar_inverse")
    grid.occupancy[0, 2, 2] = 254; grid.observed_ms[0, 2, 2] = 1500
    out = write_report(tmp_path / "dalg", summary=dict(state="RUNNING", archive="archive"),
                       evidence={"scan-lidar_inverse": grid})
    assert out == tmp_path / "dalg"
    summary = json.loads((out / "summary.json").read_text())
    assert summary["archive"] == "archive"
    entry = summary["evidence"]["scan-lidar_inverse"]
    assert (out / entry["image"]).is_file()
    assert entry["geometry"]["origin_m"] == [-3.0, -3.0]
    assert "scores" not in summary and "scores" not in entry


def _stereo_frames(count: int, spacing_m: float = 0.5) -> list[Frame]:
    """A straight sideways slide, so every frame has partners on its right."""
    rgb = np.random.default_rng(11).integers(0, 255, (120, 160, 3), np.uint8)
    frames = []
    for index in range(count):
        pose = Pose(10.0 + index * spacing_m, 10.0, 1.5, 0.0)
        frames.append(Frame(index, index * 0.2, rgb, pose, None, None, pose))
    return frames


def _sgbm_pairings(*, with_previews: bool) -> dict[int, float]:
    intrinsics = Intrinsics(160, 120, 100.0, 100.0, 80.0, 60.0)
    algorithm = ALGORITHMS["sgbm"](30, 30, intrinsics)
    chosen: dict[int, float] = {}
    algorithm._fuse_stereo = (
        lambda index, other, baseline: chosen.__setitem__(index, round(baseline, 3)))
    for frame in _stereo_frames(8):
        algorithm.observe(frame)
        if with_previews:
            algorithm.preview()
    algorithm.finish()
    return chosen


def test_sgbm_pairs_the_widest_baseline_whether_or_not_previews_ran():
    """A preview is a display, not a measurement, and must not change one.

    _process marks a frame fused so a disparity is computed once, but during a
    preview the wider partners have not been captured yet -- so previews used
    to pin every frame to the narrowest baseline that cleared min_baseline_m
    and the reported grid depended on the front end's preview cadence.
    """
    assert _sgbm_pairings(with_previews=True) == _sgbm_pairings(with_previews=False)


def test_sgbm_uses_the_widest_eligible_baseline():
    pairings = _sgbm_pairings(with_previews=True)
    # 8 frames 0.5 m apart, max_baseline_m 4.0: frame 0's widest partner is the
    # last one, 3.5 m away, not its immediate neighbour.
    assert pairings[0] == 3.5


def test_feature_triangulation_survives_a_frame_with_one_descriptor():
    """knnMatch returns min(k, available) matches, so k=2 rows can hold one."""
    intrinsics = Intrinsics(160, 120, 100.0, 100.0, 80.0, 60.0)
    algorithm = FeatureTriangulationAlgorithm(30, 30, intrinsics)
    rng = np.random.default_rng(5)

    class _OneDescriptor:
        """Second frame yields a single descriptor, as a blank wall would."""
        def __init__(self):
            self.calls = 0

        def detectAndCompute(self, gray, mask):
            self.calls += 1
            count = 8 if self.calls == 1 else 1
            keypoints = [type("K", (), {"pt": (float(i), 1.0)})()
                         for i in range(count)]
            return keypoints, rng.integers(0, 256, (count, 32), np.uint8)

    algorithm.orb = _OneDescriptor()
    rgb = rng.integers(0, 255, (120, 160, 3), np.uint8)
    algorithm.observe(Frame(0, 0.0, rgb, Pose(10.0, 10.0, 1.5, 0.0), None, None,
                            Pose(10.0, 10.0, 1.5, 0.0)))
    algorithm.observe(Frame(1, 0.2, rgb, Pose(11.0, 10.0, 1.5, 0.0), None, None,
                            Pose(11.0, 10.0, 1.5, 0.0)))

    assert algorithm.finish().diagnostics["keyframes"] == 2


def test_one_clear_sightline_is_enough_to_call_a_cell_free():
    """The free carve has to cross the threshold it is scored against.

    fuse_endpoint's delta used to be -0.55 while FREE_THRESHOLD sits at
    log-odds -0.619, so a cell carved once rendered dark but scored undecided.
    plane_sweep, which sets its own -0.7, was never affected -- which made the
    comparison between algorithms turn partly on an untuned default.
    """
    import inspect
    from dalg.algo.spatial import fuse_endpoint
    from dalg.grid import FREE_THRESHOLD

    delta = inspect.signature(fuse_endpoint).parameters["free"].default
    assert 1.0 / (1.0 + math.exp(-delta)) <= FREE_THRESHOLD

    grid = LogOddsGrid(10, 10)
    fuse_endpoint(grid, Pose(1.0, 1.0, 1.5, 90.0), (5.0, 1.0))
    result = grid.result()
    # The ray between the camera and the endpoint is decided free, and the
    # endpoint itself decided occupied.
    assert result.free[grid.cells([3.0], [1.0])[1][0],
                       grid.cells([3.0], [1.0])[0][0]]
    assert result.occupied[grid.cells([5.0], [1.0])[1][0],
                           grid.cells([5.0], [1.0])[0][0]]


def test_prediction_image_ramps_black_through_blue_to_white():
    from dalg.overlay import UNDECIDED, prediction_image

    grid = OccupancyGrid.unknown(8, 4)
    grid.probabilities[0, 0] = 1.0     # certain wall
    grid.probabilities[0, 1] = 0.0     # certain free
    pixels = np.array(prediction_image(grid, scale=1))

    assert tuple(pixels[0, 0]) == (255, 255, 255)
    assert tuple(pixels[0, 1]) == (0, 0, 0)
    assert tuple(pixels[0, 2]) == UNDECIDED         # untouched: no opinion


def test_undecided_neutral_stays_clear_of_free_and_occupied():
    """The neutral has to be legible against both ends of the ramp.

    The evaluation verdict overlay's BACKGROUND shares the palette but sits at
    luminance 38, inside the range a prediction grid paints confident free
    space -- so using it directly hides the free space a run actually carved.
    """
    from dalg.overlay import UNDECIDED
    from dtest.evaluation import BACKGROUND

    def luminance(colour):
        return 0.2126 * colour[0] + 0.7152 * colour[1] + 0.0722 * colour[2]

    assert luminance(UNDECIDED) == pytest.approx(127.0, abs=1.5)
    # Same hue as the overlay background, so the two images read as one palette.
    scale = luminance(UNDECIDED) / luminance(BACKGROUND)
    for channel, reference in zip(UNDECIDED, BACKGROUND):
        assert channel == pytest.approx(reference * scale, abs=1.0)


class _FakeBus:
    """A module bus that records what was published and replays a script."""

    def __init__(self) -> None:
        self.process_id = "dalg-test"
        #: Part of the bus contract: modules report it as a keeping-up signal.
        self.overruns = 0
        self.published: list[tuple[str, str, dict]] = []
        self.inbox: list = []
        self.closed = False

    def connect(self) -> bool:
        return True

    def publish(self, event_type, *, run_id="", payload=None):
        self.published.append((event_type, run_id, payload or {}))
        return True

    def receive(self):
        events, self.inbox = self.inbox, []
        return events

    def close(self) -> None:
        self.closed = True

    def types(self) -> list[str]:
        return [event for event, _, _ in self.published]


def _shutdown_event(run_id: str = ""):
    from dcmn.module_bus import SHUTDOWN_EVENT, ModuleEvent
    return ModuleEvent(
        event_id="e1", instance_id="area1", role="simulator",
        implementation="dsim", process_id="dsim-1", sequence=1,
        sim_time_s=42.0, type=SHUTDOWN_EVENT, run_id=run_id,
        payload={"reason": "operator requested shutdown", "scope": "instance"})


def _idle_run():
    """A DalgRun with no provider at all, and a bus that records what it says."""
    root = Path(__file__).resolve().parents[1]
    run = DalgRun("area1-" + str(time.monotonic_ns()), load_profile("optical-flow-baseline", root), root)
    run.bus = _FakeBus()
    return run


def test_shutdown_is_heard_while_waiting_for_a_provider():
    run = _idle_run()
    try:
        run.bus.inbox = [_shutdown_event()]
        run.step()
        assert run.shutdown_requested is True and run.done
    finally:
        run.close()


def test_a_waiting_run_keeps_publishing_presence_and_says_why():
    """Waiting is a state an operator can see, not silence."""
    run = _idle_run()
    try:
        run.step()
        run._last_presence = -1e9
        run.step()
        assert run.bus.types().count("module.heartbeat") == 2
        assert run.bus.types().count("module.sensor_health") == 2
        beats = [payload for kind, _, payload in run.bus.published if kind == "module.heartbeat"]
        assert beats[-1]["state"] == "WAITING_PROVIDER"
        assert beats[-1]["ready"] is False and beats[-1]["capabilities"]["maps"] is False
    finally:
        run.close()


def test_mission_lifecycle_traffic_never_starts_or_stops_perception():
    """A coordinator's run.* events are recorded; they are not dalg's lifecycle."""
    from dcmn.module_bus import ModuleEvent
    run = _idle_run()
    try:
        for kind in ("run.prepare", "run.start_scheduled", "run.completed"):
            run.bus.inbox = [ModuleEvent(event_id=kind, instance_id="area1", role="navigator",
                implementation="dway", process_id="dway-1", sequence=9, sim_time_s=9.0,
                type=kind, run_id="r1", payload={"outcome": "complete"})]
            run.step()
        assert not run.done and run.state == "WAITING_PROVIDER"
        # Not active, so no readiness claim either.
        assert "run.ready" not in run.bus.types()
    finally:
        run.close()


def test_main_keeps_stepping_a_stopped_run_while_its_window_is_open(monkeypatch):
    """The window outlives observation, and so must the bus drain behind it."""
    import dalg.dalg as dalg_module

    class _Run:
        """Stops on the third step, then reports a shutdown three steps later."""
        def __init__(self, *args, **kwargs):
            self.steps = 0
            self.steps_after_done = 0
            self.done = False
            self.shutdown_requested = False
            self.report_dir = None
            self.state = "RUNNING"
            self.closed = False

        def step(self):
            self.steps += 1
            if self.done:
                self.steps_after_done += 1
            if self.steps == 3:
                self.done = True
            if self.steps_after_done == 3:
                self.shutdown_requested = True

        def poll_delay(self): return 0.
        def finish(self, partial=False): self.done = True
        def close(self):
            self.closed = True

    class _Window:
        BUDGET = 200

        def __init__(self, run, *, show_reference=False):
            self.run = run
            self.running = True
            self.updates = 0
            self.root = type("R", (), {"destroy": lambda self: None})()

        def update(self):
            self.updates += 1
            if self.updates > self.BUDGET:
                self.running = False

        def close(self): self.running = False
        def save_geometry(self): pass

    created = {}
    monkeypatch.setattr(dalg_module, "DalgRun",
                        lambda *a, **k: created.setdefault("run", _Run()))
    monkeypatch.setattr(dalg_module, "Window", _Window)
    monkeypatch.setattr(dalg_module.time, "sleep", lambda _s: None)

    code = dalg_module.main(["--id", "area1", "--profile", "optical-flow-baseline"])
    run = created["run"]

    assert run.steps_after_done == 3, "the bus stopped being drained once the run stopped"
    assert run.shutdown_requested is True
    assert run.closed is True
    assert code == 0
