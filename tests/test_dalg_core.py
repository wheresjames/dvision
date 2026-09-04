from __future__ import annotations

import math
import time
import numpy as np
import pytest
from PIL import Image
from pathlib import Path

import dalg.run as run_module
from dalg.algo.controls import ExactRangeAlgorithm
from dalg.grid import LogOddsGrid, OccupancyGrid
from dalg.overlay import FALSE_NEGATIVE, FALSE_POSITIVE, TRUE_POSITIVE, verdict_raster
from dalg.profiles import load_profile, profile_dir
from dvision2_common import load_map, shared_names
from dalg.algo import ALGORITHMS
from dalg.algo.plane_sweep import PlaneSweepAlgorithm
from dalg.algo.features import FeatureTriangulationAlgorithm
from dalg.algo.ground_plane import GroundPlaneAlgorithm
from dalg.algo.monocular_depth import MonocularDepthAlgorithm
from dalg.algo.optical_flow import OpticalFlowTriangulationAlgorithm
from dalg.algo.spatial import triangulate_xy
from dalg.model import Frame, Intrinsics, Pose
from dcmn.module_bus import ModuleEvent, PipelineView
from dalg.run import (DalgRun, algorithm_settings, copy_video_frame,
                      matches_prepare)
from dalg.visibility import observable_mask
from dalg.score import score_occupancy


def test_default_profile_loads():
    root = Path(__file__).resolve().parents[1]
    profile = load_profile("sgbm-default", root)
    assert profile.algorithm == "sgbm"
    assert profile.tour.exists()
    assert "plane_sweep" in ALGORITHMS
    assert {"feature_triangulation", "optical_flow_triangulation",
            "ground_plane", "monocular_depth"} <= set(ALGORITHMS)


def test_range_profile_preserves_configuration():
    root = Path(__file__).resolve().parents[1]
    profile = load_profile("sgbm-flash", root)
    assert profile.sensors == ("rgb", "range")
    assert profile.sensor_config["range"] == "lidar_flash_short"
    assert matches_prepare(profile, ["algorithm:sgbm-flash"])
    assert matches_prepare(profile, ["algorithm:sgbm"])
    assert not matches_prepare(profile, ["algorithm:plane_sweep"])
    assert matches_prepare(profile, [])


def test_manual_profile_has_no_tour_constraint():
    root = Path(__file__).resolve().parents[1]
    profile = load_profile("sgbm-manual", root)
    assert profile.tour is None
    assert matches_prepare(profile, ["algorithm:sgbm-manual"])


def test_every_algorithm_has_a_maze020_profile():
    root = Path(__file__).resolve().parents[1]
    expected = {
        "sgbm": "sgbm-maze020", "constant": "constant-maze020",
        "exact_range": "exact-range-maze020",
        "plane_sweep": "plane-sweep-maze020",
        "feature_triangulation": "features-maze020",
        "optical_flow_triangulation": "optical-flow-maze020",
        "ground_plane": "ground-plane-maze020",
        "monocular_depth": "monocular-depth-maze020",
    }
    assert set(expected) == set(ALGORITHMS)
    for algorithm, profile_name in expected.items():
        profile = load_profile(profile_name, root)
        assert profile.algorithm == algorithm
        assert profile.tour.name == "maze_020.default.v1.json"


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


def test_video_frame_keeps_shared_memory_orientation():
    frame = np.array([[[1, 2, 3]], [[4, 5, 6]]], dtype=np.uint8)
    copied = copy_video_frame(frame)
    assert copied.tolist() == frame.tolist()
    copied[0, 0, 0] = 99
    assert frame[0, 0, 0] == 1


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


def test_every_profile_constructs_through_the_run_call_path():
    """The constant profile shipped broken: run.py passes intrinsics
    positionally, and ConstantAlgorithm did not accept a third argument. A
    profile that only loads is not a profile that runs."""
    root = Path(__file__).resolve().parents[1]
    intrinsics = Intrinsics(640, 480, 554.3, 554.3, 320.0, 240.0)
    truth = OccupancyGrid(np.full((8, 8), .05, np.float32), np.ones((8, 8), bool))
    for path in sorted(profile_dir(root).glob("*.json")):
        profile = load_profile(path.stem, root)
        settings = algorithm_settings(profile.algorithm, profile.settings)
        if profile.algorithm == "exact_range":
            ExactRangeAlgorithm(truth).finish()
            continue
        if profile.algorithm == "monocular_depth":
            continue  # needs a downloaded ONNX model; covered separately
        ALGORITHMS[profile.algorithm](40, 30, intrinsics, settings=settings)


def test_sensor_settings_do_not_reach_the_algorithm_configuration():
    root = Path(__file__).resolve().parents[1]
    profile = load_profile("sgbm-tof", root)
    assert profile.settings["range_stride"] == 8
    settings = algorithm_settings(profile.algorithm, profile.settings)
    assert "range_stride" not in settings
    with pytest.raises(ValueError, match="unknown settings"):
        algorithm_settings("sgbm", {"not_a_real_setting": 1})


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


def test_report_lands_in_the_module_directory_like_every_other_module(tmp_path):
    """``<report_root>/dalg/``, with no run directory nested inside it.

    dalg used to add a ``<run_id>-<profile>`` level so a second run in one
    simulator session could not overwrite the first. Every other module
    overwrites in that case, so the extra level only made the path opaque.
    """
    out = _report_fixture(tmp_path)

    assert out == tmp_path / "dalg"
    assert (out / "summary.json").is_file()
    assert not [child for child in out.iterdir() if child.is_dir()]


def test_report_html_still_finds_the_sibling_module_reports(tmp_path):
    """The flight images live one level up, and flattening moved that level.

    report.html embeds dsim's flight path and dway's track from the run root
    the modules share. That root was ``out.parent.parent`` while dalg nested a
    run directory; it is ``out.parent`` now, and getting it wrong loses the
    images silently rather than raising.
    """
    for relative in ("dsim/flight_path.png", "dway/track.png"):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (4, 4), (7, 8, 9)).save(target)

    html = (_report_fixture(tmp_path) / "report.html").read_text(encoding="utf-8")

    assert "Flight Path (dsim)" in html
    assert "Navigator Track (dway)" in html
    assert html.count("data:image/png;base64,") >= 2


def test_coordinator_watchdog_follows_the_clock_the_coordinator_paces_on():
    """Coordinators heartbeat on simulated time. Timing their silence on the
    wall clock aborted every run whose simulator lagged real time."""
    run = object.__new__(DalgRun)
    run._coordinator_seen = time.monotonic()-30.0
    run._coordinator_seen_sim = 10.0
    assert not run._coordinator_silent(11.0)
    assert run._coordinator_silent(10.0+run_module.COORDINATOR_SILENCE_SIM_S+.1)
    run._coordinator_seen = time.monotonic()-run_module.COORDINATOR_SILENCE_WALL_S-1
    assert run._coordinator_silent(10.0)


def _report_fixture(tmp_path, partial=False):
    """One finished report on disk, written the way a real run writes it."""
    from types import SimpleNamespace
    from dalg.model import Result
    from dalg.report import write_report

    truth = OccupancyGrid(np.full((6, 6), .05, np.float32), np.ones((6, 6), bool), 1.0)
    truth.probabilities[2, :] = .95
    predicted = OccupancyGrid(truth.probabilities.copy(), np.ones((6, 6), bool), 1.0)
    predicted.probabilities[2, 4:] = .05           # two walls missed
    empty = OccupancyGrid(np.full((6, 6), .05, np.float32), np.ones((6, 6), bool), 1.0)
    profile = SimpleNamespace(name="p", digest="d" * 64, algorithm="sgbm",
                              sensors=("rgb",))
    return write_report(
        tmp_path, run_id="r1", profile=profile, truth=truth,
        results={"sgbm": Result(predicted, {"frames": 3}),
                 "constant": Result(empty, {})},
        provenance={"frames": 3, "observable_cells": 30, "map_cells": 36},
        partial=partial, reason="operator requested shutdown" if partial else "",
        events=[{"type": "module.heartbeat", "role": "navigator", "sim_time_s": 0.0,
                 "payload": {}},
                {"type": "run.started", "role": "navigator", "sim_time_s": 1.0,
                 "payload": {"state": "FLYING"}}])


def test_report_writes_a_readable_html_summary(tmp_path):
    """summary.json is the record; report.html is how anyone reads it."""
    out = _report_fixture(tmp_path)
    html = (out / "report.html").read_text(encoding="utf-8")
    assert (out / "scored-region.png").exists()          # what the scores cover
    assert "overlay-sgbm.png" in html and "scored-region.png" in html
    assert "COMPLETE" in html and "sgbm" in html
    assert "0.667" in html                               # 4 of 6 walls found
    assert "run.started" in html                         # lifecycle survives
    assert "module.heartbeat" not in html                # liveness noise does not


def test_report_html_flags_an_aborted_run(tmp_path):
    html = (_report_fixture(tmp_path, partial=True) / "report.html").read_text()
    assert "ABORTED" in html
    assert "operator requested shutdown" in html


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

    The verdict overlay's own BACKGROUND shares the palette but sits at
    luminance 38, inside the range a prediction grid paints confident free
    space -- so using it directly hides the free space a run actually carved.
    """
    from dalg.overlay import BACKGROUND, UNDECIDED

    def luminance(colour):
        return 0.2126 * colour[0] + 0.7152 * colour[1] + 0.0722 * colour[2]

    assert luminance(UNDECIDED) == pytest.approx(127.0, abs=1.5)
    # Same hue as the overlay background, so the two images read as one palette.
    scale = luminance(UNDECIDED) / luminance(BACKGROUND)
    for channel, reference in zip(UNDECIDED, BACKGROUND):
        assert channel == pytest.approx(reference * scale, abs=1.0)


def test_report_publishes_a_prediction_grid_per_algorithm(tmp_path):
    """The picture that was on screen belongs beside the score it earned."""
    from dalg.report import write_report
    from dalg.report_html import _rgb as _rgb_css
    from dalg.overlay import UNDECIDED
    from dalg.truth import ground_truth
    from dalg.model import Result
    from types import SimpleNamespace

    sim_map = load_map(Path(__file__).resolve().parents[1]
                       / "assets/maps/maze_001.txt")
    truth = ground_truth(sim_map)
    grid = LogOddsGrid(sim_map.width, sim_map.height)
    grid.update([4, 5], [4, 4], 2.5)
    profile = SimpleNamespace(name="p", digest="d", sensors=("rgb",),
                              algorithm="optical_flow_triangulation")

    out = write_report(
        tmp_path, run_id="r1", profile=profile, truth=truth,
        results={"optical_flow_triangulation": Result(grid.result(), {}),
                 "constant": Result(truth, {})},
        provenance={}, partial=False, reason="", events=[], observable=None)

    for name in ("optical_flow_triangulation", "constant"):
        assert (out / f"prediction-{name}.png").is_file()
    html = (out / "report.html").read_text()
    assert "Prediction Grids" in html
    assert html.count("prediction-") == 2
    # The legend has to say what the greyscale means, and that scoring is a
    # threshold rather than the gradient the eye reads.
    assert "no opinion" in html and "undecided" in html
    assert _rgb_css(UNDECIDED) in html   # legend swatch matches the raster


class _FakeBus:
    """A module bus that records what was published and replays a script."""

    def __init__(self) -> None:
        self.process_id = "dalg-test"
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


def _finished_run():
    """A DalgRun that has completed and written its report."""
    from types import SimpleNamespace

    class _Unavailable:
        """The simulator has gone; every attach fails and connect() says so."""
        def open_existing(self, name): return False
        def open(self, name): return False

    run = object.__new__(DalgRun)
    run.bus = _FakeBus()
    run.pm = SimpleNamespace(memvid=_Unavailable, memkv=_Unavailable)
    run.names = shared_names("area1")
    run.video = run.status = None
    run.state = "COMPLETE"
    run.reason = "landed"
    run.run_id = "r1"
    run.done = True
    run.active = False
    run.shutdown_requested = False
    run.provenance = {"coordinator_outcome": "complete"}
    run.start_sim_time = 3.0
    run._coordinator_process_id = "dway-1"
    run._coordinator_seen = time.monotonic()
    run._coordinator_seen_sim = 100.0
    run._event_log = []
    run._hello_sent = True
    run._last_heartbeat = -1e9
    run._last_display_seq = -1
    run.profile = load_profile("optical-flow-maze020",
                               Path(__file__).resolve().parents[1])
    return run


def test_shutdown_is_still_heard_after_the_run_has_finished():
    """The kill-all signal has to reach a window left open on a result.

    main() used to skip step() once the run was done, and step() is the only
    thing that drains the bus -- so a finished dalg sat with its window open
    and never saw system.shutdown again.
    """
    run = _finished_run()
    run.bus.inbox = [_shutdown_event()]

    run.step()

    assert run.shutdown_requested is True


def test_shutdown_after_a_finished_run_keeps_its_outcome():
    """Stopping the process must not relabel a measurement that succeeded."""
    run = _finished_run()
    run.bus.inbox = [_shutdown_event()]

    run.step()

    assert run.provenance["coordinator_outcome"] == "complete"
    assert run.state == "COMPLETE"
    assert run.reason == "landed"


def test_a_finished_run_keeps_publishing_presence():
    """Otherwise it ages out of every PipelineView while its window is open."""
    run = _finished_run()

    run.step()
    run._last_heartbeat = -1e9      # let the next one through the 1 Hz gate
    run.step()

    assert run.bus.types().count("module.heartbeat") == 2
    for _, _, payload in run.bus.published:
        assert payload["state"] == "COMPLETE"
        assert payload["ready"] is False


def test_late_lifecycle_traffic_cannot_reopen_a_finished_run():
    """A coordinator repeating itself must not restart a closed measurement."""
    from dcmn.module_bus import ModuleEvent

    run = _finished_run()
    run.bus.inbox = [ModuleEvent(
        event_id="e2", instance_id="area1", role="navigator",
        implementation="dway", process_id="dway-1", sequence=9,
        sim_time_s=99.0, type="run.start_scheduled", run_id="r1",
        payload={"start_sim_time_s": 120.0})]

    run.step()

    assert run.state == "COMPLETE"
    assert run.start_sim_time == 3.0          # the finished run's own start
    assert run.provenance["coordinator_outcome"] == "complete"


def test_main_keeps_stepping_a_finished_run_while_its_window_is_open(monkeypatch):
    """The window outlives the tour, and so must the bus drain behind it.

    main() used to guard the step with ``if not run.done``, which stopped the
    only call that drains the bus and publishes presence. The window stayed on
    screen showing the result and quietly ignored system.shutdown.
    """
    import dalg.dalg as dalg_module

    class _Run:
        """Finishes on the third step, then reports a shutdown on the sixth."""
        def __init__(self, *args, **kwargs):
            self.steps = 0
            self.steps_after_done = 0
            self.done = False
            self.active = False
            self.shutdown_requested = False
            self.reason = ""
            self.report_dir = None
            self.provenance = {}
            self.closed = False

        def step(self):
            self.steps += 1
            if self.done:
                self.steps_after_done += 1
            if self.steps == 3:
                self.done = True
                self.provenance["coordinator_outcome"] = "complete"
            if self.steps_after_done == 3:
                self.shutdown_requested = True

        def close(self):
            self.closed = True

    class _Window:
        # update() runs every iteration whether or not step() does, so it is
        # also the safety valve: without it a regression here hangs the suite
        # instead of failing it, which is exactly what the bug did to dalg.
        BUDGET = 200

        def __init__(self, run):
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

    code = dalg_module.main(["--id", "area1",
                             "--profile", "optical-flow-maze020"])
    run = created["run"]

    assert run.steps_after_done == 3, (
        "the bus stopped being drained once the run finished")
    assert run.shutdown_requested is True
    assert run.closed is True
    # A completed run that was then shut down is still a completed run.
    assert code == 0
