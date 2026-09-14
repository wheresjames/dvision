"""Every camera algorithm, as an evaluated measurement and as an evidence source.

Two promises are held here. The algorithms' own measurements stay exactly what
the offline evaluator (dtest.evaluation) recorded for them, and every grid an
evidence source publishes is one dnav will accept, stamped with capture times
and built without ever touching truth.

The fixture is a textured wall at a fixed depth, seen from a camera sliding
sideways, with each frame shifted by the parallax a real lens would see. That
is what gives the triangulating algorithms something to find: the older
sideways fixture reuses one image for every pose, which is zero parallax, and
optical flow and feature matching fuse nothing from it.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from dalg.algo import ALGORITHMS
from dalg.grid import OccupancyGrid
from dalg.model import Frame, Intrinsics, Pose
from dtest.evaluation import score_occupancy

ROOT = Path(__file__).resolve().parents[1]
INTRINSICS = Intrinsics(160, 120, 100.0, 100.0, 80.0, 60.0)
#: The camera algorithms. The controls are evaluation-only and never registered.
CAMERA_ALGORITHMS = sorted(ALGORITHMS)


def parallax_frames(count: int = 16, *, shift_px: int = 5, depth_m: float = 6.0,
                    floor_row: int = 90, first_time_s: float = 0.0):
    """A wall ``depth_m`` north of a camera sliding east, and a floor below it.

    Each frame crops the texture ``shift_px`` further along, which is exactly
    the parallax of a plane at ``depth_m`` for a camera moving
    ``depth_m * shift_px / fx`` metres. The white floor gives the ground-plane
    algorithm a boundary to find.
    """
    spacing_m = depth_m * shift_px / INTRINSICS.fx_px
    rng = np.random.default_rng(11)
    texture = cv2.GaussianBlur(
        rng.integers(0, 255, (INTRINSICS.height_px, INTRINSICS.width_px + count * shift_px),
                     np.uint8), (5, 5), 0)
    frames = []
    for index in range(count):
        gray = texture[:, index * shift_px:index * shift_px + INTRINSICS.width_px].copy()
        gray[floor_row:] = 255
        pose = Pose(10.0 + index * spacing_m, 10.0, 1.5, 0.0)
        frames.append(Frame(index, first_time_s + index * 0.2,
                            np.repeat(gray[..., None], 3, axis=2), pose, None, None, pose))
    return frames


def settings_for(name: str) -> dict:
    """What each algorithm needs to run at all; the depth model comes from its baseline."""
    if name != "monocular_depth":
        return {}
    settings = json.loads((ROOT / "assets/algorithm_profiles/monocular-depth-baseline.json")
                          .read_text(encoding="utf-8"))["sources"][0]["settings"]
    model = ROOT / settings["model_path"]
    if not model.is_file():
        pytest.skip("the monocular depth model is not installed")
    pytest.importorskip("onnxruntime")
    return dict(settings, model_path=str(model))


def wall_truth() -> OccupancyGrid:
    """The fixture's world at the scoring resolution: a wall at y = 4 m."""
    truth = OccupancyGrid(np.full((120, 120), .05, np.float32), np.ones((120, 120), bool), .25)
    truth.probabilities[14:18, :] = .95
    return truth


def scored(name: str):
    """Run one algorithm the way a measured run does, and score what it finishes with."""
    algorithm = ALGORITHMS[name](30, 30, INTRINSICS, settings=settings_for(name))
    algorithm.start()
    for frame in parallax_frames():
        algorithm.observe(frame)
        algorithm.preview()
    return algorithm.finish()


# Captured from the algorithms before any of them could publish evidence. A
# change here is a change to what an algorithm scores, which converting it to
# publish evidence must never cause.
GOLDEN = {
    'feature_triangulation': {
        'diagnostics': {'keyframes': 8, 'matches': 1995, 'triangulated_points': 1995},
        'scores': {
            'brier': 0.063004,
            'coverage': 0.049375,
            'free_iou': 0.040189,
            'hallucination_rate': 0.00079,
            'occupied_iou': 0.203666,
            'occupied_precision': 0.900901,
            'occupied_recall': 0.208333,
        },
    },
    'ground_plane': {
        'diagnostics': {'frames': 16, 'projected_boundaries': 640},
        'scores': {
            'brier': 0.14845,
            'coverage': 0.048194,
            'free_iou': 0.040948,
            'hallucination_rate': 0.008908,
            'occupied_iou': 0.0,
            'occupied_precision': 0.0,
            'occupied_recall': 0.0,
        },
    },
    'monocular_depth': {
        'diagnostics': {'depth_points': 4800, 'frames': 16},
        'scores': {
            'brier': 0.901559,
            'coverage': 0.002847,
            'free_iou': 0.0,
            'hallucination_rate': 0.002945,
            'occupied_iou': 0.0,
            'occupied_precision': 0.0,
            'occupied_recall': 0.0,
        },
    },
    'optical_flow_triangulation': {
        'diagnostics': {'frames': 16, 'tracks': 1608, 'triangulated_points': 1584},
        'scores': {
            'brier': 0.053987,
            'coverage': 0.066806,
            'free_iou': 0.056677,
            'hallucination_rate': 0.000287,
            'occupied_iou': 0.231405,
            'occupied_precision': 0.965517,
            'occupied_recall': 0.233333,
        },
    },
    'plane_sweep': {
        'diagnostics': {'accepted_points': 728, 'depth_hypotheses': 16, 'frames': 8, 'frames_with_depth': 8},
        'scores': {
            'brier': 0.147502,
            'coverage': 0.089375,
            'free_iou': 0.075368,
            'hallucination_rate': 0.006968,
            'occupied_iou': 0.0,
            'occupied_precision': 0.0,
            'occupied_recall': 0.0,
        },
    },
    'sgbm': {
        'diagnostics': {'frames': 16, 'range_frames': 0, 'range_points': 0, 'stereo_pairs': 15, 'stereo_points': 540},
        'scores': {
            'brier': 0.002188,
            'coverage': 0.002153,
            'free_iou': 0.0,
            'hallucination_rate': 0.0,
            'occupied_iou': 0.064583,
            'occupied_precision': 1.0,
            'occupied_recall': 0.064583,
        },
    },
}


@pytest.mark.parametrize("name", CAMERA_ALGORITHMS)
def test_scoring_is_exactly_what_it_was_before_evidence(name):
    assert name in GOLDEN, f"{name} has no recorded score to hold it to"
    result = scored(name)
    expected = GOLDEN[name]
    assert result.diagnostics == expected["diagnostics"]
    scores = score_occupancy(result.grid, wall_truth())
    for key, value in expected["scores"].items():
        assert scores[key] == pytest.approx(value, abs=1e-6), key


# -- the evidence copy ---------------------------------------------------------

def _evidence_algorithms():
    from dalg.profiles import camera_evidence_algorithms
    return list(camera_evidence_algorithms())


def evidence_for(name: str, settings=None, extent=(30., 30.)):
    """One algorithm's evidence source on a 0.5 m runtime geometry, with its own plane."""
    import uuid
    from dalg.evidence import CameraEvidence
    from dalg.profiles import Source
    from dcmn.maps import GridGeometry, MapPublisher

    geometry = GridGeometry.from_extent(*extent, .5)
    publisher = MapPublisher(f"ev-{uuid.uuid4().hex[:8]}", geometry, [dict(
        id=Source("front", name).id, sensor="front", sensor_type="camera.rgb", algorithm=name)])
    evidence = CameraEvidence(geometry, INTRINSICS, "front",
                              settings_for(name) if settings is None else settings,
                              algorithm=name, publisher=publisher)
    evidence.close = publisher.close
    return evidence


def test_every_camera_algorithm_but_the_controls_is_registered():
    """The registry is the whole switch; this is where a missed one shows."""
    from dalg.profiles import camera_evidence_algorithms, source_configs
    assert set(camera_evidence_algorithms()) == set(CAMERA_ALGORITHMS)
    assert not {"constant", "exact_range", "synthetic"} & set(source_configs())


@pytest.mark.parametrize("name", _evidence_algorithms())
def test_every_published_record_is_one_dnav_accepts(name):
    """Observed cells carry capture times -- never publish times -- and decode."""
    from dcmn.maps import MapSession, stamp_ms

    evidence = evidence_for(name)
    session = MapSession(evidence.publisher.instance)
    frames = parallax_frames(first_time_s=5.0)
    try:
        for frame in frames:
            evidence.observe(frame)
            # Published long after capture, so a stamp taken at publish time
            # rather than capture time cannot pass for the right one.
            evidence.publish(frame.timestamp_s + 100.0, force=True)
            session.poll()
        state = session.states[evidence.source]
        assert state.rejected == 0, state.last_reason
        assert state.records == len(frames)
        grid = session.latest(evidence.source)
        assert grid.observed.any(), f"{name} published nothing it observed"
        captured = {int(stamp_ms(frame.timestamp_s)) for frame in frames}
        assert set(grid.observed_ms[grid.observed].tolist()) <= captured
    finally:
        session.close()
        evidence.close()


@pytest.mark.parametrize("name", _evidence_algorithms())
def test_the_evidence_path_never_touches_truth(name, monkeypatch):
    """Belief, not truth: no map file, no truth grid, and no range samples."""
    import dsim.range
    import dtest.evaluation
    import dvision2_common

    def forbidden(*args, **kwargs):
        raise AssertionError("the evidence path reached for truth")
    for module, attribute in ((dvision2_common, "load_map"), (dtest.evaluation, "ground_truth"),
                              (dtest.evaluation, "ground_truth_on"), (dsim.range, "raycast_map")):
        monkeypatch.setattr(module, attribute, forbidden)

    evidence = evidence_for(name)
    seen_range = []
    original = evidence.algorithm.observe
    def observe(frame):
        seen_range.append(frame.range_m)
        return original(frame)
    monkeypatch.setattr(evidence.algorithm, "observe", observe)
    try:
        for frame in parallax_frames(8):
            # Range is ray-cast from the map, so it is truth: offer it, and
            # require that it never arrives.
            evidence.observe(_with_range(frame))
        evidence.publish(10.0, force=True)
    finally:
        evidence.close()
    assert seen_range and all(value is None for value in seen_range)


def _with_range(frame):
    from dataclasses import replace
    return replace(frame, range_m=np.ones((12, 16)), range_confidence=np.ones((12, 16)))


def test_feature_triangulation_keeps_a_bounded_window_and_an_honest_count():
    """Memory stays flat over a long run, and the reported count does not cap."""
    from dalg.algo.features import MATCH_WINDOW

    algorithm = ALGORITHMS["feature_triangulation"](60, 30, INTRINSICS)
    for frame in parallax_frames(60, shift_px=3):
        algorithm.observe(frame)
    assert len(algorithm.frames) == MATCH_WINDOW
    assert algorithm.finish().diagnostics["keyframes"] > MATCH_WINDOW


#: Depth algorithms that marked only obstacles until they could publish.
DENSE_ALGORITHMS = ("monocular_depth", "sgbm")


@pytest.mark.parametrize("name", [n for n in DENSE_ALGORITHMS if n in _evidence_algorithms()])
def test_dense_evidence_marks_free_space_the_scored_copy_never_did(name):
    """Observed free is the evidence grid's most valuable bit; the score stays put."""
    assert int(scored(name).grid.free.sum()) == 0, "the scored copy changed"
    evidence = evidence_for(name)
    try:
        for frame in parallax_frames():
            evidence.observe(frame)
        grid = evidence.algorithm.grid.result()
        assert int(grid.free.sum()) > 0
        assert int(grid.occupied.sum()) > 0, "clearing must not erase what was seen"
    finally:
        evidence.close()


def test_plane_sweep_evidence_keeps_going_where_the_scored_copy_stops():
    """The scored copy stops observing at max_keyframes; a live source cannot."""
    from dalg.algo.plane_sweep import EVIDENCE_WINDOW

    frames = parallax_frames(48, first_time_s=1.0)
    settings = {"max_keyframes": 4}
    scored_copy = ALGORITHMS["plane_sweep"](30, 30, INTRINSICS, settings=settings)
    for frame in frames: scored_copy.observe(frame)
    assert len(scored_copy.frames) == 4

    evidence = evidence_for("plane_sweep", settings, extent=(60., 30.))
    pairs = []
    depth = evidence.algorithm._depth
    def recording(index, neighbor):
        pairs.append((index, neighbor))
        return depth(index, neighbor)
    evidence.algorithm._depth = recording
    try:
        for frame in frames: evidence.observe(frame)
        grid = evidence.algorithm.grid
        assert len(evidence.algorithm.frames) <= EVIDENCE_WINDOW
        assert len(pairs) > 4, "evidence stopped with the scored copy"
        # Causal: every partner was captured before the frame it was fused for.
        assert all(neighbor < index for index, neighbor in pairs)
        # Still observing at the end, not frozen at the fourth keyframe.
        assert grid.observed_ms.max() / 1000.0 >= frames[-4].timestamp_s
    finally:
        evidence.close()


def test_sgbm_evidence_pairs_only_with_the_past_in_a_bounded_window():
    """finish() may use the future; live evidence may not, and must not grow."""
    from dalg.algo.sgbm import EVIDENCE_WINDOW

    evidence = evidence_for("sgbm")
    algorithm = evidence.algorithm
    fused = []
    stereo = algorithm._fuse_stereo
    def recording(index, other, baseline):
        # Both frames must already be in hand, and the pair can be no newer
        # than the frame that has just arrived.
        fused.append((algorithm.times[index], algorithm.times[other],
                      algorithm.times[-1], algorithm.grid.timestamp_s))
        return stereo(index, other, baseline)
    algorithm._fuse_stereo = recording
    try:
        frames = parallax_frames(96, shift_px=2)
        for frame in frames:
            evidence.observe(frame)
        assert len(algorithm.frames) <= EVIDENCE_WINDOW
        assert fused, "sgbm fused nothing from a sideways slide"
        for reference_s, partner_s, newest_s, stamp_s in fused:
            assert max(reference_s, partner_s) == newest_s
            assert stamp_s == newest_s
    finally:
        evidence.close()


# -- scoring the published grid ------------------------------------------------

def _maze():
    from dvision2_common import load_map
    return load_map(ROOT / "assets/maps/maze_012.txt")


def test_truth_lands_on_the_evidence_grids_own_cells():
    """Offset origin, a different cell size, and area beyond the map."""
    from dtest.evaluation import ground_truth, ground_truth_on
    from dcmn.maps import GridGeometry

    sim_map = _maze()
    wall = next(o for o in sim_map.objects if o.kind == "wall")
    geometry = GridGeometry(-2.0, -2.0, .5, 100, 80)       # reaches past the map
    truth = ground_truth_on(sim_map, geometry)
    col, row = geometry.to_cell(wall.x, wall.y)
    assert truth.occupied[row, col], "a wall is not where the map puts it"
    assert not truth.observed[0, 0], "cells off the map must not be scored"
    base = ground_truth(sim_map, .5)
    assert int(truth.observed.sum()) == base.observed.size


def test_a_perfect_published_grid_scores_perfectly_and_an_empty_one_scores_nothing():
    from dtest.evaluation import ground_truth_on, score_evidence
    from dcmn.maps import EvidenceGrid, GridGeometry, quantize, stamp_ms

    sim_map = _maze()
    geometry = GridGeometry.from_extent(sim_map.width, sim_map.height, .5)
    truth = ground_truth_on(sim_map, geometry)
    occupancy = quantize(truth.probabilities)[None]
    observed = np.full(geometry.shape, stamp_ms(1.0), np.uint32)
    perfect = EvidenceGrid(geometry, occupancy, observed, "cam")
    poses = [Pose(sim_map.start_x, sim_map.start_y, 1.5, heading) for heading in range(0, 360, 45)]
    scores = score_evidence(perfect, sim_map, poses)
    assert scores["occupied_iou"] == pytest.approx(1.0)
    assert scores["free_iou"] == pytest.approx(1.0)

    empty = EvidenceGrid.blank(geometry, "cam")
    scores = score_evidence(empty, sim_map, poses)
    assert scores["coverage"] == 0.0


def test_a_shared_observation_moves_into_the_evidence_grids_origin():
    from dalg.algo.spatial import PointObservation

    observation = PointObservation(Pose(10.0, 20.0, 1.5, 90.0), np.array([12.0, 15.0]),
                                   np.array([20.0, 22.0]), np.array([1.0, 2.0]))
    local = observation.translated(-4.0, 6.0)
    assert (local.pose.x_m, local.pose.y_m, local.pose.z_m) == (14.0, 14.0, 1.5)
    assert local.xs.tolist() == [16.0, 19.0] and local.ys.tolist() == [14.0, 16.0]
    assert local.zs.tolist() == [1.0, 2.0]
