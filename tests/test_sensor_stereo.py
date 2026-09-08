"""Synchronized RGB pairs: disparity, shared capture, independent misalignment.

A stereo rig in this simulator is two ordinary cameras that share a rate, a
lens model and a ``sync_group``; everything that makes it stereo lives in the
two explicit poses. These tests check the geometry that produces against the
pinhole relation ``disparity = fx * baseline / depth``, from several headings
and depths, and check that a per-camera misalignment stays in its own image.
"""

from __future__ import annotations

import numpy as np
import pytest

from dsim.profiles import DroneProfile, default_profile, stereo_pair
from dtest.calibration_scene import (CALIBRATION_RING_MAP, CENTER_X,
                                     FRAME_HEIGHT, FRAME_WIDTH, RING_HEADINGS,
                                     RING_START_Y_M, START_Y_M)
from dtest.color_probe import color_centroid
from dtest.deterministic import DeterministicSim

pytest.importorskip("panda3d")

#: An exaggerated baseline: the geometry is the same as a 12 cm rig, but the
#: disparity is large enough to measure against centroid noise.
BASELINE_M = 0.5
LENS = dict(width_px=FRAME_WIDTH, height_px=FRAME_HEIGHT, fov_h_deg=70.0,
            near_m=0.15, far_m=150.0)
#: Landmarks on the row the calibration fixtures place 7 m ahead of the drone.
LANDMARKS = ("red", "white", "blue")
LANDMARK_ROW_Y_M = 5.5


def stereo_profile(*, baseline_m=BASELINE_M, right_pose=None):
    draft = default_profile(FRAME_WIDTH, FRAME_HEIGHT)
    draft["sensors"] = stereo_pair("nav", baseline_m=baseline_m,
                                   pose=dict(z_m=0.1, pitch_deg=-5.0),
                                   model=dict(LENS))
    if right_pose:
        draft["sensors"][1]["pose_parent"].update(right_pose)
    draft["primary_camera"] = "nav_left"
    return DroneProfile.parse(draft)


def pair(profile, **sim_kwargs):
    sim = DeterministicSim(drone_profile=profile, **sim_kwargs)
    try:
        return sim.render_group("nav_stereo")
    finally:
        sim.close()


def disparities(frames, names=LANDMARKS):
    out = {}
    for name in names:
        left = color_centroid(frames["nav_left"], name)
        right = color_centroid(frames["nav_right"], name)
        assert left is not None and right is not None, f"{name} missing from the pair"
        out[name] = left.x - right.x
    return out


def test_disparity_matches_the_baseline_and_depth():
    """The pinhole relation, at two depths, with the sign the geometry implies.

    The right camera sits toward +Y in body axes, so a point ahead lands
    further left in its image: ``x_left - x_right`` is positive.
    """
    profile = stereo_profile()
    fx = profile.primary["model"]["fx_px"]
    sim = DeterministicSim(drone_profile=profile)
    try:
        for y in (START_Y_M, START_Y_M - 3.0):
            sim.set_pose(sim.position[0], y, 1.5, 0.0)
            depth = y - LANDMARK_ROW_Y_M
            expected = fx * BASELINE_M / depth
            # Only the near-axis landmark: at 4 m the outer panels leave the
            # frame, and a clipped mask moves its own centroid.
            for name, value in disparities(sim.render_group("nav_stereo"),
                                           ("white",)).items():
                assert value == pytest.approx(expected, rel=0.05), (
                    f"{name} disparity {value:.2f} px, expected "
                    f"{expected:.2f} px at {depth:.1f} m")
    finally:
        sim.close()


@pytest.mark.parametrize("baseline_m", [0.25, 0.5, 1.0])
def test_disparity_scales_with_the_baseline(baseline_m):
    profile = stereo_profile(baseline_m=baseline_m)
    fx = profile.primary["model"]["fx_px"]
    expected = fx * baseline_m / (START_Y_M - LANDMARK_ROW_Y_M)
    measured = np.mean(list(disparities(pair(profile)).values()))
    assert measured == pytest.approx(expected, rel=0.05)


@pytest.mark.parametrize("heading", RING_HEADINGS)
def test_disparity_is_the_same_from_every_heading(heading):
    """The baseline rotates with the body, so the rig is not axis-aligned."""
    profile = stereo_profile()
    fx = profile.primary["model"]["fx_px"]
    frames = pair(profile, map_path=CALIBRATION_RING_MAP, heading_deg=heading)
    expected = fx * BASELINE_M / (RING_START_Y_M - 3.5)
    for name, value in disparities(frames).items():
        assert value == pytest.approx(expected, rel=0.06), (
            f"{name} disparity {value:.2f} px at heading {heading:.0f}")


def test_the_pair_is_rendered_from_one_vehicle_state():
    """Two renders of the same group must be identical, frame for frame."""
    profile = stereo_profile()
    sim = DeterministicSim(drone_profile=profile)
    try:
        first = {k: v.copy() for k, v in sim.render_group("nav_stereo").items()}
        second = sim.render_group("nav_stereo")
        assert set(first) == {"nav_left", "nav_right"}
        for name in first:
            assert np.array_equal(first[name], second[name])
        assert not np.array_equal(first["nav_left"], first["nav_right"])
    finally:
        sim.close()


def test_a_misaligned_member_moves_only_its_own_image():
    """Manufacturing error belongs to one camera, not to the pair."""
    aligned = pair(stereo_profile())
    skewed = pair(stereo_profile(right_pose=dict(yaw_deg=2.0)))
    assert np.array_equal(aligned["nav_left"], skewed["nav_left"])
    assert not np.array_equal(aligned["nav_right"], skewed["nav_right"])

    fx = stereo_profile().primary["model"]["fx_px"]
    for name in LANDMARKS:
        before = color_centroid(aligned["nav_right"], name).x
        after = color_centroid(skewed["nav_right"], name).x
        # Yawing the camera right sweeps the scene left in its own image, by a
        # pinhole re-projection rather than a constant pixel offset: the shift
        # grows with the landmark's angle off the optical axis.
        expected = CENTER_X + fx * np.tan(
            np.arctan((before - CENTER_X) / fx) - np.radians(2.0))
        assert after == pytest.approx(expected, abs=2.0), name
        assert after < before


def test_the_primary_camera_is_one_member_of_the_pair():
    profile = stereo_profile()
    sim = DeterministicSim(drone_profile=profile)
    try:
        assert np.array_equal(sim.render(), sim.render_group("nav_stereo")["nav_left"])
        right = sim.render(camera_id="nav_right")
        assert not np.array_equal(right, sim.render(camera_id="nav_left"))
        assert color_centroid(right, "white").x < CENTER_X + FRAME_WIDTH
    finally:
        sim.close()


@pytest.mark.parametrize("scene_preset", ["legacy", "representative"])
def test_prepared_camera_buffers_can_be_discarded_without_changing_rgb(monkeypatch, scene_preset):
    sim = DeterministicSim(drone_profile=stereo_profile(), scene_preset=scene_preset)
    try:
        before = sim.render().copy()
        renderer = sim._renderer
        active = dict(renderer._views)
        draft = stereo_profile().data
        for sensor in draft['sensors']:
            sensor['model']['width_px'] //= 2
            sensor['model']['height_px'] //= 2
        prepared = renderer.prepare_profile(DroneProfile.parse(draft))
        assert renderer._views == active
        renderer.finish_profile(prepared, commit=False)
        assert renderer._views == active
        assert np.array_equal(sim.render(), before)

        original = renderer.view
        def fail_second(camera_id, model):
            if camera_id == 'nav_right':
                raise RuntimeError('second buffer unavailable')
            return original(camera_id, model)
        monkeypatch.setattr(renderer, 'view', fail_second)
        with pytest.raises(RuntimeError, match='second buffer unavailable'):
            renderer.prepare_profile(DroneProfile.parse(draft))
        monkeypatch.undo()
        assert renderer._views == active
        assert np.array_equal(sim.render(), before)
    finally:
        sim.close()
