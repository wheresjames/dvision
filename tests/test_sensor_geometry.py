"""Analytic ray geometry, cone reduction, scan ordering and noise determinism.

The ray sensors are checked against distances that can be worked out on paper,
against the same collision geometry the physics uses, and against the rule
that a measurement depends on its logical capture rather than on how many
samples happened to be published before it.
"""

from __future__ import annotations

import dataclasses
import math
from types import SimpleNamespace

import numpy as np
import pytest

from dsim import sensor_models
from dsim.dsim import Panda3DRenderer
from dsim.headless import HeadlessSimulator
from dsim.profiles import DroneProfile, default_profile, scan_angles_deg
from dsim.range import cast_rays, scene_geometry
from dtest.calibration_scene import CALIBRATION_MAP

WALL_H = Panda3DRenderer.WALL_H
#: Every model field that would otherwise blur an analytic distance.
NOISELESS = dict(noise_std_m=0.0, quantization_m=0.0, dropout_probability=0.0,
                 limit_degradation=0.0, confidence_model="exact")


def build(*sensors, physics_hz=300.0):
    profile = default_profile()
    profile["sensors"].extend(sensors)
    return DroneProfile.parse(profile)


def beam(sensor_id, kind="range.laser", *, pitch_deg=0.0, yaw_deg=0.0, x_m=0.0,
         z_m=0.0, samples=1, fov_deg=0.0, parent="body", **model):
    return dict(id=sensor_id, type=kind, enabled=True, rate_hz=10.0, parent=parent,
                pose_parent=dict(x_m=x_m, z_m=z_m, pitch_deg=pitch_deg, yaw_deg=yaw_deg),
                model={**NOISELESS, "beam_samples": samples, "beam_fov_deg": fov_deg,
                       "min_range_m": 0.02, "max_range_m": 50.0, **model})


def sim_with(profile, objects=(), *, x=None, y=None, z=1.5, heading_deg=0.0):
    driver = HeadlessSimulator(map_path=CALIBRATION_MAP, drone_profile=profile,
                               altitude_m=z, heading_deg=heading_deg)
    driver.sim.map = dataclasses.replace(driver.sim.map, objects=list(objects))
    state = driver.sim.state
    if x is not None: state.x = x
    if y is not None: state.y = y
    state.z = z
    return driver


def wall(x, y):
    return SimpleNamespace(symbol="#", kind="wall", x=x, y=y)


# ---------------------------------------------------------------------------
# Analytic distances
# ---------------------------------------------------------------------------

def test_a_downward_beam_measures_height_above_the_ground():
    driver = sim_with(build(beam("down", pitch_deg=-90.0)), z=2.25)
    assert driver.sample("down")["value_m"] == pytest.approx(2.25)


@pytest.mark.parametrize("heading,wall_xy,expected", [
    (90.0, (4.5, 3.5), 1.0),     # +X: the near face sits at x = 4.0
    (270.0, (0.5, 3.5), 2.0),    # -X: the near face sits at x = 1.0
    (0.0, (3.5, 0.5), 2.0),      # -Y (north): the near face sits at y = 1.0
    (180.0, (3.5, 6.5), 3.0),    # +Y (south): the near face sits at y = 6.0
])
def test_a_forward_beam_measures_the_near_face_from_every_heading(heading, wall_xy, expected):
    driver = sim_with(build(beam("probe")), [wall(*wall_xy)],
                      x=3.0, y=3.0, z=1.0, heading_deg=heading)
    assert driver.sample("probe")["value_m"] == pytest.approx(expected)


def test_a_downward_beam_over_a_wall_reads_the_wall_not_the_ground():
    """The box has a top: a rangefinder above one must not see through it."""
    driver = sim_with(build(beam("down", pitch_deg=-90.0)), [wall(3.0, 3.0)],
                      x=3.0, y=3.0, z=WALL_H + 1.25)
    assert driver.sample("down")["value_m"] == pytest.approx(1.25)
    assert not driver.sim.is_blocked(3.0, 3.0, WALL_H + 1.25)
    assert driver.sim.is_blocked(3.0, 3.0, WALL_H - 0.1)


def test_a_tilted_beam_measures_the_slant_range():
    driver = sim_with(build(beam("tilt", pitch_deg=-45.0)), z=2.0)
    assert driver.sample("tilt")["value_m"] == pytest.approx(2.0 * math.sqrt(2.0))


def test_a_miss_is_invalid_rather_than_maximum_range():
    driver = sim_with(build(beam("up", pitch_deg=90.0, max_range_m=5.0)), z=1.5)
    reading = driver.sample("up")
    assert reading["value_m"] is None
    assert reading["returns"] == 0
    assert reading["value_confidence"] == 0.0
    assert np.isnan(reading["range_m"]).all()


def test_the_minimum_range_gate_rejects_a_close_return():
    profile = build(beam("blind", pitch_deg=-90.0, min_range_m=1.0))
    assert sim_with(profile, z=0.5).sample("blind")["value_m"] is None
    assert sim_with(profile, z=1.5).sample("blind")["value_m"] == pytest.approx(1.5)


def test_the_maximum_range_gate_rejects_a_distant_return():
    profile = build(beam("short", pitch_deg=-90.0, max_range_m=2.0))
    assert sim_with(profile, z=3.0).sample("short")["value_m"] is None


def test_linear_confidence_is_the_unmeasured_fraction_of_full_scale():
    """`range_linear` confidence is 255 * (1 - range / max_range): a beam
    reading 2.25 m of a 7 m scale reports 1 - 2.25/7 of it, to the eight bits
    the channel carries, and it is the miss, not the measurement at the
    limit, that reads zero."""
    profile = build(beam("down", pitch_deg=-90.0, max_range_m=7.0,
                         confidence_model="range_linear"))
    reading = sim_with(profile, z=2.25).sample("down")
    assert reading["value_confidence"] == pytest.approx(1.0 - 2.25 / 7.0,
                                                        abs=1.0 / 255.0)


def test_a_mount_translation_is_rotated_by_the_body_heading():
    """A 1 m forward mount must move the measured face by 1 m at any heading."""
    for heading, wall_xy in ((90.0, (5.5, 3.5)), (0.0, (3.5, 0.5))):
        plain = sim_with(build(beam("probe")), [wall(*wall_xy)],
                         x=3.0, y=3.0, z=1.0, heading_deg=heading)
        ahead = sim_with(build(beam("probe", x_m=1.0)), [wall(*wall_xy)],
                         x=3.0, y=3.0, z=1.0, heading_deg=heading)
        assert (plain.sample("probe")["value_m"]
                - ahead.sample("probe")["value_m"]) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Cones and reducers
# ---------------------------------------------------------------------------

def test_the_cone_spreads_within_its_half_angle_and_starts_on_the_axis():
    rays = sensor_models.cone_directions(dict(beam_fov_deg=30.0, beam_samples=16))
    assert np.allclose(np.linalg.norm(rays, axis=-1), 1.0)
    assert np.allclose(rays[0], (1.0, 0.0, 0.0))
    assert np.degrees(np.arccos(rays[:, 0])).max() == pytest.approx(15.0)


@pytest.mark.parametrize("reducer,expected", [
    ("nearest", 1.0), ("farthest", 3.0), ("median", 2.0)])
def test_the_reducer_selects_the_declared_return(reducer, expected):
    ranges = np.array([2.0, np.nan, 3.0, 1.0], np.float32)
    confidence = np.array([200, 0, 100, 255], np.uint8)
    value, level, returns = sensor_models.reduce_beam(ranges, confidence, reducer)
    assert (value, returns) == (expected, 3)
    assert 0.0 <= level <= 1.0


def test_a_wide_cone_finds_a_target_a_single_ray_would_miss():
    """The nearest valid return in the cone, which is what a sonar reports."""
    obstacles = [wall(4.5, 2.5)]
    narrow = sim_with(build(beam("narrow", fov_deg=0.0, samples=1)), obstacles,
                      x=3.0, y=3.0, z=1.0, heading_deg=90.0)
    wide = sim_with(build(beam("wide", kind="range.ultrasonic", fov_deg=60.0,
                               samples=64)), obstacles,
                    x=3.0, y=3.0, z=1.0, heading_deg=90.0)
    assert narrow.sample("narrow")["value_m"] is None
    reading = wide.sample("wide")
    assert reading["value_m"] is not None and reading["returns"] > 0
    assert reading["value_m"] >= 1.0


# ---------------------------------------------------------------------------
# Scan and range-image ordering
# ---------------------------------------------------------------------------

def test_a_full_circle_scan_steps_without_repeating_its_first_ray():
    model = dict(fov_deg=360.0, samples=4, elevation_deg=0.0)
    assert scan_angles_deg(model) == (-180.0, 90.0)
    rays = sensor_models.scan_directions(model)
    # -180, -90, 0, +90: backward, left, forward, right in forward/right/up.
    assert np.allclose(rays, [(-1, 0, 0), (0, -1, 0), (1, 0, 0), (0, 1, 0)], atol=1e-9)


def test_a_partial_scan_includes_both_of_its_endpoints():
    assert scan_angles_deg(dict(fov_deg=90.0, samples=5)) == (-45.0, 22.5)


def test_scan_ranges_are_ordered_by_angle_around_the_vehicle():
    scan = dict(id="scan", type="lidar.scan2d", enabled=True, rate_hz=10.0,
                parent="body", pose_parent={},
                model=dict(NOISELESS, fov_deg=360.0, samples=4,
                           min_range_m=0.02, max_range_m=50.0))
    driver = sim_with(build(scan), [wall(4.5, 3.5)],
                      x=3.0, y=3.0, z=1.0, heading_deg=90.0)
    ranges = driver.sample("scan")["range_m"]
    assert ranges.shape == (4,)
    # Index 2 is the forward ray, and only it sees the wall 1 m ahead.
    assert ranges[2] == pytest.approx(1.0)
    assert np.isnan(np.delete(ranges, 2)).all()


def test_a_range_image_is_row_major_and_measures_its_centre_pixel():
    image = dict(id="flash", type="lidar.range_image", enabled=True, rate_hz=10.0,
                 parent="body", pose_parent={},
                 model=dict(NOISELESS, width_px=9, height_px=7, fov_h_deg=60.0,
                            cx_px=4.0, cy_px=3.0,
                            min_range_m=0.02, max_range_m=50.0))
    driver = sim_with(build(image), [wall(4.5, 3.5)],
                      x=3.0, y=3.0, z=1.0, heading_deg=90.0)
    reading = driver.sample("flash")
    assert reading["range_m"].shape == (7, 9)
    # cx/cy default to the exact image centre, so the centre ray is the axis.
    assert reading["range_m"][3, 4] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def _noisy(sensor_id="noisy"):
    return dict(id=sensor_id, type="range.ultrasonic", enabled=True, rate_hz=10.0,
                parent="body", pose_parent=dict(pitch_deg=-90.0),
                model=dict(beam_fov_deg=25.0, beam_samples=9, noise_std_m=0.05,
                           quantization_m=0.0, dropout_probability=0.1,
                           min_range_m=0.05, max_range_m=7.0))


def test_the_same_logical_capture_produces_identical_bytes():
    driver = sim_with(build(_noisy()), z=2.0)
    first = driver.sample("noisy", index=7, seed=11)
    second = driver.sample("noisy", index=7, seed=11)
    assert (sensor_models.pack_array(first["range_m"], first["confidence"])
            == sensor_models.pack_array(second["range_m"], second["confidence"]))


def test_a_skipped_capture_does_not_shift_a_later_one():
    driver = sim_with(build(_noisy()), z=2.0)
    reference = driver.sample("noisy", index=9, seed=11)["range_m"]
    for skipped in range(9):
        driver.sample("noisy", index=skipped, seed=11)
    assert np.allclose(driver.sample("noisy", index=9, seed=11)["range_m"],
                       reference, equal_nan=True)


def test_different_sensors_and_epochs_do_not_share_a_noise_stream():
    driver = sim_with(build(_noisy("left"), _noisy("right")), z=2.0)
    left = driver.sample("left", index=3, seed=11)["range_m"]
    right = driver.sample("right", index=3, seed=11)["range_m"]
    after_reset = driver.sample("left", index=3, seed=11, reset_epoch=1)["range_m"]
    assert not np.allclose(left, right, equal_nan=True)
    assert not np.allclose(left, after_reset, equal_nan=True)


def test_the_capture_generator_is_a_frozen_scheme():
    """A test vector, so a change to the seed encoding cannot pass silently."""
    values = sensor_models.capture_rng(1, "front", 0, 0).random(3)
    assert np.allclose(values, [0.56138353, 0.86099201, 0.02213166])
    assert sensor_models.SEED_SCHEME == b"dvision2.sensor-noise.v1"


# ---------------------------------------------------------------------------
# Shared geometry
# ---------------------------------------------------------------------------

def test_the_ray_service_and_the_collision_test_agree_on_a_tree():
    from dsim.dsim import _obstacle_height_m
    tree = SimpleNamespace(symbol="T", kind="tree", x=5.5, y=3.5)
    scene = scene_geometry(SimpleNamespace(objects=[tree]))
    assert scene.tops[0] == _obstacle_height_m("tree")
    # The footprint the ray service intersects is the collision cell exactly.
    forward = np.array([[1.0, 0.0, 0.0]])
    assert cast_rays(scene, (3.0, 3.5, 1.0), forward)[0] == pytest.approx(2.0)
    assert cast_rays(scene, (3.0, 4.05, 1.0), forward)[0] == np.inf


def test_appearance_presets_cannot_move_a_measured_range():
    profile = build(_noisy())
    readings = []
    for preset in ("legacy", "representative"):
        driver = sim_with(profile, [wall(4.5, 3.5)], x=3.0, y=3.0, z=2.0)
        driver.scene_preset = preset
        readings.append(driver.sample("noisy", index=2, seed=5)["range_m"])
    assert np.allclose(*readings, equal_nan=True)
    assert "scene_preset" not in sensor_models.cast.__code__.co_varnames


def test_a_range_sensor_pans_with_its_ptz_mount():
    """The mount chain reaches ray sensors by the same path it reaches cameras."""
    def profile(pan_deg):
        d = default_profile()
        d["mounts"] = [dict(id="head", type="mount.ptz", parent="body",
                            pose_parent=dict(z_m=.2), state=dict(pan_deg=pan_deg))]
        d["sensors"].append(beam("tof", parent="head"))
        return DroneProfile.parse(d)

    obstacles = [wall(4.5, 3.5)]
    # Heading 0 is north (-Y); the wall is due east, so only a 90 degree pan
    # brings it into the beam, and the mount's own 0.2 m riser does not move it.
    assert sim_with(profile(0.0), obstacles, x=3.0, y=3.0, z=1.0).sample("tof")["value_m"] is None
    panned = sim_with(profile(90.0), obstacles, x=3.0, y=3.0, z=1.0)
    assert panned.sample("tof")["value_m"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# What sensors nobody consumes must not touch
# ---------------------------------------------------------------------------

def test_unused_lidar_and_range_sensors_change_nothing_the_camera_sees():
    """The migration acceptance rule: a profile may carry LiDAR and range
    sensors that no client consumes, and neither the primary RGB stream nor
    the fused telemetry a client reads may notice the difference."""
    extra = [
        dict(id="scan", type="lidar.scan2d", parent="body", rate_hz=10.,
             model=dict(samples=180)),
        dict(id="flash", type="lidar.range_image", parent="body", rate_hz=10.,
             model=dict(width_px=32, height_px=24, fov_h_deg=70.)),
        dict(id="sonar", type="range.ultrasonic", parent="body", rate_hz=20.,
             pose_parent=dict(pitch_deg=-90.)),
        dict(id="ir", type="range.infrared", parent="body", rate_hz=20.),
        dict(id="tof", type="range.laser", parent="body", rate_hz=25.),
    ]

    def flown(data):
        trace = []
        with HeadlessSimulator(map_path=CALIBRATION_MAP,
                               drone_profile=DroneProfile.parse(data)) as driver:
            driver.send_body_velocity(0.4, 0.0, 0.0, 20.0)
            for _ in range(4):
                driver.step(0.25)
                fields = {key: value for key, value in driver.read_telemetry().items()
                         if key.startswith(("drone.", "gps.", "est."))}
                trace.append((driver.render().tobytes(), fields))
        return trace

    plain, busy = default_profile(64, 48), default_profile(64, 48)
    busy["sensors"].extend(extra)
    plain_run, busy_run = flown(plain), flown(busy)
    assert [frame for frame, _ in plain_run] == [frame for frame, _ in busy_run], \
        "the primary camera's pixels moved because unused sensors were configured"
    assert [fields for _, fields in plain_run] == [fields for _, fields in busy_run], \
        "fused drone telemetry moved because unused sensors were configured"
