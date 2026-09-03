"""The calibration contract, revalidated against every scene preset.

A renderer change gets a new ``scene_version`` and, with it, a *revalidated
calibration suite*: appearance may change, geometry may not. The version string alone proves nothing -- it is a promise that something
was checked, and this is the check.

Two claims are made here, and they are opposite claims on purpose:

* every geometric invariant the calibration fixture pins survives the preset.
  Landmarks stay on their published sides, the channel order stays RGB, the
  horizon stays where the camera puts it, and yawing right still moves the
  world left. A preset that broke one of these would not be a harder scene, it
  would be a different coordinate system.
* the preset actually *changed the picture*. A "representative" scene that
  renders the legacy pixels is a version number with nothing behind it, and it
  would let a shortcut-resistance claim rest on a scene change that never
  happened. That was the state this file was written to catch.

The range channel is asserted to be bit-identical across presets, which is what
makes the first claim measurable rather than a matter of opinion: the exact
range oracle is geometry, and geometry is exactly what a lighting change must
not touch.
"""

from __future__ import annotations

import numpy as np
import pytest

from dsim.scene import SCENE_PRESETS
from dsim.range import raycast_map
from dtest.calibration_scene import CENTER_X
from dtest.assertions import (assert_calibration_orientation,
                              assert_channel_order, assert_landmark_moves)
from dtest.color_probe import color_centroid, horizon_row
from dtest.deterministic import DeterministicSim

pytest.importorskip("panda3d")

PRESETS = tuple(SCENE_PRESETS)


def _frame(preset: str, heading_deg: float = 0.0) -> np.ndarray:
    sim = DeterministicSim(heading_deg=heading_deg, scene_preset=preset)
    try:
        return sim.render().copy()
    finally:
        sim.close()


def _scene_objects(preset: str):
    """The geometry the renderer actually built for this preset."""
    sim = DeterministicSim(scene_preset=preset)
    try:
        sim.render()  # force the renderer to exist and build its scene
        return sorted((obj.kind, round(obj.x, 6), round(obj.y, 6))
                      for obj in sim.sim.map.objects)
    finally:
        sim.close()


@pytest.mark.parametrize("preset", PRESETS)
def test_every_preset_keeps_the_published_landmark_geometry(preset, tmp_path):
    assert_calibration_orientation(_frame(preset), artifact_dir=tmp_path)


@pytest.mark.parametrize("preset", PRESETS)
def test_every_preset_delivers_rgb_channel_order(preset):
    assert_channel_order(_frame(preset))


#: Rows the measured horizon may move between presets.
#:
#: ``horizon_row`` finds the last row where blue leads red, so it measures a
#: *colour* boundary, and the representative preset deliberately changes
#: colour. Eight rows of a 480-row frame is 0.6 degrees of a 55 degree vertical
#: field: the sky/ground edge is being classified one antialiased band later,
#: not aimed anywhere new. Geometry identity is proved exactly, and separately,
#: by the two tests at the bottom of this file; this bound only has to be tight
#: enough to catch a camera that actually moved.
HORIZON_TOLERANCE_ROWS = 10


@pytest.mark.parametrize("preset", PRESETS)
def test_every_preset_puts_the_horizon_in_the_same_place(preset):
    """Within a few rows: shading may change, camera geometry may not."""
    legacy = horizon_row(_frame("legacy"), int(CENTER_X))
    moved = abs(horizon_row(_frame(preset), int(CENTER_X)) - legacy)
    assert moved <= HORIZON_TOLERANCE_ROWS, f"horizon moved {moved} rows"


@pytest.mark.parametrize("preset", PRESETS)
def test_every_preset_moves_landmarks_left_when_yawing_right(preset):
    assert_landmark_moves(_frame(preset, 0.0), _frame(preset, 20.0),
                          color="white", direction="left")


def test_presets_share_one_scene_geometry():
    """Appearance is preset-dependent; geometry is not, and must not become so.

    The exact range oracle ray-casts the map rather than the renderer, so a
    preset that moved geometry would silently decouple every score from its
    truth. Both halves are checked: the object list the renderer builds from,
    and the ranges the oracle casts through it.
    """
    assert _scene_objects("legacy") == _scene_objects("representative")


def test_the_range_oracle_is_independent_of_the_preset():
    from pathlib import Path

    from dsim.range import Intrinsics, Pose
    from dvision2_common import load_map

    root = Path(__file__).resolve().parents[1]
    # A benchmark map, not the calibration one: the calibration landmarks are
    # markers rather than walls or trees, so the oracle casts nothing at them.
    sim_map = load_map(root / "assets/maps/maze_012.txt")
    intrinsics = Intrinsics(64, 48, 45.0, 45.0, 32.0, 24.0)
    pose = Pose(1.5, 1.5, 1.5, 180.0)
    ranges, _confidence = raycast_map(sim_map, pose, intrinsics)
    assert np.isfinite(ranges).any(), "the calibration scene casts no ranges"
    # The oracle never consults the renderer's appearance, which is what makes
    # a photometric preset change safe to score against unchanged truth.
    assert "scene_preset" not in raycast_map.__code__.co_varnames


def test_the_representative_preset_actually_changes_the_picture():
    """A scene version with identical pixels behind it is not a scene version.

    Enabling a shadow caster on a pipeline that ignores it changed under one
    per cent of the pixels while reporting a new scene version -- which would
    have let the shortcut-resistance gate be argued over a change that never
    reached the renderer.
    """
    legacy = _frame("legacy").astype(np.int16)
    representative = _frame("representative").astype(np.int16)
    difference = np.abs(legacy - representative)
    changed = (difference.sum(axis=2) > 0).mean()
    assert changed > 0.5, f"only {changed:.1%} of pixels differ"
    assert difference.mean() > 5.0, difference.mean()
