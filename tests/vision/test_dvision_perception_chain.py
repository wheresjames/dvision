"""End-to-end perception chain: rendered pixels -> detector -> occupancy map.

Every link in this chain is unit-tested in isolation, and that is not the same
as testing the chain. A convention can be applied consistently *within* two
stages and still disagree *between* them: the optical-flow detector can be
correct that an obstacle is in its `left` sector while the map is wrong about
where `left` points, and both unit suites pass. The +/-70 degree sector bearing
bug lived exactly there and was found by reading code, not by a failing test.

These tests run the production detector over frames the production renderer
actually produced, feed the result to the production map, and ask where the
obstacle ended up in world coordinates.

The fixtures are mirror images of one another. That pairing is the point: a
handedness error deep in the chain moves both answers the same way, so a test
that only ever sees one side cannot distinguish "correct" from "consistently
mirrored". Mirroring the world must mirror the perception.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from daic.local_map import LocalOccupancyMap, pose_from_status, _OCCUPIED
from daic.optical_flow_avoidance import OpticalFlowAvoidance
from dtest.calibration_scene import (
    CHAIN_FRONT_MAP,
    CHAIN_LEFT_MAP,
    CHAIN_RIGHT_MAP,
)
from dtest.deterministic import DeterministicSim

# The drone starts 5 m short of the side walls and closes at this speed. The
# window below is the approach: late enough that optical flow has a history,
# early enough that the vehicle has not drawn level with the wall and started
# seeing the far side of the arena instead.
APPROACH_SPEED_MPS = 0.8
STEP_S = 0.1
WINDOW = range(6, 14)

# Sector risk margin that counts as "the obstacle is on this side". Measured
# margins are around 0.4, so this leaves better than 2x headroom while still
# being far outside anything symmetric noise produces.
SIDE_MARGIN = 0.15


def _fly_and_perceive(map_path: Path, ticks: int):
    """Fly straight ahead, running the real detector on real rendered frames.

    Returns (per-tick sectors, final map, final pose).
    """
    sim = DeterministicSim(map_path=map_path, heading_deg=0.0)
    detector = OpticalFlowAvoidance()
    local_map = LocalOccupancyMap()
    sim.send_body_velocity(APPROACH_SPEED_MPS, 0.0, 0.0, 0.0)

    sectors = []
    pose = None
    for _ in range(ticks):
        sim.step(STEP_S)
        telemetry = sim.read_telemetry()
        detector.set_motion_from_status(telemetry)
        reading = detector.detect_obstacles(sim.render().copy())
        pose = pose_from_status(telemetry)
        local_map.update(pose, reading)
        sectors.append(reading)
    return sectors, local_map, pose


def _side_margin(sectors) -> float:
    """Mean (left + front_left) - (right + front_right) over the approach."""
    window = [sectors[i] for i in WINDOW]
    left = sum(s.left + s.front_left for s in window) / len(window)
    right = sum(s.right + s.front_right for s in window) / len(window)
    return left - right


# ---------------------------------------------------------------------------
# Rendered obstacle -> detector sectors
# ---------------------------------------------------------------------------

def test_wall_rendered_to_port_raises_the_port_sectors() -> None:
    sectors, _, _ = _fly_and_perceive(CHAIN_LEFT_MAP, max(WINDOW) + 1)

    margin = _side_margin(sectors)
    assert margin > SIDE_MARGIN, (
        f"a wall standing only to the drone's left produced a left-minus-right "
        f"sector margin of {margin:+.2f}; perception is not seeing it on the "
        "side it is actually on")


def test_wall_rendered_to_starboard_raises_the_starboard_sectors() -> None:
    sectors, _, _ = _fly_and_perceive(CHAIN_RIGHT_MAP, max(WINDOW) + 1)

    margin = _side_margin(sectors)
    assert margin < -SIDE_MARGIN, (
        f"a wall standing only to the drone's right produced a left-minus-right "
        f"sector margin of {margin:+.2f}; perception is not seeing it on the "
        "side it is actually on")


def test_mirroring_the_world_mirrors_the_perception() -> None:
    """The pair, compared directly.

    Either test above could pass on its own with a chain that is mirrored but
    self-consistent, if the arena were not perfectly symmetric. Requiring the
    margin to *change sign* between two mirrored scenes cannot be satisfied
    that way.
    """
    left_margin = _side_margin(_fly_and_perceive(CHAIN_LEFT_MAP, max(WINDOW) + 1)[0])
    right_margin = _side_margin(_fly_and_perceive(CHAIN_RIGHT_MAP, max(WINDOW) + 1)[0])

    assert left_margin > 0.0 > right_margin, (
        f"mirrored worlds gave margins {left_margin:+.2f} and {right_margin:+.2f}; "
        "they must have opposite signs")
    assert left_margin - right_margin > 2 * SIDE_MARGIN


# ---------------------------------------------------------------------------
# Rendered obstacle -> detector -> occupancy map
# ---------------------------------------------------------------------------

def test_rendered_wall_ahead_becomes_occupied_cells_ahead() -> None:
    """The link the sector tests cannot reach: bearings becoming map cells.

    Map Y is south-positive and the drone flies north, so every cell the map
    learns from a wall in front of it must have a *smaller* Y than the drone.
    """
    _, local_map, pose = _fly_and_perceive(CHAIN_FRONT_MAP, 30)

    occupied = [cell for cell, value in local_map._cells.items()
                if value >= _OCCUPIED]
    assert occupied, (
        "flying at a rendered wall taught the occupancy map nothing; the rest "
        "of this test would pass vacuously")

    drone_cell = local_map._cell(pose.x, pose.y)
    behind = [cell for cell in occupied if cell[1] >= drone_cell[1]]
    assert not behind, (
        f"{len(behind)} of {len(occupied)} cells learned from a wall dead ahead "
        f"were placed at or behind the drone at {drone_cell}: {sorted(behind)[:5]}")

    # The wall spans the drone's own column, so the marks must straddle it
    # rather than sitting off to one side.
    columns = [cell[0] for cell in occupied]
    assert min(columns) <= drone_cell[0] <= max(columns), (
        f"cells learned from a wall dead ahead span columns "
        f"{min(columns)}..{max(columns)}, which does not include the drone's "
        f"own column {drone_cell[0]}")


def test_occupancy_from_a_rendered_wall_lands_at_a_plausible_range() -> None:
    """A correct bearing at an absurd distance is still a broken chain."""
    _, local_map, pose = _fly_and_perceive(CHAIN_FRONT_MAP, 30)

    occupied = [cell for cell, value in local_map._cells.items()
                if value >= _OCCUPIED]
    assert occupied

    distances = [
        ((local_map._world(cell)[0] - pose.x) ** 2
         + (local_map._world(cell)[1] - pose.y) ** 2) ** 0.5
        for cell in occupied
    ]
    nearest = min(distances)
    assert 0.5 <= nearest <= 8.0, (
        f"nearest cell learned from the rendered wall sits {nearest:.2f} m away; "
        "that is outside any range this scene can produce")


# ---------------------------------------------------------------------------
# Fixture guards
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("map_path", (CHAIN_LEFT_MAP, CHAIN_RIGHT_MAP,
                                      CHAIN_FRONT_MAP))
def test_chain_fixture_produces_confident_perception(map_path: Path) -> None:
    """If the detector stopped reporting, the assertions above go quiet."""
    sectors, _, _ = _fly_and_perceive(map_path, max(WINDOW) + 1)

    confident = [s for s in (sectors[i] for i in WINDOW) if s.confidence > 0.5]
    assert len(confident) >= len(WINDOW) - 1, (
        f"only {len(confident)} of {len(WINDOW)} approach ticks produced a "
        "confident reading; the fixture is no longer exercising perception")
