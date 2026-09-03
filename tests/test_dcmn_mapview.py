"""The shared map: one description of the world, several surfaces to paint it.

These are the guard against a fifth private copy of "how a map looks" growing
back. They assert the geometry and the palette come from one place, not that
any particular pixel is any particular colour.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from dcmn import theme
from dcmn.mapview import MapView, draw_map_axes, map_shapes
from dvision2_common import load_map

ROOT = Path(__file__).resolve().parents[1]
MAZE_012 = ROOT / "assets/maps/maze_012.txt"


def fake_map(*kinds: str):
    return SimpleNamespace(
        width=4, height=3,
        objects=[SimpleNamespace(kind=kind, x=index + 0.5, y=0.5)
                 for index, kind in enumerate(kinds)])


class RecordingCanvas:
    """Just enough Tk canvas to record what was asked for."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def __getattr__(self, name: str):
        def record(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            return len(self.calls)
        return record


def test_the_transform_round_trips_exactly() -> None:
    view = MapView(cell=16, margin=10)
    assert view.xy(0.0, 0.0) == (10.0, 10.0)
    assert view.xy(2.0, 3.0) == (42.0, 58.0)
    assert view.to_map(*view.xy(7.25, 1.75)) == pytest.approx((7.25, 1.75))


def test_a_map_is_fitted_to_the_space_it_is_given() -> None:
    sim_map = load_map(MAZE_012)
    view = MapView.fitted(sim_map, max_edge_px=720, max_cell_px=42)
    assert view.cell == max(18, min(42, 720 // max(sim_map.width, sim_map.height)))
    width, height = view.canvas_size(sim_map)
    assert width == sim_map.width * view.cell + view.margin * 2
    assert height == sim_map.height * view.cell + view.margin * 2


def test_every_object_kind_is_drawn_including_ones_nobody_named() -> None:
    shapes = list(map_shapes(fake_map("wall", "tree", "target", "sculpture"),
                             grid=False))
    assert [s.fill for s in shapes if s.kind == "rect"] == [
        theme.WALL_FILL, theme.OBJECT_FILL]
    assert [s.fill for s in shapes if s.kind == "oval"] == [
        theme.TREE_FILL, theme.TARGET_FILL]
    # The target's crosshair, which the report used to omit along with the
    # target itself.
    assert sum(1 for s in shapes if s.kind == "line") == 2


def test_the_grid_covers_every_cell_and_can_be_turned_off() -> None:
    sim_map = fake_map()
    assert len(list(map_shapes(sim_map))) == sim_map.width * sim_map.height
    assert list(map_shapes(sim_map, grid=False)) == []


def test_both_backends_consume_the_same_shapes() -> None:
    """A window and a report draw one description of the world, not two."""
    matplotlib = pytest.importorskip("matplotlib")
    from matplotlib.figure import Figure

    sim_map = load_map(MAZE_012)
    shapes = list(map_shapes(sim_map))

    canvas = RecordingCanvas()
    MapView(cell=10, margin=0).draw_map(canvas, sim_map)
    assert len(canvas.calls) == len(shapes)

    axes = Figure().add_subplot(111)
    before = len(axes.patches) + len(axes.lines)
    draw_map_axes(axes, sim_map)
    assert len(axes.patches) + len(axes.lines) - before == len(shapes)


def test_the_drone_points_where_the_compass_says(monkeypatch) -> None:
    """North is up the canvas and east is to the right, at every quarter turn."""
    view = MapView(cell=20, margin=0)
    for heading, (dx, dy) in ((0.0, (0.0, -1.0)), (90.0, (1.0, 0.0)),
                              (180.0, (0.0, 1.0)), (270.0, (-1.0, 0.0))):
        canvas = RecordingCanvas()
        view.draw_drone(canvas, 5.0, 5.0, heading, cone=False)
        # The body polygon's first point is the nose.
        _, args, _ = canvas.calls[0]
        nose_x, nose_y = args[0], args[1]
        size = view.cell * 0.34
        assert (nose_x - 100.0, nose_y - 100.0) == pytest.approx(
            (dx * size, dy * size), abs=1e-9)


def test_a_crashed_vehicle_is_drawn_in_the_danger_colour() -> None:
    view = MapView(cell=20, margin=0)
    for crashed, expected in ((False, theme.ACCENT), (True, theme.DANGER)):
        canvas = RecordingCanvas()
        view.draw_drone(canvas, 1.0, 1.0, 0.0, crashed=crashed, cone=False)
        assert canvas.calls[0][2]["fill"] == expected
