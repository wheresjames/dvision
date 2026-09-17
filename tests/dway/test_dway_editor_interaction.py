"""Pointer interactions for the graphical tour editor."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from dcmn.mapview import MapView
from dtest.dway_rig import MAZE_012, ROOT, Rig
from dvision2_common import load_map
from dway.editor import TourEditor
from dway.mission import MissionState


def editor() -> TourEditor:
    instance = TourEditor.__new__(TourEditor)
    instance.view = MapView(cell=16, margin=10)
    instance.sim_map = SimpleNamespace(width=30, height=20)
    instance.waypoints = [{
        "x": 10.0, "y": 8.0, "z": 1.5,
        "heading_deg": 90.0, "dwell_s": 0.0,
    }]
    instance.selected = None
    instance._drag_mode = None
    instance._refresh = lambda: None
    return instance


def test_clicking_heading_arrow_then_dragging_rotates_without_moving_waypoint() -> None:
    instance = editor()
    cx, cy = instance._to_canvas(10.0, 8.0)
    original_position = (instance.waypoints[0]["x"], instance.waypoints[0]["y"])

    # The initial arrow points east. Grab its tip, then rotate it south.
    instance._on_click(SimpleNamespace(x=cx + 18, y=cy))
    assert instance.selected == 0
    assert instance._drag_mode == "heading"
    instance._on_drag(SimpleNamespace(x=cx, y=cy + 25))
    instance._on_release(SimpleNamespace())

    assert instance.waypoints[0]["heading_deg"] == pytest.approx(180.0)
    assert (instance.waypoints[0]["x"], instance.waypoints[0]["y"]) \
        == original_position
    assert instance._drag_mode is None


@pytest.mark.parametrize(("x_offset", "y_offset", "heading"), (
    (0, -30, 0.0), (30, 0, 90.0), (0, 30, 180.0), (-30, 0, 270.0),
))
def test_heading_drag_uses_compass_cardinal_convention(
        x_offset: float, y_offset: float, heading: float) -> None:
    instance = editor()
    instance.selected = 0
    instance._drag_mode = "heading"
    cx, cy = instance._to_canvas(10.0, 8.0)
    instance._on_drag(SimpleNamespace(x=cx + x_offset, y=cy + y_offset))
    assert instance.waypoints[0]["heading_deg"] == pytest.approx(heading)


def test_clicking_waypoint_marker_still_selects_position_drag() -> None:
    instance = editor()
    cx, cy = instance._to_canvas(10.0, 8.0)
    instance._on_click(SimpleNamespace(x=cx, y=cy))
    assert instance._drag_mode == "move"


# ---------------------------------------------------------------------------
# From the editor to the air
# ---------------------------------------------------------------------------

class _Var:
    """The one thing ``build_tour`` needs from a Tk variable."""

    def __init__(self, value: str = "") -> None:
        self._value = value

    def get(self) -> str:
        return self._value

    def set(self, value) -> None:
        self._value = str(value)


def headless_editor(root_dir: Path, map_path: Path) -> TourEditor:
    """The editor's own logic without its widgets.

    Placing waypoints, validating, saving and reloading are what has to be
    trusted; a Tk canvas is not available in a test run and is not what is
    being asserted.
    """
    instance = TourEditor.__new__(TourEditor)
    instance.root_dir = root_dir
    instance._on_status = None
    instance.status_var = _Var()
    instance.tour_id_var = _Var()
    instance.fields = {"default_speed_mps": _Var("1.0"),
                       "waypoint_tolerance_m": _Var("0.05"),
                       "min_clearance_m": _Var("0.4")}
    instance.view = MapView(cell=16, margin=10)
    instance.canvas = SimpleNamespace(config=lambda **_: None)
    instance.sim_map = load_map(map_path)
    instance.map_path = map_path
    instance.tour_path = None
    instance.waypoints = []
    instance.selected = None
    instance._drag_mode = None
    instance._refresh = lambda: None
    return instance


def test_a_tour_authored_in_the_editor_validates_saves_reloads_and_flies(
        monkeypatch, tmp_path) -> None:
    editor = headless_editor(ROOT, MAZE_012)
    editor.tour_id_var.set("maze_012.editor_authored")
    # Two points on the corridor the deterministic rig starts on, placed by
    # clicking the map exactly as an operator would.
    for x, y in ((35.5, 1.5), (35.5, 4.0)):
        cx, cy = editor._to_canvas(x, y)
        editor._on_click(SimpleNamespace(x=cx, y=cy))
        editor._on_release(SimpleNamespace())
    assert len(editor.waypoints) == 2

    # The editor's own verdict on the geometry, which is what an operator acts
    # on before saving. Leg zero is measured from the map's own start pose, and
    # maze_012 puts that behind a wall from this corridor, so the editor says
    # so -- which is the whole reason it measures leg zero at all.
    assert editor.geometry_diagnostics()["path_length_m"] == pytest.approx(2.5)
    clearances = editor._clearances()
    assert [leg.index for leg in clearances if leg.obstructed] == [0]
    assert clearances[1].clearance_m >= 0.4

    saved = editor._write(tmp_path / "editor_authored.json")
    assert saved is not None and saved.exists()
    assert "saved" in editor.status_var.get()

    # Reloading is the editor's own round trip...
    reopened = headless_editor(ROOT, MAZE_012)
    reopened.load_tour_file(saved)
    assert reopened.tour_id_var.get() == "maze_012.editor_authored"
    assert [(round(p["x"], 3), round(p["y"], 3)) for p in reopened.waypoints] \
        == [(35.5, 1.5), (35.5, 4.0)]

    # ...and flying it is the only check that says the file is a flight plan.
    rig = Rig(monkeypatch, tmp_path, tour_path=saved, finish_action="hold")
    assert rig.fly() is MissionState.COMPLETE, rig.mission.reason
    summary = rig.summary()
    assert summary["tour_id"] == "maze_012.editor_authored"
    assert summary["waypoints_reached"] == 2
