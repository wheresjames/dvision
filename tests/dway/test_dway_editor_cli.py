"""The tour editor can start without a simulator or vehicle identity."""

from __future__ import annotations

import pytest

import dway.dway as dway_app


def test_offline_editor_needs_neither_id_nor_tour() -> None:
    args = dway_app.parse_args(["--edit"])
    assert args.edit
    assert args.id is None
    assert args.tour is None


def test_offline_editor_accepts_initial_map_and_tour() -> None:
    args = dway_app.parse_args([
        "--edit", "--map", "assets/maps/maze_012.txt",
        "--tour", "assets/tours/maze_012.forward.v1.json"])
    assert args.edit_map == "assets/maps/maze_012.txt"
    assert args.tour == "assets/tours/maze_012.forward.v1.json"


@pytest.mark.parametrize("argv", (
    [], ["--tour", "tour.json"], ["--id", "vehicle"],
    ["--edit", "--no-ui"], ["--edit", "--id", "vehicle"],
    ["--id", "vehicle", "--tour", "tour.json", "--map", "map.txt"],
))
def test_incompatible_or_incomplete_modes_are_rejected(argv) -> None:
    with pytest.raises(SystemExit):
        dway_app.parse_args(argv)


def test_editor_mode_does_not_construct_a_flight(monkeypatch) -> None:
    opened = {}

    class FakeEditorWindow:
        def __init__(self, *, map_path=None, tour_path=None):
            opened.update(map_path=map_path, tour_path=tour_path)

        def run(self):
            return 17

    monkeypatch.setattr(dway_app, "EditorWindow", FakeEditorWindow)
    monkeypatch.setattr(
        dway_app, "Flight",
        lambda _args: pytest.fail("offline editor constructed a vehicle flight"))
    result = dway_app.main([
        "--edit", "--map", "map.txt", "--tour", "tour.json"])
    assert result == 17
    assert opened == {"map_path": "map.txt", "tour_path": "tour.json"}
