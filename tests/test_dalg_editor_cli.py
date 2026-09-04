"""The profile editor is usable without simulator IPC."""
from __future__ import annotations

import pytest

import dalg.dalg as dalg_app


def test_offline_editor_needs_no_instance_id() -> None:
    args = dalg_app.parse_args(["--edit"])
    assert args.edit
    assert args.id is None
    assert args.profile == "sgbm-manual"


def test_offline_editor_accepts_an_initial_profile() -> None:
    args = dalg_app.parse_args(["--edit", "--profile", "sgbm-aggressive"])
    assert args.profile == "sgbm-aggressive"


@pytest.mark.parametrize("argv", (
    [], ["--profile", "sgbm-default"], ["--edit", "--id", "area1"],
    ["--edit", "--no-ui"],
))
def test_incomplete_or_incompatible_modes_are_rejected(argv) -> None:
    with pytest.raises(SystemExit): dalg_app.parse_args(argv)


def test_editor_mode_does_not_construct_a_run(monkeypatch) -> None:
    opened = {}

    class FakeEditorWindow:
        def __init__(self, profile): opened["profile"] = profile.name
        def run(self): return 17

    monkeypatch.setattr(dalg_app, "EditorWindow", FakeEditorWindow)
    monkeypatch.setattr(
        dalg_app, "DalgRun",
        lambda *_args: pytest.fail("offline editor constructed a connected run"))
    assert dalg_app.main(["--edit", "--profile", "sgbm-manual"]) == 17
    assert opened == {"profile": "sgbm-manual"}


def test_connected_ui_stays_open_after_run_completion(monkeypatch) -> None:
    events = []

    class FakeRun:
        def __init__(self, *_args):
            self.done = False; self.active = True; self.report_dir = None
            self.reason = ""; self.provenance = {}
            self.shutdown_requested = False
        def step(self):
            events.append("step"); self.done = True
            self.provenance["coordinator_outcome"] = "complete"
        def close(self): events.append("close")

    class FakeRoot:
        def destroy(self): events.append("destroy")

    class FakeWindow:
        def __init__(self, _run):
            self.running = True; self.updates = 0; self.root = FakeRoot()
        def update(self):
            self.updates += 1; events.append("update")
            if self.updates == 2: self.running = False
        def save_geometry(self): pass

    monkeypatch.setattr(dalg_app, "DalgRun", FakeRun)
    monkeypatch.setattr(dalg_app, "Window", FakeWindow)
    monkeypatch.setattr(dalg_app.time, "sleep", lambda _delay: None)
    assert dalg_app.main(["--id", "test", "--profile", "sgbm-default"]) == 0
    # The window stays open on the result, and the run keeps being stepped
    # behind it: step() is what drains the bus and publishes presence, so a
    # window that outlives the tour has to keep getting one. This previously
    # read ["step", "update", "update"] -- one step and then silence, which is
    # how a finished dalg came to ignore system.shutdown.
    assert events[:4] == ["step", "update", "step", "update"]
    assert events[-2:] == ["close", "destroy"]
