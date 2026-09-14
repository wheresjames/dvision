"""The profile editor: offline, sources-only, and the same form as the running tab."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import dalg.dalg as dalg_app

ROOT = Path(__file__).resolve().parents[1]


def test_offline_editor_needs_no_instance_id() -> None:
    args = dalg_app.parse_args(["--edit"])
    assert args.edit
    assert args.id is None
    assert args.profile is None


def test_offline_editor_accepts_an_initial_profile() -> None:
    args = dalg_app.parse_args(["--edit", "--profile", "sgbm-baseline"])
    assert args.profile == "sgbm-baseline"


@pytest.mark.parametrize("argv", (
    [], ["--profile", "sgbm-baseline"], ["--edit", "--id", "area1"],
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
        lambda *_args, **_kwargs: pytest.fail("offline editor constructed a connected run"))
    assert dalg_app.main(["--edit", "--profile", "sgbm-baseline"]) == 17
    assert opened == {"profile": "sgbm-baseline"}
    assert dalg_app.main(["--edit"]) == 17
    assert opened == {"profile": "new-profile"}


def test_connected_ui_keeps_stepping_behind_its_window(monkeypatch) -> None:
    events = []

    class FakeRun:
        def __init__(self, *_args, **_kwargs):
            self.done = False; self.report_dir = None; self.state = "RUNNING"
            self.shutdown_requested = False
        def step(self): events.append("step")
        def poll_delay(self): return 0.
        def finish(self, partial=False): self.done = True
        def close(self): events.append("close")

    class FakeRoot:
        def destroy(self): events.append("destroy")

    class FakeWindow:
        def __init__(self, _run, *, show_reference=False):
            self.running = True; self.updates = 0; self.root = FakeRoot()
        def update(self):
            self.updates += 1; events.append("update")
            if self.updates == 2: self.running = False
        def save_geometry(self): pass

    monkeypatch.setattr(dalg_app, "DalgRun", FakeRun)
    monkeypatch.setattr(dalg_app, "Window", FakeWindow)
    monkeypatch.setattr(dalg_app.time, "sleep", lambda _delay: None)
    assert dalg_app.main(["--id", "test", "--profile", "sgbm-baseline"]) == 0
    assert events[:4] == ["step", "update", "step", "update"]
    assert events[-2:] == ["close", "destroy"]


def _committed_profiles():
    return sorted(path.stem for path in (ROOT / "assets" / "algorithm_profiles").glob("*.json"))


def _editor(tk_root, profile):
    import tkinter as tk
    from tkinter import ttk
    return dalg_app.build_profile_editor(ttk.Frame(tk_root), profile, tk, ttk)


def _load(name):
    from dalg.profiles import load_profile
    return load_profile(name, ROOT)


@pytest.mark.parametrize("name", _committed_profiles())
def test_every_baseline_opens_and_saves_back_byte_for_byte(name, tmp_path) -> None:
    from dtest.tkfixture import hidden_tk
    with hidden_tk() as tk_root:
        editor = _editor(tk_root, _load(name))
        assert editor.validate(), [editor.rows.item(i, "values") for i in editor.rows.get_children()]
        editor.save(tmp_path / f"{name}.json")
    original = json.loads((ROOT / "assets/algorithm_profiles" / f"{name}.json").read_text())
    assert json.loads((tmp_path / f"{name}.json").read_text()) == original


def test_the_editor_has_no_tour_or_geometry_to_save() -> None:
    from dtest.tkfixture import hidden_tk
    with hidden_tk() as tk_root:
        editor = _editor(tk_root, _load("lidar-baseline"))
        assert not hasattr(editor, "tour") and not hasattr(editor, "geometry_vars")
        assert "not resolved here" in editor.runtime.get()


def test_changing_a_rows_algorithm_starts_from_the_new_algorithms_defaults() -> None:
    from dtest.tkfixture import hidden_tk
    with hidden_tk() as tk_root:
        editor = _editor(tk_root, _load("sgbm-baseline"))
        editor.rows.selection_set("0"); editor._select()
        editor._change_algorithm(0, "plane_sweep")
        assert editor.sources[0].algorithm == "plane_sweep"
        assert editor.sources[0].settings == {}
        assert "num_disparities" not in editor.setting_vars


def test_only_settings_that_differ_from_defaults_are_written(tmp_path) -> None:
    from dtest.tkfixture import hidden_tk
    with hidden_tk() as tk_root:
        editor = _editor(tk_root, _load("lidar-baseline"))
        editor.rows.selection_set("0"); editor._select()
        editor.setting_vars["min_confidence"][0].set("3")
        editor.save(tmp_path / "edited.json")
    saved = json.loads((tmp_path / "edited.json").read_text())
    assert saved["sources"][0]["settings"] == {"min_confidence": 3}


@pytest.mark.parametrize("text,expected", [("true", True), ("False", False), ("0", False)])
def test_a_boolean_setting_parses_as_written_and_not_as_a_nonempty_string(text, expected) -> None:
    """bool("False") is True, which would save every unticked setting as on."""
    from dalg.source_editor import coerce
    assert coerce(bool, text) is expected


def test_the_primary_camera_selector_is_checked_against_the_declared_primary() -> None:
    from dalg.source_editor import row_errors
    from dalg.profiles import Source
    manifest = {"primary_camera": "nose", "sensors": {"nose": {"type": "camera.rgb"}}}
    assert row_errors([Source("primary_camera", "sgbm")], manifest) == {}
    assert "no primary camera" in row_errors([Source("primary_camera", "sgbm")],
                                             {"sensors": {"nose": {"type": "camera.rgb"}}})[0]
    assert "duplicate" in row_errors([Source("primary_camera", "sgbm"), Source("nose", "sgbm")],
                                     manifest)[1]


# -- opening an existing profile ---------------------------------------------

@pytest.fixture
def editor_window():
    """The --edit window on a hidden root, with both dialogs answered by the test."""
    from dtest.tkfixture import hidden_root

    root = hidden_root()
    window = dalg_app.EditorWindow(_load("sgbm-baseline"), root=root)
    asked = []
    window.confirm = lambda *a, **k: asked.append(a) or window.answer
    window.answer = True
    window.asked = asked
    try:
        yield window
    finally:
        root.destroy()


def _root_path(name):
    return ROOT / "assets" / "algorithm_profiles" / f"{name}.json"


def test_the_offline_editor_offers_open_and_the_running_profile_tab_does_not() -> None:
    import tkinter as tk
    from tkinter import ttk
    from dtest.tkfixture import hidden_tk

    with hidden_tk() as tk_root:
        window = dalg_app.EditorWindow(_load("sgbm-baseline"), root=tk_root)
        assert window.editor.open_button is not None
        tab = dalg_app.build_profile_editor(ttk.Frame(tk_root), _load("sgbm-baseline"), tk, ttk)
        assert tab.open_button is None


def test_opening_a_profile_replaces_the_one_being_edited(editor_window) -> None:
    window = editor_window
    assert window.open_profile(_root_path("optical-flow-baseline"))
    editor = window.editor
    assert editor.name.get() == "optical-flow-baseline"
    assert editor.rows.item("0", "values")[0] == "optical_flow_triangulation"
    assert "max_corners" in editor.setting_vars
    assert editor.path == _root_path("optical-flow-baseline").resolve()
    assert editor.notice.get().startswith("Opened optical-flow-baseline.json")
    assert len(window.page.winfo_children()) == 1


def test_a_legacy_profile_will_not_open_and_says_which_field(editor_window, tmp_path) -> None:
    window = editor_window
    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps({"name": "old", "algorithm": "sgbm", "tour": "t.json"}))
    before = window.editor
    assert not window.open_profile(legacy)
    assert window.editor is before
    assert "unsupported field" in before.notice.get() and "tour" in before.notice.get()


def test_a_file_that_will_not_load_leaves_the_current_profile_and_says_why(editor_window, tmp_path) -> None:
    window = editor_window
    broken = tmp_path / "broken.json"
    broken.write_text("{ not json", encoding="utf-8")
    before = window.editor
    assert not window.open_profile(broken)
    assert window.editor is before and before.name.get() == "sgbm-baseline"
    assert before.notice.get().startswith("Not opened: broken.json")


def test_cancelling_the_file_dialog_changes_nothing(editor_window) -> None:
    window = editor_window
    window.choose_file = lambda **kwargs: ""
    before = window.editor
    assert not window.open_profile()
    assert window.editor is before


def test_the_file_dialog_starts_in_the_committed_profiles(editor_window) -> None:
    window = editor_window
    seen = {}
    window.choose_file = lambda **kwargs: seen.update(kwargs) or ""
    window.open_profile()
    assert seen["initialdir"] == dalg_app.profile_dir(dalg_app.ROOT)


def test_unsaved_changes_are_not_thrown_away_without_asking(editor_window) -> None:
    window = editor_window
    assert not window.editor.dirty
    window.open_profile(_root_path("lidar-baseline"))
    assert window.asked == [], "an untouched profile should open without a question"

    window.editor.name.set("my-edits")
    assert window.editor.dirty
    window.answer = False
    assert not window.open_profile(_root_path("optical-flow-baseline"))
    assert window.editor.name.get() == "my-edits", "declining must keep the edits"
    assert len(window.asked) == 1

    window.answer = True
    assert window.open_profile(_root_path("optical-flow-baseline"))
    assert window.editor.name.get() == "optical-flow-baseline"


def test_a_setting_typed_but_not_yet_valid_still_counts_as_an_edit(editor_window) -> None:
    editor = editor_window.editor
    variable, _ = editor.setting_vars["num_disparities"]
    variable.set("not a number")
    assert editor.dirty


def test_saving_makes_the_profile_clean_again(editor_window, tmp_path) -> None:
    editor = editor_window.editor
    editor.name.set("renamed")
    assert editor.dirty
    editor.save(tmp_path / "renamed.json")
    assert not editor.dirty
    assert editor_window.open_profile(_root_path("lidar-baseline"))
    assert editor_window.asked == [], "a saved profile has nothing to discard"
