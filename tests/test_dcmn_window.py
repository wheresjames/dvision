from __future__ import annotations

import json
import os

from dcmn import window


class FakeWindow:
    def __init__(self, geometry="800x500+120+90"):
        self.current = geometry
        self.applied = []

    def update_idletasks(self): pass
    def wm_geometry(self, value=None):
        if value is None:
            return self.current
        self.current = value
        self.applied.append(value)

    def winfo_screenwidth(self): return 1920
    def winfo_screenheight(self): return 1080

    # Deliberately different client coordinates. These must never be saved.
    def winfo_x(self): return 128
    def winfo_y(self): return 121


def test_geometry_round_trip_uses_symmetric_wm_coordinates(tmp_path, monkeypatch):
    store = tmp_path / "window_pos.json"
    monkeypatch.setattr(window, "_STORE", store)
    monkeypatch.setattr(window, "_LOCK", tmp_path / "window_pos.lock")
    root = FakeWindow()

    window.save_window_geometry(root, "dalg.area1")
    assert json.loads(store.read_text())["dalg.area1"] == {
        "geometry": "800x500+120+90"}

    restored = FakeWindow("1x1+0+0")
    window.restore_window_geometry(restored, "dalg.area1")
    assert restored.applied == ["800x500+120+90"]


def test_legacy_position_entry_still_restores(tmp_path, monkeypatch):
    store = tmp_path / "window_pos.json"
    store.write_text(json.dumps({"dsim.area1": {"x": 40, "y": 50}}))
    monkeypatch.setattr(window, "_STORE", store)
    root = FakeWindow()
    window.restore_window_geometry(root, "dsim.area1")
    assert root.applied == ["+40+50"]


# ---------------------------------------------------------------------------
# Input method
# ---------------------------------------------------------------------------

def test_input_method_is_declined_before_a_window_is_built(monkeypatch):
    """Every UI here opts out: the XIM handshake dominates their start-up."""
    monkeypatch.delenv("XMODIFIERS", raising=False)
    monkeypatch.delenv("DVISION2_INPUT_METHOD", raising=False)
    window.disable_input_method()
    assert os.environ["XMODIFIERS"] == "@im=none"


def test_the_session_wide_input_method_is_overridden(monkeypatch):
    """A desktop exports XMODIFIERS to everything, so deferring to it declines
    nothing -- the value already there is the one that costs the time."""
    monkeypatch.delenv("DVISION2_INPUT_METHOD", raising=False)
    monkeypatch.setenv("XMODIFIERS", "@im=ibus")
    window.disable_input_method()
    assert os.environ["XMODIFIERS"] == "@im=none"


def test_an_input_method_can_still_be_asked_for(monkeypatch):
    monkeypatch.setenv("DVISION2_INPUT_METHOD", "1")
    monkeypatch.setenv("XMODIFIERS", "@im=ibus")
    window.disable_input_method()
    assert os.environ["XMODIFIERS"] == "@im=ibus"


def test_every_tk_front_end_declines_before_it_reaches_tk(monkeypatch):
    """The call has to happen in main(), not beside the Tk root it protects.

    Tk reads XMODIFIERS as it creates a root, so a front end that set it later
    -- or not at all -- would go on paying for the handshake and leaking the
    windows it opens.
    """
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for module in ("dsim/dsim.py", "dctl/dctl.py", "dway/dway.py",
                   "dalg/dalg.py", "daic/daic.py", "dfgb/dfgb.py"):
        tree = ast.parse((root / module).read_text(encoding="utf-8"))
        main = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "main")
        first = main.body[0]
        called = (isinstance(first, ast.Expr) and isinstance(first.value, ast.Call)
                  and getattr(first.value.func, "id", "") == "disable_input_method")
        assert called, f"{module}: main() must decline the input method first"
