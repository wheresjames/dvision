"""The test harness's own invariants.

A suite that flashes windows across the developer's screen is a suite people
stop running locally, and one that needs a display is one that cannot run in
CI. Widget tests are worth having -- geometry and event dispatch are exactly
what a mock gets wrong -- so the rule is that they run on a hidden root.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TEST_DIRS = (ROOT / "tests", ROOT / "dtest")
#: The helper is the only place allowed to build one.
HELPER = ROOT / "dtest/tkfixture.py"


def test_a_hidden_root_is_never_on_screen() -> None:
    from dtest.tkfixture import hidden_tk

    with hidden_tk() as root:
        root.update()
        assert not root.winfo_viewable(), "the root was mapped"
        # It is still a real root: geometry and styles work, which is the
        # entire reason for testing against widgets rather than mocks.
        assert root.winfo_screenwidth() > 0
        assert root.tk.call("info", "commands", "ttk::style")


def test_the_root_is_destroyed_on_the_way_out() -> None:
    """Leaked roots keep an event loop and an X connection alive for the run."""
    import tkinter

    from dtest.tkfixture import hidden_tk

    with hidden_tk() as root:
        pass
    # Every call into a destroyed interpreter raises, which is the proof.
    with pytest.raises(tkinter.TclError):
        root.winfo_exists()


def _bare_tk_calls(path: Path) -> list[int]:
    """Lines where this file calls ``Tk()`` directly."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [node.lineno for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and ast.unparse(node.func).split(".")[-1] == "Tk"]


@pytest.mark.parametrize("path", sorted(
    p for directory in TEST_DIRS for p in directory.glob("*.py")
    if p != HELPER), ids=lambda p: p.name)
def test_no_test_module_builds_its_own_tk_root(path: Path) -> None:
    """``Tk()`` maps a viewable window; ``hidden_root()`` is the way in."""
    lines = _bare_tk_calls(path)
    assert lines == [], (
        f"{path.name} calls Tk() at line(s) {lines}; use "
        f"dtest.tkfixture.hidden_root() so the suite stays off the screen")
