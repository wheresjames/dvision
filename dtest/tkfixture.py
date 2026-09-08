"""A Tk root for tests that does not put a window on anyone's screen.

Widget code deserves to be tested against real widgets -- geometry, style
lookups and event dispatch are exactly the parts that a mock would get wrong.
But a bare ``Tk()`` maps a viewable toplevel, so a suite that builds a few of
them flashes windows across the developer's desktop, and on a machine with no
display it fails instead of skipping.

``hidden_root()`` is the only way a test should make one: withdrawn before
anything is realised, and skipped rather than errored when there is no display
to withdraw it from.

A few tests need more than a hidden root can give them -- they must open a
*mapped* window -- and those are opt-in through :func:`mapped_root_available`.
"""

from __future__ import annotations

import os
from contextlib import contextmanager

import pytest

from dcmn.window import disable_input_method


def mapped_root_available() -> bool:
    """Whether a test may build a real, mapped toplevel.

    The device-browser tests need a mapped window: freeze selects panes by
    ``winfo_ismapped()``, pop-out is real window-manager management, and
    synthesized key events are delivered to the toplevel holding keyboard
    focus. None of that works on a withdrawn root -- but a mapped window
    flashes on the operator's desktop, so it is opt-in: the widget halves run
    when a display is present **and** ``DVISION2_GUI_TESTS`` is set, and every
    test falls back to its headless half otherwise, exactly as on a build
    machine.
    """
    return bool(os.environ.get("DISPLAY")) and bool(os.environ.get("DVISION2_GUI_TESTS"))


def hidden_root():
    """A withdrawn ``Tk`` root, or a skip when no display is available.

    The caller destroys it. Prefer :func:`hidden_tk`, which does that for you.
    """
    tkinter = pytest.importorskip("tkinter")
    disable_input_method()
    try:
        root = tkinter.Tk()
    except tkinter.TclError as exc:
        pytest.skip(f"no display: {exc}")
    # Before any update, so the window is never mapped even briefly.
    root.withdraw()
    return root


@contextmanager
def hidden_tk():
    """``with hidden_tk() as root:`` -- withdrawn, and destroyed on the way out."""
    root = hidden_root()
    try:
        yield root
    finally:
        try:
            root.destroy()
        except Exception:
            pass
