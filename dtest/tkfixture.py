"""A Tk root for tests that does not put a window on anyone's screen.

Widget code deserves to be tested against real widgets -- geometry, style
lookups and event dispatch are exactly the parts that a mock would get wrong.
But a bare ``Tk()`` maps a viewable toplevel, so a suite that builds a few of
them flashes windows across the developer's desktop, and on a machine with no
display it fails instead of skipping.

``hidden_root()`` is the only way a test should make one: withdrawn before
anything is realised, and skipped rather than errored when there is no display
to withdraw it from.
"""

from __future__ import annotations

from contextlib import contextmanager

import pytest

from dcmn.window import disable_input_method


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
