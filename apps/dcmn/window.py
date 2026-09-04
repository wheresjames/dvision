"""Shared window-manager concerns for dvision2 UIs.

Geometry that survives a restart without drifting, and the X input method
every one of these UIs opts out of before it builds a window.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
from pathlib import Path
from typing import Any


_STORE = Path.home() / ".config" / "dvision2" / "window_pos.json"
_LOCK = _STORE.with_suffix(".lock")
_GEOMETRY_RE = re.compile(
    r"^(?P<width>\d+)x(?P<height>\d+)(?P<x>[+-]\d+)(?P<y>[+-]\d+)$")


def disable_input_method() -> None:
    """Keep Tk from negotiating with an X input method as it starts.

    The XIM handshake costs one of these UIs several hundred milliseconds --
    more than the rest of Tk's start-up put together -- and ibus leaks a pair
    of windows per client that it reclaims only when the session ends, so the
    cost of every launch grows with the number of launches before it. None of
    these UIs takes input an X input method exists to compose, and libX11's
    built-in handling remains in place for the keys they do read.

    A desktop session exports XMODIFIERS to everything it starts, so this has
    to overwrite rather than default: deferring to the value already there is
    exactly the cost being avoided. ``DVISION2_INPUT_METHOD`` keeps it, for
    anyone who does need to compose text into one of these windows.
    """
    if os.environ.get("DVISION2_INPUT_METHOD"):
        return
    os.environ["XMODIFIERS"] = "@im=none"


def _read_store() -> dict:
    try:
        value = json.loads(_STORE.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def _valid_geometry(root: Any, geometry: str) -> bool:
    match = _GEOMETRY_RE.fullmatch(geometry)
    if match is None:
        return False
    width, height, x, y = (int(match[name]) for name in
                           ("width", "height", "x", "y"))
    return (1 <= width <= root.winfo_screenwidth()
            and 1 <= height <= root.winfo_screenheight()
            and 0 <= x < root.winfo_screenwidth()
            and 0 <= y < root.winfo_screenheight())


def save_window_geometry(root: Any, key: str) -> None:
    """Persist the WM geometry string, which includes size and outer position.

    Using ``winfo_x/y`` here is subtly wrong on decorated top-level windows:
    some X11 window managers report client coordinates there, then interpret
    ``geometry(+x+y)`` as outer-frame coordinates on restore. The title-bar
    offset is consequently added on every launch. ``wm_geometry`` is symmetric
    with the restore operation and does not drift.
    """
    try:
        root.update_idletasks()
        geometry = str(root.wm_geometry())
        if _GEOMETRY_RE.fullmatch(geometry) is None:
            return
        _STORE.parent.mkdir(parents=True, exist_ok=True)
        with _LOCK.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            data = _read_store()
            data[key] = {"geometry": geometry}
            _STORE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        pass


def restore_window_geometry(root: Any, key: str) -> None:
    """Restore geometry, including compatibility with legacy x/y entries."""
    try:
        value = _read_store().get(key)
        if not isinstance(value, dict):
            return
        geometry = value.get("geometry")
        if isinstance(geometry, str) and _valid_geometry(root, geometry):
            root.wm_geometry(geometry)
            return
        x, y = int(value["x"]), int(value["y"])
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        if not (0 <= x < sw and 0 <= y < sh):
            return
        if "width" in value and "height" in value:
            geometry = f"{int(value['width'])}x{int(value['height'])}+{x}+{y}"
        else:
            geometry = f"+{x}+{y}"
        root.wm_geometry(geometry)
    except Exception:
        pass


# Position-only names remain aliases for older callers. Saving the size as well
# is harmless and gives every migrated UI one consistent behavior.
save_window_pos = save_window_geometry
restore_window_pos = restore_window_geometry
