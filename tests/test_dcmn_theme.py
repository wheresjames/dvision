"""One palette, applied one way.

`dsim`, `dctl` and `dway` each carried their own copy of the same dark theme
and their own colour literals. These tests are the guard against a copy growing
back: they assert the palette is reachable from one place and that no window
module spells a colour out by hand.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from dcmn import theme

ROOT = Path(__file__).resolve().parents[1]
WINDOW_MODULES = ("apps/dsim/dsim.py", "apps/dctl/dctl.py", "apps/dway/dway.py",
                  "apps/dway/editor.py", "apps/dway/report.py")
HEX_COLOUR = re.compile(r'"#[0-9a-fA-F]{3,8}"')


@pytest.mark.parametrize("module", WINDOW_MODULES)
def test_no_window_module_spells_a_colour_out_by_hand(module: str) -> None:
    found = HEX_COLOUR.findall((ROOT / module).read_text(encoding="utf-8"))
    assert found == [], f"{module} carries its own palette: {sorted(set(found))}"


def test_every_palette_entry_is_a_colour() -> None:
    names = [name for name in dir(theme) if name.isupper()]
    assert len(names) > 20
    for name in names:
        value = getattr(theme, name)
        assert HEX_COLOUR.fullmatch(f'"{value}"'), f"{name} = {value!r}"


def test_the_map_grid_is_quieter_than_a_widget_border() -> None:
    """The cell grid is a hint of one, not a lattice drawn over the map."""
    def luminance(colour: str) -> int:
        return sum(int(colour[i:i + 2], 16) for i in (1, 3, 5))

    assert luminance(theme.MAP_GRID) < luminance(theme.GRID)
    assert luminance(theme.CELL) < luminance(theme.MAP_GRID)


def test_the_shared_theme_configures_the_styles_windows_rely_on() -> None:
    from dcmn.tktheme import apply_theme
    from dtest.tkfixture import hidden_tk

    with hidden_tk() as root:
        style = apply_theme(root)
        assert style.theme_use() == "clam"
        for name in ("TFrame", "Header.TFrame", "Panel.TFrame", "TLabel",
                     "Dim.TLabel", "Brand.TLabel", "HeaderDim.TLabel",
                     "TButton", "Accent.TButton", "Danger.TButton"):
            assert style.configure(name) is not None, name
        assert style.lookup("TButton", "background") == theme.BUTTON
        assert style.lookup("TLabel", "foreground") == theme.TEXT
        # A disabled control has to stay readable; clam's default grey is not.
        assert style.lookup("TButton", "foreground",
                            ["disabled"]) == theme.DIM


# ---------------------------------------------------------------------------
# Repaint pacing
# ---------------------------------------------------------------------------

def test_a_paced_surface_paints_immediately_then_holds_its_rate():
    """A window that waited a frame before showing anything looks broken."""
    from dcmn.pacing import Paced

    now = [100.0]
    paced = Paced(10.0, clock=lambda: now[0])

    assert paced.due()                    # the first call always paints
    assert not paced.due()                # nothing has moved
    now[0] += 0.099
    assert not paced.due()
    now[0] += 0.002                       # 0.101 s, past the 0.1 s period
    assert paced.due()


def test_paced_counts_paints_not_calls():
    from dcmn.pacing import Paced

    now = [0.0]
    paced = Paced(4.0, clock=lambda: now[0])
    painted = 0
    for _ in range(1000):                 # ten seconds of 100 Hz ticking
        now[0] += 0.01
        painted += bool(paced.due())

    # Never more than the cap allows over that span, and actually near it.
    assert painted <= 10 * 4 + 1
    assert painted >= 10 * 4 - 2


def test_paced_rejects_a_rate_that_would_never_paint():
    from dcmn.pacing import Paced
    with pytest.raises(ValueError):
        Paced(0.0)


def test_the_caps_are_ordered_video_then_map_then_text():
    """Video is the smoothest surface and text the least; nobody reads at 30 Hz."""
    from dcmn.pacing import MAP_HZ, TEXT_HZ, VIDEO_HZ

    assert VIDEO_HZ == 30.0 and MAP_HZ == 10.0 and TEXT_HZ == 4.0
    assert VIDEO_HZ > MAP_HZ > TEXT_HZ


@pytest.mark.parametrize("module,surfaces", [
    ("apps/dctl/dctl.py", ("_paint_video", "_paint_text")),
    ("apps/daic/daic.py", ("_paint_video", "_paint_map", "_paint_text")),
    ("apps/dway/dway.py", ("_paint_map", "_paint_text")),
    ("apps/dalg/dalg.py", ("_paint_video", "_paint_text")),
])
def test_every_window_paces_its_painting(module, surfaces):
    source = (ROOT / module).read_text(encoding="utf-8")
    assert "from dcmn.pacing import" in source, f"{module} keeps its own caps"
    for surface in surfaces:
        assert f"{surface}.due()" in source, f"{module} never checks {surface}"


def _guarded_by_a_paint_cap(body: str) -> set[str]:
    """Every statement that only runs when a repaint budget allows it."""
    guarded: set[str] = set()
    guard_indent: int | None = None
    for line in body.splitlines():
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        if guard_indent is not None and indent <= guard_indent:
            guard_indent = None
        if ("_paint_" in line and ".due()" in line
                and line.rstrip().endswith(":")):
            guard_indent = indent
            continue
        if guard_indent is not None:
            guarded.add(line.strip())
    return guarded


@pytest.mark.parametrize("module,anchor,end,control", [
    ("apps/dctl/dctl.py", "    def tick(self) -> None:", "self.root.after",
     ("self.send_held_velocity()", "self._maintain_control()")),
    ("apps/daic/daic.py", "    def tick(self) -> None:", "_schedule_next_tick",
     ("self._run_ai(now)", "self._maintain_control(now)")),
    ("apps/dway/dway.py", "    def tick(self) -> None:", "self.root.after",
     ("self.flight.step()",)),
])
def test_control_paths_are_not_behind_a_repaint_cap(module, anchor, end, control):
    """Capping the timer that also flies the vehicle was the hazard here.

    `dctl.tick` sends held velocity and `daic.tick` runs the planner; each
    shared one timer with its redraw, so a frame-rate cap would have been a
    control-rate cap. The calls that command the vehicle must sit outside
    every `due()` guard.
    """
    source = (ROOT / module).read_text(encoding="utf-8")
    body = source[source.index(anchor):]
    body = body[:body.index(end)]
    guarded = _guarded_by_a_paint_cap(body)

    for call in control:
        assert call in body, f"{module}: {call} is not in the tick any more"
        assert call not in guarded, (
            f"{module}: {call} only runs when a repaint budget allows it")


# ---------------------------------------------------------------------------
# Contrast
# ---------------------------------------------------------------------------

def _relative_luminance(colour: str) -> float:
    """WCAG relative luminance of a #rrggbb colour."""
    raw = colour.lstrip("#")
    channels = [int(raw[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]
    linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
              for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _contrast(foreground: str, background: str) -> float:
    a, b = _relative_luminance(foreground), _relative_luminance(background)
    return (max(a, b) + 0.05) / (min(a, b) + 0.05)


def test_the_contrast_helper_agrees_with_known_values():
    assert _contrast("#ffffff", "#000000") == pytest.approx(21.0, abs=0.01)
    assert _contrast("#ffffff", "#ffffff") == pytest.approx(1.0, abs=0.01)


def test_a_treeview_is_readable_against_this_palette():
    """clam draws a Treeview's rows from its own near-white defaults.

    The "." style does not reach them, so an unstyled tree was light grey on
    white inside an otherwise dark window -- 3.08:1, under the 4.5:1 that
    counts as readable, and the worst contrast anywhere in the application.
    """
    from tkinter import ttk
    from dcmn.tktheme import apply_theme
    from dtest.tkfixture import hidden_tk

    with hidden_tk() as root:
        style = apply_theme(root)
        ttk.Treeview(root, columns=("a",), show="headings")

    body_bg = style.lookup("Treeview", "background")
    field_bg = style.lookup("Treeview", "fieldbackground")
    body_fg = style.lookup("Treeview", "foreground")
    heading_bg = style.lookup("Treeview.Heading", "background")
    heading_fg = style.lookup("Treeview.Heading", "foreground")
    selected_bg = dict(style.map("Treeview", "background"))["selected"]
    selected_fg = dict(style.map("Treeview", "foreground"))["selected"]

    # The rows and the empty area under them are both painted; which of the
    # two options applies varies between Tk builds, so neither may be light.
    for surface in (body_bg, field_bg):
        assert _relative_luminance(surface) < 0.05, f"{surface} is not a dark background"

    assert _contrast(body_fg, body_bg) >= 4.5
    assert _contrast(heading_fg, heading_bg) >= 4.5
    # ACCENT is a light blue, so a selected row needs dark text over it --
    # the window's near-white foreground lands near 2:1 there.
    assert _contrast(selected_fg, selected_bg) >= 4.5


def test_the_treeview_style_comes_from_the_shared_palette():
    """No window keeps its own idea of what a list looks like."""
    from dcmn import theme
    from dcmn.tktheme import apply_theme
    from dtest.tkfixture import hidden_tk

    with hidden_tk() as root:
        style = apply_theme(root)
    palette = {value for name, value in vars(theme).items()
               if isinstance(value, str) and value.startswith("#")}

    for widget, option in (("Treeview", "background"),
                           ("Treeview", "fieldbackground"),
                           ("Treeview", "foreground"),
                           ("Treeview.Heading", "background"),
                           ("Treeview.Heading", "foreground")):
        assert style.lookup(widget, option) in palette, f"{widget} {option}"
