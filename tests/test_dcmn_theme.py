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
