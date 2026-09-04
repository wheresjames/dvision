"""The dvision2 palette, applied to ttk.

Every window wants the same dark theme, and each had built it by hand: `dsim`,
`dctl` and `dway` carried near-identical thirty-line style blocks that agreed
on the colours and disagreed on the details, so a disabled button was legible
in one window and not in the next. This applies the common part once.

`clam` is the base because it is the only stock theme that honours every
colour option used here. Each interactive state needs its own entry: clam's
untouched defaults are a light grey, unreadable under this palette's near-white
text at exactly the moment a control is hovered, pressed or disabled.

A window that needs more -- `dctl`'s joystick legend, `dway`'s notebook tabs --
configures its own styles on the returned object afterwards. This is the
common base, not a ceiling.
"""

from __future__ import annotations

from tkinter import ttk

from dcmn import theme


def apply_theme(root) -> ttk.Style:
    """Theme ``root`` and return the style object for anything window-specific."""
    root.configure(background=theme.BG)
    style = ttk.Style(root)
    style.theme_use("clam")

    style.configure(".", background=theme.BG, foreground=theme.TEXT,
                    bordercolor=theme.GRID, darkcolor=theme.PANEL,
                    lightcolor=theme.PANEL, troughcolor=theme.PANEL,
                    fieldbackground=theme.ENTRY,
                    selectbackground=theme.ACCENT, selectforeground=theme.TEXT)

    style.configure("TFrame", background=theme.BG)
    style.configure("Header.TFrame", background=theme.PANEL)
    style.configure("Panel.TFrame", background=theme.PANEL)

    style.configure("TLabel", background=theme.BG, foreground=theme.TEXT)
    style.configure("Dim.TLabel", background=theme.BG, foreground=theme.DIM)
    style.configure("Brand.TLabel", background=theme.PANEL, foreground=theme.TEXT,
                    font=("TkDefaultFont", 11, "bold"))
    style.configure("HeaderDim.TLabel", background=theme.PANEL,
                    foreground=theme.DIM)

    style.configure("TButton", background=theme.BUTTON, foreground=theme.TEXT,
                    bordercolor=theme.GRID, lightcolor=theme.GRID,
                    darkcolor=theme.GRID, focuscolor=theme.ACCENT,
                    padding=(10, 6), relief="flat")
    style.map("TButton",
              background=[("disabled", theme.PANEL), ("pressed", theme.PANEL),
                          ("active", theme.BUTTON_ACTIVE)],
              foreground=[("disabled", theme.DIM), ("active", theme.TEXT)],
              bordercolor=[("active", theme.ACCENT)])

    style.configure("TNotebook", background=theme.BG, borderwidth=0)
    style.configure("TNotebook.Tab", background=theme.PANEL, foreground=theme.DIM,
                    bordercolor=theme.GRID, lightcolor=theme.PANEL,
                    darkcolor=theme.PANEL, padding=(12, 6))
    style.map("TNotebook.Tab",
              background=[("selected", theme.BG), ("active", theme.BUTTON_ACTIVE)],
              foreground=[("selected", theme.TEXT), ("active", theme.TEXT)],
              lightcolor=[("selected", theme.BG)])

    style.configure("Vertical.TScrollbar", background=theme.BUTTON,
                    troughcolor=theme.BG, bordercolor=theme.GRID,
                    arrowcolor=theme.DIM, lightcolor=theme.GRID,
                    darkcolor=theme.GRID)
    style.map("Vertical.TScrollbar",
              background=[("pressed", theme.ACCENT),
                          ("active", theme.BUTTON_ACTIVE)],
              arrowcolor=[("active", theme.TEXT)])

    style.configure("TEntry", fieldbackground=theme.ENTRY, foreground=theme.TEXT,
                    bordercolor=theme.GRID, lightcolor=theme.GRID,
                    darkcolor=theme.GRID, insertcolor=theme.TEXT)
    style.map("TEntry", bordercolor=[("focus", theme.ACCENT)])

    style.configure("TCheckbutton", background=theme.BG, foreground=theme.TEXT,
                    focuscolor=theme.ACCENT,
                    indicatorbackground=theme.ENTRY,
                    indicatorforeground=theme.ON_EMPHASIS,
                    bordercolor=theme.GRID, lightcolor=theme.GRID,
                    darkcolor=theme.GRID)
    style.map("TCheckbutton",
              background=[("active", theme.BG)],
              indicatorbackground=[("selected", theme.ACCENT),
                                   ("active", theme.BUTTON_ACTIVE),
                                   ("disabled", theme.PANEL)],
              foreground=[("disabled", theme.DIM)])

    style.configure("TCombobox", fieldbackground=theme.ENTRY,
                    background=theme.BUTTON, foreground=theme.TEXT,
                    arrowcolor=theme.TEXT, bordercolor=theme.GRID,
                    lightcolor=theme.GRID, darkcolor=theme.GRID)
    style.map("TCombobox",
              fieldbackground=[("readonly", theme.ENTRY)],
              foreground=[("disabled", theme.DIM)],
              bordercolor=[("focus", theme.ACCENT)])
    # The dropdown itself is a Tk listbox, not a ttk widget, so it is reached
    # through the option database or it comes up white.
    root.option_add("*TCombobox*Listbox.background", theme.ENTRY)
    root.option_add("*TCombobox*Listbox.foreground", theme.TEXT)
    root.option_add("*TCombobox*Listbox.selectBackground", theme.ACCENT)
    root.option_add("*TCombobox*Listbox.selectForeground", theme.TEXT)

    # A Treeview keeps its own body colours: clam draws the rows and the empty
    # area below them from its own near-white defaults rather than from the "."
    # style above, so an unstyled tree is dark text on white inside an
    # otherwise dark window. `background` paints the rows and `fieldbackground`
    # the space under them -- both are needed, and which one applies varies
    # between Tk builds.
    style.configure("Treeview", background=theme.CANVAS,
                    fieldbackground=theme.CANVAS, foreground=theme.TEXT,
                    bordercolor=theme.GRID, lightcolor=theme.GRID,
                    darkcolor=theme.GRID, borderwidth=1, relief="flat",
                    rowheight=22)
    # Dark text on the selected row: ACCENT is a light blue, so the window's
    # near-white foreground over it lands around 2:1 -- worse than the light
    # grey on white this replaced. Reversing it gives 8:1 and keeps ACCENT as
    # the one selection colour the whole application uses.
    style.map("Treeview",
              background=[("selected", theme.ACCENT)],
              foreground=[("selected", theme.CANVAS)])
    # Column headings are a panel surface, like the window's own header, and
    # clam gives them a raised bevel that reads as a button under this palette.
    style.configure("Treeview.Heading", background=theme.PANEL,
                    foreground=theme.DIM, bordercolor=theme.GRID,
                    lightcolor=theme.PANEL, darkcolor=theme.PANEL,
                    relief="flat", padding=(6, 4))
    style.map("Treeview.Heading",
              background=[("active", theme.BUTTON_ACTIVE)],
              foreground=[("active", theme.TEXT)])

    for name, fill, edge, pressed in (
            ("Accent.TButton", theme.ACCENT_BUTTON, theme.ACCENT_BUTTON_EDGE,
             theme.ACCENT_BUTTON_PRESSED),
            ("Danger.TButton", theme.DANGER_BUTTON, theme.DANGER_BUTTON_EDGE,
             theme.DANGER_BUTTON_PRESSED)):
        style.configure(name, background=fill, foreground=theme.ON_EMPHASIS,
                        bordercolor=edge, lightcolor=edge, darkcolor=fill,
                        padding=(10, 6), relief="flat")
        style.map(name, background=[("disabled", theme.PANEL),
                                    ("pressed", pressed), ("active", edge)],
                  foreground=[("disabled", theme.DIM)])
    return style
