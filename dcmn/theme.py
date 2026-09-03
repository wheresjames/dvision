"""The one dvision2 palette.

Every window drew from the same GitHub-dark colours and each kept its own copy,
which is how `dsim`'s map came to look nothing like `dway`'s: the values did
not drift, but which colour was used for a wall did. Defining them once makes
that kind of divergence a visible edit rather than an accident.

Deliberately constants only -- no tkinter import -- so a headless module can
read a colour without pulling in a display library.
"""

from __future__ import annotations

BG = "#0d1117"             # window and frame background
PANEL = "#161b22"          # headers, footers, raised panels
CANVAS = "#010409"         # canvas ground, darker than the window
CELL = "#161b22"           # one empty map cell
GRID = "#30363d"           # widget borders, plot spines
MAP_GRID = "#21262d"       # cell edges: a hint of one, not a drawn lattice
TEXT = "#e6edf3"           # primary text
DIM = "#8b949e"            # secondary text
ACCENT = "#58a6ff"         # focus, selection, the vehicle
BUTTON = "#21262d"
BUTTON_ACTIVE = "#30363d"
ENTRY = "#21262d"          # entry and spinbox fields

# Emphasis buttons. The primary action in a window, and the one that ends a
# flight; both keep white text, so they carry their own foreground.
ACCENT_BUTTON = "#1f6feb"
ACCENT_BUTTON_EDGE = "#388bfd"
ACCENT_BUTTON_PRESSED = "#1158c7"
DANGER_BUTTON = "#da3633"
DANGER_BUTTON_EDGE = "#f85149"
DANGER_BUTTON_PRESSED = "#b62324"
ON_EMPHASIS = "#ffffff"
WARN = "#f2cc60"           # planned route, caution
CAUTION = "#d29922"        # inside tolerance but tighter than asked for
DANGER = "#f85149"         # failure, crash
OK = "#3fb950"             # healthy, complete

# Map objects. These are what made the two windows look like different
# programs, so they live here rather than in whichever module drew last.
WALL_FILL = "#6e7681"
WALL_EDGE = "#8b949e"
TREE_FILL = "#238636"
TREE_EDGE = "#3fb950"
TARGET_FILL = "#da3633"
TARGET_EDGE = "#f85149"
TARGET_CROSS = "#ffffff"
OBJECT_FILL = "#30363d"    # an object kind no window knows about
DRONE_EDGE = "#c9d1d9"
DRONE_HUB = "#010409"
