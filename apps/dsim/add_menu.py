"""The Add dropdown, drawn by hand.

The list of things that can be added is long and getting longer, and a
classic Tk menu cannot present it well: an entry is one line in one font, a
highlight covers one line at a time, and there is no scrolling. So the
dropdown is a borderless popup with a canvas of titled items -- a name, and
a small dim description beneath it -- where a hover highlights the item as
one thing, and the wheel and a scrollbar reach the rest when the list
outgrows the screen.
"""

import tkinter as tk
from tkinter import font as tkfont
from tkinter import ttk

from dcmn import theme
from dsim.scroll import wheel_step

#: One item is a name line with a description line beneath it.
_NAME_Y = 5
_DESC_Y = 20
_ITEM_H = 34
#: A hairline between the sensor types and the mount types.
_SEPARATOR_H = 9
#: Never open taller than this many rows; the rest scroll.
_MAX_ROWS = 12
_PAD_X = 10
_DESC_PAD_X = 8
#: Room reserved so the popup's width does not jump when the bar appears.
_BAR_RESERVE_PX = 14


class AddMenu:
    """A popup list of ``(kind, description, group)`` items under a button.

    An item's name and its description highlight as one row, picking one
    hands its kind to ``on_pick`` and closes, and clicking anywhere else --
    or pressing Escape -- just closes. A change of ``group`` draws a hairline
    between the items.
    """

    def __init__(self, button, items, on_pick):
        self.button = button
        self.items = list(items)
        self.on_pick = on_pick
        self.is_open = False
        self._highlighted = None
        self._rows = []
        self._width = 0
        self._height = 0
        self._build()

    # -- construction ---------------------------------------------------

    def _build(self):
        self._name_font = tkfont.nametofont("TkDefaultFont")
        self._desc_font = self._name_font.copy()
        self._desc_font.configure(size=8)
        # The popup floats over the window, so it needs an edge of its own:
        # a one-pixel frame in the grid colour around the panel surface --
        # and no window-manager decorations; the title bar it would get is
        # the one thing a dropdown must not have.
        self._top = tk.Toplevel(self.button)
        self._top.overrideredirect(True)
        self._top.configure(background=theme.GRID)
        self._top.withdraw()
        self._top.rowconfigure(0, weight=1)
        self._top.columnconfigure(0, weight=1)
        self._canvas = tk.Canvas(self._top, borderwidth=0,
                                 highlightthickness=0, background=theme.PANEL)
        self._canvas.grid(row=0, column=0, padx=1, pady=1, sticky="nsew")
        self._bar = ttk.Scrollbar(self._top, orient="vertical",
                                  command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=self._on_scrolled)
        self._draw()

        self._canvas.bind("<Motion>", self._on_motion)
        self._canvas.bind("<Leave>", lambda _event: self._highlight(None))
        self._canvas.bind("<Button-1>", self._on_press)
        for sequence in ("<Button-4>", "<Button-5>", "<MouseWheel>"):
            self._canvas.bind(sequence, self._on_wheel)
        # A global grab routes presses made anywhere else to this popup,
        # which is what lets a click outside dismiss it.
        self._top.bind("<Button-1>", lambda _event: self.close())
        self._top.bind("<Escape>", lambda _event: self.close())

    def _draw(self):
        """Lay the items out once; only the colours move after this."""
        canvas = self._canvas
        width = max(self._name_font.measure(kind)
                    for kind, _description, _group in self.items)
        width = max(width, max(self._desc_font.measure(description)
                               for _kind, description, _group in self.items)
                    + _DESC_PAD_X)
        width += 2 * _PAD_X + _BAR_RESERVE_PX

        y, previous_group = 0, None
        for kind, description, group in self.items:
            if previous_group is not None and group != previous_group:
                canvas.create_line(0, y + _SEPARATOR_H // 2, width,
                                   y + _SEPARATOR_H // 2, fill=theme.GRID)
                y += _SEPARATOR_H
            self._rows.append(dict(
                kind=kind, y0=y, y1=y + _ITEM_H,
                rect=canvas.create_rectangle(0, y, width, y + _ITEM_H,
                                             fill=theme.PANEL, outline=""),
                name=canvas.create_text(_PAD_X, y + _NAME_Y, anchor="nw",
                                         text=kind, fill=theme.TEXT,
                                         font=self._name_font),
                desc=canvas.create_text(_PAD_X + _DESC_PAD_X, y + _DESC_Y,
                                         anchor="nw", text=description,
                                         fill=theme.DIM,
                                         font=self._desc_font)))
            y += _ITEM_H
            previous_group = group

        # Whole rows only: an item half cut off at the bottom looks broken.
        visible = max((row["y1"] for row in self._rows
                       if row["y1"] <= _MAX_ROWS * _ITEM_H), default=0)
        self._height = visible or min(y, _ITEM_H)
        self._width = width
        # One wheel notch is three items, whatever size the window has
        # realized -- the canvas's "units" otherwise depend on its height.
        canvas.configure(width=width, height=self._height,
                         yscrollincrement=_ITEM_H,
                         scrollregion=(0, 0, width, y))

    # -- highlighting --------------------------------------------------

    def _on_motion(self, event) -> None:
        self._highlight(self._row_at(event.y))

    def _row_at(self, y: float):
        y = self._canvas.canvasy(y)
        return next((row for row in self._rows
                     if row["y0"] <= y < row["y1"]), None)

    def _highlight(self, row) -> None:
        """Move the highlight to ``row`` as one thing, or clear it.

        The colours follow the tree beside this tab: dark text on the accent
        fill, which is the readable pairing under this palette.
        """
        if row is self._highlighted:
            return
        canvas = self._canvas
        if self._highlighted is not None:
            canvas.itemconfigure(self._highlighted["rect"], fill=theme.PANEL)
            canvas.itemconfigure(self._highlighted["name"], fill=theme.TEXT)
            canvas.itemconfigure(self._highlighted["desc"], fill=theme.DIM)
        self._highlighted = row
        if row is not None:
            canvas.itemconfigure(row["rect"], fill=theme.ACCENT)
            canvas.itemconfigure(row["name"], fill=theme.CANVAS)
            canvas.itemconfigure(row["desc"], fill=theme.CANVAS)

    # -- picking and scrolling ------------------------------------------

    def _on_press(self, event) -> None:
        row = self._row_at(event.y)
        self.close()
        if row is not None:
            self.on_pick(row["kind"])

    def _on_wheel(self, event) -> None:
        first, last = self._canvas.yview()
        if first <= 0.0 and last >= 1.0:
            return
        self._canvas.yview_scroll(wheel_step(event) * 3, "units")

    def _on_scrolled(self, first: str, last: str) -> None:
        """Show the scrollbar only when there is something to scroll to."""
        if float(first) <= 0.0 and float(last) >= 1.0:
            self._bar.grid_remove()
        else:
            self._bar.grid(row=0, column=1, sticky="ns", pady=1)
        self._bar.set(first, last)

    # -- opening and closing --------------------------------------------

    def toggle(self) -> None:
        self.close() if self.is_open else self.open()

    def open(self) -> None:
        """Show the list under the button, on screen, from the top."""
        self._canvas.yview_moveto(0.0)
        self._highlight(None)
        self._place()
        self._top.deiconify()
        self._top.lift()
        try:
            self._top.grab_set_global()
        except tk.TclError:
            pass
        self.is_open = True

    def _place(self) -> None:
        """Below the button, above it when there is no room below, and
        never off the edge of the screen."""
        below = self.button.winfo_rooty() + self.button.winfo_height()
        if below + self._height <= self.button.winfo_screenheight():
            y = below
        else:
            y = max(0, self.button.winfo_rooty() - self._height)
        x = min(self.button.winfo_rootx(),
                self.button.winfo_screenwidth() - self._width)
        self._top.geometry(f"+{x}+{y}")

    def close(self) -> None:
        self.is_open = False
        try:
            self._top.grab_release()
        except tk.TclError:
            pass
        self._top.withdraw()
