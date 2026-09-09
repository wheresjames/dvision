"""A form taller than the window it sits in, and the scrollbar that implies.

A notebook is as tall as its tallest page, so a long settings form would set
the height of the whole window and push everything beside it down the screen.
The viewport instead asks for a modest height and scrolls the rest.
"""

import tkinter as tk
from tkinter import ttk

from dcmn import theme

#: Never ask for less window than this, however small the page beside it is.
_MIN_VIEWPORT_PX = 320

#: X11 reports the wheel as buttons 4 and 5; other platforms send a delta.
_WHEEL_SEQUENCES = ("<Button-4>", "<Button-5>", "<MouseWheel>")


def wheel_step(event) -> int:
    """Scroll lines for one wheel notch, whichever way the platform says it.

    X11 reports the wheel as buttons 4 and 5; every other platform sends a
    signed ``delta``.
    """
    if getattr(event, "num", 0) == 4:
        return -1
    if getattr(event, "num", 0) == 5:
        return 1
    return -1 if getattr(event, "delta", 0) > 0 else 1


class Scrollable:
    """A canvas that scrolls the frame inside it, plus a wheel that reaches it.

    The scrollbar appears only when the content is taller than the viewport,
    and the wheel scrolls this page wherever the pointer is over it -- without
    spinning a combobox, which over a form would silently edit a setting.
    """

    def __init__(self, parent: tk.Misc, *, height: int = _MIN_VIEWPORT_PX) -> None:
        self.outer = ttk.Frame(parent)
        self.outer.rowconfigure(0, weight=1)
        self.outer.columnconfigure(0, weight=1)
        self._canvas = tk.Canvas(self.outer, height=max(_MIN_VIEWPORT_PX, height),
                                borderwidth=0, highlightthickness=0,
                                background=theme.BG)
        self._canvas.grid(row=0, column=0, sticky="nsew")
        self._bar = ttk.Scrollbar(self.outer, orient="vertical",
                                  command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=self._on_scrolled)

        self.inner = ttk.Frame(self._canvas)
        self._window = self._canvas.create_window((0, 0), window=self.inner,
                                                  anchor="nw")
        self.inner.bind("<Configure>", self._on_content_resized)
        self._canvas.bind("<Configure>", self._on_viewport_resized)
        # Bound application-wide and filtered by ancestry rather than by
        # Enter/Leave on the canvas: the form's own widgets are children of the
        # canvas, so moving the pointer onto one of them fires Leave and would
        # take the wheel away exactly where it is wanted.
        for sequence in _WHEEL_SEQUENCES:
            self._canvas.bind_all(sequence, self._wheel, add="+")
        self._canvas.bind("<Destroy>", self._release_wheel)

    def _on_content_resized(self, _event) -> None:
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))

    def _on_viewport_resized(self, event) -> None:
        self._canvas.itemconfigure(self._window, width=event.width)

    def _on_scrolled(self, first: str, last: str) -> None:
        """Show the scrollbar only when there is something to scroll to."""
        if float(first) <= 0.0 and float(last) >= 1.0:
            self._bar.grid_remove()
        else:
            self._bar.grid(row=0, column=1, sticky="ns")
        self._bar.set(first, last)

    def claim_wheel(self, widget: tk.Misc) -> None:
        """Scroll the page over ``widget`` instead of whatever it would do.

        ttk's combobox spins its own value on the wheel. Over a form that is a
        trap: a scroll aimed at the page silently changes a setting. A binding
        on the widget itself runs before its class binding and breaks out of
        it, so the wheel means one thing everywhere on this page.
        """
        for sequence in _WHEEL_SEQUENCES:
            widget.bind(sequence, self._wheel_and_stop)

    def _release_wheel(self, _event) -> None:
        for sequence in _WHEEL_SEQUENCES:
            self._canvas.unbind_all(sequence)

    def _wheel_and_stop(self, event) -> str:
        self._scroll_by(event)
        return "break"

    def _wheel(self, event) -> None:
        # bind_all reaches every widget in the application, so only act when the
        # pointer is over something inside this page.
        widget = event.widget
        while widget is not None:
            if widget is self._canvas:
                break
            widget = getattr(widget, "master", None)
        else:
            return
        self._scroll_by(event)

    def _scroll_by(self, event) -> None:
        # Ask the view, not the scrollbar: whether the wheel should do anything
        # is a question about the scroll range, and a widget's mapped state
        # answers a different one.
        first, last = self._canvas.yview()
        if first <= 0.0 and last >= 1.0:
            return
        self._canvas.yview_scroll(wheel_step(event) * 3, "units")
