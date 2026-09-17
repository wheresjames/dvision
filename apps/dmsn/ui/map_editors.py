"""The Mission tab's geometry editors: the area polygon and the waypoint list (B2).

Everything the operator does to a mission's geometry goes through a
:class:`~dcmn.shape_editor.ShapeEditor` -- one for the area, one for the
waypoints, whichever the mission asks for by its ``editors`` -- and every
accepted edit is written straight into the profile document, so the file is
the source of truth and the digest moves with it. The gestures are the ones
the pane delivers: click to append, click on an edge to insert, drag to move,
right-click or Delete to remove, Undo/Hull/Clear buttons, and a reorderable
list for waypoints. Shift-drag draws a rectangle: the area becomes that
rectangle, and on a waypoint list it selects the points inside for Delete.
Points outside the current evidence coverage are accepted but flagged, since
flying to them resets the map (B1).

Editing happens between missions: while a mission is flying the geometry is
what it was launched with, and every gesture is refused with the reason.
"""
from __future__ import annotations

from typing import Callable

from dcmn.mission import MAX_VERTICES, MAX_WAYPOINTS
from dcmn.shape_editor import ShapeEditor
from dmsn import geometry as G
from dmsn.missions import MISSIONS

#: The profile keys an editor may own, and the map tools they use.
EDITOR_KEYS = ('area', 'waypoints')


def _hit_handles(editor: ShapeEditor):
    """The editor's vertices as ``(key, x, y)`` triples, for the pane's hit test."""
    return [(index, x, y) for index, (x, y) in editor.handles().items()]


class MapEditors:
    def __init__(self, parent, run, pane, notice: Callable[[str], None], *, row: int = 0) -> None:
        self.run, self.pane, self.notice = run, pane, notice
        self.parent, self.row = parent, row
        self.keys: tuple[str, ...] = ()
        self.editors: dict[str, ShapeEditor] = {}
        self.frame = None
        self.listbox = None
        self._pressed: tuple | None = None
        self._rectangle: tuple[float, float, float, float] | None = None

    # -- state ------------------------------------------------------------------

    def enabled(self) -> bool:
        """Editing happens between missions; a flying mission owns its geometry."""
        return bool(self.editors) and self.run.phase in ('idle', 'done')

    def sync(self) -> None:
        """Follow the profile: new editors for a new mission, resets for outside changes."""
        document = self.run.document
        mission = MISSIONS.get(document.value['mission'])
        keys = tuple(e for e in getattr(mission, 'editors', ()) if e in EDITOR_KEYS)
        if keys != self.keys:
            self.keys = keys
            self.editors = {key: self._editor(key, document.value[key]) for key in keys}
            self._build()
            return
        for key in keys:  # changed elsewhere: a load, a tour import, the Setup tab
            if self.editors[key].as_list() != document.value[key]:
                self.editors[key] = self._editor(key, document.value[key])
                self._build()
                return

    def _editor(self, key: str, points) -> ShapeEditor:
        return ShapeEditor(points, closed=(key == 'area'),
                           limit=MAX_VERTICES if key == 'area' else MAX_WAYPOINTS)

    def _build(self) -> None:
        if self.frame is not None:
            self.frame.destroy()
            self.frame, self.listbox = None, None
        if not self.keys: return
        from tkinter import ttk
        title = 'Area' if 'area' in self.keys else 'Waypoints'
        self.frame = ttk.LabelFrame(self.parent, text=title, padding=8)
        self.frame.grid(row=self.row, column=0, sticky='ew', pady=(0, 6))
        self.frame.columnconfigure(0, weight=1)
        if 'area' in self.keys:
            self._build_area(self.frame)
        else:
            self._build_waypoints(self.frame)
        self._paint_list()

    def _build_area(self, frame) -> None:
        from tkinter import ttk
        from dcmn.theme import DIM
        help_text = ('click: add a vertex · click an edge: insert · drag: move · '
                     'right-click: remove · shift-drag: rectangle area')
        ttk.Label(frame, text=help_text, foreground=DIM, wraplength=210, justify='left').grid(
            row=0, column=0, sticky='w')
        buttons = ttk.Frame(frame)
        buttons.grid(row=1, column=0, sticky='ew', pady=(4, 0))
        for index, (label, command) in enumerate((
                ('Hull', self.hull), ('Clear', self.clear), ('Undo', self.undo))):
            ttk.Button(buttons, text=label, command=command, width=6).grid(
                row=0, column=index, padx=(0, 4) if index < 2 else 0)

    def _build_waypoints(self, frame) -> None:
        import tkinter as tk
        from tkinter import ttk
        from dcmn.theme import DIM
        help_text = ('click: add a waypoint · click a leg: insert · drag: move · '
                     'right-click: remove · shift-drag: select · Del: remove selected')
        ttk.Label(frame, text=help_text, foreground=DIM, wraplength=210, justify='left').grid(
            row=0, column=0, sticky='w')
        self.listbox = tk.Listbox(frame, height=6, exportselection=False)
        self.listbox.grid(row=1, column=0, sticky='ew', pady=(4, 0))
        self.listbox.bind('<Double-Button-1>', lambda _e: self._remove_listed())
        buttons = ttk.Frame(frame)
        buttons.grid(row=2, column=0, sticky='ew', pady=(4, 0))
        for index, (label, command) in enumerate((
                ('Up', lambda: self._move_listed(-1)), ('Down', lambda: self._move_listed(1)),
                ('Delete', self._remove_listed), ('Undo', self.undo))):
            ttk.Button(buttons, text=label, command=command, width=6).grid(
                row=0, column=index, padx=(0, 4) if index < 3 else 0)

    def _paint_list(self) -> None:
        if self.listbox is None: return
        editor = self.editors.get('waypoints')
        if editor is None: return
        self.listbox.delete(0, 'end')
        for index, (x, y) in enumerate(editor.points):
            self.listbox.insert('end', f'{index + 1}. {x:.1f}, {y:.1f} m')

    # -- gestures (the pane calls these) -----------------------------------------------

    def press(self, x_m: float, y_m: float, modifiers: frozenset) -> None:
        self._pressed = None
        if not self.enabled(): return
        if 'shift' in modifiers:
            self._pressed = ('rectangle', x_m, y_m)
            self._rectangle = (x_m, y_m, x_m, y_m)
            return
        for key, editor in self.editors.items():
            hit = self.pane.handle_at(x_m, y_m, _hit_handles(editor))
            if hit is not None:
                self._pressed = ('handle', key, hit)
                editor.selected = frozenset({hit})
                return
        self._pressed = ('plain',)

    def consume_click(self, x_m: float, y_m: float) -> bool:
        """A click on the map: an editor adds a vertex, or the goal keeps it.

        A mission with a map editor owns every click on the map: a press on a
        handle or a shift-press is the start of a drag, which release finishes,
        so it is consumed here without adding anything.
        """
        if not self.editors: return False
        if not self.enabled():
            self.notice(f'the {" and ".join(self.editors)} can be edited when no mission is flying')
            return True
        if self._pressed is None or self._pressed[0] != 'plain': return True
        self._pressed = None
        key = 'area' if 'area' in self.editors else 'waypoints'
        editor = self.editors[key]
        problem = editor.add(x_m, y_m)
        if problem:
            self.notice(problem)
            return True
        self._apply(key, editor)
        return True

    def drag(self, x_m: float, y_m: float, modifiers: frozenset) -> None:
        if self._pressed is None: return
        if self._pressed[0] == 'handle':
            _, key, index = self._pressed
            editor = self.editors[key]
            problem = editor.move(index, x_m, y_m, coalesce=True)
            if problem == '':
                self._apply(key, editor, quiet=True)
            else:
                self.notice(problem)  # the vertex stays put, and the reason is shown
        elif self._pressed[0] == 'rectangle':
            x0, y0 = self._pressed[1], self._pressed[2]
            self._rectangle = (min(x0, x_m), min(y0, y_m), max(x0, x_m), max(y0, y_m))

    def release(self, x_m: float, y_m: float, modifiers: frozenset) -> None:
        if self._pressed is None: return
        kind = self._pressed[0]
        self._pressed = None
        if kind == 'rectangle' and self._rectangle is not None:
            bounds, self._rectangle = self._rectangle, None
            editor = self.editors.get('area')
            if editor is not None:
                x0, y0, x1, y1 = bounds
                if x1 - x0 < 0.5 or y1 - y0 < 0.5:
                    return self.notice('shift-drag to draw a rectangle area; click to add a vertex')
                problem = editor.replace(((x0, y0), (x1, y0), (x1, y1), (x0, y1)))
                if problem: return self.notice(problem)
                self._apply('area', editor)
                return
            count = 0
            if bounds[2] - bounds[0] < 1e-9 and bounds[3] - bounds[1] < 1e-9:
                return self.notice('shift-drag to select waypoints; click to add one')
            for editor in self.editors.values(): count += editor.select(bounds)
            self.notice(f'{count} vertices selected' if count else 'no vertex in that rectangle')

    def context(self, x_m: float, y_m: float, modifiers: frozenset) -> None:
        if not self.enabled(): return
        for key, editor in self.editors.items():
            hit = self.pane.handle_at(x_m, y_m, _hit_handles(editor))
            if hit is not None:
                problem = editor.remove(hit)
                if problem: return self.notice(problem)
                self._apply(key, editor)
                return

    def key(self, keysym: str, pointer) -> None:
        if not self.enabled(): return
        if keysym in ('Delete', 'KP_Delete', 'BackSpace'):
            for key, editor in self.editors.items():
                if not editor.selected: continue
                problem = editor.remove_indices(editor.selected)
                if problem: return self.notice(problem)
                self._apply(key, editor)
                return
        elif keysym == 'Escape':
            for editor in self.editors.values(): editor.selected = frozenset()

    # -- buttons -----------------------------------------------------------------------

    def hull(self) -> None:
        editor = self.editors.get('area')
        if editor is None: return
        problem = editor.replace(G.convex_hull(editor.points))
        if problem: return self.notice(problem)
        self._apply('area', editor)

    def clear(self) -> None:
        for key, editor in self.editors.items():
            problem = editor.clear()
            if problem: return self.notice(problem)
            self._apply(key, editor)
            return

    def undo(self) -> None:
        for key, editor in self.editors.items():
            problem = editor.undo()
            if problem: return self.notice(problem)
            self._apply(key, editor)
            return

    def _move_listed(self, delta: int) -> None:
        if self.listbox is None: return
        selection = self.listbox.curselection()
        if not selection: return self.notice('select a waypoint in the list first')
        index, to = selection[0], selection[0] + delta
        editor = self.editors['waypoints']
        problem = editor.reorder(index, to)
        if problem: return self.notice(problem)
        self._apply('waypoints', editor, quiet=True)
        self.listbox.selection_set(to)

    def _remove_listed(self) -> None:
        if self.listbox is None: return
        selection = self.listbox.curselection()
        if not selection: return self.notice('select a waypoint in the list first')
        editor = self.editors['waypoints']
        problem = editor.remove(selection[0])
        if problem: return self.notice(problem)
        self._apply('waypoints', editor)

    # -- applying and drawing -----------------------------------------------------------

    def _apply(self, key: str, editor: ShapeEditor, *, quiet: bool = False) -> None:
        """Write an accepted edit into the profile document; it is the source of truth."""
        if not self.run.document.set(key, editor.as_list()):
            return self.notice(f'not changed: {self.run.document.error}')
        self._paint_list()
        coverage = self.run.observation.coverage if self.run.observation is not None else None
        outside = 0 if coverage is None else sum(not G.contains(coverage, p) for p in editor.points)
        if outside:
            self.notice(f'{key}: {len(editor.points)} {"vertices" if key == "area" else "waypoints"}; '
                        f'{outside} outside evidence coverage (flying there resets the map)')
        elif not quiet:
            what = 'vertices' if key == 'area' else 'waypoints'
            self.notice(f'{key}: {len(editor.points)} {what}')

    def shapes(self) -> list:
        """The editor's own marks: vertex handles, the waypoint legs, the drag rectangle."""
        if not self.enabled(): return []
        from dcmn import theme
        from dcmn.mapview import Shape
        result = []
        scale = self.pane.metres_per_pixel() or 0.5
        radius = 4. * scale
        for key, editor in self.editors.items():
            for index, (x, y) in editor.handles().items():
                colour = theme.HIGHLIGHT if editor.in_selection(index) else theme.ACCENT
                result.append(Shape('rect', x - radius, y - radius, x + radius, y + radius,
                                    outline=colour, width=1.5))
            if key == 'waypoints':
                points = editor.points
                for (ax, ay), (bx, by) in zip(points, points[1:]):
                    result.append(Shape('line', ax, ay, bx, by, outline=theme.DIM, width=1., dash=(4, 4)))
        if self._rectangle is not None:
            x0, y0, x1, y1 = self._rectangle
            result.append(Shape('rect', x0, y0, x1, y1, outline=theme.HIGHLIGHT, width=1., dash=(3, 3)))
        return result
