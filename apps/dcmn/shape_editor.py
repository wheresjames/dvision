"""A polygon or a point list being edited: every change checked, capped and undoable.

The model the Mission tab's area and waypoint editors run on (B2). It is pure
state -- no Tk, no planes -- so a test can drive exactly the gestures the pane
delivers: click to append, click on an edge to insert, drag to move,
right-click or Delete to remove, a rectangle to select, replace
for Hull/Clear and a converted tour. Polygon edits that would make the ring
cross are refused with the reason; every list is capped the way the profile
and the mission status snapshot are.

Every mutating method returns ``''`` on success, or the reason it was refused
(also kept in ``error``). ``changed`` says a successful edit has not been
applied to the profile yet; the host clears it once it has. ``undo`` steps back
through the history of successful edits.
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

from dcmn.mission import MAX_WAYPOINTS

Point = Sequence[float]

#: Successful edits remembered for ``undo``.
MAX_UNDO = 100
#: How far from a segment, in metres, a click still inserts on it.
EDGE_SNAP_M = 0.75


def _orientation(ax, ay, bx, by, cx, cy) -> float:
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


def _on_segment(ax, ay, bx, by, px, py) -> bool:
    return (min(ax, bx) - 1e-12 <= px <= max(ax, bx) + 1e-12
            and min(ay, by) - 1e-12 <= py <= max(ay, by) + 1e-12)


def _segments_cross(a, b, c, d) -> bool:
    """Whether closed segments ``ab`` and ``cd`` touch or cross."""
    o1 = _orientation(a[0], a[1], b[0], b[1], c[0], c[1])
    o2 = _orientation(a[0], a[1], b[0], b[1], d[0], d[1])
    o3 = _orientation(c[0], c[1], d[0], d[1], a[0], a[1])
    o4 = _orientation(c[0], c[1], d[0], d[1], b[0], b[1])
    if ((o1 > 0) != (o2 > 0)) and ((o3 > 0) != (o4 > 0)) and o1 and o2 and o3 and o4: return True
    return ((abs(o1) < 1e-12 and _on_segment(a[0], a[1], b[0], b[1], c[0], c[1]))
            or (abs(o2) < 1e-12 and _on_segment(a[0], a[1], b[0], b[1], d[0], d[1]))
            or (abs(o3) < 1e-12 and _on_segment(c[0], c[1], d[0], d[1], a[0], a[1]))
            or (abs(o4) < 1e-12 and _on_segment(c[0], c[1], d[0], d[1], b[0], b[1])))


def polygon_error(points: Sequence[Point]) -> str:
    """Why a vertex ring is not a simple polygon, or ``''`` when it is.

    The same question :func:`dmsn.geometry.simple_polygon_error` answers for a
    profile, answered here so the editor refuses with a reason instead of
    letting the profile grow invalid and failing later.
    """
    n = len(points)
    if n < 3: return f'a polygon needs at least 3 vertices, not {n}'
    for p in points:
        if not all(math.isfinite(float(c)) for c in p[:2]): return 'polygon vertices must be finite'
    for i in range(n):
        a, b = points[i], points[(i + 1) % n]
        if math.dist(a[:2], b[:2]) < 1e-9: return f'vertices {i} and {(i + 1) % n} coincide'
        for j in range(i + 1, n):
            if j == i or (j + 1) % n == i or j == (i + 1) % n: continue
            if _segments_cross(a, b, points[j], points[(j + 1) % n]):
                return f'edges {i} and {j} cross'
    area = sum(points[i][0] * points[(i + 1) % n][1] - points[(i + 1) % n][0] * points[i][1] for i in range(n))
    if abs(area) < 1e-9: return 'polygon has no area'
    return ''


def _point_on_segment(px, py, ax, ay, bx, by) -> tuple[tuple[float, float], float]:
    """The closest point on segment ``ab`` to ``p``, and the distance to it."""
    dx, dy = bx - ax, by - ay
    length = math.hypot(dx, dy)
    if length < 1e-12: return (ax, ay), math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (length * length)))
    qx, qy = ax + t * dx, ay + t * dy
    return (qx, qy), math.hypot(px - qx, py - qy)


class ShapeEditor:
    """One editable geometry: the mission's area polygon (``closed``) or waypoint list."""

    def __init__(self, points: Iterable[Point] = (), *, closed: bool = False,
                 limit: int = MAX_WAYPOINTS) -> None:
        self.closed = bool(closed)
        self.limit = int(limit)
        self.points: list[tuple[float, float]] = [(float(x), float(y)) for x, y in points]
        self.selected: frozenset[int] = frozenset()
        self.error = ''
        self.changed = False
        self._history: list[list[tuple[float, float]]] = []
        self._last_key: tuple | None = None

    # -- reading ------------------------------------------------------------------

    def as_list(self) -> list[list[float]]:
        return [[x, y] for x, y in self.points]

    def handles(self) -> dict[int, tuple[float, float]]:
        """Vertex positions by index, for the pane's handle hit-testing."""
        return dict(enumerate(self.points))

    def in_selection(self, index: int) -> bool:
        return index in self.selected

    # -- the checks every edit passes ------------------------------------------------

    def _checked(self, points: Sequence[Point], *, closed: bool | None = None) -> str:
        closed = self.closed if closed is None else closed
        if len(points) > self.limit: return f'at most {self.limit} points are allowed'
        for x, y in points:
            if not (math.isfinite(x) and math.isfinite(y)): return 'points must be finite numbers'
        # Fewer than three vertices is a cleared or half-drawn ring, not a bad
        # one -- the profile accepts an empty area the same way, and only
        # checks a polygon once it could be one.
        if closed and len(points) >= 3: return polygon_error(points)
        return ''

    def _commit(self, points: Sequence[Point], selection: frozenset[int] = frozenset(),
                coalesce_key: tuple | None = None) -> str:
        problem = self._checked(points)
        if problem:
            self.error = problem
            return problem
        if coalesce_key is None or coalesce_key != self._last_key:
            self._history.append(list(self.points))
            del self._history[:-MAX_UNDO]
        self._last_key = coalesce_key  # one drag gesture is one undo step
        self.points = [(float(x), float(y)) for x, y in points]
        self.selected = frozenset(i for i in selection if i < len(self.points))
        self.error, self.changed = '', True
        return ''

    # -- the gestures ------------------------------------------------------------------

    def add(self, x: float, y: float, *, snap_m: float = EDGE_SNAP_M) -> str:
        """A click: insert on the nearest segment when it is close, otherwise add.

        An open list grows at the end. A polygon of three or more vertices
        takes the new vertex on the edge where it adds the least perimeter and
        the ring stays simple -- so a click beside an existing area pulls out
        the nearest edge instead of being refused for crossing, whatever order
        the ring was drawn in.
        """
        x, y = float(x), float(y)
        if len(self.points) >= 3:
            best, best_distance = None, snap_m
            segments = self._segments()
            for index, (a, b) in segments.items():
                point, distance = _point_on_segment(x, y, a[0], a[1], b[0], b[1])
                if distance <= best_distance:
                    if best is None or distance < best_distance:
                        best, best_distance = (index, point), distance
            if best is not None:
                index, point = best
                return self.insert(index + 1, *point)
        if self.closed and len(self.points) >= 3:
            return self._insert_best(x, y)
        return self.insert(len(self.points), x, y)

    def _insert_best(self, x: float, y: float) -> str:
        """Insert into the ring on the edge that grows it least and keeps it simple."""
        if len(self.points) >= self.limit:
            self.error = f'at most {self.limit} points are allowed'
            return self.error
        n = len(self.points)
        order = sorted(range(n), key=lambda i: (math.dist(self.points[i], (x, y))
                                                + math.dist((x, y), self.points[(i + 1) % n])
                                                - math.dist(self.points[i], self.points[(i + 1) % n]), i))
        for index in order:
            candidate = list(self.points)
            candidate.insert(index + 1, (x, y))
            if not polygon_error(candidate): return self._commit(candidate, frozenset())
        self.error = 'no edge takes that vertex without the area crossing itself'
        return self.error

    def insert(self, index: int, x: float, y: float) -> str:
        points = list(self.points)
        index = max(0, min(len(points), int(index)))
        points.insert(index, (float(x), float(y)))
        return self._commit(points, frozenset())

    def move(self, index: int, x: float, y: float, *, coalesce: bool = False) -> str:
        """Move one vertex; with ``coalesce``, a whole drag gesture is one undo step."""
        if not 0 <= int(index) < len(self.points): return f'no vertex {index}'
        points = list(self.points)
        points[int(index)] = (float(x), float(y))
        return self._commit(points, coalesce_key=('move', int(index)) if coalesce else None)

    def remove(self, index: int) -> str:
        """Remove one vertex; the checks in :meth:`_commit` still apply."""
        if not 0 <= int(index) < len(self.points): return f'no vertex {index}'
        points = list(self.points)
        del points[int(index)]
        selection = frozenset(i for i in self.selected if i != int(index))
        selection = frozenset((i if i < int(index) else i - 1) for i in selection)
        return self._commit(points, selection)

    def remove_indices(self, indices: Iterable[int]) -> str:
        """Remove many vertices at once (a selection); the checks in :meth:`_commit` still apply."""
        gone = sorted({int(i) for i in indices if 0 <= int(i) < len(self.points)}, reverse=True)
        if not gone: return 'nothing selected'
        points = list(self.points)
        for index in gone: del points[index]
        return self._commit(points, frozenset())

    def reorder(self, index: int, to: int) -> str:
        """Move one waypoint to another position in the list (the list's up/down buttons)."""
        n = len(self.points)
        if self.closed: return 'a polygon has no order to change'
        if not 0 <= int(index) < n or not 0 <= int(to) < n: return f'no waypoint {index} or {to}'
        points = list(self.points)
        points.insert(int(to), points.pop(int(index)))
        return self._commit(points)

    def select(self, bounds: Sequence[float]) -> int:
        """Select the vertices inside ``xmin,ymin,xmax,ymax`` (a shift-drag rectangle)."""
        x0, y0, x1, y1 = bounds
        self.selected = frozenset(i for i, (x, y) in enumerate(self.points)
                                  if x0 <= x <= x1 and y0 <= y <= y1)
        return len(self.selected)

    def replace(self, points: Iterable[Point] = ()) -> str:
        """Set the whole list (Hull, Clear, a converted tour, an undo of everything)."""
        try: values = [(float(x), float(y)) for x, y in points]
        except (TypeError, ValueError): return 'points must be x,y pairs'
        return self._commit(values)

    def clear(self) -> str:
        return self.replace(())

    # -- history ------------------------------------------------------------------------

    def undo(self) -> str:
        if not self._history:
            self.error = 'nothing to undo'
            return self.error
        self.points = self._history.pop()
        self.selected = frozenset()
        self.error, self.changed = '', True
        return ''

    def can_undo(self) -> bool:
        return bool(self._history)

    def take_changes(self) -> list[list[float]] | None:
        """The points when an edit has not been applied yet, or None; clears the mark."""
        if not self.changed: return None
        self.changed = False
        return self.as_list()

    # -- internals -------------------------------------------------------------------------

    def _segments(self) -> dict[int, tuple[tuple[float, float], tuple[float, float]]]:
        """Segments by the index of their first vertex: a ring closes, a list does not."""
        n = len(self.points)
        if n < 2: return {}
        count = n if self.closed else n - 1
        return {i: (self.points[i], self.points[(i + 1) % n]) for i in range(count)}
