"""Plane geometry in local map metres: bounds, simple polygons, hulls, cell masks.

Pure functions with no display or transport dependency, so the profile
validator, the runner and the editors all ask the same questions the same way.
"""
from __future__ import annotations

import math
from typing import Sequence

Point = Sequence[float]


def bbox(points: Sequence[Point]) -> tuple[float, float, float, float] | None:
    """``(xmin, ymin, xmax, ymax)`` of a point list, or None when empty."""
    if not points: return None
    xs, ys = [float(p[0]) for p in points], [float(p[1]) for p in points]
    return (min(xs), min(ys), max(xs), max(ys))


def expand(bounds: Sequence[float], margin_m: float) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = (float(v) for v in bounds)
    m = float(margin_m)
    return (x0 - m, y0 - m, x1 + m, y1 + m)


def contains(bounds: Sequence[float] | None, point: Point) -> bool:
    if bounds is None: return False
    x0, y0, x1, y1 = bounds
    return x0 <= point[0] <= x1 and y0 <= point[1] <= y1


def _orientation(a: Point, b: Point, c: Point) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a: Point, b: Point, p: Point) -> bool:
    return (min(a[0], b[0]) - 1e-12 <= p[0] <= max(a[0], b[0]) + 1e-12
            and min(a[1], b[1]) - 1e-12 <= p[1] <= max(a[1], b[1]) + 1e-12)


def segments_intersect(a: Point, b: Point, c: Point, d: Point) -> bool:
    """Whether closed segments ``ab`` and ``cd`` touch or cross."""
    o1, o2, o3, o4 = _orientation(a, b, c), _orientation(a, b, d), _orientation(c, d, a), _orientation(c, d, b)
    if ((o1 > 0) != (o2 > 0)) and ((o3 > 0) != (o4 > 0)) and o1 and o2 and o3 and o4: return True
    return ((abs(o1) < 1e-12 and _on_segment(a, b, c)) or (abs(o2) < 1e-12 and _on_segment(a, b, d))
            or (abs(o3) < 1e-12 and _on_segment(c, d, a)) or (abs(o4) < 1e-12 and _on_segment(c, d, b)))


def simple_polygon_error(points: Sequence[Point]) -> str:
    """Why a vertex list is not a simple polygon, or ``''`` when it is."""
    n = len(points)
    if n < 3: return f'a polygon needs at least 3 vertices, not {n}'
    for p in points:
        if not all(math.isfinite(float(c)) for c in p[:2]): return 'polygon vertices must be finite'
    for i in range(n):
        a, b = points[i], points[(i + 1) % n]
        if math.dist(a[:2], b[:2]) < 1e-9: return f'vertices {i} and {(i + 1) % n} coincide'
        for j in range(i + 1, n):
            if j == i or (j + 1) % n == i or j == (i + 1) % n: continue
            if segments_intersect(a, b, points[j], points[(j + 1) % n]):
                return f'edges {i} and {j} cross'
    area = sum(points[i][0] * points[(i + 1) % n][1] - points[(i + 1) % n][0] * points[i][1] for i in range(n))
    if abs(area) < 1e-9: return 'polygon has no area'
    return ''


def point_in_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    """Even-odd rule; a point on an edge counts as inside."""
    x, y = float(point[0]), float(point[1])
    inside = False
    n = len(polygon)
    for i in range(n):
        (x0, y0), (x1, y1) = polygon[i][:2], polygon[(i + 1) % n][:2]
        if abs(_orientation((x0, y0), (x1, y1), (x, y))) < 1e-12 and _on_segment((x0, y0), (x1, y1), (x, y)):
            return True
        if (y0 > y) != (y1 > y) and x < x0 + (y - y0) * (x1 - x0) / (y1 - y0):
            inside = not inside
    return inside


def convex_hull(points: Sequence[Point]) -> list[tuple[float, float]]:
    """The convex hull as a counter-clockwise ring, or the points themselves.

    Collinear points on the hull are skipped, so the result is always a strict
    ring (or fewer points than it started with). The Hull button reduces a
    sketched blob to the area it covers.
    """
    unique = sorted({(float(p[0]), float(p[1])) for p in points})
    if len(unique) <= 2: return unique

    def half(seq):
        ring: list[tuple[float, float]] = []
        for p in seq:
            while len(ring) >= 2 and _orientation(ring[-2], ring[-1], p) <= 1e-12:
                ring.pop()
            ring.append(p)
        return ring

    lower, upper = half(unique), half(reversed(unique))
    return lower[:-1] + upper[:-1]


def cell_mask(polygon: Sequence[Point], geometry) -> list[list[bool]]:
    """``mask[row][col]``: whether the cell's centre is inside the polygon.

    ``geometry`` is anything with the grid's shape -- ``origin_x_m``,
    ``origin_y_m``, ``cell_m``, ``width``, ``height`` (``GridGeometry``
    qualifies). Cells outside the polygon's bounding box are False without a
    containment test, so a small area over a large grid costs its own size,
    not the grid's.
    """
    width, height = int(geometry.width), int(geometry.height)
    mask = [[False] * width for _ in range(height)]
    if len(polygon) < 3: return mask
    bounds = bbox(polygon)
    if bounds is None: return mask
    x0, y0, x1, y1 = bounds
    step, origin_x, origin_y = float(geometry.cell_m), float(geometry.origin_x_m), float(geometry.origin_y_m)
    first_col = max(0, int((x0 - origin_x) / step))
    last_col = min(width - 1, int((x1 - origin_x) / step))
    first_row = max(0, int((y0 - origin_y) / step))
    last_row = min(height - 1, int((y1 - origin_y) / step))
    for row in range(first_row, last_row + 1):
        y = origin_y + (row + 0.5) * step
        for col in range(first_col, last_col + 1):
            x = origin_x + (col + 0.5) * step
            mask[row][col] = point_in_polygon((x, y), polygon)
    return mask
