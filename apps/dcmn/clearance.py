"""Exact horizontal distances to occupied grid squares, shared by planning and admission."""
import numpy as np


def solid_cells(occupied, never):
    """Occupied cells, plus never-observed cells touching one.

    Plan permission trusts unknown space, but an unseen cell beside an observed
    obstacle is as likely its continuation as free space. Seen in the Sep 15
    area1 crash: a wall's end cells were unobserved, and a route that cleared
    the observed end by 0.35 m ran through the real one.
    """
    occupied = np.asarray(occupied, bool)
    height, width = occupied.shape
    grown = occupied.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            grown[max(0, dy):height-max(0, -dy), max(0, dx):width-max(0, -dx)] |= \
                occupied[max(0, -dy):height-max(0, dy), max(0, -dx):width-max(0, dx)]
    return occupied | (grown & np.asarray(never, bool))


class ObstacleClearance:
    def __init__(self, geometry, occupied, radius, body_radius=0.0):
        self.geometry = geometry
        self.occupied = np.asarray(occupied, bool)
        self.radius = radius
        self.body_radius = body_radius

    def distances(self, a, b):
        """Start, end and minimum segment distance to each nearby full cell."""
        g, radius = self.geometry, self.radius
        lo = np.floor((np.minimum(a[:2], b[:2]) - radius -
                       [g.origin_x_m, g.origin_y_m]) / g.cell_m - 1e-9).astype(int)
        hi = np.floor((np.maximum(a[:2], b[:2]) + radius -
                       [g.origin_x_m, g.origin_y_m]) / g.cell_m).astype(int)
        lo = np.maximum(lo, 0)
        hi = np.minimum(hi, [g.width - 1, g.height - 1])
        if np.any(hi < lo): return np.array([]), np.array([]), np.array([])
        rows, cols = np.nonzero(self.occupied[lo[1]:hi[1]+1, lo[0]:hi[0]+1])
        lower = np.column_stack((cols + lo[0], rows + lo[1])) * g.cell_m
        lower += [g.origin_x_m, g.origin_y_m]
        upper = lower + g.cell_m
        a, b = np.asarray(a[:2], float), np.asarray(b[:2], float)
        def point_distance(point):
            return np.linalg.norm(np.maximum(np.maximum(lower-point, point-upper), 0), axis=1)
        da, db = point_distance(a), point_distance(b)
        minimum = np.minimum(da, db)
        delta = b - a
        span = float(delta @ delta)
        if span:
            # Disjoint segment/rectangle minima occur at an endpoint or vertex.
            for x, y in ((0, 0), (0, 1), (1, 0), (1, 1)):
                corner = lower + np.array([x, y]) * g.cell_m
                t = np.clip(((corner-a) @ delta) / span, 0, 1)
                minimum = np.minimum(minimum, np.linalg.norm(corner-a-t[:, None]*delta, axis=1))
            # Slab intersection also catches a segment crossing a cell interior.
            enter, leave = np.zeros(len(lower)), np.ones(len(lower))
            for axis in (0, 1):
                if delta[axis] == 0:
                    leave[(a[axis] < lower[:, axis]) | (a[axis] > upper[:, axis])] = -1
                else:
                    t0 = (lower[:, axis]-a[axis])/delta[axis]
                    t1 = (upper[:, axis]-a[axis])/delta[axis]
                    enter = np.maximum(enter, np.minimum(t0, t1))
                    leave = np.minimum(leave, np.maximum(t0, t1))
            minimum[enter <= leave] = 0
        return da, db, minimum

    def lateral_m(self, a, b):
        """Least distance from the segment to a full occupied cell, capped at ``radius``."""
        _, _, minimum = self.distances(a, b)
        return float(min(self.radius, minimum.min())) if len(minimum) else float(self.radius)

    def allows(self, a, b, *, escape=False):
        da, db, minimum = self.distances(a, b)
        clear = minimum > self.radius + 1e-9
        if escape:
            # Distance to a convex square is convex along a segment. Its minimum
            # at the start guarantees nondecreasing separation along the move.
            clear |= ((da <= self.radius + 1e-9) & (minimum > self.body_radius + 1e-9)
                      & (minimum >= da - 1e-9) & (db >= da - 1e-9))
        return bool(clear.all())
