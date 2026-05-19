"""Vision-built local occupancy map and A* route planner for daic."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Any

from daic.orb_slam3_detector import ObstacleSectors


_CELL_M = 0.5
_HALF_WIDTH_M = 14.0
_OCCUPIED = 1.6
_FREE = -1.2
_OCC_HIT = 1.0
_FREE_HIT = -0.35
_DECAY = 0.995
_DEFAULT_OBSTACLE_DIST_M = 3.0
_MIN_OBSTACLE_DIST_M = 0.7
_MAX_OBSTACLE_DIST_M = 8.0
_OBSTACLE_SPREAD_M = 0.8
_WAYPOINT_LOOKAHEAD_M = 2.0
_MAX_YAW_DPS = 18.0
_YAW_GAIN = 0.45
_NAV_SPEED = 4.5
_NAV_MIN_SPEED = 1.5
_ALIGN_DEG = 90.0
_TURN_ONLY_DEG = 135.0


@dataclass(frozen=True)
class Pose2:
    x: float
    y: float
    yaw_deg: float


@dataclass(frozen=True)
class PlannedCommand:
    fields: dict
    status: str
    path: list[tuple[float, float]]


class LocalOccupancyMap:
    """Rolling world-frame occupancy grid built from vision obstacle sectors."""

    def __init__(self, cell_m: float = _CELL_M,
                 half_width_m: float = _HALF_WIDTH_M) -> None:
        self.cell_m = cell_m
        self.half_width_m = half_width_m
        self._cells: dict[tuple[int, int], float] = {}
        self.last_path: list[tuple[float, float]] = []

    def update(self, pose: Pose2, sectors: ObstacleSectors) -> None:
        self._decay_and_prune(pose)
        self._mark_free_fan(pose)
        if sectors.confidence <= 0.0:
            return

        sector_defs = [
            (-70.0, sectors.left, sectors.left_range_m),
            (-25.0, sectors.front_left, sectors.front_left_range_m),
            (0.0, sectors.front, sectors.front_range_m),
            (25.0, sectors.front_right, sectors.front_right_range_m),
            (70.0, sectors.right, sectors.right_range_m),
        ]
        for rel_deg, risk, range_m in sector_defs:
            if risk < 0.18:
                continue
            self._mark_obstacle_sector(
                pose, rel_deg, risk * sectors.confidence, range_m,
            )

    def plan_to_target(self, pose: Pose2,
                       target_xy: tuple[float, float]) -> PlannedCommand | None:
        start = self._cell(pose.x, pose.y)
        goal = self._cell(*target_xy)
        radius_cells = max(3, int(self.half_width_m / self.cell_m))
        path_cells = self._astar(start, goal, radius_cells)
        if not path_cells:
            self.last_path = []
            return None

        path = [self._world(c) for c in path_cells]
        self.last_path = path
        waypoint = _lookahead(path, (pose.x, pose.y), _WAYPOINT_LOOKAHEAD_M)
        fields = _command_to_waypoint(pose, waypoint)
        return PlannedCommand(
            fields=fields,
            status=f"local route {len(path_cells)} cells",
            path=path,
        )

    def _mark_free_fan(self, pose: Pose2) -> None:
        for rel_deg in (-35.0, -20.0, 0.0, 20.0, 35.0):
            for dist in (1.0, 1.5, 2.0):
                self._add_relative(pose, rel_deg, dist, _FREE_HIT)

    def _mark_obstacle_sector(self, pose: Pose2,
                              rel_deg: float, strength: float,
                              range_m: float | None) -> None:
        dist = _obstacle_dist(range_m)
        radial_spread = min(_OBSTACLE_SPREAD_M, max(0.15, dist * 0.2))
        for off_deg in (-10.0, 0.0, 10.0):
            for off_m in (-radial_spread, 0.0, radial_spread):
                self._add_relative(
                    pose,
                    rel_deg + off_deg,
                    max(_MIN_OBSTACLE_DIST_M, dist + off_m),
                    _OCC_HIT * strength,
                )

    def _add_relative(self, pose: Pose2,
                      rel_deg: float, dist_m: float, delta: float) -> None:
        yaw = math.radians(pose.yaw_deg + rel_deg)
        x = pose.x + math.cos(yaw) * dist_m
        y = pose.y + math.sin(yaw) * dist_m
        cell = self._cell(x, y)
        self._cells[cell] = _clamp(self._cells.get(cell, 0.0) + delta, -3.0, 3.0)

    def _decay_and_prune(self, pose: Pose2) -> None:
        keep: dict[tuple[int, int], float] = {}
        max_cells = int(self.half_width_m / self.cell_m)
        pc = self._cell(pose.x, pose.y)
        for cell, value in self._cells.items():
            if abs(cell[0] - pc[0]) > max_cells or abs(cell[1] - pc[1]) > max_cells:
                continue
            value *= _DECAY
            if abs(value) > 0.05:
                keep[cell] = value
        self._cells = keep

    def _astar(self, start: tuple[int, int], goal: tuple[int, int],
               radius_cells: int) -> list[tuple[int, int]]:
        min_x = min(start[0], goal[0]) - radius_cells
        max_x = max(start[0], goal[0]) + radius_cells
        min_y = min(start[1], goal[1]) - radius_cells
        max_y = max(start[1], goal[1]) + radius_cells

        open_heap: list[tuple[float, tuple[int, int]]] = [(0.0, start)]
        came_from: dict[tuple[int, int], tuple[int, int]] = {}
        g_score: dict[tuple[int, int], float] = {start: 0.0}

        while open_heap:
            _, current = heapq.heappop(open_heap)
            if current == goal or _cell_dist(current, goal) <= 1.5:
                return _reconstruct(came_from, current)

            for nb in _neighbors(current):
                if not (min_x <= nb[0] <= max_x and min_y <= nb[1] <= max_y):
                    continue
                cost = self._cell_cost(nb)
                if math.isinf(cost):
                    continue
                step = math.sqrt(2.0) if nb[0] != current[0] and nb[1] != current[1] else 1.0
                tentative = g_score[current] + step * cost
                if tentative >= g_score.get(nb, math.inf):
                    continue
                came_from[nb] = current
                g_score[nb] = tentative
                f = tentative + _cell_dist(nb, goal)
                heapq.heappush(open_heap, (f, nb))

        return []

    def _cell_cost(self, cell: tuple[int, int]) -> float:
        value = self._cells.get(cell, 0.0)
        if value >= _OCCUPIED:
            return math.inf
        if value <= _FREE:
            return 0.7
        return 1.5 + max(0.0, value) * 4.0

    def _cell(self, x: float, y: float) -> tuple[int, int]:
        return (round(x / self.cell_m), round(y / self.cell_m))

    def _world(self, cell: tuple[int, int]) -> tuple[float, float]:
        return (cell[0] * self.cell_m, cell[1] * self.cell_m)

    def snapshot(self) -> dict:
        return {
            "cell_m": self.cell_m,
            "half_width_m": self.half_width_m,
            "cells": dict(self._cells),
            "path": list(self.last_path),
        }

    def diagnostics(self, pose: Pose2,
                    target_xy: tuple[float, float] | None = None) -> dict:
        occupied: list[tuple[float, float, float]] = []
        free_count = 0
        for cell, value in self._cells.items():
            if value >= 0.2:
                wx, wy = self._world(cell)
                occupied.append((wx, wy, value))
            elif value <= -0.2:
                free_count += 1

        nearest_occ = None
        front_occ = None
        yaw = math.radians(pose.yaw_deg)
        fwd_x, fwd_y = math.cos(yaw), math.sin(yaw)
        for wx, wy, _value in occupied:
            dx = wx - pose.x
            dy = wy - pose.y
            dist = math.hypot(dx, dy)
            nearest_occ = dist if nearest_occ is None else min(nearest_occ, dist)
            along = dx * fwd_x + dy * fwd_y
            lateral = abs(-dx * fwd_y + dy * fwd_x)
            if along > 0.0 and lateral <= 1.5:
                front_occ = along if front_occ is None else min(front_occ, along)

        out = {
            "cells": len(self._cells),
            "occupied_cells": len(occupied),
            "free_cells": free_count,
            "path_len": len(self.last_path),
            "nearest_occ_m": round(nearest_occ, 2) if nearest_occ is not None else None,
            "front_occ_m": round(front_occ, 2) if front_occ is not None else None,
            "default_obstacle_projection_m": _DEFAULT_OBSTACLE_DIST_M,
        }
        if target_xy is not None:
            out["target_dist_m"] = round(
                math.hypot(target_xy[0] - pose.x, target_xy[1] - pose.y), 2
            )
        return out


def pose_from_status(status: dict[str, Any]) -> Pose2 | None:
    x = _try_float(status.get("drone.x_m"))
    y = _try_float(status.get("drone.y_m"))
    yaw = _try_float(status.get("drone.heading_deg"))
    if None in (x, y, yaw):
        return None
    return Pose2(float(x), float(y), float(yaw))


def target_xy_from_status(status: dict[str, Any]) -> tuple[float, float] | None:
    x = _try_float(status.get("drone.x_m"))
    y = _try_float(status.get("drone.y_m"))
    d_lat = _try_float(status.get("drone.lat_deg"))
    d_lon = _try_float(status.get("drone.lon_deg"))
    t_lat = _try_float(status.get("target.lat_deg"))
    t_lon = _try_float(status.get("target.lon_deg"))
    if None in (x, y, d_lat, d_lon, t_lat, t_lon):
        return None
    lat_mid = math.radians((float(d_lat) + float(t_lat)) / 2.0)
    east = (float(t_lon) - float(d_lon)) * 111_320.0 * math.cos(lat_mid)
    north = (float(t_lat) - float(d_lat)) * 111_320.0
    return (float(x) + east, float(y) - north)


def _command_to_waypoint(pose: Pose2, waypoint: tuple[float, float]) -> dict:
    dx = waypoint[0] - pose.x
    dy = waypoint[1] - pose.y
    dist = math.hypot(dx, dy)
    desired_yaw = math.degrees(math.atan2(dy, dx)) % 360.0
    yaw_error = (desired_yaw - pose.yaw_deg + 180.0) % 360.0 - 180.0
    yaw = _clamp(yaw_error * _YAW_GAIN, -_MAX_YAW_DPS, _MAX_YAW_DPS)
    abs_err = abs(yaw_error)
    if abs_err < _ALIGN_DEG:
        t = _clamp(dist / _WAYPOINT_LOOKAHEAD_M, 0.0, 1.0)
        align = 1.0 - abs_err / _ALIGN_DEG
        forward = (_NAV_MIN_SPEED + t * (_NAV_SPEED - _NAV_MIN_SPEED)) * max(0.35, align)
    elif abs_err < _TURN_ONLY_DEG:
        forward = 0.8
    else:
        forward = 0.0
    return {
        "forward_mps": forward,
        "right_mps": 0.0,
        "up_mps": 0.0,
        "yaw_rate_dps": yaw,
    }


def _obstacle_dist(range_m: float | None) -> float:
    if range_m is None or not math.isfinite(range_m):
        return _DEFAULT_OBSTACLE_DIST_M
    return _clamp(range_m, _MIN_OBSTACLE_DIST_M, _MAX_OBSTACLE_DIST_M)


def _lookahead(path: list[tuple[float, float]],
               pos: tuple[float, float],
               lookahead_m: float) -> tuple[float, float]:
    for p in path[1:]:
        if math.hypot(p[0] - pos[0], p[1] - pos[1]) >= lookahead_m:
            return p
    return path[-1]


def _neighbors(cell: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    x, y = cell
    return (
        (x - 1, y - 1), (x, y - 1), (x + 1, y - 1),
        (x - 1, y),                 (x + 1, y),
        (x - 1, y + 1), (x, y + 1), (x + 1, y + 1),
    )


def _reconstruct(came_from: dict[tuple[int, int], tuple[int, int]],
                 current: tuple[int, int]) -> list[tuple[int, int]]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def _cell_dist(a: tuple[int, int], b: tuple[int, int]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _try_float(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))
