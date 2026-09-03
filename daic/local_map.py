"""Vision-built local occupancy map and A* route planner for daic."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Any

from daic.orb_slam3_detector import (ObstacleSectors, _INNER_DEG,
                                     _OUTER_DEG)


_CELL_M = 0.5
_HALF_WIDTH_M = 14.0
_OCCUPIED = 1.6
_FREE = -1.2
_OCC_HIT = 1.0
_FREE_HIT = -0.35
_DECAY = 0.995
_FLOW_SOFT_DECAY = 0.92
_FLOW_CANDIDATE_EXPIRE_TICKS = 6
_FLOW_CONFIRM_HITS = 3
_FLOW_HARD_CONFIRM_HITS = 6
_FLOW_OCC_HIT_SCALE = 0.35
_FLOW_SOFT_OCCUPIED_MAX = 1.2
# Phase 6.8: a strong front-risk signal that stays high for several consecutive
# ticks while the drone holds a steady heading is a real wall ahead, even when it
# is rangeless (mini-SLAM / flow:persist often report high front risk with no
# range). Map it as a hard, sticky barrier at a conservative near distance so A*
# routes around it instead of plowing through. Gated on a *steady heading* (not
# yaw-scanning/turning) rather than forward speed: reactive avoidance brakes the
# drone to a near-crawl right in front of a wall, so a speed gate misses exactly
# the moment we need; but a scanning/turning drone sweeping past obstacles must
# not paint a phantom 2 m halo (the Phase 6.2 failure), and that is what the
# heading-stability gate excludes.
_SUSTAINED_RISK = 0.6
_SUSTAINED_CONFIRM_TICKS = 5
_SUSTAINED_DIST_M = 2.0
_SUSTAINED_MARK_STRENGTH = 2.0
# Per-tick heading change below this is "steady". The yaw-scan / max route-turn
# rate is _MAX_YAW_DPS=18 deg/s ~= 0.6 deg/tick at 30 fps, so 0.3 cleanly
# separates a steady approach (yaw ~= 0) from an active turn or scan.
_SUSTAINED_MAX_YAW_STEP_DEG = 0.3
_DEFAULT_OBSTACLE_DIST_M = 3.0
_MIN_OBSTACLE_DIST_M = 0.7
_MAX_OBSTACLE_DIST_M = 8.0
_OBSTACLE_SPREAD_M = 0.8

# Where each sector's observations are planted, as a bearing relative to the
# heading. The detectors bin points by azimuth into bands that run out to +/-90
# deg, but the camera only ever sees +/-_CAMERA_HALF_FOV_DEG and nothing beyond
# it, so the sector bands are the shared _INNER_DEG/_OUTER_DEG split clipped to
# the field of view, and each sector is planted at the centre of its own band.
# Planting the left/right sectors at +/-70 deg - as this table previously did -
# places every off-axis observation about 40 deg outside the field of view that
# produced it, which smears a wall into a wide arc of phantom cells instead of
# a barrier.
_CAMERA_HALF_FOV_DEG = 35.0   # dsim Panda3DRenderer.CAM_FOV_H = 70.0


def _sector_band(lo_deg: float, hi_deg: float) -> tuple[float, float]:
    """Return (centre, half_width) of a sector band clipped to the camera FOV."""
    lo = max(lo_deg, -_CAMERA_HALF_FOV_DEG)
    hi = min(hi_deg, _CAMERA_HALF_FOV_DEG)
    return ((lo + hi) / 2.0, max((hi - lo) / 2.0, 2.5))


_SECTOR_BANDS = {
    "left":        _sector_band(-90.0, -_OUTER_DEG),
    "front_left":  _sector_band(-_OUTER_DEG, -_INNER_DEG),
    "front":       _sector_band(-_INNER_DEG, _INNER_DEG),
    "front_right": _sector_band(_INNER_DEG, _OUTER_DEG),
    "right":       _sector_band(_OUTER_DEG, 90.0),
}
_WAYPOINT_LOOKAHEAD_M = 2.0
_MAX_YAW_DPS = 18.0
_YAW_GAIN = 0.45
_NAV_SPEED = 0.45
_NAV_MIN_SPEED = 0.15
_ALIGN_DEG = 30.0
_TURN_ONLY_DEG = 60.0


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


@dataclass(frozen=True)
class CellProvenance:
    source: str
    sector: str
    ranged: bool
    last_hit_tick: int
    hit_count: int = 1

    def to_dict(self, current_tick: int, value: float | None = None) -> dict:
        out = {
            "source": self.source,
            "sector": self.sector,
            "ranged": self.ranged,
            "last_hit_tick": self.last_hit_tick,
            "age_ticks": max(0, current_tick - self.last_hit_tick),
            "hit_count": self.hit_count,
        }
        if value is not None:
            out["value"] = round(value, 3)
        return out


@dataclass(frozen=True)
class FlowCandidate:
    hits: int
    last_tick: int


class LocalOccupancyMap:
    """Rolling world-frame occupancy grid built from vision obstacle sectors."""

    def __init__(self, cell_m: float = _CELL_M,
                 half_width_m: float = _HALF_WIDTH_M) -> None:
        self.cell_m = cell_m
        self.half_width_m = half_width_m
        self._cells: dict[tuple[int, int], float] = {}
        self._provenance: dict[tuple[int, int], CellProvenance] = {}
        self._flow_candidates: dict[tuple[tuple[int, int], str], FlowCandidate] = {}
        self._sustained_hits: dict[str, int] = {}
        self._prev_pose: Pose2 | None = None
        self._tick = 0
        self.last_path: list[tuple[float, float]] = []

    def update(self, pose: Pose2, sectors: ObstacleSectors) -> None:
        self._tick += 1
        heading_steady = self._update_heading_steady(pose)
        self._decay_and_prune(pose)
        self._mark_free_fan(pose)
        if sectors.confidence <= 0.0:
            self._reset_sustained()
            return

        sector_defs = [
            ("left", _SECTOR_BANDS["left"][0],
             sectors.left, sectors.left_range_m),
            ("front_left", _SECTOR_BANDS["front_left"][0],
             sectors.front_left, sectors.front_left_range_m),
            ("front", _SECTOR_BANDS["front"][0],
             sectors.front, sectors.front_range_m),
            ("front_right", _SECTOR_BANDS["front_right"][0],
             sectors.front_right, sectors.front_right_range_m),
            ("right", _SECTOR_BANDS["right"][0],
             sectors.right, sectors.right_range_m),
        ]
        for sector_name, rel_deg, risk, range_m in sector_defs:
            if risk < 0.12:
                continue
            if _is_flow_method(sectors.method):
                self._record_flow_candidate(
                    pose, rel_deg, risk * sectors.confidence, range_m,
                    sectors.method, sector_name,
                )
                continue
            if not _should_map_sector(sectors.method, range_m):
                continue
            self._mark_obstacle_sector(
                pose, rel_deg, risk * sectors.confidence, range_m,
                sectors.method, sector_name,
            )

        self._update_sustained_front(pose, sectors, heading_steady)

    def _update_heading_steady(self, pose: Pose2) -> bool:
        """Return whether the drone is holding a steady heading (not turning).

        Reactive avoidance often brakes the drone to a near-crawl in front of a
        wall, so forward speed is an unreliable "confronting a wall" cue. A
        steady heading is reliable: it is True for a straight approach or a hover
        facing an obstacle, and False while yaw-scanning or turning to detour
        (which is exactly when a fresh sustained mark would smear into a halo).
        """
        prev = self._prev_pose
        self._prev_pose = pose
        if prev is None:
            return False
        dyaw = abs((pose.yaw_deg - prev.yaw_deg + 180.0) % 360.0 - 180.0)
        return dyaw <= _SUSTAINED_MAX_YAW_STEP_DEG

    def _reset_sustained(self) -> None:
        self._sustained_hits = {}

    def _update_sustained_front(self, pose: Pose2,
                                sectors: ObstacleSectors,
                                heading_steady: bool) -> None:
        front_defs = (
            ("front", _SECTOR_BANDS["front"][0], sectors.front),
            ("front_left", _SECTOR_BANDS["front_left"][0], sectors.front_left),
            ("front_right", _SECTOR_BANDS["front_right"][0], sectors.front_right),
        )
        for sector_name, rel_deg, risk in front_defs:
            if heading_steady and risk >= _SUSTAINED_RISK:
                hits = self._sustained_hits.get(sector_name, 0) + 1
            else:
                hits = 0
            self._sustained_hits[sector_name] = hits
            if hits < _SUSTAINED_CONFIRM_TICKS:
                continue
            # Commit a hard, sticky barrier at a conservative near distance. A
            # single mark exceeds _OCCUPIED so A* sees it as impassable, and the
            # normal decay rate (sustained_front: prefix is not soft-flow) keeps
            # it alive long enough for A* to commit to and finish a detour.
            self._mark_obstacle_sector(
                pose, rel_deg, _SUSTAINED_MARK_STRENGTH, _SUSTAINED_DIST_M,
                f"sustained_front:{sectors.method}", sector_name,
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
                              range_m: float | None,
                              source: str,
                              sector: str,
                              *,
                              hit_scale: float = 1.0,
                              max_value: float | None = None) -> None:
        dist = _obstacle_dist(range_m)
        radial_spread = min(_OBSTACLE_SPREAD_M, max(0.15, dist * 0.2))
        for off_deg in (-10.0, 0.0, 10.0):
            for off_m in (-radial_spread, 0.0, radial_spread):
                self._add_relative(
                    pose,
                    rel_deg + off_deg,
                    max(_MIN_OBSTACLE_DIST_M, dist + off_m),
                    _OCC_HIT * strength * hit_scale,
                    source=source,
                    sector=sector,
                    ranged=range_m is not None and math.isfinite(range_m),
                    max_value=max_value,
                )

    def _record_flow_candidate(self, pose: Pose2,
                               rel_deg: float, strength: float,
                               range_m: float | None,
                               source: str,
                               sector: str) -> None:
        if sector not in ("front", "front_left", "front_right"):
            return
        if range_m is None or not math.isfinite(range_m):
            return

        cell = self._relative_cell(pose, rel_deg, _obstacle_dist(range_m))
        key = (cell, sector)
        prev = self._flow_candidates.get(key)
        if prev is None or self._tick - prev.last_tick > _FLOW_CANDIDATE_EXPIRE_TICKS:
            candidate = FlowCandidate(hits=1, last_tick=self._tick)
        else:
            candidate = FlowCandidate(hits=prev.hits + 1, last_tick=self._tick)
        self._flow_candidates[key] = candidate

        if candidate.hits < _FLOW_CONFIRM_HITS:
            return

        # A few confirmations make a soft (traversable, fast-decaying) cell; once
        # the same world cell+sector keeps reporting a ranged front obstacle, it
        # is strong enough to be a real wall, so promote it to hard occupancy A*
        # must route around rather than a high soft cost it can plough through
        # (Phase 6.7). Hard marks use a distinct provenance prefix so they decay
        # at the normal rate (_cell_decay), not the fast soft-flow rate.
        if candidate.hits >= _FLOW_HARD_CONFIRM_HITS:
            self._mark_obstacle_sector(
                pose,
                rel_deg,
                strength,
                range_m,
                f"confirmed_flow_hard:{source}",
                sector,
            )
            return

        self._mark_obstacle_sector(
            pose,
            rel_deg,
            strength,
            range_m,
            f"confirmed_flow:{source}",
            sector,
            hit_scale=_FLOW_OCC_HIT_SCALE,
            max_value=_FLOW_SOFT_OCCUPIED_MAX,
        )

    def _add_relative(self, pose: Pose2,
                      rel_deg: float, dist_m: float, delta: float,
                      source: str | None = None,
                      sector: str | None = None,
                      ranged: bool | None = None,
                      max_value: float | None = None) -> None:
        cell = self._relative_cell(pose, rel_deg, dist_m)
        prev_value = self._cells.get(cell, 0.0)
        value = _clamp(prev_value + delta, -3.0, 3.0)
        soft_hit_saturated = False
        if max_value is not None and delta > 0.0:
            if prev_value >= max_value:
                value = prev_value
                soft_hit_saturated = True
            else:
                value = min(value, max_value)
        self._cells[cell] = value
        if (
            delta > 0.0
            and not soft_hit_saturated
            and source is not None
            and sector is not None
            and ranged is not None
        ):
            prev = self._provenance.get(cell)
            hit_count = 1 if prev is None else prev.hit_count + 1
            self._provenance[cell] = CellProvenance(
                source=source,
                sector=sector,
                ranged=ranged,
                last_hit_tick=self._tick,
                hit_count=hit_count,
            )

    def _relative_cell(self, pose: Pose2, rel_deg: float, dist_m: float) -> tuple[int, int]:
        yaw = math.radians(pose.yaw_deg + rel_deg)
        x = pose.x + math.cos(yaw) * dist_m
        y = pose.y + math.sin(yaw) * dist_m
        return self._cell(x, y)

    def _decay_and_prune(self, pose: Pose2) -> None:
        keep: dict[tuple[int, int], float] = {}
        max_cells = int(self.half_width_m / self.cell_m)
        pc = self._cell(pose.x, pose.y)
        for cell, value in self._cells.items():
            if abs(cell[0] - pc[0]) > max_cells or abs(cell[1] - pc[1]) > max_cells:
                continue
            value *= self._cell_decay(cell)
            if abs(value) > 0.05:
                keep[cell] = value
        self._cells = keep
        self._provenance = {
            cell: prov for cell, prov in self._provenance.items()
            if cell in self._cells
        }
        self._flow_candidates = {
            key: candidate
            for key, candidate in self._flow_candidates.items()
            if (
                abs(key[0][0] - pc[0]) <= max_cells
                and abs(key[0][1] - pc[1]) <= max_cells
                and self._tick - candidate.last_tick <= _FLOW_CANDIDATE_EXPIRE_TICKS
            )
        }

    def _cell_decay(self, cell: tuple[int, int]) -> float:
        # Only *soft* confirmed-flow cells ("confirmed_flow:") decay fast.
        # Hard-promoted flow walls ("confirmed_flow_hard:") and non-flow
        # obstacles use the normal rate so a confirmed wall survives long enough
        # for A* to commit to and complete a detour around it.
        prov = self._provenance.get(cell)
        if prov is not None and prov.source.startswith("confirmed_flow:"):
            return _FLOW_SOFT_DECAY
        return _DECAY

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
            "provenance": {
                cell: prov.to_dict(self._tick, self._cells.get(cell))
                for cell, prov in self._provenance.items()
            },
            "path": list(self.last_path),
        }

    def diagnostics(self, pose: Pose2,
                    target_xy: tuple[float, float] | None = None) -> dict:
        occupied: list[tuple[float, float, float, tuple[int, int]]] = []
        free_count = 0
        for cell, value in self._cells.items():
            if value >= 0.2:
                wx, wy = self._world(cell)
                occupied.append((wx, wy, value, cell))
            elif value <= -0.2:
                free_count += 1

        nearest_occ = None
        nearest_cell = None
        front_occ = None
        front_cell = None
        front_block_occ = None
        front_block_cell = None
        yaw = math.radians(pose.yaw_deg)
        fwd_x, fwd_y = math.cos(yaw), math.sin(yaw)
        for wx, wy, _value, cell in occupied:
            dx = wx - pose.x
            dy = wy - pose.y
            dist = math.hypot(dx, dy)
            if nearest_occ is None or dist < nearest_occ:
                nearest_occ = dist
                nearest_cell = cell
            along = dx * fwd_x + dy * fwd_y
            lateral = abs(-dx * fwd_y + dy * fwd_x)
            if along > 0.0 and lateral <= 1.5:
                if front_occ is None or along < front_occ:
                    front_occ = along
                    front_cell = cell
                prov = self._provenance.get(cell)
                if _is_front_blocking_provenance(prov):
                    if front_block_occ is None or along < front_block_occ:
                        front_block_occ = along
                        front_block_cell = cell

        out = {
            "cells": len(self._cells),
            "occupied_cells": len(occupied),
            "free_cells": free_count,
            "path_len": len(self.last_path),
            "nearest_occ_m": round(nearest_occ, 2) if nearest_occ is not None else None,
            "front_occ_m": round(front_occ, 2) if front_occ is not None else None,
            "front_block_occ_m": (
                round(front_block_occ, 2) if front_block_occ is not None else None
            ),
            "front_block_occ_age_ticks": self._cell_age_ticks(front_block_cell),
            "default_obstacle_projection_m": _DEFAULT_OBSTACLE_DIST_M,
        }
        nearest_prov = self._diagnostic_provenance(nearest_cell)
        front_prov = self._diagnostic_provenance(front_cell)
        front_block_prov = self._diagnostic_provenance(front_block_cell)
        if nearest_prov is not None:
            out["nearest_occ_source"] = nearest_prov
        if front_prov is not None:
            out["front_occ_source"] = front_prov
        if front_block_prov is not None:
            out["front_block_occ_source"] = front_block_prov
        out["occupied_by_source"] = self._occupied_source_counts(occupied)
        if target_xy is not None:
            out["target_dist_m"] = round(
                math.hypot(target_xy[0] - pose.x, target_xy[1] - pose.y), 2
            )
        return out

    def _cell_age_ticks(self, cell: tuple[int, int] | None) -> int | None:
        """Ticks since this cell was last hit, or None if it has no provenance."""
        if cell is None:
            return None
        prov = self._provenance.get(cell)
        if prov is None:
            return None
        return max(0, self._tick - prov.last_hit_tick)

    def _diagnostic_provenance(self, cell: tuple[int, int] | None) -> dict | None:
        if cell is None:
            return None
        prov = self._provenance.get(cell)
        if prov is None:
            return None
        return prov.to_dict(self._tick, self._cells.get(cell))

    def _occupied_source_counts(
        self,
        occupied: list[tuple[float, float, float, tuple[int, int]]],
    ) -> dict[str, int]:
        counts: dict[str, int] = {}
        for _wx, _wy, _value, cell in occupied:
            prov = self._provenance.get(cell)
            if prov is None:
                key = "unknown"
            else:
                ranged = "ranged" if prov.ranged else "default"
                key = f"{prov.source}|{prov.sector}|{ranged}"
            counts[key] = counts.get(key, 0) + 1
        return counts


def pose_from_status(status: dict[str, Any]) -> Pose2 | None:
    x = _try_float(status.get("drone.x_m"))
    y = _try_float(status.get("drone.y_m"))
    heading = _try_float(status.get("drone.heading_deg"))
    if None in (x, y, heading):
        return None
    return Pose2(float(x), float(y), _heading_to_map_yaw(float(heading)))


def _heading_to_map_yaw(heading_deg: float) -> float:
    """Convert simulator heading (0=north, clockwise) to map yaw.

    Local-map geometry uses the map's x/y axes with mathematical atan2:
    0 degrees is +x/east and 270 degrees is -y/north. Since dsim now publishes
    `drone.heading_deg` as compass-style navigation heading, convert it before
    route following or front-occupancy diagnostics use it.
    """
    return (heading_deg + 270.0) % 360.0


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
    # Public positive yaw is a clockwise/right turn, which is also an increase
    # in compass heading and the corresponding turn toward positive error.
    yaw = _clamp(yaw_error * _YAW_GAIN, -_MAX_YAW_DPS, _MAX_YAW_DPS)
    abs_err = abs(yaw_error)
    if abs_err < _ALIGN_DEG:
        t = _clamp(dist / _WAYPOINT_LOOKAHEAD_M, 0.0, 1.0)
        align = 1.0 - abs_err / _ALIGN_DEG
        forward = (_NAV_MIN_SPEED + t * (_NAV_SPEED - _NAV_MIN_SPEED)) * align
    elif abs_err < _TURN_ONLY_DEG:
        forward = 0.0
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


def _should_map_sector(method: str, range_m: float | None) -> bool:
    """Return whether a sector has enough evidence to seed A* occupancy.

    Ranged detections make a concrete spatial claim, so they can be placed in
    the local map. Rangeless mini-SLAM sectors are weaker evidence: Phase 6.2
    found they can fire near-continuously and, if planted at the 3 m default,
    create a trailing halo of phantom obstacles. They still flow through the
    reactive avoidance path; they just do not become compact A* blockers.
    """
    if _is_flow_method(method):
        return False
    if range_m is not None and math.isfinite(range_m):
        return True
    return "mini_slam:" not in (method or "")


def _is_flow_method(method: str) -> bool:
    return "flow:" in (method or "")


def _is_front_blocking_provenance(prov: CellProvenance | None) -> bool:
    if prov is None:
        return True
    return prov.sector in ("front", "front_left", "front_right")


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
