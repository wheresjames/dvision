"""Tour files: load, validate, convert between frames, and check clearance.

A tour is a file in the repository. There is no store and no database; the only
identity check is the ``map_sha`` that catches a tour flown against the wrong
map.

The loader is deliberately strict about what it will hand to a follower. The
committed tour directory holds three kinds of file -- flyable tours, an
aggregate diagnostics file, and tours explicitly marked ``not_applicable`` for
their map -- and only the first kind can be flown, so the other two are
rejected here with a reason rather than crashing later on a missing key.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dvision2_common import SimMap, load_map
from dway.frames import (
    GeoAnchor, global_to_local_ned, local_ned_to_global, local_ned_to_map,
    map_heading_to_true, map_to_local_ned,
)
from dway.link import Frame, PositionTarget

SUPPORTED_SCHEMA_VERSIONS = (1,)
FRAMES: tuple[Frame, ...] = ("map", "local_ned", "global")

# Arrival gate defaults. A tour may widen them; nothing else may.
DEFAULT_ARRIVAL_SPEED_MPS = 0.15
DEFAULT_HEADING_TOLERANCE_DEG = 5.0
DEFAULT_MAX_STATE_AGE_S = 0.5
DEFAULT_WAYPOINT_TOLERANCE_M = 0.25
DEFAULT_SPEED_MPS = 1.0
DEFAULT_MIN_CLEARANCE_M = 0.4

# Obstacles occupy their whole map cell, exactly as the simulator's collision
# test treats them; a clearance figure computed against a different footprint
# than the one that actually crashes the vehicle would be worse than none.
_OBSTACLE_HALF_EXTENT_M = 0.5
_CLEARANCE_STEP_M = 0.1
_BLOCKING_KINDS = ("wall", "tree")


class TourError(ValueError):
    """A tour file cannot be flown, and the message says why."""


@dataclass(frozen=True)
class Waypoint:
    index: int
    frame: Frame
    heading_deg: float
    dwell_s: float
    # Exactly the trio belonging to ``frame`` is populated.
    x: float | None = None
    y: float | None = None
    z: float | None = None
    north_m: float | None = None
    east_m: float | None = None
    down_m: float | None = None
    lat_deg: float | None = None
    lon_deg: float | None = None
    alt_m: float | None = None

    def target(self, max_speed_mps: float) -> PositionTarget:
        return PositionTarget(
            frame=self.frame, x=self.x, y=self.y, z=self.z,
            north_m=self.north_m, east_m=self.east_m, down_m=self.down_m,
            lat_deg=self.lat_deg, lon_deg=self.lon_deg, alt_m=self.alt_m,
            heading_deg=self.heading_deg, max_speed_mps=max_speed_mps,
        )

    def describe(self) -> dict[str, Any]:
        """The waypoint as it goes into a report: frame plus its own fields."""
        fields = {"map": ("x", "y", "z"),
                  "local_ned": ("north_m", "east_m", "down_m"),
                  "global": ("lat_deg", "lon_deg", "alt_m")}[self.frame]
        described = {"frame": self.frame, "heading_deg": self.heading_deg,
                     "dwell_s": self.dwell_s}
        described.update({name: getattr(self, name) for name in fields})
        return described


@dataclass(frozen=True)
class Tour:
    tour_id: str
    schema_version: int
    coordinate_frame: Frame
    waypoints: tuple[Waypoint, ...]
    path: Path
    map_path: str | None = None
    map_sha: str | None = None
    default_speed_mps: float = DEFAULT_SPEED_MPS
    waypoint_tolerance_m: float = DEFAULT_WAYPOINT_TOLERANCE_M
    arrival_speed_mps: float = DEFAULT_ARRIVAL_SPEED_MPS
    heading_tolerance_deg: float = DEFAULT_HEADING_TOLERANCE_DEG
    max_state_age_s: float = DEFAULT_MAX_STATE_AGE_S
    min_clearance_m: float = DEFAULT_MIN_CLEARANCE_M
    settle_s: float = 0.0
    leg_timeout_s: float | None = None
    geo_anchor: GeoAnchor | None = None

    def leg_timeout(self, distance_m: float, speed_mps: float) -> float:
        if self.leg_timeout_s is not None:
            return self.leg_timeout_s
        return max(10.0, 3.0 * distance_m / max(speed_mps, 1e-6))


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _number(payload: dict, key: str, *, where: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TourError(f"{where}: {key} must be a number")
    value = float(value)
    if not math.isfinite(value):
        raise TourError(f"{where}: {key} must be finite")
    return value


def _optional_number(payload: dict, key: str, default: float | None, *,
                     where: str) -> float | None:
    if key not in payload or payload[key] is None:
        return default
    return _number(payload, key, where=where)


_FRAME_FIELDS: dict[str, tuple[str, str, str]] = {
    "map": ("x", "y", "z"),
    "local_ned": ("north_m", "east_m", "down_m"),
    "global": ("lat_deg", "lon_deg", "alt_m"),
}


def _parse_waypoint(payload: Any, index: int, frame: Frame) -> Waypoint:
    where = f"waypoint {index}"
    if not isinstance(payload, dict):
        raise TourError(f"{where}: must be an object")
    fields = _FRAME_FIELDS[frame]
    foreign = [name for other, names in _FRAME_FIELDS.items() if other != frame
               for name in names if name in payload and name not in fields]
    if foreign:
        raise TourError(
            f"{where}: {', '.join(sorted(foreign))} do not belong to the "
            f"{frame} frame")
    values = {name: _number(payload, name, where=where) for name in fields}
    dwell = _optional_number(payload, "dwell_s", 0.0, where=where)
    if dwell < 0.0:
        raise TourError(f"{where}: dwell_s must not be negative")
    return Waypoint(
        index=index, frame=frame,
        heading_deg=_optional_number(payload, "heading_deg", 0.0,
                                     where=where) % 360.0,
        dwell_s=dwell, **values)


def _parse_geo_anchor(payload: Any) -> GeoAnchor:
    if not isinstance(payload, dict):
        raise TourError("geo_anchor: must be an object")
    required = ("origin_lat_deg", "origin_lon_deg", "origin_alt_m")
    missing = [key for key in required if key not in payload]
    if missing:
        # Older shorthand keys are deliberately not accepted: a silently
        # defaulted anchor puts a tour over the wrong piece of ground.
        raise TourError(f"geo_anchor: missing {', '.join(missing)}")
    return GeoAnchor(
        origin_lat_deg=_number(payload, "origin_lat_deg", where="geo_anchor"),
        origin_lon_deg=_number(payload, "origin_lon_deg", where="geo_anchor"),
        origin_alt_m=_number(payload, "origin_alt_m", where="geo_anchor"),
        rotation_deg=_optional_number(payload, "rotation_deg", 0.0,
                                      where="geo_anchor"),
    )


def parse_tour(payload: Any, *, path: Path) -> Tour:
    if not isinstance(payload, dict):
        raise TourError(f"{path}: not a tour object")
    if "tours" in payload and "waypoints" not in payload:
        raise TourError(f"{path}: aggregate diagnostics file, not a tour")
    version = payload.get("schema_version")
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        raise TourError(f"{path}: unsupported schema_version {version!r}")
    status = payload.get("status")
    if status not in (None, "applicable"):
        raise TourError(f"{path}: tour status is {status!r}, not applicable")
    frame = payload.get("coordinate_frame", "map")
    if frame not in FRAMES:
        raise TourError(f"{path}: unsupported coordinate_frame {frame!r}")
    raw_waypoints = payload.get("waypoints")
    if not isinstance(raw_waypoints, list) or not raw_waypoints:
        raise TourError(f"{path}: tour has no waypoints")
    waypoints = tuple(_parse_waypoint(item, index, frame)
                      for index, item in enumerate(raw_waypoints))

    tour_id = payload.get("tour_id")
    if not isinstance(tour_id, str) or not tour_id:
        raise TourError(f"{path}: missing tour_id")
    map_path = payload.get("map")
    if map_path is not None and not isinstance(map_path, str):
        raise TourError(f"{path}: map must be a path string")
    map_sha = payload.get("map_sha")
    if map_sha is not None and not isinstance(map_sha, str):
        raise TourError(f"{path}: map_sha must be a string")
    if frame == "map" and not map_path:
        raise TourError(f"{path}: a map-frame tour must name its map")

    where = str(path)
    speed = _optional_number(payload, "default_speed_mps", DEFAULT_SPEED_MPS,
                             where=where)
    if speed <= 0.0:
        raise TourError(f"{path}: default_speed_mps must be positive")
    tolerance = _optional_number(payload, "waypoint_tolerance_m",
                                 DEFAULT_WAYPOINT_TOLERANCE_M, where=where)
    if tolerance <= 0.0:
        raise TourError(f"{path}: waypoint_tolerance_m must be positive")
    leg_timeout = _optional_number(payload, "leg_timeout_s", None, where=where)
    if leg_timeout is not None and leg_timeout <= 0.0:
        raise TourError(f"{path}: leg_timeout_s must be positive")
    # The arrival gates were previously taken on trust. A tour is a file, and a
    # negative gate is not a tighter one: it can never be satisfied, so the
    # flight fails at the first waypoint with a message about vehicle health
    # rather than about the tour that asked for the impossible.
    gates = {
        "arrival_speed_mps": (_optional_number(
            payload, "arrival_speed_mps", DEFAULT_ARRIVAL_SPEED_MPS,
            where=where), True),
        "heading_tolerance_deg": (_optional_number(
            payload, "heading_tolerance_deg", DEFAULT_HEADING_TOLERANCE_DEG,
            where=where), True),
        "max_state_age_s": (_optional_number(
            payload, "max_state_age_s", DEFAULT_MAX_STATE_AGE_S,
            where=where), True),
        "min_clearance_m": (_optional_number(
            payload, "min_clearance_m", DEFAULT_MIN_CLEARANCE_M,
            where=where), False),
        "settle_s": (_optional_number(payload, "settle_s", 0.0,
                                      where=where), False),
    }
    for key, (value, strictly_positive) in gates.items():
        if strictly_positive and value <= 0.0:
            raise TourError(f"{path}: {key} must be positive")
        if not strictly_positive and value < 0.0:
            raise TourError(f"{path}: {key} must not be negative")
    anchor = (_parse_geo_anchor(payload["geo_anchor"])
              if payload.get("geo_anchor") is not None else None)

    return Tour(
        tour_id=tour_id, schema_version=int(version), coordinate_frame=frame,
        waypoints=waypoints, path=path, map_path=map_path, map_sha=map_sha,
        default_speed_mps=speed, waypoint_tolerance_m=tolerance,
        arrival_speed_mps=gates["arrival_speed_mps"][0],
        heading_tolerance_deg=gates["heading_tolerance_deg"][0],
        max_state_age_s=gates["max_state_age_s"][0],
        min_clearance_m=gates["min_clearance_m"][0],
        settle_s=gates["settle_s"][0],
        leg_timeout_s=leg_timeout, geo_anchor=anchor,
    )


def load_tour(path: str | Path) -> Tour:
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise TourError(f"{path}: no such tour file") from exc
    except json.JSONDecodeError as exc:
        raise TourError(f"{path}: invalid JSON ({exc})") from exc
    return parse_tour(payload, path=path)


def save_tour(tour: Tour, path: str | Path) -> Path:
    """Write a tour back out in the committed schema's shape."""
    path = Path(path)
    payload: dict[str, Any] = {
        "schema_version": tour.schema_version,
        "tour_id": tour.tour_id,
        "status": "applicable",
        "coordinate_frame": tour.coordinate_frame,
        "default_speed_mps": tour.default_speed_mps,
        "waypoint_tolerance_m": tour.waypoint_tolerance_m,
        "arrival_speed_mps": tour.arrival_speed_mps,
        "heading_tolerance_deg": tour.heading_tolerance_deg,
        "min_clearance_m": tour.min_clearance_m,
        "settle_s": tour.settle_s,
        "waypoints": [
            {k: v for k, v in point.describe().items() if k != "frame"}
            for point in tour.waypoints
        ],
    }
    if tour.map_path:
        payload["map"] = tour.map_path
    if tour.map_sha:
        payload["map_sha"] = tour.map_sha
    if tour.leg_timeout_s is not None:
        payload["leg_timeout_s"] = tour.leg_timeout_s
    if tour.geo_anchor is not None:
        payload["geo_anchor"] = {
            "origin_lat_deg": tour.geo_anchor.origin_lat_deg,
            "origin_lon_deg": tour.geo_anchor.origin_lon_deg,
            "origin_alt_m": tour.geo_anchor.origin_alt_m,
            "rotation_deg": tour.geo_anchor.rotation_deg,
        }
    path.write_text(json.dumps(payload, sort_keys=True,
                               separators=(",", ":")), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Map identity
# ---------------------------------------------------------------------------

def map_content_sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve_map(tour: Tour, root: Path) -> Path | None:
    """The map this tour was authored against, resolved from the repo root."""
    if not tour.map_path:
        return None
    candidate = Path(tour.map_path)
    return candidate if candidate.is_absolute() else root / candidate


def verify_map(tour: Tour, root: Path) -> Path | None:
    """Check the tour's map hash, raising :class:`TourError` on a mismatch.

    Local-NED and global tours need no map, but a hash they do carry is still
    checked: a stated expectation that is not tested is worse than none.
    """
    map_file = resolve_map(tour, root)
    if map_file is None:
        return None
    if not map_file.exists():
        raise TourError(f"{tour.tour_id}: map {map_file} is missing")
    if tour.map_sha:
        actual = map_content_sha(map_file)
        if actual != tour.map_sha:
            raise TourError(
                f"{tour.tour_id}: map hash mismatch for {map_file} "
                f"(tour {tour.map_sha[:12]}, file {actual[:12]})")
    return map_file


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def _obstacle_distance_m(sim_map: SimMap, x: float, y: float) -> float:
    """Distance from a point to the nearest obstacle cell, 0.0 when inside."""
    nearest = math.inf
    for obj in sim_map.objects:
        if obj.kind not in _BLOCKING_KINDS:
            continue
        dx = max(0.0, abs(obj.x - x) - _OBSTACLE_HALF_EXTENT_M)
        dy = max(0.0, abs(obj.y - y) - _OBSTACLE_HALF_EXTENT_M)
        nearest = min(nearest, math.hypot(dx, dy))
    return nearest


def leg_clearance_m(sim_map: SimMap, start: tuple[float, float],
                    end: tuple[float, float]) -> float:
    """The tightest obstacle clearance anywhere along one straight leg."""
    distance = math.hypot(end[0] - start[0], end[1] - start[1])
    steps = max(1, math.ceil(distance / _CLEARANCE_STEP_M))
    worst = math.inf
    for step in range(steps + 1):
        t = step / steps
        worst = min(worst, _obstacle_distance_m(
            sim_map, start[0] + (end[0] - start[0]) * t,
            start[1] + (end[1] - start[1]) * t))
    return worst


@dataclass(frozen=True)
class LegClearance:
    index: int            # 0 is the leg from the current pose to waypoint 0
    start: tuple[float, float]
    end: tuple[float, float]
    clearance_m: float

    @property
    def obstructed(self) -> bool:
        return self.clearance_m <= 0.0


def leg_clearances(tour: Tour, sim_map: SimMap,
                   start_pose: tuple[float, float] | None) -> list[LegClearance]:
    """Clearance for every leg, leg zero included when a pose is given.

    Starting away from the first waypoint is valid, and that movement is as
    collision-prone as any other leg, so it is measured here rather than
    assumed safe.
    """
    context = FrameContext.for_map(sim_map, tour.geo_anchor)
    points: list[tuple[float, float]] = []
    if start_pose is not None:
        points.append(start_pose)
    for waypoint in tour.waypoints:
        x, y, _ = context.ned_to_map(*context.waypoint_ned(waypoint))
        points.append((x, y))
    first_index = 0 if start_pose is not None else 1
    return [
        LegClearance(index=first_index + i, start=points[i], end=points[i + 1],
                     clearance_m=leg_clearance_m(sim_map, points[i], points[i + 1]))
        for i in range(len(points) - 1)
    ]


@dataclass(frozen=True)
class FrameContext:
    """Everything needed to move one tour between the three frames.

    Every conversion in ``dway`` goes through here, so a waypoint, a vehicle
    pose and a plotted track cannot disagree about which way north is.
    """

    width_m: float
    height_m: float
    anchor: GeoAnchor | None = None

    @classmethod
    def for_map(cls, sim_map: SimMap | None, anchor: GeoAnchor | None = None) -> "FrameContext":
        if sim_map is None:
            return cls(0.0, 0.0, anchor)
        return cls(float(sim_map.width), float(sim_map.height), anchor)

    # -- map <-> local NED ---------------------------------------------

    def map_to_ned(self, x: float, y: float, z: float) -> tuple[float, float, float]:
        return map_to_local_ned(x, y, z, self.width_m, self.height_m)

    def ned_to_map(self, north_m: float, east_m: float,
                   down_m: float) -> tuple[float, float, float]:
        return local_ned_to_map(north_m, east_m, down_m, self.width_m, self.height_m)

    # -- waypoints and poses -------------------------------------------

    def waypoint_ned(self, waypoint: Waypoint) -> tuple[float, float, float]:
        """One waypoint as north/east/down, whatever frame it was authored in.

        Local NED is the follower's working frame: it is the frame a real
        autopilot uses, and distances measured in it are the same distances the
        map frame measures, so nothing downstream has to know which frame the
        file used.
        """
        if waypoint.frame == "local_ned":
            return (float(waypoint.north_m), float(waypoint.east_m),
                    float(waypoint.down_m))
        if waypoint.frame == "map":
            return self.map_to_ned(float(waypoint.x), float(waypoint.y),
                                   float(waypoint.z))
        if self.anchor is None:
            raise TourError("a global tour needs a geo_anchor to be flown")
        return global_to_local_ned(float(waypoint.lat_deg), float(waypoint.lon_deg),
                                   float(waypoint.alt_m), self.anchor)

    def position_ned(self, position: PositionTarget) -> tuple[float, float, float]:
        """A published vehicle position as north/east/down."""
        if position.frame == "local_ned":
            return (float(position.north_m), float(position.east_m),
                    float(position.down_m))
        if position.frame == "map":
            return self.map_to_ned(float(position.x), float(position.y),
                                   float(position.z))
        if self.anchor is None:
            raise TourError("a global vehicle position needs a geo_anchor")
        return global_to_local_ned(float(position.lat_deg), float(position.lon_deg),
                                   float(position.alt_m), self.anchor)

    def target_from_ned(self, north_m: float, east_m: float, down_m: float, *,
                        frame: Frame, heading_deg: float,
                        max_speed_mps: float) -> PositionTarget:
        """Build a setpoint in ``frame`` from a working-frame position."""
        if frame == "local_ned":
            return PositionTarget(frame=frame, north_m=north_m, east_m=east_m,
                                  down_m=down_m, heading_deg=heading_deg,
                                  max_speed_mps=max_speed_mps)
        if frame == "map":
            x, y, z = self.ned_to_map(north_m, east_m, down_m)
            return PositionTarget(frame=frame, x=x, y=y, z=z,
                                  heading_deg=heading_deg,
                                  max_speed_mps=max_speed_mps)
        if self.anchor is None:
            raise TourError("a global setpoint needs a geo_anchor")
        lat, lon, alt = local_ned_to_global(north_m, east_m, down_m, self.anchor)
        return PositionTarget(frame=frame, lat_deg=lat, lon_deg=lon, alt_m=alt,
                              heading_deg=map_heading_to_true(heading_deg, self.anchor),
                              max_speed_mps=max_speed_mps)


def load_tour_map(tour: Tour, root: Path) -> SimMap | None:
    map_file = verify_map(tour, root)
    return None if map_file is None else load_map(map_file)
