"""Canonical coordinate transforms for waypoint links.

Three frames exist, and every position carries the one it is expressed in:

``map``        east-positive X, south-positive Y, metres above map ground Z.
``local_ned``  north/east/down metres from the geographic origin at map centre.
``global``     WGS84 latitude/longitude with AMSL altitude.

The map/local-NED equations live in :mod:`dvision2_common` because the
simulator consumes them too; everything that projects out to WGS84 lives here.
One module owning both directions is what keeps a sign error from being
rediscovered in each consumer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from dvision2_common import local_ned_to_map, local_to_gps, map_to_local_ned

__all__ = [
    "GeoAnchor", "map_to_local_ned", "local_ned_to_map", "rotate_clockwise",
    "local_ned_to_global", "global_to_local_ned", "map_heading_to_true",
    "true_heading_to_map",
]


@dataclass(frozen=True)
class GeoAnchor:
    """Where a map-frame tour sits on the Earth, and how it is turned.

    ``rotation_deg`` is the clockwise angle from map north to true north, so a
    site surveyed at an angle to the map grid needs no re-authored waypoints.
    Zero leaves the map/local-NED equations exactly as written.
    """

    origin_lat_deg: float
    origin_lon_deg: float
    origin_alt_m: float
    rotation_deg: float = 0.0


def rotate_clockwise(east_m: float, north_m: float,
                     rotation_deg: float) -> tuple[float, float]:
    """Rotate an (east, north) vector clockwise by ``rotation_deg``."""
    angle = math.radians(rotation_deg)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    return (east_m * cos_a + north_m * sin_a,
            north_m * cos_a - east_m * sin_a)


def local_ned_to_global(north_m: float, east_m: float, down_m: float,
                        anchor: GeoAnchor) -> tuple[float, float, float]:
    """Project local NED from ``anchor`` to WGS84 latitude/longitude/AMSL."""
    east_t, north_t = rotate_clockwise(east_m, north_m, anchor.rotation_deg)
    lat, lon, _ = local_to_gps(east_t, north_t, 0.0, anchor.origin_lat_deg,
                               anchor.origin_lon_deg, anchor.origin_alt_m)
    return lat, lon, anchor.origin_alt_m - down_m


def global_to_local_ned(lat_deg: float, lon_deg: float, alt_m: float,
                        anchor: GeoAnchor) -> tuple[float, float, float]:
    """The exact inverse of :func:`local_ned_to_global`."""
    north_t = (lat_deg - anchor.origin_lat_deg) * 111_320.0
    lon_scale = 111_320.0 * max(0.01, math.cos(math.radians(anchor.origin_lat_deg)))
    east_t = (lon_deg - anchor.origin_lon_deg) * lon_scale
    east_m, north_m = rotate_clockwise(east_t, north_t, -anchor.rotation_deg)
    return north_m, east_m, anchor.origin_alt_m - alt_m


def map_heading_to_true(heading_deg: float, anchor: GeoAnchor) -> float:
    """Map-grid compass heading expressed against true north."""
    return (heading_deg + anchor.rotation_deg) % 360.0


def true_heading_to_map(heading_deg: float, anchor: GeoAnchor) -> float:
    return (heading_deg - anchor.rotation_deg) % 360.0
