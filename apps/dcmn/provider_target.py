"""The mission target a vehicle provider publishes, in the planning frame.

dsim puts the map target (the ``*`` cell) on its status plane as GNSS --
``target.lat_deg``/``target.lon_deg`` beside the datum ``origin.lat_deg``/
``origin.lon_deg``/``origin.alt_m`` -- and writes the local NED origin of the
map frame into the neutral session context. Converting the one with the other
needs no world file and no simulator module, so a truth-isolated planner may
use it as a goal the provider handed over, like a waypoint uploaded to a
vehicle. A provider that publishes no target yields ``None`` and a reason.
"""
from __future__ import annotations

import math

from dvision2_common import gps_to_local, load_pymembus, shared_names

TRANSFORM_SCHEMA = 'dvision2.local-ned-transform.v1'


def read_status(instance: str) -> dict[str, str] | None:
    """The provider's status key/value plane, or None when none is published."""
    handle = load_pymembus().memkv()
    if not handle.open(shared_names(instance)['status']):
        return None
    try:
        return dict(handle.getAll())
    finally:
        handle.close()


def map_target(values: dict[str, str] | None, snapshot: dict) -> tuple[list[float] | None, str]:
    """``([x, y], '')`` in map metres (east, south), or ``(None, reason)``."""
    if values is None:
        return None, 'provider status plane unavailable'
    try:
        lat, lon = float(values.get('target.lat_deg') or 'nan'), float(values.get('target.lon_deg') or 'nan')
        lat0 = float(values.get('origin.lat_deg') or 'nan')
        lon0 = float(values.get('origin.lon_deg') or 'nan')
    except ValueError:
        return None, 'provider target is not numeric'
    if not all(math.isfinite(v) for v in (lat, lon)):
        return None, 'provider publishes no map target'
    if not all(math.isfinite(v) for v in (lat0, lon0)):
        return None, 'provider publishes no geographic origin'
    transform = (snapshot or {}).get('vehicle_transform') or {}
    origin = transform.get('origin')
    if (transform.get('schema') != TRANSFORM_SCHEMA or not isinstance(origin, list) or len(origin) != 3
            or transform.get('localization_epoch') != snapshot.get('localization_epoch')):
        return None, 'session context has no current local NED transform'
    east, north, _ = gps_to_local(lat, lon, 0.0, lat0, lon0, 0.0)
    # Local NED from map (x east, y south): north = oy - y, east = x - ox.
    return [float(origin[0]) + east, float(origin[1]) - north], ''
