#!/usr/bin/env python3

from __future__ import annotations

import json
import math
import re
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
COMMAND_MAGIC = "dvision2.command.v1"

STATUS_KEYS = [
    "sim.id",
    "sim.map",
    "sim.time_s",
    "sim.report_dir",
    "sim.camera_in_geometry",
    "vehicle.type",
    "vehicle.frames",
    "vehicle.accepts_position",
    "vehicle.accepts_velocity",
    "vehicle.accepts_attitude",
    "vehicle.supports_missions",
    "vehicle.setpoint_timeout_s",
    "vehicle.max_speed_mps",
    "vehicle.max_accel_mps2",
    "gps.fix_type",
    "gps.satellites",
    "gps.hdop",
    "gps.vdop",
    "est.attitude_valid",
    "est.local_position_valid",
    "est.global_position_valid",
    "est.velocity_valid",
    "wind.speed_mps",
    "wind.dir_deg",
    "wind.gust_mps",
    "geofence.box",
    "geofence.action",
    "realism.telemetry_latency_ms",
    "realism.telemetry_jitter_ms",
    "realism.sensor_noise",
    "realism.battery_failsafe_pct",
    "realism.battery_drain_pct_s",
    "realism.seed",
    "origin.lat_deg",
    "origin.lon_deg",
    "origin.alt_m",
    "home.lat_deg",
    "home.lon_deg",
    "home.alt_m",
    "control.owner",
    "control.lease_age_s",
    "control.lease_timeout_s",
    "setpoint.age_s",
    "failsafe.reason",
    "command.result.request_id",
    "command.result.accepted",
    "command.result.reason",
    "drone.armed",
    "drone.mode",
    "drone.x_m",
    "drone.y_m",
    "drone.z_m",
    "drone.lat_deg",
    "drone.lon_deg",
    "drone.alt_m",
    "target.lat_deg",
    "target.lon_deg",
    "target.alt_m",
    "drone.roll_deg",
    "drone.pitch_deg",
    "drone.heading_deg",
    "drone.compass_deg",
    "drone.vx_mps",
    "drone.vy_mps",
    "drone.vz_mps",
    "drone.speed_mps",
    "drone.battery_pct",
    "drone.crashed",
    "drone.last_command_s",
    "link.command_count",
    "link.last_command_type",
    "status.message",
    "camera.fov_h_deg",
    "camera.fov_v_deg",
    "camera.tx_m",
    "camera.ty_m",
    "camera.tz_m",
    "camera.roll_deg",
    "camera.pitch_deg",
    "camera.yaw_deg",
    "camera.fx_px",
    "camera.fy_px",
    "camera.cx_px",
    "camera.cy_px",
    "camera.width_px",
    "camera.height_px",
    "camera.fps",
]

def memkv_aligned_name_len(min_name_len: int, max_value_len: int) -> int:
    """Round ``min_name_len`` up so pymembus' memkv record stride is 8-byte aligned.

    pymembus computes the per-record stride as
    ``2 * sizeof(int64_t) + max_name_len + 1 + max_value_len + 1`` and rejects
    ``memkv.create``/``open`` with "invalid shared-memory layout" unless that
    stride is a multiple of ``alignof(int64_t)`` (8 on x86_64).
    """
    name_len = min_name_len
    while (2 * 8 + name_len + 1 + max_value_len + 1) % 8 != 0:
        name_len += 1
    return name_len


BERLIN_CENTER_LAT_DEG = 52.5200
BERLIN_CENTER_LON_DEG = 13.4050
BERLIN_CENTER_ALT_M = 34.0


def load_pymembus():
    try:
        import pymembus as module
    except (ModuleNotFoundError, ImportError) as exc:
        sys.modules.pop("pymembus", None)
        local_build = Path.home() / "code" / "wj" / "pymembus" / "bld" / "lib"
        if local_build.exists():
            sys.path.insert(0, str(local_build))
            try:
                import pymembus as module
            except (ModuleNotFoundError, ImportError):
                raise SystemExit(
                    "pymembus is required. Install it with `python3 -m pip install pymembus` "
                    "or set PYTHONPATH to a local pymembus build directory."
                ) from exc
        else:
            raise SystemExit(
                "pymembus is required. Install it with `python3 -m pip install pymembus` "
                "or set PYTHONPATH to a local pymembus build directory."
            ) from exc
    if hasattr(module, "pymembus"):
        return module.pymembus
    return module


def validate_id(instance_id: str) -> str:
    if not instance_id or not ID_RE.fullmatch(instance_id):
        raise ValueError("id must match [A-Za-z0-9_.-]+")
    return instance_id


#: Where a run's artifacts live when the instance has no id. An id is normally
#: required, so this is a fallback for programmatic use rather than a mode.
DEFAULT_REPORT_ID = "default"


def new_run_id() -> str:
    """A sortable, collision-resistant name for one run.

    The timestamp so runs sort and can be found by eye; the random suffix so
    two instances started in the same second cannot land in the same
    directory. Defined here rather than in the simulator so every module that
    ever mints one produces the same shape.
    """
    import datetime
    import uuid

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


def report_root(instance_id: str | None, run_id: str, *,
                root: Path | None = None) -> Path:
    """``reports/<id>/<run_id>/`` — where one run's artifacts live.

    Grouping by instance id first keeps concurrent instances apart at the top
    level: ``area1`` and ``area2`` running side by side produce two trees
    rather than one interleaved list that can only be untangled by opening
    files. Each component then writes into its own subdirectory of the
    returned root -- ``dsim/``, ``daic/``, ``dway/`` -- so a run's pieces stay
    together and no component has to coordinate a name with any other.

    The simulator owns the root and publishes it as the ``sim.report_dir``
    status key; every other module reads it from there rather than deriving
    one, which is what keeps them in the same directory.
    """
    base = Path(root) if root is not None else Path(__file__).resolve().parent / "reports"
    group = validate_id(instance_id) if instance_id else DEFAULT_REPORT_ID
    return base / group / run_id


def shared_names(instance_id: str) -> dict[str, str]:
    validate_id(instance_id)
    base = f"/dvision2.{instance_id}"
    return {
        "video": f"{base}.video",
        "command": f"{base}.control",
        "status": f"{base}.status",
    }


def encode_command(command_type: str, **fields: Any) -> str:
    payload = {"magic": COMMAND_MAGIC, "type": command_type}
    payload.update(fields)
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def new_control_identity(client_id: str) -> tuple[str, str]:
    """Return a validated controller id and an opaque proposed lease id."""
    return validate_id(client_id), uuid.uuid4().hex


def controlled_command_with_id(command_type: str, source_id: str, lease_id: str,
                               **fields: Any) -> tuple[str, str]:
    """Encode one correlated command and return its request id alongside it.

    A caller that waits for the acknowledgement needs the id it is waiting for,
    and minting one here rather than letting each client invent its own keeps
    every correlated command on the same field.
    """
    request_id = uuid.uuid4().hex
    return request_id, encode_command(
        command_type, source_id=validate_id(source_id), lease_id=lease_id,
        request_id=request_id, **fields)


def controlled_command(command_type: str, source_id: str, lease_id: str,
                       **fields: Any) -> str:
    """Encode one correlated command from a control-lease holder."""
    return controlled_command_with_id(command_type, source_id, lease_id,
                                      **fields)[1]


def map_to_local_ned(x_m: float, y_m: float, z_m: float,
                     width_m: float, height_m: float) -> tuple[float, float, float]:
    """Convert public map coordinates (east/south/up) to local NED."""
    return height_m / 2.0 - y_m, x_m - width_m / 2.0, -z_m


def local_ned_to_map(north_m: float, east_m: float, down_m: float,
                     width_m: float, height_m: float) -> tuple[float, float, float]:
    """Convert local NED at map centre back to map coordinates."""
    return east_m + width_m / 2.0, height_m / 2.0 - north_m, -down_m


def decode_command(raw: str | bytes) -> dict[str, Any] | None:
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or payload.get("magic") != COMMAND_MAGIC:
        return None
    if not isinstance(payload.get("type"), str):
        return None
    return payload


@dataclass(frozen=True)
class MapObject:
    symbol: str
    kind: str
    x: float
    y: float


@dataclass(frozen=True)
class SimMap:
    path: Path
    data: dict[str, str]
    vars: dict[str, str]
    rows: list[str]
    width: int
    height: int
    start_x: float
    start_y: float
    objects: list[MapObject]


def load_map(path: Path) -> SimMap:
    lines = path.read_text(encoding="utf-8").splitlines()
    section: str | None = None
    data_map: dict[str, str] = {}
    vars_map: dict[str, str] = {}
    rows: list[str] = []

    for line_no, line in enumerate(lines, start=1):
        if line.strip() == "--- DATA":
            section = "data"
            continue
        if line.strip() == "--- VARS":
            section = "vars"
            continue
        if line.strip() == "--- MAP":
            section = "map"
            continue
        if section is None:
            if line.strip():
                raise ValueError(f"{path}:{line_no}: content before first section")
            continue
        if section == "data":
            if not line.strip():
                continue
            if "=" not in line:
                raise ValueError(f"{path}:{line_no}: expected key=value")
            key, value = line.split("=", 1)
            data_map[key.strip()] = value.strip()
        elif section == "vars":
            if not line.strip():
                continue
            if "=" not in line:
                raise ValueError(f"{path}:{line_no}: expected SYMBOL=name")
            symbol, name = line.split("=", 1)
            if len(symbol) != 1:
                raise ValueError(f"{path}:{line_no}: map symbols must be one character")
            vars_map[symbol] = name.strip()
        elif section == "map":
            rows.append(line.rstrip("\n"))

    if not vars_map:
        raise ValueError(f"{path}: missing VARS entries")
    if not rows:
        raise ValueError(f"{path}: missing MAP rows")

    width = max(len(row) for row in rows)
    if width <= 0:
        raise ValueError(f"{path}: empty MAP")

    normalized_rows = [row.ljust(width) for row in rows]
    start_cells: list[tuple[float, float]] = []
    objects: list[MapObject] = []

    for y, row in enumerate(normalized_rows):
        for x, symbol in enumerate(row):
            if symbol == " ":
                continue
            if symbol not in vars_map:
                raise ValueError(f"{path}: unknown symbol {symbol!r} at row {y + 1}, col {x + 1}")
            kind = vars_map[symbol]
            cx = x + 0.5
            cy = y + 0.5
            if kind == "drone":
                start_cells.append((cx, cy))
                continue
            objects.append(MapObject(symbol=symbol, kind=kind, x=cx, y=cy))

    if len(start_cells) != 1:
        raise ValueError(f"{path}: expected exactly one drone start, found {len(start_cells)}")

    return SimMap(
        path=path,
        data=data_map,
        vars=vars_map,
        rows=normalized_rows,
        width=width,
        height=len(normalized_rows),
        start_x=start_cells[0][0],
        start_y=start_cells[0][1],
        objects=objects,
    )


def gps_bearing(lat1_deg: float, lon1_deg: float,
                lat2_deg: float, lon2_deg: float) -> float:
    """Compass bearing in degrees (0 = north, 90 = east) from point 1 to point 2."""
    lat1 = math.radians(lat1_deg)
    lat2 = math.radians(lat2_deg)
    dlon = math.radians(lon2_deg - lon1_deg)
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return math.degrees(math.atan2(x, y)) % 360.0


def gps_distance_m(lat1_deg: float, lon1_deg: float,
                   lat2_deg: float, lon2_deg: float) -> float:
    """Flat-earth distance in metres between two GPS points."""
    dlat = (lat2_deg - lat1_deg) * 111_320.0
    lat_mid = math.radians((lat1_deg + lat2_deg) / 2.0)
    dlon = (lon2_deg - lon1_deg) * 111_320.0 * math.cos(lat_mid)
    return math.hypot(dlat, dlon)


def local_to_gps(x_m: float, y_m: float, z_m: float, lat0: float, lon0: float, alt0: float) -> tuple[float, float, float]:
    lat = lat0 + y_m / 111_320.0
    lon_scale = 111_320.0 * max(0.01, math.cos(math.radians(lat0)))
    lon = lon0 + x_m / lon_scale
    return lat, lon, alt0 + z_m


def clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


# ---------------------------------------------------------------------------
# Window position persistence
# ---------------------------------------------------------------------------

_WIN_POS_FILE = Path.home() / ".config" / "dvision2" / "window_pos.json"


def _read_pos_store() -> dict:
    try:
        return json.loads(_WIN_POS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_pos_store(data: dict) -> None:
    try:
        _WIN_POS_FILE.parent.mkdir(parents=True, exist_ok=True)
        _WIN_POS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        pass


def save_window_pos(root: Any, key: str) -> None:
    """Save *root* window position under *key*.  Call just before destroy()."""
    try:
        x, y = root.winfo_x(), root.winfo_y()
        data = _read_pos_store()
        data[key] = {"x": x, "y": y}
        _write_pos_store(data)
    except Exception:
        pass


def restore_window_pos(root: Any, key: str) -> None:
    """Restore window position from a previous run.

    The saved top-left corner (x, y) is only applied when it falls inside the
    current virtual desktop reported by Tkinter.  If the display layout has
    changed (monitor removed, resolution reduced) the position is silently
    ignored and the OS places the window normally.
    """
    try:
        data = _read_pos_store()
        if key not in data:
            return
        x = int(data[key]["x"])
        y = int(data[key]["y"])
        sw = root.winfo_screenwidth()
        sh = root.winfo_screenheight()
        # Require the top-left corner to be fully inside the current screen so
        # a window that lived on a now-absent monitor is not half-off-screen.
        if 0 <= x < sw and 0 <= y < sh:
            root.geometry(f"+{x}+{y}")
    except Exception:
        pass
