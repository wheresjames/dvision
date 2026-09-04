from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Profile:
    name: str
    algorithm: str
    tour: Path | None
    sensors: tuple[str, ...]
    sensor_config: dict[str, str]
    settings: dict[str, Any]
    digest: str


def load_profile(name_or_path: str, root: Path) -> Profile:
    path = Path(name_or_path)
    if not path.suffix:
        path = root / "dalg" / "profiles" / f"{name_or_path}.json"
    elif not path.is_absolute():
        path = root / path
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("profile must be an object")
    for key in ("name", "algorithm"):
        if not isinstance(value.get(key), str) or not value[key]:
            raise ValueError(f"profile requires {key}")
    sensors = value.get("sensors", ["rgb"])
    sensor_config: dict[str, str] = {}
    if isinstance(sensors, dict):
        if not all(isinstance(k, str) and isinstance(v, str)
                   for k, v in sensors.items()):
            raise ValueError("profile sensor configuration must contain names")
        sensor_config = dict(sensors)
        sensors = list(sensors)
    if not isinstance(sensors, list) or not all(isinstance(x, str) for x in sensors):
        raise ValueError("profile sensors must be names")
    settings = value.get("settings", {})
    if not isinstance(settings, dict):
        raise ValueError("profile settings must be an object")
    tour_value = value.get("tour")
    if tour_value is not None and (not isinstance(tour_value, str) or not tour_value):
        raise ValueError("profile tour must be a non-empty path when present")
    tour = None if tour_value is None else Path(tour_value)
    if tour is not None and not tour.is_absolute(): tour = root / tour
    return Profile(value["name"], value["algorithm"], tour, tuple(sensors),
                   sensor_config, settings, hashlib.sha256(raw).hexdigest())


def save_profile(path: Path, *, name: str, algorithm: str, tour: str | None,
                 sensors: dict[str, str] | list[str], settings: dict[str, Any]) -> None:
    """Write the deliberately plain, diffable profile format."""
    value = {"name": name, "algorithm": algorithm,
             "sensors": sensors, "settings": settings}
    if tour is not None: value["tour"] = tour
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
