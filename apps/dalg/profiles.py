"""Source-only algorithm profiles; geometry and missions belong to runtime context."""
from __future__ import annotations
import hashlib
import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

PRIMARY_CAMERA = "primary_camera"

@dataclass(frozen=True)
class Source:
    sensor: str
    algorithm: str
    settings: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self):
        stem = f"{self.sensor}-{self.algorithm}"
        return stem if len(stem) <= 48 else stem[:35]+"-"+hashlib.sha256(stem.encode()).hexdigest()[:12]

    def as_dict(self):
        return {"sensor": self.sensor, "algorithm": self.algorithm,
                "settings": dict(self.settings)}


#: Camera algorithms that publish evidence, in the order they were converted.
#: An algorithm joins this tuple once it meets the evidence contract -- belief
#: not truth, every write stamped with its capture time, causal, bounded -- and joining it
#: is the whole switch: the run, the multi-source path, profile validation and
#: the profile editor all read it. The controls never join: ``exact_range`` is
#: built from truth and ``constant`` carries no information.
CAMERA_EVIDENCE = ("ground_plane", "optical_flow_triangulation", "feature_triangulation",
                   "monocular_depth", "plane_sweep", "sgbm")


def camera_evidence_algorithms():
    """The camera algorithms that can publish an evidence grid."""
    return CAMERA_EVIDENCE


def source_configs():
    """Every evidence algorithm, with the sensor type it reads and its settings."""
    from dalg.algo import CONFIGS
    from dalg.algo.lidar import LidarConfig
    configs = {name: ("camera.rgb", CONFIGS[name]) for name in CAMERA_EVIDENCE}
    configs["lidar_inverse"] = ("lidar.scan2d", LidarConfig)
    return configs


def source_errors(sources, manifest=None):
    """Row-addressed errors shared by profile loading, readiness and the UI."""
    from dvision2_common import validate_id
    errors = {}
    seen = set()
    for index, source in enumerate(sources):
        try:
            validate_id(source.sensor)
            validate_id(source.id)
            if source.id in seen: raise ValueError("duplicate sensor/algorithm source")
            seen.add(source.id)
            if source.algorithm not in source_configs():
                raise ValueError(f"unsupported evidence algorithm: {source.algorithm}")
            kind, config = source_configs()[source.algorithm]
            if not isinstance(source.settings, dict): raise ValueError("settings must be an object")
            for key, value in source.settings.items():
                if isinstance(value, (int, float)) and not math.isfinite(value):
                    raise ValueError(f"{key} must be finite")
            config(**source.settings)
            if source.algorithm == "ground_plane":
                configured = config(**source.settings)
                if (not isinstance(configured.column_stride, int) or configured.column_stride < 1
                        or configured.min_range_m <= 0 or configured.max_range_m <= configured.min_range_m
                        or configured.edge_threshold < 0):
                    raise ValueError("invalid camera stride, range or edge threshold")
            if manifest is not None:
                entry = manifest.get("sensors", {}).get(source.sensor)
                if entry is None: raise ValueError(f"sensor {source.sensor!r} is absent from the manifest")
                if entry['type'] != kind:
                    raise ValueError(f"{source.algorithm} needs {kind}, got {entry['type']}")
        except (ValueError, TypeError) as exc:
            errors[index] = str(exc)
    return errors


def validate_sources(sources, manifest=None):
    if not sources: raise ValueError("profile requires at least one source")
    errors = source_errors(sources, manifest)
    if errors:
        raise ValueError("; ".join(f"source {index+1}: {error}" for index, error in errors.items()))


@dataclass(frozen=True)
class Profile:
    name: str
    sources: tuple[Source, ...]
    digest: str
    path: Path | None = field(default=None, compare=False)
    components: tuple[dict, ...] = ()

    @property
    def algorithm(self): return self.sources[0].algorithm
    @property
    def sensors(self): return tuple(dict.fromkeys(s.sensor for s in self.sources))


def profile_dir(root): return Path(root) / 'assets/algorithm_profiles'


def load_profile(name_or_path, root):
    name_or_path = str(name_or_path)
    path = Path(name_or_path)
    if not path.suffix: path = profile_dir(root) / (name_or_path+'.json')
    elif not path.is_absolute():
        path = (profile_dir(root)/path if name_or_path == path.name and not (Path(root)/path).is_file()
                else Path(root)/path)
    raw = path.read_bytes(); value = json.loads(raw)
    if not isinstance(value, dict): raise ValueError('profile must be an object')
    legacy = sorted(set(value)-{'name', 'sources'})
    if legacy:
        raise ValueError(f'{path.name}: unsupported field(s) {", ".join(legacy)}; profiles hold only name '
                         'and sources -- tours belong to mission tooling, map extents to dalg --bounds/--cell-m')
    if not isinstance(value.get('name'), str) or not value['name'].strip(): raise ValueError('profile name required')
    if not isinstance(value.get('sources'), list): raise ValueError('profile sources must be a list')
    sources = []
    for row in value['sources']:
        if not isinstance(row, dict) or set(row)-{'sensor', 'algorithm', 'settings'}:
            raise ValueError('a source holds only sensor, algorithm and settings')
        if not all(isinstance(row.get(k), str) and row[k] for k in ('sensor', 'algorithm')):
            raise ValueError('source requires sensor and algorithm names')
        sources.append(Source(row['sensor'], row['algorithm'], row.get('settings', {})))
    validate_sources(sources)
    digest = hashlib.sha256(raw).hexdigest()
    return Profile(value['name'], tuple(sources), digest, path.resolve(),
                   (dict(name=value['name'], path=str(path.resolve()), digest=digest),))


def load_profiles(names, root):
    profiles = [load_profile(name, root) for name in names]
    if not profiles: raise ValueError('at least one profile is required')
    if len(profiles) == 1: return profiles[0]
    sources = tuple(s for p in profiles for s in p.sources)
    validate_sources(sources)
    digest = hashlib.sha256(json.dumps([p.digest for p in profiles]).encode()).hexdigest()
    return Profile('+'.join(p.name for p in profiles), sources, digest,
                   components=tuple(c for p in profiles for c in p.components))


def preflight(profile, root):
    """Errors a launch should refuse on before connecting: missing model files.

    No model is ever downloaded automatically; the error says how to install it.
    """
    from dalg.sources import resolved_settings
    errors = []
    for source in profile.sources:
        try: resolved_settings(source.algorithm, source.settings, root)
        except ValueError as exc: errors.append(f'{source.id}: {exc}')
    return errors


def resolve_sources(profile, manifest):
    """Bind the ``primary_camera`` selector to the manifest's declared primary camera.

    Never the first camera found: a manifest without a primary camera leaves the
    selector unresolved, and the run reports that source as unavailable.
    Duplicate resolved sensor/algorithm pairs are rejected.
    """
    primary = manifest.get('primary_camera')
    sources = tuple(replace(s, sensor=primary) if s.sensor == PRIMARY_CAMERA and primary else s
                    for s in profile.sources)
    validate_sources(sources)
    return sources


def save_sources_profile(path, *, name, sources):
    validate_sources(sources)
    if not name.strip(): raise ValueError('profile name required')
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(name=name, sources=[s.as_dict() for s in sources]),
                               sort_keys=True, indent=2)+'\n')
