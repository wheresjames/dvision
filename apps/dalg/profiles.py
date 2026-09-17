"""Algorithm profiles: evidence sources and target detectors; geometry and missions belong to runtime context.

A profile holds a name, ``sources`` (evidence algorithms) and optionally
``detectors`` (target detectors, :mod:`dalg.detect`). Mapping profiles keep
their sources only; a detector profile such as ``red-disc`` holds detectors
only and is composed beside one: ``--profiles lidar-baseline red-disc``.
Detectors run in the one dalg beside evidence, so a run needs at least one
evidence source.
"""
from __future__ import annotations
import hashlib
import json
import math
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

PRIMARY_CAMERA = "primary_camera"
#: A published detection label: short, and safe in plane names, events and file names.
LABEL_RE = re.compile(r"[A-Za-z0-9_.-]{1,120}")

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


@dataclass(frozen=True)
class Detector:
    """One target detector on one sensor, with its settings (common and its own).

    Two of the same detector on one sensor -- different settings side by side --
    are told apart by ``suffix``, which joins the ID.

    ``label`` is what the detector's detections and estimates are published as,
    and so what a mission's ``target_label`` must name. Empty keeps the label
    the detector declares (``red_disc``: ``landing_pad``); a label may only be
    set for a detector that declares exactly one.
    """
    sensor: str
    detector: str
    settings: dict[str, Any] = field(default_factory=dict)
    suffix: str = ''
    label: str = ''

    @property
    def id(self):
        stem = f"{self.sensor}-{self.detector}" + (f"-{self.suffix}" if self.suffix else "")
        return stem if len(stem) <= 48 else stem[:35]+"-"+hashlib.sha256(stem.encode()).hexdigest()[:12]

    @property
    def settings_digest(self):
        value = dict(detector=self.detector, settings=self.settings)
        if self.label: value['label'] = self.label  # what is published is part of what a run means
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

    @property
    def native_labels(self):
        """The labels the detector itself declares."""
        from dalg.detect import DETECTORS
        cls = DETECTORS.get(self.detector)
        return tuple(cls.labels) if cls is not None else ()

    @property
    def labels(self):
        """The labels this row publishes: its ``label`` when set, otherwise the detector's own."""
        return (self.label,) if self.label else self.native_labels

    def published_label(self, native):
        """The label a detection the detector emitted as ``native`` is published under."""
        return self.label or native

    def as_dict(self):
        value = {"sensor": self.sensor, "detector": self.detector, "settings": dict(self.settings)}
        if self.suffix: value["suffix"] = self.suffix
        if self.label: value["label"] = self.label
        return value


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


def validate_sources(sources, manifest=None, *, detectors=None):
    """Sources (and detectors) that can run; a profile needs at least one of either.

    Without ``detectors`` (the evidence path) at least one source is required.
    """
    if not sources and not detectors:
        raise ValueError("profile requires at least one source" if detectors is None
                         else "profile requires at least one source or detector")
    errors = source_errors(sources, manifest)
    if errors:
        raise ValueError("; ".join(f"source {index+1}: {error}" for index, error in errors.items()))
    detector_problems = detector_errors(detectors or (), manifest)
    if detector_problems:
        raise ValueError("; ".join(f"detector {index+1}: {error}" for index, error in detector_problems.items()))


def detector_errors(detectors, manifest=None):
    """Row-addressed detector errors, the same way :func:`source_errors` addresses sources."""
    from dvision2_common import validate_id
    from dalg.detect import DETECTORS, split_settings
    errors, seen = {}, set()
    for index, detector in enumerate(detectors):
        try:
            validate_id(detector.sensor)
            validate_id(detector.id)
            if detector.id in seen:
                raise ValueError("duplicate sensor/detector: give the second a distinct suffix")
            seen.add(detector.id)
            if not isinstance(detector.settings, dict): raise ValueError("settings must be an object")
            for key, value in detector.settings.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool) and not math.isfinite(value):
                    raise ValueError(f"{key} must be finite")
            split_settings(detector.detector, detector.settings)
            if not isinstance(detector.label, str): raise ValueError("detector label must be text")
            if detector.label:
                if not LABEL_RE.fullmatch(detector.label):
                    raise ValueError(f"label {detector.label!r} must be 1-120 of [A-Za-z0-9_.-]")
                if len(detector.native_labels) != 1:
                    raise ValueError(f"{detector.detector} declares {len(detector.native_labels)} labels; "
                                     "a label can only rename a single-label detector")
            if manifest is not None:
                entry = manifest.get("sensors", {}).get(detector.sensor)
                if entry is None: raise ValueError(f"sensor {detector.sensor!r} is absent from the manifest")
                kind = DETECTORS[detector.detector].sensor_type
                if entry['type'] != kind:
                    raise ValueError(f"{detector.detector} needs {kind}, got {entry['type']}")
        except (ValueError, TypeError) as exc:
            errors[index] = str(exc)
    return errors


@dataclass(frozen=True)
class Profile:
    name: str
    sources: tuple[Source, ...]
    digest: str
    path: Path | None = field(default=None, compare=False)
    components: tuple[dict, ...] = ()
    detectors: tuple[Detector, ...] = ()

    @property
    def algorithm(self): return self.sources[0].algorithm if self.sources else self.detectors[0].detector
    @property
    def sensors(self): return tuple(dict.fromkeys([*(s.sensor for s in self.sources),
                                                   *(d.sensor for d in self.detectors)]))


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
    legacy = sorted(set(value)-{'name', 'sources', 'detectors'})
    if legacy:
        raise ValueError(f'{path.name}: unsupported field(s) {", ".join(legacy)}; profiles hold only name, '
                         'sources and detectors -- tours belong to mission tooling, map extents to dalg '
                         '--bounds/--cell-m')
    if not isinstance(value.get('name'), str) or not value['name'].strip(): raise ValueError('profile name required')
    if not isinstance(value.get('sources'), list): raise ValueError('profile sources must be a list')
    if not isinstance(value.get('detectors', []), list): raise ValueError('profile detectors must be a list')
    sources = []
    for row in value['sources']:
        if not isinstance(row, dict) or set(row)-{'sensor', 'algorithm', 'settings'}:
            raise ValueError('a source holds only sensor, algorithm and settings')
        if not all(isinstance(row.get(k), str) and row[k] for k in ('sensor', 'algorithm')):
            raise ValueError('source requires sensor and algorithm names')
        sources.append(Source(row['sensor'], row['algorithm'], row.get('settings', {})))
    detectors = []
    for row in value.get('detectors', []):
        if not isinstance(row, dict) or set(row)-{'sensor', 'detector', 'settings', 'suffix', 'label'}:
            raise ValueError('a detector holds only sensor, detector, settings, suffix and label')
        if not all(isinstance(row.get(k), str) and row[k] for k in ('sensor', 'detector')):
            raise ValueError('detector requires sensor and detector names')
        if not isinstance(row.get('suffix', ''), str): raise ValueError('detector suffix must be text')
        if not isinstance(row.get('label', ''), str): raise ValueError('detector label must be text')
        detectors.append(Detector(row['sensor'], row['detector'], row.get('settings', {}), row.get('suffix', ''),
                                  row.get('label', '')))
    validate_sources(sources, detectors=detectors) if 'detectors' in value else validate_sources(sources)
    digest = hashlib.sha256(raw).hexdigest()
    return Profile(value['name'], tuple(sources), digest, path.resolve(),
                   (dict(name=value['name'], path=str(path.resolve()), digest=digest),), tuple(detectors))


def load_profiles(names, root):
    profiles = [load_profile(name, root) for name in names]
    if not profiles: raise ValueError('at least one profile is required')
    if len(profiles) == 1: return profiles[0]
    sources = tuple(s for p in profiles for s in p.sources)
    detectors = tuple(d for p in profiles for d in p.detectors)
    validate_sources(sources, detectors=detectors)
    digest = hashlib.sha256(json.dumps([p.digest for p in profiles]).encode()).hexdigest()
    return Profile('+'.join(p.name for p in profiles), sources, digest,
                   components=tuple(c for p in profiles for c in p.components), detectors=detectors)


def require_evidence(profile):
    """Detectors run beside evidence in v1: a run needs an evidence source."""
    if not profile.sources:
        raise ValueError(f'{profile.name} holds detectors only; detectors run beside evidence, so compose it '
                         'with an evidence profile, e.g. --profiles lidar-baseline red-disc')


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


def resolve_detectors(profile, manifest):
    """Bind ``primary_camera`` in detector rows, exactly as :func:`resolve_sources` does for sources."""
    primary = manifest.get('primary_camera')
    detectors = tuple(replace(d, sensor=primary) if d.sensor == PRIMARY_CAMERA and primary else d
                      for d in profile.detectors)
    problems = detector_errors(detectors)
    if problems:
        raise ValueError("; ".join(f"detector {index+1}: {error}" for index, error in problems.items()))
    return detectors


def save_sources_profile(path, *, name, sources, detectors=()):
    validate_sources(sources, detectors=detectors) if detectors else validate_sources(sources)
    if not name.strip(): raise ValueError('profile name required')
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    value = dict(name=name, sources=[s.as_dict() for s in sources])
    if detectors: value['detectors'] = [d.as_dict() for d in detectors]
    path.write_text(json.dumps(value, sort_keys=True, indent=2)+'\n')
