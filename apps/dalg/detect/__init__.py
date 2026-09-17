"""Target detectors: a second product of dalg beside evidence.

Detection is perception, not mission logic, and not every run needs it: a
mapping run composes no detector profile and runs none. A detector works in
image space only -- it never projects, fuses or publishes. The host
(:mod:`dalg.detect.host`) schedules detectors on their sensor's frames,
projects what they find onto the ground (:mod:`dalg.detect.projection`), keeps
a crude running estimate per target (:mod:`dalg.detect.estimate`), publishes
the detections plane (:mod:`dcmn.detections`) and records everything.

A new detector is one module and one line in :data:`DETECTORS`; nothing above
the registry changes. Every detector reads the common settings in
:class:`DetectorConfig` (rate, projection range, estimate association and
confirmation, archive imagery); a detector with settings of its own names a
dataclass holding them as ``settings_class``.

Detections are belief, not truth. No detector, projection or estimate reads the
world, ``dsim/truth`` or the provider's ``target.*`` keys.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, ClassVar, Protocol, runtime_checkable

#: Rejected candidates a detector result carries per frame.
CANDIDATE_CAP = 8


@dataclass(frozen=True)
class PixelDetection:
    """Something found in one frame, in pixels."""

    label: str
    cx_px: float
    cy_px: float
    #: Radius, or half the larger box side.
    extent_px: float
    #: The detector's own 0..1 confidence.
    confidence: float
    box_px: tuple[float, float, float, float] | None = None
    #: Bounded, JSON-serialisable shape measurements (fill, aspect, ...).
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return dict(label=self.label, cx_px=float(self.cx_px), cy_px=float(self.cy_px),
                    extent_px=float(self.extent_px), confidence=float(self.confidence),
                    box_px=None if self.box_px is None else [float(v) for v in self.box_px],
                    diagnostics=dict(self.diagnostics))


@dataclass(frozen=True)
class Candidate:
    """Something considered and rejected: what makes "why didn't it see the pad?" answerable."""

    cx_px: float
    cy_px: float
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return dict(cx_px=float(self.cx_px), cy_px=float(self.cy_px), reason=self.reason)


@dataclass(frozen=True)
class DetectorResult:
    detections: tuple[PixelDetection, ...] = ()
    candidates: tuple[Candidate, ...] = ()


@dataclass(frozen=True)
class DetectorConfig:
    """Settings every detector takes, whatever it detects."""

    #: Frames processed per sim second; overdue frames are skipped and counted.
    hz: float = 5.0
    #: Ground range beyond which a detection is not projectable.
    max_range_m: float = 25.0
    #: A projected detection this close to an estimate of its label joins it.
    gate_m: float = 1.5
    #: Observations before an estimate is confirmed.
    confirm_n: int = 3
    #: Sim seconds unseen before an estimate is marked stale (never deleted).
    expire_s: float = 30.0
    #: Full frames archived per sim second for offline work; 0 keeps none.
    record_frames_hz: float = 0.0
    #: Side of the square crop archived around each detection, in pixels.
    crop_px: int = 96
    #: Rejected candidates kept per frame for the overlay and the archive.
    candidates: int = CANDIDATE_CAP

    def check(self) -> 'DetectorConfig':
        if not self.hz > 0: raise ValueError('hz must be positive')
        if not self.max_range_m > 0: raise ValueError('max_range_m must be positive')
        if not self.gate_m > 0: raise ValueError('gate_m must be positive')
        if isinstance(self.confirm_n, bool) or not isinstance(self.confirm_n, int) or self.confirm_n < 1:
            raise ValueError('confirm_n must be a whole number of at least 1')
        if not self.expire_s > 0: raise ValueError('expire_s must be positive')
        if not self.record_frames_hz >= 0: raise ValueError('record_frames_hz must not be negative')
        if isinstance(self.crop_px, bool) or not isinstance(self.crop_px, int) or not 16 <= self.crop_px <= 512:
            raise ValueError('crop_px must be a whole number from 16 to 512')
        if isinstance(self.candidates, bool) or not isinstance(self.candidates, int) or not 0 <= self.candidates <= 32:
            raise ValueError('candidates must be a whole number from 0 to 32')
        return self


@runtime_checkable
class Detector(Protocol):
    name: ClassVar[str]
    #: Labels it can emit.
    labels: ClassVar[tuple[str, ...]]
    #: The sensor type it reads; ``camera.rgb`` in v1.
    sensor_type: ClassVar[str]
    #: Its own settings dataclass, or None.
    settings_class: ClassVar[type | None]

    def __init__(self, config: Any, intrinsics: Any) -> None: ...
    def detect(self, frame: Any) -> DetectorResult: ...


from dalg.detect.red_disc import RedDiscDetector  # noqa: E402 - the registry follows the protocol it satisfies

DETECTORS: dict[str, type] = {
    RedDiscDetector.name: RedDiscDetector,
}


def split_settings(name: str, settings: dict[str, Any] | None) -> tuple[DetectorConfig, Any]:
    """``(common config, detector-specific settings)`` from one profile ``settings`` object.

    Unknown keys are refused by name, so a typo never silently runs defaults.
    """
    try: cls = DETECTORS[name]
    except KeyError:
        raise ValueError(f'unsupported detector: {name}; known: {", ".join(sorted(DETECTORS))}') from None
    settings = dict(settings or {})
    common_names = {f.name for f in fields(DetectorConfig)}
    own_class = getattr(cls, 'settings_class', None)
    own_names = set() if own_class is None else {f.name for f in fields(own_class)}
    unknown = sorted(set(settings) - common_names - own_names)
    if unknown: raise ValueError(f'unknown {name} settings: {", ".join(unknown)}')
    common = DetectorConfig(**{k: v for k, v in settings.items() if k in common_names}).check()
    own = None if own_class is None else own_class(**{k: v for k, v in settings.items() if k in own_names})
    return common, own


def build(name: str, settings: dict[str, Any] | None, intrinsics: Any):
    """``(detector instance, common config)`` for one profile row."""
    common, own = split_settings(name, settings)
    return DETECTORS[name](own, intrinsics), common
