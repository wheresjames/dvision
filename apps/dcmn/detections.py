"""The detections plane: what a producer's target detectors see, and where it thinks targets are.

One retained snapshot per producer, ``/dvision2.<id>.detections.<producer>``,
schema ``dvision2.detections.v1``, carried by
:class:`dcmn.navigation.Snapshot` -- the same 64 KiB cap, lifetime writer lock
and reopen-on-read. dalg is the producer (``dalg`` by default; one dalg per
vehicle in v1). A producer owns its plane; a consumer creates nothing and reads
an absent plane as empty.

Detections are belief, not truth: stamped with their capture time, causal and
bounded. A snapshot holds

* ``detectors`` -- each detector's ID, name, labels, settings digest, state and
  rates (at most :data:`MAX_DETECTORS`);
* ``detections`` -- the latest processed frame's detections per detector, in
  pixels and, when projectable, on the ground (at most :data:`MAX_DETECTIONS`);
* ``estimates`` -- the running per-label target estimates: ID, label, ground
  position, spread, observation count, confirmed and stale flags, first and
  last sighting and a revision that moves with every update (at most
  :data:`MAX_ESTIMATES`).

``revision`` counts publications and ``time_s`` is the producer's sim time,
so a consumer judges staleness by sim time (``docs/clock.md``), never by wall
clock. Frame lists, crops and per-frame histories stay in the producer's
archive.
"""
from __future__ import annotations

import math

SCHEMA = 'dvision2.detections.v1'
DEFAULT_PRODUCER = 'dalg'
MAX_DETECTORS = 8
MAX_DETECTIONS = 16
MAX_ESTIMATES = 32
MAX_LABELS = 8
MAX_TEXT = 120
DETECTOR_STATES = ('waiting', 'running', 'failed')
CONTEXT_KEYS = ('frame_id', 'localization_epoch', 'clock_domain_id', 'clock_epoch', 'mapping_epoch')


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _optional_number(value):
    return value is None or _number(value)


def _count(value):
    return type(value) is int and value >= 0


def _text(value, *, empty=False):
    return isinstance(value, str) and (empty or bool(value)) and len(value) <= MAX_TEXT


def _xy(value):
    return isinstance(value, list) and len(value) == 2 and all(_number(v) for v in value)


def _require(condition, message):
    if not condition: raise ValueError(message)


def validate(value):
    """A detections snapshot, or ``ValueError`` saying which field is wrong."""
    _require(isinstance(value, dict) and value.get('schema') == SCHEMA, f'unsupported detections schema; expected {SCHEMA}')
    for key in ('vehicle_id', 'producer', 'session'):
        _require(_text(value.get(key)), f'invalid {key}')
    _require(_count(value.get('revision')), 'invalid revision')
    _require(_number(value.get('time_s')), 'invalid time_s')
    context = value.get('context', {})
    _require(isinstance(context, dict), 'context must be an object')
    detectors = value.get('detectors', [])
    _require(isinstance(detectors, list) and len(detectors) <= MAX_DETECTORS,
             f'at most {MAX_DETECTORS} detectors')
    for row in detectors:
        _require(isinstance(row, dict), 'invalid detector row')
        for key in ('id', 'name', 'sensor', 'settings_digest'):
            _require(_text(row.get(key)), f'invalid detector {key}')
        labels = row.get('labels')
        _require(isinstance(labels, list) and 0 < len(labels) <= MAX_LABELS and all(_text(x) for x in labels),
                 'invalid detector labels')
        _require(row.get('state') in DETECTOR_STATES, 'invalid detector state')
        _require(_text(row.get('reason', ''), empty=True), 'invalid detector reason')
        for key in ('processed', 'skipped', 'failed'):
            _require(_count(row.get(key, 0)), f'invalid detector {key}')
        for key in ('hz', 'configured_hz', 'latency_ms_p50', 'latency_ms_p95', 'last_frame_s'):
            _require(_optional_number(row.get(key)), f'invalid detector {key}')
    detections = value.get('detections', [])
    _require(isinstance(detections, list) and len(detections) <= MAX_DETECTIONS,
             f'at most {MAX_DETECTIONS} detections')
    for row in detections:
        _require(isinstance(row, dict), 'invalid detection row')
        for key in ('id', 'detector', 'label'):
            _require(_text(row.get(key)), f'invalid detection {key}')
        _require(_count(row.get('sequence')), 'invalid detection sequence')
        for key in ('capture_time_s', 'cx_px', 'cy_px', 'extent_px', 'confidence'):
            _require(_number(row.get(key)), f'invalid detection {key}')
        _require(0.0 <= row['confidence'] <= 1.0, 'detection confidence must be in 0..1')
        _require(isinstance(row.get('projectable'), bool), 'invalid detection projectable flag')
        if row['projectable']:
            _require(_xy(row.get('xy_m')), 'a projectable detection needs xy_m')
            for key in ('range_m', 'bearing_deg', 'sigma_m'):
                _require(_number(row.get(key)), f'invalid detection {key}')
        else:
            _require(_text(row.get('reason')), 'a detection that is not projectable must say why')
        _require(row.get('estimate') is None or _text(row.get('estimate')), 'invalid detection estimate')
    estimates = value.get('estimates', [])
    _require(isinstance(estimates, list) and len(estimates) <= MAX_ESTIMATES,
             f'at most {MAX_ESTIMATES} estimates')
    for row in estimates:
        _require(isinstance(row, dict), 'invalid estimate row')
        for key in ('id', 'label', 'detector'):
            _require(_text(row.get(key)), f'invalid estimate {key}')
        _require(_xy(row.get('xy_m')), 'invalid estimate xy_m')
        for key in ('spread_m', 'first_s', 'last_s', 'confidence'):
            _require(_number(row.get(key)), f'invalid estimate {key}')
        for key in ('count', 'revision'):
            _require(_count(row.get(key)), f'invalid estimate {key}')
        for key in ('confirmed', 'stale'):
            _require(isinstance(row.get(key), bool), f'invalid estimate {key}')
    return value


def empty(vehicle_id, producer, session, time_s=0.0, context=None):
    """A valid snapshot with nothing in it: the plane exists, nothing is seen yet."""
    return dict(schema=SCHEMA, vehicle_id=vehicle_id, producer=producer, session=session, revision=0,
                time_s=float(time_s), context=dict(context or {}), detectors=[], detections=[], estimates=[])


def labels(value):
    """Every label some detector in a snapshot can emit."""
    return sorted({label for row in (value or {}).get('detectors', []) for label in row.get('labels', ())})


class DetectionsPublisher:
    """The producer's side: owns the plane for its lifetime (a second producer is refused)."""

    def __init__(self, vehicle_id, producer=DEFAULT_PRODUCER):
        from dcmn.navigation import Snapshot
        self.snapshot = Snapshot(vehicle_id, 'detections', planner=producer)
        self.name = self.snapshot.name

    def start(self):
        self.snapshot.start()

    def write(self, value):
        self.snapshot.write(value)

    def close(self):
        self.snapshot.close()


class DetectionsReader:
    """A consumer's side: reopen on every read, an absent plane reads as ``{}``."""

    def __init__(self, vehicle_id, producer=DEFAULT_PRODUCER):
        from dcmn.navigation import Snapshot
        self.snapshot = Snapshot(vehicle_id, 'detections', planner=producer)
        self.name = self.snapshot.name

    def read(self):
        return self.snapshot.read()
