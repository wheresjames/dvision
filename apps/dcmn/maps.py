"""The evidence plane: the maps registry, its manifest, and payload type 110.

A producer -- `dalg` today -- publishes what it *believes* about the world as
one occupancy-evidence grid per source, and a consumer -- `dnav` next --
discovers those grids and derives cost from them under its own policy. Evidence
rather than cost is the whole point of the split: two sensors' costs cannot be
combined without double-counting the same wall, while two sensors' beliefs can,
and the distinction between *never observed* and *observed free* survives only
in an evidence grid. See DV-DNAV §2.

This module is the contract, and nothing above it:

* the registry name and its keys, mirroring the sensor plane's shape;
* the manifest a producer commits atomically and a consumer probes for;
* the DVS1 record codec for payload type 110, ``evidence.grid.v1``;
* :class:`MapPublisher`, which owns the plane, and :class:`MapSession`, which
  reads it and owns nothing.

**A producer owns its plane; consumers own nothing and create nothing.** The
publisher creates the registry, writes the manifest and creates every source
ring; a session opens derived names with retry, so start order does not matter
and a producer restart is a session change the consumer notices rather than a
handle it silently keeps reading. The transport is the same pymembus ring and
the same 128-byte DVS1 header the sensor plane uses -- what differs is the
payload and the fact that the rings size themselves from the declared geometry
rather than inheriting the sensor codec's 8 KiB compact-record cap. A 200x200
grid is about 200 KB, and the sensor plane already carries ~900 KB video
frames.

Cadence and staleness are measured in **simulated** time, per docs/clock.md;
the only wall clock here is discovery backoff, which is a liveness concern.
"""

from __future__ import annotations

if __name__ == '__main__':
    # A library that is also this plane's inspection tool. The bootstrap every
    # entry point performs has to run before the sibling imports below, so it
    # sits here rather than beside the ``main()`` call at the foot of the file.
    import sys as _sys
    from pathlib import Path as _Path
    for _path in (str(_Path(__file__).resolve().parents[2]),
                  str(_Path(__file__).resolve().parents[1])):
        if _path not in _sys.path: _sys.path.insert(0, _path)

import json
import struct
import time
import uuid
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from dcmn.sensors import (HEADER, RecordRing, STATUS_INVALID, STATUS_VALID,
                          decode_record, encode_record, json_bytes)
from dvision2_common import (load_pymembus, memkv_aligned_name_len,
                             shared_names, validate_id)

SCHEMA = 'dvision2.map-manifest.v2'
MAX_MANIFEST = 65536

#: Payload types on the maps plane. 110 is a complete grid; 111 is reserved for
#: patches, which are a pure optimization -- adopt it only when a profile's
#: extent makes full records measurably expensive. Every record being
#: self-contained is what keeps a missed record costing nothing but time, with
#: no keyframe/patch/gap machinery to get wrong.
EVIDENCE_GRID = 110
EVIDENCE_PATCH = 111
PAYLOAD_SCHEMA = {EVIDENCE_GRID: 'evidence.grid.v1',
                  EVIDENCE_PATCH: 'evidence.patch.v1'}

#: A cell nobody has looked at. The sentinel is carried twice -- occupancy 255
#: and timestamp 0 -- and the two must agree; either check alone is valid, so a
#: consumer may use whichever it has to hand, and a record where they disagree
#: is rejected rather than interpreted. Probabilities occupy 0..254 so that 255
#: stays outside the scale rather than meaning "almost certainly occupied".
NEVER_OBSERVED = 255
PROBABILITY_SCALE = 254

REGISTRY_KEYS = ('maps.schema', 'maps.manifest', 'maps.generation',
                 'maps.profile_name', 'maps.profile_digest', 'producer_id',
                 'vehicle_id', 'provider_session_id', 'clock_domain_id')

#: Publish cadence and freshness horizon, both in simulated seconds, both
#: configurable. The producer publishes on the cadence whether or not content
#: changed -- the revision advances only on change, but the sim timestamp
#: always does -- so a live map is distinguishable from a dead one without
#: wall-clock guessing.
DEFAULT_CADENCE_S = 1.0
DEFAULT_STALENESS_HORIZON_S = 5.0

#: How many revisions of history a source ring holds. A consumer polling on
#: MAP_HZ drains far faster than the 1 Hz cadence; the slack is for one that
#: stalls briefly, not for replay.
DEFAULT_RING_SLOTS = 4

#: An upper bound on a declared geometry, so a corrupt or hostile header cannot
#: ask a consumer to allocate an arbitrary amount of memory before the length
#: check that would have rejected it. Four million cells is 20 MB of payload --
#: far past the 0.25-0.5 m band a sane profile uses, and still finite.
MAX_GRID_CELLS = 4_000_000

# magic, version, header bytes, origin east/south metres, cell metres, slab
# floor and thickness, width, height, layers. Repeated in every record rather
# than only in the manifest: a record that cannot be checked against the
# geometry it claims is a record that can be misread silently.
GRID_HEADER = struct.Struct('<4sHHddfffIII')
GRID_MAGIC = b'EGR1'
GRID_VERSION = 1


# -- names -------------------------------------------------------------------

def registry_name(instance: str) -> str:
    """The stable discovery address of an instance's maps plane."""
    name = shared_names(instance)['maps']
    if len(name.encode()) > 254: raise ValueError('instance: shared-memory name exceeds 254 bytes')
    return name


def channel_name(instance: str, session: str, generation: int, source: str) -> str:
    """One source's ring, qualified by producer session and generation.

    ``m`` rather than the sensor plane's ``s`` so the two planes' names cannot
    collide even for an instance that publishes a sensor and a source under the
    same id.
    """
    name = (f'/dvision2.{validate_id(instance)}.m{uuid.UUID(session).hex}'
            f'.g{int(generation)}.map.{validate_id(source)}')
    if len(name.encode()) > 254: raise ValueError('map channel name exceeds 254 bytes')
    return name


# -- geometry ----------------------------------------------------------------

@dataclass(frozen=True)
class GridGeometry:
    """Where a grid sits in the world and how finely it divides it.

    The map frame -- x east, y south, z up -- and world-fixed, not
    drone-centred: a producer publishes beliefs about the area, not about
    itself. ``layers`` is 1 today and the format reserves nothing else for
    z-stacking; one layer is a *slab* covering ``[z0_m, z0_m + dz_m)``, not a
    plane, so full 3D later means more layers rather than a new format.
    """

    origin_x_m: float
    origin_y_m: float
    cell_m: float
    width: int
    height: int
    layers: int = 1
    z0_m: float = 0.0
    dz_m: float = 3.0

    def __post_init__(self) -> None:
        for name in ('origin_x_m', 'origin_y_m', 'cell_m', 'z0_m', 'dz_m'):
            object.__setattr__(self, name, float(getattr(self, name)))
        for name in ('width', 'height', 'layers'):
            object.__setattr__(self, name, int(getattr(self, name)))
        if not np.isfinite([self.origin_x_m, self.origin_y_m, self.cell_m,
                            self.z0_m, self.dz_m]).all():
            raise ValueError('grid geometry must be finite')
        if self.cell_m <= 0: raise ValueError('grid cell_m must be positive')
        if self.dz_m <= 0: raise ValueError('grid dz_m must be positive')
        if min(self.width, self.height, self.layers) < 1:
            raise ValueError('grid dimensions must be positive')
        if self.cells > MAX_GRID_CELLS:
            raise ValueError(f'grid declares {self.cells} cells, over the {MAX_GRID_CELLS} limit')

    @classmethod
    def from_extent(cls, width_m: float, height_m: float, cell_m: float, *,
                    origin_x_m: float = 0.0, origin_y_m: float = 0.0,
                    layers: int = 1, z0_m: float = 0.0, dz_m: float = 3.0) -> 'GridGeometry':
        """Cover at least ``width_m`` x ``height_m``, rounding cells outward."""
        cell_m = float(cell_m)
        if cell_m <= 0: raise ValueError('cell_m must be positive')
        return cls(origin_x_m, origin_y_m, cell_m,
                   int(np.ceil(float(width_m) / cell_m)),
                   int(np.ceil(float(height_m) / cell_m)),
                   layers, z0_m, dz_m)

    @property
    def shape(self) -> tuple[int, int, int]: return (self.layers, self.height, self.width)
    @property
    def cells(self) -> int: return self.layers * self.height * self.width
    @property
    def extent_m(self) -> tuple[float, float]:
        return (self.width * self.cell_m, self.height * self.cell_m)

    @property
    def record_bytes(self) -> int:
        """One complete record: DVS1 header, grid header, then five bytes a cell."""
        return HEADER.size + GRID_HEADER.size + self.cells * 5

    def bounds_m(self) -> tuple[float, float, float, float]:
        """``(x0, y0, x1, y1)`` of the covered area, in map metres."""
        width_m, height_m = self.extent_m
        return (self.origin_x_m, self.origin_y_m,
                self.origin_x_m + width_m, self.origin_y_m + height_m)

    def to_cell(self, x_m: float, y_m: float) -> tuple[int, int] | None:
        """``(column, row)`` for a point in map metres, or None if outside.

        Flooring rather than truncating: ``int()`` rounds toward zero, so every
        point in the half-cell just outside the west or north edge would land
        on column or row 0 instead of being rejected here.
        """
        col = int(np.floor((float(x_m) - self.origin_x_m) / self.cell_m))
        row = int(np.floor((float(y_m) - self.origin_y_m) / self.cell_m))
        if 0 <= col < self.width and 0 <= row < self.height: return col, row
        return None

    def cell_centre_m(self, col: int, row: int) -> tuple[float, float]:
        return (self.origin_x_m + (int(col) + .5) * self.cell_m,
                self.origin_y_m + (int(row) + .5) * self.cell_m)

    def as_dict(self) -> dict[str, Any]:
        return dict(origin_m=[self.origin_x_m, self.origin_y_m], cell_m=self.cell_m,
                    width=self.width, height=self.height, layers=self.layers,
                    z0_m=self.z0_m, dz_m=self.dz_m)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> 'GridGeometry':
        origin = value.get('origin_m', (0., 0.))
        if not isinstance(origin, (list, tuple)) or len(origin) != 2:
            raise ValueError('map geometry origin_m must be a pair of metres')
        return cls(origin[0], origin[1], value['cell_m'], value['width'],
                   value['height'], value.get('layers', 1),
                   value.get('z0_m', 0.), value.get('dz_m', 3.))


# -- quantization ------------------------------------------------------------

def quantize(probability) -> np.ndarray:
    """Occupancy probability to the wire's uint8, never onto the sentinel."""
    scaled = np.rint(np.clip(np.asarray(probability, np.float64), 0., 1.) * PROBABILITY_SCALE)
    return scaled.astype(np.uint8)


def dequantize(occupancy) -> np.ndarray:
    """The wire's uint8 back to probability, with NaN where nobody has looked."""
    values = np.asarray(occupancy, np.uint8)
    result = values.astype(np.float32) / PROBABILITY_SCALE
    result[values == NEVER_OBSERVED] = np.nan
    return result


def stamp_ms(sim_time_s) -> np.ndarray:
    """Simulated seconds to the wire's uint32 milliseconds.

    Clamped up off zero: zero is the never-observed sentinel, so a cell
    genuinely observed at simulated time zero has to round to one millisecond
    rather than claim it was never seen. The field wraps after about 49 days of
    simulated time, which no run in this repository approaches.
    """
    ms = np.rint(np.asarray(sim_time_s, np.float64) * 1000.)
    return np.clip(ms, 1., float(2**32 - 1)).astype(np.uint32)


# -- the grid ----------------------------------------------------------------

@dataclass(frozen=True)
class EvidenceGrid:
    """One source's belief about the area, as published or as received."""

    geometry: GridGeometry
    occupancy: np.ndarray        # uint8  (layers, height, width)
    observed_ms: np.ndarray      # uint32 (layers, height, width)
    source: str = ''
    revision: int = 0
    sim_time_s: float = 0.0
    status: int = STATUS_VALID
    entry: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, 'occupancy', np.ascontiguousarray(self.occupancy, np.uint8))
        object.__setattr__(self, 'observed_ms', np.ascontiguousarray(self.observed_ms, np.uint32))
        shape = self.geometry.shape
        if self.occupancy.shape != shape or self.observed_ms.shape != shape:
            raise ValueError(f'grid planes must be {shape}, got '
                             f'{self.occupancy.shape} and {self.observed_ms.shape}')
        disagree = (self.occupancy == NEVER_OBSERVED) != (self.observed_ms == 0)
        if disagree.any():
            raise ValueError(f'{int(disagree.sum())} cells disagree about being '
                             'never-observed: occupancy 255 must coincide with timestamp 0')

    @classmethod
    def blank(cls, geometry: GridGeometry, source: str = '') -> 'EvidenceGrid':
        """A grid nobody has looked at any part of yet."""
        return cls(geometry, np.full(geometry.shape, NEVER_OBSERVED, np.uint8),
                   np.zeros(geometry.shape, np.uint32), source)

    @property
    def never_observed(self) -> np.ndarray: return self.occupancy == NEVER_OBSERVED
    @property
    def observed(self) -> np.ndarray: return self.occupancy != NEVER_OBSERVED
    @property
    def probability(self) -> np.ndarray: return dequantize(self.occupancy)

    def layer(self, index: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """One layer's occupancy and timestamp planes, as ``(height, width)``."""
        return self.occupancy[int(index)], self.observed_ms[int(index)]

    def age_s(self, sim_now_s: float) -> np.ndarray:
        """Per cell, simulated seconds since it was last observed; NaN if never."""
        age = float(sim_now_s) - self.observed_ms.astype(np.float64) / 1000.
        age[self.never_observed] = np.nan
        return age.astype(np.float32)

    def newest_observation_s(self) -> float:
        """The most recent observation anywhere in the grid, or 0.0 if none."""
        return float(self.observed_ms.max()) / 1000. if self.observed.any() else 0.

    def probability_at(self, x_m: float, y_m: float, layer: int = 0) -> float | None:
        cell = self.geometry.to_cell(x_m, y_m)
        if cell is None: return None
        col, row = cell
        value = self.occupancy[int(layer), row, col]
        return None if value == NEVER_OBSERVED else float(value) / PROBABILITY_SCALE

    def observed_at(self, x_m: float, y_m: float, layer: int = 0) -> float | None:
        cell = self.geometry.to_cell(x_m, y_m)
        if cell is None: return None
        col, row = cell
        value = self.observed_ms[int(layer), row, col]
        return None if value == 0 else float(value) / 1000.


# -- codec -------------------------------------------------------------------

def encode_grid(geometry: GridGeometry, occupancy, observed_ms) -> bytes:
    """One ``evidence.grid.v1`` payload: header, occupancy plane, timestamp plane.

    Layer, row, column order in both planes, little-endian, so a consumer can
    reshape straight out of the buffer.
    """
    occupancy = np.ascontiguousarray(occupancy, np.uint8)
    observed_ms = np.ascontiguousarray(observed_ms, np.uint32)
    if occupancy.shape != geometry.shape or observed_ms.shape != geometry.shape:
        raise ValueError(f'grid planes must be {geometry.shape}')
    header = GRID_HEADER.pack(
        GRID_MAGIC, GRID_VERSION, GRID_HEADER.size, geometry.origin_x_m,
        geometry.origin_y_m, geometry.cell_m, geometry.z0_m, geometry.dz_m,
        geometry.width, geometry.height, geometry.layers)
    return header + occupancy.tobytes() + observed_ms.astype('<u4').tobytes()


def decode_grid(payload: bytes, *, expect: GridGeometry | None = None
                ) -> tuple[GridGeometry, np.ndarray, np.ndarray]:
    """Decode one payload, rejecting anything that does not add up.

    Every failure raises with a message naming what was wrong. Nothing here
    returns a partial grid: a torn, oversize or geometry-inconsistent record is
    not evidence about the world, and treating it as such is how a consumer
    ends up quietly planning through a wall.
    """
    raw = bytes(payload)
    if len(raw) < GRID_HEADER.size:
        raise ValueError(f'truncated evidence grid: {len(raw)} bytes, header needs {GRID_HEADER.size}')
    magic, version, header_bytes, ox, oy, cell, z0, dz, width, height, layers = \
        GRID_HEADER.unpack_from(raw)
    if magic != GRID_MAGIC:
        raise ValueError(f'not an evidence grid: magic {magic!r}')
    if version != GRID_VERSION or header_bytes != GRID_HEADER.size:
        raise ValueError(f'unsupported evidence grid v{version} with {header_bytes}-byte header')
    geometry = GridGeometry(ox, oy, cell, width, height, layers, z0, dz)
    expected = GRID_HEADER.size + geometry.cells * 5
    if len(raw) != expected:
        raise ValueError(f'evidence grid declares {geometry.width}x{geometry.height}'
                         f'x{geometry.layers} = {expected} bytes, record carries {len(raw)}')
    if expect is not None and geometry != expect:
        raise ValueError(f'evidence grid geometry {geometry.as_dict()} does not match '
                         f'the manifest geometry {expect.as_dict()}')
    end = GRID_HEADER.size + geometry.cells
    occupancy = np.frombuffer(raw, np.uint8, geometry.cells, GRID_HEADER.size).reshape(geometry.shape)
    observed = np.frombuffer(raw, '<u4', geometry.cells, end).reshape(geometry.shape)
    disagree = (occupancy == NEVER_OBSERVED) != (observed == 0)
    if disagree.any():
        raise ValueError(f'{int(disagree.sum())} cells disagree about being never-observed: '
                         'occupancy 255 must coincide with timestamp 0')
    return geometry, occupancy.copy(), observed.astype(np.uint32)


# -- publication -------------------------------------------------------------

class MapPublisher:
    """Owns an instance's maps plane: the registry, the manifest, every ring.

    One producer per instance id in this milestone. The multi-producer case --
    a registry flock and owner tokens -- is deferred alongside evidence fusion;
    if a second producer ever appears before then, extend this manifest rather
    than improvising a second registry.
    """

    def __init__(self, instance: str, geometry: GridGeometry,
                 sources: Iterable[dict[str, Any]], *, producer: str = 'dalg',
                 profile_name: str = '', profile_digest: str = '',
                 cadence_s: float = DEFAULT_CADENCE_S,
                 staleness_horizon_s: float = DEFAULT_STALENESS_HORIZON_S,
                 ring_slots: int = DEFAULT_RING_SLOTS, session: str | None = None,
                 generation: int = 1, context: dict | None = None,
                 activate: bool = True) -> None:
        self.pm = load_pymembus()
        self.instance = validate_id(instance)
        self.session = session or uuid.uuid4().hex
        self.generation = int(generation)
        self.geometry = geometry
        self.producer = producer
        self.cadence_s = float(cadence_s)
        self.staleness_horizon_s = float(staleness_horizon_s)
        self.rings: dict[str, RecordRing] = {}
        self.revisions: dict[str, int] = {}
        self._last_payload: dict[str, bytes] = {}
        self.closed = False
        self.active = False
        # Every source of one generation shares one frame, localization and
        # clock epoch, mapping epoch and geometry revision (DV-MAPPING Q10).
        self.context = dict(frame_id='local', localization_epoch=0, clock_epoch=0,
                            mapping_epoch=self.generation, geometry_revision=self.generation,
                            clock_domain_id=self.instance)
        self.context.update(context or {})

        capacity = (geometry.record_bytes + 4096) * max(1, int(ring_slots))
        entries: dict[str, dict[str, Any]] = {}
        try:
            for source in sources:
                sid = validate_id(str(source['id']))
                if sid in entries: raise ValueError(f'duplicate map source {sid!r}')
                name = channel_name(self.instance, self.session, self.generation, sid)
                self.rings[sid] = RecordRing(name, capacity, create=True)
                self.revisions[sid] = 0
                entries[sid] = dict(
                    id=sid, sensor=str(source.get('sensor', '')),
                    sensor_type=str(source.get('sensor_type', '')),
                    algorithm=str(source.get('algorithm', '')),
                    channel=name, capacity=capacity,
                    record_bytes=geometry.record_bytes,
                    payload_type=EVIDENCE_GRID,
                    payload_schema=PAYLOAD_SCHEMA[EVIDENCE_GRID],
                    profile_digest=profile_digest, **self.context)
            self.manifest = dict(
                schema=SCHEMA, provider_session_id=self.session,
                vehicle_id=self.instance, producer=producer,
                generation=self.generation, clock_domain_id=self.instance,
                profile_name=profile_name, profile_digest=profile_digest,
                cadence_s=self.cadence_s,
                staleness_horizon_s=self.staleness_horizon_s,
                map=geometry.as_dict(), sources=entries, context=self.context)
            encoded = json_bytes(self.manifest).decode()
            if len(encoded.encode()) > MAX_MANIFEST: raise ValueError('map manifest exceeds 64 KiB')
            if activate: self.activate()

        except Exception:
            for ring in self.rings.values():
                ring.close(); ring.unlink()
            self.rings.clear()
            raise

    def activate(self):
        if self.active: return
        self._create_registry()
        self.active = True

    def _create_registry(self):
        self.registry = self.pm.memkv()
        self.pm.memkv.remove(registry_name(self.instance))
        if not self.registry.create(registry_name(self.instance), len(REGISTRY_KEYS),
                memkv_aligned_name_len(32, MAX_MANIFEST), MAX_MANIFEST, True):
            raise RuntimeError(self.pm.last_error_message())
        for index, key in enumerate(REGISTRY_KEYS): self.registry.setName(index, key)
        values = dict(zip(REGISTRY_KEYS, (SCHEMA, json_bytes(self.manifest).decode(),
            str(self.generation), self.manifest['profile_name'], self.manifest['profile_digest'],
            self.producer, self.instance, self.session, self.context['clock_domain_id'])))
        if not self.registry.setAll(values): raise RuntimeError(self.pm.last_error_message())

    def _ensure_registry(self) -> bool:
        """Recreate the discovery registry if something removed it; True when it had to.

        pymembus unlinks a memkv *name* when the handle that created it closes,
        even after a successor has re-created that name. So when dalg stages a
        new generation and then closes the old publisher, the old handle's
        close deletes the new registry, and every consumer is left following a
        generation that no longer publishes (seen live: dnav held a stale map
        forever after a coverage reset). The live publisher checks its own
        registry each publication and puts it back.
        """
        if not self.active or self.closed: return False
        probe = self.pm.memkv()
        if probe.open(registry_name(self.instance)):
            try: owner = probe.getAll().get('provider_session_id')
            finally: probe.close()
            if owner == self.session: return False
            if owner: return False  # a newer producer owns the name: never take it back
        try: self.registry.close()
        except Exception: pass
        self._create_registry()
        return True

    @property
    def sources(self) -> dict[str, dict[str, Any]]: return self.manifest['sources']

    def publish(self, source: str, occupancy, observed_ms, sim_time_s: float, *,
                status: int = STATUS_VALID) -> int:
        """Publish one complete grid and return the revision it carries.

        The revision advances only when the content does; the simulated
        timestamp advances every time. That pair is what lets a consumer tell a
        map that is unchanged from one that is dead without consulting a wall
        clock.
        """
        if self.closed: raise RuntimeError('publish: the maps plane is closed')
        if source not in self.rings: raise KeyError(source)
        self._ensure_registry()
        payload = encode_grid(self.geometry, occupancy, observed_ms)
        if payload != self._last_payload.get(source):
            self.revisions[source] += 1
            self._last_payload[source] = payload
        revision = self.revisions[source]
        self.rings[source].write(encode_record(
            self.session, self.generation, source, revision, revision,
            int(round(float(sim_time_s) * 1e6)), payload,
            payload_type=EVIDENCE_GRID, status=status,
            reset_epoch=self.context['mapping_epoch'], clock_epoch=self.context['clock_epoch']))
        return revision

    def publish_grid(self, grid: EvidenceGrid, sim_time_s: float, *,
                     status: int = STATUS_VALID) -> int:
        if grid.geometry != self.geometry:
            raise ValueError('grid geometry does not match the published plane')
        return self.publish(grid.source, grid.occupancy, grid.observed_ms,
                            sim_time_s, status=status)

    def close(self) -> None:
        """Unlink everything this producer created. A clean shutdown leaves nothing."""
        if self.closed: return
        self.closed = True
        for ring in self.rings.values():
            ring.close(); ring.unlink()
        self.rings.clear()
        if self.active:
            # A staged successor may already own the name. Never unlink it.
            probe = self.pm.memkv()
            if probe.open(registry_name(self.instance)):
                values = probe.getAll(); probe.close()
                if values.get('provider_session_id') == self.session:
                    self.pm.memkv.remove(registry_name(self.instance))
            self.registry.close()


# -- discovery and intake ----------------------------------------------------

@dataclass
class SourceState:
    """What a consumer knows about one source it is following."""
    entry: dict[str, Any]
    grid: EvidenceGrid | None = None
    revision: int = 0
    sim_time_s: float = 0.0       # newest record, whether or not content changed
    first_sim_time_s: float | None = None
    records: int = 0
    gaps: int = 0                 # revisions this consumer never saw
    rejected: int = 0
    last_reason: str = ''

    def rate_hz(self) -> float | None:
        """Records per simulated second, or None before there is an interval."""
        if self.first_sim_time_s is None or self.records < 2: return None
        elapsed = self.sim_time_s - self.first_sim_time_s
        return (self.records - 1) / elapsed if elapsed > 0 else None


class MapSession:
    """Read-only intake for an instance's maps plane.

    Creates nothing, so start order is irrelevant: it probes the stable
    registry name with bounded wall-clock backoff until a producer commits a
    manifest, and re-discovers whenever the session or generation changes. All
    data cadence and staleness stay on the simulated clock.
    """

    def __init__(self, instance: str, *, probe_interval_s: float = .25,
                 max_probe_interval_s: float = 2.0) -> None:
        self.instance = validate_id(instance)
        self.pm = load_pymembus()
        self.manifest: dict[str, Any] = {}
        self.identity: tuple[str, int] | None = None
        self.geometry: GridGeometry | None = None
        self.states: dict[str, SourceState] = {}
        self.rings: dict[str, RecordRing] = {}
        self.probe_interval_s = float(probe_interval_s)
        self.max_probe_interval_s = float(max_probe_interval_s)
        self._backoff_s = float(probe_interval_s)
        self._last_probe = -1e9
        self.rejections: list[tuple[str, str]] = []
        self.discoveries = 0
        self.last_seen_identity: tuple[str, int] | None = None
        self.closed = False

    # -- discovery ---------------------------------------------------------

    @property
    def sources(self) -> dict[str, dict[str, Any]]: return self.manifest.get('sources', {})

    @property
    def staleness_horizon_s(self) -> float:
        return float(self.manifest.get('staleness_horizon_s', DEFAULT_STALENESS_HORIZON_S))

    @property
    def cadence_s(self) -> float:
        return float(self.manifest.get('cadence_s', DEFAULT_CADENCE_S))

    @property
    def context(self) -> dict[str, Any]:
        """Frame, localization/clock/mapping epochs and geometry revision of this generation."""
        return self.manifest.get('context', {})

    def _read_manifest(self) -> dict[str, Any]:
        kv = self.pm.memkv()
        if not kv.open(registry_name(self.instance)): return {}
        try:
            values = kv.getAll()
        finally:
            kv.close()
        if values.get('maps.schema') != SCHEMA:
            self._reject('', 'unsupported map manifest schema'); return {}
        try:
            manifest = json.loads(values['maps.manifest'])
        except (KeyError, TypeError, ValueError) as exc:
            self._reject('', f'unreadable map manifest: {exc}')
            return {}
        return manifest if isinstance(manifest, dict) else {}

    def connect(self) -> bool:
        """Attach to the current session and generation, retrying on backoff."""
        if self.closed: return False
        now = time.monotonic()
        if self.identity is not None and self.rings:
            if now - self._last_probe < self.probe_interval_s: return True
        elif now - self._last_probe < self._backoff_s:
            return False
        self._last_probe = now
        manifest = self._read_manifest()
        identity = ((str(manifest.get('provider_session_id', '')),
                     int(manifest.get('generation', 0))) if manifest else None)
        if identity == self.identity and self.rings:
            self._backoff_s = self.probe_interval_s
            return True
        self._close_rings()
        if not manifest:
            # An old mapping can stay readable after unlink, so "no manifest"
            # is the only evidence of absence there is -- and it is a reason to
            # keep retrying, not to give up.
            self._backoff_s = min(self.max_probe_interval_s, self._backoff_s * 2)
            return False
        try:
            geometry = GridGeometry.from_dict(manifest['map'])
        except (KeyError, TypeError, ValueError) as exc:
            self._reject('', f'unusable map manifest geometry: {exc}')
            self._backoff_s = min(self.max_probe_interval_s, self._backoff_s * 2)
            return False
        rings: dict[str, RecordRing] = {}
        try:
            for sid, entry in manifest.get('sources', {}).items():
                rings[sid] = RecordRing(entry['channel'], int(entry['capacity']))
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            for ring in rings.values(): ring.close()
            # A rollover racing the open is normal: retry rather than report.
            self._reject('', f'map channel unavailable: {exc}')
            self._backoff_s = min(self.max_probe_interval_s, self._backoff_s * 2)
            return False
        self.manifest, self.identity, self.geometry = manifest, identity, geometry
        self.rings = rings
        self.states = {sid: SourceState(entry) for sid, entry in manifest['sources'].items()}
        self.last_seen_identity = identity
        self._backoff_s = self.probe_interval_s
        self.discoveries += 1
        return True

    def _close_rings(self) -> None:
        """Drop the transport, keep what this consumer last knew.

        A producer that exits unlinks its plane, and forgetting its sources at
        that moment would leave a viewer with nothing to display and nothing to
        complain about -- the map would simply vanish. Keeping the last grid and
        letting its age run past the horizon is what turns a departed producer
        into a `stale_map` an operator can read. A *different* session is the
        other case entirely, and :meth:`connect` replaces the state wholesale
        when it adopts one.
        """
        for ring in self.rings.values(): ring.close()
        self.rings.clear()
        self.identity = None

    def _reject(self, source: str, reason: str) -> None:
        """Record a rejection. A failure is a message, never silence."""
        self.rejections.append((source, reason))
        del self.rejections[:-64]
        state = self.states.get(source)
        if state is not None:
            state.rejected += 1
            state.last_reason = reason

    # -- intake ------------------------------------------------------------

    def poll(self) -> None:
        """Drain every source ring, keeping the newest grid of each."""
        if not self.connect(): return
        for sid, ring in self.rings.items():
            state = self.states[sid]
            for raw in self._drain(ring, sid):
                self._admit(sid, state, raw)

    def _drain(self, ring: RecordRing, sid: str, limit: int = 64):
        """Raw records, with decode failures reported rather than swallowed.

        ``RecordRing.drain`` raises out of the generator on a bad record, which
        would abandon every good record still queued behind it. This plane owes
        the operator a message per rejection and the rest of the queue, so it
        drives the ring directly.
        """
        for _ in range(limit):
            if not ring.handle.poll(): break
            raw, overrun = ring.handle.read_bytes_with_overrun(0)
            ring.overruns += int(overrun)
            if not raw: continue
            try:
                yield decode_record(raw)
            except ValueError as exc:
                self._reject(sid, f'{sid}: {exc}')

    def _admit(self, sid: str, state: SourceState, record: dict[str, Any]) -> None:
        if (record['provider_session_id'], record['generation']) != self.identity: return
        if record['sensor_id'] != sid: return
        if record['payload_type'] != EVIDENCE_GRID:
            self._reject(sid, f'{sid}: payload type {record["payload_type"]} is not '
                              f'{EVIDENCE_GRID} ({PAYLOAD_SCHEMA[EVIDENCE_GRID]})')
            return
        try:
            geometry, occupancy, observed = decode_grid(record['payload'], expect=self.geometry)
        except ValueError as exc:
            self._reject(sid, f'{sid}: {exc}')
            return
        # One published generation shares one mapping and clock epoch; a
        # record that disagrees with its manifest is never mixed in.
        context = self.context
        if (record['reset_epoch'] != context.get('mapping_epoch') or
                record['clock_epoch'] != context.get('clock_epoch')):
            self._reject(sid, f'{sid}: evidence epoch mismatch (record mapping/clock epoch '
                              f'{record["reset_epoch"]}/{record["clock_epoch"]}, manifest '
                              f'{context.get("mapping_epoch")}/{context.get("clock_epoch")})')
            return
        sim_time_s = record['sim_time_us'] / 1e6
        revision = int(record['sequence'])
        state.records += 1
        if state.first_sim_time_s is None: state.first_sim_time_s = sim_time_s
        # Records arrive in order on one ring, so a jump in the revision is a
        # revision this consumer never saw -- which costs nothing but time,
        # because every record is complete. It is still worth counting.
        if state.revision and revision > state.revision + 1:
            state.gaps += revision - state.revision - 1
        state.sim_time_s = max(state.sim_time_s, sim_time_s)
        if revision < state.revision: return
        state.revision = revision
        state.grid = EvidenceGrid(geometry, occupancy, observed, sid, revision,
                                  sim_time_s, record['status'], state.entry)

    # -- queries -----------------------------------------------------------

    def latest(self, source: str) -> EvidenceGrid | None:
        state = self.states.get(source)
        return state.grid if state else None

    def age_s(self, source: str, sim_now_s: float) -> float | None:
        """Simulated seconds since this source last published, or None if never."""
        state = self.states.get(source)
        if state is None or not state.records: return None
        return max(0., float(sim_now_s) - state.sim_time_s)

    def stale(self, source: str, sim_now_s: float) -> bool:
        """Whether the newest record is older than the declared horizon.

        A source that has published nothing at all is stale: a consumer with no
        evidence is in the same position as one whose evidence has expired, and
        both must say so rather than plan on nothing.
        """
        age = self.age_s(source, sim_now_s)
        return True if age is None else age > self.staleness_horizon_s

    def report(self, sim_now_s: float) -> dict[str, Any]:
        sources = {}
        for sid, state in self.states.items():
            sources[sid] = dict(
                revision=state.revision, sim_time_s=state.sim_time_s,
                age_s=self.age_s(sid, sim_now_s), stale=self.stale(sid, sim_now_s),
                rate_hz=state.rate_hz(), records=state.records, gaps=state.gaps,
                rejected=state.rejected, reason=state.last_reason,
                status=(state.grid.status if state.grid else STATUS_INVALID),
                sensor=state.entry.get('sensor', ''),
                algorithm=state.entry.get('algorithm', ''))
        seen = self.identity or self.last_seen_identity
        return dict(connected=self.identity is not None,
                    provider_session_id=seen[0] if seen else '',
                    generation=seen[1] if seen else 0,
                    producer=self.manifest.get('producer', ''),
                    cadence_s=self.cadence_s,
                    staleness_horizon_s=self.staleness_horizon_s,
                    map=self.geometry.as_dict() if self.geometry else {},
                    context=dict(self.context),
                    discoveries=self.discoveries, sources=sources,
                    overruns=sum(ring.overruns for ring in self.rings.values()),
                    rejections=list(self.rejections))

    def close(self) -> None:
        self._close_rings()
        self.states.clear()
        self.manifest = {}; self.geometry = None
        self.closed = True


def open_maps(instance: str) -> MapSession | None:
    """A connected session, or None while no producer has committed a manifest."""
    session = MapSession(instance)
    if session.connect(): return session
    session.close()
    return None


# -- the dump tool -----------------------------------------------------------

def status_clock(instance: str):
    """Simulated seconds from the instance's status plane, if one is published."""
    pm = load_pymembus()
    handle = pm.memkv()
    if not handle.open(shared_names(instance)['status']):
        return None
    def clock():
        try: return float(handle.getAll().get('sim.time_s', 0.))
        except (TypeError, ValueError): return 0.
    return clock


def _dump(instance: str, seconds: float, only: str | None) -> int:
    """Print the manifest, then one line per record header, until time runs out."""
    import sys

    session = MapSession(instance)
    clock = None
    deadline = time.monotonic() + float(seconds)
    printed_manifest = False
    seen: dict[str, tuple[int, float]] = {}
    reported = 0
    stale_said: dict[str, bool] = {}
    connected = False
    try:
        while time.monotonic() < deadline:
            if clock is None: clock = status_clock(instance)
            live = session.connect()
            if live != connected and printed_manifest:
                connected = live
                print(f'producer {"returned" if live else "gone"}: '
                      f'{registry_name(instance)} '
                      f'{"committed a manifest" if live else "has no manifest"}')
            connected = live
            if not live and not printed_manifest:
                time.sleep(.1)
                continue
            if not printed_manifest:
                print(json.dumps(session.manifest, indent=2, sort_keys=True))
                print(f'record_bytes={session.geometry.record_bytes}  '
                      f'cells={session.geometry.cells}  '
                      f'cadence={session.cadence_s:g}s  '
                      f'horizon={session.staleness_horizon_s:g}s (simulated)')
                printed_manifest = True
            session.poll()
            sim_now = clock() if clock else None
            for sid, state in session.states.items():
                if only and sid != only: continue
                mark = (state.revision, state.sim_time_s)
                if state.grid is not None and seen.get(sid) != mark:
                    seen[sid] = mark
                    grid = state.grid
                    print(f'{sid}: rev={state.revision} sim={state.sim_time_s:9.3f}s '
                          f'status={"valid" if grid.status == STATUS_VALID else "invalid"} '
                          f'{grid.geometry.width}x{grid.geometry.height}x{grid.geometry.layers} '
                          f'cell={grid.geometry.cell_m:g}m '
                          f'origin=({grid.geometry.origin_x_m:g},{grid.geometry.origin_y_m:g}) '
                          f'observed={int(grid.observed.sum())}/{grid.geometry.cells} '
                          f'newest_obs={grid.newest_observation_s():.3f}s '
                          f'rate={f"{state.rate_hz():.2f}" if state.rate_hz() else "--"}/sim-s '
                          f'gaps={state.gaps}')
                if sim_now is None: continue
                is_stale = session.stale(sid, sim_now)
                if stale_said.get(sid) != is_stale:
                    stale_said[sid] = is_stale
                    age = session.age_s(sid, sim_now)
                    print(f'{sid}: {"stale_map" if is_stale else "fresh"} '
                          f'(age={"never" if age is None else f"{age:.3f}s"}, '
                          f'horizon={session.staleness_horizon_s:g}s, sim={sim_now:.3f}s)')
            while reported < len(session.rejections):
                source, reason = session.rejections[reported]
                reported += 1
                print(f'rejected: {reason}', file=sys.stderr)
            time.sleep(.05)
        if not printed_manifest:
            print(f'dcmn.maps: no map manifest on {registry_name(instance)} '
                  f'after {seconds:g}s', file=sys.stderr)
            return 1
        if clock is None:
            print('dcmn.maps: no status plane, so staleness was not evaluated',
                  file=sys.stderr)
    finally:
        session.close()
    return 0


def main(argv=None) -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description='inspect an instance\'s evidence-grid plane')
    parser.add_argument('--dump', action='store_true',
                        help='print the manifest and every record header seen')
    parser.add_argument('--id', required=True, help='instance id')
    parser.add_argument('--seconds', type=float, default=10.0,
                        help='how long to watch, in wall seconds')
    parser.add_argument('--source', default=None, help='limit output to one source')
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    if not args.dump: parser.error('--dump is the only mode')
    if args.seconds <= 0: parser.error('--seconds must be positive')
    try:
        return _dump(args.id, args.seconds, args.source)
    except (ValueError, RuntimeError) as exc:
        print(f'dcmn.maps: {exc}', file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == '__main__':
    raise SystemExit(main())
