"""The reference-imagery plane: optional PNG backgrounds, published as evidence-adjacent data.

DV-MAPPING §7 draws one line this module lives on: reference imagery -- a
rendered simulator map today, a survey or a satellite tile tomorrow -- is
**display and reporting data, never planning input**. Nothing here is
converted into occupancy or cost; a consumer that switches, replaces or
removes the image must produce numerically identical evidence and routes,
because nothing in `dalg` or `dnav`'s algorithmic paths reads this plane at
all. The only readers are the shared renderer in :mod:`dcmn.map_pane`, the
report writers and this plane's own inspection tool.

The shape mirrors the evidence plane deliberately, because the problems are
the same problems:

* a stable registry name a consumer can probe before any producer exists, and
  a small discovery manifest committed atomically with ``setAll``;
* one memmsg ring per image, qualified by producer session and generation,
  carrying DVS1 records whose payload is a complete, self-contained revision --
  length-prefixed JSON metadata followed by the PNG bytes, so a missed record
  costs nothing but time;
* revisions that advance only when content changes, and consumers that
  re-discover wholesale when the producer restarts.

What differs from evidence is worth stating. An evidence grid is republished
on a cadence; a reference image is published once per revision and then sits
there, so the ring's retained records are the *only* delivery mechanism a
late-attaching consumer has -- pymembus replays retained records to a reader
that opens after they were written, which is what makes publish-once, attach
late work at all. And an image is not a belief that improves; it is a
transformation of something outside the vehicle, so every record carries the
full 2D affine from image pixels to the map frame, a checksum, the reference
frame and its localization/clock epochs, and a source category that says
what the picture *is* -- `simulation_truth` in this repository, and
`surveyed_plan` or `satellite` when a hardware provider arrives.

**The pixel contract.** ``(col, row) = (0, 0)`` is the centre of the top-left
pixel, columns increase to the right, rows increase downward, and the affine
``[a, b, c, d, e, f]`` maps pixel centres to map-frame metres (x east, y
south) as::

    x = a*col + b*row + e
    y = c*col + d*row + f

The full affine -- rotation, offset, unequal scale and reflection -- is one
unambiguous answer, which is why it is an affine and not a corner list: two
corners cannot say whether an image is mirrored, and a mirrored map behind a
planner is worse than no map. Row 0 is the top row; a producer whose world
file draws row 0 at the north edge in a y-south frame says so with the
affine, not with a convention nobody wrote down.

**Budgets.** PNG only, at most 16 MiB encoded and 64 MiB decoded per image,
at most :data:`MAX_IMAGES` images per instance, and a consumer keeps at most
the current and one staged decoded revision per image (DV-MAPPING §9). These
are accounted apart from the mapping-memory budget: imagery is optional
decoration, and exhausting it must cost an operator their background, not
their map.
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

import hashlib
import io
import json
import struct
import time
import uuid
from dataclasses import dataclass
from typing import Any

from dcmn.sensors import (STATUS_VALID, RecordRing, decode_record,
                          encode_record, json_bytes)
from dvision2_common import (load_pymembus, memkv_aligned_name_len,
                             shared_names, validate_id)

SCHEMA = 'dvision2.imagery-manifest.v1'
MAX_MANIFEST = 65536

#: Payload type of the imagery plane, packed binary on the same DVS1 wire
#: the sensor and evidence planes use. Like payload type 110 it is one
#: complete, self-contained revision per record -- keyframes and patches would
#: be an optimization nothing here needs, and every record being whole is what
#: keeps a torn record costing nothing.
REFERENCE_IMAGE = 120
PAYLOAD_TYPES = {REFERENCE_IMAGE: 'reference.image.v1'}
IMAGE_SCHEMA = 'dvision2.reference-image.v1'

#: The category a producer says its image is. An open set -- a hardware
#: provider may name its own -- but it is never empty, because the operator
#: looking at a background is entitled to know what kind of thing they are
#: looking at, and the operational UI is entitled to refuse truth imagery.
SIMULATION_TRUTH = 'simulation_truth'

#: Budgets from DV-MAPPING §9: PNG only, at most 16 MiB encoded and 64 MiB
#: decoded per image. "Decoded" is the RGBA bytes the consumer caches, four a
#: pixel, so 64 MiB bounds the decoded plane the same way 16 MiB bounds the
#: wire. Both are checked against the *declared* dimensions before anything
#: is decoded, so a hostile or corrupt record cannot talk a consumer into an
#: allocation before the checks that would have refused it.
MAX_ENCODED_BYTES = 16 * 1024 * 1024
MAX_DECODED_BYTES = 64 * 1024 * 1024

#: How many images one instance's plane may carry. The manifest is small and
#: the rings are bounded, but "a manifest may declare a million images" is a
#: consumer-side DoS, so the plane states a number: one simulator map today,
#: and a handful of surveyed or satellite layers per floor when they arrive.
MAX_IMAGES = 16

#: Revisions of one image a ring retains. An image is published once per
#: revision, so two slots are one live revision and one predecessor -- a
#: late-attaching consumer replays the newest, and a producer that overruns
#: its own ring loses only history the consumer was too slow to want.
DEFAULT_RING_SLOTS = 2

REGISTRY_KEYS = ('imagery.schema', 'imagery.manifest', 'imagery.generation',
                 'producer_id', 'vehicle_id', 'provider_session_id')

# Payload layout: a little-endian u32 metadata length, the JSON metadata,
# then the PNG. The length is what lets a consumer reject a torn record
# before trying to decode anything.
LENGTH = struct.Struct('<I')


# -- names -------------------------------------------------------------------

def registry_name(instance: str) -> str:
    """The stable discovery address of an instance's imagery plane."""
    name = shared_names(instance)['imagery']
    if len(name.encode()) > 254: raise ValueError('instance: shared-memory name exceeds 254 bytes')
    return name


def channel_name(instance: str, session: str, generation: int, image_id: str) -> str:
    """One image's ring, qualified by producer session and generation.

    ``i`` rather than the sensor plane's ``s`` or the maps plane's ``m`` so
    the three planes' channels cannot collide even under one producer.
    """
    name = (f'/dvision2.{validate_id(instance)}.i{uuid.UUID(session).hex}'
            f'.g{int(generation)}.image.{validate_id(image_id)}')
    if len(name.encode()) > 254: raise ValueError('imagery channel name exceeds 254 bytes')
    return name


# -- the affine ---------------------------------------------------------------

def _finite(value) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return number == number and abs(number) != float('inf')


def validate_affine(affine) -> tuple[float, float, float, float, float, float]:
    """Check an image-pixel-to-frame affine and return it as six floats.

    Everything a consumer checks about an image's placement reduces to this:
    six finite numbers and a non-singular pixel-to-metres matrix. Singular
    is refused -- an image collapsed onto a line registers nowhere -- while
    a reflection (negative determinant) is accepted, because mirrored
    reference data is a legitimate thing to publish and a consumer that
    silently un-mirrored it would be lying about what was sent.
    """
    if not isinstance(affine, (list, tuple)) or len(affine) != 6:
        raise ValueError('reference image affine must be a list of 6 numbers [a,b,c,d,e,f]')
    values = tuple(float(v) for v in affine)
    if not all(_finite(v) for v in values):
        raise ValueError('reference image affine must be finite')
    a, b, c, d, _e, _f = values
    if abs(a * d - b * c) < 1e-12:
        raise ValueError('reference image affine is singular: pixels map onto a line')
    return values


def affine_xy(affine, col, row):
    """Pixel coordinates to map-frame metres, under the contract above."""
    a, b, c, d, e, f = affine
    x = a * col + b * row + e
    y = c * col + d * row + f
    return x, y


def affine_inverse(affine) -> tuple[float, float, float, float, float, float]:
    """The frame-to-pixel affine: the inverse every renderer needs.

    The inverse of a non-singular affine is an affine; expressing it in the
    same six-number contract means background sampling and forward placement
    are one transform, described once, with no renderer inventing its own
    matrix algebra.
    """
    a, b, c, d, e, f = affine
    det = a * d - b * c
    return (d / det, -b / det, -c / det, a / det,
            (b * f - d * e) / det, (c * e - a * f) / det)


# -- codec --------------------------------------------------------------------

def encode_revision(metadata: dict, png: bytes) -> bytes:
    """One ``reference.image.v1`` payload: metadata length, metadata, PNG."""
    raw = json_bytes(metadata)
    return LENGTH.pack(len(raw)) + raw + bytes(png)


def decode_revision(payload: bytes) -> tuple[dict[str, Any], bytes]:
    """Split one payload into metadata and PNG, rejecting what does not add up.

    This validates the *envelope* -- lengths, the declared size budget, the
    metadata's shape, the checksum against the bytes that travel with it.
    Decoding the PNG itself is :func:`decode_png`, kept separate because a
    consumer wants to refuse a record before paying for pixels, and a report
    writer wants the checked PNG without a decode at all.
    """
    raw = bytes(payload)
    if len(raw) < LENGTH.size:
        raise ValueError(f'truncated reference image: {len(raw)} bytes, length prefix needs {LENGTH.size}')
    (meta_len,) = LENGTH.unpack_from(raw)
    if meta_len == 0 or LENGTH.size + meta_len > len(raw):
        raise ValueError(f'reference image declares {meta_len} metadata bytes, record carries {len(raw) - LENGTH.size}')
    try:
        metadata = json.loads(raw[LENGTH.size:LENGTH.size + meta_len])
    except ValueError as exc:
        raise ValueError(f'reference image metadata is not JSON: {exc}') from exc
    if not isinstance(metadata, dict) or metadata.get('schema') != IMAGE_SCHEMA:
        raise ValueError(f'not a reference image: schema {metadata.get("schema") if isinstance(metadata, dict) else metadata!r}')
    png = raw[LENGTH.size + meta_len:]
    if len(png) > MAX_ENCODED_BYTES:
        raise ValueError(f'reference image carries {len(png)} encoded bytes, over the {MAX_ENCODED_BYTES} limit')
    for key in ('image_id', 'frame_id', 'source_category'):
        if not isinstance(metadata.get(key), str) or not metadata[key]:
            raise ValueError(f'reference image metadata {key!r} must be a non-empty string')
    if metadata.get('encoding') != 'png':
        raise ValueError(f'reference image encoding {metadata.get("encoding")!r} is not png')
    width, height = metadata.get('width'), metadata.get('height')
    for key, value in (('width', width), ('height', height)):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f'reference image {key} must be a positive integer, got {value!r}')
    if width * height * 4 > MAX_DECODED_BYTES:
        raise ValueError(f'reference image {width}x{height} decodes past the {MAX_DECODED_BYTES} decoded-byte limit')
    revision = metadata.get('revision')
    if not isinstance(revision, int) or isinstance(revision, bool) or revision <= 0:
        raise ValueError(f'reference image revision must be a positive integer, got {revision!r}')
    for key in ('localization_epoch', 'clock_epoch'):
        value = metadata.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f'reference image {key} must be a non-negative integer, got {value!r}')
    checksum = metadata.get('checksum')
    if not isinstance(checksum, str) or len(checksum) != 64:
        raise ValueError('reference image checksum must be a 64-character sha256 hex digest')
    if hashlib.sha256(png).hexdigest() != checksum:
        raise ValueError('reference image checksum does not match the PNG bytes it travels with')
    validate_affine(metadata.get('affine'))
    return metadata, png


def decode_png(png: bytes, width: int, height: int):
    """Decode validated PNG bytes to an RGBA PIL image of the declared size.

    The one place imagery becomes pixels. Both the declared dimensions and
    the decoders' opinion are checked -- a record whose metadata says one
    thing and whose bytes say another is not a record a renderer can place.
    """
    from PIL import Image

    try:
        image = Image.open(io.BytesIO(png))
        image.load()
    except Exception as exc:                       # PIL raises a zoo; all of it means "not a PNG"
        raise ValueError(f'reference image is not a decodable PNG: {exc}') from exc
    if image.size != (int(width), int(height)):
        raise ValueError(f'reference image decodes to {image.size[0]}x{image.size[1]}, '
                         f'metadata declares {width}x{height}')
    return image if image.mode == 'RGBA' else image.convert('RGBA')


def build_metadata(image_id: str, png: bytes, affine, *, revision: int,
                   provider_session_id: str, source_category: str,
                   frame_id: str = 'local', localization_epoch: int = 0,
                   clock_domain_id: str = '', clock_epoch: int = 0,
                   capture_s: float | None = None, registration: dict | None = None,
                   altitude_m: float | None = None, floor: str | None = None) -> dict[str, Any]:
    """Publisher-side validation and metadata assembly: refuse before publishing.

    A producer that publishes a malformed or oversize image has made work for
    every consumer on the plane, so the publisher decodes and measures first.
    The checksum and dimensions are *derived* here, never trusted from the
    caller, which is what makes them true rather than asserted.
    """
    png = bytes(png)
    if len(png) > MAX_ENCODED_BYTES:
        raise ValueError(f'reference image carries {len(png)} encoded bytes, over the {MAX_ENCODED_BYTES} limit')
    if not isinstance(source_category, str) or not source_category:
        raise ValueError('source_category must be a non-empty string')
    image = decode_png(png, *_png_size(png))
    width, height = image.size
    if width * height * 4 > MAX_DECODED_BYTES:
        raise ValueError(f'reference image {width}x{height} decodes past the {MAX_DECODED_BYTES} decoded-byte limit')
    metadata: dict[str, Any] = dict(
        schema=IMAGE_SCHEMA, image_id=validate_id(image_id), revision=int(revision),
        provider_session_id=str(provider_session_id), encoding='png',
        checksum=hashlib.sha256(png).hexdigest(), width=width, height=height,
        affine=list(validate_affine(affine)), frame_id=str(frame_id),
        localization_epoch=int(localization_epoch),
        clock_domain_id=str(clock_domain_id), clock_epoch=int(clock_epoch),
        source_category=source_category)
    # Optional provenance: included when meaningful, absent when not, and
    # never a required field a hardware provider must fake.
    if capture_s is not None: metadata['capture_s'] = float(capture_s)
    if registration is not None: metadata['registration'] = dict(registration)
    if altitude_m is not None: metadata['altitude_m'] = float(altitude_m)
    if floor is not None: metadata['floor'] = str(floor)
    return metadata


def _png_size(png: bytes) -> tuple[int, int]:
    """The dimensions a PNG declares in its IHDR, before any decoding.

    ``decode_png`` still checks the decoders' opinion against this; reading
    IHDR first just means the declared-size budget is enforced before the
    decode that could have been the attack.
    """
    if len(png) < 24 or png[:8] != b'\x89PNG\r\n\x1a\n':
        raise ValueError('reference image is not a PNG (bad signature)')
    width, height = struct.unpack_from('>II', png, 16)
    if width <= 0 or height <= 0:
        raise ValueError(f'reference image declares {width}x{height}')
    return width, height


# -- the revision ---------------------------------------------------------------

@dataclass(frozen=True)
class ReferenceImage:
    """One image revision as published or as received: pixels plus placement.

    ``image`` is the decoded RGBA plane, kept because the only consumers are
    renderers; the PNG bytes ride along so a report can archive the exact
    revision it displayed without re-encoding anything.
    """

    image_id: str
    revision: int
    metadata: dict[str, Any]
    png: bytes
    width: int
    height: int
    affine: tuple
    image: Any                     # PIL RGBA; dcmn already depends on PIL
    sim_time_s: float = 0.0
    status: int = STATUS_VALID
    entry: dict[str, Any] | None = None

    @property
    def source_category(self) -> str: return str(self.metadata.get('source_category', ''))
    @property
    def checksum(self) -> str: return str(self.metadata.get('checksum', ''))

    def matches(self, context: dict[str, Any] | None) -> bool:
        """Whether this revision still describes the frame the vehicle is in.

        The retention rule from DV-MAPPING §7: an image is retained only
        while its reference frame, localization epoch and clock epoch agree
        with the context snapshot; the moment they do not, there is no
        background, because a picture of a *previous* frame behind a plan in
        the current one misregisters silently. An absent or empty context
        matches nothing -- a display that cannot check had no background.
        """
        if not isinstance(context, dict) or not context: return False
        for key in ('frame_id', 'localization_epoch', 'clock_epoch'):
            if key in context and self.metadata.get(key) != context[key]: return False
        if ('clock_domain_id' in context and 'clock_domain_id' in self.metadata and
                self.metadata['clock_domain_id'] != context['clock_domain_id']): return False
        return True

    def footprint_m(self) -> tuple[tuple[float, float], ...]:
        """The image's covered rectangle in map metres, as four corners.

        Pixel centres span ``(0, 0)`` to ``(width-1, height-1)``, so the
        covered *area* is the half-pixel beyond each: ``(-0.5, -0.5)`` to
        ``(width-0.5, height-0.5)``. This is the polygon a renderer clips
        against and the registration mark it draws -- derived, never declared,
        so it cannot disagree with the affine.
        """
        w, h = float(self.width) - .5, float(self.height) - .5
        return tuple(affine_xy(self.affine, col, row)
                     for col, row in ((-.5, -.5), (w, -.5), (w, h), (-.5, h)))


# -- publication -----------------------------------------------------------------

class ImagePublisher:
    """Owns an instance's imagery plane: the registry, the manifest, every ring.

    A producer publishes an image by handing :meth:`publish` the PNG and the
    affine; the metadata -- checksum, dimensions, everything a consumer checks
    -- is derived here, never asserted by the caller. Attachment is late by
    design: the plane may exist with no images, and a consumer that attached
    before the first publish discovers it on its next probe. Withdrawal is
    equally explicit: the image leaves the manifest and the consumer drops it,
    which is how an operator turns a background off at the source.

    Each ring is sized from the first revision of its image and does not
    grow: a bounded plane must state where the bound is, and a producer whose
    image outgrew its own staging withdraws and publishes it again rather
    than the plane inventing a second, quieter bound.
    """

    def __init__(self, instance: str, *, producer: str = 'dsim',
                 session: str | None = None, generation: int = 1,
                 ring_slots: int = DEFAULT_RING_SLOTS, activate: bool = True) -> None:
        self.pm = load_pymembus()
        self.instance = validate_id(instance)
        self.session = session or uuid.uuid4().hex
        self.generation = int(generation)
        self.producer = producer
        self.ring_slots = max(1, int(ring_slots))
        self.rings: dict[str, RecordRing] = {}
        self.entries: dict[str, dict[str, Any]] = {}
        self.revisions: dict[str, int] = {}
        self._change: dict[str, bytes] = {}
        self.closed = False
        self.active = False
        self.manifest: dict[str, Any] = {}
        if activate: self.activate()

    # -- the manifest ---------------------------------------------------------

    def _manifest(self) -> dict[str, Any]:
        return dict(
            schema=SCHEMA, provider_session_id=self.session,
            vehicle_id=self.instance, producer=self.producer,
            generation=self.generation,
            limits=dict(max_encoded_bytes=MAX_ENCODED_BYTES,
                        max_decoded_bytes=MAX_DECODED_BYTES,
                        max_images=MAX_IMAGES),
            images=dict(self.entries))

    def activate(self) -> None:
        if self.active: return
        self.registry = self.pm.memkv()
        self.pm.memkv.remove(registry_name(self.instance))
        if not self.registry.create(registry_name(self.instance), len(REGISTRY_KEYS),
                memkv_aligned_name_len(32, MAX_MANIFEST), MAX_MANIFEST, True):
            raise RuntimeError(self.pm.last_error_message())
        for index, key in enumerate(REGISTRY_KEYS): self.registry.setName(index, key)
        self.manifest = self._manifest()
        if not self.registry.setAll(self._values()): raise RuntimeError(self.pm.last_error_message())
        self.active = True

    def _values(self) -> dict[str, str]:
        return dict(zip(REGISTRY_KEYS, (
            SCHEMA, json_bytes(self.manifest).decode(), str(self.generation),
            self.producer, self.instance, self.session)))

    def _commit(self) -> None:
        """Publish a new manifest atomically: readers see all of it or none."""
        self.manifest = self._manifest()
        encoded = json_bytes(self.manifest).decode()
        if len(encoded.encode()) > MAX_MANIFEST: raise ValueError('imagery manifest exceeds 64 KiB')
        if not self.registry.setAll(self._values()): raise RuntimeError(self.pm.last_error_message())

    # -- publication and withdrawal ------------------------------------------

    def publish(self, image_id: str, png: bytes, affine, *, source_category: str,
                frame_id: str = 'local', localization_epoch: int = 0,
                clock_domain_id: str = '', clock_epoch: int = 0,
                capture_s: float | None = None, registration: dict | None = None,
                altitude_m: float | None = None, floor: str | None = None,
                sim_time_s: float = 0.0, status: int = STATUS_VALID) -> int:
        """Publish one immutable revision of an image and return its revision.

        The revision advances only when the image or its placement changes,
        because a reference image is not a belief that improves; the record is
        still written every call, so republishing is a heartbeat a consumer
        that overran its ring can catch up on.
        """
        if self.closed: raise RuntimeError('publish: the imagery plane is closed')
        image_id = validate_id(image_id)
        if image_id not in self.entries and len(self.entries) >= MAX_IMAGES:
            raise ValueError(f'the imagery plane holds at most {MAX_IMAGES} images')
        candidate = build_metadata(
            image_id, png, affine, revision=0, provider_session_id=self.session,
            source_category=source_category, frame_id=frame_id,
            localization_epoch=localization_epoch, clock_domain_id=clock_domain_id,
            clock_epoch=clock_epoch, capture_s=capture_s, registration=registration,
            altitude_m=altitude_m, floor=floor)
        change = json_bytes({key: value for key, value in candidate.items()
                             if key != 'revision'})
        if change != self._change.get(image_id):
            self.revisions[image_id] = self.revisions.get(image_id, 0) + 1
            self._change[image_id] = change
        revision = self.revisions[image_id]
        metadata = dict(candidate, revision=revision)
        payload = encode_revision(metadata, bytes(png))
        ring = self.rings.get(image_id)
        if ring is None:
            capacity = max(65536, (len(payload) + 4096) * self.ring_slots)
            name = channel_name(self.instance, self.session, self.generation, image_id)
            ring = RecordRing(name, capacity, create=True)
            self.rings[image_id] = ring
            self.entries[image_id] = dict(id=image_id, channel=name,
                                          capacity=capacity, encoding='png')
            self._commit()
        elif len(payload) + 64 >= ring.size:
            raise ValueError(f'reference image for {image_id} is {len(payload)} bytes, over the '
                             f'{ring.size - 64} staged for it; withdraw the image and publish it again')
        ring.write(encode_record(
            self.session, self.generation, image_id, revision, revision,
            int(round(float(sim_time_s) * 1e6)), payload,
            payload_type=REFERENCE_IMAGE, status=status,
            reset_epoch=localization_epoch, clock_epoch=clock_epoch))
        return revision

    def withdraw(self, image_id: str) -> bool:
        """Remove an image from the plane. Explicit, and the opposite of ambient.

        Returns whether there was anything to withdraw. The manifest commits
        before the ring is unlinked, so a consumer that reads the manifest
        between the two never opens a ring the producer is about to remove.
        """
        if self.closed: raise RuntimeError('withdraw: the imagery plane is closed')
        image_id = validate_id(image_id)
        if image_id not in self.entries: return False
        del self.entries[image_id]
        self.revisions.pop(image_id, None)
        self._change.pop(image_id, None)
        self._commit()
        ring = self.rings.pop(image_id)
        ring.close(); ring.unlink()
        return True

    @property
    def images(self) -> dict[str, dict[str, Any]]: return self.entries

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
            self.active = False


# -- discovery and intake --------------------------------------------------------

@dataclass
class ImageState:
    """What a consumer knows about one image it is following.

    ``staged`` is the predecessor of ``current``: the one revision of slack
    §9 allows a display cache, so a renderer comparing a replacement has the
    old pixels to fall back on, and nothing older is kept.
    """
    entry: dict[str, Any]
    current: ReferenceImage | None = None
    staged: ReferenceImage | None = None
    revision: int = 0
    sim_time_s: float = 0.0
    first_sim_time_s: float | None = None
    records: int = 0
    gaps: int = 0
    rejected: int = 0
    last_reason: str = ''


class ImageSession:
    """Read-only intake for an instance's imagery plane.

    Creates nothing, so start order is irrelevant, and it is the only cache:
    the newest validated revision of each image plus one predecessor, which
    is the whole of §9's display-cache allowance. A record that fails any
    check is rejected *with a message* and changes nothing -- an invalid
    replacement never displaces the revision it was trying to replace, and
    the only things that ever remove a good image are withdrawal, a producer
    restart, or the frame no longer matching.
    """

    def __init__(self, instance: str, *, probe_interval_s: float = .25,
                 max_probe_interval_s: float = 2.0) -> None:
        self.instance = validate_id(instance)
        self.pm = load_pymembus()
        self.manifest: dict[str, Any] = {}
        self.identity: tuple[str, int] | None = None
        self.states: dict[str, ImageState] = {}
        self.rings: dict[str, RecordRing] = {}
        self.probe_interval_s = float(probe_interval_s)
        self.max_probe_interval_s = float(max_probe_interval_s)
        self._backoff_s = float(probe_interval_s)
        self._last_probe = -1e9
        self.rejections: list[tuple[str, str]] = []
        self.discoveries = 0
        self.last_seen_identity: tuple[str, int] | None = None
        self.closed = False

    # -- discovery -----------------------------------------------------------

    @property
    def limits(self) -> dict[str, Any]:
        return dict(self.manifest.get('limits', {}))

    def _read_manifest(self) -> dict[str, Any]:
        kv = self.pm.memkv()
        if not kv.open(registry_name(self.instance)): return {}
        try:
            values = kv.getAll()
        finally:
            kv.close()
        if values.get('imagery.schema') != SCHEMA:
            self._reject('', 'unsupported imagery manifest schema'); return {}
        try:
            manifest = json.loads(values['imagery.manifest'])
        except (KeyError, TypeError, ValueError) as exc:
            self._reject('', f'unreadable imagery manifest: {exc}'); return {}
        if not isinstance(manifest, dict): return {}
        images = manifest.get('images')
        if not isinstance(images, dict):
            self._reject('', 'imagery manifest declares no images table')
            return {}
        if len(images) > MAX_IMAGES:
            self._reject('', f'imagery manifest declares {len(images)} images, over the {MAX_IMAGES} limit')
            return {}
        return manifest

    def connect(self) -> bool:
        """Attach to the current session and generation, retrying on backoff.

        A *new* session or generation is a wholesale re-discovery: the old
        images belong to a producer that is gone. The *same* session is the
        interesting case this plane adds -- images attach and withdraw within
        one producer's lifetime, so the manifest is diffed: entries that
        appeared get rings and entries that vanished lose them, and an image
        that survived keeps its state.
        """
        if self.closed: return False
        now = time.monotonic()
        if self.identity is not None:
            if now - self._last_probe < self.probe_interval_s: return True
        elif now - self._last_probe < self._backoff_s:
            return False
        self._last_probe = now
        manifest = self._read_manifest()
        identity = ((str(manifest.get('provider_session_id', '')),
                     int(manifest.get('generation', 0))) if manifest else None)
        if not manifest:
            # No manifest is the only evidence of absence there is, and it is
            # a reason to keep retrying, not to give up.
            self._backoff_s = min(self.max_probe_interval_s, self._backoff_s * 2)
            return False
        if identity == self.identity:
            self._backoff_s = self.probe_interval_s
            self._sync_images(manifest)
            return True
        try:
            self._adopt(manifest, identity)
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            self._reject('', f'imagery channel unavailable: {exc}')
            self._backoff_s = min(self.max_probe_interval_s, self._backoff_s * 2)
            return False
        return True

    def _adopt(self, manifest: dict[str, Any], identity: tuple[str, int]) -> None:
        """Open every channel a manifest declares, abandoning the old session."""
        for ring in self.rings.values(): ring.close()
        self.rings.clear()
        self.states.clear()
        self._open_new(manifest)
        self.manifest, self.identity = manifest, identity
        self.last_seen_identity = identity
        self._backoff_s = self.probe_interval_s
        self.discoveries += 1

    def _sync_images(self, manifest: dict[str, Any]) -> None:
        """Apply one session's attach/withdraw diff to the rings this consumer holds."""
        images = manifest.get('images', {})
        for image_id in list(self.rings):
            if image_id not in images:
                # A withdrawn image is gone: its state goes with it, so the
                # display shows no background rather than the last thing the
                # producer meant to take back.
                self.rings.pop(image_id).close()
                self.states.pop(image_id, None)
        self._open_new(manifest)
        self.manifest = manifest

    def _open_new(self, manifest: dict[str, Any]) -> None:
        for image_id, entry in manifest.get('images', {}).items():
            if image_id in self.rings: continue
            self.rings[image_id] = RecordRing(entry['channel'], int(entry['capacity']))
            self.states[image_id] = ImageState(entry)

    def _reject(self, image_id: str, reason: str) -> None:
        """Record a rejection. A failure is a message, never silence."""
        self.rejections.append((image_id, reason))
        del self.rejections[:-64]
        state = self.states.get(image_id)
        if state is not None:
            state.rejected += 1
            state.last_reason = reason

    # -- intake ----------------------------------------------------------------

    def poll(self) -> None:
        """Drain every image ring, keeping the newest validated revision of each."""
        if not self.connect(): return
        for image_id, ring in self.rings.items():
            state = self.states[image_id]
            for raw in self._drain(ring, image_id):
                self._admit(image_id, state, raw)

    def _drain(self, ring: RecordRing, image_id: str, limit: int = 16):
        """Raw records, with decode failures reported rather than swallowed.

        An image record is up to 16 MiB, so the drain bound is tight: a
        producer that floods its own ring costs a consumer one bounded pass,
        not an unbounded one.
        """
        for _ in range(limit):
            if not ring.handle.poll(): break
            raw, overrun = ring.handle.read_bytes_with_overrun(0)
            ring.overruns += int(overrun)
            if not raw: continue
            try:
                yield decode_record(raw)
            except ValueError as exc:
                self._reject(image_id, f'{image_id}: {exc}')

    def _admit(self, image_id: str, state: ImageState, record: dict[str, Any]) -> None:
        if (record['provider_session_id'], record['generation']) != self.identity: return
        if record['sensor_id'] != image_id: return
        if record['payload_type'] != REFERENCE_IMAGE:
            self._reject(image_id, f'{image_id}: payload type {record["payload_type"]} is not '
                                   f'{REFERENCE_IMAGE} ({PAYLOAD_TYPES[REFERENCE_IMAGE]})')
            return
        try:
            metadata, png = decode_revision(record['payload'])
            image = decode_png(png, metadata['width'], metadata['height'])
        except ValueError as exc:
            self._reject(image_id, f'{image_id}: {exc}')
            return
        if metadata.get('image_id') != image_id:
            self._reject(image_id, f'{image_id}: record names image {metadata.get("image_id")!r}')
            return
        if int(metadata.get('revision', 0)) != int(record['sequence']):
            self._reject(image_id, f'{image_id}: metadata revision {metadata.get("revision")} '
                                   f'disagrees with the record sequence {record["sequence"]}')
            return
        # The record envelope and the metadata must tell the same story about
        # which frame this image describes, or neither is trustworthy.
        if (record['reset_epoch'] != metadata.get('localization_epoch') or
                record['clock_epoch'] != metadata.get('clock_epoch')):
            self._reject(image_id, f'{image_id}: reference image epoch mismatch (record '
                                   f'{record["reset_epoch"]}/{record["clock_epoch"]}, metadata '
                                   f'{metadata.get("localization_epoch")}/{metadata.get("clock_epoch")})')
            return
        sim_time_s = record['sim_time_us'] / 1e6
        revision = int(record['sequence'])
        state.records += 1
        if state.first_sim_time_s is None: state.first_sim_time_s = sim_time_s
        if state.revision and revision > state.revision + 1:
            state.gaps += revision - state.revision - 1
        state.sim_time_s = max(state.sim_time_s, sim_time_s)
        if revision < state.revision: return
        state.revision = revision
        reference = ReferenceImage(
            image_id=image_id, revision=revision, metadata=metadata, png=png,
            width=int(metadata['width']), height=int(metadata['height']),
            affine=tuple(float(v) for v in metadata['affine']), image=image,
            sim_time_s=sim_time_s, status=record['status'], entry=state.entry)
        # The whole display cache: the newest validated revision and one
        # predecessor, per §9. A replacement of the same revision refreshes
        # in place; a new revision demotes the old current to staged.
        if state.current is not None and state.current.revision != revision:
            state.staged = state.current
        state.current = reference

    # -- queries ---------------------------------------------------------------

    def latest(self, image_id: str) -> ReferenceImage | None:
        """The newest validated revision of an image, or None if there is none."""
        state = self.states.get(image_id)
        return state.current if state else None

    def display(self, image_id: str, context: dict[str, Any] | None) -> ReferenceImage | None:
        """The newest revision that still matches the frame the vehicle is in.

        The one query a renderer should use: it applies the retention rule
        (§7) so no caller can forget it -- a background whose frame or epochs
        disagree with the context snapshot is no background, and an image
        that was never checked is never shown.
        """
        reference = self.latest(image_id)
        return reference if reference is not None and reference.matches(context) else None

    def report(self) -> dict[str, Any]:
        images = {}
        for image_id, state in self.states.items():
            images[image_id] = dict(
                revision=state.revision, sim_time_s=state.sim_time_s,
                records=state.records, gaps=state.gaps, rejected=state.rejected,
                reason=state.last_reason,
                cached=(state.current is not None) + (state.staged is not None),
                width=state.current.width if state.current else 0,
                height=state.current.height if state.current else 0,
                checksum=state.current.checksum if state.current else '',
                source_category=state.current.source_category if state.current else '')
        seen = self.identity or self.last_seen_identity
        return dict(connected=self.identity is not None,
                    provider_session_id=seen[0] if seen else '',
                    generation=seen[1] if seen else 0,
                    producer=self.manifest.get('producer', ''),
                    limits=self.limits,
                    discoveries=self.discoveries, images=images,
                    overruns=sum(ring.overruns for ring in self.rings.values()),
                    rejections=list(self.rejections))

    def close(self) -> None:
        for ring in self.rings.values(): ring.close()
        self.rings.clear()
        self.states.clear()
        self.manifest = {}
        self.identity = None
        self.closed = True


def open_imagery(instance: str) -> ImageSession | None:
    """A connected session, or None while no producer has committed a manifest."""
    session = ImageSession(instance)
    if session.connect(): return session
    session.close()
    return None


# -- the dump tool -----------------------------------------------------------

def _dump(instance: str, seconds: float, only: str | None) -> int:
    """Print the manifest, then one line per revision seen, until time runs out."""
    import sys

    session = ImageSession(instance)
    deadline = time.monotonic() + float(seconds)
    printed_manifest = False
    seen: dict[str, tuple[int, str]] = {}
    reported = 0
    connected = False
    try:
        while time.monotonic() < deadline:
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
                printed_manifest = True
            session.poll()
            for image_id, state in session.states.items():
                if only and image_id != only: continue
                if state.current is None: continue
                mark = (state.current.revision, state.current.checksum)
                if seen.get(image_id) != mark:
                    seen[image_id] = mark
                    reference = state.current
                    a = reference.affine
                    print(f'{image_id}: rev={reference.revision} '
                          f'sim={state.sim_time_s:.3f}s '
                          f'{reference.width}x{reference.height} '
                          f'category={reference.source_category} '
                          f'checksum={reference.checksum[:16]}... '
                          f'affine=[{a[0]:.4f},{a[1]:.4f},{a[2]:.4f},'
                          f'{a[3]:.4f},{a[4]:.3f},{a[5]:.3f}] '
                          f'gaps={state.gaps}')
            while reported < len(session.rejections):
                image_id, reason = session.rejections[reported]
                reported += 1
                print(f'rejected: {reason}', file=sys.stderr)
            time.sleep(.05)
        if not printed_manifest:
            print(f'dcmn.imagery: no imagery manifest on {registry_name(instance)} '
                  f'after {seconds:g}s', file=sys.stderr)
            return 1
    finally:
        session.close()
    return 0


def main(argv=None) -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description='inspect an instance\'s reference-imagery plane')
    parser.add_argument('--dump', action='store_true',
                        help='print the manifest and every revision seen')
    parser.add_argument('--id', required=True, help='instance id')
    parser.add_argument('--seconds', type=float, default=10.0,
                        help='how long to watch, in wall seconds')
    parser.add_argument('--image', default=None, help='limit output to one image')
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    if not args.dump: parser.error('--dump is the only mode')
    if args.seconds <= 0: parser.error('--seconds must be positive')
    try:
        return _dump(args.id, args.seconds, args.image)
    except (ValueError, RuntimeError) as exc:
        print(f'dcmn.imagery: {exc}', file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == '__main__':
    raise SystemExit(main())