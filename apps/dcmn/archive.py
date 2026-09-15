"""Bounded asynchronous numeric output recording, and its independent reader.

What a module actually published or consumed, kept losslessly so it can be
evaluated later -- not screenshots, not a summary written on close.

Layout of one archive directory (one per module per recording session)::

    archive.json   manifest: schema, metadata, state, counters, completeness
    chunks/        chunk-NNNNNN.npz, numeric arrays only (no pickles)
    chunks.jsonl   one line per committed chunk: file, sha256, bytes, contents
    index.jsonl    one line per event, in recording-sequence order

Commit order is the whole crash story. A chunk's payload file is written to a
temporary name, flushed, fsynced and renamed; only then is its ``chunks.jsonl``
line appended and flushed; only then are the index lines of the events that
reference it appended. An index line therefore never references a payload that
is not on disk, and a crash loses at most the chunk in flight. A reader of an
archive whose manifest still says ``recording`` treats it as unclean and
recovers every complete chunk.

Evidence content is stored once and referenced by content digest (sha256 over
the grid's metadata and exact occupancy/timestamp bytes); every index reference
also names the chunk holding it. Chunks commit every ``commit_s`` wall seconds
or ``chunk_bytes`` of payload, whichever comes first.

A reference image displayed behind a report rides the same
machinery as an *optional* payload: its PNG bytes are stored once under the
content digest of metadata-plus-bytes, and its declared checksum (sha256 of
the PNG alone) is verified before it is admitted. Image references live beside
grid references, not among them -- a reader reconstructs every number without
the image, and a missing or corrupt image costs a background, never a decision.

Bounds: at most ``queue_bytes`` of copied payload waits in memory; beyond that a
record is dropped, never blocking the caller, and an ``archive.gap`` marker says
which sequence numbers are missing. At ``disk_bytes`` payloads stop, a small
reserve keeps room for the final manifest, and the archive is incomplete. An
archive with any gap, error or unclean end is never labelled complete.
"""
from __future__ import annotations

if __name__ == '__main__':
    import sys as _sys
    from pathlib import Path as _Path
    for _path in (str(_Path(__file__).resolve().parents[2]),
                  str(_Path(__file__).resolve().parents[1])):
        if _path not in _sys.path: _sys.path.insert(0, _path)

import hashlib
import io
import json
import os
import queue
import threading
import time
from pathlib import Path

import numpy as np

SCHEMA = 'dvision2.archive.v2'
RESERVE_BYTES = 1 << 20
DEFAULT_QUEUE_BYTES = 64 << 20
DEFAULT_DISK_BYTES = 4 << 30
DEFAULT_CHUNK_BYTES = 32 << 20
DEFAULT_COMMIT_S = 1.
DEFAULT_DRAIN_S = 5.
#: Occupancy is the wire's quantization: uint8, 0..254 probability x 254,
#: 255 never observed. Timestamps are uint32 milliseconds of the data clock,
#: 0 never observed. Both (layers, height, width), C order.
PAYLOAD_SEMANTICS = dict(occupancy='uint8 p*254, 255=never observed',
                         observed_ms='uint32 data-clock ms, 0=never observed',
                         order='layers,height,width C-contiguous',
                         reference_image='uint8 PNG bytes; sha256 of the bytes in metadata.checksum')


def digest_grid(grid):
    """The content identity of one grid: metadata plus its exact bytes."""
    metadata = dict(source=grid.source, geometry=grid.geometry.as_dict(), entry=grid.entry,
                    revision=int(grid.revision), status=int(grid.status))
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True, allow_nan=False).encode())
    digest.update(np.ascontiguousarray(grid.occupancy, np.uint8).tobytes())
    digest.update(np.ascontiguousarray(grid.observed_ms, np.uint32).tobytes())
    return digest.hexdigest(), metadata


def digest_image(image):
    """The content identity of one reference-image revision: metadata plus bytes.

    The plane's own checksum -- sha256 of the PNG alone -- is checked first:
    bytes that do not match the checksum they were published under never
    enter the archive under an identity that says they do.
    """
    png = bytes(image.png)
    metadata = _jsonable(dict(image.metadata))
    if metadata.get('checksum') != hashlib.sha256(png).hexdigest():
        raise ValueError(f'{image.image_id}: declared checksum does not match the image bytes')
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True, allow_nan=False).encode())
    digest.update(png)
    return digest.hexdigest(), metadata


def _copy_parts(entry):
    """The chunk arrays one recorded payload contributes, copied synchronously.

    The copy happens under the caller's lock so a caller may mutate its arrays
    the moment ``record`` returns -- grids as their exact occupancy/timestamp
    bytes, images as their exact PNG bytes.
    """
    kind, payload = entry
    if kind == 'grid':
        return dict(occupancy=np.array(payload.occupancy, np.uint8, copy=True),
                    observed_ms=np.array(payload.observed_ms, np.uint32, copy=True))
    return dict(blob=np.frombuffer(bytes(payload.png), np.uint8).copy())


def next_archive_dir(module_dir):
    """``archive``, or ``archive-2``... when a restarted module finds one already there.

    Archives are append-once: a second process in the same session never
    reopens, and never overwrites, the first one's records.
    """
    module_dir = Path(module_dir)
    for index in range(1, 10000):
        candidate = module_dir/('archive' if index == 1 else f'archive-{index}')
        if not candidate.exists(): return candidate
    raise FileExistsError(f'{module_dir}: too many archives')


def _jsonable(value):
    return json.loads(json.dumps(value, allow_nan=False, default=str))


class Recorder:
    """One module's archive writer. ``record`` never blocks on disk."""

    def __init__(self, directory, metadata=None, *, module='', queue_bytes=DEFAULT_QUEUE_BYTES,
                 disk_bytes=DEFAULT_DISK_BYTES, chunk_bytes=DEFAULT_CHUNK_BYTES,
                 commit_s=DEFAULT_COMMIT_S, drain_s=DEFAULT_DRAIN_S):
        self.directory = Path(directory)
        (self.directory/'chunks').mkdir(parents=True, exist_ok=True)
        if (self.directory/'archive.json').exists():
            raise FileExistsError(f'{self.directory}: an archive already exists; archives are append-once')
        self.queue_limit, self.disk_limit = int(queue_bytes), int(disk_bytes)
        self.chunk_limit, self.commit_s, self.drain_s = int(chunk_bytes), float(commit_s), float(drain_s)
        self.module = module
        self.metadata = _jsonable(metadata or {})
        self.lock = threading.Lock()
        self.queue = queue.Queue()
        self.stop = threading.Event()
        self.pending_bytes = self.bytes_written = self.sequence = 0
        self.dropped = 0; self.dropped_by = {}
        self.committed_events = self.committed_chunks = 0
        self.errors = []; self.state = 'recording'; self.quota_exhausted = False
        self.closed = False; self._abandoned = False
        self._queued = set()     # contents copied into the queue, not yet committed
        self._known = {}         # content digest -> chunk file, once committed
        self._gap = None         # [first, last] sequences dropped since the last accepted record
        self.created_wall = time.time()
        self._manifest()
        self.thread = threading.Thread(target=self._worker, name=f'{module or "output"}-recorder',
                                       daemon=True)
        self.thread.start()

    # -- producer side -----------------------------------------------------------

    def report(self):
        return dict(state=self.state, queued_bytes=self.pending_bytes,
                    queue_limit_bytes=self.queue_limit, bytes_written=self.bytes_written,
                    disk_limit_bytes=self.disk_limit, events_recorded=self.sequence,
                    events_committed=self.committed_events, chunks=self.committed_chunks,
                    dropped=self.dropped, dropped_by=dict(self.dropped_by),
                    quota_exhausted=self.quota_exhausted, errors=list(self.errors),
                    complete=self.state == 'finalized' and not self.dropped and not self.errors)

    def _drop(self, sequence, why):
        self.dropped += 1
        self.dropped_by[why] = self.dropped_by.get(why, 0) + 1
        self._gap = [sequence, sequence] if self._gap is None else [self._gap[0], sequence]

    def record(self, kind, data=None, grids=None, images=None, *, vital=False):
        """Queue one event; returns its sequence, or None if it had to be dropped.

        Grids and reference images are digested and copied synchronously, so a
        caller may mutate its arrays the moment this returns. Vital events
        (what the module decided, not what it observed) are admitted even with
        the queue full, like the gap markers: the reader must be told what
        happened, not left to infer it from a count of the missing.
        """
        if self.closed: return None
        grids = grids or {}
        images = images or {}
        refs, image_refs, payloads = {}, {}, {}
        for sid, grid in grids.items():
            key, meta = digest_grid(grid)
            refs[sid] = dict(content=key, sim_time_s=float(grid.sim_time_s), **meta)
            payloads[key] = ('grid', grid)
        for sid, image in images.items():
            key, meta = digest_image(image)
            image_refs[sid] = dict(content=key, image_id=str(image.image_id),
                                   revision=int(image.revision), bytes=len(image.png),
                                   sim_time_s=float(image.sim_time_s),
                                   status=int(image.status), metadata=meta)
            payloads[key] = ('image', image)
        try: data = _jsonable(data or {})
        except (TypeError, ValueError) as exc:
            data = dict(unserializable=str(exc))
        with self.lock:
            self.sequence += 1
            sequence = self.sequence
            if self.quota_exhausted:
                self._drop(sequence, 'disk quota'); return None
            fresh = {k: v for k, v in payloads.items() if k not in self._known and k not in self._queued}
            copies = {k: _copy_parts(v) for k, v in fresh.items()}
            size = (512 + len(json.dumps(data)) +
                    sum(part.nbytes for parts in copies.values() for part in parts.values()))
            if not vital and self.pending_bytes + size > self.queue_limit:
                self._drop(sequence, 'queue full'); return None
            if self._gap is not None:
                # A tiny marker, admitted even at the limit: the reader must be
                # told what is missing, not left to infer it.
                self.queue.put((dict(sequence=None, type='archive.gap', grids={}, data=dict(
                    missing=list(self._gap), reasons=dict(self.dropped_by))), {}, 0))
                self._gap = None
            self._queued.update(copies)
            self.pending_bytes += size
            event = dict(sequence=sequence, type=str(kind), recorded_wall=time.time(),
                         data=data, grids=refs)
            if image_refs: event['images'] = image_refs
            self.queue.put((event, copies, size))
        return sequence

    # -- worker side --------------------------------------------------------------

    def _manifest(self):
        if self._abandoned: return
        value = dict(schema=SCHEMA, module=self.module, metadata=self.metadata,
                     created_wall=self.created_wall, payload_semantics=PAYLOAD_SEMANTICS,
                     chunk_policy=dict(commit_s=self.commit_s, chunk_bytes=self.chunk_limit),
                     **self.report())
        temp = self.directory/'archive.json.tmp'
        temp.write_text(json.dumps(value, allow_nan=False, indent=2)+'\n')
        temp.replace(self.directory/'archive.json')

    def _commit(self, chunk, index, chunks_log):
        """Write one chunk's payloads, then its log line, then its index lines."""
        events, payloads = chunk['events'], chunk['payloads']
        if not events: return
        name = None
        if payloads:
            name = f'chunk-{self.committed_chunks+1:06d}.npz'
            buffer = io.BytesIO()
            np.savez(buffer, **{f'{key}_{part}': array for key, parts in payloads.items()
                                for part, array in parts.items()})
            raw = buffer.getvalue()
            if self.bytes_written + len(raw) + RESERVE_BYTES > self.disk_limit:
                self.quota_exhausted = True
                with self.lock:
                    for event in events:
                        if event['sequence'] is not None: self._drop(event['sequence'], 'disk quota')
                    self._queued.difference_update(payloads)
                if 'recording disk quota exceeded' not in self.errors:
                    self.errors.append('recording disk quota exceeded')
                return
            temp = self.directory/'chunks'/(name+'.tmp')
            with temp.open('wb') as handle:
                handle.write(raw); handle.flush(); os.fsync(handle.fileno())
            temp.replace(self.directory/'chunks'/name)
            line = json.dumps(dict(chunk=self.committed_chunks+1, file=f'chunks/{name}',
                sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw), contents=sorted(payloads),
                first_sequence=min(e['sequence'] or 0 for e in events),
                last_sequence=max(e['sequence'] or 0 for e in events)))+'\n'
            chunks_log.write(line); chunks_log.flush(); os.fsync(chunks_log.fileno())
            self.committed_chunks += 1
            self.bytes_written += len(raw) + len(line)
            with self.lock:
                for key in payloads: self._known[key] = f'chunks/{name}'
                self._queued.difference_update(payloads)
        lines = []
        for event in events:
            for refs in (event['grids'], event.get('images', {})):
                for ref in refs.values(): ref['chunk'] = self._known.get(ref['content'])
            lines.append(json.dumps(event, allow_nan=False)+'\n')
        text = ''.join(lines)
        # Gap markers may use the reserve: they are the diagnostics it is for.
        reserve = 0 if all(e['sequence'] is None for e in events) else RESERVE_BYTES
        if self.bytes_written + len(text) + reserve > self.disk_limit:
            self.quota_exhausted = True
            with self.lock:
                for event in events:
                    if event['sequence'] is not None: self._drop(event['sequence'], 'disk quota')
            return
        index.write(text); index.flush(); os.fsync(index.fileno())
        self.bytes_written += len(text)
        self.committed_events += sum(1 for e in events if e['sequence'] is not None)

    def _worker(self):
        chunk = dict(events=[], payloads={}, bytes=0, started=None, size=0)
        try:
            with (self.directory/'index.jsonl').open('a') as index, \
                 (self.directory/'chunks.jsonl').open('a') as chunks_log:
                while True:
                    try:
                        event, payloads, size = self.queue.get(timeout=.05)
                        if chunk['started'] is None: chunk['started'] = time.monotonic()
                        chunk['events'].append(event); chunk['size'] += size
                        for key, parts in payloads.items():
                            if key not in chunk['payloads'] and key not in self._known:
                                chunk['payloads'][key] = parts
                                chunk['bytes'] += sum(part.nbytes for part in parts.values())
                    except queue.Empty:
                        pass
                    stopping = self.stop.is_set() and self.queue.empty()
                    due = chunk['started'] is not None and (
                        stopping or chunk['bytes'] >= self.chunk_limit
                        or time.monotonic()-chunk['started'] >= self.commit_s)
                    if due:
                        try: self._commit(chunk, index, chunks_log)
                        except Exception as exc:
                            with self.lock:
                                for e in chunk['events']:
                                    if e['sequence'] is not None: self._drop(e['sequence'], 'write error')
                                self._queued.difference_update(chunk['payloads'])
                            if len(self.errors) < 16: self.errors.append(f'{type(exc).__name__}: {exc}')
                        with self.lock: self.pending_bytes -= chunk['size']
                        chunk = dict(events=[], payloads={}, bytes=0, started=None, size=0)
                        self._manifest()
                    if stopping and chunk['started'] is None: break
        except Exception as exc:
            self.errors.append(f'{type(exc).__name__}: {exc}')

    def close(self):
        """Drain within the deadline, then finalize. Idempotent."""
        if self.closed: return self.report()
        with self.lock:
            self.closed = True
            if self._gap is not None:
                self.queue.put((dict(sequence=None, type='archive.gap', grids={}, data=dict(
                    missing=list(self._gap), reasons=dict(self.dropped_by))), {}, 0))
                self._gap = None
        self.stop.set()
        self.thread.join(timeout=self.drain_s)
        if self.thread.is_alive():
            with self.lock:
                self.dropped += self.queue.qsize()
                self.dropped_by['shutdown deadline'] = self.queue.qsize()
            self.errors.append(f'recording shutdown deadline of {self.drain_s:g} s exceeded')
            self.state = 'unfinished'
            self._manifest(); self._abandoned = True
            return self.report()
        self.state = 'finalized'
        self._manifest()
        return self.report()


class IncompleteInput(ValueError):
    """A recorded decision whose full input set is not in the archive."""


class ArchiveReader:
    """Validate an archive and reconstruct recorded events from it alone."""

    def __init__(self, directory):
        from dcmn.maps import EvidenceGrid, GridGeometry
        self._grid_types = (EvidenceGrid, GridGeometry)
        self.directory = Path(directory)
        manifest_path = self.directory/'archive.json'
        if not manifest_path.exists(): raise ValueError(f'{self.directory}: no archive.json')
        self.manifest = json.loads(manifest_path.read_text())
        if self.manifest.get('schema') != SCHEMA:
            raise ValueError(f'unsupported archive schema {self.manifest.get("schema")!r}; expected {SCHEMA}')
        self.errors = []
        self.warnings = []
        self.clean = self.manifest.get('state') == 'finalized'
        if not self.clean:
            self.warnings.append(f'archive state is {self.manifest.get("state")!r}: unclean end; '
                                 'recovering committed chunks only')
        self.chunks = self._chunks()
        self._verified = {}
        self._events = None

    def _lines(self, name):
        path = self.directory/name
        if not path.exists(): return []
        lines = path.read_bytes().split(b'\n')
        out = []
        for number, line in enumerate(lines, 1):
            if not line.strip(): continue
            try: out.append(json.loads(line))
            except ValueError:
                last = number == len(lines) or all(not rest.strip() for rest in lines[number:])
                (self.warnings if last and not self.clean else self.errors).append(
                    f'{name}: truncated or corrupt line {number}')
                break
        return out

    def _chunks(self):
        chunks = {entry['file']: entry for entry in self._lines('chunks.jsonl')}
        for path in sorted((self.directory/'chunks').glob('chunk-*.npz')):
            relative = f'chunks/{path.name}'
            if relative not in chunks:
                self.warnings.append(f'{relative}: payload written but never committed; ignored')
        return chunks

    def events(self):
        """Committed events in order, with gaps and markers accounted for."""
        if self._events is not None: return self._events
        lines = self._lines('index.jsonl')
        events, previous, explained = [], 0, set()
        # Markers first: a worker-side drop can be announced after later
        # sequences were already committed.
        for event in lines:
            if event.get('type') == 'archive.gap':
                first, last = event['data']['missing']
                explained.update(range(first, last+1))
                self.errors.append(f'recorder dropped sequences {first}..{last}: {event["data"].get("reasons")}')
        for event in lines:
            if event.get('type') == 'archive.gap': continue
            sequence = event['sequence']
            missing = set(range(previous+1, sequence)) - explained
            if missing:
                self.errors.append(f'unexplained index gap: sequences {min(missing)}..{max(missing)}')
            previous = max(previous, sequence)
            events.append(event)
        self._events = events
        return events

    def event(self, sequence):
        for event in self.events():
            if event['sequence'] == sequence: return event
        raise KeyError(f'no committed event {sequence}')

    def _chunk(self, relative):
        if relative in self._verified:
            cached = self._verified[relative]
            if isinstance(cached, Exception): raise cached
            return cached
        try:
            entry = self.chunks.get(relative)
            if entry is None: raise IncompleteInput(f'{relative}: chunk not committed')
            raw = (self.directory/relative).read_bytes()
            if hashlib.sha256(raw).hexdigest() != entry['sha256']:
                raise ValueError(f'{relative}: payload checksum mismatch (corrupt chunk)')
            with np.load(io.BytesIO(raw), allow_pickle=False) as data:
                arrays = {key: data[key] for key in data.files}
        except FileNotFoundError:
            error = IncompleteInput(f'{relative}: chunk file missing')
            self._verified[relative] = error; raise error from None
        except (IncompleteInput, ValueError, OSError) as exc:
            self._verified[relative] = exc; raise
        self._verified[relative] = arrays
        return arrays

    def grid(self, sid, ref):
        """One referenced grid, verified against its chunk checksum and content digest."""
        EvidenceGrid, GridGeometry = self._grid_types
        if not ref.get('chunk'): raise IncompleteInput(f'{sid}: payload never committed')
        arrays = self._chunk(ref['chunk'])
        key = ref['content']
        try: occupancy, observed = arrays[f'{key}_occupancy'], arrays[f'{key}_observed_ms']
        except KeyError: raise IncompleteInput(f'{sid}: content {key[:12]} absent from {ref["chunk"]}') from None
        grid = EvidenceGrid(GridGeometry.from_dict(ref['geometry']), occupancy, observed, ref['source'],
                            ref['revision'], ref['sim_time_s'], ref['status'], ref['entry'])
        if digest_grid(grid)[0] != key: raise ValueError(f'{sid}: content digest mismatch')
        return grid

    def image(self, sid, ref):
        """One archived reference-image revision, verified, as the plane sees one.

        Both identities are checked: the declared checksum (sha256 of the PNG
        alone, the identity the imagery plane publishes) and the content
        digest under which the recorder stored it. A mismatch is corruption,
        and the caller decides -- for an image, that decision is always "no
        background", never "no report".
        """
        from dcmn.imagery import ReferenceImage, decode_png
        if not ref.get('chunk'): raise IncompleteInput(f'{sid}: payload never committed')
        arrays = self._chunk(ref['chunk'])
        key = ref['content']
        try: blob = arrays[f'{key}_blob']
        except KeyError: raise IncompleteInput(f'{sid}: content {key[:12]} absent from {ref["chunk"]}') from None
        png = np.ascontiguousarray(blob).tobytes()
        metadata = dict(ref['metadata'])
        if metadata.get('checksum') != hashlib.sha256(png).hexdigest():
            raise ValueError(f'{sid}: PNG checksum mismatch (corrupt image)')
        digest = hashlib.sha256(json.dumps(metadata, sort_keys=True, allow_nan=False).encode())
        digest.update(png)
        if digest.hexdigest() != key: raise ValueError(f'{sid}: content digest mismatch')
        width, height = int(metadata['width']), int(metadata['height'])
        return ReferenceImage(image_id=ref['image_id'], revision=int(ref['revision']),
                              metadata=metadata, png=png, width=width, height=height,
                              affine=tuple(float(v) for v in metadata['affine']),
                              image=decode_png(png, width, height),
                              sim_time_s=float(ref.get('sim_time_s', 0.)),
                              status=int(ref.get('status', 0)),
                              entry=None)

    def reconstruct(self, event):
        """The event with every referenced grid loaded; raises if any is unavailable.

        Reference images are loaded alongside but never required: a missing
        or corrupt image is a warning, because it costs a background, not a
        decision. Numeric reconstruction succeeds without it.
        """
        grids = {sid: self.grid(sid, ref) for sid, ref in event.get('grids', {}).items()}
        images = {}
        for sid, ref in event.get('images', {}).items():
            try: images[sid] = self.image(sid, ref)
            except (IncompleteInput, ValueError, OSError) as exc:
                self.warnings.append(f'{sid}: reference image unavailable: {exc}')
        return dict(event, reconstructed_grids=grids, reconstructed_images=images)

    def attempts(self):
        return [e for e in self.events() if e['type'] == 'planning.attempt']

    def reconstruct_attempt(self, sequence):
        """A planning attempt resolved to its exact pose, goal, policy and grids."""
        event = self.event(sequence)
        if event['type'] != 'planning.attempt': raise ValueError(f'event {sequence} is {event["type"]}')
        data = event['data']
        declared = set(data.get('inputs', {}).get('sources', []))
        if declared - set(event['grids']):
            raise IncompleteInput(f'attempt {sequence}: inputs {sorted(declared-set(event["grids"]))} not recorded')
        rebuilt = self.reconstruct(event)
        return dict(sequence=sequence, pose=data.get('pose'), goal=data.get('goal'),
                    policy=data.get('policy'), planner=data.get('planner'),
                    route=data.get('route'), inputs=data.get('inputs'),
                    grids=rebuilt['reconstructed_grids'], images=rebuilt['reconstructed_images'])

    def validate(self):
        """Every committed event checked; which decisions are reconstructable."""
        counted, reconstructable, incomplete = 0, [], {}
        for event in self.events():
            try:
                if event['type'] == 'planning.attempt': self.reconstruct_attempt(event['sequence'])
                else: self.reconstruct(event)
                counted += 1
                if event['type'] == 'planning.attempt': reconstructable.append(event['sequence'])
            except (OSError, KeyError, ValueError) as exc:
                (incomplete.__setitem__(event['sequence'], str(exc)))
                if not isinstance(exc, IncompleteInput): self.errors.append(f'event {event["sequence"]}: {exc}')
        errors = list(dict.fromkeys(self.errors + list(self.manifest.get('errors', []))))
        complete = (self.clean and bool(self.manifest.get('complete')) and not errors
                    and not incomplete and not self.manifest.get('dropped'))
        return dict(events=counted, complete=complete, clean=self.clean, errors=errors,
                    warnings=self.warnings, reconstructable_attempts=reconstructable,
                    incomplete_events=incomplete, recording=self.manifest)


def displayed_background(reader):
    """The reference image an archived report displayed, with its opacity.

    The module's run records a ``report.background`` event when it files a
    picture over a reference image; the newest such event is
    the revision that report used, so it wins even when its bytes did not
    survive -- an older revision's picture would misregister, which is worse
    than no background. Absent, unreadable or corrupt imagery comes back as
    ``(None, None)``: it costs a background, never the numbers around it.
    """
    for event in reversed(reader.events()):
        if event.get('type') != 'report.background': continue
        data = event.get('data', {})
        for sid in sorted(event.get('images', {})):
            try: return reader.image(sid, event['images'][sid]), float(data.get('opacity', 1.))
            except (IncompleteInput, ValueError, OSError) as exc:
                reader.warnings.append(f'{sid}: reference image unavailable: {exc}')
        return None, None
    return None, None


def render_attempt(reader, sequence, out_path):
    """One archived planning attempt as a PNG, from the archive alone.

    No dsim, no world file: the grid, route, pose and goal all come out of the
    numeric record, drawn by the same renderer the live pane used. The
    reference image -- when one was displayed beside the report and its bytes
    committed -- composes under the never-observed cells through the same
    sampler and opacity, so an archived overlay registers exactly as the live
    one did, and a world re-rendered later could never silently replace it.
    """
    from dcmn.map_pane import Background, Overlay, snapshot_image
    attempt = reader.reconstruct_attempt(sequence)
    grids = attempt['grids']
    if not grids: raise IncompleteInput(f'attempt {sequence} recorded no grids to draw')
    source = sorted(grids)[0]
    route, pose, goal, policy = attempt['route'] or {}, attempt['pose'] or {}, \
        attempt['goal'] or {}, attempt['policy'] or {}
    overlay = Overlay(
        route=[(float(w['x']), float(w['y'])) for w in route.get('waypoints', ())],
        vehicle=None if not pose else (pose['x_m'], pose['y_m'], pose['heading_deg']),
        goal=None if not goal.get('position') else (goal['position'][0], goal['position'][1]),
        start=(tuple(route['start'][:2]) if route.get('start') else
               None if not pose else (pose['x_m'], pose['y_m'])),
        inflation_m=float(policy.get('inflation_m') or 0.))
    reference, opacity = displayed_background(reader)
    background = None if reference is None else Background(reference, opacity=opacity)
    image = snapshot_image(grids[source], mode='occupancy', overlay=overlay,
                           sim_now_s=grids[source].sim_time_s, background=background)
    if image is None: raise IncompleteInput(f'attempt {sequence}: grid {source} has no layer to draw')
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    return out_path


def main(argv=None):
    import argparse
    import sys
    parser = argparse.ArgumentParser(description='validate a numeric output archive, '
                                     'reconstruct one planning attempt from it, or render '
                                     'that attempt to a PNG -- all without dsim')
    parser.add_argument('directory')
    parser.add_argument('--attempt', type=int, default=None, help='planning attempt sequence to resolve')
    parser.add_argument('--render', type=int, default=None, metavar='ATTEMPT',
                        help='render one planning attempt to a PNG from the archive alone, '
                             'with the archived reference image behind it when one was '
                             'displayed and its bytes survived')
    parser.add_argument('--out', default=None,
                        help='output path for --render (default: render-ATTEMPT.png '
                             'beside the archive)')
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    try: reader = ArchiveReader(args.directory)
    except (OSError, ValueError) as exc:
        print(f'dcmn.archive: {exc}', file=sys.stderr); return 2
    if args.render is not None:
        try:
            path = render_attempt(reader, args.render,
                                  args.out or reader.directory/f'render-{args.render}.png')
        except (KeyError, ValueError, OSError) as exc:
            print(f'dcmn.archive: {exc}', file=sys.stderr); return 1
        print(f'dcmn.archive: rendered attempt {args.render} -> {path}')
        return 0
    if args.attempt is not None:
        try: attempt = reader.reconstruct_attempt(args.attempt)
        except (KeyError, ValueError) as exc:
            print(f'dcmn.archive: {exc}', file=sys.stderr); return 1
        attempt['grids'] = {sid: dict(revision=g.revision, sim_time_s=g.sim_time_s,
                                      geometry=g.geometry.as_dict(), observed=int(g.observed.sum()))
                            for sid, g in attempt['grids'].items()}
        attempt['images'] = {sid: dict(image_id=i.image_id, revision=i.revision,
                                        checksum=i.checksum, width=i.width, height=i.height)
                             for sid, i in attempt.pop('images', {}).items()}
        print(json.dumps(attempt, indent=2, default=str)); return 0
    report = reader.validate()
    report['recording'] = {k: report['recording'].get(k) for k in
                           ('module', 'state', 'complete', 'dropped', 'events_committed', 'bytes_written')}
    print(json.dumps(report, indent=2))
    return 0 if report['complete'] else 1


if __name__ == '__main__': raise SystemExit(main())
