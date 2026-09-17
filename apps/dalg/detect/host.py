"""DetectorSources: run the profile's detectors, project, estimate, publish and record.

The host is to detectors what :class:`dalg.sources.EvidenceSources` is to
evidence. Per detector it keeps its own rate (``hz``) against the data clock;
a detector that falls behind skips frames and counts them, so a slow detector
never holds evidence up. Each processed frame is the newest one its sensor
delivered, checked by the run exactly as evidence frames are (capture pose
present, same frame and epochs, not from the future).

For every processed frame -- **empty ones included**, since a miss rate and
"it stopped seeing the pad at 14.2 s" need them -- the archive gets a
``detector.frame``; each detection a ``detector.detection`` with its pixel
fields, projection inputs and result and the estimate it joined; rejected
candidates a ``detector.candidates``; overruns and exceptions
``detector.skipped`` / ``detector.failed``; every estimate change an
``estimate.*`` event; and every publication a ``detections.published``.
Imagery rides as optional archive payloads: a crop around each detection, the
full frame of each estimate's first observation, and full frames at
``record_frames_hz`` when set.

The detections plane (:mod:`dcmn.detections`) is published whenever a frame
was processed or an estimate changed, and at least once a sim second so a
consumer can judge its freshness by sim time.
"""
from __future__ import annotations

import collections
import hashlib
import io
import time
import traceback
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Callable

import numpy as np

from dalg.detect import DetectorResult, build
from dalg.detect.estimate import Estimates
from dalg.detect.projection import inputs as projection_inputs, project
from dalg.model import Intrinsics
from dcmn import detections as D
from dcmn.pacing import PeriodicDeadline

#: Processed frames kept per detector for the Live overlay and the sim-time strip.
STRIP_S = 120.0
#: Projected detections kept for the Detections tab map, fading with age.
RECENT_DETECTIONS = 200
RECENT_S = 30.0
#: Latencies kept for the p50/p95 readout.
LATENCY_WINDOW = 200
#: The plane is refreshed at least this often in sim seconds, even with nothing new.
PUBLISH_EVERY_S = 1.0


def _percentile(values, q):
    return None if not values else float(np.percentile(np.asarray(values, float), q))


def reason_key(reason: str) -> str:
    """A not-projectable reason without its numbers, for counting by kind."""
    return 'beyond max range' if reason.startswith('range ') else reason


def png_image(image_id: str, rgb, *, revision: int, sim_time_s: float, category: str, extra=None):
    """An archive image payload (the recorder's reference-image shape) from an RGB array."""
    from PIL import Image
    from dcmn.imagery import ReferenceImage
    array = np.ascontiguousarray(np.asarray(rgb, np.uint8))
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, 'PNG')
    png = buffer.getvalue()
    height, width = array.shape[:2]
    metadata = dict(checksum=hashlib.sha256(png).hexdigest(), width=int(width), height=int(height),
                    affine=[1.0, 0.0, 0.0, 1.0, 0.5, 0.5], source_category=category, **(extra or {}))
    return ReferenceImage(image_id=image_id, revision=int(revision), metadata=metadata, png=png, width=int(width),
                          height=int(height), affine=tuple(metadata['affine']), image=None,
                          sim_time_s=float(sim_time_s))


def crop(rgb, cx_px: float, cy_px: float, extent_px: float, size_px: int):
    """A square crop around a detection, scaled to ``size_px``, and the box it came from."""
    from PIL import Image
    array = np.asarray(rgb, np.uint8)
    height, width = array.shape[:2]
    side = int(max(size_px // 2, 4 * max(extent_px, 1.0)))
    x0 = int(max(0, min(width - 1, round(cx_px - side / 2))))
    y0 = int(max(0, min(height - 1, round(cy_px - side / 2))))
    x1, y1 = int(min(width, x0 + side)), int(min(height, y0 + side))
    piece = Image.fromarray(array[y0:y1, x0:x1]).resize((size_px, size_px), Image.Resampling.NEAREST)
    return np.asarray(piece), (x0, y0, x1, y1)


@dataclass
class DetectorState:
    config: Any                       # dalg.profiles.Detector
    detector: Any
    settings: Any                     # dalg.detect.DetectorConfig
    stream: Any
    intrinsics: Intrinsics
    estimates: Estimates
    deadline: PeriodicDeadline
    frames_deadline: PeriodicDeadline | None
    last_sequence: int = 0
    processed: int = 0
    skipped: int = 0
    failed: int = 0
    rejected_samples: int = 0
    frames_with_detections: int = 0
    detections_by_label: dict[str, int] = field(default_factory=dict)
    not_projectable: dict[str, int] = field(default_factory=dict)
    latencies_ms: collections.deque = field(default_factory=lambda: collections.deque(maxlen=LATENCY_WINDOW))
    processed_times: collections.deque = field(default_factory=lambda: collections.deque(maxlen=64))
    #: ``(sim time, 'detect' | 'empty' | 'skip' | 'fail')`` for the strip.
    strip: collections.deque = field(default_factory=lambda: collections.deque(maxlen=4000))
    state: str = 'waiting'
    reason: str = 'no frame processed yet'
    last_error: str = ''
    #: The last processed frame, its result and projections: the Live overlay draws exactly this.
    last: dict[str, Any] | None = None
    #: This detector's latest detections as the plane carries them.
    latest: list[dict[str, Any]] = field(default_factory=list)

    @property
    def effective_hz(self) -> float | None:
        times = self.processed_times
        if len(times) < 2 or times[-1] <= times[0]: return None
        return (len(times) - 1) / (times[-1] - times[0])

    def health(self) -> dict[str, Any]:
        return dict(id=self.config.id, name=self.config.detector, sensor=self.config.sensor,
                    labels=list(self.config.labels), settings_digest=self.config.settings_digest,
                    state=self.state, reason=self.reason[:D.MAX_TEXT], processed=self.processed,
                    skipped=self.skipped, failed=self.failed, rejected_samples=self.rejected_samples,
                    configured_hz=float(self.settings.hz),
                    hz=None if self.effective_hz is None else round(self.effective_hz, 3),
                    latency_ms_p50=_round(_percentile(self.latencies_ms, 50)),
                    latency_ms_p95=_round(_percentile(self.latencies_ms, 95)),
                    last_frame_s=None if self.last is None else round(self.last['capture_time_s'], 4),
                    frames_with_detections=self.frames_with_detections,
                    detections_per_frame=round(sum(self.detections_by_label.values()) / max(self.processed, 1), 4),
                    detections_by_label=dict(self.detections_by_label),
                    not_projectable=dict(self.not_projectable), last_error=self.last_error[:D.MAX_TEXT])


def _round(value, digits=3):
    return None if value is None else round(float(value), digits)


class DetectorSources:
    def __init__(self, instance: str, detectors, session, streams: dict, *, producer: str = D.DEFAULT_PRODUCER,
                 z0_m: float = 0.0, record: Callable[..., Any] | None = None,
                 bus_publish: Callable[..., Any] | None = None, context: dict | None = None,
                 session_id: str = 'dalg', code_version: str = '', clock=time.perf_counter) -> None:
        self.instance, self.producer = instance, producer
        self.z0_m = float(z0_m)
        self._record = record or (lambda *args, **kwargs: None)
        self._bus = bus_publish or (lambda *args, **kwargs: None)
        self.context = {k: v for k, v in (context or {}).items() if k in D.CONTEXT_KEYS}
        self.session_id = str(session_id or 'dalg')[:D.MAX_TEXT]
        self.clock = clock
        self.states: dict[str, DetectorState] = {}
        self.revision = 0
        self.detection_count = 0
        self.last_publish_s: float | None = None
        self.recent: collections.deque = collections.deque(maxlen=RECENT_DETECTIONS)
        #: Per estimate: ``(confidence, crop PNG bytes, detection id)`` of its best detection.
        self.best_crops: dict[str, tuple[float, bytes, str]] = {}
        self.publisher = D.DetectionsPublisher(instance, producer)
        self.plane = self.publisher.name
        single = len(detectors) == 1
        for detector in detectors:
            stream = streams.get(detector.sensor)
            if stream is None:
                stream = streams[detector.sensor] = session.open(detector.sensor, required=False)
            model = stream.model
            intrinsics = Intrinsics(*(model[k] for k in ('width_px', 'height_px', 'fx_px', 'fy_px', 'cx_px', 'cy_px')))
            instance_, settings = build(detector.detector, detector.settings, intrinsics)
            estimates = Estimates(gate_m=settings.gate_m, confirm_n=settings.confirm_n, expire_s=settings.expire_s,
                                  id_prefix='' if single else f'{detector.id}/')
            self.states[detector.id] = DetectorState(
                detector, instance_, settings, stream, intrinsics, estimates, PeriodicDeadline(settings.hz),
                PeriodicDeadline(settings.record_frames_hz) if settings.record_frames_hz > 0 else None)
        try:
            self.publisher.start()
        except Exception:
            self.publisher.close()
            raise
        for state in self.states.values():
            self.record('detector.configured', dict(
                detector=state.config.id, name=state.config.detector, sensor=state.config.sensor,
                labels=list(state.config.labels), native_labels=list(state.config.native_labels),
                label=state.config.label, settings=dict(state.config.settings),
                resolved_settings=asdict(state.settings), settings_digest=state.config.settings_digest,
                intrinsics=asdict(state.intrinsics), z0_m=self.z0_m, plane=self.plane, code_version=code_version),
                vital=True)
        self.publish(0.0, force=True)

    # -- plumbing ---------------------------------------------------------------------

    def record(self, kind, data, images=None, *, vital=False):
        try:
            return self._record(kind, data, images=images, vital=vital)
        except Exception:          # recording never changes what a detector does
            return None

    def bus(self, event_type, payload):
        try: self._bus(event_type, payload)
        except Exception: pass

    @property
    def labels(self) -> list[str]:
        return sorted({label for s in self.states.values() for label in s.config.labels})

    # -- the loop -----------------------------------------------------------------------

    def step(self, now: float, check_sample: Callable[[Any], dict]) -> bool:
        """Process whatever is due at sim time ``now``; returns whether anything changed."""
        from dalg.sources import camera_frame
        changed = False
        for state in self.states.values():
            if not state.deadline.due(now): continue
            state.deadline.advance(now)
            try: state.stream.refresh()
            except Exception as exc:
                self._fail(state, now, None, f'stream refresh failed: {exc}', exc)
                changed = True
                continue
            samples = [sample for sample, _ in state.stream.history.values() if sample.sequence > state.last_sequence]
            if not samples: continue
            overdue = len(samples) - 1
            sample = samples[-1]
            state.last_sequence = sample.sequence
            if overdue:
                state.skipped += overdue
                state.strip.append((float(sample.sim_time_s), 'skip'))
                self.record('detector.skipped', dict(detector=state.config.id, reason='overdue frames',
                                                     count=overdue, skipped=state.skipped,
                                                     through_sequence=int(sample.sequence) - 1, time_s=now))
            try:
                meta = check_sample(sample)
                frame = camera_frame(sample)
            except (ValueError, KeyError, TypeError) as exc:
                state.rejected_samples += 1
                state.skipped += 1
                state.strip.append((float(sample.sim_time_s), 'skip'))
                self.record('detector.skipped', dict(detector=state.config.id, reason=f'frame rejected: {exc}',
                                                     count=1, sequence=int(sample.sequence), time_s=now))
                continue
            started = self.clock()
            try:
                result = state.detector.detect(frame)
            except Exception as exc:  # noqa: BLE001 - one detector failing never stops the others or evidence
                self._fail(state, now, sample, f'{type(exc).__name__}: {exc}', exc)
                changed = True
                continue
            latency_ms = (self.clock() - started) * 1000.0
            self._processed(state, now, sample, meta, frame, result, latency_ms)
            changed = True
        for state in self.states.values():
            for kind, data in state.estimates.expire(now):
                self._estimate_event(kind, data)
                changed = True
        self.publish(now, force=changed)
        return changed

    def _fail(self, state, now, sample, reason, exc):
        state.failed += 1
        state.state, state.reason, state.last_error = 'failed', reason, reason
        state.strip.append((float(now if sample is None else sample.sim_time_s), 'fail'))
        summary = ''.join(traceback.format_exception_only(type(exc), exc)).strip()[-500:]
        frames = traceback.extract_tb(exc.__traceback__)[-3:]
        self.record('detector.failed', dict(detector=state.config.id, reason=reason, count=state.failed,
                                            sequence=None if sample is None else int(sample.sequence),
                                            exception=summary,
                                            traceback=[f'{f.filename}:{f.lineno} {f.name}' for f in frames],
                                            time_s=now), vital=True)
        self.bus('detection.detector_failed', dict(detector=state.config.id, reason=reason, count=state.failed))

    def _processed(self, state, now, sample, meta, frame, result, latency_ms):
        capture = float(sample.sim_time_s)
        state.processed += 1
        state.processed_times.append(capture)
        state.latencies_ms.append(latency_ms)
        state.state, state.reason = 'running', ''
        # The profile row names what is published: a detector's own label, or the row's ``label``.
        result = DetectorResult(tuple(replace(p, label=state.config.published_label(p.label)) for p in result.detections),
                                result.candidates)
        candidates = tuple(result.candidates[:state.settings.candidates])
        camera = frame.camera_pose if frame.camera is not None else None
        frame_ref = f'{state.config.id}:{int(sample.sequence)}'
        self.record('detector.frame', dict(
            detector=state.config.id, frame=frame_ref, sequence=int(sample.sequence), capture_id=sample.capture_id,
            capture_time_s=capture, time_s=now, pose_context=meta, camera_pose=None if camera is None else asdict(camera),
            body_pose=asdict(frame.pose), latency_ms=round(latency_ms, 3), detections=len(result.detections),
            candidates=len(result.candidates)))
        if candidates:
            self.record('detector.candidates', dict(detector=state.config.id, frame=frame_ref,
                                                    sequence=int(sample.sequence), capture_time_s=capture,
                                                    total=len(result.candidates),
                                                    candidates=[c.as_dict() for c in candidates]))
        if state.frames_deadline is not None and state.frames_deadline.due(capture):
            state.frames_deadline.advance(capture)
            self.record('detector.recorded_frame', dict(detector=state.config.id, frame=frame_ref,
                                                        sequence=int(sample.sequence), capture_time_s=capture),
                        images={'frame': png_image(f'{frame_ref}:frame', frame.rgb, revision=sample.sequence,
                                                   sim_time_s=capture, category='detector_frame')})
        latest, projections = [], []
        for pixel in result.detections:
            self.detection_count += 1
            detection_id = f'D{self.detection_count}'
            projection = project(pixel.cx_px, pixel.cy_px, pixel.extent_px, camera, state.intrinsics,
                                 z0_m=self.z0_m, max_range_m=state.settings.max_range_m)
            state.detections_by_label[pixel.label] = state.detections_by_label.get(pixel.label, 0) + 1
            estimate, events = None, []
            if projection.projectable:
                estimate, events = state.estimates.associate(pixel.label, state.config.id, projection.xy_m,
                                                             pixel.confidence, capture, detection_id)
                self.recent.append(dict(xy_m=list(projection.xy_m), time_s=capture, label=pixel.label,
                                        sigma_m=projection.sigma_m, detector=state.config.id))
            else:
                key = reason_key(projection.reason)
                state.not_projectable[key] = state.not_projectable.get(key, 0) + 1
            crop_rgb, box = crop(frame.rgb, pixel.cx_px, pixel.cy_px, pixel.extent_px, state.settings.crop_px)
            images = {'crop': png_image(f'{detection_id}:crop', crop_rgb, revision=sample.sequence, sim_time_s=capture,
                                        category='detection_crop', extra=dict(box_px=list(box)))}
            if any(kind == 'opened' for kind, _ in events):
                images['frame'] = png_image(f'{detection_id}:frame', frame.rgb, revision=sample.sequence,
                                            sim_time_s=capture, category='detection_frame')
            record = dict(id=detection_id, detector=state.config.id, frame=frame_ref, sequence=int(sample.sequence),
                          capture_time_s=capture, time_s=now, **pixel.as_dict(),
                          projection_inputs=projection_inputs(pixel.cx_px, pixel.cy_px, pixel.extent_px, camera,
                                                              state.intrinsics, z0_m=self.z0_m,
                                                              max_range_m=state.settings.max_range_m),
                          projection=projection.as_dict(), estimate=None if estimate is None else estimate.id,
                          crop_box_px=list(box))
            self.record('detector.detection', record, images=images)
            if estimate is not None:
                best = self.best_crops.get(estimate.id)
                if best is None or pixel.confidence > best[0]:
                    self.best_crops[estimate.id] = (float(pixel.confidence), images['crop'].png, detection_id)
            for kind, data in events: self._estimate_event(kind, data)
            row = dict(id=detection_id, detector=state.config.id, label=pixel.label, sequence=int(sample.sequence),
                       capture_time_s=capture, cx_px=float(pixel.cx_px), cy_px=float(pixel.cy_px),
                       extent_px=float(pixel.extent_px), confidence=float(pixel.confidence),
                       estimate=None if estimate is None else estimate.id, **projection.as_dict())
            latest.append(row)
            projections.append(projection)
        state.latest = latest
        if result.detections:
            state.frames_with_detections += 1
        state.strip.append((capture, 'detect' if result.detections else 'empty'))
        while state.strip and capture - state.strip[0][0] > STRIP_S: state.strip.popleft()
        state.last = dict(image=frame.rgb, capture_time_s=capture, sequence=int(sample.sequence), result=result,
                          candidates=candidates, projections=projections, latency_ms=latency_ms, camera=camera,
                          detections=latest)

    def _estimate_event(self, kind, data):
        estimate = data.get('estimate') or {}
        self.record(f'estimate.{kind}', data, vital=kind in ('confirmed', 'cleared'))
        if kind in ('opened', 'confirmed', 'stale'):
            self.bus(f'detection.estimate_{kind}', dict(id=estimate.get('id'), label=estimate.get('label'),
                                                        xy_m=estimate.get('xy_m'), count=estimate.get('count'),
                                                        detector=estimate.get('detector')))

    # -- epochs --------------------------------------------------------------------------

    def clear(self, cause: str, now: float = 0.0, context: dict | None = None) -> None:
        """A mapping reset or epoch change: estimates start over, and say why."""
        for state in self.states.values():
            for kind, data in state.estimates.clear(cause): self._estimate_event(kind, data)
            state.latest = []
        self.recent.clear()
        self.best_crops.clear()
        if context is not None:
            self.context = {k: v for k, v in context.items() if k in D.CONTEXT_KEYS}
        self.publish(now, force=True)

    # -- publication ----------------------------------------------------------------------

    def estimates(self) -> list[dict[str, Any]]:
        rows = [e for s in self.states.values() for e in s.estimates.published(D.MAX_ESTIMATES)]
        rows.sort(key=lambda e: (not e['confirmed'], e['stale'], -e['last_s'], e['id']))
        return rows[:D.MAX_ESTIMATES]

    def value(self, now: float) -> dict[str, Any]:
        detections = [row for s in self.states.values() for row in s.latest]
        detections.sort(key=lambda row: (-row['capture_time_s'], row['id']))
        return dict(schema=D.SCHEMA, vehicle_id=self.instance, producer=self.producer, session=self.session_id,
                    revision=self.revision, time_s=float(now), context=dict(self.context),
                    detectors=[{k: v for k, v in s.health().items()
                                if k in ('id', 'name', 'sensor', 'labels', 'settings_digest', 'state', 'reason',
                                         'processed', 'skipped', 'failed', 'hz', 'configured_hz',
                                         'latency_ms_p50', 'latency_ms_p95', 'last_frame_s')}
                               for s in list(self.states.values())[:D.MAX_DETECTORS]],
                    detections=detections[:D.MAX_DETECTIONS], estimates=self.estimates())

    def publish(self, now: float, *, force: bool = False) -> None:
        if not force and self.last_publish_s is not None and now - self.last_publish_s < PUBLISH_EVERY_S: return
        self.revision += 1
        value = self.value(now)
        try:
            self.publisher.write(value)
        except (ValueError, RuntimeError) as exc:
            self.record('detections.publish_failed', dict(reason=str(exc), revision=self.revision, time_s=now),
                        vital=True)
            return
        self.last_publish_s = now
        self.record('detections.published', dict(revision=self.revision, time_s=float(now),
                                                 detections=[d['id'] for d in value['detections']],
                                                 estimates=[[e['id'], e['revision']] for e in value['estimates']]))

    # -- reports ---------------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        return dict(plane=self.plane, z0_m=self.z0_m,
                    detectors=[s.health() for s in self.states.values()],
                    estimates=[e.as_dict() for s in self.states.values()
                               for e in sorted(s.estimates.items.values(), key=lambda e: e.id)])

    def write_report(self, directory, *, grid=None, background=None, now_s: float = 0.0) -> dict[str, Any]:
        """``detections.png`` and one crop per estimate under ``detections/``; returns the summary section."""
        from pathlib import Path
        directory = Path(directory)
        section = self.summary()
        crops = {}
        if self.best_crops:
            folder = directory / 'detections'
            folder.mkdir(parents=True, exist_ok=True)
            for estimate_id, (confidence, png, detection_id) in sorted(self.best_crops.items()):
                name = estimate_id.replace('/', '_') + '.png'
                (folder / name).write_bytes(png)
                crops[estimate_id] = dict(image=f'detections/{name}', detection=detection_id,
                                          confidence=round(confidence, 4))
        section['crops'] = crops
        try:
            from dalg.detection_view import detections_image
            image = detections_image(self, grid=grid, background=background, now_s=now_s)
            if image is not None:
                image.save(directory / 'detections.png')
                section['image'] = 'detections.png'
        except Exception as exc:     # noqa: BLE001 - a picture never costs the numbers
            section['image_error'] = str(exc)
        return section

    def close(self) -> None:
        self.publisher.close()
