"""Continuous evidence observation from neutral provider interfaces.

dalg is a long-lived sensor-evidence producer. It needs a session context, a
valid pose, resolved geometry and at least one compatible sensor; it does not
need a goal, a tour, a world file or a coordinator, and it survives mission
completion (which it only records). Liveness (the heartbeat), input availability
(per-sensor health) and evidence freshness (publication time and per-cell
observation ages) are reported separately.

States:

``WAITING_PROVIDER``  no session context, or its provider stopped updating it
``WAITING_POSE``      no valid, fresh pose in the current frame (before mapping)
``WAITING_SENSORS``   no configured sensor present yet
``CONFIGURATION_ERROR``  malformed/incompatible source or missing model
``ALLOCATION_FAILED`` requested coverage exceeds the cell cap or budget
``RUNNING``           publishing; ``admission`` says whether samples are being
                      admitted or paused on an invalid/stale pose

Resets (DV-MAPPING §9): an operator mapping reset (context ``reset_revision``)
stages the new generation before retiring the old one and is rejected without
harm if it does not fit; a provider, localization, clock or sensor-transport
epoch change retires the current generation first -- the old frame is not
valid any more -- and rebuilds from new valid samples. Either way cells restart
unknown, temporal algorithm state is discarded, and the reason is recorded.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import time
from pathlib import Path

from dcmn.archive import Recorder, next_archive_dir
from dcmn.context import Context, validate_pose
from dcmn.health import IntakeMeter, SensorIntake
from dcmn.mapping import AllocationError, MappingConfig, contains
from dcmn.module_bus import PymembusModuleBus, SENSOR_HEALTH_EVENT, requests_shutdown
from dcmn.pacing import PeriodicDeadline
from dcmn.sensors import SensorSession
from dalg.profiles import (PRIMARY_CAMERA, camera_evidence_algorithms, resolve_sources,
                           validate_sources)
from dalg.sources import EvidenceSources, camera_frame

#: Bus traffic worth keeping in the archive: lifecycle, not presence chatter.
RECORDED_EVENTS = ('run.', 'mission.', 'system.', 'route.accepted', 'route.rejected')
#: Non-camera samples admitted per source per loop turn.
MAX_SAMPLES_PER_TURN = 128


def matches_prepare(profile, requirements, process_id=''):
    """Whether this instance satisfies an ``algorithm:<selector>`` readiness requirement."""
    selectors = [str(value).partition(':')[2] for value in requirements
                 if str(value).partition(':')[0] == 'algorithm']
    if not selectors: return True
    return any(not selector or selector in (profile.name, process_id,
                                             *(source.algorithm for source in profile.sources))
               for selector in selectors)


class DalgRun:
    """Observer lifecycle shared by the headless and Tk front ends."""

    def __init__(self, instance_id, profile, root, *, mapping=None, camera_hz=5., pose_max_age=.5,
                 recording_queue=64 << 20, recording_disk=4 << 30):
        if camera_hz <= 0 or pose_max_age <= 0: raise ValueError('rates and freshness must be positive')
        self.id, self.profile, self.root = instance_id, profile, Path(root)
        self.mapping = mapping or MappingConfig()
        self.geometry = None; self.geometry_basis = ''
        self.camera_hz, self.pose_max_age = float(camera_hz), float(pose_max_age)
        self.recording_queue, self.recording_disk = recording_queue, recording_disk
        self.context = Context(instance_id); self.snapshot = {}
        self.sensor_session = SensorSession(instance_id); self.source_streams = {}
        self.sources = None
        self.bus = PymembusModuleBus(instance_id, 'algorithm', 'dalg', sim_time=self.sim_time_s)
        self.state = 'WAITING_PROVIDER'; self.reason = 'no session context yet'
        self.admission = 'paused'; self.coverage = ''
        self.active = False; self.done = False; self.shutdown_requested = False
        self.run_id = ''; self.frames = 0
        self.unavailable_sources = {}; self.source_health = {}; self.provenance = {}
        self.last_frame = self.preview_result = None; self.report_dir = None
        self.intake = IntakeMeter(); self.sensor_intake = SensorIntake()
        self.recorder = None; self._session_id = None
        self._time_s = 0.; self._last_presence = -1e9; self._hello_sent = False
        self._epoch = None; self._mapping_epoch = 0; self._reset_revision = 0
        self._membership = None; self._failed_attempt = None; self._allocated = 0
        self._capture = PeriodicDeadline(self.camera_hz); self._last_record = None
        self._coverage_goal = None; self.counters = dict(admitted=0, rejected=0, skipped=0,
                                                        backlog=0, resets=0, publications=0)
        self._closed = False
        self.background_provider = None; self._displayed_background = None

    # -- clocks and connections ---------------------------------------------------

    def sim_time_s(self): return self._time_s
    def poll_delay(self): return .01

    def note(self, kind, **data):
        if self.recorder:
            self.recorder.record(kind, dict(time_s=self.sim_time_s(), **data))

    def note_display(self, mode, **data):
        """Record an operator display decision in this session's provenance.

        A reference background is optional decoration, but *showing* one is a
        fact about how the session was conducted: an operator who planned over
        a truth map was assisted, and a report that could not say so would
        make assisted and unassisted sessions comparable (DV-MAPPING §7).
        Archived as an event and kept in the summary's provenance, where the
        evaluator reads it.
        """
        self.provenance['reference_display'] = dict(mode=mode, **data)
        self.note('display.reference', mode=mode, **data)

    def _archive_displayed_background(self):
        """File the exact reference revision the operator's picture showed (§7).

        Called before the recorder seals, so the archived revision is the one
        the report renders by construction, not by coincidence. Nothing here
        may raise into the shutdown path: a decoration that cannot be filed is
        a warning, never a lost report.
        """
        background = None
        if self.background_provider is not None:
            try: background = self.background_provider()
            except Exception: background = None    # noqa: BLE001 - decoration never kills the report
        self._displayed_background = background
        if background is None or self.recorder is None: return
        image = background.image
        try:
            self.recorder.record('report.background', dict(
                time_s=self.sim_time_s(), image_id=image.image_id, revision=image.revision,
                checksum=image.checksum, opacity=background.opacity,
                label=background.label(), source_category=image.source_category),
                images={'reference': image})
        except ValueError:
            self.note('report.background.unverified',
                      reason='image bytes do not match their declared checksum')

    def _reference_image_record(self):
        background = self._displayed_background
        if background is None: return None
        image = background.image
        return dict(image_id=image.image_id, revision=image.revision, checksum=image.checksum,
                    opacity=background.opacity, label=background.label(),
                    source_category=image.source_category)

    def _recording(self):
        """Open the current session's archive; roll over when the session does."""
        sid = self.snapshot.get('session_id')
        if not sid or sid == self._session_id: return
        if self.recorder:
            self.note('session.rollover', new_session=sid)
            self._archive_displayed_background()
            self.recorder.close(); self._write_report()
        self._session_id = sid; self.run_id = sid
        self.report_dir = Path(self.snapshot['report_root'])/'dalg'
        metadata = dict(module='dalg', session_id=sid, instance=self.id,
            provider_id=self.snapshot.get('provider_id'), provider_kind=self.snapshot.get('provider_kind'),
            frame=self.snapshot.get('frame'), clock_domain_id=self.snapshot.get('clock_domain_id'),
            profiles=list(self.profile.components), profile_digest=self.profile.digest,
            configuration=[s.as_dict() for s in self.profile.sources],
            mapping=self.mapping.as_dict(), camera_hz=self.camera_hz, pose_max_age_s=self.pose_max_age,
            code_version=_code_version(self.root))
        try:
            self.recorder = Recorder(next_archive_dir(self.report_dir), metadata, module='dalg',
                                     queue_bytes=self.recording_queue, disk_bytes=self.recording_disk)
        except (OSError, FileExistsError) as exc:
            self.recorder = None; self.reason = f'recording unavailable: {exc}'
            return
        self._last_record = None
        if self.sources:
            # A new archive owes no reader a trip to the previous one.
            self.recorder.record('retained_state', dict(context=self.sources.context,
                geometry=self.geometry.as_dict(), sources=sorted(self.sources.states)), self.evidence_grids())

    def connect(self):
        try: self.snapshot = self.context.read()
        except ValueError as exc:
            self.snapshot = {}; self.reason = str(exc)
        if self.snapshot: self._time_s = float(self.snapshot.get('time_s', self._time_s))
        self.bus.connect(); self.sensor_session.poll()
        self._recording()
        return bool(self.snapshot) and self.context.alive()

    # -- mapping generations ---------------------------------------------------------

    def _retire(self, reason):
        self.note('mapping.retired', reason=reason, mapping_epoch=self._mapping_epoch)
        if self.sources: self.sources.close()
        self.sources = None; self.active = False; self._allocated = 0
        self._failed_attempt = None; self._last_record = None
        self.counters['resets'] += 1
        self.bus.publish('mapping.invalidated', run_id=self.run_id, payload=dict(
            reason=reason, mapping_epoch=self._mapping_epoch))

    def _available(self, voluntary):
        manifest = self.sensor_session.manifest
        configured = resolve_sources(self.profile, manifest)
        devices = self.sensor_session.devices
        available, omitted = [], {}
        for source in configured:
            if source.sensor == PRIMARY_CAMERA:
                omitted[source.id] = 'the sensor manifest declares no primary camera'
            elif source.sensor not in devices:
                omitted[source.id] = f'sensor {source.sensor!r} absent'
            elif not voluntary and self._membership is not None and source.id not in self._membership:
                omitted[source.id] = 'newly available; joins after an explicit mapping reset or restart'
            else: available.append(source)
        return configured, tuple(available), omitted

    def _initialize(self, pose, *, voluntary=False):
        configured, available, omitted = self._available(voluntary)
        if self._membership is not None and not voluntary:
            removed = sorted(self._membership - {s.id for s in available})
            for sid in removed: omitted.setdefault(sid, 'removed from the sensor manifest')
        self.unavailable_sources = omitted
        if not available:
            self.state = 'WAITING_SENSORS'; self.reason = 'no configured sensor is available'
            return False
        validate_sources(available, self.sensor_session.manifest)
        for source in available:
            if source.sensor not in self.source_streams:
                self.source_streams[source.sensor] = self.sensor_session.open(source.sensor, required=False)
        config = self.mapping
        request = self.snapshot.get('mapping_request')
        # The latest operator request is durable state: a dalg started after it
        # was made honours it as surely as one that heard it happen.
        if (request and request.get('bounds') is not None
                and int(request.get('revision', 0)) > self._reset_revision):
            config = replace(config, bounds=tuple(request['bounds']))
        goal = self.snapshot.get('goal')
        target = (goal['position'][:2] if goal and goal.get('localization_epoch') == pose['localization_epoch']
                  and goal.get('frame_id') == pose['frame_id'] else None)
        if voluntary or self.geometry is None:
            geometry = config.resolve((pose['x_m'], pose['y_m']), target)
            basis = config.requested_bounds((pose['x_m'], pose['y_m']), target)[1]
        else:
            geometry, basis = self.geometry, self.geometry_basis
        # Every configured source counts: membership may be rebuilt later, and
        # a budget that only fit today's sensors is not a budget.
        allocation = config.admit(geometry, len(configured), retained_bytes=self._allocated if voluntary else 0)
        epoch = self._mapping_epoch+1
        context = {k: self.snapshot[k] for k in ('frame_id', 'localization_epoch', 'clock_domain_id', 'clock_epoch')}
        context.update(mapping_epoch=epoch, geometry_revision=epoch,
                       pose_provider_id=self.snapshot['provider_id'], session_id=self.snapshot['session_id'])
        new = EvidenceSources(self.id, replace(self.profile, sources=available), self.sensor_session,
            self.source_streams, geometry, root=self.root, context=context, generation=epoch, activate=False)
        previous = self.sources
        try:
            # The successor is fully built before it replaces anything; the
            # registry swap itself is the atomic commit.
            new.publisher.activate()
        except Exception:
            new.close(); raise
        if previous is not None:
            self.note('mapping.retired', reason='operator mapping reset', mapping_epoch=self._mapping_epoch)
            previous.close()
            self.counters['resets'] += 1
        self.sources = new
        self.mapping = config; self.geometry = geometry; self.geometry_basis = basis
        self._allocated = allocation
        self._mapping_epoch = epoch; self.active = True
        self._capture = PeriodicDeadline(self.camera_hz)
        # Membership is the active set: a later manifest change rebuilds only
        # these, and anything else waits for an explicit reset or a restart.
        self._membership = {s.id for s in available}
        self.state = 'RUNNING'; self.reason = ''; self._last_record = None; self._coverage_goal = None
        models = {}
        for state in new.states.values():
            model = getattr(getattr(state.evidence.algorithm, 'config', None), 'model_path', '')
            if model: models[model] = hashlib.sha256(Path(model).read_bytes()).hexdigest()
        self.provenance.update(geometry=geometry.as_dict(), geometry_basis=basis, mapping_context=context,
                               model_digests=models, allocation_estimate_bytes=allocation)
        self.note('mapping.initialized', geometry=geometry.as_dict(), geometry_basis=basis,
                  context=context, voluntary=voluntary, mapping=config.as_dict(),
                  allocation_estimate_bytes=allocation, omitted=self.unavailable_sources,
                  model_digests=models, sources=[s.as_dict() for s in available],
                  sensor_manifest_identity=list(self.sensor_session.identity or ()),
                  sensor_manifest_digest=self.sensor_session.manifest.get('profile_digest'),
                  start_pose={k: pose[k] for k in ('x_m', 'y_m', 'z_m', 'heading_deg', 'capture_time_s')},
                  goal=goal)
        self.bus.publish('mapping.initialized', run_id=self.run_id, payload=dict(
            geometry=geometry.as_dict(), mapping_epoch=epoch, basis=basis, sources=sorted(new.states)))
        return True

    # -- observation ------------------------------------------------------------------

    def _check_sample(self, state, sample, camera):
        if (sample.provider_session_id, sample.generation) != self.sensor_session.identity:
            raise ValueError('sample from another sensor generation')
        if sample.reset_epoch != self.sensor_session.reset_epoch:
            raise ValueError('sample from another sensor reset epoch')
        if not sample.status: raise ValueError('sensor marked the sample invalid')
        meta = sample.payload.get('pose_context')
        if not meta: raise ValueError('capture-associated pose metadata missing')
        for key in ('frame_id', 'localization_epoch', 'clock_domain_id', 'clock_epoch'):
            if meta.get(key) != self.snapshot.get(key): raise ValueError(f'sample {key} mismatch')
        if meta.get('valid') is not True: raise ValueError('capture pose invalid')
        if abs(float(meta['capture_time_s'])-sample.sim_time_s) > 1e-6: raise ValueError('capture timestamp mismatch')
        if sample.sim_time_s > self.sim_time_s()+1e-6: raise ValueError('sample from the future')
        if camera:
            # The capture pose itself, never the latest telemetry in its place.
            at = dict(self.snapshot, time_s=sample.sim_time_s)
            validate_pose(dict(sample.payload['body'], **meta), at, max_age=0.)
            validate_pose(dict(sample.payload['pose'], **meta), at, max_age=0.)
        return meta

    def _observe(self):
        now = self.sim_time_s()
        due = self._capture.due(now)
        backlog = 0
        for state in self.sources.states.values():
            camera = state.config.algorithm in camera_evidence_algorithms()
            if camera and not due: continue
            state.stream.refresh()
            samples = [sample for sample, _ in state.stream.history.values() if sample.sequence > state.last_sequence]
            if camera:
                # The latest eligible matched frame; the rest are counted, not queued.
                state.skipped += max(0, len(samples)-1); self.counters['skipped'] += max(0, len(samples)-1)
                samples = samples[-1:]
            else:
                backlog += max(0, len(samples)-MAX_SAMPLES_PER_TURN)
                samples = samples[:MAX_SAMPLES_PER_TURN]
            for sample in samples:
                state.last_sequence = sample.sequence
                try:
                    meta = self._check_sample(state, sample, camera)
                    state.evidence.observe(camera_frame(sample) if camera else sample)
                except (ValueError, KeyError, TypeError) as exc:
                    state.error = str(exc); state.rejected += 1; self.counters['rejected'] += 1
                    self.note('sample.rejected', source=state.config.id, sequence=sample.sequence,
                              capture_id=sample.capture_id, reason=str(exc))
                    continue
                state.samples += 1; self.frames += 1; self.intake.record(); self.counters['admitted'] += 1
                state.preview = state.evidence.algorithm.preview()
                state.error = ''; state.last_image = sample.image
                if camera: self.last_frame = sample.image
                observation = state.stream.observation()
                if observation: self.sensor_intake.observe(observation)
                # The capture pose actually delivered with the sample: what a
                # later evaluator aligns against truth, never the truth itself.
                self.note('sample.admitted', source=state.config.id, sequence=sample.sequence,
                          capture_id=sample.capture_id, capture_time_s=sample.sim_time_s,
                          sensor_generation=sample.generation, sensor_reset_epoch=sample.reset_epoch,
                          pose_context=meta, pose=sample.payload.get('pose'),
                          body=sample.payload.get('body'))
        self.counters['backlog'] = backlog
        if due: self._capture.advance(now)

    def _publish(self):
        self.sources.publish(self.sim_time_s())
        grids = self.evidence_grids()
        key = tuple((sid, g.revision, g.sim_time_s) for sid, g in grids.items())
        if key != self._last_record:
            # Every publication, including unchanged content: the archive
            # stores content once and the publication event every time.
            self.counters['publications'] += 1
            if self.recorder:
                self.recorder.record('evidence.published', dict(context=self.sources.context,
                    time_s=self.sim_time_s(), admission=self.admission), grids)
            self._last_record = key

    def _check_coverage(self):
        goal = self.snapshot.get('goal')
        revision = None if goal is None else goal.get('revision')
        if goal is None or self.geometry is None:
            self.coverage = ''; return
        inside = contains(self.geometry, goal['position'])
        self.coverage = '' if inside else 'goal outside coverage'
        if revision != self._coverage_goal:
            self._coverage_goal = revision
            if not inside:
                self.note('coverage.goal_outside', goal=goal, geometry=self.geometry.as_dict())

    # -- the loop -------------------------------------------------------------------------

    def _presence(self):
        now = time.monotonic()
        if now-self._last_presence < 1: return
        self._last_presence = now
        self.sensor_intake.follow(self.sensor_session.subscriptions())
        self.source_health = self.sensor_intake.report(self.sim_time_s())
        payload = dict(state=self.state, ready=self.active, profile=self.profile.name,
            profile_digest=self.profile.digest, reason=self.reason, admission=self.admission,
            coverage=self.coverage, geometry=None if self.geometry is None else self.geometry.as_dict(),
            mapping_epoch=self._mapping_epoch, allocation_estimate_bytes=self._allocated,
            capabilities=dict(maps=self.sources is not None,
                              algorithms=[s.algorithm for s in self.profile.sources],
                              sensors=list(self.profile.sensors)),
            unavailable_sources=self.unavailable_sources, counters=dict(self.counters),
            recording=None if not self.recorder else self.recorder.report())
        if not self._hello_sent:
            self.bus.publish('module.hello', run_id=self.run_id, payload=payload); self._hello_sent = True
        self.bus.publish('module.heartbeat', run_id=self.run_id, payload=payload)
        self.bus.publish(SENSOR_HEALTH_EVENT, run_id=self.run_id, payload=dict(
            state=self.state, sensor_inputs=self.source_health))

    def _events(self):
        for event in self.bus.receive():
            if event.type.startswith(RECORDED_EVENTS):
                self.note('bus.'+event.type, role=event.role, run_id=event.run_id, payload=event.payload)
            if requests_shutdown(event): self.shutdown_requested = True; self.done = True
            if event.type == 'run.prepare' and self.active and matches_prepare(
                    self.profile, event.payload.get('required_roles', ()), self.bus.process_id):
                self.bus.publish('run.ready', run_id=event.run_id, payload=dict(profile=self.profile.name,
                    capabilities={'algorithms': [s.config.algorithm for s in self.sources.states.values()],
                                  'maps': True},
                    configuration_digest=self.profile.digest))
            if event.type == 'run.completed':
                # Mission completion is a recording event, never a perception stop.
                self.provenance['last_mission_outcome'] = event.payload.get('outcome', '')

    def step(self):
        try:
            connected = self.connect()
            self._events()
            if self.done: return self.state
            if not connected:
                self.state = 'WAITING_PROVIDER'; self.reason = 'session provider unavailable'
                self.admission = 'paused'
                return self.state
            identity = self.sensor_session.identity
            provider = (self.snapshot['provider_id'], self.snapshot['localization_epoch'],
                        self.snapshot['clock_epoch'])
            epoch = (provider, identity, self.sensor_session.reset_epoch)
            if self._epoch is not None and self._epoch != epoch:
                old_provider, old_identity, _ = self._epoch
                cause = ('provider changed' if old_provider[0] != provider[0] else
                         'localization epoch changed' if old_provider[1] != provider[1] else
                         'clock epoch changed' if old_provider[2] != provider[2] else
                         'sensor manifest changed' if old_identity != identity else
                         'sensor transport reset')
                if old_provider != provider:
                    # The old frame no longer holds: bounds are resolved again
                    # from the first valid pose of the new one.
                    self.geometry = None
                if self.sources: self._retire(cause)
                self.note('epoch.changed', cause=cause, provider=list(provider),
                          sensor_identity=list(identity or ()), sensor_reset_epoch=self.sensor_session.reset_epoch)
            self._epoch = epoch
            try: pose = validate_pose(self.snapshot.get('pose'), self.snapshot, max_age=self.pose_max_age)
            except (ValueError, KeyError, TypeError) as exc:
                self.admission = 'paused'
                if self.sources is None:
                    self.state = 'WAITING_POSE'; self.reason = str(exc)
                else:
                    # Evidence is kept and republished with its true ages;
                    # nothing new is admitted until a valid same-epoch pose.
                    self.reason = f'admission paused: {exc}'; self._publish()
                return self.state
            revision = self.snapshot['reset_revision']
            voluntary = revision != self._reset_revision and self.sources is not None
            attempt = (epoch, revision)
            if identity is None or self.sensor_session.reset_epoch is None:
                if self.sources is None:
                    self.state = 'WAITING_SENSORS'; self.reason = 'waiting for a sensor manifest and first samples'
                    return self.state
            elif (self.sources is None or voluntary) and attempt != self._failed_attempt:
                try: self._initialize(pose, voluntary=voluntary)
                except AllocationError as exc:
                    self._failed(attempt, 'ALLOCATION_FAILED', exc, voluntary)
                except (ValueError, RuntimeError, TypeError, OSError, KeyError) as exc:
                    self._failed(attempt, 'CONFIGURATION_ERROR', exc, voluntary)
                self._reset_revision = revision
            if self.sources:
                self.admission = 'admitting'
                if self._failed_attempt != attempt: self.state = 'RUNNING'; self.reason = ''
                self._observe(); self._publish(); self._check_coverage()
                if self.coverage and not self.reason: self.reason = self.coverage
        finally: self._presence()
        return self.state

    def _failed(self, attempt, state, exc, voluntary):
        self._failed_attempt = attempt
        if voluntary and self.sources is not None:
            # A voluntary reset that cannot be staged leaves the valid map alone.
            self.state = 'RUNNING'; self.reason = f'mapping reset rejected: {exc}'
        else:
            self.state = state; self.reason = str(exc)
        self.note('mapping.rejected', state=state, reason=str(exc), voluntary=voluntary)

    def evidence_grids(self): return {} if self.sources is None else self.sources.grids()

    # -- reports and shutdown --------------------------------------------------------------

    def _write_report(self):
        if self.report_dir is None: return
        from dalg.report import write_report
        try:
            write_report(self.report_dir, summary=dict(schema_version=3, module='dalg', session_id=self.run_id,
                instance=self.id, profile=self.profile.name, profiles=list(self.profile.components),
                profile_digest=self.profile.digest, sources=[s.as_dict() for s in self.profile.sources],
                frames=self.frames, state=self.state, reason=self.reason, admission=self.admission,
                coverage=self.coverage, unavailable_sources=self.unavailable_sources,
                geometry=None if self.geometry is None else self.geometry.as_dict(),
                geometry_basis=self.geometry_basis, mapping=self.mapping.as_dict(),
                mapping_epoch=self._mapping_epoch, counters=dict(self.counters),
                recording=None if self.recorder is None else self.recorder.report(),
                archive=None if self.recorder is None else self.recorder.directory.name,
                allocation_estimate_bytes=self._allocated, provenance=self.provenance,
                reference_image=self._reference_image_record(),
                pose_provider=self.snapshot.get('provider_kind', '')),
                evidence=self.evidence_grids(), background=self._displayed_background)
        except Exception as exc:                          # noqa: BLE001 - never raise out of reporting
            import sys
            print(f'dalg: report failed: {exc}', file=sys.stderr)

    def finish(self, partial=False):
        """Stop observing; the process decides whether to exit."""
        self.done = True; self.active = False
        self.provenance['stopped'] = 'timeout' if partial else 'requested'

    def close(self):
        if self._closed: return
        self._closed = True
        self.note('module.closed', state=self.state, reason=self.reason)
        # The background is filed before the recorder seals, so the report
        # and its record can never disagree about which reference revision
        # was displayed (DV-MAPPING §7).
        self._archive_displayed_background()
        if self.recorder: self.recorder.close()
        self._write_report()
        if self.sources: self.sources.close()
        self.sensor_session.close(); self.context.close()
        self.bus.publish('module.goodbye', run_id=self.run_id, payload={'state': 'stopped'})
        self.bus.close(); self.done = True; self.active = False


def _code_version(root):
    head = Path(root)/'.git'/'HEAD'
    try:
        ref = head.read_text().strip()
        if ref.startswith('ref: '): return (Path(root)/'.git'/ref[5:]).read_text().strip()
        return ref
    except OSError:
        return ''
