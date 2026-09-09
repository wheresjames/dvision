"""Simulated-time scheduling and publication for every sensor in a profile.

The manager is the only thing that knows when a sensor is due, what state it
sampled, and which channel its record belongs on. Sensor models supply
measurements, the transport layer supplies channels, and this ties a physics
tick to both: one frozen vehicle snapshot per capture, one capture id shared by
everything sampled at that instant, and one deterministic random stream per
sensor and logical capture.

None of that is about any particular simulator, and this module deliberately
imports none. Everything world-specific -- the geometry a beam meets, the
renderer that fills a camera frame, the vehicle datum a state sensor reads --
arrives through one :class:`dcmn.sensor_backend.SensorBackend`, so a second
provider supplies measurements rather than reimplementing the transport.
"""

from __future__ import annotations

import time
from dataclasses import replace

import numpy as np

from dcmn import health
from dcmn.sensors import (CAMERA_FRAME, LIDAR_FRAME, RANGE_SAMPLE, STATE_PAYLOAD,
                          STATUS_INVALID, STATUS_VALID, SensorPublisher)
from dsim import sensor_models, state_sensors
from dsim.profiles import ARRAY_TYPES, STATE_TYPES
from dsim.transforms import pose_angles, resolve


class SensorManager:
    """Owns the sensor plane of one running simulator."""

    def __init__(self, instance, profile, backend, *, seed=0, session=None):
        #: The one simulator-specific surface: geometry, camera views and the
        #: vehicle datum the state sensors read. See
        #: :class:`dcmn.sensor_backend.SensorBackend`.
        self.backend = backend
        self.seed = int(seed)
        self.publisher = SensorPublisher(instance, profile, session=session)
        self.capture_id = 0
        self._install(profile)

    # -- lifecycle ---------------------------------------------------------

    def _install(self, profile):
        self.profile = profile
        data = profile.data
        self.physics_hz = data['physics_hz']
        # Every rate divides the physics rate exactly, so a capture always
        # lands on a physics snapshot and nothing has to be interpolated.
        self.schedule = [(sensor, round(self.physics_hz / sensor['rate_hz']))
                         for sensor in data['sensors'] if sensor['enabled']]
        self.counts = {sensor['id']: dict(scheduled=0, generated=0, published=0, invalid=0,
                                          drops=0, work_s=0.0,
                                          configured_hz=sensor['rate_hz'],
                                          last_sequence=0, last_sim_time_s=None,
                                          fault='') for sensor, _ in self.schedule}
        self.types = {sensor['id']: sensor['type'] for sensor, _ in self.schedule}
        # An inertial sensor differentiates successive snapshots, so the
        # previous one is kept -- but only when something actually needs it.
        self._needs_history = any(sensor['type'] == 'motion.imu'
                                  for sensor, _ in self.schedule)
        self._previous = None
        self._previous_time = None
        self.tick_index = 0
        self.elapsed_ticks = 0
        # Cumulative work, so a run can say where its time went even though
        # the health sampler only averages the cost of published frames.
        self.render_total_s = 0.0
        self.sample_total_s = 0.0

    def apply(self, profile):
        """Swap in a validated profile as one new generation."""
        previous = {sensor['id'] for sensor, _ in self.schedule
                    if sensor['type'] == 'camera.rgb'}
        prepared = self.backend.prepare_profile(profile)
        try:
            self.publisher.apply(profile)
        except Exception:
            self.backend.finish_profile(prepared, commit=False)
            raise
        self.backend.finish_profile(prepared, commit=True)
        self._install(profile)
        self.capture_id = 0
        self.backend.drop_views(previous - {sensor['id'] for sensor, _ in self.schedule})

    def reset(self):
        """Drone reset: cadence and noise restart, transport identity does not."""
        self.publisher.reset()
        self.tick_index = 0
        self._previous = self._previous_time = None

    def close(self):
        self.publisher.close()

    @property
    def generation(self): return self.publisher.generation
    @property
    def manifest(self): return self.publisher.manifest
    @property
    def reset_epoch(self): return self.publisher.reset_epoch
    @property
    def sequence(self): return sum(self.publisher.sequences.values())

    # -- scheduling --------------------------------------------------------

    def due(self):
        """The sensors sampling on this tick, with each one's logical index."""
        return [(sensor, self.tick_index // divisor)
                for sensor, divisor in self.schedule if self.tick_index % divisor == 0]

    def tick(self, state, sim_time_s):
        """Advance one physics tick and publish everything due at its timestamp.

        Returns the seconds spent rendering when a camera published, and
        ``None`` otherwise: that is the frame cost the health sampler averages,
        and a tick that only sampled a rangefinder did not produce a frame.
        """
        self.tick_index += 1
        self.elapsed_ticks += 1
        due = self.due()
        if not due:
            self._remember(state, sim_time_s)
            return None
        self.capture_id += 1
        # One immutable snapshot for the whole capture: a synchronized pair
        # cannot straddle two vehicle states, and neither can a camera and the
        # range sensor beside it.
        snapshot = replace(state)
        sim_us = round(sim_time_s * 1_000_000)
        cameras = [(s, i) for s, i in due if s['type'] == 'camera.rgb']
        started = time.monotonic()
        rendered = self._publish_cameras(cameras, snapshot, sim_us)
        render_s = time.monotonic() - started
        self.render_total_s += render_s
        # One pass renders every due camera, so its cost is split evenly
        # between the cameras that took part: the sum over sensors is the wall
        # time the sensor plane actually spent, never more, and a lone camera
        # still carries its whole pass.
        if cameras:
            share = render_s / len(cameras)
            for sensor, _index in cameras:
                self.counts[sensor['id']]['work_s'] += share
        for sensor, index in due:
            kind = sensor['type']
            if kind == 'camera.rgb':
                continue  # accounted for with the render pass above
            began = time.monotonic()
            try:
                if kind in STATE_TYPES:
                    self._publish_state(sensor, index, snapshot, sim_time_s, sim_us)
                else:
                    self._publish_rays(sensor, index, snapshot, sim_us)
            except (RuntimeError, ValueError) as exc:
                self._failed_capture(sensor['id'], exc)
            spent = time.monotonic() - began
            self.sample_total_s += spent
            self.counts[sensor['id']]['work_s'] += spent
        self._remember(state, sim_time_s)
        return render_s if rendered else None

    def _remember(self, state, sim_time_s):
        if self._needs_history:
            self._previous, self._previous_time = replace(state), sim_time_s

    # -- publication -------------------------------------------------------

    def _body(self, state):
        return dict(x_m=state.x, y_m=state.y, z_m=state.z,
                    heading_deg=self.backend.compass_heading(state.yaw_deg),
                    roll_deg=state.roll_deg, pitch_deg=state.pitch_deg,
                    vx_mps=state.vx, vy_mps=state.vy, vz_mps=state.vz)

    def _publish_cameras(self, due, state, sim_us):
        if not due:
            return False
        requests = []
        try:
            # The body datum reaches the record through the backend as well, so
            # it belongs inside the guard: a backend that cannot answer it
            # fails these cameras rather than the physics loop.
            body = self._body(state)
            poses = {sensor['id']: resolve(self.profile.data, sensor['id'], state)
                     for sensor, _index in due}
            requests = [(sensor['id'], sensor['model'], poses[sensor['id']],
                         self.publisher.camera_slot(sensor['id'])) for sensor, _index in due]
            self.backend.render_views(requests)
        except (RuntimeError, ValueError) as exc:
            for sensor, _index in due:
                self._failed_capture(sensor['id'], exc)
            return False
        finally:
            # Drop the shared-memory views before the slots advance under them.
            del requests
        for sensor, _index in due:
            sensor_id = sensor['id']
            try:
                video_sequence = self.publisher.commit_camera(sensor_id, sim_us)
            except (RuntimeError, ValueError) as exc:
                self._failed_capture(sensor_id, exc, generated=True)
                continue
            self._record(sensor_id, sim_us, lambda sequence, sid=sensor_id,
                         video_sequence=video_sequence, sensor=sensor:
                         self.publisher.write_compact(
                             sid, sequence, self.capture_id, sim_us, CAMERA_FRAME,
                             dict(schema='camera.frame.v1',
                                  video_sequence=video_sequence,
                                  pose_world=poses[sid].tolist(),
                                  pose=pose_angles(poses[sid]), body=body,
                                  calibration_revision=self.profile.digest,
                                  sync_group=sensor.get('sync_group'))))
        return True

    def _publish_rays(self, sensor, index, state, sim_us):
        sensor_id, kind, model = sensor['id'], sensor['type'], sensor['model']
        pose_world = resolve(self.profile.data, sensor_id, state)
        rng = sensor_models.capture_rng(self.seed, sensor_id, self.publisher.reset_epoch, index)
        truth = self.backend.range_truth(pose_world, kind, model)
        ranges, confidence = sensor_models.measure(truth, model, rng)
        common = dict(pose=pose_angles(pose_world), pose_world=pose_world.tolist(),
                      body=self._body(state), min_range_m=model['min_range_m'],
                      max_range_m=model['max_range_m'])
        if kind in ARRAY_TYPES:
            returns = int(np.isfinite(ranges).sum())

            def publish(sequence):
                self.publisher.write_array(
                    sensor_id, sequence, self.capture_id, sim_us,
                    sensor_models.pack_array(ranges, confidence),
                    status=STATUS_VALID if returns else STATUS_INVALID)
                self.publisher.write_compact(
                    sensor_id, sequence, self.capture_id, sim_us, LIDAR_FRAME,
                    dict(schema='lidar.frame.v1', array_sequence=sequence, type=kind,
                         returns=returns, samples=int(ranges.size),
                         calibration=sensor_models.calibration(kind, model), **common),
                    status=STATUS_VALID if returns else STATUS_INVALID)

            self._record(sensor_id, sim_us, publish, valid=bool(returns))
            return
        value, level, returns = sensor_models.reduce_beam(ranges, confidence, model['reducer'])
        self._record(sensor_id, sim_us, lambda sequence: self.publisher.write_compact(
            sensor_id, sequence, self.capture_id, sim_us, RANGE_SAMPLE,
            dict(schema='range.sample.v1', type=kind, range_m=value, confidence=level,
                 returns=returns, samples=int(ranges.size), reducer=model['reducer'],
                 **common),
            status=STATUS_VALID if value is not None else STATUS_INVALID),
            valid=value is not None)

    def _publish_state(self, sensor, index, state, sim_time_s, sim_us):
        """One state-sensor record: a measurement, and nothing else.

        These carry no pose and no vehicle datum. A GNSS record beside the true
        position is not a measurement, and the sensor's axes are static
        configuration a consumer already has from the manifest.
        """
        sensor_id, kind, model = sensor['id'], sensor['type'], sensor['model']
        rng = sensor_models.capture_rng(self.seed, sensor_id,
                                        self.publisher.reset_epoch, index)
        if kind == 'motion.imu':
            dt = 0.0 if self._previous_time is None else sim_time_s - self._previous_time
            payload, valid = state_sensors.imu(model, self.profile.data, sensor_id,
                                               state, self._previous, dt, rng)
        elif kind == 'position.gnss':
            payload, valid = state_sensors.gnss(model, self.backend, state, rng)
        elif kind == 'altimeter.barometric':
            payload, valid = state_sensors.barometer(model, self.backend, state, rng)
        elif kind == 'heading.magnetometer':
            payload, valid = state_sensors.magnetometer(model, self.backend, state, rng)
        else:
            payload, valid = state_sensors.temperature(model, self.backend, state, rng)
        self._record(sensor_id, sim_us, lambda sequence: self.publisher.write_compact(
            sensor_id, sequence, self.capture_id, sim_us, STATE_PAYLOAD[kind], payload,
            status=STATUS_VALID if valid else STATUS_INVALID), valid=valid)

    def _record(self, sensor_id, sim_us, publish, *, valid=True):
        """Run one sensor's publication and account for what happened.

        A transport failure is counted as a publisher drop and its message
        retained as the sensor's fault; it never stops the simulation, which is
        the whole point of a provider nobody is allowed to block.
        """
        counts = self.counts[sensor_id]
        counts['scheduled'] += 1
        counts['generated'] += 1
        sequence = self.publisher.next_sequence(sensor_id)
        try:
            publish(sequence)
        except (RuntimeError, ValueError) as exc:
            counts['drops'] += 1
            counts['fault'] = str(exc)
            return False
        counts['published'] += 1
        counts['last_sequence'] = sequence
        counts['last_sim_time_s'] = sim_us / 1e6
        counts['fault'] = ''
        if not valid:
            counts['invalid'] += 1
        return True

    def _failed_capture(self, sensor_id, exc, *, generated=False):
        counts = self.counts[sensor_id]
        counts['scheduled'] += 1
        counts['generated'] += int(generated)
        counts['drops'] += 1
        counts['fault'] = str(exc)
        self.publisher.next_sequence(sensor_id)

    # -- diagnostics -------------------------------------------------------

    def costs(self):
        """Where the sensor plane spent its wall time, in seconds."""
        return dict(render_s=round(self.render_total_s, 4),
                    sample_s=round(self.sample_total_s, 4))

    def production(self):
        """Per-sensor production health, for run artifacts and the monitor.

        This is the provider's half of the health contract: what dsim
        configured, what it actually generated, and what went wrong. The
        consumers' half -- what each of them managed to take in -- is reported
        separately and correlated for display; neither side is derived from
        the other, because that is exactly how a stalled reader comes to look
        like a healthy producer.
        """
        elapsed = self.elapsed_ticks / self.physics_hz if self.physics_hz else 0.0
        out = {}
        for sensor_id, counts in self.counts.items():
            observed = counts['published'] / elapsed if elapsed > 0 else 0.0
            out[sensor_id] = dict(counts, observed_hz=observed,
                                  type=self.types[sensor_id],
                                  state=self._grade(counts, observed, elapsed))
        return out

    def _grade(self, counts, observed_hz, elapsed):
        """A sensor is starting until its first period has had time to pass."""
        if counts['drops']:
            return health.BAD
        if elapsed <= 0.0 or counts['published'] == 0 and elapsed < 2.0 / counts['configured_hz']:
            return 'starting'
        return health.grade(observed_hz, counts['configured_hz'])
