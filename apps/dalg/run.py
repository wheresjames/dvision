from __future__ import annotations

import hashlib
import json
import time
from dataclasses import fields
from pathlib import Path

import numpy as np

from dcmn.health import IntakeMeter
from dcmn.module_bus import PymembusModuleBus, requests_shutdown
from dcmn.pacing import PeriodicDeadline, simulated_poll_delay
from dvision2_common import load_map, load_pymembus, shared_names
from dalg.algo import ALGORITHMS, CONFIGS
from dalg.algo.controls import ConstantAlgorithm, ExactRangeAlgorithm
from dalg.model import Frame, Intrinsics, Pose
from dalg.report import write_report
from dalg.truth import ground_truth
from dalg.visibility import observable_mask

# Settings that configure the sensor rather than the algorithm. They live in
# the profile's settings block for the same reason everything else does -- one
# flat, diffable object -- but they must not be handed to an algorithm config,
# which would reject them as unknown fields.
SENSOR_SETTINGS = ("range_stride",)

# The coordinator paces its heartbeat on simulated time, so that is the clock
# its silence has to be measured against; under a fixed-timestep harness the
# wall clock runs far ahead of it. The wall-clock bound is only a backstop for
# a pipeline that has stopped entirely, sim clock included.
COORDINATOR_SILENCE_SIM_S = 3.5
COORDINATOR_SILENCE_WALL_S = 60.0


def copy_video_frame(frame: np.ndarray) -> np.ndarray:
    """Copy a shared-memory RGB frame without changing its published orientation."""
    return np.array(frame, copy=True)


def matches_prepare(profile, requirements, process_id: str = "") -> bool:
    """Whether this instance satisfies an exclusive algorithm selector."""
    selectors = [str(value).partition(":")[2] for value in requirements
                 if str(value).partition(":")[0] == "algorithm"]
    # A coordinator that did not require an algorithm still broadcasts a
    # useful lifecycle. One DALG may observe it opportunistically; readiness
    # is advisory because the coordinator has no algorithm barrier to wait on.
    if not selectors: return True
    return any(not selector or selector in (profile.name, profile.algorithm, process_id)
               for selector in selectors)


def algorithm_settings(algorithm: str, settings: dict) -> dict:
    """Profile settings this algorithm's configuration actually declares."""
    config = CONFIGS.get(algorithm)
    if config is None: return {}
    declared = {field.name for field in fields(config)}
    unknown = sorted(set(settings) - declared - set(SENSOR_SETTINGS))
    if unknown:
        raise ValueError(f"{algorithm} profile has unknown settings: "
                         + ", ".join(unknown))
    return {name: value for name, value in settings.items() if name in declared}


class DalgRun:
    """Observer lifecycle shared by headless and Tk front ends."""

    def __init__(self, instance_id: str, profile, root: Path) -> None:
        self.id, self.profile, self.root = instance_id, profile, root
        self.pm = load_pymembus()
        self.names = shared_names(instance_id)
        self.video = self.status = None
        self.bus = PymembusModuleBus(instance_id, "algorithm", "dalg",
                                     sim_time=self.sim_time_s)
        self.state = "CONNECTING"
        self.reason = ""
        self.run_id = ""
        self.start_sim_time: float | None = None
        self.active = False
        self.done = False
        self.shutdown_requested = False
        self.last_seq = -1
        self._last_display_seq = -1
        self.last_frame = None
        self.frames = 0
        self.algorithms = {}
        self.truth = None
        self.sim_map = None
        self.intrinsics = None
        self.provenance = {}
        self._last_heartbeat = -1e9
        self._last_preview = -1e9
        self.preview_result = None
        self.report_dir = None
        self._event_log = []
        self._hello_sent = False
        self._coordinator_seen = time.monotonic()
        self._coordinator_seen_sim = None
        self._coordinator_process_id = ""
        self._camera_poses = []
        self._fov_h_deg = 70.0
        self.intake = IntakeMeter()
        self._tour_value = ({} if profile.tour is None else
                            json.loads(profile.tour.read_text(encoding="utf-8")))
        self._capture_cadence = PeriodicDeadline(float(
            self._tour_value.get("capture_fps", 5.0)))
        self._tour_digest = ("" if profile.tour is None else
                             hashlib.sha256(profile.tour.read_bytes()).hexdigest())

    def sim_time_s(self) -> float:
        if self.status is None:
            return 0.0
        try:
            return float(self.status.getAll().get("sim.time_s", 0.0))
        except (TypeError, ValueError):
            return 0.0

    def poll_delay(self) -> float:
        """Wall delay for checking whether simulated-time work is due."""
        values = {} if self.status is None else self.status.getAll()
        rate = float(self._tour_value.get("capture_fps", 5.0))
        return simulated_poll_delay(max(rate, 0.1), values)

    def connect(self) -> bool:
        if self.video is None:
            handle = self.pm.memvid()
            if handle.open_existing(self.names["video"]): self.video = handle
        if self.status is None:
            handle = self.pm.memkv()
            if handle.open(self.names["status"]): self.status = handle
        self.bus.connect()
        if self.video is not None and self.status is not None:
            if self.state == "CONNECTING": self.state = "READY"
            return True
        return False

    def _publish_presence(self, now: float) -> None:
        """Announce liveness on the wall clock.

        Presence is projected by PipelineView, which expires members on
        time.monotonic(). Pacing this on simulated time made DALG vanish from
        every registry the moment the simulator lagged -- or, before the
        simulator appeared at all, after a single heartbeat.
        """
        if not self._hello_sent:
            self.bus.publish("module.hello", payload={
                "state": self.state, "profile": self.profile.name,
                "capabilities": {"algorithms": list(ALGORITHMS),
                                 "sensors": list(self.profile.sensors)}})
            self._hello_sent = True
        if now - self._last_heartbeat < 1.0: return
        # An idle demonstrator owes no samples. Treating its deliberate zero
        # as a shortfall would make every pipeline red before a run starts.
        self.intake.set_wanted(float(self._tour_value.get("capture_fps", 5.0))
                               if self.active else 0.0)
        self.bus.publish("module.heartbeat", run_id=self.run_id, payload={
            "intake": self.intake.report(self.sim_time_s(),
                                         overruns=self.bus.overruns),
            "state": self.state, "ready": self.state in ("READY", "SCHEDULED"),
            "profile": self.profile.name, "profile_digest": self.profile.digest,
            "capabilities": {"algorithms": list(ALGORITHMS),
                             "sensors": list(self.profile.sensors)},
        })
        self._last_heartbeat = now

    def _reject(self, run_id: str, reason: str) -> None:
        self.bus.publish("run.reject", run_id=run_id, payload={"reason": reason})

    def _prepare(self, event) -> None:
        # Coordinators repeat the preparation snapshot every second for late
        # joiners, so staying quiet until the shared memory is open is better
        # than rejecting: a rejection aborts their whole mission.
        if self.status is None or self.active or self.done: return
        p = event.payload
        if not matches_prepare(self.profile, p.get("required_roles", ()),
                               self.bus.process_id):
            return
        expected_id = self._tour_value.get("tour_id")
        if self.profile.tour is not None:
            if p.get("tour_id") != expected_id:
                self._reject(event.run_id, f"tour mismatch: expected {expected_id}")
                return
            if p.get("tour_digest") != self._tour_digest:
                self._reject(event.run_id, "tour digest mismatch")
                return
        expected_map = self._tour_value.get("map_sha", "") or p.get("map_digest", "")
        values = self.status.getAll()
        published = str(values.get("sim.map", "")).strip()
        if not published:
            self._reject(event.run_id, "simulator has not published a map")
            return
        published_map = Path(published)
        if not published_map.is_absolute():
            published_map = self.root / published_map
        if expected_map and (not published_map.is_file() or
                hashlib.sha256(published_map.read_bytes()).hexdigest() != expected_map):
            self._reject(event.run_id, "simulator map digest mismatch")
            return
        if "rgb" not in self.profile.sensors:
            self._reject(event.run_id, "selected algorithm requires rgb")
            return
        expected_range = self.profile.sensor_config.get("range")
        actual_range = values.get("range.config", "none")
        if expected_range and actual_range != expected_range:
            self._reject(event.run_id, "range sensor mismatch: expected "
                         f"{expected_range}, got {actual_range}")
            return
        if not expected_range and "range" in self.profile.sensors:
            self._reject(event.run_id, "range sensor profile requires a configuration")
            return
        self.run_id = event.run_id
        self._coordinator_process_id = event.process_id
        self.state = "READY"
        self.provenance.update({
            "tour_id": expected_id, "tour_digest": self._tour_digest,
            "map_digest": expected_map,
            "sensor_config": dict(self.profile.sensor_config),
            "flight_mode": "tour" if self.profile.tour is not None else "manual",
            "coordinator_role": event.role,
            "coordinator_implementation": event.implementation,
            "navigator_implementation": (event.implementation
                                           if event.role == "navigator" else ""),
        })
        self.bus.publish("run.ready", run_id=self.run_id, payload={
            "profile": self.profile.name, "algorithm": self.profile.algorithm,
            "capabilities": {"algorithms": [self.profile.algorithm],
                             "sensors": list(self.profile.sensors)},
            "configuration_digest": self.profile.digest})

    def _initialize(self) -> None:
        values = self.status.getAll()
        map_path = Path(values["sim.map"])
        if not map_path.is_absolute(): map_path = self.root / map_path
        self.sim_map = load_map(map_path)
        self.truth = ground_truth(self.sim_map)
        self.intrinsics = Intrinsics(*[int(float(values[name])) if name in
            ("camera.width_px", "camera.height_px") else float(values[name])
            for name in ("camera.width_px", "camera.height_px", "camera.fx_px",
                         "camera.fy_px", "camera.cx_px", "camera.cy_px")])
        try:
            self._fov_h_deg = float(values["camera.fov_h_deg"])
        except (KeyError, TypeError, ValueError):
            pass
        if self.profile.algorithm == "exact_range":
            selected = ExactRangeAlgorithm(self.truth)
        else:
            settings = algorithm_settings(self.profile.algorithm,
                                          self.profile.settings)
            if self.profile.algorithm == "monocular_depth" and settings.get("model_path"):
                model_path = Path(settings["model_path"]).expanduser()
                if not model_path.is_absolute(): model_path = self.root/model_path
                settings["model_path"] = str(model_path)
            selected = ALGORITHMS[self.profile.algorithm](
                self.sim_map.width, self.sim_map.height, self.intrinsics,
                settings=settings)
        self.algorithms = {self.profile.algorithm: selected}
        self.algorithms.setdefault(
            "constant", ConstantAlgorithm(self.sim_map.width, self.sim_map.height))
        self.algorithms.setdefault("exact_range", ExactRangeAlgorithm(self.truth))
        for algorithm in self.algorithms.values(): algorithm.start()

    def _camera_pose(self, values, pose: Pose) -> Pose:
        """The lens pose the simulator publishes, not the vehicle datum.

        camera.pitch_deg is already absolute -- it folds in the fixed mount
        tilt and the body's own pitch -- while camera.t*_m is an offset from
        the datum.
        """
        try:
            return Pose(pose.x_m+float(values["camera.tx_m"]),
                        pose.y_m+float(values["camera.ty_m"]),
                        pose.z_m+float(values["camera.tz_m"]),
                        pose.heading_deg+float(values["camera.yaw_deg"]),
                        float(values["camera.roll_deg"]),
                        float(values["camera.pitch_deg"]))
        except (KeyError, TypeError, ValueError):
            return pose

    def _observe_frame(self, now: float) -> None:
        capture_fps = float(self._tour_value.get("capture_fps", 5.0))
        self._capture_cadence.set_rate(max(capture_fps, 0.1))
        if not self._capture_cadence.due(now): return
        seq = self.video.getSeq()
        self.intake.note_sequence(seq)
        if seq == self.last_seq: return
        self.last_seq = seq
        slot = self.video.getPtr(-1)
        rgb = copy_video_frame(self.video[slot])
        values = self.status.getAll()
        pose = Pose(*[float(values[name]) for name in
            ("drone.x_m", "drone.y_m", "drone.z_m", "drone.heading_deg",
             "drone.roll_deg", "drone.pitch_deg")])
        camera = self._camera_pose(values, pose)
        ranges = confidence = None
        range_name = self.profile.sensor_config.get("range")
        if range_name:
            from dsim.range import range_config, raycast_map
            stride = int(self.profile.settings.get("range_stride", 8))
            # raycast_map applies the camera mount tilt and height itself, so
            # it takes the body pose; the camera pose is what inverts it.
            ranges, confidence = raycast_map(
                self.sim_map, pose, self.intrinsics,
                config=range_config(range_name), stride=stride)
        frame = Frame(seq, now, rgb, pose, ranges, confidence, camera)
        self.algorithms[self.profile.algorithm].observe(frame)
        self._capture_cadence.advance(now)
        self.intake.record()
        self.frames += 1
        self._camera_poses.append(camera)
        self.last_frame = rgb
        selected = self.algorithms[self.profile.algorithm]
        interval = float(getattr(getattr(selected, "config", None),
                                 "preview_interval_s", .5))
        if now - self._last_preview >= interval:
            self.preview_result = selected.preview()
            self._last_preview = now

    def _update_camera(self) -> None:
        """Display connected video without admitting it to a measurement."""
        if self.video is None: return
        seq = self.video.getSeq()
        if seq <= 0 or seq == self._last_display_seq: return
        self._last_display_seq = seq
        slot = self.video.getPtr(-1)
        self.last_frame = copy_video_frame(self.video[slot])

    def _coordinator_silent(self, now: float) -> bool:
        wall = time.monotonic() - self._coordinator_seen
        if self._coordinator_seen_sim is None:
            return wall > COORDINATOR_SILENCE_WALL_S
        return (now - self._coordinator_seen_sim > COORDINATOR_SILENCE_SIM_S
                or wall > COORDINATOR_SILENCE_WALL_S)

    def _handle(self, event, now: float) -> bool:
        """Apply one bus event. Returns False to stop draining the queue."""
        if requests_shutdown(event):
            self.shutdown_requested = True
            if self.done:
                # The report is already on disk and its outcome is settled.
                # Shutdown still has to stop this process -- that is the point
                # of it -- but relabelling a completed run "aborted" on the way
                # out would rewrite a measurement that had already succeeded,
                # and turn its exit status into a failure.
                return False
            self.reason = str(event.payload.get(
                "reason", "instance shutdown requested"))
            self.state = "STOPPING"
            self.provenance["coordinator_outcome"] = "aborted"
            if self.active: self.finish(partial=True)
            else: self.done = True
            return False
        # One DalgRun measures one run. Once it is finished the record is
        # closed, so later lifecycle traffic -- a coordinator repeating itself,
        # or a second run starting on the same bus -- is observed but never
        # allowed to edit it. Draining continues so shutdown is still heard.
        if self.done: return True
        if event.role not in ("navigator", "controller"): return True
        if (self._coordinator_process_id and
                event.process_id != self._coordinator_process_id): return True
        self._coordinator_seen = time.monotonic()
        self._coordinator_seen_sim = now
        if event.type == "run.prepare":
            self._prepare(event)
            return True
        # An empty run_id is "no run", not a wildcard: without this an
        # untargeted lifecycle event would schedule a run we never prepared.
        if not self.run_id or event.run_id != self.run_id: return True
        if event.type == "run.start_scheduled":
            self.start_sim_time = float(event.payload["start_sim_time_s"])
            self.state = "SCHEDULED"
        elif event.type in ("run.completed", "run.aborted"):
            self.reason = str(event.payload.get("reason", ""))
            outcome = event.payload.get("outcome", "")
            terminal_state = event.payload.get("state", "")
            self.provenance["coordinator_outcome"] = outcome
            self.provenance["coordinator_terminal_state"] = terminal_state
            if event.role == "navigator":
                self.provenance["navigator_outcome"] = outcome
                self.provenance["navigator_terminal_state"] = terminal_state
            self.finish(partial=event.type == "run.aborted")
        elif event.type == "run.state":
            scheduled = event.payload.get("start_sim_time_s")
            if scheduled is not None and self.start_sim_time is None:
                self.start_sim_time = float(scheduled)
                self.state = "SCHEDULED"
        return True

    def step(self) -> str:
        connected = self.connect()
        if not self.active: self._update_camera()
        now = self.sim_time_s()
        self._publish_presence(time.monotonic())
        for event in self.bus.receive():
            self._event_log.append({"type": event.type, "run_id": event.run_id,
                                    "role": event.role,
                                    "sim_time_s": event.sim_time_s,
                                    "payload": event.payload})
            if not self._handle(event, now): break
        # Nothing below can run without the video and status segments, and
        # reaching for them anyway is how a late simulator became a traceback.
        if not connected or self.done: return self.state
        if not self.active and self.start_sim_time is not None and now >= self.start_sim_time:
            self._initialize()
            self.active = True
            self.state = "RUNNING"
            self.bus.publish("run.started", run_id=self.run_id,
                             payload={"start_sim_time_s": self.start_sim_time})
        if self.active:
            if self._coordinator_silent(now):
                self.reason = "measurement coordinator heartbeat expired"
                self.provenance["coordinator_outcome"] = "aborted"
                self.finish(partial=True)
            else:
                self._observe_frame(now)
        return self.state

    def finish(self, partial=False) -> None:
        if self.done: return
        self.done = True
        self.active = False
        self.state = "ABORTED" if partial else "COMPLETE"
        try:
            if not self.algorithms:
                if self.status is None: return
                self._initialize()
            results = {name: algorithm.finish()
                       for name, algorithm in self.algorithms.items()}
            self.preview_result = results[self.profile.algorithm]
            observable = observable_mask(self.truth, self._camera_poses,
                                         fov_h_deg=self._fov_h_deg)
            self.provenance.update({
                "start_sim_time_s": self.start_sim_time, "frames": self.frames,
                "observable_cells": int(np.count_nonzero(observable)),
                "map_cells": int(observable.size)})
            values = {} if self.status is None else self.status.getAll()
            configured = str(values.get("sim.report_dir", "")).strip()
            root = Path(configured) if configured else self.root / "reports"
            self.report_dir = write_report(
                root, run_id=self.run_id, profile=self.profile, truth=self.truth,
                results=results, provenance=self.provenance, partial=partial,
                reason=self.reason, events=self._event_log, observable=observable)
        finally:
            self.bus.publish("run.completed" if not partial else "run.aborted",
                             run_id=self.run_id,
                             payload={"state": self.state, "reason": self.reason})

    def close(self) -> None:
        if self.active and not self.done:
            self.reason = "dalg shutdown"
            self.finish(partial=True)
        self.bus.publish("module.goodbye", run_id=self.run_id,
                         payload={"state": self.state})
        self.bus.close()
        for handle in (self.status, self.video):
            if handle is not None: handle.close()
