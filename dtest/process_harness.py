"""Real-process DSIM harness for transport-level contract tests."""

from __future__ import annotations

import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Callable

import numpy as np

from dtest.artifacts import save_failure_bundle
from dtest.backend import BackendCapabilities
from dtest.calibration_scene import CALIBRATION_MAP, ROOT
from dvision2_common import controlled_command, load_pymembus, shared_names


# Timeline samples retained per instance before the record is decimated.
_TIMELINE_LIMIT = 2000


class DsimProcessHarness:
    """Own a uniquely named DSIM process and its real IPC clients."""

    def __init__(self, artifact_dir: Path, *, map_path: Path = CALIBRATION_MAP,
                 start_heading: float = 0.0, fps: int = 30,
                 setpoint_timeout_s: float | None = 0.0) -> None:
        self.id = f"dtest-{uuid.uuid4().hex[:12]}"
        self.names = shared_names(self.id)
        self.control_source = self.id
        self.control_lease = uuid.uuid4().hex
        self.artifact_dir = Path(artifact_dir)
        self.report_dir = self.artifact_dir / "report"
        self.map_path = Path(map_path)
        self.start_heading = start_heading
        self.fps = fps
        # None keeps the simulator's own default; a test that needs the guided
        # setpoint failsafe asks for it explicitly.
        self.setpoint_timeout_s = setpoint_timeout_s
        self.pm = load_pymembus()
        self.process: subprocess.Popen | None = None
        self._stderr_fh = None
        self._stderr_path = self.artifact_dir / "dsim.stderr.log"
        self.video = None
        self.command = None
        self.status = None
        self.last_status: dict = {}
        self.last_frame: np.ndarray | None = None
        self.first_frame: np.ndarray | None = None
        # Structured evidence retained for failure bundles.
        self.commands: list[dict] = []
        self.timeline: list[dict] = []
        self.positions: list[tuple[float, float]] = []
        self._lease_acquired = False
        self._last_lease_heartbeat = 0.0

    def __enter__(self) -> "DsimProcessHarness":
        try:
            self.start()
            return self
        except Exception:
            self.close()
            raise

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc is not None:
            self.save_failure(error=repr(exc))
        self.close()

    def save_failure(self, **details) -> Path:
        """Write the full structured failure bundle for this instance."""
        return save_failure_bundle(
            self.artifact_dir,
            frames={"initial_frame": self.first_frame, "final_frame": self.last_frame},
            frame_rgb=self.last_frame,
            telemetry=self.last_status,
            commands=self.commands,
            timeline=self.timeline,
            positions=self.positions,
            fixture={
                "instance_id": self.id,
                "map": str(self.map_path),
                "start_heading_deg": self.start_heading,
                "fps": self.fps,
                "shared_names": self.names,
                "camera": {
                    k: self.last_status.get(k)
                    for k in ("camera.width_px", "camera.height_px",
                              "camera.fov_h_deg", "camera.fov_v_deg",
                              "camera.pitch_deg")
                },
            },
            details={"instance_id": self.id, **details},
        )

    def start(self) -> None:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, str(ROOT / "apps/dsim/dsim.py"),
            "--id", self.id,
            "--map", str(self.map_path),
            "--start-heading", str(self.start_heading),
            "--fps", str(self.fps),
            "--no-ui",
            "--report-dir", str(self.report_dir),
        ]
        if self.setpoint_timeout_s is not None:
            cmd += ["--setpoint-timeout", str(self.setpoint_timeout_s)]
        self._stderr_fh = self._stderr_path.open("w", encoding="utf-8")
        self.process = subprocess.Popen(
            cmd, cwd=str(ROOT), stdout=subprocess.DEVNULL,
            stderr=self._stderr_fh, text=True,
        )
        self._wait_until(self._connect, timeout=10.0, description="DSIM IPC readiness")
        self._wait_until(
            lambda: self.read_status().get("sim.id") == self.id,
            timeout=5.0,
            description="matching simulator status",
        )
        self.send("acquire_control")
        self.wait_status(
            lambda s: s.get("control.owner") == self.control_source,
            description="test harness control lease",
        )
        self._lease_acquired = True
        self._wait_until(
            lambda: self.video.getSeq() > 0,
            timeout=5.0,
            description="first video frame",
        )

    def _connect(self) -> bool:
        self._assert_running()
        if self.video is None:
            handle = self.pm.memvid()
            if handle.open_existing(self.names["video"]):
                self.video = handle
        if self.command is None:
            handle = self.pm.memcmd()
            if handle.open(self.names["command"], 65536):
                self.command = handle
        if self.status is None:
            handle = self.pm.memkv()
            if handle.open(self.names["status"]):
                self.status = handle
        return all((self.video is not None, self.command is not None, self.status is not None))

    def _assert_running(self) -> None:
        if self.process is not None and self.process.poll() is not None:
            if self._stderr_fh is not None:
                self._stderr_fh.flush()
            stderr = (
                self._stderr_path.read_text(encoding="utf-8")
                if self._stderr_path.exists() else ""
            )
            raise AssertionError(
                f"DSIM exited early with {self.process.returncode}: {stderr[-2000:]}"
            )

    def _wait_until(self, predicate: Callable[[], bool], *, timeout: float,
                    description: str) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self._assert_running()
            if predicate():
                return
            time.sleep(0.02)
        raise AssertionError(
            f"timeout waiting for {description}; status={self.last_status!r}; "
            f"frame_seq={self.video.getSeq() if self.video is not None else None}"
        )

    def send(self, command_type: str, **fields) -> None:
        assert self.command is not None
        if command_type not in ("acquire_control", "release_control", "heartbeat"):
            heartbeat = controlled_command(
                "heartbeat", self.control_source, self.control_lease)
            if not self.command.write(heartbeat):
                raise AssertionError("failed to write lease heartbeat")
        self.commands.append({
            "sim_time_s": self.last_status.get("sim.time_s"),
            "type": command_type, **fields,
        })
        payload = controlled_command(
            command_type, self.control_source, self.control_lease, **fields)
        if not self.command.write(payload):
            raise AssertionError(f"failed to write {command_type} command")

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(deterministic=False, physical_vehicle=False)

    def arm(self) -> None:
        self.send("arm", armed=True)

    def disarm(self) -> None:
        self.send("arm", armed=False)

    def zero(self) -> None:
        self.send("zero")

    def send_body_velocity(self, forward_mps: float, right_mps: float,
                           up_mps: float, yaw_rate_dps: float) -> None:
        self.send(
            "velocity", forward_mps=forward_mps, right_mps=right_mps,
            up_mps=up_mps, yaw_rate_dps=yaw_rate_dps,
        )

    def read_telemetry(self) -> dict:
        return self.read_status()

    def settle(self, seconds: float, *, timeout: float = 10.0) -> None:
        """Advance the running simulator by ``seconds`` of its published clock.

        This is a bounded readiness wait on ``sim.time_s`` rather than a bare
        sleep, so a stalled simulator fails with its last status instead of
        silently passing on host timing.
        """
        start = float(self.read_status()["sim.time_s"])
        self._wait_until(
            lambda: float(self.read_status()["sim.time_s"]) - start >= seconds,
            timeout=max(timeout, seconds * 4.0),
            description=f"{seconds:.2f}s of simulated time",
        )

    def read_status(self) -> dict:
        assert self.status is not None
        now = time.monotonic()
        if (self._lease_acquired and self.command is not None
                and now - self._last_lease_heartbeat >= 1.0):
            self.command.write(controlled_command(
                "heartbeat", self.control_source, self.control_lease))
            self._last_lease_heartbeat = now
        self.last_status = dict(self.status.getAll())
        self._record(self.last_status)
        return self.last_status

    def _record(self, status: dict) -> None:
        """Append one pose/velocity/heading/epoch/sequence timeline sample."""
        sample = {
            key: status.get(key) for key in (
                "sim.time_s", "drone.x_m", "drone.y_m", "drone.z_m",
                "drone.vx_mps", "drone.vy_mps", "drone.vz_mps",
                "drone.heading_deg", "drone.mode", "drone.armed",
                "link.command_count", "link.last_command_type", "status.message",
            )
        }
        sample["status_epoch"] = (
            self.status.getEpoch() if self.status is not None else None
        )
        sample["frame_seq"] = self.video.getSeq() if self.video is not None else None
        if self.timeline and self.timeline[-1] == sample:
            return
        self.timeline.append(sample)
        try:
            self.positions.append((float(status["drone.x_m"]), float(status["drone.y_m"])))
        except (KeyError, TypeError, ValueError):
            pass
        if len(self.timeline) > _TIMELINE_LIMIT:
            # Halve the resolution rather than truncate, so a long run's bundle
            # still spans the whole flight.
            self.timeline = self.timeline[::2]
            self.positions = self.positions[::2]

    def wait_status(self, predicate: Callable[[dict], bool], *, timeout: float = 5.0,
                    description: str = "status condition") -> dict:
        result: dict = {}

        def check() -> bool:
            nonlocal result
            result = self.read_status()
            return predicate(result)

        self._wait_until(check, timeout=timeout, description=description)
        return result

    def read_frame(self, *, newer_than: int | None = None,
                   timeout: float = 5.0) -> tuple[int, np.ndarray]:
        assert self.video is not None
        if newer_than is not None:
            self._wait_until(
                lambda: self.video.getSeq() > newer_than,
                timeout=timeout,
                description=f"video sequence newer than {newer_than}",
            )
        seq = int(self.video.getSeq())
        slot = self.video.getPtr(-1)
        self.last_frame = np.array(self.video[slot], copy=True)
        if self.first_frame is None:
            self.first_frame = self.last_frame.copy()
        return seq, self.last_frame

    def close(self) -> None:
        if self.command is not None:
            try:
                self.send("release_control")
                self._lease_acquired = False
            except Exception:
                pass
        for handle_name in ("status", "command", "video"):
            handle = getattr(self, handle_name)
            if handle is not None:
                try:
                    handle.close()
                except Exception:
                    pass
                setattr(self, handle_name, None)
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5.0)
        self.process = None
        if self._stderr_fh is not None:
            self._stderr_fh.close()
            self._stderr_fh = None
        # DSIM owns unlinking during normal lifetime only indirectly; remove
        # names explicitly so interrupted/parallel test runs cannot collide.
        for factory, name in (
            (self.pm.memkv, self.names["status"]),
            (self.pm.memcmd, self.names["command"]),
            (self.pm.memvid, self.names["video"]),
        ):
            try:
                factory.remove(name)
            except Exception:
                pass
