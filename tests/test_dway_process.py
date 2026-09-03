"""dway against a real dsim process, over the live shared-memory transports."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from dtest.artifacts import artifact_directory
from dtest.dway_rig import write_corridor_tour
from dtest.process_harness import DsimProcessHarness

ROOT = Path(__file__).resolve().parents[1]
MAZE_012 = ROOT / "assets/maps/maze_012.txt"
SETPOINT_TIMEOUT_S = 2.0


def start_dway(harness: DsimProcessHarness, tour: Path, stderr_path: Path,
               *extra: str) -> tuple[subprocess.Popen, object]:
    handle = stderr_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        [sys.executable, str(ROOT / "dway/dway.py"), "--id", harness.id,
         "--tour", str(tour), "--no-ui", "--finish-action", "hold", *extra],
        cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=handle,
        start_new_session=True)
    return process, handle


def test_killing_dway_mid_flight_leaves_the_vehicle_holding(tmp_path) -> None:
    artifact_dir = artifact_directory(tmp_path, "dway-process")
    tour = write_corridor_tour(tmp_path / "corridor.json")
    harness = DsimProcessHarness(artifact_dir, map_path=MAZE_012,
                                 setpoint_timeout_s=SETPOINT_TIMEOUT_S)
    dway = None
    stderr_path = artifact_dir / "dway.stderr.log"
    stderr_handle = None
    try:
        harness.start()
        # The test harness takes control on startup; dway is the client under
        # test, so it must be the one holding the lease.
        harness.send("release_control")
        harness._lease_acquired = False
        harness.wait_status(lambda s: s.get("control.owner") == "",
                            description="released control lease")

        dway, stderr_handle = start_dway(harness, tour, stderr_path)
        harness.wait_status(
            lambda s: s.get("control.owner") == f"dway-{harness.id}",
            timeout=20.0, description="dway control lease")
        harness.wait_status(
            lambda s: (s.get("drone.mode") == "GUIDED"
                       and float(s.get("drone.speed_mps", 0.0)) > 0.2),
            timeout=20.0, description="dway flying the first leg")

        # Killed outright: no hold, no release, nothing but a stopped stream.
        os.kill(dway.pid, signal.SIGKILL)
        dway.wait(timeout=5.0)
        held = harness.wait_status(
            lambda s: s.get("drone.mode") == "HOLD",
            timeout=SETPOINT_TIMEOUT_S + 8.0,
            description="simulator failsafe HOLD after the stream stopped")
        assert held["failsafe.reason"] in ("setpoint_timeout", "control_lease_expired")
        settled = harness.wait_status(
            lambda s: float(s.get("drone.speed_mps", 1.0)) < 0.05,
            timeout=5.0, description="vehicle stopped")
        assert settled["drone.armed"] == "1"
    finally:
        if dway is not None and dway.poll() is None:
            dway.kill()
            dway.wait(timeout=5.0)
        if stderr_handle is not None:
            stderr_handle.close()
        harness.close()


def test_dway_flies_a_tour_through_the_real_transports(tmp_path) -> None:
    artifact_dir = artifact_directory(tmp_path, "dway-flight")
    tour = write_corridor_tour(tmp_path / "corridor.json")
    harness = DsimProcessHarness(artifact_dir, map_path=MAZE_012,
                                 setpoint_timeout_s=SETPOINT_TIMEOUT_S)
    dway = None
    stderr_path = artifact_dir / "dway.stderr.log"
    stderr_handle = None
    try:
        harness.start()
        report_root = Path(harness.read_status()["sim.report_dir"])
        harness.send("release_control")
        harness._lease_acquired = False
        harness.wait_status(lambda s: s.get("control.owner") == "",
                            description="released control lease")

        dway, stderr_handle = start_dway(harness, tour, stderr_path,
                                         "--timeout", "90")
        returncode = dway.wait(timeout=120.0)
        stderr_handle.flush()
        stderr = stderr_path.read_text(encoding="utf-8")
        assert returncode == 0, stderr

        summary = json.loads((report_root / "dway" / "summary.json").read_text())
        assert summary["outcome"] == "complete", summary["reason"]
        assert summary["strategy"] == "position"
        assert summary["waypoints_reached"] == 2
        assert summary["partial"] is False
        assert (report_root / "dway" / "flight.jsonl").exists()
        final = harness.read_status()
        assert float(final["drone.x_m"]) == pytest.approx(14.5, abs=0.3)
        assert float(final["drone.y_m"]) == pytest.approx(1.5, abs=0.3)
    finally:
        if dway is not None and dway.poll() is None:
            dway.kill()
            dway.wait(timeout=5.0)
        if stderr_handle is not None:
            stderr_handle.close()
        harness.close()
