"""Tests across the real DSIM/DAIC command, status, and video transports."""

import contextlib
import json
import os
import signal
import subprocess
import sys
import time
import shutil

import numpy as np
import pytest

from daic.daic import (
    _annotate,
    _client_rgb_frame as daic_rgb_frame,
    _display_rgb_frame as daic_display_frame,
)
from daic.detector import detect
from dctl.dctl import _client_rgb_frame as dctl_rgb_frame, format_status_value
from dsim.dsim import TopDownUi
from dtest.color_probe import color_centroid
from dtest.backend import VehicleBackend
from dtest.artifacts import artifact_directory
from dtest.conformance import run_conformance
from dtest.contract import EXPECTED_COMMAND_SIGN, circular_delta_deg
from dtest.process_harness import DsimProcessHarness
from dtest.calibration_scene import DIRECT_MAP, ROOT
from dvision2_common import decode_command, encode_command


def test_real_dsim_ipc_preserves_control_heading_and_video_contract(tmp_path) -> None:
    artifact_dir = artifact_directory(tmp_path, "dsim-ipc")
    harness = DsimProcessHarness(artifact_dir)
    names = dict(harness.names)
    pm = harness.pm

    with harness:
        assert isinstance(harness, VehicleBackend)
        initial = harness.read_status()
        initial_epoch = harness.status.getEpoch()
        seq0, frame0 = harness.read_frame()

        assert frame0.shape == (480, 640, 3)
        assert np.array_equal(dctl_rgb_frame(frame0), frame0)
        assert np.array_equal(daic_rgb_frame(frame0), frame0)
        blue0 = color_centroid(frame0, "blue")
        assert blue0.x > frame0.shape[1] / 2

        harness.send("arm", armed=True)
        harness.wait_status(
            lambda s: s.get("drone.armed") == "1",
            description="armed state",
        )
        harness.send("velocity", forward_mps=0.0, right_mps=0.0,
                     up_mps=0.0, yaw_rate_dps=30.0)
        yawed = harness.wait_status(
            lambda s: circular_delta_deg(
                float(s["drone.heading_deg"]), float(initial["drone.heading_deg"])
            ) > 8.0,
            description="positive compass-heading response",
        )
        harness.send("zero")
        seq1, frame1 = harness.read_frame(newer_than=seq0)
        assert seq1 > seq0
        assert color_centroid(frame1, "blue").x < blue0.x - 10.0
        assert int(yawed["link.command_count"]) >= 2
        assert harness.status.getEpoch() > initial_epoch

        harness.send("reset")
        reset = harness.wait_status(
            lambda s: s.get("status.message") == "reset",
            description="reset state",
        )
        harness.send("arm", armed=True)
        harness.wait_status(lambda s: s.get("drone.armed") == "1",
                            description="re-armed state")
        y0 = float(reset["drone.y_m"])
        harness.send("velocity", forward_mps=0.5, right_mps=0.0,
                     up_mps=0.0, yaw_rate_dps=0.0)
        moved = harness.wait_status(
            lambda s: float(s["drone.y_m"]) < y0 - 0.05,
            description="northward forward displacement",
        )
        harness.send("zero")
        assert float(moved["drone.vy_mps"]) < 0.0

    video = pm.memvid()
    command = pm.memcmd()
    status = pm.memkv()
    assert not video.open_existing(names["video"])
    assert not command.open(names["command"], 65536)
    assert not status.open(names["status"])


def test_running_daic_connects_and_drives_real_dsim_transport(tmp_path) -> None:
    artifact_dir = artifact_directory(tmp_path, "daic-process")
    direct_map = ROOT / "assets/maps/test_direct.txt"
    harness = DsimProcessHarness(artifact_dir, map_path=direct_map)
    daic = None
    stderr_path = artifact_dir / "daic.stderr.log"
    stderr_fh = None
    try:
        harness.start()
        initial = harness.read_status()
        harness.send("release_control")
        harness._lease_acquired = False
        stderr_fh = stderr_path.open("w", encoding="utf-8")
        daic = subprocess.Popen(
            [
                sys.executable, str(ROOT / "apps/daic/daic.py"),
                "--id", harness.id,
                "--no-ui",
                "--enable-ai",
                "--fps", "30",
            ],
            cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=stderr_fh,
            start_new_session=True,
        )
        active = harness.wait_status(
            lambda s: (
                s.get("drone.armed") == "1"
                and int(s.get("link.command_count", "0")) >= 3
            ),
            timeout=10.0,
            description="DAIC arming and command stream",
        )
        assert daic.poll() is None
        assert int(active["link.command_count"]) > int(initial["link.command_count"])
        assert active["link.last_command_type"] in {"arm", "zero", "velocity", "heartbeat"}

        log_path = harness.report_dir / "daic/flight.jsonl"
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and (
            not log_path.exists() or log_path.stat().st_size == 0
        ):
            time.sleep(0.05)
        assert log_path.exists() and log_path.stat().st_size > 0
    except Exception as exc:
        harness.save_failure(component="daic", error=repr(exc))
        raise
    finally:
        if daic is not None and daic.poll() is None:
            daic.terminate()
            try:
                daic.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                daic.kill()
                daic.wait(timeout=5.0)
        if stderr_fh is not None:
            stderr_fh.close()
        harness.close()


def test_dctl_gui_starts_against_live_backend_under_virtual_display(tmp_path) -> None:
    xvfb_run = shutil.which("xvfb-run")
    if xvfb_run is None:
        pytest.skip("xvfb-run is required for the GUI smoke test")

    artifact_dir = artifact_directory(tmp_path, "dctl-gui")
    harness = DsimProcessHarness(artifact_dir)
    dctl = None
    stderr_path = artifact_dir / "dctl.stderr.log"
    stderr_fh = None
    try:
        harness.start()
        stderr_fh = stderr_path.open("w", encoding="utf-8")
        dctl = subprocess.Popen(
            [
                xvfb_run, "-a", sys.executable, str(ROOT / "apps/dctl/dctl.py"),
                "--id", harness.id,
                "--no-joystick",
                "--fps", "20",
            ],
            cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=stderr_fh,
            # Its own session, so the pid is a process-group leader and the
            # teardown below can take down xvfb-run, its Xvfb, and the client
            # in one signal. Without this, killpg raises ESRCH and the whole
            # tree -- simulator included -- is orphaned.
            start_new_session=True,
        )
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and dctl.poll() is None:
            # A live status epoch proves the backend remains healthy while the
            # actual Tk client opens and consumes it.
            if harness.read_status().get("sim.id") == harness.id:
                time.sleep(0.5)
                break
            time.sleep(0.05)
        if dctl.poll() is not None:
            stderr_fh.flush()
            error = stderr_path.read_text(encoding="utf-8")
            if "couldn't connect to display" in error:
                pytest.skip("virtual display is unavailable in this environment")
            pytest.fail(f"DCTL GUI exited early:\n{error[-2000:]}")
    except Exception as exc:
        harness.save_failure(component="dctl", error=repr(exc))
        raise
    finally:
        if dctl is not None and dctl.poll() is None:
            # The group can disappear between the poll and the signal; that is
            # the outcome this wants anyway, and it must not stop the
            # simulator from being shut down below.
            try:
                os.killpg(dctl.pid, signal.SIGTERM)
                dctl.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(dctl.pid, signal.SIGKILL)
                dctl.wait(timeout=5.0)
            except ProcessLookupError:
                pass
        if stderr_fh is not None:
            stderr_fh.close()
        harness.close()


def test_dsim_process_passes_the_backend_conformance_suite(tmp_path) -> None:
    """The real transport path is the reference conformance run for backends."""
    artifact_dir = artifact_directory(tmp_path, "dsim-conformance")
    with DsimProcessHarness(artifact_dir) as harness:
        summary = run_conformance(harness, harness.settle)
    assert summary["frame_shape"] == [480, 640, 3]
    assert summary["capabilities"]["physical_vehicle"] is False


def test_wire_command_semantics_survive_the_real_transport(tmp_path) -> None:
    """Semantic action -> encoded command -> decoded command -> observed motion."""
    for action, (field, sign) in EXPECTED_COMMAND_SIGN.items():
        raw = encode_command("velocity", **{field: sign * 1.0})
        payload = decode_command(raw)
        assert payload is not None, f"{action} did not survive encode/decode"
        assert payload[field] * sign > 0.0, (
            f"semantic {action} encoded {field}={payload[field]}"
        )

    artifact_dir = artifact_directory(tmp_path, "wire-semantics")
    with DsimProcessHarness(artifact_dir) as harness:
        harness.arm()
        harness.wait_status(lambda s: s.get("drone.armed") == "1",
                            description="armed state")
        before = float(harness.read_status()["drone.heading_deg"])
        harness.send("velocity", yaw_rate_dps=30.0)
        yawed = harness.wait_status(
            lambda s: circular_delta_deg(float(s["drone.heading_deg"]), before) > 8.0,
            description="yaw-right increasing compass heading over the real bus",
        )
        harness.zero()
        assert yawed["link.last_command_type"] == "velocity"


def test_both_clients_report_the_same_compass_heading(tmp_path) -> None:
    """Correct telemetry paired with a wrong label is still a defect."""
    artifact_dir = artifact_directory(tmp_path, "shared-heading")
    with DsimProcessHarness(artifact_dir, start_heading=137.0) as harness:
        status = harness.wait_status(
            lambda s: s.get("drone.heading_deg") not in (None, ""),
            description="published heading",
        )
        published = status["drone.heading_deg"]
        assert status["drone.compass_deg"] == published

        # DCTL formats the same published value it receives from the bus.
        assert format_status_value("drone.heading_deg", published) == (
            f"{float(published):.1f} deg"
        )
        # DSIM's own status line shows that value, not the internal renderer yaw.
        harness_heading = float(published)
        assert f"heading={harness_heading:.1f}" in TopDownUi.status_text(
            _state_at_heading(harness_heading)
        )
        assert harness_heading == pytest.approx(137.0, abs=0.5)


def _state_at_heading(heading_deg: float):
    from dsim.dsim import DroneState, compass_heading_to_sim_yaw

    return DroneState(0.0, 0.0, 1.0, yaw_deg=compass_heading_to_sim_yaw(heading_deg))


def test_live_daic_detections_and_overlays_refer_to_displayed_pixels(tmp_path) -> None:
    """Both clients read the same live frame; the overlay lands on the target."""
    artifact_dir = artifact_directory(tmp_path, "live-overlay")
    with DsimProcessHarness(artifact_dir, map_path=DIRECT_MAP) as harness:
        # Close to within the detector's usable range over the real command bus.
        harness.arm()
        harness.wait_status(lambda s: s.get("drone.armed") == "1",
                            description="armed state")
        harness.send_body_velocity(1.0, 0.0, 0.0, 0.0)
        harness.wait_status(
            lambda s: float(s["drone.y_m"]) <= 13.0,
            timeout=15.0,
            description="approach to the landing target",
        )
        harness.zero()
        harness.settle(0.5)

        _, frame = harness.read_frame()
        vision = daic_rgb_frame(frame)
        display = daic_display_frame(frame)
        assert np.array_equal(vision, dctl_rgb_frame(frame))
        assert np.array_equal(vision, display)

        detection = detect(vision)
        assert detection.visible, "no landing target in the live approach frame"
        annotated = _annotate(display, detection)
        assert annotated.shape == frame.shape
        assert not np.array_equal(annotated, display)

        redetected = detect(annotated)
        assert redetected.visible
        assert redetected.cx == pytest.approx(detection.cx, abs=6.0), (
            f"overlay moved the target from x={detection.cx:.1f} to "
            f"x={redetected.cx:.1f}; annotation is not aligned with the frame"
        )


def test_process_failure_writes_a_locatable_artifact_bundle(tmp_path) -> None:
    """A failing process test must leave enough evidence to find the boundary."""
    artifact_dir = artifact_directory(tmp_path, "artifact-bundle")
    with pytest.raises(AssertionError):
        with DsimProcessHarness(artifact_dir) as harness:
            harness.arm()
            harness.wait_status(lambda s: s.get("drone.armed") == "1",
                                description="armed state")
            harness.send_body_velocity(0.6, 0.0, 0.0, 0.0)
            harness.wait_status(
                lambda s: float(s["drone.y_m"]) < 11.0,
                description="northward displacement",
            )
            harness.read_frame()
            raise AssertionError("deliberate failure to exercise the bundle")

    bundle = json.loads((artifact_dir / "result.json").read_text(encoding="utf-8"))
    assert bundle["details"]["instance_id"] == harness.id
    assert bundle["fixture"]["camera"]["camera.width_px"] == "640"
    assert bundle["telemetry"]["drone.heading_deg"]
    for name in ("initial_frame.png", "final_frame.png", "path.png",
                 "commands.jsonl", "timeline.jsonl"):
        assert (artifact_dir / name).stat().st_size > 0, f"missing artifact {name}"

    commands = [json.loads(line) for line in
                (artifact_dir / "commands.jsonl").read_text().splitlines()]
    assert any(c["type"] == "velocity" and c["forward_mps"] > 0 for c in commands)
    timeline = [json.loads(line) for line in
                (artifact_dir / "timeline.jsonl").read_text().splitlines()]
    assert len(timeline) >= 2
    assert timeline[-1]["frame_seq"] is not None
    assert timeline[-1]["status_epoch"] >= timeline[0]["status_epoch"]
