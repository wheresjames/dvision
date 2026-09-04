from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

from dcmn.module_bus import PymembusModuleBus
from dtest.process_harness import DsimProcessHarness

ROOT = Path(__file__).resolve().parents[1]
MAP = ROOT / "assets/maps/maze_012.txt"
TOUR = ROOT / "assets/tours/maze_012.forward.v1.json"


def _only_summary(report_dir: Path) -> Path:
    """The one report DALG wrote. Runs get a directory each, so this asserts
    the session produced exactly one rather than assuming a fixed path."""
    found = sorted((report_dir / "dalg").glob("*/summary.json"))
    assert len(found) == 1, f"expected one dalg report, found {found}"
    return found[0]


def _wait(predicate, timeout=15.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value: return value
        time.sleep(.05)
    raise AssertionError("timed out")


def test_dalg_observes_protocol_role_without_taking_control(tmp_path):
    with DsimProcessHarness(tmp_path, map_path=MAP) as harness:
        bus = PymembusModuleBus(
            harness.id, "navigator", "dway2-test",
            sim_time=lambda: float(harness.read_status().get("sim.time_s", 0)))
        assert bus.connect()
        process = subprocess.Popen(
            [sys.executable, str(ROOT / "dalg/dalg.py"), "--id", harness.id,
             "--profile", "sgbm-default", "--no-ui", "--timeout", "30"],
            cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        run_id = "mock-navigator-run"
        tour = json.loads(TOUR.read_text(encoding="utf-8"))
        prepare = {
            "tour_id": tour["tour_id"],
            "tour_digest": hashlib.sha256(TOUR.read_bytes()).hexdigest(),
            "map_digest": tour["map_sha"], "coordinate_frame": "map",
            # DALG may observe a run even when the coordinator does not make
            # it a required readiness-barrier participant.
            "required_roles": [],
        }
        try:
            ready = None
            def prepare_until_ready():
                nonlocal ready
                bus.publish("run.prepare", run_id=run_id, payload=prepare)
                for event in bus.receive():
                    if event.type == "run.ready" and event.run_id == run_id:
                        ready = event
                return ready
            assert _wait(prepare_until_ready)
            start = float(harness.read_status()["sim.time_s"]) + .5
            bus.publish("run.start_scheduled", run_id=run_id,
                        payload={"start_sim_time_s": start})
            _wait(lambda: float(harness.read_status()["sim.time_s"]) > start + 1.0)
            bus.publish("run.completed", run_id=run_id, payload={
                "state": "COMPLETE", "outcome": "complete", "reason": "tour complete"})
            _wait(lambda: process.poll() is not None, timeout=30)
            assert process.returncode == 0, process.stderr.read()
            assert harness.read_status()["control.owner"] == harness.control_source
            summary_path = _only_summary(harness.report_dir)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            assert summary["run_id"] == run_id
            assert summary["provenance"]["navigator_implementation"] == "dway2-test"
            assert set(summary["scores"]) == {"sgbm", "constant", "exact_range"}
            assert (summary_path.parent / "overlay-sgbm.png").exists()
        finally:
            bus.close()
            if process.poll() is None:
                process.terminate(); process.wait(timeout=5)


def test_dalg_records_a_controller_coordinated_manual_flight(tmp_path):
    with DsimProcessHarness(tmp_path, map_path=MAP) as harness:
        bus = PymembusModuleBus(
            harness.id, "controller", "manual-test",
            sim_time=lambda: float(harness.read_status().get("sim.time_s", 0)))
        assert bus.connect()
        process = subprocess.Popen(
            [sys.executable, str(ROOT / "dalg/dalg.py"), "--id", harness.id,
             "--profile", "sgbm-manual", "--no-ui", "--timeout", "30"],
            cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        run_id = "manual-controller-run"
        prepare = {
            "tour_id": "", "tour_digest": "",
            "map_digest": hashlib.sha256(MAP.read_bytes()).hexdigest(),
            "coordinate_frame": "map", "flight_mode": "manual",
            "required_roles": ["algorithm:sgbm-manual"],
        }
        try:
            ready = None
            def prepare_until_ready():
                nonlocal ready
                bus.publish("run.prepare", run_id=run_id, payload=prepare)
                for event in bus.receive():
                    if event.type == "run.ready" and event.run_id == run_id:
                        ready = event
                return ready
            assert _wait(prepare_until_ready)
            start = float(harness.read_status()["sim.time_s"]) + .25
            bus.publish("run.start_scheduled", run_id=run_id,
                        payload={"start_sim_time_s": start})
            _wait(lambda: float(harness.read_status()["sim.time_s"]) > start + .5)
            bus.publish("run.completed", run_id=run_id, payload={
                "state": "COMPLETE", "outcome": "complete",
                "reason": "manual measurement stopped"})
            _wait(lambda: process.poll() is not None, timeout=30)
            assert process.returncode == 0, process.stderr.read()
            summary = json.loads(_only_summary(harness.report_dir).read_text())
            assert summary["profile"] == "sgbm-manual"
            assert summary["provenance"]["flight_mode"] == "manual"
            assert summary["provenance"]["coordinator_role"] == "controller"
            assert summary["provenance"]["tour_id"] is None
        finally:
            bus.close()
            if process.poll() is None:
                process.terminate(); process.wait(timeout=5)
