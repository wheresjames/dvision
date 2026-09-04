"""The same flight, at two speeds, must produce the same report.

This is the safety criterion made executable: a module is fast-forward safe
only if every timer that affects its output reads simulated time. Fly one tour
against a real simulator at real-time pacing and again at four times that, and
the flight the report describes has to be the same flight. A timer that slipped
back onto the wall clock changes the answer here and nowhere else in the suite.

Two real processes, because the property is about what they publish to each
other -- an in-process rig shares a clock and so cannot fail this test.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

import os

import pytest

from dvision2_common import load_pymembus, shared_names

# Two complete flights of a committed tour, one of them at real-time pacing, so
# this costs minutes rather than seconds. It is opted into the same way the
# other long-running coverage is, and gates a nightly rather than every commit.
pytestmark = [
    pytest.mark.nightly,
    pytest.mark.skipif(os.environ.get("DVISION_NIGHTLY") != "1",
                       reason="set DVISION_NIGHTLY=1 to fly the conformance pair"),
]

ROOT = Path(__file__).resolve().parents[1]
MAP = ROOT / "assets/maps/maze_001.txt"
TOUR = ROOT / "assets/tours/maze_001.default.v1.json"


def _await_simulator(instance_id: str, timeout: float = 40.0) -> None:
    pm = load_pymembus()
    name = shared_names(instance_id)["status"]
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        handle = pm.memkv()
        if handle.open(name):
            values = handle.getAll()
            handle.close()
            if values.get("sim.map"):
                return
        time.sleep(0.05)
    raise AssertionError("the simulator never published a status area")


def _fly(tmp_path: Path, sim_speed: str) -> dict:
    """One complete tour, and the summary dway wrote for it."""
    instance = f"speed-{uuid.uuid4().hex[:10]}"
    report_dir = tmp_path / f"run-{sim_speed}"
    simulator = subprocess.Popen(
        [sys.executable, str(ROOT / "apps/dsim/dsim.py"), "--id", instance,
         "--map", str(MAP), "--no-ui", "--report-dir", str(report_dir),
         "--sim-speed", sim_speed, "--video-hz", "5"],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        _await_simulator(instance)
        navigator = subprocess.Popen(
            [sys.executable, str(ROOT / "apps/dway/dway.py"), "--id", instance,
             "--tour", str(TOUR), "--no-ui", "--exit-on-finish"],
            cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            text=True)
        try:
            _, errors = navigator.communicate(timeout=300)
        except subprocess.TimeoutExpired:
            navigator.kill()
            _, errors = navigator.communicate()
        assert navigator.returncode == 0, errors[-800:]
    finally:
        simulator.terminate()
        try:
            simulator.wait(timeout=20)
        except subprocess.TimeoutExpired:
            simulator.kill()
    return json.loads((report_dir / "dway" / "summary.json").read_text())


def test_a_tour_flies_the_same_at_one_and_four_times_real_time(tmp_path):
    with tempfile.TemporaryDirectory() as scratch:
        scratch = Path(scratch)
        paced = _fly(scratch, "1")
        fast = _fly(scratch, "4")

    assert paced["outcome"] == fast["outcome"] == "complete"
    assert paced["waypoints_reached"] == fast["waypoints_reached"] \
        == paced["waypoint_count"]
    assert fast["failsafes"] == paced["failsafes"] == []

    # The flight itself: same path, same accuracy, same duration in the only
    # units that describe a flight rather than a machine.
    assert fast["path_length_m"] == pytest.approx(paced["path_length_m"], rel=0.02)
    assert fast["duration_s"] == pytest.approx(paced["duration_s"], rel=0.02)
    assert fast["max_cross_track_error_m"] == pytest.approx(
        paced["max_cross_track_error_m"], abs=0.05)

    # Per-waypoint arrivals carry a skew the aggregate figures do not. The
    # navigator polls on its own loop while simulated time runs four times
    # faster, so it samples the vehicle at four times coarser simulated
    # granularity and notices arrival a fraction of a second later. That is
    # the documented cost of running without a tick barrier, and it is
    # bounded: a timer that had slipped back onto the wall clock would move
    # these by the speed factor -- tens of seconds -- not by a fraction of one.
    for slow_wp, fast_wp in zip(paced["waypoints"], fast["waypoints"]):
        assert fast_wp["arrival_s"] == pytest.approx(slow_wp["arrival_s"], abs=2.0)
        assert fast_wp["overshoot_m"] == pytest.approx(
            slow_wp["overshoot_m"], abs=0.05)


def test_the_fast_run_actually_took_less_wall_time(tmp_path):
    """Guards the test above from passing because nothing was scaled at all."""
    with tempfile.TemporaryDirectory() as scratch:
        scratch = Path(scratch)
        started = time.monotonic()
        _fly(scratch, "1")
        paced_wall = time.monotonic() - started
        started = time.monotonic()
        _fly(scratch, "4")
        fast_wall = time.monotonic() - started

    assert fast_wall < paced_wall * 0.75, (
        f"scaled run took {fast_wall:.1f}s against {paced_wall:.1f}s paced")
