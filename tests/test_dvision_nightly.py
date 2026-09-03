"""Opt-in longer process calibration checks.

Run with ``DVISION_NIGHTLY=1 pytest -m nightly -q``.
"""

import os
import time

import pytest

from dtest.artifacts import artifact_directory
from dtest.contract import circular_delta_deg
from dtest.process_harness import DsimProcessHarness


pytestmark = pytest.mark.nightly


@pytest.mark.skipif(os.environ.get("DVISION_NIGHTLY") != "1",
                    reason="set DVISION_NIGHTLY=1 to run endurance coverage")
def test_longer_live_calibration_stream_remains_consistent(tmp_path) -> None:
    duration = max(5.0, float(os.environ.get("DVISION_NIGHTLY_SECONDS", "20")))
    artifacts = artifact_directory(tmp_path, "nightly-calibration")
    with DsimProcessHarness(artifacts) as harness:
        initial = harness.read_status()
        seq0, _ = harness.read_frame()
        harness.arm()
        harness.wait_status(lambda s: s.get("drone.armed") == "1",
                            description="nightly armed state")

        deadline = time.monotonic() + duration
        direction = 1.0
        samples = 0
        while time.monotonic() < deadline:
            before = harness.read_status()
            harness.send_body_velocity(0.0, 0.0, 0.0, direction * 20.0)
            changed = harness.wait_status(
                lambda s: abs(circular_delta_deg(
                    float(s["drone.heading_deg"]),
                    float(before["drone.heading_deg"]),
                )) >= 3.0,
                timeout=3.0,
                description="nightly alternating yaw response",
            )
            delta = circular_delta_deg(
                float(changed["drone.heading_deg"]),
                float(before["drone.heading_deg"]),
            )
            assert delta * direction > 0.0
            direction *= -1.0
            samples += 1

        harness.zero()
        seq1, frame = harness.read_frame(newer_than=seq0)
        assert seq1 > seq0
        assert frame.shape == (480, 640, 3)
        assert samples >= 2
        assert int(harness.read_status()["link.command_count"]) > int(
            initial["link.command_count"]
        )
