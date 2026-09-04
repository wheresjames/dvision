"""Guardrails for DAIC runtime navigation inputs."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


RUNTIME_MODULES = [
    "apps/daic/daic.py",
    "apps/daic/planner.py",
    "apps/daic/local_map.py",
    "apps/daic/avoidance.py",
    "apps/daic/optical_flow_avoidance.py",
    "apps/daic/mini_slam_detector.py",
    "apps/daic/orb_slam3_detector.py",
]


FORBIDDEN_RUNTIME_REFERENCES = [
    "load_map(",
    "\"sim.map\"",
    "'sim.map'",
    "assets/maps",
]


def test_daic_runtime_navigation_does_not_read_simulator_maps() -> None:
    """DAIC may use shared video and status, but not simulator map geometry."""
    violations: list[str] = []
    for rel in RUNTIME_MODULES:
        text = (ROOT / rel).read_text(encoding="utf-8")
        for forbidden in FORBIDDEN_RUNTIME_REFERENCES:
            if forbidden in text:
                violations.append(f"{rel}: {forbidden}")

    assert not violations, "forbidden runtime map access:\n" + "\n".join(violations)
