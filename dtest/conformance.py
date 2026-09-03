"""Backend-neutral conformance checks.

Any object implementing the normalized ``VehicleBackend`` contract must pass
these. They are written against the public surface only, so the same suite will
run unchanged against the in-process deterministic simulator, the real DSIM
process over its transports, and a future MAVLink backend.

Every check raises ``AssertionError`` naming the violated boundary. Waiting is
the caller's problem: a backend that cannot respond instantly supplies a
``settle`` callable that advances it by roughly the requested seconds.
"""

from __future__ import annotations

import inspect
from typing import Callable

import numpy as np

from dtest.backend import BackendCapabilities, VehicleBackend
from dtest.contract import circular_delta_deg

REQUIRED_METHODS = (
    "capabilities", "arm", "disarm", "zero",
    "send_body_velocity", "read_telemetry", "read_frame",
)

REQUIRED_TELEMETRY_KEYS = (
    "drone.x_m", "drone.y_m", "drone.z_m",
    "drone.heading_deg", "drone.compass_deg", "drone.armed",
)

BODY_VELOCITY_PARAMETERS = ("forward_mps", "right_mps", "up_mps", "yaw_rate_dps")


def assert_backend_surface(backend: object) -> None:
    """Structural conformance: methods, signature, and capability reporting.

    ``runtime_checkable`` protocols only compare method *names*, so the
    body-velocity signature is checked explicitly. A backend that renamed or
    reordered those arguments would otherwise silently reinterpret commands.
    """
    assert isinstance(backend, VehicleBackend), (
        f"{type(backend).__name__} does not implement the VehicleBackend protocol"
    )
    for name in REQUIRED_METHODS:
        assert callable(getattr(backend, name, None)), (
            f"{type(backend).__name__} is missing backend method {name!r}"
        )
    params = list(
        inspect.signature(backend.send_body_velocity).parameters
    )
    assert params[:4] == list(BODY_VELOCITY_PARAMETERS), (
        f"{type(backend).__name__}.send_body_velocity takes {params[:4]}, "
        f"expected {list(BODY_VELOCITY_PARAMETERS)}; a backend may not reorder "
        "or rename the normalized body axes"
    )
    caps = backend.capabilities()
    assert isinstance(caps, BackendCapabilities), (
        f"{type(backend).__name__}.capabilities() returned {type(caps).__name__}, "
        "expected BackendCapabilities"
    )


def assert_telemetry_contract(backend: object) -> dict:
    """Telemetry publishes the required normalized keys with usable values."""
    telemetry = backend.read_telemetry()
    assert isinstance(telemetry, dict), (
        f"read_telemetry() returned {type(telemetry).__name__}, expected dict"
    )
    for key in REQUIRED_TELEMETRY_KEYS:
        assert key in telemetry, f"telemetry is missing required key {key!r}"
    heading = float(telemetry["drone.heading_deg"])
    compass = float(telemetry["drone.compass_deg"])
    assert 0.0 <= heading < 360.0, (
        f"drone.heading_deg={heading} is not normalized to [0, 360)"
    )
    assert abs(circular_delta_deg(compass, heading)) < 1e-6, (
        "drone.compass_deg and drone.heading_deg must be the same public "
        f"compass value, got {compass} and {heading}"
    )
    return telemetry


def assert_frame_contract(backend: object) -> np.ndarray:
    """Video is an RGB24, top-left-origin, HxWx3 uint8 array."""
    seq, frame = backend.read_frame()
    assert isinstance(seq, int), f"frame sequence {seq!r} is not an int"
    assert isinstance(frame, np.ndarray), (
        f"read_frame() returned {type(frame).__name__}, expected numpy.ndarray"
    )
    assert frame.dtype == np.uint8, f"frame dtype {frame.dtype} is not uint8"
    assert frame.ndim == 3 and frame.shape[2] == 3, (
        f"frame shape {frame.shape} is not HxWx3 RGB24"
    )
    return frame


def assert_yaw_polarity(backend: object, settle: Callable[[float], None],
                        *, rate_dps: float = 30.0, seconds: float = 1.0,
                        minimum_deg: float = 5.0) -> None:
    """Positive wire yaw rate must increase the published compass heading."""
    from dtest.assertions import assert_heading_change

    backend.arm()
    settle(0.2)
    for direction, sign in (("right", 1.0), ("left", -1.0)):
        backend.zero()
        settle(0.4)
        before = float(backend.read_telemetry()["drone.heading_deg"])
        backend.send_body_velocity(0.0, 0.0, 0.0, sign * rate_dps)
        settle(seconds)
        after = float(backend.read_telemetry()["drone.heading_deg"])
        backend.zero()
        assert_heading_change(after, before, direction, minimum_deg=minimum_deg)


def run_conformance(backend: object, settle: Callable[[float], None]) -> dict:
    """Run every backend-neutral check and return a small result summary."""
    assert_backend_surface(backend)
    telemetry = assert_telemetry_contract(backend)
    frame = assert_frame_contract(backend)
    assert_yaw_polarity(backend, settle)
    return {
        "backend": type(backend).__name__,
        "capabilities": vars(backend.capabilities()),
        "frame_shape": list(frame.shape),
        "initial_heading_deg": float(telemetry["drone.heading_deg"]),
    }
