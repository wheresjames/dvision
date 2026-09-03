"""Tests for dctl manual control mappings."""

import pytest

from dctl.dctl import (
    DroneController, JoystickManager, _MANUAL_YAW_RATE_DPS, _manual_yaw_rate,
)


def test_manual_yaw_right_maps_to_positive_command_rate() -> None:
    assert _manual_yaw_rate(1.0) == _MANUAL_YAW_RATE_DPS


def test_manual_yaw_left_maps_to_negative_command_rate() -> None:
    assert _manual_yaw_rate(-1.0) == -_MANUAL_YAW_RATE_DPS


def test_manual_yaw_rate_clamps_normalized_input() -> None:
    assert _manual_yaw_rate(2.0) == _MANUAL_YAW_RATE_DPS
    assert _manual_yaw_rate(-2.0) == -_MANUAL_YAW_RATE_DPS


def test_controller_starts_as_observer_when_vehicle_is_unowned() -> None:
    controller = DroneController.__new__(DroneController)
    controller.command = object()
    controller.status = type("Status", (), {"getAll": lambda self: {
        "control.owner": ""
    }})()
    controller.control_source = "dctl-test"
    controller._last_heartbeat = 0.0
    sent = []
    controller.send_command = lambda typ, **fields: sent.append(typ)

    controller._maintain_control()

    assert sent == []


def test_controller_does_not_contend_with_another_owner() -> None:
    controller = DroneController.__new__(DroneController)
    controller.command = object()
    controller.status = type("Status", (), {"getAll": lambda self: {
        "control.owner": "dway-test"
    }})()
    controller.control_source = "dctl-test"
    controller._last_heartbeat = 0.0
    sent = []
    controller.send_command = lambda typ, **fields: sent.append(typ)

    controller._maintain_control()

    assert sent == []


def test_controller_heartbeats_only_its_own_lease(monkeypatch) -> None:
    controller = DroneController.__new__(DroneController)
    controller.command = object()
    controller.status = type("Status", (), {"getAll": lambda self: {
        "control.owner": "dctl-test"
    }})()
    controller.control_source = "dctl-test"
    controller._last_heartbeat = 0.0
    sent = []
    controller.send_command = lambda typ, **fields: sent.append(typ)
    monkeypatch.setattr("dctl.dctl.time.monotonic", lambda: 2.0)

    controller._maintain_control()

    assert sent == ["heartbeat"]


def test_right_stick_x_reports_human_facing_yaw_right() -> None:
    joy = JoystickManager.__new__(JoystickManager)
    joy._axes = [0.0, 0.0, 0.0, 0.5]

    assert joy.yaw == pytest.approx(0.5)
    assert _manual_yaw_rate(joy.yaw) == pytest.approx(22.5)


# Documented in the README keyboard table: key -> (axis index, expected sign).
# 0 = forward, 1 = right, 2 = up, 3 = yaw-right.
_DOCUMENTED_KEYS = {
    "w": (0, 1.0), "Up": (0, 1.0),
    "s": (0, -1.0), "Down": (0, -1.0),
    "d": (1, 1.0), "Right": (1, 1.0),
    "a": (1, -1.0), "Left": (1, -1.0),
    "r": (2, 1.0), "Prior": (2, 1.0), "Page_Up": (2, 1.0),
    "f": (2, -1.0), "Next": (2, -1.0), "Page_Down": (2, -1.0),
    "e": (3, 1.0), "End": (3, 1.0),
    "q": (3, -1.0), "Home": (3, -1.0),
}


@pytest.mark.parametrize("keysym", sorted(_DOCUMENTED_KEYS))
def test_each_documented_key_drives_only_its_documented_axis(keysym: str) -> None:
    from dctl.dctl import _control_key, _held_axes

    index, sign = _DOCUMENTED_KEYS[keysym]
    axes = _held_axes({_control_key(keysym)})
    assert axes[index] == sign, (
        f"key {keysym!r} produced axes {axes}, expected axis {index} = {sign:+.0f}"
    )
    for other, value in enumerate(axes):
        if other != index:
            assert value == 0.0, f"key {keysym!r} also moved axis {other}"


def test_no_keys_held_produces_no_motion() -> None:
    from dctl.dctl import _held_axes

    assert _held_axes(set()) == (0.0, 0.0, 0.0, 0.0)


# Documented in the JoystickManager docstring: axis index -> (property, sign
# for a positive raw reading). Xbox sticks report Y down-positive, so the two
# vertical axes are negated on the way out; the table records the *resulting*
# semantic sign, not the raw one.
_DOCUMENTED_AXES = {
    0: ("right", +1.0),     # left-stick X, right = +1 -> strafe right
    1: ("forward", -1.0),   # left-stick Y, down  = +1 -> negate for forward
    3: ("yaw", +1.0),       # right-stick X, right = +1 -> yaw right
    4: ("up", -1.0),        # right-stick Y, down  = +1 -> negate for ascend
}

_SEMANTIC_AXES = ("forward", "right", "up", "yaw")


def _joystick_with(index: int, value: float) -> JoystickManager:
    joy = JoystickManager.__new__(JoystickManager)
    joy._axes = [0.0] * 6
    joy._axes[index] = value
    return joy


@pytest.mark.parametrize("index", sorted(_DOCUMENTED_AXES))
def test_each_documented_stick_axis_drives_only_its_documented_control(index) -> None:
    name, sign = _DOCUMENTED_AXES[index]
    joy = _joystick_with(index, 1.0)

    assert getattr(joy, name) == pytest.approx(sign), (
        f"axis {index} at +1.0 gave {name}={getattr(joy, name)}, expected "
        f"{sign:+.0f}; the stick mapping is inverted or crossed")
    for other in _SEMANTIC_AXES:
        if other != name:
            assert getattr(joy, other) == 0.0, (
                f"axis {index} also moved {other}")


@pytest.mark.parametrize("index", sorted(_DOCUMENTED_AXES))
def test_each_documented_stick_axis_reverses_with_its_input(index) -> None:
    name, sign = _DOCUMENTED_AXES[index]

    assert getattr(_joystick_with(index, -1.0), name) == pytest.approx(-sign)


def test_stick_deadzone_suppresses_small_readings_on_every_axis() -> None:
    for index, (name, _) in _DOCUMENTED_AXES.items():
        joy = _joystick_with(index, JoystickManager.DEADZONE * 0.5)
        assert getattr(joy, name) == 0.0, (
            f"axis {index} passed a sub-deadzone reading through to {name}")


def test_missing_axes_report_neutral_rather_than_raising() -> None:
    joy = JoystickManager.__new__(JoystickManager)
    joy._axes = []

    assert (joy.forward, joy.right, joy.up, joy.yaw) == (0.0, 0.0, 0.0, 0.0)


def test_stick_yaw_right_reaches_the_wire_as_a_positive_rate() -> None:
    """The joystick half of the same chain the keyboard test covers."""
    joy = _joystick_with(3, 1.0)

    assert _manual_yaw_rate(joy.yaw) == pytest.approx(_MANUAL_YAW_RATE_DPS)
