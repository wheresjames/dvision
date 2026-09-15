"""Tests for dctl manual control mappings."""

import pytest
from types import SimpleNamespace

from dctl.dctl import (
    DroneController, JoystickManager, _MANUAL_YAW_RATE_DPS, _manual_yaw_rate,
    parse_args,
)


def test_manual_yaw_right_maps_to_positive_command_rate() -> None:
    assert _manual_yaw_rate(1.0) == _MANUAL_YAW_RATE_DPS


def test_manual_yaw_left_maps_to_negative_command_rate() -> None:
    assert _manual_yaw_rate(-1.0) == -_MANUAL_YAW_RATE_DPS


def test_manual_yaw_rate_clamps_normalized_input() -> None:
    assert _manual_yaw_rate(2.0) == _MANUAL_YAW_RATE_DPS
    assert _manual_yaw_rate(-2.0) == -_MANUAL_YAW_RATE_DPS


def _controller(owner: str, *, released: bool = False,
                lease_age: str = "", lease_timeout: str = "3.000",
                sent: list) -> DroneController:
    controller = DroneController.__new__(DroneController)
    controller.command = object()
    controller.status = type("Status", (), {"getAll": lambda self: {
        "control.owner": owner,
        "control.lease_age_s": lease_age,
        "control.lease_timeout_s": lease_timeout,
    }})()
    controller.control_source = "dctl-test"
    controller._last_heartbeat = 0.0
    controller._control_released = released
    controller._startup_take_control = False
    controller.held = set()
    controller._held_velocity_active = False
    controller.log = lambda message: None
    controller.send_command = lambda typ, **fields: sent.append(typ)
    return controller


def test_controller_observes_an_unowned_vehicle_by_default() -> None:
    sent: list[str] = []
    _controller("", sent=sent)._maintain_control()

    assert sent == []


def test_controller_does_not_reclaim_after_release() -> None:
    sent: list[str] = []
    _controller("", released=True, sent=sent)._maintain_control()

    assert sent == []


def test_controller_does_not_contend_with_another_owner() -> None:
    sent: list[str] = []
    _controller("dway-test", sent=sent)._maintain_control()

    assert sent == []


def test_cli_defaults_to_observer_and_supports_explicit_acquisition():
    assert not parse_args(['--id', 'area1']).take_control
    assert parse_args(['--id', 'area1', '--take-control']).take_control


@pytest.mark.parametrize('owner', ['', 'dway-test'])
def test_startup_acquisition_is_attempted_only_once(owner):
    sent = []
    controller = _controller(owner, released=True, sent=sent)
    controller._startup_take_control = True
    controller._maintain_control()
    assert sent == ([] if owner else ['acquire_control'])
    assert not controller._startup_take_control
    # A busy owner's later departure (or a failed acquisition) must not turn
    # this observer into a competing controller.
    controller.status.getAll = lambda: {'control.owner': ''}
    for _ in range(3): controller._maintain_control()
    assert sent == ([] if owner else ['acquire_control'])


def test_startup_waits_for_status_but_release_cancels_the_request():
    sent = []
    controller = _controller('', released=True, sent=sent)
    controller._startup_take_control = True
    controller.status.getAll = lambda: {}
    controller._maintain_control()
    assert controller._startup_take_control and not sent
    controller.release_control()
    controller.status.getAll = lambda: {'control.owner': ''}
    controller._maintain_control()
    assert not sent and not controller._startup_take_control


def test_release_stops_renewal_immediately_even_with_stale_owner_status():
    sent = []
    controller = _controller('dctl-test', sent=sent)
    controller.held.add('w')
    controller.release_control()
    controller._maintain_control()
    assert sent == ['release_control']
    assert not controller._owns_control() and not controller.held
    controller.status.getAll = lambda: {'control.owner': ''}
    controller.take_control()
    assert sent == ['release_control', 'acquire_control']


def test_lost_lease_does_not_trigger_reacquisition():
    sent = []
    controller = _controller('dctl-test', sent=sent)
    controller.status.getAll = lambda: {'control.owner': ''}
    controller._maintain_control()
    assert not sent and not controller._owns_control()


@pytest.mark.parametrize('command', ['velocity', 'zero', 'land', 'takeoff', 'arm', 'heartbeat'])
def test_observer_manual_commands_never_reach_the_wire(command):
    sent = []
    controller = _controller('dway-test', released=True, sent=sent)
    controller.command = SimpleNamespace(write=lambda payload: sent.append(payload) or True)
    DroneController.send_command(controller, command)
    assert not sent


def test_owned_manual_command_reaches_the_wire():
    sent = []
    controller = _controller('dctl-test', sent=sent)
    controller.control_lease = 'test-lease'
    controller.command = SimpleNamespace(write=lambda payload: sent.append(payload) or True)
    DroneController.send_command(controller, 'velocity', forward_mps=1.)
    assert len(sent) == 1


def test_observer_ignores_keyboard_and_joystick_without_queuing_motion():
    sent = []
    controller = _controller('', released=True, sent=sent)
    controller.key_down(SimpleNamespace(keysym='w', widget=None))
    assert not controller.held
    # No joystick polling or command dispatch is needed while observing.
    controller.send_held_velocity(force=True)
    controller._handle_joy_buttons()
    controller.key_up(SimpleNamespace(keysym='w'))
    assert not sent and not controller._held_velocity_active


def test_controller_heartbeats_only_its_own_lease(monkeypatch) -> None:
    sent: list[str] = []
    controller = _controller("dctl-test", sent=sent)
    monkeypatch.setattr("dctl.dctl.time.monotonic", lambda: 2.0)

    controller._maintain_control()

    assert sent == ["heartbeat"]


def test_controller_renews_early_when_the_lease_is_burning_fast(monkeypatch) -> None:
    # Accelerated simulation: the vehicle has already spent a third of a
    # three-second lease although well under a wall-clock second has passed.
    sent: list[str] = []
    controller = _controller("dctl-test", lease_age="1.200", sent=sent)
    monkeypatch.setattr("dctl.dctl.time.monotonic", lambda: 0.4)

    controller._maintain_control()

    assert sent == ["heartbeat"]


def test_controller_does_not_heartbeat_every_tick(monkeypatch) -> None:
    sent: list[str] = []
    controller = _controller("dctl-test", lease_age="2.900", sent=sent)
    monkeypatch.setattr("dctl.dctl.time.monotonic", lambda: 0.05)

    controller._maintain_control()

    assert sent == []


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
