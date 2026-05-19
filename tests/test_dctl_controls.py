"""Tests for dctl manual control mappings."""

import pytest

from dctl.dctl import JoystickManager, _MANUAL_YAW_RATE_DPS, _manual_yaw_rate


def test_manual_yaw_right_maps_to_negative_command_rate() -> None:
    assert _manual_yaw_rate(1.0) == -_MANUAL_YAW_RATE_DPS


def test_manual_yaw_left_maps_to_positive_command_rate() -> None:
    assert _manual_yaw_rate(-1.0) == _MANUAL_YAW_RATE_DPS


def test_manual_yaw_rate_clamps_normalized_input() -> None:
    assert _manual_yaw_rate(2.0) == -_MANUAL_YAW_RATE_DPS
    assert _manual_yaw_rate(-2.0) == _MANUAL_YAW_RATE_DPS


def test_right_stick_x_reports_human_facing_yaw_right() -> None:
    joy = JoystickManager.__new__(JoystickManager)
    joy._axes = [0.0, 0.0, 0.0, 0.5]

    assert joy.yaw == pytest.approx(0.5)
    assert _manual_yaw_rate(joy.yaw) == pytest.approx(-22.5)
