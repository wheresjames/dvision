"""Deterministic control, heading, and UI-semantics contract tests.

Everything here runs on a fixed simulated timestep. Nothing sleeps or depends
on wall-clock scheduling, and no expectation is produced by a production
coordinate transform.
"""

import math

import pytest

from dctl.dctl import (
    _MANUAL_YAW_RATE_DPS,
    _manual_yaw_rate,
    format_status_value,
)
from dsim.dsim import (
    TopDownUi,
    compass_heading_to_sim_yaw,
    sim_yaw_to_compass_heading,
)
from dsim.dsim import DroneState
from dtest.assertions import assert_heading_change
from dtest.contract import (
    CARDINAL_HEADINGS,
    EXPECTED_BACKWARD,
    EXPECTED_COMMAND_SIGN,
    EXPECTED_FORWARD,
    EXPECTED_LEFT,
    EXPECTED_RIGHT,
    assert_direction,
    assert_stationary,
    circular_delta_deg,
)
from dtest.deterministic import DeterministicSim
from dtest.faults import COMMAND_FAULTS, invert_strafe, invert_yaw
from dtest.preflight import DETERMINISTIC_MODULES, missing_dependencies
from dvision2_common import decode_command, encode_command


# ---------------------------------------------------------------------------
# Control impulse matrix
# ---------------------------------------------------------------------------

_HORIZONTAL_AXES = {
    "forward": ({"forward_mps": 0.5}, EXPECTED_FORWARD),
    "backward": ({"forward_mps": -0.5}, EXPECTED_BACKWARD),
    "right": ({"right_mps": 0.5}, EXPECTED_RIGHT),
    "left": ({"right_mps": -0.5}, EXPECTED_LEFT),
}


@pytest.mark.parametrize("heading", CARDINAL_HEADINGS)
@pytest.mark.parametrize("axis", sorted(_HORIZONTAL_AXES))
def test_horizontal_impulse_matches_literal_cardinal_contract(
    axis: str, heading: int,
) -> None:
    """Each body axis moves along its literal map direction, and no other."""
    command, expected = _HORIZONTAL_AXES[axis]
    sim = DeterministicSim(heading_deg=heading)
    dx, dy = sim.impulse(1.0, settle=0.2, **command)
    assert_direction(dx, dy, expected[heading])


@pytest.mark.parametrize("heading", CARDINAL_HEADINGS)
def test_horizontal_impulse_leaves_heading_unchanged(heading: int) -> None:
    """A pure strafe must not also yaw; crossed axes would show up here."""
    sim = DeterministicSim(heading_deg=heading)
    before = sim.heading_deg
    sim.impulse(1.0, settle=0.2, right_mps=0.5)
    assert abs(circular_delta_deg(sim.heading_deg, before)) < 0.5, (
        f"right strafe at heading {heading} changed compass heading by "
        f"{circular_delta_deg(sim.heading_deg, before):+.3f} deg"
    )


@pytest.mark.parametrize("heading", CARDINAL_HEADINGS)
@pytest.mark.parametrize("axis,sign", [("up", 1.0), ("down", -1.0)])
def test_vertical_impulse_is_isolated_from_horizontal_motion(
    heading: int, axis: str, sign: float,
) -> None:
    sim = DeterministicSim(heading_deg=heading, altitude_m=8.0)
    z0 = sim.state.z
    dx, dy = sim.impulse(1.0, settle=0.2, up_mps=sign * 0.8)
    dz = sim.state.z - z0
    assert dz * sign > 0.25, (
        f"{axis} command produced dz={dz:+.3f} m at heading {heading}"
    )
    assert_stationary(dx, dy)


@pytest.mark.parametrize("heading", CARDINAL_HEADINGS)
def test_zero_command_holds_position_and_heading(heading: int) -> None:
    sim = DeterministicSim(heading_deg=heading)
    before = sim.heading_deg
    dx, dy = sim.impulse(1.0, settle=0.2)
    assert_stationary(dx, dy)
    assert abs(circular_delta_deg(sim.heading_deg, before)) < 1e-6


@pytest.mark.parametrize("heading", CARDINAL_HEADINGS)
@pytest.mark.parametrize("direction,sign", [("right", 1.0), ("left", -1.0)])
def test_yaw_impulse_matches_semantic_compass_direction(
    heading: int, direction: str, sign: float,
) -> None:
    sim = DeterministicSim(heading_deg=heading)
    before = sim.heading_deg
    sim.impulse(1.0, settle=0.2, yaw_rate_dps=sign * 30.0)
    assert_heading_change(sim.heading_deg, before, direction)


@pytest.mark.parametrize("start,direction,sign", [
    (350.0, "right", 1.0),   # crosses 360 -> 0
    (10.0, "left", -1.0),    # crosses 0 -> 360
])
def test_yaw_across_the_zero_boundary_uses_circular_difference(
    start: float, direction: str, sign: float,
) -> None:
    sim = DeterministicSim(heading_deg=start)
    before = sim.heading_deg
    sim.impulse(1.0, settle=0.2, yaw_rate_dps=sign * 30.0)
    assert_heading_change(sim.heading_deg, before, direction)


def test_yaw_does_not_translate_the_drone() -> None:
    sim = DeterministicSim(heading_deg=0)
    dx, dy = sim.impulse(1.0, settle=0.2, yaw_rate_dps=45.0)
    assert_stationary(dx, dy)


# ---------------------------------------------------------------------------
# Command units and the three semantic stages
# ---------------------------------------------------------------------------

def test_velocity_command_fields_are_actual_mps_without_hidden_scale() -> None:
    sim = DeterministicSim(heading_deg=0)
    sim.send_body_velocity(0.7, -0.2, 0.3, 12.0)
    assert sim.state.cmd_forward == pytest.approx(0.7)
    assert sim.state.cmd_right == pytest.approx(-0.2)
    assert sim.state.cmd_up == pytest.approx(0.3)
    assert sim.state.cmd_yaw_rate == pytest.approx(12.0)


@pytest.mark.parametrize("action", sorted(EXPECTED_COMMAND_SIGN))
def test_semantic_action_encodes_to_the_contracted_wire_sign(action: str) -> None:
    """Semantic action to numeric wire command.

    The wire value is checked on its own so a client conversion and a
    simulator conversion cannot be wrong in mutually cancelling ways.
    """
    field, sign = EXPECTED_COMMAND_SIGN[action]
    payload = decode_command(encode_command("velocity", **{field: sign * 1.0}))
    assert payload is not None
    assert payload["type"] == "velocity"
    assert payload[field] * sign > 0.0, (
        f"semantic {action} encoded {field}={payload[field]}, expected sign {sign:+d}"
    )


def test_command_round_trip_preserves_every_semantic_field() -> None:
    fields = {"forward_mps": 0.5, "right_mps": -0.25,
              "up_mps": 0.1, "yaw_rate_dps": -30.0}
    payload = decode_command(encode_command("velocity", **fields))
    assert payload is not None
    for key, value in fields.items():
        assert payload[key] == pytest.approx(value)


def test_decode_rejects_foreign_and_malformed_payloads() -> None:
    assert decode_command("not json") is None
    assert decode_command('{"type":"velocity"}') is None
    assert decode_command('{"magic":"dvision2.command.v1"}') is None


def test_wire_yaw_sign_survives_the_full_semantic_chain() -> None:
    """Semantic action to wire value to observed motion, checked end to end:
    yaw-right encodes positive and increases heading."""
    payload = decode_command(encode_command("velocity", yaw_rate_dps=30.0))
    assert payload is not None and payload["yaw_rate_dps"] > 0.0
    sim = DeterministicSim(heading_deg=0)
    before = sim.heading_deg
    sim._command("velocity", yaw_rate_dps=payload["yaw_rate_dps"])
    sim.step(1.0)
    assert_heading_change(sim.heading_deg, before, "right")


# ---------------------------------------------------------------------------
# Heading conventions
# ---------------------------------------------------------------------------

def test_heading_conversion_round_trips_cardinal_values() -> None:
    for heading in CARDINAL_HEADINGS:
        assert sim_yaw_to_compass_heading(
            compass_heading_to_sim_yaw(heading)
        ) == pytest.approx(heading)


def test_published_heading_and_compass_are_the_same_public_value() -> None:
    for heading in (0.0, 37.5, 180.0, 359.9):
        sim = DeterministicSim(heading_deg=heading)
        telemetry = sim.read_telemetry()
        assert telemetry["drone.heading_deg"] == telemetry["drone.compass_deg"]
        assert float(telemetry["drone.heading_deg"]) == pytest.approx(heading, abs=0.01)


# ---------------------------------------------------------------------------
# Published telemetry signs
#
# Everything below reads the dict the production publisher builds
# (DroneSimulator.status_fields, via DeterministicSim.read_telemetry). These
# keys are DAIC's only view of the vehicle: pose_from_status navigates on the
# position keys, set_motion_from_status turns the velocity keys into a flow
# range, and the attitude keys carry the camera's orientation. A sign inverted
# here is invisible in the rendered image, so the image oracles cannot catch it.
# ---------------------------------------------------------------------------

def test_published_position_keys_are_not_transposed() -> None:
    """A north leg must move drone.y_m alone, an east leg drone.x_m alone."""
    sim = DeterministicSim(heading_deg=0.0)
    before = sim.read_telemetry()
    sim.impulse(1.0, forward_mps=1.0)
    sim.step(0.5)
    after = sim.read_telemetry()
    dx = float(after["drone.x_m"]) - float(before["drone.x_m"])
    dy = float(after["drone.y_m"]) - float(before["drone.y_m"])
    assert dy < -0.05, f"north leg moved drone.y_m by {dy:+.3f}, expected decrease"
    assert abs(dx) <= 0.04, f"north leg also moved drone.x_m by {dx:+.3f}"

    sim = DeterministicSim(heading_deg=90.0)
    before = sim.read_telemetry()
    sim.impulse(1.0, forward_mps=1.0)
    sim.step(0.5)
    after = sim.read_telemetry()
    dx = float(after["drone.x_m"]) - float(before["drone.x_m"])
    dy = float(after["drone.y_m"]) - float(before["drone.y_m"])
    assert dx > 0.05, f"east leg moved drone.x_m by {dx:+.3f}, expected increase"
    assert abs(dy) <= 0.04, f"east leg also moved drone.y_m by {dy:+.3f}"


def test_published_velocity_keys_carry_the_world_frame_sign() -> None:
    """drone.vx_mps is east-positive and drone.vy_mps is south-positive."""
    sim = DeterministicSim(heading_deg=90.0)   # nose east
    sim.send_body_velocity(1.0, 0.0, 0.0, 0.0)
    sim.step(1.0)
    telemetry = sim.read_telemetry()
    assert float(telemetry["drone.vx_mps"]) > 0.1, (
        f"flying east published vx {telemetry['drone.vx_mps']}, expected positive")
    assert abs(float(telemetry["drone.vy_mps"])) <= 0.1

    sim = DeterministicSim(heading_deg=180.0)  # nose south
    sim.send_body_velocity(1.0, 0.0, 0.0, 0.0)
    sim.step(1.0)
    telemetry = sim.read_telemetry()
    assert float(telemetry["drone.vy_mps"]) > 0.1, (
        f"flying south published vy {telemetry['drone.vy_mps']}, expected positive")
    assert abs(float(telemetry["drone.vx_mps"])) <= 0.1


def test_published_speed_is_the_magnitude_of_the_published_velocity() -> None:
    sim = DeterministicSim(heading_deg=45.0)
    sim.send_body_velocity(1.0, 0.4, 0.0, 0.0)
    sim.step(1.0)
    t = sim.read_telemetry()
    expected = math.sqrt(sum(float(t[k]) ** 2
                             for k in ("drone.vx_mps", "drone.vy_mps", "drone.vz_mps")))
    assert float(t["drone.speed_mps"]) == pytest.approx(expected, abs=0.01)


def test_published_roll_is_positive_for_a_right_wing_down_bank() -> None:
    sim = DeterministicSim(heading_deg=0.0)
    sim.send_body_velocity(0.0, 1.0, 0.0, 0.0)
    sim.step(1.0)
    right = float(sim.read_telemetry()["drone.roll_deg"])

    sim = DeterministicSim(heading_deg=0.0)
    sim.send_body_velocity(0.0, -1.0, 0.0, 0.0)
    sim.step(1.0)
    left = float(sim.read_telemetry()["drone.roll_deg"])

    assert right > 2.0, (
        f"right strafe published drone.roll_deg {right:+.2f}; positive roll "
        "must mean a right-wing-down bank")
    assert left < -2.0, f"left strafe published drone.roll_deg {left:+.2f}"


def test_published_pitch_is_negative_nose_down_in_forward_flight() -> None:
    sim = DeterministicSim(heading_deg=0.0)
    sim.send_body_velocity(1.0, 0.0, 0.0, 0.0)
    sim.step(1.0)
    forward = float(sim.read_telemetry()["drone.pitch_deg"])

    sim = DeterministicSim(heading_deg=0.0)
    sim.send_body_velocity(-1.0, 0.0, 0.0, 0.0)
    sim.step(1.0)
    backward = float(sim.read_telemetry()["drone.pitch_deg"])

    assert forward < -2.0, (
        f"forward flight published drone.pitch_deg {forward:+.2f}; positive "
        "pitch is nose-up, so flying forward must publish a negative pitch")
    assert backward > 2.0, f"reverse published drone.pitch_deg {backward:+.2f}"


def test_step_rejects_nonpositive_timestep() -> None:
    with pytest.raises(ValueError):
        DeterministicSim(heading_deg=0).sim.step(0.0)


# ---------------------------------------------------------------------------
# UI semantics
# ---------------------------------------------------------------------------

def test_dsim_displayed_heading_equals_published_heading() -> None:
    """Physics can be right while the label shows internal renderer yaw."""
    for heading in (0.0, 90.0, 217.0):
        sim = DeterministicSim(heading_deg=heading)
        published = float(sim.read_telemetry()["drone.heading_deg"])
        text = TopDownUi.status_text(sim.state)
        assert f"heading={published:.1f}" in text, (
            f"DSIM displayed {text!r}, which does not show the published "
            f"compass heading {published:.1f}"
        )


def test_dctl_displays_the_same_published_heading_as_dsim() -> None:
    sim = DeterministicSim(heading_deg=123.0)
    published = sim.read_telemetry()["drone.heading_deg"]
    assert format_status_value("drone.heading_deg", published) == (
        f"{float(published):.1f} deg"
    )


def test_ui_direction_labels_match_public_command_semantics() -> None:
    labels = {
        "move forward": DroneState(0.0, 0.0, 1.0, cmd_forward=1.0),
        "move back": DroneState(0.0, 0.0, 1.0, cmd_forward=-1.0),
        "move right": DroneState(0.0, 0.0, 1.0, cmd_right=1.0),
        "move left": DroneState(0.0, 0.0, 1.0, cmd_right=-1.0),
        "move up": DroneState(0.0, 0.0, 1.0, cmd_up=1.0),
        "move down": DroneState(0.0, 0.0, 1.0, cmd_up=-1.0),
        "yaw right": DroneState(0.0, 0.0, 1.0, cmd_yaw_rate=10.0),
        "yaw left": DroneState(0.0, 0.0, 1.0, cmd_yaw_rate=-10.0),
    }
    for label, state in labels.items():
        assert label in TopDownUi.command_text(state)
    assert TopDownUi.command_text(DroneState(0.0, 0.0, 1.0)) == "hover"


def test_yaw_right_label_corresponds_to_increasing_compass_heading() -> None:
    """The label and the observed motion are checked against each other."""
    sim = DeterministicSim(heading_deg=0)
    before = sim.heading_deg
    sim.send_body_velocity(0.0, 0.0, 0.0, 30.0)
    assert "yaw right" in TopDownUi.command_text(sim.state)
    sim.step(1.0)
    assert_heading_change(sim.heading_deg, before, "right")


def test_dctl_manual_yaw_binding_maps_right_to_positive_wire_rate() -> None:
    assert _manual_yaw_rate(1.0) == _MANUAL_YAW_RATE_DPS
    assert _manual_yaw_rate(-1.0) == -_MANUAL_YAW_RATE_DPS


# ---------------------------------------------------------------------------
# Fault injection: the oracle must reject each deliberate inversion
# ---------------------------------------------------------------------------

def test_oracle_rejects_injected_strafe_sign_inversion() -> None:
    sim = DeterministicSim(heading_deg=0)
    corrupted = invert_strafe({"right_mps": 0.5})
    dx, dy = sim.impulse(1.0, settle=0.2, **corrupted)
    with pytest.raises(AssertionError):
        assert_direction(dx, dy, EXPECTED_RIGHT[0])


def test_oracle_rejects_injected_yaw_sign_inversion() -> None:
    sim = DeterministicSim(heading_deg=0)
    before = sim.heading_deg
    corrupted = invert_yaw({"yaw_rate_dps": 30.0})
    sim.impulse(1.0, settle=0.2, **corrupted)
    with pytest.raises(AssertionError):
        assert_heading_change(sim.heading_deg, before, "right")


def test_oracle_rejects_injected_forward_right_axis_exchange() -> None:
    sim = DeterministicSim(heading_deg=0)
    corrupted = COMMAND_FAULTS["exchange_forward_and_right"](
        {"forward_mps": 0.5, "right_mps": 0.0}
    )
    dx, dy = sim.impulse(1.0, settle=0.2, **corrupted)
    with pytest.raises(AssertionError):
        assert_direction(dx, dy, EXPECTED_FORWARD[0])


def test_deterministic_group_dependencies_are_present() -> None:
    """This group may never be skipped, so its dependencies are mandatory."""
    assert missing_dependencies(DETERMINISTIC_MODULES) == []
