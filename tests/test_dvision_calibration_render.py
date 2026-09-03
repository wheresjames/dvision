"""Rendered, client-adapter, and closed-loop image orientation contract tests.

The static cases render the calibration scene once and measure it with
independent RGB masks.

The closed-loop cases step commands through deterministic physics and re-render
the resulting *world geometry*, so command semantics, telemetry, camera pose,
and pixels are all proved against one another.
"""

import numpy as np
import pytest

from daic.controller import servo
from daic.daic import _annotate, _client_rgb_frame, _display_rgb_frame
from daic.detector import detect
from dctl.dctl import _client_rgb_frame as dctl_client_rgb_frame
from dtest.artifacts import artifact_directory
from dtest.assertions import (
    assert_calibration_orientation,
    assert_channel_order,
    assert_heading_change,
    assert_landmark_moves,
)
from dtest.calibration_scene import (
    CALIBRATION_RING_MAP,
    CENTER_X,
    DIRECT_MAP,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    MINIMUM_MARKER_PIXELS,
    RING_HEADINGS,
)
from dtest.contract import circular_delta_deg
from dtest.color_probe import color_centroid, horizon_row
from dtest.conformance import run_conformance
from dtest.deterministic import DeterministicSim
from dtest.faults import IMAGE_FAULTS, asymmetric_probe_frame


@pytest.fixture(scope="module")
def calibration_sim():
    sim = DeterministicSim(heading_deg=0.0)
    try:
        yield sim
    finally:
        sim.close()


@pytest.fixture(scope="module")
def calibration_frame(calibration_sim) -> np.ndarray:
    return calibration_sim.render().copy()


# ---------------------------------------------------------------------------
# Static render orientation
# ---------------------------------------------------------------------------

def test_static_calibration_landmarks_prove_orientation(
    calibration_frame: np.ndarray, tmp_path,
) -> None:
    assert_calibration_orientation(
        calibration_frame,
        artifact_dir=artifact_directory(tmp_path, "static-orientation"),
    )


def test_neutral_forward_landmark_sits_near_the_image_centre(
    calibration_frame: np.ndarray,
) -> None:
    """The white panel is one map cell east of the nose at heading 0."""
    white = color_centroid(calibration_frame, "white")
    green = color_centroid(calibration_frame, "green")
    assert green.x == pytest.approx(CENTER_X, abs=25.0), (
        f"green landmark on the forward axis is at x={green.x:.1f}, "
        f"expected near the image centre {CENTER_X:.0f}"
    )
    assert CENTER_X < white.x < CENTER_X + 130.0, (
        f"white landmark expected just right of centre, observed x={white.x:.1f}"
    )


# ---------------------------------------------------------------------------
# Orientation at every cardinal heading
#
# Everything above sits at heading 0. An orientation error that is a function
# of yaw -- a rotation applied with the wrong sign, a frame composed in the
# wrong order -- vanishes there and would pass the whole static group. The ring
# fixture is built so the *same* literal expectations must hold from all four
# sides; see the diagram in dtest/calibration_scene.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("heading", RING_HEADINGS)
def test_landmark_orientation_holds_at_every_cardinal_heading(
    heading: float, tmp_path,
) -> None:
    sim = DeterministicSim(map_path=CALIBRATION_RING_MAP, heading_deg=heading)
    frame = sim.render().copy()

    assert_calibration_orientation(
        frame,
        artifact_dir=artifact_directory(tmp_path, f"ring-orientation-{heading:.0f}"),
    )


@pytest.mark.parametrize("heading", RING_HEADINGS)
def test_ring_landmarks_are_fully_visible_at_every_heading(heading: float) -> None:
    """Guards the fixture itself: a group turned edge-on would leave a sliver.

    Without this a shrinking mask could quietly erode the orientation check
    above into a test of almost nothing.
    """
    sim = DeterministicSim(map_path=CALIBRATION_RING_MAP, heading_deg=heading)
    frame = sim.render()

    for name in ("red", "yellow", "green", "white", "blue"):
        centroid = color_centroid(frame, name)
        assert centroid is not None, f"{name} landmark missing at heading {heading:.0f}"
        assert centroid.pixels >= MINIMUM_MARKER_PIXELS, (
            f"{name} landmark covered only {centroid.pixels} px at heading "
            f"{heading:.0f}; expected at least {MINIMUM_MARKER_PIXELS}")


def test_ring_fixture_measures_the_same_from_all_four_sides() -> None:
    """The fixture's premise, asserted rather than assumed.

    If the four groups did not present the same geometry, the parameterised
    test above would be four different tests wearing one name, and a heading
    where the check happened to be weaker would not be visible.
    """
    positions = {}
    for heading in RING_HEADINGS:
        frame = DeterministicSim(map_path=CALIBRATION_RING_MAP,
                                 heading_deg=heading).render()
        positions[heading] = {
            name: color_centroid(frame, name)
            for name in ("red", "yellow", "green", "white", "blue")
        }

    reference = positions[RING_HEADINGS[0]]
    for heading in RING_HEADINGS[1:]:
        for name, centroid in positions[heading].items():
            expected = reference[name]
            assert centroid.x == pytest.approx(expected.x, abs=6.0), (
                f"{name} sits at x={centroid.x:.1f} at heading {heading:.0f} but "
                f"x={expected.x:.1f} at heading {RING_HEADINGS[0]:.0f}")
            assert centroid.y == pytest.approx(expected.y, abs=6.0), (
                f"{name} sits at y={centroid.y:.1f} at heading {heading:.0f} but "
                f"y={expected.y:.1f} at heading {RING_HEADINGS[0]:.0f}")


@pytest.mark.parametrize("heading", RING_HEADINGS)
def test_yaw_right_moves_landmarks_left_from_every_heading(heading: float) -> None:
    """The dynamic check the static group only makes at heading 0."""
    sim = DeterministicSim(map_path=CALIBRATION_RING_MAP, heading_deg=heading)
    before = color_centroid(sim.render().copy(), "green")
    heading_before = sim.heading_deg

    sim.send_body_velocity(0.0, 0.0, 0.0, 6.0)   # yaw right
    sim.step(1.0)
    sim.zero()
    after = color_centroid(sim.render().copy(), "green")

    assert circular_delta_deg(sim.heading_deg, heading_before) > 1.0, (
        "a positive yaw rate must raise the compass heading")
    assert after is not None and before is not None
    assert after.x < before.x - 10.0, (
        f"yawing right from heading {heading:.0f} moved the forward landmark "
        f"from x={before.x:.1f} to x={after.x:.1f}; it must sweep left")


def test_rendered_channel_order_is_rgb24(calibration_frame: np.ndarray) -> None:
    assert_channel_order(calibration_frame)


def test_calibration_markers_do_not_trigger_the_landing_target_detector(
    calibration_frame: np.ndarray,
) -> None:
    """A diagnostic panel must never be mistaken for a real target."""
    detection = detect(calibration_frame)
    assert not detection.visible, (
        "calibration markers were detected as a landing target at "
        f"({detection.cx:.1f}, {detection.cy:.1f}); marker colours must stay "
        "outside the detector's saturation band"
    )


@pytest.mark.parametrize("fault", sorted(IMAGE_FAULTS))
def test_injected_image_fault_is_detected(
    calibration_frame: np.ndarray, fault: str,
) -> None:
    """Flip, transpose, rotation, and RGB/BGR exchange must all fail."""
    corrupted = IMAGE_FAULTS[fault](calibration_frame)
    with pytest.raises(AssertionError):
        assert_calibration_orientation(corrupted)


def test_failed_orientation_assertion_writes_visual_evidence(
    calibration_frame: np.ndarray, tmp_path,
) -> None:
    artifacts = tmp_path / "forced-orientation-failure"
    with pytest.raises(AssertionError):
        assert_calibration_orientation(
            IMAGE_FAULTS["horizontal_flip"](calibration_frame),
            artifact_dir=artifacts,
        )
    assert (artifacts / "frame.png").stat().st_size > 0
    assert (artifacts / "result.json").stat().st_size > 0


# ---------------------------------------------------------------------------
# Client frame adapters
# ---------------------------------------------------------------------------

_ADAPTERS = {
    "dctl_display": dctl_client_rgb_frame,
    "daic_vision": _client_rgb_frame,
    "daic_display": _display_rgb_frame,
}


@pytest.mark.parametrize("adapter", sorted(_ADAPTERS))
def test_adapters_preserve_a_synthetic_asymmetric_array(adapter: str) -> None:
    """Asymmetric in x, in y, in axis order, and across channels at once."""
    probe = asymmetric_probe_frame()
    out = _ADAPTERS[adapter](probe)
    assert out.shape == probe.shape
    assert np.array_equal(out, probe), (
        f"{adapter} altered the frame; a client may annotate pixels but must "
        "never change orientation or channel order"
    )
    assert out.flags["C_CONTIGUOUS"]


def test_adapters_agree_on_a_real_shared_memory_calibration_frame(
    calibration_frame: np.ndarray,
) -> None:
    """One captured array through every adapter, compared byte for byte."""
    shared = np.ascontiguousarray(calibration_frame)
    outputs = {name: fn(shared) for name, fn in _ADAPTERS.items()}
    for name, out in outputs.items():
        assert np.array_equal(out, shared), f"{name} changed the shared frame"
    assert len({out.tobytes() for out in outputs.values()}) == 1


@pytest.mark.parametrize("fault", sorted(IMAGE_FAULTS))
@pytest.mark.parametrize("adapter", sorted(_ADAPTERS))
def test_adapter_output_fails_the_oracle_under_injected_faults(
    calibration_frame: np.ndarray, adapter: str, fault: str,
) -> None:
    """Corrupt after the adapter; the oracle must still reject it."""
    corrupted = IMAGE_FAULTS[fault](_ADAPTERS[adapter](calibration_frame))
    with pytest.raises(AssertionError):
        assert_calibration_orientation(corrupted)


def test_annotation_changes_pixels_but_not_landmark_sides() -> None:
    """Overlays must refer to the pixels they are drawn on."""
    sim = DeterministicSim(map_path=DIRECT_MAP, heading_deg=0.0, altitude_m=3.0)
    sim.state.y = 12.5
    sim.state.x = 11.5
    frame = sim.render().copy()
    detection = detect(_client_rgb_frame(frame))
    assert detection.visible

    annotated = _annotate(_display_rgb_frame(frame), detection)
    assert annotated.shape == frame.shape
    assert not np.array_equal(annotated, frame), "annotation drew nothing"

    after = detect(annotated)
    assert after.visible
    assert after.cx == pytest.approx(detection.cx, abs=6.0), (
        f"detection moved from x={detection.cx:.1f} to x={after.cx:.1f} across "
        "annotation; overlay and detector disagree about displayed pixels"
    )
    assert (after.cx - CENTER_X) * (detection.cx - CENTER_X) > 0.0


# ---------------------------------------------------------------------------
# Closed-loop image motion
# ---------------------------------------------------------------------------

def test_yaw_right_increases_heading_and_moves_blue_landmark_left(tmp_path) -> None:
    sim = DeterministicSim(heading_deg=0.0)
    before_heading = sim.heading_deg
    before = sim.render().copy()
    blue_before = color_centroid(before, "blue")
    assert blue_before.x > CENTER_X

    sim.send_body_velocity(0.0, 0.0, 0.0, 30.0)
    sim.step(0.6)
    sim.zero()
    middle = sim.render().copy()
    assert_heading_change(sim.heading_deg, before_heading, "right")
    assert_landmark_moves(
        before, middle, "blue", "left",
        artifact_dir=artifact_directory(tmp_path, "yaw-right"),
    )

    # Continue past the forward axis: the landmark must end up left of centre.
    sim.send_body_velocity(0.0, 0.0, 0.0, 30.0)
    sim.step(1.0)
    sim.zero()
    after = sim.render().copy()
    blue_after = color_centroid(after, "blue")
    assert blue_after.x < CENTER_X, (
        "after yawing right past the landmark bearing, blue was expected left "
        f"of centre, observed x={blue_after.x:.1f}"
    )
    assert_heading_change(sim.heading_deg, before_heading, "right", minimum_deg=25.0)


def test_yaw_left_decreases_heading_and_moves_red_landmark_right(tmp_path) -> None:
    sim = DeterministicSim(heading_deg=0.0)
    before_heading = sim.heading_deg
    before = sim.render().copy()
    assert color_centroid(before, "red").x < CENTER_X

    sim.send_body_velocity(0.0, 0.0, 0.0, -30.0)
    sim.step(0.6)
    sim.zero()
    after = sim.render().copy()
    assert_heading_change(sim.heading_deg, before_heading, "left")
    assert_landmark_moves(
        before, after, "red", "right",
        artifact_dir=artifact_directory(tmp_path, "yaw-left"),
    )


@pytest.mark.parametrize("direction,sign,expected_image_motion", [
    ("right", 1.0, "left"),
    ("left", -1.0, "right"),
])
def test_strafe_produces_world_displacement_and_landmark_parallax(
    direction: str, sign: float, expected_image_motion: str, tmp_path,
) -> None:
    """Secondary image check: with yaw locked, a strafe shifts the scene.

    Yaw gives a cleaner visual oracle, so the world-displacement assertion is
    the primary one here and the parallax check is deliberately tolerant.
    """
    sim = DeterministicSim(heading_deg=0.0)
    before_heading = sim.heading_deg
    before = sim.render().copy()

    x0 = sim.state.x
    sim.send_body_velocity(0.0, sign * 1.0, 0.0, 0.0)
    sim.step(1.5)
    sim.zero()
    after = sim.render().copy()

    dx = sim.state.x - x0
    assert dx * sign > 0.3, (
        f"strafe {direction} produced dx={dx:+.3f} m at heading 0; "
        f"expected motion toward map {'east' if sign > 0 else 'west'}"
    )
    assert abs(sim.heading_deg - before_heading) < 0.5, "strafe changed heading"
    assert_landmark_moves(
        before, after, "white", expected_image_motion, minimum_px=10.0,
        artifact_dir=artifact_directory(tmp_path, f"strafe-{direction}"),
    )


# ---------------------------------------------------------------------------
# DAIC visual-servo polarity from real rendered frames
# ---------------------------------------------------------------------------

def _servo_scenario(lateral_offset_m: float) -> DeterministicSim:
    """Unobstructed world, fixed pose, genuine target deliberately off centre."""
    sim = DeterministicSim(map_path=DIRECT_MAP, heading_deg=0.0, altitude_m=3.0)
    sim.state.y = 12.5
    sim.state.x = 10.5 + lateral_offset_m
    return sim


@pytest.mark.parametrize("offset_m", [-1.0, 1.0])
def test_daic_servo_reduces_horizontal_pixel_error_from_either_side(
    offset_m: float, tmp_path,
) -> None:
    sim = _servo_scenario(offset_m)
    frame = sim.render().copy()
    detection = detect(_client_rgb_frame(frame))
    assert detection.visible, "calibration approach scenario produced no detection"
    error_before = abs(detection.cx - CENTER_X)
    assert error_before > 30.0, "target was not placed far enough off centre"

    command = servo(detection, FRAME_WIDTH, FRAME_HEIGHT, altitude_m=sim.state.z)
    sim.send_body_velocity(
        command.forward_mps, command.right_mps, 0.0, command.yaw_rate_dps,
    )
    sim.step(1.0)

    after = detect(_client_rgb_frame(sim.render()))
    assert after.visible, "target left the frame while servoing toward it"
    error_after = abs(after.cx - CENTER_X)
    assert error_after < error_before, (
        f"DAIC steered away from the target: horizontal pixel error went "
        f"{error_before:.1f} -> {error_after:.1f} with the target "
        f"{'right' if detection.cx > CENTER_X else 'left'} of centre"
    )


@pytest.mark.parametrize("offset_m", [-1.0, 1.0])
def test_daic_servo_strafe_and_yaw_axes_are_individually_correct(
    offset_m: float,
) -> None:
    """Inspect each output separately so two wrong signs cannot cancel."""
    sim = _servo_scenario(offset_m)
    detection = detect(_client_rgb_frame(sim.render()))
    assert detection.visible
    command = servo(detection, FRAME_WIDTH, FRAME_HEIGHT, altitude_m=sim.state.z)

    pixel_error = detection.cx - CENTER_X  # + = target right of centre
    assert command.right_mps * pixel_error > 0.0, (
        f"target {pixel_error:+.1f}px from centre but strafe command is "
        f"{command.right_mps:+.3f} m/s"
    )
    assert command.yaw_rate_dps * pixel_error > 0.0, (
        f"target {pixel_error:+.1f}px from centre but yaw command is "
        f"{command.yaw_rate_dps:+.3f} deg/s"
    )


def test_daic_servo_world_motion_agrees_with_the_image_correction() -> None:
    """The command must also move the drone the right way in the world."""
    sim = _servo_scenario(1.0)  # drone east of the target
    detection = detect(_client_rgb_frame(sim.render()))
    assert detection.visible and detection.cx < CENTER_X

    x0 = sim.state.x
    command = servo(detection, FRAME_WIDTH, FRAME_HEIGHT, altitude_m=sim.state.z)
    sim.send_body_velocity(
        command.forward_mps, command.right_mps, 0.0, command.yaw_rate_dps,
    )
    sim.step(1.0)
    assert sim.state.x < x0, (
        f"target was west of the drone but map x went {x0:.2f} -> {sim.state.x:.2f}"
    )


# ---------------------------------------------------------------------------
# Roll polarity
# ---------------------------------------------------------------------------

def test_right_strafe_publishes_a_positive_right_wing_down_roll() -> None:
    sim = DeterministicSim(heading_deg=0.0)
    sim.send_body_velocity(0.0, 1.0, 0.0, 0.0)
    sim.step(1.0)
    right_roll = float(sim.read_telemetry()["drone.roll_deg"])

    sim = DeterministicSim(heading_deg=0.0)
    sim.send_body_velocity(0.0, -1.0, 0.0, 0.0)
    sim.step(1.0)
    left_roll = float(sim.read_telemetry()["drone.roll_deg"])

    assert right_roll > 2.0, (
        f"right strafe published roll {right_roll:+.2f} deg; positive roll must "
        "mean a right-wing-down bank"
    )
    assert left_roll < -2.0, f"left strafe published roll {left_roll:+.2f} deg"


def test_positive_roll_lifts_the_right_end_of_the_rendered_horizon() -> None:
    """A right bank rotates the scene the other way, so the horizon tilts up
    on the right."""
    sim = DeterministicSim(heading_deg=0.0)
    level = sim.render().copy()
    assert horizon_row(level, 60) == pytest.approx(horizon_row(level, 580), abs=3)

    sim.state.roll_deg = 20.0
    banked = sim.render().copy()
    left = horizon_row(banked, 60)
    right = horizon_row(banked, 580)
    assert right < left - 40, (
        f"right bank put the horizon at row {right} on the right and {left} on "
        "the left; a positive roll must lift the right side"
    )


def test_forward_flight_publishes_a_nose_down_negative_pitch() -> None:
    sim = DeterministicSim(heading_deg=0.0)
    sim.send_body_velocity(1.0, 0.0, 0.0, 0.0)
    sim.step(1.0)
    forward_pitch = float(sim.read_telemetry()["drone.pitch_deg"])

    sim = DeterministicSim(heading_deg=0.0)
    sim.send_body_velocity(-1.0, 0.0, 0.0, 0.0)
    sim.step(1.0)
    backward_pitch = float(sim.read_telemetry()["drone.pitch_deg"])

    assert forward_pitch < -2.0, (
        f"forward flight published pitch {forward_pitch:+.2f} deg; positive "
        "pitch must mean nose-up"
    )
    assert backward_pitch > 2.0, (
        f"backward flight published pitch {backward_pitch:+.2f} deg"
    )


def test_nose_down_pitch_aims_the_camera_at_the_ground() -> None:
    """Tilting the nose down aims lower, so the horizon rises in the image."""
    sim = DeterministicSim(heading_deg=0.0)
    level = horizon_row(sim.render().copy(), 320)
    sim.state.pitch_deg = -15.0
    nose_down = horizon_row(sim.render().copy(), 320)
    sim.state.pitch_deg = 15.0
    nose_up = horizon_row(sim.render().copy(), 320)
    assert nose_down < level - 40, (
        f"nose-down pitch put the horizon at row {nose_down} versus {level} "
        "level; aiming lower must move the horizon up the image"
    )
    assert nose_up > level + 40, (
        f"nose-up pitch put the horizon at row {nose_up} versus {level} level"
    )


# ---------------------------------------------------------------------------
# Coupled attitude, at every heading
#
# The attitude tests above move one axis at a time from heading 0, and each
# reads back only the quantity its own axis changes: the roll test measures
# tilt and ignores horizon height, the pitch test samples a single column and
# so cannot see tilt at all. Between them the cross terms go unmeasured -- a
# bank that also aims the camera up, a pitch that also rolls the horizon --
# because neither shows up until both axes are non-zero at once.
#
# The ring fixture is rotationally symmetric, so a given attitude must produce
# identical image geometry from all four sides. That makes heading independence
# an exact assertion rather than a tolerance.
# ---------------------------------------------------------------------------

_ATTITUDE_COLUMNS = (60, 320, 580)
_TEST_ROLL_DEG = 20.0
_TEST_PITCH_DEG = -12.0


def _horizon_geometry(heading: float, roll: float, pitch: float) -> tuple[float, float]:
    """(tilt, mean row) of the horizon; tilt is the left row minus the right.

    A positive tilt means the right end of the horizon sits higher up the
    image, which is what a right-wing-down bank must produce.
    """
    sim = DeterministicSim(map_path=CALIBRATION_RING_MAP, heading_deg=heading)
    sim.state.roll_deg = roll
    sim.state.pitch_deg = pitch
    frame = sim.render().copy()
    rows = [horizon_row(frame, column) for column in _ATTITUDE_COLUMNS]
    return float(rows[0] - rows[-1]), sum(rows) / len(rows)


@pytest.mark.parametrize("heading", RING_HEADINGS)
def test_pitch_alone_does_not_tilt_the_horizon(heading: float) -> None:
    """Pitch must stay out of the roll channel.

    The single-column pitch test cannot see tilt, so a pitch that also rolled
    the camera passes it.
    """
    tilt, _ = _horizon_geometry(heading, 0.0, _TEST_PITCH_DEG)

    assert abs(tilt) <= 5.0, (
        f"pitching {_TEST_PITCH_DEG:+.0f} deg at heading {heading:.0f} tilted the "
        f"horizon by {tilt:+.0f} rows; pitch must not roll the camera")


@pytest.mark.parametrize("heading", RING_HEADINGS)
def test_roll_alone_does_not_raise_or_lower_the_horizon(heading: float) -> None:
    """Roll must stay out of the pitch channel.

    The roll test measures only tilt, so a bank that also aimed the camera up
    or down passes it.
    """
    _, level_mean = _horizon_geometry(heading, 0.0, 0.0)
    _, rolled_mean = _horizon_geometry(heading, _TEST_ROLL_DEG, 0.0)

    assert abs(rolled_mean - level_mean) <= 15.0, (
        f"banking {_TEST_ROLL_DEG:+.0f} deg at heading {heading:.0f} moved the "
        f"horizon from row {level_mean:.0f} to {rolled_mean:.0f}; a roll must "
        "not aim the camera up or down")


@pytest.mark.parametrize("heading", RING_HEADINGS)
def test_coupled_roll_and_pitch_keep_their_separate_effects(heading: float) -> None:
    """Both axes at once must still decompose into the two single-axis results.

    This puts the composition itself under test: an attitude assembled in the
    wrong order, or a roll applied about a world axis rather than the camera's
    own, agrees with both single-axis cases and diverges only here.
    """
    _, level_mean = _horizon_geometry(heading, 0.0, 0.0)
    roll_tilt, _ = _horizon_geometry(heading, _TEST_ROLL_DEG, 0.0)
    _, pitch_mean = _horizon_geometry(heading, 0.0, _TEST_PITCH_DEG)
    both_tilt, both_mean = _horizon_geometry(heading, _TEST_ROLL_DEG, _TEST_PITCH_DEG)

    assert both_tilt == pytest.approx(roll_tilt, abs=25.0), (
        f"tilt with roll and pitch together is {both_tilt:+.0f} rows but "
        f"{roll_tilt:+.0f} with roll alone; pitch is leaking into the roll channel")
    assert both_mean == pytest.approx(pitch_mean, abs=30.0), (
        f"horizon height with roll and pitch together is row {both_mean:.0f} but "
        f"{pitch_mean:.0f} with pitch alone (level {level_mean:.0f}); roll is "
        "leaking into the pitch channel")


@pytest.mark.parametrize("heading", RING_HEADINGS)
def test_roll_sign_survives_a_simultaneous_pitch(heading: float) -> None:
    """A bank must lift the same side whether or not the nose is down."""
    right_bank, _ = _horizon_geometry(heading, _TEST_ROLL_DEG, _TEST_PITCH_DEG)
    left_bank, _ = _horizon_geometry(heading, -_TEST_ROLL_DEG, _TEST_PITCH_DEG)

    assert right_bank > 40.0, (
        f"a right bank with the nose down gave tilt {right_bank:+.0f} at heading "
        f"{heading:.0f}; it must still lift the right end of the horizon")
    assert left_bank < -40.0, (
        f"a left bank with the nose down gave tilt {left_bank:+.0f} at heading "
        f"{heading:.0f}")


def test_coupled_attitude_looks_the_same_from_every_heading() -> None:
    """The fixture is symmetric, so attitude geometry must not depend on yaw."""
    results = {
        heading: _horizon_geometry(heading, _TEST_ROLL_DEG, _TEST_PITCH_DEG)
        for heading in RING_HEADINGS
    }
    reference_tilt, reference_mean = results[RING_HEADINGS[0]]

    for heading, (tilt, mean) in results.items():
        assert tilt == pytest.approx(reference_tilt, abs=6.0), (
            f"the same attitude tilted the horizon {tilt:+.0f} rows at heading "
            f"{heading:.0f} but {reference_tilt:+.0f} at {RING_HEADINGS[0]:.0f}")
        assert mean == pytest.approx(reference_mean, abs=6.0), (
            f"the same attitude put the horizon at row {mean:.0f} at heading "
            f"{heading:.0f} but {reference_mean:.0f} at {RING_HEADINGS[0]:.0f}")


@pytest.mark.parametrize("heading", RING_HEADINGS)
def test_diagonal_flight_publishes_both_attitude_axes(heading: float) -> None:
    """A forward-and-right leg banks right and pitches nose down at once.

    Published attitude is derived in the body frame, so heading must not enter
    it; every cardinal heading has to give the same answer.
    """
    def attitude(forward: float, right: float) -> tuple[float, float]:
        sim = DeterministicSim(heading_deg=heading)
        sim.send_body_velocity(forward, right, 0.0, 0.0)
        sim.step(1.0)
        telemetry = sim.read_telemetry()
        return (float(telemetry["drone.roll_deg"]),
                float(telemetry["drone.pitch_deg"]))

    roll, pitch = attitude(1.0, 1.0)
    assert roll > 2.0 and pitch < -2.0, (
        f"flying forward-right at heading {heading:.0f} published roll {roll:+.2f} "
        f"pitch {pitch:+.2f}; expected a right bank and a nose-down pitch")

    roll, pitch = attitude(1.0, -1.0)
    assert roll < -2.0 and pitch < -2.0, (
        f"flying forward-left at heading {heading:.0f} published roll {roll:+.2f} "
        f"pitch {pitch:+.2f}")

    roll, pitch = attitude(-1.0, 1.0)
    assert roll > 2.0 and pitch > 2.0, (
        f"flying back-right at heading {heading:.0f} published roll {roll:+.2f} "
        f"pitch {pitch:+.2f}; the two axes must be independent")


# ---------------------------------------------------------------------------
# Backend conformance
# ---------------------------------------------------------------------------

def test_deterministic_backend_passes_the_conformance_suite() -> None:
    """The same backend-neutral suite that gates the real DSIM process and,
    later, any MAVLink backend."""
    sim = DeterministicSim(heading_deg=0.0)
    try:
        summary = run_conformance(sim, sim.step)
    finally:
        sim.close()
    assert summary["capabilities"]["deterministic"] is True
    assert summary["frame_shape"] == [FRAME_HEIGHT, FRAME_WIDTH, 3]


def test_full_pinned_vision_environment_is_installed() -> None:
    from dtest.preflight import missing_dependencies

    assert missing_dependencies() == []
