"""Tests for daic.planner — state machine logic, no I/O."""

import time

import pytest

from daic.planner import (
    Planner, State,
    SEARCH_ALT_M,
    _APPROACH_LOCK_FRAMES, _LANDING_LOCK_FRAMES,
    _LOST_TARGET_TIMEOUT_S,
    _LOW_BATTERY_PCT, _ARM_TIMEOUT_S, _ARM_RETRY_S,
    _SEARCH_TURN_DPS, _SEARCH_TURN_DEG,
    _SEARCH_LEG_S_BASE, _SEARCH_LEG_S_INC,
)
from daic.detector import Detection


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _det(visible=True, cx=320.0, cy=240.0, radius=50.0, confidence=0.85):
    return Detection(visible=visible, cx=cx, cy=cy,
                     radius=radius, confidence=confidence)

def _no_det():
    return Detection(visible=False, cx=0, cy=0, radius=0, confidence=0)

def _status(armed="1", mode="GUIDED", z=3.0, battery=100.0, heading=270.0):
    return {
        "drone.armed":       armed,
        "drone.mode":        mode,
        "drone.z_m":         str(z),
        "drone.battery_pct": str(battery),
        "drone.heading_deg": str(heading),
    }

def _status_with_target_gps():
    status = _status()
    status.update({
        "drone.lat_deg": "52.0000000",
        "drone.lon_deg": "13.0000000",
        "target.lat_deg": "52.0000800",
        "target.lon_deg": "13.0000000",
        "drone.compass_deg": "0.0",
    })
    return status

def _new_planner():
    return Planner(img_w=640, img_h=480)


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------

def test_initial_state_is_idle():
    p = _new_planner()
    assert p.state == State.IDLE


def test_idle_tick_does_not_send_command():
    p = _new_planner()
    out = p.tick(_no_det(), {})
    assert not out.send_command


def test_disable_on_idle_stays_idle():
    p = _new_planner()
    out = p.disable()
    assert p.state == State.IDLE
    assert out.command_type == "zero"


# ---------------------------------------------------------------------------
# Enable → ARMING
# ---------------------------------------------------------------------------

def test_enable_transitions_to_arming():
    p = _new_planner()
    p.enable({})
    assert p.state == State.ARMING


def test_arming_sends_arm_command_immediately():
    p = _new_planner()
    p.enable({})
    out = p.tick(_no_det(), _status(armed="0", mode="DISARMED", z=0.0))
    assert out.command_type == "arm"
    assert out.command_fields.get("armed") is True


def test_arming_transitions_to_search_when_armed_without_takeoff():
    p = _new_planner()
    p.enable({})
    # First tick sends arm.
    p.tick(_no_det(), _status(armed="0", mode="DISARMED", z=0.0))
    # Subsequent tick with confirmed arm and elapsed > 0.2 s.
    # We patch the state entry time to simulate elapsed time.
    p._state_entered -= 0.5
    out = p.tick(_no_det(), _status(armed="1", mode="GUIDED", z=0.0))
    assert p.state == State.SEARCH
    assert out.command_type == "zero"


def test_arming_arms_even_when_the_first_tick_is_late():
    """The first tick waits on the first frame, so it can land seconds late.

    Arming used to be a one-shot request valid only for 200 ms after the state
    was entered. When camera start-up pushed the first tick past that window
    the request was never sent at all, and the run sat disarmed until the arm
    timeout fired. Six of ten benchmark runs died this way.
    """
    p = _new_planner()
    p.enable({})
    p._state_entered -= 1.5          # first frame arrived 1.5 s after enable

    out = p.tick(_no_det(), _status(armed="0", mode="DISARMED", z=0.0))

    assert out.command_type == "arm"
    assert out.command_fields.get("armed") is True
    assert p.state == State.ARMING


def test_arming_repeats_the_request_until_the_vehicle_agrees():
    """A dropped arm command must not cost the whole run."""
    p = _new_planner()
    p.enable({})
    disarmed = _status(armed="0", mode="DISARMED", z=0.0)

    sent = 0
    for _ in range(6):               # ~1.5 s of ticks at the retry period
        if p.tick(_no_det(), disarmed).command_type == "arm":
            sent += 1
        p._last_arm_command -= _ARM_RETRY_S

    assert sent == 6
    assert p.state == State.ARMING


def test_arming_stops_asking_once_armed():
    p = _new_planner()
    p.enable({})
    p.tick(_no_det(), _status(armed="0", mode="DISARMED", z=0.0))

    out = p.tick(_no_det(), _status(armed="1", mode="GUIDED", z=0.0))

    assert p.state == State.SEARCH
    assert out.command_type != "arm"


def test_arming_does_not_flood_the_link_with_arm_commands():
    """Repeats are rate limited; the link carries heartbeats in between."""
    p = _new_planner()
    p.enable({})
    disarmed = _status(armed="0", mode="DISARMED", z=0.0)

    types = [p.tick(_no_det(), disarmed).command_type for _ in range(5)]

    assert types[0] == "arm"
    assert all(t == "heartbeat" for t in types[1:])


def test_arming_failsafe_on_timeout():
    p = _new_planner()
    p.enable({})
    p._last_arm_command = p._state_entered
    p._state_entered -= _ARM_TIMEOUT_S + 1.0
    out = p.tick(_no_det(), _status(armed="0", mode="DISARMED", z=0.0))
    assert p.state == State.FAILSAFE


def test_slow_gui_startup_cannot_timeout_before_first_arm_request():
    p = _new_planner()
    p.enable({})
    p._state_entered -= _ARM_TIMEOUT_S + 10.0

    out = p.tick(_no_det(), _status(armed="0", mode="DISARMED", z=0.0))

    assert p.state == State.ARMING
    assert out.command_type == "arm"
    assert out.command_fields == {"armed": True}


# ---------------------------------------------------------------------------
# TAKEOFF compatibility
# ---------------------------------------------------------------------------

def test_takeoff_state_transitions_to_search_without_climbing():
    p = _new_planner()
    p._state = State.TAKEOFF
    p._state_entered = time.monotonic()
    out = p.tick(_no_det(), _status(armed="1", mode="GUIDED", z=0.1))
    assert p.state == State.SEARCH
    assert out.command_type == "zero"


# ---------------------------------------------------------------------------
# SEARCH
# ---------------------------------------------------------------------------

def test_search_sends_velocity():
    p = _new_planner()
    p._state = State.SEARCH
    p._state_entered = time.monotonic()
    out = p.tick(_no_det(), _status(z=SEARCH_ALT_M))
    assert out.command_type == "velocity"


def test_search_does_not_emit_gps_nav_velocity_when_target_gps_known():
    p = _new_planner()
    p._state = State.SEARCH
    p._state_entered = time.monotonic()

    out = p.tick(_no_det(), _status_with_target_gps())

    assert out.command_type == "velocity"
    assert "GPS nav" not in out.status_text


def test_search_transitions_to_approach_on_lock():
    p = _new_planner()
    p._state = State.SEARCH
    p._state_entered = time.monotonic()
    # Feed enough consecutive detections to trigger lock.
    for _ in range(_APPROACH_LOCK_FRAMES):
        out = p.tick(_det(), _status(z=SEARCH_ALT_M))
    assert p.state == State.APPROACH


def test_search_does_not_transition_on_single_detection():
    p = _new_planner()
    p._state = State.SEARCH
    p._state_entered = time.monotonic()
    p.tick(_det(), _status(z=SEARCH_ALT_M))
    assert p.state == State.SEARCH


def test_search_turn_lasts_full_ninety_degrees():
    p = _new_planner()
    p._state = State.SEARCH
    now = time.monotonic()
    p._search.leg_start = now - p._search.leg_duration - 0.1

    first = p.tick(_no_det(), _status(z=SEARCH_ALT_M))
    assert first.command_fields["yaw_rate_dps"] == _SEARCH_TURN_DPS
    assert p._search.turning

    half_turn_s = (_SEARCH_TURN_DEG / _SEARCH_TURN_DPS) / 2.0
    p._search.leg_start = time.monotonic() - half_turn_s
    mid = p.tick(_no_det(), _status(z=SEARCH_ALT_M))
    assert mid.command_fields["yaw_rate_dps"] == _SEARCH_TURN_DPS
    assert p._search.turning

    p._search.leg_start = time.monotonic() - (_SEARCH_TURN_DEG / _SEARCH_TURN_DPS)
    done = p.tick(_no_det(), _status(z=SEARCH_ALT_M))
    assert done.command_fields["forward_mps"] > 0
    assert not p._search.turning


def test_search_expands_quickly_after_two_legs():
    p = _new_planner()
    p._state = State.SEARCH
    assert p._search.leg_duration == _SEARCH_LEG_S_BASE

    now = time.monotonic()
    p._search.leg_start = now - p._search.leg_duration - 0.1
    p.tick(_no_det(), _status(z=SEARCH_ALT_M))
    assert p._search.leg_duration == _SEARCH_LEG_S_BASE

    p._search.turning = False
    p._search.leg_start = time.monotonic() - p._search.leg_duration - 0.1
    p.tick(_no_det(), _status(z=SEARCH_ALT_M))
    assert p._search.leg_duration == _SEARCH_LEG_S_BASE + _SEARCH_LEG_S_INC


# ---------------------------------------------------------------------------
# APPROACH
# ---------------------------------------------------------------------------

def test_approach_sends_velocity():
    p = _new_planner()
    p._state = State.APPROACH
    p._state_entered = time.monotonic()
    out = p.tick(_det(), _status(z=SEARCH_ALT_M))
    assert out.command_type == "velocity"


def test_approach_continues_toward_last_known_on_loss():
    """Within the timeout window, the drone must keep moving (velocity, not zero)."""
    p = _new_planner()
    p._state = State.APPROACH
    p._state_entered = time.monotonic()
    p._target_last_seen = time.monotonic()
    p._last_valid_detection = _det(cx=400, cy=240)   # target was to the right
    out = p.tick(_no_det(), _status(z=SEARCH_ALT_M))
    assert p.state == State.APPROACH
    assert out.command_type == "velocity"
    # Should still be moving toward where the target was (right_mps > 0).
    assert out.command_fields.get("right_mps", 0) > 0


def test_approach_bottom_edge_loss_stops_reacquires():
    p = _new_planner()
    p._state = State.APPROACH
    p._state_entered = time.monotonic()
    p._target_last_seen = time.monotonic()
    p._last_valid_detection = _det(cx=320, cy=470, radius=55)
    out = p.tick(_no_det(), _status(z=SEARCH_ALT_M))
    assert p.state == State.SEARCH
    assert out.command_type == "zero"


def test_approach_returns_to_search_after_timeout():
    """After _LOST_TARGET_TIMEOUT_S has elapsed with no detection, give up."""
    p = _new_planner()
    p._state = State.APPROACH
    p._state_entered = time.monotonic()
    # Back-date last-seen so the timeout has already expired.
    p._target_last_seen = time.monotonic() - _LOST_TARGET_TIMEOUT_S - 0.1
    out = p.tick(_no_det(), _status(z=SEARCH_ALT_M))
    assert p.state == State.SEARCH


def test_approach_still_holding_just_before_timeout():
    """Just under the timeout threshold should still hold."""
    p = _new_planner()
    p._state = State.APPROACH
    p._state_entered = time.monotonic()
    p._target_last_seen = time.monotonic() - _LOST_TARGET_TIMEOUT_S + 0.2
    out = p.tick(_no_det(), _status(z=SEARCH_ALT_M))
    assert p.state == State.APPROACH


def test_approach_lost_timestamp_resets_on_reacquisition():
    """Regaining the target updates _target_last_seen to now."""
    p = _new_planner()
    p._state = State.APPROACH
    p._state_entered = time.monotonic()
    # Simulate target was lost a while ago.
    p._target_last_seen = time.monotonic() - 1.0
    before = p._target_last_seen
    p.tick(_det(), _status(z=SEARCH_ALT_M))
    assert p._target_last_seen > before
    assert p.state == State.APPROACH


def test_approach_descent_does_not_trigger_landing():
    p = _new_planner()
    p._state = State.APPROACH
    p._state_entered = time.monotonic()
    det = _det(cx=320.0, cy=465.0, radius=35.0)
    for _ in range(_LANDING_LOCK_FRAMES + 1):
        out = p.tick(det, _status(z=3.0))
    assert p.state == State.APPROACH
    assert out.command_type == "velocity"
    assert out.command_fields.get("up_mps", 0.0) < 0.0


def test_approach_low_centered_target_transitions_to_landing():
    p = _new_planner()
    p._state = State.APPROACH
    p._state_entered = time.monotonic()
    det = _det(cx=320.0, cy=465.0, radius=50.0)
    for _ in range(_LANDING_LOCK_FRAMES + 1):
        out = p.tick(det, _status(z=2.0))
    assert p.state == State.LANDING
    assert out.command_type == "velocity"
    assert out.command_fields.get("up_mps", 0.0) < 0.0


# ---------------------------------------------------------------------------
# LANDING
# ---------------------------------------------------------------------------

def test_landing_completes_at_low_altitude():
    p = _new_planner()
    p._state = State.LANDING
    p._state_entered = time.monotonic()
    out = p.tick(_det(), _status(z=0.1))
    assert p.state == State.COMPLETE
    assert out.command_type == "land"


def test_landing_continues_toward_last_known_on_loss():
    """Within the timeout window during landing, keep moving toward last known position."""
    p = _new_planner()
    p._state = State.LANDING
    p._state_entered = time.monotonic()
    p._target_last_seen = time.monotonic()
    p._last_valid_detection = _det(cx=320, cy=240)   # centred
    out = p.tick(_no_det(), _status(z=1.5))
    assert out.command_type == "velocity"
    assert p.state == State.LANDING


def test_landing_climbs_after_timeout():
    """After _LOST_TARGET_TIMEOUT_S with no detection, climb to reacquire."""
    p = _new_planner()
    p._state = State.LANDING
    p._state_entered = time.monotonic()
    p._target_last_seen = time.monotonic() - _LOST_TARGET_TIMEOUT_S - 0.1
    out = p.tick(_no_det(), _status(z=1.5))
    assert out.command_type == "velocity"
    assert out.command_fields.get("up_mps", 0) > 0


# ---------------------------------------------------------------------------
# Failsafe
# ---------------------------------------------------------------------------

def test_failsafe_on_low_battery():
    p = _new_planner()
    p._state = State.SEARCH
    p._state_entered = time.monotonic()
    out = p.tick(_no_det(), _status(battery=_LOW_BATTERY_PCT - 1.0))
    assert p.state == State.FAILSAFE
    assert out.command_type == "zero"


def test_failsafe_on_stale_status_marker():
    p = _new_planner()
    p._state = State.SEARCH
    p._state_entered = time.monotonic()
    status = _status()
    status["_stale"] = "1"
    out = p.tick(_no_det(), status)
    assert p.state == State.FAILSAFE
    assert out.command_type == "zero"
    assert "status stale" in out.status_text


def test_disable_from_search_returns_zero():
    p = _new_planner()
    p._state = State.SEARCH
    out = p.disable()
    assert out.command_type == "zero"
    assert p.state == State.IDLE


# ---------------------------------------------------------------------------
# PlannerOutput fields
# ---------------------------------------------------------------------------

def test_planner_output_has_status_text():
    p = _new_planner()
    out = p.tick(_no_det(), {})
    assert isinstance(out.status_text, str)
    assert len(out.status_text) > 0


def test_planner_state_matches_output_state():
    p = _new_planner()
    out = p.tick(_no_det(), {})
    assert out.state == p.state


# ---------------------------------------------------------------------------
# Crash handling and obstacle-gated target lock
#
# Run 20260826-105908-5c26161f locked onto a target that sat behind a wall,
# entered APPROACH with front_risk=0.64, and hit the wall 0.64 s later. It then
# kept commanding forward velocity for a further 5.6 s because nothing in the
# state machine looked at drone.crashed.
# ---------------------------------------------------------------------------

def _searching_planner():
    p = Planner()
    p.enable(_status())
    p.tick(_no_det(), _status())
    time.sleep(0.25)
    p.tick(_no_det(), _status())
    assert p.state is State.SEARCH
    return p


def test_crash_transitions_to_failsafe_and_stops_commanding() -> None:
    p = _searching_planner()
    status = _status()
    status["drone.crashed"] = "1"

    out = p.tick(_det(), status)

    assert p.state is State.FAILSAFE
    assert out.command_type == "zero"
    assert "crash" in out.status_text


def test_crash_failsafe_sticks_on_later_ticks() -> None:
    p = _searching_planner()
    status = _status()
    status["drone.crashed"] = "1"
    p.tick(_no_det(), status)

    out = p.tick(_det(), _status())

    assert p.state is State.FAILSAFE
    assert out.command_type == "zero"


def test_uncrashed_status_does_not_failsafe() -> None:
    p = _searching_planner()
    status = _status()
    status["drone.crashed"] = "0"

    p.tick(_no_det(), status)

    assert p.state is State.SEARCH


def test_high_front_risk_blocks_the_approach_lock() -> None:
    p = _searching_planner()

    for _ in range(_APPROACH_LOCK_FRAMES * 3):
        out = p.tick(_det(), _status(), front_risk=0.64)

    assert p.state is State.SEARCH
    assert "approach lock held" in out.status_text


def test_approach_lock_needs_clear_air_not_a_lucky_gap() -> None:
    """A momentary dip in risk must not complete a lock built up behind a wall."""
    p = _searching_planner()

    for _ in range(_APPROACH_LOCK_FRAMES - 1):
        p.tick(_det(), _status(), front_risk=0.0)
    p.tick(_det(), _status(), front_risk=0.64)      # obstacle appears
    out = p.tick(_det(), _status(), front_risk=0.0)  # risk dips again

    assert p.state is State.SEARCH
    assert out.state is State.SEARCH


def test_clear_air_still_locks_on() -> None:
    p = _searching_planner()

    for _ in range(_APPROACH_LOCK_FRAMES):
        out = p.tick(_det(), _status(), front_risk=0.0)

    assert p.state is State.APPROACH
    assert out.state is State.APPROACH


def test_front_risk_defaults_to_clear_for_existing_callers() -> None:
    """tick() without front_risk keeps its old behaviour."""
    p = _searching_planner()

    for _ in range(_APPROACH_LOCK_FRAMES):
        p.tick(_det(), _status())

    assert p.state is State.APPROACH
