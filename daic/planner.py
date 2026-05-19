"""Behavior state machine and search planner for daic.

States
------
IDLE        AI disabled or not yet connected.
ARMING      Sent arm command, waiting for confirmation.
TAKEOFF     Legacy/manual state; transitions to search without changing height.
SEARCH      No target visible; executing expanding-square search pattern.
APPROACH    Target visible; visual-servo toward it.
LANDING     Target centred and low enough; descending to land.
COMPLETE    Landed successfully.
FAILSAFE    Something went wrong; hovering or landing safely.

The planner is pure logic (no I/O).  The main loop reads its output and sends
the appropriate velocity / control commands.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from dvision2_common import gps_bearing, gps_distance_m
from .controller import (ControlOutput, hover, navigate_to_bearing,
                         servo, search_step, turn, estimate_horiz_dist)
from .detector import Detection


class State(Enum):
    IDLE      = auto()
    ARMING    = auto()
    TAKEOFF   = auto()
    SEARCH    = auto()
    APPROACH  = auto()
    LANDING   = auto()
    COMPLETE  = auto()
    FAILSAFE  = auto()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEARCH_ALT_M    = 3.0   # nominal map/test altitude; daic does not climb to it
TAKEOFF_ALT_M   = 3.0   # legacy constant; daic no longer commands takeoff by default

# How many consecutive frames the target must be visible before transitioning
# SEARCH → APPROACH and APPROACH → LANDING.
_APPROACH_LOCK_FRAMES  = 5
_LANDING_LOCK_FRAMES   = 10

# Seconds without a valid detection before giving up and acting on the loss.
# During this window the drone hovers in place and waits for reacquisition.
_LOST_TARGET_TIMEOUT_S = 2.5

# Centring tolerance for landing gate (pixels).
_LAND_CENTRE_TOL_PX = 30.0
_LAND_MAX_HORIZ_M = 2.0
_LAND_LOW_FRAME_HORIZ_M = 3.2
_LAND_LOW_FRAME_CY_NORM = 0.88

# If the target disappears immediately after reaching the bottom of the image,
# the drone has likely flown over it.  Continuing on the stale detection would
# push it farther past the target.
_BOTTOM_LOSS_CY_NORM = 0.85

# Altitude below which landing is considered complete.
_LAND_COMPLETE_ALT_M = 0.25

# Max seconds to wait for arm / takeoff confirmation before FAILSAFE.
_ARM_TIMEOUT_S    = 5.0
_TAKEOFF_TIMEOUT_S = 15.0

# Stale-status timeout.
_STATUS_STALE_S = 2.0

# Battery threshold for forced landing.
_LOW_BATTERY_PCT = 15.0

# Search pattern: expanding square.
# The drone flies each leg at a constant heading, then turns 90°, then
# increases leg length after two turns.
_SEARCH_TURN_DPS  = 45.0
_SEARCH_TURN_DEG  = 90.0
_SEARCH_LEG_S_BASE = 3.0   # seconds per leg at initial size
_SEARCH_LEG_S_INC  = 3.0   # extra seconds added every two turns


@dataclass
class PlannerOutput:
    """Everything the main loop needs from one planner tick."""
    state: State
    command_type: str          # "velocity", "arm", "takeoff", "land", "zero", "heartbeat"
    command_fields: dict       # extra fields for encode_command
    status_text: str           # human-readable, shown in UI
    send_command: bool = True  # False = nothing to send this tick


@dataclass
class _SearchState:
    leg_index: int   = 0
    leg_start: float = 0.0    # monotonic time
    leg_duration: float = _SEARCH_LEG_S_BASE
    heading_target: float = 270.0  # degrees; matches dsim default start heading
    turning: bool = False


class Planner:
    def __init__(self, img_w: int = 640, img_h: int = 480) -> None:
        self.img_w = img_w
        self.img_h = img_h
        self._state = State.IDLE
        self._state_entered: float = time.monotonic()
        self._approach_count = 0
        self._landing_count  = 0
        self._target_last_seen: float = 0.0   # monotonic; 0 = not yet seen
        self._last_valid_detection = Detection(False, 0.0, 0.0, 0.0, 0.0)
        self._last_d_horiz: float | None = None  # last reliable distance estimate
        self._search = _SearchState()
        self._status_text = "idle"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def state(self) -> State:
        return self._state

    def enable(self, status: dict) -> None:
        """Called when the operator enables AI control."""
        if self._state == State.IDLE:
            self._transition(State.ARMING)

    def disable(self) -> PlannerOutput:
        """Called when the operator disables AI control.  Returns a zero command."""
        self._transition(State.IDLE)
        return PlannerOutput(state=State.IDLE, command_type="zero",
                             command_fields={}, status_text="AI disabled")

    def tick(self, detection: Detection, status: dict) -> PlannerOutput:
        """Advance the state machine one step.

        *status* is the dict returned by memkv.getAll(); missing keys default
        to safe values so the planner can run even before all fields arrive.
        """
        now = time.monotonic()

        # ── Global failsafe checks ────────────────────────────────────
        if self._state not in (State.IDLE, State.COMPLETE, State.FAILSAFE):
            fail_reason = self._check_failsafes(status, now)
            if fail_reason:
                self._transition(State.FAILSAFE)
                self._status_text = f"failsafe: {fail_reason}"
                return PlannerOutput(state=State.FAILSAFE,
                                     command_type="zero", command_fields={},
                                     status_text=self._status_text)

        # ── Per-state logic ───────────────────────────────────────────
        s = self._state

        if s == State.IDLE:
            return PlannerOutput(state=s, command_type="heartbeat",
                                 command_fields={}, status_text="idle",
                                 send_command=False)

        if s == State.ARMING:
            elapsed = now - self._state_entered
            if elapsed < 0.2:
                return PlannerOutput(state=s, command_type="arm",
                                     command_fields={"armed": True},
                                     status_text="arming…")
            if _armed(status):
                self._transition(State.SEARCH)
                self._search = _SearchState(heading_target=_heading(status))
                return PlannerOutput(state=State.SEARCH,
                                     command_type="zero", command_fields={},
                                     status_text="armed, starting search")
            if elapsed > _ARM_TIMEOUT_S:
                self._transition(State.FAILSAFE)
                self._status_text = "failsafe: arm timeout"
                return PlannerOutput(state=State.FAILSAFE,
                                     command_type="zero", command_fields={},
                                     status_text=self._status_text)
            return PlannerOutput(state=s, command_type="heartbeat",
                                 command_fields={}, status_text="waiting for arm…",
                                 send_command=False)

        if s == State.TAKEOFF:
            self._transition(State.SEARCH)
            self._search = _SearchState(heading_target=_heading(status))
            return PlannerOutput(state=State.SEARCH,
                                 command_type="zero", command_fields={},
                                 status_text="starting search")

        if s == State.SEARCH:
            return self._search_tick(detection, status, now)

        if s == State.APPROACH:
            return self._approach_tick(detection, status, now)

        if s == State.LANDING:
            return self._landing_tick(detection, status, now)

        if s == State.COMPLETE:
            return PlannerOutput(state=s, command_type="heartbeat",
                                 command_fields={}, status_text="landed ✓",
                                 send_command=False)

        # FAILSAFE
        return PlannerOutput(state=s, command_type="zero",
                             command_fields={}, status_text=self._status_text)

    # ------------------------------------------------------------------
    # Per-state helpers
    # ------------------------------------------------------------------

    def _search_tick(self, detection: Detection,
                     status: dict, now: float) -> PlannerOutput:
        if detection.visible and detection.confidence > 0.4:
            self._approach_count += 1
            if self._approach_count >= _APPROACH_LOCK_FRAMES:
                self._approach_count = 0
                self._transition(State.APPROACH)
                # Send zero first to shed search-pattern velocity before
                # the servo takes over.  Next tick will begin servo.
                return PlannerOutput(state=State.APPROACH,
                                     command_type="zero", command_fields={},
                                     status_text="target locked, stopping")
        else:
            self._approach_count = 0

        # GPS-guided navigation when target coordinates are known.
        gps_ctrl = self._gps_nav_to_target(status)
        if gps_ctrl is not None:
            dist = _gps_dist_to_target(status)
            return PlannerOutput(state=State.SEARCH,
                                 command_type="velocity",
                                 command_fields=gps_ctrl.as_command_fields(),
                                 status_text=f"GPS nav {dist:.0f} m to target")

        ctrl = self._expanding_square(now)
        return PlannerOutput(state=State.SEARCH,
                             command_type="velocity",
                             command_fields=ctrl.as_command_fields(),
                             status_text=f"searching (leg {self._search.leg_index})")

    def _approach_tick(self, detection: Detection,
                       status: dict, now: float) -> PlannerOutput:
        if not detection.visible or detection.confidence < 0.35:
            self._approach_count = 0
            lost_s = now - self._target_last_seen
            if lost_s < _LOST_TARGET_TIMEOUT_S:
                last_cy_norm = self._last_valid_detection.cy / max(float(self.img_h), 1.0)
                if last_cy_norm >= _BOTTOM_LOSS_CY_NORM:
                    self._transition(State.SEARCH)
                    self._search = _SearchState(heading_target=_heading(status))
                    return PlannerOutput(state=State.SEARCH,
                                         command_type="zero", command_fields={},
                                         status_text="target passed under camera, reacquiring")
                # Keep flying toward the last known position rather than stopping.
                alt  = _alt(status)
                ctrl = servo(self._last_valid_detection, self.img_w, self.img_h, alt)
                return PlannerOutput(
                    state=State.APPROACH,
                    command_type="velocity",
                    command_fields=ctrl.as_command_fields(),
                    status_text=f"target lost, continuing "
                                f"{lost_s:.1f}s / {_LOST_TARGET_TIMEOUT_S:.0f}s",
                )
            # Timeout exhausted — resume search.
            self._transition(State.SEARCH)
            self._search = _SearchState(heading_target=_heading(status))
            return PlannerOutput(state=State.SEARCH,
                                 command_type="zero", command_fields={},
                                 status_text="target lost, resuming search")

        self._target_last_seen = now
        self._last_valid_detection = detection

        alt = _alt(status)

        # Update the last reliable horizontal-distance estimate.
        # estimate_horiz_dist returns 0 when the target is too close/large;
        # we keep the previous value so the trajectory correction stays active.
        d_est = estimate_horiz_dist(detection.radius, alt, self.img_w)
        if 0.05 < d_est < 20.0:
            self._last_d_horiz = d_est

        ctrl = servo(detection, self.img_w, self.img_h, alt,
                     d_horiz_override=self._last_d_horiz)

        # Ground-level shortcut: if the drone is essentially on the ground and
        # the target is still visible, commit to landing immediately rather than
        # waiting for centring counters to accumulate.
        if alt <= _LAND_COMPLETE_ALT_M and detection.visible:
            self._transition(State.COMPLETE)
            return PlannerOutput(state=State.COMPLETE, command_type="land",
                                 command_fields={}, status_text="landed ✓")

        # Gate to LANDING: lateral centring only (cx_err).
        # With a forward-facing camera, cy_err represents depth/distance,
        # not lateral misalignment — a large cy_err just means the target is
        # close below you, which is exactly when landing should proceed.
        cx_err = abs(detection.cx - self.img_w / 2.0)
        centred = cx_err < _LAND_CENTRE_TOL_PX
        cy_norm = detection.cy / max(float(self.img_h), 1.0)
        low_in_frame = cy_norm >= _LAND_LOW_FRAME_CY_NORM

        landing_ready = (
            (ctrl.descending and ctrl.horiz_dist_m <= _LAND_MAX_HORIZ_M)
            or (low_in_frame and ctrl.horiz_dist_m <= _LAND_LOW_FRAME_HORIZ_M)
        )
        if centred and (landing_ready or alt < 0.5):
            self._landing_count += 1
            if self._landing_count >= _LANDING_LOCK_FRAMES:
                self._landing_count = 0
                self._transition(State.LANDING)
                return self._landing_tick(detection, status, now)
        else:
            self._landing_count = 0

        cx_disp = detection.cx - self.img_w / 2.0
        cy_disp = detection.cy - self.img_h / 2.0
        return PlannerOutput(state=State.APPROACH,
                             command_type="velocity",
                             command_fields=ctrl.as_command_fields(),
                             status_text=f"approach "
                                         f"Δx={cx_disp:.0f} Δy={cy_disp:.0f} "
                                         f"r={detection.radius:.0f} "
                                         f"z={alt:.1f}m")

    def _landing_tick(self, detection: Detection,
                      status: dict, now: float) -> PlannerOutput:
        alt = _alt(status)

        if alt <= _LAND_COMPLETE_ALT_M:
            self._transition(State.COMPLETE)
            return PlannerOutput(state=State.COMPLETE,
                                 command_type="land", command_fields={},
                                 status_text="landed ✓")

        if not detection.visible:
            lost_s = now - self._target_last_seen
            if lost_s < _LOST_TARGET_TIMEOUT_S:
                # Continue descending toward last known position.
                ctrl = servo(self._last_valid_detection, self.img_w, self.img_h, alt)
                return PlannerOutput(
                    state=State.LANDING,
                    command_type="velocity",
                    command_fields=ctrl.as_command_fields(),
                    status_text=f"landing: target lost, continuing "
                                f"{lost_s:.1f}s / {_LOST_TARGET_TIMEOUT_S:.0f}s",
                )
            # Timeout exhausted — climb to reacquire.
            return PlannerOutput(state=State.LANDING,
                                 command_type="velocity",
                                 command_fields={"forward_mps": 0.0,
                                                 "right_mps": 0.0,
                                                 "up_mps": 2.0,
                                                 "yaw_rate_dps": 0.0},
                                 status_text=f"landing: target lost, climbing ({alt:.1f} m)")

        self._target_last_seen = now
        self._last_valid_detection = detection
        d_est = estimate_horiz_dist(detection.radius, alt, self.img_w)
        if 0.05 < d_est < 20.0:
            self._last_d_horiz = d_est
        ctrl = servo(detection, self.img_w, self.img_h, alt,
                     d_horiz_override=self._last_d_horiz)
        return PlannerOutput(state=State.LANDING,
                             command_type="velocity",
                             command_fields=ctrl.as_command_fields(),
                             status_text=f"landing… {alt:.1f} m")

    # ------------------------------------------------------------------
    # GPS-guided navigation
    # ------------------------------------------------------------------

    def _gps_nav_to_target(self, status: dict) -> ControlOutput | None:
        """Return a velocity command toward the target's GPS position, or None
        if GPS coordinates are unavailable or the target is visually close."""
        d_lat = _try_float(status.get("drone.lat_deg"))
        d_lon = _try_float(status.get("drone.lon_deg"))
        t_lat = _try_float(status.get("target.lat_deg"))
        t_lon = _try_float(status.get("target.lon_deg"))
        if None in (d_lat, d_lon, t_lat, t_lon):
            return None
        if t_lat == 0.0 and t_lon == 0.0:
            return None
        dist_m = gps_distance_m(d_lat, d_lon, t_lat, t_lon)
        if dist_m < 1.5:
            # Close enough for visual servo; fall through to expanding square.
            return None
        bearing = gps_bearing(d_lat, d_lon, t_lat, t_lon)
        # drone.compass_deg is 0=north, 90=east (true compass heading).
        compass = _try_float(status.get("drone.compass_deg"))
        if compass is None:
            # Fallback: convert sim yaw to compass bearing.
            compass = (_heading(status) + 90.0) % 360.0
        yaw_error = (bearing - compass + 360.0) % 360.0
        return navigate_to_bearing(yaw_error, dist_m)

    # ------------------------------------------------------------------
    # Expanding square search
    # ------------------------------------------------------------------

    def _expanding_square(self, now: float) -> ControlOutput:
        sq = self._search
        elapsed = now - sq.leg_start

        if sq.turning:
            if elapsed >= _SEARCH_TURN_DEG / _SEARCH_TURN_DPS:
                sq.turning = False
                sq.leg_start = now
                elapsed = 0.0
            else:
                return turn(_SEARCH_TURN_DPS)

        if elapsed >= sq.leg_duration:
            sq.leg_index += 1
            sq.turning = True
            sq.leg_start = now
            sq.heading_target = (sq.heading_target + 90.0) % 360.0
            # Increase leg length every two turns.
            if sq.leg_index % 2 == 0:
                sq.leg_duration += _SEARCH_LEG_S_INC
            return turn(_SEARCH_TURN_DPS)

        return search_step(sq.heading_target)

    # ------------------------------------------------------------------
    # Failsafe checks
    # ------------------------------------------------------------------

    def _check_failsafes(self, status: dict, now: float) -> str | None:
        last_s = _last_status_age(status, now)
        if last_s > _STATUS_STALE_S:
            return "status stale"
        batt = _battery(status)
        if batt < _LOW_BATTERY_PCT:
            return f"low battery {batt:.0f}%"
        return None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _transition(self, new_state: State) -> None:
        now = time.monotonic()
        self._state = new_state
        self._state_entered = now
        self._approach_count  = 0
        self._landing_count   = 0
        # _last_d_horiz is intentionally NOT reset here so the trajectory
        # correction carries the last good estimate across state transitions.
        self._target_last_seen = now   # full timeout budget from state entry


# ---------------------------------------------------------------------------
# Status dict helpers
# ---------------------------------------------------------------------------

def _try_float(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None

def _gps_dist_to_target(s: dict) -> float:
    d_lat = _try_float(s.get("drone.lat_deg"))
    d_lon = _try_float(s.get("drone.lon_deg"))
    t_lat = _try_float(s.get("target.lat_deg"))
    t_lon = _try_float(s.get("target.lon_deg"))
    if None in (d_lat, d_lon, t_lat, t_lon):
        return 0.0
    return gps_distance_m(d_lat, d_lon, t_lat, t_lon)

def _armed(s: dict) -> bool:
    return s.get("drone.armed", "0") == "1"

def _mode(s: dict) -> str:
    return s.get("drone.mode", "DISARMED")

def _alt(s: dict) -> float:
    try:
        return float(s.get("drone.z_m", 0.0))
    except (ValueError, TypeError):
        return 0.0

def _heading(s: dict) -> float:
    try:
        return float(s.get("drone.heading_deg", 270.0))
    except (ValueError, TypeError):
        return 270.0

def _battery(s: dict) -> float:
    try:
        return float(s.get("drone.battery_pct", 100.0))
    except (ValueError, TypeError):
        return 100.0

def _last_status_age(s: dict, now: float) -> float:
    """Seconds since the status dict was last updated."""
    if s.get("_stale") == "1":
        return _STATUS_STALE_S + 1.0
    return 0.0
