"""Waypoint sequencing, arrival rules, control strategies, and the lifecycle.

Sequencing lives off-vehicle: publish the current target continuously, advance
only on arrival. The working frame is local NED throughout, because that is
the frame a real autopilot uses and it makes the follower independent of the
frame a tour happened to be authored in.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from dway.link import (
    CommandResult, Frame, PositionTarget, VehicleCapabilities, VehicleState,
    VelocityTarget,
)
from dway.tour import FrameContext, Tour, Waypoint

#: Above this height the vehicle is treated as already airborne and a tour
#: starts by flying rather than by climbing.
AIRBORNE_MIN_M = 0.3
#: Lowest altitude a takeoff may be commanded to; vehicles refuse less.
MIN_TAKEOFF_ALT_M = 0.5

# Velocity-backend gains. The horizontal gain matches the proportional
# approach a position-capable vehicle runs onboard, so the fallback converges
# the same way rather than with a second, differently tuned character. The
# trim term matches it too: proportional alone parks the vehicle at an offset
# of exactly the disturbance divided by the gain, which in any wind at all
# means it hovers downwind of the waypoint and never satisfies the arrival
# gate. Trim accumulates only near the target and at hover speed -- during an
# approach the error is distance still to travel, not disturbance, and
# integrating it throws the vehicle past the waypoint.
_POSITION_GAIN = 1.0
_POSITION_TRIM_GAIN = 0.5
_MAX_POSITION_TRIM_MPS = 2.0
_POSITION_TRIM_BAND_M = 1.0
_POSITION_TRIM_SPEED_MPS = 0.5
_YAW_GAIN = 2.0
_MAX_YAW_RATE_DPS = 90.0


def wrap_deg(angle: float) -> float:
    """Signed shortest angular difference, in (-180, 180]."""
    return (float(angle) + 180.0) % 360.0 - 180.0


def _clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


@dataclass(frozen=True)
class Sample:
    """One vehicle observation reduced to the follower's working frame.

    ``clock_s`` is mission time, not the wall clock: the follower only ever
    takes differences of it -- a dwell, a leg deadline, a trim step -- and none
    of those should run while the tour is paused.
    """

    north_m: float
    east_m: float
    down_m: float
    heading_deg: float
    speed_mps: float
    clock_s: float

    @classmethod
    def from_state(cls, state: VehicleState, context: FrameContext,
                   now: float) -> "Sample":
        north, east, down = context.position_ned(state.position)
        return cls(north, east, down, state.heading_deg,
                   math.sqrt(state.vx_mps ** 2 + state.vy_mps ** 2
                             + state.vz_mps ** 2), now)


@dataclass(frozen=True)
class Leg:
    """One waypoint, ready to command in the negotiated setpoint frame."""

    index: int
    north_m: float
    east_m: float
    down_m: float
    heading_deg: float
    dwell_s: float
    target: PositionTarget
    waypoint: Waypoint

    def distance_to(self, sample: Sample) -> float:
        return math.dist((self.north_m, self.east_m, self.down_m),
                         (sample.north_m, sample.east_m, sample.down_m))


def build_legs(tour: Tour, context: FrameContext, *, frame: Frame,
               speed_mps: float) -> list[Leg]:
    legs: list[Leg] = []
    for waypoint in tour.waypoints:
        north, east, down = context.waypoint_ned(waypoint)
        legs.append(Leg(
            index=waypoint.index, north_m=north, east_m=east, down_m=down,
            heading_deg=waypoint.heading_deg, dwell_s=waypoint.dwell_s,
            target=context.target_from_ned(north, east, down, frame=frame,
                                           heading_deg=waypoint.heading_deg,
                                           max_speed_mps=speed_mps),
            waypoint=waypoint))
    return legs


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

class StrategyError(RuntimeError):
    """No strategy can fly this tour on this vehicle, and the message says why."""


@dataclass(frozen=True)
class PositionStrategy:
    """Stream position targets and let the vehicle close its own loop."""

    frame: Frame
    name: str = "position"

    def describe(self) -> str:
        return f"position targets in the {self.frame} frame"

    def send(self, link, leg: Leg, sample: Sample, speed_mps: float) -> CommandResult:
        return link.send_position_target(leg.target)


@dataclass
class VelocityStrategy:
    """Close the position loop here and command body-frame velocity.

    Positive ``yaw_rate_dps`` is clockwise and increases compass heading. That
    is the public convention; a vehicle whose internal yaw runs the other way
    inverts it inside its own link, never here.

    The loop carries trim, so this backend holds a waypoint against a steady
    disturbance exactly as a position-capable vehicle does onboard. The trim
    is per-leg: it is the disturbance at *this* waypoint, and carrying it into
    the next leg would be a guess.
    """

    speed_mps: float
    name: str = "velocity"
    _trim_north: float = 0.0
    _trim_east: float = 0.0
    _trim_index: int = -1
    _trim_sample_s: float | None = None

    def describe(self) -> str:
        return "body-frame velocity with an external position loop"

    def send(self, link, leg: Leg, sample: Sample, speed_mps: float) -> CommandResult:
        return link.send_velocity_target(self.velocity_for(leg, sample, speed_mps))

    def _update_trim(self, leg: Leg, sample: Sample, north_err: float,
                     east_err: float) -> None:
        if leg.index != self._trim_index:
            self._trim_index = leg.index
            self._trim_north = self._trim_east = 0.0
            self._trim_sample_s = None
        previous, self._trim_sample_s = self._trim_sample_s, sample.clock_s
        if previous is None:
            return
        dt = sample.clock_s - previous
        if dt <= 0.0:
            return
        if (math.hypot(north_err, east_err) > _POSITION_TRIM_BAND_M
                or sample.speed_mps > _POSITION_TRIM_SPEED_MPS):
            return
        self._trim_north = _clamp(
            self._trim_north + north_err * _POSITION_TRIM_GAIN * dt,
            _MAX_POSITION_TRIM_MPS)
        self._trim_east = _clamp(
            self._trim_east + east_err * _POSITION_TRIM_GAIN * dt,
            _MAX_POSITION_TRIM_MPS)

    def velocity_for(self, leg: Leg, sample: Sample,
                     speed_mps: float) -> VelocityTarget:
        north_err = leg.north_m - sample.north_m
        east_err = leg.east_m - sample.east_m
        down_err = leg.down_m - sample.down_m
        self._update_trim(leg, sample, north_err, east_err)
        v_north = _clamp(north_err * _POSITION_GAIN + self._trim_north, speed_mps)
        v_east = _clamp(east_err * _POSITION_GAIN + self._trim_east, speed_mps)
        horizontal = math.hypot(v_north, v_east)
        if horizontal > speed_mps:
            scale = speed_mps / horizontal
            v_north, v_east = v_north * scale, v_east * scale
        heading = math.radians(sample.heading_deg)
        cos_h, sin_h = math.cos(heading), math.sin(heading)
        return VelocityTarget(
            forward_mps=v_north * cos_h + v_east * sin_h,
            right_mps=-v_north * sin_h + v_east * cos_h,
            up_mps=_clamp(-down_err * _POSITION_GAIN, speed_mps),
            yaw_rate_dps=_clamp(
                wrap_deg(leg.heading_deg - sample.heading_deg) * _YAW_GAIN,
                _MAX_YAW_RATE_DPS),
        )


_FRAME_PREFERENCE: tuple[Frame, ...] = ("map", "local_ned", "global")


def select_setpoint_frame(capabilities: VehicleCapabilities,
                          tour_frame: Frame) -> Frame:
    """The frame position targets go out in: the tour's, or a converted one."""
    accepted = tuple(capabilities.frames)
    if tour_frame in accepted:
        return tour_frame
    for frame in _FRAME_PREFERENCE:
        if frame in accepted:
            return frame
    raise StrategyError(
        f"vehicle accepts no position frame this tour can be converted to "
        f"(vehicle frames: {', '.join(accepted) or 'none'})")


def select_strategy(capabilities: VehicleCapabilities, tour_frame: Frame, *,
                    speed_mps: float, forced: str = "auto"):
    """Choose a control strategy from capabilities, never from vehicle identity.

    The ladder is short on purpose: stream position targets when the vehicle
    accepts them, otherwise close the loop here and command velocity,
    otherwise refuse to fly and say which capability is missing.
    """
    if forced not in ("auto", "position", "velocity"):
        raise StrategyError(f"unknown strategy {forced!r}")
    if forced != "velocity" and capabilities.accepts_position_target:
        return PositionStrategy(select_setpoint_frame(capabilities, tour_frame))
    if forced == "position":
        raise StrategyError("vehicle does not accept position targets")
    if capabilities.accepts_velocity_target:
        return VelocityStrategy(speed_mps)
    raise StrategyError(
        "vehicle accepts neither position nor velocity targets; nothing to fly with")


# ---------------------------------------------------------------------------
# Following
# ---------------------------------------------------------------------------

@dataclass
class WaypointProgress:
    index: int
    first_target_s: float | None = None
    arrival_s: float | None = None
    dwell_s: float = 0.0
    overshoot_m: float = 0.0
    max_cross_track_error_m: float = 0.0


class FollowerEvent(str, Enum):
    NONE = "none"
    ARRIVED = "arrived"
    COMPLETE = "complete"
    LEG_TIMEOUT = "leg_timeout"


def _point_segment_distance(point: tuple[float, float, float],
                            start: tuple[float, float, float],
                            end: tuple[float, float, float]) -> float:
    """Perpendicular distance to the finite planned leg."""
    sx, sy, sz = start
    dx, dy, dz = end[0] - sx, end[1] - sy, end[2] - sz
    length_sq = dx * dx + dy * dy + dz * dz
    if length_sq <= 1e-12:
        return math.dist(point, start)
    t = ((point[0] - sx) * dx + (point[1] - sy) * dy + (point[2] - sz) * dz) / length_sq
    t = max(0.0, min(1.0, t))
    return math.dist(point, (sx + dx * t, sy + dy * t, sz + dz * t))


class Follower:
    """Advance through a tour's legs on arrival, and measure the flying.

    Arrival is not distance alone: distance, total speed and heading error must
    all stay inside their gates continuously for the waypoint's dwell. Leaving
    the gate resets the dwell clock, and a zero-dwell waypoint advances on the
    first in-gate sample rather than being flown through.
    """

    def __init__(self, tour: Tour, legs: Sequence[Leg], *, speed_mps: float) -> None:
        self.tour = tour
        self.legs = list(legs)
        self.speed_mps = float(speed_mps)
        self.index = 0
        self.complete = False
        self.progress = [WaypointProgress(leg.index) for leg in self.legs]
        self.path_length_m = 0.0
        self.max_cross_track_error_m = 0.0
        self._leg_start: tuple[float, float, float] | None = None
        self._leg_started_s: float | None = None
        self._leg_deadline_s: float | None = None
        self._in_gate_since: float | None = None
        self._previous: tuple[float, float, float] | None = None

    # -- geometry ------------------------------------------------------

    @property
    def leg(self) -> Leg | None:
        return None if self.complete else self.legs[self.index]

    def begin_leg(self, sample: Sample) -> None:
        """Anchor the current leg at this pose and start its clocks."""
        leg = self.leg
        if leg is None:
            return
        self._leg_start = (sample.north_m, sample.east_m, sample.down_m)
        self._leg_started_s = sample.clock_s
        self._in_gate_since = None
        distance = leg.distance_to(sample)
        self._leg_deadline_s = sample.clock_s + self.tour.leg_timeout(
            distance, self.speed_mps)
        entry = self.progress[self.index]
        if entry.first_target_s is None:
            entry.first_target_s = sample.clock_s

    def in_gate(self, sample: Sample) -> bool:
        leg = self.leg
        if leg is None:
            return False
        return (leg.distance_to(sample) <= self.tour.waypoint_tolerance_m
                and sample.speed_mps <= self.tour.arrival_speed_mps
                and abs(wrap_deg(leg.heading_deg - sample.heading_deg))
                <= self.tour.heading_tolerance_deg)

    def dwell_remaining_s(self, now: float) -> float | None:
        """Seconds of dwell still owed, or ``None`` when not inside the gate."""
        if self.complete or self._in_gate_since is None:
            return None
        leg = self.legs[self.index]
        return max(0.0, leg.dwell_s - (now - self._in_gate_since))

    # -- stepping ------------------------------------------------------

    def update(self, sample: Sample) -> FollowerEvent:
        if self.complete:
            return FollowerEvent.COMPLETE
        if self._leg_started_s is None:
            self.begin_leg(sample)
        self._measure(sample)

        if self.in_gate(sample):
            if self._in_gate_since is None:
                self._in_gate_since = sample.clock_s
            held = sample.clock_s - self._in_gate_since
            if held >= self.legs[self.index].dwell_s:
                return self._arrive(sample, held)
        else:
            self._in_gate_since = None

        if (self._leg_deadline_s is not None
                and sample.clock_s > self._leg_deadline_s):
            return FollowerEvent.LEG_TIMEOUT
        return FollowerEvent.NONE

    def _measure(self, sample: Sample) -> None:
        point = (sample.north_m, sample.east_m, sample.down_m)
        if self._previous is not None:
            self.path_length_m += math.dist(point, self._previous)
        self._previous = point
        leg = self.legs[self.index]
        entry = self.progress[self.index]
        if self._leg_start is None:
            return
        target = (leg.north_m, leg.east_m, leg.down_m)
        cross_track = _point_segment_distance(point, self._leg_start, target)
        entry.max_cross_track_error_m = max(entry.max_cross_track_error_m, cross_track)
        self.max_cross_track_error_m = max(self.max_cross_track_error_m, cross_track)
        # Overshoot is travel past the waypoint along the incoming leg, so it
        # is the projection onto that leg's direction beyond its end.
        dn = target[0] - self._leg_start[0]
        de = target[1] - self._leg_start[1]
        dd = target[2] - self._leg_start[2]
        length = math.sqrt(dn * dn + de * de + dd * dd)
        if length > 1e-9:
            past = ((point[0] - target[0]) * dn + (point[1] - target[1]) * de
                    + (point[2] - target[2]) * dd) / length
            entry.overshoot_m = max(entry.overshoot_m, past)

    def _arrive(self, sample: Sample, held_s: float) -> FollowerEvent:
        entry = self.progress[self.index]
        entry.arrival_s = sample.clock_s
        entry.dwell_s = held_s
        self.index += 1
        self._leg_start = None
        self._leg_started_s = None
        self._leg_deadline_s = None
        self._in_gate_since = None
        if self.index >= len(self.legs):
            self.complete = True
            return FollowerEvent.COMPLETE
        self.begin_leg(sample)
        return FollowerEvent.ARRIVED
