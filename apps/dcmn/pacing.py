"""How often a window redraws, and nothing else.

Repaint rate is the one thing in dvision2 that is genuinely about the person
watching rather than about the vehicle, so it is the one thing measured on the
wall clock in every mode. A simulation running at ten times real time still
only has one operator, and that operator resolves no more than about thirty
frames of video a second and far less text than that.

The caps live here rather than in each window because four of them need the
same answer, and because a rate limiter beside the palette and the map view is
where a reader looks for "what every view shares".

**These pace painting, never control.** In `dctl` and `daic` the redraw and the
control step were originally the same timer, so capping the timer would have
capped the control loop with it. Each window splits the two first and applies
these to the painting half only.
"""

from __future__ import annotations

import time
import math
from typing import Callable, Mapping

#: Video panes. Beyond this an operator sees no additional motion, and every
#: frame costs a colour conversion, a resize and a Tk image upload.
VIDEO_HZ = 30.0

#: Maps and overlays. They cost more than the whole control path and change
#: slowly enough that ten a second looks continuous.
MAP_HZ = 10.0

#: Text, telemetry and status lines. Numbers replaced thirty times a second
#: are not readable; four is, and it is still faster than anyone reacts.
TEXT_HZ = 4.0


def simulation_speed(values: Mapping[str, object] | None) -> float:
    """Best available simulated-seconds-per-wall-second estimate."""
    values = values or {}
    try:
        requested = float(values.get("sim.speed", 1.0) or 0.0)
    except (TypeError, ValueError):
        requested = 1.0
    if requested > 0.0:
        return requested
    try:
        achieved = float(values.get("sim.speed_achieved", 1.0) or 1.0)
    except (TypeError, ValueError):
        achieved = 1.0
    return achieved if achieved > 0.0 else 1.0


def simulated_poll_delay(rate_hz: float, values: Mapping[str, object] | None,
                         *, checks_per_period: float = 2.0,
                         minimum_s: float = 0.002,
                         maximum_s: float = 0.25) -> float:
    """Return a bounded wall delay for checking simulated-time work.

    This only schedules a check. The vehicle clock remains authoritative for
    deciding whether work is due, so a stale speed estimate cannot send early.
    """
    rate_hz = float(rate_hz)
    checks_per_period = float(checks_per_period)
    if rate_hz <= 0.0 or checks_per_period <= 0.0:
        raise ValueError("rate and checks_per_period must be positive")
    delay = 1.0 / (rate_hz * checks_per_period * simulation_speed(values))
    return max(float(minimum_s), min(float(maximum_s), delay))


class PeriodicDeadline:
    """A periodic schedule whose phase does not drift after a late check.

    The caller supplies the clock, so this works for both simulated-time work
    and wall-time loops. ``advance`` skips elapsed slots instead of producing
    a catch-up burst and, crucially, advances from the previous deadline rather
    than from when the work happened to finish.
    """

    def __init__(self, rate_hz: float) -> None:
        self.deadline: float | None = None
        self.set_rate(rate_hz)

    def set_rate(self, rate_hz: float) -> None:
        rate_hz = float(rate_hz)
        if rate_hz <= 0.0:
            raise ValueError("rate must be positive")
        self.period_s = 1.0 / rate_hz

    def reset(self, now: float, *, immediate: bool = True) -> None:
        now = float(now)
        self.deadline = now if immediate else now + self.period_s

    def due(self, now: float) -> bool:
        return self.deadline is None or float(now) + 1e-9 >= self.deadline

    def advance(self, now: float) -> int:
        """Advance past ``now`` and return the number of skipped slots."""
        now = float(now)
        if self.deadline is None:
            self.deadline = now + self.period_s
            return 0
        if not self.due(now):
            return 0
        elapsed = max(0.0, now - self.deadline + 1e-9)
        steps = int(elapsed / self.period_s) + 1
        self.deadline += steps * self.period_s
        return steps - 1

    def delay(self, now: float, *, maximum_s: float | None = None) -> float:
        delay = 0.0 if self.deadline is None else max(
            0.0, self.deadline - float(now))
        return delay if maximum_s is None else min(delay, float(maximum_s))

    def delay_ms(self, now: float, *, minimum_ms: int = 1) -> int:
        """Whole milliseconds for GUI timers, rounded away from early firing.

        Truncating 33.33 ms to 33 lets a callback execute real work before its
        deadline and then execute again for the remaining fraction, defeating
        the cap. Ceiling preserves the deadline's no-earlier-than contract.
        """
        return max(int(minimum_ms), math.ceil(self.delay(now) * 1000.0))


class Paced:
    """A wall-clock rate limiter for one repaint surface.

    ``due()`` answers "has enough real time passed to paint again?" and marks
    the surface painted when it says yes, so the caller is a plain ``if``.
    Nothing here reads simulated time: a window is not part of the flight.
    """

    def __init__(self, hz: float,
                 clock: Callable[[], float] = time.monotonic) -> None:
        if hz <= 0.0:
            raise ValueError("repaint rate must be positive")
        self.period_s = 1.0 / float(hz)
        self._clock = clock
        # Far enough back that the first call always paints: a window that
        # waited a frame before showing anything would look broken on startup.
        self._painted_s = -1e9

    def due(self) -> bool:
        now = self._clock()
        if now - self._painted_s < self.period_s:
            return False
        self._painted_s = now
        return True

    def reset(self) -> None:
        """Force the next :meth:`due` to paint, whatever the interval says."""
        self._painted_s = -1e9
