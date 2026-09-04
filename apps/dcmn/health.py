"""Whether a module is keeping up, and how anyone would know.

Two things can fall behind. The simulator can fail to advance simulated time as
fast as it was asked to, which is a fact about the simulator. A client can fail
to sample as often as it meant to, which is a fact about that client. Both are
reported the same way -- what was wanted against what was achieved -- so one
indicator can speak for the whole pipeline.

**Nothing here ever gates anything.** The simulator publishes and moves on, at
any speed, and a module that cannot keep up drops samples. That is the contract
a real vehicle needs. These numbers exist so the shortfall is visible rather
than silent; the moment one of them could hold the simulator back it would be
the barrier the design deliberately does not have.

Sensor and control-work rates are in *simulated* Hz. A module that samples five
times a simulated second is doing its job whether that second took one wall
second or a tenth of one. Operator-facing work that is deliberately paced for
a person instead declares a ``wall`` basis; comparing that UI with simulated
time would report a fast run as a broken one.
"""

from __future__ import annotations

from typing import Any

#: At or above this fraction of the intended rate, a module is keeping up.
#: The figure is not new: ``daic.flight_log`` already uses it to decide, after a
#: run, whether the host dropped frames. This makes the same judgement live.
GOOD_RATIO = 0.9

#: Below this, the shortfall is bad enough that a measurement built from it
#: should not be trusted without saying so.
POOR_RATIO = 0.5

OK, WARN, BAD, UNKNOWN = "ok", "warn", "bad", "unknown"

#: Worst-first, so aggregating a pipeline is a max over this order.
SEVERITY = {UNKNOWN: 0, OK: 1, WARN: 2, BAD: 3}


def grade(achieved_hz: float | None, wanted_hz: float | None) -> str:
    """How well a rate met its intention, as ``ok``/``warn``/``bad``.

    ``unknown`` when there is nothing to compare -- a module that has not
    declared an intended rate is not thereby failing to meet it.
    """
    if not wanted_hz or wanted_hz <= 0.0 or achieved_hz is None:
        return UNKNOWN
    ratio = achieved_hz / wanted_hz
    if ratio >= GOOD_RATIO:
        return OK
    return WARN if ratio >= POOR_RATIO else BAD


def worst(grades) -> str:
    """The grade a single indicator should show for a whole pipeline."""
    found = [g for g in grades if g in SEVERITY]
    if not found:
        return UNKNOWN
    return max(found, key=lambda g: SEVERITY[g])


class SteadyGrade:
    """A grade that must repeat before it is believed.

    A bare threshold flickers whenever a module sits on the boundary, and an
    indicator that changes colour twice a second is worse than none: it trains
    the operator to ignore it. A change has to hold for ``runs`` samples before
    it is shown, so a single slow tick does not repaint the header.
    """

    def __init__(self, runs: int = 3, initial: str = UNKNOWN) -> None:
        self.runs = max(1, int(runs))
        self.value = initial
        self._candidate = initial
        self._seen = 0

    def update(self, observed: str) -> str:
        if observed == self.value:
            self._candidate, self._seen = observed, 0
            return self.value
        if observed != self._candidate:
            self._candidate, self._seen = observed, 1
        else:
            self._seen += 1
        if self._seen >= self.runs:
            self.value = observed
            self._seen = 0
        return self.value


class IntakeMeter:
    """What one module meant to sample, and what it managed.

    The module counts its own work -- one ``record()`` per loop it completed or
    frame it took -- and closes a window each time it reports. The window is
    measured against the declared clock basis. Most modules use simulated time;
    a deliberately wall-paced operator UI may use wall time.
    """

    def __init__(self, wanted_hz: float = 0.0, *, basis: str = "sim") -> None:
        if basis not in ("sim", "wall"):
            raise ValueError("intake clock basis must be 'sim' or 'wall'")
        self.wanted_hz = float(wanted_hz)
        self.basis = basis
        self.achieved_hz: float | None = None
        self.skipped = 0            #: frames published while we were not looking
        self.total_skipped = 0
        self._events = 0
        self._window_start_s: float | None = None
        self._last_seq: int | None = None

    def set_wanted(self, wanted_hz: float) -> None:
        """Update the intended rate, for a module whose target can change."""
        self.wanted_hz = float(wanted_hz)

    def record(self, count: int = 1) -> None:
        self._events += int(count)

    def note_sequence(self, seq: int) -> None:
        """Count frames that went past between two reads of the video ring.

        Every consumer already tracks the last sequence number it handled, so
        the gap is free: it is exactly the frames published while this module
        was busy elsewhere. A ring that restarts -- a new simulator -- goes
        backwards, which is not a gap.
        """
        if self._last_seq is not None and seq > self._last_seq + 1:
            missed = seq - self._last_seq - 1
            self.skipped += missed
            self.total_skipped += missed
        self._last_seq = seq

    def report(self, sim_time_s: float, *, overruns: int = 0) -> dict[str, Any]:
        """Close the window and describe it; safe to call on a frozen clock."""
        start, self._window_start_s = self._window_start_s, sim_time_s
        elapsed = None if start is None else sim_time_s - start
        # A clock that has not moved is not an interval, and a clock that has
        # gone backwards is a restarted simulator. Either way the previous
        # achieved rate stands rather than a division nobody can defend.
        if elapsed is not None and elapsed > 0.0:
            self.achieved_hz = self._events / elapsed
        self._events = 0
        skipped, self.skipped = self.skipped, 0
        return {
            "basis": self.basis,
            "wanted_hz": round(self.wanted_hz, 4),
            "achieved_hz": (None if self.achieved_hz is None
                            else round(self.achieved_hz, 4)),
            "skipped": skipped,
            "overruns": int(overruns),
            "grade": grade(self.achieved_hz, self.wanted_hz),
        }


#: What a module that reports nothing looks like.
UNREPORTED: dict[str, Any] = {"basis": "sim", "wanted_hz": None,
                              "achieved_hz": None,
                              "skipped": 0, "overruns": 0, "grade": UNKNOWN}


def describe(block: Any) -> dict[str, Any]:
    """Normalise an ``intake`` block, tolerating a module that sends none.

    Modules are versioned separately and a pipeline may hold one that predates
    this, so a missing or malformed block is ``unknown`` rather than an error.
    """
    if not isinstance(block, dict):
        return dict(UNREPORTED)
    achieved = block.get("achieved_hz")
    wanted = block.get("wanted_hz")
    return {
        "basis": block.get("basis") if block.get("basis") in ("sim", "wall")
                 else "sim",
        "wanted_hz": wanted,
        "achieved_hz": achieved,
        "skipped": int(block.get("skipped", 0) or 0),
        "overruns": int(block.get("overruns", 0) or 0),
        "grade": block.get("grade") or grade(achieved, wanted),
    }
