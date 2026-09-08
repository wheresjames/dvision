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


#: A sensor whose first period has not had time to pass yet. Deliberately not
#: in ``SEVERITY``: a module that has only just connected is not failing, and
#: aggregating it as a grade would paint every startup red for a second.
STARTING = "starting"

#: How late a sample may be, in multiples of its own period, before lateness
#: is worth reporting on its own. A sensor at its configured rate is always
#: within one; three absorbs a scheduling boundary, ten is a stopped stream.
_LATE_PERIODS = 3.0
_STALLED_PERIODS = 10.0


class SensorIntake:
    """What one module takes in from each sensor it actually uses.

    This is the consumer half of the sensor health contract. It is not the
    same measurement as :class:`IntakeMeter`, which says whether a module
    completed its own work often enough: a module can run its loop at full
    rate while one of its cameras has stopped arriving, and one number cannot
    say both. Nor is it derivable from the provider's production counts --
    that is the whole point, because a producer that is publishing perfectly
    is exactly what a stalled reader looks like from the other side.

    Rates and ages are in *simulated* time, because sample cadence is a
    property of the data; module liveness is a wall-clock question and is
    answered elsewhere.
    """

    def __init__(self) -> None:
        self._inputs: dict[str, dict[str, Any]] = {}

    def follow(self, subscriptions) -> None:
        """Declare the complete set of sensors in use, replacing the last set.

        A sensor that disappears from a new generation's manifest disappears
        from the report with it, rather than lingering as a permanent fault
        for a stream nobody is subscribed to any more.
        """
        wanted = {}
        for subscription in subscriptions:
            sensor_id = subscription["sensor_id"]
            entry = self._inputs.get(sensor_id)
            generation = subscription.get("generation")
            identity = (subscription.get('provider_session_id'), generation,
                        subscription.get('reset_epoch'))
            if entry is None or entry.get('identity') != identity:
                # A new generation is a new stream: sequences restart and the
                # rate window means nothing across the boundary.
                entry = _fresh_input(generation)
                entry['identity'] = identity
            entry.update(required=bool(subscription.get("required", True)),
                         expected_hz=subscription.get("expected_hz"),
                         sync_group=subscription.get("sync_group"))
            wanted[sensor_id] = entry
        self._inputs = wanted

    def track(self, *handles, required: bool = True) -> None:
        """Declare and sample every open sensor handle, in one call.

        A handle is anything exposing ``subscription()`` and ``observation()``
        -- :class:`dcmn.sensors.SensorVideo` is the one every consumer holds.
        Call it wherever frames are polled rather than once a report: the
        observed rate is counted from the samples actually admitted, so
        sampling it on the reporting cadence would measure the report.
        """
        subscriptions, observations = [], []
        for handle in handles:
            if handle is None:
                continue
            subscription = handle.subscription(required=required)
            if subscription is None:
                continue
            subscriptions.append(subscription)
            observations.append(handle.observation())
        self.follow(subscriptions)
        for observation in observations:
            if observation is not None:
                self.observe(observation)

    def observe(self, observation) -> None:
        """Note one sample that this module actually took in.

        Sequences are per sensor and per generation, so a repeat is ignored
        and a gap is counted: those are records published while this module
        was busy elsewhere, which is a different number from the transport's
        own overrun count and is kept separately.
        """
        entry = self._inputs.get(observation["sensor_id"])
        if entry is None:
            return
        sequence = int(observation["sequence"])
        if entry["last_sequence"] is not None:
            if sequence <= entry["last_sequence"]:
                return
            entry["skipped"] += sequence - entry["last_sequence"] - 1
        entry["last_sequence"] = sequence
        entry["last_sim_time_s"] = float(observation["sim_time_s"])
        entry["capture_id"] = observation.get("capture_id")
        entry["overruns"] = int(observation.get("overruns", 0) or 0)
        entry["drops"] = int(observation.get("drops", 0) or 0)
        entry['cache_bytes'] = int(observation.get('cache_bytes', 0) or 0)
        entry["events"] += 1

    def report(self, sim_now: float) -> dict[str, dict[str, Any]]:
        """Close the window and describe every declared input."""
        groups: dict[str, set] = {}
        for entry in self._inputs.values():
            if entry["sync_group"] and entry["capture_id"] is not None:
                groups.setdefault(entry["sync_group"], set()).add(entry["capture_id"])
        out = {}
        for sensor_id, entry in self._inputs.items():
            out[sensor_id] = _close(entry, sim_now, groups)
        return out

    def grade(self) -> str:
        """The worst state among the inputs this module says it requires."""
        return worst(entry["state"] for entry in self._inputs.values()
                     if entry["required"])


def _fresh_input(generation) -> dict[str, Any]:
    return {"generation": generation, "required": True, "expected_hz": None,
            "sync_group": None, "last_sequence": None, "last_sim_time_s": None,
            "capture_id": None, "skipped": 0, "overruns": 0, "drops": 0,
            "events": 0, "window_start_s": None, "observed_hz": None, "cache_bytes": 0,
            "state": STARTING, "started_s": None}


def _close(entry: dict[str, Any], sim_now: float, groups) -> dict[str, Any]:
    start, entry["window_start_s"] = entry["window_start_s"], sim_now
    if entry["started_s"] is None:
        entry["started_s"] = sim_now
    if start is not None and sim_now > start:
        entry["observed_hz"] = entry["events"] / (sim_now - start)
    entry["events"] = 0
    skipped, entry["skipped"] = entry["skipped"], 0
    expected = entry["expected_hz"]
    age = (None if entry["last_sim_time_s"] is None
           else max(0.0, sim_now - entry["last_sim_time_s"]))
    sync = "alone"
    if entry["sync_group"]:
        captures = groups.get(entry["sync_group"], set())
        sync = "diverged" if len(captures) > 1 else "ok"
    entry["state"] = _state(entry, age, expected, sim_now, sync)
    return {"generation": entry["generation"], "required": entry["required"],
            "expected_hz": expected,
            "observed_hz": (None if entry["observed_hz"] is None
                            else round(entry["observed_hz"], 3)),
            "last_sequence": entry["last_sequence"],
            "age_s": None if age is None else round(age, 4),
            "skipped": skipped, "overruns": entry["overruns"],
            "drops": entry["drops"], "cache_bytes": entry['cache_bytes'],
            "sync": sync, "state": entry["state"]}


def _state(entry, age, expected, sim_now, sync) -> str:
    """Whether an input is healthy, still starting, or failing, and why.

    Rate alone is not enough: a stream that stopped a moment ago still
    averages its configured rate over the window it stopped in, so lateness is
    graded beside it and the worse of the two wins. A synchronized group whose
    members report different captures is unhealthy however good both rates
    look, because the geometry that made them a group no longer holds.
    """
    if not expected or expected <= 0.0:
        return UNKNOWN
    period = 1.0 / expected
    if entry["last_sim_time_s"] is None:
        started = entry["started_s"]
        early = started is None or sim_now - started < 2.0 * period
        return STARTING if early else BAD
    if entry["observed_hz"] is None:
        return STARTING
    lateness = OK if age <= _LATE_PERIODS * period else (
        WARN if age <= _STALLED_PERIODS * period else BAD)
    return worst([grade(entry["observed_hz"], expected), lateness,
                  BAD if sync == "diverged" else OK])


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
