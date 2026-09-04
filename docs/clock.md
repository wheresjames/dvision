# Clocks and synchronization

Every dvision2 module measures time, and almost every module measures it for
two unrelated reasons at once: what the vehicle did, and whether the process
next door is still alive. Those are different clocks. Confusing them produces
a class of bug that is invisible on a quiet machine and wrong on a busy one.

This document is the contract: which clocks exist, who owns them, what each
one may be used for, how modules stay in step in real time and under a
time-scaled simulation, and the failure modes that follow from getting it
wrong. Two bugs of exactly this shape have already shipped and been fixed
here; §6 names them.

---

## 1. The two clocks

| | Simulated time | Wall time |
|---|---|---|
| Read via | `sim.time_s` status key | `time.monotonic()` |
| Owned by | the vehicle provider (`dsim`, or a bridge) | the operating system |
| Counts | seconds the *vehicle* experienced | seconds that passed *in the room* |
| Governs | everything the vehicle did or that describes what it did | liveness between processes, and repaint rate |

**The rule:**

> Anything that affects a module's *output* reads simulated time. Only
> liveness and repaint rate may read the wall clock.

Output means: physics, control loops, arrival gates, leg timeouts, failsafes,
sampling rates, and anything written into a report. If a measurement changes
because the machine was busy, it was measuring the wrong thing.

A third category exists and is easy to miss: **the interval between two pieces
of data is a property of the data, not of any clock.** See F3.

---

## 2. How simulated time is produced

`dsim` owns it. `DroneSimulator.integrate()` advances one counter:

```python
self.sim_time_s += dt
```

`DroneSimulator.clock()` returns it, and every vehicle-side timer reads that
rather than the wall clock: the control lease, the guided setpoint timeout,
how long the vehicle has been in GUIDED, and the telemetry delay ring.

In real time `dt` is measured from the wall clock, so simulated and wall time
track each other: if the host stalls, the vehicle is owed the truth about how
long that took. Under `--sim-speed` the step is fixed at `1/fps` instead — a
scaled clock is not a measurement of the room, so measuring it would be
meaningless, and a fixed step is what makes a scaled run repeatable. A
fixed-timestep harness does the same in-process, and can advance a second of
flight in a millisecond or stall for a second without the vehicle ageing.

It is published once per frame as the `sim.time_s` status key.

### 2.1 What a client actually receives

`sim.time_s` is built inside `status_fields()` and pushed through the **same
telemetry delay ring as every other key**. A client's view of it is therefore:

- **Quantised** to the publish rate — 33 ms at the default 30 Hz.
- **Delayed** by the configured telemetry latency.
- **Occasionally frozen**: when the delay ring releases nothing,
  `published_fields()` returns `None` and the previous values stand.

Measured over one simulated second at 30 Hz:

| | last `sim.time_s` readable | lag | publishes returning nothing |
|---|---|---|---|
| no telemetry latency | 1.000 | 0 | 0/30 |
| `--telemetry-latency-ms 200` | 0.767 | 233 ms | 13/30 |

This is not a defect. A delayed clock arriving alongside delayed state is
causally correct: "how long since I saw this" is then measured against the
instant the state actually describes. And because every consumer asks *"has
enough vehicle time passed?"*, a lagging or frozen clock makes a module
**wait**, never misbehave.

### 2.2 Continuity

- A **drone reset** (`reset` command) does *not* move `sim.time_s`. The
  vehicle is repositioned; time does not rewind.
- A **simulator restart** does. A new `dsim` process starts at 0, so a client
  that reconnects sees the clock jump backwards. Modules must tolerate this;
  see F6.
- `dfgb` publishes its own `sim.time_s` as `monotonic() - started`. A bridge to
  a real vehicle is its own time authority and is always in real time.

---

## 3. What reads which clock

Concrete, current, and the reference for anything new.

**Simulated time:**

| Reader | Timer |
|---|---|
| `dsim` | control lease expiry, guided setpoint timeout, GUIDED entry, telemetry delay ring, battery drain |
| `dway` | the whole mission clock -- dwell, arrival gates, leg timeouts, setpoint stream -- plus the readiness deadline, scheduled start instant and bus event timestamps |
| `dalg` | frame capture interval (`capture_fps`), coordinator-silence watchdog, scheduled start instant |
| `daic` | planner state timers (arming, target loss, search legs), the optical-flow frame interval, and the SLAM frame timestamp |
| `dctl` | measurement-run scheduling, bus event timestamps |

**Wall time:**

| Reader | Timer |
|---|---|
| all | `module.hello` / `module.heartbeat` cadence and `PipelineView` expiry |
| `dway` | command-acknowledgement deadline (stretched for a slow-motion vehicle), vehicle-state staleness, the `--timeout` abort |
| `dalg` | presence heartbeat, coordinator-silence wall backstop |
| `daic`, `dctl` | status staleness ("is the simulator still publishing?") |
| all UIs | repaint pacing -- 30 Hz video, 10 Hz maps, 4 Hz text, from `dcmn.pacing` |

Presence is deliberately on the wall clock everywhere. `PipelineView` expires
members on `time.monotonic()`, so a module that heartbeats on simulated time
vanishes from every registry the moment the simulator lags. `dalg/run.py`
carries a comment recording that this was tried and reverted.

---

## 4. Synchronization in real time

**Real time cannot be delayed.** The bus will eventually carry real vehicles,
and a real vehicle does not wait for a subscriber to catch up. Every
synchronization mechanism below is therefore one-way: the vehicle publishes,
and consumers keep up or miss data.

**Video** is a ring of latest frames. A consumer tracks the sequence number it
last handled (`getSeq()`) and reads the newest slot. A consumer that falls
behind skips frames; nothing blocks and nothing queues on its behalf.

**Status** is a retained latest-value store with an epoch. A consumer detects
change via `getEpoch()`. It sees the current values, not a history — a
transition it was too slow to observe is simply not observed. The
`command.results` key exists precisely because one-latest-value was not enough
for command acknowledgement, and carries a short history instead.

**Liveness** is heartbeat plus expiry, on the wall clock, in both directions.
A module that stops heartbeating is dropped from `PipelineView` after its
expiry window. This is the only mechanism that may use the wall clock to make
a decision, because it is a fact about processes rather than about the vehicle.

**Run coordination** is the one place modules genuinely agree on time, and it
does so with **absolute simulated timestamps rather than delays**. The
coordinator publishes `run.prepare`, waits for `run.ready` from every required
participant, then publishes `run.start_scheduled` carrying
`start_sim_time_s` — an absolute instant on the vehicle's clock. Each
participant independently starts when its own reading of `sim.time_s` passes
that instant. Nobody counts down; nobody waits for anybody at the moment of
starting.

That design is what makes coordination work without blocking, and it is the
reason §5 needs no new coordination protocol.

---

## 5. Synchronization under time scaling

A time-scaled simulation advances `sim.time_s` faster or slower than the wall
clock. `dsim --sim-speed <multiplier|max>` selects it, and the monitor's
header menu changes it while the simulation runs; omitting the option is real
time, and real time is never made to wait. Changing speed mid-run is sound for
the same reason scaling is: no consumer reads the rate, only the clock. `--video-hz` publishes video
below the physics rate, which is what makes a scaled run fast: rendering is
the whole per-tick cost, and publishing at the rate consumers actually sample
preserves the simulated interval between frames while removing most of the
work.

**Nothing about the protocol changes.** Only the mapping from simulated
seconds to wall seconds changes. Specifically:

- Publishing stays one-way. Consumers still keep up or drop frames.
- Liveness stays on the wall clock, unscaled. A peer that has died has died,
  at whatever speed the simulation is running.
- Run coordination already uses absolute simulated instants, so it is
  speed-agnostic by construction.

**The criterion for a module to be safe under scaling:**

> Every timer that affects the module's output reads simulated time.

A module that satisfies it needs no knowledge of scaling to decide whether
work is due. This is the whole reason modules read `sim.time_s` rather than
keeping local clocks of their own: local clocks would drift apart and could
not agree on an absolute `start_sim_time_s`.

It may nevertheless need the speed as a **wall-clock wake-up hint**. A process
that owes ten checks per simulated second but sleeps a fixed 50 ms in the room
only wakes ten times per simulated second at 2x, leaving no scheduling margin.
`dsim` therefore publishes `sim.speed` (the requested pace) and
`sim.speed_achieved` (the measured pace). `dcmn.pacing.simulated_poll_delay()`
maps a simulated target rate to a bounded wall delay. Paced modes use the
requested speed so an under-performing simulator causes harmless early polls;
unpaced `max` mode uses the achieved speed. This value may decide when to wake,
but never whether to produce output. The latter decision still reads
`sim.time_s`.

Periodic work also uses `dcmn.pacing.PeriodicDeadline`. Its deadlines form an
absolute grid on the authoritative clock. A late check skips deadlines already
in the past without rebasing the next one and without emitting a catch-up
burst. `simulated_poll_delay` and `PeriodicDeadline` solve different halves of
the problem: the first makes checks frequent enough in wall time; the second
prevents their inevitable lateness from accumulating in simulated time.
GUI schedulers must use `PeriodicDeadline.delay_ms()`: Tk accepts whole
milliseconds, and truncating 33.33 ms to 33 ms permits an early callback and
an extra update before the same deadline. The helper rounds upward so a cap is
never exceeded merely because its delay was quantised.

**What scaling costs a consumer.** Publishing is one-way at any speed, so a
module that polls on its own wall-clock loop sees the same frames arrive in a
shorter wall interval and drops the ones it cannot reach. Measured, `dalg`
asking for 5 frames per simulated second: 4.9 in real time, 4.0 at
`--sim-speed 4`, 2.3 at `--sim-speed max`. The flight is unaffected — arrival,
path and duration hold — but anything *measuring* the flight is built from
fewer samples the faster it runs. Pick a speed with headroom over the
consumer's poll rate when the samples matter.

**What scaling does not give you.** Without a tick barrier — a protocol where
the simulator waits for every participant to acknowledge each frame — the
interleaving of client commands against simulator ticks still varies run to
run. Scaled time is more *repeatable* than real time, because the timestep
stops depending on host load; it is not bit-exact across processes. Do not
claim determinism the design does not provide.

---

## 6. Failure modes

Each is stated as symptom, cause and rule. The first two have shipped here.

### F1 — A wall-clock timer measuring flight

**Symptom.** Behaviour that is correct on an idle machine and wrong on a busy
one; a failsafe that fires "randomly"; results that differ between a laptop
and CI.
**Cause.** A timer that gates flight reads `time.monotonic()`. When the loop
stalls, the vehicle has not aged but the timer has.
**Shipped here twice.** The guided setpoint failsafe fired after two seconds
*in the room* rather than two seconds of flight. And `daic`'s optical-flow
detector turned image expansion into a distance using the wall time between
two frames rather than the distance the camera actually travelled — making
every range estimate a function of machine load.
**Rule.** If it gates, measures, or is reported, it reads `sim.time_s`.

### F2 — A simulated-clock timer measuring liveness

**Symptom.** A module disappears from the pipeline view, or is declared dead,
whenever the simulator lags — or never, when the simulator has stopped
entirely.
**Cause.** The inverse mistake. Presence and expiry are facts about processes.
If the simulator freezes, simulated time freezes with it, and a heartbeat
paced on simulated time stops arriving while an expiry measured in simulated
time never fires.
**Seen here.** `dalg`'s presence heartbeat was paced on simulated time and the
module vanished from every registry as soon as the simulator lagged — or,
before the simulator appeared at all, after a single heartbeat.
**Rule.** Liveness reads the wall clock, always. A simulated-time watchdog
needs a wall-clock backstop for the case where the clock itself has stopped;
`dalg`'s coordinator-silence check carries both.

### F3 — An interval taken from a clock read at processing time

**Symptom.** Velocities, ranges or rates that are subtly wrong, and get worse
under load or at higher speed.
**Cause.** Computing `dt` by reading a clock when two samples are *processed*,
rather than from the timestamps attached to the samples. Under any queueing,
delay or batching those differ.
**Rule.** The interval between two pieces of data is a property of the data.
Carry a timestamp with the sample and subtract those.
`OpticalFlowAvoidance._frame_interval` is the reference implementation.

### F4 — Mixing two clocks in one subtraction

**Symptom.** An age or duration that is nonsensically large or negative,
often only in one configuration.
**Cause.** Subtracting a timestamp taken from one clock from a reading of
another. This is easy to introduce when a clock is made injectable and only
some call sites are switched.
**Live hazard.** `Mission._health` computes
`age = now - state.sample_monotonic_s`, where `now` comes from `Mission.clock`
and `sample_monotonic_s` was stamped by `DsimLink.clock`. Today both default
to `time.monotonic` and are consistent. **`Mission` and `DsimLink` must always
be given the same clock**; changing one alone silently corrupts every
staleness check.
**Rule.** A subtraction may only involve two readings of the same clock. When
a field can carry either, name it for its role (`clock_s`), not for one
implementation (`monotonic_s`).

### F5 — Assuming the published clock is fresh, smooth or always advancing

**Symptom.** A control loop that stutters, a rate limiter that fires in
bursts, or a division by a zero interval.
**Cause.** Treating `sim.time_s` as a continuous high-resolution clock. It is
quantised to the publish rate, delayed by telemetry latency, and stalls
outright on ticks where the delay ring releases nothing (§2.1).
**Rule.** Use it for "has enough time passed?" comparisons, which degrade to
waiting. Never divide by an interval derived from it without checking the
interval is positive.

### F6 — Assuming the published clock is monotonic across a restart

**Symptom.** A module wedges, or reports a huge negative interval, after the
simulator is restarted underneath it.
**Cause.** A new `dsim` process restarts `sim.time_s` at 0. Any module holding
a previous reading now sees time run backwards.
**Rule.** Treat a backwards clock as "no interval available" and fall back, or
re-initialise. `OpticalFlowAvoidance._frame_interval` does exactly this and
says so. Note that a *drone* reset does not rewind the clock — only a
simulator restart does.

### F7 — Rebasing a periodic schedule after every late update

**Symptom.** A nominal 10 Hz stream settles at 7.5 Hz even though the process
has ample CPU and checks more often than 10 Hz.
**Cause.** Testing `now - last >= period` and assigning `last = now` after the
work. With a quantised 30 Hz vehicle clock, observing a 100 ms deadline one
tick late turns that interval into 133 ms, and the lateness permanently shifts
every deadline that follows.
**Seen here.** `dway` setpoint streaming exhibited exactly this distribution;
`dalg` frame capture used the same relative-time pattern.
**Rule.** Target-frequency work advances an absolute `PeriodicDeadline` from
its previous deadline. Skip elapsed slots rather than bursting to catch up.
Best-effort caps such as UI repaint and presence heartbeat may deliberately
remain relative because they promise “no more often than,” not a target rate.

### F8 — Blocking on a peer that is blocked on you

**Symptom.** Deadlock, or a spurious acknowledgement timeout under load.
**Cause.** `DsimLink._await_result` blocks the calling process until the
vehicle publishes a command result, and the vehicle only drains its command
queue inside its own frame loop. Any mechanism that makes the vehicle wait for
that client before advancing closes the loop.
**Rule.** The vehicle must never wait for a client in order to service that
client. If a frame barrier is ever added, the simulator must keep draining
commands and publishing status while blocked on it — time frozen, commands
still answered. Acknowledgement deadlines stay on the wall clock, because they
bound a peer's liveness, not a flight.

### F9 — Coordinating with delays instead of instants

**Symptom.** Participants start a run at visibly different points; the
skew grows with the number of participants and with load.
**Cause.** Telling participants "start in three seconds" rather than "start at
simulated instant T". Each participant's countdown begins when it happens to
receive the message.
**Rule.** Coordinate on absolute simulated instants. `start_sim_time_s` is
absolute for this reason.

### F10 — Pacing control on the repaint timer

**Symptom.** Control rate drops when the window is busy; capping the UI
frame rate silently caps the control loop.
**Cause.** One timer driving both the redraw and the control step.
`dctl.tick()` both repaints and calls `send_held_velocity()`, and `daic`'s
`_update_video` runs the detector and the SLAM front end before it uploads an
image — so capping either timer would have capped perception or manual flight
with it.
**Rule.** Control and repaint are separate timers on separate clocks. Control
paces on simulated time; repaint paces on the wall clock and may be capped
freely — no operator resolves more than about 30 Hz of video or a few Hz of
text. `dcmn.pacing` holds the caps and the limiter; each window applies them
to its painting only, and a test asserts the control calls sit outside every
`due()` guard.

---

## 7. Checklist for a new module

1. List every timer you have. For each, answer: does this affect what I
   output, or only whether someone is alive / when I repaint?
2. Output timers read `sim.time_s`. Liveness and repaint read
   `time.monotonic()`. Nothing reads both.
3. Intervals between samples come from timestamps carried with the samples.
4. Handle the clock being absent (no vehicle attached), stalled (delay ring
   released nothing), and going backwards (simulator restarted).
5. Never divide by an interval without checking it is positive.
6. Heartbeat on the wall clock. Give any simulated-time watchdog a wall-clock
   backstop.
7. Coordinate on absolute simulated instants, never on delays.
8. Schedule target-frequency work from absolute periodic deadlines, not from
   when the previous iteration finished.
9. Use simulation speed only to schedule the next wall-clock wake-up; use
   simulated time to decide whether work is due.
10. Do not keep a private simulated clock. Read the published one.

---

## 8. How this is enforced

`tests/test_dvision_wall_clock_independence.py` drives the simulator with a
wall clock that can be stopped or made to lurch, and asserts that behaviour
does not change: the published clock counts flight rather than room time, the
guided failsafe counts seconds of flight, the perception chain learns the same
map however slow the loop was, and the flow detector takes its interval from
the vehicle. It also asserts the correct fallbacks — an explicit interval
still wins, and a vehicle that publishes no clock leaves the wall clock as the
only estimate.

Its header states the standard plainly, and it is the right place to add a
case when a new timer is introduced:

> Nothing the vehicle does may depend on how busy the machine is. [...] That
> is not a flaky test — it is a wrong answer that happens to be right when the
> machine is quiet.
