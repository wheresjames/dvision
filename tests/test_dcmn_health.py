from __future__ import annotations

from types import SimpleNamespace

import pytest

from dcmn.health import BAD, OK, UNKNOWN, WARN, IntakeMeter, SteadyGrade, describe
from dcmn.pacing import PeriodicDeadline, simulated_poll_delay, simulation_speed
from dcmn.series import EnvelopeSeries
from dsim.health import SimulationHealth


def test_intake_meter_uses_simulated_time_and_counts_sequence_gaps():
    meter = IntakeMeter(5)
    meter.record(5)
    assert meter.report(10)["achieved_hz"] is None
    meter.note_sequence(4)
    meter.note_sequence(7)
    meter.record(4)
    report = meter.report(12)
    assert report == {"basis": "sim", "wanted_hz": 5.0, "achieved_hz": 2.0,
                      "skipped": 2, "overruns": 0, "grade": BAD}
    assert meter.report(12)["achieved_hz"] == 2.0


def test_health_helpers_tolerate_old_modules_and_debounce_changes():
    assert describe(None)["grade"] == UNKNOWN
    assert describe({"wanted_hz": 2, "achieved_hz": 2})["basis"] == "sim"
    assert IntakeMeter(30, basis="wall").report(1)["basis"] == "wall"
    steady = SteadyGrade(runs=2)
    assert steady.update(WARN) == UNKNOWN
    assert steady.update(WARN) == WARN
    assert steady.update(OK) == WARN
    assert steady.update(OK) == OK


def test_envelope_series_is_bounded_and_preserves_peaks():
    series = EnvelopeSeries(4)
    for x, value in enumerate((1, 50, 2, 3, -20, 4, 5)):
        series.add(x, value)
    assert len(series) < series.capacity
    assert series.samples == 7
    assert series.extremes() == (-20, 50)
    assert sum(span.count for span in series.spans()) == 7


def test_simulated_poll_delay_scales_and_handles_unpaced_simulation():
    assert simulated_poll_delay(10, {"sim.speed": "2"}) == 0.025
    assert simulation_speed({"sim.speed": "0",
                             "sim.speed_achieved": "4.0"}) == 4.0
    assert simulated_poll_delay(10, {"sim.speed": "0",
                                     "sim.speed_achieved": "4.0"}) == 0.0125
    assert simulation_speed({}) == 1.0
    assert simulated_poll_delay(1000, {"sim.speed": "100"}) == 0.002


def test_periodic_deadline_keeps_phase_and_skips_missed_slots():
    cadence = PeriodicDeadline(10)
    cadence.reset(1.0)
    assert cadence.due(1.0)
    assert cadence.advance(1.034) == 0
    assert cadence.deadline == 1.1
    assert not cadence.due(1.099)
    assert cadence.advance(1.234) == 1
    assert cadence.deadline == pytest.approx(1.3)
    assert cadence.delay(1.25) == pytest.approx(.05)
    cadence.deadline = 2.0 + 1.0 / 30.0
    assert cadence.delay_ms(2.0) == 34


def test_simulation_health_marks_expired_members_bad_and_writes_report(tmp_path):
    monitor = SimulationHealth(tmp_path, capacity=4)
    member = SimpleNamespace(role="controller", implementation="daic",
                             process_id="p1", state="RUNNING",
                             intake={"wanted_hz": 5, "achieved_hz": 5,
                                     "grade": OK})
    monitor.note_tick(.002)
    monitor.sample(wall_now=1, sim_now=1, requested=2,
                   members=[(member, 4)], member_expiry_s=3)
    for wall, sim in ((2, 3), (3, 5)):
        monitor.note_tick(.003)
        record = monitor.sample(wall_now=wall, sim_now=sim, requested=2,
                                members=[(member, 4)], member_expiry_s=3)
    assert record["modules"][0]["state"] == "expired"
    assert record["modules"][0]["grade"] == BAD
    assert monitor.grade.value == BAD
    output = monitor.write_report(tmp_path)
    monitor.close()
    assert output.is_file()
    assert "Simulation health" in output.read_text(encoding="utf-8")
    assert (tmp_path / "health.jsonl").is_file()
