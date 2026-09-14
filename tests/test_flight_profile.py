"""The dynamic flight profile must stay inside what was actually measured.

The checked-in envelope in ``dsim-position-v1.json`` is not documentation --
it is a claim about the vehicle, and this file re-measures the vehicle with
the same deterministic rig that produced it. If the simulator, the link or
the calibration harness changes, a profile that no longer covers reality
must fail here rather than authorize flight it cannot stop.

Extrapolation is refused outright: a profile whose limits exceed its own
measured envelope is invalid at load time, in both dnav and dway.
"""

import json
from pathlib import Path

import pytest

from dcmn.navigation import ExecutionProfile
from dtest.flight_calibration import envelope, measure_all

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / 'assets/execution_profiles/dsim-position-v1.json'
DRY_RUN = ROOT / 'assets/execution_profiles/dry-run.json'


def profile(**changes) -> ExecutionProfile:
    value = json.loads(PROFILE.read_text())
    value.update(changes)
    path = PROFILE.with_name('profile-under-test.json')
    path.write_text(json.dumps(value))
    try:
        return ExecutionProfile.load(path)
    finally:
        path.unlink()


def test_the_checked_in_envelope_is_what_the_vehicle_measures(tmp_path) -> None:
    """Re-measure HOLD stopping; the profile's declared envelope must cover it."""
    measured = envelope(measure_all(tmp_path))
    declared = json.loads(PROFILE.read_text())['calibration']['envelope']
    for key, value in measured.items():
        assert declared[key] >= value - 1e-9, f'{key}: declared {declared[key]} < measured {value}'
    # The re-measurement must actually be the run the profile cites: same rig,
    # same conditions, so a drift in the simulator shows up as a mismatch.
    for key in ('stop_distance_m', 'stop_time_s', 'lateral_m', 'cruise_cross_track_m'):
        assert measured[key] == pytest.approx(declared[key], abs=1e-6), key


def test_a_synthetic_profile_cannot_authorize_flight() -> None:
    with pytest.raises(ValueError, match='synthetic'):
        ExecutionProfile.load(DRY_RUN).require_flight()
    executor = ExecutionProfile.load(DRY_RUN)
    assert not executor.calibrated


def test_uncalibrated_profile_still_admits_observation() -> None:
    p = ExecutionProfile.load(DRY_RUN)
    assert p.speed_mps > 0 and p.calibration is None


def test_profile_without_a_calibration_record_is_not_calibrated() -> None:
    with pytest.raises(ValueError, match='no calibration record'):
        profile(calibration={})


def test_speed_above_the_measured_conditions_is_refused() -> None:
    with pytest.raises(ValueError, match='speed above measured'):
        profile(speed_mps=0.75)


def test_stopping_distance_below_the_measured_envelope_is_refused() -> None:
    with pytest.raises(ValueError, match='stopping_m below'):
        profile(stopping_m=0.05)


def test_stopping_time_below_the_measured_stop_time_is_refused() -> None:
    with pytest.raises(ValueError, match='stopping_s below'):
        profile(stopping_s=0.1)


def test_tracking_below_the_measured_lateral_error_is_refused() -> None:
    with pytest.raises(ValueError, match='tracking_m below'):
        profile(tracking_m=0.1)


def test_join_and_arrival_must_accept_a_vehicle_stopped_short() -> None:
    with pytest.raises(ValueError, match='join_m cannot'):
        profile(join_m=0.1)
    with pytest.raises(ValueError, match='arrival_m cannot'):
        profile(arrival_m=0.1)


def test_wind_and_latency_above_the_measured_conditions_are_refused() -> None:
    with pytest.raises(ValueError, match='wind above measured'):
        profile(max_wind_mps=0.5)
    with pytest.raises(ValueError, match='telemetry latency above'):
        profile(max_telemetry_latency_ms=100.0)


def test_an_undeclared_vertical_assumption_cannot_authorize_flight() -> None:
    with pytest.raises(ValueError, match='vertical assumption'):
        profile(slab_assumption='none')


def test_arrival_gate_covers_the_measured_overshoot_consequence() -> None:
    """Position-target overshoot (0.62 m) exceeds stopping_m (0.2 m), so the
    executor must stop by HOLD before targets -- and the profile's own note
    must still say so, or the reasoning is silently gone."""
    p = ExecutionProfile.load(PROFILE)
    value = json.loads(PROFILE.read_text())
    assert value['calibration']['envelope']['overshoot_m'] > p.stopping_m
    assert 'never' in value['note'] and 'HOLD' in value['note']


def test_the_measured_envelope_supports_the_profile_limits_exactly() -> None:
    p = json.loads(PROFILE.read_text())
    envelope_values, margins = p['calibration']['envelope'], p['calibration']['margins']
    # stopping_m covers measured stop distance + one control period + margin.
    assert (p['stopping_m'] >= envelope_values['stop_distance_m']
            + p['speed_mps'] / p['stream_hz'] + margins['stop_m'] - 1e-9)
    # stopping_s covers measured stop time + reaction.
    assert p['stopping_s'] >= envelope_values['stop_time_s'] + p['reaction_s'] - 1e-9
