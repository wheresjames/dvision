"""The mission status and intent planes: schemas, bounds, and one live writer per vehicle."""
from __future__ import annotations

import uuid

import pytest

from dcmn.mission import INTENT_SCHEMA, MAX_TEXT, STATUS_SCHEMA, clip, validate_intent, validate_status
from dcmn.navigation import Snapshot


def status(**fields):
    value = dict(schema=STATUS_SCHEMA, vehicle_id='v', session='s', sequence=1, time_s=0., mission='goto',
                 profile='p', profile_digest='d', phase='running', outcome='', reason='', blocking='')
    value.update(fields)
    return value


def intent(**fields):
    value = dict(schema=INTENT_SCHEMA, vehicle_id='v', session='s', sequence=1, time_s=0., revision=1,
                 kind='launch', args={}, reason='')
    value.update(fields)
    return value


def test_valid_records_pass_and_malformed_ones_say_why():
    assert validate_status(status(goal=dict(xy=[1., 2.], tag='goal'), area=[[0., 0.], [1., 0.], [1., 1.]]))
    assert validate_intent(intent())
    for bad, message in ((status(phase='flying'), 'phase'), (status(outcome='won'), 'outcome'),
                         (status(reason='x' * (MAX_TEXT + 1)), 'reason'),
                         (status(area=[[0., 0.]] * 257), '256'), (status(waypoints=[{'xy': [0, 0]}] * 257), '256'),
                         (status(goal=dict(xy=[1.])), 'goal'), (status(sequence=-1), 'sequence')):
        with pytest.raises(ValueError, match=message):
            validate_status(bad)
    for bad, message in ((intent(kind='fly'), 'kind'), (intent(revision='1'), 'revision'),
                         (intent(schema='x'), 'schema')):
        with pytest.raises(ValueError, match=message):
            validate_intent(bad)


def test_clip_marks_what_it_cut():
    assert clip('short') == 'short'
    text = clip('y' * 500)
    assert len(text) == MAX_TEXT and text.endswith('…')


def test_mission_planes_round_trip_and_refuse_a_second_live_writer():
    vehicle = 'mission-' + uuid.uuid4().hex[:8]
    writer, reader = Snapshot(vehicle, 'mission'), Snapshot(vehicle, 'mission')
    intents, intent_reader = Snapshot(vehicle, 'mission_intent'), Snapshot(vehicle, 'mission_intent')
    try:
        assert writer.name == f'/dvision2.{vehicle}.mission'
        assert intents.name == f'/dvision2.{vehicle}.mission_intent'
        assert reader.read() == {}
        writer.write(status(vehicle_id=vehicle))
        intents.write(intent(vehicle_id=vehicle, kind='hold'))
        assert reader.read()['phase'] == 'running'
        assert intent_reader.read()['kind'] == 'hold'
        with pytest.raises(ValueError):
            writer.write(status(phase='nope'))
        assert reader.read()['phase'] == 'running', 'a refused write never replaces the retained record'
        with pytest.raises(ValueError, match='live writer already owns'):
            Snapshot(vehicle, 'mission').start()
    finally:
        writer.close(); intents.close()


def test_a_second_dmsn_for_the_same_vehicle_fails_before_sending_anything(tmp_path):
    from dmsn.ports import LivePorts
    from dmsn.runner import MissionRun
    vehicle = 'mission-' + uuid.uuid4().hex[:8]
    first = MissionRun(vehicle, ports=LivePorts(vehicle, bus=False), record=False)
    try:
        with pytest.raises(ValueError, match='live writer already owns'):
            MissionRun(vehicle, ports=LivePorts(vehicle, bus=False), record=False)
    finally:
        first.close()
    MissionRun(vehicle, ports=LivePorts(vehicle, bus=False), record=False).close()  # released on close


def test_restart_recovers_the_profile_and_previous_status_without_resuming():
    from dmsn import profiles as P
    from dmsn.ports import LivePorts
    from dmsn.runner import MissionRun
    vehicle = 'restart-' + uuid.uuid4().hex[:8]
    profile = P.apply_overrides(P.blank('goto'), ['goal=12,4', 'budgets.time_s=90'])
    first = MissionRun(vehicle, profile, ports=LivePorts(vehicle, bus=False), record=False)
    first.ports.write_status(first.status_value(first.observation))
    # A crashed writer leaves the shared record but loses its lifetime lock.
    # Keep the old handle open here: memkv.close() normally unlinks its area.
    snapshot = first.ports.status_out
    snapshot.lock.close(); snapshot.lock = None
    first.ports.intent_out.close()
    second = None
    try:
        second = MissionRun(vehicle, ports=LivePorts(vehicle, bus=False), record=False)
        assert second.document.value == profile
        assert second.previous_status['profile_digest'] == P.digest(profile)
        assert second.phase == 'idle' and second.mission is None
        assert second.intents.last is None
    finally:
        if second is not None: second.close()
        first.close()
