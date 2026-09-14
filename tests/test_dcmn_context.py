"""The neutral session context: provider ownership, pose validity, goal authority."""
from __future__ import annotations

import math
import uuid

import pytest

from dcmn.context import Context, main as context_cli, parse_bounds, validate_goal, validate_pose

POSE = dict(x_m=1., y_m=-2., z_m=1.5, heading_deg=90., roll_deg=0., pitch_deg=0.)


@pytest.fixture
def context(tmp_path):
    provider = Context('ctx-' + uuid.uuid4().hex[:8])
    provider.start(tmp_path / 'session', clock_epoch=5)
    try: yield provider
    finally: provider.close()


def test_the_registry_outlives_every_writer_but_its_provider(context, tmp_path):
    reader = Context(context.instance)
    value = reader.read()
    assert value['provider_id'] == context.owner and value['frame']['axes'] == 'x_east,y_south,z_up'
    assert value['report_root'] == str((tmp_path / 'session').resolve())
    context.publish_pose(POSE, 1.)
    assert reader.read()['pose']['sequence'] == 1
    context.close()
    assert Context(context.instance).read() == {}


def test_only_the_provider_publishes_poses(context):
    intruder = Context(context.instance)
    intruder.owner = 'someone-else'
    with pytest.raises(ValueError, match='ownership'):
        intruder.publish_pose(POSE, 1.)


def test_a_pose_carries_frame_epochs_time_validity_and_unknown_uncertainty(context):
    context.publish_pose(POSE, 2., localization_epoch=0, clock_epoch=5)
    snapshot = context.read()
    pose = validate_pose(snapshot['pose'], snapshot)
    assert pose['frame_id'] == 'local' and pose['capture_time_s'] == 2.
    assert pose['uncertainty'] is None and pose['uncertainty_state'] == 'unknown'
    assert pose['estimate_kind'] == 'ideal' and pose['provider_id'] == context.owner


def test_stale_invalid_nonfinite_and_mismatched_poses_are_refused(context):
    context.publish_pose(POSE, 2., clock_epoch=5)
    snapshot = context.read()
    with pytest.raises(ValueError, match='stale'):
        validate_pose(snapshot['pose'], dict(snapshot, time_s=3.), max_age=.5)
    with pytest.raises(ValueError, match='future'):
        validate_pose(snapshot['pose'], dict(snapshot, time_s=1.))
    with pytest.raises(ValueError, match='invalid'):
        validate_pose(dict(snapshot['pose'], valid=False), snapshot)
    with pytest.raises(ValueError, match='nonfinite'):
        validate_pose(dict(snapshot['pose'], x_m=math.nan), snapshot)
    with pytest.raises(ValueError, match='clock_epoch'):
        validate_pose(snapshot['pose'], dict(snapshot, clock_epoch=6))
    with pytest.raises(ValueError, match='localization_epoch'):
        validate_pose(snapshot['pose'], dict(snapshot, localization_epoch=1))


def test_a_paused_clock_does_not_age_the_pose(context):
    """Freshness is data-clock age; wall time passing changes nothing."""
    import time
    context.publish_pose(POSE, 2., clock_epoch=5)
    time.sleep(.6)
    snapshot = context.read()
    assert validate_pose(snapshot['pose'], snapshot, max_age=.5)


def test_goal_authority_requires_an_explicit_handoff(context):
    ui = Context(context.instance)
    goal = ui.set_goal('ui-1', (4., 5.))
    assert goal['revision'] == 1 and goal['authority_epoch'] == 1 and goal['role'] == 'ui'
    with pytest.raises(ValueError, match='explicit handoff'):
        ui.set_goal('mission-1', (7., 7.), role='mission')
    assert ui.read()['goal']['position'] == [4., 5.]
    goal = ui.set_goal('mission-1', (7., 7., 2.), role='mission', handoff=True)
    assert goal['authority_epoch'] == 2 and goal['position'] == [7., 7., 2.]
    assert ui.set_goal('mission-1', None, role='mission') is None
    snapshot = ui.read()
    assert snapshot['goal'] is None and snapshot['goal_revision'] == 3
    kinds = [entry['kind'] for entry in snapshot['history']]
    assert kinds.count('goal.authority') == 2 and 'goal.cleared' in kinds


@pytest.mark.parametrize('position', [(1.,), (1., math.inf), ('a', 1.), (1., 2., 3., 4.)])
def test_malformed_goals_are_refused(context, position):
    with pytest.raises(ValueError):
        Context(context.instance).set_goal('ui', position)


def test_a_localization_or_clock_change_withdraws_the_goal_and_says_why(context):
    writer = Context(context.instance)
    writer.set_goal('ui', (4., 5.))
    context.publish_pose(POSE, 1., clock_epoch=5)
    snapshot = context.read()
    assert validate_goal(snapshot['goal'], snapshot)
    context.publish_pose(POSE, 1.1, localization_epoch=1, clock_epoch=5)
    snapshot = context.read()
    assert snapshot['goal'] is None
    reasons = [h.get('reason') for h in snapshot['history'] if h['kind'] == 'goal.invalidated']
    assert reasons == ['localization epoch changed']
    # A goal kept from before the change would not validate either.
    with pytest.raises(ValueError, match='localization_epoch'):
        validate_goal(dict(position=[4., 5.], frame_id='local', localization_epoch=0, clock_epoch=5),
                      snapshot)


def test_mapping_requests_and_rollover_are_durable_for_late_readers(context, tmp_path):
    requester = Context(context.instance)
    assert requester.request_reset((-10., -10., 10., 10.)) == 1
    old = context.read()['session_id']
    context.rollover(tmp_path / 'second')
    late = Context(context.instance).read()
    assert late['mapping_request'] == dict(bounds=[-10., -10., 10., 10.], revision=1, by='operator')
    assert late['session_id'] != old and late['report_root'].endswith('second')
    assert [h['kind'] for h in late['history']][-2:] == ['mapping.reset_requested', 'session.rollover']
    with pytest.raises(ValueError):
        requester.request_reset((0., 0., -1., 1.))


def test_an_incompatible_schema_is_refused_explicitly(context):
    value = context.read()
    context._write(dict(value, schema='dvision2.context.v0'))
    with pytest.raises(ValueError, match='unsupported context schema'):
        Context(context.instance).read()
    context._write(value)


def test_bounds_parse_negative_and_reject_nonsense():
    assert parse_bounds('-10,-5.5,30,25') == (-10., -5.5, 30., 25.)
    for text in ('1,2,3', '0,0,0,1', '0,0,nan,1', 'a,b,c,d'):
        with pytest.raises(ValueError): parse_bounds(text)


def test_the_cli_sets_goals_and_requests_resets(context, capsys):
    assert context_cli(['--id', context.instance, 'goal', '-3,4']) == 0
    assert Context(context.instance).read()['goal']['position'] == [-3., 4.]
    assert context_cli(['--id', context.instance, 'reset', '--bounds', '-20,-20,20,20']) == 0
    assert Context(context.instance).read()['mapping_request']['bounds'] == [-20., -20., 20., 20.]
    assert context_cli(['--id', context.instance, 'goal', '1,1', '--writer', 'other']) == 1
    assert 'explicit handoff' in capsys.readouterr().err
