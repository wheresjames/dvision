"""The chain waits to be told: dnav adopts no provider goal by default, dway can wait for a mission,
dalg sizes its first grid to cover every committed map, and dnav says when a mission loses the goal."""
from __future__ import annotations

import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_dnav_adopts_the_provider_goal_only_when_asked():
    from dnav.dnav import parse_args
    assert parse_args(['--id', 'a']).provider_goal is False
    assert parse_args(['--id', 'a', '--provider-goal']).provider_goal is True
    legacy = parse_args(['--id', 'a', '--no-provider-goal'])
    assert legacy.provider_goal is False, '--no-provider-goal is still accepted and changes nothing'


def test_dway_waits_for_a_mission_only_in_target_mode():
    from dway.dway import parse_args
    from dway.flightui import executor_options
    args = parse_args(['--id', 'a', '--mode', 'target', '--wait-for-mission'])
    assert executor_options(args)['wait_for_mission'] is True
    assert executor_options(parse_args(['--id', 'a', '--mode', 'target']))['wait_for_mission'] is False
    with pytest.raises(SystemExit):
        parse_args(['--id', 'a', '--mode', 'dynamic', '--wait-for-mission'])


def test_a_gated_executor_requires_target_mode():
    from dway.executor import DynamicExecutor
    from dcmn.navigation import ExecutionProfile
    with pytest.raises(ValueError, match='target mode'):
        DynamicExecutor('gate-' + uuid.uuid4().hex[:6], object(), ExecutionProfile(), allow_uncalibrated=True,
                        wait_for_mission=True)


def test_dalg_first_grid_defaults_to_100_m():
    from dalg.dalg import parse_args
    assert parse_args(['--id', 'a', '--profile', 'lidar-baseline']).min_side_m == 100.


@pytest.mark.parametrize('gated', [False, True])
def test_mission_authority_prevents_automatic_mapping_resets(gated):
    from types import SimpleNamespace
    from dway.executor import DynamicExecutor
    executor = SimpleNamespace(snapshot=dict(authority=dict(role='mission')), wait_for_mission=gated)
    DynamicExecutor._request_coverage(executor)


def test_dnav_says_when_goal_authority_moves_between_a_mission_and_an_operator(tmp_path):
    from dcmn.context import Context
    from dnav.plan import NavRun
    from dnav.policy import load_policy
    instance = 'auth-' + uuid.uuid4().hex[:8]
    context = Context(instance)
    context.start(tmp_path / 'run')
    run = NavRun(instance, ROOT, policy=load_policy('default', ROOT))
    try:
        context.set_goal('dmsn:mission', [5., 5.], role='mission')
        run.connect()
        assert run.authority_notice == ''
        run.set_goal(1., 1., handoff=True)
        run.connect()
        assert 'from mission to ui (this dnav)' in run.authority_notice
        assert 'mission is paused' in run.authority_notice
        assert any(e['type'] == 'goal.authority_changed' for e in run._events)
    finally:
        run.close(); context.close()
