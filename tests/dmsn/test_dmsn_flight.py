"""dmsn and dway on the deterministic flight rig: real executor, real planes, one virtual clock."""
from __future__ import annotations

import math
import uuid

import pytest

from dcmn.navigation import Snapshot
from dmsn import profiles as P
from dmsn.intents import IntentWriter
from dmsn.ports import LivePorts
from dmsn.runner import MissionRun
from dtest.dynamic_rig import ROOT, DynamicRig

TARGET_PROFILE = ROOT / 'assets/execution_profiles/sim-target.json'
GATED = dict(auto=True, allow_uncalibrated=True, wait_for_mission=True, bus_enabled=False)


def parked(tmp_path, **options):
    rig = DynamicRig(tmp_path, profile=TARGET_PROFILE, executor_options=dict(GATED, **options))
    rig.sim.state.armed, rig.sim.state.mode = False, 'DISARMED'
    # The rig installs a fixture goal; a mission chain starts with none.
    rig.context.set_goal('fixture', None, role='mission')
    rig.context.release_authority('fixture')
    return rig


def ack(rig):
    return rig.executor.status.get('mission_intent') or {}


@pytest.fixture
def intents():
    opened = []

    def writer(rig):
        output = Snapshot(rig.id, 'mission_intent')
        opened.append(output)
        return IntentWriter(output, rig.id, 'test-' + uuid.uuid4().hex[:8])
    yield writer
    for output in opened: output.close()


def test_a_gated_target_executor_stays_on_the_ground_until_a_launch_intent(tmp_path, intents):
    rig = parked(tmp_path)
    try:
        rig.set_goal(12.5, 6.0)
        rig.fly(limit_s=10.0, until=lambda r: False)
        ex = rig.executor
        assert ex.state == 'WAITING' and ex.reason == 'waiting for a mission launch intent'
        assert not rig.sim.state.armed and ex.targets_sent == 0 and not ex.owns_control
        status = rig.status()
        assert status['wait_for_mission'] and not status['launch_authorized']
        writer = intents(rig)
        writer.send('launch', rig.sim.sim_time_s, 'test launch')
        assert rig.fly(limit_s=60.0, until=lambda r: ack(r).get('state') == 'completed'), ack(rig)
        assert ack(rig)['revision'] == writer.revision and ack(rig)['kind'] == 'launch'
        assert ex.launch_phase == 'airborne' and rig.sim.state.armed
        assert rig.fly_to_state('COMPLETE', limit_s=120.0), ex.reason
    finally:
        rig.close()


def test_hold_resume_and_cancel_intents_are_acknowledged_with_their_outcome(tmp_path, intents):
    rig = parked(tmp_path)
    try:
        rig.set_goal(20.5, 6.0)
        writer = intents(rig)
        writer.send('launch', rig.sim.sim_time_s)
        assert rig.fly(limit_s=90.0, until=lambda r: r.executor.state == 'EXECUTING'), rig.executor.reason
        writer.send('hold', rig.sim.sim_time_s, 'operator pause')
        assert rig.fly(limit_s=20.0, until=lambda r: r.executor.state == 'HOLDING'), rig.executor.reason
        assert ack(rig)['kind'] == 'hold' and ack(rig)['state'] == 'accepted'
        assert rig.executor.mission_hold and rig.executor.hold_kind == 'pause'
        rig.fly(limit_s=5.0, until=lambda r: False)
        assert rig.executor.state == 'HOLDING', 'a mission hold is not resumed automatically'
        writer.send('resume', rig.sim.sim_time_s)
        assert rig.fly(limit_s=20.0, until=lambda r: r.executor.state == 'EXECUTING'), (rig.executor.reason, ack(rig))
        assert ack(rig)['kind'] == 'resume' and ack(rig)['state'] == 'accepted'
        writer.send('cancel', rig.sim.sim_time_s)
        assert rig.fly(limit_s=20.0, until=lambda r: r.executor.state == 'CANCELLED'), rig.executor.reason
        assert ack(rig)['kind'] == 'cancel' and ack(rig)['state'] == 'accepted'
        assert not rig.targets_outside_permission()
    finally:
        rig.close()


def test_an_ungated_executor_launches_on_its_own_and_reads_no_intents(tmp_path):
    rig = DynamicRig(tmp_path, profile=TARGET_PROFILE,
                     executor_options=dict(auto=True, allow_uncalibrated=True, bus_enabled=False))
    rig.sim.state.armed, rig.sim.state.mode = False, 'DISARMED'
    try:
        assert rig.executor.intent is None
        assert rig.fly_to_state('COMPLETE', limit_s=120.0), rig.executor.reason
        assert rig.status()['mission_intent'] is None
    finally:
        rig.close()


def mission(rig, *overrides):
    profile = P.apply_overrides(P.blank('goto'), list(overrides))
    return MissionRun(rig.id, profile, ports=LivePorts(rig.id, bus=False), wall=rig.clock.read)


def fly(rig, run, *, limit_s, until):
    steps = int(limit_s / 0.05)
    for _ in range(steps):
        if until(): return True
        rig.step(); run.step()
    return until()


def test_dmsn_launches_flies_a_goto_and_completes_as_the_mission_authority(tmp_path):
    rig = parked(tmp_path)
    run = mission(rig, 'goal_source=xy', 'goal=12.5,6.0')
    try:
        fly(rig, run, limit_s=5.0, until=lambda: False)
        assert not rig.sim.state.armed, 'nothing flies before Start'
        run.start()
        assert fly(rig, run, limit_s=180.0, until=lambda: run.phase == 'done'), (run.blocking, rig.executor.reason)
        assert run.outcome == 'complete'
        assert math.dist(rig.executor.pose[:2], (12.5, 6.0)) <= rig.profile.arrival_m
        assert rig.context.read()['authority']['role'] == 'mission'
        assert [i['kind'] for i in run.acks if i['state'] == 'completed'] == ['launch']
        assert rig.sim.state.armed and rig.sim.state.mode == 'HOLD', 'finish action hold: airborne in HOLD'
    finally:
        run.close(); rig.close()


def test_retarget_while_flying_and_an_operator_takeover_pause_the_mission(tmp_path):
    rig = parked(tmp_path)
    run = mission(rig, 'goal_source=xy', 'goal=24.0,6.0')
    try:
        run.start()
        assert fly(rig, run, limit_s=90.0, until=lambda: rig.executor.state == 'EXECUTING'), run.blocking
        run.click(8.0, 6.0)
        assert fly(rig, run, limit_s=120.0, until=lambda: run.phase == 'done'), (run.blocking, rig.executor.reason)
        assert run.outcome == 'complete' and 'arrived at 8.00, 6.00' in run.reason
        assert [g['outcome'] for g in run.goals] == ['replaced', 'reached']
    except Exception:
        run.close(); rig.close()
        raise
    run.close()
    second = mission(rig, 'goal_source=xy', 'goal=20.0,6.0')
    try:
        second.start()
        assert fly(rig, second, limit_s=60.0, until=lambda: rig.executor.state == 'EXECUTING'), second.blocking
        rig.context.set_goal('dnav:operator', [12.0, 6.0, rig.profile.altitude_m], role='ui', handoff=True)
        assert fly(rig, second, limit_s=5.0, until=lambda: second.phase == 'paused')
        assert 'goal authority taken by ui' in second.blocking
        assert fly(rig, second, limit_s=120.0, until=lambda: rig.executor.state == 'COMPLETE'), rig.executor.reason
        assert math.dist(rig.executor.pose[:2], (12.0, 6.0)) <= rig.profile.arrival_m, \
            'the operator\'s goal flew; the paused mission did not hold the vehicle under them'
        assert rig.context.read()['authority']['id'] == 'dnav:operator'
        second.resume()
        assert rig.context.read()['authority']['id'] == second.writer_id
        assert fly(rig, second, limit_s=120.0, until=lambda: second.phase == 'done'), second.blocking
        assert second.outcome == 'complete'
    finally:
        second.close(); rig.close()


def walled(tmp_path, walls, **options):
    """The parked rig on its open map, with interior wall cells added."""
    rig = DynamicRig(tmp_path, walls=walls, profile=TARGET_PROFILE,
                      executor_options=dict(GATED, **options))
    rig.sim.state.armed, rig.sim.state.mode = False, 'DISARMED'
    rig.context.set_goal('fixture', None, role='mission')
    rig.context.release_authority('fixture')
    return rig


def test_an_unreachable_waypoint_aborts_the_mission(tmp_path):
    rig = walled(tmp_path, [(20, 6)])  # the second waypoint sits inside this wall
    run = mission(rig, 'mission=waypoints', 'waypoints=10,6;20.5,6.5', 'failure_policy=abort', 'stuck_s=10')
    try:
        run.start()
        assert fly(rig, run, limit_s=240.0, until=lambda: run.phase == 'done'), (run.blocking, rig.executor.reason)
        assert run.outcome == 'failed'
        assert 'waypoint 2' in run.reason and 'unreachable' in run.reason
        assert run.mission.states == ['visited', 'failed']
        assert rig.sim.state.armed, 'an aborted mission leaves the vehicle where it is'
    finally:
        run.close(); rig.close()


def test_the_skip_policy_drops_an_unreachable_waypoint_and_finishes(tmp_path):
    rig = walled(tmp_path, [(20, 6)])
    run = mission(rig, 'mission=waypoints', 'waypoints=10,6;20.5,6.5;8,3', 'failure_policy=skip', 'stuck_s=10')
    try:
        run.start()
        assert fly(rig, run, limit_s=300.0, until=lambda: run.phase == 'done'), (run.blocking, rig.executor.reason)
        assert run.outcome == 'complete' and run.reason == 'visited 2 waypoints, skipped 1'
        assert run.mission.states == ['visited', 'skipped', 'visited']
        assert math.dist(rig.executor.pose[:2], (8.0, 3.0)) <= rig.profile.arrival_m
    finally:
        run.close(); rig.close()


def test_a_landing_finish_puts_the_vehicle_on_the_ground_and_only_a_launch_flies_again(tmp_path):
    rig = parked(tmp_path)
    run = mission(rig, 'goal_source=xy', 'goal=12.5,6.0', 'finish_action=land')
    try:
        run.start()
        assert fly(rig, run, limit_s=240.0, until=lambda: run.phase == 'done'), (run.blocking, rig.executor.reason)
        assert run.outcome == 'complete'
        assert not rig.sim.state.armed and rig.executor.state == 'WAITING'
        assert 'landed' in rig.executor.reason and 'launch intent' in rig.executor.reason
        assert [i['kind'] for i in run.acks if i['state'] == 'completed'] == ['launch', 'land']
        assert rig.executor.launch_phase is None and not rig.executor.launch_authorized
    finally:
        run.close(); rig.close()


def test_a_landing_that_never_confirms_fails_the_mission_with_the_reason(tmp_path):
    rig = parked(tmp_path, land_timeout_s=0.05)
    run = mission(rig, 'goal_source=xy', 'goal=12.5,6.0', 'finish_action=land')
    try:
        run.start()
        assert fly(rig, run, limit_s=240.0, until=lambda: run.phase == 'done'), (run.blocking, rig.executor.reason)
        assert run.outcome == 'failed'
        assert 'landing failed' in run.reason and 'not confirmed within 0.05 s' in run.reason
        assert rig.executor.state == 'FAILED'
    finally:
        run.close(); rig.close()


def test_a_yaw_to_intent_turns_to_the_heading_from_a_confirmed_hold(tmp_path):
    from dway.executor import heading_error_deg
    rig = parked(tmp_path)
    run = mission(rig, 'goal_source=xy', 'goal=12.5,6.0')
    try:
        run.start()
        assert fly(rig, run, limit_s=240.0,
                   until=lambda: run.phase == 'done' and rig.executor.state in ('HOLDING', 'COMPLETE')), \
            (run.blocking, rig.executor.reason)
        held = rig.executor.state
        # the mission is done but still owns the intent plane: its writer asks for the turn
        run.intents.send('yaw_to', rig.sim.sim_time_s, 'face back the way it came', heading_deg=180.)
        deadline = int(60.0 / 0.05)
        for _ in range(deadline):  # the mission is done; the rig flies the turn alone
            if ack(rig).get('kind') == 'yaw_to' and ack(rig)['state'] == 'completed': break
            rig.step()
        assert ack(rig)['state'] == 'completed', (rig.executor.reason, ack(rig))
        assert abs(heading_error_deg(180., rig.executor.vehicle.heading_deg)) <= rig.profile.turn_tolerance_deg
        assert rig.executor.state == held and 'heading 180' in rig.executor.reason, 'the turn ends where it began'
        assert rig.sim.state.armed and rig.executor.owns_control
    finally:
        run.close(); rig.close()
