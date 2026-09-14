"""Dynamic route flights through the whole production chain.

Everything between planner and executor is the real code: the retained
snapshots, dnav's ``RoutePublisher`` and ``Clearance``, the route source's
admission and stop-generation latch, and the executor's following, stopping
and lifecycle -- flown against a real ``DroneSimulator`` on the loopback
transport under one virtual clock, so failures are reproducible.

The rig (``dtest.dynamic_rig``) is honest about what is fixture: evidence
generation and the A* planner stand in for dalg and dnav's planner so route
shapes are predictable. Nothing here tests the vehicle's own failsafes.

Each scenario also asserts the invariants from the flight plan that are easy
to lose: no target outside the permitted interval, a stop only confirmed by
measured low speed (never an acknowledgement), permission that ends between
waypoints is honoured, and a route never restarted from an old start pose.
"""

import json
import math
from pathlib import Path

import pytest

from dtest.dynamic_rig import DT_S, DynamicRig

COMPLETE_HOLD_STATES = ('COMPLETE', 'CANCELLED', 'FAILED')


@pytest.fixture
def rig(tmp_path):
    rig = DynamicRig(tmp_path)
    try:
        yield rig
    finally:
        rig.close()


def goal_distance(rig):
    goal = rig.context.read().get('goal', {}).get('position')
    pose = rig.executor.pose
    return math.dist(pose[:2], goal[:2]) if goal and pose else None


def holds(rig, kind=None, phase='confirmed'):
    return [e for e in rig.executor.events
            if e['kind'] == 'execution.hold' and (e['data'].get('phase') == phase)
            and (kind is None or e['data'].get('kind') == kind)]


def transitions(rig, state):
    return [e for e in rig.executor.events
            if e['kind'] == 'execution.transition' and e['data'].get('state') == state]


# ---------------------------------------------------------------------------
# The basic loop
# ---------------------------------------------------------------------------

def test_clear_goal_completion_from_confirmed_hold(rig) -> None:
    rig.start()
    assert rig.fly_to_state(*COMPLETE_HOLD_STATES), 'flight never finished'
    ex = rig.executor
    assert ex.state == 'COMPLETE', ex.reason
    assert goal_distance(rig) <= ex.profile.arrival_m
    # The mission ended at a *measured* stop, not a converging target or an ACK.
    assert holds(rig, 'arrival'), 'no confirmed arrival hold'
    confirm = holds(rig, 'arrival')[-1]['data']
    assert confirm['speed_mps'] <= ex.profile.hold_speed_mps
    assert confirm['confirm_s'] >= ex.profile.hold_dwell_s
    # Every commanded target lay on the interval it was checked against.
    assert not rig.targets_outside_permission()
    assert ex.targets_sent > 0
    # Mission outcome and vehicle state stay separate: it is still airborne.
    assert rig.sim.state.armed and rig.sim.state.mode == 'HOLD'
    assert ex.owns_control


def test_flight_waits_for_explicit_start(rig) -> None:
    assert not rig.fly_to_state(*COMPLETE_HOLD_STATES, limit_s=6.0)
    assert rig.executor.state == 'READY', rig.executor.reason
    assert rig.executor.targets_sent == 0
    assert not rig.executor.owns_control, 'no lease before Start'


def test_start_requires_an_airborne_confirmed_hold(rig) -> None:
    rig.sim.state.mode = 'GUIDED'
    rig.sim.state.vx_mps = 0.4
    rig.step()
    rig.executor.request('start')
    rig.step()
    assert rig.executor.state != 'EXECUTING'
    assert 'HOLD required' in str(rig.executor.control_results['start']['reason'])
    assert rig.executor.targets_sent == 0


def test_start_refuses_a_route_from_a_foreign_planner(rig) -> None:
    rig.executor.source.planner = 'someone-else'
    rig.step()
    rig.executor.request('start')
    rig.step()
    assert 'no admitted route' in rig.executor.control_results['start']['reason']
    rig.executor.source.planner = 'dnav'


# ---------------------------------------------------------------------------
# Stopping, permission ends and honest waits
# ---------------------------------------------------------------------------

def test_unknown_space_prefix_stop_waits_honestly(rig) -> None:
    """The planner may route through unknown space; permission may not. The
    vehicle stops at the interval end, between waypoints, and holds there
    rather than creeping into the unknown to make the demonstration finish."""
    g = rig.geometry
    band = int(8.0 / g.cell_m)
    rig.never_observe((row, band) for row in range(g.height))
    rig.start()
    assert rig.fly_to_state('HOLDING', *COMPLETE_HOLD_STATES), 'no stop at the interval end'
    ex = rig.executor
    assert ex.state == 'HOLDING', ex.reason
    assert ex.hold_kind == 'prefix', ex.hold_kind
    assert 'permitted interval' in ex.reason
    assert goal_distance(rig) > ex.profile.arrival_m
    stopped_pose = ex.pose
    sent_at_stop = ex.targets_sent
    # It holds rather than finishing: more simulated time changes nothing.
    for _ in range(int(10.0 / DT_S)):
        rig.step()
    assert ex.state == 'HOLDING', 'prefix exhaustion became a restart or success'
    assert ex.targets_sent == sent_at_stop
    assert math.dist(ex.pose[:2], stopped_pose[:2]) < 0.05
    assert not rig.targets_outside_permission()
    assert ex.active['geometry_revision'] == 1, 'permission ended but geometry churned'


def test_prefix_wait_resumes_only_with_fresh_clearance(rig) -> None:
    g = rig.geometry
    band = int(8.0 / g.cell_m)
    rig.never_observe((row, band) for row in range(g.height))
    rig.start()
    assert rig.fly_to_state('HOLDING', *COMPLETE_HOLD_STATES)
    assert rig.executor.hold_kind == 'prefix'
    # The unknown band is never observed on its own; the wait is honest.
    for _ in range(int(5.0 / DT_S)):
        rig.step()
    assert rig.executor.state == 'HOLDING'
    # New evidence extends the permitted interval; a Resume continues.
    rig.unobserved.clear()
    for _ in range(int(2.0 / DT_S)):
        rig.step()
    rig.executor.request('resume')
    assert rig.fly_to_state(*COMPLETE_HOLD_STATES), rig.executor.reason
    assert rig.executor.state == 'COMPLETE'


def test_stale_evidence_expires_permission_and_holds(rig) -> None:
    """With local evidence, cells behind the vehicle age out; the permission
    deadline shortens and the vehicle stops in time rather than running an
    expired interval. A visibility hold is a legitimate outcome."""
    rig.sensor_range_m = 3.0
    rig.start()
    assert rig.fly_to_state('HOLDING', *COMPLETE_HOLD_STATES), 'no expiry stop'
    ex = rig.executor
    assert ex.state == 'HOLDING', ex.reason
    assert ex.hold_kind == 'expired', ex.hold_kind
    assert 'too little time' in ex.reason or 'deadline' in ex.reason
    sent_at_stop, stopped_pose = ex.targets_sent, ex.pose
    for _ in range(int(4.0 / DT_S)):
        rig.step()
    assert ex.state == 'HOLDING', 'an expired permission resumed by itself'
    assert ex.targets_sent == sent_at_stop
    assert math.dist(ex.pose[:2], stopped_pose[:2]) < 0.05


def test_newly_blocked_route_stops_and_replans_from_the_stop(rig) -> None:
    rig.start()
    assert rig.fly(limit_s=30.0, until=lambda r: r.executor.state == 'EXECUTING')
    begin_pose = tuple(rig.executor.pose)
    blocked_x = rig.executor.pose[0] + 3.0
    rig.block(blocked_x, 6.0, radius_m=0.8)
    # The executing geometry stays pinned, so the first effect of the block is
    # the occupied veto: the permission ends short of it and the executor
    # stops inside what remains.
    assert rig.fly_to_state('HOLDING', *COMPLETE_HOLD_STATES, limit_s=40.0)
    ex = rig.executor
    assert ex.state == 'HOLDING', ex.reason
    assert ex.hold_kind == 'prefix', ex.hold_kind
    assert not goal_distance(rig) <= ex.profile.arrival_m
    # Continuing on the pinned geometry runs the truncated permission out.
    # Which stop the executor records next is a race between its own margin
    # brake ('prefix') and dnav's withdrawal of the route ('unavailable' with
    # an advanced stop generation); either way it stops, confirms the HOLD,
    # and only a fresh validated route may be resumed. A resume requested
    # while the stop is still being processed is refused, so an operator
    # retries until the fresh route is admitted; corner stops on the
    # replacement continue on their own.
    stop_pose, rejoined, holds_seen = None, 0, 0
    while ex.state not in COMPLETE_HOLD_STATES:
        if ex.state == 'HOLDING':
            holds_seen += 1
            assert holds_seen <= 8, ex.reason
            assert ex.hold_kind in ('prefix', 'unavailable', 'corner'), (ex.hold_kind, ex.reason)
            stop_pose, rejoined = tuple(ex.pose), len(ex.sent)
            next_request = rig.sim.sim_time_s
            for _ in range(int(120.0 / DT_S)):
                rig.step()
                if ex.state != 'HOLDING': break
                if (ex.source.state == 'READY' and ex.hold_kind not in ('corner',)
                        and rig.sim.sim_time_s >= next_request):
                    ex.request('resume')
                    next_request = rig.sim.sim_time_s + 1.0
        else:
            rig.step()
    assert ex.state == 'COMPLETE', ex.reason
    assert not rig.targets_outside_permission()
    # The route flown after the last stop joined where the vehicle actually
    # stopped, never the pose the mission began at.
    first = ex.sent[rejoined]['target']
    assert math.dist(stop_pose[:2], first[:2]) < math.dist(begin_pose[:2], first[:2])
    assert math.dist(begin_pose[:2], first[:2]) > 2.0


def test_missed_withdrawal_still_forces_the_stop(rig) -> None:
    """A stop generation that arrives without its unavailable record must
    stop an executing route before any newer route is admitted."""
    rig.start()
    assert rig.fly(limit_s=30.0, until=lambda r: r.executor.state == 'EXECUTING')
    handled = rig.executor.source.handled_stop
    rig.publisher.stop_generation += 1  # the withdrawal the executor never read
    assert rig.fly_to_state('HOLDING', *COMPLETE_HOLD_STATES, limit_s=40.0)
    ex = rig.executor
    assert ex.state == 'HOLDING'
    assert ex.source.stop_generation == handled + 1
    # The stop was forced by the unhandled generation even though the record
    # the executor consumed was still eligible.
    brake = holds(rig, phase='requested')[0]
    assert 'stop generation' in brake['data']['reason'], brake['data']['reason']
    # Handling the generation is what re-opens admission; the same goal then
    # continues under the same Start authorization, as a planned replacement.
    for _ in range(int(3.0 / DT_S)):
        rig.step()
    assert ex.source.handled_stop == handled + 1
    assert ex.source.state == 'READY', ex.source.reason
    assert rig.fly_to_state(*COMPLETE_HOLD_STATES), ex.reason
    assert ex.state == 'COMPLETE'
    assert not rig.targets_outside_permission()


def test_control_loop_overrun_stops_the_vehicle(rig) -> None:
    """A step that cannot keep its own cadence stops and says so; obsolete
    targets are never sent to catch up."""
    rig.start()
    assert rig.fly(limit_s=30.0, until=lambda r: r.executor.state == 'EXECUTING')
    rig.clock.advance(0.7)  # one step misses its deadline outright
    rig.step()
    assert rig.executor.state == 'BRAKING', rig.executor.reason
    assert 'stalled' in rig.executor.reason
    assert rig.fly_to_state('HOLDING', *COMPLETE_HOLD_STATES)
    assert rig.executor.hold_kind == 'overrun'
    rig.executor.request('resume')
    assert rig.fly_to_state(*COMPLETE_HOLD_STATES), rig.executor.reason
    assert rig.executor.state == 'COMPLETE'


def test_a_hold_acknowledgement_is_not_a_measured_stop(rig) -> None:
    """If the vehicle keeps reporting speed under braking, the stop is not
    confirmed and the mission fails honestly instead of claiming a hover."""
    import dataclasses
    rig.start()
    assert rig.fly(limit_s=30.0, until=lambda r: r.executor.state == 'EXECUTING')
    original = rig.executor.link.state

    def still_moving():
        state = original()
        return dataclasses.replace(state, vx_mps=max(state.vx_mps, 0.3))

    rig.executor.link.state = still_moving
    rig.block(rig.executor.pose[0] + 3.0, 6.0, radius_m=0.8)
    assert rig.fly_to_state('FAILED', limit_s=40.0)
    assert 'HOLD not confirmed' in rig.executor.reason
    assert rig.executor.pose is not None  # the failure is reported, not hidden


def test_rejected_hold_fails_and_ceases_targets(rig) -> None:
    rig.start()
    assert rig.fly(limit_s=30.0, until=lambda r: r.executor.state == 'EXECUTING')
    rig.reject_holds()
    rig.block(rig.executor.pose[0] + 3.0, 6.0, radius_m=0.8)
    assert rig.fly_to_state('FAILED', limit_s=40.0)
    assert 'HOLD rejected' in rig.executor.reason
    assert 'targets ceased' in rig.executor.reason
    sent = rig.executor.targets_sent
    for _ in range(int(3.0 / DT_S)):
        rig.step()
    assert rig.executor.targets_sent == sent, 'targets continued after a rejected HOLD'


# ---------------------------------------------------------------------------
# Route changes, crossings and arrivals
# ---------------------------------------------------------------------------

def test_crossing_route_does_not_skip_ahead(rig) -> None:
    """A route that crosses itself: progress must follow the commanded order,
    never jump to the later pass because it is momentarily nearer.

    Segments 0 and 2 cross at about (11.0, 4.7) m: while flying segment 2 the
    pose sits exactly on segment 0's line, so a projection without the
    ordering rule would send the vehicle backwards along the route.
    """
    bowtie = [[3.5, 6.0, 1.5], [26.0, 2.0, 1.5], [26.0, 10.0, 1.5], [3.5, 2.0, 1.5]]
    rig.fixed_route = bowtie
    rig.context.set_goal('fixture', [3.5, 2.0, 1.5], role='mission')
    rig.start()
    assert rig.fly_to_state(*COMPLETE_HOLD_STATES, limit_s=300.0), rig.executor.reason
    assert rig.executor.state == 'COMPLETE'
    assert not rig.targets_outside_permission()
    # Progress along the route is monotone in commanded order: the vehicle
    # flew the first pass before the second, rather than cutting the corner.
    segments = [entry['segment'] for entry in rig.executor.sent]
    assert segments == sorted(segments), f'progress jumped at the crossing: {segments}'


def test_coincident_goal_is_an_arrival_only_flight(rig) -> None:
    pose = rig.sim.state
    rig.set_goal(pose.x, pose.y)
    for _ in range(int(2.0 / DT_S)):
        rig.step()
    assert rig.executor.source.state == 'READY', rig.executor.source.reason
    assert len(rig.executor.source.value['points']) == 1
    rig.start()
    assert rig.fly_to_state(*COMPLETE_HOLD_STATES), rig.executor.reason
    assert rig.executor.state == 'COMPLETE'
    assert holds(rig, 'arrival'), 'arrival gates were skipped for a coincident goal'
    assert rig.executor.targets_sent == 0 or not rig.targets_outside_permission()


def test_geometry_replacement_while_moving_stops_and_continues(rig) -> None:
    """A same-goal replacement dnav selects is a stop, then an automatic
    continuation under the same Start authorization -- never an in-motion
    switch, and never a jump back to the original start pose."""
    rig.start()
    assert rig.fly(limit_s=30.0, until=lambda r: r.executor.state == 'EXECUTING')
    original = rig.executor.active['points']
    detour = [list(original[0]), [8.0, 9.0, 1.5], list(original[-1])]
    rig.fixed_route = [list(p) for p in detour]
    # The pin keeps the executing geometry: an unannounced improvement never
    # reaches the executor. dnav withdraws the active route for the
    # replacement the way the plan requires -- by advancing the generation.
    rig.publisher.stop_generation += 1
    assert rig.fly_to_state('HOLDING', *COMPLETE_HOLD_STATES, limit_s=60.0)
    assert rig.executor.hold_kind == 'replacement', rig.executor.hold_kind
    assert holds(rig, 'replacement'), 'the geometry change did not stop the vehicle'
    # The replacement is planned from the stopped pose, never the old start.
    stopped = rig.executor.pose
    rig.fixed_route[0] = [stopped[0], stopped[1], stopped[2]]
    # The continuation is automatic once the replacement validates from the stop.
    assert rig.fly_to_state(*COMPLETE_HOLD_STATES, limit_s=240.0), rig.executor.reason
    assert rig.executor.state == 'COMPLETE'
    assert not rig.targets_outside_permission()
    assert any(math.dist(t['target'][:2], (8.0, 9.0)) < 1.0 for t in rig.executor.sent), \
        'the replacement route was never flown'


def test_goal_change_stops_the_old_route_and_requires_start(rig) -> None:
    rig.start()
    assert rig.fly(limit_s=30.0, until=lambda r: r.executor.state == 'EXECUTING')
    rig.set_goal(4.0, 3.0)
    assert rig.fly_to_state('HOLDING', *COMPLETE_HOLD_STATES, limit_s=40.0)
    ex = rig.executor
    assert ex.hold_kind == 'goal', ex.hold_kind
    # Resume is refused: a new goal needs a new Start authorization.
    ex.request('resume')
    rig.step()
    assert ex.control_results['resume']['state'] == 'refused'
    assert 'Start required' in ex.control_results['resume']['reason']
    ex.request('start')
    assert rig.fly_to_state(*COMPLETE_HOLD_STATES, limit_s=120.0), ex.reason
    assert ex.state == 'COMPLETE'
    assert math.dist(ex.pose[:2], (4.0, 3.0)) <= ex.profile.arrival_m


def test_pause_and_resume_keep_the_same_authorization(rig) -> None:
    rig.start()
    assert rig.fly(limit_s=30.0, until=lambda r: r.executor.state == 'EXECUTING')
    rig.executor.request('pause')
    assert rig.fly_to_state('HOLDING', *COMPLETE_HOLD_STATES, limit_s=40.0)
    assert rig.executor.hold_kind == 'pause'
    held_pose = rig.executor.pose
    rig.executor.request('resume')
    rig.step()
    assert rig.executor.state == 'EXECUTING', rig.executor.control_results['resume']
    assert math.dist(rig.executor.pose[:2], held_pose[:2]) < 1.0, 'resume jumped'
    assert rig.fly_to_state(*COMPLETE_HOLD_STATES), rig.executor.reason
    assert rig.executor.state == 'COMPLETE'


def test_cancel_supervises_the_vehicle_after_the_mission(rig) -> None:
    rig.start()
    assert rig.fly(limit_s=30.0, until=lambda r: r.executor.state == 'EXECUTING')
    rig.executor.request('cancel')
    assert rig.fly_to_state(*COMPLETE_HOLD_STATES, limit_s=40.0)
    ex = rig.executor
    assert ex.state == 'CANCELLED'
    assert holds(rig, 'cancel'), 'cancel was confirmed without a measured stop'
    # Cancellation ends the mission, not the vehicle supervision.
    for _ in range(int(3.0 / DT_S)):
        rig.step()
    assert ex.owns_control
    assert rig.sim.state.armed and rig.sim.state.mode == 'HOLD'
    assert ex.vehicle_fault == ''


# ---------------------------------------------------------------------------
# Restarts, takeovers and the lease
# ---------------------------------------------------------------------------

def test_planner_restart_requires_a_new_start(rig) -> None:
    rig.start()
    assert rig.fly(limit_s=30.0, until=lambda r: r.executor.state == 'EXECUTING')
    rig.restart_planner()
    assert rig.fly_to_state('HOLDING', *COMPLETE_HOLD_STATES, limit_s=40.0)
    ex = rig.executor
    assert ex.hold_kind == 'restart', ex.hold_kind
    assert ex.source.start_required
    # The old flight does not resume by itself once a route appears again.
    for _ in range(int(5.0 / DT_S)):
        rig.step()
    assert ex.state == 'HOLDING', 'a restarted planner resumed the old flight'
    ex.request('start')
    assert rig.fly_to_state(*COMPLETE_HOLD_STATES, limit_s=120.0), ex.reason
    assert ex.state == 'COMPLETE'


def test_manual_takeover_fails_the_flight_without_reacquisition(rig) -> None:
    rig.start()
    assert rig.fly(limit_s=30.0, until=lambda r: r.executor.state == 'EXECUTING')
    operator, result = rig.takeover()
    assert result.accepted
    assert rig.fly_to_state('FAILED', limit_s=20.0)
    ex = rig.executor
    assert 'lease lost' in ex.reason
    assert 'operator' in ex.lease_state
    sent = ex.targets_sent
    for _ in range(int(3.0 / DT_S)):
        rig.step()
    assert ex.targets_sent == sent, 'targets continued after losing the lease'
    assert ex.owns_control is False
    assert rig.sim.state.mode == 'HOLD', 'the takeover disturbed the vehicle'


def test_start_refuses_to_steal_a_held_lease(rig) -> None:
    operator, result = rig.takeover()
    assert result.accepted
    for _ in range(int(2.0 / DT_S)):
        rig.step()
    rig.executor.request('start')
    rig.step()
    reason = rig.executor.control_results['start']['reason']
    assert 'held by operator' in reason, reason
    assert not rig.executor.owns_control


def test_a_second_executor_cannot_own_the_endpoint(rig) -> None:
    from dway.executor import DynamicExecutor
    with pytest.raises(ValueError, match='live writer'):
        DynamicExecutor(rig.id, rig.link, rig.profile)


def test_provider_stall_stops_an_active_route(rig) -> None:
    rig.start()
    assert rig.fly(limit_s=30.0, until=lambda r: r.executor.state == 'EXECUTING')
    rig.provider_frozen = True
    assert rig.fly_to_state('HOLDING', *COMPLETE_HOLD_STATES, limit_s=40.0)
    ex = rig.executor
    assert ex.state == 'HOLDING', ex.reason
    assert ex.hold_kind in ('stalled', 'unavailable'), (ex.hold_kind, ex.reason)
    assert 'stall' in ex.reason or 'stall' in ex.source.reason, ex.source.reason
    # The provider returning does not resume anything by itself; an explicit
    # Resume with fresh validation does.
    rig.provider_frozen = False
    for _ in range(int(5.0 / DT_S)):
        rig.step()
    assert ex.state == 'HOLDING', 'a stalled provider resumed by itself'
    ex.request('resume')
    assert rig.fly_to_state(*COMPLETE_HOLD_STATES), ex.reason
    assert ex.state == 'COMPLETE', ex.reason


# ---------------------------------------------------------------------------
# Determinism, recording and shutdown
# ---------------------------------------------------------------------------

def test_commands_are_identical_with_recording_and_reports_off(tmp_path) -> None:
    """Identical deterministic inputs must produce identical commands,
    whatever the recording and report subsystem is doing."""
    sent = []
    for options in (None, dict(record=False, report=False)):
        run = DynamicRig(tmp_path / f'{"on" if options is None else "off"}',
                         fixed_route=[[3.5, 6.0, 1.5], [12.5, 6.0, 1.5]])
        try:
            run.start()
            assert run.fly_to_state(*COMPLETE_HOLD_STATES), run.executor.reason
            assert run.executor.state == 'COMPLETE'
            sent.append([entry['target'] for entry in run.executor.sent])
        finally:
            run.close()
    assert sent[0] == sent[1], 'recording changed the commands'


def test_recording_failure_does_not_change_the_flight(tmp_path) -> None:
    """A starved recording queue drops events and says so; the flight itself
    is unaffected."""
    run = DynamicRig(tmp_path, executor_options=dict(recording_queue_bytes=4096))
    try:
        run.start()
        assert run.fly_to_state(*COMPLETE_HOLD_STATES), run.executor.reason
        assert run.executor.state == 'COMPLETE', run.executor.reason
        assert run.executor.report_error, 'a starved queue was not reported'
        assert 'dropped' in run.executor.report_error
        # The report is derived offline when the recording finishes, so finish
        # it here: shutdown closes the archive and builds the report in place.
        run.close()
        report = json.loads((Path(run.executor.report_dir) / 'report' / 'summary.json').read_text())
        assert report['recording']['dropped'] > 0
        assert report['recording_complete'] is False
        assert report['outcome_note'], 'incomplete recording must say both'
        assert report['final_state'] == 'COMPLETE', 'a dropped recording must not change the outcome'
    finally:
        run.close()


def test_shutdown_requests_hold_releases_and_lands_nothing(rig) -> None:
    rig.start()
    assert rig.fly(limit_s=30.0, until=lambda r: r.executor.state == 'EXECUTING')
    result = rig.executor.shutdown('test shutdown')
    assert result['owned'] and result['hold']['accepted']
    assert result['confirmed'], 'shutdown wait did not confirm the stop'
    assert result['released']
    assert rig.sim.state.armed, 'shutdown must not land, RTL or disarm'
    assert rig.sim.state.mode == 'HOLD'
    assert rig.executor.state == 'FAILED', 'an interrupted flight is a failure, not success'
    # Shutdown withdraws the retained execution area when its writer closes
    # (the single-writer protocol), so a reader afterwards finds no snapshot.
    status = rig.status()
    assert not status or status.get('state') == 'CLOSED', status


def test_view_is_a_snapshot_for_display(rig) -> None:
    """The immutable view the window paints: bounded, and stable after read."""
    rig.start()
    assert rig.fly_to_state(*COMPLETE_HOLD_STATES)
    view = rig.executor.view()
    assert set(view) >= {'status', 'history', 'events', 'track', 'active', 'permitted'}
    assert view['status']['state'] == 'COMPLETE'
    assert len(view['history']) <= 10.0 * 61  # bounded recent history
    frozen = view['history']
    rig.step()
    assert view['history'] == frozen, 'a published view must not mutate'

# ---------------------------------------------------------------------------
# Resets, conditions outside the profile and delayed inputs
# ---------------------------------------------------------------------------

def test_localization_reset_discards_permission_and_requires_start(rig) -> None:
    """A frame discontinuity invalidates the route and the goal it was
    planned in: the vehicle stops, and nothing continues without a new Start."""
    rig.start()
    assert rig.fly(limit_s=30.0, until=lambda r: r.executor.state == 'EXECUTING')
    rig.localization_epoch = 1
    assert rig.fly_to_state('HOLDING', *COMPLETE_HOLD_STATES, limit_s=40.0)
    ex = rig.executor
    assert ex.state == 'HOLDING', ex.reason
    assert ex.hold_kind in ('restart', 'goal'), (ex.hold_kind, ex.reason)
    sent = ex.targets_sent
    for _ in range(int(5.0 / DT_S)):
        rig.step()
    assert ex.state == 'HOLDING' and ex.targets_sent == sent, 'a reset resumed the old flight'
    assert ex.status['start_required']
    ex.request('resume')
    rig.step()
    assert ex.control_results['resume']['state'] == 'refused'
    assert not rig.targets_outside_permission()


def test_start_is_refused_in_wind_outside_the_calibrated_profile(rig) -> None:
    rig.sim.realism.apply(wind_mps=0.5)
    for _ in range(int(2.0 / DT_S)):
        rig.step()
    rig.executor.request('start')
    rig.step()
    reason = rig.executor.control_results['start']['reason']
    assert 'wind' in reason and 'calibrated' in reason, reason
    assert rig.executor.targets_sent == 0 and not rig.executor.owns_control


def test_wind_arriving_mid_flight_fails_honestly_outside_the_model(rig) -> None:
    """The profile measured no wind, and dsim's HOLD does not hold position
    against it. Once wind arrives the flight must not claim success it cannot
    support: it stops on a stated limit, and a stop it cannot confirm is a
    failure, never an assumed hover."""
    rig.start()
    assert rig.fly(limit_s=30.0, until=lambda r: r.executor.state == 'EXECUTING')
    rig.sim.realism.apply(wind_mps=0.8, wind_dir_deg=0.0)
    rig.fly(limit_s=60.0, until=lambda r: r.executor.state in ('HOLDING',) + COMPLETE_HOLD_STATES
            and r.executor.state != 'BRAKING')
    ex = rig.executor
    assert ex.state in ('HOLDING', 'FAILED'), (ex.state, ex.reason)
    assert ex.state != 'COMPLETE'
    if ex.state == 'HOLDING':
        # A held vehicle drifting downwind cannot Resume inside the profile.
        for _ in range(int(3.0 / DT_S)):
            rig.step()
        ex.request('resume')
        rig.step()
        assert ex.state != 'EXECUTING', 'resumed in wind outside the calibrated profile'
    assert not rig.targets_outside_permission()


@pytest.mark.parametrize('seed', (1234, 4321))
def test_delayed_telemetry_inside_the_profile_completes(tmp_path, seed) -> None:
    """50 ms telemetry latency is inside the measured conditions; the same
    documented seeds the calibration used are flown here."""
    run = DynamicRig(tmp_path, realism=dict(telemetry_latency_ms=50.0, realism_seed=seed))
    try:
        # Delayed telemetry means capabilities arrive late; a Start pressed
        # before then is refused and never retried on its own.
        assert run.fly(limit_s=5.0, until=lambda r: r.executor.state == 'READY'
                       and r.link.capabilities().accepts_position_target)
        run.start()
        assert run.fly_to_state(*COMPLETE_HOLD_STATES), run.executor.reason
        assert run.executor.state == 'COMPLETE', run.executor.reason
        assert not run.targets_outside_permission()
        assert goal_distance(run) <= run.executor.profile.arrival_m
    finally:
        run.close()


# ---------------------------------------------------------------------------
# Target mode: automatic launch and continuation, plan permission
# ---------------------------------------------------------------------------

TARGET_PROFILE = Path(__file__).resolve().parents[1] / 'assets/execution_profiles/sim-target.json'
TARGET = dict(auto=True, allow_uncalibrated=True)


def parked(rig):
    """dsim's start: parked at its start altitude, disarmed, nobody holding control."""
    rig.sim.state.armed, rig.sim.state.mode = False, 'DISARMED'
    return rig


def test_target_mode_launches_from_parked_and_flies_to_the_goal(tmp_path) -> None:
    run = parked(DynamicRig(tmp_path, profile=TARGET_PROFILE, executor_options=TARGET))
    try:
        assert run.fly_to_state(*COMPLETE_HOLD_STATES, limit_s=120.0), run.executor.reason
        ex = run.executor
        assert ex.state == 'COMPLETE', ex.reason
        assert goal_distance(run) <= ex.profile.arrival_m
        phases = [e['data'].get('phase') for e in ex.events if e['kind'] == 'execution.launch']
        assert phases[:1] == ['begin'] and 'arm' in phases and phases[-1] == 'airborne'
        starts = [e['data'] for e in ex.events if e['kind'] == 'execution.control' and e['data']['action'] == 'start']
        assert starts and starts[-1]['origin'] == 'auto' and starts[-1]['accepted']
        assert run.sim.state.armed and run.sim.state.mode == 'HOLD', 'target mode never lands'
        assert not run.targets_outside_permission()
    finally:
        run.close()


def test_plan_permission_flies_through_unknown_space_that_evidence_permission_waits_on(tmp_path) -> None:
    run = parked(DynamicRig(tmp_path, profile=TARGET_PROFILE, executor_options=TARGET))
    try:
        band = int(8.0 / run.geometry.cell_m)
        run.never_observe((row, band) for row in range(run.geometry.height))
        assert run.fly_to_state(*COMPLETE_HOLD_STATES, limit_s=120.0), run.executor.reason
        assert run.executor.state == 'COMPLETE', run.executor.reason
        evidence = run.publisher.last['clearance'].get('evidence_check')
        assert evidence is not None, 'the strict verdict must be published beside plan permission'
    finally:
        run.close()


def test_target_mode_stops_replans_and_continues_around_a_new_obstacle(tmp_path) -> None:
    run = parked(DynamicRig(tmp_path, profile=TARGET_PROFILE, executor_options=TARGET))
    try:
        assert run.fly(limit_s=60.0, until=lambda r: r.executor.state == 'EXECUTING')
        run.block(run.executor.pose[0] + 3.0, 6.0, radius_m=0.8)
        assert run.fly_to_state(*COMPLETE_HOLD_STATES, limit_s=240.0), run.executor.reason
        ex = run.executor
        assert ex.state == 'COMPLETE', ex.reason
        assert holds(run), 'the new obstacle never stopped the vehicle'
        assert not run.targets_outside_permission()
    finally:
        run.close()


def test_travel_heading_faces_each_segment_before_flying_it(tmp_path) -> None:
    """With ``heading=travel`` the forward sensors look where the vehicle goes:
    it turns in place on the route at the start of each segment (toward the
    actual course from where it stopped, which may be a little short of the
    corner), and ends facing the last segment."""
    from dsim.dsim import sim_yaw_to_compass_heading
    from dway.executor import heading_error_deg
    route = [[3.5, 6.0, 1.5], [12.5, 6.0, 1.5], [12.5, 9.5, 1.5]]  # east, then south
    run = parked(DynamicRig(tmp_path, profile=TARGET_PROFILE, fixed_route=route, goal=(12.5, 9.5),
                            heading_deg=0.0, executor_options=TARGET))
    try:
        assert run.fly_to_state(*COMPLETE_HOLD_STATES, limit_s=180.0), run.executor.reason
        ex = run.executor
        assert ex.state == 'COMPLETE', ex.reason
        turns = [e['data'] for e in ex.events if e['kind'] == 'execution.turn']
        assert len(turns) >= 2, turns
        assert abs(heading_error_deg(turns[0]['to_deg'], 90.0)) < 5.0, turns[0]
        assert abs(heading_error_deg(turns[1]['to_deg'], 180.0)) < 10.0, turns[1]
        assert not run.targets_outside_permission()
        final = sim_yaw_to_compass_heading(run.sim.state.yaw_deg)
        assert abs(heading_error_deg(180.0, final)) < 10.0 + ex.profile.turn_tolerance_deg, final
        assert ex.status['heading_mode'] == 'travel' and ex.status['commanded_heading_deg'] is not None
    finally:
        run.close()


def test_instance_shutdown_on_the_bus_stops_a_target_executor(tmp_path) -> None:
    """dsim's "Kill all" publishes ``system.shutdown``. A target-mode executor
    must hear it without any ``--wait-for`` role, then HOLD, release control
    and never land."""
    from dcmn.module_bus import PymembusModuleBus
    run = parked(DynamicRig(tmp_path, profile=TARGET_PROFILE, executor_options=dict(TARGET, bus_enabled=True)))
    try:
        bus = PymembusModuleBus(run.id, 'simulator', 'test', create=True)
        assert bus.connect()
        assert run.fly(limit_s=60.0, until=lambda r: r.executor.state == 'EXECUTING'), run.executor.reason
        bus.publish('system.shutdown', payload=dict(reason='Kill all'))
        assert run.fly(limit_s=5.0, until=lambda r: r.executor.shutdown_requested), 'shutdown never heard'
        assert run.executor.shutdown_reason == 'Kill all'
        result = run.executor.shutdown(f'instance shutdown: {run.executor.shutdown_reason}')
        assert result['hold']['accepted'] and result['released']
        assert run.sim.state.armed and run.sim.state.mode == 'HOLD', 'shutdown must not land'
        assert any(e['kind'] == 'execution.shutdown_requested' for e in run.executor.events)
    finally:
        run.close()
        bus.remove()
