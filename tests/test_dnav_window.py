"""The dnav tabs: what the operator can do, and what they are told.

Built against real widgets on a hidden root, because geometry, variable
tracing and event dispatch are exactly the parts a mock gets wrong.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from dtest.synthetic import SOURCE_ID, SyntheticRoom
from dcmn.maps import GridGeometry, MapPublisher
from dnav import route as R
from dnav.plan import NavRun
from dnav.policy import load_policy
from dtest.tkfixture import hidden_tk

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def run(tmp_path):
    """A NavRun with a session context and a swept room already published."""
    from dcmn.context import Context
    instance = f'nav-{uuid.uuid4().hex[:8]}'
    context = Context(instance)
    context.start(tmp_path / '20260101-000000-abcdef12', clock_epoch=3)
    context.publish_pose(dict(x_m=1.5, y_m=1.5, z_m=1.5, heading_deg=90., roll_deg=0., pitch_deg=0.),
                         12., clock_epoch=3)
    geometry = GridGeometry.from_extent(20., 20., .5)
    publisher = MapPublisher(instance, geometry,
                             [dict(id=SOURCE_ID, sensor=SOURCE_ID,
                                   sensor_type='fixture', algorithm='synthetic')],
                             context=dict(clock_epoch=3, clock_domain_id=instance))
    room = SyntheticRoom(geometry)
    for step in range(12):
        room.sweep(1. + step)
        publisher.publish(SOURCE_ID, room.occupancy, room.observed_ms, 1. + step)
    session = NavRun(instance, ROOT, policy=load_policy('default', ROOT),
                     goal=(15.5, 10.0, None))
    session.step()
    session.replan(force=True)
    try:
        yield session
    finally:
        session.close(); publisher.close(); context.close()


@pytest.fixture
def tabs(run):
    """The Plan and Cost tabs on a hidden root, sharing one run."""
    import tkinter as tk
    from tkinter import ttk
    from dnav.dnav import CostTab, PlanTab

    with hidden_tk() as root:
        plan_frame, cost_frame = ttk.Frame(root), ttk.Frame(root)
        plan = PlanTab(plan_frame, run, tk, ttk)
        cost = CostTab(cost_frame, run, tk, ttk)
        root.update_idletasks()
        yield root, plan, cost


# -- the Plan tab ------------------------------------------------------------

def test_a_goal_from_the_command_line_shows_in_the_form(tabs, run):
    """A goal from the context -- here, the command line's -- is as real as one typed in."""
    _, plan, _ = tabs
    assert plan.goal_vars['x'].get() == '15.50'
    assert plan.goal_vars['y'].get() == '10.00'
    assert plan.goal_vars['z'].get() == ''


def test_clicking_the_map_sets_the_goal_in_metres(tabs, run):
    _, plan, _ = tabs
    plan.pane.set_grid(run.session.latest(SOURCE_ID), sim_now_s=12.)
    plan.pane.refresh(force=True)
    plan._clicked(17.25, 4.5)
    assert plan.goal_vars['x'].get() == '17.25'
    assert run.goal_xy == (17.25, 4.5)
    assert run.route.waypoints[-1].x_m == pytest.approx(17.25)


def test_a_goal_that_is_not_a_number_is_refused_in_the_form(tabs, run):
    _, plan, _ = tabs
    before = run.goal_xy
    plan.goal_vars['x'].set('over there')
    plan._set_goal()
    assert run.goal_xy == before
    assert 'numbers' in plan.route_rows['reason'].get()


def test_clearing_the_goal_empties_the_form_and_stops_planning(tabs, run):
    _, plan, _ = tabs
    plan._clear_goal()
    assert plan.goal_vars['x'].get() == ''
    assert run.route.status == R.NO_GOAL
    plan.paint_text(12.)
    assert plan.route_rows['status'].get() == R.NO_GOAL


def test_the_route_readout_says_what_was_planned(tabs, run):
    _, plan, _ = tabs
    plan.paint_text(12.)
    assert plan.route_rows['status'].get() == R.OK
    assert plan.route_rows['reason'].get() == 'planned'
    assert plan.route_rows['length'].get().endswith(' m')
    assert 'ms' in plan.route_rows['planning'].get()
    assert plan.route_rows['vs straight line'].get() != '--'


def test_a_failure_is_shown_as_text_rather_than_an_empty_map(tabs, run):
    """The whole point of the status vocabulary: no silent blank canvas."""
    _, plan, _ = tabs
    run.policy = load_policy('cautious', ROOT)
    run.replan(force=True)
    plan.paint_text(12.)
    assert plan.route_rows['status'].get() == R.NO_ROUTE
    assert plan.route_rows['reason'].get().strip()
    assert plan.route_rows['length'].get() == '--'


def test_the_map_source_panel_names_the_producer_and_its_freshness(tabs, run):
    _, plan, _ = tabs
    plan.paint_text(12.)
    assert plan.source_rows['producer'].get() == 'dalg'
    assert plan.source_rows['source'].get() == SOURCE_ID
    assert plan.source_rows['state'].get() in ('fresh', 'stale')
    assert plan.source_rows['cell'].get() == '0.5 m'
    assert plan.source_rows['extent'].get() == '20 x 20 m'


def test_choosing_a_planner_replans_with_it(tabs, run):
    _, plan, _ = tabs
    plan.planner_var.set('control')
    plan._planner_changed()
    assert run.planner_name == 'control'
    assert run.route.planner == 'control'


def test_the_plan_pane_draws_the_cost_map_and_the_route(tabs, run):
    _, plan, _ = tabs
    plan.paint_map(12.)
    assert plan.pane.photo is not None
    assert plan.pane.state.mode == 'cost'
    assert plan.pane.state.cost is not None
    assert len(plan.pane.state.overlay.route) >= 2
    assert plan.pane.state.overlay.goal == run.goal_xy
    assert plan.pane.state.overlay.inflation_m == run.policy.inflation_m


def test_the_report_image_shows_the_final_route_even_if_the_tab_was_not_visible(tabs, run):
    """The pane only repaints while its tab is showing.

    A report that snapshotted whatever was last painted filed a route from the
    middle of a flight under a summary describing its end.
    """
    _, plan, _ = tabs
    plan.paint_map(12.)                     # the operator looks once...
    early = tuple(plan.pane.state.overlay.route)
    run.set_goal(17.25, 4.5)                # ...then the plan moves on unseen
    run.replan(force=True)
    assert tuple(plan.pane.state.overlay.route) == early, 'precondition: pane is stale'

    image = plan.snapshot()
    assert image is not None
    assert tuple(plan.pane.state.overlay.route) == tuple(run.route.points)
    assert plan.pane.state.overlay.goal == (17.25, 4.5)


# -- the Cost tab ------------------------------------------------------------

def test_the_cost_tab_offers_every_layer_and_the_stack(tabs, run):
    from dnav.dnav import STACK

    _, _, cost = tabs
    cost.paint_map(12.)
    choices = list(cost.view_combo['values'])
    assert choices[0] == STACK
    assert f'evidence: {SOURCE_ID}' in choices
    assert f'cost: {SOURCE_ID}' in choices
    assert f'blocked on evidence: {SOURCE_ID}' in choices


def test_selecting_evidence_shows_belief_and_selecting_cost_shows_the_policy(tabs, run):
    _, _, cost = tabs
    cost.view_var.set(f'evidence: {SOURCE_ID}')
    cost.paint_map(12.)
    assert cost.pane.state.mode == 'occupancy'
    cost.view_var.set(f'cost: {SOURCE_ID}')
    cost.paint_map(12.)
    assert cost.pane.state.mode == 'cost'
    assert cost.pane.state.cost is not None


def test_blocked_on_evidence_tints_the_source_margin_over_its_own_evidence(tabs, run):
    _, _, cost = tabs
    cost.view_var.set(f'blocked on evidence: {SOURCE_ID}')
    cost.paint_map(12.)
    assert cost.pane.state.mode == 'blocked'
    layer = run.cost_map.layer(SOURCE_ID)
    assert cost.pane.state.cost is not None
    assert (cost.pane.state.cost == layer.cost).all()
    assert cost.pane.photo is not None
    assert 'blocked by the policy' in cost.pane.header.get()


def test_with_one_source_the_stack_says_why_it_matches_the_cost_view(tabs, run):
    """Two identical pictures under two names read as a bug unless named."""
    from dnav.dnav import STACK

    _, _, cost = tabs
    cost.view_var.set(STACK)
    cost.paint_map(12.)
    assert len(run.cost_map.layers) == 1
    header = cost.pane.header.get()
    assert 'one source' in header and f'cost: {SOURCE_ID}' in header


def test_the_policy_values_are_listed_beside_the_layers(tabs, run):
    _, _, cost = tabs
    cost.paint_text(12.)
    assert cost.policy_rows['name'].get() == 'default'
    assert cost.policy_rows['combine'].get() == 'max'
    assert cost.policy_rows['digest'].get() == run.policy.digest[:12]
    assert cost.policy_rows['occupied_threshold'].get() == '0.5'
    assert ' of ' in cost.policy_rows['blocked cells'].get()


def test_reloading_a_widened_margin_blocks_more_and_moves_the_route(tabs, run, tmp_path):
    """The review gate: a bigger margin visibly fattens what a route must avoid."""
    _, plan, cost = tabs
    cost.paint_text(12.)
    before = int(run.cost_map.blocked.sum())
    before_length = run.route.length_m
    assert run.route.ok

    path = tmp_path / 'wide.json'
    path.write_text(json.dumps({'schema': 'dvision2.cost-policy.v1',
                                'name': 'wide', 'inflation_m': 0.9}), encoding='utf-8')
    run.policy = load_policy(str(path), ROOT)
    cost._reload()

    cost.paint_text(12.)
    assert 'reloaded' in cost.notice.get()
    assert int(run.cost_map.blocked.sum()) > before
    assert cost.policy_rows['inflation_m'].get() == '0.9'
    assert run.route.ok and run.route.length_m != before_length


def test_a_policy_that_cannot_be_reloaded_says_so_rather_than_silently_keeping_the_old(
        tabs, run, tmp_path):
    _, _, cost = tabs
    path = tmp_path / 'broken.json'
    path.write_text('{ not json', encoding='utf-8')
    run.policy = run.policy.__class__(name='x', path=path, digest='old')
    cost._reload()
    assert cost.notice.get().startswith('not reloaded')


def test_a_vehicle_already_at_its_goal_is_described_not_crashed_on(tabs, run):
    """Every tour ends here: a zero-length control has no ratio to format."""
    _, plan, _ = tabs
    run.set_goal(1.6, 1.6)                   # the same cell the vehicle is in
    run.replan(force=True)
    assert run.route.ok
    plan.paint_text(12.)
    assert plan.route_rows['vs straight line'].get() == 'at the goal'


@pytest.mark.parametrize('detour,expected', [
    (None, '--'),
    ({'control_blocked': True}, 'straight line blocked'),
    ({'control_blocked': False, 'length_ratio': None, 'cost_ratio': None}, 'at the goal'),
    ({'control_blocked': False, 'length_ratio': 1.25, 'cost_ratio': 0.5},
     '1.25x length, 0.50x cost'),
])
def test_the_detour_readout_handles_every_shape_it_can_be_given(detour, expected):
    from dnav.dnav import describe_detour

    route = R.Route(status=R.OK, waypoints=(R.Waypoint(0., 0., 0.), R.Waypoint(1., 0., 0.)),
                    cost=1., length_m=1.)
    assert expected in describe_detour(route, detour)


def test_a_run_that_crashes_is_reported_as_partial_with_the_reason(tmp_path, monkeypatch):
    """A traceback must not be filed as a clean, complete run."""
    import dnav.dnav as entry

    written = {}

    class Probe:
        partial, reason, id = False, '', 'probe'
        def set_goal(self, *args, **kwargs): return False
        def close(self, partial=False):
            written.update(partial=self.partial or partial, reason=self.reason)
            return None
    monkeypatch.setattr(entry, 'NavRun', lambda *a, **k: Probe())
    def boom(run, args): raise RuntimeError('paint failed')
    monkeypatch.setattr(entry, 'run_headless', boom)
    with pytest.raises(RuntimeError):
        entry.main(['--id', 'probe', '--no-ui', '--goal', '1,1'])
    assert written['partial'] is True
    assert 'paint failed' in written['reason']


# -- the status line ---------------------------------------------------------

def test_the_report_path_is_shortened_to_the_part_that_identifies_it():
    from dnav.dnav import _short_report

    assert _short_report(None) == 'waiting for the session provider'
    assert _short_report(Path('/a/b/reports/area1/20260101-000000-aa/dnav')) == \
        'area1/20260101-000000-aa/dnav'


def test_the_planner_reports_its_health_in_the_shared_vocabulary(run):
    from dcmn.health import DOTS

    assert run.health() in DOTS
    run.policy = load_policy('cautious', ROOT)
    run.replan(force=True)
    assert run.route.status == R.NO_ROUTE
    assert run.health() == 'warn', 'a planner that cannot find a route is not ok'
