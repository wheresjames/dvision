"""dmsn's tabs on a hidden root: the Setup form, the Mission tab's map and controls, the Status strip."""
from __future__ import annotations

import json
import uuid

import pytest

from dmsn import profiles as P
from dmsn.runner import MissionRun
from dtest.mission_fixture import FakeExecutor, FakePorts, drive
from dtest.tkfixture import hidden_tk


@pytest.fixture(autouse=True)
def user_dir(tmp_path, monkeypatch):
    monkeypatch.setenv('DVISION2_MISSION_DIR', str(tmp_path / 'user'))
    return tmp_path / 'user'


@pytest.fixture
def chain(tmp_path):
    ports = FakePorts(tmp_path / 'run', instance='win-' + uuid.uuid4().hex[:8], pose=(12., 8., 1.5, 90.),
                      coverage=None)
    run = MissionRun(ports.instance, P.load('maze_020-goto-click')[0], profile_path=P.resolve('maze_020-goto-click'),
                     ports=ports)
    run.step()
    yield run, ports
    run.close()


@pytest.fixture
def geometry_tabs(tmp_path, request):
    """A Mission tab over one of the geometry missions (area or waypoints)."""
    import tkinter as tk
    from tkinter import ttk
    from dmsn.ui.mission_tab import MissionTab
    ports = FakePorts(tmp_path / 'run', instance='win-' + uuid.uuid4().hex[:8], pose=(12., 8., 1.5, 90.),
                      coverage=None)
    name = request.param
    run = MissionRun(ports.instance, P.load(name)[0], profile_path=P.resolve(name), ports=ports)
    run.step()
    with hidden_tk() as root:
        tab = MissionTab(root, run, tk, ttk)
        tab.pane.canvas.configure(width=300, height=300)
        root.update_idletasks()
        tab.paint_map()
        yield root, tab, run, ports
        tab.reference.close()
    run.close()


@pytest.fixture
def tabs(chain):
    import tkinter as tk
    from tkinter import ttk
    from dmsn.ui.mission_tab import MissionTab
    from dmsn.ui.setup_tab import SetupTab
    from dmsn.ui.status_tab import StatusTab
    run, _ = chain
    with hidden_tk() as root:
        frames = [ttk.Frame(root) for _ in range(3)]
        mission = MissionTab(frames[0], run, tk, ttk)
        setup = SetupTab(frames[1], run, tk, ttk)
        status = StatusTab(frames[2], run, tk, ttk)
        root.update_idletasks()
        yield root, mission, setup, status
        mission.reference.close()


def test_the_form_is_generated_from_the_schemas(tabs):
    _, _, setup, _ = tabs
    for key in ('params.goal_source', 'params.goal', 'finish_action', 'workspace', 'budgets.time_s',
                'debounce.min_commit_s', 'assisted.reference_background'):
        assert key in setup.rows, key
    assert setup.rows['params.goal_source'][1].get() == 'click'
    assert 'saved' in setup.identity.get() and 'modified' not in setup.identity.get()


def test_setup_edits_mark_the_profile_dirty_refuse_bad_values_and_save(tabs, chain, user_dir):
    _, _, setup, _ = tabs
    run, _ = chain
    assert setup.set_field('budgets.time_s', '300')
    assert run.document.value['budgets']['time_s'] == 300. and run.document.dirty
    assert 'modified' in setup.identity.get()
    assert not setup.set_field('budgets.time_s', 'later')
    assert 'not changed' in setup.notice.get()
    assert setup.rows['budgets.time_s'][1].get() == '300', 'a refused value is put back'
    saved = setup.save(user_dir / 'edited.json')
    assert saved is not None and json.loads(saved.read_text())['budgets']['time_s'] == 300.
    assert not run.document.dirty and 'saved' in setup.identity.get()
    assert any(path == saved for _, path in P.available())
    assert run.document.value['finish_action'] == 'home'
    setup.set_field('finish_action', 'hold')
    assert run.document.dirty
    setup.revert()
    assert run.document.value['finish_action'] == 'home' and not run.document.dirty
    setup.new()
    assert run.document.path is None and run.document.dirty


def test_the_strategy_is_picked_from_the_registry_and_its_parameters_get_a_form(tabs, chain):
    _, _, setup, _ = tabs
    run, _ = chain
    setup.mission_var.set('explore')
    setup._mission_changed()
    assert run.document.value['strategy'] == 'frontier'
    widget = setup.rows['strategy'][2]
    assert set(widget.cget('values')) == {'', 'frontier', 'wall_follow'}, 'a choice, not free text'
    assert not any(key.startswith('strategy_params.') for key in setup.rows), 'frontier has no parameters'
    assert setup.set_field('strategy', 'wall_follow')
    assert setup.rows['strategy_params.hand'][1].get() == 'right', 'the form grows the strategy section'
    assert setup.set_field('strategy_params.step_m', '6')
    assert run.document.value['strategy_params']['step_m'] == 6. and run.document.dirty


def test_a_mission_tab_click_before_any_evidence_sets_the_goal_through_the_fallback_extent(tabs, chain):
    root, mission, setup, _ = tabs
    run, _ = chain
    mission.pane.canvas.configure(width=300, height=300)
    root.update_idletasks()
    mission.paint_map()
    assert mission.pane.state.grid is None and mission.pane.state.fallback is not None
    assert 'around the vehicle' in mission.pane.state.fallback_label
    event = type('E', (), {'x': 150, 'y': 150, 'state': 0})()
    mission.pane._click(event)
    goal = run.document.value['params']['goal']
    assert goal is not None and abs(goal[0] - 12.) < 5. and abs(goal[1] - 8.) < 5.
    assert 'goal set' in mission.notice.get()
    setup.refresh()
    assert setup.rows['params.goal'][1].get() == f'{goal[0]:g},{goal[1]:g}'
    mission.paint_text()
    assert '(profile)' in mission.rows['goal'].get()
    assert mission.snapshot() is not None


def test_controls_follow_the_phase_and_the_status_strip_draws_it(tabs, chain):
    root, mission, _, status = tabs
    run, ports = chain
    run.click(14., 8.)
    ex = FakeExecutor(ports)
    mission.paint_text()
    assert mission.buttons['Pause'].instate(['disabled']), 'nothing to pause while idle'
    assert mission.buttons['Start'].instate(['!disabled'])
    assert mission.buttons['Land'].instate(['disabled']), 'nothing to land while the vehicle is on the ground'
    mission._command(mission.start)
    assert run.phase == 'starting'
    drive(run, ex, ports=ports, until=lambda r: r.goal_sent is not None)
    mission.paint_text()
    assert mission.buttons['Pause'].instate(['!disabled']) and mission.buttons['Start'].instate(['disabled'])
    assert mission.buttons['Land'].instate(['!disabled']), 'Land is armed once the vehicle is airborne'
    assert 'launch' in mission.rows['intent'].get()
    drive(run, ex, ports=ports, until=lambda r: r.phase == 'done')
    mission.paint_text()
    status.paint()
    root.update_idletasks()
    assert status.canvas.find_all(), 'the phase strip is drawn'
    assert status.table.get_children(), 'the timeline table has rows'
    assert 'phase done' in status.readout.get()


# -- the geometry editors (the area polygon and the waypoint list) --------------------------

@pytest.mark.parametrize('geometry_tabs', ['maze_020-explore'], indirect=True)
def test_the_area_editor_appends_moves_refuses_a_crossing_and_undoes(geometry_tabs):
    from dataclasses import replace
    root, mission, run, _ = geometry_tabs
    editors = mission.editors
    assert set(editors.editors) == {'area'}, 'the explore mission edits an area'
    assert not run.document.dirty
    # a click inside the area notches the edge it grows least, and marks the profile dirty
    mission.pane.on_press(30., 20., frozenset())
    mission.pane.on_click(30., 20.)
    area = run.document.value['area']
    assert len(area) == 5 and area[3] == [30., 20.], area
    assert run.document.dirty and 'area: 5 vertices' in mission.notice.get()
    # dragging the new vertex across the ring is refused, with the reason shown
    mission.pane.on_press(30., 20., frozenset())  # the press lands on its handle
    mission.pane.on_drag(66., 30., frozenset())
    assert 'cross' in mission.notice.get()
    assert run.document.value['area'][3] == [30., 20.], 'a refused edit changes nothing'
    # a legal drag moves it; one gesture is one undo step
    mission.pane.on_drag(40., 25., frozenset())
    assert run.document.value['area'][3] == [40., 25.]
    mission.editors.undo()
    assert run.document.value['area'][3] == [30., 20.]
    # right-click removes the vertex under the pointer
    mission.pane.on_context(2., 2., frozenset())
    assert run.document.value['area'] == [[65., 2.], [65., 33.], [30., 20.], [2., 33.]]
    # a shift-drag draws a rectangle area in one gesture, undoable like any edit
    mission.pane.on_press(5., 5., frozenset({'shift'}))
    mission.pane.on_drag(45., 32., frozenset({'shift'}))
    mission.pane.on_release(45., 32., frozenset({'shift'}))
    assert run.document.value['area'] == [[5., 5.], [45., 5.], [45., 32.], [5., 32.]]
    mission.editors.undo()
    assert run.document.value['area'] == [[65., 2.], [65., 33.], [30., 20.], [2., 33.]]
    # a pressed handle is selected, and Delete removes the selection
    mission.pane.on_press(30., 20., frozenset())
    mission.pane.on_release(30., 20., frozenset())
    mission.editors.key('Delete', None)
    assert len(run.document.value['area']) == 3
    # a point beyond the evidence coverage is kept, and flagged
    run.observation = replace(run.observation, coverage=(0., 0., 70., 40.))
    mission.pane.on_press(10., -5., frozenset())
    mission.pane.on_click(10., -5.)
    assert len(run.document.value['area']) == 4 and '1 outside evidence coverage' in mission.notice.get()
    # a change made outside the editor (a load, a tour import) rebuilds it
    run.document.set('area', [[2., 2.], [40., 2.], [40., 30.], [2., 30.]])
    mission.paint_map()
    assert editors.editors['area'].as_list() == [[2., 2.], [40., 2.], [40., 30.], [2., 30.]]


@pytest.mark.parametrize('geometry_tabs', ['maze_020-explore'], indirect=True)
def test_an_area_mission_owns_every_map_click_and_never_answers_with_the_goal(geometry_tabs):
    root, mission, run, _ = geometry_tabs
    run.document.set('area', [])
    mission.paint_map()
    for x, y in ((10., 10.), (30., 10.), (30., 25.), (10., 25.)):  # clicks on a cleared area draw it
        mission.pane.on_press(x, y, frozenset())
        mission._clicked(x, y)
        mission.pane.on_release(x, y, frozenset())
    assert run.document.value['area'] == [[10., 10.], [30., 10.], [30., 25.], [10., 25.]]
    mission.pane.on_press(20., 5., frozenset({'shift'}))  # a shift-click without a drag
    mission._clicked(20., 5.)
    mission.pane.on_release(20., 5., frozenset({'shift'}))
    assert 'goal' not in mission.notice.get() and 'shift-drag' in mission.notice.get()
    assert len(run.document.value['area']) == 4 and mission.editors._rectangle is None, 'nothing left drawn'
    run.phase = 'running'
    mission._clicked(20., 5.)
    assert 'when no mission is flying' in mission.notice.get()
    run.phase = 'idle'


@pytest.mark.parametrize('geometry_tabs', ['maze_020-waypoints'], indirect=True)
def test_the_waypoint_editor_grows_reorders_and_deletes_from_the_list(geometry_tabs):
    root, mission, run, _ = geometry_tabs
    editors = mission.editors
    assert set(editors.editors) == {'waypoints'} and editors.listbox is not None
    before = [list(w) for w in run.document.value['waypoints']]
    # a click clear of every leg and waypoint appends one
    mission.pane.on_press(5., 30., frozenset())
    mission.pane.on_click(5., 30.)
    grown = run.document.value['waypoints']
    assert len(grown) == len(before) + 1 and run.document.dirty
    assert editors.listbox.size() == len(grown), 'the list follows the profile'
    # the list reorders and deletes the selected waypoint
    editors.listbox.selection_set(0)
    editors._move_listed(1)
    moved = run.document.value['waypoints']
    assert moved[0] == grown[1] and moved[1] == grown[0]
    editors.listbox.selection_set(0)
    editors._remove_listed()
    assert run.document.value['waypoints'] == moved[1:]
    # and the editor follows an outside change, not just its own gestures
    run.document.set('waypoints', before)
    mission.paint_map()
    assert editors.editors['waypoints'].as_list() == before


def _canvas_event(x, y, *, num=0, delta=0):
    return type('E', (), {'x': x, 'y': y, 'state': 0, 'num': num, 'delta': delta,
                          'keysym': 'Escape'})()


@pytest.mark.parametrize('geometry_tabs', ['maze_020-explore'], indirect=True)
def test_the_mission_map_zooms_and_its_reset_badge_is_never_a_map_click(geometry_tabs):
    """The gesture that would otherwise cost an operator a stray polygon vertex."""
    _root, mission, run, _ports = geometry_tabs
    pane = mission.pane
    assert pane.zoomable, 'the Mission tab asked for a zoomable pane'
    area = list(run.document.value['area'])
    pane._wheel(_canvas_event(150, 150, num=4, delta=120))
    assert pane.zoom > 1.0 and not pane.default_view
    assert pane._badge is not None, 'a view to reset puts the badge on the image'

    # Editing still works while zoomed, at the metres the pane reads out.
    where = pane.to_map(150, 150)
    pane._click(_canvas_event(150, 150))
    pane._release(_canvas_event(150, 150))
    grown = run.document.value['area']
    assert len(grown) == len(area) + 1
    # The editor notches a new vertex into the edge it grows least, so what
    # matters is that the point is the one the pane read out under the pointer.
    assert [point for point in grown if point not in area] == [pytest.approx(list(where), abs=1e-6)]
    zoomed = list(run.document.value['area'])

    # And a press on the badge resets the view instead of reaching the editor.
    x0, y0, x1, y1 = pane._badge
    pane._click(_canvas_event((x0 + x1) / 2, (y0 + y1) / 2))
    pane._release(_canvas_event((x0 + x1) / 2, (y0 + y1) / 2))
    assert run.document.value['area'] == zoomed, 'the badge added nothing'
    assert pane.default_view and pane._badge is None
