"""The live dynamic-flight window: it explains the flight and never flies it.

The window is built on a hidden Tk root over a deterministic rig. The rig
drives the executor; the window only paints the immutable view and queues
operator requests, so these tests check that painting cannot step control,
that the four controls reach the executor, and that the readouts separate
proposal, permission, command and motion.
"""

import pytest

from dtest.dynamic_rig import DT_S, DynamicRig
from dtest.tkfixture import hidden_tk


@pytest.fixture
def window(tmp_path):
    rig = DynamicRig(tmp_path)
    try:
        with hidden_tk() as root:
            from dway.flightui import FlightWindow
            win = FlightWindow(rig.executor, root, control_thread=False)
            win.closed = True  # the fixture ticks by hand; no after() loop runs
            yield rig, win
    finally:
        rig.close()


def paint(win):
    win.closed = False
    try:
        win._text.reset()
        win.pane._paint.reset()
        win.tick()
    finally:
        win.closed = True


def test_painting_never_steps_the_executor(window, monkeypatch) -> None:
    rig, win = window
    for _ in range(int(2.0 / DT_S)):
        rig.step()
    steps = []
    monkeypatch.setattr(rig.executor, 'step', lambda: steps.append(1))
    for _ in range(5):
        paint(win)
    assert steps == [], 'the window stepped control from the Tk thread'


def test_buttons_queue_the_same_requests_the_headless_runner_uses(window) -> None:
    rig, win = window
    for _ in range(int(2.0 / DT_S)):
        rig.step()
    win.buttons['start'].invoke()
    assert ('start', 'window') in rig.executor.pending
    assert rig.executor.control_results['start']['state'] == 'pending'
    rig.step()
    assert rig.executor.control_results['start']['state'] == 'accepted', rig.executor.control_results
    win.buttons['pause'].invoke()
    rig.step()
    assert rig.executor.state in ('BRAKING', 'HOLDING')


def test_readouts_and_overlays_explain_a_flight(window) -> None:
    rig, win = window
    rig.start()
    assert rig.fly(limit_s=30.0, until=lambda r: r.executor.state == 'EXECUTING')
    for _ in range(10):
        rig.step()
    paint(win)
    text = win.readout_text()
    for words in ('EXECUTING', 'Remaining', 'Stop margin', 'Cross-track', 'Control: held',
                  'calibrated', 'Recording: healthy', 'slab', 'Heading'):
        assert words in text, f'{words!r} missing from the readouts:\n{text}'
    overlay = win.pane.state.overlay
    # Proposal, permission, command and motion are separate symbols.
    assert overlay.proposal and overlay.route and overlay.permitted
    assert overlay.target is not None and overlay.vehicle is not None and overlay.track
    assert overlay.target != overlay.vehicle[:2]
    assert overlay.highlight, 'the segment being flown is not highlighted'
    assert win.timeline.find_all(), 'the timeline drew nothing'
    assert win.events.get_children(), 'the event table is empty'
    assert 'painted' in win.drawing_var.get()
    # Selecting an event shows its route and explanation.
    win.events.selection_set(win.events.get_children()[-1])
    win._select_event(None)
    assert win.selected_event is not None and ' at ' in win.event_var.get()
    assert win.state_badge.cget('text') == 'EXECUTING'
    paint(win)
    if win.selected_event.get('points'):
        assert win.pane.state.overlay.candidate


def test_closing_the_window_holds_and_releases_without_landing(window) -> None:
    rig, win = window
    rig.start()
    assert rig.fly(limit_s=30.0, until=lambda r: r.executor.state == 'EXECUTING')
    win.closed = False
    win.close()
    result = rig.executor.shutdown_result
    assert result['hold']['accepted'] and result['confirmed'] and result['released']
    assert rig.sim.state.armed and rig.sim.state.mode == 'HOLD'


def test_the_right_column_scrolls_instead_of_squashing_the_timeline(window) -> None:
    rig, win = window
    from dway.flightui import TIMELINE_HEIGHT
    win.root.deiconify()
    win.root.geometry('1200x520')
    for _ in range(3):
        paint(win)
        win.root.update()
    assert win.timeline.winfo_height() >= TIMELINE_HEIGHT - 2, 'the timeline was squashed'
    top, bottom = win.side_canvas.yview()
    assert bottom < 1.0, 'the cards fit a 520 px window, so nothing was tested'
    assert win.side_scroll.winfo_ismapped(), 'the scrollbar is hidden although the cards overflow'
    win.side_canvas.yview_moveto(1.0)
    assert win.side_canvas.yview()[1] == 1.0
    win.root.withdraw()
