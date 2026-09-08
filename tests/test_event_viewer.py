"""Passive event inspection must not consume control, publish, or stop draining."""
from types import SimpleNamespace
import uuid

import pytest

from dcmn.event_viewer import EventHistory, EventViewer
from dcmn.module_bus import ModuleEvent, PymembusModuleBus
from dtest.tkfixture import hidden_root


def event(sequence, kind='run.state', source='dway', run='tour'):
    return ModuleEvent(str(sequence), 'test', 'navigator', source, 'process',
                       sequence, sequence / 10, kind, run, {'state': 'FLYING', 'n': sequence})


def test_history_bounds_and_filters_do_not_erase_received_events():
    history = EventHistory(max_rows=3)
    for n in range(5):
        history.append(event(n, 'module.heartbeat' if n == 4 else 'run.state'))
    assert history.received == 5 and history.discarded == 2
    assert [r.event.sequence for r in history.matching()] == [2, 3]
    assert len(list(history.matching(source='DWAY', kind='state', run='tour'))) == 2
    assert not list(history.matching(run='other'))
    assert len(list(history.matching(hide_heartbeats=False))) == 3


def test_byte_limit_also_bounds_large_events():
    history = EventHistory(max_bytes=1)
    history.append(event(1))
    assert history.discarded == 1 and not history.rows


class Reader:
    session_id = 1
    overruns = 2
    def __init__(self): self.events = []; self.closed = False
    def receive(self, *, limit):
        assert limit == 128
        result, self.events = self.events[:limit], self.events[limit:]
        return result
    def close(self): self.closed = True


@pytest.fixture
def viewer():
    root = hidden_root()
    reader = Reader()
    panel = EventViewer(root, 'test', reader=reader, history=EventHistory(max_rows=3))
    panel.page.pack(fill='both', expand=True)
    try:
        yield panel, reader
    finally:
        panel.close()
        root.destroy()


def test_pause_keeps_draining_but_freezes_rows_and_details(viewer):
    panel, reader = viewer
    reader.events = [event(1)]
    panel.poll(); panel.render(force=True)
    first = panel.table.get_children()[0]
    panel.table.selection_set(first); panel._select()
    detail = panel.detail.get('1.0', 'end')
    panel.paused.set(True)
    reader.events = [event(n) for n in range(2, 8)]
    panel.poll(); panel.render(force=True)
    assert not reader.events and panel.history.received == 7
    assert panel.history.discarded == 4
    assert panel.table.get_children() == (first,)
    assert panel.detail.get('1.0', 'end') == detail
    assert '2 reader overruns' in panel.status.get()
    panel.paused.set(False); panel.render(force=True)
    assert len(panel.table.get_children()) == 3
    assert first not in panel.visible


def test_filtering_and_clear_are_local_and_follow_can_be_stopped(viewer):
    panel, reader = viewer
    reader.events = [event(1), event(2, 'module.heartbeat'), event(3, source='dalg')]
    panel.poll(); panel.render(force=True)
    assert len(panel.visible) == 2
    panel.source.set('dalg')
    assert len(panel.visible) == 1
    panel.table.selection_set(next(iter(panel.visible))); panel._select()
    assert '"implementation": "dalg"' in panel.detail.get('1.0', 'end')
    panel._stop_follow(SimpleNamespace(delta=120))
    assert not panel.follow.get()
    # Clicking a row or arrowing down must cancel follow through the real
    # bindings, or see() keeps snapping the view back to the newest row.
    panel.follow.set(True)
    panel.table.event_generate('<Button-1>', when='now')
    assert not panel.follow.get()
    # Key events need a focused, viewable window, which the hidden test root
    # cannot offer; assert the wiring instead of the dispatch.
    assert panel.table.bind('<KeyPress-Down>')
    # A filter that hides an earlier row and then lifts must restore arrival
    # order among the rows that come back.
    panel.source.set('')
    panel.clear()
    panel.kind.set('run.state')
    reader.events = [event(1, kind='nav.other'), event(2), event(3)]
    panel.poll(); panel.render(force=True)
    assert tuple(panel.visible) == ('5', '6')
    panel.kind.set('')  # the write trace re-renders with force
    assert panel.table.get_children() == ('4', '5', '6')
    panel.clear()
    assert not panel.visible and not panel.history.rows
    assert panel.history.received == 6


def test_reader_has_its_own_cursor_and_cannot_publish():
    instance = 'events-test-' + uuid.uuid4().hex[:10]
    owner = PymembusModuleBus(instance, 'simulator', 'test', create=True)
    observer = PymembusModuleBus(instance, 'observer', 'viewer', read_only=True)
    client = PymembusModuleBus(instance, 'controller', 'test')
    try:
        assert owner.connect() and observer.connect() and client.connect()
        with pytest.raises(RuntimeError, match='read-only'):
            observer.publish('system.shutdown')
        for n in range(3):
            owner.publish('run.state', payload={'n': n})
        assert len(observer.receive(limit=1)) == 1
        assert len(client.receive()) == 3
        assert len(observer.receive(limit=10)) == 2
        owner.remove()
        observer._last_session_probe = -1e9
        assert observer.receive(limit=10) == []
        assert observer.session_id is None
        assert owner.connect()
        assert observer.connect()
        owner.publish('run.started')
        assert observer.receive(limit=10)[0].type == 'run.started'
    finally:
        observer.close(); client.close(); owner.remove()


def test_sensor_health_updates_hidden_behind_their_own_checkbox(viewer):
    panel, reader = viewer
    reader.events = [event(1, 'module.sensor_health'), event(2)]
    panel.poll(); panel.render(force=True)
    assert tuple(panel.visible) == ('2',)
    panel.hide_health.set(False); panel.render(force=True)
    assert tuple(panel.visible) == ('1', '2')
    assert panel.hide_heartbeats.get()  # the boxes stay independent
    panel.hide_health.set(True); panel.render(force=True)
    assert tuple(panel.visible) == ('2',)
    # No two toolbar children may claim the same grid cell: a checkbox added
    # without moving Clear used to render underneath the button.
    bar = panel.page.winfo_children()[0]
    children = bar.winfo_children()
    cells = [(c.grid_info()['row'], c.grid_info()['column']) for c in children]
    assert len(cells) == len(set(cells))


def test_event_tab_blocks_flight_shortcuts_and_held_keys():
    from dctl.dctl import DroneController
    controller = DroneController.__new__(DroneController)
    controller.held = set()
    controller.flight_page = 'flight'
    controller.notebook = SimpleNamespace(select=lambda: 'events')
    key = SimpleNamespace(keysym='w', widget=None)
    controller.key_down(key)
    sent = []
    controller._flight_shortcut(key, lambda: sent.append('takeoff'))
    assert not sent and not controller.held
    controller.notebook = SimpleNamespace(select=lambda: 'flight')
    key.widget = SimpleNamespace(winfo_class=lambda: 'TEntry')
    controller.key_down(key)
    assert not controller.held
