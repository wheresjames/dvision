"""Persisted arrangement repair and concurrent state writers, without Tk."""
import multiprocessing

import pytest

from dcmn import layout, window


def test_absent_added_restored_and_digest_change():
    saved = layout.auto_layout(['left', 'scan'])
    saved['profile_digest'] = 'old'
    saved['panes'][0]['options'] = {'fit': 'native'}
    changed = layout.reconcile(saved, ['scan', 'new'])
    assert changed == saved
    restored = layout.reconcile(changed, ['left', 'scan', 'new'])
    assert restored == saved
    assert 'new' not in [p['id'] for p in restored['panes']]
    key, resolved = layout.resolve_layout(lambda store, key: saved if key == 'profile' else None,
        dict(profile={'name': 'profile'}, profile_digest='new', sensors={'scan': {}}), 'vehicle')
    assert key == 'profile' and resolved == saved


def test_instance_fallback_and_named_override():
    saved = layout.auto_layout(['scan'])
    loader = lambda store, key: saved if key == 'vehicle' else None
    key, record = layout.resolve_layout(loader, {'profile': {'name': 'profile'}}, 'vehicle')
    assert key == 'profile' and record['panes'][0]['id'] == 'scan'
    key, record = layout.resolve_layout(loader, {}, 'vehicle', 'inspection')
    assert key == 'inspection' and record['panes'][0]['id'] == 'scan'


def test_collision_and_out_of_range_keep_every_pane():
    record = layout.reconcile(dict(columns=2, rows=2, panes=[
        dict(id='a', row=0, col=0, rowspan=2), dict(id='b', row=0, col=0),
        dict(id='c', row=1, col=1, colspan=3), dict(id='d', row=-1, col=8)]))
    occupied = set()
    for pane in record['panes']:
        assert not occupied & layout.cells(pane)
        occupied |= layout.cells(pane)
    assert len(record['panes']) == 4
    assert record['panes'][1]['col'] == 1
    assert layout.reconcile(record) == record


def test_devices_checked_one_at_a_time_build_columns_not_a_stack():
    """The never-placed marker grows the grid toward auto layout's shape.

    Checking devices with no saved arrangement used to place every pane in
    the first free cell of a one-column grid, so the bench stacked vertically
    no matter how many devices were opened.
    """
    record = layout.auto_layout([])
    placed = []
    for sid in ('a', 'b', 'c', 'd'):
        record['panes'].append(dict(id=sid, row=-1, col=-1))
        record = layout.reconcile(record)
        placed.append((record['panes'][-1]['row'], record['panes'][-1]['col']))
    # The bench starts three by three, so early devices fill its top rows.
    assert placed == [(0, 0), (0, 1), (0, 2), (1, 0)]
    assert record['columns'] == record['rows'] == 3
    record['panes'].append(dict(id='e', row=-1, col=-1))
    record = layout.reconcile(record)
    assert record['panes'][-1]['row'] == 1 and record['panes'][-1]['col'] == 1
    # Past nine panes the grid grows toward the near-square shape.
    for sid in 'fghij':
        record['panes'].append(dict(id=sid, row=-1, col=-1))
        record = layout.reconcile(record)
    assert record['columns'] == 4
    # A broken saved cell is damage, not a new pane: it repairs into the
    # operator's existing shape rather than growing columns under it.
    stacked = layout.reconcile(dict(columns=1, rows=3, panes=[
        dict(id='a', row=0, col=0), dict(id='b', row=1, col=0),
        dict(id='c', row=2, col=0)]))
    stacked['panes'][1].update(row=9, col=9)
    repaired = layout.reconcile(stacked)
    assert repaired['columns'] == 1
    assert [(p['row'], p['col']) for p in repaired['panes']] == [(0, 0), (1, 0), (2, 0)]


def test_swap_rectangles_resize_and_weight_roundtrip():
    record = layout.auto_layout(['a', 'b', 'c', 'd'])
    record['column_weights'] = [2, 1]
    grown = layout.span(record, 'a', 1, 1)
    assert grown['panes'][0]['rowspan'] == grown['panes'][0]['colspan'] == 2
    swapped = layout.swap(grown, 'a', 'b')
    assert next(p for p in swapped['panes'] if p['id'] == 'b')['rowspan'] == 2
    assert len(set.union(*(layout.cells(p) for p in swapped['panes']))) == 7
    assert swapped['column_weights'] == [2., 1., 1.]
    shrunk = layout.unspan(swapped, 'b')
    assert next(p for p in shrunk['panes'] if p['id'] == 'b')['rowspan'] == 1
    assert layout.reconcile(shrunk) == shrunk


def test_place_grows_the_grid_and_displaces_the_occupant():
    """The seating primitive a drop uses once the grid has evicted the occupant."""
    record = layout.auto_layout(['a', 'b'])
    # A seated pane takes the exact cell it is given; any pane still standing
    # there is the one that moves, never the seated one.
    dropped = layout.place(record, 'c', 0, 1)
    assert [(p['id'], p['row'], p['col']) for p in dropped['panes']] == [('c', 0, 1), ('a', 0, 0), ('b', 0, 2)]
    assert dropped['rows'] == 3 and layout.reconcile(dropped) == dropped
    # Dropping beyond the grid grows it deterministically to that cell.
    grown = layout.place(record, 'x', 2, 3)
    assert (grown['rows'], grown['columns']) == (3, 4)
    assert layout.occupied(grown) == {(0, 0), (0, 1), (2, 3)}
    # Moving an open pane keeps its span where it fits and clamps it where not.
    spanned = layout.reconcile(dict(columns=3, rows=2, panes=[
        dict(id='a', row=0, col=0, rowspan=2, colspan=2), dict(id='b', row=0, col=2)]))
    moved = layout.place(spanned, 'a', 0, 2)
    assert next(p for p in moved['panes'] if p['id'] == 'a')['colspan'] == 1
    assert next(p for p in moved['panes'] if p['id'] == 'a')['col'] == 2
    assert next(p for p in moved['panes'] if p['id'] == 'b')['col'] != 2
    for bad in ((0, -1), (-1, 0), (None, 0), (0.5, 0)):
        with pytest.raises(ValueError): layout.place(record, 'z', *bad)


def test_insert_and_remove_rows_and_columns_move_no_pane():
    record = layout.auto_layout(['a', 'b', 'c', 'd'])
    record['column_weights'] = [2., 1.]
    # Appending grows the edge without touching any pane.
    grown = layout.insert_row(record, record['rows'])
    assert grown['rows'] == 4 and [(p['row'], p['col']) for p in grown['panes']] == [(0, 0), (0, 1), (0, 2), (1, 0)]
    assert grown['row_weights'] == [1., 1., 1., 1.]
    # Inserting between rows shifts the panes and weights below it.
    middle = layout.insert_column(grown, 1)
    assert [(p['row'], p['col']) for p in middle['panes']] == [(0, 0), (0, 2), (0, 3), (1, 0)]
    assert middle['column_weights'] == [2., 1., 1., 1.]
    assert layout.reconcile(middle) == middle
    # Removing restores the same arrangement: an exact round trip.
    assert layout.remove_column(middle, 1)['panes'] == grown['panes']
    # Deleting a row or column that still holds a pane is refused, not repaired.
    for operation, index in ((layout.remove_row, 0), (layout.remove_column, 0)):
        with pytest.raises(ValueError): operation(middle, index)
    empty = layout.insert_row(layout.auto_layout(['a']), 0)
    single = layout.remove_row(empty, 0)
    assert single == layout.auto_layout(['a'])
    with pytest.raises(ValueError): layout.remove_row(single, 0)  # the last one stays
    with pytest.raises(ValueError): layout.remove_row(empty, 9)


def write_state(directory, prefix):
    from pathlib import Path
    window._STORE = Path(directory)/'window_pos.json'
    for i in range(20): window.save_state('device_layouts', f'{prefix}{i}', {'value': i})


def test_two_writers_preserve_other_keys(tmp_path, monkeypatch):
    monkeypatch.setattr(window, '_STORE', tmp_path/'window_pos.json')
    processes = [multiprocessing.get_context('spawn').Process(target=write_state, args=(str(tmp_path), prefix)) for prefix in ('a', 'b')]
    for process in processes: process.start()
    for process in processes:
        process.join(10); assert process.exitcode == 0
    for prefix in ('a', 'b'):
        for i in range(20): assert window.load_state('device_layouts', f'{prefix}{i}') == {'value': i}
