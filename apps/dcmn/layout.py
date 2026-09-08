"""Pure layout records: absent devices reserve their cells across revisions."""
from copy import deepcopy
import math


def layout_key(manifest, instance):
    profile = manifest.get('profile') or {}
    return profile.get('name') or instance


def resolve_layout(load, manifest, instance, name=None):
    key = name or layout_key(manifest, instance)
    value = load('device_layouts', key)
    if value is None and key != instance:
        value = load('device_layouts', instance)
    return key, reconcile(value or {}, manifest.get('sensors', {}))


def cells(pane):
    return {(r, c) for r in range(pane['row'], pane['row']+pane['rowspan'])
            for c in range(pane['col'], pane['col']+pane['colspan'])}


def auto_layout(ids):
    """Seat ids near-square on a grid that starts at three by three.

    One open device is a cell in a bench with empty cells to drop into, not
    a single pane that owns the whole window.
    """
    ids = list(dict.fromkeys(ids))
    columns = max(3, math.ceil(math.sqrt(len(ids))))
    rows = max(3, math.ceil(len(ids)/columns))
    return reconcile(dict(columns=columns, rows=rows,
        panes=[dict(id=sid, row=i//columns, col=i%columns) for i, sid in enumerate(ids)]), ids)


def reconcile(record, devices=()):
    """Repair collisions and invalid spans, retaining absent and unknown options."""
    record = deepcopy(record) if isinstance(record, dict) else {}
    def positive(value, default):
        return value if isinstance(value, int) and not isinstance(value, bool) and 0 < value <= 128 else default
    cols, rows = positive(record.get('columns'), 1), positive(record.get('rows'), 1)
    occupied, panes, ids = set(), [], set()
    raw_panes = record.get('panes', [])
    for raw in raw_panes if isinstance(raw_panes, list) else []:
        if not isinstance(raw, dict) or not isinstance(raw.get('id'), str) or raw['id'] in ids: continue
        pane = dict(raw); ids.add(pane['id'])
        pane['rowspan'] = positive(pane.get('rowspan'), 1)
        pane['colspan'] = positive(pane.get('colspan'), 1)
        pane['options'] = pane.get('options', {}) if isinstance(pane.get('options', {}), dict) else {}
        row, col = pane.get('row'), pane.get('col')
        # A negative position is the "never placed" marker, not damage: grow
        # the arrangement toward the near-square shape auto_layout would give
        # this many panes, so checking devices one at a time builds a grid
        # rather than a single ever-growing column. A pane whose saved cell
        # merely broke keeps the shape it had.
        unplaced = (isinstance(row, int) and isinstance(col, int)
                    and row < 0 and col < 0)
        if unplaced:
            cols = max(cols, math.ceil(math.sqrt(len(ids))))
            rows = max(rows, math.ceil(len(ids)/cols))
        valid = (isinstance(row, int) and isinstance(col, int) and row >= 0 and col >= 0
                 and row+pane['rowspan'] <= rows and col+pane['colspan'] <= cols)
        if not valid or cells(pane) & occupied:
            pane.update(rowspan=1, colspan=1)
            index = 0
            while (index//cols, index%cols) in occupied: index += 1
            pane.update(row=index//cols, col=index%cols)
            rows = max(rows, pane['row']+1)
        occupied |= cells(pane); panes.append(pane)
    record.update(columns=cols, rows=rows, panes=panes)
    for key, count in (('column_weights', cols), ('row_weights', rows)):
        weights = record.get(key, [])
        if not isinstance(weights, list): weights = []
        record[key] = [float(weights[i]) if i < len(weights) and isinstance(weights[i], (int, float))
                       and math.isfinite(weights[i]) and weights[i] > 0 else 1. for i in range(count)]
    popped = record.get('popped', {})
    record['popped'] = popped if isinstance(popped, dict) else {}
    return record


def swap(record, first, second):
    result = deepcopy(record)
    a = next(p for p in result['panes'] if p['id'] == first)
    b = next(p for p in result['panes'] if p['id'] == second)
    # Swap the whole occupied rectangles so spans cannot overlap neighbours.
    for key in ('row', 'col', 'rowspan', 'colspan'): a[key], b[key] = b[key], a[key]
    return reconcile(result)


def occupied(record):
    """Every cell covered by any pane, popped-out panes included."""
    return set().union(*(cells(pane) for pane in record['panes'])) if record['panes'] else set()


def place(record, sid, row, col):
    """Seat a pane at an explicit cell, growing the grid to reach it.

    The placed pane keeps priority: a neighbour whose cells it takes is
    displaced to the first free cell, never the other way round. A pane
    that did not exist is created there as a single cell.
    """
    if not isinstance(row, int) or isinstance(row, bool) or not isinstance(col, int) or isinstance(col, bool):
        raise ValueError('cell must be a (row, column) pair of integers')
    result = deepcopy(record)
    if row < 0 or col < 0: raise ValueError('cell must be non-negative')
    result['rows'] = max(result['rows'], row+1)
    result['columns'] = max(result['columns'], col+1)
    pane = next((p for p in result['panes'] if p['id'] == sid), None)
    if pane is None:
        result['panes'].insert(0, dict(id=sid, row=row, col=col, rowspan=1, colspan=1, options={}))
    else:
        # Clamp before reconcile: an overhanging span would read as damage
        # and drag the pane off its target cell.
        pane.update(row=row, col=col,
                    rowspan=min(pane.get('rowspan', 1), result['rows']-row),
                    colspan=min(pane.get('colspan', 1), result['columns']-col))
        result['panes'].remove(pane); result['panes'].insert(0, pane)
    return reconcile(result)


def _insert_axis(record, axis, index):
    """Grow rows or columns at ``index``, shifting panes past it down."""
    if not isinstance(index, int) or isinstance(index, bool):
        raise ValueError('index must be an integer')
    result = deepcopy(record)
    count = result[axis]
    index = max(0, min(index, count))
    result[axis] = count+1
    cell = 'row' if axis == 'rows' else 'col'
    for pane in result['panes']:
        if pane[cell] >= index: pane[cell] += 1
    weights = result['column_weights' if axis == 'columns' else 'row_weights']
    weights.insert(min(index, len(weights)), 1.)
    return reconcile(result)


def _remove_axis(record, axis, index):
    """Delete rows or columns number ``index``; refuse one that holds a pane.

    A structural edit never moves a pane: the operator drags it out first.
    The last row or column stays, so a grid is always at least one by one.
    """
    if not isinstance(index, int) or isinstance(index, bool):
        raise ValueError('index must be an integer')
    result = deepcopy(record)
    if not 0 <= index < result[axis]: raise ValueError(f"{axis[:-1]} {index} is outside the grid")
    cell, length = (('row', 'rowspan') if axis == 'rows' else ('col', 'colspan'))
    if result[axis] == 1 or any(pane[cell] <= index < pane[cell]+pane[length] for pane in result['panes']):
        raise ValueError(f"{axis[:-1]} {index} still holds a pane")
    for pane in result['panes']:
        if pane[cell] > index: pane[cell] -= 1
    result[axis] -= 1
    weights = result['column_weights' if axis == 'columns' else 'row_weights']
    if index < len(weights): weights.pop(index)
    return reconcile(result)


def insert_row(record, index): return _insert_axis(record, 'rows', index)
def remove_row(record, index): return _remove_axis(record, 'rows', index)
def insert_column(record, index): return _insert_axis(record, 'columns', index)
def remove_column(record, index): return _remove_axis(record, 'columns', index)


def span(record, sid, row, col):
    result = deepcopy(record)
    pane = next(p for p in result['panes'] if p['id'] == sid)
    pane['rowspan'] = max(1, min(result['rows']-pane['row'], row-pane['row']+1))
    pane['colspan'] = max(1, min(result['columns']-pane['col'], col-pane['col']+1))
    # The resized pane gets priority; collided neighbours move to free cells.
    result['panes'].remove(pane); result['panes'].insert(0, pane)
    return reconcile(result)


def unspan(record, sid):
    """Shrink a pane back to a single cell, anchored where it stands."""
    result = deepcopy(record)
    pane = next(p for p in result['panes'] if p['id'] == sid)
    pane.update(rowspan=1, colspan=1)
    return reconcile(result)
