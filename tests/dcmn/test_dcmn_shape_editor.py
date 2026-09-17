"""The shape editor: every gesture checked, capped, and undoable (B2).

Pure state, no window: the tests drive exactly the gestures the pane delivers.
"""
from __future__ import annotations

from dcmn.mission import MAX_VERTICES, MAX_WAYPOINTS
from dcmn.shape_editor import ShapeEditor, polygon_error


def square():
    return ShapeEditor([(0., 0.), (10., 0.), (10., 10.), (0., 10.)], closed=True)


# -- a polygon refuses to cross ------------------------------------------------------------

def test_a_simple_polygon_has_no_error_and_a_crossing_ring_names_its_edges():
    assert polygon_error([(0., 0.), (10., 0.), (10., 10.), (0., 10.)]) == ''
    bow = [(0., 0.), (10., 0.), (0., 10.), (10., 10.)]
    assert 'cross' in polygon_error(bow)
    assert 'at least 3' in polygon_error([(0., 0.), (1., 1.)])
    assert 'no area' in polygon_error([(0., 0.), (1., 1.), (2., 2.)])
    assert 'coincide' in polygon_error([(0., 0.), (0., 0.), (1., 1.), (2., 2.)])


def test_an_edit_that_would_cross_is_refused_with_the_reason():
    editor = square()
    problem = editor.move(2, -1., 5.)  # drags a vertex across the opposite edge
    assert 'cross' in problem and problem == editor.error
    assert editor.points[2] == (10., 10.), 'a refused edit changes nothing'
    pocket = ShapeEditor([(0., 0.), (10., 0.), (10., 10.), (7., 10.), (7., 4.),
                          (3., 4.), (3., 10.), (0., 10.)], closed=True)
    assert pocket.add(5., 7.) == '', 'a click in the pocket joins the pocket\'s own edge'
    assert (5., 7.) in pocket.points and len(pocket.points) == 9 and polygon_error(pocket.points) == ''


def test_clicking_an_edge_inserts_on_it_and_clicking_far_appends():
    editor = square()
    # the vertex lands on the edge itself, at the point the click projects to
    assert editor.add(5.0, 0.1) == '' and editor.points[1] == (5.0, 0.0)
    assert editor.add(-5., 5.) == '' and editor.points[-1] == (-5., 5.), \
        'a click beyond the closing edge appends outside the ring'


def test_removing_a_vertex_keeps_the_ring_simple_or_refuses():
    editor = square()
    assert editor.remove(0) == '' and len(editor.points) == 3
    editor = ShapeEditor([(0., 0.), (10., 0.), (10., 10.), (5., 5.), (0., 10.)], closed=True)
    assert 'cross' in editor.remove(1), 'the closing edge would cut the ring'
    # fewer than three vertices is a half-drawn ring, not a bad one -- the
    # profile accepts an empty area the same way, and Clear needs to reach it
    spike = ShapeEditor([(0., 0.), (10., 0.), (5., 10.)], closed=True)
    assert spike.remove(0) == '' and spike.remove(0) == '' and spike.remove(0) == ''
    assert 'no vertex' in spike.remove(0), 'an empty ring has nothing left to remove'


# -- the point list ----------------------------------------------------------------------

def test_a_waypoint_list_grows_at_the_end_and_inserts_on_a_leg():
    editor = ShapeEditor([(0., 0.), (10., 0.), (20., 0.)])
    assert editor.add(15., 0.2) == '' and editor.points[2] == (15., 0.0)  # inserted on the leg
    assert editor.add(5., 5.) == '' and editor.points[-1] == (5., 5.)  # far from any leg: appended
    assert editor.reorder(4, 1) == '' and editor.points[1] == (5., 5.)
    assert 'no order' in square().reorder(0, 1), 'a polygon has no order to change'


def test_moving_dragging_and_undoing():
    editor = square()
    assert editor.move(1, 12., 0.) == '' and editor.points[1] == (12., 0.)
    assert editor.move(9, 0., 0.) == 'no vertex 9'
    # a whole drag gesture is one undo step, not one per pixel
    editor.move(1, 13., 0., coalesce=True)
    editor.move(1, 14., 0., coalesce=True)
    editor.move(1, 15., 0., coalesce=True)
    assert editor.can_undo()
    assert editor.undo() == '' and editor.points[1] == (12., 0.)
    assert editor.undo() == '' and editor.points == [(0., 0.), (10., 0.), (10., 10.), (0., 10.)]
    assert editor.undo() == 'nothing to undo'


def test_selection_rectangle_and_removal():
    editor = square()
    editor.move(2, 5., 5.)  # inside the rectangle below
    assert editor.select((1., 1., 6., 6.)) == 1 and editor.selected == {2}
    assert editor.remove_indices(editor.selected) == ''
    assert editor.points == [(0., 0.), (10., 0.), (0., 10.)]
    assert editor.selected == frozenset(), 'a commit clears the selection'
    assert editor.remove_indices(editor.selected) == 'nothing selected'
    empty = ShapeEditor()
    assert empty.remove_indices({}) == 'nothing selected'


def test_hull_clear_and_replace():
    editor = square()
    # what the Hull button writes: a dented ring replaced by its convex outline
    assert editor.replace(((0., 0.), (10., 0.), (10., 10.), (5., 5.), (0., 10.))) == ''
    assert editor.replace(((0., 0.), (10., 0.), (10., 10.), (0., 10.))) == ''
    assert editor.clear() == '' and editor.points == [], 'a polygon editor can be cleared'
    assert ShapeEditor([(0., 0.), (10., 0.), (10., 10.), (0., 10.)]).replace('nonsense') \
        == 'points must be x,y pairs'


def test_the_caps_are_enforced():
    editor = ShapeEditor([(float(i), 0.) for i in range(MAX_WAYPOINTS)])
    assert 'at most' in editor.add(1., 1.), 'the point list stops at its cap'
    ring = ShapeEditor([(float(i), 0.) for i in range(MAX_VERTICES)], closed=True, limit=MAX_VERTICES)
    assert 'at most' in ring.add(1., 1.), 'the polygon stops at its cap'
    assert 'must be finite' in ShapeEditor().insert(0, float('nan'), 0.)


def test_changes_are_marked_until_taken():
    editor = square()
    assert editor.take_changes() is None, 'nothing changed yet'
    editor.add(5., 5.)
    assert editor.take_changes() == editor.as_list() and editor.take_changes() is None


def test_a_click_beside_a_drawn_area_joins_the_nearest_edge_instead_of_crossing():
    editor = ShapeEditor([(0., 0.), (10., 0.), (10., 10.), (0., 10.)], closed=True)
    assert editor.add(5., -3.) == '', 'below the first edge: appending last-to-first would cross'
    assert editor.points == [(0., 0.), (5., -3.), (10., 0.), (10., 10.), (0., 10.)]
    assert editor.add(4., 4.) == '', 'inside the area: a notch in the cheapest edge'
    assert polygon_error(editor.points) == ''


def test_clicks_on_a_cleared_area_draw_a_polygon_point_by_point():
    editor = ShapeEditor(closed=True)
    for x, y in ((0., 0.), (10., 0.), (10., 10.), (0., 10.), (-4., 5.)):
        assert editor.add(x, y) == '', editor.error
    assert len(editor.points) == 5 and polygon_error(editor.points) == ''

