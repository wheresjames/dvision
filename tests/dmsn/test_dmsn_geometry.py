"""dmsn plane geometry: the convex hull the Hull button writes, and the cell
mask the explore mission's coverage is measured over."""
from __future__ import annotations

from dcmn.maps import GridGeometry
from dmsn.geometry import cell_mask, convex_hull


def ring_area(ring):
    n = len(ring)
    return sum(ring[i][0] * ring[(i + 1) % n][1] - ring[(i + 1) % n][0] * ring[i][1] for i in range(n))


def test_the_hull_drops_interior_and_collinear_points_and_stays_counter_clockwise():
    square = [(0., 0.), (10., 0.), (10., 10.), (0., 10.)]
    hull = convex_hull(square + [(5., 5.), (10., 5.), (5., 11.)])
    assert set(hull) == {(0., 0.), (10., 0.), (10., 10.), (5., 11.), (0., 10.)}
    assert ring_area(hull) > 0, 'the hull is a counter-clockwise ring'
    assert convex_hull(square + [(5., 0.)]) == [(0., 0.), (10., 0.), (10., 10.), (0., 10.)], \
        'a collinear point on the hull edge is skipped'


def test_the_hull_of_degenerate_inputs_stays_well_formed():
    assert convex_hull([]) == []
    assert convex_hull([(3., 4.)]) == [(3., 4.)]
    assert convex_hull([(0., 0.), (0., 0.), (1., 1.)]) == [(0., 0.), (1., 1.)]
    assert convex_hull([(0., 0.), (5., 5.), (10., 10.)]) == [(0., 0.), (10., 10.)], \
        'a straight line has no ring in it'


def test_the_mask_covers_the_cells_whose_centres_are_inside():
    geometry = GridGeometry(0., 0., 1., 8, 6)
    mask = cell_mask([[2., 2.], [5., 2.], [5., 5.], [2., 5.]], geometry)
    assert len(mask) == 6 and len(mask[0]) == 8, 'mask[row][col]: rows are y, columns are x'
    inside = {(row, col) for row, cells in enumerate(mask) for col, hit in enumerate(cells) if hit}
    assert inside == {(row, col) for row in range(2, 5) for col in range(2, 5)}, \
        'the cells whose centres fall in the square, and only those'


def test_the_mask_of_an_empty_or_off_grid_area_is_empty():
    geometry = GridGeometry(0., 0., 1., 4, 4)
    assert not any(any(row) for row in cell_mask([], geometry))
    beyond = cell_mask([[20., 20.], [30., 20.], [30., 30.], [20., 30.]], geometry)
    assert not any(any(row) for row in beyond), 'an area outside the grid masks nothing'
    straddling = cell_mask([[-2., -2.], [3., -2.], [3., 3.], [-2., 3.]], geometry)
    hit = {(row, col) for row, cells in enumerate(straddling) for col, c in enumerate(cells) if c}
    assert hit == {(row, col) for row in range(3) for col in range(3)}, \
        'an area larger than the grid clips to it (the far cells\' centres fall outside)'


def test_the_mask_of_a_small_area_over_a_large_grid_costs_its_own_size():
    geometry = GridGeometry(0., 0., .5, 200, 200)
    mask = cell_mask([[40., 40.], [42., 40.], [42., 42.], [40., 42.]], geometry)
    hit = {(row, col) for row, cells in enumerate(mask) for col, c in enumerate(cells) if c}
    assert hit == {(row, col) for row in range(80, 84) for col in range(80, 84)}
