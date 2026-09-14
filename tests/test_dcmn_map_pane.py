"""The shared map pane: what it paints, and what it refuses to paint.

The map pane's review gate is that the synthetic room renders correctly, that
hover reads out plausible metres and probabilities, and that never-observed is
visibly distinct from free. The rule with teeth is the last one: a pane that
paints an unobserved cell as empty floor throws away the whole point of the
evidence format.
"""

from __future__ import annotations

import numpy as np
import pytest

from dcmn import theme
from dcmn.map_pane import (MODES, MapPane, Overlay, occupancy_rgb, overlay_shapes,
                           project, render_raster, snapshot_image, unproject)
from dcmn.mapview import MapView
from dcmn.maps import EvidenceGrid, GridGeometry, quantize, stamp_ms
from dtest.tkfixture import hidden_tk


def geometry(cell_m=.5, width_m=10., height_m=8., origin=(0., 0.)):
    return GridGeometry.from_extent(width_m, height_m, cell_m,
                                    origin_x_m=origin[0], origin_y_m=origin[1])


def sample_grid(geo=None):
    """Free floor, one wall block, and a corner nobody has looked at."""
    geo = geo or geometry()
    grid = EvidenceGrid.blank(geo)
    occupancy = grid.occupancy.copy()
    observed = grid.observed_ms.copy()
    occupancy[0, :, :] = quantize(.05)
    observed[0, :, :] = stamp_ms(10.)
    occupancy[0, 2:5, 2:5] = quantize(.95)
    occupancy[0, -3:, -3:] = 255           # never observed, and its clock with it
    observed[0, -3:, -3:] = 0
    observed[0, 0, :] = stamp_ms(2.)       # an older row, for the age mode
    return EvidenceGrid(geo, occupancy, observed, 'cam', revision=7, sim_time_s=10.)


# -- rasters -----------------------------------------------------------------

def test_never_observed_is_its_own_colour_and_not_free_space():
    grid = sample_grid()
    rgb = occupancy_rgb(grid)
    never = grid.never_observed[0]
    unobserved = np.array(theme.rgb(theme.UNOBSERVED), np.uint8)
    assert (rgb[never] == unobserved).all()
    free = rgb[~never & (grid.occupancy[0] < 64)]
    assert not (free == unobserved).any(axis=-1).all(), \
        'free space is painted the never-observed colour'
    assert len(np.unique(rgb[~never].reshape(-1, 3), axis=0)) >= 2


def test_occupied_and_free_land_at_opposite_ends_of_the_ramp():
    grid = sample_grid()
    rgb = occupancy_rgb(grid)
    free, occupied = rgb[10, 10], rgb[3, 3]
    assert not np.array_equal(free, occupied)
    # The diverging ramp runs cool to warm, so an obstacle is the redder end.
    assert int(occupied[0]) - int(occupied[2]) > int(free[0]) - int(free[2])


@pytest.mark.parametrize('mode', MODES)
def test_every_mode_paints_a_full_size_raster(mode):
    grid = sample_grid()
    cost = np.where(grid.occupancy[0] > 127, np.inf, 1.)
    raster = render_raster(grid, mode=mode, sim_now_s=12., cost=cost)
    assert raster.shape == (grid.geometry.height, grid.geometry.width, 3)
    assert raster.dtype == np.uint8


def test_the_age_mode_separates_a_fresh_cell_from_an_old_one():
    grid = sample_grid()
    raster = render_raster(grid, mode='age', sim_now_s=12., horizon_s=5.)
    assert not np.array_equal(raster[0, 5], raster[6, 5])
    assert (raster[-1, -1] == np.array(theme.rgb(theme.UNOBSERVED))).all()


def test_the_cost_mode_says_forbidden_rather_than_merely_expensive():
    grid = sample_grid()
    cost = np.where(grid.occupancy[0] > 127, np.inf, np.linspace(
        0., 1., grid.geometry.width)[None, :].repeat(grid.geometry.height, 0))
    raster = render_raster(grid, mode='cost', cost=cost)
    assert (raster[3, 3] == np.array(theme.rgb(theme.DANGER))).all()
    assert not (raster[10, 10] == np.array(theme.rgb(theme.DANGER))).all()


def test_blocked_tints_the_margin_differently_from_what_the_sensor_saw():
    """The band a margin adds must be visible against the obstacle it came from."""
    from dcmn.map_pane import blocked_rgb

    grid = sample_grid()
    cost = np.ones((grid.geometry.height, grid.geometry.width), np.float32)
    cost[2:5, 2:5] = np.inf                     # the obstacle itself
    cost[5, 3] = np.inf                         # margin over observed-free floor
    cost[-1, -1] = np.inf                       # margin reaching unobserved cells
    evidence = occupancy_rgb(grid)
    tinted = blocked_rgb(grid, cost)
    assert np.array_equal(tinted[10, 10], evidence[10, 10]), 'an unblocked cell changed'
    margin, wall, unseen = tinted[5, 3], tinted[3, 3], tinted[-1, -1]
    assert not np.array_equal(margin, evidence[5, 3])
    assert not np.array_equal(margin, wall), 'the margin reads as more wall'
    assert not np.array_equal(unseen, margin)


def test_a_blocked_view_with_no_cost_paints_nothing():
    assert render_raster(sample_grid(), mode='blocked', cost=None) is None


def test_a_cost_mode_with_no_cost_layer_paints_nothing_rather_than_zeros():
    """No cost is not zero cost, and a pane may not invent the difference."""
    assert render_raster(sample_grid(), mode='cost', cost=None) is None


def test_the_no_cost_notice_goes_away_once_the_pane_paints_again():
    """A notice left over from a mode nobody is in says the wrong thing."""
    from dcmn.map_pane import HOVER_PROMPT

    grid = sample_grid()
    with hidden_tk() as root:
        pane = MapPane(root, mode='cost')
        pane.set_grid(grid)
        root.update_idletasks()
        pane.refresh(force=True)
        assert 'no cost layer' in pane.readout.get()
        pane.set_mode('occupancy')
        assert pane.readout.get() == HOVER_PROMPT
        assert pane.photo is not None


def test_an_unknown_mode_is_refused():
    with pytest.raises(ValueError, match='display mode'):
        render_raster(sample_grid(), mode='temperature')


# -- geometry ----------------------------------------------------------------

def test_the_pixel_transform_inverts_itself_including_a_shifted_origin():
    geo = geometry(origin=(100., -50.))
    view = MapView(cell=12., margin=0)
    view.offset_x, view.offset_y = 7., 3.
    for point in ((100., -50.), (104.25, -46.5), (110., -42.)):
        assert unproject(view, geo, *project(view, geo, *point)) == \
            pytest.approx(point)


def test_a_snapshot_is_the_raster_at_cell_resolution_with_the_overlay_on_it():
    grid = sample_grid()
    plain = snapshot_image(grid, cell_px=4)
    assert plain.size == (grid.geometry.width * 4, grid.geometry.height * 4)
    drawn = snapshot_image(grid, cell_px=4, overlay=Overlay(
        route=((1., 1.), (8., 6.)), goal=(8., 6.), start=(1., 1.),
        vehicle=(4., 4., 90.)))
    assert np.asarray(plain).shape == np.asarray(drawn).shape
    assert not np.array_equal(np.asarray(plain), np.asarray(drawn))


def test_a_route_and_its_furniture_are_all_drawable_shapes():
    shapes = list(overlay_shapes(Overlay(
        route=((1., 1.), (4., 2.), (8., 6.)), goal=(8., 6.), start=(1., 1.),
        vehicle=(4., 4., 45.), inflation_m=.75), cell_m=.5))
    kinds = {shape.kind for shape in shapes}
    assert kinds <= {'rect', 'oval', 'line'}, 'a shape no PIL backend can draw'
    # Two segments, three waypoints, a start, a goal and a vehicle at minimum.
    assert sum(shape.kind == 'line' for shape in shapes) >= 2
    assert theme.ROUTE in {shape.outline for shape in shapes}


def test_an_empty_overlay_draws_nothing():
    assert list(overlay_shapes(Overlay(), cell_m=.5)) == []


# -- the widget --------------------------------------------------------------

def test_the_pane_paints_a_grid_and_reads_a_cell_back_out():
    grid = sample_grid()
    with hidden_tk() as root:
        pane = MapPane(root, width=200, height=160)
        pane.set_grid(grid, sim_now_s=12.)
        root.update_idletasks()
        assert pane.refresh(force=True)
        assert pane.photo is not None
        # A point inside the wall block, and one nobody has observed.
        wall = grid.geometry.cell_centre_m(3, 3)
        assert 'p=0.9' in pane.describe(*wall)
        assert 'cell 3,3' in pane.describe(*wall)
        corner = grid.geometry.cell_centre_m(grid.geometry.width - 1,
                                             grid.geometry.height - 1)
        assert 'never observed' in pane.describe(*corner)
        assert 'outside the grid' in pane.describe(-5., -5.)


def test_a_click_hands_back_map_metres_not_pixels():
    grid = sample_grid()
    clicks = []
    with hidden_tk() as root:
        pane = MapPane(root, width=200, height=160, on_click=lambda x, y: clicks.append((x, y)))
        pane.set_grid(grid)
        pane.canvas.configure(width=200, height=160)
        root.update_idletasks()
        pane.refresh(force=True)
        pane._click(type('E', (), {'x': 100, 'y': 80})())
    assert len(clicks) == 1
    x, y = clicks[0]
    x0, y0, x1, y1 = grid.geometry.bounds_m()
    assert x0 <= x <= x1 and y0 <= y <= y1


def test_a_pane_with_no_grid_paints_nothing_and_says_so():
    with hidden_tk() as root:
        pane = MapPane(root)
        assert pane.refresh(force=True)
        assert pane.photo is None
        assert pane.describe(0., 0.) == 'no grid'
        assert pane.to_map(10, 10) is None


def test_the_pane_repaints_at_map_hz_and_not_faster():
    from dcmn.pacing import MAP_HZ
    grid = sample_grid()
    now = [0.]
    with hidden_tk() as root:
        pane = MapPane(root)
        pane._paint._clock = lambda: now[0]
        pane._paint.reset()
        pane.set_grid(grid)
        root.update_idletasks()
        painted = 0
        for _ in range(200):                     # two seconds of 100 Hz ticking
            now[0] += .01
            painted += bool(pane.refresh())
    assert painted <= 2 * MAP_HZ + 1


def test_a_resize_drag_does_not_escape_the_repaint_cap():
    """<Configure> fires continuously while a window is dragged."""
    from dcmn.pacing import MAP_HZ
    now = [0.]
    with hidden_tk() as root:
        pane = MapPane(root)
        pane._paint._clock = lambda: now[0]
        pane._paint.reset()
        pane.set_grid(sample_grid())
        root.update_idletasks()
        painted = 0
        original = pane.refresh
        def counted(**kwargs):
            nonlocal painted
            drawn = original(**kwargs)
            painted += bool(drawn)
            return drawn
        pane.refresh = counted
        for _ in range(100):                      # one second of 100 Hz resizing
            now[0] += .01
            pane._resized(None)
        pane.canvas.after_cancel(pane._resize_after)
    assert painted <= MAP_HZ + 1


def test_the_pane_renders_the_synthetic_room_the_fixture_publishes():
    """The review gate, in one test: the room, its doorway and its shadow."""
    from dtest.synthetic import SyntheticRoom

    geo = GridGeometry.from_extent(20., 20., .5)
    room = SyntheticRoom(geo)
    for step in range(12): room.sweep(1. + step)
    grid = room.grid()
    with hidden_tk() as root:
        pane = MapPane(root, width=300, height=300)
        pane.set_grid(grid, sim_now_s=13.)
        root.update_idletasks()
        assert pane.refresh(force=True)
        image = np.asarray(pane.snapshot(cell_px=3))
    colours = {tuple(c) for c in image.reshape(-1, 3)}
    assert theme.rgb(theme.UNOBSERVED) in colours, 'the shadow is not drawn'
    assert len(colours) >= 3, 'wall, floor and unobserved are not three colours'
