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
from dtest.tkfixture import hidden_tk, mapped_root_available


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


# -- mission shapes, pointer input and the fallback extent ------------------------------------------

def test_polygon_and_text_shapes_paint_in_both_backends_at_the_same_place():
    from PIL import Image
    from dcmn.map_pane import choose_fallback_extent, nearest_handle
    from dcmn.mapview import Shape
    grid = sample_grid()
    square = Shape.polygon([(6., 1.), (9., 1.), (9., 4.), (6., 4.)], fill=theme.GOAL, outline=theme.GOAL)
    label = Shape.label(2., 6., 'home', outline=theme.TEXT)
    overlay = Overlay(shapes=(square, label))
    shapes = list(overlay_shapes(overlay, grid.geometry.cell_m))
    assert square in shapes and label in shapes
    image = snapshot_image(grid, overlay=overlay, cell_px=10)
    inside = image.getpixel((int(7.5 / .5 * 10), int(2.5 / .5 * 10)))
    assert inside[:3] == theme.rgb(theme.GOAL), 'PIL filled the polygon where it is'
    with hidden_tk() as root:
        pane = MapPane(root, width=200, height=160)
        pane.set_grid(grid)
        pane.set_overlay(shapes=(square, label))
        root.update_idletasks()
        pane.refresh(force=True)
        items = pane.canvas.find_withtag('overlay')
        kinds = {pane.canvas.type(item) for item in items}
        assert {'polygon', 'text'} <= kinds
        polygon = next(i for i in items if pane.canvas.type(i) == 'polygon')
        coords = pane.canvas.coords(polygon)
        expected = [c for x, y in square.points for c in project(pane.view, grid.geometry, x, y)]
        assert coords == pytest.approx(expected, abs=1.)
        text = next(i for i in items if pane.canvas.type(i) == 'text')
        assert pane.canvas.itemcget(text, 'text') == 'home'
    assert isinstance(image, Image.Image)
    assert nearest_handle((1., 1.), [('a', 1.2, 1.), ('b', 5., 5.)], .5) == 'a'
    assert nearest_handle((1., 1.), [('a', 1.8, 1.)], .5) is None
    assert choose_fallback_extent() == (None, choose_fallback_extent()[1])


def test_fallback_extent_order_is_area_workspace_reference_then_pose():
    from dcmn.map_pane import choose_fallback_extent
    area, workspace, reference = (0, 0, 10, 10), (5, 5, 50, 50), (-1, -1, 1, 1)
    assert choose_fallback_extent(area_bounds=area, workspace=workspace, reference_bounds=reference,
                                  pose=(3, 3))[1] == 'mission area'
    assert choose_fallback_extent(workspace=workspace, reference_bounds=reference, pose=(3, 3))[1] == 'profile workspace'
    assert choose_fallback_extent(reference_bounds=reference, pose=(3, 3))[1] == 'reference image'
    bounds, label = choose_fallback_extent(pose=(3, 4), min_side_m=100.)
    assert bounds == (-47., -46., 53., 54.) and 'around the vehicle' in label
    bounds, label = choose_fallback_extent()
    assert bounds is None and label


def test_a_pane_without_evidence_takes_clicks_and_pointer_events_over_its_fallback_extent():
    from dcmn.map_pane import reference_footprint
    presses, drags, releases, contexts, keys, clicks = [], [], [], [], [], []
    event = lambda x, y, state=0: type('E', (), {'x': x, 'y': y, 'state': state, 'keysym': 'Delete'})()
    with hidden_tk() as root:
        pane = MapPane(root, width=200, height=200, on_click=lambda x, y: clicks.append((x, y)),
                       on_press=lambda x, y, m: presses.append((x, y, m)),
                       on_drag=lambda x, y, m: drags.append((x, y)),
                       on_release=lambda x, y, m: releases.append((x, y)),
                       on_context=lambda x, y, m: contexts.append((x, y)),
                       on_key=lambda key, point: keys.append((key, point)))
        assert pane.to_map(100, 100) is None
        pane.set_fallback_extent((10., 20., 30., 40.), 'profile workspace')
        root.update_idletasks()
        assert pane.refresh(force=True) and pane.photo is not None, 'the fallback is drawn as unobserved ground'
        pane._click(event(100, 100, 0x1))
        pane._pointer_event(pane.on_drag, event(120, 100))
        pane._pointer_event(pane.on_release, event(120, 100))
        pane._pointer_event(pane.on_context, event(50, 50))
        pane._key(event(0, 0))
        x, y = clicks[0]
        assert 10. <= x <= 30. and 20. <= y <= 40.
        assert presses[0][2] == frozenset({'shift'})
        assert drags and releases and contexts and keys[0][0] == 'Delete'
        assert 'no evidence yet (profile workspace)' in pane.describe(x, y)
        scale = pane.metres_per_pixel()
        assert pane.handle_at(x + 5 * scale, y, [('v', x, y)]) == 'v'
        assert pane.handle_at(x + 30 * scale, y, [('v', x, y)]) is None
        pane.set_grid(sample_grid())
        pane.refresh(force=True)
        assert pane.describe(1., 1.).startswith('1.00, 1.00 m  cell'), 'evidence wins once it exists'
        assert pane.snapshot() is not None
    image = type('I', (), dict(affine=(.5, 0., 0., .5, 10., 20.), width=40, height=20))()
    assert reference_footprint(image) == pytest.approx((9.75, 19.75, 29.75, 29.75))


# -- zooming and panning -----------------------------------------------------
#
# The gate here is that the view is the operator's and the data is not: a zoom
# moves the transform, never the grid, and a snapshot is unaffected by it. The
# rule with teeth is that the reset badge is a control drawn on the map, so a
# press on it must never reach the host as a press on the map -- in the Mission
# tab that press would add a polygon vertex or move the goal.

def _pane_event(x, y, state=0, num=0, delta=0):
    return type('E', (), {'x': x, 'y': y, 'state': state, 'num': num, 'delta': delta,
                          'keysym': 'Escape'})()


def test_a_pane_is_not_zoomable_unless_the_host_asks():
    from dcmn.map_pane import HOVER_PROMPT

    with hidden_tk() as root:
        pane = MapPane(root)
        assert not pane.zoomable and pane.prompt == HOVER_PROMPT
        bound = pane.canvas.bind()
        assert '<Button-4>' not in bound and '<Button-2>' not in bound
        pane.set_grid(sample_grid())
        root.update_idletasks()
        pane.refresh(force=True)
        pane.zoom_at(100, 100, 2.0)
        pane.pan_by(40, 40)
        assert pane.default_view, 'a plain pane has no view of its own to lose'
        assert pane._badge is None


def test_the_wheel_zooms_about_the_pointer_and_keeps_that_point_under_it():
    """The one property that makes a wheel zoom a magnifier rather than a jump.

    Measured once the image is larger than the canvas on both axes. Until it
    is, containment wins: a letterboxed axis has nowhere to move to, so the
    anchor gives way to the clamp rather than pulling empty canvas into view.
    """
    with hidden_tk() as root:
        pane = MapPane(root, zoom=True)
        pane.set_grid(sample_grid(geometry(cell_m=.05)))   # fine cells: room to zoom
        root.update_idletasks()
        pane.refresh(force=True)
        width, height = pane.canvas_size()
        for _ in range(3): pane.zoom_at(width / 2, height / 2, 2.0)
        pointer = (140, 150)
        before = pane.to_map(*pointer)
        pane._wheel(_pane_event(*pointer, num=4))      # X11 wheel up
        zoomed = pane.zoom
        assert zoomed == pytest.approx(8.0 * 1.25) and not pane.default_view
        assert pane.to_map(*pointer) == pytest.approx(before, abs=1e-6)
        pane._wheel(_pane_event(*pointer, num=5))      # and back down
        assert pane.zoom == pytest.approx(8.0)
        assert pane.to_map(*pointer) == pytest.approx(before, abs=1e-6)


def test_one_notch_each_way_leaves_the_view_it_started_in():
    with hidden_tk() as root:
        pane = MapPane(root, zoom=True)
        pane.set_grid(sample_grid())
        root.update_idletasks()
        pane.refresh(force=True)
        pane._wheel(_pane_event(140, 150, num=4))
        assert not pane.default_view
        pane._wheel(_pane_event(140, 150, num=5))
        assert pane.zoom == pytest.approx(1.0)
        assert pane.default_view, 'and the clamp put the pan back with it'


def test_the_fit_is_the_floor_and_a_cell_is_the_ceiling():
    """The limit belongs to the grid: magnifying a cell past MAX_CELL_PX shows
    a bigger square of one measurement, not more evidence. So a coarse grid
    stops sooner than a fine one on the same canvas."""
    from dcmn.map_pane import MAX_CELL_PX, MAX_ZOOM

    with hidden_tk() as root:
        pane = MapPane(root, zoom=True)
        pane.set_grid(sample_grid())                       # 0.5 m cells
        root.update_idletasks()
        pane.refresh(force=True)
        pane.zoom_at(160, 160, 0.25)
        assert pane.zoom == 1.0 and pane.default_view, 'zooming out past the whole extent shows nothing more'
        coarse = pane.max_zoom
        for _ in range(40): pane.zoom_at(160, 160, 2.0)
        assert pane.zoom == pytest.approx(coarse), 'it stops at the ceiling rather than anywhere it likes'
        assert pane.view.cell * pane.state.shown.geometry.cell_m == pytest.approx(MAX_CELL_PX)

        pane.reset_view()
        pane.set_grid(sample_grid(geometry(cell_m=.05)))    # ten times finer
        pane.refresh(force=True)
        assert pane.max_zoom > coarse * 5, 'a finer grid is worth more magnification'
        assert pane.max_zoom <= MAX_ZOOM


def test_a_pan_cannot_drag_the_image_off_its_own_canvas():
    with hidden_tk() as root:
        pane = MapPane(root, zoom=True)
        pane.set_grid(sample_grid())
        root.update_idletasks()
        pane.refresh(force=True)
        pane.pan_by(400, 400)
        assert pane.default_view, 'at the fitted zoom the image is contained: there is nothing to pan'
        pane.zoom_at(160, 160, 4.0)
        pane.pan_by(10_000, 10_000)
        width, height = pane.canvas_size()
        # The image still covers the canvas: its top-left cannot come inside it.
        assert pane.view.offset_x <= 1e-6 and pane.view.offset_y <= 1e-6
        assert pane.view.offset_x + sample_grid().geometry.extent_m[0] * pane.view.cell >= width - 1e-6
        assert pane.view.offset_y + sample_grid().geometry.extent_m[1] * pane.view.cell >= height - 1e-6


def test_the_reset_badge_appears_only_when_there_is_a_view_to_reset():
    with hidden_tk() as root:
        pane = MapPane(root, zoom=True)
        pane.set_grid(sample_grid())
        root.update_idletasks()
        pane.refresh(force=True)
        assert pane._badge is None and not pane.canvas.find_withtag('viewreset')
        pane.zoom_at(160, 160, 2.0)
        assert pane._badge is not None and pane.canvas.find_withtag('viewreset')
        x0, y0, x1, y1 = pane._badge
        width, _ = pane.canvas_size()
        assert x1 <= width and y0 >= 0, 'the badge sits inside the canvas'
        assert x0 > width / 2, 'and in its top-right corner'
        pane.reset_view()
        assert pane.default_view and pane._badge is None
        assert not pane.canvas.find_withtag('viewreset')


def test_a_press_on_the_badge_resets_the_view_and_never_reaches_the_host():
    """In the Mission tab that press would otherwise add a vertex or set a goal."""
    clicks, presses, releases = [], [], []
    with hidden_tk() as root:
        pane = MapPane(root, zoom=True, on_click=lambda x, y: clicks.append((x, y)),
                       on_press=lambda x, y, m: presses.append((x, y)),
                       on_release=lambda x, y, m: releases.append((x, y)))
        pane.set_grid(sample_grid())
        root.update_idletasks()
        pane.refresh(force=True)
        pane.zoom_at(160, 160, 2.0)
        x0, y0, x1, y1 = pane._badge
        middle = ((x0 + x1) / 2, (y0 + y1) / 2)
        pane._click(_pane_event(*middle))
        pane._release(_pane_event(*middle))
        assert pane.default_view, 'the badge reset the view'
        assert not clicks and not presses and not releases, 'the badge swallowed the gesture'
        # And a press anywhere else is still the host's.
        pane._click(_pane_event(20, 200))
        pane._release(_pane_event(20, 200))
        assert clicks and presses and releases


def test_panning_moves_the_view_without_re_rendering_the_grid():
    """A pan changes which cells are on screen, not what the raster says."""
    with hidden_tk() as root:
        pane = MapPane(root, zoom=True)
        pane.set_grid(sample_grid(geometry(cell_m=.05)))
        root.update_idletasks()
        pane.refresh(force=True)
        pane.zoom_at(160, 160, 3.0)
        serial, zoom = pane._image_serial, pane.zoom
        offset = (pane.view.offset_x, pane.view.offset_y)
        pane._pan_press(_pane_event(100, 100))
        pane._pan_move(_pane_event(130, 90))
        pane._pan_release(_pane_event(130, 90))
        assert pane.view.offset_x == pytest.approx(offset[0] + 30)
        assert pane.view.offset_y == pytest.approx(offset[1] - 10)
        assert pane._image_serial == serial, 'the grid was rendered again for a pan'
        assert pane.zoom == pytest.approx(zoom), 'a pan is not a zoom'


def test_only_the_visible_cells_are_scaled_and_uploaded_however_far_it_zooms():
    """The regression that made a zoomed pane a slideshow.

    Scaling the whole extent costs the square of the zoom: a 200x160 cell grid
    at 20x is a 6400x5120 image, of which a 320x320 canvas shows 0.3%. The
    uploaded tile has to stay the size of the canvas, plus the part-cells at
    its edges.
    """
    with hidden_tk() as root:
        pane = MapPane(root, zoom=True)
        pane.set_grid(sample_grid(geometry(cell_m=.05)))
        root.update_idletasks()
        pane.refresh(force=True)
        width, height = pane.canvas_size()
        assert pane.photo.width() == pytest.approx(320, abs=2), 'the fitted view is the whole grid'
        for factor in (2.0, 4.0, 8.0, 64.0):
            pane.reset_view()
            pane.zoom_at(width / 2, height / 2, factor)
            pane.refresh(force=True)
            drawn = pane.state.shown.geometry.extent_m[0] * pane.view.cell
            cell_px = pane.view.cell * pane.state.shown.geometry.cell_m
            assert drawn > width, 'the view really is larger than the canvas'
            assert pane.photo.width() <= width + 2 * cell_px + 2
            assert pane.photo.height() <= height + 2 * cell_px + 2


def test_a_snapshot_ignores_where_the_operator_happened_to_be_looking():
    """A report is of the mission, not of the pane."""
    grid = sample_grid()
    with hidden_tk() as root:
        pane = MapPane(root, zoom=True)
        pane.set_grid(grid)
        root.update_idletasks()
        pane.refresh(force=True)
        plain = np.asarray(pane.snapshot())
        pane.zoom_at(160, 160, 4.0)
        pane.pan_by(60, -40)
        assert np.array_equal(np.asarray(pane.snapshot()), plain)


def test_the_visible_tile_is_placed_exactly_where_its_first_cell_belongs():
    """What keeps the raster and the overlay on top of each other when zoomed.

    The crop is on cell boundaries and the tile is placed at the canvas
    position of the crop's own first cell, so a point drawn by the overlay and
    the cell painted under it come from the same transform. A tile placed one
    cell out would put a wall half a metre from where the route says it is.
    """
    with hidden_tk() as root:
        pane = MapPane(root, zoom=True)
        pane.set_grid(sample_grid(geometry(cell_m=.05)))
        root.update_idletasks()
        pane.refresh(force=True)
        width, height = pane.canvas_size()
        for factor in (1.0, 3.0, 12.0):
            pane.reset_view()
            if factor > 1.0: pane.zoom_at(width / 2, height / 2, factor)
            pane.refresh(force=True)
            geo = pane.state.shown.geometry
            (col0, row0, col1, row1) = pane._photo_key[1]
            placed = pane.canvas.coords(pane.item)
            assert placed == pytest.approx(list(project(
                pane.view, geo, geo.origin_x_m + col0 * geo.cell_m,
                geo.origin_y_m + row0 * geo.cell_m)))
            # And the crop really is the visible window, not the whole grid.
            if factor > 1.0:
                assert (col1 - col0) < geo.width and (row1 - row0) < geo.height


def test_the_left_button_pans_only_when_no_host_callback_wants_it():
    """The gesture is the host's business, and the pane reads it off what it was given.

    A map that is only looked at pans with a plain drag, which is what people
    try first. One with a click or an editor behind it keeps the button, and
    pans with the middle one or control instead.
    """
    with hidden_tk() as root:
        viewing = MapPane(root, zoom=True)
        assert viewing._left_pans and 'drag pans' in viewing.prompt
        assert 'middle-drag' not in viewing.prompt
        clicked = MapPane(root, zoom=True, on_click=lambda x, y: None)
        assert not clicked._left_pans and 'middle-drag or ctrl-drag' in clicked.prompt

        # Whichever button it is, it drives the same pan.
        for pane in (viewing, clicked):
            pane.set_grid(sample_grid(geometry(cell_m=.05)))
            root.update_idletasks()
            pane.refresh(force=True)
            pane.zoom_at(160, 160, 4.0)
            offset = pane.view.offset_x
            pane._pan_press(_pane_event(100, 100))
            pane._pan_move(_pane_event(140, 100))
            pane._pan_release(_pane_event(140, 100))
            assert pane.view.offset_x == pytest.approx(offset + 40)
        # And a click on the pane that took one still reaches its host.
        assert clicked.to_map(100, 100) is not None


def test_a_left_drag_on_a_viewing_pane_really_pans_it():
    """The binding, not just the policy: synthesized clicks need a mapped window."""
    with hidden_tk() as root:
        pane = MapPane(root, zoom=True)
        if not mapped_root_available():
            assert pane._left_pans, 'the policy half runs everywhere'
            return
        # Mapped from the hidden root rather than a fresh Tk(), which the suite
        # forbids: pointer events are not dispatched to an unmapped widget.
        pane.widget.pack(fill='both', expand=True)
        pane.set_grid(sample_grid(geometry(cell_m=.05)))
        root.deiconify()
        root.update_idletasks(); root.update()
        pane.refresh(force=True)
        width, height = pane.canvas_size()
        pane.zoom_at(width / 2, height / 2, 4.0)
        offset = pane.view.offset_x
        pane.canvas.event_generate('<Button-1>', x=100, y=100)
        pane.canvas.event_generate('<B1-Motion>', x=140, y=100)
        pane.canvas.event_generate('<ButtonRelease-1>', x=140, y=100)
        root.update()
        assert pane.view.offset_x == pytest.approx(offset + 40), 'a plain drag did not pan'
        root.withdraw()


def test_a_viewing_pane_still_resets_from_its_badge_with_the_left_button():
    with hidden_tk() as root:
        pane = MapPane(root, zoom=True)
        pane.set_grid(sample_grid(geometry(cell_m=.05)))
        root.update_idletasks()
        pane.refresh(force=True)
        pane.zoom_at(160, 160, 4.0)
        x0, y0, x1, y1 = pane._badge
        pane._pan_press(_pane_event((x0 + x1) / 2, (y0 + y1) / 2))
        pane._pan_move(_pane_event((x0 + x1) / 2 + 40, (y0 + y1) / 2))
        pane._pan_release(_pane_event((x0 + x1) / 2 + 40, (y0 + y1) / 2))
        assert pane.default_view, 'a press on the badge reset instead of starting a pan'
