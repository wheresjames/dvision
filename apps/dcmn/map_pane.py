"""`MapPane`: the one way an evidence grid and a route are drawn.

`daic` kept its local-map canvas private, and the piece worth having -- an
occupancy grid with a planned path over it -- could not be reused, reported, or
looked at from anywhere else. This is that widget, lifted into dcmn, world-
framed and raster-based: `dalg`'s Grids tab, `dnav`'s Plan and Cost tabs and
both modules' reports draw through it, so a tab that needs a primitive this
lacks adds it here rather than forking a second drawing of a map.

Two implementation notes carry most of the value:

**Raster, not canvas items.** `daic` created one canvas rectangle per cell,
which is fine for a small rolling grid and hopeless at 200x200 = 40 000 items.
The grid becomes a numpy array, a PIL image and one ``PhotoImage`` with NEAREST
resampling -- the pattern `device_view.show_image` already uses -- and only the
overlays are canvas items, of which there are a handful.

**One renderer for the screen and the report.** The overlays are described once
as shapes in map metres, exactly as :mod:`dcmn.mapview` describes a map, and two
thin backends paint them: a Tk canvas for the window and PIL for
:meth:`MapPane.snapshot`. A report image and a screenshot cannot drift apart
because there is nothing to drift.

And one rule with teeth: **never-observed renders as its own colour, never as
free space.** The sentinel in :mod:`dcmn.maps` exists so an operator can see
what has not been looked at; a pane that painted it as empty floor would throw
away the single most valuable bit in the format.
"""

from __future__ import annotations

if __name__ == '__main__':
    # A widget that is also its own minimal host window; the bootstrap has to
    # precede the sibling imports below.
    import sys as _sys
    from pathlib import Path as _Path
    for _path in (str(_Path(__file__).resolve().parents[2]),
                  str(_Path(__file__).resolve().parents[1])):
        if _path not in _sys.path: _sys.path.insert(0, _path)

import math
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterator, Sequence

import numpy as np

from dcmn import theme
from dcmn.imagery import ReferenceImage, affine_inverse
from dcmn.maps import (DEFAULT_STALENESS_HORIZON_S, EvidenceGrid, GridGeometry,
                       PROBABILITY_SCALE)
from dcmn.mapview import MapView, Shape
from dcmn.pacing import MAP_HZ, Paced

#: What a pane can show of one grid. ``never`` is the observation debugger --
#: what has been looked at at all -- and ``cost`` is the derived layer a
#: consumer supplies; the pane never derives cost itself, because the policy
#: that does belongs to `dnav`. ``blocked`` is the two at once:
#: the evidence as the sensor believes it, with the cells the consumer's cost
#: forbids tinted over it, so what a margin added is visible against what was
#: actually seen.
MODES = ('occupancy', 'age', 'never', 'cost', 'blocked')

#: How strongly a blocked cell is tinted over the evidence beneath it. Half:
#: enough that the margin reads as a band around what the sensor saw, while
#: free, occupied and never-observed stay distinguishable underneath it.
BLOCKED_TINT = 0.5

#: What the hover readout says before the pointer has been anywhere.
HOVER_PROMPT = 'hover for a cell'

#: What a zoomable pane adds to that prompt, because a gesture nobody mentions
#: is a gesture nobody finds. Which one it says depends on whether the host has
#: claimed the left button for editing or clicking.
ZOOM_HINT = ' · wheel zooms · middle-drag or ctrl-drag pans'
ZOOM_HINT_LEFT = ' · wheel zooms · drag pans'

#: One wheel notch. 1.0 is the fitted view every pane starts in, and it is also
#: the floor: zooming out past the whole extent would only shrink it inside the
#: canvas, which shows nothing more.
ZOOM_STEP = 1.25
MIN_ZOOM = 1.0

#: How large one cell is worth magnifying to, in screen pixels. Past this the
#: pane is drawing a bigger square of the same single measurement: the evidence
#: has no more detail in it, and a goal can already be placed anywhere inside a
#: cell. So the ceiling belongs to the *grid*, not to the widget -- a 0.25 m
#: grid on a wide canvas runs out of detail sooner than a fine one on a narrow
#: one -- and :attr:`MapPane.max_zoom` reads it off the fit rather than taking a
#: number on faith.
MAX_CELL_PX = 32.0
#: A backstop for a pathologically fine grid, so the factor on the badge stays
#: a number an operator can reason about.
MAX_ZOOM = 64.0

#: The reset badge: how far its corner sits inside the image's top-right, and
#: the padding around its text. Drawn stippled rather than with a real alpha
#: channel, because a Tk canvas has no translucent fill -- the same trick the
#: vehicle's view cone uses.
BADGE_INSET_PX = 6
BADGE_PAD_PX = 4

#: Pixels per cell in a report snapshot, before the overlays are drawn. Large
#: enough that a 0.5 m cell is legible on a page and small enough that a
#: 200x200 grid stays a reasonable image.
SNAPSHOT_CELL_PX = 6


# -- rasters -----------------------------------------------------------------

def _unobserved(shape) -> np.ndarray:
    return np.broadcast_to(np.array(theme.rgb(theme.UNOBSERVED), np.uint8),
                           (*shape, 3)).copy()


def occupancy_rgb(grid: EvidenceGrid, layer: int = 0) -> np.ndarray:
    """Probability through the diverging ramp; never-observed its own colour."""
    occupancy, _ = grid.layer(layer)
    rgb = theme.CONFIDENCE_LUT[
        (occupancy.astype(np.uint16) * 255 // PROBABILITY_SCALE).clip(0, 255).astype(np.uint8)]
    return np.where((occupancy == 255)[..., None], _unobserved(occupancy.shape), rgb)


def age_rgb(grid: EvidenceGrid, sim_now_s: float,
            horizon_s: float = DEFAULT_STALENESS_HORIZON_S, layer: int = 0) -> np.ndarray:
    """Simulated seconds since each cell was observed, clamped at the horizon.

    Fresh is the dark end and stale the bright one, so a map going stale
    brightens rather than fading -- an operator notices something arriving far
    more reliably than something leaving.
    """
    _, observed = grid.layer(layer)
    age = np.maximum(0., float(sim_now_s) - observed.astype(np.float64) / 1000.)
    scaled = np.clip(age / max(1e-6, float(horizon_s)), 0., 1.)
    rgb = theme.RANGE_LUT[(scaled * 255).astype(np.uint8)]
    return np.where((observed == 0)[..., None], _unobserved(observed.shape), rgb)


def never_rgb(grid: EvidenceGrid, layer: int = 0) -> np.ndarray:
    """Looked at or not, and nothing else: the sentinel on its own."""
    occupancy, _ = grid.layer(layer)
    rgb = np.broadcast_to(np.array(theme.rgb(theme.PANEL), np.uint8),
                          (*occupancy.shape, 3)).copy()
    rgb[occupancy == 255] = theme.rgb(theme.WARN)
    return rgb


def cost_rgb(cost, never_observed=None) -> np.ndarray:
    """A consumer's derived cost layer, normalized over its own finite range.

    Infinite cost -- an obstacle a planner may not enter -- is painted as the
    danger colour rather than being folded into the ramp, because "expensive"
    and "forbidden" are different answers and a policy debugger that blurs them
    is not one.
    """
    values = np.asarray(cost, np.float64)
    finite = np.isfinite(values)
    scaled = np.zeros(values.shape)
    if finite.any():
        low, high = values[finite].min(), values[finite].max()
        scaled[finite] = (values[finite] - low) / (high - low) if high > low else .5
    rgb = theme.RANGE_LUT[(scaled * 255).astype(np.uint8)]
    rgb[~finite] = theme.rgb(theme.DANGER)
    if never_observed is not None:
        rgb[np.asarray(never_observed, bool) & finite] = theme.rgb(theme.UNOBSERVED)
    return rgb


def blocked_rgb(grid: EvidenceGrid, cost, layer: int = 0,
                tint: float = BLOCKED_TINT) -> np.ndarray:
    """The evidence, with every cell the cost forbids tinted toward danger.

    An obstacle the sensor saw stays red, because the tint only deepens it; a
    cell the sensor saw as free but the policy's margin blocks turns a
    distinctly different colour. That band is the inflation, drawn against the
    thing it was inflated from -- which neither the evidence view nor the cost
    view can show on its own.
    """
    rgb = occupancy_rgb(grid, layer).astype(np.float32)
    values = np.asarray(cost)
    values = values[layer] if values.ndim == 3 else values
    blocked = ~np.isfinite(values)
    danger = np.array(theme.rgb(theme.DANGER), np.float32)
    rgb[blocked] = rgb[blocked] * (1.0 - tint) + danger * tint
    return rgb.round().astype(np.uint8)


def render_raster(grid: EvidenceGrid | None, *, mode: str = 'occupancy',
                  layer: int = 0, sim_now_s: float = 0.0,
                  horizon_s: float = DEFAULT_STALENESS_HORIZON_S,
                  cost: np.ndarray | None = None,
                  never_observed: np.ndarray | None = None) -> np.ndarray | None:
    """One ``(height, width, 3)`` uint8 image of a grid, or None if there is none."""
    if grid is None: return None
    if mode not in MODES: raise ValueError(f'unknown map display mode {mode!r}')
    if mode == 'occupancy': return occupancy_rgb(grid, layer)
    if mode == 'age': return age_rgb(grid, sim_now_s, horizon_s, layer)
    if mode == 'never': return never_rgb(grid, layer)
    if cost is None: return None
    if mode == 'blocked': return blocked_rgb(grid, cost, layer)
    return cost_rgb(np.asarray(cost)[layer] if np.ndim(cost) == 3 else cost,
                    grid.layer(layer)[0] == 255 if never_observed is None else never_observed)


# -- reference backgrounds ----------------------------------------------------

#: What the visible label says, ahead of the image's source category. The
#: picture behind a pane is *reference*, not evidence: an operator glancing at
#: a background must never mistake a rendered truth map for something the
#: vehicle observed, and the label is how the pane keeps that distinction in
#: front of the person looking at it.
REFERENCE_LABEL = 'reference only'

#: The default opacity of a reference layer under evidence. Below one, so the
#: first thing an operator sees when a background appears is that something
#: changed; the control is there to take it the rest of the way either way.
DEFAULT_BACKGROUND_OPACITY = 0.8


@dataclass(frozen=True)
class Background:
    """One reference image placed for display, and how it is drawn.

    This is the *display* description -- the image plus the operator's
    opacity -- and it is the only thing that crosses from the imagery plane
    into rendering. Constructing it from a :class:`~dcmn.imagery.ReferenceImage`
    a host has already frame-checked is the whole integration; the pane never
    opens the imagery plane itself, because a widget that could discover its
    own truth imagery is a widget that could quietly show it.
    """

    image: ReferenceImage
    opacity: float = DEFAULT_BACKGROUND_OPACITY

    def __post_init__(self) -> None:
        opacity = min(1., max(0., float(self.opacity)))
        object.__setattr__(self, 'opacity', opacity)

    @property
    def identity(self) -> tuple:
        """What identifies one rendering: image, revision, checksum, opacity."""
        return (self.image.image_id, self.image.revision, self.image.checksum,
                self.opacity)

    def label(self) -> str:
        """The visible caption: reference, and what kind of reference it is."""
        category = self.image.source_category
        return f'{REFERENCE_LABEL} · {category}' if category else REFERENCE_LABEL


def background_raster(background: Background, geometry: GridGeometry,
                      *, layer: int = 0) -> np.ndarray:
    """One ``(height, width, 3)`` uint8 plane of the reference under a grid.

    Each cell samples the image at the cell's *centre*, nearest pixel, under
    the inverse of the image's affine -- one sample per cell, no interpolation
    and no stretching: a coarse grid shows the image as coarse blocks, and a
    mismatch between image and grid resolution stays visible as the mismatch
    it is rather than being smoothed into a registration nobody checked.
    Cells whose centres fall outside the image footprint keep the pane's
    ground colour, exactly as §7 requires: the image is clipped, never
    stretched to fill.
    """
    reference = background.image
    inverse = affine_inverse(reference.affine)
    a, b, c, d, e, f = inverse
    width, height = geometry.shape[2], geometry.shape[1]
    # Cell centres along each axis, broadcast to a full plane.
    xs = geometry.origin_x_m + (np.arange(width, dtype=np.float64) + .5) * geometry.cell_m
    ys = geometry.origin_y_m + (np.arange(height, dtype=np.float64) + .5) * geometry.cell_m
    col = a * xs[None, :] + b * ys[:, None] + e      # image columns, (height, width)
    row = c * xs[None, :] + d * ys[:, None] + f      # image rows, (height, width)
    inside = ((col >= -.5) & (col < reference.width - .5) &
              (row >= -.5) & (row < reference.height - .5))
    # Nearest pixel centre: the sample index is floor(coordinate + 0.5).
    nearest_col = np.clip(np.floor(col + .5), 0, reference.width - 1).astype(np.intp)
    nearest_row = np.clip(np.floor(row + .5), 0, reference.height - 1).astype(np.intp)
    pixels = np.asarray(reference.image)
    sample = pixels[nearest_row, nearest_col]        # (height, width, 4) RGBA
    alpha = sample[..., 3:4].astype(np.float32) * (background.opacity / 255.)
    ground = np.array(theme.rgb(theme.CANVAS), np.float32)
    colour = (sample[..., :3].astype(np.float32) * alpha +
              ground * (1. - alpha))
    plane = np.broadcast_to(ground.astype(np.uint8), (height, width, 3)).copy()
    plane[inside] = colour[inside].round().astype(np.uint8)
    return plane


def compose_background(raster: np.ndarray, background: Background,
                      geometry: GridGeometry, never_mask: np.ndarray) -> np.ndarray:
    """Put the reference under the evidence, in exactly the cells it belongs.

    The background appears where nobody has looked -- never-observed cells --
    and observed cells keep their evidence colours, because a cell the
    vehicle has measured is that measurement and a picture under it would
    only dilute it. Not the other way around: the picture never paints over
    evidence. The ``never`` mode is exempt in its entirety, because that
    mode *is* the sentinel and a background would erase the thing it exists
    to show.
    """
    mask = np.asarray(never_mask, bool)
    if mask.ndim == 3: mask = mask[int(0)]            # one layer of a slab
    composed = np.asarray(raster).copy()
    composed[mask] = background_raster(background, geometry)[mask]
    return composed


def background_shapes(background: Background, cell_m: float) -> Iterator[Shape]:
    """The registration marks: the image's footprint and its axes, in metres.

    Four lines around :meth:`ReferenceImage.footprint_m` -- which is derived
    from the affine, never declared separately, so what an operator checks is
    where the image claims to sit -- and an L at the top-left pixel corner
    showing which way the image's columns and rows run, so a rotated or
    mirrored image reads as one at a glance instead of as a map that looks
    slightly wrong.
    """
    reference = background.image
    corners = reference.footprint_m()
    edge = theme.blend(theme.DIM, theme.CANVAS, .45)
    for (x0, y0), (x1, y1) in zip(corners, corners[1:] + corners[:1]):
        yield Shape('line', x0, y0, x1, y1, outline=edge, width=1.)
    # The axis L: column direction then row direction, from the top-left
    # pixel corner. A metre is long enough to read at any sane cell size.
    length = max(1., float(cell_m) * 2.)
    (x, y), (xc, yc), _, (xr, yr) = corners
    def _step(to_x, to_y):
        span = math.hypot(to_x - x, to_y - y)
        return ((to_x - x) / span * length, (to_y - y) / span * length) if span else (0., 0.)
    dx, dy = _step(xc, yc)
    yield Shape('line', x, y, x + dx, y + dy, outline=theme.ACCENT, width=2.)
    dx, dy = _step(xr, yr)
    yield Shape('line', x, y, x + dx, y + dy, outline=theme.DIM, width=2.)


def never_cells(grid: EvidenceGrid | None, *, mode: str, layer: int = 0,
                never_observed: np.ndarray | None = None) -> np.ndarray | None:
    """Which cells a background may fill: the never-observed ones, as a mask.

    The same never-observed definition :func:`render_raster` paints from:
    the layer's occupancy sentinel, or -- in cost mode, where the raster is
    the consumer's cost -- the mask the consumer supplied, so a cost view
    composites under exactly the cells its policy called never-observed.
    """
    if grid is None or mode == 'never': return None
    if mode == 'cost' and never_observed is not None:
        return np.asarray(never_observed, bool)
    occupancy, _ = grid.layer(layer)
    return occupancy == 255


# -- overlays ----------------------------------------------------------------

@dataclass(frozen=True)
class Overlay:
    """Everything drawn on top of the raster, in **map metres**.

    Described once and painted by two backends, which is what keeps a report
    image and the screen identical. A route is the ordered waypoints of
    the plan with their speeds dropped: this pane draws where a plan goes,
    not how fast it goes there.
    """

    route: Sequence[tuple[float, float]] = ()
    permitted: Sequence[tuple[float, float]] = ()
    track: Sequence[tuple[float, float]] = ()
    target: tuple[float, float] | None = None
    vehicle: tuple[float, float, float] | None = None   # x, y, heading degrees
    goal: tuple[float, float] | None = None
    start: tuple[float, float] | None = None
    inflation_m: float = 0.0
    #: Draw ``route`` as a thin dashed proposal: planning intent, not permission.
    proposal: bool = False
    #: A second, dimmer dashed line: a diagnostic candidate or a selected past route.
    candidate: Sequence[tuple[float, float]] = ()
    #: The segment currently being flown, drawn over the route.
    highlight: Sequence[tuple[float, float]] = ()
    #: A host's own primitives -- a mission area, waypoints, a home marker --
    #: drawn above the clearance ring and below the route and the vehicle, so
    #: what the vehicle is doing always reads over what it was asked to do.
    shapes: Sequence[Shape] = ()


def overlay_shapes(overlay: Overlay, cell_m: float) -> Iterator[Shape]:
    """Every primitive an overlay is made of, in map metres.

    The vehicle is a disc with a heading spoke rather than
    :meth:`MapView.draw_drone`'s nose-forward triangle, so it stays the same
    small mark at every zoom in both backends.
    """
    route = [(float(x), float(y)) for x, y in overlay.route]
    radius = max(cell_m * .6, 0.15)
    if overlay.inflation_m > 0:
        # One ring, at the vehicle, showing the clearance the cost policy is
        # enforcing. The inflated cells are already visible in a cost raster;
        # what a raster cannot show is *how far* the margin reaches in metres,
        # and that is a question about scale, so it is answered with a circle
        # of that radius drawn where the vehicle actually is. It goes down
        # first, under the route, so a route that clips it reads as a route
        # over a ring rather than disappearing beneath one.
        anchor = (overlay.vehicle[:2] if overlay.vehicle is not None else
                  overlay.start if overlay.start is not None else
                  route[0] if route else None)
        if anchor is not None:
            x, y = anchor
            r = float(overlay.inflation_m)
            yield Shape('oval', x - r, y - r, x + r, y + r,
                        outline=theme.blend(theme.DANGER, theme.CANVAS, .55), width=1.)
    yield from overlay.shapes
    dot = radius * (.3 if overlay.proposal else .5)
    for (x0, y0), (x1, y1) in zip(route, route[1:]):
        yield Shape('line', x0, y0, x1, y1, outline=theme.ROUTE,
                    width=1. if overlay.proposal else 2., dash=(6, 4) if overlay.proposal else ())
    for x, y in route:
        yield Shape('oval', x - dot, y - dot, x + dot, y + dot, fill=theme.ROUTE, outline=theme.ROUTE)
    for (x0, y0), (x1, y1) in zip(overlay.candidate, overlay.candidate[1:]):
        yield Shape('line', x0, y0, x1, y1, outline=theme.blend(theme.ROUTE, theme.CANVAS, .5),
                    width=1., dash=(3, 5))
    for points, color, width in ((overlay.track, theme.ACCENT, 1.),
                                  (overlay.permitted, theme.GOAL, 4.),
                                  (overlay.highlight, theme.ACCENT, 3.)):
        for (x0,y0), (x1,y1) in zip(points, points[1:]):
            yield Shape('line', x0,y0,x1,y1, outline=color, width=width)
    if overlay.permitted:
        x,y = overlay.permitted[-1]
        yield Shape('rect', x-radius,y-radius,x+radius,y+radius, outline=theme.GOAL, width=2.)
    if overlay.target is not None:
        x,y = overlay.target
        yield Shape('line', x-radius,y-radius,x+radius,y+radius, outline=theme.DANGER, width=2.)
        yield Shape('line', x-radius,y+radius,x+radius,y-radius, outline=theme.DANGER, width=2.)
    if overlay.start is not None:
        x, y = overlay.start
        yield Shape('oval', x - radius, y - radius, x + radius, y + radius,
                    fill=theme.START, outline=theme.START)
    if overlay.goal is not None:
        x, y = overlay.goal
        yield Shape('oval', x - radius, y - radius, x + radius, y + radius,
                    outline=theme.GOAL, width=2.)
        yield Shape('line', x - radius, y, x + radius, y, outline=theme.GOAL, width=2.)
        yield Shape('line', x, y - radius, x, y + radius, outline=theme.GOAL, width=2.)
    if overlay.vehicle is not None:
        x, y, heading_deg = overlay.vehicle
        yield Shape('oval', x - radius, y - radius, x + radius, y + radius,
                    fill=theme.ACCENT, outline=theme.DRONE_EDGE, width=1.)
        # The public compass heading -- north 0, clockwise positive -- never a
        # renderer yaw, and north is -y in the map frame.
        angle = math.radians(float(heading_deg) - 90.)
        yield Shape('line', x, y, x + math.cos(angle) * radius * 2.4,
                    y + math.sin(angle) * radius * 2.4,
                    outline=theme.DRONE_EDGE, width=2.)


class _Extent:
    """A grid's covered area as the width and height :class:`MapView` wants.

    `MapView` measures a map in cells of its own; a grid measures itself in
    metres. Handing it metres makes ``cell`` pixels-per-metre and leaves ``xy``
    and ``to_map`` exactly inverse, which is what the click-to-set goal picker
    needs -- so the transform is adapted rather than modified.
    """

    def __init__(self, geometry: GridGeometry) -> None:
        self.width, self.height = geometry.extent_m


def project(view: MapView, geometry: GridGeometry, x_m: float, y_m: float) -> tuple[float, float]:
    """Absolute map metres to pixels in this pane."""
    return view.xy(float(x_m) - geometry.origin_x_m, float(y_m) - geometry.origin_y_m)


def unproject(view: MapView, geometry: GridGeometry, px: float, py: float) -> tuple[float, float]:
    """Pixels back to absolute map metres; the exact inverse of :func:`project`."""
    x, y = view.to_map(px, py)
    return x + geometry.origin_x_m, y + geometry.origin_y_m


def _paint_shapes_pil(draw, view: MapView, geometry: GridGeometry, shapes) -> None:
    for shape in shapes:
        x0, y0 = project(view, geometry, shape.x0, shape.y0)
        x1, y1 = project(view, geometry, shape.x1, shape.y1)
        width = max(1, int(round(shape.width)))
        if shape.kind == 'polygon':
            points = [project(view, geometry, x, y) for x, y in shape.points]
            draw.polygon(points, fill=shape.fill, outline=shape.outline, width=width)
        elif shape.kind == 'text':
            draw.text((x0, y0), shape.text, fill=shape.outline or shape.fill or theme.TEXT, anchor='mm')
        elif shape.kind == 'rect':
            draw.rectangle((x0, y0, x1, y1), fill=shape.fill, outline=shape.outline, width=width)
        elif shape.kind == 'oval':
            draw.ellipse((min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)),
                         fill=shape.fill, outline=shape.outline, width=width)
        elif shape.dash:
            # PIL has no dash style; paint the on-lengths along the segment.
            length = math.hypot(x1 - x0, y1 - y0)
            position, index = 0., 0
            while position < length:
                end = min(length, position + shape.dash[index % len(shape.dash)])
                if index % 2 == 0:
                    a, b = position / length, end / length
                    draw.line((x0 + (x1 - x0) * a, y0 + (y1 - y0) * a,
                               x0 + (x1 - x0) * b, y0 + (y1 - y0) * b), fill=shape.outline, width=width)
                position, index = end, index + 1
        else:
            draw.line((x0, y0, x1, y1), fill=shape.outline, width=width)


def draw_overlay_pil(image, view: MapView, geometry: GridGeometry, overlay: Overlay) -> None:
    """Paint an overlay onto a PIL image already showing the raster."""
    from PIL import ImageDraw

    _paint_shapes_pil(ImageDraw.Draw(image), view, geometry,
                      overlay_shapes(overlay, geometry.cell_m))


def draw_background_pil(image, view: MapView, geometry: GridGeometry,
                        background: Background) -> None:
    """Paint a background's registration marks and label onto a PIL image.

    The marks and the label are described once in map metres and one string;
    this and the Tk backend in :meth:`MapPane._draw_overlay` are two thin
    paintings of the same description, which is the rule that keeps a
    report's image and the screen it came from identical.
    """
    from PIL import ImageDraw

    draw = ImageDraw.Draw(image)
    _paint_shapes_pil(draw, view, geometry,
                      background_shapes(background, geometry.cell_m))
    if background.label():
        draw.text((4, 4), background.label(), fill=theme.DIM)


def snapshot_image(grid: EvidenceGrid | None, *, mode: str = 'occupancy', layer: int = 0,
                   sim_now_s: float = 0.0, horizon_s: float = DEFAULT_STALENESS_HORIZON_S,
                   cost: np.ndarray | None = None, overlay: Overlay | None = None,
                   cell_px: int = SNAPSHOT_CELL_PX, never_observed=None,
                   background: Background | None = None):
    """The pane's rendering as a PIL image, for a report or a test.

    The same raster, the same background and the same overlay shapes the
    window paints, so what a report files and what an operator saw are one
    rendering rather than two that agree. The background is what the caller
    was displaying -- frame-checked, host-supplied -- and missing imagery is
    simply a ``None`` that renders exactly the evidence-only picture.
    """
    from PIL import Image

    raster = render_raster(grid, mode=mode, layer=layer, sim_now_s=sim_now_s,
                           horizon_s=horizon_s, cost=cost, never_observed=never_observed)
    if raster is None: return None
    mask = never_cells(grid, mode=mode, layer=layer, never_observed=never_observed)
    if background is not None and mask is not None:
        raster = compose_background(raster, background, grid.geometry, mask)
    cell_px = max(1, int(cell_px))
    image = Image.fromarray(raster).resize(
        (grid.geometry.width * cell_px, grid.geometry.height * cell_px),
        Image.Resampling.NEAREST)
    if overlay is not None or background is not None:
        view = MapView(cell=cell_px / grid.geometry.cell_m, margin=0)
        view.offset_x = view.offset_y = 0.
        if background is not None:
            draw_background_pil(image, view, grid.geometry, background)
        if overlay is not None:
            draw_overlay_pil(image, view, grid.geometry, overlay)
    return image


# -- editing support -----------------------------------------------------------

#: Most cells a fallback extent is divided into along its longer side. The
#: blank grid behind it only has to carry a transform and read as unobserved,
#: so it stays coarse however large the extent is.
FALLBACK_CELLS = 200

#: How near, in screen pixels, a press must land to pick up a handle.
HANDLE_RADIUS_PX = 8.0


def reference_footprint(image) -> tuple[float, float, float, float]:
    """The map-metre bounding box a reference image covers, from its affine."""
    from dcmn.imagery import affine_xy

    corners = [affine_xy(image.affine, col, row)
               for col in (-.5, image.width - .5) for row in (-.5, image.height - .5)]
    xs, ys = [c[0] for c in corners], [c[1] for c in corners]
    return (min(xs), min(ys), max(xs), max(ys))


def choose_fallback_extent(*, area_bounds=None, workspace=None, reference_bounds=None,
                           pose=None, min_side_m: float = 100.0) -> tuple[tuple[float, ...] | None, str]:
    """Where a pane with no evidence yet lets an operator point, and why there.

    In order: the mission's own area or workspace -- what the operator is
    about to work in; the reference image, only while it is displayed; a
    square of dalg's default size around the vehicle. ``(None, reason)``
    when none of them is known, and the pane stays blank.
    """
    for bounds, label in ((area_bounds, 'mission area'), (workspace, 'profile workspace'),
                          (reference_bounds, 'reference image')):
        if bounds is not None and bounds[2] > bounds[0] and bounds[3] > bounds[1]:
            return tuple(float(v) for v in bounds), label
    if pose is not None:
        half = float(min_side_m) / 2.
        x, y = float(pose[0]), float(pose[1])
        return (x - half, y - half, x + half, y + half), f'{min_side_m:g} m around the vehicle'
    return None, 'no evidence, mission area, workspace, reference image or pose yet'


def fallback_grid(bounds) -> EvidenceGrid:
    """A never-observed grid covering ``bounds``: a transform and nothing else."""
    x0, y0, x1, y1 = (float(v) for v in bounds)
    side = max(x1 - x0, y1 - y0)
    cell = max(side / FALLBACK_CELLS, 0.05)
    geometry = GridGeometry.from_extent(x1 - x0, y1 - y0, cell, origin_x_m=x0, origin_y_m=y0)
    return EvidenceGrid.blank(geometry)


def nearest_handle(point_m, handles, radius_m: float):
    """The key of the handle nearest ``point_m`` within ``radius_m``, or None.

    ``handles`` is ``(key, x_m, y_m)`` triples. Pure, so an editor's hit test
    can be checked without a window.
    """
    best, best_d = None, float(radius_m)
    for key, x, y in handles:
        d = math.hypot(float(x) - point_m[0], float(y) - point_m[1])
        if d <= best_d: best, best_d = key, d
    return best


# -- the widget --------------------------------------------------------------

@dataclass
class PaneState:
    """What the pane is currently showing. One object so a host can log it."""
    mode: str = 'occupancy'
    layer: int = 0
    sim_now_s: float = 0.0
    horizon_s: float = DEFAULT_STALENESS_HORIZON_S
    grid: EvidenceGrid | None = None
    cost: np.ndarray | None = None
    never_observed: np.ndarray | None = None
    overlay: Overlay = field(default_factory=Overlay)
    background: Background | None = None
    #: The never-observed stand-in drawn while there is no grid, and why that extent.
    fallback: EvidenceGrid | None = None
    fallback_label: str = ''

    @property
    def shown(self) -> EvidenceGrid | None:
        """The grid the pane draws: the evidence, or its fallback stand-in."""
        return self.grid if self.grid is not None else self.fallback


class MapPane:
    """A grid, its overlays, a hover readout and a click callback.

    Owns no data and no clock: a host sets a grid and a simulated time and
    calls :meth:`refresh`, which repaints at most at ``MAP_HZ`` because a
    window is not part of the flight.

    With ``zoom=True`` the operator can also scroll to zoom about the pointer
    and drag to pan, and a stippled **Reset view** badge appears in the
    image's top-right corner whenever the view is not the fitted one. The zoom
    is a property of the *pane*, never of the data: :meth:`snapshot` renders
    the whole grid at its own scale, so what a report contains does not depend
    on where an operator happened to be looking. Left-drag and right-click
    stay with the host's editing callbacks; panning is the middle button, or
    control with the left, neither of which any editor uses. A pane that took
    no pointer callbacks at all pans with the plain left button as well.
    """

    def __init__(self, parent, *, on_click: Callable[[float, float], None] | None = None,
                 mode: str = 'occupancy', hz: float = MAP_HZ, controls: bool = True,
                 title: str = '', width: int = 320, height: int = 320,
                 zoom: bool = False,
                 on_press: Callable[[float, float, frozenset], None] | None = None,
                 on_drag: Callable[[float, float, frozenset], None] | None = None,
                 on_release: Callable[[float, float, frozenset], None] | None = None,
                 on_context: Callable[[float, float, frozenset], None] | None = None,
                 on_key: Callable[[str, tuple[float, float] | None], None] | None = None) -> None:
        import tkinter as tk
        from tkinter import ttk
        from PIL import ImageTk

        if mode not in MODES: raise ValueError(f'unknown map display mode {mode!r}')
        self.tk, self.ttk, self.ImageTk = tk, ttk, ImageTk
        self.state = PaneState(mode=mode)
        self.on_click = on_click
        # Editing callbacks, each given map metres and the held modifiers
        # ('shift', 'control'); a key callback gets the key and the last
        # pointer position over the map, if any.
        self.on_press, self.on_drag, self.on_release = on_press, on_drag, on_release
        self.on_context, self.on_key = on_context, on_key
        self._pointer: tuple[float, float] | None = None
        self.view = MapView(cell=8., margin=0)
        self.photo = None
        self._paint = Paced(hz)
        self._raster_key: tuple | None = None
        self._image = None
        #: Bumped whenever ``_image`` is rebuilt, so the scaled photo below can
        #: be cached across repaints that only moved the view or the overlay.
        self._image_serial = 0
        self._photo_key: tuple | None = None
        self._resize_after: str | None = None
        self._notice = ''
        # The view on top of the fit: a zoom factor, and a pan in pixels away
        # from the fitted, centred position. Both are the operator's, so
        # neither survives into a snapshot or a report.
        self.zoomable = bool(zoom)
        # A pane nobody clicks, edits or keys has no other use for the left
        # button, so it pans with that too: on a map that is only ever looked
        # at -- dway's live map, a replay -- dragging is the gesture people try
        # first. Where a host has claimed the button (dmsn's editors, dnav's
        # click-to-set-goal) it keeps every gesture it had, and panning is the
        # middle button or control.
        self._left_pans = self.zoomable and not any((on_click, on_press, on_drag, on_release))
        self._zoom = 1.0
        self._pan = (0.0, 0.0)
        #: Where a pan drag started: ``(event x, event y, pan at press)``.
        self._pan_from: tuple[float, float, tuple[float, float]] | None = None
        #: Screen pixels per cell at the fitted scale, set by the next fit.
        self._fitted_cell_px: float | None = None
        #: The reset badge's canvas rectangle while it is on screen.
        self._badge: tuple[float, float, float, float] | None = None
        self._press_on_badge = False
        self.prompt = HOVER_PROMPT + ((ZOOM_HINT_LEFT if self._left_pans else ZOOM_HINT)
                                      if self.zoomable else '')

        self.widget = ttk.Frame(parent, style='Panel.TFrame')
        self.header = tk.StringVar(value=title)
        self.readout = tk.StringVar(value=self.prompt)
        if title or controls:
            bar = ttk.Frame(self.widget, style='Panel.TFrame')
            bar.pack(fill='x', padx=4, pady=(4, 0))
            ttk.Label(bar, textvariable=self.header, style='Brand.TLabel').pack(side='left')
            if controls:
                self.mode_var = tk.StringVar(value=mode)
                combo = ttk.Combobox(bar, textvariable=self.mode_var, values=MODES,
                                     state='readonly', width=10)
                combo.pack(side='right')
                combo.bind('<<ComboboxSelected>>',
                           lambda _e: self.set_mode(self.mode_var.get()))
        self.canvas = tk.Canvas(self.widget, width=width, height=height,
                                background=theme.CANVAS, highlightthickness=1,
                                highlightbackground=theme.GRID,
                                highlightcolor=theme.ACCENT)
        self.canvas.pack(fill='both', expand=True, padx=4, pady=4)
        self.item = self.canvas.create_image(0, 0, anchor='nw')
        ttk.Label(self.widget, textvariable=self.readout, style='Dim.TLabel',
                  anchor='w').pack(fill='x', padx=6, pady=(0, 4))
        self.canvas.bind('<Motion>', self._hover)
        self.canvas.bind('<Leave>', lambda _e: self.readout.set(self._notice or self.prompt))
        self.canvas.bind('<Button-1>', self._click)
        self.canvas.bind('<B1-Motion>', lambda e: self._pointer_event(self.on_drag, e))
        self.canvas.bind('<ButtonRelease-1>', self._release)
        self.canvas.bind('<Button-3>', lambda e: self._pointer_event(self.on_context, e))
        self.canvas.bind('<Key>', self._key)
        self.canvas.bind('<Configure>', self._resized)
        if self.zoomable:
            # X11 reports the wheel as buttons 4 and 5; every other platform
            # sends a signed delta. Each handler breaks, so an ancestor's
            # bind_all scroller never also scrolls the page under the map.
            for sequence in ('<MouseWheel>', '<Button-4>', '<Button-5>'):
                self.canvas.bind(sequence, self._wheel)
            # The middle button, and control with the left for a mouse that
            # has no middle button. A more specific sequence wins over the
            # plain one on the same widget, so the editing bindings above keep
            # every gesture they had.
            gestures = [('<Button-2>', '<B2-Motion>', '<ButtonRelease-2>'),
                        ('<Control-Button-1>', '<Control-B1-Motion>', '<Control-ButtonRelease-1>')]
            if self._left_pans:
                gestures.append(('<Button-1>', '<B1-Motion>', '<ButtonRelease-1>'))
            for press, motion, release in gestures:
                self.canvas.bind(press, self._pan_press)
                self.canvas.bind(motion, self._pan_move)
                self.canvas.bind(release, self._pan_release)
            self.canvas.tag_bind('viewreset', '<Enter>',
                                 lambda _e: self.canvas.configure(cursor='hand2'))
            self.canvas.tag_bind('viewreset', '<Leave>',
                                 lambda _e: self.canvas.configure(cursor=''))

    # -- inputs ------------------------------------------------------------

    def set_mode(self, mode: str, *, refresh: bool = True) -> None:
        """Change what the pane shows.

        ``refresh=False`` is for a host that is about to change the grid or
        cost as well: repainting here would draw the new mode over the old
        data, and spend the frame budget the host's own repaint needed.
        """
        if mode not in MODES: raise ValueError(f'unknown map display mode {mode!r}')
        changed = mode != self.state.mode
        self.state.mode = mode
        if refresh and changed: self.refresh(force=True)

    def set_layer(self, layer: int) -> None:
        self.state.layer = max(0, int(layer))
        self.refresh(force=True)

    def set_grid(self, grid: EvidenceGrid | None, *, sim_now_s: float | None = None,
                 horizon_s: float | None = None) -> None:
        self.state.grid = grid
        if sim_now_s is not None: self.state.sim_now_s = float(sim_now_s)
        if horizon_s is not None: self.state.horizon_s = float(horizon_s)

    def set_sim_time(self, sim_now_s: float) -> None:
        self.state.sim_now_s = float(sim_now_s)

    def set_cost(self, cost, *, never_observed=None) -> None:
        self.state.cost = None if cost is None else np.asarray(cost)
        self.state.never_observed = None if never_observed is None else np.asarray(never_observed, bool)

    def set_overlay(self, **fields: Any) -> None:
        self.state.overlay = replace(self.state.overlay, **fields)

    def set_background(self, background: Background | None, *,
                       refresh: bool = True) -> None:
        """Set the reference background, or clear it with None.

        The host has already frame-checked the image -- this widget takes a
        placement, not a discovery -- and clearing it is the one thing that
        ever happens to a background here, because an image that stopped
        matching the vehicle's frame is not this pane's decision to keep.
        """
        self.state.background = background
        if refresh: self.refresh(force=True)

    def set_header(self, text: str) -> None:
        self.header.set(text)

    def set_fallback_extent(self, bounds, label: str = '') -> None:
        """The extent to draw and accept pointer input over while there is no grid.

        Evidence always wins once it exists; the fallback only lets an
        operator place a goal or a vertex before dalg has published. ``None``
        clears it. The label says which extent it is (see
        :func:`choose_fallback_extent`) and is shown in the readout.
        """
        current = self.state.fallback
        if bounds is None:
            self.state.fallback, self.state.fallback_label = None, label
            return
        wanted = tuple(round(float(v), 6) for v in bounds)
        if current is None or tuple(round(v, 6) for v in current.geometry.bounds_m()) != wanted:
            self.state.fallback = fallback_grid(bounds)
        self.state.fallback_label = label

    def metres_per_pixel(self) -> float | None:
        """The current scale, for converting a pixel radius into map metres."""
        if self._fit() is None: return None
        return 1.0 / max(self.view.cell, 1e-9)

    def handle_at(self, x_m: float, y_m: float, handles, *, radius_px: float = HANDLE_RADIUS_PX):
        """The key of the handle under a pointer at map metres, or None."""
        scale = self.metres_per_pixel()
        if scale is None: return None
        return nearest_handle((x_m, y_m), handles, radius_px * scale)

    # -- geometry ----------------------------------------------------------

    def _fit(self) -> GridGeometry | None:
        """Size the transform to the canvas as it is right now.

        Tk reports a width of 1 until a widget has been mapped, and fitting a
        whole extent into one pixel puts every metre a hundred times where it
        belongs -- so a pane asked for a coordinate before its first
        ``<Configure>`` falls back to the size it asked for.
        """
        grid = self.state.shown
        if grid is None: return None
        width, height = self.canvas_size()
        self.view.fit_canvas(_Extent(grid.geometry), width, height)
        # Screen pixels per cell at the fitted scale: the zoom ceiling is a
        # multiple of this, so it follows the grid and the canvas rather than
        # being a constant that is wrong for both.
        self._fitted_cell_px = self.view.cell * grid.geometry.cell_m
        if self.zoomable and not self.default_view:
            self._apply_view(grid.geometry, width, height)
        return grid.geometry

    def canvas_size(self) -> tuple[int, int]:
        """The canvas as it is right now, falling back to the size it asked for."""
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        if width <= 1: width = max(1, self.canvas.winfo_reqwidth())
        if height <= 1: height = max(1, self.canvas.winfo_reqheight())
        return width, height

    # -- the operator's view ------------------------------------------------

    @property
    def default_view(self) -> bool:
        """Whether the pane is showing the fitted, centred view it starts in."""
        return self._zoom == 1.0 and self._pan == (0.0, 0.0)

    @property
    def zoom(self) -> float:
        return self._zoom

    @property
    def max_zoom(self) -> float:
        """How far this grid is worth zooming: until one cell fills MAX_CELL_PX.

        A property of the grid and the canvas, not a constant: a coarse grid
        runs out of detail at a few times the fit, a fine one has further to
        go. Before the first fit there is no grid to answer for, and the fitted
        view is the only one on offer.
        """
        if not self._fitted_cell_px: return MIN_ZOOM
        return max(MIN_ZOOM, min(MAX_ZOOM, MAX_CELL_PX / self._fitted_cell_px))

    def _apply_view(self, geometry: GridGeometry, width: int, height: int) -> None:
        """Scale the fitted transform about the canvas centre, then pan it.

        Composed on top of :meth:`MapView.fit_canvas` rather than replacing it,
        so a resize, a new grid and a change of extent all keep working exactly
        as they did and the zoom rides on whatever they produce.
        """
        # The fit may have changed under the zoom -- a resize, a new extent --
        # so the ceiling is re-applied here rather than only where a gesture
        # asked for it.
        self._zoom = max(MIN_ZOOM, min(self.max_zoom, self._zoom))
        extent_w, extent_h = geometry.extent_m
        self._pan = self._clamp_pan(self._pan, extent_w * self.view.cell * self._zoom,
                                    extent_h * self.view.cell * self._zoom, width, height)
        centre_x, centre_y = width / 2.0, height / 2.0
        self.view.offset_x = centre_x + self._zoom * (self.view.offset_x - centre_x) + self._pan[0]
        self.view.offset_y = centre_y + self._zoom * (self.view.offset_y - centre_y) + self._pan[1]
        self.view.cell *= self._zoom

    @staticmethod
    def _clamp_pan(pan, drawn_w: float, drawn_h: float, width: int, height: int):
        """A pan that cannot drag the image off its own canvas.

        The image may be moved only as far as its own edges: at the fitted zoom
        it is contained, so there is nothing to pan and the clamp returns the
        centred view -- which is what makes "not the default view" the same
        question as "is there anything to reset".
        """
        limit_x = max(0.0, (drawn_w - width) / 2.0)
        limit_y = max(0.0, (drawn_h - height) / 2.0)
        return (min(limit_x, max(-limit_x, pan[0])), min(limit_y, max(-limit_y, pan[1])))

    def zoom_at(self, px: float, py: float, factor: float) -> None:
        """Zoom by ``factor`` about the canvas point ``(px, py)``.

        The map point under the pointer stays under the pointer, which is the
        one property that makes a wheel zoom feel like a magnifier rather than
        a jump.
        """
        if not self.zoomable or self._fit() is None: return
        previous = self._zoom
        self._zoom = max(MIN_ZOOM, min(self.max_zoom, previous * float(factor)))
        if self._zoom == previous: return
        ratio = self._zoom / previous
        width, height = self.canvas_size()
        # The pointer's offset from the centre, scaled about the pointer: the
        # inverse of _apply_view, solved for the pan that keeps px, py fixed.
        offset_x, offset_y = px - width / 2.0, py - height / 2.0
        self._pan = (offset_x - ratio * (offset_x - self._pan[0]),
                     offset_y - ratio * (offset_y - self._pan[1]))
        self.refresh(force=True)

    def pan_by(self, dx: float, dy: float) -> None:
        """Move the view by a pixel offset, clamped to the image's own edges."""
        if not self.zoomable: return
        self._pan = (self._pan[0] + float(dx), self._pan[1] + float(dy))
        self.refresh(force=True)

    def reset_view(self) -> None:
        """Back to the fitted, centred view. Idempotent, and cheap when it is a no-op."""
        if not self.zoomable or self.default_view: return
        self._zoom, self._pan, self._pan_from = 1.0, (0.0, 0.0), None
        self.refresh(force=True)

    def to_map(self, px: float, py: float) -> tuple[float, float] | None:
        """Canvas pixels to absolute map metres -- the goal picker's whole job.

        Fitted first, so a click that lands between a resize and the next
        repaint is answered against the canvas the operator actually clicked.
        """
        geometry = self._fit()
        if geometry is None: return None
        return unproject(self.view, geometry, px, py)

    # -- painting ----------------------------------------------------------

    def _resized(self, _event) -> None:
        """Repaint after a resize, at the cap and exactly once at the end.

        Tk fires ``<Configure>`` continuously while a window is dragged.
        Repainting on each one would put a full rescale and image upload
        outside the repaint budget, which is the one thing the budget exists to
        prevent; skipping the last one would leave the pane the wrong size
        until something else happened to repaint it. So: a paced repaint now,
        and one guaranteed repaint a frame after the drag stops.
        """
        self.refresh()
        if self._resize_after is not None:
            self.canvas.after_cancel(self._resize_after)
        self._resize_after = self.canvas.after(
            max(1, round(self._paint.period_s * 1000.)), self._settled)

    def _settled(self) -> None:
        self._resize_after = None
        self.refresh(force=True)

    def refresh(self, *, force: bool = False) -> bool:
        """Repaint if the budget allows. Returns whether anything was drawn.

        ``force`` spends the budget as well as ignoring it, so a forced repaint
        cannot be followed a millisecond later by a paced one.
        """
        due = self._paint.due()
        if not force and not due: return False
        from PIL import Image

        geometry = self._fit()
        self.canvas.delete('overlay')
        self._badge = None
        if geometry is None:
            self.canvas.itemconfigure(self.item, image='')
            self.photo = None
            return True
        state = self.state
        shown = state.shown
        # The age mode's raster changes with the clock even when the grid does
        # not, so simulated time is part of what identifies a rendering; a
        # background is part of it too, identified by revision and checksum so
        # a replacement image is a new rendering even at the same resolution.
        key = (id(shown), shown.revision, state.mode, state.layer,
               round(state.sim_now_s, 2) if state.mode == 'age' else 0,
               None if state.cost is None else state.cost.tobytes(),
               None if state.never_observed is None else state.never_observed.tobytes(),
               None if state.background is None else state.background.identity)
        # A notice belongs to the state that produced it; leaving one up after
        # the pane has painted something says the wrong thing about what is on
        # screen.
        if self._notice:
            self._notice = ''
            self.readout.set(self.prompt)
        # A fallback stand-in has no cost layer, whatever mode the host set.
        mode = state.mode if state.grid is not None else 'occupancy'
        cost = state.cost if state.grid is not None else None
        never = state.never_observed if state.grid is not None else None
        if key != self._raster_key or self._image is None:
            raster = render_raster(shown, mode=mode, layer=state.layer,
                                   sim_now_s=state.sim_now_s,
                                   horizon_s=state.horizon_s, cost=cost,
                                   never_observed=never)
            if raster is not None and state.background is not None:
                mask = never_cells(shown, mode=mode, layer=state.layer,
                                   never_observed=never)
                if mask is not None:
                    raster = compose_background(raster, state.background,
                                                shown.geometry, mask)
            self._raster_key = key
            self._image = None if raster is None else Image.fromarray(raster)
            self._image_serial += 1
        if self._image is None:
            self.canvas.itemconfigure(self.item, image='')
            self.photo = None
            self._notice = 'no cost layer for this source'
            self.readout.set(self._notice)
            return True
        # Only the part of the raster the canvas can actually show is scaled
        # and uploaded. Scaling the whole thing costs the square of the zoom --
        # a 250x125 cell grid at 17x is an 8194x4032 image, 33 megapixels of
        # which a 480x480 canvas displays 0.7% -- and at that size building one
        # per repaint is what turns a zoomed pane into a slideshow. The crop is
        # snapped out to whole cells, so every drawn pixel still belongs to
        # exactly one measured cell, and at the fitted view it is the whole
        # image again: an unzoomed pane draws exactly what it always did.
        cell_px = self.view.cell * geometry.cell_m
        width_px, height_px = self.canvas_size()
        col0 = max(0, int(math.floor(-self.view.offset_x / cell_px)))
        row0 = max(0, int(math.floor(-self.view.offset_y / cell_px)))
        col1 = min(geometry.width, int(math.ceil((width_px - self.view.offset_x) / cell_px)))
        row1 = min(geometry.height, int(math.ceil((height_px - self.view.offset_y) / cell_px)))
        if col1 <= col0 or row1 <= row0:
            # Nothing of it is on the canvas: reachable only in the middle of a
            # resize, and drawing the overlay is still right.
            self.canvas.itemconfigure(self.item, image='')
            self.photo, self._photo_key = None, None
            self._draw_overlay(geometry)
            return True
        box = (col0, row0, col1, row1)
        size = (max(1, round((col1 - col0) * cell_px)), max(1, round((row1 - row0) * cell_px)))
        # Rescaling is the expensive half of a repaint, so the scaled tile is
        # kept until the raster, the visible cells or the scale actually change.
        photo_key = (self._image_serial, box, size)
        if self.photo is None or photo_key != self._photo_key:
            tile = self._image if box == (0, 0, geometry.width, geometry.height) else self._image.crop(box)
            # NEAREST: every pixel of a measured cell has to stay that cell, or
            # the pane invents evidence at the boundaries.
            self.photo = self.ImageTk.PhotoImage(tile.resize(size, Image.Resampling.NEAREST))
            self._photo_key = photo_key
            self.canvas.itemconfigure(self.item, image=self.photo)
        self.canvas.coords(self.item, self.view.offset_x + col0 * cell_px,
                           self.view.offset_y + row0 * cell_px)
        self._draw_overlay(geometry)
        return True

    def _paint_shapes_tk(self, geometry: GridGeometry, shapes) -> None:
        for shape in shapes:
            x0, y0 = project(self.view, geometry, shape.x0, shape.y0)
            x1, y1 = project(self.view, geometry, shape.x1, shape.y1)
            if shape.kind == 'polygon':
                flat = [c for x, y in shape.points for c in project(self.view, geometry, x, y)]
                self.canvas.create_polygon(*flat, fill=shape.fill or '', outline=shape.outline or '',
                                           width=shape.width, tags='overlay')
            elif shape.kind == 'text':
                self.canvas.create_text(x0, y0, text=shape.text, anchor='center',
                                        fill=shape.outline or shape.fill or theme.TEXT, tags='overlay')
            elif shape.kind == 'rect':
                self.canvas.create_rectangle(x0, y0, x1, y1, fill=shape.fill or '',
                                             outline=shape.outline or '',
                                             width=shape.width, tags='overlay')
            elif shape.kind == 'oval':
                self.canvas.create_oval(x0, y0, x1, y1, fill=shape.fill or '',
                                        outline=shape.outline or '',
                                        width=shape.width, tags='overlay')
            else:
                self.canvas.create_line(x0, y0, x1, y1, fill=shape.outline,
                                        width=shape.width, tags='overlay',
                                        **({'dash': shape.dash} if shape.dash else {}))

    def _draw_overlay(self, geometry: GridGeometry) -> None:
        background = self.state.background
        if background is not None:
            # Registration marks first, so a route reads over them; and the
            # label, so what the picture is stays in front of the operator.
            self._paint_shapes_tk(geometry, background_shapes(background, geometry.cell_m))
            if background.label():
                # Clamped into the canvas: zoomed in, the image's own corner is
                # off screen, and a label nobody can see says nothing.
                self.canvas.create_text(max(4.0, self.view.offset_x + 4),
                                        max(2.0, self.view.offset_y + 2),
                                        text=background.label(), anchor='nw',
                                        fill=theme.DIM, tags='overlay')
        self._paint_shapes_tk(geometry, overlay_shapes(self.state.overlay, geometry.cell_m))
        if self.zoomable and not self.default_view: self._draw_reset_badge(geometry)

    def _draw_reset_badge(self, geometry: GridGeometry) -> None:
        """A translucent Reset view button in the image's top-right corner.

        Only while the view is not the fitted one: a control that does nothing
        is a control an operator has to think about. Stippled over whatever it
        covers, because a Tk canvas has no alpha -- and placed inside both the
        image and the canvas, so it is on the picture when the picture is
        letterboxed and in the corner when the picture is larger than the pane.
        """
        width, height = self.canvas_size()
        extent_w, extent_h = geometry.extent_m
        right = min(float(width), self.view.offset_x + extent_w * self.view.cell) - BADGE_INSET_PX
        top = max(0.0, self.view.offset_y) + BADGE_INSET_PX
        label = f'Reset view ({self._zoom:.1f}x)'
        text = self.canvas.create_text(right - BADGE_PAD_PX, top + BADGE_PAD_PX, text=label,
                                       anchor='ne', fill=theme.TEXT, tags=('overlay', 'viewreset'))
        bounds = self.canvas.bbox(text)
        if bounds is None: return
        x0, y0, x1, y1 = bounds
        box = (x0 - BADGE_PAD_PX, y0 - BADGE_PAD_PX, x1 + BADGE_PAD_PX, y1 + BADGE_PAD_PX)
        plate = self.canvas.create_rectangle(*box, fill=theme.PANEL, stipple='gray50',
                                             outline=theme.GRID, width=1.,
                                             tags=('overlay', 'viewreset'))
        self.canvas.tag_lower(plate, text)
        self._badge = box

    def snapshot(self, *, cell_px: int = SNAPSHOT_CELL_PX):
        """The same rendering as a PIL image, for a report."""
        evidence = self.state.grid is not None
        return snapshot_image(self.state.shown, mode=self.state.mode if evidence else 'occupancy',
                              layer=self.state.layer, sim_now_s=self.state.sim_now_s,
                              horizon_s=self.state.horizon_s, cost=self.state.cost if evidence else None,
                              overlay=self.state.overlay, cell_px=cell_px,
                              never_observed=self.state.never_observed if evidence else None,
                              background=self.state.background)

    # -- interaction -------------------------------------------------------

    def describe(self, x_m: float, y_m: float) -> str:
        """The hover readout: where, how likely, and how long ago."""
        grid = self.state.grid
        if grid is None:
            if self.state.fallback is None: return 'no grid'
            label = self.state.fallback_label
            return f'{x_m:.2f}, {y_m:.2f} m  no evidence yet' + (f' ({label})' if label else '')
        cell = grid.geometry.to_cell(x_m, y_m)
        if cell is None: return f'{x_m:.2f}, {y_m:.2f} m  outside the grid'
        col, row = cell
        probability = grid.probability_at(x_m, y_m, self.state.layer)
        observed = grid.observed_at(x_m, y_m, self.state.layer)
        where = f'{x_m:.2f}, {y_m:.2f} m  cell {col},{row}'
        if self.state.mode == 'cost' and self.state.cost is not None:
            cost = np.asarray(self.state.cost)
            value = float(cost[self.state.layer, row, col] if cost.ndim == 3 else cost[row, col])
            if not np.isfinite(value): return f'{where}  blocked (inflated cost)'
            never = (probability is None if self.state.never_observed is None else
                     bool(self.state.never_observed[row, col]))
            return f'{where}  cost={value:g}' + ('  never observed' if never else '')
        suffix = ''
        if self.state.mode == 'blocked' and self.state.cost is not None:
            cost = np.asarray(self.state.cost)
            value = float(cost[self.state.layer, row, col] if cost.ndim == 3 else cost[row, col])
            if not np.isfinite(value): suffix = '  · blocked by the policy'
        if probability is None: return f'{where}  never observed{suffix}'
        age = max(0., self.state.sim_now_s - observed)
        return (f'{where}  p={probability:.2f}  observed {observed:.2f}s '
                f'({age:.1f}s ago){suffix}')

    def _hover(self, event) -> None:
        point = self.to_map(event.x, event.y)
        self._pointer = point
        self.readout.set((self.state.fallback_label or 'no grid') if point is None
                         else self.describe(*point))

    @staticmethod
    def modifiers(event) -> frozenset:
        """The held modifiers of a Tk event, by name."""
        state = int(getattr(event, 'state', 0) or 0)
        return frozenset(name for name, bit in (('shift', 0x1), ('control', 0x4)) if state & bit)

    def _pointer_event(self, callback, event) -> None:
        if self._press_on_badge: return
        point = self.to_map(event.x, event.y)
        self._pointer = point
        if point is not None and callback is not None: callback(point[0], point[1], self.modifiers(event))

    def _click(self, event) -> None:
        # Focus follows a press, so Delete and Escape reach the editor the
        # operator is pointing at.
        try: self.canvas.focus_set()
        except Exception: pass
        # The badge is a control drawn on the map, so a press on it is not a
        # press on the map: an editor must never get a vertex out of it.
        self._press_on_badge = self._hit_badge(event.x, event.y)
        if self._press_on_badge:
            self.reset_view()
            return
        point = self.to_map(event.x, event.y)
        self._pointer = point
        if point is None: return
        if self.on_press is not None: self.on_press(point[0], point[1], self.modifiers(event))
        if self.on_click is not None: self.on_click(*point)

    def _release(self, event) -> None:
        self._pointer_event(self.on_release, event)
        self._press_on_badge = False
        # Any button coming up ends a pan, so control let go mid-drag cannot
        # leave one half-started for the next gesture to inherit.
        self._pan_from = None

    def _key(self, event) -> None:
        if self.on_key is not None: self.on_key(str(event.keysym), self._pointer)

    # -- zooming and panning -----------------------------------------------

    def _hit_badge(self, px: float, py: float) -> bool:
        """Whether a canvas point is on the reset badge, when one is showing."""
        if self._badge is None: return False
        x0, y0, x1, y1 = self._badge
        return x0 <= px <= x1 and y0 <= py <= y1

    def _wheel(self, event) -> str:
        """One notch of the wheel, whichever way the platform reports it."""
        from dcmn.scroll import wheel_step
        step = wheel_step(event)          # negative is up, which is closer in
        if step: self.zoom_at(event.x, event.y, ZOOM_STEP ** -step)
        return 'break'

    def _pan_press(self, event) -> str:
        # Reached with the left button on a viewing-only pane, so the badge is
        # checked here too rather than only in _click.
        if self._hit_badge(event.x, event.y):
            self._press_on_badge = True
            self.reset_view()
            return 'break'
        self._press_on_badge = False
        self._pan_from = (float(event.x), float(event.y), self._pan)
        return 'break'

    def _pan_move(self, event) -> str | None:
        if self._pan_from is None:
            # Control pressed part-way through somebody else's drag: that drag
            # is still theirs.
            self._pointer_event(self.on_drag, event)
            return None
        x, y, pan = self._pan_from
        self._pan = (pan[0] + (event.x - x), pan[1] + (event.y - y))
        self.refresh(force=True)
        return 'break'

    def _pan_release(self, event) -> str | None:
        if self._pan_from is None:
            self._release(event)
            return None
        self._pan_from = None
        self._press_on_badge = False
        return 'break'


# -- the opt-in background host -------------------------------------------------

class ReferenceBackgroundHost:
    """The toggle, the opacity knob and the discovery behind a background.

    One object per window: it owns the imagery session, applies the
    frame/epoch check, and produces the :class:`Background` the window's
    panes display. The toggle starts **off** and never defaults on -- a
    truth map behind live evidence is a display decision an operator makes
    -- keep truth imagery out of the operational view or label the session --
    which is why the host reports every change through
    ``on_change`` and the app records it in its session provenance.

    The host never invents a background: no imagery plane, no image, or a
    frame the vehicle is no longer in all come back as ``None``, and the
    panes render exactly the evidence-only picture.
    """

    #: What the toggle says. The word "debug" is deliberate: an operator
    #: enabling it should read the session as assisted, and the label is
    #: where that reading starts.
    TOGGLE_TEXT = 'reference image (debug)'

    def __init__(self, parent, instance: str, *, enabled: bool = False,
                 opacity: float = DEFAULT_BACKGROUND_OPACITY,
                 on_change: Callable[[bool, float], None] | None = None) -> None:
        import tkinter as tk
        from tkinter import ttk
        from dcmn.context import Context
        from dcmn.imagery import ImageSession

        self.instance = instance
        self.enabled = bool(enabled)
        self.opacity = min(1., max(0., float(opacity)))
        self.on_change = on_change
        self.session = ImageSession(instance)
        self.context = Context(instance)
        self.tk, self.ttk = tk, ttk

        self.widget = ttk.Frame(parent)
        self.var = tk.BooleanVar(value=self.enabled)
        ttk.Checkbutton(self.widget, text=self.TOGGLE_TEXT, variable=self.var,
                        command=self._toggled).pack(side='left', padx=(6, 8))
        ttk.Label(self.widget, text='opacity', style='Dim.TLabel').pack(side='left')
        self.scale_var = tk.DoubleVar(value=self.opacity)
        self.scale = ttk.Scale(self.widget, from_=0., to=1.,
                               variable=self.scale_var, command=self._opacity)
        self.scale.pack(side='left', fill='x', expand=True, padx=6)

    def _toggled(self) -> None:
        self.enabled = bool(self.var.get())
        if self.on_change is not None:
            self.on_change(self.enabled, self.opacity)

    def _opacity(self, value) -> None:
        self.opacity = min(1., max(0., float(value)))
        # A drag fires this continuously; the panes repaint cheaply off the
        # change, and provenance records the mode, not every pixel of a knob.

    def background(self) -> Background | None:
        """The frame-checked background the panes should show right now.

        The newest image that still matches the context snapshot, at the
        operator's opacity -- or ``None``, which renders without a
        background and is what imagery absence, corruption and frame
        mismatch all look like. The first matching image of the plane wins
        when a provider ever publishes several; one world image is all this
        repository's providers have.
        """
        if not self.enabled: return None
        self.session.poll()
        try:
            snapshot = self.context.read()
        except ValueError:
            snapshot = {}                     # an unreadable context matches nothing
        for image_id in sorted(self.session.states):
            reference = self.session.display(image_id, snapshot)
            if reference is not None:
                return Background(reference, opacity=self.opacity)
        return None

    def close(self) -> None:
        self.session.close()


# -- the minimal host window -------------------------------------------------

class MapPaneWindow:
    """One pane per published source, and nothing else.

    The smallest thing that makes the widget reviewable on its own: it is not
    `dalg`'s Grids tab and must not grow into one -- that tab arrives with the
    first real sensor source, and it will be this widget in a layout rather
    than a second copy of it.
    """

    def __init__(self, instance: str, *, source: str | None = None,
                 mode: str = 'occupancy', background: bool = False) -> None:
        import tkinter as tk
        from tkinter import ttk
        from dcmn.context import Context
        from dcmn.imagery import ImageSession
        from dcmn.maps import MapSession, status_clock
        from dcmn.tktheme import apply_theme
        from dcmn.window import restore_window_geometry, save_window_geometry

        self.tk, self.ttk = tk, ttk
        self.instance, self.only, self.mode = instance, source, mode
        self.session = MapSession(instance)
        # The reference background is off unless the operator asked for it,
        # and this window is the debug view that asking opens: a truth map
        # behind live evidence is a display decision, recorded by being an
        # explicit flag rather than a default anyone drifts into.
        self.show_background = bool(background)
        self.imagery = ImageSession(instance) if self.show_background else None
        self.context = Context(instance) if self.show_background else None
        self.clock = status_clock(instance)
        self._save_geometry = save_window_geometry
        self.root = tk.Tk()
        apply_theme(self.root)
        self.root.title(f'dcmn map pane — {instance}')
        self.root.geometry('760x820')
        self.root.minsize(360, 360)
        self.root.protocol('WM_DELETE_WINDOW', self.close)
        self.running = True
        self.panes: dict[str, MapPane] = {}
        self.frame = ttk.Frame(self.root)
        self.frame.grid(row=0, column=0, sticky='nsew')
        self.status = tk.StringVar(value=f'waiting for a map manifest on {instance}')
        ttk.Label(self.root, textvariable=self.status, style='Dim.TLabel',
                  anchor='w').grid(row=1, column=0, sticky='ew', padx=6, pady=4)
        self.root.columnconfigure(0, weight=1); self.root.rowconfigure(0, weight=1)
        restore_window_geometry(self.root, f'dcmn.map_pane.{instance}')

    def _sim_time(self) -> float:
        """The simulator's clock, retried until it is there: order is irrelevant."""
        if self.clock is None:
            from dcmn.maps import status_clock
            self.clock = status_clock(self.instance)
        return self.clock() if self.clock else 0.

    def _background(self) -> Background | None:
        """The newest reference image that still matches the vehicle's frame.

        Missing imagery, a plane nobody publishes to and an image whose frame
        or epochs disagree with the context snapshot all come back the same
        way -- ``None``, a pane with no background -- because imagery is
        decoration and nothing here may depend on it.
        """
        if self.imagery is None: return None
        self.imagery.poll()
        snapshot = self.context.read() if self.context is not None else {}
        for image_id in sorted(self.imagery.states):
            reference = self.imagery.display(image_id, snapshot)
            if reference is not None: return Background(reference)
        return None

    def _pane_for(self, source: str) -> MapPane:
        if source not in self.panes:
            pane = MapPane(self.frame, title=source, mode=self.mode,
                           on_click=lambda x, y, s=source: self.status.set(
                               f'{s}: clicked {x:.2f}, {y:.2f} m'))
            index = len(self.panes)
            pane.widget.grid(row=index // 2, column=index % 2, sticky='nsew',
                             padx=3, pady=3)
            self.frame.columnconfigure(index % 2, weight=1)
            self.frame.rowconfigure(index // 2, weight=1)
            self.panes[source] = pane
        return self.panes[source]

    def tick(self) -> None:
        self.session.poll()
        sim_now = self._sim_time()
        report = self.session.report(sim_now)
        background = self._background()
        for source, state in sorted(self.session.states.items()):
            if self.only and source != self.only: continue
            pane = self._pane_for(source)
            pane.set_grid(state.grid, sim_now_s=sim_now,
                          horizon_s=self.session.staleness_horizon_s)
            pane.set_background(background, refresh=False)
            entry = report['sources'][source]
            age = entry['age_s']
            rate = entry['rate_hz']
            pane.set_header(
                f'{source}  rev {entry["revision"]}  '
                f'{"stale" if entry["stale"] else "fresh"} '
                f'{"--" if age is None else format(age, ".1f") + "s"}  '
                f'{"--" if rate is None else format(rate, ".2f")}/sim-s')
            pane.refresh()
        rejected = self.session.rejections[-1][1] if self.session.rejections else ''
        self.status.set(
            f'{"connected" if report["connected"] else "no producer"}  '
            f'sim {sim_now:.2f}s  sources {len(self.session.states)}  '
            f'discoveries {report["discoveries"]}'
            + (f'  last rejection: {rejected}' if rejected else ''))

    def run(self) -> int:
        # The loop spins faster than MAP_HZ so the window stays responsive;
        # each pane's own Paced() is what caps the repainting.
        while self.running:
            self.tick()
            self.root.update_idletasks(); self.root.update()
            time.sleep(1. / MAP_HZ / 4)
        return 0

    def close(self) -> None:
        self._save_geometry(self.root, f'dcmn.map_pane.{self.instance}')
        self.running = False
        self.session.close()
        if self.imagery is not None: self.imagery.close()
        try: self.root.destroy()
        except Exception: pass


def main(argv=None) -> int:
    import argparse
    import sys

    from dcmn.window import disable_input_method

    parser = argparse.ArgumentParser(
        description='show an instance\'s evidence grids in the shared map pane')
    parser.add_argument('--id', required=True, help='instance id')
    parser.add_argument('--source', default=None, help='show only this source')
    parser.add_argument('--mode', default='occupancy', choices=MODES,
                        help='which display mode the panes open in')
    parser.add_argument('--background', action='store_true',
                        help='debug display: show the reference-imagery background '
                             'behind the evidence (off by default)')
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    disable_input_method()
    try:
        return MapPaneWindow(args.id, source=args.source, mode=args.mode,
                              background=args.background).run()
    except ImportError as exc:
        print(f'dcmn.map_pane: tkinter is unavailable: {exc}', file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == '__main__':
    raise SystemExit(main())
