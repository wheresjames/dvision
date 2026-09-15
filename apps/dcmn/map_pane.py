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


def overlay_shapes(overlay: Overlay, cell_m: float) -> Iterator[Shape]:
    """Every primitive an overlay is made of, in map metres.

    The vehicle is a disc with a heading spoke rather than
    :meth:`MapView.draw_drone`'s nose-forward triangle: a polygon has no
    representation in :class:`~dcmn.mapview.Shape`, and a shape the PIL backend
    could not draw would be a shape a report could not contain.
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
        if shape.kind == 'rect':
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


class MapPane:
    """A grid, its overlays, a hover readout and a click callback.

    Owns no data and no clock: a host sets a grid and a simulated time and
    calls :meth:`refresh`, which repaints at most at ``MAP_HZ`` because a
    window is not part of the flight.
    """

    def __init__(self, parent, *, on_click: Callable[[float, float], None] | None = None,
                 mode: str = 'occupancy', hz: float = MAP_HZ, controls: bool = True,
                 title: str = '', width: int = 320, height: int = 320) -> None:
        import tkinter as tk
        from tkinter import ttk
        from PIL import ImageTk

        if mode not in MODES: raise ValueError(f'unknown map display mode {mode!r}')
        self.tk, self.ttk, self.ImageTk = tk, ttk, ImageTk
        self.state = PaneState(mode=mode)
        self.on_click = on_click
        self.view = MapView(cell=8., margin=0)
        self.photo = None
        self._paint = Paced(hz)
        self._raster_key: tuple | None = None
        self._image = None
        self._resize_after: str | None = None
        self._notice = ''

        self.widget = ttk.Frame(parent, style='Panel.TFrame')
        self.header = tk.StringVar(value=title)
        self.readout = tk.StringVar(value=HOVER_PROMPT)
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
        self.canvas.bind('<Leave>', lambda _e: self.readout.set(self._notice or HOVER_PROMPT))
        self.canvas.bind('<Button-1>', self._click)
        self.canvas.bind('<Configure>', self._resized)

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

    # -- geometry ----------------------------------------------------------

    def _fit(self) -> GridGeometry | None:
        """Size the transform to the canvas as it is right now.

        Tk reports a width of 1 until a widget has been mapped, and fitting a
        whole extent into one pixel puts every metre a hundred times where it
        belongs -- so a pane asked for a coordinate before its first
        ``<Configure>`` falls back to the size it asked for.
        """
        grid = self.state.grid
        if grid is None: return None
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        if width <= 1: width = max(1, self.canvas.winfo_reqwidth())
        if height <= 1: height = max(1, self.canvas.winfo_reqheight())
        self.view.fit_canvas(_Extent(grid.geometry), width, height)
        return grid.geometry

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
        if geometry is None:
            self.canvas.itemconfigure(self.item, image='')
            self.photo = None
            return True
        state = self.state
        # The age mode's raster changes with the clock even when the grid does
        # not, so simulated time is part of what identifies a rendering; a
        # background is part of it too, identified by revision and checksum so
        # a replacement image is a new rendering even at the same resolution.
        key = (id(state.grid), state.grid.revision, state.mode, state.layer,
               round(state.sim_now_s, 2) if state.mode == 'age' else 0,
               None if state.cost is None else state.cost.tobytes(),
               None if state.never_observed is None else state.never_observed.tobytes(),
               None if state.background is None else state.background.identity)
        # A notice belongs to the state that produced it; leaving one up after
        # the pane has painted something says the wrong thing about what is on
        # screen.
        if self._notice:
            self._notice = ''
            self.readout.set(HOVER_PROMPT)
        if key != self._raster_key or self._image is None:
            raster = render_raster(state.grid, mode=state.mode, layer=state.layer,
                                   sim_now_s=state.sim_now_s,
                                   horizon_s=state.horizon_s, cost=state.cost,
                                   never_observed=state.never_observed)
            if raster is not None and state.background is not None:
                mask = never_cells(state.grid, mode=state.mode, layer=state.layer,
                                   never_observed=state.never_observed)
                if mask is not None:
                    raster = compose_background(raster, state.background,
                                                state.grid.geometry, mask)
            self._raster_key = key
            self._image = None if raster is None else Image.fromarray(raster)
        if self._image is None:
            self.canvas.itemconfigure(self.item, image='')
            self.photo = None
            self._notice = 'no cost layer for this source'
            self.readout.set(self._notice)
            return True
        size = (max(1, round(geometry.width * self.view.cell * geometry.cell_m)),
                max(1, round(geometry.height * self.view.cell * geometry.cell_m)))
        # NEAREST: every pixel of a measured cell has to stay that cell, or the
        # pane invents evidence at the boundaries.
        self.photo = self.ImageTk.PhotoImage(
            self._image.resize(size, Image.Resampling.NEAREST))
        self.canvas.coords(self.item, self.view.offset_x, self.view.offset_y)
        self.canvas.itemconfigure(self.item, image=self.photo)
        self._draw_overlay(geometry)
        return True

    def _paint_shapes_tk(self, geometry: GridGeometry, shapes) -> None:
        for shape in shapes:
            x0, y0 = project(self.view, geometry, shape.x0, shape.y0)
            x1, y1 = project(self.view, geometry, shape.x1, shape.y1)
            if shape.kind == 'rect':
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
                self.canvas.create_text(self.view.offset_x + 4, self.view.offset_y + 2,
                                        text=background.label(), anchor='nw',
                                        fill=theme.DIM, tags='overlay')
        self._paint_shapes_tk(geometry, overlay_shapes(self.state.overlay, geometry.cell_m))

    def snapshot(self, *, cell_px: int = SNAPSHOT_CELL_PX):
        """The same rendering as a PIL image, for a report."""
        return snapshot_image(self.state.grid, mode=self.state.mode,
                              layer=self.state.layer, sim_now_s=self.state.sim_now_s,
                              horizon_s=self.state.horizon_s, cost=self.state.cost,
                              overlay=self.state.overlay, cell_px=cell_px,
                              never_observed=self.state.never_observed,
                              background=self.state.background)

    # -- interaction -------------------------------------------------------

    def describe(self, x_m: float, y_m: float) -> str:
        """The hover readout: where, how likely, and how long ago."""
        grid = self.state.grid
        if grid is None: return 'no grid'
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
        self.readout.set('no grid' if point is None else self.describe(*point))

    def _click(self, event) -> None:
        point = self.to_map(event.x, event.y)
        if point is not None and self.on_click is not None: self.on_click(*point)


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
