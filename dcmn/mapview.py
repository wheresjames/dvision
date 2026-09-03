"""The top-down map, drawn the same way by every window that shows one.

`dsim`, `dway`'s Fly tab, `dway`'s tour editor and `dway`'s flight report all
draw the same world, and all four had drawn it differently: the same wall was
light grey in one and dark in another, a tree was a circle in one and a square
in the next, only one drew the ground grid that makes a map readable, and the
report left the targets out entirely. This module is the single answer, so a
map looks like a map wherever it appears.

The geometry lives in :func:`map_shapes`, in map metres, and the backends are
thin: :meth:`MapView.draw_map` applies the pixel transform and paints onto a Tk
canvas, :func:`draw_map_axes` paints the identical shapes onto matplotlib axes
that are already in map coordinates. Adding a third surface means writing
another short adapter, not another drawing of a map.

The transform is deliberately trivial and public: map metres are canvas pixels
scaled by ``cell`` and offset by ``margin``, and ``to_map`` inverts it exactly,
because the editor turns pointer positions back into waypoints.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator

from dcmn import theme

#: Longest canvas edge a map is fitted into, before clamping to the cell range.
DEFAULT_MAX_EDGE_PX = 720
DEFAULT_MIN_CELL_PX = 18
DEFAULT_MAX_CELL_PX = 42
DEFAULT_MARGIN_PX = 18


@dataclass(frozen=True)
class Shape:
    """One primitive of a drawn map, in **map metres**.

    Map space rather than pixels, because the report plots straight into map
    coordinates while a Tk canvas needs the transform applied first. Keeping
    the geometry here is what makes "the same map" true by construction
    instead of by two implementations agreeing to stay in step.
    """

    kind: str                      # "rect" | "oval" | "line"
    x0: float
    y0: float
    x1: float
    y1: float
    fill: str | None = None
    outline: str | None = None
    width: float = 1.0


def map_shapes(sim_map, *, grid: bool = True) -> Iterator[Shape]:
    """Every primitive a drawn map is made of: ground first, objects on top."""
    if grid:
        for y in range(sim_map.height):
            for x in range(sim_map.width):
                yield Shape("rect", x, y, x + 1, y + 1,
                            fill=theme.CELL, outline=theme.MAP_GRID, width=1.0)
    for obj in sim_map.objects:
        if obj.kind == "wall":
            yield Shape("rect", obj.x - 0.5, obj.y - 0.5, obj.x + 0.5, obj.y + 0.5,
                        fill=theme.WALL_FILL, outline=theme.WALL_EDGE)
        elif obj.kind == "tree":
            yield Shape("oval", obj.x - 0.36, obj.y - 0.36, obj.x + 0.36, obj.y + 0.36,
                        fill=theme.TREE_FILL, outline=theme.TREE_EDGE)
        elif obj.kind == "target":
            r = 0.34
            yield Shape("oval", obj.x - r, obj.y - r, obj.x + r, obj.y + r,
                        fill=theme.TARGET_FILL, outline=theme.TARGET_EDGE)
            yield Shape("line", obj.x - r, obj.y, obj.x + r, obj.y,
                        outline=theme.TARGET_CROSS, width=2.0)
            yield Shape("line", obj.x, obj.y - r, obj.x, obj.y + r,
                        outline=theme.TARGET_CROSS, width=2.0)
        else:
            yield Shape("rect", obj.x - 0.5, obj.y - 0.5, obj.x + 0.5, obj.y + 0.5,
                        fill=theme.OBJECT_FILL, outline=theme.GRID)


def draw_map_axes(axes, sim_map, *, grid: bool = True, zorder: float = 1.0) -> None:
    """Paint the same map onto matplotlib axes already in map coordinates.

    The report's track plot is the one place a map is drawn without a Tk
    canvas. It draws the identical shapes so a report and a live window show
    the same world -- including the targets and the ground grid the plot used
    to leave out.
    """
    from matplotlib.patches import Ellipse, Rectangle

    for shape in map_shapes(sim_map, grid=grid):
        if shape.kind == "rect":
            axes.add_patch(Rectangle(
                (shape.x0, shape.y0), shape.x1 - shape.x0, shape.y1 - shape.y0,
                facecolor=shape.fill or "none",
                edgecolor=shape.outline or "none",
                linewidth=shape.width * 0.5, zorder=zorder))
        elif shape.kind == "oval":
            axes.add_patch(Ellipse(
                ((shape.x0 + shape.x1) / 2.0, (shape.y0 + shape.y1) / 2.0),
                shape.x1 - shape.x0, shape.y1 - shape.y0,
                facecolor=shape.fill or "none",
                edgecolor=shape.outline or "none",
                linewidth=shape.width * 0.5, zorder=zorder + 0.1))
        else:
            axes.plot([shape.x0, shape.x1], [shape.y0, shape.y1],
                      color=shape.outline, linewidth=shape.width * 0.5,
                      zorder=zorder + 0.2)


class MapView:
    """A map-to-canvas transform, plus the drawing that goes with it.

    Holds no canvas and no map: a window owns those and passes them in, which
    is what lets the editor redraw into a canvas it clears wholesale while
    `dsim` keeps its static map and repaints only the vehicle.
    """

    def __init__(self, *, cell: int = DEFAULT_MIN_CELL_PX,
                 margin: int = DEFAULT_MARGIN_PX) -> None:
        self.cell = int(cell)
        self.margin = int(margin)

    @classmethod
    def fitted(cls, sim_map, *, max_edge_px: int = DEFAULT_MAX_EDGE_PX,
               min_cell_px: int = DEFAULT_MIN_CELL_PX,
               max_cell_px: int = DEFAULT_MAX_CELL_PX,
               margin: int = DEFAULT_MARGIN_PX) -> "MapView":
        """Sized so the map's longer side fills roughly ``max_edge_px``."""
        span = max(sim_map.width, sim_map.height, 1)
        cell = max(min_cell_px, min(max_cell_px, int(max_edge_px / span)))
        return cls(cell=cell, margin=margin)

    def fit(self, sim_map, **kwargs) -> "MapView":
        """Resize in place for a map that arrived after the window was built."""
        sized = self.fitted(sim_map, margin=self.margin, **kwargs)
        self.cell = sized.cell
        return self

    # -- geometry -------------------------------------------------------

    def canvas_size(self, sim_map) -> tuple[int, int]:
        return (sim_map.width * self.cell + self.margin * 2,
                sim_map.height * self.cell + self.margin * 2)

    def xy(self, x: float, y: float) -> tuple[float, float]:
        """Map metres to canvas pixels."""
        return self.margin + x * self.cell, self.margin + y * self.cell

    def to_map(self, px: float, py: float) -> tuple[float, float]:
        """Canvas pixels back to map metres; the exact inverse of ``xy``."""
        return (px - self.margin) / self.cell, (py - self.margin) / self.cell

    # -- drawing --------------------------------------------------------

    def draw_map(self, canvas, sim_map, *, tags: str | tuple[str, ...] = (),
                 grid: bool = True) -> None:
        """Ground, then objects. Everything a client draws goes on top."""
        for shape in map_shapes(sim_map, grid=grid):
            x0, y0 = self.xy(shape.x0, shape.y0)
            x1, y1 = self.xy(shape.x1, shape.y1)
            if shape.kind == "rect":
                canvas.create_rectangle(x0, y0, x1, y1, fill=shape.fill,
                                        outline=shape.outline or "", tags=tags)
            elif shape.kind == "oval":
                canvas.create_oval(x0, y0, x1, y1, fill=shape.fill,
                                   outline=shape.outline or "", tags=tags)
            else:
                canvas.create_line(x0, y0, x1, y1, fill=shape.outline,
                                   width=shape.width, tags=tags)

    def draw_drone(self, canvas, x: float, y: float, heading_deg: float, *,
                   crashed: bool = False, cone: bool = True,
                   tags: str | tuple[str, ...] = ()) -> list[int]:
        """The vehicle: a view cone, a nose-forward body, and a hub.

        ``heading_deg`` is the public compass heading -- north 0, clockwise
        positive -- never a renderer yaw. A caller holding an internal yaw
        converts it before calling, so the sign lives in one place.
        """
        cx, cy = self.xy(x, y)
        # Compass heading to canvas direction: north is -y, east is +x.
        angle = math.radians(float(heading_deg) - 90.0)
        size = self.cell * 0.34
        nose = (cx + math.cos(angle) * size, cy + math.sin(angle) * size)
        left = (cx + math.cos(angle + 2.45) * size,
                cy + math.sin(angle + 2.45) * size)
        right = (cx + math.cos(angle - 2.45) * size,
                 cy + math.sin(angle - 2.45) * size)
        items: list[int] = []
        if cone:
            length = self.cell * 1.8
            vl = (cx + math.cos(angle - 0.45) * length,
                  cy + math.sin(angle - 0.45) * length)
            vr = (cx + math.cos(angle + 0.45) * length,
                  cy + math.sin(angle + 0.45) * length)
            items.append(canvas.create_polygon(
                cx, cy, vl[0], vl[1], vr[0], vr[1],
                fill=theme.WARN, outline="", stipple="gray50", tags=tags))
        items.append(canvas.create_polygon(
            nose[0], nose[1], left[0], left[1], right[0], right[1],
            fill=theme.DANGER if crashed else theme.ACCENT,
            outline=theme.DRONE_EDGE, width=2, tags=tags))
        items.append(canvas.create_oval(cx - 3, cy - 3, cx + 3, cy + 3,
                                        fill=theme.DRONE_HUB, outline="",
                                        tags=tags))
        return items
