"""The tour editor: place a flight plan on a map and see whether it is flyable.

A tour is a file, so this is a file editor with a map behind it. What it adds
over a text editor is the two things that cannot be read off JSON -- where the
waypoints actually are, and how close the legs pass to the walls -- and it
computes the second with the same code the follower uses at preflight, so a
tour that looks clear here is one that will not be refused there.
"""

from __future__ import annotations

import math
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Callable

from dcmn import theme
from dcmn.mapview import MapView
from dvision2_common import SimMap, load_map
from dway.tour import (
    DEFAULT_MIN_CLEARANCE_M, DEFAULT_SPEED_MPS, DEFAULT_WAYPOINT_TOLERANCE_M,
    Tour, TourError, Waypoint, leg_clearances, load_tour, map_content_sha,
    resolve_map, save_tour,
)

# One palette, in dcmn.theme, so every window's map is the same colour.
_BG = theme.BG
_PANEL = theme.PANEL
_CANVAS = theme.CANVAS
_TEXT = theme.TEXT
_DIM = theme.DIM
_ACCENT = theme.ACCENT
_WARN = theme.WARN
_DANGER = theme.DANGER
_OK = theme.OK
_HIT_RADIUS_PX = 9
_HEADING_HIT_PX = 6
_HEADING_MIN_LENGTH_PX = 18


class TourEditor:
    """A map, a list of waypoints, and a live verdict on the geometry."""

    def __init__(self, parent: tk.Misc, *, root_dir: Path,
                 on_status: Callable[[str], None] | None = None) -> None:
        self.root_dir = Path(root_dir)
        self._on_status = on_status
        self.frame = ttk.Frame(parent, padding=8)
        self.sim_map: SimMap | None = None
        self.map_path: Path | None = None
        self.tour_path: Path | None = None
        self.waypoints: list[dict[str, float]] = []
        self.selected: int | None = None
        self.view = MapView(cell=16, margin=10)
        self._drag_mode: str | None = None

        self._build_toolbar()
        self._build_canvas()
        self._build_side_panel()
        self.frame.columnconfigure(0, weight=1)
        self.frame.rowconfigure(1, weight=1)
        self._refresh()

    # ------------------------------------------------------------------
    # Widgets
    # ------------------------------------------------------------------

    def _build_toolbar(self) -> None:
        bar = ttk.Frame(self.frame)
        bar.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        for column, (text, action) in enumerate((
                ("Open map", self.open_map),
                ("Open tour", self.open_tour),
                ("Save tour", self.save),
                ("Save as", self.save_as),
                ("Clear waypoints", self.clear_waypoints))):
            ttk.Button(bar, text=text, command=action).grid(
                row=0, column=column, padx=(0, 6))
        self.status_var = tk.StringVar(value="open a map to start a tour")
        ttk.Label(bar, textvariable=self.status_var, style="Dim.TLabel").grid(
            row=0, column=9, sticky="w", padx=(12, 0))
        bar.columnconfigure(9, weight=1)

    def _build_canvas(self) -> None:
        self.canvas = tk.Canvas(self.frame, width=640, height=420,
                                background=_CANVAS, highlightthickness=0)
        self.canvas.grid(row=1, column=0, sticky="nsew")
        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<Button-3>", self._on_right_click)

    def _build_side_panel(self) -> None:
        side = ttk.Frame(self.frame, padding=(12, 0, 0, 0))
        side.grid(row=1, column=1, sticky="nsew")

        ttk.Label(side, text="tour", style="Dim.TLabel").grid(
            row=0, column=0, sticky="w")
        self.tour_id_var = tk.StringVar(value="untitled.v1")
        ttk.Entry(side, textvariable=self.tour_id_var, width=28).grid(
            row=0, column=1, sticky="w", pady=2)

        self.fields: dict[str, tk.StringVar] = {}
        for row, (name, label, default) in enumerate((
                ("default_speed_mps", "speed m/s", DEFAULT_SPEED_MPS),
                ("waypoint_tolerance_m", "tolerance m", DEFAULT_WAYPOINT_TOLERANCE_M),
                ("min_clearance_m", "clearance m", DEFAULT_MIN_CLEARANCE_M)), start=1):
            ttk.Label(side, text=label, style="Dim.TLabel").grid(
                row=row, column=0, sticky="w")
            var = tk.StringVar(value=str(default))
            var.trace_add("write", lambda *_: self._refresh())
            self.fields[name] = var
            ttk.Entry(side, textvariable=var, width=10).grid(
                row=row, column=1, sticky="w", pady=2)

        ttk.Label(side, text="waypoints", style="Dim.TLabel").grid(
            row=4, column=0, columnspan=2, sticky="w", pady=(10, 2))
        self.listbox = tk.Listbox(side, height=9, width=38, background=_PANEL,
                                  foreground=_TEXT, selectbackground=_ACCENT,
                                  highlightthickness=0, borderwidth=0,
                                  activestyle="none", exportselection=False)
        self.listbox.grid(row=5, column=0, columnspan=2, sticky="ew")
        self.listbox.bind("<<ListboxSelect>>", self._on_select)

        editor = ttk.Frame(side)
        editor.grid(row=6, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        self.point_fields: dict[str, tk.StringVar] = {}
        for column, (name, label) in enumerate((
                ("z", "alt m"), ("heading_deg", "heading"), ("dwell_s", "dwell s"))):
            ttk.Label(editor, text=label, style="Dim.TLabel").grid(
                row=0, column=column, sticky="w", padx=(0, 6))
            var = tk.StringVar(value="")
            self.point_fields[name] = var
            entry = ttk.Entry(editor, textvariable=var, width=8)
            entry.grid(row=1, column=column, sticky="w", padx=(0, 6))
            entry.bind("<Return>", lambda _event: self._apply_point_fields())
            entry.bind("<FocusOut>", lambda _event: self._apply_point_fields())
        ttk.Button(editor, text="Delete", command=self.delete_selected).grid(
            row=1, column=3, padx=(6, 0))

        ttk.Label(side, text="diagnostics", style="Dim.TLabel").grid(
            row=7, column=0, columnspan=2, sticky="w", pady=(10, 2))
        self.diagnostics = tk.Text(side, height=10, width=40, background=_PANEL,
                                   foreground=_TEXT, highlightthickness=0,
                                   borderwidth=0, wrap="word")
        self.diagnostics.grid(row=8, column=0, columnspan=2, sticky="nsew")
        self.diagnostics.configure(state="disabled")
        side.rowconfigure(8, weight=1)

    # ------------------------------------------------------------------
    # Files
    # ------------------------------------------------------------------

    def _status(self, message: str) -> None:
        self.status_var.set(message)
        if self._on_status is not None:
            self._on_status(message)

    def open_map(self, path: str | Path | None = None) -> None:
        if path is None:
            path = filedialog.askopenfilename(
                title="Open map", initialdir=str(self.root_dir / "assets/maps"),
                filetypes=[("dvision2 maps", "*.txt"), ("all files", "*")])
            if not path:
                return
        try:
            self.load_map_file(Path(path))
        except (OSError, ValueError) as exc:
            messagebox.showerror("Map", str(exc))

    def load_map_file(self, path: Path) -> None:
        self.sim_map = load_map(Path(path))
        self.map_path = Path(path)
        self.view.fit(self.sim_map, max_edge_px=760, max_cell_px=30)
        width, height = self.view.canvas_size(self.sim_map)
        self.canvas.config(width=width, height=height)
        self._status(f"map {path}")
        self._refresh()

    def open_tour(self, path: str | Path | None = None) -> None:
        if path is None:
            path = filedialog.askopenfilename(
                title="Open tour", initialdir=str(self.root_dir / "assets/tours"),
                filetypes=[("tours", "*.json"), ("all files", "*")])
            if not path:
                return
        try:
            self.load_tour_file(Path(path))
        except (OSError, TourError) as exc:
            messagebox.showerror("Tour", str(exc))

    def load_tour_file(self, path: Path) -> None:
        tour = load_tour(path)
        if tour.coordinate_frame != "map":
            raise TourError(
                f"{path}: the editor edits map-frame tours; this one is "
                f"{tour.coordinate_frame}")
        map_file = resolve_map(tour, self.root_dir)
        if map_file is not None and map_file.exists():
            self.load_map_file(map_file)
        self.tour_path = Path(path)
        self.tour_id_var.set(tour.tour_id)
        self.fields["default_speed_mps"].set(str(tour.default_speed_mps))
        self.fields["waypoint_tolerance_m"].set(str(tour.waypoint_tolerance_m))
        self.fields["min_clearance_m"].set(str(tour.min_clearance_m))
        self.waypoints = [
            {"x": float(point.x), "y": float(point.y), "z": float(point.z),
             "heading_deg": float(point.heading_deg), "dwell_s": float(point.dwell_s)}
            for point in tour.waypoints
        ]
        self.selected = 0 if self.waypoints else None
        self._status(f"tour {path}")
        self._refresh()

    def build_tour(self) -> Tour:
        """The tour as edited, validated the same way the loader validates it."""
        if self.sim_map is None or self.map_path is None:
            raise TourError("open a map before saving a tour")
        if not self.waypoints:
            raise TourError("a tour needs at least one waypoint")
        try:
            speed = float(self.fields["default_speed_mps"].get())
            tolerance = float(self.fields["waypoint_tolerance_m"].get())
            clearance = float(self.fields["min_clearance_m"].get())
        except ValueError as exc:
            raise TourError(f"tour settings must be numbers: {exc}") from exc
        tour_id = self.tour_id_var.get().strip()
        if not tour_id:
            raise TourError("a tour needs an id")
        try:
            relative = str(self.map_path.resolve().relative_to(self.root_dir))
        except ValueError:
            relative = str(self.map_path)
        return Tour(
            tour_id=tour_id, schema_version=1, coordinate_frame="map",
            waypoints=tuple(
                Waypoint(index=index, frame="map", x=point["x"], y=point["y"],
                         z=point["z"], heading_deg=point["heading_deg"] % 360.0,
                         dwell_s=point["dwell_s"])
                for index, point in enumerate(self.waypoints)),
            path=self.tour_path or Path(f"{tour_id}.json"),
            map_path=relative, map_sha=map_content_sha(self.map_path),
            default_speed_mps=speed, waypoint_tolerance_m=tolerance,
            min_clearance_m=clearance)

    def save(self) -> Path | None:
        if self.tour_path is None:
            return self.save_as()
        return self._write(self.tour_path)

    def save_as(self) -> Path | None:
        path = filedialog.asksaveasfilename(
            title="Save tour", defaultextension=".json",
            initialdir=str(self.root_dir / "assets/tours"),
            initialfile=f"{self.tour_id_var.get().strip() or 'untitled'}.json",
            filetypes=[("tours", "*.json")])
        return self._write(Path(path)) if path else None

    def _write(self, path: Path) -> Path | None:
        try:
            tour = self.build_tour()
            save_tour(tour, path)
            # Saving is not the same as being loadable; reading it straight
            # back is the only check that says the file will fly.
            load_tour(path)
        except (TourError, OSError) as exc:
            messagebox.showerror("Save tour", str(exc))
            self._status(f"not saved: {exc}")
            return None
        self.tour_path = path
        self._status(f"saved {path}")
        return path

    def clear_waypoints(self) -> None:
        self.waypoints = []
        self.selected = None
        self._refresh()

    def delete_selected(self) -> None:
        if self.selected is None:
            return
        del self.waypoints[self.selected]
        self.selected = min(self.selected, len(self.waypoints) - 1)
        if self.selected < 0:
            self.selected = None
        self._refresh()

    # ------------------------------------------------------------------
    # Canvas interaction
    # ------------------------------------------------------------------

    def _to_canvas(self, x: float, y: float) -> tuple[float, float]:
        return self.view.xy(x, y)

    def _to_map(self, cx: float, cy: float) -> tuple[float, float]:
        return self.view.to_map(cx, cy)

    def _hit(self, cx: float, cy: float) -> int | None:
        for index, point in enumerate(self.waypoints):
            px, py = self._to_canvas(point["x"], point["y"])
            if math.hypot(px - cx, py - cy) <= _HIT_RADIUS_PX:
                return index
        return None

    def _heading_length(self) -> float:
        return float(max(self.view.cell, _HEADING_MIN_LENGTH_PX))

    def _heading_hit(self, cx: float, cy: float) -> int | None:
        """Return the arrow under the pointer, excluding its waypoint marker."""
        nearest: tuple[float, int] | None = None
        for index, point in enumerate(self.waypoints):
            px, py = self._to_canvas(point["x"], point["y"])
            heading = math.radians(point["heading_deg"])
            ux, uy = math.sin(heading), -math.cos(heading)
            length = self._heading_length()
            # Start outside the marker so clicking the waypoint still moves it.
            start = min(float(_HIT_RADIUS_PX), length * 0.5)
            projection = (cx - px) * ux + (cy - py) * uy
            if not start <= projection <= length + _HEADING_HIT_PX:
                continue
            perpendicular = abs((cx - px) * uy - (cy - py) * ux)
            if perpendicular <= _HEADING_HIT_PX \
                    and (nearest is None or perpendicular < nearest[0]):
                nearest = perpendicular, index
        return None if nearest is None else nearest[1]

    def _on_click(self, event: tk.Event) -> None:
        if self.sim_map is None:
            return
        hit = self._hit(event.x, event.y)
        if hit is not None:
            self.selected = hit
            self._drag_mode = "move"
        else:
            heading_hit = self._heading_hit(event.x, event.y)
            if heading_hit is not None:
                self.selected = heading_hit
                self._drag_mode = "heading"
                self._refresh()
                return
            x, y = self._to_map(event.x, event.y)
            if not (0.0 <= x <= self.sim_map.width and 0.0 <= y <= self.sim_map.height):
                return
            previous = self.waypoints[-1] if self.waypoints else None
            self.waypoints.append({
                "x": round(x, 3), "y": round(y, 3),
                "z": previous["z"] if previous else 1.5,
                "heading_deg": previous["heading_deg"] if previous else 0.0,
                "dwell_s": previous["dwell_s"] if previous else 0.0,
            })
            self.selected = len(self.waypoints) - 1
            self._drag_mode = None
        self._refresh()

    def _on_drag(self, event: tk.Event) -> None:
        if self._drag_mode is None or self.selected is None or self.sim_map is None:
            return
        point = self.waypoints[self.selected]
        if self._drag_mode == "heading":
            px, py = self._to_canvas(point["x"], point["y"])
            dx, dy = event.x - px, event.y - py
            if math.hypot(dx, dy) > 1.0:
                # Canvas Y points south, so atan2(east, north) is atan2(dx, -dy).
                point["heading_deg"] = round(math.degrees(
                    math.atan2(dx, -dy)) % 360.0, 2)
        else:
            x, y = self._to_map(event.x, event.y)
            point["x"] = round(min(max(x, 0.0), float(self.sim_map.width)), 3)
            point["y"] = round(min(max(y, 0.0), float(self.sim_map.height)), 3)
        self._refresh()

    def _on_release(self, _event: tk.Event) -> None:
        self._drag_mode = None

    def _on_right_click(self, event: tk.Event) -> None:
        hit = self._hit(event.x, event.y)
        if hit is not None:
            self.selected = hit
            self.delete_selected()

    def _on_select(self, _event: tk.Event) -> None:
        selection = self.listbox.curselection()
        if selection:
            self.selected = int(selection[0])
            self._refresh()

    def _apply_point_fields(self) -> None:
        if self.selected is None:
            return
        point = self.waypoints[self.selected]
        for name, var in self.point_fields.items():
            try:
                point[name] = float(var.get())
            except ValueError:
                continue
        self._refresh()

    # ------------------------------------------------------------------
    # Drawing and diagnostics
    # ------------------------------------------------------------------

    def _refresh(self) -> None:
        self._draw()
        self._refresh_list()
        self._refresh_diagnostics()

    def _draw(self) -> None:
        self.canvas.delete("all")
        if self.sim_map is not None:
            self.view.draw_map(self.canvas, self.sim_map)
            sx, sy = self._to_canvas(self.sim_map.start_x, self.sim_map.start_y)
            self.canvas.create_oval(sx - 5, sy - 5, sx + 5, sy + 5,
                                    outline=_OK, width=2)
            self.canvas.create_text(sx, sy - 12, text="start", fill=_OK,
                                    font=("TkDefaultFont", 7))
        clearances = self._clearances()
        for index, point in enumerate(self.waypoints):
            cx, cy = self._to_canvas(point["x"], point["y"])
            if index:
                px, py = self._to_canvas(self.waypoints[index - 1]["x"],
                                         self.waypoints[index - 1]["y"])
                leg = clearances[index] if index < len(clearances) else None
                color = _WARN
                if leg is not None and leg.obstructed:
                    color = _DANGER
                elif leg is not None and leg.clearance_m < self._min_clearance():
                    color = theme.CAUTION
                self.canvas.create_line(px, py, cx, cy, fill=color, width=2,
                                        dash=(5, 3))
            fill = _ACCENT if index == self.selected else ""
            self.canvas.create_oval(cx - 5, cy - 5, cx + 5, cy + 5,
                                    outline=_WARN, fill=fill, width=2)
            self.canvas.create_text(cx + 10, cy - 10, text=str(index),
                                    fill=_WARN, font=("TkDefaultFont", 7))
            heading = math.radians(point["heading_deg"])
            length = self._heading_length()
            self.canvas.create_line(
                cx, cy, cx + math.sin(heading) * length,
                cy - math.cos(heading) * length, fill=_TEXT, width=2,
                arrow="last")

    def _refresh_list(self) -> None:
        self.listbox.delete(0, tk.END)
        for index, point in enumerate(self.waypoints):
            self.listbox.insert(
                tk.END,
                f"{index}  x={point['x']:.2f} y={point['y']:.2f} "
                f"z={point['z']:.2f}  hdg {point['heading_deg']:.0f}  "
                f"dwell {point['dwell_s']:.1f}s")
        if self.selected is not None and 0 <= self.selected < len(self.waypoints):
            self.listbox.selection_clear(0, tk.END)
            self.listbox.selection_set(self.selected)
            point = self.waypoints[self.selected]
            for name, var in self.point_fields.items():
                var.set(f"{point[name]:g}")
        else:
            for var in self.point_fields.values():
                var.set("")

    def _min_clearance(self) -> float:
        try:
            return float(self.fields["min_clearance_m"].get())
        except ValueError:
            return DEFAULT_MIN_CLEARANCE_M

    def _clearances(self):
        if self.sim_map is None or len(self.waypoints) < 1:
            return []
        try:
            tour = self.build_tour()
        except TourError:
            return []
        # Leg zero is the flight from the map's own start pose, which is where
        # a vehicle actually begins; leaving it out is how a tour that cannot
        # be flown looks fine in an editor.
        return leg_clearances(tour, self.sim_map,
                              (self.sim_map.start_x, self.sim_map.start_y))

    def geometry_diagnostics(self) -> dict[str, float]:
        """Path length and the longest straight run, as the tour files record."""
        points = [(point["x"], point["y"]) for point in self.waypoints]
        total = sum(math.dist(points[i], points[i + 1])
                    for i in range(len(points) - 1))
        longest = run = 0.0
        previous_bearing: float | None = None
        for i in range(len(points) - 1):
            length = math.dist(points[i], points[i + 1])
            bearing = math.degrees(math.atan2(points[i + 1][0] - points[i][0],
                                              points[i][1] - points[i + 1][1]))
            if (previous_bearing is not None
                    and abs((bearing - previous_bearing + 180.0) % 360.0 - 180.0) > 5.0):
                run = 0.0
            run += length
            longest = max(longest, run)
            previous_bearing = bearing
        return {"path_length_m": round(total, 3),
                "longest_straight_run_m": round(longest, 3)}

    def _refresh_diagnostics(self) -> None:
        lines: list[str] = []
        if self.sim_map is None:
            lines.append("no map open")
        elif not self.waypoints:
            lines.append("click the map to place the first waypoint")
        else:
            geometry = self.geometry_diagnostics()
            lines.append(f"waypoints        {len(self.waypoints)}")
            lines.append(f"path length      {geometry['path_length_m']:.2f} m")
            lines.append(f"longest straight {geometry['longest_straight_run_m']:.2f} m")
            minimum = self._min_clearance()
            worst = None
            for leg in self._clearances():
                if worst is None or leg.clearance_m < worst.clearance_m:
                    worst = leg
                if leg.obstructed:
                    lines.append(f"leg {leg.index}: BLOCKED by map geometry")
                elif leg.clearance_m < minimum:
                    lines.append(f"leg {leg.index}: clears {leg.clearance_m:.2f} m,"
                                 f" under {minimum:.2f} m")
            if worst is not None:
                lines.append(f"tightest leg     {worst.clearance_m:.2f} m "
                             f"(leg {worst.index})")
            if worst is not None and not worst.obstructed \
                    and worst.clearance_m >= minimum:
                lines.append("every leg is clear, leg zero included")
        self.diagnostics.configure(state="normal")
        self.diagnostics.delete("1.0", tk.END)
        self.diagnostics.insert("1.0", "\n".join(lines))
        self.diagnostics.configure(state="disabled")
