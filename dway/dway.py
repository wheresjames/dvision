#!/usr/bin/env python3
"""dway -- fly a tour on a vehicle.

    ./dsim/dsim.py --id area1 --map ./assets/maps/maze_012.txt &
    ./dway/dway.py --id area1 --tour assets/tours/maze_012.forward.v1.json

The importable modules (`dway.link`, `dway.follower`, `dway.tour`,
`dway.mission`) hold everything a flight needs; this file adds a command line,
a window, and the loop that drives them.
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dcmn import theme
from dcmn.mapview import MapView
from dcmn.tktheme import apply_theme
from dvision2_common import validate_id
from dway.follower import PositionStrategy, Sample
from dway.link import DsimLink
from dway.mission import (
    FINISH_ACTIONS, TERMINAL_STATES, Mission, MissionConfig, MissionState,
    mission_report_dir,
)
from dway.report import FlightRecorder
from dway.tour import TourError, load_tour

# One palette, in dcmn.theme, so every window's map is the same colour.
_UI_BG = theme.BG
_UI_PANEL = theme.PANEL
_UI_CANVAS = theme.CANVAS
_UI_GRID = theme.GRID
_UI_TEXT = theme.TEXT
_UI_DIM = theme.DIM
_UI_ACCENT = theme.ACCENT
_UI_BUTTON = theme.BUTTON
_UI_BUTTON_ACTIVE = theme.BUTTON_ACTIVE
_UI_WARN = theme.WARN
_UI_DANGER = theme.DANGER
_UI_OK = theme.OK


def _describe_waypoint(waypoint) -> str:
    """A waypoint as a line to read, not a dict to decode."""
    described = waypoint.describe()
    frame = described.pop("frame")
    heading = described.pop("heading_deg")
    dwell = described.pop("dwell_s")
    coordinates = "  ".join(f"{name}={value:g}"
                            for name, value in described.items())
    return (f"{frame}  {coordinates}   heading {heading:g} deg"
            f"   dwell {dwell:g}s")


class Flight:
    """One mission, its link, and the artefacts it leaves behind."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.tour = load_tour(args.tour)
        self.link = DsimLink(args.id, client_id=args.client_id or f"dway-{args.id}",
                             ack_timeout_s=args.ack_timeout)
        self.mission = Mission(
            self.link, self.tour, root=ROOT,
            config=MissionConfig(
                strategy=args.strategy, speed_mps=args.speed,
                stream_hz=args.stream_hz, finish_action=args.finish_action,
                autostart=not args.wait_for_start,
            ))
        self.report_dir: Path | None = None
        self.running = True
        self._deadline = time.monotonic() + args.timeout if args.timeout else None

    # -- lifecycle ------------------------------------------------------

    def step(self) -> MissionState:
        state = self.mission.step()
        if self.report_dir is None and state is not MissionState.DISCONNECTED:
            self.report_dir = mission_report_dir(self.link, self.args.id)
            self.mission.recorder = FlightRecorder(self.report_dir)
            print(f"dway: report directory -> {self.report_dir}", file=sys.stderr)
        if (self._deadline is not None and time.monotonic() > self._deadline
                and state not in TERMINAL_STATES):
            self.mission.abort("mission timeout")
        return state

    def stop(self, reason: str = "shutdown") -> None:
        """Hold, keep the vehicle where it is, and give control back."""
        self.running = False
        self.mission.abort(reason)

    def finish(self) -> int:
        if self.mission.state not in TERMINAL_STATES:
            self.mission.abort("shutdown")
        else:
            self.mission.release()
        summary = self.mission.summary()
        if self.report_dir is not None:
            self.mission.write_report(self.report_dir)
        if self.mission.recorder is not None:
            self.mission.recorder.close()
        self.link.close()
        for warning in summary["warnings"]:
            print(f"dway: warning: {warning}", file=sys.stderr)
        print(f"dway: {summary['outcome']}"
              + (f" ({summary['reason']})" if summary["reason"] else "")
              + f"; {summary['waypoints_reached']}/{summary['waypoint_count']} "
                f"waypoints in {summary['duration_s']:.1f}s", file=sys.stderr)
        return 0 if summary["outcome"] == "complete" else 1

    def run_headless(self) -> int:
        period = 0.5 / max(self.args.stream_hz, 1.0)
        while self.running and self.mission.state not in TERMINAL_STATES:
            self.step()
            time.sleep(period)
        return self.finish()


# ---------------------------------------------------------------------------
# Fly tab
# ---------------------------------------------------------------------------

class FlyWindow:
    """Three tabs: fly the tour, read the vehicle, edit the plan.

    *Fly* is the map with the tour on it, the vehicle, the current leg and the
    controls. *Vehicle* answers "why will it not fly" from the published
    capability, health, ownership and failsafe facts, so that question never
    needs a log. *Tour editor* is the same widget an offline ``--edit``
    session opens, imported rather than copied.
    """

    def __init__(self, flight: Flight) -> None:
        import tkinter as tk
        from tkinter import ttk

        from dvision2_common import restore_window_pos

        self.flight = flight
        self.closed = False
        self.root = tk.Tk()
        self.root.title(f"dway {flight.args.id}")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        apply_theme(self.root)

        notebook = ttk.Notebook(self.root)
        notebook.grid(row=0, column=0, sticky="nsew")
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        fly = ttk.Frame(notebook, padding=8)
        notebook.add(fly, text="Fly")

        # The map arrives with preflight, so the canvas starts at a neutral
        # size and is resized to the map the first time one is available.
        self.view = MapView(cell=16, margin=12)
        self.canvas = tk.Canvas(fly, width=640, height=480,
                                background=_UI_CANVAS, highlightthickness=1,
                                highlightbackground=_UI_GRID)
        self.canvas.grid(row=0, column=0, rowspan=2, sticky="nw")

        info = ttk.Frame(fly, padding=(12, 0))
        info.grid(row=0, column=1, sticky="nw")
        self.vars = {}
        self.labels = {}
        for row, name in enumerate((
                "state", "strategy", "progress", "target", "pose", "telemetry",
                "vehicle", "notice")):
            ttk.Label(info, text=name, style="Dim.TLabel").grid(
                row=row, column=0, sticky="w", padx=(0, 10))
            var = tk.StringVar(value="-")
            self.vars[name] = var
            label = ttk.Label(info, textvariable=var, wraplength=360)
            label.grid(row=row, column=1, sticky="w")
            self.labels[name] = label

        # Three across, two down: six in a row set the window's minimum width
        # wider than the map beside them needs. The rows also group the
        # controls -- running the tour above, ending it below.
        buttons = ttk.Frame(fly, padding=(12, 12))
        buttons.grid(row=1, column=1, sticky="nw")
        for index, (text, action) in enumerate((
                ("Start", flight.mission.start), ("Pause", flight.mission.pause),
                ("Resume", flight.mission.resume), ("Hold", flight.mission.hold),
                ("RTL", flight.mission.rtl), ("Land", flight.mission.land))):
            row, column = divmod(index, 3)
            ttk.Button(buttons, text=text,
                       command=self._announce(text, action)).grid(
                row=row, column=column, padx=3, pady=3, sticky="ew")
        for column in range(3):
            buttons.columnconfigure(column, weight=1, uniform="control")

        self._map_drawn = False
        self._dynamic: list[int] = []

        vehicle = ttk.Frame(notebook, padding=12)
        notebook.add(vehicle, text="Vehicle")
        self.vehicle_vars: dict[str, tk.StringVar] = {}
        vehicle_fields = (
            ("blocking", "flight blocker"), ("ownership", "control"),
            ("capabilities", "capabilities"), ("estimators", "estimators"),
            ("gps", "GPS"), ("battery", "battery"), ("setpoint", "setpoint"),
            ("wind", "wind"), ("geofence", "geofence"),
        )
        for row, (name, label) in enumerate(vehicle_fields):
            ttk.Label(vehicle, text=label, style="Dim.TLabel").grid(
                row=row, column=0, sticky="nw", padx=(0, 14), pady=3)
            var = tk.StringVar(value="-")
            self.vehicle_vars[name] = var
            ttk.Label(vehicle, textvariable=var, wraplength=650).grid(
                row=row, column=1, sticky="nw", pady=3)
        vehicle.columnconfigure(1, weight=1)

        # The editor owns its own frame so it remains usable independently in
        # tests and by later clients that import it without the flight window.
        from dway.editor import TourEditor
        self.editor = TourEditor(notebook, root_dir=ROOT,
                                 on_status=lambda message:
                                 self.vars["notice"].set(message))
        notebook.add(self.editor.frame, text="Tour editor")
        restore_window_pos(self.root, f"dway.{flight.args.id}")

    def _announce(self, name: str, action):
        """Run a control and say so when the mission or vehicle refused it.

        A button that silently does nothing reads as a hung program; the
        commonest reason one is ignored -- the mission is not in a state that
        accepts it -- is exactly the thing worth putting on screen.
        """

        def run() -> None:
            outcome = action()
            refused = outcome is False or (
                hasattr(outcome, "accepted") and not outcome.accepted)
            if not refused:
                self.vars["notice"].set(f"{name} accepted")
                return
            detail = getattr(outcome, "reason", "") or (
                f"not available while {self.flight.mission.state.value}")
            self.vars["notice"].set(f"{name} ignored: {detail}")

        return run

    # -- drawing --------------------------------------------------------

    def _to_canvas(self, x: float, y: float) -> tuple[float, float]:
        return self.view.xy(x, y)

    def _draw_map(self) -> None:
        sim_map = self.flight.mission.sim_map
        if sim_map is None or self._map_drawn:
            return
        # The map arrives with preflight, so the view is sized here rather
        # than when the window was built. Cells are capped smaller than dsim's
        # because the map shares this tab with the telemetry column.
        self.view.fit(sim_map, max_cell_px=28)
        width, height = self.view.canvas_size(sim_map)
        self.canvas.config(width=width, height=height)
        self.view.draw_map(self.canvas, sim_map)
        planned = self.flight.mission.planned
        for index, (x, y) in enumerate(planned):
            cx, cy = self._to_canvas(x, y)
            if index:
                px, py = self._to_canvas(*planned[index - 1])
                self.canvas.create_line(px, py, cx, cy, fill=_UI_WARN, dash=(4, 3))
            self.canvas.create_oval(cx - 4, cy - 4, cx + 4, cy + 4,
                                    outline=_UI_WARN, width=2)
            self.canvas.create_text(cx + 9, cy - 9, text=str(index),
                                    fill=_UI_WARN, font=("TkDefaultFont", 7))
        self._map_drawn = True

    def _draw_vehicle(self) -> None:
        for item in self._dynamic:
            self.canvas.delete(item)
        self._dynamic = []
        mission = self.flight.mission
        track = mission.track
        if len(track) > 1:
            points: list[float] = []
            for x, y in track[-3000:]:
                points.extend(self._to_canvas(x, y))
            self._dynamic.append(self.canvas.create_line(*points, fill=_UI_ACCENT))
        state = mission.last_state
        if state is None:
            return
        try:
            x, y, _ = mission.context.ned_to_map(
                *mission.context.position_ned(state.position))
        except Exception:
            return
        cx, cy = self._to_canvas(x, y)
        self._dynamic.extend(self.view.draw_drone(
            self.canvas, x, y, state.heading_deg,
            crashed=state.mode == "CRASHED"))
        follower = mission.follower
        if follower is not None and follower.leg is not None:
            leg = follower.leg
            tx, ty, _ = mission.context.ned_to_map(leg.north_m, leg.east_m, leg.down_m)
            lx, ly = self._to_canvas(tx, ty)
            self._dynamic.append(self.canvas.create_line(
                cx, cy, lx, ly, fill=_UI_ACCENT, dash=(2, 2)))

    _STATE_COLORS = {"FAILED": _UI_DANGER, "PAUSED": _UI_WARN,
                     "FLYING": _UI_OK, "COMPLETE": _UI_OK}

    def _refresh_text(self) -> None:
        mission = self.flight.mission
        name = mission.state.value
        self.vars["state"].set(f"{name} -- {mission.reason}"
                               if mission.reason else name)
        self.labels["state"].configure(
            foreground=self._STATE_COLORS.get(name, _UI_TEXT))

        strategy, capabilities = mission.strategy, mission.capabilities
        if strategy is not None and capabilities is not None:
            chose = ("accepts_position_target=1"
                     if isinstance(strategy, PositionStrategy)
                     else "accepts_position_target=0, accepts_velocity_target=1")
            self.vars["strategy"].set(f"{strategy.describe()}  ({chose})")

        vehicle = mission.last_state
        follower = mission.follower
        if follower is not None:
            total = len(follower.legs)
            leg = follower.legs[min(follower.index, total - 1)]
            distance = "-"
            if vehicle is not None:
                sample = Sample.from_state(vehicle, mission.context, mission.clock())
                distance = f"{leg.distance_to(sample):.2f} m"
            self.vars["progress"].set(
                f"waypoint {min(follower.index + 1, total)} of {total}"
                f"   distance {distance}")
            self.vars["target"].set(
                f"{_describe_waypoint(leg.waypoint)}"
                f"   speed {mission.speed_mps:.2f} m/s")

        if vehicle is not None:
            position = vehicle.position
            self.vars["pose"].set(
                f"x={position.x:.2f} y={position.y:.2f} z={position.z:.2f} "
                f"heading={vehicle.heading_deg:.1f} deg")
            age = vehicle.last_setpoint_age_s
            self.vars["telemetry"].set(
                f"v=({vehicle.vx_mps:.2f}, {vehicle.vy_mps:.2f}, "
                f"{vehicle.vz_mps:.2f}) m/s   setpoint age "
                f"{'-' if age is None else f'{age:.2f}s'}")
            self.vars["vehicle"].set(
                f"mode {vehicle.mode}  armed {'yes' if vehicle.armed else 'no'}"
                f"  failsafe {vehicle.failsafe_reason or 'none'}")

        self._refresh_vehicle()

    def _refresh_vehicle(self) -> None:
        """Make the exact published fact blocking flight visible at a glance."""
        mission = self.flight.mission
        state = mission.last_state
        diagnostics = self.flight.link.diagnostics()
        capabilities = mission.capabilities
        blocker = mission.blocking_fact()
        self.vehicle_vars["blocking"].set(blocker or "none -- ready to fly")
        owner = diagnostics.get("control.owner", "")
        self.vehicle_vars["ownership"].set(
            f"owner {owner or 'none'}; this client {self.flight.link.client_id}")
        if capabilities is not None:
            self.vehicle_vars["capabilities"].set(
                f"{capabilities.vehicle}; frames {', '.join(capabilities.frames) or 'none'}; "
                f"position={int(capabilities.accepts_position_target)} "
                f"velocity={int(capabilities.accepts_velocity_target)}; "
                f"max {capabilities.max_speed_mps:g} m/s, "
                f"{capabilities.max_accel_mps2:g} m/s²")
        self.vehicle_vars["estimators"].set(
            "attitude={0} local={1} global={2} velocity={3}".format(
                diagnostics.get("est.attitude_valid", "?"),
                diagnostics.get("est.local_position_valid", "?"),
                diagnostics.get("est.global_position_valid", "?"),
                diagnostics.get("est.velocity_valid", "?")))
        self.vehicle_vars["gps"].set(
            f"fix {diagnostics.get('gps.fix_type', '?')}; "
            f"satellites {diagnostics.get('gps.satellites', '?')}; "
            f"HDOP {diagnostics.get('gps.hdop', '?')}, "
            f"VDOP {diagnostics.get('gps.vdop', '?')}")
        self.vehicle_vars["battery"].set(
            f"{diagnostics.get('drone.battery_pct', '?')}%")
        timeout = (capabilities.setpoint_timeout_s if capabilities else None)
        age = state.last_setpoint_age_s if state is not None else None
        self.vehicle_vars["setpoint"].set(
            f"age {'-' if age is None else f'{age:.2f}s'}; timeout "
            f"{'disabled' if timeout is None else f'{timeout:g}s'}")
        self.vehicle_vars["wind"].set(
            f"{diagnostics.get('wind.speed_mps', '?')} m/s from "
            f"{diagnostics.get('wind.dir_deg', '?')}°; gust setting "
            f"{diagnostics.get('wind.gust_mps', '?')} m/s")
        fence = diagnostics.get("geofence.box", "")
        self.vehicle_vars["geofence"].set(
            "none" if not fence else
            f"{fence}; action {diagnostics.get('geofence.action', '?')}")

    # -- loop -----------------------------------------------------------

    def tick(self) -> None:
        if self.closed:
            return
        self.flight.step()
        self._draw_map()
        self._draw_vehicle()
        self._refresh_text()
        if self.flight.mission.state in TERMINAL_STATES and self.flight.args.exit_on_finish:
            self.close()
            return
        self.root.after(int(500 / max(self.flight.args.stream_hz, 1.0)), self.tick)

    def run(self) -> int:
        self.root.after(50, self.tick)
        self.root.mainloop()
        return self.flight.finish()

    def close(self) -> None:
        from dvision2_common import save_window_pos

        if self.closed:
            return
        self.closed = True
        save_window_pos(self.root, f"dway.{self.flight.args.id}")
        self.root.quit()
        self.root.destroy()


# ---------------------------------------------------------------------------
# Offline editor
# ---------------------------------------------------------------------------

class EditorWindow:
    """A tour editor with no vehicle link, instance id, or flight lifecycle."""

    def __init__(self, *, map_path: str | None = None,
                 tour_path: str | None = None) -> None:
        import tkinter as tk

        from dvision2_common import restore_window_pos
        from dway.editor import TourEditor

        self.root = tk.Tk()
        self.root.title("dway tour editor")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        apply_theme(self.root)
        self.editor = TourEditor(self.root, root_dir=ROOT)
        self.editor.frame.grid(row=0, column=0, sticky="nsew")
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        if map_path:
            self.editor.load_map_file(Path(map_path))
        if tour_path:
            self.editor.load_tour_file(Path(tour_path))
        restore_window_pos(self.root, "dway.editor")

    def run(self) -> int:
        self.root.mainloop()
        return 0

    def close(self) -> None:
        from dvision2_common import save_window_pos

        save_window_pos(self.root, "dway.editor")
        self.root.destroy()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="dvision2 waypoint follower")
    parser.add_argument("--edit", action="store_true",
                        help="open the tour editor without connecting to a vehicle")
    parser.add_argument("--id", help="vehicle instance id (required for flight)")
    parser.add_argument("--tour", help="tour JSON file to fly or edit")
    parser.add_argument("--map", dest="edit_map",
                        help="map to open initially in offline editor mode")
    parser.add_argument("--strategy", choices=("auto", "position", "velocity"),
                        default="auto",
                        help="force a control strategy instead of negotiating one")
    parser.add_argument("--speed", type=float, default=None,
                        help="override the tour's default speed in m/s")
    parser.add_argument("--stream-hz", type=float, default=10.0,
                        help="setpoint stream rate (default: 10)")
    parser.add_argument("--finish-action", choices=FINISH_ACTIONS, default="land",
                        help="what to do once the last waypoint is reached")
    parser.add_argument("--wait-for-start", action="store_true",
                        help="stay in READY until Start is pressed")
    parser.add_argument("--client-id", default=None,
                        help="control-lease identity (default: dway-<id>)")
    parser.add_argument("--ack-timeout", type=float, default=1.0,
                        help="seconds to wait for a command acknowledgement")
    parser.add_argument("--timeout", type=float, default=0.0,
                        help="abort the flight after this many seconds")
    parser.add_argument("--exit-on-finish", action="store_true",
                        help="close the window when the flight ends")
    parser.add_argument("--no-ui", action="store_true",
                        help="run headless, for scripted flights")
    args = parser.parse_args(argv)
    if args.edit:
        if args.no_ui:
            parser.error("--edit cannot be combined with --no-ui")
        if args.id:
            parser.error("--edit does not use --id")
    else:
        if not args.id:
            parser.error("--id is required unless --edit is used")
        if not args.tour:
            parser.error("--tour is required unless --edit is used")
        if args.edit_map:
            parser.error("--map is only available with --edit")
        validate_id(args.id)
    if args.stream_hz <= 0.0:
        raise SystemExit("stream rate must be positive")
    if args.timeout < 0.0:
        raise SystemExit("timeout must not be negative")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.edit:
        try:
            import tkinter as tk
            window = EditorWindow(map_path=args.edit_map, tour_path=args.tour)
        except ImportError as exc:
            print(f"dway: tkinter is unavailable: {exc}", file=sys.stderr)
            return 2
        except tk.TclError as exc:
            print(f"dway: no display available: {exc}", file=sys.stderr)
            return 2
        except (OSError, TourError, ValueError) as exc:
            print(f"dway: cannot open editor input: {exc}", file=sys.stderr)
            return 2
        return window.run()

    try:
        flight = Flight(args)
    except TourError as exc:
        print(f"dway: {exc}", file=sys.stderr)
        return 2

    def stop(_signum, _frame):
        flight.stop("shutdown")

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    if args.no_ui:
        return flight.run_headless()
    try:
        import tkinter as tk
    except ImportError as exc:
        print(f"dway: tkinter is unavailable; use --no-ui: {exc}", file=sys.stderr)
        return 2
    try:
        window = FlyWindow(flight)
    except tk.TclError as exc:
        print(f"dway: no display available; use --no-ui: {exc}", file=sys.stderr)
        return 2
    return window.run()


if __name__ == "__main__":
    raise SystemExit(main())
