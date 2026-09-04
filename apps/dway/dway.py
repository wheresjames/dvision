#!/usr/bin/env python3
"""dway -- fly a tour on a vehicle.

    ./apps/dsim/dsim.py --id area1 --map ./assets/maps/maze_012.txt &
    ./apps/dway/dway.py --id area1 --tour assets/tours/maze_012.forward.v1.json

The importable modules (`dway.link`, `dway.follower`, `dway.tour`,
`dway.mission`) hold everything a flight needs; this file adds a command line,
a window, and the loop that drives them.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import signal
import sys
import time
import uuid
from pathlib import Path

# The repository root, two levels up now that the applications live under
# ``apps/``. ``apps`` itself is a source root rather than a package -- like a
# ``src/`` directory -- so sibling applications keep importing each other as
# ``dsim.dsim`` and ``dcmn.window`` with no ``apps.`` prefix anywhere.
ROOT = Path(__file__).resolve().parents[2]
APPS = ROOT / "apps"
for _path in (str(ROOT), str(APPS)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from dcmn import theme
from dcmn.health import IntakeMeter
from dcmn.module_bus import PymembusModuleBus, requests_shutdown
from dcmn.pacing import MAP_HZ, Paced, TEXT_HZ, simulated_poll_delay
from dcmn.mapview import MapView
from dcmn.tktheme import apply_theme
from dcmn.window import (disable_input_method, restore_window_pos,
                          save_window_pos)
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


def fit_fly_map(view: MapView, sim_map, *, max_cell_px: int = 28) -> int:
    """Fit the map into a square whose side follows its natural fitted height."""
    view.fit(sim_map, max_cell_px=max_cell_px)
    _width, height = view.canvas_size(sim_map)
    side = max(1, round(height))
    view.fit_canvas(sim_map, side, side)
    return side


def fly_canvas_side(width: int, height: int, controls_width: int) -> int:
    """Square map size available beside the controls in the Fly tab."""
    return max(120, min(max(1, height - 16),
                        max(1, width - controls_width - 28)))


class Flight:
    """One mission, its link, and the artefacts it leaves behind."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.tour = load_tour(args.tour)
        self.link = DsimLink(args.id, client_id=args.client_id or f"dway-{args.id}",
                             ack_timeout_s=args.ack_timeout)
        self.mission = Mission(
            self.link, self.tour, root=ROOT,
            # The flight is measured on the vehicle's clock, so a busy host --
            # or a simulator running faster or slower than real time -- cannot
            # shorten a dwell, expire a leg, or starve the setpoint stream.
            # `wall` stays the default: staleness is a fact about processes.
            clock=self.link.sim_time_s,
            config=MissionConfig(
                strategy=args.strategy, speed_mps=args.speed,
                stream_hz=args.stream_hz, finish_action=args.finish_action,
                autostart=False,
            ))
        self.report_dir: Path | None = None
        self.running = True
        self._deadline = time.monotonic() + args.timeout if args.timeout else None
        self.run_id = uuid.uuid4().hex
        self.bus = PymembusModuleBus(
            args.id, "navigator", "dway", sim_time=self.link.sim_time_s)
        self._prepare_sent = False
        self._ready_roles: set[str] = set()
        self._ready_processes: dict[str, set[str]] = {}
        self._ready_deadline: float | None = None
        self._start_sim_time: float | None = None
        self._started_announced = False
        self._terminal_announced = False
        self._last_bus_heartbeat = -1e9
        self._last_prepare = -1e9
        self._hello_sent = False
        # What dway owes the vehicle is a setpoint stream at --stream-hz; the
        # rate it actually achieves in simulated seconds is what a fast run
        # erodes, and what this reports.
        self.intake = IntakeMeter(float(args.stream_hz))
        self._streamed = 0

    # -- lifecycle ------------------------------------------------------

    def step(self) -> MissionState:
        state = self.mission.step()
        if self.report_dir is None and state is not MissionState.DISCONNECTED:
            self.report_dir = mission_report_dir(self.link, self.args.id)
            self.mission.recorder = FlightRecorder(self.report_dir)
            print(f"dway: report directory -> {self.report_dir}", file=sys.stderr)
        self._coordinate(state)
        if (self._deadline is not None and time.monotonic() > self._deadline
                and state not in TERMINAL_STATES):
            self.mission.abort("mission timeout")
        return state

    def _prepare_payload(self) -> dict:
        tour_bytes = self.tour.path.read_bytes()
        return {
            "tour_id": self.tour.tour_id,
            "tour_digest": hashlib.sha256(tour_bytes).hexdigest(),
            "map_digest": self.tour.map_sha or "",
            "coordinate_frame": self.tour.coordinate_frame,
            "required_roles": list(self.args.wait_for),
        }

    def _coordinate(self, state: MissionState) -> None:
        if not self.bus.connect():
            return
        now = self.link.sim_time_s()
        if not self._hello_sent:
            self.bus.publish("module.hello", run_id=self.run_id, payload={
                "state": state.value, "capabilities": ["waypoint_navigation"]})
            self._hello_sent = True
        for event in self.bus.receive():
            if requests_shutdown(event):
                self.stop("instance shutdown requested")
                return
            if event.run_id != self.run_id:
                continue
            if event.type == "run.ready":
                for requirement in self.args.wait_for:
                    role, _, selector = requirement.partition(":")
                    matches_selector = (not selector or selector in (
                        event.payload.get("profile"), event.payload.get("algorithm"),
                        event.process_id) or selector in event.payload.get(
                            "capabilities", {}).get("algorithms", ()))
                    if event.role == role and matches_selector:
                        self._ready_roles.add(requirement)
                        matches = self._ready_processes.setdefault(requirement, set())
                        matches.add(event.process_id)
                        if len(matches) > 1:
                            self.mission.abort(
                                f"ambiguous exclusive participant: {requirement}")
            elif event.type == "run.reject":
                self.mission.abort(
                    f"{event.role} rejected run: "
                    f"{event.payload.get('reason', 'unspecified reason')}")
        if now - self._last_bus_heartbeat >= 1.0:
            sent = getattr(self.mission, "setpoints_sent", 0)
            self.intake.record(max(0, sent - self._streamed))
            self._streamed = sent
            # Only FLYING streams waypoint setpoints. Waiting, takeoff and a
            # deliberate pause are not failures to meet the stream target.
            self.intake.set_wanted(float(self.args.stream_hz)
                                   if state is MissionState.FLYING else 0.0)
            self.bus.publish("module.heartbeat", run_id=self.run_id, payload={
                "intake": self.intake.report(now, overruns=self.bus.overruns),
                "state": state.value, "ready": state is MissionState.READY,
                "capabilities": ["waypoint_navigation"],
            })
            self.bus.publish("run.state", run_id=self.run_id, payload={
                "state": state.value, "tour_id": self.tour.tour_id,
                "start_sim_time_s": self._start_sim_time,
                "waypoint_index": getattr(self.mission, "_current_index", -1),
                "waypoint_count": len(self.tour.waypoints),
            })
            self._last_bus_heartbeat = now
        if state is MissionState.READY and not self.args.wait_for_start:
            if not self._prepare_sent:
                self._prepare_sent = True
                self._ready_deadline = now + self.args.ready_timeout_s
            # Repeat the complete preparation snapshot for late joiners.
            if now - self._last_prepare >= 1.0:
                self.bus.publish("run.prepare", run_id=self.run_id,
                                 payload=self._prepare_payload())
                self._last_prepare = now
            missing = set(self.args.wait_for) - self._ready_roles
            if missing and self._ready_deadline is not None \
                    and now >= self._ready_deadline:
                self.mission.abort("readiness timeout waiting for "
                                   + ", ".join(sorted(missing)))
            elif not missing and self._start_sim_time is None:
                self._start_sim_time = now + self.args.start_delay_s
                self.bus.publish("run.start_scheduled", run_id=self.run_id,
                                 payload={"start_sim_time_s": self._start_sim_time})
            elif self._start_sim_time is not None and now >= self._start_sim_time:
                if self.mission.start():
                    self._started_announced = True
                    self.bus.publish("run.started", run_id=self.run_id,
                                     payload={"start_sim_time_s": self._start_sim_time})
        if state in TERMINAL_STATES and not self._terminal_announced:
            self._terminal_announced = True
            kind = "run.completed" if state is MissionState.COMPLETE else "run.aborted"
            self.bus.publish(kind, run_id=self.run_id, payload={
                "state": state.value, "outcome": self.mission.outcome,
                "reason": self.mission.reason,
            })

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
        self.bus.publish("module.goodbye", run_id=self.run_id,
                         payload={"state": self.mission.state.value})
        self.bus.close()
        for warning in summary["warnings"]:
            print(f"dway: warning: {warning}", file=sys.stderr)
        print(f"dway: {summary['outcome']}"
              + (f" ({summary['reason']})" if summary["reason"] else "")
              + f"; {summary['waypoints_reached']}/{summary['waypoint_count']} "
                f"waypoints in {summary['duration_s']:.1f}s", file=sys.stderr)
        return 0 if summary["outcome"] == "complete" else 1

    def run_headless(self) -> int:
        while self.running and self.mission.state not in TERMINAL_STATES:
            self.step()
            time.sleep(simulated_poll_delay(
                max(self.args.stream_hz, 1.0), self.link.diagnostics()))
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

        self.flight = flight
        self.closed = False
        # Painting is capped on the wall clock; flight.step() is not.
        self._paint_map = Paced(MAP_HZ)
        self._paint_text = Paced(TEXT_HZ)
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
        self.fly = fly

        # The map arrives with preflight, so the canvas starts at a neutral
        # size and is resized to the map the first time one is available.
        self.view = MapView(cell=16, margin=12)
        self.canvas = tk.Canvas(fly, width=480, height=480,
                                background=_UI_CANVAS, highlightthickness=1,
                                highlightbackground=_UI_GRID)
        self.canvas.grid(row=0, column=0, rowspan=2, sticky="nw")

        info = ttk.Frame(fly, padding=(12, 0))
        self.info = info
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
        self.buttons = buttons
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
        self._map_side = 480
        self._dynamic: list[int] = []
        fly.bind("<Configure>", self._resize_map)

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

    def _draw_map(self, *, force: bool = False, side: int | None = None) -> None:
        sim_map = self.flight.mission.sim_map
        if sim_map is None or (self._map_drawn and not force):
            return
        # Keep the Fly tab compact for wide maps: use the fitted map height as
        # both canvas dimensions, then contain and centre the complete map.
        if side is None:
            side = fit_fly_map(self.view, sim_map)
        else:
            self.view.fit_canvas(sim_map, side, side)
        self._map_side = side
        self.canvas.config(width=side, height=side)
        self.canvas.delete("map-static")
        self.view.draw_map(self.canvas, sim_map, tags="map-static")
        planned = self.flight.mission.planned
        for index, (x, y) in enumerate(planned):
            cx, cy = self._to_canvas(x, y)
            if index:
                px, py = self._to_canvas(*planned[index - 1])
                self.canvas.create_line(px, py, cx, cy, fill=_UI_WARN,
                                        dash=(4, 3), tags="map-static")
            self.canvas.create_oval(cx - 4, cy - 4, cx + 4, cy + 4,
                                    outline=_UI_WARN, width=2, tags="map-static")
            self.canvas.create_text(cx + 9, cy - 9, text=str(index),
                                    fill=_UI_WARN, font=("TkDefaultFont", 7),
                                    tags="map-static")
        self._map_drawn = True

    def _resize_map(self, event) -> None:
        sim_map = self.flight.mission.sim_map
        if sim_map is None: return
        controls_width = max(self.info.winfo_reqwidth(), self.buttons.winfo_reqwidth())
        side = fly_canvas_side(event.width, event.height, controls_width)
        if abs(side - self._map_side) < 2: return
        self._draw_map(force=True, side=side)
        self._draw_vehicle()

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
        if not self.flight.running:
            self.close()
            return
        # flight.step() above is the mission; everything here is for looking
        # at, and is capped on the wall clock.
        if self._paint_map.due():
            self._draw_map()
            self._draw_vehicle()
        if self._paint_text.due():
            self._refresh_text()
        if self.flight.mission.state in TERMINAL_STATES and self.flight.args.exit_on_finish:
            self.close()
            return
        delay = simulated_poll_delay(
            max(self.flight.args.stream_hz, 1.0),
            self.flight.link.diagnostics())
        # Tk accepts whole milliseconds; round upward so this polling hint
        # never creates an early extra iteration.
        self.root.after(max(1, math.ceil(delay * 1000)), self.tick)

    def run(self) -> int:
        self.root.after(50, self.tick)
        self.root.mainloop()
        return self.flight.finish()

    def close(self) -> None:
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
    parser.add_argument("--wait-for", action="append", default=[], metavar="ROLE",
                        help="require a module role to acknowledge this run")
    parser.add_argument("--ready-timeout-s", type=float, default=15.0,
                        help="simulated seconds to wait for required modules")
    parser.add_argument("--start-delay-s", type=float, default=3.0,
                        help="simulated seconds between readiness and start")
    parser.add_argument("--client-id", default=None,
                        help="control-lease identity (default: dway-<id>)")
    parser.add_argument("--ack-timeout", type=float, default=3.0,
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
    if args.ready_timeout_s <= 0.0 or args.start_delay_s < 0.0:
        raise SystemExit("readiness timeout must be positive and start delay non-negative")
    return args


def main(argv: list[str] | None = None) -> int:
    disable_input_method()
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
