#!/usr/bin/env python3
"""dnav: the route planner's window, and the headless run behind it.

Three tabs and a status bar, assembled almost entirely from dcmn parts. Plan is
where the work is: a map pane showing the combined cost surface with the route
over it, and a side panel of the same `LabelFrame` groups `dctl` uses, so one
glance reads the same across the applications. Cost is the policy debugger --
the tab that makes threshold and inflation tunable numbers instead of magic,
which is the whole reason the policy is an artifact. Events is `dcmn`'s viewer,
unchanged.

dnav holds no control lease and sends no command. It publishes what it plans
and shows what it published.
"""

from __future__ import annotations

import sys
import time
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

from dvision2_common import validate_id
from dcmn import theme
from dcmn.health import DOTS
from dcmn.pacing import MAP_HZ, Paced, TEXT_HZ
from dnav import route as R
from dcmn.navigation import permitted_points
from dnav.plan import NavRun, parse_goal
from dnav.planners import DEFAULT_PLANNER, PLANNERS
from dnav.policy import load_policy

#: What the Cost tab can put in its pane. One entry per evidence layer, one per
#: derived cost layer, one per source showing the cells its cost blocks tinted
#: over its evidence, and the combined stack the planner actually searched.
STACK = 'stack (combined)'
EVIDENCE_VIEW = 'evidence: '
COST_VIEW = 'cost: '
BLOCKED_VIEW = 'blocked on evidence: '


def parse_args(argv):
    import argparse

    from dalg.dalg import join_negative_values
    argv = join_negative_values(argv, ("--goal",))
    parser = argparse.ArgumentParser(description="dvision2 route planner")
    parser.add_argument("--id", required=True, help="instance id")
    parser.add_argument("--goal", default=None,
                        help="goal as x,y or x,y,z in local-frame metres, submitted to the "
                             "session context as this dnav's goal authority")
    parser.add_argument("--no-provider-goal", action="store_true",
                        help="without --goal, do not adopt the map target the simulator publishes")
    parser.add_argument("--goal-role", choices=("ui", "mission"), default="ui",
                        help="the authority role this dnav claims for goals it submits")
    parser.add_argument("--take-goal-authority", action="store_true",
                        help="take the goal from its current authority (an explicit handoff)")
    parser.add_argument("--pose-max-age", type=float, default=0.5,
                        help="freshness limit for the vehicle pose, in data-clock seconds")
    parser.add_argument("--policy", default="default",
                        help="cost policy name or path")
    parser.add_argument("--planner", default=DEFAULT_PLANNER,
                        choices=sorted(PLANNERS))
    parser.add_argument("--execution-profile",
                        help="execution profile JSON path or name under assets/execution_profiles "
                             "(e.g. sim-target); dway must load the same profile")
    parser.add_argument("--profile-set", action="append", default=[], metavar="KEY=VALUE",
                        help="override one execution profile field (repeatable; pass the same to dway)")
    parser.add_argument("--navigation-name", default="dnav", help="selected navigation publisher name")
    parser.add_argument("--source", default=None,
                        help="plan on one evidence source instead of every one")
    parser.add_argument("--show-reference", action="store_true",
                        help="open with the reference-image background enabled "
                             "(debug display; recorded in the session provenance)")
    parser.add_argument("--no-ui", action="store_true",
                        help="run headless and print each route as it changes")
    parser.add_argument("--timeout", type=float, default=0.0,
                        help="stop after this many wall seconds; 0 runs until closed")
    args = parser.parse_args(argv)
    validate_id(args.id)
    if args.timeout < 0: parser.error("--timeout may not be negative")
    if args.pose_max_age <= 0: parser.error("--pose-max-age must be positive")
    return args


def _metres(value, digits=2):
    return "--" if value is None else f"{value:.{digits}f}"


def describe_detour(route, detour):
    """The plan against its straight-line control, in words, never a crash.

    Either ratio can be missing: a blocked control has no finite cost, and a
    vehicle already standing at its goal has a control of zero length -- which
    is exactly where every tour ends, so the readout meets it on every run.
    """
    if detour is None: return "--"
    if detour["control_blocked"]:
        return f"straight line blocked; plan {route.length_m:.2f} m"
    length, cost = detour.get("length_ratio"), detour.get("cost_ratio")
    if length is None or cost is None:
        return "at the goal"
    return f"{length:.2f}x length, {cost:.2f}x cost"


def _short_report(directory):
    """The run and module directory, without the absolute path in front of it."""
    if directory is None: return "waiting for the session provider"
    parts = Path(directory).parts
    return "/".join(parts[-3:]) if len(parts) >= 3 else str(directory)


def _cost(value):
    import math
    if value is None: return "--"
    return "blocked" if math.isinf(value) else f"{value:.2f}"


class PlanTab:
    """The map, the goal, the planner, and what the plan came out as."""

    def __init__(self, parent, run, tk, ttk, *, show_reference=False):
        from dcmn.map_pane import MapPane, ReferenceBackgroundHost
        from dcmn.scroll import Scrollable

        self.run, self.tk, self.ttk = run, tk, ttk
        page = ttk.Frame(parent)
        page.grid(row=0, column=0, sticky="nsew")
        parent.rowconfigure(0, weight=1); parent.columnconfigure(0, weight=1)
        page.columnconfigure(0, weight=1); page.rowconfigure(1, weight=1)
        # The reference background is opt-in and its every change recorded:
        # a truth map behind live evidence marks the session as assisted,
        # and the provenance entry is how that marking
        # survives past the window. One host serves both map tabs.
        self.reference = ReferenceBackgroundHost(
            page, run.id, enabled=show_reference, on_change=self._reference_changed)
        self.reference.widget.grid(row=0, column=0, columnspan=2, sticky="ew")
        self.pane = MapPane(page, mode="cost", controls=False,
                            title="cost and route", on_click=self._clicked,
                            width=440, height=440)
        self.pane.widget.grid(row=1, column=0, sticky="nsew", padx=(0, 6))
        # Four groups is more than a short window has room for, and a side
        # panel that sets the height of the whole window is how the map ends
        # up smaller than the form beside it.
        self.scroll = Scrollable(page)
        self.scroll.outer.grid(row=1, column=1, sticky="ns")
        side = ttk.Frame(self.scroll.inner)
        side.grid(row=0, column=0, sticky="ns")

        goal = ttk.LabelFrame(side, text="Goal", padding=8)
        goal.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        self.goal_vars = {}
        for row, name in enumerate(("x", "y", "z")):
            ttk.Label(goal, text=f"{name} (m)", style="Dim.TLabel").grid(
                row=row, column=0, sticky="w", padx=(0, 8))
            var = tk.StringVar(value="")
            self.goal_vars[name] = var
            entry = ttk.Entry(goal, textvariable=var, width=10)
            entry.grid(row=row, column=1, sticky="ew")
            entry.bind("<Return>", lambda _e: self._set_goal())
        ttk.Label(goal, text="or click the map", style="Dim.TLabel").grid(
            row=3, column=0, columnspan=2, sticky="w", pady=(4, 4))
        ttk.Button(goal, text="Set", style="Accent.TButton",
                   command=self._set_goal).grid(row=4, column=0, sticky="ew")
        ttk.Button(goal, text="Clear", command=self._clear_goal).grid(
            row=4, column=1, sticky="ew")
        self.handoff = tk.BooleanVar(value=False)
        ttk.Checkbutton(goal, text="take goal authority", variable=self.handoff).grid(
            row=5, column=0, columnspan=2, sticky="w", pady=(4, 0))
        self.authority = tk.StringVar(value="--")
        ttk.Label(goal, textvariable=self.authority, style="Dim.TLabel", wraplength=210,
                  justify="left").grid(row=6, column=0, columnspan=2, sticky="w")
        goal.columnconfigure(1, weight=1)

        planner = ttk.LabelFrame(side, text="Planner", padding=8)
        planner.grid(row=1, column=0, sticky="ew", pady=(0, 6))
        self.planner_var = tk.StringVar(value=run.planner_name)
        combo = ttk.Combobox(planner, textvariable=self.planner_var,
                             values=sorted(PLANNERS), state="readonly", width=12)
        combo.grid(row=0, column=0, columnspan=2, sticky="ew")
        combo.bind("<<ComboboxSelected>>", self._planner_changed)
        self.auto = tk.BooleanVar(value=True)
        ttk.Checkbutton(planner, text="re-plan on new evidence",
                        variable=self.auto).grid(row=1, column=0, columnspan=2,
                                                 sticky="w", pady=(4, 4))
        ttk.Button(planner, text="Replan now",
                   command=lambda: run.replan(force=True)).grid(
            row=2, column=0, columnspan=2, sticky="ew")
        planner.columnconfigure(0, weight=1)

        self.route_rows = self._readout(side, 2, "Route", (
            "status", "reason", "waypoints", "length", "cost", "vs straight line",
            "planning"))
        self.source_rows = self._readout(side, 3, "Map source", (
            "producer", "source", "revision", "age", "state", "cell", "extent"))
        self.show_goal()
        for widget in (combo,):
            self.scroll.claim_wheel(widget)

    def _readout(self, side, row, title, names):
        frame = self.ttk.LabelFrame(side, text=title, padding=8)
        frame.grid(row=row, column=0, sticky="ew", pady=(0, 6))
        frame.columnconfigure(1, weight=1)
        rows = {}
        for index, name in enumerate(names):
            self.ttk.Label(frame, text=name, style="Dim.TLabel").grid(
                row=index, column=0, sticky="nw", padx=(0, 8))
            var = self.tk.StringVar(value="--")
            label = self.ttk.Label(frame, textvariable=var, anchor="w",
                                   wraplength=210, justify="left")
            label.grid(row=index, column=1, sticky="ew")
            rows[name] = var
        return rows

    # -- input ---------------------------------------------------------

    def show_goal(self):
        """Put the current goal into the entries, whoever set it.

        The goal lives in the session context: one given on the command line,
        by a mission coordinator or typed here is equally real, and a form that
        showed another authority's goal as blank would make the planner look
        like it was working on nothing.
        """
        run = self.run
        self.goal_vars["x"].set("" if run.goal_xy is None else f"{run.goal_xy[0]:.2f}")
        self.goal_vars["y"].set("" if run.goal_xy is None else f"{run.goal_xy[1]:.2f}")
        self.goal_vars["z"].set("" if run.goal_z is None else f"{run.goal_z:.2f}")

    def _clicked(self, x_m, y_m):
        self.goal_vars["x"].set(f"{x_m:.2f}")
        self.goal_vars["y"].set(f"{y_m:.2f}")
        self._set_goal()

    def _set_goal(self):
        try:
            x = float(self.goal_vars["x"].get())
            y = float(self.goal_vars["y"].get())
        except ValueError:
            self.route_rows["reason"].set("goal x and y must be numbers in metres")
            return
        text = self.goal_vars["z"].get().strip()
        self.run.set_goal(x, y, float(text) if text else None, handoff=self.handoff.get())
        self.handoff.set(False)
        self.run.replan(force=True)

    def _clear_goal(self):
        self.run.clear_goal(handoff=self.handoff.get())
        self.handoff.set(False)
        self.show_goal()
        self.run.replan(force=True)

    def _planner_changed(self, _event=None):
        self.run.set_planner(self.planner_var.get())
        self.run.replan(force=True)

    def _reference_changed(self, enabled, opacity):
        self.run.note_display("reference-on" if enabled else "reference-off",
                              opacity=round(opacity, 3))

    # -- painting ------------------------------------------------------

    def sync(self, sim_now):
        """Put the run's current state into the pane, without drawing it.

        Kept apart from painting because the report needs the one without the
        other. The pane is only repainted while its tab is visible, so a report
        that snapshotted whatever was last painted would file a picture from
        whenever the operator last looked at this tab -- a route from the
        middle of the flight under a summary describing its end.
        """
        run = self.run
        grids = run._grids()
        source = sorted(grids)[0] if grids else None
        self.pane.set_grid(None if source is None else grids[source],
                           sim_now_s=sim_now,
                           horizon_s=run.session.staleness_horizon_s)
        self.pane.set_cost(None if run.cost_map is None else run.cost_map.cost,
                           never_observed=None if run.cost_map is None else run.cost_map.never_observed)
        # The background goes in here, not in paint_map, so the report
        # snapshot -- which calls sync -- archives exactly what the operator
        # was shown, reference revision and all.
        self.pane.set_background(self.reference.background(), refresh=False)
        vehicle = run.vehicle()
        self.pane.set_overlay(
            route=tuple(run.route.points), proposal=True,
            permitted=tuple(p[:2] for p in permitted_points(run.navigation.last)),
            vehicle=None if vehicle is None else (vehicle[0], vehicle[1], vehicle[3]),
            goal=None if run.goal_xy is None else run.goal_xy,
            start=None if vehicle is None else (vehicle[0], vehicle[1]),
            inflation_m=run.policy.inflation_m)

    def paint_map(self, sim_now):
        self.sync(sim_now)
        self.pane.refresh()

    def snapshot(self):
        """The report image: this pane's renderer, over the run's final state."""
        self.sync(self.run.sim_time_s())
        return self.pane.snapshot()

    def displayed_background(self):
        """The background the pane last rendered -- the exact revision a report filed."""
        return self.pane.state.background

    def paint_text(self, sim_now):
        run, route = self.run, self.run.route
        detour = run.summary_detour()
        authority = run.authority
        goal = run.goal_descriptor
        self.authority.set(
            (run.goal_error + "\n" if run.goal_error else "")
            + ("authority: none" if not authority else
               f"authority: {authority['role']} {'(this dnav)' if authority['id'] == run.writer_id else authority['id'][:18]}")
            + ("" if goal is None else f"\nrevision {goal['revision']}, epoch {goal['localization_epoch']}"))
        if self.run.goal_xy != getattr(self, "_shown_goal", None):
            self._shown_goal = self.run.goal_xy
            self.show_goal()
        self.route_rows["status"].set(route.status)
        self.route_rows["reason"].set(route.reason or "planned")
        self.route_rows["waypoints"].set(str(len(route.waypoints)))
        self.route_rows["length"].set(f"{route.length_m:.2f} m" if route.ok else "--")
        self.route_rows["cost"].set(_cost(route.cost if route.ok else None))
        self.route_rows["vs straight line"].set(describe_detour(route, detour))
        self.route_rows["planning"].set(
            f"{route.plan_time_s * 1000:.1f} ms, {route.expanded} cells")

        rows = run.source_rows(sim_now)
        geometry = run.session.geometry
        self.source_rows["producer"].set(
            run.session.manifest.get("producer", "--") or "--")
        self.source_rows["source"].set(
            ", ".join(row["source"] for row in rows) or "none")
        self.source_rows["revision"].set(
            ", ".join(str(row["revision"]) for row in rows) or "--")
        self.source_rows["age"].set(
            ", ".join(_metres(row["age_s"], 1) + " s" for row in rows) or "--")
        self.source_rows["state"].set(
            "stale" if any(row["stale"] for row in rows) else
            ("fresh" if rows else "no producer"))
        self.source_rows["cell"].set(
            "--" if geometry is None else f"{geometry.cell_m:g} m")
        self.source_rows["extent"].set(
            "--" if geometry is None else
            f"{geometry.extent_m[0]:g} x {geometry.extent_m[1]:g} m")


class CostTab:
    """The policy debugger: every layer, what it costs, and why."""

    def __init__(self, parent, run, tk, ttk, *, reference=None):
        from dcmn.map_pane import MapPane
        from dcmn.scroll import Scrollable

        self.run, self.tk, self.ttk, self.reference = run, tk, ttk, reference
        page = ttk.Frame(parent)
        page.grid(row=0, column=0, sticky="nsew")
        parent.rowconfigure(0, weight=1); parent.columnconfigure(0, weight=1)
        page.columnconfigure(0, weight=1); page.rowconfigure(0, weight=1)
        self.pane = MapPane(page, mode="cost", controls=False,
                            title=STACK, width=440, height=440)
        self.pane.widget.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        self.scroll = Scrollable(page)
        self.scroll.outer.grid(row=0, column=1, sticky="ns")
        side = ttk.Frame(self.scroll.inner)
        side.grid(row=0, column=0, sticky="ns")

        view = ttk.LabelFrame(side, text="Layer", padding=8)
        view.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        self.view_var = tk.StringVar(value=STACK)
        self.view_combo = ttk.Combobox(view, textvariable=self.view_var,
                                       values=(STACK,), state="readonly", width=22)
        self.view_combo.grid(row=0, column=0, sticky="ew")
        # The wheel over the selector scrolls the panel rather than silently
        # changing which layer is on screen.
        self.scroll.claim_wheel(self.view_combo)
        ttk.Label(view, style="Dim.TLabel", wraplength=210, justify="left",
                  text="evidence is what a sensor believes; cost is what this "
                       "policy makes of it. Blocked cells are drawn in the "
                       "danger colour, never-observed in its own.").grid(
            row=1, column=0, sticky="ew", pady=(6, 0))
        view.columnconfigure(0, weight=1)

        policy = ttk.LabelFrame(side, text="Cost policy", padding=8)
        policy.grid(row=1, column=0, sticky="ew", pady=(0, 6))
        self.policy_rows = {}
        for index, name in enumerate(("name", "digest", "occupied_threshold",
                                      "inflation_m", "combine", "free_cost",
                                      "unobserved_cost", "blocked cells")):
            ttk.Label(policy, text=name, style="Dim.TLabel").grid(
                row=index, column=0, sticky="w", padx=(0, 8))
            var = tk.StringVar(value="--")
            self.policy_rows[name] = var
            ttk.Label(policy, textvariable=var, anchor="w").grid(
                row=index, column=1, sticky="ew")
        self.notice = tk.StringVar(value="the file is the source of truth")
        ttk.Button(policy, text="Reload", command=self._reload).grid(
            row=9, column=0, columnspan=2, sticky="ew", pady=(6, 4))
        ttk.Label(policy, textvariable=self.notice, style="Dim.TLabel",
                  wraplength=210, justify="left").grid(
            row=10, column=0, columnspan=2, sticky="w")
        policy.columnconfigure(1, weight=1)

    def _reload(self):
        try:
            policy = self.run.reload_policy()
            self.run.replan(force=True)
            self.notice.set(f"reloaded {policy.digest[:12]}")
        except (OSError, ValueError) as exc:
            self.notice.set(f"not reloaded: {exc}")

    def _choices(self):
        sources = self.run.sources()
        return ([STACK] + [EVIDENCE_VIEW + sid for sid in sources]
                + [COST_VIEW + sid for sid in sources]
                + [BLOCKED_VIEW + sid for sid in sources])

    def paint_map(self, sim_now):
        run = self.run
        choices = self._choices()
        if tuple(self.view_combo["values"]) != tuple(choices):
            self.view_combo.configure(values=choices)
            if self.view_var.get() not in choices: self.view_var.set(STACK)
        selection = self.view_var.get()
        grids = run._grids()
        view, source = STACK, (sorted(grids)[0] if grids else None)
        for prefix in (EVIDENCE_VIEW, COST_VIEW, BLOCKED_VIEW):
            if selection.startswith(prefix):
                view, source = prefix, selection[len(prefix):]
        grid = grids.get(source) if source else None
        self.pane.set_grid(grid, sim_now_s=sim_now,
                           horizon_s=run.session.staleness_horizon_s)
        self.pane.set_header(self._header(selection, view, source))
        layer = (None if run.cost_map is None or view == STACK
                 else run.cost_map.layer(source))
        if view == EVIDENCE_VIEW:
            self.pane.set_mode("occupancy", refresh=False)
            self.pane.set_cost(None)
        elif view == BLOCKED_VIEW:
            # The source's own margin over its own evidence: the combined stack
            # would tint cells another sensor blocked, which is not what this
            # source's picture is asking about.
            self.pane.set_mode("blocked", refresh=False)
            self.pane.set_cost(None if layer is None else layer.cost)
        else:
            self.pane.set_mode("cost", refresh=False)
            self.pane.set_cost(
                None if run.cost_map is None else
                (layer.cost if layer is not None else run.cost_map.cost),
                never_observed=(None if run.cost_map is None else
                                layer.never_observed if layer is not None else run.cost_map.never_observed))
        vehicle = run.vehicle()
        self.pane.set_overlay(
            route=tuple(run.route.points), proposal=True,
            permitted=tuple(p[:2] for p in permitted_points(run.navigation.last)),
            vehicle=None if vehicle is None else (vehicle[0], vehicle[1], vehicle[3]),
            goal=None if run.goal_xy is None else run.goal_xy,
            inflation_m=run.policy.inflation_m)
        # The Plan tab's host, so both tabs show the same background at the
        # same opacity and one poll serves both panes.
        self.pane.set_background(None if self.reference is None else self.reference.background(),
                                 refresh=False)
        # Forced: the window already paces this call to MAP_HZ, and the mode,
        # grid and cost were all just set, so there is exactly one correct
        # picture to draw and no reason to let a second budget skip it.
        self.pane.refresh(force=True)

    def _header(self, selection, view, source):
        """The pane's title, saying so when two views are one by construction.

        The stack is the max over every source's cost layer, so with a single
        source it is that layer exactly. Two identical pictures under two names
        read as a bug; naming the reason is what keeps them from reading that
        way until a second sensor makes them differ.
        """
        cost_map = self.run.cost_map
        if view == STACK and cost_map is not None and len(cost_map.layers) == 1:
            return f"{STACK} — one source, so identical to {COST_VIEW}{cost_map.layers[0].source}"
        if view == BLOCKED_VIEW:
            return f"{selection} — tinted cells are blocked by the policy"
        return selection

    def paint_text(self, sim_now):
        policy = self.run.policy
        values = policy.values()
        self.policy_rows["name"].set(policy.name)
        self.policy_rows["digest"].set(policy.digest[:12] or "--")
        for name in ("occupied_threshold", "inflation_m", "combine",
                     "free_cost", "unobserved_cost"):
            self.policy_rows[name].set(str(values[name]))
        cost_map = self.run.cost_map
        self.policy_rows["blocked cells"].set(
            "--" if cost_map is None else
            f"{int(cost_map.blocked.sum())} of {cost_map.cost.size}")


class Window:
    """The dnav window: Plan, Cost, Events, and a status bar."""

    def __init__(self, run, *, show_reference=False):
        import tkinter as tk
        from tkinter import ttk
        from dcmn.event_viewer import EventViewer
        from dcmn.tktheme import apply_theme
        from dcmn.window import restore_window_geometry, save_window_geometry

        self.tk, self.ttk, self.run = tk, ttk, run
        self._save_geometry = save_window_geometry
        self.root = tk.Tk()
        apply_theme(self.root)
        self.root.title(f"dnav route planner — {run.id}")
        self.root.geometry("1120x800")
        self.root.minsize(720, 480)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.running = True

        notebook = ttk.Notebook(self.root)
        notebook.grid(row=0, column=0, sticky="nsew")
        plan_page = ttk.Frame(notebook, padding=8)
        notebook.add(plan_page, text="Plan")
        cost_page = ttk.Frame(notebook, padding=8)
        notebook.add(cost_page, text="Cost")
        events_page = ttk.Frame(notebook)
        notebook.add(events_page, text="Events")
        self.notebook = notebook
        self.plan = PlanTab(plan_page, run, tk, ttk, show_reference=show_reference)
        self.cost = CostTab(cost_page, run, tk, ttk, reference=self.plan.reference)
        self.events = EventViewer(events_page, run.id)
        self.events.page.grid(row=0, column=0, sticky="nsew")
        events_page.rowconfigure(0, weight=1); events_page.columnconfigure(0, weight=1)
        if show_reference:
            run.note_display("reference-on", source="--show-reference")

        self.status = tk.StringVar(value="connecting")
        ttk.Label(self.root, textvariable=self.status, style="Dim.TLabel",
                  anchor="w").grid(row=1, column=0, sticky="ew", padx=8, pady=(2, 6))
        self.root.columnconfigure(0, weight=1); self.root.rowconfigure(0, weight=1)
        self._paint_map = Paced(MAP_HZ)
        self._paint_text = Paced(TEXT_HZ)
        run.snapshot_provider = self.plan.snapshot
        run.background_provider = self.plan.displayed_background
        restore_window_geometry(self.root, f"dnav.{run.id}")

    def save_geometry(self):
        self._save_geometry(self.root, f"dnav.{self.run.id}")

    def close(self):
        self.save_geometry()
        self.events.close()
        self.plan.reference.close()
        self.running = False

    def update(self):
        self.root.update_idletasks(); self.root.update()
        # The viewer keeps its own 5 Hz paint budget and has to drain its
        # reader every turn regardless, so it sits outside the guards below.
        self.events.poll()
        sim_now = self.run.sim_time_s()
        if self._paint_map.due():
            active = self.notebook.index(self.notebook.select())
            # Only the visible tab is painted: two map panes rendering the same
            # grid twice a tick is the repaint budget spent on a surface nobody
            # is looking at.
            if active == 0: self.plan.paint_map(sim_now)
            elif active == 1: self.cost.paint_map(sim_now)
        if self._paint_text.due():
            self.plan.paint_text(sim_now)
            self.cost.paint_text(sim_now)
            self._paint_status(sim_now)

    def _paint_status(self, sim_now):
        run = self.run
        connected = run.session.identity is not None
        self.status.set(
            f"{DOTS[run.health()]} planner {run.health()}  ·  "
            f"bus {'connected' if run.bus.session_id is not None else 'waiting'}  ·  "
            f"maps {'connected' if connected else 'no producer'}  ·  "
            f"pose {'ok' if not run.pose_error else run.pose_error}  ·  "
            f"t {sim_now:.2f}s  ·  plans {run.plans}/{run.attempts}  ·  "
            f"run {run.run_id or '--'}  ·  "
            f"clearance {run.navigation.last.get('clearance', {}).get('reason', 'waiting')}  ·  "
            f"report {_short_report(run.report_dir)}")


def run_headless(run, args):
    """Print each route as its status changes; the offline half of the window."""
    deadline = None if args.timeout <= 0 else time.monotonic() + args.timeout
    last = last_permit = provider_goal = None
    try:
        while not run.shutdown_requested:
            run.step()
            if run.provider_goal != provider_goal and run.provider_goal:
                provider_goal = run.provider_goal
                if provider_goal.get('adopted'):
                    x, y = provider_goal['position']
                    print(f"[sim {run.sim_time_s():8.2f}] goal: adopted the provider's map target at "
                          f"{x:.2f}, {y:.2f} m", flush=True)
                else:
                    print(f"[sim {run.sim_time_s():8.2f}] goal: no provider target ({provider_goal['reason']})",
                          flush=True)
            route = run.route
            # The route itself, not the revision it was planned against: new
            # evidence that leaves the plan unchanged is not news, and a line a
            # second saying so would bury the moment it does change.
            mark = (route.status, route.reason, route.points, round(route.cost, 6))
            clearance = (run.navigation.last.get('clearance') or {}) if run.navigation is not None else {}
            permit = (clearance.get('eligible'), clearance.get('reason'),
                      (clearance.get('evidence_check') or {}).get('reason'))
            if permit != last_permit and clearance:
                last_permit = permit
                evidence = clearance.get('evidence_check')
                print(f"[sim {route.sim_time_s:8.2f}] permission {'GRANTED' if permit[0] else 'withheld'}: "
                      f"{permit[1]}" + (f" | evidence check: {permit[2]}" if evidence else ""), flush=True)
            if mark != last:
                last = mark
                detour = run.summary_detour()
                extra = ""
                if route.ok:
                    extra = (f" cost={route.cost:.2f} length={route.length_m:.2f}m "
                             f"waypoints={len(route.waypoints)} "
                             f"plan={route.plan_time_s * 1000:.1f}ms")
                    if detour is not None:
                        extra += f" vs-control: {describe_detour(route, detour)}"
                print(f"[sim {route.sim_time_s:8.2f}] {route.status}: "
                      f"{route.reason or 'planned'}{extra}", flush=True)
                if route.ok:
                    print("    " + " -> ".join(f"({w.x_m:.2f},{w.y_m:.2f})"
                                               for w in route.waypoints), flush=True)
            if deadline is not None and time.monotonic() >= deadline:
                run.reason = "dnav timeout"
                run.partial = True
                break
            time.sleep(run.poll_delay())
    except KeyboardInterrupt:
        run.reason = "interrupted"
        run.partial = True
    return 0


def main(argv=None):
    from dcmn.window import disable_input_method

    disable_input_method()
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        policy = load_policy(args.policy, ROOT)
    except (OSError, ValueError) as exc:
        print(f"dnav: cannot load cost policy {args.policy!r}: {exc}", file=sys.stderr)
        return 1
    goal = None
    try:
        if args.goal: goal = parse_goal(args.goal)
    except ValueError as exc:
        print(f"dnav: {exc}", file=sys.stderr)
        return 1

    try:
        run = NavRun(args.id, ROOT, policy=policy, planner=args.planner, source=args.source,
                     pose_max_age=args.pose_max_age, goal_role=args.goal_role,
                     execution_profile=args.execution_profile, navigation_name=args.navigation_name,
                     profile_overrides=args.profile_set,
                     goal_from_provider=args.goal is None and not args.no_provider_goal)
    except (OSError, ValueError, TypeError) as exc:
        print(f"dnav: cannot start navigation: {exc}", file=sys.stderr)
        return 1
    if goal is not None:
        # Queued until the session provider appears; submitted as this
        # process's authority, and refused rather than overwriting another's.
        run.set_goal(*goal, handoff=args.take_goal_authority)
    window = None
    if not args.no_ui:
        try:
            import tkinter as tk
            window = Window(run, show_reference=args.show_reference)
        except ImportError as exc:
            print(f"dnav: tkinter is unavailable: {exc}", file=sys.stderr)
            run.close(partial=True)
            return 1
        except tk.TclError as exc:
            print(f"dnav: cannot open a window: {exc}", file=sys.stderr)
            run.close(partial=True)
            return 1
    try:
        if window is None:
            status = run_headless(run, args)
        else:
            status = _run_window(run, window, args)
    except Exception as exc:
        # Filed before re-raising: the report is written in the finally below,
        # and a run that died on a traceback must say so rather than reading
        # as a clean, complete one.
        run.partial = True
        run.reason = f"dnav crashed: {type(exc).__name__}: {exc}"
        raise
    finally:
        directory = run.close(partial=run.partial)
        if directory is not None:
            print(f"dnav: report -> {directory}", file=sys.stderr)
        if window is not None:
            try:
                window.save_geometry(); window.root.destroy()
            except Exception: pass
    return status


def _run_window(run, window, args):
    deadline = None if args.timeout <= 0 else time.monotonic() + args.timeout
    while window.running:
        # Outside every repaint guard: planning is the work, and a budget for
        # painting must never become a budget for it.
        run.step()
        window.update()
        if run.shutdown_requested:
            run.reason = run.reason or "instance shutdown requested"
            window.close()
        if deadline is not None and time.monotonic() >= deadline:
            run.reason = "dnav timeout"
            run.partial = True
            break
        time.sleep(min(run.poll_delay(), 1.0 / MAP_HZ / 2))
    return 0


#: Named so a linter can tell a deliberate re-export from an unused import.
_REEXPORTS = (R, theme)

if __name__ == "__main__":
    raise SystemExit(main())
