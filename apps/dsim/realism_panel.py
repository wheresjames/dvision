"""The realism tab: what the environment is set to, and changing it in flight.

Every knob here is one the command line already accepts. The point of putting
them in the window is that a fault is far more informative when you can switch
it on against a vehicle that is already flying -- deny GPS mid-leg, raise the
wind while a controller is holding station, narrow the fence under a drone --
and none of that is reachable from a flag you had to choose before takeoff.

Changes are not persisted anywhere. The command line remains the record of how
a run started, and "Reset to command line" puts it back.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any

from dcmn import theme
from dsim.realism import (
    GEOFENCE_ACTIONS, GPS_MODES, REALISM_DEFAULTS, SENSOR_NOISE_PROFILES,
)
from dsim.scroll import Scrollable

#: One row of the form: setting name, label, and how it is edited.
#: ``choices`` makes a combobox, ``None`` an entry.
_FIELDS: tuple[tuple[str, str, tuple[str, ...] | None, str], ...] = (
    ("gps", "GPS fix", tuple(GPS_MODES), "quality of the published fix"),
    ("gps_noise_m", "GPS noise m", None, "blank uses the fix mode's own figure"),
    ("local_estimator", "Local estimator", ("on", "off"),
     "VIO/SLAM/flow pose, independent of GPS"),
    ("wind_mps", "Wind m/s", None, "steady speed"),
    ("wind_dir_deg", "Wind from deg", None, "compass direction it blows from"),
    ("wind_gust_mps", "Gust m/s", None, "correlated, on top of the steady wind"),
    ("telemetry_latency_ms", "Telemetry lag ms", None, "delay on published status"),
    ("telemetry_jitter_ms", "Telemetry jitter ms", None, "never reorders samples"),
    ("sensor_noise", "Sensor noise", tuple(SENSOR_NOISE_PROFILES),
     "compass, barometer and velocity"),
    ("ambient_temp_c", "Ambient temp C", None, "at ground level; the lapse rate cools with height"),
    ("battery_failsafe_pct", "Battery failsafe %", None, "0 disables; triggers RTL then LAND"),
    ("battery_drain_pct_s", "Battery drain %/s", None, "while armed"),
    ("geofence", "Geofence", None, "x0,y0,x1,y1[,max_alt_m]; blank for none"),
    ("geofence_action", "Fence action", GEOFENCE_ACTIONS, "on crossing it"),
    ("realism_seed", "Seed", None, "changing it re-rolls every noise process"),
)

#: Estimators that can be faulted independently of what caused it.
_ESTIMATORS = ("attitude", "local", "global", "velocity")


class RealismPanel:
    """A form over ``sim.realism``, plus a readout of what is actually in effect."""

    def __init__(self, parent: tk.Misc, sim, *, height: int = 640) -> None:
        self.sim = sim
        #: The settings the process started with, for Reset.
        self._command_line = dict(sim.realism.settings())
        self._scroll = Scrollable(parent, height=height)
        #: What the notebook adds as the page.
        self.page = self._scroll.outer
        self.frame = ttk.Frame(self._scroll.inner, padding=12)
        self.frame.grid(row=0, column=0, sticky="nsew")
        self._scroll.inner.columnconfigure(0, weight=1)
        self.frame.columnconfigure(2, weight=1)

        self.vars: dict[str, tk.StringVar] = {}
        self.faults: dict[str, tk.BooleanVar] = {}
        self.message = tk.StringVar(value="")
        self._build_form()
        self._build_live_readout()
        self.revert()
        # Paint the readout now rather than showing a column of dashes until
        # the first frame lands.
        self.refresh()

    # -- construction ---------------------------------------------------

    def _build_form(self) -> None:
        ttk.Label(self.frame, text="Environment", style="Brand.TLabel").grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))
        for row, (name, label, choices, hint) in enumerate(_FIELDS, start=1):
            ttk.Label(self.frame, text=label).grid(
                row=row, column=0, sticky="w", padx=(0, 10), pady=2)
            var = tk.StringVar()
            self.vars[name] = var
            if choices is None:
                widget: tk.Widget = ttk.Entry(self.frame, textvariable=var, width=14)
            else:
                widget = ttk.Combobox(self.frame, textvariable=var, width=12,
                                      values=list(choices), state="readonly")
            widget.grid(row=row, column=1, sticky="w", pady=2)
            self._scroll.claim_wheel(widget)
            # Enter applies, so a value can be committed without reaching for
            # the button; a combobox commits on selection for the same reason.
            widget.bind("<Return>", lambda _event: self.apply())
            if choices is not None:
                widget.bind("<<ComboboxSelected>>", lambda _event: self.apply())
            ttk.Label(self.frame, text=hint, style="Dim.TLabel").grid(
                row=row, column=2, sticky="w", padx=(12, 0), pady=2)

        faults_row = len(_FIELDS) + 1
        ttk.Label(self.frame, text="Estimators", style="Brand.TLabel").grid(
            row=faults_row, column=0, columnspan=3, sticky="w", pady=(12, 4))
        box = ttk.Frame(self.frame)
        box.grid(row=faults_row + 1, column=0, columnspan=3, sticky="w")
        for column, name in enumerate(_ESTIMATORS):
            var = tk.BooleanVar(value=True)
            self.faults[name] = var
            ttk.Checkbutton(box, text=name, variable=var,
                            command=self._apply_estimators).grid(
                row=0, column=column, sticky="w", padx=(0, 16))
        ttk.Label(self.frame,
                  text="unticking one faults that estimator, as a sensor "
                       "failure would; what is valid right now is below",
                  style="Dim.TLabel").grid(row=faults_row + 2, column=0,
                                           columnspan=3, sticky="w", pady=(2, 0))

        buttons = ttk.Frame(self.frame)
        buttons.grid(row=faults_row + 3, column=0, columnspan=3,
                     sticky="w", pady=(14, 0))
        for column, (text, command, style) in enumerate((
                ("Apply", self.apply, "Accent.TButton"),
                ("Revert", self.revert, "TButton"),
                ("Reset to command line", self.reset, "TButton"))):
            ttk.Button(buttons, text=text, command=command, style=style).grid(
                row=0, column=column, padx=(0, 8))
        self._message_label = ttk.Label(self.frame, textvariable=self.message,
                                        style="Dim.TLabel")
        self._message_label.grid(row=faults_row + 4, column=0, columnspan=3,
                                 sticky="w", pady=(8, 0))

    def _build_live_readout(self) -> None:
        row = len(_FIELDS) + 6
        ttk.Label(self.frame, text="In effect", style="Brand.TLabel").grid(
            row=row, column=0, columnspan=3, sticky="w", pady=(16, 4))
        self.live: dict[str, tk.StringVar] = {}
        for offset, (key, label) in enumerate((
                ("gps", "GPS"), ("estimators", "estimators"),
                ("wind", "wind"), ("battery", "battery"),
                ("fence", "geofence"), ("failsafe", "failsafe"))):
            ttk.Label(self.frame, text=label, style="Dim.TLabel").grid(
                row=row + 1 + offset, column=0, sticky="w", padx=(0, 10))
            var = tk.StringVar(value="-")
            self.live[key] = var
            ttk.Label(self.frame, textvariable=var).grid(
                row=row + 1 + offset, column=1, columnspan=2, sticky="w")

    # -- actions --------------------------------------------------------

    def revert(self) -> None:
        """Fill the form from what the simulator is actually running."""
        for name, value in self.sim.realism.settings().items():
            self.vars[name].set("" if value is None else _text(value))
        # These are the fault switches, not live validity: a 2-D fix makes the
        # global estimate invalid for its own reasons, and showing that here as
        # an unticked box would invite Apply to latch a fault nobody asked for.
        # What is actually valid right now is in the readout below.
        faults = self.sim.realism.faults
        for name, var in self.faults.items():
            var.set(faults.get(name, True))
        self._say("showing the values in effect")

    def reset(self) -> None:
        """Back to the settings the command line asked for."""
        try:
            self.sim.realism.apply(**self._command_line)
        except ValueError as exc:          # pragma: no cover - cannot happen
            self._say(f"reset failed: {exc}", error=True)
            return
        self.sim.realism.faults.clear()
        self.revert()
        self._say("reset to the command line")

    def apply(self) -> None:
        """Push the form onto the running simulator, or say why it will not go."""
        settings: dict[str, Any] = {}
        for name, _label, choices, _hint in _FIELDS:
            text = self.vars[name].get().strip()
            if choices is not None or name == "geofence":
                settings[name] = text
            elif not text:
                settings[name] = None if name == "gps_noise_m" \
                    else REALISM_DEFAULTS[name]
            else:
                try:
                    settings[name] = float(text)
                except ValueError:
                    self._say(f"{name}: {text!r} is not a number", error=True)
                    return
        try:
            self.sim.realism.apply(**settings)
        except (TypeError, ValueError) as exc:
            self._say(str(exc), error=True)
            return
        self._apply_estimators()
        self.revert()
        self._say("applied")

    def _apply_estimators(self) -> None:
        self.sim.realism.set_estimator(
            **{name: var.get() for name, var in self.faults.items()})

    def _say(self, message: str, *, error: bool = False) -> None:
        """A rejected change has to say so where the eye already is."""
        self.message.set(message)
        self._message_label.configure(
            foreground=theme.DANGER if error else theme.DIM)

    # -- live readout ---------------------------------------------------

    def refresh(self) -> None:
        """Called every frame: what the vehicle is publishing right now."""
        realism = self.sim.realism
        state = self.sim.state
        fields = realism.status_fields()
        self.live["gps"].set(
            f"fix {fields['gps.fix_type']}  sats {fields['gps.satellites']}  "
            f"hdop {fields['gps.hdop']}")
        estimators = realism.estimators()
        self.live["estimators"].set("  ".join(
            f"{name}={'ok' if estimators[name] else 'INVALID'}"
            for name in _ESTIMATORS))
        self.live["wind"].set(
            f"{realism.wind_speed_mps:.2f} m/s from {realism.wind_dir_deg:.0f} deg"
            if realism.wind_mps or realism.wind_gust_mps else "calm")
        self.live["battery"].set(f"{state.battery_pct:.1f}%")
        self.live["fence"].set(
            "none" if realism.geofence is None
            else f"{realism.geofence.describe()}  ({realism.geofence_action})")
        self.live["failsafe"].set(state.failsafe_reason or "none")


def _text(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)
