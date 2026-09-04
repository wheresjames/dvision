"""Changing the environment while the simulation runs.

The realism tab edits ``sim.realism`` directly -- it is in the same process as
the physics -- so what has to hold is that a change is validated before it is
applied, that it survives into the vehicle's behaviour, and that a rejected one
leaves the environment untouched. The widgets are not the subject; the model
underneath them is.
"""

from __future__ import annotations

import pytest

from dsim.realism import REALISM_DEFAULTS, Realism


def realism(**settings) -> Realism:
    return Realism.from_settings({**REALISM_DEFAULTS, **settings})


def test_settings_round_trip_without_changing_anything() -> None:
    model = realism(gps="degraded", wind_mps=1.5, wind_gust_mps=0.4,
                    sensor_noise="light", geofence="1,1,9,9",
                    geofence_action="rtl", telemetry_latency_ms=40.0)
    before = model.settings()
    model.update(0.5)
    wander = (model._gust.value, model._gps_north.value)

    model.apply(**before)

    assert model.settings() == before
    # Re-applying what is already set must not re-roll the dice: turning a knob
    # should not teleport the gust.
    assert (model._gust.value, model._gps_north.value) == wander


def test_a_change_retunes_the_noise_it_governs() -> None:
    model = realism(sensor_noise="none", wind_gust_mps=0.0, gps="rtk")
    assert model._baro_drift.sigma == 0.0
    assert model._gust.sigma == 0.0

    model.apply(sensor_noise="heavy", wind_gust_mps=1.5, gps="degraded")

    assert model._baro_drift.sigma == pytest.approx(0.3)
    assert model._gust.sigma == pytest.approx(1.5)
    # GPS noise follows the mode unless it was overridden.
    assert model._gps_north.sigma == pytest.approx(2.5)
    model.apply(gps_noise_m=0.1)
    assert model._gps_north.sigma == pytest.approx(0.1)


def test_a_rejected_change_leaves_the_environment_exactly_as_it_was() -> None:
    model = realism(gps="good", wind_mps=2.0)
    before = model.settings()

    for bad in ({"gps": "banana"}, {"sensor_noise": "extreme"},
                {"geofence_action": "explode"}, {"wind_mps": -1.0},
                {"battery_failsafe_pct": 140.0}, {"geofence": "1,2,3"},
                {"telemetry_latency_ms": -5.0}, {"nonsense": 1}):
        with pytest.raises(ValueError):
            model.apply(**bad)
        assert model.settings() == before


def test_the_seed_is_how_you_ask_for_the_dice_back() -> None:
    model = realism(wind_gust_mps=1.0, realism_seed=1234)
    for _ in range(10):
        model.update(0.1)
    wandered = model._gust.value
    assert wandered != 0.0

    model.apply(realism_seed=99)
    assert model.seed == 99
    assert model._gust.value == 0.0
    assert model._gust.sigma == pytest.approx(1.0)


def test_denying_gps_in_flight_invalidates_only_the_global_estimate() -> None:
    model = realism(gps="good", local_estimator="on")
    assert model.estimators()["global"] and model.estimators()["local"]

    model.apply(gps="off")

    assert not model.estimators()["global"]
    assert model.estimators()["local"], "VIO/SLAM does not depend on GPS"
    assert model.status_fields()["gps.fix_type"] == "0"


def test_a_fence_narrowed_under_a_flying_vehicle_puts_it_outside() -> None:
    model = realism(geofence="")
    assert not model.outside_geofence(20.0, 20.0, 1.5)

    model.apply(geofence="0,0,5,5", geofence_action="rtl")

    assert model.outside_geofence(20.0, 20.0, 1.5)
    assert model.geofence_action == "rtl"


def test_telemetry_latency_can_be_switched_on_and_off_mid_run() -> None:
    model = realism(telemetry_latency_ms=0.0)
    assert not model.telemetry.enabled

    model.apply(telemetry_latency_ms=120.0, telemetry_jitter_ms=30.0)
    assert model.telemetry.enabled
    assert model.telemetry.latency_s == pytest.approx(0.120)

    model.apply(telemetry_latency_ms=0.0, telemetry_jitter_ms=0.0)
    assert not model.telemetry.enabled


def test_estimator_faults_are_separate_from_the_settings() -> None:
    """A fault is a runtime event; re-applying the configuration keeps it."""
    model = realism(gps="good")
    model.set_estimator(local=False)
    assert not model.estimators()["local"]

    model.apply(wind_mps=1.0)

    assert not model.estimators()["local"], "applying settings cleared a fault"
    assert "local_estimator" not in model.faults


# ---------------------------------------------------------------------------
# Reaching the vehicle
# ---------------------------------------------------------------------------

def test_a_fence_narrowed_mid_flight_is_acted_on_by_the_vehicle(
        monkeypatch, tmp_path) -> None:
    """The whole point of the tab: a fault switched on against a flying drone."""
    from dtest.dway_rig import Rig
    from dway.mission import MissionState

    rig = Rig(monkeypatch, tmp_path, finish_action="hold",
              realism={"geofence_action": "rtl"})
    rig.fly(until=lambda m: m.state is MissionState.FLYING)
    assert rig.sim.state.mode == "GUIDED"

    rig.sim.realism.apply(geofence="0,0,2,2")

    assert rig.fly(limit_s=60.0) is MissionState.FAILED
    assert "geofence" in rig.mission.reason
    assert rig.sim.state.mode == "RTL"


def test_gps_denied_mid_flight_reaches_the_client_as_state(
        monkeypatch, tmp_path) -> None:
    from dtest.dway_rig import Rig
    from dway.mission import MissionState

    rig = Rig(monkeypatch, tmp_path, realism={"gps": "good"})
    rig.fly(until=lambda m: m.state is MissionState.FLYING)
    assert rig.mission.last_state.global_position_valid

    rig.sim.realism.apply(gps="off")
    rig.step()

    assert not rig.mission.last_state.global_position_valid
    # Local is untouched, so a map-frame tour keeps flying.
    assert rig.mission.last_state.local_position_valid
    assert rig.fly() is MissionState.COMPLETE, rig.mission.reason


# ---------------------------------------------------------------------------
# The panel itself
# ---------------------------------------------------------------------------

#: The viewport the simulator gives the page: its map canvas's height.
VIEWPORT_PX = 400


def panel_and_sim(*, height: int = VIEWPORT_PX):
    """A real panel over a real simulator, without the rest of the window.

    Real widgets, because geometry and event dispatch are exactly what is being
    asserted, but on a withdrawn root so the suite does not flash windows
    across the screen.
    """
    from types import SimpleNamespace

    from dcmn.tktheme import apply_theme
    from dsim.realism_panel import RealismPanel
    from dtest.tkfixture import hidden_root

    root = hidden_root()
    apply_theme(root)
    sim = SimpleNamespace(
        realism=realism(gps="degraded", wind_mps=1.2, sensor_noise="light"),
        state=SimpleNamespace(battery_pct=88.0, failsafe_reason=""))
    panel = RealismPanel(root, sim, height=height)
    # The notebook maps the page in the real window; do the same here so the
    # geometry the tests read is the geometry the window would produce.
    panel.page.grid(row=0, column=0, sticky="nsew")
    root.update_idletasks()
    return panel, sim, root


def test_the_form_opens_showing_what_the_command_line_asked_for() -> None:
    panel, sim, root = panel_and_sim()
    try:
        assert panel.vars["gps"].get() == "degraded"
        assert panel.vars["wind_mps"].get() == "1.2"
        assert panel.vars["sensor_noise"].get() == "light"
        # Blank means "use the fix mode's own figure", not zero.
        assert panel.vars["gps_noise_m"].get() == ""
        assert all(var.get() for var in panel.faults.values())
    finally:
        root.destroy()


def test_editing_a_field_and_applying_changes_the_running_simulator() -> None:
    panel, sim, root = panel_and_sim()
    try:
        panel.vars["wind_mps"].set("4.5")
        panel.vars["gps"].set("off")
        panel.apply()

        assert sim.realism.wind_mps == pytest.approx(4.5)
        assert sim.realism.gps_mode == "off"
        assert "applied" in panel.message.get()
        # The form is refilled from the model, so it shows what took effect.
        assert panel.vars["wind_mps"].get() == "4.5"
    finally:
        root.destroy()


def test_a_bad_value_is_refused_and_says_so_without_touching_the_vehicle() -> None:
    panel, sim, root = panel_and_sim()
    try:
        panel.vars["wind_mps"].set("gusty")
        panel.apply()
        assert sim.realism.wind_mps == pytest.approx(1.2)
        assert "not a number" in panel.message.get()

        panel.vars["wind_mps"].set("-3")
        panel.apply()
        assert sim.realism.wind_mps == pytest.approx(1.2)
        assert "must not be negative" in panel.message.get()
    finally:
        root.destroy()


def test_unticking_an_estimator_faults_it_immediately() -> None:
    panel, sim, root = panel_and_sim()
    try:
        assert sim.realism.estimators()["local"]
        panel.faults["local"].set(False)
        panel._apply_estimators()
        assert not sim.realism.estimators()["local"]

        # Ticking it back clears the fault rather than latching it.
        panel.faults["local"].set(True)
        panel._apply_estimators()
        assert sim.realism.estimators()["local"]
    finally:
        root.destroy()


def test_a_faulted_estimator_stays_ticked_when_something_else_invalidates_it() -> None:
    """A 2-D fix makes global invalid; that is not the operator's fault switch."""
    panel, sim, root = panel_and_sim()
    try:
        assert not sim.realism.estimators()["global"], "degraded fix is 2-D"
        panel.revert()
        assert panel.faults["global"].get() is True
        assert "global=INVALID" in _live(panel, "estimators")
    finally:
        root.destroy()


def test_reset_puts_back_the_settings_the_process_started_with() -> None:
    panel, sim, root = panel_and_sim()
    try:
        panel.vars["wind_mps"].set("9")
        panel.vars["gps"].set("rtk")
        panel.apply()
        panel.faults["attitude"].set(False)
        panel._apply_estimators()

        panel.reset()

        assert sim.realism.wind_mps == pytest.approx(1.2)
        assert sim.realism.gps_mode == "degraded"
        assert sim.realism.estimators()["attitude"], "reset left a fault behind"
        assert panel.vars["wind_mps"].get() == "1.2"
    finally:
        root.destroy()


def test_the_readout_reports_what_is_in_effect_not_what_was_typed() -> None:
    panel, sim, root = panel_and_sim()
    try:
        panel.refresh()
        assert "fix 2" in _live(panel, "gps")
        assert "1.20 m/s" in _live(panel, "wind")
        assert _live(panel, "battery") == "88.0%"
        assert _live(panel, "failsafe") == "none"

        # Typing without applying changes nothing.
        panel.vars["wind_mps"].set("9")
        panel.refresh()
        assert "1.20 m/s" in _live(panel, "wind")
    finally:
        root.destroy()


def _live(panel, key: str) -> str:
    return panel.live[key].get()


# ---------------------------------------------------------------------------
# Staying compact
# ---------------------------------------------------------------------------

def test_the_page_asks_for_the_height_it_was_given_not_the_form_s() -> None:
    """A notebook is as tall as its tallest page, so this one must not grow.

    Without the scrolling viewport the realism form set the height of the whole
    simulator window and pushed the map monitor down the screen.
    """
    panel, _sim, root = panel_and_sim()
    try:
        content = panel.frame.winfo_reqheight()
        page = panel.page.winfo_reqheight()
        assert content > VIEWPORT_PX, "the form is the tall thing being scrolled"
        assert page == VIEWPORT_PX, (
            f"the page asked for {page}, not the {VIEWPORT_PX} it was given")
    finally:
        root.destroy()


def test_a_short_form_hides_the_scrollbar() -> None:
    """Nothing to scroll to, nothing to scroll with."""
    panel, _sim, root = panel_and_sim()
    try:
        root.update()
        # grid_info() rather than winfo_ismapped(): the question is whether the
        # geometry manager is showing the bar, which is what the panel decides,
        # not whether X has it on screen -- and these run on a hidden root.
        assert panel._scroll._bar.grid_info(), "the form overflows already"

        # A viewport taller than the form takes the bar away again.
        panel._scroll._canvas.configure(height=4000)
        root.update()
        assert not panel._scroll._bar.grid_info()
    finally:
        root.destroy()


def test_the_wheel_over_a_combobox_scrolls_instead_of_spinning_it() -> None:
    """ttk spins a combobox on the wheel, which over a form silently edits it."""
    tkinter = pytest.importorskip("tkinter")
    panel, _sim, root = panel_and_sim()
    try:
        combobox = _widget_for(panel, "sensor_noise")
        assert isinstance(combobox, tkinter.ttk.Combobox)
        before = panel.vars["sensor_noise"].get()

        # The page's own binding runs first and breaks out of the class one.
        bound = combobox.bind("<Button-4>")
        assert bound, "no wheel binding on the combobox"
        top = panel._scroll._canvas.yview()[0]
        combobox.event_generate("<Button-5>", when="now")
        root.update()

        assert panel.vars["sensor_noise"].get() == before, "the wheel spun it"
        assert panel._scroll._canvas.yview()[0] > top, "the page did not scroll"
    finally:
        root.destroy()


def _widget_for(panel, name: str):
    variable = str(panel.vars[name])
    for child in panel.frame.winfo_children():
        if str(child.cget("textvariable") or "") == variable:
            return child
    raise AssertionError(f"no widget bound to {name}")


def test_reseeding_reseeds_telemetry_jitter_too():
    """``realism_seed`` asks for the dice back -- all of them.

    TelemetryDelay used to keep the generator it was built with, so a reseed
    changed every random process except the one it did not reach.
    """
    realism = Realism.from_settings({"telemetry_jitter_ms": 40.0,
                                     "realism_seed": 1})

    def jitter_stream(env):
        stream = []
        for tick in range(6):
            env.telemetry.push(float(tick), {})
            stream.append(round(env.telemetry._last_release, 6))
        return stream

    first = jitter_stream(realism)
    realism.apply(realism_seed=99)
    reseeded = jitter_stream(realism)
    realism.apply(realism_seed=1)

    assert reseeded != first
    assert realism.telemetry._rng is realism._rng


def test_telemetry_and_sensor_noise_share_one_seeded_stream():
    """Two generators seeded alike are not one stream; they are the same one
    drawn twice, which is a correlation nobody asked for."""
    realism = Realism.from_settings({"telemetry_jitter_ms": 40.0,
                                     "sensor_noise": "heavy"})

    assert realism.telemetry._rng is realism._rng
