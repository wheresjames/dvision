"""The Sensors tab: a validated draft, a tree of it, and a safe Apply.

The widgets are not the subject. What has to hold is that the tab never
touches the running simulation until Apply, that Apply is refused while armed
and after a failed validation, that the tree it draws is the parent links the
profile actually declares, and that every structural edit leaves a draft the
loader will still accept.
"""

from __future__ import annotations

from tkinter import ttk
from types import SimpleNamespace

import pytest

from dcmn import theme

from dsim.profiles import (DroneProfile, MOUNT_TYPES, SENSOR_TYPES,
                          camera_profile)
from dsim.sensors_panel import component_tree


def reference_profile():
    draft = camera_profile(64, 48)
    draft["mounts"] = [dict(id="head", type="mount.ptz", parent="body",
                            pose_parent=dict(z_m=.1), state=dict(tilt_deg=-5.))]
    draft["sensors"][0]["parent"] = "head"
    draft["sensors"].append(dict(id="scan", type="lidar.scan2d", parent="body",
                                 rate_hz=10., model=dict(samples=180)))
    return DroneProfile.parse(draft)


def test_the_tree_shows_the_parent_links_the_profile_declares():
    lines = component_tree(reference_profile()).splitlines()
    assert lines[0] == "body"
    assert lines[1].startswith("|-- head (mount.ptz)")
    assert "front (camera.rgb, primary, 30 Hz)" in lines[2]
    assert lines[2].startswith("|   ")
    assert "scan (lidar.scan2d, 10 Hz, array ring)" in lines[3]


def panel_and_sim():
    """A real panel over a stand-in simulator, on a withdrawn root."""
    from dcmn.tktheme import apply_theme
    from dsim.sensors_panel import SensorsPanel
    from dtest.tkfixture import hidden_root

    root = hidden_root()
    apply_theme(root)
    applied = []
    sim = SimpleNamespace(profile=reference_profile(),
                          sensors=SimpleNamespace(generation=1),
                          state=SimpleNamespace(armed=False),
                          apply_profile=applied.append)
    panel = SensorsPanel(root, sim)
    panel.page.grid(row=0, column=0, sticky="nsew")
    root.update_idletasks()
    return panel, sim, applied, root


@pytest.fixture
def panel():
    built = panel_and_sim()
    try:
        yield built
    finally:
        built[3].destroy()


def field(panel, name):
    """The editor variable for one field of the selected component."""
    return next(var for path, var in panel._vars.items() if path[-1] == name)


def ids(panel):
    return [node["id"] for node in panel.components()]


# ---------------------------------------------------------------------------
# Opening state
# ---------------------------------------------------------------------------

def test_the_tab_opens_on_the_running_profile(panel):
    tab, sim, _applied, _root = panel
    assert tab.resolved.digest == sim.profile.digest
    assert "active" in tab.info.get()
    assert "1 camera(s)" in tab.info.get() and "1 array ring(s)" in tab.info.get()
    assert "30 render Hz" in tab.info.get()
    assert tab.tree.exists("head") and tab.tree.parent("front") == "head"
    assert tab.selected == "front"


def test_the_readout_resolves_the_selected_camera(panel):
    tab, _sim, _applied, _root = panel
    text = tab.resolved_var.get()
    assert "fx " in text and "fov 70.00" in text
    # Body axes: the PTZ raises the camera 0.1 m and tilts it 5 degrees down,
    # and the camera adds another of each on top of the mount.
    assert "z +0.200" in text
    assert "pitch -10.00" in text
    assert "x +0.009" in text, "the child offset rotates with the tilted mount"


# ---------------------------------------------------------------------------
# Editing
# ---------------------------------------------------------------------------

def test_editing_a_model_field_reaches_the_resolved_profile(panel):
    tab, sim, applied, _root = panel
    # A resolved profile carries fx rather than the FOV it came from, so that
    # is the field the editor offers.
    before = sim.profile.primary["model"]["fx_px"]
    field(tab, "fx_px").set(str(before * 2))

    assert applied == [], "editing must not touch the running simulation"
    assert tab.resolved.primary["model"]["fx_px"] == pytest.approx(before * 2)
    assert "modified" in tab.info.get()
    assert sim.profile.primary["model"]["fx_px"] == before


def test_a_blank_field_means_the_default_rather_than_an_empty_value(panel):
    tab, _sim, _applied, _root = panel
    field(tab, "near_m").set("0.5")
    assert tab.resolved.primary["model"]["near_m"] == 0.5
    field(tab, "near_m").set("")
    assert "near_m" not in tab.component("front")["model"]
    assert tab.resolved.primary["model"]["near_m"] == 0.15


def test_an_invalid_field_reports_beside_itself_and_blocks_apply(panel):
    tab, _sim, applied, _root = panel
    field(tab, "rate_hz").set("7")

    assert tab.resolved is None
    assert "rate_hz" in tab.info.get()
    assert tab._errors["rate_hz"].cget("text")
    tab.apply()
    assert applied == []


def test_renaming_carries_the_children_and_the_primary_selection(panel):
    tab, _sim, _applied, _root = panel
    tab.selected = "head"
    tab._refresh_editor()
    field(tab, "id").set("gimbal")

    assert tab.component("front")["parent"] == "gimbal"
    assert tab.resolved is not None
    tab.selected = "front"
    tab._refresh_editor()
    field(tab, "id").set("nav")
    assert tab.draft["primary_camera"] == "nav"
    assert tab.resolved is not None


def test_reparenting_offers_no_cycle(panel):
    tab, _sim, _applied, _root = panel
    tab.selected = "head"
    tab._refresh_editor()
    # A mount may not hang from itself, and "head" is the only mount.
    assert tab._parent_choices("head") == ["body"]
    tab.selected = "front"
    tab._refresh_editor()
    field(tab, "parent").set("body")
    assert tab.resolved is not None
    assert tab.tree.parent("front") == "body"


def test_disabling_a_sensor_removes_it_from_the_resolved_plan(panel):
    tab, _sim, _applied, _root = panel
    tab.selected = "scan"
    tab._refresh_editor()
    field(tab, "enabled").set("false")

    assert tab.resolved is not None
    assert "scan" not in tab.resolved.plan["arrays"]
    assert "0 array ring(s)" in tab.info.get()


# ---------------------------------------------------------------------------
# Structural actions
# ---------------------------------------------------------------------------

def test_add_creates_a_valid_component_of_the_requested_type(panel):
    tab, _sim, _applied, _root = panel
    tab.add("range.ultrasonic")
    assert tab.resolved is not None
    assert tab.component(tab.selected)["type"] == "range.ultrasonic"
    assert tab.selected in ids(tab)


def test_duplicate_copies_the_selection_under_a_free_id(panel):
    tab, _sim, _applied, _root = panel
    tab.selected = "scan"
    tab.duplicate()
    assert tab.selected == "scan_copy"
    assert tab.resolved is not None
    assert tab.component("scan_copy")["model"]["samples"] == 180


def test_remove_lifts_the_children_rather_than_orphaning_them(panel):
    tab, _sim, _applied, _root = panel
    tab.selected = "head"
    tab.remove()

    assert "head" not in ids(tab)
    assert tab.component("front")["parent"] == "body"
    assert tab.resolved is not None, "removing a mount must leave a valid draft"


def test_add_stereo_pair_extends_the_draft_without_applying_it(panel):
    tab, _sim, applied, _root = panel
    tab.add_stereo()

    assert applied == []
    left, right = tab.draft["sensors"][-2:]
    assert [left["id"], right["id"]] == ["stereo1_left", "stereo1_right"]
    assert left["sync_group"] == right["sync_group"]
    assert left["pose_parent"]["y_m"] == -right["pose_parent"]["y_m"] != 0
    assert tab.resolved is not None
    assert "3 camera(s)" in tab.info.get()

    tab.add_stereo()
    assert [s["id"] for s in tab.draft["sensors"][-2:]] == [
        "stereo2_left", "stereo2_right"]


def test_the_readout_reports_the_stereo_baseline(panel):
    tab, _sim, _applied, _root = panel
    tab.add_stereo()
    tab._refresh_editor()
    assert "baseline 0.1200 m" in tab.resolved_var.get()


def test_the_readout_reports_the_pairs_relative_rotation(panel):
    tab, _sim, _applied, _root = panel
    tab.add_stereo()
    # A misalignment only the right member carries. The pair shares a -5
    # degree pitch, which must cancel: roll is the innermost factor of the
    # composition, so the readout has to answer with the misalignment alone
    # and none of the mount the two cameras hang from.
    tab.component("stereo1_right")["pose_parent"]["roll_deg"] = 3.0
    tab.refresh_all()
    text = tab.resolved_var.get()
    assert "baseline 0.1200 m" in text
    line = next(l for l in text.splitlines() if l.startswith("relative"))
    yaw, pitch, roll = (float(line.split()[i]) for i in (2, 4, 6))
    assert yaw == pytest.approx(0.0, abs=0.005)
    assert pitch == pytest.approx(0.0, abs=0.005), "the mount they share cancels"
    assert roll == pytest.approx(3.0, abs=0.005)


def test_make_primary_moves_the_badge_and_the_selection(panel):
    tab, _sim, _applied, _root = panel
    tab.add_stereo()
    tab.make_primary()
    assert tab.draft["primary_camera"] == "stereo1_left"
    assert tab.resolved.data["primary_camera"] == "stereo1_left"
    assert "primary" in tab._badges(tab.component("stereo1_left"))


# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------

def test_apply_is_disabled_while_armed(panel):
    tab, sim, applied, _root = panel
    sim.state.armed = True
    tab.refresh()
    assert str(tab.apply_button["state"]) == "disabled"
    sim.state.armed = False
    tab.refresh()
    assert str(tab.apply_button["state"]) == "normal"

    tab.apply()
    assert [p.digest for p in applied] == [sim.profile.digest]


def test_a_failed_apply_leaves_the_draft_and_says_why(panel):
    tab, sim, _applied, _root = panel

    def refuse(profile):
        raise RuntimeError("staged channel construction failed")

    sim.apply_profile = refuse
    tab.add_stereo()
    tab.apply()

    assert "Apply failed: staged channel" in tab.info.get()
    assert "stereo1_left" in ids(tab), "the draft must survive a failed apply"


def test_revert_discards_the_draft(panel):
    tab, sim, _applied, _root = panel
    tab.add_stereo()
    assert "modified" in tab.info.get()
    tab.revert()
    assert tab.resolved.digest == sim.profile.digest
    assert "active" in tab.info.get()
    assert "stereo1_left" not in ids(tab)


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------

def test_save_writes_the_resolved_draft_where_it_was_pointed(panel, tmp_path,
                                                             monkeypatch):
    tab, _sim, _applied, _root = panel
    tab.add_stereo()
    path = tmp_path / "hardware.json"
    monkeypatch.setattr("dsim.sensors_panel.filedialog.asksaveasfilename",
                        lambda **kwargs: str(path))
    tab.save()

    assert tab.path == str(path)
    # What lands on disk is the resolved draft -- the thing Apply would fly --
    # and it reloads as itself.
    assert DroneProfile.load(path).digest == tab.resolved.digest


def test_load_replaces_the_draft_with_the_file_on_disk(panel, tmp_path,
                                                       monkeypatch):
    tab, sim, _applied, _root = panel
    path = tmp_path / "hardware.json"
    reference_profile().save(path)
    tab.add_stereo()
    assert "modified" in tab.info.get()
    monkeypatch.setattr("dsim.sensors_panel.filedialog.askopenfilename",
                        lambda **kwargs: str(path))
    tab.load()

    assert tab.path == str(path)
    assert tab.resolved.digest == sim.profile.digest
    assert "active" in tab.info.get()
    assert "stereo1_left" not in ids(tab), "the file on disk replaced the draft"


def test_a_file_that_is_not_a_profile_says_so_rather_than_breaking_the_draft(
        panel, tmp_path, monkeypatch):
    tab, _sim, _applied, _root = panel
    tab.add_stereo()
    path = tmp_path / "broken.json"
    path.write_text('{"schema": "dvision2.drone-profile.v9"}')
    monkeypatch.setattr("dsim.sensors_panel.filedialog.askopenfilename",
                        lambda **kwargs: str(path))
    tab.load()

    assert "unsupported profile version" in tab.info.get()
    assert "stereo1_left" in ids(tab), "a bad file leaves the draft alone"


# ---------------------------------------------------------------------------
# The settings scroll
# ---------------------------------------------------------------------------

def combobox_for(tab, name):
    """The combobox editing one field of the selected component."""
    var = field(tab, name)
    return next(w for w in tab.editor.winfo_children()
                if isinstance(w, ttk.Combobox)
                and str(w.cget("textvariable")) == str(var))


def test_the_settings_get_a_scrollbar_when_they_outgrow_the_viewport(panel):
    """The editor can run well past a window's height; every field has to
    stay reachable, and a scrollbar that appears only when it is needed is
    how a short form stays clean."""
    tab, _sim, _applied, root = panel
    root.update()
    assert tab.editor.winfo_reqheight() > tab._scroll._canvas.winfo_reqheight(), \
        "the camera's settings are the tall thing being scrolled"
    assert tab._scroll._bar.grid_info(), "no scrollbar for an overflowing form"

    # A viewport taller than the settings takes the bar away again.
    tab._scroll._canvas.configure(height=4000)
    root.update()
    assert not tab._scroll._bar.grid_info()


def test_the_column_asks_for_the_viewports_height_not_the_editors(panel):
    """A notebook is as tall as its tallest page, so a settings column that
    asked for the editor's full height would set the height of the whole
    window. The viewport caps the column, whatever is selected."""
    tab, _sim, _applied, root = panel
    root.update_idletasks()
    viewport = tab._scroll.outer.winfo_reqheight()
    assert tab.editor.winfo_reqheight() > viewport, \
        "the camera's settings are the tall thing being scrolled"
    # The overflow belongs to the scrollbar, never to the window.
    for selected in ("front", "scan"):
        tab.selected = selected
        tab._refresh_editor()
        root.update_idletasks()
        assert tab._scroll.outer.winfo_reqheight() == viewport


def test_the_wheel_over_a_combobox_scrolls_the_settings_instead_of_spinning_it(
        panel):
    """ttk spins a combobox on the wheel, which over this form would silently
    edit the draft."""
    tab, _sim, _applied, root = panel
    combobox = combobox_for(tab, "enabled")
    before = field(tab, "enabled").get()
    top = tab._scroll._canvas.yview()[0]

    combobox.event_generate("<Button-5>", when="now")
    root.update()

    assert field(tab, "enabled").get() == before, "the wheel spun the draft"
    assert tab._scroll._canvas.yview()[0] > top, "the settings did not scroll"


# ---------------------------------------------------------------------------
# Theming
# ---------------------------------------------------------------------------

def test_the_add_dropdown_wears_the_application_palette(panel):
    """The popup is drawn by hand, so the palette has to be asked for by
    name; the button face is a ttk style."""
    tab, _sim, _applied, root = panel
    assert ttk.Style(root).lookup("TMenubutton", "background") == theme.BUTTON
    menu = tab.add_menu
    assert menu._top.overrideredirect(), "the window manager gave it a title bar"
    assert str(menu._canvas.cget("background")) == theme.PANEL
    assert str(menu._top.cget("background")) == theme.GRID, "no hairline edge"


# ---------------------------------------------------------------------------
# The Add menu
# ---------------------------------------------------------------------------

def test_the_add_menu_lists_every_type_with_its_description(panel):
    """The Add menu is the one place that enumerates every type, so it says
    what each one is -- and the sensor types and the mount types are divided
    by a hairline."""
    tab, _sim, _applied, _root = panel
    menu = tab.add_menu
    assert [item[0] for item in menu.items] == [*SENSOR_TYPES, *MOUNT_TYPES]
    assert all(item[1] for item in menu.items), "a type with no description"
    lines = [i for i in menu._canvas.find_all()
             if menu._canvas.type(i) == "line"]
    assert len(lines) == 1, "the sensors and the mounts are not divided"


def test_hovering_highlights_the_name_and_the_description_together(panel):
    """An item is one thing: its two lines take the highlight as a unit."""
    tab, _sim, _applied, _root = panel
    menu, canvas = tab.add_menu, tab.add_menu._canvas
    row = menu._rows[0]
    assert canvas.itemcget(row["rect"], "fill") == theme.PANEL
    assert canvas.itemcget(row["name"], "fill") == theme.TEXT
    assert canvas.itemcget(row["desc"], "fill") == theme.DIM

    menu._highlight(row)
    assert canvas.itemcget(row["rect"], "fill") == theme.ACCENT
    assert canvas.itemcget(row["name"], "fill") == theme.CANVAS
    assert canvas.itemcget(row["desc"], "fill") == theme.CANVAS

    menu._highlight(menu._rows[1])
    assert canvas.itemcget(row["rect"], "fill") == theme.PANEL, \
        "two items are highlighted at once"
    assert canvas.itemcget(row["name"], "fill") == theme.TEXT
    assert canvas.itemcget(menu._rows[1]["rect"], "fill") == theme.ACCENT


def test_picking_an_item_adds_that_type_and_closes_the_menu(panel):
    tab, sim, applied, _root = panel
    menu = tab.add_menu
    menu.open()
    row = next(r for r in menu._rows if r["kind"] == "range.laser")

    menu._on_press(SimpleNamespace(y=(row["y0"] + row["y1"]) / 2))

    assert not menu.is_open
    assert applied == [], "adding still never touches the running simulation"
    assert tab.component(tab.selected)["type"] == "range.laser"


def test_a_press_on_the_divider_closes_without_adding(panel):
    tab, _sim, _applied, _root = panel
    menu = tab.add_menu
    menu.open()
    before = len(tab.components())

    menu._on_press(SimpleNamespace(y=menu._rows[10]["y1"] + 4))

    assert not menu.is_open
    assert len(tab.components()) == before


def test_a_press_outside_the_items_closes_without_adding(panel):
    tab, _sim, _applied, _root = panel
    menu = tab.add_menu
    menu.open()
    before = len(tab.components())

    menu._top.event_generate("<Button-1>", when="now")

    assert not menu.is_open
    assert len(tab.components()) == before


def test_the_menu_scrolls_when_it_outgrows_the_screen(panel):
    """Fourteen types already open taller than the screen will spare them,
    so the wheel and a scrollbar have to reach the rest."""
    tab, _sim, _applied, root = panel
    menu = tab.add_menu
    root.update_idletasks()  # the fractions need the viewport's real size
    first, last = menu._canvas.yview()
    assert last < 1.0, "everything already fits on screen"

    menu._on_wheel(SimpleNamespace(num=5, delta=0))

    assert menu._canvas.yview()[0] > first, "the wheel moved nothing"
    assert menu._bar.grid_info(), "there is nothing to scroll with"
    for _ in range(3):
        menu._on_wheel(SimpleNamespace(num=4, delta=0))
    assert menu._canvas.yview()[0] == first, "the wheel ran away past the top"


@pytest.mark.parametrize('kind', SENSOR_TYPES)
def test_every_add_menu_sensor_starts_with_a_valid_model(panel, kind):
    tab, _sim, _applied, _root = panel
    tab.add(kind)
    assert tab.resolved is not None, tab.error


def test_profile_name_and_physics_cadence_are_editable_without_applying(panel):
    tab, sim, applied, _root = panel
    field(tab, 'name').set('inspection')
    field(tab, 'physics_hz').set('600')
    assert tab.resolved.data['name'] == 'inspection'
    assert tab.resolved.data['physics_hz'] == 600
    assert sim.profile.data['physics_hz'] == 300
    assert applied == []


def test_fov_edit_updates_intrinsics_and_keeps_a_single_lens_parameter(panel):
    tab, sim, _applied, _root = panel
    field(tab, 'fov_h_deg').set('40')
    assert tab.resolved.primary['model']['fx_px'] > sim.profile.primary['model']['fx_px']
    assert tab.resolved.primary['model']['fy_px'] == tab.resolved.primary['model']['fx_px']
    assert 'fx_px' not in tab.component('front')['model']
    assert 'fov 40.00' in tab.resolved_var.get()


def test_ptz_limits_validate_the_draft_state(panel):
    tab, _sim, _applied, _root = panel
    tab.selected = 'head'
    tab._refresh_editor()
    tab._vars[('limits', 'tilt_deg')].set('-2, 30')
    assert tab.resolved is None
    tab._vars[('limits', 'tilt_deg')].set('-90, 30')
    assert tab.resolved.data['mounts'][0]['limits']['tilt_deg'] == [-90, 30]


def test_dragging_reparents_without_allowing_cycles(panel):
    tab, _sim, _applied, _root = panel
    tab.reparent('front', 'body')
    assert tab.tree.parent('front') == 'body'
    tab.reparent('front', 'head')
    assert tab.tree.parent('front') == 'head'
    tab.reparent('head', 'front')
    assert tab.component('head')['parent'] == 'body'
    assert tab.resolved is not None


def test_duplicate_id_edit_is_visible_and_does_not_break_the_tree(panel):
    tab, _sim, _applied, _root = panel
    field(tab, 'id').set('head')
    assert tab.resolved is None
    assert 'duplicate' in tab.info.get()
    assert tab.tree.exists('front') and tab.tree.exists('head')
    field(tab, 'id').set('nav')
    assert tab.resolved is not None
    assert tab.tree.exists('nav')


def test_revert_refreshes_fields_even_when_selection_and_model_keys_match(panel):
    tab, sim, _applied, _root = panel
    field(tab, 'fx_px').set('200')
    tab.revert()
    assert float(field(tab, 'fx_px').get()) == sim.profile.primary['model']['fx_px']


def test_nonnumeric_rate_can_be_repaired_without_a_widget_callback_error(panel):
    tab, _sim, _applied, _root = panel
    field(tab, 'rate_hz').set('unfinished')
    assert tab.resolved is None
    assert 'invalid' in tab.tree.item('front', 'values')[0]
    assert tab._errors['rate_hz'].cget('text')
    field(tab, 'rate_hz').set('30')
    assert tab.resolved is not None
