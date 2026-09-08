"""Device browser contract; widget halves run when DVISION2_GUI_TESTS opts in."""
import json
from pathlib import Path
import tkinter as tk
from tkinter import ttk

import numpy as np
import pytest

from dcmn.device_view import DevicesView, GenericRenderer, device_rows, heatmap, resolve_renderer
from dcmn.sensors import SensorSession
from dctl.dctl import DroneController, parse_args
from dsim.dsim import DroneState
from dsim.profiles import DroneProfile, camera_profile
from dtest.tkfixture import mapped_root_available
from tests.test_sensor_contract import manager, _Vehicle
from tests.test_dcmn_session import numeric_profile


@pytest.fixture(autouse=True)
def isolated_layout_store(tmp_path, monkeypatch):
    monkeypatch.setattr('dcmn.window._STORE', tmp_path/'window_pos.json')
    monkeypatch.setattr('dcmn.window._LOCK', tmp_path/'window_pos.lock')


def test_cli_and_unknown_renderer():
    args = parse_args(['--id', 'test', '--no-sensors', '--camera', 'other',
                       '--devices', 'imu,scan', '--sensor-cache-mb', '4'])
    assert args.no_sensors and args.camera == 'other' and args.sensor_cache_mb == 4
    assert resolve_renderer('future.unknown') is GenericRenderer
    for value in ('0', '-1', 'nan', 'inf'):
        with pytest.raises(SystemExit): parse_args(['--id', 'test', '--sensor-cache-mb', value])


def test_unfamiliar_layout_and_flat_tree():
    layout = [dict(name='new', dtype='<f8', shape=[2, 3, 4]),
              dict(name='scalar', dtype='u1', shape=[])]
    im = heatmap(dict(new=np.arange(24.).reshape(2, 3, 4), scalar=np.array(5)), layout)
    assert im is not None and im.width == 4
    for profile in (None, {}, {'mounts': None}):
        rows = device_rows(dict(sensors={'new': {'type': 'future.new'}}, profile=profile))
        assert rows == [('new', '', 'new (future.new)')]


def controller_stub(args):
    from dvision2_common import load_pymembus, shared_names
    ctl = DroneController.__new__(DroneController)
    ctl.args = args; ctl.pymembus = load_pymembus(); ctl.names = shared_names(args.id)
    ctl._last_open_attempt = -1e9; ctl.video = ctl.session = ctl.devices_view = None
    ctl.command = ctl.status = None; ctl.log = lambda text: None
    return ctl


def test_no_sensors_does_not_probe(monkeypatch):
    ctl = controller_stub(parse_args(['--id', 'no-sensor-test', '--no-sensors']))
    monkeypatch.setattr('dctl.dctl.SensorSession', lambda *a, **k: pytest.fail('sensor probe'))
    ctl.open_missing()
    assert ctl.session is None and ctl.video is None


def test_zero_camera_manifest_still_connects_numeric_streams():
    sensors = manager(numeric_profile(), vehicle=_Vehicle())
    manifest = dict(sensors.publisher.manifest)
    manifest.update(primary_camera=None, sensors={sid: entry for sid, entry in manifest['sensors'].items()
                                                if entry['transport'] != 'video'})
    sensors.publisher.registry.setAll({'sensors.manifest': json.dumps(manifest)})
    ctl = controller_stub(parse_args(['--id', sensors.publisher.instance]))
    try:
        ctl.open_missing()
        assert ctl.session.records is not None and ctl.video is None
        ctl.session.open('beam', accounting='from_attach')
        sensors.tick(DroneState(0, 0, 1), .1); ctl.session.poll()
        assert ctl.session.latest('beam') is not None
    finally: ctl.session.close(); sensors.close()


def test_reference_profile_browsable_and_apply_reconciles():
    profile = DroneProfile.load(Path('assets/drone_profiles/stereo-nav-and-proximity.json'))
    sensors = manager(profile, vehicle=_Vehicle())
    session = SensorSession(sensors.publisher.instance)
    root = None
    try:
        session.connect()
        assert len(session.devices) == 12
        rows = device_rows(session.manifest)
        assert set(session.devices).issubset({r[0] for r in rows})
        if mapped_root_available():
            root = tk.Tk(); root.geometry('1200x800')
            notebook = ttk.Notebook(root); notebook.pack(fill='both', expand=True)
            view = DevicesView(notebook, session, session.devices)
            notebook.add(view.page, text='Devices'); root.update()
            view.update(True)
            assert len(view.grid.panes) == 12
        else:
            for sid in session.devices: session.open(sid, accounting='from_attach')
        for i in range(600):
            sensors.tick(DroneState(0, 0, 1), i/300); session.poll()
            for stream in session.streams.values(): stream.refresh()
        assert all(session.latest(sid) is not None for sid in session.devices)
        before = {sid: session.latest(sid).capture_id for sid in session.devices}
        for i in range(600, 1200):
            sensors.tick(DroneState(0, 0, 1), i/300); session.poll()
            for stream in session.streams.values(): stream.refresh()
        assert all(session.latest(sid).capture_id > before[sid] for sid in session.devices)
        if root:
            # Exercise all generic renderers, independent of the paint-round budget.
            for sid, pane in view.grid.panes.items():
                pane.paint(session.report()['devices'][sid], 30)
            root.update()
            view.update(False)
            assert not session.streams and not view.grid.panes
            view.update(True); assert len(view.grid.panes) == 12
        sensors.apply(DroneProfile.parse(camera_profile(16, 12, physics_hz=30.)))
        session.probe.last_probe = -1e9; session.poll()
        if root:
            view.update(True)
            assert not view.grid.panes
            assert view.tree.exists('front')
        assert set(session.devices) == {'front'}
        sensors.apply(profile); session.probe.last_probe = -1e9; session.poll()
        if root:
            view.update(True)
            assert len(view.grid.panes) == 12
            assert all(p.changed for p in view.grid.panes.values())
    finally:
        session.close(); sensors.close()
        if root: root.destroy()


@pytest.mark.parametrize('disabled', [False, True])
def test_controller_window_starts_and_ticks(monkeypatch, disabled):
    if not mapped_root_available():
        assert parse_args(['--id', 'headless', '--no-sensors']).no_sensors
        return
    import dctl.dctl as module
    monkeypatch.setattr(module, 'save_window_pos', lambda *a: None)
    monkeypatch.setattr(module, 'restore_window_pos', lambda *a: None)
    sensors = manager(numeric_profile(), vehicle=_Vehicle())
    args = ['--id', sensors.publisher.instance, '--no-joystick', '--devices', 'beam']
    if disabled: args.append('--no-sensors')
    ctl = DroneController(parse_args(args))
    try:
        ctl._initialize_interfaces(); ctl.open_missing(); ctl._finish_window()
        sensors.tick(DroneState(0, 0, 1), .1)
        ctl.tick(); ctl.root.update_idletasks()
        if disabled:
            assert ctl.session is None and ctl.devices_view is None
        else:
            assert ctl.video is ctl.session.streams['front']
            assert ctl.video.required
            ctl.notebook.select(ctl.devices_view.page)
            ctl.tick(); ctl.root.update_idletasks()
            assert 'beam' in ctl.devices_view.grid.panes
            assert not ctl.session.streams['beam'].required
    finally: ctl.close(); sensors.close()


def test_camera_free_startup_schedules_controls():
    from types import SimpleNamespace
    for disabled, manifest in ((True, {}), (False, {'sensors': {}, 'primary_camera': None})):
        called = []
        ctl = DroneController.__new__(DroneController)
        ctl.args = SimpleNamespace(no_sensors=disabled, camera=None)
        ctl.video = None; ctl.session = SimpleNamespace(manifest=manifest)
        ctl.video_label = SimpleNamespace(configure=lambda **kw: None)
        ctl._schedule_dashboard = lambda: called.append(True)
        ctl.update_video()
        assert called == [True]


def test_flight_camera_switch_preserves_primary_requirement(monkeypatch):
    draft = camera_profile(16, 12, physics_hz=30.)
    draft['sensors'].append(dict(draft['sensors'][0], id='aux'))
    sensors = manager(DroneProfile.parse(draft))
    ctl = controller_stub(parse_args(['--id', sensors.publisher.instance]))
    try:
        ctl.open_missing()
        primary = ctl.video
        ctl.args.camera = 'aux'; ctl._last_open_attempt = -1e9; ctl.open_missing()
        assert ctl.video.selected == 'aux'
        assert primary.required and not ctl.video.required
        sensors.tick(DroneState(0, 0, 1), .1); ctl.session.poll()
        assert ctl.video.getSeq() > 0
        assert np.all(ctl.video[0] == [13, 34, 56])
        ctl.args.camera = 'front'; ctl._last_open_attempt = -1e9; ctl.open_missing()
        assert ctl.video is primary
        assert ctl.session.refs == {'front': 2}
    finally: ctl.session.close(); sensors.close()


def test_apply_removing_the_selected_camera_closes_the_stale_flight_stream():
    sensors = manager(DroneProfile.parse(camera_profile(16, 12, physics_hz=30.)))
    ctl = controller_stub(parse_args(['--id', sensors.publisher.instance, '--camera', 'front']))
    try:
        ctl.open_missing()
        assert ctl.video is not None and ctl.video.selected == 'front'
        # The selection never changes, but the applied manifest removes the
        # camera it points at: the stream left behind is dead, not retained.
        removed = camera_profile(16, 12, physics_hz=30.)
        removed['sensors'][0]['id'] = 'other'; removed['primary_camera'] = 'other'
        sensors.apply(DroneProfile.parse(removed))
        ctl.session.probe.last_probe = -1e9; ctl.session.poll()
        ctl._last_open_attempt = -1e9; ctl.open_missing()
        assert ctl.video is None
        assert 'front' not in ctl.session.streams and 'front' not in ctl.session.refs
        assert ctl.session.streams['other'].required
    finally: ctl.session.close(); sensors.close()


def test_apply_regrouping_the_stereo_pair_closes_the_stale_pair_stream():
    from dcmn.device_view import StereoStream
    draft = camera_profile(16, 12, physics_hz=30.)
    draft['sensors'].append(dict(draft['sensors'][0], id='aux', sync_group='pair'))
    draft['sensors'][0]['sync_group'] = 'pair'
    sensors = manager(DroneProfile.parse(draft))
    ctl = controller_stub(parse_args(['--id', sensors.publisher.instance, '--camera', 'stereo:pair']))
    try:
        ctl.open_missing()
        stale = ctl.video
        assert isinstance(stale, StereoStream) and stale.closed is False
        # Membership changes under the same pair name: the held pair streams
        # the old members and must close even though the id still resolves.
        regrouped = camera_profile(16, 12, physics_hz=30.)
        regrouped['sensors'].append(dict(regrouped['sensors'][0], id='other', sync_group='pair'))
        regrouped['sensors'].append(dict(regrouped['sensors'][0], id='aux'))
        regrouped['sensors'][0]['sync_group'] = 'pair'
        sensors.apply(DroneProfile.parse(regrouped))
        ctl.session.probe.last_probe = -1e9; ctl.session.poll()
        ctl._last_open_attempt = -1e9; ctl.open_missing()
        assert ctl.video is not stale and stale.closed
        assert tuple(ctl.video.members) == ('front', 'other')
        assert 'aux' not in ctl.session.streams
    finally: ctl.session.close(); sensors.close()


def stereo_controller(monkeypatch, sensors, *extra):
    import dctl.dctl as module
    monkeypatch.setattr(module, 'save_window_pos', lambda *a: None)
    monkeypatch.setattr(module, 'restore_window_pos', lambda *a: None)
    ctl = DroneController(parse_args(['--id', sensors.publisher.instance, '--no-joystick', *extra]))
    ctl._initialize_interfaces(); ctl.open_missing(); ctl._finish_window()
    ctl._last_open_attempt = -1e9; ctl.open_missing()
    return ctl


def test_flight_source_offers_the_stereo_pair(monkeypatch):
    from dcmn.device_view import ImageRenderer, StereoRenderer, StereoStream
    profile = DroneProfile.load(Path('assets/drone_profiles/stereo-nav-and-proximity.json'))
    if not mapped_root_available():
        from dcmn.device_view import stereo_groups
        manifest = dict(sensors={s['id']: dict(type=s['type'], sync_group=s.get('sync_group'))
                                 for s in profile.data['sensors']})
        assert stereo_groups(manifest) == {'stereo:nav_stereo': ('nav_left', 'nav_right')}
        return
    sensors = manager(profile, vehicle=_Vehicle())
    ctl = stereo_controller(monkeypatch, sensors)
    try:
        for i in range(30): sensors.tick(DroneState(0, 0, 1), i/300)
        ctl.tick(); ctl.update_video(); ctl.root.update_idletasks()
        assert ctl.video is ctl.session.streams['nav_left'] and ctl.video.required
        assert type(ctl.flight_renderer) is ImageRenderer
        assert 'stereo:nav_stereo' in ctl.camera_selector['values']
        ctl.camera_choice.set('stereo:nav_stereo'); ctl._select_camera()
        for i in range(30, 60): sensors.tick(DroneState(0, 0, 1), i/300)
        ctl.tick(); ctl.update_video(); ctl.root.update_idletasks()
        assert isinstance(ctl.video, StereoStream) and type(ctl.flight_renderer) is StereoRenderer
        # The pair is browsed, and the primary camera stays the required input.
        assert ctl.session.streams['nav_left'].required and not ctl.session.streams['nav_right'].required
        assert 'stereo:nav_stereo · 640×480' in ctl.video_label['text']
        ctl.camera_choice.set('nav_right'); ctl._select_camera()
        for i in range(60, 90): sensors.tick(DroneState(0, 0, 1), i/300)
        ctl.tick(); ctl.update_video(); ctl.root.update_idletasks()
        assert type(ctl.flight_renderer) is ImageRenderer
        assert ctl.video is ctl.session.streams['nav_right']
        assert 'nav_right' not in ctl.session.streams or not ctl.session.streams['nav_right'].required
        assert ctl.session.streams['nav_left'].required
    finally: ctl.close(); sensors.close()


def test_popped_out_pane_cannot_fly_the_drone(monkeypatch):
    if not mapped_root_available():
        assert parse_args(['--id', 'headless']).devices == ''
        return
    sensors = manager(numeric_profile(), vehicle=_Vehicle())
    ctl = stereo_controller(monkeypatch, sensors, '--devices', 'beam')
    try:
        ctl.notebook.select(ctl.devices_view.page)
        sensors.tick(DroneState(0, 0, 1), .1)
        ctl.tick(); ctl.root.update_idletasks()
        pane = ctl.devices_view.grid.panes['beam']
        ctl.devices_view.grid.pop_out('beam'); ctl.root.update()
        sent = []
        monkeypatch.setattr(ctl, 'send_command', lambda typ, quiet=False, **fields: sent.append(typ))
        # Tk delivers key events to the toplevel that holds keyboard focus,
        # regardless of the widget they name, so focusing the floating pane is
        # what makes this half a real test: the events reach its bindtags,
        # which no longer include the flying window's, so nothing may fly.
        pane.focus_force(); ctl.root.update()
        keys = ('<KeyPress-w>', '<KeyPress-t>', '<KeyPress-m>', '<KeyPress-l>', '<space>')
        for widget in (pane, pane.readout, pane.renderer.widget):
            for key in keys: widget.event_generate(key, when='now')
        assert not ctl.held and sent == []
        # The same keys, delivered to the focused flying window, still fly it.
        ctl.notebook.select(ctl.flight_page); ctl.root.update()
        ctl.root.focus_force(); ctl.root.update()
        ctl.video_label.event_generate('<KeyPress-w>', when='now')
        assert ctl.held == {'w'}
        ctl.video_label.event_generate('<KeyPress-l>', when='now')
        # Generated events are handled synchronously; an update() here would
        # let a scheduled tick send the held velocity and blur the assertion.
        assert ctl.held == {'w'} and sent == ['land']
        ctl.devices_view.grid.return_to_grid('beam')
    finally: ctl.close(); sensors.close()
