"""Renderer geometry, samples, persistence and paced bulk intake."""
from dataclasses import replace
import math
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np
import pytest

from dcmn import theme, window, layout
from dcmn.device_export import dump_samples, export_directory, load_samples, snapshot_png
from dcmn.device_view import (BeamRenderer, ImageRenderer, ScalarRenderer, PolarRenderer,
    RangeImageRenderer, GenericRenderer, CircleButton, DeviceGrid, DevicesView, DOTS, FixRenderer,
    StereoRenderer, VectorRenderer, colour_field, envelope_arrays, fix_state, open_device,
    polar_points, resolve_renderer, stereo_groups, stereo_sample, synchronized_capture,
    _wants_video)
from dcmn.pacing import TEXT_HZ
from dcmn.sensors import Sample, SensorSession
from dtest.tkfixture import mapped_root_available
from dsim.profiles import DroneProfile, camera_profile
from dsim.dsim import DroneState
from tests.test_sensor_contract import manager, _Vehicle


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setattr(window, '_STORE', tmp_path/'window_pos.json')
    monkeypatch.setattr(window, '_LOCK', tmp_path/'window_pos.lock')
    if not mapped_root_available(): yield None; return
    import tkinter as tk
    root = tk.Tk(); root.geometry('900x700'); root.update()
    try: yield root
    finally: root.destroy()


def sample(kind, payload=None, fields=None, image=None, status=1):
    entry = dict(type=kind, rate_hz=30, model=dict(min_range_m=1., max_range_m=20.,
        width_px=4, height_px=3, fx_px=2, fy_px=2), calibration=dict(angle_min_deg=-90,
        angle_increment_deg=90, elevation_deg=0), layout=[])
    return Sample('test', kind, 1, 1, 1., status, payload or {}, fields, image, entry)


def test_ramps_invalid_values_and_polar_coordinates():
    assert theme.RANGE_LUT.shape == theme.CONFIDENCE_LUT.shape == (256, 3)
    assert theme.RANGE_LUT.dtype == np.uint8
    values = colour_field(np.array([1., 20., np.nan]), 1., 20., theme.RANGE_LUT)
    assert np.array_equal(values[0], theme.RANGE_LUT[0])
    assert np.array_equal(values[1], theme.RANGE_LUT[-1])
    assert np.array_equal(values[2], [32, 32, 32])
    scan = sample('lidar.scan2d', fields={'range_m': np.array([2., 3., np.nan])})
    points, angles = polar_points(scan)
    assert np.allclose(points[:2], [[-2., 0.], [0., 3.]])
    assert list(angles) == [-90, 0, 90]


@pytest.mark.parametrize('kind,renderer,payload', [
    ('camera.rgb', ImageRenderer, {}), ('range.infrared', BeamRenderer, {'range_m': 5}),
    ('altimeter.barometric', ScalarRenderer, {'altitude_m': 3}),
    ('heading.magnetometer', ScalarRenderer, {'heading_deg': 90}),
    ('environment.temperature', ScalarRenderer, {'temperature_c': 22}),
    ('lidar.scan2d', PolarRenderer, {}), ('lidar.range_image', RangeImageRenderer, {}),
    ('future.unknown', GenericRenderer, {'reading': 5})])
def test_renderers_and_options(root, kind, renderer, payload):
    assert resolve_renderer(kind) is renderer
    fields = {'range_m': np.array([[2., 3., np.nan]]), 'confidence': np.array([[255, 128, 0]], dtype='u1')}
    value = sample(kind, payload, fields if kind.startswith('lidar') else None, np.zeros((3,4,3),dtype='u1') if kind.startswith('camera') else None)
    if root is None: return
    widget = renderer(root, value.entry)
    root.update(); widget.draw(value)
    options = dict(widget.options); widget.set_options(options)
    assert widget.options == options
    if renderer is RangeImageRenderer:
        left, top, width, height = widget.bounds
        assert widget.value_at(left+width/6, top+height/2) == 2.
        assert widget.value_at(left+width*5/6, top+height/2) is None
        widget.set_options({'field': 'confidence'}); widget.draw(value)
        assert widget.value_at(left+width/2, top+height/2) == 3.
    if renderer is BeamRenderer:
        widget.draw(sample(kind, {'range_m': None}, status=0))
        assert widget.value['text'] == 'no return'
    if renderer is ScalarRenderer and kind == 'heading.magnetometer':
        # The rose fills the pane, centred on it, instead of the old corner dial.
        box = widget.canvas.bbox('all')
        size = widget.canvas.winfo_width(), widget.canvas.winfo_height()
        assert abs((box[0]+box[2])/2-size[0]/2) < 5 and abs((box[1]+box[3])/2-size[1]/2) < 5
        assert min(box[2]-box[0], box[3]-box[1]) > .6*min(size)
        assert not widget.choices  # no plot window option on a wrap-around value
    variables = list(widget.variables.values())
    widget.destroy()
    assert all(not variable.trace_info() for variable in variables)


def test_polar_scan_agrees_with_world_rays():
    from dsim import sensor_models
    from dsim.transforms import resolve
    from dsim.range import scene_geometry
    from types import SimpleNamespace
    draft = camera_profile(16, 12, physics_hz=30.)
    draft['sensors'].append(dict(id='scan', type='lidar.scan2d', parent='body', rate_hz=10., model={'samples': 8}))
    profile = DroneProfile.parse(draft); sensor = profile.data['sensors'][1]
    pose = resolve(profile.data, 'scan', DroneState(0, 0, 1, yaw_deg=90))
    # Compare projection against the same calibrated rays used by the range oracle.
    directions = sensor_models.scan_directions(sensor['model']) @ pose[:3, :3].T
    from dsim.range import cast_rays
    scene = scene_geometry(SimpleNamespace(objects=[SimpleNamespace(kind='wall', x=5., y=0.)]))
    ranges = cast_rays(scene, pose[:3, 3], directions, max_range_m=20.)
    ranges[~np.isfinite(ranges)] = np.nan
    entry = dict(model=sensor['model'], calibration=sensor_models.calibration('lidar.scan2d', sensor['model']))
    value = Sample('scan', 'lidar.scan2d', 1, 1, 1., 1,
        {'pose_world': pose.tolist(), 'body': {'heading_deg': 180}}, {'range_m': ranges}, entry=entry)
    assert np.isfinite(ranges).any()
    points, _ = polar_points(value, 'north')
    assert np.allclose(points[:, 0], directions[:, 0]*ranges, equal_nan=True)
    assert np.allclose(points[:, 1], -directions[:, 1]*ranges, equal_nan=True)


def test_array_open_close_hundred_times_has_no_retained_cache():
    draft = camera_profile(16, 12, physics_hz=30.)
    draft['sensors'].append(dict(id='scan', type='lidar.scan2d', parent='body', rate_hz=10., model={'samples': 8}))
    sensors = manager(DroneProfile.parse(draft)); session = SensorSession(sensors.publisher.instance)
    try:
        for i in range(100):
            stream = session.open('scan', accounting='from_attach')
            for j in range(3): sensors.tick(DroneState(0, 0, 1), (3*i+j)/30)
            session.poll(); stream.refresh(); assert stream.latest is not None
            session.release('scan')
            assert stream.array is None and session.cache_bytes == 0 and not session.streams
    finally: session.close(); sensors.close()


def test_grid_layout_and_options_survive_visibility_and_reload(root):
    if root is None:
        # The default bench is three by three, even for a single device.
        fresh = layout.reconcile(layout.auto_layout(['a', 'b']))
        assert (fresh['columns'], fresh['rows']) == (3, 3)
        return
    from tkinter import ttk
    sensors = manager(DroneProfile.parse(camera_profile(16, 12, physics_hz=30.)))
    session = SensorSession(sensors.publisher.instance)
    try:
        session.connect(); notebook = ttk.Notebook(root); notebook.pack(fill='both', expand=True)
        view = DevicesView(notebook, session, ['front']); notebook.add(view.page, text='Devices')
        view.update(True); root.update()
        view.grid.panes['front'].renderer.set_options({'fit': 'native'})
        record = view.grid.serialize(); record.update(columns=2, column_weights=[2., 1.])
        view.grid.restore(record); view.grid.resize('front', 0, 1)
        view.update(False); view.save()
        saved = window.load_state('device_layouts', sensors.profile.data['name'])
        assert saved['panes'][0]['colspan'] == 2
        assert saved['panes'][0]['options']['fit'] == 'native'
        view.update(True)
        assert view.grid.panes['front'].renderer.options['fit'] == 'native'
    finally: session.close(); sensors.close()


def test_four_cameras_share_one_copy_budget(root, monkeypatch):
    if root is None:
        assert 4*max(5., 30./4) == 30.
        return
    draft = camera_profile(64, 48, physics_hz=30.)
    first = draft['sensors'][0]
    draft['sensors'] += [dict(first, id=f'camera{i}') for i in range(1, 4)]
    sensors = manager(DroneProfile.parse(draft)); session = SensorSession(sensors.publisher.instance)
    clock = [0.]
    monkeypatch.setattr('dcmn.device_view.time.monotonic', lambda: clock[0])
    try:
        session.connect()
        copied = []
        for ids in (['front'], ['front', 'camera1', 'camera2', 'camera3']):
            grid = DeviceGrid(root, session); grid.pack(fill='both', expand=True)
            grid.wanted = ids; grid.set_visible(True); root.update()
            count = 0
            for _ in range(61):
                clock[0] += 1/30
                sensors.tick(DroneState(0, 0, 1), clock[0]); session.poll()
                before = {sid: stream.last_seen_video for sid, stream in session.streams.items()}
                grid.paint(session.report(), budget_s=1.)
                count += sum(stream.last_seen_video != before[sid] for sid, stream in session.streams.items())
            copied.append(count*64*48*3)
            grid.set_visible(False); grid.destroy()
        assert copied[1] <= copied[0]*1.2
        assert copied[1] > copied[0]*.8
    finally: session.close(); sensors.close()


def test_same_profile_apply_retains_cell_options_and_grey_absent_device(root):
    if root is None:
        assert layout.reconcile(layout.auto_layout(['front']), [])['panes'][0]['id'] == 'front'
        return
    from tkinter import ttk
    draft = camera_profile(16, 12, physics_hz=30.)
    draft['sensors'].append(dict(draft['sensors'][0], id='aux'))
    profile = DroneProfile.parse(draft)
    sensors = manager(profile); session = SensorSession(sensors.publisher.instance)
    try:
        session.connect(); notebook = ttk.Notebook(root); notebook.pack(fill='both', expand=True)
        view = DevicesView(notebook, session, ['front', 'aux']); notebook.add(view.page, text='Devices')
        view.update(True); root.update()
        view.grid.swap('front', 'aux')
        view.grid.panes['aux'].renderer.set_options({'fit': 'native'})
        saved = next(p for p in view.grid.serialize()['panes'] if p['id'] == 'aux')
        draft['sensors'].pop()
        sensors.apply(DroneProfile.parse(draft)); session.probe.last_probe = -1e9; session.poll(); view.update(True)
        assert 'aux' not in view.grid.panes and view.tree.item('aux', 'tags') == ('absent',)
        sensors.apply(profile); session.probe.last_probe = -1e9; session.poll(); view.update(True)
        restored = next(p for p in view.grid.serialize()['panes'] if p['id'] == 'aux')
        assert restored == saved
    finally: session.close(); sensors.close()


def imu_history(count, rate=100.):
    entry = dict(type='motion.imu', rate_hz=rate, model={}, calibration={}, layout=[])
    return [Sample('imu', 'motion.imu', i, i, i/rate, 1,
        dict(angular_rate_dps=dict(x=math.sin(i/50), y=.5, z=-.25),
             specific_force_mps2=dict(x=0., y=0., z=9.81+.01*(i % 7))), entry=entry)
            for i in range(1, count+1)]


def gnss_sample(i=1, fix_type=3, status=1, valid=True):
    entry = dict(type='position.gnss', rate_hz=10., model={}, calibration={}, layout=[])
    return Sample('gnss', 'position.gnss', i, i, i/10., status,
        dict(fix_type=fix_type, valid=valid, satellites=11, hdop=.9, vdop=1.4,
             lat_deg=47.1, lon_deg=-122.3, alt_m=71.5, vel_north_mps=.2, vel_east_mps=-.1,
             vel_down_mps=0., error_north_m=.4*math.sin(i/5), error_east_m=.3*math.cos(i/7)),
        entry=entry)


def test_imu_pane_holds_text_rate_over_a_full_window(root):
    history = imu_history(6000)
    series = envelope_arrays(history, 'angular_rate_dps', 'x', history[-1].sim_time_s, 60., 100.)
    stamps, mean, low, high = series.plot_arrays()
    # The whole window is admitted, then compressed to the drawn width.
    assert series.samples == 6000 and 2 <= len(stamps) <= 200
    assert min(low) <= min(mean) and max(high) >= max(mean)
    if root is None: return
    widget = VectorRenderer(root, history[-1].entry); root.update()
    widget.history = history
    start = time.perf_counter()
    widget.draw(history[-1])
    assert time.perf_counter()-start < 1/TEXT_HZ
    assert widget.canvas.find_all()
    assert dict(widget.describe(history[-1]))['angular_rate_dps'].startswith('X ')
    assert widget.describe(replace(history[-1], status=0)) == [('state', 'invalid IMU sample')]
    widget.destroy()


def test_gnss_invalid_states_are_three_different_layouts(root):
    assert fix_state(None) == 'no receiver'
    assert fix_state(gnss_sample(fix_type=0)) == 'no fix'
    assert fix_state(gnss_sample(status=0)) == 'fix rejected'
    assert fix_state(gnss_sample(valid=False)) == 'fix rejected'
    assert fix_state(gnss_sample()) == 'fix accepted'
    if root is None: return
    widget = FixRenderer(root, gnss_sample().entry); root.update()
    widget.history = [gnss_sample(i) for i in range(1, 601)]
    widget.draw(widget.history[-1])
    assert widget.quality['text'] == 'fix accepted' and widget.canvas.find_all()
    assert widget.envelopes['north'].samples == 600
    rows = dict(widget.describe(widget.history[-1]))
    assert rows['satellites'] == '11' and rows['state'] == 'fix accepted'
    for sample in (gnss_sample(fix_type=0), gnss_sample(status=0)):
        widget.draw(sample)
        # An invalid fix draws no scatter rather than a position of zero.
        assert widget.quality['text'] == fix_state(sample) and not widget.canvas.find_all()
    widget.destroy()


REFERENCE = Path('assets/drone_profiles/stereo-nav-and-proximity.json')


def reference_session():
    sensors = manager(DroneProfile.load(REFERENCE), vehicle=_Vehicle())
    session = SensorSession(sensors.publisher.instance)
    session.connect()
    return sensors, session


def test_stereo_pair_matches_captures_and_reports_desynchronization(root):
    sensors, session = reference_session()
    try:
        assert stereo_groups(session.manifest) == {'stereo:nav_stereo': ('nav_left', 'nav_right')}
        stream = open_device(session, 'stereo:nav_stereo')
        for i in range(30):
            sensors.tick(DroneState(0, 0, 1), i/300); session.poll(); stream.refresh()
        sample = stream.latest
        left, right = session.latest('nav_left'), session.latest('nav_right')
        assert sample.capture_id == left.capture_id == right.capture_id
        assert sample.payload['baseline_m'] == pytest.approx(.12, abs=1e-6)
        entry = stream.entry
        # A pair that does not share a capture, generation or epoch is never composed.
        assert stereo_sample('stereo:nav_stereo', left, replace(right, capture_id=right.capture_id+1), entry) is None
        assert stereo_sample('stereo:nav_stereo', left, replace(right, generation=right.generation+1), entry) is None
        assert stereo_sample('stereo:nav_stereo', left, replace(right, reset_epoch=right.reset_epoch+1), entry) is None
        if root is None: return
        widget = StereoRenderer(root, entry); root.update()
        widget.set_options({'fit': 'native'})
        height, width = left.image.shape[:2]
        for mode, expected in (('side-by-side', 2*width), ('anaglyph', width), ('difference', width)):
            widget.set_options({'mode': mode}); widget.draw(sample)
            assert (widget.photo.width(), widget.photo.height()) == (expected, height)
        assert dict(widget.describe(sample))['sync'] == 'matched'
        stale = replace(sample, payload=dict(sample.payload, latest_captures=[sample.capture_id+1, sample.capture_id]))
        assert 'waiting for matching capture' in dict(widget.describe(stale))['sync']
        # A mismatched pair renders both eyes at natural scale instead of resampling one.
        odd = replace(sample, fields=dict(left=left.image, right=right.image[:height//2, :width//2]))
        widget.draw(odd)
        assert widget.photo.width() == width+width//2 and widget.options['fit'] == 'native'
        assert 'mismatch' in dict(widget.describe(odd))['resolution']
        widget.destroy()
    finally: session.close(); sensors.close()


def test_freeze_holds_one_capture_and_names_the_panes_without_it(root):
    if root is None:
        assert synchronized_capture([]) == (None, False)
        return
    sensors, session = reference_session()
    try:
        grid = DeviceGrid(root, session); grid.pack(fill='both', expand=True)
        grid.wanted = ['nav_left', 'imu', 'scan']; grid.set_visible(True); root.update()
        for i in range(30):
            sensors.tick(DroneState(0, 0, 1), i/300); session.poll()
            for pane in grid.panes.values(): pane.stream.refresh()
        grid.freeze_all(True)
        grid.paint(session.report(), budget_s=1.)
        held = {sid: pane.display_capture for sid, pane in grid.panes.items()}
        assert len(set(held.values())) == 1 and None not in held.values()
        assert 'shared capture' in grid.freeze_message
        # The drain continues while the paint is held.
        decoded = session.report()['decoded_records']
        for i in range(30, 60):
            sensors.tick(DroneState(0, 0, 1), i/300); session.poll()
        grid.paint(session.report(), budget_s=1.)
        assert session.report()['decoded_records'] > decoded
        assert {sid: pane.display_capture for sid, pane in grid.panes.items()} == held
        # A pane with no record at the held capture says so instead of showing a neighbour's.
        grid.toggle('ambient_temp'); root.update()
        # Opened after the freeze was already armed, it inherits the held
        # capture instead of showing "waiting for sample" forever.
        grid.paint(session.report(), budget_s=1.)
        assert 'ambient_temp' in grid.freeze_targets
        assert grid.panes['ambient_temp'].readout['text'] != 'waiting for sample'
        grid.freeze_all(True)
        grid.paint(session.report(), budget_s=1.)
        assert 'no shared capture' in grid.freeze_message
        capture = grid.freeze_targets['nav_left']
        assert grid.panes['ambient_temp'].display_capture is None
        assert grid.panes['ambient_temp'].readout['text'] == f'no sample at capture {capture}'
        assert grid.panes['nav_left'].display_capture == capture
        # Unsynchronized freeze holds each pane's own newest capture.
        newest = {}
        for i in range(60, 120):
            for pane in grid.panes.values(): pane.stream.refresh()
            newest = {sid: pane.stream.latest.capture_id for sid, pane in grid.panes.items() if pane.stream.latest}
            if len(set(newest.values())) > 1: break
            sensors.tick(DroneState(0, 0, 1), i/300); session.poll()
        assert len(set(newest.values())) > 1, 'devices never diverged'
        grid.freeze_all(True, synchronized=False)
        grid.paint(session.report(), budget_s=1.)
        assert {sid: grid.panes[sid].display_capture for sid in newest} == newest
        grid.freeze_all(False)
        assert grid.freeze_message == 'live'
        grid.destroy()
    finally: session.close(); sensors.close()


def test_popped_pane_keeps_its_window_and_returns_to_its_cell(root):
    if root is None:
        record = layout.reconcile(dict(columns=1, rows=1, panes=[dict(id='front', row=0, col=0)],
                                       popped={'front': {'geometry': '640x520+2100+180'}}))
        assert record['popped'] == {'front': {'geometry': '640x520+2100+180'}}
        return
    from tkinter import ttk
    draft = camera_profile(16, 12, physics_hz=30.)
    draft['sensors'].append(dict(draft['sensors'][0], id='aux'))
    sensors = manager(DroneProfile.parse(draft)); session = SensorSession(sensors.publisher.instance)
    try:
        session.connect(); notebook = ttk.Notebook(root); notebook.pack(fill='both', expand=True)
        view = DevicesView(notebook, session, ['front', 'aux']); notebook.add(view.page, text='Devices')
        view.update(True); root.update()
        view.grid.swap('front', 'aux')
        cell = next(p for p in view.grid.serialize()['panes'] if p['id'] == 'front')
        pane = view.grid.panes['front']; stream, renderer = pane.stream, pane.renderer
        view.grid.pop_out('front'); root.update()
        assert view.grid.popped == {'front': pane} and not pane.grid_info()
        # The floating window carries no binding tag from the flying window.
        assert str(root) not in pane.bindtags()
        assert str(root) not in pane.renderer.widget.bindtags()
        # Its transport and renderer instance survive the move.
        assert pane.stream is stream and pane.renderer is renderer
        pane.wm_geometry('320x240+140+130'); root.update()
        view.grid.return_to_grid('front'); root.update()
        saved = window.load_state('window_pos', view.grid._geometry_key('front'))['geometry']
        assert saved.endswith('+140+130')
        assert str(root) in view.grid.panes['front'].bindtags()
        restored = view.grid.panes['front'].grid_info()
        assert (restored['row'], restored['column']) == (cell['row'], cell['col'])
        # Popping out again reopens where it was left.
        view.grid.pop_out('front'); root.update()
        assert view.grid.panes['front'].wm_geometry() == saved
        assert view.grid.serialize()['popped']['front']['geometry'] == saved
        view.grid.return_to_grid('front')
    finally: session.close(); sensors.close()


def export_history(count=6):
    entry = dict(type='lidar.range_image', rate_hz=10., model=dict(min_range_m=.5, max_range_m=30.),
                 calibration={}, layout=[dict(name='range_m', dtype='<f4', shape=[2, 2])])
    return [Sample('flash', 'lidar.range_image', i, i, i/10., 1,
        dict(returns=3, samples=4, nested=dict(pose=[1., 2., 3.])),
        dict(range_m=np.array([[1.5, np.nan], [2.5, 3.5]], dtype='f4'),
             confidence=np.array([[255, 0], [128, 64]], dtype='u1')),
        np.full((2, 2, 3), i, dtype='u1'), entry) for i in range(1, count+1)]


def same_sample(restored, original):
    assert (restored.sensor_id, restored.type, restored.capture_id, restored.sequence,
            restored.sim_time_s, restored.status) == (original.sensor_id, original.type,
            original.capture_id, original.sequence, original.sim_time_s, original.status)
    assert restored.payload == original.payload and restored.entry == original.entry
    assert np.array_equal(restored.image, original.image)
    if original.fields is None:
        assert restored.fields is None
        return
    assert set(restored.fields) == set(original.fields)
    for key, value in original.fields.items():
        assert restored.fields[key].dtype == value.dtype
        assert np.array_equal(restored.fields[key], value, equal_nan=value.dtype.kind == 'f')


@pytest.mark.parametrize('form', ['json', 'csv'])
def test_export_round_trips_and_states_what_it_omitted(tmp_path, form):
    assert export_directory({}) is None and export_directory({'sim.report_dir': '  '}) is None
    assert export_directory({'sim.report_dir': str(tmp_path)}) == tmp_path/'dctl'
    directory = export_directory({'sim.report_dir': str(tmp_path)})
    history = export_history()
    path, count, omitted = dump_samples(directory, 'flash', history, format=form)
    assert path.parent == directory and path.suffix == '.'+form and (count, omitted) == (6, 0)
    for restored, original in zip(load_samples(path), history):
        same_sample(restored, original)
    # Both ceilings keep the newest samples and report the rest as omitted.
    path, count, omitted = dump_samples(directory, 'flash', history, format=form, max_samples=2)
    assert (count, omitted) == (2, 4)
    for restored, original in zip(load_samples(path), history[-2:]):
        same_sample(restored, original)
    small = dump_samples(directory, 'flash', history, format=form, max_bytes=1200)
    assert 0 < small[1] < 6 and small[1]+small[2] == 6
    with pytest.raises(ValueError): dump_samples(directory, 'flash', history, format=form, max_bytes=64)
    with pytest.raises(ValueError): dump_samples(directory, 'flash', [], format=form)
    with pytest.raises(ValueError): dump_samples(None, 'flash', history, format=form)
    with pytest.raises(ValueError): dump_samples(directory, 'flash', history, format='exe')


def test_export_controls_follow_the_published_report_root(root, tmp_path):
    if root is None:
        assert export_directory({'sim.report_dir': str(tmp_path)}) == tmp_path/'dctl'
        return
    from tkinter import ttk
    sensors = manager(DroneProfile.parse(camera_profile(16, 12, physics_hz=30.)))
    session = SensorSession(sensors.publisher.instance)
    values = {}
    try:
        session.connect(); notebook = ttk.Notebook(root); notebook.pack(fill='both', expand=True)
        view = DevicesView(notebook, session, ['front'], status_values=lambda: dict(values))
        notebook.add(view.page, text='Devices'); view.update(True); root.update()
        # Without the published key the controls are disabled rather than inventing a path.
        assert not view.grid.export_enabled()
        assert all(str(button['state']) == 'disabled' for button in view.export_buttons)
        view.export('front', 'json')
        assert view.export_message == 'sim.report_dir is unavailable'
        values['sim.report_dir'] = str(tmp_path)
        for i in range(10):
            sensors.tick(DroneState(0, 0, 1), i/30); session.poll()
        view.update(True); root.update()
        assert view.grid.export_enabled()
        assert all(str(button['state']) == 'normal' for button in view.export_buttons)
        pane = view.grid.panes['front']
        assert pane.display_capture is not None
        view.export('front', 'json')
        dumped = sorted((tmp_path/'dctl').glob('front-*.json'))
        assert len(dumped) == 1 and view.export_message.startswith('Saved '+dumped[0].name)
        restored = load_samples(dumped[0])
        held = [s for s, _ in pane.stream.history.values() if s.capture_id <= pane.display_capture]
        assert [s.capture_id for s in restored] == [s.capture_id for s in held]
        same_sample(restored[-1], held[-1])
        view.export('front', 'png')
        shot = sorted((tmp_path/'dctl').glob('front-*.png'))
        assert len(shot) == 1 and view.export_message == 'Saved '+shot[0].name
        from PIL import Image
        with Image.open(shot[0]) as image:
            assert image.text['capture_id'] == str(pane.display_capture)
            assert image.text['sensor_id'] == 'front'
        view.export('aux', 'json')
        assert view.export_message == 'Open a device pane first'
        with pytest.raises(ValueError): snapshot_png(None, pane)
        # A pane swapped to its text body has no drawn graphic to rasterize,
        # so the PNG is refused by name rather than saving whatever the canvas
        # last held at some other size. The sample dumps are unaffected.
        pane.text_button.invoke()
        view.export('front', 'png')
        assert view.export_message == 'the pane is showing its readout, not a graphic'
        assert len(sorted((tmp_path/'dctl').glob('front-*.png'))) == 1
        view.export('front', 'csv')
        assert view.export_message.startswith('Saved ')
        pane.text_button.invoke()
    finally: session.close(); sensors.close()


def test_tree_drag_opens_devices_and_the_close_button_unchecks(root):
    if root is None:
        # The pure half: an explicit drop target beats auto placement, and
        # the occupant of a taken cell is the pane that moves.
        assert layout.place(layout.auto_layout([]), 'a', 0, 0)['panes'][0] == dict(
            id='a', row=0, col=0, rowspan=1, colspan=1, options={})
        return
    from tkinter import ttk
    draft = camera_profile(16, 12, physics_hz=30.)
    draft['sensors'].append(dict(draft['sensors'][0], id='aux'))
    draft['sensors'].append(dict(id='scan', type='lidar.scan2d', parent='body', rate_hz=10., model={'samples': 8}))
    sensors = manager(DroneProfile.parse(draft)); session = SensorSession(sensors.publisher.instance)
    try:
        session.connect(); notebook = ttk.Notebook(root); notebook.pack(fill='both', expand=True)
        view = DevicesView(notebook, session, ['front']); notebook.add(view.page, text='Devices')
        view.update(True); root.update()
        grid = view.grid

        def click(sid):
            y = view.tree.bbox(sid)[1]+2
            for sequence in ('<ButtonPress-1>', '<ButtonRelease-1>'):
                view.tree.event_generate(sequence, x=10, y=y, rootx=100, rooty=100)

        def drag(sid, x_root, y_root):
            y = view.tree.bbox(sid)[1]+2
            view.tree.event_generate('<ButtonPress-1>', x=10, y=y, rootx=100, rooty=100)
            view.tree.event_generate('<B1-Motion>', x=40, y=y+6, rootx=130, rooty=106)
            assert view._preview is not None  # the press became a drag, not a click
            view.tree.event_generate('<ButtonRelease-1>', x=10, y=y, rootx=x_root, rooty=y_root)
            root.update()

        # The bench starts three by three, so an empty cell waits at the centre.
        assert (grid.state['rows'], grid.state['columns']) == (3, 3)
        centre = (grid.winfo_rootx()+grid.winfo_width()//2, grid.winfo_rooty()+grid.winfo_height()//2)
        # Drag aux onto that empty cell: it lands exactly there and nothing
        # else moves or closes.
        drag('aux', *centre)
        assert view._preview is None and 'aux' in grid.panes and 'aux' in grid.wanted
        cells = {p['id']: (p['row'], p['col']) for p in grid.serialize()['panes']}
        assert cells['aux'] == (1, 1) and cells['front'] == (0, 0)
        assert view.tree.set('aux', 'checked') == '☑'
        assert 'front' in grid.panes and view.tree.set('front', 'checked') == '☑'
        # A drop on an occupied cell closes the occupant and takes its cell:
        # no row or column appears, and no other pane is moved.
        corner = (grid.winfo_rootx()+grid.winfo_width()//6, grid.winfo_rooty()+grid.winfo_height()//6)
        drag('scan', *corner)
        cells = {p['id']: (p['row'], p['col']) for p in grid.serialize()['panes']}
        assert cells['scan'] == (0, 0) and 'front' not in cells
        assert grid.state['rows'] == grid.state['columns'] == 3
        assert 'front' not in grid.panes and 'front' not in grid.wanted
        assert view.tree.set('front', 'checked') == '☐'
        assert cells['aux'] == (1, 1)  # the bystander kept its cell
        # A click without a drag is not a control: the check mark is an indicator.
        click('front')
        assert 'front' not in grid.wanted and 'front' not in grid.panes
        # An absent device's grey row is the one click that acts: it forgets
        # the reservation instead of holding the cell for a lost device.
        grid.wanted.append('ghost'); view.update(True); root.update()
        assert view.tree.exists('ghost') and view.tree.set('ghost', 'checked') == '☑'
        click('ghost')
        assert 'ghost' not in grid.wanted and not view.tree.exists('ghost')
        # The pane's × is the close gesture, and the indicator follows it.
        grid.panes['aux'].close_button.invoke()
        assert 'aux' not in grid.panes and 'aux' not in grid.wanted
        assert view.tree.set('aux', 'checked') == '☐'
    finally: session.close(); sensors.close()


def test_pane_controls_are_discs_that_survive_a_narrow_header(root):
    if root is None:
        # The pure half: the tint a disc paints itself with, since Tk has no
        # alpha. Both ends of the blend are the colours themselves.
        assert theme.blend(theme.PANEL, theme.TEXT, 0) == theme.PANEL.lower()
        assert theme.blend(theme.PANEL, theme.TEXT, 1) == theme.TEXT.lower()
        assert theme.blend('#000000', '#ffffff', .5) == '#808080'
        return
    from tkinter import ttk
    sensors = manager(DroneProfile.parse(camera_profile(16, 12, physics_hz=30.)))
    session = SensorSession(sensors.publisher.instance)
    try:
        session.connect(); notebook = ttk.Notebook(root); notebook.pack(fill='both', expand=True)
        view = DevicesView(notebook, session, ['front']); notebook.add(view.page, text='Devices')
        view.update(True); root.update()
        pane = view.grid.panes['front']
        discs = (pane.close_button, pane.freeze_button, pane.text_button)
        # The title is packed last on purpose: pack hands out the cavity in
        # packing order, so the discs are served before the name is.
        assert pane.header.pack_slaves() == [*discs, pane.title]
        assert all(disc.winfo_reqwidth() == 2*CircleButton.RADIUS for disc in discs)
        # Six columns across the same bench leave a header far too narrow for
        # the device name. The name is what gives way; the controls never do.
        for _ in range(3): view.grid.grow('columns')
        root.update(); root.update_idletasks()
        assert pane.header.winfo_width() < pane.title.winfo_reqwidth()
        assert pane.title.winfo_width() < pane.title.winfo_reqwidth()
        assert all(disc.winfo_ismapped() for disc in discs)
        # Freeze is one of those discs now, and it lights while it holds.
        assert not pane.frozen.get() and not pane.freeze_button.active
        pane.freeze_button.invoke()
        assert pane.frozen.get() and pane.freeze_button.active
        pane.freeze_button.invoke()
        assert not pane.frozen.get() and not pane.freeze_button.active
        # The info disc lights with the strip it opens, from either direction.
        pane.text_button.invoke()
        assert pane.show_readout and pane.text_button.active
        pane.text_button.invoke()
        assert not pane.show_readout and not pane.text_button.active
    finally: session.close(); sensors.close()


def test_the_info_disc_swaps_the_graphic_for_the_text(root):
    if root is None:
        # The pure half: only a pane actually painting frames wants a share of
        # the video budget, so a text body drops out of the divisor entirely.
        stream = SimpleNamespace(entry={'transport': 'stream'})
        assert _wants_video(SimpleNamespace(stream=stream, show_readout=False))
        assert not _wants_video(SimpleNamespace(stream=stream, show_readout=True))
        assert not _wants_video(SimpleNamespace(
            stream=SimpleNamespace(entry={'transport': 'compact'}), show_readout=False))
        return
    from tkinter import ttk
    sensors = manager(DroneProfile.parse(camera_profile(16, 12, physics_hz=30.)))
    session = SensorSession(sensors.publisher.instance)
    try:
        session.connect(); notebook = ttk.Notebook(root); notebook.pack(fill='both', expand=True)
        view = DevicesView(notebook, session, ['front']); notebook.add(view.page, text='Devices')
        view.update(True); root.update()
        for i in range(30): sensors.tick(DroneState(0, 0, 1), i/30); session.poll()
        grid = view.grid; pane = grid.panes['front']
        # Settle first: the opening paint rebuilds the renderer on the manifest
        # revision, which would throw the counter away with it.
        grid.paint(session.report(), budget_s=1.); root.update()
        drawn = []
        undecorated = pane.renderer.draw
        pane.renderer.draw = lambda sample: (drawn.append(sample.capture_id), undecorated(sample))[1]
        pane.last_paint = -1e9; grid.paint(session.report(), budget_s=1.); root.update()
        assert drawn and pane.renderer.widget.winfo_manager() == 'pack'
        graphic_height = pane.renderer.widget.winfo_height()

        # The disc swaps the bodies: the text takes the cell the graphic had,
        # and the graphic is unpacked rather than left underneath.
        pane.text_button.invoke(); root.update()
        assert pane.renderer.widget.winfo_manager() == '' and pane.readout.winfo_manager() == 'pack'
        assert pane.readout.winfo_height() >= graphic_height
        # It wraps to the cell it now owns, not to the strip's fixed width.
        assert int(pane.readout['wraplength']) == max(80, pane.readout.winfo_width()-8)

        # Nothing is drawn into the body that left, but the readout keeps
        # following the stream: the numbers advance while the canvas rests.
        for i in range(30, 60): sensors.tick(DroneState(0, 0, 1), i/30); session.poll()
        held, count = pane.display_capture, len(drawn)
        pane.last_paint = -1e9; grid.paint(session.report(), budget_s=1.); root.update()
        assert len(drawn) == count and pane.display_capture > held

        # Swapping back redraws at once, at the size the graphic actually has.
        pane.text_button.invoke()
        grid.paint(session.report(), budget_s=1.); root.update()
        assert len(drawn) > count and pane.renderer.widget.winfo_height() > 1

        # Even frozen, where the held capture would otherwise short-circuit the
        # paint, the returning graphic is redrawn instead of coming back blank.
        pane.frozen.set(True)
        pane.last_paint = -1e9; grid.paint(session.report(), budget_s=1.)
        pane.text_button.invoke(); grid.paint(session.report(), budget_s=1.); root.update()
        count = len(drawn)
        pane.text_button.invoke(); grid.paint(session.report(), budget_s=1.); root.update()
        assert len(drawn) == count+1 and drawn[-1] == pane.display_capture
    finally: session.close(); sensors.close()


def test_arrange_mode_edits_the_grid_while_panes_stay_paused(root):
    if root is None:
        assert layout.occupied(layout.auto_layout(['a'])) == {(0, 0)}
        return
    from tkinter import ttk
    draft = camera_profile(16, 12, physics_hz=30.)
    draft['sensors'].append(dict(draft['sensors'][0], id='aux'))
    sensors = manager(DroneProfile.parse(draft)); session = SensorSession(sensors.publisher.instance)
    try:
        session.connect(); notebook = ttk.Notebook(root); notebook.pack(fill='both', expand=True)
        view = DevicesView(notebook, session, ['front']); notebook.add(view.page, text='Devices')
        view.update(True); root.update()
        for i in range(30): sensors.tick(DroneState(0, 0, 1), i/30); session.poll()
        grid = view.grid
        grid.paint(session.report(), budget_s=1.); root.update()
        pane = grid.panes['front']; painted = pane.display_capture
        assert pane.renderer.widget.winfo_manager() == 'pack'
        # The graphic owns the cell by default; the info disc swaps the text
        # in for it, rather than squeezing a strip in underneath it.
        assert pane.readout.winfo_manager() == ''
        pane.text_button.invoke()
        assert pane.readout.winfo_manager() == 'pack' and pane.renderer.widget.winfo_manager() == ''
        root.update()
        # The text takes the whole body: under the header, above the grip,
        # which keeps its slab however short the pane gets.
        assert pane.readout.winfo_y() < pane.resize_grip.winfo_y()
        assert pane.readout.winfo_height() > pane.header.winfo_height()
        assert pane.resize_grip.winfo_manager() == 'pack'
        assert next(p for p in grid.serialize()['panes'] if p['id'] == 'front')['readout'] is True
        pane.text_button.invoke()
        assert pane.readout.winfo_manager() == '' and pane.renderer.widget.winfo_manager() == 'pack'
        assert pane.resize_grip.winfo_manager() == 'pack'
        pane.text_button.invoke()  # leave the readout shown for the assertions below
        # Arranging collapses the bench to its skeleton and pauses painting.
        view.arrange.set(True); view._arrange_toggled(); root.update()
        assert pane.renderer.widget.winfo_manager() == '' and not pane.readout.winfo_manager()
        # The info disc is remembered while collapsed, but never unpacks a
        # strip onto the skeleton.
        pane.text_button.invoke()
        assert pane.show_readout is False and pane.readout.winfo_manager() == ''
        pane.text_button.invoke()  # shown again: the strip returns with the body at exit
        assert all(button.winfo_ismapped() for button, _, _ in view.structure)
        # The default three-by-three leaves empty rows and columns to remove.
        assert str(view.structure[1][0]['state']) == 'normal' and str(view.structure[3][0]['state']) == 'normal'
        grid.grow('columns'); grid.grow('columns'); grid.grow('rows'); root.update()
        assert (grid.state['rows'], grid.state['columns']) == (4, 5)
        # The sashes fatten and the empty cells outline themselves as targets.
        assert grid.sashes[('col', 0)].winfo_width() == 8 and grid.sashes[('row', 0)].winfo_height() == 8
        assert set(grid.targets) == {(r, c) for r in range(4) for c in range(5)} - {(0, 0)}
        # A device dropped while arranging is born collapsed, in its target cell.
        view._press(SimpleNamespace(y=view.tree.bbox('aux')[1]+2, x_root=100, y_root=100))
        view._drag(SimpleNamespace(x_root=200, y_root=140))
        view._release(SimpleNamespace(x_root=grid.winfo_rootx()+grid.winfo_width()*5//6,
                                      y_root=grid.winfo_rooty()+grid.winfo_height()*7//8))
        root.update()
        assert grid.panes['aux'].renderer.widget.winfo_manager() == ''
        assert next(p for p in grid.serialize()['panes'] if p['id'] == 'aux')['row'] == 3
        assert next(p for p in grid.serialize()['panes'] if p['id'] == 'aux')['col'] == 4
        # Both edge row and column now hold a pane, so − refuses to remove them.
        assert not grid.edge_is_empty('rows') and not grid.edge_is_empty('columns')
        assert str(view.structure[1][0]['state']) == 'disabled'
        assert str(view.structure[3][0]['state']) == 'disabled'
        # The menu span commands merge cells explicitly.
        grid.grow_span('front', 0, 1); grid.grow_span('front', 1, 0)
        front = next(p for p in grid.serialize()['panes'] if p['id'] == 'front')
        assert (front['rowspan'], front['colspan']) == (2, 2)
        grid.shrink_span('front')
        assert (next(p for p in grid.serialize()['panes'] if p['id'] == 'front')['rowspan'],
                next(p for p in grid.serialize()['panes'] if p['id'] == 'front')['colspan']) == (1, 1)
        # Painting is paused while intake keeps draining, as under freeze.
        for i in range(30, 60): sensors.tick(DroneState(0, 0, 1), i/30); session.poll()
        grid.paint(session.report(), budget_s=1.)
        assert pane.display_capture == painted
        # Leaving arrange restores each pane's chosen body -- the text one
        # here, the graphic next door -- thins the sashes, and repaints.
        view.arrange.set(False); view._arrange_toggled(); root.update()
        assert pane.show_readout
        assert pane.readout.winfo_manager() == 'pack' and pane.renderer.widget.winfo_manager() == ''
        assert grid.panes['aux'].renderer.widget.winfo_manager() == 'pack'
        assert grid.panes['aux'].readout.winfo_manager() == ''
        assert not grid.targets and grid.sashes[('col', 0)].winfo_width() == 4
        assert not any(button.winfo_ismapped() for button, _, _ in view.structure)
        grid.paint(session.report(), budget_s=1.)
        assert pane.display_capture > painted
    finally: session.close(); sensors.close()


def test_pop_out_during_arrange_keeps_its_body(root):
    if root is None: return
    from tkinter import ttk
    sensors = manager(DroneProfile.parse(camera_profile(16, 12, physics_hz=30.)))
    session = SensorSession(sensors.publisher.instance)
    try:
        session.connect(); notebook = ttk.Notebook(root); notebook.pack(fill='both', expand=True)
        view = DevicesView(notebook, session, ['front']); notebook.add(view.page, text='Devices')
        view.update(True); root.update()
        grid = view.grid
        view.arrange.set(True); view._arrange_toggled(); root.update()
        assert grid.panes['front'].renderer.widget.winfo_manager() == ''  # a skeleton cell
        grid.pop_out('front'); root.update()
        # A floating window is never part of the skeleton: it shows its body.
        assert grid.panes['front'].renderer.widget.winfo_manager() == 'pack'
        # A profile change while arranging recreates the panes collapsed and
        # pops the recorded one back out -- still with its body, not a strip.
        sensors.apply(sensors.profile)
        session.probe.last_probe = -1e9; session.connect()
        view.update(True); root.update()
        assert 'front' in grid.popped
        assert grid.panes['front'].renderer.widget.winfo_manager() == 'pack'
        # Leaving arrange restores the grid bodies without touching the window.
        view.arrange.set(False); view._arrange_toggled(); root.update()
        assert grid.panes['front'].renderer.widget.winfo_manager() == 'pack'
        grid.return_to_grid('front'); root.update()
        assert grid.panes['front'].renderer.widget.winfo_manager() == 'pack'
    finally: session.close(); sensors.close()


def test_freeze_armed_before_any_record_is_not_invalidated_by_the_first_epoch(root):
    if root is None:
        assert synchronized_capture([]) == (None, False)
        return
    sensors, session = reference_session()
    try:
        grid = DeviceGrid(root, session); grid.pack(fill='both', expand=True)
        grid.wanted = ['nav_left', 'imu']; grid.set_visible(True); root.update()
        grid.freeze_all(True)
        assert grid.freeze_message == 'no shared capture; holding None'
        for i in range(30):
            sensors.tick(DroneState(0, 0, 1), i/300); session.poll()
        grid.paint(session.report(), budget_s=1.)
        # The first epoch arriving is not a reset: the armed freeze survives it.
        assert 'invalidated' not in grid.freeze_message
        grid.freeze_all(True); grid.paint(session.report(), budget_s=1.)
        assert 'shared capture' in grid.freeze_message and grid.freeze_capture is not None
        grid.destroy()
    finally: session.close(); sensors.close()


def test_reset_is_counted_a_profile_change_is_labelled(root):
    if root is None: return
    sensors, session = reference_session()
    try:
        grid = DeviceGrid(root, session); grid.pack(fill='both', expand=True)
        grid.wanted = ['imu']; grid.set_visible(True); root.update()
        for i in range(30):
            sensors.tick(DroneState(0, 0, 1), i/300); session.poll()
        grid.panes['imu'].last_paint = -1e9
        grid.paint(session.report(), budget_s=1.)
        assert 'profile changed' not in grid.panes['imu'].readout['text']
        # A simulator reset bumps the revision but not the generation: its
        # loss is counted as drops, never announced as a profile change.
        sensors.reset()
        for i in range(30):
            sensors.tick(DroneState(0, 0, 1), .1+i/300); session.poll()
        grid.panes['imu'].last_paint = -1e9
        grid.paint(session.report(), budget_s=1.)
        assert 'profile changed' not in grid.panes['imu'].readout['text']
        sensors.apply(sensors.profile)
        session.probe.last_probe = -1e9; session.connect()
        grid.reconcile(); root.update()
        for i in range(30):
            sensors.tick(DroneState(0, 0, 1), .2+i/300); session.poll()
        grid.paint(session.report(), budget_s=1.)
        assert 'profile changed' in grid.panes['imu'].readout['text']
        grid.destroy()
    finally: session.close(); sensors.close()


def test_the_tree_reports_a_pairs_worst_member(root):
    if root is None: return
    from tkinter import ttk
    sensors, session = reference_session()
    try:
        notebook = ttk.Notebook(root); notebook.pack(fill='both', expand=True)
        view = DevicesView(notebook, session, ['nav_left', 'nav_right'])
        notebook.add(view.page, text='Devices')
        for i in range(90):
            sensors.tick(DroneState(0, 0, 1), i/300); session.poll()
        view.update(True); root.update()
        # The pair row shows its members' grade; the session report has no
        # stereo rows, so a pair that graded from nothing showed '○' before.
        assert view.tree.set('stereo:nav_stereo', 'health') == DOTS['ok']
    finally: session.close(); sensors.close()
