"""Independent rays, one manifest, a source builder that round-trips, and the panes."""
from dataclasses import replace
import json
from pathlib import Path
import uuid

import numpy as np
import pytest

from dalg.algo.lidar import LidarInverseModel
from dalg.profiles import (Source, load_profile, load_profiles, save_sources_profile,
                          source_errors)
from dcmn.maps import GridGeometry
from dcmn.sensors import Sample

ROOT = Path(__file__).resolve().parents[1]


def scan_sample(ranges=(3.,), *, angles=0., increment=0., confidence=None, z=1.5):
    pose = np.eye(4)
    pose[:3, 3] = (1.5, 2.5, z)  # Sensor forward is map east.
    return Sample('scan', 'lidar.scan2d', 1, 1, 2., 1,
        {'pose_world': pose.tolist(), 'calibration': {'samples': len(ranges),
         'angle_min_deg': angles, 'angle_increment_deg': increment, 'elevation_deg': 0.}},
        fields={'range_m': np.array(ranges), 'confidence': np.array(
            [255]*len(ranges) if confidence is None else confidence)},
        entry={'model': {'min_range_m': .1, 'max_range_m': 10.}})


def test_lidar_clears_only_valid_ray_cells_and_retains_unknown():
    model = LidarInverseModel(GridGeometry.from_extent(8, 6, 1))
    model.observe(scan_sample((3., np.nan, 4.), angles=0, increment=90, confidence=[255, 0, 0]))
    result = model.preview().grid
    assert result.free[2, 1:4].all()
    assert result.occupied[2, 4]
    assert result.observed.sum() == 4
    assert (model.grid.observed_ms[result.observed] == 2000).all()
    model.observe(replace(scan_sample(), sequence=2, sim_time_s=3))
    assert (model.grid.observed_ms[result.observed] == 3000).all()


def test_lidar_pose_slab_and_extent_clipping():
    geometry = GridGeometry.from_extent(4, 4, 1, origin_x_m=-2, origin_y_m=-2)
    model = LidarInverseModel(geometry)
    sample = scan_sample((8.,))
    pose = np.eye(4); pose[:3, 3] = (-4., -.5, 1.5)
    model.observe(replace(sample, payload=dict(sample.payload, pose_world=pose.tolist())))
    assert model.preview().grid.free[1].all()
    assert not model.preview().grid.occupied.any()  # Return beyond extent.
    above = LidarInverseModel(geometry)
    above.observe(scan_sample(z=3.))
    assert not above.grid.observed.any()  # Slab's top is exclusive.
    # Rotated forward vector points south; no heading/body-pose guessing.
    rotated = LidarInverseModel(GridGeometry.from_extent(8, 8, 1))
    pose[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    pose[:3, 3] = (1.5, 2.5, 1.5)
    rotated.observe(replace(scan_sample(), payload=dict(scan_sample().payload, pose_world=pose.tolist())))
    assert rotated.preview().grid.occupied[5, 1]


def test_invalid_lidar_record_does_not_change_evidence():
    model = LidarInverseModel(GridGeometry.from_extent(8, 6, 1))
    sample = scan_sample()
    with pytest.raises(ValueError, match='arrays'):
        model.observe(replace(sample, fields={'range_m': np.ones(2), 'confidence': np.ones(2)}))
    assert not model.grid.observed.any()


def test_profile_sources_validate_and_round_trip(tmp_path):
    original = load_profiles(['ground-plane-baseline', 'lidar-baseline'], ROOT)
    path = tmp_path/'sources.json'
    save_sources_profile(path, name='camera-lidar', sources=original.sources)
    loaded = load_profile(str(path), ROOT)
    assert loaded.sources == original.sources
    assert set(json.loads(path.read_text())) == {'name', 'sources'}
    manifest = {'sensors': {'front': {'type': 'camera.rgb'}, 'scan': {'type': 'lidar.scan2d'}}}
    explicit = [Source('front', 'ground_plane'), Source('scan', 'lidar_inverse')]
    assert source_errors(explicit, manifest) == {}
    assert 'absent' in source_errors([Source('missing', 'lidar_inverse')], manifest)[0]
    assert 'needs' in source_errors([Source('front', 'lidar_inverse')], manifest)[0]
    assert 'duplicate' in source_errors([explicit[1]]*2, manifest)[1]
    assert len(Source('a'*48, 'lidar_inverse').id) == 48
    assert Source('a'*48, 'lidar_inverse').id != Source('a'*47+'b', 'lidar_inverse').id
    assert source_errors([Source('scan', 'lidar_inverse', {'min_confidence': 0})])
    assert source_errors([Source('front', 'ground_plane', {'column_stride': 0})])


def test_two_sources_publish_one_manifest_and_max_cost(tmp_path):
    from dalg.profiles import Profile
    from dalg.sources import EvidenceSources
    from dcmn.maps import MapSession
    from dnav.policy import load_policy, build_cost_map
    from dnav.planners import build
    from types import SimpleNamespace
    profile = Profile('pair', (Source('front', 'ground_plane'), Source('scan', 'lidar_inverse')), 'd')
    model = dict(width_px=64, height_px=48, fx_px=50, fy_px=50, cx_px=32, cy_px=24)
    streams = {'front': SimpleNamespace(model=model), 'scan': SimpleNamespace()}
    session = SimpleNamespace(devices={'front': {'type': 'camera.rgb'}, 'scan': {'type': 'lidar.scan2d'}},
                              identity=('test', 1), reset_epoch=0)
    producer = EvidenceSources('sources-'+uuid.uuid4().hex[:8], profile, session, streams,
                               GridGeometry.from_extent(8., 6., 1.), context=dict(clock_epoch=4))
    consumer = MapSession(producer.publisher.instance)
    try:
        producer.states['scan-lidar_inverse'].evidence.observe(scan_sample())
        producer.publish(2)
        consumer.poll()
        assert len(consumer.sources) == 2
        # One generation, one context: every source carries the same epochs.
        assert {entry['mapping_epoch'] for entry in consumer.sources.values()} == {1}
        assert consumer.context['clock_epoch'] == 4
        grids = {source: consumer.latest(source) for source in consumer.sources}
        assert grids['front-ground_plane'].never_observed.all()
        lidar = grids['scan-lidar_inverse']
        assert lidar.observed.any() and (lidar.occupancy == 255).any()
        cost = build_cost_map(grids, load_policy('default', ROOT))
        assert np.array_equal(cost.cost, np.maximum(*(layer.cost for layer in cost.layers)))
        assert cost.blocked[2, 4]
        route = build('astar').plan(cost, (1.5, 2.5, 1.5), (6.5, 2.5, 1.5), cost.policy)
        assert route.status == 'ok'
        assert all(not cost.is_blocked(point[0], point[1]) for point in route.points)
        assert route.length_m > 5
    finally:
        consumer.close(); producer.close()


def test_source_builder_save_reload_and_validation(tmp_path):
    import tkinter as tk
    from tkinter import ttk
    from dtest.tkfixture import hidden_tk
    from dalg.source_editor import SourceEditor

    profile = load_profile('ground-plane-baseline', ROOT)
    manifest = {'primary_camera': 'front', 'sensors': {
        'front': {'type': 'camera.rgb', 'rate_hz': 30},
        'scan': {'type': 'lidar.scan2d', 'rate_hz': 5}}}
    with hidden_tk() as root:
        page = ttk.Frame(root)
        editor = SourceEditor(page, profile, tk, ttk, root=ROOT)
        editor.set_manifest(manifest)
        editor.add_source('scan', 'lidar_inverse')
        assert editor.validate()
        path = tmp_path/'saved.json'
        editor.save(path)
        saved = load_profile(str(path), ROOT)
        assert saved.sources == (Source('primary_camera', 'ground_plane'), Source('scan', 'lidar_inverse'))
        assert set(json.loads(path.read_text())) == {'name', 'sources'}
        editor.rows.selection_set('1')
        editor.move_selected(-1)
        assert editor.sources[0].sensor == 'scan'
        editor.move_selected(1)
        editor.add_source('front', 'lidar_inverse')
        assert not editor.validate()
        assert 'needs lidar.scan2d' in editor.rows.item('2', 'values')[1]
        editor.remove_selected()
        # The selector and the camera it resolves to are the same source.
        editor.add_source('front', 'ground_plane')
        assert not editor.validate()
        assert 'duplicate' in editor.rows.item('2', 'values')[1]
        editor.remove_selected()
        editor.rows.selection_set('1'); editor._select()
        editor.setting_vars['min_confidence'][0].set('0')
        assert not editor.validate()
        with pytest.raises(ValueError): editor.save(tmp_path/'invalid.json')
        editor.setting_vars['min_confidence'][0].set('1')
        assert editor.validate()


def test_the_editor_shows_runtime_geometry_and_never_saves_it(tmp_path):
    import tkinter as tk
    from tkinter import ttk
    from dtest.tkfixture import hidden_tk
    from dalg.source_editor import SourceEditor

    with hidden_tk() as root:
        editor = SourceEditor(ttk.Frame(root), load_profile('lidar-baseline', ROOT), tk, ttk, root=ROOT)
        editor.set_runtime(geometry=GridGeometry.from_extent(40, 40, .5, origin_x_m=-20, origin_y_m=-20),
                           basis='goal', state='RUNNING')
        assert '[-20, -20] - [20, 20] m' in editor.runtime.get()
        editor.save(tmp_path/'p.json')
        assert json.loads((tmp_path/'p.json').read_text()) == {
            'name': 'lidar-baseline', 'sources': [{'algorithm': 'lidar_inverse', 'sensor': 'scan', 'settings': {}}]}


def test_combined_cost_renderer_keeps_lidar_obstacles_visible():
    from dcmn.maps import EvidenceGrid
    from dcmn.map_pane import render_raster
    from dcmn import theme
    from dnav.policy import build_cost_map, load_policy
    geometry = GridGeometry.from_extent(8, 6, 1)
    camera = EvidenceGrid.blank(geometry, 'camera')
    lidar = EvidenceGrid.blank(geometry, 'lidar')
    lidar.occupancy[0, 2, 4] = 254; lidar.observed_ms[0, 2, 4] = 1000
    lidar.occupancy[0, 2, 2] = 0; lidar.observed_ms[0, 2, 2] = 1000
    cost = build_cost_map({'camera': camera, 'lidar': lidar}, load_policy('default', ROOT))
    rgb = render_raster(camera, mode='cost', cost=cost.cost, never_observed=cost.never_observed)
    assert tuple(rgb[2, 4]) == theme.rgb(theme.DANGER)
    assert tuple(rgb[2, 2]) != theme.rgb(theme.UNOBSERVED)
    assert tuple(rgb[0, 0]) == theme.rgb(theme.UNOBSERVED)


def test_the_grid_panes_the_live_selector_and_the_cost_layers(tmp_path, monkeypatch):
    import tkinter as tk
    from tkinter import ttk
    import dalg.dalg as ui
    from dalg.run import DalgRun
    from dnav.dnav import CostTab, STACK
    from dnav.plan import NavRun
    from dnav.policy import load_policy
    from dtest.provider import FixtureProvider
    from dtest.tkfixture import hidden_root

    root = hidden_root()
    monkeypatch.setattr(tk, 'Tk', lambda: root)
    monkeypatch.setattr(ui, 'restore_window_geometry', lambda *a: None)
    monkeypatch.setattr(ui, 'save_window_geometry', lambda *a: None)
    instance = 'panes-'+uuid.uuid4().hex[:8]
    provider = FixtureProvider(instance, tmp_path, sensors=('scan', 'front'))
    run = DalgRun(instance, load_profiles(['ground-plane-baseline', 'lidar-baseline'], ROOT), ROOT)
    nav = NavRun(instance, ROOT, policy=load_policy('default', ROOT), goal=(8., 2., None))
    window = None
    try:
        provider.fly([(2., 0.), (2., 2.5)])
        for _ in range(90): provider.step(); run.step(); nav.step()
        window = ui.Window(run)
        window.notebook.select(window.grids); window.update()
        assert set(window.map_panes) == {'front-ground_plane', 'scan-lidar_inverse'}
        for sid, pane in window.map_panes.items():
            assert pane.state.grid is run.sources.states[sid].evidence.latest
            pane.snapshot().save(tmp_path/f'{sid}.png')
        window.source.set('scan-lidar_inverse')
        window.notebook.select(window.live); window.update()
        assert window.source.get() == 'scan-lidar_inverse'
        assert 'scan-lidar_inverse' in window.lidar_views
        assert 'coverage=' in window.status.get() and 'truth' not in window.status.get()
        nav.replan(force=True)
        tab = CostTab(ttk.Frame(root), nav, tk, ttk)
        now = nav.sim_time_s()
        tab.paint_map(now)
        # The stack, then evidence, cost and blocked-on-evidence per source.
        assert len(tab.view_combo['values']) == 1 + 3 * len(window.map_panes)
        for sid in window.map_panes:
            tab.view_var.set(f'evidence: {sid}'); tab.paint_map(now)
            assert tab.pane.state.grid.source == sid
            tab.view_var.set(f'cost: {sid}'); tab.paint_map(now)
            assert np.array_equal(tab.pane.state.cost, nav.cost_map.layer(sid).cost)
            tab.view_var.set(f'blocked on evidence: {sid}'); tab.paint_map(now)
            assert tab.pane.state.mode == 'blocked' and tab.pane.state.grid.source == sid
        tab.view_var.set(STACK); tab.paint_map(now)
        assert np.array_equal(tab.pane.state.never_observed, nav.cost_map.never_observed)
        tab.pane.snapshot().save(tmp_path/'combined-cost.png')
    finally:
        if window is not None: window.close()
        root.destroy(); nav.close(); run.close(); provider.close()


def test_planner_waits_for_every_declared_source(tmp_path):
    from dcmn.context import Context
    from dcmn.maps import MapPublisher, EvidenceGrid
    from dnav.plan import NavRun
    from dnav.policy import load_policy
    geometry = GridGeometry.from_extent(8, 6, 1)
    instance = 'missing-'+uuid.uuid4().hex[:8]
    context = Context(instance); context.start(tmp_path)
    context.publish_pose(dict(x_m=1.5, y_m=2.5, z_m=1.5, heading_deg=90., roll_deg=0., pitch_deg=0.), 2.)
    publisher = MapPublisher(instance, geometry, [{'id': 'camera'}, {'id': 'lidar'}])
    run = NavRun(instance, ROOT, policy=load_policy('default', ROOT), goal=(6.5, 2.5, None))
    try:
        publisher.publish_grid(EvidenceGrid.blank(geometry, 'camera'), 2.)
        run.step(); run.replan(force=True)
        assert run.route.status == 'stale_map'
        assert 'waiting for evidence from: lidar' in run.route.reason
        assert run.plans == 0
    finally:
        run.close(); publisher.close(); context.close()


def test_a_short_window_scrolls_the_profile_form_instead_of_crushing_its_lists():
    """At 1124x550 grid shrank both sensor lists to nine pixels.

    They took their entry, the algorithm box and every add/remove button with
    them, and the strip painted leftover pixels. A form taller than its window
    has to scroll; the lists must always get the height they ask for.
    """
    import tkinter as tk
    from tkinter import ttk
    from dalg.source_editor import SourceEditor
    from dcmn.scroll import Scrollable
    from dtest.tkfixture import hidden_tk, mapped_root_available

    profile = load_profile('ground-plane-baseline', ROOT)
    if not mapped_root_available():
        # Headless half: the structure that makes the mapped half true.
        with hidden_tk() as root:
            page = ttk.Frame(root)
            editor = SourceEditor(page, profile, tk, ttk, root=ROOT)
            assert isinstance(editor._scroll, Scrollable)
            ancestors, widget = [], editor.sensors
            while widget is not None:
                ancestors.append(widget); widget = widget.master
            assert editor._scroll.inner in ancestors
        return
    from dcmn.tktheme import apply_theme
    from dtest.tkfixture import hidden_root

    # Through the fixture like every other root; mapped only because this
    # half is opted into, and the collapse only exists on a mapped window.
    root = hidden_root()
    root.deiconify()
    try:
        apply_theme(root)
        root.geometry('1124x550+0+0')
        notebook = ttk.Notebook(root)
        notebook.grid(row=0, column=0, sticky='nsew')
        root.rowconfigure(0, weight=1); root.columnconfigure(0, weight=1)
        page = ttk.Frame(notebook, padding=12)
        notebook.add(page, text='Profile')
        editor = SourceEditor(page, profile, tk, ttk, root=ROOT)
        for _ in range(10): root.update()
        for tree in (editor.sensors, editor.rows):
            assert tree.winfo_height() >= tree.winfo_reqheight(), \
                f'{tree} was given {tree.winfo_height()} px of {tree.winfo_reqheight()}'
    finally:
        root.destroy()


