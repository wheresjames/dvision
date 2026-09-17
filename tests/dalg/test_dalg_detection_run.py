"""dalg running a detector beside evidence on the ideal fixture, with one red pad on its floor.

A real session context, sensor plane, detections plane and archive; no
simulator, world file or truth channel. The provider paints the pad (B14); the
only way dalg knows about it is its camera. One loose check guards the
projection convention -- the estimate lies within 2 m of the pad, which a sign
or pitch error would miss by far more. It is not a detector-quality metric.
"""
from __future__ import annotations

import json
import math
import uuid
from collections import Counter
from pathlib import Path

import pytest

from dalg.detect.projection import replay
from dalg.profiles import load_profiles
from dalg.run import DalgRun
from dcmn.archive import ArchiveReader, detections_listing, estimate_trace, render_detection
from dcmn.context import Context
from dcmn.detections import DetectionsReader
from dtest.provider import FixtureProvider
from dtest.tkfixture import hidden_tk

ROOT = Path(__file__).resolve().parents[2]
PAD = (4.0, 0.0, 0.6)


class Rig:
    def __init__(self, tmp_path, profiles=('lidar-baseline', 'red-disc'), *, heading=90., pads=(PAD,)):
        self.instance = 'det-' + uuid.uuid4().hex[:8]
        self.provider = FixtureProvider(self.instance, tmp_path / 'session', sensors=('scan', 'front'),
                                        start=(0., 0., 1.5, heading), pads=pads)
        self.run = DalgRun(self.instance, load_profiles(list(profiles), ROOT), ROOT)

    def tick(self, count=1):
        for _ in range(count):
            self.provider.step()
            self.run.sensor_session.probe.last_probe = -1e9
            self.run.step()

    def close(self):
        self.run.close()
        self.provider.close()


@pytest.fixture
def rig(tmp_path):
    made = []

    def build(*args, **kwargs):
        made.append(Rig(tmp_path, *args, **kwargs))
        return made[-1]
    yield build
    for item in made:
        try: item.close()
        except Exception: pass


def events(run_or_directory):
    directory = run_or_directory if isinstance(run_or_directory, Path) else run_or_directory.recorder.directory
    return ArchiveReader(directory)


def test_a_seen_pad_becomes_a_confirmed_estimate_on_the_plane_near_the_pad(rig):
    r = rig()
    r.tick(120)
    value = DetectionsReader(r.instance).read()
    assert value['producer'] == 'dalg' and value['revision'] >= 1
    (detector,) = value['detectors']
    assert detector['id'] == 'front-red_disc' and detector['labels'] == ['landing_pad']
    assert detector['state'] == 'running' and detector['processed'] >= 15
    (estimate,) = value['estimates']
    assert estimate['confirmed'] and not estimate['stale'] and estimate['count'] >= 3
    assert math.dist(estimate['xy_m'], PAD[:2]) < 2.0, 'the projection convention (loose, never tightened)'
    assert value['detections'] and value['detections'][0]['estimate'] == estimate['id']
    assert abs(value['time_s'] - r.run.sim_time_s()) < 1.5, 'the plane is stamped in sim time'


def test_every_frame_is_archived_empty_ones_too_and_projections_replay(rig):
    r = rig(heading=270.)  # facing away from the pad
    r.tick(45)
    r.provider.teleport(0., 0., heading_deg=90.)
    r.tick(45)
    directory = r.run.recorder.directory
    r.run.recorder.close()
    reader = events(directory)
    assert reader.validate()['complete']
    kinds = Counter(e['type'] for e in reader.events())
    frames = [e['data'] for e in reader.events() if e['type'] == 'detector.frame']
    assert kinds['detector.configured'] == 1
    assert any(f['detections'] == 0 for f in frames) and any(f['detections'] for f in frames), \
        'empty frames are recorded on purpose'
    assert kinds['detector.frame'] == r.run.detectors.states['front-red_disc'].processed
    assert kinds['estimate.opened'] == 1 and kinds['estimate.confirmed'] == 1
    assert kinds['detections.published'] >= kinds['detector.frame']
    detections = [e for e in reader.events() if e['type'] == 'detector.detection']
    for event in detections:
        data = event['data']
        assert replay(data['projection_inputs']).as_dict() == data['projection'], data['id']
        assert data['projection_inputs']['camera']['pitch_deg'] == pytest.approx(-8.), 'the camera pose, not the body'
        assert 'crop' in event.get('images', {})
    assert 'frame' in detections[0]['images'], 'the full frame of the observation that opened the estimate'
    listing = detections_listing(reader)
    assert listing['frames']['front-red_disc']['processed'] == len(frames)
    assert listing['frames']['front-red_disc']['with_detections'] == len(detections)
    trace = estimate_trace(reader, 'landing_pad-1')
    assert [row['event'] for row in trace[:2]] == ['observation', 'opened']
    out = render_detection(reader, 'D1', directory.parent / 'd1.png')
    assert out.is_file() and out.stat().st_size > 1000
    with pytest.raises(KeyError):
        estimate_trace(reader, 'landing_pad-99')


def test_the_report_gains_a_detectors_section_an_image_and_one_crop_per_estimate(rig):
    r = rig()
    r.tick(90)
    report = Path(r.run.report_dir)
    r.close()
    summary = json.loads((report / 'summary.json').read_text())
    assert summary['schema_version'] == 4
    section = summary['detectors']
    (health,) = section['detectors']
    assert health['id'] == 'front-red_disc' and health['processed'] > 0 and health['frames_with_detections'] > 0
    assert health['detections_by_label'] == {'landing_pad': health['frames_with_detections']}
    assert health['latency_ms_p50'] is not None and health['hz'] == pytest.approx(5.0, rel=.2)
    (estimate,) = section['estimates']
    assert estimate['confirmed'] and estimate['id'] in section['crops']
    assert (report / section['crops'][estimate['id']]['image']).is_file()
    assert section['image'] == 'detections.png' and (report / 'detections.png').is_file()


def test_a_mapping_reset_clears_estimates_and_says_so(rig):
    r = rig()
    r.tick(60)
    assert r.run.detectors.estimates()
    Context(r.instance).request_reset((-10., -10., 20., 20.))
    r.tick(3)
    assert r.run._mapping_epoch == 2
    r.run.recorder.close()
    reader = events(r.run)
    reasons = [e['data']['cause'] for e in reader.events() if e['type'] == 'estimate.cleared']
    assert reasons == ['operator mapping reset']
    assert DetectionsReader(r.instance).read()['context']['mapping_epoch'] == 2


def test_evidence_alone_publishes_no_detections_plane(rig):
    r = rig(('lidar-baseline',))
    r.tick(30)
    assert r.run.detectors is None and DetectionsReader(r.instance).read() == {}
    assert not any(e['type'].startswith(('detector.', 'estimate.')) for e in events(r.run).events())


def test_a_failing_detector_is_counted_and_reported_while_evidence_carries_on(rig, monkeypatch):
    r = rig()
    r.tick(30)
    state = r.run.detectors.states['front-red_disc']

    def broken(frame): raise RuntimeError('lens cap on')
    monkeypatch.setattr(state.detector, 'detect', broken)
    published = r.run.counters['publications']
    r.tick(30)
    assert state.failed >= 3 and state.state == 'failed' and 'lens cap on' in state.last_error
    assert r.run.counters['publications'] > published, 'evidence kept publishing'
    value = DetectionsReader(r.instance).read()
    assert value['detectors'][0]['state'] == 'failed' and value['detectors'][0]['failed'] == state.failed
    from dalg.detection_view import status_text
    assert 'FAILED: front-red_disc' in status_text(r.run.detectors)
    r.run.recorder.close()
    failures = [e['data'] for e in events(r.run).events() if e['type'] == 'detector.failed']
    assert failures and 'lens cap on' in failures[0]['exception'] and failures[0]['traceback']


# -- the operator's view ---------------------------------------------------------------------

def test_the_detections_tab_and_live_overlay_render_from_a_run(rig):
    import tkinter as tk
    from tkinter import ttk
    from dalg.detection_view import (DetectionsTab, detection_shapes, frame_overlay, health_rows,
                                     selector_values)
    r = rig()
    r.tick(90)
    state = r.run.detectors.states['front-red_disc']
    image, reasons = frame_overlay(state, r.run.sim_time_s())
    assert image is not None and image.size == (160, 120) and isinstance(reasons, list)
    assert selector_values(r.run) == ['scan-lidar_inverse', 'front-red_disc']
    assert health_rows(r.run.detectors)[0][0] == 'front-red_disc'
    with hidden_tk() as root:
        notebook = ttk.Notebook(root)
        tab = DetectionsTab(notebook, r.run, tk, ttk)
        root.update_idletasks()
        tab.paint()
        tab.estimates.selection_set('landing_pad-1')
        tab._select()
        tab.paint()
        root.update_idletasks()
        tab.pane.refresh(force=True)
        assert tab.estimates.get_children() == ('landing_pad-1',)
        assert tab.health.get_children() == ('front-red_disc',)
        kinds = {tab.pane.canvas.type(item) for item in tab.pane.canvas.find_withtag('overlay')}
        assert {'oval', 'text', 'polygon'} <= kinds, 'estimate, label and camera footprint'
        assert tab.strip.find_all(), 'the sim-time strip draws its ticks'
        assert 'estimates=1/1' in tab.pane.header.get()
    shapes = detection_shapes(r.run.detectors, r.run.sim_time_s(), selected='landing_pad-1')
    assert any(s.kind == 'text' and s.text.startswith('landing_pad-1') for s in shapes)


def test_a_second_detector_runs_with_nothing_but_its_module_and_registry_line(tmp_path, monkeypatch):
    from dataclasses import replace
    from typing import ClassVar
    from dalg import detect as DET
    from dalg.profiles import Detector

    class Corner:
        """Sees a 'corner' in the top-left pixel of every frame, whatever the frame shows."""
        name: ClassVar[str] = 'corner'
        labels: ClassVar[tuple[str, ...]] = ('corner',)
        sensor_type: ClassVar[str] = 'camera.rgb'
        settings_class: ClassVar[type | None] = None

        def __init__(self, config, intrinsics): pass
        def detect(self, frame): return DET.DetectorResult((DET.PixelDetection('corner', 80., 119., 2., .9),))

    monkeypatch.setitem(DET.DETECTORS, 'corner', Corner)
    profile = load_profiles(['lidar-baseline', 'red-disc'], ROOT)
    profile = replace(profile, detectors=profile.detectors + (Detector('primary_camera', 'corner'),))
    r = Rig.__new__(Rig)
    r.instance = 'det-' + uuid.uuid4().hex[:8]
    r.provider = FixtureProvider(r.instance, tmp_path / 'session', sensors=('scan', 'front'), pads=(PAD,))
    r.run = DalgRun(r.instance, profile, ROOT)
    try:
        r.tick(60)
        value = DetectionsReader(r.instance).read()
        assert sorted(d['id'] for d in value['detectors']) == ['front-corner', 'front-red_disc']
        labels = {e['label'] for e in value['estimates']}
        assert labels == {'corner', 'landing_pad'}
        assert all('/' in e['id'] for e in value['estimates']), 'side-by-side detectors namespace their estimates'
    finally:
        r.close()


def test_detection_shapes_land_in_the_same_place_in_tk_and_in_the_report_image(rig):
    from dcmn import theme
    from dcmn.map_pane import MapPane, Overlay, project, snapshot_image
    from dalg.detection_view import detection_shapes
    r = rig()
    r.tick(90)
    now = r.run.sim_time_s()
    grids = r.run.evidence_grids()
    grid = grids[sorted(grids)[0]]
    shapes = tuple(detection_shapes(r.run.detectors, now))
    estimate = next(s for s in shapes if s.kind == 'oval' and s.fill == theme.TARGET_EDGE)
    cx, cy = (estimate.x0 + estimate.x1) / 2, (estimate.y0 + estimate.y1) / 2
    image = snapshot_image(grid, overlay=Overlay(shapes=shapes), sim_now_s=now, cell_px=10)
    column = int((cx - grid.geometry.origin_x_m) / grid.geometry.cell_m * 10)
    row = int((cy - grid.geometry.origin_y_m) / grid.geometry.cell_m * 10)
    assert image.getpixel((column, row))[:3] == theme.rgb(theme.TARGET_EDGE), 'PIL filled the confirmed estimate'
    with hidden_tk() as root:
        pane = MapPane(root, width=400, height=400)
        pane.set_grid(grid, sim_now_s=now)
        pane.set_overlay(shapes=shapes)
        root.update_idletasks()
        pane.refresh(force=True)
        ovals = [item for item in pane.canvas.find_withtag('overlay') if pane.canvas.type(item) == 'oval'
                 and pane.canvas.itemcget(item, 'fill') == theme.TARGET_EDGE]
        assert ovals, 'Tk drew the confirmed estimate'
        x0, y0, x1, y1 = pane.canvas.coords(ovals[0])
        expected = project(pane.view, grid.geometry, cx, cy)
        assert ((x0 + x1) / 2, (y0 + y1) / 2) == pytest.approx(expected, abs=1.5)


def test_the_dalg_window_shows_a_detector_in_live_and_detections_tabs(rig):
    pytest.importorskip('tkinter')
    from dalg.dalg import Window
    r = rig()
    r.tick(60)
    try:
        window = Window(r.run)
    except Exception as exc:  # noqa: BLE001 - no display
        pytest.skip(f'no display: {exc}')
    try:
        window.root.withdraw()
        window.update()
        assert 'front-red_disc' in window.source.cget('values')
        window.source.set('front-red_disc')
        window._paint_detector(r.run.detectors.states['front-red_disc'])
        window.root.update_idletasks()
        assert window.canvases[0].find_withtag('preview'), 'the processed frame is drawn'
        texts = [window.canvases[1].itemcget(i, 'text') for i in window.canvases[1].find_withtag('preview')]
        assert any('rejected candidates' in t for t in texts)
        window.notebook.select(window.detections.page)
        window.detections.paint()
        assert window.detections.estimates.get_children()
        assert 'detectors=1 estimates=1/1' in window.status.get(), window.status.get()
    finally:
        window.close()
        window.root.destroy()


def test_a_renamed_label_is_what_the_plane_and_the_estimates_publish(tmp_path):
    from dataclasses import replace
    profile = load_profiles(['lidar-baseline', 'red-disc'], ROOT)
    profile = replace(profile, detectors=(replace(profile.detectors[0], label='red_pad'),))
    r = Rig.__new__(Rig)
    r.instance = 'det-' + uuid.uuid4().hex[:8]
    r.provider = FixtureProvider(r.instance, tmp_path / 'session', sensors=('scan', 'front'), pads=(PAD,))
    r.run = DalgRun(r.instance, profile, ROOT)
    try:
        r.tick(60)
        value = DetectionsReader(r.instance).read()
        assert value['detectors'][0]['labels'] == ['red_pad']
        assert {d['label'] for d in value['detections']} == {'red_pad'}
        assert [e['id'] for e in value['estimates']] == ['red_pad-1'] and value['estimates'][0]['label'] == 'red_pad'
    finally:
        r.close()


def test_the_profile_editor_edits_and_saves_a_detector_label(tmp_path):
    import tkinter as tk
    from tkinter import ttk
    from dalg.source_editor import SourceEditor
    with hidden_tk() as root:
        page = ttk.Frame(root)
        editor = SourceEditor(page, load_profile_named('red-disc'), tk, ttk, root=ROOT)
        editor.rows.selection_set('0')
        editor._select()
        assert editor.label_var.get() == 'landing_pad'
        editor.label_var.set('red_pad')
        assert editor.dirty and editor.validate()
        editor.save(tmp_path / 'renamed.json')
    saved = json.loads((tmp_path / 'renamed.json').read_text())
    assert saved['detectors'][0]['label'] == 'red_pad' and saved['sources'] == []


def load_profile_named(name):
    from dalg.profiles import load_profile
    return load_profile(name, ROOT)
