"""Camera evidence through the real transport, a real flight, and the shared renderer.

The flight test is the operational stack end to end: dsim provides the session,
the ideal pose and the camera; dway flies a tour externally; dalg publishes from
a baseline profile with no tour or world file; dnav plans on it. Scoring the
result against the world is then done here, by the evaluator, from dalg's own
archive -- never inside dalg.
"""
import json
import time
import uuid
from pathlib import Path

import numpy as np
import pytest

from dalg.evidence import CameraEvidence
from dalg.model import Frame, Intrinsics, Pose
from dalg.profiles import Source
from dcmn.maps import GridGeometry, MapPublisher, MapSession

ROOT = Path(__file__).resolve().parents[1]


def camera_evidence(geometry, algorithm='ground_plane'):
    instance = 'cam-'+uuid.uuid4().hex[:8]
    publisher = MapPublisher(instance, geometry, [dict(id=Source('front', algorithm).id, sensor='front',
                                                       sensor_type='camera.rgb', algorithm=algorithm)])
    return CameraEvidence(geometry, Intrinsics(64, 48, 50., 50., 32., 24.), 'front', {},
                          algorithm=algorithm, publisher=publisher)


def test_camera_ray_timestamps_negative_geometry_and_cadence():
    evidence = camera_evidence(GridGeometry.from_extent(12., 12., .5, origin_x_m=-6., origin_y_m=-6.))
    session = MapSession(evidence.publisher.instance)
    try:
        rgb = np.zeros((48, 64, 3), np.uint8)
        rgb[38:] = 255
        pose = Pose(0., 2., 1., 0.)
        evidence.observe(Frame(1, 2., rgb, pose, camera=pose))
        evidence.publish(2.)
        session.poll()
        grid = session.latest(evidence.source)
        assert grid is not None and grid.observed.any()
        assert grid.geometry.origin_x_m == -6.
        assert grid.geometry.cell_m == .5
        assert set(grid.observed_ms[grid.observed]) == {2000}
        assert (grid.occupancy[grid.never_observed] == 255).all()
        evidence.publish(2.5)
        assert evidence.published == 1
        evidence.publish(3.)
        assert evidence.latest.revision == grid.revision
        assert np.array_equal(evidence.latest.observed_ms, grid.observed_ms)
        # A repeated ray dates saturated cells too; untouched cells retain age.
        for seq in range(2, 12):
            evidence.observe(Frame(seq, 4., rgb, pose, camera=pose))
        evidence.publish(4.)
        assert evidence.latest.revision > grid.revision
        assert set(evidence.latest.observed_ms[grid.observed]) == {4000}
        from dcmn.map_pane import snapshot_image
        assert snapshot_image(evidence.latest, cell_px=8).size == (192, 192)
    finally:
        session.close()
        evidence.publisher.close()


def _nightly(name):
    """The dense algorithms take minutes on a real flight; they opt in."""
    import os
    return pytest.param(name, marks=[
        pytest.mark.nightly,
        pytest.mark.skipif(os.environ.get('DVISION_NIGHTLY') != '1',
                           reason='set DVISION_NIGHTLY=1 to fly the dense algorithms')])


@pytest.mark.parametrize('algorithm', [
    'ground_plane', 'optical_flow_triangulation', 'feature_triangulation',
    _nightly('monocular_depth'), _nightly('plane_sweep'), _nightly('sgbm')])
def test_tour_flown_camera_evidence_reaches_dnav_and_is_scored_offline(tmp_path, algorithm):
    """A real flight with motion supplied externally by a tour; evaluation afterwards."""
    from dalg.run import DalgRun
    from dalg.profiles import Profile
    from dcmn.archive import ArchiveReader
    from dnav.plan import NavRun
    from dnav.policy import load_policy
    from dtest.dway_rig import write_corridor_tour
    from dtest.evaluation import score_evidence
    from dtest.process_harness import DsimProcessHarness
    from dvision2_common import load_map
    from tests.test_dalg_evidence_algorithms import settings_for
    from tests.test_dway_process import start_dway

    settings = settings_for(algorithm)
    if algorithm == 'sgbm':
        # Stereo from motion pairs only sideways motion: crab facing the wall,
        # with a baseline the matcher's disparity range can resolve.
        tour = write_corridor_tour(tmp_path/'tour.json', heading_deg=180.0)
        settings = dict(settings, max_baseline_m=.5)
    else:
        tour = write_corridor_tour(tmp_path/'tour.json')
    profile = Profile(f'{algorithm}-flight', (Source('primary_camera', algorithm, settings),), 'digest')
    with DsimProcessHarness(tmp_path, map_path=ROOT/'assets/maps/maze_012.txt') as harness:
        run = DalgRun(harness.id, profile, ROOT)
        nav = NavRun(harness.id, ROOT, policy=load_policy('default', ROOT), goal=(14.5, 1.5, None))
        harness.send('release_control')
        harness._lease_acquired = False
        harness.wait_status(lambda s: s.get('control.owner') == '', description='released control')
        process, log = start_dway(harness, tour, tmp_path/'dway.log',
            '--wait-for', f'algorithm:{algorithm}', '--timeout', '90')
        seen = False
        try:
            deadline = time.monotonic()+100
            while process.poll() is None and time.monotonic() < deadline:
                run.step()
                nav.step()
                if nav.sources():
                    seen |= any(nav.session.latest(s) is not None for s in nav.sources())
                time.sleep(.01)
            assert process.returncode == 0, (tmp_path/'dway.log').read_text()
            # The mission finished; perception did not.
            assert run.state == 'RUNNING' and not run.done, run.reason
            assert run.snapshot['provider_kind'] == 'ideal-simulation'
            assert seen and nav.plans > 0
            assert any(e['status'] == 'ok' for e in nav.history.entries)
            source = f'front-{algorithm}'
            final = run.evidence_grids()[source]
            assert final.observed.any()
            assert harness.read_status()['control.owner'] != f'dnav-{harness.id}'
        finally:
            nav.close(partial=process.poll() is None)
            run.close()
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=5)
            log.close()
    summary = json.loads((run.report_dir/'summary.json').read_text())
    assert 'scores' not in summary and summary['pose_provider'] == 'ideal-simulation'
    assert (run.report_dir/summary['evidence'][source]['image']).is_file()
    # The evaluator's half: the archive, the delivered poses, and the world.
    reader = ArchiveReader(run.report_dir/'archive')
    assert reader.validate()['complete']
    published = [e for e in reader.events() if e['type'] == 'evidence.published']
    grid = reader.reconstruct(published[-1])['reconstructed_grids'][source]
    poses = [Pose(**{k: e['data']['pose'][k] for k in ('x_m', 'y_m', 'z_m', 'heading_deg')})
             for e in reader.events() if e['type'] == 'sample.admitted']
    assert poses
    scores = score_evidence(grid, load_map(ROOT/'assets/maps/maze_012.txt'), poses)
    assert scores['coverage'] is not None and scores['coverage'] > 0


def test_grids_window_and_report_share_renderer(tmp_path, monkeypatch):
    import tkinter as tk
    from PIL import Image
    import dalg.dalg as ui
    from dalg.run import DalgRun
    from dalg.profiles import load_profile
    from dcmn.map_pane import snapshot_image
    from dtest.provider import FixtureProvider
    from dtest.tkfixture import hidden_root

    root = hidden_root()
    monkeypatch.setattr(tk, 'Tk', lambda: root)
    monkeypatch.setattr(ui, 'restore_window_geometry', lambda *a: None)
    monkeypatch.setattr(ui, 'save_window_geometry', lambda *a: None)
    instance = 'ui-'+uuid.uuid4().hex[:8]
    provider = FixtureProvider(instance, tmp_path, sensors=('front',))
    run = DalgRun(instance, load_profile('ground-plane-baseline', ROOT), ROOT)
    window = ui.Window(run)
    try:
        for _ in range(60): provider.step(); run.step()
        window.notebook.select(window.grids)
        window.update()
        pane = window.map_pane
        latest = run.evidence_grids()['front-ground_plane']
        assert pane.state.grid is latest
        for mode in ('occupancy', 'age', 'never'):
            pane.set_mode(mode)
            window.update()
            assert np.array_equal(np.asarray(pane.snapshot()), np.asarray(
                snapshot_image(latest, mode=mode, sim_now_s=run.sim_time_s())))
        pane.set_mode('occupancy')
        pane.snapshot().save(tmp_path/'camera-evidence.png')
        assert Image.open(tmp_path/'camera-evidence.png').size[0] > 0
        assert [window.notebook.tab(tab, 'text') for tab in window.notebook.tabs()] == [
            'Live', 'Grids', 'Profile', 'Events']
        # Two live panes: the sensor and the algorithm's belief. No truth.
        assert len(window.canvases) == 2
    finally:
        window.close()
        window.root.destroy()
        run.close()
        provider.close()


def test_shutdown_reports_evidence_and_a_finalized_archive(tmp_path):
    from dalg.run import DalgRun
    from dalg.profiles import load_profile
    from dcmn.archive import ArchiveReader
    from dtest.provider import FixtureProvider

    instance = 'abort-'+uuid.uuid4().hex[:8]
    provider = FixtureProvider(instance, tmp_path, sensors=('front',))
    run = DalgRun(instance, load_profile('ground-plane-baseline', ROOT), ROOT)
    try:
        for _ in range(45): provider.step(); run.step()
        run.finish(partial=True)
        run.close()
    finally:
        provider.close()
    summary = json.loads((tmp_path/'dalg/summary.json').read_text())
    assert summary['state'] == 'RUNNING' and summary['provenance']['stopped'] == 'timeout'
    assert summary['evidence']['front-ground_plane']['time_s'] > 0
    assert (tmp_path/'dalg/evidence-front-ground_plane.png').is_file()
    assert summary['recording']['complete'] is True
    assert ArchiveReader(tmp_path/'dalg'/summary['archive']).validate()['complete']
    assert run.sources is None or run.sources.publisher.closed
