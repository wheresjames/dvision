"""Reports and replay built from recorded dynamic flights, offline.

The report and replay read archives only: no live module, no vehicle
endpoint, no memkv area. These tests build real recorded flights with the
dynamic rig (a completed one shared across the module, plus a genuinely
failed one for the comparison table), then check what the derived artefacts
say -- and what they honestly refuse to say when a value is unavailable or
a recording is incomplete.

The gap and epoch behaviour of replay is pinned with a small synthetic
archive whose sample times, epochs and holes are known exactly, so "last
recorded value, never interpolated" is checked against arithmetic, not
against a flight that happens to have flown somewhere.
"""

import csv
import json
import os
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from dcmn.archive import Recorder
from dtest.dynamic_rig import DynamicRig
from dway.flightlog import REPORT_SCHEMA, build_report, compare_runs
from dway.replay import ReplayModel

TERMINAL = ('COMPLETE', 'CANCELLED', 'FAILED')


@pytest.fixture(scope='module')
def flight(tmp_path_factory):
    """One complete recorded flight with a sibling planner archive under one root.

    The rig is closed inside setup (the report is derived at close), so no
    transport, writer lock or patched clock survives into the tests; only
    the archive and its derived report remain on disk.
    """
    tmp = tmp_path_factory.mktemp('flight')
    rig = DynamicRig(tmp, nav_archive=True)
    try:
        rig.start()
        assert rig.fly_to_state(*TERMINAL), rig.executor.reason
        assert rig.executor.state == 'COMPLETE', rig.executor.reason
        assert not rig.executor.report_error, rig.executor.report_error
    finally:
        rig.close()
    summary = json.loads((Path(rig.executor.report_dir) / 'report' / 'summary.json').read_text())
    yield SimpleNamespace(root=tmp, run_root=tmp / 'run', dway=Path(rig.executor.report_dir),
                          summary=summary)


def test_report_artifacts_explain_a_completed_flight(flight) -> None:
    report = flight.dway / 'report'
    for name in ('summary.json', 'events.jsonl', 'samples.csv', 'routes.csv', 'holds.csv',
                 'report.html', 'manifest.json', 'map.png', 'timeline.png'):
        assert (report / name).is_file(), f'{name} missing from the report'
    summary = flight.summary
    assert summary['schema'] == REPORT_SCHEMA
    assert summary['outcome'] == 'complete' and summary['final_state'] == 'COMPLETE'
    assert summary['recording_complete'] is True
    # 1. outcome, goal, pose, times and interventions
    assert summary['goal'] and summary['goal']['position'] == [12.5, 6.0, 1.5]
    assert summary['final_pose'] and summary['elapsed_s'] > 0
    assert summary['executing_s'] > 0 and summary['state_durations_s']['EXECUTING'] > 0
    assert isinstance(summary['interventions'], list)
    assert any(i['action'] == 'start' and i['accepted'] for i in summary['interventions'])
    # 2. proposed versus active routes and the flown track, attributed to the
    #    route active at the time (not the last one in the file)
    assert summary['routes'], 'no per-route attribution rows'
    route = summary['routes'][0]
    assert route['points'] and route['planner_session'] and route['geometry_revision'] is not None
    assert route['targets'] > 0 and route['samples'] > 0 and route['estimated_track_m'] > 0
    assert route['dispositions'] and route['permitted_ends']
    assert summary['consumed_routes'] and summary['proposed_routes'], 'dnav proposals missing'
    assert summary['estimated_track_m'] > 0
    # 4. speed, tracking, margins, cadence -- with units and definitions (5)
    assert summary['speed']['max_mps'] > 0 and summary['speed']['cap_mps'] is not None
    assert summary['tracking']['max_m'] is not None and summary['tracking']['allowance_m'] is not None
    assert summary['tracking']['mean_time_weighted_m'] is not None
    assert summary['stopping']['min_margin_m'] is not None
    assert summary['cadence']['targets_sent'] > 0
    assert summary['cadence']['targets_accepted'] == summary['cadence']['targets_sent']
    assert summary['cadence']['skipped_slots'] == 0 and summary['cadence']['deadline_misses'] == 0
    assert summary['route_age_s']['max'] is not None
    assert summary['units'] and summary['definitions'] and summary['profile']
    assert summary['conditions'], 'simulator conditions not recorded'
    # the derived files agree with the summary they came from
    with (report / 'samples.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == summary['samples']
    lines = (report / 'events.jsonl').read_text().splitlines()
    assert len(lines) == summary['recording']['events']
    html = (report / 'report.html').read_text()
    assert 'complete' in html and 'Definitions' in html
    manifest = json.loads((report / 'manifest.json').read_text())
    assert manifest['inputs'] and not manifest['missing'], manifest['missing']


def test_values_the_report_cannot_know_are_null_with_a_reason(tmp_path) -> None:
    """Without a sibling planner archive the proposals are null, and the report says why."""
    rig = DynamicRig(tmp_path)
    try:
        rig.start()
        assert rig.fly_to_state(*TERMINAL), rig.executor.reason
        assert rig.executor.state == 'COMPLETE', rig.executor.reason
        rig.close()
        summary = json.loads((Path(rig.executor.report_dir) / 'report' / 'summary.json').read_text())
    finally:
        rig.close()
    assert summary['outcome'] == 'complete'
    assert summary['proposed_routes'] is None
    assert summary['proposed_routes_unavailable'] == 'no sibling dnav archive under this report root'
    manifest = json.loads((Path(rig.executor.report_dir) / 'report' / 'manifest.json').read_text())
    assert any(m['item'] == 'proposed routes' and m['reason'] == summary['proposed_routes_unavailable']
               for m in manifest['missing'])


def test_replay_reads_what_the_executor_consumed(flight) -> None:
    model = ReplayModel(flight.dway)
    assert model.samples and model.times and model.transitions and model.routes
    # the last recorded value at or before the selected time, never interpolation
    sample, age = model.sample_at(model.times[0])
    assert sample is not None and age == 0.0
    sample, age = model.sample_at(model.times[0] - 1.0)
    assert sample is None and age is None
    for a, b in zip(model.times, model.times[1:]):
        sample, age = model.sample_at((a + b) / 2)
        assert sample['t_s'] == a and age == pytest.approx((a + b) / 2 - a, abs=1e-9)
        break
    executing = next(data for _, data in reversed(model.samples) if data['state'] == 'EXECUTING')
    frame = model.frame(executing['t_s'])
    assert frame['state'] == 'EXECUTING' and frame['sample_age_s'] == 0.0
    assert frame['route_points'] and frame['permitted'] and frame['target'] and frame['pose']
    assert frame['gap'] is None and frame['track']
    assert frame['route_key'] in model.revisions()
    assert frame['route_key'].endswith(f":{frame['route_revision']}")
    final = model.frame(model.end)
    assert final['state'] == 'COMPLETE'
    # event stepping walks the decision log in order
    t, kinds = model.start, []
    while True:
        found = model.next_event(t)
        if found is None: break
        assert found[0] > t
        t, _ = found
        kinds.append(t)
    assert kinds and model.previous_event(model.start) is None
    assert model.previous_event(t) is not None
    # the evidence the executor saw at the time, reconstructable offline
    grid, note = model.grid_at(executing['t_s'])
    assert grid is not None and note is None
    assert model.gaps == [], model.gaps


def test_replay_marks_gaps_and_breaks_the_track_at_epoch_changes(tmp_path) -> None:
    """Sample times, epochs and holes are known exactly here, so the lookup rules
    are checked against arithmetic: last value at or before the time, gaps
    reported as gaps, the track broken at epoch changes and never interpolated."""
    recorder = Recorder(tmp_path / 'archive', dict(profile=dict(stream_hz=10.0)), module='dway')
    for i in range(11):  # 0.0 .. 1.0 s, epoch A; the 1.0 s sample already says HOLDING
        recorder.record('execution.sample', dict(t_s=0.1 * i, state='EXECUTING' if i < 10 else 'HOLDING',
                                                 pose=[1.0 + 0.1 * i, 2.0, 1.5], epoch=['p', 0, 0]))
    for i in range(5):  # 1.5 .. 1.9 s, epoch B: an epoch change at the seam
        recorder.record('execution.sample', dict(t_s=1.5 + 0.1 * i, state='EXECUTING',
                                                 pose=[3.0 + 0.1 * i, 2.0, 1.5], epoch=['p', 1, 0]))
    recorder.record('execution.sample', dict(t_s=6.0, state='HOLDING',
                                             pose=[10.0, 2.0, 1.5], epoch=['p', 1, 0]))  # a 4.1 s hole
    recorder.record('execution.transition', dict(t_s=0.05, state='READY', reason='route ready', hold_kind=None))
    recorder.record('execution.transition', dict(t_s=1.0, state='HOLDING', reason='pause', hold_kind='pause'))
    recorder.close()
    model = ReplayModel(tmp_path / 'archive')
    reasons = {g['reason'] for g in model.gaps}
    assert reasons == {'epoch change', 'missing samples'}
    assert model.in_gap(1.2)['reason'] == 'epoch change'
    assert model.in_gap(3.0)['reason'] == 'missing samples'
    assert model.in_gap(0.5) is None
    # no interpolation into a gap: the 1.0 s sample is what 1.2 s shows
    sample, age = model.sample_at(1.2)
    assert sample['t_s'] == 1.0 and sample['pose'][0] == 2.0 and age == pytest.approx(0.2)
    # the track never joins across the epoch change or the hole
    lines = model.track_until(10.0)
    assert len(lines) == 2
    assert [pt[0] for pt in lines[0]] == pytest.approx([1.0 + 0.1 * i for i in range(11)])
    assert [pt[0] for pt in lines[1]] == pytest.approx([3.0 + 0.1 * i for i in range(5)])
    frame = model.frame(1.2)
    assert frame['state'] == 'HOLDING' and frame['hold_kind'] == 'pause' and frame['reason'] == 'pause'
    assert model.state_at(0.5)['state'] == 'READY'
    # a route that was never consumed is absent, not invented
    assert model.route_at(1.2) is None and frame['route_points'] == []


def test_replay_export_writes_the_selected_instant(flight, tmp_path) -> None:
    model = ReplayModel(flight.dway)
    out = model.export(model.times[len(model.times) // 2], tmp_path / 'export')
    frame_files = list(out.glob('frame-*.json'))
    assert len(frame_files) == 1
    exported = json.loads(frame_files[0].read_text())
    assert exported['sample'] and exported['sample_age_s'] is not None
    assert exported['route_points'] and exported['track']
    assert exported['evidence_note'] is None
    assert list(out.glob('frame-*.png')), 'the exported frame must include the evidence map'


def test_a_corrupted_recording_says_complete_flight_and_incomplete_recording(flight, tmp_path) -> None:
    """Outcome and recording completeness are independent: a flight that
    finished is reported as finished even when its archive cannot be trusted."""
    corrupt = tmp_path / 'corrupt'
    shutil.copytree(flight.dway, corrupt, ignore=shutil.ignore_patterns('report'))
    index = corrupt / 'index.jsonl'
    lines = index.read_text().splitlines(keepends=True)
    index.write_text(''.join(lines[:-1]) + lines[-1][:len(lines[-1]) // 2])  # torn final line
    out = build_report(corrupt, tmp_path / 'report')
    summary = json.loads((out / 'summary.json').read_text())
    assert summary['outcome'] == 'complete' and summary['final_state'] == 'COMPLETE'
    assert summary['recording_complete'] is False
    assert summary['recording']['errors'], 'a torn archive must name its errors'
    assert summary['outcome_note'], 'an incomplete recording must say both'
    assert 'INCOMPLETE' in (out / 'report.html').read_text()


def test_a_copied_run_bundle_opens_elsewhere(flight, tmp_path) -> None:
    """The whole run root -- planner archive, execution archive, derived report
    -- copied elsewhere is self-contained: proposals found, evidence rebuilt,
    replay and export working from the copy alone."""
    bundle = tmp_path / 'bundle'
    shutil.copytree(flight.run_root, bundle)
    dway = next((bundle / 'dway').glob('archive*'))
    out = build_report(dway)
    summary = json.loads((out / 'summary.json').read_text())
    assert summary['proposed_routes'], 'the copied planner archive was not found'
    assert summary['archive'] == str(dway)
    manifest = json.loads((out / 'manifest.json').read_text())
    assert {i['kind'] for i in manifest['inputs']} == {'dway archive', 'dnav archive'}
    assert all(str(bundle) in i['path'] for i in manifest['inputs'])
    model = ReplayModel(dway)
    grid, note = model.grid_at(model.times[len(model.times) // 2])
    assert grid is not None and note is None
    exported = model.export(model.end, tmp_path / 'elsewhere')
    assert list(exported.glob('frame-*.json'))


def test_the_comparison_table_keeps_failures_and_interventions(tmp_path) -> None:
    """Every outcome is kept in the table, failures included, with the counts
    that explain them -- and no efficiency score across different evidence."""
    failed_rig = DynamicRig(tmp_path / 'failed')
    try:
        failed_rig.start()
        assert failed_rig.fly(limit_s=30.0, until=lambda r: r.executor.state == 'EXECUTING')
        failed_rig.reject_holds()
        failed_rig.block(failed_rig.executor.pose[0] + 3.0, 6.0, radius_m=0.8)
        assert failed_rig.fly_to_state('FAILED', limit_s=40.0)
        failed_rig.close()
        failed = json.loads((Path(failed_rig.executor.report_dir) / 'report' / 'summary.json').read_text())
    finally:
        failed_rig.close()
    assert failed['outcome'] == 'failed'
    summary_path = Path(failed_rig.executor.report_dir) / 'report' / 'summary.json'
    rows = compare_runs([failed, summary_path], tmp_path / 'compare')
    assert len(rows) == 2 and {r['outcome'] for r in rows} == {'failed'}
    assert rows[0]['reason'] and rows[0]['interventions'] > 0 and rows[0]['stops'] >= 0
    table = json.loads((tmp_path / 'compare' / 'comparison.json').read_text())
    assert table['runs'] == 2 and table['outcomes'] == {'failed': 2}
    assert 'all runs included' in table['note']
    with (tmp_path / 'compare' / 'comparison.csv').open() as handle:
        header = next(csv.reader(handle))
    assert 'outcome' in header and 'recording_complete' in header and 'profile' in header


@pytest.mark.nightly
@pytest.mark.skipif(os.environ.get('DVISION_NIGHTLY') != '1',
                    reason='set DVISION_NIGHTLY=1 to fly the thirty-minute recording')
def test_a_thirty_minute_recording_stays_bounded_seeks_and_reports(tmp_path) -> None:
    """A representative long recording: repeated goals with supervision holds
    between them, thirty minutes of data clock. The executor's live-memory
    bounds must hold at the end, the replay must still seek usefully, and the
    report must be generated from the full archive."""
    rig = DynamicRig(tmp_path, nav_archive=True)
    try:
        started = time.perf_counter()
        rig.start()
        assert rig.fly_to_state(*TERMINAL), rig.executor.reason
        for x, y in ((6.0, 6.0), (12.5, 6.0), (5.0, 9.0), (12.5, 6.0)):
            for _ in range(int(450.0 / 0.05)):  # supervised hovering between missions
                rig.step()
            rig.set_goal(x, y)
            rig.executor.request('start', 'nightly')
            assert rig.fly_to_state(*TERMINAL, limit_s=120.0), rig.executor.reason
            assert rig.executor.state == 'COMPLETE', rig.executor.reason
        span = rig.sim.sim_time_s
        rig.close()
        wall = time.perf_counter() - started
        ex_rate, report = span / wall, Path(rig.executor.report_dir) / 'report'
        summary = json.loads((report / 'summary.json').read_text())
        assert span >= 1800.0, span
        assert summary['outcome'] == 'complete' and summary['recording_complete'] is True
        assert summary['samples'] >= 10 * span * 0.9, 'the recording must cover the half hour'
        assert summary['elapsed_s'] == pytest.approx(span, abs=1.0)
        import resource
        peak_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        print(f'nightly recording: {span:.0f} s of data clock in {wall:.0f} s wall ({ex_rate:.0f}x), '
              f'{summary["samples"]} samples, {summary["recording"]["events"]} events, '
              f'peak RSS {peak_mib:.0f} MiB')
        # bounded live memory at the end of a long session
        assert len(rig.executor.history) <= 61 * 10
        assert len(rig.executor.track) <= 6000
        assert len(rig.executor.events) <= 300
        assert len(rig.executor.sent) <= 50000
        # a useful seek: any mid-flight instant still reconstructs
        model = ReplayModel(rig.executor.report_dir)
        mid = model.times[len(model.times) // 2]
        frame = model.frame(mid)
        assert frame['sample'] is not None and frame['sample_age_s'] <= 0.3
        assert frame['gap'] is None and frame['track']
        exported = model.export(mid, tmp_path / 'export')
        assert list(exported.glob('frame-*.json'))
    finally:
        rig.close()
