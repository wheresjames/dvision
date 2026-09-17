"""dmsn as a process: truth isolation, one coordinator per vehicle, the arena check, --validate.

The fixture provider stands in for the simulator (no world file, no renderer),
so these run everywhere. It has no vehicle link, so a mission here gets as far
as asking for a launch nobody answers -- which is exactly what these tests need
to check the process surface. Flying a mission end to end is the flight rig's
job (test_dmsn_flight) and the nightly chain's (test_dmsn_nightly).
"""
from __future__ import annotations

import json
import math
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest

from dcmn.archive import ArchiveReader
from dtest.isolation import isolated_env, violations

ROOT = Path(__file__).resolve().parents[2]
DMSN = ROOT / 'apps/dmsn/dmsn.py'
PROVIDER = ROOT / 'dtest/provider.py'


@pytest.fixture
def launcher(tmp_path):
    processes = []

    def start(args, env=None, name=None):
        log = tmp_path / f'{name or len(processes)}.log'
        with log.open('w') as out:
            process = subprocess.Popen([sys.executable, *map(str, args)], cwd=ROOT, env=env, stdout=out,
                                       stderr=subprocess.STDOUT)
        process.log = log
        processes.append(process)
        return process

    yield start
    for process in processes:
        if process.poll() is None:
            process.terminate()
            try: process.wait(timeout=5)
            except subprocess.TimeoutExpired: process.kill(); process.wait(timeout=5)


def provider(start, instance, report, *extra):
    process = start([PROVIDER, '--id', instance, '--report-dir', report, '--sensors', '', '--timeout', '40', *extra],
                    name='provider')
    time.sleep(1.0)
    return process


def test_headless_goto_under_isolation_asks_for_launch_reports_and_refuses_a_second_dmsn(tmp_path, launcher,
                                                                                          monkeypatch):
    monkeypatch.setenv('DVISION2_MISSION_DIR', str(tmp_path / 'user'))
    instance = 'msn-' + uuid.uuid4().hex[:8]
    report = tmp_path / 'run'
    env = isolated_env(tmp_path / 'isolation')
    env['DVISION2_MISSION_DIR'] = str(tmp_path / 'user')
    mission = launcher([DMSN, '--id', instance, '--no-ui', '--mission', 'goto', '--goal', '4,2', '--start',
                        '--timeout', '8'], env, name='dmsn')
    time.sleep(0.5)  # a late provider: dmsn waits for it
    provider(launcher, instance, report)
    deadline = time.monotonic() + 15
    while not (tmp_path / 'dmsn.log').read_text().count('starting') and time.monotonic() < deadline:
        time.sleep(0.2)
    second = launcher([DMSN, '--id', instance, '--no-ui', '--mission', 'goto', '--timeout', '3'], env, name='second')
    assert second.wait(timeout=30) == 2, second.log.read_text()
    assert 'another dmsn is already coordinating' in second.log.read_text()
    code = mission.wait(timeout=40)
    log = mission.log.read_text()
    assert code == 1, log  # no dway answered: the mission cannot finish before the timeout
    assert violations(tmp_path / 'isolation') == [], log
    assert 'no dway execution status' in log
    summary = json.loads((report / 'dmsn/summary.json').read_text())
    assert summary['phase'] == 'starting' and summary['partial'] and summary['reason'] == 'dmsn timeout'
    assert summary['profile']['params'] == dict(goal_source='xy', goal=[4.0, 2.0])
    assert summary['overrides'] == ['goal_source=xy', 'goal=4,2']
    reader = ArchiveReader(report / 'dmsn/archive')
    assert reader.validate()['complete']
    types = [e['type'] for e in reader.events()]
    assert 'mission.start' in types and 'mission.unfinished' in types
    for name in ('events.jsonl', 'mission.png', 'report.html'):
        assert (report / 'dmsn' / name).exists(), name


def test_a_headless_start_on_the_wrong_arena_exits_2_unless_allowed(tmp_path, launcher, monkeypatch):
    instance = 'arena-' + uuid.uuid4().hex[:8]
    report = tmp_path / 'run'
    provider(launcher, instance, report, '--map-name', 'maze_012')
    refused = launcher([DMSN, '--id', instance, '--no-ui', '--profile', 'maze_020-goto', '--start', '--timeout', '10'],
                       name='refused')
    assert refused.wait(timeout=30) == 2, refused.log.read_text()
    text = refused.log.read_text()
    assert "'maze_020' differs from the running map 'maze_012'" in text and '--allow-arena-mismatch' in text
    allowed = launcher([DMSN, '--id', instance, '--no-ui', '--profile', 'maze_020-goto', '--start',
                        '--allow-arena-mismatch', '--timeout', '3'], name='allowed')
    assert allowed.wait(timeout=30) == 1, allowed.log.read_text()
    summaries = sorted((report / 'dmsn').glob('summary.json'))
    summary = json.loads(summaries[-1].read_text())
    assert summary['arena'] == dict(profile='maze_020', provider='maze_012', mismatch=True, allowed=True)


def test_validate_reports_the_digest_or_the_problem(tmp_path):
    good = subprocess.run([sys.executable, DMSN, '--validate', 'maze_020-goto'], cwd=ROOT, capture_output=True,
                          text=True, timeout=30)
    assert good.returncode == 0 and 'digest' in good.stdout, good.stderr
    broken = tmp_path / 'broken.json'
    broken.write_text(json.dumps(dict(mission='goto', params=dict(goal_source='teleport'))))
    bad = subprocess.run([sys.executable, DMSN, '--validate', str(broken)], cwd=ROOT, capture_output=True,
                         text=True, timeout=30)
    assert bad.returncode == 2 and 'goal_source must be one of' in bad.stderr


CHILD_IMPORT = """
import sys
from pathlib import Path
sys.path[:0] = [sys.argv[1], sys.argv[1] + '/apps']
from dmsn import profiles as P
from dmsn.tour_import import run_converter
tour, out = sys.argv[2], sys.argv[3]
try:
    open(tour).close()
except PermissionError:
    print('window-open-refused')
converted, message = run_converter(tour, out)
print('converted' if converted else 'not-converted', message)
profile, _ = P.load(out)
print('loaded', profile['mission'], len(profile['waypoints']), profile['assisted']['tour_import'])
"""


def test_the_setup_tab_tour_import_succeeds_under_isolation_only_through_the_child_converter(tmp_path):
    """The window is isolated and cannot open a tour; the converter child it starts can (B9)."""
    tour = ROOT / 'assets/tours/maze_020.default.v1.json'
    if not tour.exists(): pytest.skip('no committed maze_020 tour')
    env = isolated_env(tmp_path / 'isolation')
    env['DVISION2_MISSION_DIR'] = str(tmp_path / 'missions')
    out = tmp_path / 'missions' / 'maze_020-tour.json'
    script = tmp_path / 'window_import.py'
    script.write_text(CHILD_IMPORT, encoding='utf-8')
    result = subprocess.run([sys.executable, str(script), str(ROOT), str(tour), str(out)], cwd=ROOT, env=env,
                            capture_output=True, text=True, timeout=120)
    log = result.stdout + result.stderr
    assert result.returncode == 0, log
    assert 'window-open-refused' in result.stdout, f'the isolation hook was not live in the window: {log}'
    assert 'converted' in result.stdout.split() and 'not-converted' not in result.stdout, log
    assert 'loaded waypoints' in result.stdout and 'True' in result.stdout, log
    assert violations(tmp_path / 'isolation') == [f'open {tour}'], 'only the deliberate probe was refused'
    assert json.loads(out.read_text())['imported_from']['kind'] == 'tour'


SEARCH_AREA = '-10,-10;3.5,-10;3.5,10;-10,10'


def search_chain(tmp_path, launcher, profiles):
    """Provider with a pad, an isolated dalg with ``profiles``, and a search profile in the user directory."""
    from dcmn.detections import DetectionsReader
    from dmsn import profiles as P
    instance = 'srch-' + uuid.uuid4().hex[:8]
    report = tmp_path / 'run'
    env = isolated_env(tmp_path / 'isolation')
    env['DVISION2_MISSION_DIR'] = str(tmp_path / 'user')
    launcher([PROVIDER, '--id', instance, '--report-dir', report, '--sensors', 'scan,front', '--pads', '3,0',
              '--timeout', '60'], name='provider')
    dalg = launcher([ROOT / 'apps/dalg/dalg.py', '--id', instance, '--profiles', *profiles, '--no-ui',
                     '--timeout', '40'], env, name='dalg')
    profile = P.apply_overrides(P.blank('search', name='fixture-search'), [f'area={SEARCH_AREA}'])
    P.save(profile, tmp_path / 'user' / 'fixture-search.json')
    reader = DetectionsReader(instance)
    return instance, report, env, dalg, reader


def test_search_starts_only_beside_a_detector_and_dalg_reports_what_it_saw(tmp_path, launcher):
    instance, report, env, dalg, reader = search_chain(tmp_path, launcher, ('lidar-baseline', 'red-disc'))
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and not (reader.read() or {}).get('estimates'):
        time.sleep(0.2)
    value = reader.read()
    assert value and value['estimates'], (tmp_path / 'dalg.log').read_text()
    search = launcher([DMSN, '--id', instance, '--no-ui', '--profile', 'fixture-search', '--start',
                       '--timeout', '6'], env, name='search')
    code = search.wait(timeout=40)
    log = search.log.read_text()
    assert code == 1, log  # started; no dway answered, so it cannot finish before the timeout
    assert violations(tmp_path / 'isolation') == [], log
    summary = json.loads((report / 'dmsn/summary.json').read_text())
    assert summary['phase'] == 'starting' and summary['profile']['mission'] == 'search'
    checks = [e['data'] for e in ArchiveReader(report / 'dmsn/archive').events() if e['type'] == 'mission.target_check']
    assert checks and checks[0]['found'] == ['front-red_disc']
    dalg.send_signal(signal.SIGINT)  # a clean stop writes the report
    assert dalg.wait(timeout=30) == 0, (tmp_path / 'dalg.log').read_text()
    dalg_summary = json.loads((report / 'dalg/summary.json').read_text())
    section = dalg_summary['detectors']
    assert dalg_summary['schema_version'] == 4 and section['detectors'][0]['processed'] > 0
    assert section['estimates'] and math.dist(section['estimates'][0]['xy_m'], (3., 0.)) < 2.0
    assert (report / 'dalg/detections.png').is_file() and (report / 'dalg/detections').is_dir()
    reader_dalg = ArchiveReader(report / 'dalg/archive')
    types = {e['type'] for e in reader_dalg.events()}
    assert {'detector.configured', 'detector.frame', 'detector.detection', 'estimate.confirmed',
            'detections.published'} <= types


def test_search_refuses_to_start_without_a_detector(tmp_path, launcher):
    instance, report, env, dalg, reader = search_chain(tmp_path, launcher, ('lidar-baseline',))
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and not (report / 'dalg').exists():
        time.sleep(0.2)
    time.sleep(2.0)
    assert reader.read() == {}, 'no detector profile, no detections plane'
    refused = launcher([DMSN, '--id', instance, '--no-ui', '--profile', 'fixture-search', '--start',
                        '--detector-wait-s', '2', '--timeout', '10'], env, name='refused')
    assert refused.wait(timeout=40) == 2, refused.log.read_text()
    text = refused.log.read_text()
    assert 'no detector publishing `landing_pad`' in text and 'waiting up to 2 s' in text
