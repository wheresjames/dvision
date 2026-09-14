"""dalg and dnav as processes, with world, tour and truth access forbidden.

Each operational process runs under an audit hook (dtest.isolation) that
refuses to open a world file, a tour, a planner query or anything under a
simulator truth directory, and refuses to import the evaluator or the
simulator's world modules. The provider -- the deterministic fixture, or dsim --
is the only process allowed truth-equivalent state, and it supplies it only as
labelled ideal poses and sensor samples.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
import uuid
from pathlib import Path

from dcmn.archive import ArchiveReader
from dtest.isolation import isolated_env, violations

ROOT = Path(__file__).resolve().parents[1]


def _wait(predicate, timeout=20.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value: return value
        time.sleep(.05)
    raise AssertionError("timed out")


def _start(args, env, log):
    return subprocess.Popen([sys.executable, *map(str, args)], cwd=ROOT, env=env,
                            stdout=subprocess.DEVNULL, stderr=log.open('w'), start_new_session=True)


def _stop(*processes):
    for process in processes:
        if process is not None and process.poll() is None:
            process.terminate()
            try: process.wait(timeout=5)
            except subprocess.TimeoutExpired: process.kill()


def test_late_provider_isolated_dalg_and_dnav_plan_and_archive(tmp_path):
    """Modules first, provider late: waiting is visible, then evidence and routes flow."""
    instance = 'iso-' + uuid.uuid4().hex[:8]
    report = tmp_path / 'session'
    env = isolated_env(tmp_path / 'hook')
    dalg = dnav = provider = None
    try:
        dalg = _start([ROOT/'apps/dalg/dalg.py', '--id', instance, '--profile', 'lidar-baseline',
                       '--no-ui', '--timeout', '14'], env, tmp_path/'dalg.log')
        dnav = _start([ROOT/'apps/dnav/dnav.py', '--id', instance, '--no-ui', '--goal', '8,2',
                       '--timeout', '14'], env, tmp_path/'dnav.log')
        time.sleep(1.5)
        assert dalg.poll() is None and dnav.poll() is None, (tmp_path/'dalg.log').read_text()
        provider = _start([ROOT/'dtest/provider.py', '--id', instance, '--report-dir', report,
                           '--sensors', 'scan', '--path', '2,0;2,2.5;2,2.5', '--timeout', '30'],
                          None, tmp_path/'provider.log')
        _wait(lambda: dalg.poll() is not None and dnav.poll() is not None, timeout=40)
        assert dalg.returncode == 0, (tmp_path/'dalg.log').read_text()
        assert dnav.returncode == 0, (tmp_path/'dnav.log').read_text()
    finally:
        _stop(dalg, dnav, provider)
    assert violations(tmp_path/'hook') == []
    dalg_summary = json.loads((report/'dalg/summary.json').read_text())
    assert dalg_summary['state'] == 'RUNNING' and dalg_summary['pose_provider'] == 'ideal-fixture'
    assert dalg_summary['geometry_basis'] == 'goal'
    dnav_summary = json.loads((report/'dnav/summary.json').read_text())
    assert dnav_summary['status_counts'].get('ok', 0) >= 1, dnav_summary['statuses']
    assert dnav_summary['authority']['role'] == 'ui'
    for module in ('dalg', 'dnav'):
        check = ArchiveReader(report/module/'archive').validate()
        assert check['complete'], (module, check['errors'])
    attempts = ArchiveReader(report/'dnav/archive').attempts()
    assert any(a['data']['route']['status'] == 'ok' for a in attempts)
    assert (report/'dnav/route.png').is_file()


def test_isolation_hook_catches_a_world_read(tmp_path):
    """The hook is real: a process that opens a world file is stopped and logged."""
    env = isolated_env(tmp_path / 'hook')
    result = subprocess.run([sys.executable, '-c', 'open("assets/maps/maze_012.txt").read()'],
                            cwd=ROOT, env=env, capture_output=True, text=True)
    assert result.returncode != 0 and 'truth isolation' in result.stderr
    assert violations(tmp_path/'hook') == [f'open {ROOT}/assets/maps/maze_012.txt']


def test_no_sensors_is_an_observable_waiting_process_that_exits_cleanly(tmp_path):
    instance = 'iso-' + uuid.uuid4().hex[:8]
    env = isolated_env(tmp_path / 'hook')
    provider = dalg = None
    try:
        provider = _start([ROOT/'dtest/provider.py', '--id', instance, '--report-dir', tmp_path/'s',
                           '--sensors', '', '--timeout', '20'], None, tmp_path/'provider.log')
        dalg = _start([ROOT/'apps/dalg/dalg.py', '--id', instance, '--profile', 'lidar-baseline',
                       '--no-ui', '--timeout', '4'], env, tmp_path/'dalg.log')
        _wait(lambda: dalg.poll() is not None)
        assert dalg.returncode == 0, (tmp_path/'dalg.log').read_text()
    finally:
        _stop(dalg, provider)
    summary = json.loads((tmp_path/'s/dalg/summary.json').read_text())
    assert summary['state'] == 'WAITING_SENSORS' and summary['geometry'] is None
    assert violations(tmp_path/'hook') == []


def test_isolated_dalg_and_dnav_against_real_dsim(tmp_path):
    """The dsim provider path: its ideal pose and camera, a ground-plane baseline, a goal."""
    from dtest.process_harness import DsimProcessHarness

    env = isolated_env(tmp_path / 'hook')
    with DsimProcessHarness(tmp_path, map_path=ROOT/'assets/maps/maze_012.txt') as harness:
        dalg = dnav = None
        try:
            dalg = _start([ROOT/'apps/dalg/dalg.py', '--id', harness.id, '--profile',
                           'ground-plane-baseline', '--no-ui', '--timeout', '8'], env, tmp_path/'dalg.log')
            dnav = _start([ROOT/'apps/dnav/dnav.py', '--id', harness.id, '--no-ui', '--goal', '5.5,1.5',
                           '--timeout', '8'], env, tmp_path/'dnav.log')
            _wait(lambda: dalg.poll() is not None and dnav.poll() is not None, timeout=30)
            assert dalg.returncode == 0, (tmp_path/'dalg.log').read_text()
            assert dnav.returncode == 0, (tmp_path/'dnav.log').read_text()
        finally:
            _stop(dalg, dnav)
        report = harness.report_dir
        assert (report/'dsim/truth/trajectory.jsonl').is_file()
    assert violations(tmp_path/'hook') == []
    dalg_summary = json.loads((report/'dalg/summary.json').read_text())
    assert dalg_summary['state'] == 'RUNNING', dalg_summary['reason']
    assert dalg_summary['pose_provider'] == 'ideal-simulation'
    assert dalg_summary['evidence']['front-ground_plane']['revision'] >= 1
    dnav_summary = json.loads((report/'dnav/summary.json').read_text())
    assert dnav_summary['attempts'] >= 1
    assert ArchiveReader(report/'dnav/archive').validate()['complete']


def test_dsim_declares_its_real_clock_epoch_so_an_early_goal_survives(tmp_path):
    """A goal set as soon as dsim is up must not be withdrawn by a startup "discontinuity".

    dsim once started its context with a placeholder clock epoch and corrected it
    on the first pose, which read as a clock reset and dropped any goal dnav had
    already submitted -- so dalg sized coverage without it.
    """
    from dcmn.context import Context
    from dtest.process_harness import DsimProcessHarness

    with DsimProcessHarness(tmp_path, map_path=ROOT/'assets/maps/maze_020.txt') as harness:
        context = Context(harness.id)
        goal = context.set_goal('early-ui', (52.444, 2.389))
        _wait(lambda: (context.read().get('pose') or {}).get('sequence', 0) > 10)
        snapshot = context.read()
        assert snapshot['goal'] is not None and snapshot['goal']['revision'] == goal['revision']
        assert snapshot['pose']['clock_epoch'] == snapshot['clock_epoch'] == goal['clock_epoch']
        assert not [h for h in snapshot['history'] if h['kind'].endswith('epoch_changed')]
