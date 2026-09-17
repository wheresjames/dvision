"""The mission chain as separate processes: dsim, dalg, dnav, gated dway and dmsn on maze_020.

The chain an operator runs, headless: dway waits on the ground until dmsn asks
for a launch, dmsn sets the goal as the mission authority, and the mission ends
on dway's COMPLETE. It checks the wiring end to end and that no operational
module read the world. Like the target-flight chain it takes minutes of wall
time, so it runs with ``DVISION_NIGHTLY=1``. Arrival is asserted here because
the committed goto profile is the documented "done" command for the mission
framework; how long it takes is printed, not judged.

The area and waypoint missions are smoke runs: explore flies for a fixed sim
time on ``lidar-baseline`` and the imported waypoint tour flies to its end, and
each must move, set goals, end with a reason and write its report. How much of
the maze explore covers, or how many waypoints are reached, is not a threshold.
"""
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import time
import uuid

import pytest

from dcmn.archive import ArchiveReader
from dtest.isolation import isolated_env, violations

ROOT = Path(__file__).resolve().parents[2]
MAP = ROOT / 'assets/maps/maze_020.txt'
MISSION_TIMEOUT_S = 420

pytestmark = [
    pytest.mark.nightly,
    pytest.mark.skipif(os.environ.get('DVISION_NIGHTLY') != '1',
                       reason='set DVISION_NIGHTLY=1 to fly the multi-process mission chain'),
]


def fly_chain(tmp_path, profile, *sets, timeout_s=MISSION_TIMEOUT_S, algorithms=('lidar-baseline',)):
    """Run the whole chain with dmsn headless on ``profile``: ``(exit code, summary)``."""
    instance = 'mission-' + uuid.uuid4().hex[:8]
    report = tmp_path / 'reports'
    processes, logs = [], {}

    def start(name, args, env=None):
        log = tmp_path / f'{name}.log'
        with log.open('w') as out:
            processes.append(subprocess.Popen([sys.executable, *map(str, args)], cwd=ROOT, env=env,
                                              stdout=out, stderr=subprocess.STDOUT))
        logs[name] = log
        return processes[-1]

    def tails():
        return '\n'.join(f'--- {name}\n' + ''.join(log.read_text().splitlines(True)[-40:])
                         for name, log in logs.items())

    try:
        start('dsim', [ROOT / 'apps/dsim/dsim.py', '--id', instance, '--map', MAP, '--drone-profile', 'camera-lidar',
                       '--sim-speed', '1.5', '--no-ui', '--report-dir', report])
        time.sleep(3)
        env = isolated_env(tmp_path / 'isolation')
        env['DVISION2_MISSION_DIR'] = str(tmp_path / 'missions')
        start('dalg', [ROOT / 'apps/dalg/dalg.py', '--id', instance, '--profiles', *algorithms, '--no-ui'], env)
        start('dnav', [ROOT / 'apps/dnav/dnav.py', '--id', instance, '--execution-profile', 'sim-target', '--no-ui'], env)
        start('dway', [ROOT / 'apps/dway/dway.py', '--id', instance, '--mode', 'target', '--execution-profile',
                       'sim-target', '--wait-for-mission', '--no-ui'], env)
        time.sleep(2)
        overrides = [item for value in sets for item in ('--set', value)]
        dmsn = start('dmsn', [ROOT / 'apps/dmsn/dmsn.py', '--id', instance, '--no-ui', '--profile', profile,
                              *overrides, '--start', '--timeout', timeout_s], env)
        try:
            code = dmsn.wait(timeout=timeout_s + 60)
        except subprocess.TimeoutExpired:
            pytest.fail('dmsn did not finish\n' + tails())
        assert violations(tmp_path / 'isolation') == [], tails()
    finally:
        for process in reversed(processes):
            if process.poll() is None:
                process.send_signal(signal.SIGINT)  # a clean stop: every module writes its report
                try: process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try: process.wait(timeout=10)
                    except subprocess.TimeoutExpired: process.kill(); process.wait(timeout=5)
    summaries = sorted(report.glob('**/dmsn/summary.json'))
    assert summaries, 'dmsn wrote no report\n' + tails()
    summary = json.loads(summaries[-1].read_text())
    summary['_directory'], summary['_tails'] = summaries[-1].parent, tails()
    summary['_report'] = report
    print(f"{profile}: {summary['outcome'] or summary['phase']}: {summary['reason']}; {summary['elapsed_s']} s sim, "
          f"{summary['track_m']} m flown, {len(summary['goals'])} goals")
    return code, summary


def assert_smoke(code, summary):
    """Moved, set goals, ended with a reason, wrote a report -- nothing about how well."""
    tails = summary['_tails']
    assert code in (0, 1), (code, tails)
    assert summary['outcome'] and summary['reason'], tails
    assert summary['goals'], ('no goal was set', tails)
    assert summary['track_m'] > 1.0, ('the vehicle did not move', tails)
    assert (summary['_directory'] / 'mission.png').exists()
    assert ArchiveReader(summary['_directory'] / 'archive').validate()['complete']


def test_goto_on_maze_020_launches_on_intent_and_completes(tmp_path) -> None:
    code, summary = fly_chain(tmp_path, 'maze_020-goto')
    tails = summary['_tails']
    assert code == 0 and summary['outcome'] == 'complete', (summary['reason'], tails)
    assert [p['phase'] for p in summary['phases']][:3] == ['idle', 'starting', 'running']
    assert any(a['kind'] == 'launch' and a['state'] == 'completed' for a in summary['acks']), summary['acks']
    assert summary['goals'][-1]['outcome'] == 'reached'
    assert summary['arena']['mismatch'] is False
    assert ArchiveReader(summary['_directory'] / 'archive').validate()['complete']
    from dmsn.runner import replay
    assert replay(summary['_directory'] / 'archive')['mismatches'] == []


def test_explore_on_maze_020_moves_sets_goals_and_ends_with_a_reason(tmp_path) -> None:
    code, summary = fly_chain(tmp_path, 'maze_020-explore', 'budgets.time_s=180', 'budgets.time_action=land')
    assert_smoke(code, summary)
    assert any(a['kind'] == 'launch' and a['state'] == 'completed' for a in summary['acks']), summary['acks']
    from dmsn.runner import replay
    assert replay(summary['_directory'] / 'archive')['mismatches'] == []


def test_the_imported_waypoint_tour_on_maze_020_ends_with_a_reason(tmp_path) -> None:
    code, summary = fly_chain(tmp_path, 'maze_020-waypoints', 'budgets.time_s=300')
    assert_smoke(code, summary)


def test_search_on_maze_020_ends_with_a_reason_and_both_reports(tmp_path) -> None:
    """Search on the rendered maze: it must end with a reason and leave reports; landing on the pad is not asserted."""
    code, summary = fly_chain(tmp_path, 'maze_020-search', 'budgets.time_s=300',
                              algorithms=('lidar-baseline', 'red-disc'))
    tails = summary['_tails']
    assert code in (0, 1), (code, tails)
    assert summary['outcome'] and summary['reason'], tails
    assert summary['profile']['mission'] == 'search'
    assert 'estimate_log' in summary['mission_summary']
    dalg = sorted(summary['_report'].glob('**/dalg/summary.json'))
    assert dalg, tails
    section = json.loads(dalg[-1].read_text())['detectors']
    assert section['detectors'] and section['detectors'][0]['processed'] > 0, section
    print(f"search: estimates {[(e['id'], e['xy_m'], e['confirmed']) for e in section['estimates']]}")
