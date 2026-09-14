"""The whole chain as separate processes: dsim, dalg, dnav and dway in target mode.

This is the holistic wiring test for flying to a target: the same four commands
an operator runs, headless, on the maze from the README. It checks that the chain
is connected end to end -- dway takes off by itself, dnav's route is admitted and
started automatically, targets reach the vehicle, the vehicle makes progress
toward the goal, and a report is written without any consumer reading the world.
Arrival is recorded, not required: how far the flight gets is a performance
result (evidence quality, stop churn), printed for comparison between runs.

It takes several minutes of wall time, so it runs with ``DVISION_NIGHTLY=1``.
On failure the assertion message carries the tail of every module's log.
"""
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

import pytest

from dtest.isolation import isolated_env, violations

ROOT = Path(__file__).resolve().parents[1]
MAP = ROOT / 'assets/maps/maze_020.txt'
GOAL = (52.444, 2.389)
FLIGHT_TIMEOUT_S = 240
START = (31.5, 27.5)

pytestmark = [
    pytest.mark.nightly,
    pytest.mark.skipif(os.environ.get('DVISION_NIGHTLY') != '1',
                       reason='set DVISION_NIGHTLY=1 to fly the multi-process target chain'),
]


def test_target_mode_flies_the_maze_to_the_goal(tmp_path) -> None:
    instance = 'target-' + uuid.uuid4().hex[:8]
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
        # The consumers must not read the world: the isolation hook records any attempt.
        env = isolated_env(tmp_path / 'isolation')
        start('dalg', [ROOT / 'apps/dalg/dalg.py', '--id', instance, '--profiles', 'lidar-baseline.json',
                       'optical-flow-baseline.json', '--no-ui'], env)
        start('dnav', [ROOT / 'apps/dnav/dnav.py', '--id', instance, '--goal', f'{GOAL[0]},{GOAL[1]}', '--no-ui',
                       '--execution-profile', 'sim-target'], env)
        dway = start('dway', [ROOT / 'apps/dway/dway.py', '--id', instance, '--mode', 'target', '--wait-for',
                              'algorithm', '--no-ui', '--exit-on-finish', '--timeout', FLIGHT_TIMEOUT_S], env)
        try:
            code = dway.wait(timeout=FLIGHT_TIMEOUT_S + 120)
        except subprocess.TimeoutExpired:
            pytest.fail('dway did not finish\n' + tails())
        assert code in (0, 2), f'dway exit {code}\n' + tails()
        assert violations(tmp_path / 'isolation') == [], tails()
    finally:
        for process in reversed(processes):
            if process.poll() is None:
                process.terminate()
                try: process.wait(timeout=10)
                except subprocess.TimeoutExpired: process.kill(); process.wait(timeout=5)
    summaries = sorted((report / 'dway').glob('archive*/report/summary.json'))
    assert summaries, 'dway wrote no flight report\n' + tails()
    summary = json.loads(summaries[-1].read_text())
    events = [json.loads(line) for line in (summaries[-1].parent / 'events.jsonl').read_text().splitlines()]
    phases = [e['data'].get('phase') for e in events if e['type'] == 'execution.launch']
    assert 'airborne' in phases, (phases, tails())
    starts = [e['data'] for e in events if e['type'] == 'execution.control' and e['data'].get('action') == 'start']
    assert any(s.get('origin') == 'auto' and s.get('accepted') for s in starts), (starts, tails())
    assert summary['cadence']['targets_sent'] > 0 and summary['cadence']['targets_accepted'] > 0
    final = summary['final_pose']
    assert final and math.dist(final[:2], START) >= 3.0, ('the vehicle did not move', final, tails())
    assert math.dist(final[:2], GOAL) < math.dist(START, GOAL), ('no progress toward the goal', final)
    arrived = summary['outcome'] == 'complete'
    print(f"target flight: {'ARRIVED' if arrived else 'not arrived: ' + str(summary['reason'])}; "
          f"{summary['elapsed_s']:.0f} s data clock, {math.dist(final[:2], GOAL):.1f} m from the goal, "
          f"{sum(1 for h in summary['holds'] if h.get('phase') == 'confirmed')} stops, "
          f"{summary['cadence']['targets_sent']} targets, estimated track {summary['estimated_track_m']:.1f} m")
