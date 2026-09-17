"""Real child lifetimes when diagnostics, startup, or the runner fail."""

import os
from pathlib import Path
import select
import signal
import subprocess
import sys

import pytest

from dtest.owned_process import popen_owned
from dtest.process_harness import DsimProcessHarness


ROOT = Path(__file__).resolve().parents[2]
SLEEP = [sys.executable, '-c', 'import time; time.sleep(60)']


@pytest.fixture
def harness(tmp_path, monkeypatch):
    instance = DsimProcessHarness(tmp_path)
    children = []

    def start():
        instance._stderr_fh = instance._stderr_path.open('w')
        instance.process = popen_owned(SLEEP, stderr=instance._stderr_fh)
        children.append(instance.process)

    monkeypatch.setattr(instance, '_start', start)
    try:
        yield instance, children
    finally:
        instance.close()
        for child in children:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=5)


def test_normal_context_exit_stops_and_reaps_child(harness):
    instance, children = harness
    with instance:
        assert children[-1].poll() is None
    assert children[-1].poll() is not None
    assert instance.process is None and instance._stderr_fh is None
    instance.close()  # Cleanup remains safe to repeat.


def test_failure_reporting_error_preserves_assertion_and_stops_child(harness, monkeypatch):
    instance, children = harness

    def broken_report(**details):
        raise OSError('disk full')

    monkeypatch.setattr(instance, 'save_failure', broken_report)
    with pytest.raises(AssertionError, match='original failure') as caught:
        with instance:
            raise AssertionError('original failure')
    assert 'disk full' in caught.value.__notes__[0]
    assert children[-1].poll() is not None
    assert instance.process is None and instance._stderr_fh is None


def test_interrupt_during_failure_reporting_still_stops_child(harness, monkeypatch):
    instance, children = harness

    def interrupted_report(**details):
        raise KeyboardInterrupt

    monkeypatch.setattr(instance, 'save_failure', interrupted_report)
    with pytest.raises(KeyboardInterrupt):
        with instance:
            raise AssertionError('original failure')
    assert children[-1].poll() is not None


@pytest.mark.parametrize('error', [RuntimeError, KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize('use_context', [False, True])
def test_failed_start_stops_child_even_before_context_entry(harness, monkeypatch, error, use_context):
    instance, children = harness
    start = instance._start

    def failed_start():
        start()
        raise error('startup interrupted')

    monkeypatch.setattr(instance, '_start', failed_start)
    with pytest.raises(error, match='startup interrupted'):
        if use_context:
            with instance:
                pytest.fail('startup should not succeed')
        else:
            instance.start()
    assert children[-1].poll() is not None
    assert instance.process is None and instance._stderr_fh is None


@pytest.mark.skipif(sys.platform != 'linux', reason='Linux parent-death signal')
@pytest.mark.parametrize('exit_mode', ['exit', 'terminate', 'kill'])
def test_simulator_child_dies_when_runner_exits_without_cleanup(exit_mode):
    # A real stand-in runner owns a child that ignores graceful shutdown.
    # EOF on its inherited stdout proves it exited even before it is reaped.
    child_code = (
        'import os, signal, time; '
        'signal.signal(signal.SIGTERM, signal.SIG_IGN); '
        'print(os.getpid(), flush=True); time.sleep(60)'
    )
    owner_code = (
        'import sys; from dtest.owned_process import popen_owned; '
        f'popen_owned([sys.executable, "-c", {child_code!r}]); '
        'sys.stdin.readline()'
    )
    owner = subprocess.Popen([sys.executable, '-c', owner_code], cwd=ROOT,
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    child_pid = None
    try:
        assert select.select([owner.stdout], [], [], 10)[0], 'child did not become ready'
        child_pid = int(owner.stdout.readline())
        assert not select.select([owner.stdout], [], [], 0)[0], 'child exited while runner was alive'
        if exit_mode == 'exit':
            owner.stdin.write('\n')
            owner.stdin.flush()
        elif exit_mode == 'terminate':
            owner.terminate()
        else:
            owner.kill()
        owner.wait(timeout=5)
        assert select.select([owner.stdout], [], [], 5)[0], 'child survived its test runner'
        assert owner.stdout.read() == ''
    finally:
        if owner.poll() is None:
            owner.kill()
        owner.wait(timeout=5)
        owner.stdin.close()
        owner.stdout.close()
        if child_pid is not None:
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.skipif(sys.platform != 'linux', reason='Linux parent-death signal')
def test_launcher_refuses_to_run_if_parent_already_died(tmp_path):
    marker = tmp_path / 'should-not-exist'
    done = subprocess.run([
        sys.executable, str(ROOT / 'dtest/owned_process.py'), '-1',
        sys.executable, '-c', f'from pathlib import Path; Path({str(marker)!r}).touch()',
    ], timeout=5)
    assert done.returncode != 0
    assert not marker.exists()
