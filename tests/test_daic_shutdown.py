"""Shutdown must always close the window, however badly cleanup goes.

Regression: a shared-memory handle raised while daic was tearing down (a numpy
view into the video buffer was still alive because a signal interrupted a tick).
The exception escaped ``close()`` before ``root.destroy()``, so the window stayed
open -- and because ``_closed`` had already been latched at the top of the
method, every later close-button click returned immediately and did nothing.
The window became impossible to close by any means short of SIGKILL.
"""

from __future__ import annotations

import argparse
import sys
import types

import pytest

from daic.daic import DaicController, HeadlessAgent


class FakeRoot:
    """Stands in for the Tk root; records whether the window was destroyed."""

    def __init__(self) -> None:
        self.destroyed = False
        self.idle_calls: list = []

    def destroy(self) -> None:
        self.destroyed = True

    def after_idle(self, callback) -> None:
        self.idle_calls.append(callback)

    # save_window_pos() probes these before writing the position store.
    def winfo_x(self) -> int:
        return 10

    def winfo_y(self) -> int:
        return 20


class ExplodingHandle:
    """A shared-memory handle that cannot be closed, as memvid does in use."""

    def __init__(self, message: str = "cannot close memvid while exported "
                                      "buffer views exist") -> None:
        self.message = message
        self.closed = False

    def close(self) -> None:
        raise RuntimeError(self.message)

    def getAll(self) -> dict:
        return {}


class GoodHandle:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True

    def getAll(self) -> dict:
        return {}


def _controller(**overrides) -> DaicController:
    """Build a controller without Tk or shared memory, for shutdown only."""
    agent = DaicController.__new__(DaicController)
    agent.args = argparse.Namespace(id="test")
    agent.root = FakeRoot()
    agent._closed = False
    agent.running = True
    agent.slam_detector = None
    agent.logger = None
    agent.reporter = None
    agent.status = None
    agent.command = None
    agent.video = None
    agent._last_planner_state_name = "IDLE"
    agent._last_target_dist_m = 0.0
    agent._send = lambda *a, **k: None
    for key, value in overrides.items():
        setattr(agent, key, value)
    return agent


def test_close_destroys_the_window_when_a_handle_cannot_be_closed():
    """The exact failure that made the window unclosable."""
    agent = _controller(video=ExplodingHandle())
    agent.close()
    assert agent.root.destroyed, "a failing handle must not keep the window open"


def test_close_destroys_the_window_when_every_step_fails():
    agent = _controller(
        status=ExplodingHandle("status gone"),
        command=ExplodingHandle("command gone"),
        video=ExplodingHandle("video gone"),
        logger=types.SimpleNamespace(close=_raise),
        slam_detector=types.SimpleNamespace(stop=_raise),
    )
    agent._send = _raise
    agent.close()
    assert agent.root.destroyed


def _raise(*_args, **_kwargs):
    raise RuntimeError("boom")


def test_a_failing_step_does_not_skip_the_remaining_cleanup():
    """One broken resource must not strand the others open."""
    good = GoodHandle()
    agent = _controller(status=ExplodingHandle(), command=good, video=GoodHandle())
    agent.close()
    assert good.closed, "cleanup stopped at the first failure"
    assert agent.video.closed
    assert agent.root.destroyed


def test_close_reports_what_failed_rather_than_swallowing_it(capsys):
    agent = _controller(video=ExplodingHandle("video is busy"))
    agent.close()
    assert "video is busy" in capsys.readouterr().err


def test_close_is_idempotent():
    agent = _controller(command=GoodHandle())
    agent.close()
    agent.root.destroyed = False
    agent.close()          # second close-button click
    assert not agent.root.destroyed, "close must not run twice"


def test_close_stops_the_tick_loop():
    agent = _controller()
    agent.close()
    assert agent.running is False


def test_request_stop_defers_the_close_to_a_safe_point():
    """A signal can land mid-tick; the teardown must wait for the idle queue.

    Closing the video buffer while an interrupted tick still holds a numpy view
    into it is what raised in the first place, so the signal handler must not
    tear anything down directly.
    """
    agent = _controller(video=GoodHandle())
    agent.request_stop()
    assert agent.running is False
    assert agent.root.idle_calls == [agent.close]
    assert not agent.root.destroyed, "teardown ran inside the signal handler"

    agent.root.idle_calls[0]()      # Tk drains the idle queue
    assert agent.root.destroyed


def test_request_stop_falls_back_to_closing_directly():
    """If the idle queue is unavailable, shutdown must still happen."""
    agent = _controller()
    agent.root.after_idle = _raise
    agent.request_stop()
    assert agent.root.destroyed


# --- headless agent ---------------------------------------------------------

def _headless() -> HeadlessAgent:
    agent = HeadlessAgent.__new__(HeadlessAgent)
    agent.args = argparse.Namespace(id="test", verbose=False, fps=10)
    agent._closed = False
    agent.running = True
    agent.slam_detector = None
    agent.logger = None
    agent.reporter = None
    agent.status = None
    agent.command = None
    agent.video = None
    agent._last_planner_state_name = "IDLE"
    agent._last_target_dist_m = 0.0
    agent._send = lambda *a, **k: None
    return agent


def test_headless_request_stop_ends_the_run_loop():
    agent = _headless()
    agent.request_stop()
    assert agent.running is False


def test_headless_run_closes_even_when_the_loop_exits_on_its_own():
    """The report must be written when the video buffer disappears, too."""
    agent = _headless()
    ticks = []

    def tick(now):
        ticks.append(now)
        agent.running = False          # e.g. the video buffer vanished

    agent._tick = tick
    agent.run()
    assert ticks, "the loop never ran"
    assert agent._closed, "run() exited without closing"


def test_headless_close_is_idempotent():
    agent = _headless()
    handle = GoodHandle()
    agent.command = handle
    agent.close()
    assert agent._closed
    agent.close()          # must not raise on a second call


def test_headless_close_survives_a_broken_logger():
    agent = _headless()
    agent.logger = types.SimpleNamespace(close=_raise)
    agent.close()
    assert agent._closed


@pytest.mark.parametrize("factory", [_controller, _headless])
def test_control_maintenance_reacquires_a_lost_or_expired_lease(factory):
    agent = factory()
    agent.command = object()
    agent.status = types.SimpleNamespace(getAll=lambda: {"control.owner": ""})
    agent.control_source = "daic-test"
    agent.control_lease = "lease"
    agent._last_heartbeat = 0.0
    sent = []
    agent._send = lambda typ, **fields: sent.append(typ)

    agent._maintain_control(10.0)

    assert sent == ["acquire_control"]
