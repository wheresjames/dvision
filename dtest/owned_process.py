"""Launch test children that cannot outlive their runner on Linux.

Use a fresh interpreter to set the parent-death signal before exec, rather
than preexec_fn: the test runner also has recorder threads. Exec preserves
the PID, exit status, streams and signal handling expected by the harness.
Other platforms retain normal Popen behavior and explicit harness cleanup.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
import signal
import subprocess
import sys


def popen_owned(command, **kwargs) -> subprocess.Popen:
    """Start from the runner's main thread; kill the child if that thread dies."""
    if sys.platform == 'linux':
        command = [sys.executable, str(Path(__file__).resolve()), str(os.getpid()), *command]
    return subprocess.Popen(command, **kwargs)


def main() -> None:
    parent_pid = int(sys.argv[1])
    command = sys.argv[2:]
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl.argtypes = [ctypes.c_int, *([ctypes.c_ulong] * 4)]
    libc.prctl.restype = ctypes.c_int
    # PR_SET_PDEATHSIG survives exec. SIGKILL also covers a simulator stuck
    # inside native rendering code or ignoring SIGTERM.
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    # The runner may have died between Popen and installing the signal.
    if os.getppid() != parent_pid:
        raise SystemExit(1)
    os.execvp(command[0], command)


if __name__ == '__main__':
    main()
