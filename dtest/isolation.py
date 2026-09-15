"""Process-level truth isolation for operational modules under test.

``isolated_env`` returns environment variables that make a child Python
process install an audit hook (via a generated ``sitecustomize``) before any
application code runs. The hook refuses -- and logs -- any attempt to open a
world file, a tour, a planner-query sidecar or anything under a simulator
``truth`` directory, and any import of the evaluator, the query tooling or the
simulator's world/raycast modules. A dalg or dnav that passes under it did not
read privileged knowledge, whatever its helpers do internally.

The deterministic provider is exempt: it is the one component allowed to hold
truth-equivalent state.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_DIRS = (ROOT / 'assets/maps', ROOT / 'assets/tours', ROOT / 'tests/assets/maps',
                  ROOT / 'tests/assets/tours', ROOT / 'tests/assets/planner_queries')
FORBIDDEN_FRAGMENTS = (f'{os.sep}dsim{os.sep}truth',)
FORBIDDEN_MODULES = ('dtest.evaluation', 'dtest.queries', 'dsim.range', 'dsim.backend',
                     'dsim.scene', 'dsim.dsim', 'dway.tour', 'dway.mission')

SITECUSTOMIZE = '''
import os, sys
_log = os.environ["DVISION2_ISOLATION_LOG"]
_dirs = tuple(p for p in os.environ["DVISION2_FORBID_DIRS"].split(os.pathsep) if p)
_frags = tuple(p for p in os.environ["DVISION2_FORBID_FRAGMENTS"].split(os.pathsep) if p)
_mods = tuple(p for p in os.environ["DVISION2_FORBID_MODULES"].split(",") if p)

def _violation(what):
    with open(_log, "a", encoding="utf-8") as handle:
        handle.write(what + "\\n")
    raise PermissionError("truth isolation: " + what)

def _hook(event, args):
    if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        path = os.path.abspath(os.fsdecode(args[0]))
        if path == _log: return
        if any(path == d or path.startswith(d + os.sep) for d in _dirs) or any(f in path for f in _frags):
            _violation("open " + path)
    elif event == "import" and args and args[0] in _mods:
        _violation("import " + args[0])

sys.addaudithook(_hook)
'''


def isolated_env(directory: Path, base: dict | None = None) -> dict:
    """Environment for a child process that must not touch truth. Violations go to ``directory/violations.log``."""
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    (directory / 'sitecustomize.py').write_text(SITECUSTOMIZE, encoding='utf-8')
    env = dict(os.environ if base is None else base)
    env['PYTHONPATH'] = os.pathsep.join([str(directory), env.get('PYTHONPATH', '')]).rstrip(os.pathsep)
    env['DVISION2_ISOLATION_LOG'] = str(directory / 'violations.log')
    env['DVISION2_FORBID_DIRS'] = os.pathsep.join(str(p) for p in FORBIDDEN_DIRS)
    env['DVISION2_FORBID_FRAGMENTS'] = os.pathsep.join(FORBIDDEN_FRAGMENTS)
    env['DVISION2_FORBID_MODULES'] = ','.join(FORBIDDEN_MODULES)
    return env


def violations(directory: Path) -> list[str]:
    log = Path(directory) / 'violations.log'
    return log.read_text(encoding='utf-8').splitlines() if log.exists() else []
