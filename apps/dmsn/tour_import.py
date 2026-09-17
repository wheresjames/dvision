"""Tour import: a recorded tour becomes a waypoints mission profile (B9).

The tour files in ``assets/tours`` are how a fixed circuit is described: one
map, a list of x,y,heading stops. This module converts one into the profile
the mission stack flies it with -- a ``waypoints`` mission, the tour's map as
the arena, ``land`` at the end (a circuit that ends in the air is a vehicle
somebody has to catch) -- so the only thing left to edit is what the tour
could not say: budgets, the failure policy.

Used headless (``dmsn --import-tour PATH --out PROFILE``) and from the Setup
tab's Import button, which runs that same command as a child process
(:func:`run_converter`) and opens the profile it wrote -- the isolation B9 asks
for: the window never opens the tour, and a bad tour never touches it. A tour
is map-bound, so the converter is the one dmsn process that may not run under
the truth-isolation hook; the child is started without it.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from dcmn.mission import MAX_TEXT, MAX_WAYPOINTS
from dmsn import profiles as P

#: Wall seconds the child converter may take.
CONVERTER_TIMEOUT_S = 60.0
#: Environment the truth-isolation hook (``dtest.isolation``) is installed
#: through; the converter child is started without it.
ISOLATION_LOG = 'DVISION2_ISOLATION_LOG'
ISOLATION_PREFIXES = ('DVISION2_ISOLATION_', 'DVISION2_FORBID_')


def digest(tour: dict[str, Any]) -> str:
    """The identity of a tour document: its canonical JSON, hashed."""
    return hashlib.sha256(json.dumps(tour, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def convert(tour: dict[str, Any], *, name: str = '') -> dict[str, Any]:
    """A validated waypoints profile from a parsed tour document."""
    raw = tour.get('waypoints')
    if not isinstance(raw, list) or not raw:
        raise ValueError('the tour has no waypoints')
    points: list[list[float]] = []
    for index, stop in enumerate(raw):
        if not isinstance(stop, dict):
            raise ValueError(f'tour waypoint {index + 1} is not an object')
        try:
            points.append([float(stop['x']), float(stop['y'])])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f'tour waypoint {index + 1} has no numeric x,y') from None
    if len(points) > MAX_WAYPOINTS:
        raise ValueError(f'the tour has {len(points)} waypoints; at most {MAX_WAYPOINTS} fit a profile')
    map_path = str(tour.get('map') or '')
    if not map_path:
        raise ValueError('the tour names no map')
    tour_id = str(tour.get('tour_id') or '')
    profile = P.blank('waypoints', name=(name or tour_id or 'imported tour')[:MAX_TEXT])
    profile.update(
        waypoints=points,
        finish_action='land',
        arena=Path(map_path).stem,
        imported_from=dict(kind='tour', tour_id=tour_id, digest=digest(tour), map=map_path,
                          map_sha=str(tour.get('map_sha') or ''), waypoints=len(points)),
    )
    profile['assisted']['tour_import'] = True
    return P.validate(profile)


def convert_file(tour_path: str | Path, out_path: str | Path, *, name: str = '') -> tuple[Path, dict[str, Any]]:
    """Read a tour, write the profile, and return ``(written path, profile)``."""
    tour_path = Path(tour_path).expanduser()
    try:
        tour = json.loads(tour_path.read_text(encoding='utf-8'))
    except FileNotFoundError:
        raise ValueError(f'no tour at {str(tour_path)!r}') from None
    except json.JSONDecodeError as exc:
        raise ValueError(f'{tour_path}: not JSON: {exc}') from None
    if not isinstance(tour, dict):
        raise ValueError(f'{tour_path}: a tour is a JSON object')
    profile = convert(tour, name=name or tour_path.stem)
    return P.save(profile, out_path), profile

def converter_env(environ: dict[str, str] | None = None) -> dict[str, str]:
    """The environment for the converter child: this one, minus the truth-isolation hook.

    The hook is a ``sitecustomize`` on ``PYTHONPATH`` in the directory holding
    its violation log, configured by ``DVISION2_ISOLATION_*`` and
    ``DVISION2_FORBID_*`` variables; all of it is dropped, and nothing else.
    """
    env = dict(os.environ if environ is None else environ)
    log = env.get(ISOLATION_LOG)
    if log:
        hook_dir = os.path.abspath(os.path.dirname(log))
        paths = [p for p in env.get('PYTHONPATH', '').split(os.pathsep)
                 if p and os.path.abspath(p) != hook_dir]
        if paths: env['PYTHONPATH'] = os.pathsep.join(paths)
        else: env.pop('PYTHONPATH', None)
    for key in [k for k in env if k.startswith(ISOLATION_PREFIXES)]: del env[key]
    return env


def run_converter(tour_path: str | Path, out_path: str | Path) -> tuple[bool, str]:
    """Run ``dmsn --import-tour`` as a child process: ``(converted, message)``.

    This process never opens the tour; it only learns whether the child wrote
    ``out_path``, and the child's last line of output says why when it did not.
    """
    command = [sys.executable, str(P.ROOT / 'apps' / 'dmsn' / 'dmsn.py'),
               '--import-tour', str(tour_path), '--out', str(out_path)]
    try:
        result = subprocess.run(command, cwd=str(P.ROOT), capture_output=True, text=True,
                                timeout=CONVERTER_TIMEOUT_S, env=converter_env())
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    lines = (result.stdout if result.returncode == 0 else (result.stderr or result.stdout)).strip().splitlines()
    if result.returncode != 0: return False, lines[-1] if lines else 'converter failed'
    return True, lines[-1] if lines else f'wrote {out_path}'
