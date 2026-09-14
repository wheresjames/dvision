"""Planner query sidecars: scenario fixtures, parsed by mission/test tooling only.

A committed query (``tests/assets/planner_queries/*.json``) names a start, a goal and
the world it was authored against. That is scenario metadata, so it lives here
rather than in dnav: dnav receives only a normal framed goal through the session
context. This tool submits a query's goal as the ``mission`` authority, exactly
as a mission coordinator would; the map digest stays scenario provenance and is
never checked against anything by an operational module.

    python dtest/queries.py --id NAME tests/assets/planner_queries/maze_013.v1.json
"""
from __future__ import annotations

if __name__ == '__main__':
    import sys as _sys
    from pathlib import Path as _Path
    for _path in (str(_Path(__file__).resolve().parents[1]),
                  str(_Path(__file__).resolve().parents[1] / 'apps')):
        if _path not in _sys.path: _sys.path.insert(0, _path)

import json
from pathlib import Path
from typing import Any


def load_query(path: Path) -> dict[str, Any]:
    """The first query of one committed sidecar: start, goal and scenario provenance."""
    value = json.loads(Path(path).read_bytes())
    if not isinstance(value, dict) or not isinstance(value.get('queries'), list):
        raise ValueError(f'{path}: not a planner query sidecar')
    if not value['queries']:
        raise ValueError(f'{path}: carries no queries')
    first = value['queries'][0]
    for key in ('start', 'goal'):
        if not (isinstance(first.get(key), (list, tuple)) and len(first[key]) >= 2):
            raise ValueError(f'{path}: query {key} must be at least x and y')
    return dict(map=value.get('map', ''), map_sha=value.get('map_sha', ''),
                start=tuple(float(v) for v in first['start'][:2]),
                goal=tuple(float(v) for v in first['goal'][:2]),
                schema_version=value.get('schema_version'), path=str(path))


def submit(instance: str, path: Path, *, writer: str = 'query-fixture', handoff: bool = False):
    """Install a query's goal in the instance context as the mission authority."""
    from dcmn.context import Context
    query = load_query(path)
    goal = Context(instance).set_goal(writer, query['goal'], role='mission', handoff=handoff)
    return query, goal


def main(argv=None) -> int:
    import argparse
    import sys
    parser = argparse.ArgumentParser(description='submit a planner query goal as the mission authority')
    parser.add_argument('--id', required=True)
    parser.add_argument('query')
    parser.add_argument('--handoff', action='store_true', help='take authority from its current holder')
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    try:
        query, goal = submit(args.id, Path(args.query), handoff=args.handoff)
    except (OSError, ValueError) as exc:
        print(f'queries: {exc}', file=sys.stderr); return 1
    print(json.dumps(dict(goal=goal, scenario=query), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
