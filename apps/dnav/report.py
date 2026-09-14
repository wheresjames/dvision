"""What dnav leaves behind: the summary, the event log, and the route image.

The standard contract (docs/reports.md): one directory named after the module,
inside the report root the session provider publishes in the neutral context
and this module never constructs. `summary.json` holds the numbers,
`events.jsonl` holds what happened in order, `report.html` is the dark page a
person opens first (built from the summary alone), `route.png` is the pane's own
rendering over evidence-derived cost -- with the reference background the
operator displayed, when one was displayed, whose exact revision the archive
also holds, referenced by checksum -- and `archive/` holds every planning
attempt with its exact numeric inputs. No report ever reads a world file.

Nothing here may raise into the run. A missing report is a nuisance; a planner
that died because a disk filled up is a lost experiment.
"""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path
from typing import Any, Iterable

#: Statuses shown in the page's history table; the full list stays in summary.json.
HISTORY_ROWS = 60


def write_report(report_dir: str | Path, *, summary: dict[str, Any],
                 events: Iterable[dict[str, Any]] = (),
                 image=None) -> Path:
    """Write every artefact, reporting -- never raising -- on each failure."""
    directory = Path(report_dir)
    directory.mkdir(parents=True, exist_ok=True)
    _guard('summary', lambda: (directory / 'summary.json').write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + '\n',
        encoding='utf-8'))
    _guard('events', lambda: _write_events(directory / 'events.jsonl', events))
    if image is not None:
        _guard('route image', lambda: image.save(directory / 'route.png'))
    _guard('html', lambda: write_html(directory, summary))
    return directory


def _json(value: Any) -> str:
    return json.dumps(value, default=str)


def _pre(value: Any) -> str:
    return '<pre>' + html.escape(json.dumps(value, indent=2, sort_keys=True, default=str)) + '</pre>'


def _num(value: Any, digits: int = 2) -> str:
    return f'{value:.{digits}f}' if isinstance(value, (int, float)) and not isinstance(value, bool) else '--'


def write_html(report_dir: str | Path, summary: dict[str, Any]) -> Path:
    """``report.html`` from the summary (and ``route.png`` when present). Tolerates partial summaries."""
    from dcmn import report_html as page
    directory = Path(report_dir)
    route = summary.get('route') or {}
    navigation = summary.get('navigation') or {}
    clearance = navigation.get('clearance') or {}
    policy = summary.get('policy') or {}
    detour = summary.get('detour') or {}
    goal = (summary.get('goal') or {}).get('position')
    status = route.get('status') or 'unknown'
    grade = 'ok' if status == 'ok' else 'warn' if status in ('stale_map', 'no_goal', 'stale_pose') else 'bad'
    health = summary.get('health') or 'unknown'
    recording = summary.get('recording') or {}
    permission = clearance.get('eligible')
    evidence = clearance.get('evidence_check') or {}
    blocks = ['<style>pre{white-space:pre-wrap;word-break:break-word;font-size:12px;margin:0}'
              '.section table td{vertical-align:top}</style>']
    blocks.append(page.section('Outcome', page.facts([
        ('Route', page.graded(status, grade)),
        ('Health', page.graded(health, 'ok' if health == 'ok' else 'warn')),
        ('Goal', page.esc(_json(goal))),
        ('Planner', page.esc(summary.get('planner'))),
        ('Attempts / plans', page.esc(f"{summary.get('attempts', '--')} / {summary.get('plans', '--')}")),
        ('Length m', page.esc(_num(route.get('length_m')))),
        ('Cost', page.esc(_num(route.get('cost')))),
        ('vs control (length, cost)', page.esc(f"{_num(detour.get('length_ratio'))}x, {_num(detour.get('cost_ratio'))}x")),
        ('Recording', page.graded(recording.get('state', 'none'), 'ok' if recording.get('complete') else 'warn')),
    ]), f"<p>{page.esc(route.get('reason') or summary.get('reason') or '')}</p>",
        f"<p class=\"muted\">Session <code>{page.esc(summary.get('session_id'))}</code>; the control route is a "
        f"straight-line diagnostic on the same evidence-derived cost, not an oracle.</p>"))
    image = directory / 'route.png'
    blocks.append(page.section('Route', page.figure(image, 'final plan over evidence-derived cost')
                               if image.exists() else '<p class="muted">no route image was written</p>'))
    if navigation:
        blocks.append(page.section('Execution permission', page.facts([
            ('Permission', page.graded('granted' if permission else 'withheld', 'ok' if permission else 'warn')),
            ('Mode', page.esc(clearance.get('permission', 'evidence'))),
            ('Stop generation', page.esc(navigation.get('stop_generation'))),
            ('Geometry revision', page.esc(navigation.get('geometry_revision'))),
            ('Permitted m', page.esc(_num(clearance.get('distance_m')))),
        ]), f"<p>{page.esc(clearance.get('reason'))}</p>",
            f"<p class=\"muted\">Strict evidence check on the same route: {page.esc(evidence.get('reason'))}</p>"
            if evidence else ''))
    counts = summary.get('status_counts') or {}
    if counts:
        blocks.append(page.section('Status counts', page.table(
            ('status', 'publications'), ([page.esc(k), page.esc(v)] for k, v in sorted(counts.items())),
            numeric=(1,))))
    statuses = list(summary.get('statuses') or [])
    if statuses:
        rows = ([page.esc(_num(entry.get('sim_time_s'))), page.esc(entry.get('status')), page.esc(entry.get('planner')),
                 page.esc(entry.get('map_revision')), page.esc(entry.get('reason'))]
                for entry in statuses[-HISTORY_ROWS:])
        blocks.append(page.section(f'Status history (last {min(len(statuses), HISTORY_ROWS)} of {len(statuses)})',
                                   page.table(('sim s', 'status', 'planner', 'map revision', 'reason'), rows,
                                              numeric=(0, 3))))
    sources = ((summary.get('map') or {}).get('sources') or {})
    if sources:
        keys = sorted({k for info in sources.values() if isinstance(info, dict) for k in info})
        blocks.append(page.section('Evidence sources', page.table(
            ('source', *keys), ([page.esc(sid)] + [page.esc(_json(info.get(k)) if isinstance(info.get(k), (dict, list))
                                                            else info.get(k)) for k in keys]
                                for sid, info in sorted(sources.items()) if isinstance(info, dict)))))
    blocks.append(page.section('Recording', _pre(recording)))
    blocks.append(page.section('Policy', _pre(policy)))
    blocks.append(page.section('Final navigation record', _pre(navigation)))
    out = directory / 'report.html'
    out.write_text(page.document(f"dnav planning — {status}", subtitle=str(summary.get('instance') or ''),
                                 blocks=blocks), encoding='utf-8')
    return out


def _write_events(path: Path, events: Iterable[dict[str, Any]]) -> None:
    with path.open('w', encoding='utf-8') as handle:
        for event in events:
            handle.write(json.dumps(event, sort_keys=True, default=str) + '\n')
            # Flushed per record: the run whose log matters most is the one
            # that ended in a crash.
            handle.flush()


def _guard(what: str, write) -> None:
    try:
        write()
    except Exception as exc:                          # noqa: BLE001
        print(f'dnav report: {what}: {exc}', file=sys.stderr)
