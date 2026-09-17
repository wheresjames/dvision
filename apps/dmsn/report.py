"""What dmsn leaves behind: the summary, the event log, the mission image and a page.

The standard contract (docs/reports.md): ``<report_root>/dmsn/``, inside the
report root the session provider publishes. ``summary.json`` holds the numbers
(schema ``dvision2.dmsn-report.v1``), ``events.jsonl`` what happened in order,
``mission.png`` the Mission tab's rendering at the end -- evidence, route,
track, goals and home -- and ``report.html`` the page a person opens first,
built from the summary alone. ``archive/`` (written by the runner) holds every
observation, event and decision, which is what a replay reads.

Nothing here may raise into the run.
"""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path
from typing import Any, Iterable


def write_report(report_dir: str | Path, *, summary: dict[str, Any], events: Iterable[dict[str, Any]] = (),
                 image=None) -> Path:
    directory = Path(report_dir)
    directory.mkdir(parents=True, exist_ok=True)
    _guard('summary', lambda: (directory / 'summary.json').write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + '\n', encoding='utf-8'))
    _guard('events', lambda: _write_events(directory / 'events.jsonl', events))
    if image is not None:
        _guard('mission image', lambda: image.save(directory / 'mission.png'))
    _guard('html', lambda: write_html(directory, summary))
    return directory


def _write_events(path: Path, events: Iterable[dict[str, Any]]) -> None:
    with path.open('w', encoding='utf-8') as handle:
        for event in events:
            handle.write(json.dumps(event, default=str) + '\n')


def _guard(what: str, action) -> None:
    try:
        action()
    except Exception as exc:  # noqa: BLE001 - a report failure is a warning, never a crash
        print(f'dmsn report: {what} not written: {type(exc).__name__}: {exc}', file=sys.stderr)


def _num(value: Any, digits: int = 1) -> str:
    return f'{value:.{digits}f}' if isinstance(value, (int, float)) and not isinstance(value, bool) else '--'


def write_html(report_dir: str | Path, summary: dict[str, Any]) -> Path:
    from dcmn import report_html as page
    directory = Path(report_dir)
    outcome = summary.get('outcome') or f"not finished ({summary.get('phase', 'unknown')})"
    grade = 'ok' if outcome == 'complete' else 'warn' if outcome in ('aborted', 'budget') else 'bad'
    profile = summary.get('profile') or {}
    arena = summary.get('arena') or {}
    assisted = summary.get('assisted') or {}
    recording = summary.get('recording') or {}
    blocks = ['<style>pre{white-space:pre-wrap;word-break:break-word;font-size:12px;margin:0}</style>']
    blocks.append(page.section('Outcome', page.facts([
        ('Outcome', page.graded(outcome, grade)),
        ('Reason', page.esc(summary.get('reason'))),
        ('Mission', page.esc(f"{summary.get('mission')} · profile {profile.get('name')} "
                             f"({str(summary.get('profile_digest', ''))[:12]})")),
        ('Elapsed (sim)', page.esc(f"{_num(summary.get('elapsed_s'))} s")),
        ('Track', page.esc(f"{_num(summary.get('track_m'))} m")),
        ('Goals set / suppressed', page.esc(f"{len(summary.get('goals') or [])} / {summary.get('suppressed_goals', 0)}")),
        ('Finish action', page.esc(summary.get('finish_action'))),
        ('Arena', page.graded(f"profile {arena.get('profile') or '--'} · running {arena.get('provider') or '--'}",
                              'warn' if arena.get('mismatch') else 'ok')),
        ('Assisted', page.esc(json.dumps(assisted, default=str))),
        ('Recording', page.graded(recording.get('state', 'none'), 'ok' if recording.get('complete') else 'warn')),
    ])))
    figure = directory / 'mission.png'
    if figure.exists():
        blocks.append(page.section('Mission map', page.figure(figure, 'evidence, route, track, goals and home at the end')))
    phases = summary.get('phases') or []
    if phases:
        blocks.append(page.section('Phases', page.table(
            ('phase', 'start (s)', 'duration (s)', 'reason'),
            [(p.get('phase'), _num(p.get('start_s'), 2), _num(p.get('duration_s'), 2), p.get('reason')) for p in phases])))
    goals = summary.get('goals') or []
    if goals:
        blocks.append(page.section('Goals', page.table(
            ('set (s)', 'x, y (m)', 'tag', 'revision', 'operator', 'outcome'),
            [(_num(g.get('set_s'), 2), f"{_num(g['xy'][0], 2)}, {_num(g['xy'][1], 2)}", g.get('tag'),
              g.get('revision'), 'yes' if g.get('operator') else 'no', g.get('outcome')) for g in goals])))
    timeline = summary.get('timeline') or []
    if timeline:
        blocks.append(page.section('Timeline', page.table(
            ('time (s)', 'kind', 'what'), [(_num(t.get('time_s'), 2), t.get('kind'), t.get('text')) for t in timeline])))
    blocks.append(page.section('Profile', '<pre>' + html.escape(json.dumps(profile, indent=2, default=str)) + '</pre>'))
    path = directory / 'report.html'
    path.write_text(page.document(f"dmsn mission — {summary.get('mission', '')}",
                                  subtitle=f"session {summary.get('session_id', '')}", blocks=blocks),
                    encoding='utf-8')
    return path
