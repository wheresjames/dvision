"""Offline reports for dynamic dway flights, built only from recorded archives.

Runtime control never reads anything here. ``build_report`` reads one dway
execution archive (and, when present, the sibling dnav archives under the same
report root) and writes derived output next to it:

``summary.json``   versioned metrics, with units, denominators and weighting
``events.jsonl``   every committed event, in order
``samples.csv``    control-cadence vehicle samples
``routes.csv``     each consumed route revision and what was flown on it
``holds.csv``      every stop: requested, confirmed, rejected
``map.png``        proposed, active and permitted routes, flown track, targets, stops
``timeline.png``   speed vs cap, cross-track vs allowance, stopping margin, states
``report.html``    the above, readable offline
``manifest.json``  inputs used and inputs missing

Measurements are attributed to the route revision active when they were taken.
Unavailable values are ``null`` with a reason. Estimated track length comes from
the vehicle's published pose and is not simulator truth.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path
import statistics
import sys

if __name__ == '__main__':
    sys.path[:0] = [str(Path(__file__).resolve().parents[2]), str(Path(__file__).resolve().parents[1])]

from dcmn.archive import ArchiveReader  # noqa: E402
from dcmn import report_html as page  # noqa: E402

#: Long JSON blocks on the dark page: scroll inside their panel instead of widening it.
PRE_STYLE = ('<style>pre{white-space:pre-wrap;word-break:break-word;font-size:12px;margin:0}'
             '.section table td{vertical-align:top}</style>')

REPORT_SCHEMA = 'dvision2.dway-dynamic-report.v1'
OUTCOMES = {'COMPLETE': 'complete', 'CANCELLED': 'cancelled', 'FAILED': 'failed'}

DEFINITIONS = {
    'time': 'seconds of provider (data-clock) time; durations are sums of intervals between consecutive samples '
            'in the same epoch, excluding intervals longer than gap_s',
    'speed_mps': 'horizontal ground speed as published by the vehicle link, m/s; max over samples',
    'cross_track_m': 'distance from the published pose to the ordered active route, m; '
                     'mean is time-weighted over EXECUTING intervals, max over EXECUTING samples',
    'stopping_margin_m': 'remaining permitted distance minus the profile stopping distance while moving, m; '
                         'min over EXECUTING samples; negative means a violated margin',
    'estimated_track_m': 'sum of distances between consecutive published poses in one epoch, m; not truth',
    'target_interval_s': 'data-clock time between consecutive accepted position targets within one '
                         'uninterrupted EXECUTING stretch; deadline miss when above 1.5 stream periods',
    'route_age_s': 'provider time at consumption minus the navigation snapshot time (same provider clock)',
}


def _t(event):
    return (event.get('data') or {}).get('t_s')


def _num(values):
    return [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]


def _stat(values, fn):
    values = _num(values)
    return None if not values else round(fn(values), 6)


def _sibling_dnav(archive_dir):
    root = Path(archive_dir).resolve().parent.parent / 'dnav'
    return sorted(p for p in root.glob('archive*') if (p / 'archive.json').exists()) if root.is_dir() else []


def analyse(archive_dir, *, dnav_archives=None):
    """Everything the report says, as one JSON-able dict (also used by replay and comparison)."""
    archive_dir = Path(archive_dir)
    reader = ArchiveReader(archive_dir)
    validation = reader.validate()
    events = reader.events()
    metadata = reader.manifest.get('metadata', {})
    profile = metadata.get('profile') or {}
    stream_hz = float(profile.get('stream_hz') or 10.)
    gap_s = 3. / stream_hz
    by_type = {}
    for event in events: by_type.setdefault(event['type'], []).append(event)
    samples = [e['data'] for e in by_type.get('execution.sample', []) if e['data'].get('t_s') is not None]
    transitions = [e['data'] for e in by_type.get('execution.transition', [])]
    holds = [e['data'] for e in by_type.get('execution.hold', [])]
    controls = [e['data'] for e in by_type.get('execution.control', [])]
    leases = [e['data'] for e in by_type.get('execution.lease', [])]
    health = [e['data'] for e in by_type.get('execution.health', [])]
    targets = [e['data'] for e in by_type.get('execution.target', [])]
    shutdown = [e['data'] for e in by_type.get('execution.shutdown', [])]
    route_events = [e for e in by_type.get('execution.route', []) if (e['data'].get('snapshot') or {}).get('points') is not None]

    durations, gaps, per_revision = {}, [], {}
    track_m = 0.
    exec_weighted = []
    for a, b in zip(samples, samples[1:]):
        dt = b['t_s'] - a['t_s']
        if a.get('epoch') != b.get('epoch') or dt < 0 or dt > gap_s:
            gaps.append(dict(from_s=a['t_s'], to_s=b['t_s'],
                             reason='epoch change' if a.get('epoch') != b.get('epoch') else 'sample gap'))
            continue
        durations[a['state']] = durations.get(a['state'], 0.) + dt
        if a.get('pose') and b.get('pose'): track_m += math.dist(a['pose'][:2], b['pose'][:2])
        if a['state'] == 'EXECUTING' and a.get('cross_track_m') is not None:
            exec_weighted.append((a['cross_track_m'], dt))
        if a.get('geometry_revision') is not None:
            key = f"{a.get('planner_session')}:{a['geometry_revision']}"
            entry = per_revision.setdefault(key, dict(planner_session=a.get('planner_session'),
                geometry_revision=a['geometry_revision'], first_s=a['t_s'], last_s=b['t_s'], active_s=0.,
                samples=0, speeds=[], cross=[], margins=[], estimated_track_m=0.))
            entry['last_s'] = b['t_s']; entry['active_s'] += dt; entry['samples'] += 1
            entry['speeds'].append(a.get('speed_mps'))
            if a['state'] == 'EXECUTING':
                entry['cross'].append(a.get('cross_track_m')); entry['margins'].append(a.get('stopping_margin_m'))
            if a.get('pose') and b.get('pose'): entry['estimated_track_m'] += math.dist(a['pose'][:2], b['pose'][:2])
    routes = {}
    for event in route_events:
        data = event['data']; snap = data['snapshot']
        key = f"{snap.get('session')}:{snap.get('geometry_revision')}"
        entry = routes.setdefault(key, dict(planner_session=snap.get('session'),
                                            geometry_revision=snap.get('geometry_revision'),
                                            first_consumed_s=data.get('t_s'), points=snap.get('points'),
                                            goal=snap.get('goal'), evidence=snap.get('evidence'),
                                            dispositions=[], permitted_ends=[]))
        disposition = (data.get('disposition') or {}).get('value')
        if disposition and (not entry['dispositions'] or entry['dispositions'][-1] != disposition):
            entry['dispositions'].append(disposition)
        end = (snap.get('clearance') or {}).get('end') if (snap.get('clearance') or {}).get('eligible') else None
        if end and (not entry['permitted_ends'] or entry['permitted_ends'][-1] != end): entry['permitted_ends'].append(end)
    for key, entry in per_revision.items():
        entry.update(max_speed_mps=_stat(entry.pop('speeds'), max), max_cross_track_m=_stat(entry.pop('cross'), max),
                     min_stopping_margin_m=_stat(entry.pop('margins'), min),
                     targets=sum(1 for t in targets if f"{t.get('planner_session')}:{t.get('geometry_revision')}" == key),
                     estimated_track_m=round(entry['estimated_track_m'], 4))
        entry.update({k: v for k, v in routes.get(key, {}).items() if k in ('points', 'dispositions', 'permitted_ends')})

    accepted = [t for t in targets if t.get('accepted')]
    change_times = sorted(_num(tr.get('t_s') for tr in transitions))
    intervals = []
    for a, b in zip(accepted, accepted[1:]):
        if any(a['t_s'] < c <= b['t_s'] for c in change_times): continue
        intervals.append(b['t_s'] - a['t_s'])
    period = 1. / stream_hz
    final_state = transitions[-1]['state'] if transitions else 'WAITING'
    if shutdown and final_state not in OUTCOMES: final_state = shutdown[-1].get('state', final_state)
    last = samples[-1] if samples else {}
    first_t = samples[0]['t_s'] if samples else None
    goal = next((r['goal'] for r in reversed(list(routes.values())) if r.get('goal')), None)
    proposals, proposal_reason = [], None
    dnav_archives = _sibling_dnav(archive_dir) if dnav_archives is None else [Path(p) for p in dnav_archives]
    if not dnav_archives:
        proposal_reason = 'no sibling dnav archive under this report root'
    for directory in dnav_archives:
        try:
            seen = set()
            for event in ArchiveReader(directory).events():
                if event['type'] != 'navigation.snapshot': continue
                snap = event['data'].get('snapshot') or {}
                key = (snap.get('session'), snap.get('geometry_revision'))
                if key in seen or not snap.get('points'): continue
                seen.add(key)
                proposals.append(dict(planner_session=key[0], geometry_revision=key[1], time_s=snap.get('time_s'),
                                      points=snap['points'], eligible=(snap.get('clearance') or {}).get('eligible'),
                                      reason=(snap.get('clearance') or {}).get('reason')))
        except (OSError, ValueError) as exc:
            proposal_reason = f'{directory}: {exc}'
    route_ages = _num(e['data'].get('route_age_s') for e in route_events)
    recording = dict(complete=validation['complete'], clean=validation['clean'], errors=validation['errors'],
                     warnings=validation['warnings'], dropped=reader.manifest.get('dropped', 0),
                     events=len(events), state=reader.manifest.get('state'))
    summary = dict(
        schema=REPORT_SCHEMA, archive=str(archive_dir), metadata=metadata,
        outcome=OUTCOMES.get(final_state, 'incomplete'), final_state=final_state,
        reason=transitions[-1]['reason'] if transitions else None,
        recording_complete=bool(validation['complete']),
        outcome_note=('flight outcome and recording completeness are independent'
                      if not validation['complete'] else None),
        goal=goal, final_pose=last.get('pose'),
        elapsed_s=None if not samples else round(samples[-1]['t_s'] - first_t, 6),
        executing_s=round(durations.get('EXECUTING', 0.), 6), braking_s=round(durations.get('BRAKING', 0.), 6),
        holding_s=round(durations.get('HOLDING', 0.), 6), state_durations_s={k: round(v, 6) for k, v in durations.items()},
        interventions=[dict(action=c.get('action'), origin=c.get('origin'), accepted=c.get('accepted'),
                            reason=c.get('reason'), t_s=c.get('t_s')) for c in controls]
                      + [dict(action='lease-lost', owner=lease.get('owner'), t_s=lease.get('t_s')) for lease in leases
                         if lease.get('event') == 'lost'],
        routes=list(per_revision.values()), consumed_routes=list(routes.values()),
        proposed_routes=proposals if proposals else None, proposed_routes_unavailable=proposal_reason if not proposals else None,
        holds=holds, transitions=transitions, health=health, leases=leases, shutdown=shutdown[-1] if shutdown else None,
        speed=dict(max_mps=_stat((s.get('speed_mps') for s in samples), max), cap_mps=profile.get('speed_mps')),
        tracking=dict(max_m=_stat((s.get('cross_track_m') for s in samples if s['state'] == 'EXECUTING'), max),
                      mean_time_weighted_m=None if not exec_weighted else round(
                          sum(v * dt for v, dt in exec_weighted) / sum(dt for _, dt in exec_weighted), 6),
                      allowance_m=profile.get('tracking_m')),
        stopping=dict(min_margin_m=_stat((s.get('stopping_margin_m') for s in samples if s['state'] == 'EXECUTING'), min),
                      violations=[h for h in holds if h.get('violated_margin_m') is not None and h.get('phase') == 'requested'],
                      stopping_m=profile.get('stopping_m')),
        cadence=dict(targets_sent=len(targets), targets_accepted=len(accepted), stream_hz=stream_hz,
                     mean_interval_s=_stat(intervals, statistics.fmean), max_interval_s=_stat(intervals, max),
                     deadline_misses=sum(1 for i in intervals if i > 1.5 * period),
                     skipped_slots=sum(int(t.get('skipped_slots') or 0) for t in targets)),
        route_age_s=dict(min=_stat(route_ages, min), mean=_stat(route_ages, statistics.fmean), max=_stat(route_ages, max)),
        estimated_track_m=round(track_m, 4), samples=len(samples), gaps=gaps, recording=recording,
        profile=profile, conditions=[e['data'] for e in by_type.get('execution.conditions', [])],
        definitions=DEFINITIONS, units='SI: metres, seconds, metres per second')
    return summary, reader, events


def _write_csv(path, rows, fields):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for row in rows: writer.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()})


def _evidence_grid(reader, events):
    for event in reversed(events):
        if event['type'] in ('execution.route', 'retained_state') and event.get('grids'):
            try:
                grids = reader.reconstruct(event)['reconstructed_grids']
                if grids: return next(iter(grids.values())), None
            except (OSError, ValueError) as exc:
                return None, f'evidence unavailable: {exc}'
    return None, 'no evidence grid was recorded with any consumed route'


def _plot_map(path, summary, reader, events):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from dcmn import theme
    import numpy as np
    figure = Figure(figsize=(9, 6), facecolor=theme.BG)
    FigureCanvasAgg(figure)
    axes = figure.add_subplot(111, facecolor=theme.CANVAS)
    grid, note = _evidence_grid(reader, events)
    if grid is not None:
        g = grid.geometry
        occupancy, _ = grid.layer(0)
        image = np.where(occupancy == 255, .55, 1. - occupancy / 254.)
        axes.imshow(image, cmap='gray', vmin=0, vmax=1, origin='upper', alpha=.55,
                    extent=(g.origin_x_m, g.origin_x_m + g.width * g.cell_m, g.origin_y_m + g.height * g.cell_m, g.origin_y_m))
    else:
        axes.text(.01, .99, note, transform=axes.transAxes, va='top', color=theme.WARN, fontsize=8)
    for proposal in (summary.get('proposed_routes') or [])[-60:]:
        pts = proposal['points']
        axes.plot([p[0] for p in pts], [p[1] for p in pts], ':', color=theme.DIM, linewidth=.8)
    for route in summary['routes']:
        pts = route.get('points') or []
        if pts: axes.plot([p[0] for p in pts], [p[1] for p in pts], '-', color=theme.ROUTE, linewidth=1.4)
    xs, ys = [], []
    for event in events:
        if event['type'] != 'execution.sample': continue
        pose = event['data'].get('pose')
        if pose is None:
            if xs: axes.plot(xs, ys, '-', color=theme.ACCENT, linewidth=1.8); xs, ys = [], []
            continue
        xs.append(pose[0]); ys.append(pose[1])
    if xs: axes.plot(xs, ys, '-', color=theme.ACCENT, linewidth=1.8, label='flown (published pose)')
    # Each distinct permitted interval once: consecutive targets on the same
    # route share it, and one plot call per target made long flights slow to report.
    drawn = set()
    for event in events:
        if event['type'] == 'execution.target' and event['data'].get('accepted'):
            data = event['data']
            permitted = data.get('permitted') or []
            key = (data.get('planner_session'), data.get('geometry_revision'), tuple(data.get('end') or ()))
            if permitted and key not in drawn:
                drawn.add(key)
                axes.plot([p[0] for p in permitted], [p[1] for p in permitted], '-', color=theme.GOAL,
                          linewidth=3, alpha=.15)
    targets = [e['data']['target'] for e in events if e['type'] == 'execution.target']
    if targets: axes.plot([t[0] for t in targets], [t[1] for t in targets], 'x', color=theme.DANGER, markersize=5, label='commanded targets')
    for hold in summary['holds']:
        if hold.get('phase') == 'confirmed' and hold.get('pose'):
            axes.plot(hold['pose'][0], hold['pose'][1], 's', color=theme.WARN, markersize=6)
            axes.annotate(hold.get('kind') or '', hold['pose'][:2], fontsize=7, color=theme.WARN, xytext=(4, 4),
                          textcoords='offset points')
    goal = (summary.get('goal') or {}).get('position')
    if goal: axes.plot(goal[0], goal[1], '*', color=theme.GOAL, markersize=14, label='goal')
    axes.set_aspect('equal'); axes.invert_yaxis() if not axes.yaxis_inverted() else None
    axes.set_xlabel('x east (m)', color=theme.DIM); axes.set_ylabel('y south (m)', color=theme.DIM)
    axes.tick_params(colors=theme.DIM)
    axes.set_title(f"{summary['outcome']} — {summary.get('reason') or ''}"[:110], color=theme.TEXT, fontsize=9)
    axes.legend(loc='lower right', fontsize=7, facecolor=theme.BUTTON, labelcolor=theme.TEXT)
    figure.savefig(path, dpi=120, bbox_inches='tight', facecolor=theme.BG)


def _state_runs(samples, gap_s):
    """(state, start, end) for each uninterrupted stretch of one state: one shaded span per
    stretch instead of one per sample pair, which grew with flight length."""
    runs = []
    for a, b in zip(samples, samples[1:]):
        if b['t_s'] - a['t_s'] > gap_s or b['t_s'] < a['t_s']:
            continue
        if runs and runs[-1][0] == a['state'] and abs(runs[-1][2] - a['t_s']) < 1e-9:
            runs[-1][2] = b['t_s']
        else:
            runs.append([a['state'], a['t_s'], b['t_s']])
    return [tuple(run) for run in runs]


def _plot_timeline(path, summary, events):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from dcmn import theme
    figure = Figure(figsize=(9, 6.5), facecolor=theme.BG)
    FigureCanvasAgg(figure)
    samples = [e['data'] for e in events if e['type'] == 'execution.sample' and e['data'].get('t_s') is not None]
    t = [s['t_s'] for s in samples]
    profile = summary['profile']
    panels = (('speed m/s', 'speed_mps', profile.get('speed_mps')),
              ('cross-track m', 'cross_track_m', profile.get('tracking_m')),
              ('stopping margin m', 'stopping_margin_m', 0.))
    colors = {'EXECUTING': theme.OK, 'BRAKING': theme.WARN, 'HOLDING': theme.DIM, 'FAILED': theme.DANGER}
    for index, (label, key, limit) in enumerate(panels):
        axes = figure.add_subplot(3, 1, index + 1, facecolor=theme.CANVAS)
        axes.plot(t, [s.get(key) if s.get(key) is not None else math.nan for s in samples], color=theme.ACCENT, linewidth=1.2)
        if limit is not None: axes.axhline(limit, color=theme.DANGER, linestyle='--', linewidth=.8)
        for state, start, end in _state_runs(samples, 3. / float(profile.get('stream_hz') or 10.)):
            if state in colors:
                axes.axvspan(start, end, color=colors[state], alpha=.08, linewidth=0)
        for gap in summary['gaps']:
            axes.axvspan(gap['from_s'], gap['to_s'], color=theme.DANGER, alpha=.2, linewidth=0)
        for hold in summary['holds']:
            if hold.get('phase') == 'requested' and hold.get('t_s') is not None:
                axes.axvline(hold['t_s'], color=theme.WARN, linewidth=.6)
        axes.set_ylabel(label, color=theme.DIM, fontsize=8); axes.tick_params(colors=theme.DIM, labelsize=7)
    figure.axes[-1].set_xlabel('provider time (s)', color=theme.DIM)
    figure.savefig(path, dpi=120, bbox_inches='tight', facecolor=theme.BG)


def _cell(value):
    return html.escape(json.dumps(value) if isinstance(value, (list, dict)) else str(value))


def _table(rows, fields):
    if not rows: return '<p class="muted">none</p>'
    return page.table(fields, ([_cell(r.get(f)) for f in fields] for r in rows))


def _pre(value):
    return '<pre>' + html.escape(json.dumps(value, indent=2, default=str)) + '</pre>'


def build_report(archive_dir, out_dir=None, *, dnav_archives=None):
    """Write every derived artefact for one execution archive. Returns the output directory."""
    archive_dir = Path(archive_dir)
    out = Path(out_dir) if out_dir is not None else archive_dir / 'report'
    out.mkdir(parents=True, exist_ok=True)
    summary, reader, events = analyse(archive_dir, dnav_archives=dnav_archives)
    missing = []
    temp = out / 'summary.json.tmp'
    temp.write_text(json.dumps(summary, indent=2, allow_nan=False, default=str) + '\n')
    temp.replace(out / 'summary.json')
    with (out / 'events.jsonl').open('w') as handle:
        for event in events:
            handle.write(json.dumps(dict(sequence=event['sequence'], type=event['type'], data=event['data']),
                                    default=str) + '\n')
    samples = [e['data'] for e in events if e['type'] == 'execution.sample']
    _write_csv(out / 'samples.csv', samples, ('t_s', 'state', 'hold_kind', 'pose', 'speed_mps', 'speed_cap_mps',
               'cross_track_m', 'tracking_allowance_m', 'remaining_permitted_m', 'stopping_margin_m',
               'validity_remaining_s', 'planner_session', 'geometry_revision', 'segment', 'progress', 'target',
               'heading_deg', 'commanded_heading_deg', 'turning', 'mode', 'owns_control', 'epoch'))
    _write_csv(out / 'routes.csv', summary['routes'], ('planner_session', 'geometry_revision', 'first_s', 'last_s',
               'active_s', 'samples', 'targets', 'max_speed_mps', 'max_cross_track_m', 'min_stopping_margin_m',
               'estimated_track_m', 'dispositions', 'permitted_ends', 'points'))
    _write_csv(out / 'holds.csv', summary['holds'], ('t_s', 'phase', 'kind', 'reason', 'accepted', 'result', 'pose',
               'progress', 'speed_mps', 'stop_generation', 'violated_margin_m', 'confirm_s'))
    images = {}
    for name, plot in (('map.png', lambda p: _plot_map(p, summary, reader, events)),
                       ('timeline.png', lambda p: _plot_timeline(p, summary, events))):
        try:
            plot(out / name); images[name] = True
        except ImportError as exc:
            missing.append(dict(item=name, reason=f'matplotlib unavailable: {exc}'))
    grade = {'complete': 'ok', 'cancelled': 'warn', 'failed': 'bad'}.get(summary['outcome'], 'unknown')
    recording = ('complete' if summary['recording_complete'] else 'INCOMPLETE')
    blocks = [
        PRE_STYLE,
        page.section('Outcome', page.facts([
            ('Outcome', page.graded(summary['outcome'], grade)),
            ('Recording', page.graded(f"{recording} ({summary['recording']['state']}, dropped "
                                      f"{summary['recording']['dropped']})",
                                      'ok' if summary['recording_complete'] else 'bad')),
            ('Goal', page.esc(json.dumps((summary.get('goal') or {}).get('position')))),
            ('Final pose', page.esc(json.dumps(summary.get('final_pose')))),
            ('Elapsed s', page.esc(summary['elapsed_s'])),
            ('Executing / braking / holding s',
             page.esc(f"{summary['executing_s']} / {summary['braking_s']} / {summary['holding_s']}")),
            ('Estimated track m', page.esc(summary['estimated_track_m'])),
        ]), f"<p>{page.esc(summary.get('reason'))}</p>",
            f"<p class=\"muted\">Profile <code>{page.esc(summary['metadata'].get('profile_digest'))}</code>. "
            "Estimated track length uses published poses and is not simulator truth.</p>"),
        page.section('Map', page.figure(out / 'map.png', 'evidence, proposed/active/permitted routes, '
                                                         'flown track, commanded targets and stops')
                     if images.get('map.png') else '<p class="muted">map unavailable</p>'),
        page.section('Timeline', page.figure(out / 'timeline.png', 'speed vs cap, cross-track vs allowance, '
                                                                   'stopping margin, state bands and gaps')
                     if images.get('timeline.png') else '<p class="muted">timeline unavailable</p>'),
        page.section('Routes flown', _table(summary['routes'], ('planner_session', 'geometry_revision', 'active_s',
            'samples', 'targets', 'max_speed_mps', 'max_cross_track_m', 'min_stopping_margin_m', 'estimated_track_m',
            'dispositions'))),
        page.section('Stops', _table(summary['holds'], ('t_s', 'phase', 'kind', 'reason', 'violated_margin_m',
                                                        'confirm_s'))),
        page.section('Interventions', _table(summary['interventions'], ('t_s', 'action', 'origin', 'accepted',
                                                                        'reason'))),
        page.section('Transitions', _table(summary['transitions'], ('t_s', 'previous', 'state', 'hold_kind',
                                                                    'reason'))),
        page.section('Cadence, tracking and stopping', _pre(dict(
            cadence=summary['cadence'], speed=summary['speed'], tracking=summary['tracking'],
            stopping=dict(summary['stopping'], violations=len(summary['stopping']['violations'])),
            route_age_s=summary['route_age_s'], gaps=summary['gaps']))),
        page.section('Proposed routes', '<p>' + page.esc(
            f"{len(summary['proposed_routes'])} dnav revisions" if summary['proposed_routes']
            else f"unavailable: {summary['proposed_routes_unavailable']}") + '</p>'),
        page.section('Recording completeness', _pre(summary['recording'])),
        page.section('Definitions', _pre(summary['definitions'])),
        page.section('Configuration', _pre(dict(profile=summary['profile'], conditions=summary['conditions']))),
    ]
    (out / 'report.html').write_text(page.document(
        f"dway dynamic flight — {summary['outcome']}",
        subtitle=str(summary.get('archive', '')), blocks=blocks))
    inputs = [dict(kind='dway archive', path=str(archive_dir), complete=summary['recording_complete'])]
    for directory in (_sibling_dnav(archive_dir) if dnav_archives is None else dnav_archives):
        inputs.append(dict(kind='dnav archive', path=str(directory)))
    if not summary['proposed_routes']:
        missing.append(dict(item='proposed routes', reason=summary['proposed_routes_unavailable']))
    grid, note = _evidence_grid(reader, events)
    if grid is None: missing.append(dict(item='evidence background', reason=note))
    (out / 'manifest.json').write_text(json.dumps(dict(schema='dvision2.dway-report-manifest.v1', inputs=inputs,
        missing=missing, outputs=sorted(p.name for p in out.iterdir() if p.is_file())), indent=2) + '\n')
    return out


COMPARISON_FIELDS = ('run', 'outcome', 'reason', 'recording_complete', 'elapsed_s', 'executing_s', 'holding_s',
                     'stops', 'interventions', 'max_speed_mps', 'max_cross_track_m', 'min_stopping_margin_m',
                     'margin_violations', 'targets', 'deadline_misses', 'samples', 'profile')


def compare_runs(summaries, out_dir):
    """A plain table over repeated runs. Every outcome is kept, failures included."""
    rows = []
    for item in summaries:
        path = Path(item) if not isinstance(item, dict) else None
        summary = item if isinstance(item, dict) else json.loads(
            (path / 'summary.json' if path.is_dir() else path).read_text())
        rows.append(dict(run=str(path or summary.get('archive')), outcome=summary['outcome'], reason=summary.get('reason'),
                         recording_complete=summary['recording_complete'], elapsed_s=summary['elapsed_s'],
                         executing_s=summary['executing_s'], holding_s=summary['holding_s'],
                         stops=sum(1 for h in summary['holds'] if h.get('phase') == 'confirmed'),
                         interventions=len(summary['interventions']), max_speed_mps=summary['speed']['max_mps'],
                         max_cross_track_m=summary['tracking']['max_m'],
                         min_stopping_margin_m=summary['stopping']['min_margin_m'],
                         margin_violations=len(summary['stopping']['violations']),
                         targets=summary['cadence']['targets_sent'], deadline_misses=summary['cadence']['deadline_misses'],
                         samples=summary['samples'], profile=summary['metadata'].get('profile_digest')))
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / 'comparison.csv', rows, COMPARISON_FIELDS)
    counts = {}
    for row in rows: counts[row['outcome']] = counts.get(row['outcome'], 0) + 1
    (out / 'comparison.json').write_text(json.dumps(dict(schema='dvision2.dway-comparison.v1', runs=len(rows),
        outcomes=counts, rows=rows, note='all runs included; raw planner cost is not compared'), indent=2) + '\n')
    (out / 'comparison.html').write_text(page.document(
        'dway run comparison', subtitle=f'{len(rows)} runs: {json.dumps(counts)}; all outcomes included',
        blocks=[PRE_STYLE, page.section('Runs', _table(rows, COMPARISON_FIELDS))]))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description='dway dynamic flight reports (offline)')
    sub = parser.add_subparsers(dest='command', required=True)
    report = sub.add_parser('report', help='build a report from one dway execution archive')
    report.add_argument('archive'); report.add_argument('--out')
    compare = sub.add_parser('compare', help='compare report directories or summary.json files')
    compare.add_argument('runs', nargs='+'); compare.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    if args.command == 'report':
        print(build_report(args.archive, args.out))
    else:
        rows = compare_runs(args.runs, args.out)
        print(f'{len(rows)} runs -> {args.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
