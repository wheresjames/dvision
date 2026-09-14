"""Offline replay of a dynamic dway flight from its archive. No vehicle endpoint is opened.

Replay shows what the executor consumed at the time -- the route snapshot, the
permitted interval, the commanded target and the published pose -- never
evidence that arrived later. Values are the last recorded sample at or before
the selected time; nothing is interpolated, gaps are reported as gaps, and the
track is broken at epoch changes.

    python3 apps/dway/replay.py reports/<id>/<run>/dway/archive           # Tk viewer
    python3 apps/dway/replay.py ARCHIVE --export out/ --at 12.5 --no-ui    # PNG + JSON of one instant
"""
from __future__ import annotations

import argparse
import bisect
import json
from pathlib import Path
import sys

if __name__ == '__main__':
    sys.path[:0] = [str(Path(__file__).resolve().parents[2]), str(Path(__file__).resolve().parents[1])]

from dcmn.archive import ArchiveReader, IncompleteInput  # noqa: E402
from dcmn.navigation import permitted_points  # noqa: E402

EVENT_KINDS = ('execution.transition', 'execution.hold', 'execution.control', 'execution.lease',
               'execution.health', 'execution.shutdown', 'execution.route')


class ReplayModel:
    def __init__(self, archive_dir, *, window_events=None):
        self.directory = Path(archive_dir)
        self.reader = ArchiveReader(self.directory)
        events = self.reader.events() if window_events is None else window_events
        self.metadata = self.reader.manifest.get('metadata', {})
        stream_hz = float((self.metadata.get('profile') or {}).get('stream_hz') or 10.)
        self.gap_s = 3. / stream_hz
        self.samples = sorted(((e['data']['t_s'], e['data']) for e in events
                               if e['type'] == 'execution.sample' and e['data'].get('t_s') is not None),
                              key=lambda item: item[0])
        self.times = [t for t, _ in self.samples]
        self.notable = [(e['data']['t_s'], e) for e in events
                        if e['type'] in EVENT_KINDS and e['data'].get('t_s') is not None]
        self.notable.sort(key=lambda item: item[0])
        self.notable_times = [t for t, _ in self.notable]
        self.routes = [(t, e) for t, e in self.notable if e['type'] == 'execution.route']
        self.route_times = [t for t, _ in self.routes]
        self.transitions = [(t, e['data']) for t, e in self.notable if e['type'] == 'execution.transition']
        self.gaps = []
        for (ta, a), (tb, b) in zip(self.samples, self.samples[1:]):
            if a.get('epoch') != b.get('epoch'): self.gaps.append(dict(from_s=ta, to_s=tb, reason='epoch change'))
            elif tb - ta > self.gap_s: self.gaps.append(dict(from_s=ta, to_s=tb, reason='missing samples'))
        for error in self.reader.errors:
            self.gaps.append(dict(from_s=None, to_s=None, reason=error))
        self.start = self.times[0] if self.times else 0.
        # A decision recorded after the last sample (the final COMPLETE, say) is still part of the flight.
        self.end = max(self.times[-1:] + self.notable_times[-1:], default=0.)
        self._grid_cache = {}

    # -- lookup, never interpolation -------------------------------------

    def sample_at(self, t):
        index = bisect.bisect_right(self.times, t) - 1
        if index < 0: return None, None
        ts, data = self.samples[index]
        return data, t - ts

    def in_gap(self, t):
        return next((g for g in self.gaps if g['from_s'] is not None and g['from_s'] < t < g['to_s']), None)

    def route_at(self, t):
        index = bisect.bisect_right(self.route_times, t) - 1
        return None if index < 0 else self.routes[index][1]

    def state_at(self, t):
        return self._transition_at(t)[1]

    def _transition_at(self, t):
        found = (None, None)
        for ts, data in self.transitions:
            if ts > t: break
            found = (ts, data)
        return found

    def track_until(self, t):
        lines, current, previous = [], [], None
        for ts, data in self.samples:
            if ts > t: break
            pose = data.get('pose')
            broken = previous is not None and (data.get('epoch') != previous[1].get('epoch') or ts - previous[0] > self.gap_s)
            if pose is None or broken:
                if len(current) > 1: lines.append(current)
                current = []
            if pose is not None: current.append((pose[0], pose[1]))
            previous = (ts, data)
        if len(current) > 1: lines.append(current)
        return lines

    def next_event(self, t):
        index = bisect.bisect_right(self.notable_times, t + 1e-9)
        return None if index >= len(self.notable) else self.notable[index]

    def previous_event(self, t):
        index = bisect.bisect_left(self.notable_times, t - 1e-9) - 1
        return None if index < 0 else self.notable[index]

    def revisions(self):
        seen = {}
        for t, event in self.routes:
            snap = event['data'].get('snapshot') or {}
            key = f"{snap.get('session')}:{snap.get('geometry_revision')}"
            if snap.get('points') is not None: seen.setdefault(key, t)
        return seen

    def grid_at(self, t):
        """The last evidence grid recorded at or before ``t``; None with a reason otherwise."""
        chosen = None
        for ts, event in self.routes:
            if ts > t: break
            if event.get('grids'): chosen = event
        if chosen is None: return None, 'no evidence recorded at or before this time'
        key = chosen['sequence']
        if key not in self._grid_cache:
            try:
                grids = self.reader.reconstruct(chosen)['reconstructed_grids']
                self._grid_cache[key] = (next(iter(grids.values())), None) if grids else (None, 'empty grid set')
            except (IncompleteInput, ValueError, OSError) as exc:
                self._grid_cache[key] = (None, f'evidence unavailable: {exc}')
        return self._grid_cache[key]

    def frame(self, t):
        sample, age = self.sample_at(t)
        route = self.route_at(t)
        snap = (route or {}).get('data', {}).get('snapshot') or {}
        changed_s, transition = self._transition_at(t)
        # The state is the latest recorded decision: a transition newer than the
        # last sample (samples are paced, transitions immediate) wins.
        state = None if sample is None else sample.get('state')
        if transition is not None and (sample is None or changed_s >= sample['t_s']):
            state = transition.get('state')
        return dict(t_s=t, sample=sample, sample_age_s=age, gap=self.in_gap(t),
                    state=state,
                    reason=None if transition is None else transition.get('reason'),
                    hold_kind=None if transition is None else transition.get('hold_kind'),
                    route_revision=snap.get('geometry_revision'),
                    route_key=None if not snap else f"{snap.get('session')}:{snap.get('geometry_revision')}",
                    route_points=snap.get('points') or [],
                    permitted=permitted_points(snap) if (route or {}).get('data', {}).get('source_state') in
                    ('READY', None) else [], source_state=(route or {}).get('data', {}).get('source_state'),
                    source_reason=(route or {}).get('data', {}).get('source_reason'),
                    target=None if sample is None else sample.get('target'),
                    pose=None if sample is None else sample.get('pose'), goal=(snap.get('goal') or {}).get('position'),
                    track=self.track_until(t))

    def overlay(self, t):
        from dcmn.map_pane import Overlay
        f = self.frame(t)
        pose = f['pose']
        track = [pt for line in f['track'] for pt in line][-3000:]
        return Overlay(route=[p[:2] for p in f['route_points']], permitted=[p[:2] for p in f['permitted']],
                       track=track, target=None if f['target'] is None else tuple(f['target'][:2]),
                       vehicle=None if pose is None else (pose[0], pose[1], 0.),
                       goal=None if not f['goal'] else tuple(f['goal'][:2]))

    def export(self, t, out_dir):
        out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
        frame = self.frame(t)
        grid, note = self.grid_at(t)
        (out / f'frame-{t:.3f}.json').write_text(json.dumps(dict(frame, evidence_note=note), indent=2, default=str) + '\n')
        if grid is not None:
            from dcmn.map_pane import snapshot_image
            snapshot_image(grid, sim_now_s=t, overlay=self.overlay(t)).save(out / f'frame-{t:.3f}.png')
        return out


class ReplayWindow:
    def __init__(self, model, root=None):
        import tkinter as tk
        from tkinter import ttk
        from dcmn.map_pane import MapPane
        from dcmn.tktheme import apply_theme
        self.model, self.tk = model, tk
        self.root = root if root is not None else tk.Tk()
        self.root.title(f'dway replay — {model.directory}')
        apply_theme(self.root)
        self.t = model.start
        self.playing = False
        bar = ttk.Frame(self.root); bar.pack(fill='x')
        for text, command in (('⏮ event', self.previous_event), ('Play/Pause', self.toggle), ('event ⏭', self.next_event)):
            ttk.Button(bar, text=text, command=command).pack(side='left', padx=2)
        self.revision = tk.StringVar()
        revisions = list(model.revisions())
        box = ttk.Combobox(bar, textvariable=self.revision, values=revisions, width=40, state='readonly')
        box.pack(side='left', padx=6)
        box.bind('<<ComboboxSelected>>', lambda _e: self.seek(model.revisions()[self.revision.get()]))
        self.scale = tk.Scale(self.root, from_=model.start, to=max(model.end, model.start + 1e-3), resolution=.05,
                              orient='horizontal', command=lambda v: self.seek(float(v), from_scale=True))
        self.scale.pack(fill='x')
        self.text = tk.StringVar()
        ttk.Label(self.root, textvariable=self.text, justify='left', wraplength=900).pack(fill='x')
        self.pane = MapPane(self.root, width=760, height=520, title='Replay (recorded, not live)')
        self.pane.widget.pack(fill='both', expand=True)
        self.seek(self.t)

    def seek(self, t, from_scale=False):
        self.t = max(self.model.start, min(self.model.end, float(t)))
        if not from_scale: self.scale.set(self.t)
        frame = self.model.frame(self.t)
        grid, note = self.model.grid_at(self.t)
        gap = frame['gap']
        age = 'n/a' if frame['sample_age_s'] is None else f"{frame['sample_age_s']:.2f} s"
        self.text.set(f"t={self.t:.2f} s  state {frame['state']} ({frame['hold_kind'] or '-'})  sample age {age}\n"
                      f"reason: {frame['reason']}\nroute revision {frame['route_revision']} ({frame['source_state']}: "
                      f"{frame['source_reason']})  target {frame['target']}\n"
                      f"{'GAP: ' + gap['reason'] if gap else ''}  {note or ''}")
        self.pane.set_grid(grid, sim_now_s=self.t)
        self.pane.state.overlay = self.model.overlay(self.t)
        self.pane.refresh()

    def next_event(self):
        event = self.model.next_event(self.t)
        if event: self.seek(event[0])

    def previous_event(self):
        event = self.model.previous_event(self.t)
        if event: self.seek(event[0])

    def toggle(self):
        self.playing = not self.playing
        if self.playing: self.root.after(100, self._play)

    def _play(self):
        if not self.playing: return
        if self.t >= self.model.end: self.playing = False; return
        self.seek(self.t + .1)
        self.root.after(100, self._play)


def main(argv=None):
    parser = argparse.ArgumentParser(description='replay a dynamic dway flight archive (offline)')
    parser.add_argument('archive')
    parser.add_argument('--export', help='write the frame at --at as JSON (and PNG when evidence exists)')
    parser.add_argument('--at', type=float, default=None, help='data-clock time to export or open at')
    parser.add_argument('--no-ui', action='store_true')
    args = parser.parse_args(argv)
    model = ReplayModel(args.archive)
    t = model.end if args.at is None else args.at
    if args.export:
        print(model.export(t, args.export))
    if args.no_ui: return 0
    window = ReplayWindow(model)
    window.seek(t)
    window.root.mainloop()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
