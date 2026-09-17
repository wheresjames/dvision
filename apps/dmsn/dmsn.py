#!/usr/bin/env python3
"""dmsn: the mission coordinator's window, and the headless run behind it.

dmsn decides where the vehicle goes next and when the job is finished. It
holds no control lease: it sets the session goal as the ``mission`` authority,
publishes intents dway may refuse, and reports. dnav plans and dway flies.

Exit codes: 0 the mission completed; 1 it failed, was aborted, spent a budget or
did not finish before ``--timeout``; 2 a usage or startup problem -- a bad
profile, a second dmsn on the same vehicle, or an arena mismatch refused.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
APPS = ROOT / 'apps'
for _path in (str(ROOT), str(APPS)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from dcmn.pacing import MAP_HZ  # noqa: E402
from dvision2_common import validate_id  # noqa: E402


def parse_args(argv):
    import argparse
    from dalg.dalg import join_negative_values
    argv = join_negative_values(argv, ('--goal',))
    parser = argparse.ArgumentParser(description='dvision2 mission coordinator')
    parser.add_argument('--id', help='vehicle instance id (not needed with --validate)')
    parser.add_argument('--profile', help='mission profile name or path (user directory first, then examples)')
    parser.add_argument('--mission', help='start from a new profile for this mission type')
    parser.add_argument('--goal', help='x,y in local metres; shorthand for --set goal=x,y (and goal_source=xy)')
    parser.add_argument('--set', action='append', default=[], metavar='KEY=VALUE', dest='overrides',
                        help='override one profile field, e.g. budgets.time_s=600 or goal_source=provider_target '
                             '(repeatable; recorded in the report)')
    parser.add_argument('--start', action='store_true', help='start the mission once the session provider is up')
    parser.add_argument('--allow-arena-mismatch', action='store_true',
                        help='headless: fly even when the profile arena differs from the running map')
    parser.add_argument('--planner', default='dnav', help='selected navigation publisher name')
    parser.add_argument('--show-reference', action='store_true',
                        help='open with the reference-image background enabled (recorded)')
    parser.add_argument('--hide-reference', action='store_true',
                        help='open with the reference-image background disabled, whatever the mission default')
    parser.add_argument('--no-ui', action='store_true', help='run headless and print each change')
    parser.add_argument('--timeout', type=float, default=0.0,
                        help='stop after this many wall seconds; 0 runs until the mission ends or is closed')
    parser.add_argument('--detector-wait-s', type=float, default=30.0,
                        help='headless --start of a mission with a target: wall seconds to wait for a detector '
                             'publishing its label before refusing (default: 30)')
    parser.add_argument('--validate', metavar='PROFILE', help='validate a mission profile, print its digest and exit')
    parser.add_argument('--import-tour', metavar='TOUR',
                        help='convert a tour JSON into a waypoints mission profile and exit (needs --out)')
    parser.add_argument('--out', metavar='PROFILE', help='where --import-tour writes its profile')
    args = parser.parse_args(argv)
    if args.validate is None and args.import_tour is None:
        if not args.id: parser.error('--id is required unless --validate or --import-tour is used')
        validate_id(args.id)
    if args.import_tour is not None and not args.out:
        parser.error('--import-tour needs --out PROFILE')
    if args.profile and args.mission: parser.error('--profile and --mission are exclusive')
    if args.timeout < 0: parser.error('--timeout may not be negative')
    if args.detector_wait_s < 0: parser.error('--detector-wait-s may not be negative')
    if args.show_reference and args.hide_reference: parser.error('--show-reference and --hide-reference are exclusive')
    return args


def build_profile(args):
    """The profile to open, with the command line's overrides applied and listed."""
    from dmsn import profiles as P
    if args.profile:
        profile, path = P.load(args.profile)
    else:
        profile, path = P.blank(args.mission or 'goto', name=f'{args.mission or "goto"}-cli'), None
    overrides = list(args.overrides)
    if args.goal:
        overrides.append(f'goal={args.goal}')
        if profile['params'].get('goal_source') == 'click' and not any(o.startswith('goal_source=') for o in overrides):
            overrides.insert(0, 'goal_source=xy')
    return P.apply_overrides(profile, overrides), path, overrides


def validate_command(target) -> int:
    from dmsn import profiles as P
    try:
        profile, path = P.load(target)
    except ValueError as exc:
        print(f'dmsn: invalid: {exc}', file=sys.stderr)
        return 2
    print(f'{path}: valid {profile["mission"]} profile {profile["name"]!r}, digest {P.digest(profile)}')
    return 0


def import_tour_command(tour, out) -> int:
    """``--import-tour``: convert and exit; the window's Import runs this as a child process (B9)."""
    from dmsn import profiles as P
    from dmsn.tour_import import convert_file
    try:
        path, profile = convert_file(tour, out)
    except ValueError as exc:
        print(f'dmsn: tour import failed: {exc}', file=sys.stderr)
        return 2
    print(f'{path}: {len(profile["waypoints"])} waypoints from {tour} '
          f'({profile["name"]!r}, digest {P.digest(profile)})')
    return 0


def wait_for_provider(run, deadline) -> bool:
    while not run.snapshot:
        run.step()
        if run.shutdown_requested or (deadline is not None and time.monotonic() >= deadline): return False
        time.sleep(0.05)
    # One provider status poll, so the arena check has the running map's name if there is one.
    run.provider = run.ports.provider_status()
    return True


def run_headless(run, args) -> int:
    from dmsn.runner import StartRefused
    deadline = None if args.timeout <= 0 else time.monotonic() + args.timeout
    if args.start:
        if not wait_for_provider(run, deadline):
            run.reason = 'no session provider before the timeout'
            print('dmsn: no session provider appeared; not started', flush=True)
            return 1
        detector_deadline = time.monotonic() + max(0.0, args.detector_wait_s)
        waiting = False
        while True:
            try:
                run.start()
                break
            except StartRefused as exc:
                # A chain launched together races dalg's first publication: a missing
                # detector is waited for (bounded); every other refusal is final.
                late = 'no detector publishing' in str(exc) and time.monotonic() < detector_deadline
                if late and (deadline is None or time.monotonic() < deadline) and not run.shutdown_requested:
                    if not waiting:
                        print(f'dmsn: waiting up to {args.detector_wait_s:g} s: {exc}', flush=True)
                        waiting = True
                    run.step()
                    time.sleep(0.25)
                    continue
                run.reason = f'start refused: {exc}'
                print(f'dmsn: start refused: {exc}', file=sys.stderr, flush=True)
                return 2
        print(f"[sim {run.sim_time_s():8.2f}] started {run.active_profile['mission']} "
              f"({run.active_profile['name']}, digest {run.document.digest[:12]})", flush=True)
    last = None
    try:
        while not run.shutdown_requested:
            run.step()
            goal = run.goal_sent
            mark = (run.phase, run.outcome, None if goal is None else (tuple(goal['xy']), goal['revision']),
                    run.blocking)
            if mark != last:
                if last is None or mark[0] != last[0] or mark[1] != last[1]:
                    print(f"[sim {run.sim_time_s():8.2f}] {run.phase}"
                          f"{': ' + run.outcome if run.outcome else ''}: {run.reason}", flush=True)
                if goal is not None and (last is None or mark[2] != last[2]):
                    print(f"[sim {run.sim_time_s():8.2f}] goal {goal['tag']} {goal['xy'][0]:.2f}, "
                          f"{goal['xy'][1]:.2f} m (revision {goal['revision']})", flush=True)
                if run.blocking and (last is None or mark[3] != last[3]):
                    print(f'[sim {run.sim_time_s():8.2f}]   {run.blocking}', flush=True)
                last = mark
            if run.phase == 'done' and args.start: break
            if deadline is not None and time.monotonic() >= deadline:
                run.reason = 'dmsn timeout'
                run.partial = True
                break
            time.sleep(run.poll_delay())
    except KeyboardInterrupt:
        run.reason = 'interrupted'
        run.partial = True
    return run.exit_code() if args.start else 0


def _run_window(run, window, args) -> int:
    from dmsn.runner import StartRefused
    deadline = None if args.timeout <= 0 else time.monotonic() + args.timeout
    pending_start = args.start
    while window.running:
        run.step()
        if pending_start and run.snapshot:
            pending_start = False
            try: window.mission.notice.set(run.start())
            except StartRefused as exc: window.mission.notice.set(f'start refused: {exc}')
        window.update()
        if run.shutdown_requested:
            window.close()
        if deadline is not None and time.monotonic() >= deadline:
            run.reason = 'dmsn timeout'
            run.partial = True
            break
        time.sleep(min(run.poll_delay(), 1.0 / MAP_HZ / 2))
    return run.exit_code() if run.phase == 'done' else 0


def main(argv=None) -> int:
    from dcmn.window import disable_input_method
    disable_input_method()
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.validate is not None: return validate_command(args.validate)
    if args.import_tour is not None: return import_tour_command(args.import_tour, args.out)
    try:
        profile, path, overrides = build_profile(args)
    except ValueError as exc:
        print(f'dmsn: {exc}', file=sys.stderr)
        return 2
    from dmsn.runner import MissionRun
    try:
        run = MissionRun(args.id, profile if args.profile or args.mission or overrides else None,
                         profile_path=path, overrides=overrides, headless=args.no_ui,
                         allow_arena_mismatch=args.allow_arena_mismatch, planner=args.planner)
    except (ValueError, RuntimeError, OSError) as exc:
        print(f'dmsn: cannot start: {exc}' + ('; another dmsn is already coordinating this vehicle'
                                               if 'live writer' in str(exc) else ''), file=sys.stderr)
        return 2
    window = None
    if not args.no_ui:
        try:
            import tkinter as tk
            from dmsn.ui.window import Window
            reference = True if args.show_reference else False if args.hide_reference else None
            window = Window(run, show_reference=reference)
        except ImportError as exc:
            print(f'dmsn: tkinter is unavailable: {exc}', file=sys.stderr)
            run.close(partial=True)
            return 2
        except tk.TclError as exc:
            print(f'dmsn: cannot open a window: {exc}', file=sys.stderr)
            run.close(partial=True)
            return 2
    status = 1
    try:
        status = run_headless(run, args) if window is None else _run_window(run, window, args)
    except Exception as exc:
        run.partial = True
        run.reason = f'dmsn crashed: {type(exc).__name__}: {exc}'
        raise
    finally:
        directory = run.close(partial=run.partial)
        if directory is not None: print(f'dmsn: report -> {directory}', file=sys.stderr)
        if window is not None:
            try: window.save_geometry(); window.root.destroy()
            except Exception: pass
    return status


if __name__ == '__main__':
    raise SystemExit(main())
