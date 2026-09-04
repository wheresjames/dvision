#!/usr/bin/env python3
"""Automated flight test runner for daic.

Launches dsim (headless) and daic (headless, AI enabled), waits for the
mission to complete or time out, then prints a structured report.

Usage
-----
    python3 tests/flight_test.py [options]

    # Quick run with defaults (maze_001, 90 s budget):
    python3 tests/flight_test.py

    # Custom map and duration:
    python3 tests/flight_test.py --map assets/maps/maze_002.txt --duration 120

    # Keep the log file for later inspection:
    python3 tests/flight_test.py --log /tmp/flight.jsonl

Exit codes
----------
    0  drone reached COMPLETE (landed on target)
    1  mission timed out, failed, or errored
"""

from __future__ import annotations

import argparse
import html
import io
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from daic.flight_log import analyze_log, diagnose_log, print_diagnosis, print_report
from daic.run_reporter import _generate_html_report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wait_for_file(path: Path, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists() and path.stat().st_size > 0:
            return True
        time.sleep(0.2)
    return False


def _resolve_report_dir(path: str | None) -> Path | None:
    if not path:
        return None
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    return p


def _git_value(*args: str) -> str | None:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def run_test(map_file: str, duration_s: int, log_path: Path | None,
             fps: int, verbose: bool,
             report_dir: Path | None = None,
             instance_id: str = "flighttest") -> dict:
    frames = duration_s * fps
    dsim_cmd = [
        sys.executable, str(ROOT / "apps/dsim" / "dsim.py"),
        "--id",    instance_id,
        "--map",   map_file,
        "--no-ui",
        "--fps",   str(fps),
        "--frames", str(frames),
    ]
    if report_dir is not None:
        report_dir.mkdir(parents=True, exist_ok=True)
        dsim_cmd.extend(["--report-dir", str(report_dir)])
    daic_cmd = [
        sys.executable, str(ROOT / "apps/daic" / "daic.py"),
        "--id",        instance_id,
        "--no-ui",
        "--enable-ai",
        "--fps",       str(fps),
    ]
    if log_path is not None:
        daic_cmd.extend(["--log-file", str(log_path)])
    if verbose:
        daic_cmd.append("--verbose")

    if verbose:
        print(f"dsim: {' '.join(dsim_cmd)}", file=sys.stderr)
        print(f"daic: {' '.join(daic_cmd)}", file=sys.stderr)

    # Launch dsim first so shared memory exists before daic connects.
    dsim_started = time.monotonic()
    dsim_proc = subprocess.Popen(
        dsim_cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE if not verbose else None,
        cwd=str(ROOT),
    )

    # Give dsim a moment to create the shared-memory buffers.
    time.sleep(1.5)

    daic_proc = subprocess.Popen(
        daic_cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE if not verbose else None,
        cwd=str(ROOT),
    )

    # Wait for the log file to appear (daic connected and started logging).
    wait_log = log_path
    if wait_log is None and report_dir is not None:
        wait_log = report_dir / "daic" / "flight.jsonl"
    if wait_log is not None and not _wait_for_file(wait_log, timeout=15.0):
        print("WARNING: log file did not appear within 15 s", file=sys.stderr)

    # Stop DAIC early enough for its final zero command to decelerate the
    # vehicle while dsim is still alive. Previously dsim outlived DAIC logging
    # by ~3 s with the last non-zero setpoint latched, so the simulator crashed
    # after DAIC had already written `crashed:false`.
    shutdown_grace_s = min(3.0, max(1.0, duration_s * 0.1))
    control_deadline = dsim_started + duration_s - shutdown_grace_s
    while (time.monotonic() < control_deadline
           and dsim_proc.poll() is None and daic_proc.poll() is None):
        time.sleep(0.1)
    if daic_proc.poll() is None:
        daic_proc.terminate()  # signal handler sends zero before closing IPC
        try:
            daic_proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            daic_proc.kill()
            daic_proc.wait()

    # Let dsim integrate the stop and produce the authoritative final summary.
    deadline = shutdown_grace_s + 10
    try:
        dsim_proc.wait(timeout=deadline)
    except subprocess.TimeoutExpired:
        print("WARNING: dsim did not exit within budget, killing", file=sys.stderr)
        dsim_proc.kill()
        dsim_proc.wait()

    if daic_proc.poll() is None:
        daic_proc.terminate()
        daic_proc.wait(timeout=5)

    if report_dir is not None:
        try:
            _generate_html_report(report_dir / "daic")
        except Exception as exc:
            print(f"WARNING: final report reconciliation failed: {exc}",
                  file=sys.stderr)

    return {
        "dsim_cmd": dsim_cmd,
        "daic_cmd": daic_cmd,
        "dsim_returncode": dsim_proc.returncode,
        "daic_returncode": daic_proc.returncode,
    }


def _capture_vision_debug(log_path: Path) -> str:
    proc = subprocess.run(
        [sys.executable, str(ROOT / "tests" / "vision_debug_report.py"), str(log_path)],
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    return proc.stdout


def _selected_daic_path_images(report_dir: Path) -> list[tuple[str, str, str]]:
    """Return representative DAIC occupancy/A* path snapshots for reports."""
    occ_paths = sorted((report_dir / "daic").glob("occ_*.png"))
    if not occ_paths:
        return []

    picks: list[tuple[str, Path]] = []
    picks.append(("Initial local route", occ_paths[0]))
    if len(occ_paths) >= 3:
        picks.append(("Mid-run local route", occ_paths[len(occ_paths) // 2]))
    if len(occ_paths) >= 2:
        picks.append(("Final local route", occ_paths[-1]))

    seen: set[Path] = set()
    out: list[tuple[str, str, str]] = []
    explanations = {
        "Initial local route": (
            "First DAIC occupancy-map snapshot. It shows the early local A* "
            "route before much obstacle evidence has accumulated."
        ),
        "Mid-run local route": (
            "Middle DAIC occupancy-map snapshot. Use this to see whether "
            "detected obstacles are changing the route around the wall."
        ),
        "Final local route": (
            "Last DAIC occupancy-map snapshot. This shows the route and "
            "occupancy state near landing, timeout, or crash."
        ),
    }
    for title, path in picks:
        if path in seen:
            continue
        seen.add(path)
        rel = path.relative_to(report_dir).as_posix()
        out.append((title, rel, explanations[title]))
    return out


def _daic_occ_images(report_dir: Path) -> list[tuple[str, str]]:
    """Return all DAIC occupancy-map snapshots as (label, relative path)."""
    out: list[tuple[str, str]] = []
    for idx, path in enumerate(sorted((report_dir / "daic").glob("occ_*.png"))):
        label = f"Occupancy snapshot {idx + 1}"
        out.append((label, path.relative_to(report_dir).as_posix()))
    return out


def _diagnosis_md_lines(diagnosis: dict) -> list[str]:
    source_lines = []
    ages = diagnosis.get("front_occ_close_source_age") or {}
    for key, count in sorted((diagnosis.get("front_occ_close_sources") or {}).items()):
        age = ages.get(key) or {}
        suffix = ""
        if age:
            suffix = f" (age median={age.get('median')}, max={age.get('max')} ticks)"
        source_lines.append(f"- {key}: {count}{suffix}")

    lines = [
        "## Diagnosis",
        "",
        f"- Target-visible ticks: {diagnosis.get('target_visible_ticks')} "
        f"({diagnosis.get('target_visible_rate')})",
        f"- Approach-gated ticks: {diagnosis.get('approach_gated_ticks')}",
        f"- Yaw-scan ticks (local route unavailable): {diagnosis.get('yaw_scan_ticks')}",
        f"- Local-route active ticks: {diagnosis.get('route_active_ticks')}",
        f"- Forward-zero ticks while aligning to a waypoint: "
        f"{diagnosis.get('forward_zero_alignment_ticks')}",
        f"- Avoidance-clamped ticks: {diagnosis.get('avoidance_clamped_ticks')}",
        f"- Route-active stalled ticks (forward~zero or avoidance-clamped, union): "
        f"{diagnosis.get('route_stalled_ticks')}",
        f"- Ticks with front_occ_m <= {diagnosis.get('front_occ_close_threshold_m')} m: "
        f"{diagnosis.get('front_occ_close_ticks')}",
        f"- Ticks with front_block_occ_m <= {diagnosis.get('front_occ_close_threshold_m')} m: "
        f"{diagnosis.get('front_block_occ_close_ticks')}",
        f"- Longest continuous front-occupancy-close run: "
        f"{diagnosis.get('longest_front_occ_close_run_s')} s",
        *(["", "Close-front occupancy sources:", "", *source_lines]
          if source_lines else []),
        f"- A* route changes: {diagnosis.get('route_changes')}",
        f"- Straight-path / detour-path ticks: {diagnosis.get('straight_path_ticks')} / "
        f"{diagnosis.get('detour_path_ticks')}",
        "",
        "Classification hints (confirm against the occupancy snapshot gallery below "
        "— heuristic pointers, not a verdict):",
        "",
    ]
    for hint in diagnosis.get("classification_hints", []):
        lines.append(f"- **{hint.get('label')}**: {hint.get('evidence')}")
    lines.append("")
    return lines


def _write_report_md(report_dir: Path, summary: dict, metadata: dict,
                     vision_debug: str, diagnosis: dict) -> None:
    result = "LANDED" if summary.get("landed") else "DID NOT LAND"
    lines = [
        f"# DAIC Benchmark - {report_dir.name}",
        "",
    ]
    if summary.get("frame_drop_suspected"):
        lines.extend([
            f"> **WARNING: dropped frames suspected.** Effective fps "
            f"({summary.get('effective_fps')}) was only "
            f"{summary['frame_drop_ratio']:.0%} of the configured "
            f"{summary.get('target_fps')} fps -- the test host was likely "
            f"overloaded (e.g. too many concurrent benchmark runs) and this "
            f"run's results may be unreliable.",
            "",
        ])
    lines.extend([
        f"**Result:** {result}",
        f"**Final state:** {summary.get('final_state', 'unknown')}",
        f"**Map:** `{metadata.get('map', '')}`",
        f"**Duration budget:** {metadata.get('duration_s')} s @ {metadata.get('fps')} fps",
        f"**Git commit:** `{metadata.get('git_commit') or 'unknown'}`",
        f"**Dirty worktree:** {metadata.get('git_dirty')}",
        "",
        "## Summary",
        "",
        f"- Ticks logged: {summary.get('tick_count')}",
        f"- Effective fps: {summary.get('effective_fps')}"
        + (f" ({summary['frame_drop_ratio']:.0%} of {summary.get('target_fps')} target"
           f"{' -- DROPPED FRAMES SUSPECTED' if summary.get('frame_drop_suspected') else ''})"
           if summary.get("frame_drop_ratio") is not None else ""),
        f"- Detection rate: {summary.get('detection_rate')}",
        f"- Landing error m: {summary.get('landing_error_m')}",
        f"- Closest pass m: {summary.get('min_dist_to_target_m')}",
        f"- Peak approach forward mps: {summary.get('peak_approach_fwd_mps')}",
        f"- Final position: `{summary.get('final_position')}`",
        f"- Target position: `{summary.get('target_position')}`",
        "",
        *_diagnosis_md_lines(diagnosis),
        "## Images",
        "",
    ])
    images = [
        ("Overhead flight path", "dsim/flight_path.png"),
        ("DAIC sector timeline", "daic/sector_timeline.png"),
    ]
    for title, rel in images:
        if (report_dir / rel).exists():
            lines.extend([f"### {title}", "", f"![{title}]({rel})", ""])
    daic_path_images = _selected_daic_path_images(report_dir)
    if daic_path_images:
        lines.extend([
            "## DAIC Local Route Snapshots",
            "",
            "These images come from `reports/.../daic/occ_*.png`. Red cells are "
            "occupied, green cells are free, the blue line is DAIC's local A* "
            "path, the triangle is the drone, and the yellow marker is the "
            "status-derived target.",
            "",
        ])
        for title, rel, explanation in daic_path_images:
            lines.extend([f"### {title}", "", explanation, "", f"![{title}]({rel})", ""])
    occ_images = _daic_occ_images(report_dir)
    if occ_images:
        lines.extend([
            "## Occupancy Snapshot Series",
            "",
            "The full `occ_*.png` series shows DAIC's local map over time. Read "
            "it as a timeline: early images show initial assumptions, middle "
            "images show whether wall detections persisted, and later images "
            "show the map state near the final outcome. The most useful thing "
            "to compare is whether the blue A* path bends around red occupied "
            "cells or keeps pointing through them.",
            "",
        ])
        for label, rel in occ_images:
            lines.extend([f"### {label}", "", f"![{label}]({rel})", ""])
    if (report_dir / "daic" / "report.html").exists():
        lines.extend(["## DAIC Runtime Report", "", "[Open DAIC HTML report](daic/report.html)", ""])
    lines.extend([
        "## Vision Debug",
        "",
        "```text",
        vision_debug.rstrip(),
        "```",
        "",
    ])
    (report_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def _diagnosis_html(diagnosis: dict) -> str:
    ages = diagnosis.get("front_occ_close_source_age") or {}
    close_sources = "".join(
        "<li>"
        f"{html.escape(str(key))}: {html.escape(str(count))}"
        + (
            f" (age median={html.escape(str((ages.get(key) or {}).get('median')))}, "
            f"max={html.escape(str((ages.get(key) or {}).get('max')))} ticks)"
            if key in ages else ""
        )
        + "</li>"
        for key, count in sorted((diagnosis.get("front_occ_close_sources") or {}).items())
    )
    rows = "".join(
        f"<tr><th>{html.escape(str(label))}</th><td>{html.escape(str(value))}</td></tr>"
        for label, value in (
            ("Target-visible ticks",
             f"{diagnosis.get('target_visible_ticks')} ({diagnosis.get('target_visible_rate')})"),
            ("Approach-gated ticks", diagnosis.get("approach_gated_ticks")),
            ("Yaw-scan ticks", diagnosis.get("yaw_scan_ticks")),
            ("Local-route active ticks", diagnosis.get("route_active_ticks")),
            ("Forward-zero (alignment) ticks", diagnosis.get("forward_zero_alignment_ticks")),
            ("Avoidance-clamped ticks", diagnosis.get("avoidance_clamped_ticks")),
            ("Route-active stalled ticks (union)", diagnosis.get("route_stalled_ticks")),
            (f"front_occ_m <= {diagnosis.get('front_occ_close_threshold_m')} m ticks",
             diagnosis.get("front_occ_close_ticks")),
            (f"front_block_occ_m <= {diagnosis.get('front_occ_close_threshold_m')} m ticks",
             diagnosis.get("front_block_occ_close_ticks")),
            ("Longest close-occupancy run (s)", diagnosis.get("longest_front_occ_close_run_s")),
            ("A* route changes", diagnosis.get("route_changes")),
            ("Straight / detour path ticks",
             f"{diagnosis.get('straight_path_ticks')} / {diagnosis.get('detour_path_ticks')}"),
        )
    )
    hints = "".join(
        f"<li><strong>{html.escape(str(hint.get('label')))}</strong>: "
        f"{html.escape(str(hint.get('evidence')))}</li>"
        for hint in diagnosis.get("classification_hints", [])
    )
    return (
        "<h2>Diagnosis</h2>"
        f"<table>{rows}</table>"
        + (f"<p>Close-front occupancy sources:</p><ul>{close_sources}</ul>"
           if close_sources else "")
        + "<p>Classification hints (confirm against the occupancy snapshot gallery "
        "below — heuristic pointers, not a verdict):</p>"
        f"<ul>{hints}</ul>"
    )


def _write_report_html(report_dir: Path, summary: dict, metadata: dict,
                       vision_debug: str, diagnosis: dict) -> None:
    result = "LANDED" if summary.get("landed") else "DID NOT LAND"
    frame_warning_html = ""
    if summary.get("frame_drop_suspected"):
        frame_warning_html = (
            "<div class=\"warning-banner\"><strong>WARNING: dropped frames "
            f"suspected.</strong> Effective fps ({summary.get('effective_fps')}) "
            f"was only {summary['frame_drop_ratio']:.0%} of the configured "
            f"{summary.get('target_fps')} fps -- the test host was likely "
            "overloaded (e.g. too many concurrent benchmark runs) and this "
            "run's results may be unreliable.</div>"
        )
    effective_fps_html = str(summary.get("effective_fps"))
    if summary.get("frame_drop_ratio") is not None:
        effective_fps_html += (
            f" ({summary['frame_drop_ratio']:.0%} of {summary.get('target_fps')} target)"
        )
        if summary.get("frame_drop_suspected"):
            effective_fps_html = (
                f"<span style=\"color:var(--danger)\">{effective_fps_html} "
                "&mdash; DROPPED FRAMES SUSPECTED</span>"
            )
    image_html = []
    for title, rel in (
        ("Overhead flight path", "dsim/flight_path.png"),
        ("DAIC sector timeline", "daic/sector_timeline.png"),
    ):
        if (report_dir / rel).exists():
            image_html.append(
                f"<h2>{html.escape(title)}</h2><img src=\"{html.escape(rel)}\" "
                "style=\"max-width:100%;height:auto\">"
            )
    daic_path_html = []
    for title, rel, explanation in _selected_daic_path_images(report_dir):
        daic_path_html.append(
            f"<h3>{html.escape(title)}</h3>"
            f"<p>{html.escape(explanation)}</p>"
            f"<img src=\"{html.escape(rel)}\" alt=\"{html.escape(title)}\">"
        )
    if daic_path_html:
        image_html.append(
            "<h2>DAIC Local Route Snapshots</h2>"
            "<p>These images come from <code>reports/.../daic/occ_*.png</code>. "
            "Red cells are occupied, green cells are free, the blue line is "
            "DAIC's local A* path, the triangle is the drone, and the yellow "
            "marker is the status-derived target.</p>"
            + "".join(daic_path_html)
        )
    occ_cards = []
    for label, rel in _daic_occ_images(report_dir):
        occ_cards.append(
            "<figure>"
            f"<img src=\"{html.escape(rel)}\" alt=\"{html.escape(label)}\">"
            f"<figcaption>{html.escape(label)}</figcaption>"
            "</figure>"
        )
    if occ_cards:
        image_html.append(
            "<h2>Occupancy Snapshot Series</h2>"
            "<p>The full <code>occ_*.png</code> series shows DAIC's local map "
            "over time. Read it as a timeline: early images show initial "
            "assumptions, middle images show whether wall detections persisted, "
            "and later images show the map state near the final outcome. The "
            "most useful comparison is whether the blue A* path bends around "
            "red occupied cells or keeps pointing through them.</p>"
            "<div class=\"occ-grid\">"
            + "".join(occ_cards)
            + "</div>"
        )
    daic_link = ""
    if (report_dir / "daic" / "report.html").exists():
        daic_link = '<p><a href="daic/report.html">Open DAIC runtime report</a></p>'
    body = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>DAIC Benchmark - {html.escape(report_dir.name)}</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #0d1117;
      --panel: #161b22;
      --panel-2: #1c2128;
      --border: #30363d;
      --text: #c9d1d9;
      --muted: #8b949e;
      --strong: #e6edf3;
      --accent: #58a6ff;
      --danger: #f85149;
      --ok: #3fb950;
    }}
    body {{
      margin: 0;
      padding: 28px 32px;
      max-width: 1120px;
      background: var(--bg);
      color: var(--text);
      font-family: "Segoe UI", system-ui, sans-serif;
      font-size: 14px;
      line-height: 1.5;
    }}
    h1 {{ margin: 0 0 18px; color: var(--strong); font-size: 2.1em; }}
    h2 {{
      margin: 30px 0 12px;
      padding-bottom: 6px;
      border-bottom: 1px solid var(--border);
      color: var(--strong);
    }}
    a {{ color: var(--accent); }}
    code {{
      background: var(--panel);
      color: var(--strong);
      padding: 1px 6px;
      border-radius: 4px;
    }}
    pre {{
      padding: 14px;
      overflow: auto;
      background: var(--panel);
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 6px;
    }}
    table {{
      border-collapse: collapse;
      min-width: 520px;
      background: var(--panel);
      border: 1px solid var(--border);
    }}
    td, th {{
      border-bottom: 1px solid var(--border);
      padding: 7px 10px;
      text-align: left;
    }}
    th {{ color: var(--muted); font-weight: 500; }}
    img {{
      max-width: 100%;
      height: auto;
      background: var(--panel-2);
      border: 1px solid var(--border);
      border-radius: 6px;
    }}
    .warning-banner {{
      margin: 0 0 18px;
      padding: 12px 16px;
      background: rgba(248, 81, 73, 0.12);
      border: 1px solid var(--danger);
      border-radius: 6px;
      color: var(--strong);
    }}
    figure {{ margin: 0; }}
    figcaption {{ margin-top: 6px; color: var(--muted); font-size: 12px; }}
    .occ-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 18px;
      align-items: start;
    }}
  </style>
</head>
<body>
  <h1>DAIC Benchmark - {html.escape(report_dir.name)}</h1>
  {frame_warning_html}
  <table>
    <tr><th>Result</th><td>{html.escape(result)}</td></tr>
    <tr><th>Final state</th><td>{html.escape(str(summary.get('final_state', 'unknown')))}</td></tr>
    <tr><th>Map</th><td><code>{html.escape(str(metadata.get('map', '')))}</code></td></tr>
    <tr><th>Duration</th><td>{metadata.get('duration_s')} s @ {metadata.get('fps')} fps</td></tr>
    <tr><th>Effective fps</th><td>{effective_fps_html}</td></tr>
    <tr><th>Git commit</th><td><code>{html.escape(str(metadata.get('git_commit') or 'unknown'))}</code></td></tr>
    <tr><th>Dirty worktree</th><td>{metadata.get('git_dirty')}</td></tr>
    <tr><th>Landing error</th><td>{summary.get('landing_error_m')}</td></tr>
    <tr><th>Closest pass</th><td>{summary.get('min_dist_to_target_m')}</td></tr>
  </table>
  {daic_link}
  {_diagnosis_html(diagnosis)}
  {''.join(image_html)}
  <h2>Vision Debug</h2>
  <pre>{html.escape(vision_debug.rstrip())}</pre>
</body>
</html>
"""
    (report_dir / "report.html").write_text(body, encoding="utf-8")


def _write_benchmark_outputs(report_dir: Path, log_path: Path, summary: dict,
                             metadata: dict, command_info: dict) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    root_log = report_dir / "flight.jsonl"
    if log_path.exists() and log_path.resolve() != root_log.resolve():
        shutil.copy2(log_path, root_log)

    daic_dir          = report_dir / "daic"
    route_log_path    = daic_dir / "route_log.jsonl"
    daic_summary_path = daic_dir / "summary.json"
    diagnosis = diagnose_log(
        root_log,
        route_log_path=route_log_path if route_log_path.exists() else None,
        daic_summary_path=daic_summary_path if daic_summary_path.exists() else None,
    )
    diagnosis_buf = io.StringIO()
    print_diagnosis(diagnosis, file=diagnosis_buf)
    (report_dir / "diagnosis.txt").write_text(diagnosis_buf.getvalue(), encoding="utf-8")

    summary_out = dict(summary)
    summary_out["diagnosis"] = diagnosis
    (report_dir / "summary.json").write_text(
        json.dumps(summary_out, indent=2) + "\n", encoding="utf-8"
    )
    (report_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    command_text = "\n".join([
        "dsim:",
        " ".join(command_info.get("dsim_cmd", [])),
        "",
        "daic:",
        " ".join(command_info.get("daic_cmd", [])),
        "",
    ])
    (report_dir / "command.txt").write_text(command_text, encoding="utf-8")
    vision_debug = _capture_vision_debug(root_log)
    (report_dir / "vision_debug.txt").write_text(vision_debug, encoding="utf-8")
    _write_report_md(report_dir, summary, metadata, vision_debug, diagnosis)
    _write_report_html(report_dir, summary, metadata, vision_debug, diagnosis)

    index_dir = ROOT / "reports" / "benchmarks"
    index_dir.mkdir(parents=True, exist_ok=True)
    index_rec = {
        "run_id": report_dir.name,
        "report_dir": str(report_dir.relative_to(ROOT) if report_dir.is_relative_to(ROOT) else report_dir),
        "map": metadata.get("map"),
        "duration_s": metadata.get("duration_s"),
        "fps": metadata.get("fps"),
        "effective_fps": summary.get("effective_fps"),
        "frame_drop_ratio": summary.get("frame_drop_ratio"),
        "frame_drop_suspected": summary.get("frame_drop_suspected"),
        "git_commit": metadata.get("git_commit"),
        "git_dirty": metadata.get("git_dirty"),
        "landed": summary.get("landed"),
        "final_state": summary.get("final_state"),
        "landing_error_m": summary.get("landing_error_m"),
        "min_dist_to_target_m": summary.get("min_dist_to_target_m"),
    }
    with (index_dir / "index.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(index_rec, separators=(",", ":")) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="daic automated flight test")
    parser.add_argument("--map",      default="assets/maps/maze_001.txt",
                        help="map file (relative to project root)")
    parser.add_argument("--duration", type=int, default=90,
                        help="maximum flight time in seconds")
    parser.add_argument("--fps",      type=int, default=30,
                        help="simulation fps (affects --frames budget)")
    parser.add_argument("--log",      default=None,
                        help="path for the JSONL log (default: /tmp/daic_flight_<ts>.jsonl)")
    parser.add_argument("--report-dir", default=None,
                        help="write permanent benchmark outputs to this directory")
    parser.add_argument("--id", default="flighttest",
                        help="shared-memory instance id for dsim/daic — give "
                             "each concurrently-running test a unique id "
                             "(default: flighttest)")
    parser.add_argument("--verbose",  action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    map_path = ROOT / args.map
    if not map_path.exists():
        print(f"error: map not found: {map_path}", file=sys.stderr)
        return 1

    ts = int(time.time())
    report_dir = _resolve_report_dir(args.report_dir)
    if report_dir is not None:
        log_path = report_dir / "daic" / "flight.jsonl"
        explicit_log_path = None
    else:
        log_path = Path(args.log) if args.log else Path(f"/tmp/daic_flight_{ts}.jsonl")
        explicit_log_path = log_path

    print(f"Id:       {args.id}", file=sys.stderr)
    print(f"Map:      {args.map}", file=sys.stderr)
    print(f"Budget:   {args.duration} s  ({args.duration * args.fps} frames @ {args.fps} fps)", file=sys.stderr)
    print(f"Log:      {log_path}", file=sys.stderr)
    if report_dir is not None:
        print(f"Report:   {report_dir}", file=sys.stderr)
    print(file=sys.stderr)

    command_info = run_test(
        str(map_path),
        args.duration,
        explicit_log_path,
        args.fps,
        args.verbose,
        report_dir=report_dir,
        instance_id=args.id,
    )

    if not log_path.exists():
        print("error: no log file produced", file=sys.stderr)
        return 1

    summary = analyze_log(log_path, map_path=str(map_path), target_fps=float(args.fps))
    print_report(summary)
    if summary.get("frame_drop_suspected"):
        print(f"warning: effective fps ({summary.get('effective_fps')}) is only "
              f"{summary['frame_drop_ratio']:.0%} of the configured {args.fps} fps -- "
              "dropped frames suspected (host overloaded?); this run's results may "
              "be unreliable", file=sys.stderr)

    if report_dir is not None:
        metadata = {
            "timestamp_unix": ts,
            "map": args.map,
            "map_path": str(map_path),
            "duration_s": args.duration,
            "fps": args.fps,
            "python": sys.version,
            "platform": platform.platform(),
            "git_commit": _git_value("rev-parse", "HEAD"),
            "git_dirty": bool(_git_value("status", "--porcelain")),
            "dsim_returncode": command_info.get("dsim_returncode"),
            "daic_returncode": command_info.get("daic_returncode"),
        }
        _write_benchmark_outputs(report_dir, log_path, summary, metadata, command_info)

    # Also emit the raw summary as JSON for programmatic consumption.
    print()
    print("JSON summary:")
    print(json.dumps(summary, indent=2))

    return 0 if summary.get("landed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
