"""A human-readable view of one DALG run, written next to summary.json.

summary.json is the machine record and stays authoritative; this file exists
because reading a nested score object in a terminal tells you almost nothing
about whether an algorithm worked. The report puts the run's scores next to the
two controls that bracket them, next to the overlays that show *where* the
prediction went wrong, and next to the flight that produced the evidence.
"""

from __future__ import annotations

import base64
import html
from pathlib import Path

# The legend borrows the overlay's own palette so the swatches in the report and
# the pixels in the images cannot drift apart.
from dalg.grid import FREE_THRESHOLD, OCCUPIED_THRESHOLD
from dalg.overlay import (BACKGROUND, FALSE_NEGATIVE, FALSE_POSITIVE, SEEN_FREE,
                          SEEN_WALL, TRUE_POSITIVE, UNDECIDED, UNSEEN_WALL)

# (key, column heading, what it means)
METRICS = (
    ("occupied_iou", "Occ IoU",
     "Overlap between predicted and true walls -- the headline number."),
    ("occupied_precision", "Occ prec", "Of the cells called wall, how many are wall."),
    ("occupied_recall", "Occ recall", "Of the true walls, how many were found."),
    ("free_iou", "Free IoU", "The same overlap for free space."),
    ("coverage", "Coverage", "Share of scored cells the algorithm committed to at all."),
    ("brier", "Brier", "Mean squared probability error; 0 is perfect, lower is better."),
    ("hallucination_rate", "Halluc.",
     "Free cells wrongly called wall -- the ones that stop a drone dead."),
)

# Every run scores these two alongside the profile's algorithm, so the headline
# number always has a floor and a ceiling to be read against.
CONTROLS = {
    "constant": "control · assumes the map is empty (score floor)",
    "exact_range": "control · reads the ground truth (score ceiling)",
}

GOOD, WARN, BAD, MUTED, TEXT = "#3fb950", "#e09440", "#f85149", "#8b949e", "#e6edf3"


def _rgb(colour: tuple[int, int, int]) -> str:
    return "rgb(%d,%d,%d)" % colour


def _esc(value) -> str:
    return html.escape("" if value is None else str(value))


def _num(value, digits: int = 3) -> str:
    if value is None: return "—"
    if isinstance(value, float): return f"{value:.{digits}f}"
    return str(value)


def _pct(value) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"


def _grade(key: str, value, baseline) -> str:
    """Traffic-light colour for one metric on the subject row.

    Overlap metrics grade against a fixed scale, but a Brier score has no such
    scale -- 0.08 looks small and is still worse than declaring the whole map
    free -- so it is graded against the empty-map control instead.
    """
    if value is None: return MUTED
    if key == "brier":
        if baseline is None: return MUTED
        return GOOD if value <= baseline * .75 else (WARN if value <= baseline else BAD)
    if key == "hallucination_rate":
        return GOOD if value <= .02 else (WARN if value <= .05 else BAD)
    return GOOD if value >= .7 else (WARN if value >= .35 else BAD)


# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------

def _findings(summary: dict) -> list[dict]:
    """Read the scores the way a reviewer would, and say so out loud."""
    findings: list[dict] = []
    selected = summary.get("algorithm", "")
    scores = summary.get("scores", {})
    subject = scores.get(selected, {}) or {}
    floor = scores.get("constant", {}) or {}
    ceiling = scores.get("exact_range", {}) or {}
    provenance = summary.get("provenance", {}) or {}
    frames = provenance.get("frames", 0) or 0

    if summary.get("partial"):
        findings.append({"level": "warning", "title": "Run did not finish",
                         "body": "The run was recorded as <b>aborted</b>"
                         + (f": {_esc(summary.get('reason'))}. " if summary.get("reason") else ". ")
                         + "The scores below cover only the part of the flight that "
                           "completed, so a low recall may be missing evidence rather "
                           "than a missing wall."})
    if not frames:
        findings.append({"level": "critical", "title": "No frames were observed",
                         "body": "The algorithm never received a frame, so every score "
                                 "here describes its empty initial grid."})

    def better(key: str, higher: bool) -> bool | None:
        mine, theirs = subject.get(key), floor.get(key)
        if mine is None or theirs is None: return None
        return mine > theirs if higher else mine < theirs

    if selected not in CONTROLS and better("occupied_iou", True) is False:
        findings.append({"level": "critical",
                         "title": "No better than the empty-map control",
                         "body": f"<code>{_esc(selected)}</code> scored "
                                 f"{_num(subject.get('occupied_iou'))} occupied IoU against "
                                 f"{_num(floor.get('occupied_iou'))} for the "
                                 "<code>constant</code> control, which predicts nothing "
                                 "at all. Whatever the algorithm found, it is not walls."})
    if selected not in CONTROLS and better("brier", False) is False:
        findings.append({"level": "warning", "title": "Worse calibrated than the floor",
                         "body": f"Brier {_num(subject.get('brier'))} is no better than the "
                                 f"{_num(floor.get('brier'))} you get by declaring the whole "
                                 "map free. The probabilities are confident in the wrong places."})

    coverage = subject.get("coverage")
    if coverage is not None and coverage < .5:
        findings.append({"level": "warning", "title": "Sparse coverage",
                         "body": f"Only {_pct(coverage)} of the scored cells were decided. "
                                 "Precision on a sparse grid is easy; read it together with "
                                 "recall before calling the result good."})
    hallucination = subject.get("hallucination_rate")
    if hallucination is not None and hallucination > .05:
        findings.append({"level": "critical", "title": "Hallucinated obstacles",
                         "body": f"{_pct(hallucination)} of genuinely free cells were called "
                                 "wall. These are the errors that wall a drone into a corner "
                                 "of an empty room."})
    if subject.get("occupied_precision") is None and selected not in CONTROLS:
        findings.append({"level": "warning", "title": "Nothing was called occupied",
                         "body": "The algorithm never marked a single cell as a wall inside "
                                 "the scored region, so precision is undefined."})

    observable, cells = provenance.get("observable_cells"), provenance.get("map_cells")
    if observable and cells:
        share = observable / cells
        findings.append({"level": "info", "title": "Scored region",
                         "body": f"{observable:,} of {cells:,} map cells ({share * 100:.1f}%) "
                                 "were visible from the flight path and are all that is "
                                 "scored. The rest of the map is neither credited nor blamed."})
    mine, top = subject.get("occupied_iou"), ceiling.get("occupied_iou")
    if selected not in CONTROLS and mine is not None and top:
        findings.append({"level": "info", "title": "Against the ceiling",
                         "body": f"The run reached {mine / top * 100:.1f}% of the occupied IoU "
                                 "that the <code>exact_range</code> control gets by reading the "
                                 "ground truth through the same flight path."})
    return findings


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

def _lifecycle(events: list[dict]) -> list[dict]:
    """Run-lifecycle events only.

    Heartbeats are liveness, not history, and they outnumber everything else
    ten to one; run.state repeats once a second and is only interesting when
    the navigator's state or waypoint actually moves on.
    """
    rows, last_state = [], None
    for event in events or []:
        kind = event.get("type", "")
        if kind.startswith("module.heartbeat"): continue
        payload = event.get("payload", {}) or {}
        if kind == "run.state":
            key = (payload.get("state"), payload.get("waypoint_index"))
            if key == last_state: continue
            last_state = key
        note = " · ".join(
            f"{name}={payload[name]}" for name in
            ("state", "outcome", "reason", "waypoint_index", "waypoint_count",
             "start_sim_time_s", "algorithm")
            if payload.get(name) not in (None, ""))
        rows.append({"t": event.get("sim_time_s"), "type": kind,
                     "role": event.get("role", ""), "note": note})
    return rows


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _img_b64(path: Path) -> str | None:
    """Sibling modules' images are inlined so the report survives being copied."""
    try:
        return base64.b64encode(path.read_bytes()).decode() if path.is_file() else None
    except OSError:
        return None


def _swatch(colour: tuple[int, int, int], label: str) -> str:
    return (f"<span class='swatch'><i style='background:{_rgb(colour)}'></i>"
            f"{_esc(label)}</span>")


def _score_table(summary: dict) -> str:
    selected = summary.get("algorithm", "")
    scores = summary.get("scores", {}) or {}
    floor = scores.get("constant", {}) or {}
    order = ([selected] if selected in scores else []) + \
            [name for name in scores if name != selected]
    head = "".join(f"<th title='{_esc(text)}'>{_esc(label)}</th>"
                   for _, label, text in METRICS)
    body = ""
    for name in order:
        row = scores.get(name, {}) or {}
        subject = name == selected
        note = CONTROLS.get(name, "") if not subject else "this profile's algorithm"
        cells = "".join(
            f"<td style='color:"
            f"{_grade(key, row.get(key), floor.get(key)) if subject else MUTED}'>"
            f"{_num(row.get(key))}</td>"
            for key, _, _ in METRICS)
        body += (f"<tr class='{'subject' if subject else ''}'>"
                 f"<td><b>{_esc(name)}</b><div class='muted small'>{_esc(note)}</div></td>"
                 f"{cells}</tr>")
    return f"<table><tr><th>Algorithm</th>{head}</tr>{body}</table>"


def _diagnostics_table(summary: dict) -> str:
    rows = ""
    for name, diagnostics in (summary.get("diagnostics", {}) or {}).items():
        if not diagnostics: continue
        items = " · ".join(f"{_esc(k)}=<b>{_esc(v)}</b>"
                           for k, v in sorted(diagnostics.items()))
        rows += f"<tr><td><code>{_esc(name)}</code></td><td>{items}</td></tr>"
    return f"<table style='width:auto'>{rows}</table>" if rows else ""


def _ordered(summary: dict, images: dict[str, str]) -> list[str]:
    """This profile's own algorithm first, then the controls that bracket it."""
    selected = summary.get("algorithm", "")
    return ([selected] if selected in images else []) + \
           [name for name in images if name != selected]


def _prediction_gallery(summary: dict, predictions: dict[str, str]) -> str:
    """What the live window showed, beside the verdict that scored it."""
    selected = summary.get("algorithm", "")
    cards = ""
    for name in _ordered(summary, predictions):
        label = "this profile's algorithm" if name == selected else CONTROLS.get(name, "")
        coverage = ((summary.get("scores", {}) or {}).get(name, {}) or {}).get("coverage")
        undecided = ("" if coverage is None else
                     f" {_pct(1.0 - coverage)} of the scored region is blue"
                     " — undecided, not free.")
        cards += f"""
      <figure>
        <img src="{_esc(predictions[name])}" alt="Prediction grid for {_esc(name)}">
        <figcaption><b>{_esc(name)}</b>{' — ' + _esc(label) if label else ''}.
        {undecided}</figcaption>
      </figure>"""
    return cards


def _gallery(summary: dict, overlays: dict[str, str], region: str | None) -> str:
    selected = summary.get("algorithm", "")
    cards = ""
    if region:
        cards += f"""
      <figure>
        <img src="{_esc(region)}" alt="Scored region">
        <figcaption><b>Scored region</b> — cells visible from the flight path.
        Everything outside it is excluded from every score above.</figcaption>
      </figure>"""
    for name in _ordered(summary, overlays):
        label = "this profile's algorithm" if name == selected else CONTROLS.get(name, "")
        cards += f"""
      <figure>
        <img src="{_esc(overlays[name])}" alt="Overlay for {_esc(name)}">
        <figcaption><b>{_esc(name)}</b>{' — ' + _esc(label) if label else ''}</figcaption>
      </figure>"""
    return cards


def render_html(summary: dict, overlays: dict[str, str], region: str | None,
                events: list[dict], flight_images: dict[str, str],
                predictions: dict[str, str] | None = None) -> str:
    partial = bool(summary.get("partial"))
    outcome = "ABORTED" if partial else "COMPLETE ✓"
    colour = WARN if partial else GOOD
    provenance = summary.get("provenance", {}) or {}
    selected = summary.get("algorithm", "")
    subject = (summary.get("scores", {}) or {}).get(selected, {}) or {}

    facts = [
        ("Run id", summary.get("run_id") or "—"),
        ("Profile", summary.get("profile") or "—"),
        ("Algorithm", selected or "—"),
        ("Sensors", ", ".join(summary.get("sensors", [])) or "—"),
        ("Frames observed", provenance.get("frames", "—")),
        ("Occupied IoU", _num(subject.get("occupied_iou"))),
        ("Coverage", _pct(subject.get("coverage"))),
        ("Scored region", summary.get("scored_region", "—")),
        ("Observable cells", f"{provenance.get('observable_cells', 0):,} of "
                             f"{provenance.get('map_cells', 0):,}"),
        ("Flight mode", provenance.get("flight_mode", "—")),
        ("Tour", provenance.get("tour_id") or "—"),
        ("Navigator", provenance.get("navigator_implementation")
                      or provenance.get("coordinator_implementation") or "—"),
        ("Coordinator outcome", provenance.get("coordinator_outcome") or "—"),
        ("Reason", summary.get("reason") or "—"),
        ("Finished at", summary.get("finished_at") or "—"),
        ("Profile digest", (summary.get("profile_digest") or "—")[:16]),
        ("Map digest", (provenance.get("map_digest") or "—")[:16]),
    ]
    facts_html = "".join(f"<tr><td class='muted'>{_esc(k)}</td>"
                         f"<td style='color:{TEXT}'>{_esc(v)}</td></tr>" for k, v in facts)

    findings_html = ""
    for finding in _findings(summary):
        border = {"critical": BAD, "warning": WARN, "info": "#58a6ff"}[finding["level"]]
        icon = {"critical": "⛔", "warning": "⚠️", "info": "ℹ️"}[finding["level"]]
        findings_html += f"""
      <div class="finding" style="border-left-color:{border}">
        <div style="color:{border};font-weight:bold">{icon} {_esc(finding['title'])}</div>
        <div>{finding['body']}</div>
      </div>"""

    event_rows = "".join(
        f"<tr><td>{_num(row['t'], 3)}s</td><td><code>{_esc(row['type'])}</code></td>"
        f"<td class='muted'>{_esc(row['role'])}</td><td>{_esc(row['note'])}</td></tr>"
        for row in _lifecycle(events))

    flight_html = "".join(f"""
      <div class="section">
        <h2>{_esc(title)}</h2>
        <img src="data:image/png;base64,{b64}">
      </div>""" for title, b64 in flight_images.items())

    # The picture the operator watched while the run flew. It goes above the
    # verdict overlays because reading it first is what makes the overlays
    # legible: the grid it shows is the one every score below was computed
    # from, and its mid grey is the reason a convincing-looking run can score
    # badly.
    prediction_html = "" if not predictions else f"""
<div class="section">
  <h2>Prediction Grids</h2>
  <p class="muted">The occupancy grid exactly as dalg's live <b>prediction</b>
  pane drew it — brightness is the probability a cell is occupied.<br>
  {_swatch((255, 255, 255), "occupied (p → 1)")}
  {_swatch(UNDECIDED, "no opinion (p = 0.5)")}
  {_swatch((0, 0, 0), "free (p → 0)")}<br>
  Scoring is not a gradient: a cell counts as occupied only at
  p&nbsp;≥&nbsp;{OCCUPIED_THRESHOLD:g} and free only at
  p&nbsp;≤&nbsp;{FREE_THRESHOLD:g}. Everything between is undecided and
  contributes to no score, however dark or bright it looks here — which is
  what <code>coverage</code> measures.</p>
  <div class="grid">{_prediction_gallery(summary, predictions)}</div>
</div>"""

    metric_help = "".join(
        f"<tr><td><code>{_esc(key)}</code></td><td class='muted'>{_esc(text)}</td></tr>"
        for key, _, text in METRICS)

    diagnostics_html = _diagnostics_table(summary)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>DALG run · {_esc(summary.get('profile', ''))}</title>
<style>
  body{{margin:0;padding:20px 28px;font-family:'Segoe UI',system-ui,sans-serif;
       background:#0d1117;color:#c9d1d9;font-size:14px;line-height:1.5}}
  h1{{margin:0 0 4px;font-size:2.2em;color:{colour}}}
  h2{{color:{TEXT};border-bottom:1px solid #30363d;padding-bottom:6px;margin-top:32px}}
  .section{{margin-bottom:28px}}
  .muted{{color:{MUTED}}}
  .small{{font-size:12px}}
  table{{border-collapse:collapse;width:100%}}
  td,th{{padding:5px 10px;text-align:left;border-bottom:1px solid #21262d}}
  th{{color:{MUTED};font-weight:normal;font-size:12px}}
  tr.subject{{background:#161b22}}
  code{{background:#161b22;padding:1px 6px;border-radius:4px;font-size:13px}}
  img{{max-width:100%;border-radius:6px;border:1px solid #30363d;background:#161b22}}
  figure{{margin:0}}
  figcaption{{color:{MUTED};font-size:12px;margin-top:6px}}
  .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));
        gap:18px;align-items:start}}
  .finding{{border-left:4px solid {MUTED};padding:10px 14px;margin:10px 0;
           background:#161b22;border-radius:0 6px 6px 0}}
  .swatch{{display:inline-flex;align-items:center;gap:6px;margin-right:16px;
          font-size:12px;color:{MUTED}}}
  .swatch i{{width:12px;height:12px;border-radius:3px;display:inline-block;
            border:1px solid #30363d}}
</style>
</head>
<body>

<div class="section">
  <h1>{outcome}</h1>
  <div class="muted">{_esc(summary.get('profile', ''))} ·
    <code>{_esc(selected)}</code> · occupied IoU
    {_num(subject.get('occupied_iou'))} · {_esc(provenance.get('frames', 0))} frames</div>
</div>

<div class="section">
  <h2>Run</h2>
  <table style="width:auto">{facts_html}</table>
</div>

{'<div class="section"><h2>Findings</h2>' + findings_html + '</div>' if findings_html else ''}

<div class="section">
  <h2>Scores</h2>
  <p class="muted">Every run scores its algorithm against two controls on the same
  cells: <code>constant</code> predicts an empty map, <code>exact_range</code> reads
  the ground truth. A result only means something between those two.</p>
  {_score_table(summary)}
  <table style="width:auto;margin-top:14px">{metric_help}</table>
</div>

{'<div class="section"><h2>Diagnostics</h2>' + diagnostics_html + '</div>' if diagnostics_html else ''}

{prediction_html}

<div class="section">
  <h2>Occupancy Overlays</h2>
  <p class="muted">Each overlay compares the predicted grid with the ground truth
  over the scored region.<br>
  {_swatch(TRUE_POSITIVE, "wall found")}{_swatch(FALSE_POSITIVE, "wall hallucinated")}
  {_swatch(FALSE_NEGATIVE, "wall missed")}{_swatch(BACKGROUND, "free / unscored")}<br>
  Scored region: {_swatch(SEEN_WALL, "wall seen")}{_swatch(SEEN_FREE, "free seen")}
  {_swatch(UNSEEN_WALL, "wall never in view")}</p>
  <div class="grid">{_gallery(summary, overlays, region)}</div>
</div>

{flight_html}

{'<div class="section"><h2>Run Lifecycle</h2><table><tr><th>Sim time</th><th>Event</th><th>Role</th><th>Detail</th></tr>' + event_rows + '</table></div>' if event_rows else ''}

<div class="section muted small">
  Generated by dalg · the machine-readable record of this run is
  <code>summary.json</code> beside this file.
</div>

</body>
</html>"""


def write_html_report(out: Path, summary: dict, *, overlays: dict[str, str],
                      region: str | None = None, events: list[dict] | None = None,
                      report_root: Path | None = None,
                      predictions: dict[str, str] | None = None) -> Path:
    """Write report.html into a finished report directory.

    `report_root` is the run's report root -- the directory holding the other
    modules' output -- so the flight that produced the evidence can be shown
    beside the scores it produced.
    """
    flight_images = {}
    if report_root is not None:
        for title, relative in (("Flight Path (dsim)", "dsim/flight_path.png"),
                                ("Navigator Track (dway)", "dway/track.png")):
            encoded = _img_b64(report_root / relative)
            if encoded: flight_images[title] = encoded
    path = out / "report.html"
    path.write_text(render_html(summary, overlays, region, events or [],
                                flight_images, predictions), encoding="utf-8")
    return path
