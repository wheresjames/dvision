from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

from dalg.overlay import observable_image, overlay_image, prediction_image
from dalg.report_html import write_html_report
from dalg.score import score_occupancy


def run_directory(run_id: str, profile_name: str) -> str:
    """A filesystem-safe directory name for one run.

    Reports used to land on a single fixed path, so a second run -- or a second
    profile -- in the same simulator session silently overwrote the first. The
    run id comes off the bus, so it is sanitised rather than trusted.
    """
    raw = f"{run_id or 'unidentified'}-{profile_name}"
    cleaned = "".join(c if c.isalnum() or c in "-_" else "-" for c in raw)
    return cleaned.strip("-")[:96] or "unidentified"


def write_report(report_root: Path, *, run_id: str, profile, truth,
                 results: dict, provenance: dict, partial: bool,
                 reason: str = "", events: list[dict] | None = None,
                 observable=None) -> Path:
    out = Path(report_root) / "dalg" / run_directory(run_id, profile.name)
    out.mkdir(parents=True, exist_ok=True)
    scores = {}
    diagnostics = {}
    overlays = {}
    predictions = {}
    for name, result in results.items():
        scores[name] = score_occupancy(result.grid, truth, observable)
        diagnostics[name] = result.diagnostics
        overlays[name] = f"overlay-{name}.png"
        overlay_image(truth, result.grid, observable=observable).save(
            out / overlays[name])
        # The raw grid as the live window drew it, so a report can be read
        # against the picture that was on screen while the run was flying.
        predictions[name] = f"prediction-{name}.png"
        prediction_image(result.grid).save(out / predictions[name])
    region = "scored-region.png"
    observable_image(truth, observable).save(out / region)
    summary = {
        "schema_version": 1, "run_id": run_id,
        "profile": profile.name, "profile_digest": profile.digest,
        "algorithm": profile.algorithm, "sensors": list(profile.sensors),
        "outcome": "aborted" if partial else "complete", "partial": partial,
        "reason": reason, "finished_at": datetime.datetime.now().astimezone().isoformat(),
        # Scores cover only the cells the flight could see; see dalg.visibility.
        "scored_region": "observable" if observable is not None else "whole_map",
        "provenance": provenance, "scores": scores, "diagnostics": diagnostics,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True),
                                      encoding="utf-8")
    with (out / "events.jsonl").open("w", encoding="utf-8") as handle:
        for event in events or []:
            handle.write(json.dumps(event, sort_keys=True,
                                    separators=(",", ":")) + "\n")
    # A broken renderer must not cost the run its machine-readable record, which
    # is already on disk by this point.
    try:
        write_html_report(out, summary, overlays=overlays,
                          predictions=predictions, region=region,
                          events=events, report_root=out.parent.parent)
    except Exception as exc:  # pragma: no cover - reporting is best effort
        print(f"dalg: html report failed: {exc}", file=sys.stderr)
    return out
