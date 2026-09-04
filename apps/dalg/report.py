from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

from dalg.overlay import observable_image, overlay_image, prediction_image
from dalg.report_html import write_html_report
from dalg.score import score_occupancy


def write_report(report_root: Path, *, run_id: str, profile, truth,
                 results: dict, provenance: dict, partial: bool,
                 reason: str = "", events: list[dict] | None = None,
                 observable=None) -> Path:
    # One directory per module, named after the module, exactly like dsim,
    # daic and dway. This used to nest a <run_id>-<profile> directory inside
    # it so a second run in one simulator session could not overwrite the
    # first -- but every other module overwrites in that case too, so the
    # uniqueness bought nothing that the report as a whole provides, at the
    # cost of an opaque path. If runs ever need to be kept apart, that belongs
    # in the run root every module shares, not in one module's subdirectory.
    out = Path(report_root) / "dalg"
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
                          events=events, report_root=out.parent)
    except Exception as exc:  # pragma: no cover - reporting is best effort
        print(f"dalg: html report failed: {exc}", file=sys.stderr)
    return out
