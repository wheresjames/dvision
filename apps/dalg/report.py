"""Truth-independent summaries and evidence images; numeric data lives in archive.

The evidence images are the pane's own rendering: when the operator displayed
a reference background, the report composes the exact same revision under the
never-observed cells, and the run archives those bytes beside
the record, referenced by checksum.
"""
import json
from pathlib import Path
from dcmn.map_pane import snapshot_image


def write_report(directory, *, summary, evidence, background=None):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    rows = {}
    for sid, grid in evidence.items():
        filename = f'evidence-{sid}.png'
        rows[sid] = dict(image=filename, geometry=grid.geometry.as_dict(), revision=grid.revision,
                         time_s=grid.sim_time_s, provenance=grid.entry)
        try: snapshot_image(grid, background=background).save(directory/filename)
        except Exception as exc: rows[sid]['image_error'] = str(exc)
    (directory/'summary.json').write_text(json.dumps(dict(summary, evidence=rows), indent=2)+'\n')
    return directory
