"""The page a report is written on, without knowing what the report is about.

`dalg` and `daic` each grew their own HTML writer, and both begin the same way:
a doctype, a style block restating the palette, and a handful of helpers for
sections, tables and figures. They agree on the colours and disagree on the
spacing, which is exactly how `dcmn.theme` and `dcmn.tktheme` came about for the
windows.

This is the common part, and only the common part. It knows about documents,
sections, tables and figures; it knows nothing about runs, scores or flights.
A report builds its own content and hands it here to be made into a page.

The existing two are deliberately left alone. This exists so the third one does
not add a third dialect, and so there is something for them to converge on when
somebody has reason to touch them.
"""

from __future__ import annotations

import base64
import html
from pathlib import Path
from typing import Any, Iterable, Sequence

from dcmn import theme


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _css() -> str:
    """The palette as a stylesheet. One source, so a report matches a window."""
    return f"""
  :root {{ color-scheme: dark; }}
  body {{ background:{theme.BG}; color:{theme.TEXT}; margin:0;
         font:14px/1.55 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif; }}
  .page {{ max-width:1080px; margin:0 auto; padding:28px 22px 56px; }}
  h1 {{ font-size:21px; margin:0 0 2px; }}
  h2 {{ font-size:15px; margin:0 0 12px; color:{theme.TEXT};
        text-transform:uppercase; letter-spacing:.08em; }}
  .subtitle {{ color:{theme.DIM}; margin:0 0 22px; }}
  .section {{ background:{theme.PANEL}; border:1px solid {theme.GRID};
              border-radius:8px; padding:16px 18px; margin:0 0 16px; }}
  table {{ border-collapse:collapse; width:100%; font-size:13px; }}
  th, td {{ text-align:left; padding:6px 10px;
            border-bottom:1px solid {theme.GRID}; }}
  th {{ color:{theme.DIM}; font-weight:600; white-space:nowrap; }}
  td.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
  tr:last-child td {{ border-bottom:none; }}
  .facts {{ display:flex; flex-wrap:wrap; gap:10px 26px; margin:0 0 4px; }}
  .fact b {{ display:block; color:{theme.DIM}; font-weight:600; font-size:11px;
             text-transform:uppercase; letter-spacing:.06em; }}
  .fact span {{ font-size:16px; }}
  figure {{ margin:0; }}
  figure img {{ width:100%; height:auto; border:1px solid {theme.GRID};
                border-radius:6px; background:{theme.CANVAS}; }}
  figcaption {{ color:{theme.DIM}; font-size:12px; margin-top:8px; }}
  .muted {{ color:{theme.DIM}; }}
  .ok {{ color:{theme.OK}; }} .warn {{ color:{theme.WARN}; }}
  .bad {{ color:{theme.DANGER}; }} .unknown {{ color:{theme.DIM}; }}
  code {{ background:{theme.ENTRY}; border-radius:4px; padding:1px 5px; }}
"""


def document(title: str, *, subtitle: str = "", blocks: Iterable[str] = ()) -> str:
    body = "\n".join(blocks)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)}</title>
<style>{_css()}</style>
</head>
<body>
<div class="page">
<h1>{esc(title)}</h1>
<p class="subtitle">{esc(subtitle)}</p>
{body}
</div>
</body>
</html>
"""


def section(heading: str, *body: str) -> str:
    inner = "\n".join(part for part in body if part)
    return f'<div class="section">\n<h2>{esc(heading)}</h2>\n{inner}\n</div>'


def facts(pairs: Sequence[tuple[str, Any]]) -> str:
    cells = "".join(
        f'<div class="fact"><b>{esc(name)}</b><span>{value}</span></div>'
        for name, value in pairs)
    return f'<div class="facts">{cells}</div>'


def table(headers: Sequence[str], rows: Iterable[Sequence[Any]], *,
          numeric: Sequence[int] = ()) -> str:
    """A table. ``numeric`` names the column indices to right-align.

    Cell values pass through untouched, so a caller may hand in a span with a
    grade class; anything from outside must be escaped by the caller.
    """
    numeric = set(numeric)
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = []
    for row in rows:
        cells = "".join(
            f'<td class="num">{cell}</td>' if i in numeric else f"<td>{cell}</td>"
            for i, cell in enumerate(row))
        body.append(f"<tr>{cells}</tr>")
    return (f"<table><thead><tr>{head}</tr></thead>"
            f"<tbody>{''.join(body)}</tbody></table>")


def figure(path: Path, caption: str = "") -> str:
    """An image embedded in the page, so the file travels on its own.

    Returns an empty string when the image is missing: a report is worth having
    without its illustration, and a broken image is worse than none.
    """
    encoded = embed(path)
    if encoded is None:
        return ""
    tail = f"<figcaption>{esc(caption)}</figcaption>" if caption else ""
    return f'<figure><img src="{encoded}" alt="{esc(caption)}">{tail}</figure>'


def embed(path: Path) -> str | None:
    try:
        raw = Path(path).read_bytes()
    except OSError:
        return None
    return "data:image/png;base64," + base64.b64encode(raw).decode("ascii")


def graded(text: Any, grade: str) -> str:
    """A value coloured by its grade, using the same names the palette uses."""
    known = {"ok", "warn", "bad", "unknown"}
    return f'<span class="{grade if grade in known else "unknown"}">{esc(text)}</span>'
