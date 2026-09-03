"""Failure artifacts.

A failed closed-loop or process test should leave behind enough evidence to
locate the broken boundary without an interactive reproduction: the raw frames,
an annotated frame showing expected versus observed landmarks, the commands
that were sent, a pose/telemetry timeline, the fixture and camera parameters,
a concise JSON result, and a top-down path plot when the drone moved.

These are diagnostics, never the correctness oracle.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def artifact_directory(tmp_path: Path, test_name: str) -> Path:
    """Return a per-test artifact path, optionally rooted for CI retention.

    Set ``DVISION_TEST_ARTIFACTS`` to a persistent CI-upload directory.
    Otherwise pytest's isolated temporary directory is used.
    """
    root = os.environ.get("DVISION_TEST_ARTIFACTS")
    if root:
        return Path(root) / f"{test_name}-{uuid.uuid4().hex[:8]}"
    return Path(tmp_path) / test_name


def _save_frame(path: Path, frame_rgb: np.ndarray) -> None:
    Image.fromarray(np.ascontiguousarray(frame_rgb), "RGB").save(path)


def save_annotated_frame(path: Path, frame_rgb: np.ndarray, *,
                         observed: dict | None = None,
                         expected: dict | None = None) -> None:
    """Draw observed centroids as crosses and expected regions as boxes.

    ``observed`` maps a label to ``(x, y)``; ``expected`` maps a label to an
    ``(x_lo, x_hi, y_lo, y_hi)`` region.
    """
    image = Image.fromarray(np.ascontiguousarray(frame_rgb), "RGB").convert("RGB")
    draw = ImageDraw.Draw(image)
    for label, (x_lo, x_hi, y_lo, y_hi) in (expected or {}).items():
        draw.rectangle([x_lo, y_lo, x_hi, y_hi], outline=(255, 255, 255))
        draw.text((x_lo + 2, y_lo + 2), f"expect {label}", fill=(255, 255, 255))
    for label, (x, y) in (observed or {}).items():
        draw.line([x - 9, y, x + 9, y], fill=(0, 0, 0), width=3)
        draw.line([x, y - 9, x, y + 9], fill=(0, 0, 0), width=3)
        draw.line([x - 8, y, x + 8, y], fill=(255, 255, 0), width=1)
        draw.line([x, y - 8, x, y + 8], fill=(255, 255, 0), width=1)
        draw.text((x + 10, y + 2), label, fill=(255, 255, 0))
    image.save(path)


def save_path_plot(path: Path, positions: list) -> bool:
    """Top-down plot of ``(x, y)`` map positions. Returns False if not plotted."""
    points = [(float(p[0]), float(p[1])) for p in positions if p is not None]
    if len(points) < 2:
        return False
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    if max(xs) - min(xs) < 1e-6 and max(ys) - min(ys) < 1e-6:
        return False
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
    except ImportError:
        return False
    figure = Figure(figsize=(4.5, 4.5), dpi=110)
    FigureCanvasAgg(figure)
    axes = figure.add_subplot(111)
    axes.plot(xs, ys, "-o", markersize=2.5, linewidth=1.0)
    axes.plot(xs[0], ys[0], "go", label="start")
    axes.plot(xs[-1], ys[-1], "ro", label="end")
    axes.set_xlabel("map x (east, m)")
    axes.set_ylabel("map y (south, m)")
    axes.invert_yaxis()  # map rows grow south, so draw north upward
    axes.set_aspect("equal", adjustable="datalim")
    axes.grid(True, alpha=0.3)
    axes.legend(loc="best", fontsize="small")
    figure.tight_layout()
    figure.savefig(path)
    return True


def save_failure_bundle(directory: Path, *, frame_rgb: np.ndarray | None = None,
                        telemetry: dict | None = None, details: dict | None = None,
                        frames: dict | None = None,
                        observed: dict | None = None,
                        expected: dict | None = None,
                        commands: list | None = None,
                        timeline: list | None = None,
                        positions: list | None = None,
                        fixture: dict | None = None) -> Path:
    """Save the reproducible visual and telemetry evidence for one failure.

    Every argument is optional so a cheap in-process assertion can save just a
    frame while a process test saves the full bundle.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []

    for name, frame in (frames or {}).items():
        if frame is None:
            continue
        _save_frame(directory / f"{name}.png", frame)
        saved.append(f"{name}.png")
    if frame_rgb is not None:
        _save_frame(directory / "frame.png", frame_rgb)
        saved.append("frame.png")
        if observed or expected:
            save_annotated_frame(directory / "frame_annotated.png", frame_rgb,
                                 observed=observed, expected=expected)
            saved.append("frame_annotated.png")

    if positions and save_path_plot(directory / "path.png", positions):
        saved.append("path.png")

    if commands:
        (directory / "commands.jsonl").write_text(
            "".join(json.dumps(c, sort_keys=True) + "\n" for c in commands),
            encoding="utf-8",
        )
        saved.append("commands.jsonl")
    if timeline:
        (directory / "timeline.jsonl").write_text(
            "".join(json.dumps(t, sort_keys=True) + "\n" for t in timeline),
            encoding="utf-8",
        )
        saved.append("timeline.jsonl")

    payload = {
        "telemetry": telemetry or {},
        "details": details or {},
        "fixture": fixture or {},
        "observed": {k: list(v) for k, v in (observed or {}).items()},
        "expected": {k: list(v) for k, v in (expected or {}).items()},
        "artifacts": saved,
    }
    (directory / "result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return directory
