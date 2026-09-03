"""Scene presets the renderer can build, and the version each one identifies.

A preset changes appearance only. The geometry -- and therefore the ranges the
exact oracle casts through it -- is identical across presets, which is what
makes a lighting change safe to measure against unchanged truth.

Consumers that record which scene a result came from should record the
*version*, not the preset name, so that a renderer change is visible in the
record rather than hidden behind a stable label.

This lives in its own module rather than in ``dsim/__init__.py`` because
``.gitignore`` excludes ``_*``: no ``__init__.py`` in this repository is
tracked, so a constant placed in one would not survive a fresh clone.
"""

from __future__ import annotations

SCENE_VERSION = "legacy-fog-corrected-v1"
REPRESENTATIVE_SCENE_VERSION = "representative-pbr-shadows-v1"

SCENE_PRESETS = {
    "legacy": SCENE_VERSION,
    "representative": REPRESENTATIVE_SCENE_VERSION,
}

#: Bumped when the built geometry changes, independently of appearance.
SCENE_GEOMETRY_VERSION = "geometry-v1"

__all__ = ["REPRESENTATIVE_SCENE_VERSION", "SCENE_GEOMETRY_VERSION",
           "SCENE_PRESETS", "SCENE_VERSION"]
