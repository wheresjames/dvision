"""The algorithm's own belief as an image. No truth, no verdicts."""
from __future__ import annotations

import numpy as np
from PIL import Image

#: What "no opinion" looks like: a mid blue, so the middle of the probability
#: ramp never reads as dark grey -- and so as free space.
UNDECIDED = (106, 130, 160)


def prediction_raster(grid) -> np.ndarray:
    """One RGB pixel per cell: black free, white occupied, blue undecided.

    Brightness is the probability a cell is occupied, so the picture still
    carries how *strongly* the algorithm believes each cell rather than only
    which side of a threshold it fell. The ramp runs black at p=0 through
    :data:`UNDECIDED` at p=0.5 to white at p=1, which is what stops the middle
    of the range reading as a dark grey and therefore as free space -- the
    misreading that makes a sparse run look convincing on screen.

    ``observed`` is deliberately not folded in, so this stays exactly what
    dalg's live "prediction" pane draws: a cell nobody looked at and a cell the
    algorithm looked at and remains split on are both blue; the evidence grid
    (Grids tab) is where never-observed is kept distinct.
    """
    p = np.clip(np.asarray(grid.probabilities, np.float64), 0.0, 1.0)[..., None]
    neutral = np.asarray(UNDECIDED, np.float64)
    below = p * 2.0 * neutral                                  # black -> neutral
    above = neutral + (p - 0.5) * 2.0 * (255.0 - neutral)      # neutral -> white
    return np.where(p <= 0.5, below, above).round().astype(np.uint8)


def prediction_image(grid, scale: int = 4) -> Image.Image:
    """:func:`prediction_raster`, enlarged without smoothing away single cells."""
    image = Image.fromarray(prediction_raster(grid))
    return image.resize((image.width * scale, image.height * scale),
                        Image.Resampling.NEAREST)
