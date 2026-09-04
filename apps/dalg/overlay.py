from __future__ import annotations

import numpy as np
from PIL import Image

BACKGROUND = (32, 39, 48)
TRUE_POSITIVE = (76, 195, 138)
FALSE_POSITIVE = (255, 107, 107)
FALSE_NEGATIVE = (94, 150, 255)


def verdict_raster(truth, predicted, observable=None) -> np.ndarray:
    seen = np.ones_like(truth.observed) if observable is None else np.asarray(observable, bool)
    rgb = np.full((*truth.observed.shape, 3), BACKGROUND, np.uint8)
    gt, pred = truth.occupied, predicted.occupied
    rgb[gt & pred & seen] = TRUE_POSITIVE
    rgb[~gt & pred & seen] = FALSE_POSITIVE
    rgb[gt & ~pred & seen] = FALSE_NEGATIVE
    return rgb


def overlay_image(truth, predicted, scale: int = 4, observable=None) -> Image.Image:
    image = Image.fromarray(verdict_raster(truth, predicted, observable))
    return image.resize((image.width * scale, image.height * scale),
                        Image.Resampling.NEAREST)


#: What "no opinion" looks like: the verdict overlay's own blue, so a
#: prediction grid and the overlay beside it read as one palette.
#:
#: The overlay's exact BACKGROUND is too dark to serve here. There it means
#: "free or unscored" against saturated verdict colours, but a prediction grid
#: paints confident free space black -- and at luminance 38 the background sits
#: inside that range, so the free space a run actually carved disappears into
#: the cells it never decided. Same hue, scaled to the midpoint of the ramp it
#: has to sit in the middle of.
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
    algorithm looked at and remains split on are both blue, and only the
    scoring separates them.
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


# The scored-region image answers the question every metric depends on: which
# cells did the flight actually see? Coverage and recall are computed over this
# mask, so a low score against a thin mask means something quite different from
# a low score against the whole map.
UNSEEN_WALL = (74, 82, 94)
SEEN_FREE = (54, 84, 116)
SEEN_WALL = (226, 232, 240)


def observable_image(truth, observable=None, scale: int = 4) -> Image.Image:
    seen = np.ones_like(truth.observed) if observable is None else np.asarray(observable, bool)
    rgb = np.full((*truth.observed.shape, 3), BACKGROUND, np.uint8)
    rgb[truth.occupied] = UNSEEN_WALL
    rgb[seen & ~truth.occupied] = SEEN_FREE
    rgb[seen & truth.occupied] = SEEN_WALL
    image = Image.fromarray(rgb)
    return image.resize((image.width * scale, image.height * scale),
                        Image.Resampling.NEAREST)
