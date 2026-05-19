"""Tests for daic client frame orientation."""

import numpy as np

from daic.daic import _client_rgb_frame


def test_client_rgb_frame_flips_vertical_and_horizontal_axes() -> None:
    frame = np.array([
        [[1, 0, 0], [2, 0, 0]],
        [[3, 0, 0], [4, 0, 0]],
    ], dtype=np.uint8)

    out = _client_rgb_frame(frame)

    assert out[:, :, 0].tolist() == [
        [4, 3],
        [2, 1],
    ]
    assert out.flags["C_CONTIGUOUS"]
