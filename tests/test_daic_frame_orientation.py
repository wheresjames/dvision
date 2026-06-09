"""Tests for daic client frame orientation."""

import numpy as np

from daic.daic import (
    _client_rgb_frame, _display_detection, _display_rgb_frame, _display_sectors,
)
from daic.detector import Detection
from daic.orb_slam3_detector import ObstacleSectors


def test_client_rgb_frame_keeps_shared_frame_orientation() -> None:
    frame = np.array([
        [[1, 0, 0], [2, 0, 0]],
        [[3, 0, 0], [4, 0, 0]],
    ], dtype=np.uint8)

    out = _client_rgb_frame(frame)

    assert out[:, :, 0].tolist() == [
        [1, 2],
        [3, 4],
    ]
    assert out.flags["C_CONTIGUOUS"]


def test_display_rgb_frame_keeps_shared_frame_orientation() -> None:
    frame = np.array([
        [[1, 0, 0], [2, 0, 0]],
        [[3, 0, 0], [4, 0, 0]],
    ], dtype=np.uint8)

    out = _display_rgb_frame(frame)

    assert out[:, :, 0].tolist() == [
        [1, 2],
        [3, 4],
    ]
    assert out.flags["C_CONTIGUOUS"]


def test_display_detection_keeps_ai_x_coordinate() -> None:
    det = Detection(True, cx=1.0, cy=4.0, radius=2.0, confidence=0.8)

    out = _display_detection(det, width=10)

    assert out.visible
    assert out.cx == 1.0
    assert out.cy == 4.0
    assert out.radius == 2.0
    assert out.confidence == 0.8


def test_display_sectors_keep_left_and_right() -> None:
    sectors = ObstacleSectors(
        front=0.3,
        front_left=0.2,
        front_right=0.4,
        left=0.1,
        right=0.5,
        confidence=0.9,
        method="test",
        front_range_m=3.0,
        front_left_range_m=2.0,
        front_right_range_m=4.0,
        left_range_m=1.0,
        right_range_m=5.0,
    )

    out = _display_sectors(sectors)

    assert out.front == 0.3
    assert out.front_left == 0.2
    assert out.front_right == 0.4
    assert out.left == 0.1
    assert out.right == 0.5
    assert out.front_left_range_m == 2.0
    assert out.front_right_range_m == 4.0
