"""Test-local fault injection.

These wrappers corrupt data *after* it leaves a production component and
*before* the oracle observes it. Nothing here modifies repository files or
production behaviour, so a fault-injection test can never leak into a real
run. Their only purpose is to prove the oracle rejects each named corruption.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def horizontal_flip(frame_rgb: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(frame_rgb[:, ::-1])


def vertical_flip(frame_rgb: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(frame_rgb[::-1])


def transpose(frame_rgb: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(frame_rgb.transpose(1, 0, 2))


def swap_channels(frame_rgb: np.ndarray) -> np.ndarray:
    """RGB/BGR exchange."""
    return np.ascontiguousarray(frame_rgb[:, :, ::-1])


def rotate_180(frame_rgb: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(frame_rgb[::-1, ::-1])


IMAGE_FAULTS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "horizontal_flip": horizontal_flip,
    "vertical_flip": vertical_flip,
    "transpose": transpose,
    "swap_channels": swap_channels,
    "rotate_180": rotate_180,
}


def invert_yaw(command: dict) -> dict:
    """Return a copy of a velocity command with the yaw-rate sign reversed."""
    corrupted = dict(command)
    corrupted["yaw_rate_dps"] = -float(command.get("yaw_rate_dps", 0.0))
    return corrupted


def invert_strafe(command: dict) -> dict:
    """Return a copy of a velocity command with the right/left sign reversed."""
    corrupted = dict(command)
    corrupted["right_mps"] = -float(command.get("right_mps", 0.0))
    return corrupted


def exchange_forward_and_right(command: dict) -> dict:
    """Return a copy of a velocity command with the two horizontal axes crossed."""
    corrupted = dict(command)
    corrupted["forward_mps"] = float(command.get("right_mps", 0.0))
    corrupted["right_mps"] = float(command.get("forward_mps", 0.0))
    return corrupted


COMMAND_FAULTS: dict[str, Callable[[dict], dict]] = {
    "invert_yaw": invert_yaw,
    "invert_strafe": invert_strafe,
    "exchange_forward_and_right": exchange_forward_and_right,
}


def asymmetric_probe_frame(height: int = 8, width: int = 12) -> np.ndarray:
    """A synthetic frame that is unique under every fault in ``IMAGE_FAULTS``.

    A horizontal gradient in R, a vertical gradient in G, and a constant B
    make the array asymmetric in x, in y, in the x/y axis order, and across
    channels at once.
    """
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :, 0] = np.linspace(10, 240, width, dtype=np.uint8)[None, :]
    frame[:, :, 1] = np.linspace(240, 10, height, dtype=np.uint8)[:, None]
    frame[:, :, 2] = 7
    return frame
