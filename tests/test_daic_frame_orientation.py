"""Tests for daic client frame orientation."""

import math

import numpy as np
import pytest

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


def test_slam_minimap_draws_starboard_points_on_the_right() -> None:
    """The top-down SLAM map must use the same handedness as ObstacleSectors."""
    from daic.daic import _slam_canvas_point

    bounds = dict(x_min=-10.0, x_max=10.0, z_min=-10.0, z_max=10.0,
                  margin=6, use=100)
    centre_x, centre_y = _slam_canvas_point(0.0, 0.0, **bounds)
    right_x, _ = _slam_canvas_point(5.0, 0.0, **bounds)
    left_x, _ = _slam_canvas_point(-5.0, 0.0, **bounds)

    assert right_x > centre_x, (
        f"camera X+ (starboard) drew at x={right_x} instead of right of "
        f"centre x={centre_x}"
    )
    assert left_x < centre_x


def test_slam_drone_marker_puts_starboard_right_of_port() -> None:
    """The marker must agree with the point cloud it sits inside."""
    from daic.daic import _slam_drone_marker

    tip, port, starboard = _slam_drone_marker(100, 100, 0.0)

    assert starboard[0] > port[0], (
        f"starboard vertex drew at x={starboard[0]} and port at x={port[0]}; "
        "the drone marker is mirrored against the point cloud")
    assert tip[1] < port[1], (
        f"nose drew at y={tip[1]} and the tail at y={port[1]}; heading 0 must "
        "point up the canvas")


def test_slam_drone_marker_turns_the_same_way_as_the_point_cloud() -> None:
    """Yawing right must swing the nose toward the canvas right."""
    from daic.daic import _slam_drone_marker

    ahead = _slam_drone_marker(100, 100, 0.0)[0]
    turned_right = _slam_drone_marker(100, 100, math.radians(45.0))[0]
    turned_left = _slam_drone_marker(100, 100, math.radians(-45.0))[0]

    assert turned_right[0] > ahead[0], "a right turn must swing the nose right"
    assert turned_left[0] < ahead[0], "a left turn must swing the nose left"


def test_slam_minimap_draws_forward_points_upward() -> None:
    from daic.daic import _slam_canvas_point

    bounds = dict(x_min=-10.0, x_max=10.0, z_min=-10.0, z_max=10.0,
                  margin=6, use=100)
    _, centre_y = _slam_canvas_point(0.0, 0.0, **bounds)
    _, forward_y = _slam_canvas_point(0.0, 5.0, **bounds)
    _, behind_y = _slam_canvas_point(0.0, -5.0, **bounds)

    assert forward_y < centre_y, "camera Z+ (forward) must draw up the canvas"
    assert behind_y > centre_y


# ---------------------------------------------------------------------------
# Perception sector handedness
#
# Every detector reduces the frame to five named sectors, and those names are
# what the local map turns into world bearings. If a detector puts image-left
# content in the `right` sector the whole map is mirrored, and no image-level
# oracle can see it: the frame is correct, only its interpretation is not.
# orb_slam3 already has this coverage; these are its two siblings, which are
# the detectors that actually run.
# ---------------------------------------------------------------------------

def _half_expansion_flow(side: str, w: int = 160, h: int = 120,
                         scale: float = 0.05) -> np.ndarray:
    """Radial expansion present on one image half only."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    flow = np.zeros((h, w, 2), dtype=np.float32)
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    flow[:, :, 0] = (xx - cx) * scale
    flow[:, :, 1] = (yy - cy) * scale
    keep = xx < cx if side == "left" else xx >= cx
    flow[~keep] = 0.0
    return flow


def test_optical_flow_puts_image_left_expansion_in_the_left_sectors() -> None:
    from daic.optical_flow_avoidance import _flow_to_sectors

    s = _flow_to_sectors(_half_expansion_flow("left"),
                         forward_speed_mps=0.5, dt_s=0.05)

    assert s.left > s.right, (
        f"expansion on the image left produced left={s.left:.2f} "
        f"right={s.right:.2f}; the sectors are mirrored")
    assert s.front_left > s.front_right


def test_optical_flow_puts_image_right_expansion_in_the_right_sectors() -> None:
    from daic.optical_flow_avoidance import _flow_to_sectors

    s = _flow_to_sectors(_half_expansion_flow("right"),
                         forward_speed_mps=0.5, dt_s=0.05)

    assert s.right > s.left, (
        f"expansion on the image right produced left={s.left:.2f} "
        f"right={s.right:.2f}; the sectors are mirrored")
    assert s.front_right > s.front_left


def _camera_points_at_azimuth(az_deg: float, n: int = 40) -> np.ndarray:
    """N points at a fixed azimuth in the OpenCV camera frame (x right, z fwd)."""
    from daic.mini_slam_detector import SLAM_CLOSE_UNITS

    z = np.full(n, SLAM_CLOSE_UNITS + 0.1) + np.linspace(0.0, 0.02, n)
    x = z * np.tan(np.radians(az_deg))
    return np.stack([x, np.zeros(n), z], axis=1)


def _mini_slam_sectors(az_deg: float):
    from daic.mini_slam_detector import MiniSLAMDetector

    detector = MiniSLAMDetector.__new__(MiniSLAMDetector)
    detector._scale = None
    return detector._project_sectors(np.eye(4), _camera_points_at_azimuth(az_deg))


def test_mini_slam_puts_camera_left_points_in_the_left_sector() -> None:
    """Camera X+ is starboard, so a negative azimuth is the drone's left."""
    s = _mini_slam_sectors(-50.0)

    assert s.left > s.right, (
        f"points 50 deg to the camera's left produced left={s.left:.2f} "
        f"right={s.right:.2f}; the sectors are mirrored")


def test_mini_slam_puts_camera_right_points_in_the_right_sector() -> None:
    s = _mini_slam_sectors(50.0)

    assert s.right > s.left, (
        f"points 50 deg to the camera's right produced left={s.left:.2f} "
        f"right={s.right:.2f}; the sectors are mirrored")


def test_mini_slam_forward_sectors_keep_their_side() -> None:
    left = _mini_slam_sectors(-20.0)
    right = _mini_slam_sectors(20.0)

    assert left.front_left > left.front_right
    assert right.front_right > right.front_left


# ---------------------------------------------------------------------------
# Camera intrinsics
#
# K is where pixels become geometry. Nothing downstream can detect a
# transposed principal point or exchanged focal lengths: recoverPose and
# triangulation consume K silently and return a skewed but well-formed point
# cloud, so the image oracles see a perfect frame and the sector oracles see
# plausible bearings. It has to be pinned here or not at all.
# ---------------------------------------------------------------------------

def test_camera_matrix_puts_each_intrinsic_in_its_own_slot() -> None:
    """Deliberately distinct values, so any transpose changes the result."""
    from daic.mini_slam_detector import MiniSLAMDetector

    K = MiniSLAMDetector._build_K({
        "camera.fx_px": "600.0",
        "camera.fy_px": "500.0",
        "camera.cx_px": "320.0",
        "camera.cy_px": "240.0",
    })

    assert K[0][0] == pytest.approx(600.0), "K[0][0] must be the horizontal focal length"
    assert K[1][1] == pytest.approx(500.0), "K[1][1] must be the vertical focal length"
    assert K[0][2] == pytest.approx(320.0), "K[0][2] must be the principal point x"
    assert K[1][2] == pytest.approx(240.0), "K[1][2] must be the principal point y"
    assert K[0][1] == 0.0 and K[1][0] == 0.0, "no skew term"
    assert K[2][0] == 0.0 and K[2][1] == 0.0 and K[2][2] == pytest.approx(1.0)


def test_camera_matrix_falls_back_to_defaults_without_telemetry() -> None:
    from daic.mini_slam_detector import MiniSLAMDetector

    K = MiniSLAMDetector._build_K({})

    assert K[0][2] > K[1][2], (
        "the default principal point must keep cx (half of a 640-wide frame) "
        "larger than cy (half of 480); equal or inverted defaults would hide a "
        "transpose in every test that omits camera telemetry")


def test_camera_matrix_matches_the_published_camera_geometry() -> None:
    """K must agree with what dsim publishes, not with a hard-coded guess."""
    from dtest.deterministic import DeterministicSim
    from daic.mini_slam_detector import MiniSLAMDetector

    telemetry = DeterministicSim(heading_deg=0.0).read_telemetry()
    K = MiniSLAMDetector._build_K(telemetry)
    width = float(telemetry["camera.width_px"])
    height = float(telemetry["camera.height_px"])

    assert K[0][2] == pytest.approx(width / 2.0), (
        f"principal point x {K[0][2]} is not the centre of a {width:.0f}px frame")
    assert K[1][2] == pytest.approx(height / 2.0), (
        f"principal point y {K[1][2]} is not the centre of a {height:.0f}px frame")

    # fx is fixed by the published horizontal field of view; deriving it from
    # the FOV rather than reading camera.fx_px keeps this an independent check.
    fov_h = math.radians(float(telemetry["camera.fov_h_deg"]))
    assert K[0][0] == pytest.approx((width / 2.0) / math.tan(fov_h / 2.0), rel=1e-4)


def test_camera_matrix_projects_starboard_right_and_below_lower() -> None:
    """The projection must land in the image the way the frame is stored.

    OpenCV camera axes are x right, y down, z forward, and DVision frames are
    top-left origin. A point off the optical axis to starboard must therefore
    project right of the principal point, and one below it must project to a
    larger row.
    """
    from dtest.deterministic import DeterministicSim
    from daic.mini_slam_detector import MiniSLAMDetector

    K = MiniSLAMDetector._build_K(DeterministicSim(heading_deg=0.0).read_telemetry())
    cx, cy = K[0][2], K[1][2]

    def project(point):
        homogeneous = K @ np.asarray(point, dtype=float)
        return homogeneous[0] / homogeneous[2], homogeneous[1] / homogeneous[2]

    on_axis = project((0.0, 0.0, 4.0))
    starboard = project((1.0, 0.0, 4.0))
    below = project((0.0, 1.0, 4.0))

    assert on_axis == pytest.approx((cx, cy)), "the optical axis must hit the centre"
    assert starboard[0] > cx, "camera x+ (starboard) must project right of centre"
    assert starboard[1] == pytest.approx(cy), "a starboard point must not change row"
    assert below[1] > cy, "camera y+ (down) must project to a larger row"
    assert below[0] == pytest.approx(cx), "a low point must not change column"
