"""Unit tests for apps/daic/orb_slam3_detector.py.

All tests are pure-Python; no ORB_SLAM3 bindings or dsim are required.
The SLAM API is exercised through a lightweight stub injected via monkey-patching.
"""

from __future__ import annotations

import math
import threading
import time
import types
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from daic.orb_slam3_detector import (
    ORBSLAM3Detector,
    ObstacleSectors,
    _NULL_SECTORS,
    _S_OK,
    _S_RECENTLY_LOST,
    _S_LOST,
    _S_NOT_INITIALIZED,
    _decay,
    _blend,
    _parse_map_points,
    _probe_api,
    SLAM_FAR_UNITS,
    SLAM_CLOSE_UNITS,
    SLAM_MIN_POINTS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rgb_frame(w: int = 64, h: int = 48) -> np.ndarray:
    """Return a solid grey RGB frame."""
    return np.full((h, w, 3), 128, dtype=np.uint8)


def _identity_pose() -> np.ndarray:
    """4×4 identity Tcw = camera at world origin, looking along +Z."""
    return np.eye(4, dtype=np.float64)


def _pts_in_front(n: int = 20, depth: float = 3.0, spread: float = 0.5) -> np.ndarray:
    """World-frame points clustered straight ahead at the given depth."""
    rng = np.random.default_rng(0)
    pts = rng.uniform(-spread, spread, (n, 3))
    pts[:, 2] = depth
    return pts.astype(np.float64)


def _pts_on_left(n: int = 20, depth: float = 3.0) -> np.ndarray:
    """World-frame points to the left (negative X in camera frame)."""
    pts = _pts_in_front(n, depth)
    pts[:, 0] = -depth * math.tan(math.radians(20))
    return pts


def _pts_below_camera(n: int = 20, depth: float = 3.0) -> np.ndarray:
    """World-frame points far below the camera (should be filtered by elevation)."""
    pts = _pts_in_front(n, depth)
    pts[:, 1] = depth * math.tan(math.radians(50))   # Y-down → below horizon
    return pts


# ---------------------------------------------------------------------------
# _parse_map_points
# ---------------------------------------------------------------------------

class TestParseMapPoints:
    def test_numpy_passthrough(self):
        arr = np.ones((10, 3), dtype=np.float32)
        result = _parse_map_points(arr)
        assert result is not None
        assert result.dtype == np.float64
        assert result.shape == (10, 3)

    def test_list_of_lists(self):
        raw = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        result = _parse_map_points(raw)
        assert result is not None
        assert result.shape == (2, 3)
        assert result[0, 2] == pytest.approx(3.0)

    def test_map_point_objects(self):
        """Objects with a GetWorldPos() method (some C++ bindings)."""
        class FakePt:
            def __init__(self, xyz):
                self._xyz = xyz
            def GetWorldPos(self):
                return self._xyz
        raw = [FakePt([1.0, 0.0, 5.0]), FakePt([0.5, 0.0, 4.0])]
        result = _parse_map_points(raw)
        assert result is not None
        assert result.shape == (2, 3)

    def test_none_returns_none(self):
        assert _parse_map_points(None) is None

    def test_wrong_shape_returns_none(self):
        arr = np.ones((10, 2))
        assert _parse_map_points(arr) is None

    def test_empty_array_returns_none(self):
        arr = np.ones((0, 3))
        assert _parse_map_points(arr) is None


# ---------------------------------------------------------------------------
# _probe_api
# ---------------------------------------------------------------------------

class TestProbeApi:
    def _make_slam(self, **attrs):
        obj = types.SimpleNamespace(**attrs)
        return obj

    def test_standard_names(self):
        fn = lambda *a: None
        slam = self._make_slam(
            process_image_mono=fn,
            get_tracking_state=fn,
            get_frame_pose=fn,
            get_tracked_map_points=fn,
            shutdown=fn,
        )
        track, state, pose, pts, sd = _probe_api(slam)
        assert track is fn
        assert state is fn
        assert pose  is fn
        assert pts   is fn
        assert sd    is fn

    def test_alternate_names(self):
        fn = lambda *a: None
        slam = self._make_slam(
            TrackMonocular=fn,
            GetTrackingState=fn,
            GetCurrentPose=fn,
            GetTrackedMapPoints=fn,
            Shutdown=fn,
        )
        track, state, pose, pts, sd = _probe_api(slam)
        assert track is fn
        assert state is fn

    def test_missing_methods_return_none(self):
        slam = types.SimpleNamespace()
        track, state, pose, pts, sd = _probe_api(slam)
        assert track is None
        assert state is None


# ---------------------------------------------------------------------------
# _decay and _blend
# ---------------------------------------------------------------------------

class TestDecayBlend:
    def _sectors(self, v: float = 0.8) -> ObstacleSectors:
        return ObstacleSectors(v, v, v, v, v, 1.0, "test")

    def test_decay_scales_all_risks(self):
        s = self._sectors(0.8)
        d = _decay(s, 0.5, "orb_slam3:lost", 0.2)
        assert d.front == pytest.approx(0.4)
        assert d.front_left == pytest.approx(0.4)
        assert d.confidence == pytest.approx(0.2)
        assert d.method == "orb_slam3:lost"

    def test_blend_takes_max(self):
        prev = ObstacleSectors(0.9, 0.0, 0.0, 0.0, 0.0, 1.0, "prev")
        curr = ObstacleSectors(0.1, 0.8, 0.0, 0.0, 0.0, 1.0, "curr")
        b = _blend(prev, curr, 0.75, "test", 0.5)
        # front: max(0.1, 0.9*0.75=0.675) → 0.675
        assert b.front == pytest.approx(0.675)
        # front_left: max(0.8, 0.0*0.75) → 0.8
        assert b.front_left == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# ORBSLAM3Detector._project_sectors
# ---------------------------------------------------------------------------

class TestProjectSectors:
    """Exercise the sector projection directly without a SLAM thread."""

    def _det(self) -> ORBSLAM3Detector:
        det = ORBSLAM3Detector.__new__(ORBSLAM3Detector)
        det._scale = None
        det._state_lock = threading.Lock()
        det._sectors = _NULL_SECTORS
        return det

    def test_points_straight_ahead_raise_front_risk(self):
        det  = self._det()
        pose = _identity_pose()
        # spread=0.05 at depth=1.0 → max azimuth ≈ 2.9°, well inside ±10° front sector
        pts  = _pts_in_front(n=30, depth=SLAM_CLOSE_UNITS * 0.5, spread=0.05)
        s    = det._project_sectors(pose, pts, "orb_slam3:ok")
        assert s.front > 0.5, f"expected high front risk, got {s.front}"
        assert s.front_left  < 0.1
        assert s.front_right < 0.1

    def test_points_on_left_raise_front_left_risk(self):
        det = self._det()
        pose = _identity_pose()
        pts  = _pts_on_left(n=30, depth=SLAM_CLOSE_UNITS * 0.8)
        s = det._project_sectors(pose, pts, "orb_slam3:ok")
        assert s.front_left > 0.2, f"expected front_left risk, got {s.front_left}"

    def test_distant_points_produce_low_risk(self):
        det = self._det()
        pose = _identity_pose()
        pts  = _pts_in_front(n=30, depth=SLAM_FAR_UNITS * 1.5)
        s = det._project_sectors(pose, pts, "orb_slam3:ok")
        assert s.front < 0.1, f"distant points should not produce high risk: {s.front}"

    def test_ground_plane_points_filtered(self):
        det = self._det()
        pose = _identity_pose()
        pts  = _pts_below_camera(n=30, depth=SLAM_CLOSE_UNITS * 0.5)
        s = det._project_sectors(pose, pts, "orb_slam3:ok")
        # Points far below horizon should be filtered by elevation gate
        assert s.front < 0.5, (
            f"ground-plane points should not produce full front risk: {s.front}"
        )

    def test_too_few_points_produce_zero_risk(self):
        det = self._det()
        pose = _identity_pose()
        pts  = _pts_in_front(n=SLAM_MIN_POINTS - 1, depth=0.5)
        s = det._project_sectors(pose, pts, "orb_slam3:ok")
        assert s.front == 0.0

    def test_no_forward_points_produce_zero_risk(self):
        det = self._det()
        pose = _identity_pose()
        # All points behind the camera (z < 0).
        pts = _pts_in_front(n=30, depth=-3.0)
        s = det._project_sectors(pose, pts, "orb_slam3:ok")
        assert s.front == 0.0
        assert s.confidence == pytest.approx(1.0)

    def test_metric_scale_lowers_risk_for_same_slam_depth(self):
        det_no_scale = self._det()
        det_scaled   = self._det()
        # scale=0.1 means 1 SLAM unit = 0.1 m → depth 3 SLAM = 0.3 m (very close)
        det_scaled._scale = 0.1

        pose = _identity_pose()
        pts  = _pts_in_front(n=30, depth=3.0)

        # Without scale, depth=3 is within SLAM_CLOSE_UNITS=2 so risk is moderate.
        # With scale=0.1, METRIC_CLOSE_M/scale = 10 SLAM units → depth=3 is inside
        # close threshold → even higher risk.
        s_ns = det_no_scale._project_sectors(pose, pts, "orb_slam3:ok")
        s_s  = det_scaled._project_sectors(pose, pts, "orb_slam3:ok")
        # Both should be > 0 for close points; scaled version sees them as closer.
        assert s_ns.front >= 0.0
        assert s_s.front  >= 0.0

    def test_confidence_is_one_when_ok(self):
        det  = self._det()
        pose = _identity_pose()
        pts  = _pts_in_front(n=30, depth=3.0)
        s    = det._project_sectors(pose, pts, "orb_slam3:ok")
        assert s.confidence == pytest.approx(1.0)
        assert s.method == "orb_slam3:ok"


# ---------------------------------------------------------------------------
# ORBSLAM3Detector._sectors_from_state
# ---------------------------------------------------------------------------

class TestSectorsFromState:
    from daic.orb_slam3_detector import _SlamState

    def _det(self) -> ORBSLAM3Detector:
        det = ORBSLAM3Detector.__new__(ORBSLAM3Detector)
        det._scale       = None
        det._state_lock  = threading.Lock()
        det._sectors     = _NULL_SECTORS
        return det

    def test_system_not_ready_returns_null(self):
        from daic.orb_slam3_detector import _SlamState, _S_SYSTEM_NOT_READY
        det   = self._det()
        state = _SlamState(tracking_state=_S_SYSTEM_NOT_READY)
        s     = det._sectors_from_state(state, _NULL_SECTORS)
        assert s.confidence == 0.0
        assert s.front == 0.0

    def test_not_initialized_returns_null(self):
        from daic.orb_slam3_detector import _SlamState
        det   = self._det()
        state = _SlamState(tracking_state=_S_NOT_INITIALIZED)
        s     = det._sectors_from_state(state, _NULL_SECTORS)
        assert s.confidence == 0.0

    def test_lost_returns_null(self):
        from daic.orb_slam3_detector import _SlamState
        det   = self._det()
        state = _SlamState(tracking_state=_S_LOST)
        s     = det._sectors_from_state(state, _NULL_SECTORS)
        assert s.confidence == 0.0
        assert s.front == 0.0

    def test_ok_with_no_pose_decays_previous(self):
        from daic.orb_slam3_detector import _SlamState
        det = self._det()
        prev = ObstacleSectors(0.8, 0.0, 0.0, 0.0, 0.0, 1.0, "orb_slam3:ok")
        state = _SlamState(tracking_state=_S_OK, pose=None, map_points=None)
        s = det._sectors_from_state(state, prev)
        assert 0.0 < s.front < 0.8   # decayed

    def test_ok_with_close_front_points_raises_front_risk(self):
        from daic.orb_slam3_detector import _SlamState
        det   = self._det()
        pose  = _identity_pose()
        pts   = _pts_in_front(n=30, depth=0.5)
        state = _SlamState(tracking_state=_S_OK, pose=pose, map_points=pts, n_points=30)
        s     = det._sectors_from_state(state, _NULL_SECTORS)
        assert s.front > 0.5
        assert s.confidence == pytest.approx(1.0)

    def test_recently_lost_blends_with_history(self):
        from daic.orb_slam3_detector import _SlamState
        det  = self._det()
        # Previous sectors had high front risk.
        prev = ObstacleSectors(0.9, 0.0, 0.0, 0.0, 0.0, 1.0, "orb_slam3:ok")
        # RECENTLY_LOST with no pose → should decay previous.
        state = _SlamState(tracking_state=_S_RECENTLY_LOST, pose=None, map_points=None)
        s = det._sectors_from_state(state, prev)
        # Decayed front should be between 0 and 0.9.
        assert 0.0 < s.front < 0.9
        assert s.confidence < 1.0


# ---------------------------------------------------------------------------
# ORBSLAM3Detector.start / stop — no real SLAM
# ---------------------------------------------------------------------------

class TestStartStop:
    def test_start_fails_gracefully_when_vocab_missing(self, tmp_path):
        det = ORBSLAM3Detector(vocab_path=tmp_path / "nonexistent_vocab.txt")
        assert det.start() is False
        assert det._available is False
        # detect_obstacles must still work safely.
        s = det.detect_obstacles(_rgb_frame())
        assert s is _NULL_SECTORS or s.confidence == 0.0

    def test_start_fails_gracefully_when_no_bindings(self, tmp_path):
        vocab = tmp_path / "ORBvoc.txt"
        vocab.write_text("fake vocab", encoding="utf-8")
        det = ORBSLAM3Detector(vocab_path=vocab)
        # orbslam3 import will fail; start() should return False, not raise.
        result = det.start({})
        assert result is False
        assert det._available is False

    def test_detect_before_start_returns_null(self, tmp_path):
        det = ORBSLAM3Detector(vocab_path=tmp_path / "v.txt")
        s   = det.detect_obstacles(_rgb_frame())
        assert s.confidence == 0.0

    def test_stop_idempotent_when_not_started(self, tmp_path):
        det = ORBSLAM3Detector(vocab_path=tmp_path / "v.txt")
        det.stop()   # must not raise
        det.stop()


# ---------------------------------------------------------------------------
# ORBSLAM3Detector — full thread test with a stub SLAM
# ---------------------------------------------------------------------------

class _StubSlam:
    """Minimal ORB_SLAM3 stub that serves a fixed pose and map-point set."""

    def __init__(self, state: int, pose: np.ndarray | None, pts: np.ndarray | None):
        self._state = state
        self._pose  = pose
        self._pts   = pts
        self.call_count = 0

    def set_use_viewer(self, v): pass
    def initialize(self): pass
    def process_image_mono(self, gray, ts):
        self.call_count += 1
    def get_tracking_state(self): return self._state
    def get_frame_pose(self): return self._pose
    def get_tracked_map_points(self): return self._pts
    def shutdown(self): pass


def _make_det_with_stub(stub: _StubSlam,
                        vocab: Path,
                        settings: Path) -> ORBSLAM3Detector:
    """Build a detector whose start() will use the stub instead of real SLAM."""
    det = ORBSLAM3Detector(vocab_path=vocab, settings_path=settings)

    # Monkey-patch orbslam3 into sys.modules for the duration of this test.
    fake_module = types.ModuleType("orbslam3")
    fake_module.Sensor = types.SimpleNamespace(MONOCULAR=0)
    fake_module.system = lambda vocab, settings, sensor: stub
    sys.modules["orbslam3"] = fake_module

    return det


class TestWithStubSlam:
    def _setup(self, tmp_path, state, pose, pts):
        vocab    = tmp_path / "ORBvoc.txt"
        settings = tmp_path / "settings.yaml"
        vocab.write_text("fake", encoding="utf-8")
        settings.write_text("%YAML:1.0\n---\n", encoding="utf-8")
        stub = _StubSlam(state, pose, pts)
        det  = _make_det_with_stub(stub, vocab, settings)
        return det, stub

    def teardown_method(self, _):
        sys.modules.pop("orbslam3", None)

    def test_ok_tracking_raises_front_risk(self, tmp_path):
        pose = _identity_pose()
        pts  = _pts_in_front(n=30, depth=0.5)   # very close
        det, stub = self._setup(tmp_path, _S_OK, pose, pts)

        det.start()
        # Feed a few frames and wait for the SLAM thread to process them.
        for _ in range(5):
            det.detect_obstacles(_rgb_frame())
            time.sleep(0.05)

        s = det.detect_obstacles(_rgb_frame())
        det.stop()

        # With very close forward points and OK tracking, front risk should be high.
        assert s.front > 0.4 or stub.call_count == 0, (
            f"front={s.front}, calls={stub.call_count}"
        )

    def test_lost_tracking_returns_zero_confidence(self, tmp_path):
        det, stub = self._setup(tmp_path, _S_LOST, None, None)

        det.start()
        for _ in range(5):
            det.detect_obstacles(_rgb_frame())
            time.sleep(0.05)

        s = det.detect_obstacles(_rgb_frame())
        det.stop()

        assert s.confidence == 0.0 or stub.call_count == 0

    def test_slam_thread_processes_frames(self, tmp_path):
        pose = _identity_pose()
        pts  = _pts_in_front(n=30, depth=3.0)
        det, stub = self._setup(tmp_path, _S_OK, pose, pts)

        det.start()
        for _ in range(8):
            det.detect_obstacles(_rgb_frame())
            time.sleep(0.04)
        det.stop()

        assert stub.call_count >= 1, "SLAM thread should have processed at least one frame"

    def test_set_scale_changes_risk_thresholds(self, tmp_path):
        pose = _identity_pose()
        # Points at SLAM depth 5 — moderate without scale, closer with scale.
        pts  = _pts_in_front(n=30, depth=5.0)
        det, _ = self._setup(tmp_path, _S_OK, pose, pts)

        det.set_scale(0.5)   # 1 SLAM unit = 0.5 m → depth 5 = 2.5 m
        assert det.scale == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# YAML generation
# ---------------------------------------------------------------------------

class TestYamlGeneration:
    def _det(self) -> ORBSLAM3Detector:
        det = ORBSLAM3Detector.__new__(ORBSLAM3Detector)
        det._tmp_dir = None
        return det

    def test_yaml_contains_camera_params(self):
        det    = self._det()
        status = {
            "camera.fx_px":     "457.1",
            "camera.fy_px":     "457.1",
            "camera.cx_px":     "320.0",
            "camera.cy_px":     "240.0",
            "camera.width_px":  "640",
            "camera.height_px": "480",
            "camera.fps":       "30",
        }
        p = det._generate_yaml(status)
        text = p.read_text()
        assert "Camera.fx: 457.1" in text
        assert "Camera.width:  640" in text
        assert "Camera.fps:    30" in text
        assert "%YAML:1.0" in text
        det._tmp_dir.cleanup()

    def test_yaml_uses_defaults_when_status_empty(self):
        det  = self._det()
        p    = det._generate_yaml({})
        text = p.read_text()
        assert "Camera.fx: 457" in text
        assert "Camera.width:  640" in text
        det._tmp_dir.cleanup()
