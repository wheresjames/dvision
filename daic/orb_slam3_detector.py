"""ORB_SLAM3-based obstacle detector for daic.

Runs ORB_SLAM3 in a background thread.  detect_obstacles() is non-blocking and
returns immediately with the sectors computed from the most recent SLAM state.
Sectors carry confidence=0.0 while SLAM is initialising, tracking is lost, or
the bindings are unavailable — the caller should treat low-confidence output as
a pass-through (no avoidance action).

Python bindings
---------------
Expects an ``orbslam3`` package that exposes the following interface (several
community wheels satisfy this; exact method names vary, so the module probes at
startup)::

    orbslam3.system(vocab_path, settings_path, orbslam3.Sensor.MONOCULAR)
      .set_use_viewer(False)
      .initialize()
      .process_image_mono(gray_uint8_ndarray, timestamp_s)  # or track_monocular / TrackMonocular
      .get_tracking_state()       -> int    # 2=OK 3=RECENTLY_LOST 4=LOST
      .get_frame_pose()           -> ndarray | None   # 4×4 Tcw (world→camera)
      .get_tracked_map_points()   -> ndarray | None   # N×3 world-frame XYZ
      .shutdown()

If the import fails the detector degrades silently: detect_obstacles() always
returns zero-confidence sectors so the caller falls back to its OpenCV path.

Coordinate conventions
----------------------
ORB_SLAM3 uses a right-handed camera frame: X right, Y down, Z forward.
Pose is returned as Tcw — the transform from world to camera — so:

    p_camera = Tcw[:3,:3] @ p_world + Tcw[:3,3]

Map points are in the world frame (SLAM coordinate system).

Scale
-----
Monocular SLAM has no absolute scale.  Sector depth is normalised against
SLAM_FAR_UNITS (raw SLAM depth units treated as the far-plane) until
slam_planner.py anchors the metric scale.  Call set_scale() once the scale
(metres per SLAM unit) has been confirmed.
"""

from __future__ import annotations

import queue
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np


# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ObstacleSectors:
    """Per-azimuth obstacle risk.  0.0 = clear, 1.0 = fully blocked."""
    front:       float
    front_left:  float
    front_right: float
    left:        float
    right:       float
    confidence:  float   # 0.0 when SLAM is not tracking; avoidance should pass-through
    method:      str     # e.g. "orb_slam3:ok", "orb_slam3:lost", "none"
    front_range_m:       float | None = None
    front_left_range_m:  float | None = None
    front_right_range_m: float | None = None
    left_range_m:        float | None = None
    right_range_m:       float | None = None

    def as_dict(self) -> dict:
        return {
            "front":       self.front,
            "front_left":  self.front_left,
            "front_right": self.front_right,
            "left":        self.left,
            "right":       self.right,
            "confidence":  self.confidence,
            "method":      self.method,
            "front_range_m":       self.front_range_m,
            "front_left_range_m":  self.front_left_range_m,
            "front_right_range_m": self.front_right_range_m,
            "left_range_m":        self.left_range_m,
            "right_range_m":       self.right_range_m,
        }


_NULL_SECTORS = ObstacleSectors(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "none")


# ---------------------------------------------------------------------------
# Tracking state codes (standard across ORB_SLAM3 bindings)
# ---------------------------------------------------------------------------

_S_SYSTEM_NOT_READY = -1
_S_NO_IMAGES        =  0
_S_NOT_INITIALIZED  =  1
_S_OK               =  2
_S_RECENTLY_LOST    =  3
_S_LOST             =  4

_TRACKING_LABELS: dict[int, str] = {
    _S_SYSTEM_NOT_READY: "system_not_ready",
    _S_NO_IMAGES:        "no_images",
    _S_NOT_INITIALIZED:  "not_initialized",
    _S_OK:               "ok",
    _S_RECENTLY_LOST:    "recently_lost",
    _S_LOST:             "lost",
}


# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

# Far-plane depth in raw SLAM units (used before metric scale is set).
SLAM_FAR_UNITS: float = 20.0

# Points closer than this (SLAM units) produce maximum risk contribution.
SLAM_CLOSE_UNITS: float = 2.0

# Metric far-plane and close threshold once scale is anchored (metres).
METRIC_FAR_M: float  = 10.0
METRIC_CLOSE_M: float = 1.0

# Minimum map points required to assign non-zero risk to a sector.
SLAM_MIN_POINTS: int = 5

# Sector azimuth half-widths (degrees from camera forward axis, +right).
#   ±_INNER_DEG          → front
#   ±_INNER..±_OUTER_DEG → front-left / front-right
#   ±_OUTER..±90°        → left / right
_INNER_DEG: float = 10.0
_OUTER_DEG: float = 30.0

# Elevation filter: exclude map points more than this many degrees below the
# camera forward axis to suppress ground-plane feature false positives.
# Camera Y-axis is down, so elev = atan2(-y_cam, z_cam).
_MAX_BELOW_DEG: float = 40.0

# Temporal decay factor applied to previous sectors when tracking is RECENTLY_LOST.
RISK_DECAY: float = 0.75

# ORB feature extractor parameters written into the generated settings YAML.
_ORB_N_FEATURES:   int   = 1000
_ORB_SCALE_FACTOR: float = 1.2
_ORB_N_LEVELS:     int   = 8
_ORB_INI_FAST:     int   = 20
_ORB_MIN_FAST:     int   = 7

# Max frames held in the inter-thread queue (older frames are dropped when full).
_QUEUE_SIZE: int = 2


# ---------------------------------------------------------------------------
# Internal thread-shared snapshot
# ---------------------------------------------------------------------------

@dataclass
class _SlamState:
    tracking_state: int              = _S_SYSTEM_NOT_READY
    pose:           np.ndarray | None = None   # 4×4 Tcw
    map_points:     np.ndarray | None = None   # Nx3 world frame
    n_points:       int              = 0
    timestamp:      float            = 0.0


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class ORBSLAM3Detector:
    """Wraps ORB_SLAM3 and converts its map-point output into ObstacleSectors."""

    def __init__(self,
                 vocab_path: str | Path,
                 settings_path: str | Path | None = None) -> None:
        """
        vocab_path    : Path to ORBvoc.txt (must be pre-extracted from .tar.gz).
        settings_path : Path to an existing ORB_SLAM3 YAML file.  When None,
                        start() generates one from the dsim status buffer.
        """
        self._vocab_path    = Path(vocab_path)
        self._settings_path = Path(settings_path) if settings_path else None
        self._slam: Any     = None
        self._available     = False

        # Probed method handles, filled in start().
        self._fn_track: Callable | None       = None
        self._fn_pose: Callable | None        = None
        self._fn_pts: Callable | None         = None
        self._fn_state: Callable | None       = None
        self._fn_shutdown: Callable | None    = None

        self._state_lock  = threading.Lock()
        self._slam_state  = _SlamState()
        self._sectors     = _NULL_SECTORS
        self._status_text = "not started"

        self._frame_queue: queue.Queue[tuple[np.ndarray, float]] = (
            queue.Queue(maxsize=_QUEUE_SIZE)
        )
        self._thread: threading.Thread | None = None
        self._stop_event  = threading.Event()

        self._scale: float | None = None
        self._tmp_dir: tempfile.TemporaryDirectory | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, status: dict | None = None) -> bool:
        """Initialise ORB_SLAM3 and launch the processing thread.

        status : dsim status dict (from memkv.getAll()) used to build the YAML
                 when settings_path was not provided.  Pass None or {} to use
                 640×480 70°-FOV defaults.

        Returns True when SLAM started successfully, False when the orbslam3
        package is unavailable or initialisation fails.  In both cases
        detect_obstacles() is safe to call and returns null sectors.
        """
        if not self._vocab_path.exists():
            self._status_text = f"vocab not found: {self._vocab_path}"
            print(f"orb_slam3_detector: {self._status_text}")
            return False

        settings_path = self._settings_path or self._generate_yaml(status or {})

        try:
            import orbslam3  # type: ignore[import]
            slam = orbslam3.system(
                str(self._vocab_path),
                str(settings_path),
                orbslam3.Sensor.MONOCULAR,
            )
            slam.set_use_viewer(False)
            slam.initialize()
        except Exception as exc:
            self._status_text = f"orbslam3 unavailable: {exc}"
            print(f"orb_slam3_detector: {self._status_text}")
            return False

        self._slam = slam
        self._fn_track, self._fn_state, self._fn_pose, self._fn_pts, self._fn_shutdown = (
            _probe_api(slam)
        )
        if self._fn_track is None:
            self._status_text = "no tracking method found on orbslam3.system"
            print(f"orb_slam3_detector: {self._status_text}")
            self._slam = None
            return False

        self._available = True
        self._status_text = "initialising"
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._slam_thread, daemon=True, name="orb_slam3"
        )
        self._thread.start()
        return True

    def stop(self) -> None:
        """Shut down the SLAM thread and free resources."""
        self._available = False
        self._stop_event.set()
        if self._thread is not None:
            try:
                self._frame_queue.put_nowait(
                    (np.zeros((1, 1), dtype=np.uint8), -1.0)
                )
            except queue.Full:
                pass
            self._thread.join(timeout=4.0)
            self._thread = None
        if self._slam is not None and self._fn_shutdown is not None:
            try:
                self._fn_shutdown()
            except Exception:
                pass
            self._slam = None
        if self._tmp_dir is not None:
            self._tmp_dir.cleanup()
            self._tmp_dir = None
        self._status_text = "stopped"

    def set_scale(self, metres_per_slam_unit: float) -> None:
        """Provide a confirmed metric scale from slam_planner.py."""
        if metres_per_slam_unit > 0.0:
            self._scale = metres_per_slam_unit

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def detect_obstacles(self, frame_rgb: np.ndarray) -> ObstacleSectors:
        """Feed a new RGB frame and return the current obstacle sectors.

        Non-blocking: the frame is queued for the SLAM thread; this method
        returns immediately with the most recently computed sectors (at most
        one frame stale).
        """
        if not self._available:
            return _NULL_SECTORS

        ts   = time.monotonic()
        gray = _to_gray(frame_rgb)

        # Drop oldest frame if queue is full — we prefer fresh frames.
        try:
            self._frame_queue.put_nowait((gray, ts))
        except queue.Full:
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._frame_queue.put_nowait((gray, ts))
            except queue.Full:
                pass

        with self._state_lock:
            return self._sectors

    # ------------------------------------------------------------------
    # Properties for health reporting
    # ------------------------------------------------------------------

    @property
    def tracking_state(self) -> int:
        with self._state_lock:
            return self._slam_state.tracking_state

    @property
    def n_map_points(self) -> int:
        with self._state_lock:
            return self._slam_state.n_points

    @property
    def scale(self) -> float | None:
        return self._scale

    @property
    def status_text(self) -> str:
        return self._status_text

    def get_map_snapshot(self) -> tuple[np.ndarray | None, np.ndarray | None, int]:
        """Return (pose, map_points, tracking_state) as thread-safe copies.

        pose       : 4×4 Tcw float64 ndarray, or None when not tracking.
        map_points : Nx3 float64 ndarray of world-frame positions, or None.
        tracking_state : one of the _S_* constants.
        """
        with self._state_lock:
            st  = self._slam_state
            pose = st.pose.copy()       if st.pose       is not None else None
            pts  = st.map_points.copy() if st.map_points is not None else None
            return pose, pts, st.tracking_state

    # ------------------------------------------------------------------
    # SLAM background thread
    # ------------------------------------------------------------------

    def _slam_thread(self) -> None:
        while not self._stop_event.is_set():
            try:
                gray, ts = self._frame_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if ts < 0:  # stop() sentinel
                break

            try:
                self._fn_track(gray, ts)
                state    = self._fn_state() if self._fn_state else _S_SYSTEM_NOT_READY
                pose     = self._fn_pose()  if self._fn_pose  else None
                pts_raw  = self._fn_pts()   if self._fn_pts   else None
            except Exception as exc:
                self._status_text = f"slam error: {exc}"
                with self._state_lock:
                    self._slam_state.tracking_state = _S_LOST
                    self._sectors = _NULL_SECTORS
                continue

            map_pts = _parse_map_points(pts_raw)
            n_pts   = len(map_pts) if map_pts is not None else 0

            new_state = _SlamState(
                tracking_state=state,
                pose=pose,
                map_points=map_pts,
                n_points=n_pts,
                timestamp=ts,
            )

            # Read previous sectors before acquiring the lock for the update.
            with self._state_lock:
                prev_sectors = self._sectors

            sectors = self._sectors_from_state(new_state, prev_sectors)

            with self._state_lock:
                self._slam_state = new_state
                self._sectors    = sectors

            self._status_text = (
                f"{_TRACKING_LABELS.get(state, str(state))} pts={n_pts}"
            )

    # ------------------------------------------------------------------
    # Sector computation
    # ------------------------------------------------------------------

    def _sectors_from_state(self, state: _SlamState,
                            prev: ObstacleSectors) -> ObstacleSectors:
        code  = state.tracking_state
        label = _TRACKING_LABELS.get(code, f"state_{code}")
        method = f"orb_slam3:{label}"

        # No data yet — return silent zeros.
        if code in (_S_SYSTEM_NOT_READY, _S_NO_IMAGES, _S_NOT_INITIALIZED):
            return ObstacleSectors(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, method)

        # Fully lost — zero with zero confidence.
        if code == _S_LOST:
            return ObstacleSectors(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, method)

        pose    = state.pose
        map_pts = state.map_points

        # Pose or points missing — decay previous sectors.
        if pose is None or map_pts is None or len(map_pts) == 0:
            conf = 0.3 if code == _S_RECENTLY_LOST else 0.5
            return _decay(prev, RISK_DECAY, method, conf)

        current = self._project_sectors(pose, map_pts, method)

        # RECENTLY_LOST: blend fresh sectors with decayed history at low confidence.
        if code == _S_RECENTLY_LOST:
            return _blend(prev, current, RISK_DECAY, method, 0.4)

        return current

    def _project_sectors(self, pose: np.ndarray,
                         map_pts: np.ndarray,
                         method: str) -> ObstacleSectors:
        """Project world-frame map points into camera-frame sectors.

        pose    : 4×4 Tcw matrix (world→camera).
        map_pts : Nx3 float64 world-frame positions.
        """
        # Transform to camera frame: p_cam = R * p_world + t
        R = pose[:3, :3]
        t = pose[:3, 3]
        pts_cam = map_pts @ R.T + t      # Nx3

        x_c = pts_cam[:, 0]
        y_c = pts_cam[:, 1]
        z_c = pts_cam[:, 2]

        # Forward hemisphere only.
        fwd  = z_c > 0.05
        x_c  = x_c[fwd]
        y_c  = y_c[fwd]
        z_c  = z_c[fwd]

        if len(z_c) == 0:
            return ObstacleSectors(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, method)

        # Elevation filter: suppress ground-plane map points.
        # Camera Y is down, so elevation above forward = atan2(-y, z).
        elev_deg = np.degrees(np.arctan2(-y_c, z_c))
        above    = elev_deg > -_MAX_BELOW_DEG
        x_c      = x_c[above]
        z_c      = z_c[above]

        if len(z_c) == 0:
            return ObstacleSectors(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, method)

        # Depth normalisation: risk = 1 at close threshold, 0 at far plane.
        if self._scale is not None:
            far_d   = METRIC_FAR_M   / self._scale
            close_d = METRIC_CLOSE_M / self._scale
        else:
            far_d   = SLAM_FAR_UNITS
            close_d = SLAM_CLOSE_UNITS

        span         = max(far_d - close_d, 1e-6)
        depth_norm   = np.clip((z_c - close_d) / span, 0.0, 1.0)
        risk_per_pt  = 1.0 - depth_norm   # close → 1.0, far → 0.0

        # Azimuth angles (degrees, +right).
        az_deg = np.degrees(np.arctan2(x_c, z_c))

        def _sector_risk(lo: float, hi: float) -> float:
            mask = (az_deg >= lo) & (az_deg < hi)
            n    = int(mask.sum())
            if n < SLAM_MIN_POINTS:
                return 0.0
            risks = risk_per_pt[mask]
            # Mean of the top quartile: robust to single stray noisy points
            # yet responsive to genuine clusters.
            top_n = max(1, n // 4)
            return float(np.sort(risks)[-top_n:].mean())

        return ObstacleSectors(
            front       = _sector_risk(-_INNER_DEG,  _INNER_DEG),
            front_left  = _sector_risk(-_OUTER_DEG, -_INNER_DEG),
            front_right = _sector_risk( _INNER_DEG,  _OUTER_DEG),
            left        = _sector_risk(-90.0,        -_OUTER_DEG),
            right       = _sector_risk( _OUTER_DEG,   90.0),
            confidence  = 1.0,
            method      = method,
        )

    # ------------------------------------------------------------------
    # Settings YAML generation
    # ------------------------------------------------------------------

    def _generate_yaml(self, status: dict) -> Path:
        """Write a temporary ORB_SLAM3 settings YAML derived from dsim camera status."""
        fx  = _f(status, "camera.fx_px",     457.0)
        fy  = _f(status, "camera.fy_px",     457.0)
        cx  = _f(status, "camera.cx_px",     320.0)
        cy  = _f(status, "camera.cy_px",     240.0)
        w   = int(_f(status, "camera.width_px",  640))
        h   = int(_f(status, "camera.height_px", 480))
        fps = int(_f(status, "camera.fps",        30))

        # ORB_SLAM3 YAML must begin with the %YAML:1.0 header.
        yaml_text = "\n".join([
            "%YAML:1.0",
            "---",
            "",
            'Camera.type: "PinHole"',
            f"Camera.fx: {fx:.6f}",
            f"Camera.fy: {fy:.6f}",
            f"Camera.cx: {cx:.6f}",
            f"Camera.cy: {cy:.6f}",
            "Camera.k1: 0.0",
            "Camera.k2: 0.0",
            "Camera.p1: 0.0",
            "Camera.p2: 0.0",
            f"Camera.width:  {w}",
            f"Camera.height: {h}",
            f"Camera.fps:    {fps}",
            "Camera.RGB: 1",
            "",
            f"ORBextractor.nFeatures:   {_ORB_N_FEATURES}",
            f"ORBextractor.scaleFactor: {_ORB_SCALE_FACTOR}",
            f"ORBextractor.nLevels:     {_ORB_N_LEVELS}",
            f"ORBextractor.iniThFAST:   {_ORB_INI_FAST}",
            f"ORBextractor.minThFAST:   {_ORB_MIN_FAST}",
            "",
            # Viewer section required even when set_use_viewer(False).
            "Viewer.KeyFrameSize: 0.05",
            "Viewer.KeyFrameLineWidth: 1",
            "Viewer.GraphLineWidth: 0.9",
            "Viewer.PointSize: 2",
            "Viewer.CameraSize: 0.08",
            "Viewer.CameraLineWidth: 3",
            "Viewer.ViewpointX: 0",
            "Viewer.ViewpointY: -0.7",
            "Viewer.ViewpointZ: -3.5",
            "Viewer.ViewpointF: 500",
        ])

        self._tmp_dir = tempfile.TemporaryDirectory(prefix="daic_slam_")
        out_path = Path(self._tmp_dir.name) / "orb_slam3_sim.yaml"
        out_path.write_text(yaml_text, encoding="utf-8")
        return out_path


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _probe_api(slam: Any) -> tuple[
    Callable | None,   # track
    Callable | None,   # get_tracking_state
    Callable | None,   # get_frame_pose
    Callable | None,   # get_tracked_map_points
    Callable | None,   # shutdown
]:
    """Probe a slam object for the method names used by various binding packages."""
    track = (
        getattr(slam, "process_image_mono", None)
        or getattr(slam, "track_monocular",  None)
        or getattr(slam, "TrackMonocular",   None)
    )
    state = (
        getattr(slam, "get_tracking_state",    None)
        or getattr(slam, "GetTrackingState",   None)
    )
    pose = (
        getattr(slam, "get_frame_pose",        None)
        or getattr(slam, "GetCurrentPose",     None)
        or getattr(slam, "get_current_pose",   None)
    )
    pts = (
        getattr(slam, "get_tracked_map_points",  None)
        or getattr(slam, "GetTrackedMapPoints",  None)
        or getattr(slam, "get_map_points",       None)
    )
    shutdown = (
        getattr(slam, "shutdown",  None)
        or getattr(slam, "Shutdown", None)
    )
    return track, state, pose, pts, shutdown


def _to_gray(frame_rgb: np.ndarray) -> np.ndarray:
    import cv2
    bgr = np.ascontiguousarray(frame_rgb[:, :, ::-1])
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def _parse_map_points(raw: Any) -> np.ndarray | None:
    """Normalise whatever the binding returns to an Nx3 float64 array or None."""
    if raw is None:
        return None
    if isinstance(raw, np.ndarray):
        if raw.ndim == 2 and raw.shape[1] == 3 and raw.shape[0] > 0:
            return raw.astype(np.float64, copy=False)
        return None
    # List of [x,y,z] sequences.
    try:
        arr = np.array([[p[0], p[1], p[2]] for p in raw], dtype=np.float64)
        return arr if arr.shape[0] > 0 else None
    except (TypeError, IndexError):
        pass
    # MapPoint objects with a GetWorldPos() method (some C++ bindings).
    try:
        arr = np.array([list(p.GetWorldPos()) for p in raw], dtype=np.float64)
        return arr if arr.shape[0] > 0 else None
    except Exception:
        return None


def _f(d: dict, key: str, default: float) -> float:
    try:
        return float(d.get(key, default))
    except (TypeError, ValueError):
        return default


def _decay(prev: ObstacleSectors, factor: float,
           method: str, confidence: float) -> ObstacleSectors:
    return ObstacleSectors(
        front       = prev.front       * factor,
        front_left  = prev.front_left  * factor,
        front_right = prev.front_right * factor,
        left        = prev.left        * factor,
        right       = prev.right       * factor,
        confidence  = confidence,
        method      = method,
    )


def _blend(prev: ObstacleSectors, curr: ObstacleSectors,
           prev_weight: float, method: str, confidence: float) -> ObstacleSectors:
    """Max-blend of current sectors with decayed previous sectors."""
    w = prev_weight
    return ObstacleSectors(
        front       = max(curr.front,       prev.front       * w),
        front_left  = max(curr.front_left,  prev.front_left  * w),
        front_right = max(curr.front_right, prev.front_right * w),
        left        = max(curr.left,        prev.left        * w),
        right       = max(curr.right,       prev.right       * w),
        confidence  = confidence,
        method      = method,
    )
