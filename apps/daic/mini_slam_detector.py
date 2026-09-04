"""Pure-OpenCV monocular visual odometry obstacle detector.

Requires only OpenCV and NumPy — no external SLAM library.  Provides the same
interface as ORBSLAM3Detector so the daic UI and avoidance layer work without
any changes.

Algorithm — sliding-window triangulation
-----------------------------------------
A ring buffer holds the last WINDOW grayscale frames.  Each new frame is
matched against the oldest frame in the buffer (the "reference") using ORB
descriptor matching.  The Essential Matrix is estimated with RANSAC to recover
the relative pose (R, t) between reference and current.  Inlier correspondences
are triangulated to produce a local point cloud in the reference camera frame.

That pose (Tcw reference→current) and point cloud are stored as the SLAM state
and consumed by the sector-risk projection (inherited from ORBSLAM3Detector's
logic).

Limitations vs ORB_SLAM3
-------------------------
- No loop closure; map is local to the current sliding window.
- Scale is arbitrary (unit-norm translation per window step) until set_scale()
  is called by slam_planner.
- No keyframe reuse or global bundle adjustment.

These limitations are acceptable for the reactive obstacle-avoidance use case.
"""

from __future__ import annotations

import collections
import queue
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

from daic.orb_slam3_detector import (
    ObstacleSectors, _NULL_SECTORS,
    _S_SYSTEM_NOT_READY, _S_NOT_INITIALIZED, _S_OK, _S_RECENTLY_LOST,
    _decay, _f,
    SLAM_FAR_UNITS, SLAM_CLOSE_UNITS, SLAM_MIN_POINTS,
    _INNER_DEG, _OUTER_DEG, _MAX_BELOW_DEG, RISK_DECAY,
)


# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

# Ring buffer size: reference frame is WINDOW frames behind the current.
# At 30 fps → ~0.33 s baseline, giving 10–30 cm at typical drone speeds.
WINDOW: int = 10

# Minimum number of descriptor matches to attempt pose estimation.
MIN_MATCHES: int = 40

# Minimum inliers after RANSAC Essential Matrix estimation.
MIN_INLIERS: int = 25

# Minimum mean pixel disparity between reference and current matched points.
# Below this the baseline is too small for reliable triangulation.
MIN_DISPARITY_PX: float = 5.0

# Discard triangulated points farther than this multiple of the median depth.
MAX_DEPTH_MULT: float = 30.0

# Number of ORB features to detect per frame.
ORB_N_FEATURES: int = 800

# Lowe ratio-test threshold for ORB descriptor matching.
MATCH_RATIO: float = 0.75


# ---------------------------------------------------------------------------
# Internal state snapshot
# ---------------------------------------------------------------------------

@dataclass
class _VO_State:
    tracking_state: int               = _S_SYSTEM_NOT_READY
    pose:           np.ndarray | None = None   # 4×4 Tcw (reference→camera)
    map_points:     np.ndarray | None = None   # Nx3 in reference camera frame
    n_points:       int               = 0
    n_inliers:      int               = 0
    disparity_px:   float             = 0.0


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class MiniSLAMDetector:
    """Minimal visual odometry with the same public API as ORBSLAM3Detector."""

    def __init__(self) -> None:
        self._K: np.ndarray | None = None     # 3×3 camera intrinsic matrix

        self._orb  = cv2.ORB_create(nfeatures=ORB_N_FEATURES,
                                     fastThreshold=15, edgeThreshold=15)
        self._bf   = cv2.BFMatcher(cv2.NORM_HAMMING)

        # Ring buffer: each entry is (gray_frame, keypoints_xy, descriptors)
        self._buf: collections.deque = collections.deque(maxlen=WINDOW)

        self._state_lock  = threading.Lock()
        self._vo_state    = _VO_State()
        self._sectors     = _NULL_SECTORS
        self._status_text = "not started"

        self._frame_queue: queue.Queue[tuple[np.ndarray, float]] = (
            queue.Queue(maxsize=2)
        )
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        self._scale: float | None = None
        self._available = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, status: dict | None = None) -> bool:
        """Initialise camera matrix from dsim status and launch the thread."""
        self._K = self._build_K(status or {})
        self._available = True
        self._status_text = "initialising"
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._worker, daemon=True,
                                        name="mini_slam")
        self._thread.start()
        return True

    def stop(self) -> None:
        self._available = False
        self._stop_event.set()
        if self._thread is not None:
            try:
                self._frame_queue.put_nowait(
                    (np.zeros((1, 1), dtype=np.uint8), -1.0)
                )
            except queue.Full:
                pass
            self._thread.join(timeout=3.0)
            self._thread = None

    def set_scale(self, metres_per_unit: float) -> None:
        if metres_per_unit > 0.0:
            self._scale = metres_per_unit

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_obstacles(self, frame_rgb: np.ndarray) -> ObstacleSectors:
        """Queue the frame and return the most recent obstacle sectors."""
        if not self._available:
            return _NULL_SECTORS

        # Queued alongside the frame for symmetry with the ORB_SLAM3 wrapper;
        # this detector's worker never reads it, so it stays a wall stamp.
        ts   = time.monotonic()
        gray = _to_gray(frame_rgb)

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

    def get_map_snapshot(self) -> tuple[np.ndarray | None, np.ndarray | None, int]:
        """Return (pose, map_points, tracking_state) as thread-safe copies."""
        with self._state_lock:
            s    = self._vo_state
            pose = s.pose.copy()       if s.pose       is not None else None
            pts  = s.map_points.copy() if s.map_points is not None else None
            return pose, pts, s.tracking_state

    @property
    def tracking_state(self) -> int:
        with self._state_lock:
            return self._vo_state.tracking_state

    @property
    def n_map_points(self) -> int:
        with self._state_lock:
            return self._vo_state.n_points

    @property
    def scale(self) -> float | None:
        return self._scale

    @property
    def status_text(self) -> str:
        return self._status_text

    # ------------------------------------------------------------------
    # Background worker
    # ------------------------------------------------------------------

    def _worker(self) -> None:
        while not self._stop_event.is_set():
            try:
                gray, ts = self._frame_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if ts < 0:
                break

            try:
                self._process_frame(gray, ts)
            except Exception as exc:
                self._status_text = f"vo error: {exc}"
                with self._state_lock:
                    self._vo_state.tracking_state = _S_RECENTLY_LOST

    # ------------------------------------------------------------------
    # Core VO pipeline
    # ------------------------------------------------------------------

    def _process_frame(self, gray: np.ndarray, ts: float) -> None:
        if self._K is None:
            with self._state_lock:
                self._vo_state.tracking_state = _S_SYSTEM_NOT_READY
            return

        # Detect ORB features in the current frame.
        kps, descs = self._orb.detectAndCompute(gray, None)
        if descs is None or len(kps) < MIN_MATCHES:
            self._buf.append((gray, None, None))
            with self._state_lock:
                self._vo_state.tracking_state = _S_NOT_INITIALIZED
            self._status_text = f"init (feats={len(kps) if kps else 0})"
            return

        pts_curr = np.array([kp.pt for kp in kps], dtype=np.float32)
        self._buf.append((gray, pts_curr, descs))

        if len(self._buf) < WINDOW:
            with self._state_lock:
                self._vo_state.tracking_state = _S_NOT_INITIALIZED
            self._status_text = f"init (buf={len(self._buf)}/{WINDOW})"
            return

        # Reference = oldest frame in buffer.
        ref_gray, ref_pts, ref_descs = self._buf[0]
        if ref_descs is None:
            with self._state_lock:
                self._vo_state.tracking_state = _S_NOT_INITIALIZED
            return

        # Match descriptors (Lowe ratio test).
        raw_matches = self._bf.knnMatch(ref_descs, descs, k=2)
        good = [pair[0] for pair in raw_matches
                if len(pair) == 2 and pair[0].distance < MATCH_RATIO * pair[1].distance]

        if len(good) < MIN_MATCHES:
            with self._state_lock:
                self._vo_state.tracking_state = _S_RECENTLY_LOST
                prev = self._sectors
            self._sectors = _decay(prev, RISK_DECAY, "mini_slam:lost+", 0.3)
            self._status_text = f"recently_lost (matches={len(good)})"
            return

        src = np.array([ref_pts[m.queryIdx] for m in good], dtype=np.float32)
        dst = np.array([pts_curr[m.trainIdx] for m in good], dtype=np.float32)

        # Check that there is enough parallax to triangulate.
        disparity = float(np.mean(np.linalg.norm(dst - src, axis=1)))
        if disparity < MIN_DISPARITY_PX:
            # Drone is nearly stationary; keep previous sectors, signal low confidence.
            with self._state_lock:
                prev = self._sectors
            self._sectors = _decay(prev, RISK_DECAY * 0.9, "mini_slam:hover", 0.5)
            self._status_text = f"hover (disp={disparity:.1f}px)"
            # Update state to OK with previous map if we have one.
            with self._state_lock:
                if self._vo_state.tracking_state == _S_OK:
                    pass  # keep existing map
                else:
                    self._vo_state.tracking_state = _S_NOT_INITIALIZED
            return

        # Essential Matrix with RANSAC.
        E, mask_e = cv2.findEssentialMat(
            src, dst, self._K, method=cv2.RANSAC, prob=0.999, threshold=1.5
        )
        if E is None or mask_e is None:
            with self._state_lock:
                self._vo_state.tracking_state = _S_RECENTLY_LOST
            self._sectors = _decay(self._sectors, RISK_DECAY, "mini_slam:no_E", 0.2)
            self._status_text = "recently_lost (E failed)"
            return

        inl = mask_e.ravel() == 1
        if inl.sum() < MIN_INLIERS:
            with self._state_lock:
                self._vo_state.tracking_state = _S_RECENTLY_LOST
            self._sectors = _decay(self._sectors, RISK_DECAY, "mini_slam:few_inl", 0.2)
            self._status_text = f"recently_lost (inliers={inl.sum()})"
            return

        _, R, t, _ = cv2.recoverPose(E, src[inl], dst[inl], self._K)

        # Triangulate: P1 = K[I|0] (reference = world), P2 = K[R|t] (current).
        P1 = self._K @ np.eye(3, 4)
        P2 = self._K @ np.hstack([R, t])
        pts4d = cv2.triangulatePoints(P1, P2, src[inl].T, dst[inl].T)
        pts4d /= pts4d[3:4]
        pts3d = pts4d[:3].T           # Nx3 in reference camera frame

        # Keep only points with positive depth in both cameras.
        z_ref  = pts3d[:, 2]
        z_curr = (R[2:3, :] @ pts3d.T + t[2]).ravel()
        valid  = (z_ref > 0.05) & (z_curr > 0.05)
        pts3d  = pts3d[valid]

        if len(pts3d) == 0:
            with self._state_lock:
                self._vo_state.tracking_state = _S_RECENTLY_LOST
            self._status_text = "recently_lost (no valid pts)"
            return

        # Reject extreme outliers by depth.
        med_z  = float(np.median(pts3d[:, 2]))
        pts3d  = pts3d[pts3d[:, 2] < med_z * MAX_DEPTH_MULT]

        # Build current-frame Tcw (reference frame = world, so Tcw = [R|t]).
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3]  = t.ravel()

        # Compute obstacle sectors.
        sectors = self._project_sectors(pose, pts3d)

        with self._state_lock:
            self._vo_state = _VO_State(
                tracking_state = _S_OK,
                pose           = pose,
                map_points     = pts3d,
                n_points       = len(pts3d),
                n_inliers      = int(inl.sum()),
                disparity_px   = disparity,
            )
            self._sectors = sectors

        self._status_text = (
            f"ok pts={len(pts3d)} inl={inl.sum()} disp={disparity:.0f}px"
        )

    # ------------------------------------------------------------------
    # Sector projection (mirrors ORBSLAM3Detector._project_sectors)
    # ------------------------------------------------------------------

    def _project_sectors(self, pose: np.ndarray,
                         map_pts: np.ndarray) -> ObstacleSectors:
        """Project reference-frame map points into current-camera sectors."""
        R = pose[:3, :3]
        t = pose[:3, 3]
        pts_cam = map_pts @ R.T + t   # Nx3 in current camera frame

        x_c = pts_cam[:, 0]
        y_c = pts_cam[:, 1]
        z_c = pts_cam[:, 2]

        fwd  = z_c > 0.05
        x_c  = x_c[fwd];  y_c = y_c[fwd];  z_c = z_c[fwd]

        if len(z_c) == 0:
            return ObstacleSectors(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, "mini_slam:ok")

        elev_deg = np.degrees(np.arctan2(-y_c, z_c))
        above    = elev_deg > -_MAX_BELOW_DEG
        x_c      = x_c[above];  z_c = z_c[above]

        if len(z_c) == 0:
            return ObstacleSectors(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, "mini_slam:ok")

        far_d   = SLAM_FAR_UNITS   if self._scale is None else (10.0 / self._scale)
        close_d = SLAM_CLOSE_UNITS if self._scale is None else (1.0  / self._scale)

        span        = max(far_d - close_d, 1e-6)
        depth_norm  = np.clip((z_c - close_d) / span, 0.0, 1.0)
        risk_per_pt = 1.0 - depth_norm

        az_deg = np.degrees(np.arctan2(x_c, z_c))

        def _sector(lo: float, hi: float) -> float:
            m = (az_deg >= lo) & (az_deg < hi)
            n = int(m.sum())
            if n < SLAM_MIN_POINTS:
                return 0.0
            top_n = max(1, n // 4)
            return float(np.sort(risk_per_pt[m])[-top_n:].mean())

        return ObstacleSectors(
            front       = _sector(-_INNER_DEG,  _INNER_DEG),
            front_left  = _sector(-_OUTER_DEG, -_INNER_DEG),
            front_right = _sector( _INNER_DEG,  _OUTER_DEG),
            left        = _sector(-90.0,        -_OUTER_DEG),
            right       = _sector( _OUTER_DEG,   90.0),
            confidence  = 1.0,
            method      = "mini_slam:ok",
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_K(status: dict) -> np.ndarray:
        fx = _f(status, "camera.fx_px",  457.0)
        fy = _f(status, "camera.fy_px",  457.0)
        cx = _f(status, "camera.cx_px",  320.0)
        cy = _f(status, "camera.cy_px",  240.0)
        return np.array([[fx, 0,  cx],
                         [0,  fy, cy],
                         [0,  0,  1.0]], dtype=np.float64)


def _to_gray(frame_rgb: np.ndarray) -> np.ndarray:
    bgr = np.ascontiguousarray(frame_rgb[:, :, ::-1])
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
