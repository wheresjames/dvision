"""Per-run debug artefact writer for daic.

Written to <report_root>/daic/ when --enable-ai is active:

  occ_NNN.png          occupancy-grid snapshots, one every 5 s
  slam_NNNNN.npz       SLAM point-cloud + pose snapshots, one every 10 s
  route_log.jsonl      A* path-change events
  frames/              annotated camera frames at triggered moments
  sector_timeline.png  obstacle-sector risk strip chart over the run
  summary.json         end-of-run key metrics

All image saves run in background threads so they never stall the control loop.
"""

from __future__ import annotations

import json
import math
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np


class RunReporter:
    _OCC_INTERVAL_S     = 5.0
    _SLAM_INTERVAL_S    = 10.0
    _PERIODIC_FRAME_S   = 15.0
    _MAX_FRAMES         = 40
    _ROUTE_LEN_THRESH   = 2      # cells: min path-length change to log
    _ROUTE_DIR_THRESH   = 20.0   # degrees: min waypoint-direction change to log
    _DETOUR_DEG         = 30.0   # deviation from direct bearing → call it a detour

    def __init__(self, report_dir: Path) -> None:
        self._dir = report_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        (self._dir / "frames").mkdir(exist_ok=True)

        self._t0 = time.monotonic()

        # Route-change log (written synchronously — fast text writes only)
        self._route_fh = (self._dir / "route_log.jsonl").open("w", encoding="utf-8")

        # Sector risk timeline: list of (t, front, fl, fr, left, right, conf)
        self._timeline: list[tuple[float, ...]] = []

        # Periodic timers
        self._last_occ_t          = -self._OCC_INTERVAL_S   # trigger immediately
        self._last_slam_t         = 0.0
        self._last_periodic_t     = 0.0

        # Sequence counters
        self._occ_seq   = 0
        self._slam_seq  = 0
        self._frame_cnt = 0

        # Route-change tracking state
        self._prev_path_len: int | None = None
        self._prev_wp_bearing: float | None = None

        # Triggered-frame flags
        self._first_obstacle_saved = False

        # Summary accumulators
        self._total_ticks        = 0
        self._avoidance_ticks    = 0
        self._wall_detect_ticks  = 0   # front risk > 0.40 with confidence > 0.3
        self._flow_conf_sum      = 0.0
        self._occ_peak_cells     = 0
        self._route_changes      = 0
        self._straight_ticks     = 0
        self._detour_ticks       = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tick(
        self,
        pose: Any,                              # Pose2 | None
        target_xy: tuple[float, float] | None,
        sectors: Any,                           # ObstacleSectors
        local_map_snap: dict,
        slam_snapshot: tuple,                   # (pose_mat, pts, state)
        annotated_frame: np.ndarray | None,
        avoiding: bool,
    ) -> None:
        elapsed = time.monotonic() - self._t0
        self._total_ticks += 1

        # --- Sector timeline (always) ---
        self._timeline.append((
            elapsed,
            sectors.front, sectors.front_left, sectors.front_right,
            sectors.left,  sectors.right, sectors.confidence,
        ))

        # --- Occupancy map snapshot ---
        if elapsed - self._last_occ_t >= self._OCC_INTERVAL_S:
            self._last_occ_t = elapsed
            if pose is not None:
                snap_copy = {
                    "cells":        dict(local_map_snap.get("cells", {})),
                    "path":         list(local_map_snap.get("path", [])),
                    "half_width_m": local_map_snap.get("half_width_m", 14.0),
                    "cell_m":       local_map_snap.get("cell_m", 0.5),
                }
                seq = self._occ_seq
                self._occ_seq += 1
                _run_bg(self._save_occ_image, elapsed, pose, target_xy, snap_copy, seq)

        # --- SLAM snapshot ---
        if elapsed - self._last_slam_t >= self._SLAM_INTERVAL_S:
            self._last_slam_t = elapsed
            try:
                pose_mat, pts, state = slam_snapshot
            except (TypeError, ValueError):
                pose_mat, pts, state = None, None, -1
            if pts is not None and pose_mat is not None:
                pts_copy  = np.array(pts,      dtype=np.float32)
                pose_copy = np.array(pose_mat, dtype=np.float32)
                seq = self._slam_seq
                self._slam_seq += 1
                _run_bg(self._save_slam_ply, pose_copy, pts_copy, int(state), seq)

        # --- Route change log ---
        path = local_map_snap.get("path", [])
        self._check_route_change(elapsed, pose, target_xy, path)

        # --- Triggered frame captures ---
        if annotated_frame is not None and self._frame_cnt < self._MAX_FRAMES:
            reason = self._frame_trigger(elapsed, sectors, avoiding)
            if reason:
                frame_copy = annotated_frame.copy()
                t_tag = f"{int(elapsed * 10):06d}"
                fname = self._dir / "frames" / f"{reason}_{t_tag}.jpg"
                self._frame_cnt += 1
                _run_bg(self._save_frame_jpg, frame_copy, fname)

        # --- Summary stats ---
        if avoiding:
            self._avoidance_ticks += 1
        if sectors.front > 0.40 and sectors.confidence > 0.3:
            self._wall_detect_ticks += 1
        self._flow_conf_sum += sectors.confidence
        occ_cells = sum(1 for v in local_map_snap.get("cells", {}).values() if v >= 1.6)
        if occ_cells > self._occ_peak_cells:
            self._occ_peak_cells = occ_cells

    def close(
        self,
        final_state: str,
        crashed: bool,
        target_dist_m: float | None,
    ) -> None:
        try:
            self._route_fh.close()
        except Exception:
            pass
        try:
            self._save_sector_timeline()
        except Exception as exc:
            print(f"daic reporter: sector timeline: {exc}", file=sys.stderr)
        try:
            self._save_summary(final_state, crashed, target_dist_m)
        except Exception as exc:
            print(f"daic reporter: summary: {exc}", file=sys.stderr)
        try:
            _generate_html_report(self._dir)
        except Exception as exc:
            print(f"daic reporter: html report: {exc}", file=sys.stderr)

    # ------------------------------------------------------------------
    # Route-change log
    # ------------------------------------------------------------------

    def _check_route_change(
        self,
        elapsed: float,
        pose: Any,
        target_xy: tuple[float, float] | None,
        path: list[tuple[float, float]],
    ) -> None:
        path_len = len(path)

        # Bearing to first waypoint ≥ 1 m away
        wp_bearing: float | None = None
        if pose is not None and len(path) >= 2:
            for wx, wy in path[1:]:
                if math.hypot(wx - pose.x, wy - pose.y) >= 1.0:
                    wp_bearing = math.degrees(
                        math.atan2(wy - pose.y, wx - pose.x)
                    ) % 360.0
                    break

        # Classify straight vs detour
        if pose is not None and target_xy is not None and wp_bearing is not None:
            direct = math.degrees(
                math.atan2(target_xy[1] - pose.y, target_xy[0] - pose.x)
            ) % 360.0
            dev = abs((wp_bearing - direct + 180.0) % 360.0 - 180.0)
            if dev < self._DETOUR_DEG:
                self._straight_ticks += 1
            elif dev > self._DETOUR_DEG * 2:
                self._detour_ticks += 1

        if self._prev_path_len is None:
            self._prev_path_len    = path_len
            self._prev_wp_bearing  = wp_bearing
            return

        len_change = abs(path_len - self._prev_path_len)
        bearing_change = 0.0
        if wp_bearing is not None and self._prev_wp_bearing is not None:
            bearing_change = abs((wp_bearing - self._prev_wp_bearing + 180.0) % 360.0 - 180.0)

        changed = (len_change >= self._ROUTE_LEN_THRESH
                   or bearing_change >= self._ROUTE_DIR_THRESH)
        if changed:
            self._route_changes += 1
            rec: dict[str, Any] = {
                "t":         round(elapsed, 2),
                "path_len":  path_len,
                "prev_len":  self._prev_path_len,
                "bearing":   round(wp_bearing, 1) if wp_bearing is not None else None,
            }
            if pose is not None:
                rec["x"] = round(pose.x, 2)
                rec["y"] = round(pose.y, 2)
            if target_xy is not None:
                rec["target_x"] = round(target_xy[0], 2)
                rec["target_y"] = round(target_xy[1], 2)
                if pose is not None:
                    rec["target_dist_m"] = round(
                        math.hypot(target_xy[0] - pose.x, target_xy[1] - pose.y), 2
                    )
            self._route_fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
            self._route_fh.flush()

        self._prev_path_len   = path_len
        self._prev_wp_bearing = wp_bearing

    # ------------------------------------------------------------------
    # Frame trigger logic
    # ------------------------------------------------------------------

    def _frame_trigger(self, elapsed: float, sectors: Any, avoiding: bool) -> str | None:
        if not self._first_obstacle_saved and sectors.front > 0.18 and sectors.confidence > 0.3:
            self._first_obstacle_saved = True
            return "obstacle_first"
        if avoiding:
            return "avoid"
        if sectors.front > 0.40 and sectors.confidence > 0.3:
            return "wall_risk"
        if elapsed - self._last_periodic_t >= self._PERIODIC_FRAME_S:
            self._last_periodic_t = elapsed
            return "periodic"
        return None

    # ------------------------------------------------------------------
    # Background-thread save functions
    # ------------------------------------------------------------------

    def _save_occ_image(
        self,
        elapsed: float,
        pose: Any,
        target_xy: tuple[float, float] | None,
        snap: dict,
        seq: int,
    ) -> None:
        try:
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            from matplotlib.patches import Rectangle, Circle
        except ImportError:
            return

        half    = float(snap["half_width_m"])
        cell_m  = float(snap["cell_m"])
        cells   = snap["cells"]
        path    = snap["path"]

        fig = Figure(figsize=(6, 6), facecolor="#0d1117")
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot(111, facecolor="#161b22")

        x0, x1 = pose.x - half, pose.x + half
        y0, y1 = pose.y - half, pose.y + half
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)   # invert Y: row 0 at top
        ax.set_aspect("equal")
        ax.tick_params(colors="#8b949e", labelsize=6)
        for sp in ax.spines.values():
            sp.set_color("#30363d")

        # Cells
        for (cx, cy), val in cells.items():
            wx, wy = cx * cell_m, cy * cell_m
            if abs(wx - pose.x) > half or abs(wy - pose.y) > half:
                continue
            if val >= 1.6:
                col, alpha = "#f85149", 0.85
            elif val >= 0.2:
                col, alpha = "#e09440", 0.55
            elif val <= -0.2:
                col, alpha = "#238636", 0.45
            else:
                continue
            ax.add_patch(Rectangle(
                (wx - cell_m * 0.5, wy - cell_m * 0.5), cell_m, cell_m,
                facecolor=col, alpha=alpha, linewidth=0, zorder=1,
            ))

        # A* path
        if len(path) >= 2:
            ax.plot([p[0] for p in path], [p[1] for p in path],
                    "-", color="#58a6ff", linewidth=1.5, zorder=3)

        # Target
        if target_xy is not None:
            ax.add_patch(Circle(target_xy, 0.35,
                                facecolor="#f2cc60", edgecolor="#e6edf3",
                                linewidth=1.2, zorder=4))

        # Drone triangle
        yaw = math.radians(pose.yaw_deg)
        sz  = 0.55
        nose = (pose.x + math.cos(yaw) * sz,       pose.y + math.sin(yaw) * sz)
        lft  = (pose.x + math.cos(yaw + 2.45) * sz * 0.6,
                pose.y + math.sin(yaw + 2.45) * sz * 0.6)
        rgt  = (pose.x + math.cos(yaw - 2.45) * sz * 0.6,
                pose.y + math.sin(yaw - 2.45) * sz * 0.6)
        ax.fill([nose[0], lft[0], rgt[0]], [nose[1], lft[1], rgt[1]],
                color="#58a6ff", zorder=5)

        occ = sum(1 for v in cells.values() if v >= 1.6)
        ax.set_title(
            f"t={elapsed:.1f}s  occ={occ}  path={len(path)}  "
            f"({pose.x:.1f}, {pose.y:.1f})",
            color="#e6edf3", fontsize=7, pad=4,
        )
        canvas.print_figure(
            str(self._dir / f"occ_{seq:03d}.png"),
            dpi=110, bbox_inches="tight", facecolor="#0d1117",
        )

    def _save_slam_ply(
        self,
        pose_mat: np.ndarray,
        pts: np.ndarray,
        state: int,
        seq: int,
    ) -> None:
        try:
            stem = self._dir / f"slam_{seq:05d}"

            # Binary little-endian PLY — readable by MeshLab, CloudCompare, Blender, Open3D
            n = len(pts)
            header = (
                "ply\n"
                "format binary_little_endian 1.0\n"
                f"element vertex {n}\n"
                "property float x\n"
                "property float y\n"
                "property float z\n"
                "end_header\n"
            ).encode()
            with open(str(stem) + ".ply", "wb") as fh:
                fh.write(header)
                fh.write(pts.astype(np.float32).tobytes())

            # Companion JSON for pose matrix and tracking state
            pose_list = pose_mat.tolist()
            with open(str(stem) + ".json", "w", encoding="utf-8") as fh:
                json.dump({"state": state, "pose": pose_list}, fh)
        except Exception:
            pass

    @staticmethod
    def _save_frame_jpg(frame: np.ndarray, path: Path) -> None:
        try:
            from PIL import Image
            Image.fromarray(frame, "RGB").save(str(path), quality=82)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # End-of-run artefacts (synchronous, called from close())
    # ------------------------------------------------------------------

    def _save_sector_timeline(self) -> None:
        if len(self._timeline) < 2:
            return
        try:
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_agg import FigureCanvasAgg
        except ImportError:
            return

        data  = np.array(self._timeline, dtype=np.float32)
        t     = data[:, 0]
        front = data[:, 1]
        fl    = data[:, 2]
        fr    = data[:, 3]
        left  = data[:, 4]
        right = data[:, 5]
        conf  = data[:, 6]

        fig = Figure(figsize=(12, 4), facecolor="#0d1117")
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot(111, facecolor="#161b22")

        ax.fill_between(t, conf * 0.12, alpha=0.4, color="#30363d", label="confidence ×0.12")
        ax.plot(t, front, color="#f85149", lw=1.2, label="front")
        ax.plot(t, fl,    color="#e09440", lw=0.9, label="front-left",  alpha=0.85)
        ax.plot(t, fr,    color="#e09440", lw=0.9, label="front-right", alpha=0.85, ls="--")
        ax.plot(t, left,  color="#58a6ff", lw=0.8, label="left",  alpha=0.7)
        ax.plot(t, right, color="#58a6ff", lw=0.8, label="right", alpha=0.7, ls="--")

        ax.axhline(0.12, color="#3fb950", lw=0.8, ls=":", alpha=0.8, label="mark thresh 0.12")
        ax.axhline(0.25, color="#f85149", lw=0.8, ls=":", alpha=0.8, label="avoid thresh 0.25")

        ax.set_xlim(float(t[0]), float(t[-1]))
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("elapsed (s)", color="#8b949e", fontsize=8)
        ax.set_ylabel("sector risk", color="#8b949e", fontsize=8)
        ax.set_title("Obstacle Sector Risk Timeline", color="#e6edf3", fontsize=9)
        ax.tick_params(colors="#8b949e", labelsize=7)
        for sp in ax.spines.values():
            sp.set_color("#30363d")
        ax.legend(facecolor="#21262d", edgecolor="#30363d",
                  labelcolor="#e6edf3", fontsize=7, ncol=4, loc="upper right")

        canvas.print_figure(
            str(self._dir / "sector_timeline.png"),
            dpi=130, bbox_inches="tight", facecolor="#0d1117",
        )

    def _save_summary(
        self,
        final_state: str,
        crashed: bool,
        target_dist_m: float | None,
    ) -> None:
        elapsed   = time.monotonic() - self._t0
        mean_conf = self._flow_conf_sum / max(1, self._total_ticks)
        summary   = {
            "duration_s":          round(elapsed, 1),
            "final_state":         final_state,
            "crashed":             crashed,
            "target_dist_final_m": (round(target_dist_m, 2)
                                    if target_dist_m is not None else None),
            "total_ticks":         self._total_ticks,
            "route_changes":       self._route_changes,
            "straight_path_ticks": self._straight_ticks,
            "detour_path_ticks":   self._detour_ticks,
            "wall_detect_ticks":   self._wall_detect_ticks,
            "avoidance_ticks":     self._avoidance_ticks,
            "occ_peak_cells":      self._occ_peak_cells,
            "flow_conf_mean":      round(mean_conf, 3),
            "occ_snapshots":       self._occ_seq,
            "slam_snapshots":      self._slam_seq,
            "frame_captures":      self._frame_cnt,
        }
        (self._dir / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_bg(fn, *args) -> None:
    """Run fn(*args) in a daemon background thread."""
    threading.Thread(target=fn, args=args, daemon=True).start()


# ---------------------------------------------------------------------------
# HTML report generation
# ---------------------------------------------------------------------------

_STATE_CSS = {
    "ARMING":   "#8b949e",
    "SEARCH":   "#58a6ff",
    "APPROACH": "#e09440",
    "LANDING":  "#3fb950",
    "COMPLETE": "#3fb950",
    "FAILSAFE": "#f85149",
    "IDLE":     "#30363d",
}


def _generate_html_report(report_dir: Path) -> None:
    log_path  = report_dir / "flight.jsonl"
    html_path = report_dir / "report.html"
    if not log_path.exists():
        return

    ticks, events, transitions = _parse_log(log_path)
    if not ticks:
        return

    summary  = _load_json(report_dir / "summary.json")
    findings = _analyse_findings(ticks, transitions)
    path_b64 = _make_state_path_b64(ticks)
    tline_b64 = _encode_img_b64(report_dir / "sector_timeline.png")
    dsim_b64  = _encode_img_b64(report_dir.parent / "dsim" / "flight_path.png")
    occ_images = [p.name for p in sorted(report_dir.glob("occ_*.png"))]

    html_path.write_text(
        _render_html(
            summary, findings, transitions, path_b64, tline_b64, dsim_b64,
            occ_images,
        ),
        encoding="utf-8",
    )
    print(f"daic: report → {html_path}", file=sys.stderr)


# ------------------------------------------------------------------
# Log parsing
# ------------------------------------------------------------------

def _parse_log(log_path: Path):
    ticks: list[dict] = []
    events: list[dict] = []
    transitions: list[dict] = []
    prev_state: str | None = None

    with open(log_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "event" in r:
                events.append(r)
            elif "state" in r and "telem" in r:
                ticks.append(r)
                state = r["state"]
                if state != prev_state:
                    vis = r.get("vision", {})
                    fused = vis.get("fused", {})
                    telem = r.get("telem", {})
                    cmd   = r.get("cmd", {})
                    transitions.append({
                        "t":          r["t"],
                        "from_state": prev_state or "—",
                        "to_state":   state,
                        "x":          _ff(telem, "drone.x_m"),
                        "y":          _ff(telem, "drone.y_m"),
                        "hdg":        _ff(telem, "drone.heading_deg"),
                        "front_risk": fused.get("front", 0.0),
                        "conf":       fused.get("confidence", 0.0),
                        "fwd":        cmd.get("forward_mps", 0.0),
                    })
                    prev_state = state

    return ticks, events, transitions


def _ff(d: dict, key: str) -> float:
    try:
        return float(d.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


# ------------------------------------------------------------------
# Failure analysis
# ------------------------------------------------------------------

def _analyse_findings(ticks: list[dict], transitions: list[dict]) -> list[dict]:
    """Return a list of finding dicts: {level, title, body}."""
    findings: list[dict] = []

    # --- Approach-with-obstacle ---
    for i, tr in enumerate(transitions):
        if tr["to_state"] == "APPROACH" and tr["front_risk"] > 0.5:
            # What was fwd on the tick AFTER the transition?
            next_fwd = _first_approach_fwd(ticks, tr["t"])
            findings.append({
                "level": "critical",
                "title": "APPROACH transition with high obstacle risk",
                "body": (
                    f"State changed to APPROACH at t={tr['t']:.2f}s with "
                    f"front_risk={tr['front_risk']:.3f} (confidence={tr['conf']:.2f}). "
                    f"The approach servo immediately commanded fwd={next_fwd:.2f} "
                    f"(pre-scaled; actual ≈{next_fwd * 0.1:.2f} m/s). "
                    f"Obstacle avoidance is only applied in SEARCH state — "
                    f"it did not fire in APPROACH, so the full forward command "
                    f"was sent despite the obstacle directly ahead."
                ),
            })

    # --- Avoidance active just before APPROACH ---
    if _avoidance_before_approach(ticks):
        findings.append({
            "level": "warning",
            "title": "Avoidance was clamping speed immediately before APPROACH",
            "body": (
                "The obstacle avoidance layer was reducing forward speed to near zero "
                "in the final SEARCH ticks (front_risk near 1.0). The detection lock "
                "completed while the drone was still adjacent to the obstacle. "
                "Consider resetting the approach lock counter when front_risk is high, "
                "so APPROACH cannot be entered until the path is clear."
            ),
        })

    # --- Crashed in APPROACH with high fwd ---
    crashed_ticks = [r for r in ticks if r["state"] == "APPROACH"]
    if crashed_ticks:
        last = crashed_ticks[-1]
        fwd = last.get("cmd", {}).get("forward_mps", 0.0)
        risk = last.get("vision", {}).get("fused", {}).get("front", 0.0)
        if fwd > 3.0 and risk > 0.5:
            findings.append({
                "level": "info",
                "title": "High forward command persisted through crash",
                "body": (
                    f"At t={last['t']:.2f}s (final APPROACH tick): "
                    f"fwd={fwd:.2f}, front_risk={risk:.3f}. "
                    f"The servo continued to command full approach speed even "
                    f"while pressed against the wall (optical flow risk decaying "
                    f"because the drone was stationary)."
                ),
            })

    # --- Path always straight (no detour) ---
    detour_ticks = sum(
        1 for r in ticks
        if r.get("vision", {}).get("local_map", {}).get("path_len", 0) > 0
    )
    if detour_ticks == 0 and len(ticks) > 50:
        findings.append({
            "level": "warning",
            "title": "No A* path was computed during the run",
            "body": (
                "local_map.path_len was zero throughout. Either the occupancy map "
                "had no obstacle data (walls not detected by vision) or the local "
                "route command was not applied. Navigation relied entirely on GPS nav."
            ),
        })

    return findings


def _first_approach_fwd(ticks: list[dict], t_transition: float) -> float:
    for r in ticks:
        if r["t"] > t_transition and r["state"] == "APPROACH":
            return float(r.get("cmd", {}).get("forward_mps", 0.0))
    return 0.0


def _avoidance_before_approach(ticks: list[dict]) -> bool:
    """True if the last few SEARCH ticks had near-zero forward speed from avoidance."""
    search_before = [r for r in ticks
                     if r["state"] == "SEARCH"][-10:]
    if not search_before:
        return False
    low_fwd = sum(
        1 for r in search_before
        if abs(r.get("cmd", {}).get("forward_mps", 1.0)) < 0.5
        and r.get("vision", {}).get("fused", {}).get("front", 0.0) > 0.5
    )
    return low_fwd >= 3


# ------------------------------------------------------------------
# State-coloured flight path image
# ------------------------------------------------------------------

def _make_state_path_b64(ticks: list[dict]) -> str | None:
    try:
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        import io, base64
    except ImportError:
        return None

    positions: list[tuple[float, float, str]] = []
    for r in ticks:
        t2 = r.get("telem", {})
        try:
            x = float(t2["drone.x_m"])
            y = float(t2["drone.y_m"])
        except (KeyError, TypeError, ValueError):
            continue
        positions.append((x, y, r["state"]))

    if len(positions) < 2:
        return None

    fig = Figure(figsize=(7, 7), facecolor="#0d1117")
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111, facecolor="#161b22")
    ax.set_aspect("equal")
    ax.tick_params(colors="#8b949e", labelsize=7)
    for sp in ax.spines.values():
        sp.set_color("#30363d")
    ax.set_xlabel("X (m)", color="#8b949e", fontsize=8)
    ax.set_ylabel("Y (m)", color="#8b949e", fontsize=8)
    ax.invert_yaxis()

    # Draw path segments coloured by state
    plotted_states: set[str] = set()
    i = 0
    while i < len(positions) - 1:
        state = positions[i][2]
        j = i + 1
        while j < len(positions) and positions[j][2] == state:
            j += 1
        seg_x = [p[0] for p in positions[i:j]]
        seg_y = [p[1] for p in positions[i:j]]
        col   = _STATE_CSS.get(state, "#8b949e")
        label = state if state not in plotted_states else None
        ax.plot(seg_x, seg_y, color=col, linewidth=2.0,
                alpha=0.9, label=label, solid_capstyle="round")
        plotted_states.add(state)
        # Share an endpoint with the next segment so the path looks continuous,
        # but always advance — a single-tick state run has j == i + 1, and
        # j - 1 == i would spin forever.
        i = max(i + 1, j - 1)

    # Start and end markers
    ax.plot(positions[0][0],  positions[0][1],  "o",
            color="#3fb950", markersize=9, markeredgecolor="#e6edf3",
            markeredgewidth=1.5, zorder=5, label="Start")
    last_state = positions[-1][2]
    end_col = "#f85149" if last_state in ("APPROACH", "FAILSAFE") else "#f2cc60"
    ax.plot(positions[-1][0], positions[-1][1], "x" if last_state == "APPROACH" else "s",
            color=end_col, markersize=10, markeredgewidth=2.5,
            zorder=5, label="Crash" if last_state == "APPROACH" else "End")

    ax.set_title("Flight Path — coloured by planner state",
                 color="#e6edf3", fontsize=9, pad=6)
    ax.legend(facecolor="#21262d", edgecolor="#30363d",
              labelcolor="#e6edf3", fontsize=7)

    buf = io.BytesIO()
    canvas.print_figure(buf, format="png", dpi=130, bbox_inches="tight",
                        facecolor="#0d1117")
    return base64.b64encode(buf.getvalue()).decode()


def _encode_img_b64(path: Path) -> str | None:
    try:
        import base64
        if not path.exists():
            return None
        return base64.b64encode(path.read_bytes()).decode()
    except Exception:
        return None


# ------------------------------------------------------------------
# HTML renderer
# ------------------------------------------------------------------

def _render_html(
    summary: dict,
    findings: list[dict],
    transitions: list[dict],
    path_b64: str | None,
    tline_b64: str | None,
    dsim_b64: str | None,
    occ_images: list[str],
) -> str:
    crashed   = summary.get("crashed", False)
    state     = summary.get("final_state", "?")
    duration  = summary.get("duration_s", 0)
    dist      = summary.get("target_dist_final_m")

    result_label = "CRASHED" if crashed else ("LANDED ✓" if state == "COMPLETE" else state)
    result_color = "#f85149" if crashed else ("#3fb950" if state == "COMPLETE" else "#e09440")

    finding_html = ""
    for f in findings:
        border = {"critical": "#f85149", "warning": "#e09440", "info": "#58a6ff"}.get(f["level"], "#8b949e")
        icon   = {"critical": "⛔", "warning": "⚠️", "info": "ℹ️"}.get(f["level"], "•")
        finding_html += f"""
        <div style="border-left:4px solid {border};padding:10px 14px;margin:10px 0;background:#161b22;border-radius:0 6px 6px 0">
          <div style="color:{border};font-weight:bold;margin-bottom:6px">{icon} {f['title']}</div>
          <div style="color:#c9d1d9;line-height:1.6">{f['body']}</div>
        </div>"""

    tr_rows = ""
    for tr in transitions:
        risk  = tr["front_risk"]
        rcol  = "#f85149" if risk > 0.7 else ("#e09440" if risk > 0.35 else "#3fb950")
        scol  = _STATE_CSS.get(tr["to_state"], "#8b949e")
        tr_rows += f"""
        <tr>
          <td>{tr['t']:.2f}s</td>
          <td style="color:#8b949e">{tr['from_state']}</td>
          <td>→</td>
          <td style="color:{scol};font-weight:bold">{tr['to_state']}</td>
          <td>({tr['x']:.1f}, {tr['y']:.1f})</td>
          <td style="color:{rcol}">{risk:.3f}</td>
          <td>{tr['conf']:.2f}</td>
          <td>{tr['fwd']:.2f}</td>
        </tr>"""

    def img_section(title: str, b64: str | None) -> str:
        if not b64:
            return ""
        return f"""
        <div class="section">
          <h2>{title}</h2>
          <img src="data:image/png;base64,{b64}" style="max-width:100%;border-radius:6px">
        </div>"""

    stats_items = [
        ("Duration",        f"{duration:.1f} s"),
        ("Final state",     state),
        ("Target dist",     f"{dist:.2f} m" if dist is not None else "—"),
        ("Total ticks",     str(summary.get("total_ticks", "?"))),
        ("Route changes",   str(summary.get("route_changes", "?"))),
        ("Wall detections", str(summary.get("wall_detect_ticks", "?"))),
        ("Avoidance ticks", str(summary.get("avoidance_ticks", "?"))),
        ("Occ peak cells",  str(summary.get("occ_peak_cells", "?"))),
        ("Flow conf mean",  str(summary.get("flow_conf_mean", "?"))),
    ]
    stats_html = "".join(
        f"<tr><td style='color:#8b949e;padding:3px 12px 3px 0'>{k}</td>"
        f"<td style='color:#e6edf3'>{v}</td></tr>"
        for k, v in stats_items
    )
    occ_html = ""
    if occ_images:
        cards = "".join(
            f"""
            <figure>
              <img src="{name}" alt="Occupancy snapshot {i + 1}">
              <figcaption>Occupancy snapshot {i + 1} · {name}</figcaption>
            </figure>"""
            for i, name in enumerate(occ_images)
        )
        occ_html = f"""
        <div class="section">
          <h2>Occupancy Snapshot Series</h2>
          <p class="muted">
            These <code>occ_*.png</code> images show DAIC's local occupancy map
            over time. Red cells are occupied, green cells are free, the blue
            line is the local A* path, the triangle is the drone, and the yellow
            marker is the status-derived target. Read the gallery as a timeline:
            the useful question is whether the blue route bends around red
            occupied cells or continues through them.
          </p>
          <div class="occ-grid">{cards}</div>
        </div>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>daic Flight Report</title>
<style>
  body{{margin:0;padding:20px 28px;font-family:'Segoe UI',system-ui,sans-serif;
       background:#0d1117;color:#c9d1d9;font-size:14px;line-height:1.5}}
  h1{{margin:0 0 4px;font-size:2.2em;color:{result_color}}}
  h2{{color:#e6edf3;border-bottom:1px solid #30363d;padding-bottom:6px;margin-top:32px}}
  .section{{margin-bottom:28px}}
  .muted{{color:#8b949e}}
  table{{border-collapse:collapse;width:100%}}
  td,th{{padding:5px 10px;text-align:left;border-bottom:1px solid #21262d}}
  th{{color:#8b949e;font-weight:normal;font-size:12px}}
  code{{background:#161b22;padding:1px 6px;border-radius:4px;font-size:13px}}
  img{{max-width:100%;border-radius:6px;border:1px solid #30363d;background:#161b22}}
  figure{{margin:0}}
  figcaption{{color:#8b949e;font-size:12px;margin-top:6px}}
  .occ-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:18px;align-items:start}}
</style>
</head>
<body>

<div class="section">
  <h1>{result_label}</h1>
  <div style="color:#8b949e">{summary.get('final_state','')} · {duration:.1f} s</div>
</div>

<div class="section">
  <h2>Statistics</h2>
  <table style="width:auto">{stats_html}</table>
</div>

{'<div class="section"><h2>Findings</h2>' + finding_html + '</div>' if findings else ''}

<div class="section">
  <h2>State Transitions</h2>
  <table>
    <tr>
      <th>Time</th><th>From</th><th></th><th>To</th>
      <th>Position</th><th>Front risk</th><th>Conf</th><th>Fwd cmd</th>
    </tr>
    {tr_rows}
  </table>
</div>

{img_section("Flight Path (by planner state)", path_b64)}
{img_section("Sector Risk Timeline", tline_b64)}
{img_section("dsim Flight Path", dsim_b64)}
{occ_html}

</body>
</html>"""
