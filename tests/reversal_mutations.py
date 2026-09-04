#!/usr/bin/env python3
"""Audit the suite's ability to catch sign, axis and orientation reversals.

Every reversal bug this project has hit looks the same from the outside: the
data is well-formed and only its *interpretation* is mirrored. A frame with
correct pixels can still be read left-for-right; a telemetry key with a
plausible value can still carry the wrong sign. Tests that only assert
"something happened" pass straight through all of it.

This script measures whether the suite actually notices. It applies one
reversal at a time to a production source file, runs pytest, and records
whether anything failed:

    CAUGHT   a test failed -- that reversal is guarded
    MISSED   the suite passed with the bug in place -- a real coverage hole

A MISSED row is the useful output. It names a boundary where a mirrored axis
or an inverted sign would ship silently.

Usage
-----
    python3 tests/reversal_mutations.py              # audit everything
    python3 tests/reversal_mutations.py --list       # show the catalogue
    python3 tests/reversal_mutations.py -k slam      # only matching mutations
    python3 tests/reversal_mutations.py -k publish --verbose

Exit status is 1 if any mutation was MISSED, so this can gate a merge.

Safety
------
Target files are copied to a temporary directory before anything is edited and
restored in a ``finally``, including on Ctrl-C. The restore is verified by
digest afterwards; if any file does not match, the script says so loudly and
prints the backup path rather than exiting quietly. It never touches git.

Adding a mutation
-----------------
Append a ``Mutation`` below. ``old`` must appear exactly once in the file --
the script refuses to guess, and reports ANCHOR when a snippet drifts, which
is itself a useful signal that the code moved.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Mutation:
    name: str
    path: str
    old: str
    new: str
    note: str = ""


MUTATIONS: tuple[Mutation, ...] = (
    # -- dsim physics and rendering -----------------------------------
    Mutation(
        "dsim/yaw-integration",
        "apps/dsim/dsim.py",
        "st.yaw_deg  = (st.yaw_deg - st.yaw_rate * dt) % 360.0",
        "st.yaw_deg  = (st.yaw_deg + st.yaw_rate * dt) % 360.0",
        "positive yaw would lower the compass heading",
    ),
    Mutation(
        "dsim/commanded-strafe",
        "apps/dsim/dsim.py",
        'self.state.cmd_right    = float(payload.get("right_mps",   0.0))',
        'self.state.cmd_right    = -float(payload.get("right_mps",   0.0))',
        "a right strafe command would fly left",
    ),
    Mutation(
        "dsim/camera-roll",
        "apps/dsim/dsim.py",
        "cam.setHpr(panda_h, self.CAM_PITCH + state.pitch_deg, -state.roll_deg)",
        "cam.setHpr(panda_h, self.CAM_PITCH + state.pitch_deg, state.roll_deg)",
        "a right bank would tilt the rendered horizon the wrong way",
    ),
    Mutation(
        "dsim/roll-leaks-into-pitch",
        "apps/dsim/dsim.py",
        "cam.setHpr(panda_h, self.CAM_PITCH + state.pitch_deg, -state.roll_deg)",
        "cam.setHpr(panda_h, self.CAM_PITCH + state.pitch_deg "
        "+ 0.3 * state.roll_deg, -state.roll_deg)",
        "banking would also aim the camera up or down",
    ),
    Mutation(
        "dsim/pitch-leaks-into-roll",
        "apps/dsim/dsim.py",
        "cam.setHpr(panda_h, self.CAM_PITCH + state.pitch_deg, -state.roll_deg)",
        "cam.setHpr(panda_h, self.CAM_PITCH + state.pitch_deg, "
        "-state.roll_deg + 0.3 * state.pitch_deg)",
        "pitching would also tilt the horizon",
    ),
    Mutation(
        "dsim/attitude-axes-exchanged",
        "apps/dsim/dsim.py",
        "cam.setHpr(panda_h, self.CAM_PITCH + state.pitch_deg, -state.roll_deg)",
        "cam.setHpr(panda_h, self.CAM_PITCH - state.roll_deg, state.pitch_deg)",
        "roll and pitch driving each other's channel",
    ),
    Mutation(
        "dsim/publish-attitude-axes-exchanged",
        "apps/dsim/dsim.py",
        '"drone.roll_deg":        f"{st.roll_deg:.2f}"',
        '"drone.roll_deg":        f"{st.pitch_deg:.2f}"',
        "published roll would report the pitch",
    ),
    Mutation(
        "dsim/framebuffer-column-reversal",
        "apps/dsim/dsim.py",
        "[::-1, ::-1]",
        "[::-1, :]",
        "every delivered frame would be mirrored left-for-right",
    ),
    # -- dsim published telemetry -------------------------------------
    Mutation(
        "dsim/publish-roll",
        "apps/dsim/dsim.py",
        '"drone.roll_deg":        f"{st.roll_deg:.2f}"',
        '"drone.roll_deg":        f"{-st.roll_deg:.2f}"',
        "DAIC would read a right bank as a left one",
    ),
    Mutation(
        "dsim/publish-pitch",
        "apps/dsim/dsim.py",
        '"drone.pitch_deg":       f"{st.pitch_deg:.2f}"',
        '"drone.pitch_deg":       f"{-st.pitch_deg:.2f}"',
        "nose-up and nose-down would be exchanged",
    ),
    Mutation(
        "dsim/publish-vx",
        "apps/dsim/dsim.py",
        '"drone.vx_mps":          f"{st.vx:.3f}"',
        '"drone.vx_mps":          f"{-st.vx:.3f}"',
        "flow ranging and SLAM scale anchoring read this",
    ),
    Mutation(
        "dsim/publish-vy",
        "apps/dsim/dsim.py",
        '"drone.vy_mps":          f"{st.vy:.3f}"',
        '"drone.vy_mps":          f"{-st.vy:.3f}"',
        "as above, on the north/south axis",
    ),
    Mutation(
        "dsim/publish-position-transpose",
        "apps/dsim/dsim.py",
        '"drone.x_m":             f"{st.x:.3f}"',
        '"drone.x_m":             f"{st.y:.3f}"',
        "pose_from_status navigates on these two keys",
    ),
    # -- dctl client ---------------------------------------------------
    Mutation(
        "dctl/display-frame-flip",
        "apps/dctl/dctl.py",
        '''def _client_rgb_frame(frame: np.ndarray) -> np.ndarray:
    """Normalize a shared RGB frame without changing pixel orientation."""
    return np.ascontiguousarray(frame)''',
        '''def _client_rgb_frame(frame: np.ndarray) -> np.ndarray:
    """Normalize a shared RGB frame without changing pixel orientation."""
    return np.ascontiguousarray(frame[:, ::-1])''',
        "the operator would see a mirrored world",
    ),
    Mutation(
        "dctl/manual-yaw-sign",
        "apps/dctl/dctl.py",
        "return _clamp(yaw_right_norm) * _MANUAL_YAW_RATE_DPS",
        "return -_clamp(yaw_right_norm) * _MANUAL_YAW_RATE_DPS",
        "pressing yaw-right would turn the drone left",
    ),
    Mutation(
        "dctl/joystick-strafe-axis",
        "apps/dctl/dctl.py",
        """    @property
    def right(self) -> float:
        return self._axis(0)""",
        """    @property
    def right(self) -> float:
        return -self._axis(0)""",
        "the left stick would strafe the wrong way",
    ),
    Mutation(
        "dctl/joystick-forward-axis",
        "apps/dctl/dctl.py",
        """    @property
    def forward(self) -> float:
        return -self._axis(1)""",
        """    @property
    def forward(self) -> float:
        return self._axis(1)""",
        "stick forward would fly backwards",
    ),
    Mutation(
        "dctl/joystick-axis-exchange",
        "apps/dctl/dctl.py",
        """    @property
    def up(self) -> float:
        return -self._axis(4)""",
        """    @property
    def up(self) -> float:
        return -self._axis(3)""",
        "the vertical control would read the yaw stick",
    ),
    # -- daic frame adapters and display -------------------------------
    Mutation(
        "daic/interpreted-frame-flip",
        "apps/daic/daic.py",
        '''def _client_rgb_frame(frame: np.ndarray) -> np.ndarray:
    """Convert shared-memory renderer output to daic's interpreted RGB frame."""
    return np.ascontiguousarray(frame)''',
        '''def _client_rgb_frame(frame: np.ndarray) -> np.ndarray:
    """Convert shared-memory renderer output to daic's interpreted RGB frame."""
    return np.ascontiguousarray(frame[:, ::-1])''',
        "perception would run on a mirrored frame",
    ),
    Mutation(
        "daic/display-sector-swap",
        "apps/daic/daic.py",
        '''def _display_sectors(sectors: ObstacleSectors) -> ObstacleSectors:
    """Convert interpreted-frame sector risks into display HUD coordinates."""
    return sectors''',
        '''def _display_sectors(sectors: ObstacleSectors) -> ObstacleSectors:
    """Convert interpreted-frame sector risks into display HUD coordinates."""
    import dataclasses
    return dataclasses.replace(sectors, left=sectors.right, right=sectors.left)''',
        "the HUD would blame the wrong side for an obstacle",
    ),
    Mutation(
        "daic/minimap-x-mirror",
        "apps/daic/daic.py",
        "px = margin + int(((wx - x_min) / (x_max - x_min)) * use)",
        "px = margin + int((1.0 - (wx - x_min) / (x_max - x_min)) * use)",
        "the point cloud would draw starboard points to port",
    ),
    Mutation(
        "daic/minimap-marker-mirror",
        "apps/daic/daic.py",
        "        return (cx + int(math.sin(angle) * radius),",
        "        return (cx - int(math.sin(angle) * radius),",
        "the drone marker would disagree with its own point cloud",
    ),
    # -- daic perception sectors ---------------------------------------
    Mutation(
        "flow/side-sector-swap",
        "apps/daic/optical_flow_avoidance.py",
        "    left_mask = roi & (x_norm < 0.28)\n"
        "    right_mask = roi & (x_norm > 0.72)",
        "    left_mask = roi & (x_norm > 0.72)\n"
        "    right_mask = roi & (x_norm < 0.28)",
        "an obstacle on the left would be mapped on the right",
    ),
    Mutation(
        "flow/forward-sector-swap",
        "apps/daic/optical_flow_avoidance.py",
        "    front_left_mask = roi & (x_norm >= 0.18) & (x_norm < 0.45)\n"
        "    front_right_mask = roi & (x_norm > 0.55) & (x_norm <= 0.82)",
        "    front_left_mask = roi & (x_norm > 0.55) & (x_norm <= 0.82)\n"
        "    front_right_mask = roi & (x_norm >= 0.18) & (x_norm < 0.45)",
        "as above, for the two forward-oblique sectors",
    ),
    Mutation(
        "mini_slam/azimuth-mirror",
        "apps/daic/mini_slam_detector.py",
        "az_deg = np.degrees(np.arctan2(x_c, z_c))",
        "az_deg = np.degrees(np.arctan2(-x_c, z_c))",
        "every triangulated point would land on the wrong side",
    ),
    Mutation(
        "mini_slam/side-sector-swap",
        "apps/daic/mini_slam_detector.py",
        "            left        = _sector(-90.0,        -_OUTER_DEG),\n"
        "            right       = _sector( _OUTER_DEG,   90.0),",
        "            left        = _sector( _OUTER_DEG,   90.0),\n"
        "            right       = _sector(-90.0,        -_OUTER_DEG),",
        "as above, at the sector-naming step",
    ),
    Mutation(
        "orb_slam3/azimuth-mirror",
        "apps/daic/orb_slam3_detector.py",
        "az_deg = np.degrees(np.arctan2(x_c, z_c))",
        "az_deg = np.degrees(np.arctan2(-x_c, z_c))",
        "the third detector's copy of the same reduction",
    ),
    # -- daic camera intrinsics ----------------------------------------
    Mutation(
        "mini_slam/K-principal-point-transpose",
        "apps/daic/mini_slam_detector.py",
        '        cx = _f(status, "camera.cx_px",  320.0)\n'
        '        cy = _f(status, "camera.cy_px",  240.0)',
        '        cx = _f(status, "camera.cy_px",  240.0)\n'
        '        cy = _f(status, "camera.cx_px",  320.0)',
        "triangulation would skew while the image stayed perfect",
    ),
    Mutation(
        "mini_slam/K-focal-transpose",
        "apps/daic/mini_slam_detector.py",
        "        return np.array([[fx, 0,  cx],\n"
        "                         [0,  fy, cy],",
        "        return np.array([[fy, 0,  cy],\n"
        "                         [0,  fx, cx],",
        "focal lengths and principal point both exchanged",
    ),
    Mutation(
        "mini_slam/K-default-transpose",
        "apps/daic/mini_slam_detector.py",
        '        cx = _f(status, "camera.cx_px",  320.0)',
        '        cx = _f(status, "camera.cx_px",  240.0)',
        "a wrong default would hide a transpose whenever telemetry is absent",
    ),

    # -- daic mapping and control --------------------------------------
    Mutation(
        "local_map/heading-to-map-yaw",
        "apps/daic/local_map.py",
        "return (heading_deg + 270.0) % 360.0",
        "return (heading_deg + 90.0) % 360.0",
        "the whole map would be rotated 180 degrees",
    ),
    Mutation(
        "local_map/sector-bearing-mirror",
        "apps/daic/local_map.py",
        '"left":        _sector_band(-90.0, -_OUTER_DEG),',
        '"left":        _sector_band(_OUTER_DEG, 90.0),',
        "left observations would be planted to starboard",
    ),
    Mutation(
        "local_map/relative-bearing-sign",
        "apps/daic/local_map.py",
        "yaw = math.radians(pose.yaw_deg + rel_deg)",
        "yaw = math.radians(pose.yaw_deg - rel_deg)",
        "every off-axis observation would mirror about the nose",
    ),
    Mutation(
        "local_map/route-yaw-sign",
        "apps/daic/local_map.py",
        "yaw = _clamp(yaw_error * _YAW_GAIN, -_MAX_YAW_DPS, _MAX_YAW_DPS)",
        "yaw = _clamp(-yaw_error * _YAW_GAIN, -_MAX_YAW_DPS, _MAX_YAW_DPS)",
        "the route follower would turn away from each waypoint",
    ),
    # -- yaw-dependent and chain-level ---------------------------------
    Mutation(
        "dsim/camera-yaw-frozen",
        "apps/dsim/dsim.py",
        "panda_h = (90.0 - state.yaw_deg) % 360.0",
        "panda_h = (90.0 - 270.0) % 360.0",
        "camera ignores yaw -- correct at heading 0, wrong everywhere else",
    ),
    Mutation(
        "local_map/camera-fov-drift",
        "apps/daic/local_map.py",
        "_CAMERA_HALF_FOV_DEG = 35.0   # dsim Panda3DRenderer.CAM_FOV_H = 70.0",
        "_CAMERA_HALF_FOV_DEG = 70.0   # dsim Panda3DRenderer.CAM_FOV_H = 70.0",
        "observations planted outside the view that produced them",
    ),
    Mutation(
        "local_map/asymmetric-obstacle-smear",
        "apps/daic/local_map.py",
        "for off_deg in (-10.0, 0.0, 10.0):",
        "for off_deg in (0.0, 10.0, 20.0):",
        "every obstacle smeared to starboard",
    ),
    Mutation(
        "controller/lateral-servo-gain",
        "apps/daic/controller.py",
        "_K_LATERAL = 0.0047",
        "_K_LATERAL = -0.0047",
        "the visual servo would steer away from the target",
    ),
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_pytest(verbose: bool) -> tuple[bool, list[str]]:
    """Run the suite. Returns (something_failed, failing test ids)."""
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-x", "--no-header",
         "-p", "no:cacheprovider"],
        cwd=ROOT, capture_output=True, text=True,
    )
    failures = [
        line.split("::")[-1].split()[0]
        for line in proc.stdout.splitlines()
        if line.startswith("FAILED")
    ]
    if verbose and proc.returncode == 0:
        print(proc.stdout[-2000:])
    return proc.returncode != 0, failures


def audit(selected: tuple[Mutation, ...], verbose: bool) -> int:
    paths = sorted({m.path for m in selected})
    backup = Path(tempfile.mkdtemp(prefix="dvision-reversal-"))
    for rel in paths:
        dest = backup / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / rel, dest)
    baseline = {rel: _digest(backup / rel) for rel in paths}

    def restore() -> None:
        for rel in paths:
            shutil.copy(backup / rel, ROOT / rel)

    width = max(len(m.name) for m in selected)
    missed: list[Mutation] = []
    anchor_failures: list[Mutation] = []

    try:
        print(f"Auditing {len(selected)} reversal(s) against the suite.\n")
        for mutation in selected:
            restore()
            target = ROOT / mutation.path
            source = target.read_text()
            if source.count(mutation.old) != 1:
                anchor_failures.append(mutation)
                print(f"  {mutation.name:<{width}}  ANCHOR "
                      f"(matched {source.count(mutation.old)}x, expected 1)")
                continue
            target.write_text(source.replace(mutation.old, mutation.new, 1))
            caught, failures = _run_pytest(verbose)
            if caught:
                detail = failures[0] if failures else ""
                print(f"  {mutation.name:<{width}}  CAUGHT  {detail[:48]}")
            else:
                missed.append(mutation)
                print(f"  {mutation.name:<{width}}  MISSED  <- {mutation.note}")
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    finally:
        restore()
        drifted = [rel for rel in paths if _digest(ROOT / rel) != baseline[rel]]
        if drifted:
            print("\n*** RESTORE FAILED -- these files are still mutated:",
                  file=sys.stderr)
            for rel in drifted:
                print(f"      {rel}", file=sys.stderr)
            print(f"    originals are in {backup}", file=sys.stderr)
            return 2
        shutil.rmtree(backup, ignore_errors=True)

    print()
    if anchor_failures:
        print(f"{len(anchor_failures)} mutation(s) no longer match their anchor; "
              "the code moved and the catalogue needs updating.")
    if missed:
        print(f"{len(missed)} reversal(s) MISSED -- these boundaries are "
              "unguarded:")
        for mutation in missed:
            print(f"  {mutation.path}: {mutation.name}")
        return 1
    if anchor_failures:
        return 1
    print(f"All {len(selected)} reversals are caught by the suite.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check that the suite catches sign and orientation reversals.")
    parser.add_argument("-k", dest="pattern", default=None,
                        help="only mutations whose name or path contains this")
    parser.add_argument("--list", action="store_true",
                        help="print the catalogue and exit")
    parser.add_argument("--verbose", action="store_true",
                        help="print pytest output for mutations that pass")
    args = parser.parse_args(argv)

    selected = MUTATIONS
    if args.pattern:
        needle = args.pattern.lower()
        selected = tuple(m for m in MUTATIONS
                         if needle in m.name.lower() or needle in m.path.lower())
    if not selected:
        print(f"no mutations match {args.pattern!r}", file=sys.stderr)
        return 2

    if args.list:
        width = max(len(m.name) for m in selected)
        for mutation in selected:
            print(f"  {mutation.name:<{width}}  {mutation.path}")
            if mutation.note:
                print(f"  {'':<{width}}  -> {mutation.note}")
        return 0

    return audit(selected, args.verbose)


if __name__ == "__main__":
    raise SystemExit(main())
