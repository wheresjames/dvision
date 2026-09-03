"""Calibration fixture paths and literal expectations.

Top-down diagram of ``calibration_orientation.txt`` (map X east, map Y south).
The drone starts at the ``+`` cell facing compass 0 (north, map -Y), so every
landmark is forward of it:

    map x:    7     9  10  11     13
              r     y   g   w      b        <- row y = 5.5
                        +                   <- row y = 12.5, heading 0

    r red    west  of the nose -> image left
    b blue   east  of the nose -> image right
    y yellow elevated panel    -> image top
    g green  short  panel      -> image bottom
    w white  neutral, just east of the forward axis -> image centre-right

These numbers are the independent public oracle. They are literals reviewed
against the diagram above, never values produced by a production transform.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_MAP = ROOT / "assets/maps/calibration_orientation.txt"
CALIBRATION_RING_MAP = ROOT / "assets/maps/calibration_orientation_ring.txt"
DIRECT_MAP = ROOT / "assets/maps/test_direct.txt"

# Perception-chain fixtures. The left/right pair are mirror images of one
# another: an identical wall run, reflected about the drone's forward axis, so
# a handedness error anywhere in the chain shows up as two answers that fail to
# change sign. The front fixture puts the same wall squarely ahead, which is the
# case that carries a range and therefore reaches the occupancy map.
CHAIN_LEFT_MAP = ROOT / "assets/maps/chain_left_obstacle.txt"
CHAIN_RIGHT_MAP = ROOT / "assets/maps/chain_right_obstacle.txt"
CHAIN_FRONT_MAP = ROOT / "assets/maps/chain_front_obstacle.txt"
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
START_HEADING_DEG = 0.0

# Drone pose the static expectations below are measured at.
START_X_M = 10.5
START_Y_M = 12.5
START_ALT_M = 1.5

CENTER_X = FRAME_WIDTH / 2.0
CENTER_Y = FRAME_HEIGHT / 2.0

# Coarse expected image regions as (x_lo, x_hi, y_lo, y_hi) pixel boxes for the
# landmark centroid at the calibration pose.  Bounds are deliberately wide so
# harmless raster differences between graphics backends cannot fail the suite,
# but narrow enough that a mirror, flip, rotation, or transpose falls outside.
EXPECTED_REGIONS = {
    "red":    (40.0, 210.0, 170.0, 330.0),
    "yellow": (180.0, 300.0, 30.0, 200.0),
    "green":  (270.0, 370.0, 220.0, 380.0),
    "white":  (330.0, 450.0, 170.0, 330.0),
    "blue":   (430.0, 600.0, 170.0, 330.0),
}

# ---------------------------------------------------------------------------
# Rotationally symmetric fixture
#
# Every static expectation above is measured at heading 0. That is exactly the
# shape of blind spot that has bitten this project before -- a chain error that
# vanishes on the optical axis, covered only by tests that sit on it. The ring
# fixture puts an identical landmark group 7 m out on all four sides, ordered so
# each one reads r-y-g-w-b left-to-right *from the drone*, which means the same
# literal EXPECTED_REGIONS must hold at every cardinal heading.
#
#              map x:     3      7  9 10 11   13        17
#     row  3:                    r  y  g  w    b              <- seen at 0
#     row  7:            b                              r
#     row  9:            w                              y
#     row 10:            g          +                   g     <- drone, 10.5/10.5
#     row 11:            y                              w
#     row 13:            r                              b
#     row 17:                    b  w  g  y    r              <- seen at 180
#              seen at 270 ^                    ^ seen at 90
#
# The side groups run in opposite row order because "image left" is north when
# facing east and south when facing west. These orderings are literals reviewed
# against the diagram, never values read back from a render.
# ---------------------------------------------------------------------------

RING_START_X_M = 10.5
RING_START_Y_M = 10.5
RING_HEADINGS = (0.0, 90.0, 180.0, 270.0)

# Minimum landmark mask area; a panel that only half-passes a colour threshold
# indicates a lighting/texture regression rather than an orientation one.
MINIMUM_MARKER_PIXELS = 500

# Fixture and camera parameters recorded beside every failure bundle, so a CI
# artifact says which scene and camera produced the frame.
CALIBRATION_FIXTURE = {
    "map": str(CALIBRATION_MAP),
    "start_x_m": START_X_M,
    "start_y_m": START_Y_M,
    "start_alt_m": START_ALT_M,
    "start_heading_deg": START_HEADING_DEG,
    "frame_width_px": FRAME_WIDTH,
    "frame_height_px": FRAME_HEIGHT,
    "frame_format": "RGB24, top-left origin",
}
