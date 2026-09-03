"""Literal DVision coordinate expectations.

Do not derive these values with production coordinate conversion helpers.
They are the independent public contract oracle.
"""

# Unit map-frame displacement produced by a positive command on each body axis,
# as (dx, dy) with map X east and map Y south.
EXPECTED_FORWARD = {
    0: (0, -1),
    90: (1, 0),
    180: (0, 1),
    270: (-1, 0),
}

EXPECTED_RIGHT = {
    0: (1, 0),
    90: (0, 1),
    180: (-1, 0),
    270: (0, -1),
}

EXPECTED_BACKWARD = {h: (-x, -y) for h, (x, y) in EXPECTED_FORWARD.items()}
EXPECTED_LEFT = {h: (-x, -y) for h, (x, y) in EXPECTED_RIGHT.items()}

CARDINAL_HEADINGS = (0, 90, 180, 270)

# Semantic action -> the sign the wire-level field must carry. The wire value is
# a separate layer from both the semantic action and the observed motion, so a
# UI conversion and a simulator conversion cannot be wrong in cancelling ways.
EXPECTED_COMMAND_SIGN = {
    "forward": ("forward_mps", +1),
    "backward": ("forward_mps", -1),
    "right": ("right_mps", +1),
    "left": ("right_mps", -1),
    "up": ("up_mps", +1),
    "down": ("up_mps", -1),
    "yaw_right": ("yaw_rate_dps", +1),
    "yaw_left": ("yaw_rate_dps", -1),
}


def circular_delta_deg(after: float, before: float) -> float:
    """Signed shortest compass change from before to after."""
    return (float(after) - float(before) + 180.0) % 360.0 - 180.0


def assert_direction(dx: float, dy: float, expected: tuple[int, int],
                     *, minimum: float = 0.05, cross_limit: float = 0.04) -> None:
    """Assert motion along one literal cardinal axis with little cross-axis drift."""
    ex, ey = expected
    along = dx * ex + dy * ey
    cross = dx * -ey + dy * ex
    assert along >= minimum, (
        f"displacement dx={dx:+.4f} dy={dy:+.4f} does not travel along the "
        f"expected map direction {expected} (along-axis {along:+.4f} < {minimum})"
    )
    assert abs(cross) <= cross_limit, (
        f"cross-axis drift {cross:+.4f} exceeds {cross_limit} for expected "
        f"direction {expected} (dx={dx:+.4f} dy={dy:+.4f})"
    )


def assert_stationary(dx: float, dy: float, *, limit: float = 0.02) -> None:
    """Assert a zero/hover command produced no material horizontal motion."""
    assert abs(dx) <= limit and abs(dy) <= limit, (
        f"expected no horizontal motion, observed dx={dx:+.4f} dy={dy:+.4f} "
        f"(limit {limit})"
    )
