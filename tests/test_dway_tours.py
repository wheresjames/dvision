"""Tour loading, coordinate transforms, arrival rules, and strategy choice."""

import json
import math
from pathlib import Path

import pytest

from dvision2_common import load_map
from dway.follower import (
    Follower, FollowerEvent, PositionStrategy, Sample, StrategyError,
    VelocityStrategy, build_legs, select_strategy, wrap_deg,
)
from dway.frames import (
    GeoAnchor, global_to_local_ned, local_ned_to_global, map_heading_to_true,
    rotate_clockwise,
)
from dway.link import VehicleCapabilities
from dway.tour import (
    FrameContext, TourError, leg_clearances, load_tour, load_tour_map,
    parse_tour, save_tour,
)

ROOT = Path(__file__).resolve().parents[1]
FORWARD_TOUR = ROOT / "assets/tours/maze_012.forward.v1.json"
MAZE_012 = ROOT / "assets/maps/maze_012.txt"


def capabilities(**overrides) -> VehicleCapabilities:
    fields = dict(vehicle="dsim", frames=("map", "local_ned"),
                  accepts_position_target=True, accepts_velocity_target=True,
                  accepts_attitude_target=False, supports_missions=False,
                  setpoint_timeout_s=2.0, max_speed_mps=5.0, max_accel_mps2=4.0)
    fields.update(overrides)
    return VehicleCapabilities(**fields)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def test_committed_forward_tour_loads_with_its_gates() -> None:
    tour = load_tour(FORWARD_TOUR)
    assert tour.tour_id == "maze_012.forward.v1"
    assert tour.coordinate_frame == "map"
    assert len(tour.waypoints) == 2
    assert tour.waypoint_tolerance_m == 0.05
    assert tour.arrival_speed_mps == 0.15
    assert tour.heading_tolerance_deg == 5.0
    assert tour.waypoints[0].dwell_s == 0.2


def test_aggregate_and_not_applicable_files_are_rejected_with_a_reason() -> None:
    with pytest.raises(TourError, match="aggregate diagnostics"):
        load_tour(ROOT / "assets/tours/diagnostics.v1.json")
    with pytest.raises(TourError, match="not applicable"):
        load_tour(ROOT / "assets/tours/maze_012.orbit.v1.json")


@pytest.mark.parametrize("payload,message", [
    ({"schema_version": 2, "tour_id": "t", "waypoints": [{"x": 1, "y": 1, "z": 1}]},
     "unsupported schema_version"),
    ({"schema_version": 1, "tour_id": "t", "waypoints": []}, "no waypoints"),
    ({"schema_version": 1, "tour_id": "t", "coordinate_frame": "body",
      "waypoints": [{"x": 1, "y": 1, "z": 1}]}, "unsupported coordinate_frame"),
    ({"schema_version": 1, "tour_id": "t", "map": "m.txt",
      "waypoints": [{"x": 1, "y": 1}]}, "z must be a number"),
    ({"schema_version": 1, "tour_id": "t", "coordinate_frame": "local_ned",
      "waypoints": [{"north_m": 1, "east_m": 1, "down_m": -1, "x": 3}]},
     "do not belong to the local_ned frame"),
])
def test_malformed_tours_are_refused(payload, message) -> None:
    with pytest.raises(TourError, match=message):
        parse_tour(payload, path=Path("memory.json"))


def test_map_hash_mismatch_is_refused_before_anything_else(tmp_path) -> None:
    payload = json.loads(FORWARD_TOUR.read_text())
    payload["map_sha"] = "0" * 64
    path = tmp_path / "wrong.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(TourError, match="map hash mismatch"):
        load_tour_map(load_tour(path), ROOT)


def test_tours_round_trip_through_save(tmp_path) -> None:
    tour = load_tour(FORWARD_TOUR)
    reloaded = load_tour(save_tour(tour, tmp_path / "again.json"))
    assert reloaded.waypoints == tour.waypoints
    assert reloaded.map_sha == tour.map_sha
    assert reloaded.waypoint_tolerance_m == tour.waypoint_tolerance_m


def test_local_ned_and_global_tours_need_no_map() -> None:
    tour = parse_tour({
        "schema_version": 1, "tour_id": "ned", "coordinate_frame": "local_ned",
        "waypoints": [{"north_m": 3.0, "east_m": -2.0, "down_m": -1.5,
                       "heading_deg": 90.0, "dwell_s": 0.0}],
    }, path=Path("ned.json"))
    assert load_tour_map(tour, ROOT) is None
    assert tour.waypoints[0].north_m == 3.0


def test_global_tour_requires_a_geo_anchor_to_be_flown() -> None:
    tour = parse_tour({
        "schema_version": 1, "tour_id": "geo", "coordinate_frame": "global",
        "waypoints": [{"lat_deg": 52.52, "lon_deg": 13.405, "alt_m": 40.0}],
    }, path=Path("geo.json"))
    with pytest.raises(TourError, match="geo_anchor"):
        FrameContext(39.0, 29.0).waypoint_ned(tour.waypoints[0])


def test_geo_anchor_shorthand_keys_are_not_accepted() -> None:
    with pytest.raises(TourError, match="missing origin_lat_deg"):
        parse_tour({
            "schema_version": 1, "tour_id": "t", "map": "m.txt",
            "waypoints": [{"x": 1, "y": 1, "z": 1}],
            "geo_anchor": {"lat": 52.0, "lon": 13.0, "alt": 30.0},
        }, path=Path("t.json"))


# ---------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------

def test_map_and_local_ned_agree_on_the_cardinal_directions() -> None:
    context = FrameContext(40.0, 30.0)
    assert context.map_to_ned(20.0, 15.0, 2.0) == (0.0, 0.0, -2.0)
    # Map Y is south-positive, so a smaller Y is further north.
    assert context.map_to_ned(20.0, 14.0, 2.0)[0] == pytest.approx(1.0)
    assert context.map_to_ned(21.0, 15.0, 2.0)[1] == pytest.approx(1.0)
    assert context.ned_to_map(*context.map_to_ned(7.25, 22.5, 3.0)) == \
        pytest.approx((7.25, 22.5, 3.0))


def test_clockwise_rotation_turns_north_into_east() -> None:
    east, north = rotate_clockwise(0.0, 1.0, 90.0)
    assert (east, north) == pytest.approx((1.0, 0.0), abs=1e-9)
    east, north = rotate_clockwise(1.0, 0.0, 90.0)
    assert (east, north) == pytest.approx((0.0, -1.0), abs=1e-9)


@pytest.mark.parametrize("rotation", [0.0, 37.0, 180.0, 315.0])
def test_global_projection_round_trips_and_keeps_bearings(rotation: float) -> None:
    anchor = GeoAnchor(52.52, 13.405, 34.0, rotation)
    north, east, down = 12.5, -7.25, -3.0
    lat, lon, alt = local_ned_to_global(north, east, down, anchor)
    assert alt == pytest.approx(anchor.origin_alt_m - down)
    assert global_to_local_ned(lat, lon, alt, anchor) == \
        pytest.approx((north, east, down), abs=1e-6)
    # A map-north step must land north of the anchor only when the site is not
    # rotated; with rotation it lands along true north turned clockwise by it.
    lat_n, lon_n, _ = local_ned_to_global(10.0, 0.0, 0.0, anchor)
    bearing = math.degrees(math.atan2(
        (lon_n - anchor.origin_lon_deg) * math.cos(math.radians(anchor.origin_lat_deg)),
        lat_n - anchor.origin_lat_deg)) % 360.0
    assert bearing == pytest.approx(rotation % 360.0, abs=0.5)
    assert map_heading_to_true(0.0, anchor) == pytest.approx(rotation % 360.0)


def test_setpoints_convert_into_whatever_frame_the_vehicle_accepts() -> None:
    context = FrameContext(39.0, 29.0, GeoAnchor(52.52, 13.405, 34.0, 0.0))
    tour = load_tour(FORWARD_TOUR)
    ned = context.waypoint_ned(tour.waypoints[0])
    as_map = context.target_from_ned(*ned, frame="map", heading_deg=180.0,
                                     max_speed_mps=1.0)
    assert (as_map.x, as_map.y, as_map.z) == pytest.approx((35.875, 1.375, 1.5))
    as_ned = context.target_from_ned(*ned, frame="local_ned", heading_deg=180.0,
                                     max_speed_mps=1.0)
    assert (as_ned.north_m, as_ned.east_m, as_ned.down_m) == pytest.approx(ned)


# ---------------------------------------------------------------------------
# Clearance
# ---------------------------------------------------------------------------

def test_first_leg_clearance_is_measured_from_the_current_pose() -> None:
    tour = load_tour(FORWARD_TOUR)
    sim_map = load_map(MAZE_012)
    from_start = leg_clearances(tour, sim_map, (1.5, 1.5))
    assert [leg.index for leg in from_start] == [0, 1]
    # The map's own start is walled off from the first waypoint, and that leg
    # is exactly the one a follower would fly first.
    assert from_start[0].obstructed
    from_near = leg_clearances(tour, sim_map, (34.5, 3.5))
    assert not any(leg.obstructed for leg in from_near)
    assert from_near[0].clearance_m == pytest.approx(0.375, abs=1e-6)
    assert [leg.index for leg in leg_clearances(tour, sim_map, None)] == [1]


# ---------------------------------------------------------------------------
# Strategy selection
# ---------------------------------------------------------------------------

def test_strategy_ladder_reads_capabilities_not_vehicle_identity() -> None:
    chosen = select_strategy(capabilities(), "map", speed_mps=1.0)
    assert isinstance(chosen, PositionStrategy) and chosen.frame == "map"
    fallback = select_strategy(capabilities(accepts_position_target=False),
                               "map", speed_mps=1.0)
    assert isinstance(fallback, VelocityStrategy)
    forced = select_strategy(capabilities(), "map", speed_mps=1.0, forced="velocity")
    assert isinstance(forced, VelocityStrategy)
    with pytest.raises(StrategyError, match="does not accept position targets"):
        select_strategy(capabilities(accepts_position_target=False), "map",
                        speed_mps=1.0, forced="position")
    with pytest.raises(StrategyError, match="nothing to fly with"):
        select_strategy(capabilities(accepts_position_target=False,
                                     accepts_velocity_target=False),
                        "map", speed_mps=1.0)


def test_a_tour_frame_the_vehicle_lacks_is_converted_not_refused() -> None:
    chosen = select_strategy(capabilities(frames=("local_ned",)), "map",
                             speed_mps=1.0)
    assert isinstance(chosen, PositionStrategy) and chosen.frame == "local_ned"
    with pytest.raises(StrategyError, match="accepts no position frame"):
        select_strategy(capabilities(frames=()), "map", speed_mps=1.0)


def test_velocity_backend_signs_follow_the_public_convention() -> None:
    tour = load_tour(FORWARD_TOUR)
    context = FrameContext(39.0, 29.0)
    legs = build_legs(tour, context, frame="map", speed_mps=1.0)
    strategy = VelocityStrategy(1.0)
    # Facing north with the waypoint due east: strafe right, no forward.
    leg = legs[0]
    sample = Sample(leg.north_m, leg.east_m - 5.0, leg.down_m, 0.0, 0.0, 0.0)
    command = strategy.velocity_for(leg, sample, 1.0)
    assert command.forward_mps == pytest.approx(0.0, abs=1e-9)
    assert command.right_mps == pytest.approx(1.0)
    # Facing east at the same offset: the same displacement is now forward.
    command = strategy.velocity_for(
        leg, Sample(leg.north_m, leg.east_m - 5.0, leg.down_m, 90.0, 0.0, 0.0), 1.0)
    assert command.forward_mps == pytest.approx(1.0)
    assert command.right_mps == pytest.approx(0.0, abs=1e-9)
    # Positive yaw rate is clockwise, and increases compass heading.
    turning = strategy.velocity_for(
        legs[0], Sample(leg.north_m, leg.east_m, leg.down_m, 170.0, 0.0, 0.0), 1.0)
    assert turning.yaw_rate_dps > 0.0


# ---------------------------------------------------------------------------
# Arrival rules
# ---------------------------------------------------------------------------

def gate_follower(*, dwell_s: float, heading_deg: float = 0.0) -> Follower:
    tour = parse_tour({
        "schema_version": 1, "tour_id": "gate", "coordinate_frame": "local_ned",
        "waypoint_tolerance_m": 0.05, "arrival_speed_mps": 0.15,
        "heading_tolerance_deg": 5.0,
        "waypoints": [{"north_m": 0.0, "east_m": 0.0, "down_m": -1.5,
                       "heading_deg": heading_deg, "dwell_s": dwell_s},
                      {"north_m": 5.0, "east_m": 0.0, "down_m": -1.5,
                       "heading_deg": heading_deg, "dwell_s": 0.0}],
    }, path=Path("gate.json"))
    legs = build_legs(tour, FrameContext(0.0, 0.0), frame="local_ned",
                      speed_mps=1.0)
    return Follower(tour, legs, speed_mps=1.0)


def sample_at(distance_m: float, t: float, *, speed: float = 0.0,
              heading: float = 0.0) -> Sample:
    return Sample(distance_m, 0.0, -1.5, heading, speed, t)


def test_a_zero_dwell_waypoint_is_not_a_fly_through_point() -> None:
    follower = gate_follower(dwell_s=0.0)
    follower.begin_leg(sample_at(2.0, 0.0, speed=1.0))
    assert follower.update(sample_at(1.0, 1.0, speed=1.0)) is FollowerEvent.NONE
    # Inside the distance gate but still moving: not an arrival.
    assert follower.update(sample_at(0.02, 2.0, speed=0.9)) is FollowerEvent.NONE
    assert follower.update(sample_at(0.02, 2.1, speed=0.1)) is FollowerEvent.ARRIVED
    assert follower.index == 1


def test_dwell_must_be_held_continuously_and_resets_on_leaving_the_gate() -> None:
    follower = gate_follower(dwell_s=1.0)
    follower.begin_leg(sample_at(1.0, 0.0, speed=1.0))
    assert follower.update(sample_at(0.01, 1.0)) is FollowerEvent.NONE
    assert follower.update(sample_at(0.01, 1.6)) is FollowerEvent.NONE
    # Blown out of the gate: the dwell clock starts again, so the sample that
    # would otherwise have completed the dwell does not.
    assert follower.update(sample_at(0.4, 1.8)) is FollowerEvent.NONE
    assert follower.update(sample_at(0.01, 2.1)) is FollowerEvent.NONE
    assert follower.update(sample_at(0.01, 3.2)) is FollowerEvent.ARRIVED
    assert follower.progress[0].dwell_s == pytest.approx(1.1)


def test_arrival_gates_speed_and_wrapped_heading() -> None:
    assert wrap_deg(350.0) == pytest.approx(-10.0)
    assert wrap_deg(-350.0) == pytest.approx(10.0)
    follower = gate_follower(dwell_s=0.0, heading_deg=2.0)
    follower.begin_leg(sample_at(1.0, 0.0))
    # 356 degrees is six degrees from a two-degree target the short way round,
    # which is outside a five-degree gate even though the raw difference is 354.
    assert follower.update(sample_at(0.01, 1.0, heading=356.0)) is FollowerEvent.NONE
    assert follower.update(sample_at(0.01, 1.1, heading=358.5)) is FollowerEvent.ARRIVED


def test_report_metrics_measure_overshoot_and_cross_track() -> None:
    follower = gate_follower(dwell_s=0.0)
    follower.begin_leg(Sample(-4.0, 0.0, -1.5, 0.0, 1.0, 0.0))
    follower.update(Sample(-2.0, 0.3, -1.5, 0.0, 1.0, 1.0))    # 0.3 m off the leg
    follower.update(Sample(0.25, 0.0, -1.5, 0.0, 1.0, 2.0))    # 0.25 m past it
    follower.update(sample_at(0.0, 3.0))
    assert follower.progress[0].max_cross_track_error_m == pytest.approx(0.3)
    assert follower.progress[0].overshoot_m == pytest.approx(0.25)
    assert follower.max_cross_track_error_m == pytest.approx(0.3)
    # Path length is summed between observed samples, not along the plan.
    assert follower.path_length_m == pytest.approx(
        math.hypot(2.25, 0.3) + 0.25, abs=1e-6)


@pytest.mark.parametrize("key,value", [
    ("arrival_speed_mps", 0.0),
    ("arrival_speed_mps", -0.1),
    ("heading_tolerance_deg", 0.0),
    ("max_state_age_s", -1.0),
    ("min_clearance_m", -0.5),
    ("settle_s", -1.0),
])
def test_unsatisfiable_arrival_gates_are_refused(tmp_path, key, value):
    """A negative gate is not a tighter gate; it can never be satisfied.

    Left unchecked, these reached the follower and the flight failed at the
    first waypoint with a message about vehicle health rather than about the
    tour that asked for the impossible.
    """
    payload = json.loads(FORWARD_TOUR.read_text())
    payload[key] = value
    path = tmp_path / "bad_gate.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(TourError) as excinfo:
        load_tour(path)

    assert key in str(excinfo.value)
