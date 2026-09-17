from __future__ import annotations

import threading
import math
import time
import unittest
import xml.etree.ElementTree as ET

from dfgb.dfgb import (
    FGBridge,
    FG_CTRL_PORT,
    FGControl,
    FGState,
    FPS_TO_MPS,
    METERS_PER_DEG_LAT,
    PROTOCOLS_DIR,
    compute_drone_control,
    integrate_axis_velocity,
    integrate_drone_pose,
    integrate_visual_attitude,
)


class FakeSocket:
    def __init__(self) -> None:
        self.sent: list[tuple[bytes, tuple[str, int]]] = []

    def sendto(self, data: bytes, addr: tuple[str, int]) -> None:
        self.sent.append((data, addr))

    @property
    def last_line(self) -> str:
        return self.sent[-1][0].decode("ascii")

    @property
    def last_values(self) -> tuple[float, ...]:
        return tuple(
            float(part) for part in self.last_line.strip().split(",")
        )


def make_bridge() -> tuple[FGBridge, FakeSocket]:
    bridge = FGBridge.__new__(FGBridge)
    bridge.armed = False
    bridge.mode = "DISARMED"
    bridge.cmd_forward = 0.0
    bridge.cmd_right = 0.0
    bridge.cmd_up = 0.0
    bridge.cmd_yaw_rate = 0.0
    bridge.target_alt = None
    bridge.origin_alt_m = None
    bridge._desired_lat_deg = None
    bridge._desired_lon_deg = None
    bridge._desired_alt_m = None
    bridge._desired_heading_deg = None
    bridge._body_forward_mps = 0.0
    bridge._body_right_mps = 0.0
    bridge._visual_roll_deg = 0.0
    bridge._visual_pitch_deg = 0.0
    bridge._last_control_time = time.monotonic()
    bridge.last_cmd_monotonic = None
    bridge.last_cmd_type = ""
    bridge.command_count = 0
    bridge.status_message = "ready"
    bridge.fg_state = FGState()
    bridge._state_lock = threading.Lock()
    sock = FakeSocket()
    bridge._ctrl_sock = sock
    return bridge, sock


class DfgbControlMathTest(unittest.TestCase):
    def test_protocol_xml_order_matches_emitted_csv(self) -> None:
        control_xml = PROTOCOLS_DIR / "dvision2-ctrl.xml"
        root = ET.parse(control_xml).getroot()
        names = [chunk.findtext("name") for chunk in root.findall(".//input/chunk")]

        self.assertEqual(
            names,
            [
                "throttle",
                "aileron",
                "elevator",
                "rudder",
                "latitude",
                "longitude",
                "altitude-ft",
                "heading-deg",
                "roll-deg",
                "pitch-deg",
            ],
        )
        self.assertEqual(
            FGControl(0.1, 0.2, -0.3, 0.4).csv_line(),
            "0.1000,0.2000,-0.3000,0.4000\n",
        )
        self.assertEqual(
            FGControl(
                0.0, 0.0, 0.0, 0.0,
                37.1, -122.2, 123.4567, 90.0, 4.0, -6.0,
            ).csv_line(),
            "0.0000,0.0000,0.0000,0.0000,37.100000000,"
            "-122.200000000,123.4567,90.0000,4.0000,-6.0000\n",
        )

    def test_zero_velocity_holds_position_with_centered_surfaces(self) -> None:
        control = compute_drone_control(
            lat_deg=37.0,
            lon_deg=-122.0,
            altitude_ft=321.0,
            heading_deg=90.0,
            roll_deg=4.0,
            pitch_deg=-2.0,
        )

        self.assertEqual(
            control,
            FGControl(0.0, 0.0, 0.0, 0.0, 37.0, -122.0, 321.0, 90.0, 4.0, -2.0),
        )

    def test_pose_integration_moves_forward_east_at_heading_90(self) -> None:
        lat, lon, alt, heading = integrate_drone_pose(
            cmd_forward=2.0,
            cmd_right=0.0,
            cmd_up=0.0,
            cmd_yaw_rate=0.0,
            lat_deg=37.0,
            lon_deg=-122.0,
            alt_m=100.0,
            heading_deg=90.0,
            dt=1.0,
        )

        lon_scale = METERS_PER_DEG_LAT * math.cos(math.radians(37.0))
        self.assertAlmostEqual(lat, 37.0)
        self.assertAlmostEqual(lon, -122.0 + 2.0 / lon_scale)
        self.assertAlmostEqual(alt, 100.0)
        self.assertAlmostEqual(heading, 90.0)

    def test_pose_integration_strafes_right_south_at_heading_90(self) -> None:
        lat, lon, alt, heading = integrate_drone_pose(
            cmd_forward=0.0,
            cmd_right=3.0,
            cmd_up=1.0,
            cmd_yaw_rate=-15.0,
            lat_deg=37.0,
            lon_deg=-122.0,
            alt_m=100.0,
            heading_deg=90.0,
            dt=2.0,
        )

        self.assertAlmostEqual(lat, 37.0 - 6.0 / METERS_PER_DEG_LAT)
        self.assertAlmostEqual(lon, -122.0)
        self.assertAlmostEqual(alt, 102.0)
        self.assertAlmostEqual(heading, 60.0)

    def test_visual_attitude_holds_commanded_forward_and_right_lean(self) -> None:
        roll, pitch = integrate_visual_attitude(
            cmd_forward=15.0,
            cmd_right=-10.0,
            roll_deg=0.0,
            pitch_deg=0.0,
            dt=1.0,
        )

        self.assertAlmostEqual(roll, -12.0, places=2)
        self.assertAlmostEqual(pitch, -18.0, places=2)

    def test_axis_velocity_coasts_after_command_returns_to_zero(self) -> None:
        accelerated = integrate_axis_velocity(
            current_mps=0.0,
            target_mps=15.0,
            dt=0.25,
        )
        coasting = integrate_axis_velocity(
            current_mps=accelerated,
            target_mps=0.0,
            dt=0.25,
        )

        self.assertGreater(accelerated, 0.0)
        self.assertGreater(coasting, 0.0)
        self.assertLess(coasting, accelerated)


class DfgbBridgeControlTest(unittest.TestCase):
    def test_disarmed_bridge_sends_zeroed_controls(self) -> None:
        bridge, sock = make_bridge()

        bridge._send_control()

        self.assertEqual(
            sock.sent,
            [(b"0.0000,0.0000,0.0000,0.0000\n", ("127.0.0.1", FG_CTRL_PORT))],
        )

    def test_velocity_command_is_ignored_while_disarmed(self) -> None:
        bridge, sock = make_bridge()

        bridge._apply_command({
            "type": "velocity",
            "forward_mps": 4.0,
            "right_mps": 3.0,
            "up_mps": 2.0,
            "yaw_rate_dps": 30.0,
        })
        bridge._send_control()

        self.assertEqual(
            (bridge.cmd_forward, bridge.cmd_right,
             bridge.cmd_up, bridge.cmd_yaw_rate),
            (0.0, 0.0, 0.0, 0.0),
        )
        self.assertEqual(sock.last_values, (0.0, 0.0, 0.0, 0.0))

    def test_armed_velocity_command_publishes_kinematic_pose(self) -> None:
        bridge, sock = make_bridge()
        bridge._apply_command({"type": "arm", "armed": True})
        bridge._desired_lat_deg = 37.0
        bridge._desired_lon_deg = -122.0
        bridge._desired_alt_m = 100.0
        bridge._desired_heading_deg = 0.0
        bridge._last_control_time = time.monotonic() - 1.0

        bridge._apply_command({
            "type": "velocity",
            "forward_mps": 2.0,
            "right_mps": 1.0,
            "up_mps": 0.75,
            "yaw_rate_dps": 10.0,
        })
        bridge._send_control()

        (
            throttle, aileron, elevator, rudder,
            lat, lon, altitude_ft, heading, roll, pitch,
        ) = sock.last_values
        self.assertEqual((throttle, aileron, elevator, rudder), (0.0, 0.0, 0.0, 0.0))
        self.assertGreater(lat, 37.0)
        self.assertLess(lat, 37.0 + 0.5 / METERS_PER_DEG_LAT)
        self.assertGreater(lon, -122.0)
        self.assertAlmostEqual(altitude_ft, 100.1875 / FPS_TO_MPS, places=4)
        self.assertAlmostEqual(heading, 2.5, places=4)
        self.assertGreater(roll, 0.0)
        self.assertLess(pitch, 0.0)

    def test_neutral_velocity_command_decelerates_without_stopping_instantly(self) -> None:
        bridge, sock = make_bridge()
        bridge._apply_command({"type": "arm", "armed": True})
        bridge._desired_lat_deg = 37.0
        bridge._desired_lon_deg = -122.0
        bridge._desired_alt_m = 100.0
        bridge._desired_heading_deg = 0.0
        bridge._body_forward_mps = 8.0
        bridge._last_control_time = time.monotonic() - 1.0

        bridge._apply_command({"type": "velocity", "forward_mps": 0.0})
        bridge._send_control()

        self.assertGreater(bridge._body_forward_mps, 0.0)
        self.assertLess(bridge._body_forward_mps, 8.0)
        self.assertGreater(sock.last_values[4], 37.0)

    def test_up_velocity_integrates_direct_altitude_hold(self) -> None:
        bridge, sock = make_bridge()
        bridge._apply_command({"type": "arm", "armed": True})
        bridge._desired_lat_deg = 37.0
        bridge._desired_lon_deg = -122.0
        bridge._desired_alt_m = 100.0
        bridge._desired_heading_deg = 0.0
        bridge._last_control_time = time.monotonic() - 1.0
        bridge._apply_command({"type": "velocity", "up_mps": 2.0})

        bridge._send_control()

        self.assertAlmostEqual(sock.last_values[6], 100.5 / FPS_TO_MPS, places=4)
        self.assertAlmostEqual(bridge._desired_alt_m, 100.5, places=4)

    def test_takeoff_target_commands_absolute_altitude_then_returns_to_guided(self) -> None:
        bridge, sock = make_bridge()
        bridge._apply_command({"type": "arm", "armed": True})
        bridge.origin_alt_m = 100.0
        bridge.fg_state.alt_ft = 100.0 / FPS_TO_MPS
        bridge._apply_command({"type": "takeoff", "alt_m": 3.0})

        bridge._send_control()
        self.assertEqual(bridge.mode, "TAKEOFF")
        self.assertAlmostEqual(sock.last_values[6], 103.0 / FPS_TO_MPS, places=4)

        bridge.fg_state.alt_ft = 103.0 / FPS_TO_MPS
        bridge.fg_state.speed_down_fps = 0.0
        bridge._send_control()
        self.assertIsNone(bridge.target_alt)
        self.assertEqual(bridge.mode, "GUIDED")
        self.assertAlmostEqual(sock.last_values[6], 103.0 / FPS_TO_MPS, places=4)

    def test_land_target_commands_origin_altitude_and_disarms_when_settled(self) -> None:
        bridge, sock = make_bridge()
        bridge._apply_command({"type": "arm", "armed": True})
        bridge.origin_alt_m = 100.0
        bridge.fg_state.alt_ft = 103.0 / FPS_TO_MPS
        bridge._apply_command({"type": "land"})

        bridge._send_control()
        self.assertEqual(bridge.mode, "LAND")
        self.assertAlmostEqual(sock.last_values[6], 100.0 / FPS_TO_MPS, places=4)

        bridge.fg_state.alt_ft = 100.0 / FPS_TO_MPS
        bridge.fg_state.speed_down_fps = 0.0
        bridge._send_control()
        self.assertFalse(bridge.armed)
        self.assertEqual(bridge.mode, "DISARMED")
        self.assertEqual(sock.last_values, (0.0, 0.0, 0.0, 0.0))

    def test_zero_command_holds_position_with_zero_speed_and_current_altitude(self) -> None:
        bridge, sock = make_bridge()
        bridge._apply_command({"type": "arm", "armed": True})
        bridge._desired_lat_deg = 37.0
        bridge._desired_lon_deg = -122.0
        bridge._desired_alt_m = 100.0
        bridge._desired_heading_deg = 0.0
        bridge._apply_command({
            "type": "velocity",
            "forward_mps": 1.0,
            "right_mps": 1.0,
            "up_mps": 1.0,
            "yaw_rate_dps": 10.0,
        })
        bridge._apply_command({"type": "zero"})

        bridge._send_control()

        self.assertEqual(bridge.mode, "HOLD")
        self.assertEqual(
            (bridge.cmd_forward, bridge.cmd_right,
             bridge.cmd_up, bridge.cmd_yaw_rate),
            (0.0, 0.0, 0.0, 0.0),
        )
        self.assertEqual(sock.last_values[:4], (0.0, 0.0, -0.0, 0.0))
        self.assertAlmostEqual(sock.last_values[6], 100.0 / FPS_TO_MPS, places=4)

    def test_state_datagram_populates_velocity_units_for_control_loop(self) -> None:
        bridge, _sock = make_bridge()
        bridge._state_sock = DatagramSocket(
            b"37.0,-122.0,328.0840,1.0,2.0,90.0,3.2808,"
            b"6.5617,-1.6404,4.0,9.8425,-3.2808\n"
        )

        bridge._recv_state()

        self.assertAlmostEqual(bridge.fg_state.vx_mps, 2.0, places=3)
        self.assertAlmostEqual(bridge.fg_state.vy_mps, 1.0, places=3)
        self.assertAlmostEqual(bridge.fg_state.vz_mps, 0.5, places=3)
        self.assertAlmostEqual(bridge.fg_state.u_fps * FPS_TO_MPS, 3.0, places=3)
        self.assertAlmostEqual(bridge.fg_state.v_fps * FPS_TO_MPS, -1.0, places=3)
        self.assertIsNotNone(bridge.origin_alt_m)
        self.assertEqual(bridge._desired_lat_deg, bridge.fg_state.lat_deg)
        self.assertEqual(bridge._desired_lon_deg, bridge.fg_state.lon_deg)
        self.assertEqual(bridge._desired_alt_m, bridge.fg_state.alt_m)
        self.assertEqual(bridge._desired_heading_deg, bridge.fg_state.heading_deg)
        self.assertEqual(bridge._visual_roll_deg, bridge.fg_state.roll_deg)
        self.assertEqual(bridge._visual_pitch_deg, bridge.fg_state.pitch_deg)


class DatagramSocket:
    def __init__(self, datagram: bytes) -> None:
        self._datagrams = [datagram]

    def recvfrom(self, _size: int) -> tuple[bytes, tuple[str, int]]:
        if not self._datagrams:
            raise BlockingIOError
        return self._datagrams.pop(0), ("127.0.0.1", 5501)


if __name__ == "__main__":
    unittest.main()
