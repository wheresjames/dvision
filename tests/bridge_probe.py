"""A full sensor run driven by a backend that is not this simulator.

Run as a script in a fresh interpreter by ``test_sensor_backend.py``, and not
importable as a test: the whole point is a process where nothing has imported
``dsim.dsim`` yet, so an import that only happens on the first capture of some
sensor cannot hide behind one the test file already did.

Every sensor kind the manager schedules is exercised, because the coupling
this guards against is per-sensor -- a function-local ``import dsim.dsim`` in
one state sensor stays invisible until that sensor fires.
"""

from __future__ import annotations

import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'apps')]

import dsim.sensor_manager  # noqa: E402,F401  -- imported for its side effect on sys.modules

#: What the reusable half must never reach: the simulator, the ray service it
#: owns, and the UI toolkit the simulator drags along behind it.
SIMULATOR = ('dsim.dsim', 'dsim.range', 'tkinter')


def leaked():
    return sorted(module for module in SIMULATOR if module in sys.modules)


assert leaked() == [], f'importing the scheduler pulled in {leaked()}'

import numpy as np  # noqa: E402

from dcmn.sensor_backend import BaseSensorBackend  # noqa: E402
from dsim.profiles import DroneProfile, camera_profile  # noqa: E402
from dsim.realism import REALISM_DEFAULTS, Realism  # noqa: E402
from dsim.sensor_manager import SensorManager  # noqa: E402
from dsim.sensor_models import directions  # noqa: E402


@dataclass
class State:
    """Only the fields the sensor plane reads, so no simulator is needed.

    A dataclass rather than a stub object because the manager freezes each
    capture's snapshot with ``dataclasses.replace``.
    """

    x: float = 3.0
    y: float = 3.0
    z: float = 2.0
    roll_deg: float = 0.0
    pitch_deg: float = 0.0
    yaw_deg: float = 90.0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0


class Bridge(BaseSensorBackend):
    """A provider with its own geometry, its own datum and its own yaw.

    The yaw convention is deliberately not dsim's -- this backend's headings
    are already compass -- so anything that converts by importing dsim's
    conversion instead of asking here produces a visibly different number.
    """

    def range_truth(self, pose_world, kind, model):
        return np.full(len(directions(kind, model)), 5.0)

    def render_views(self, requests):
        for _sensor_id, _model, _pose, frame in requests:
            frame[:] = 0

    def compass_heading(self, yaw_deg):
        return float(yaw_deg) % 360.0

    @property
    def realism(self):
        return Realism.from_settings(REALISM_DEFAULTS)

    @property
    def origin_alt_m(self):
        return 34.0

    def map_to_gps(self, x, y, z):
        return 52.0 + y * 1e-5, 13.0 + x * 1e-5, 34.0 + z


def profile():
    """One of every sensor kind the manager knows how to schedule."""
    draft = camera_profile(16, 12, physics_hz=30.)
    draft['sensors'] += [
        dict(id='sonar', type='range.ultrasonic', parent='body', rate_hz=30.,
             pose_parent=dict(pitch_deg=-90.)),
        dict(id='imu', type='motion.imu', parent='body', rate_hz=30.),
        dict(id='gnss', type='position.gnss', parent='body', rate_hz=30.),
        dict(id='baro', type='altimeter.barometric', parent='body', rate_hz=30.),
        dict(id='mag', type='heading.magnetometer', parent='body', rate_hz=30.),
        dict(id='temp', type='environment.temperature', parent='body', rate_hz=30.),
    ]
    return DroneProfile.parse(draft)


def main():
    backend = Bridge()
    sensors = SensorManager('bridge-' + uuid.uuid4().hex[:8], profile(), backend)
    try:
        # Two ticks: the IMU differentiates successive snapshots, so a single
        # tick never reaches the branch that resolves a previous state.
        sensors.tick(State(), 1 / 30)
        sensors.tick(State(), 2 / 30)
        production = sensors.production()
    finally:
        sensors.close()
    for sensor_id, counts in production.items():
        assert counts['published'] == 2 and not counts['drops'], (sensor_id, counts)
    assert leaked() == [], f'a capture pulled in {leaked()}'
    print('CLEAN ' + ' '.join(sorted(production)))


if __name__ == '__main__':
    main()
