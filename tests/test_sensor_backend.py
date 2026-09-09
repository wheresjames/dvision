"""The sensor plane's one simulator-shaped seam, and what it is worth.

These check the boundary rather than any measurement: that the surface is
countable, that `dsim` covers it, that the scheduling half no longer reaches
through it into the simulator, and that a backend which cannot answer a seam
fails at the sensor that needed it instead of stopping the run.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dcmn.sensor_backend import BaseSensorBackend, SensorBackend
from dsim.backend import SimulatorBackend
from dsim.profiles import DroneProfile, camera_profile, default_profile
from dsim.sensor_manager import SensorManager
from dsim.dsim import DroneState

from tests.test_sensor_contract import Renderer, _Vehicle, instance
from tests.test_sensor_geometry import wall

ROOT = Path(__file__).resolve().parents[1]


def test_dsim_implements_the_backend_protocol():
    backend = SimulatorBackend(SimpleNamespace(objects=[]), Renderer(), vehicle=_Vehicle())
    # runtime_checkable proves presence, not signatures -- the rest of this
    # module is what checks that the seams mean what the manager expects.
    assert isinstance(backend, SensorBackend)
    assert isinstance(BaseSensorBackend(), SensorBackend)
    # Every name the manager reaches for, and no more: a seam added here
    # without a reason is a seam every future backend has to reimplement, so
    # the count is worth pinning.
    surface = {n for n in dir(SensorBackend) if not n.startswith('_')}
    assert surface == {'range_truth', 'render_views', 'drop_views', 'prepare_profile',
                       'finish_profile', 'compass_heading', 'realism', 'origin_alt_m',
                       'map_to_gps'}


def test_the_scheduler_never_reaches_the_simulator():
    """`sensor_manager` is the reusable half, so dsim must stay out of it.

    A bridge to another simulator reuses the schedule, capture ids, sequences
    and record envelope. That should not cost it a renderer, the ray service
    or a Tk window -- not at import, and not at the first capture of any
    sensor kind either. Checking only the import is what let the compass go on
    importing `dsim.dsim` from inside its own function body, invisibly, for as
    long as no test looked at `sys.modules` after a capture.

    `bridge_probe` runs a whole profile through a backend that is not dsim, in
    a fresh interpreter, because an import already done in this process would
    hide the coupling completely.
    """
    done = subprocess.run([sys.executable, str(ROOT / 'tests' / 'bridge_probe.py')],
                          capture_output=True, text=True, cwd=ROOT)
    assert done.returncode == 0, done.stderr
    assert done.stdout.startswith('CLEAN'), done.stdout


def test_a_backend_that_cannot_answer_a_seam_never_reaches_the_physics_loop():
    """A camera capture reads the body datum through the backend as well.

    That datum is read once for the whole render pass rather than once per
    sensor, which is exactly what makes it easy to leave outside the manager's
    accounting -- and outside it, a backend with no yaw convention raises
    through `tick` and takes the simulator down instead of failing its own
    cameras.
    """
    class NoHeading(BaseSensorBackend):
        """Renders views, but has no vehicle and so no compass convention."""

        def __init__(self):
            self.passes = 0

        def render_views(self, requests):
            self.passes += 1
            for _sensor_id, _model, _pose, frame in requests:
                frame[:] = 0

    backend = NoHeading()
    profile = DroneProfile.parse(camera_profile(16, 12, physics_hz=30.))
    sensors = SensorManager(instance(), profile, backend)
    try:
        assert sensors.tick(DroneState(3.0, 3.0, 2.0), 1 / 30) is None
        counts = sensors.production()['front']
        assert counts['published'] == 0 and counts['drops'] == 1
        assert 'no compass heading' in counts['fault']
        # And nothing was rendered into a slot whose record could never be
        # written: the seam fails before the pass, not halfway through it.
        assert backend.passes == 0
    finally:
        sensors.close()


def test_a_backend_answers_the_three_seams_the_manager_uses():
    """One tick, and every seam is exercised: camera, ray and state sensor."""
    draft = camera_profile(16, 12, physics_hz=30.)
    draft['sensors'].append(dict(id='sonar', type='range.ultrasonic', parent='body',
                                 rate_hz=30., pose_parent=dict(pitch_deg=-90.)))
    draft['sensors'].append(dict(id='gnss', type='position.gnss', parent='body',
                                 rate_hz=30.))
    profile = DroneProfile.parse(draft)
    renderer = Renderer()
    backend = SimulatorBackend(SimpleNamespace(objects=[wall(4.5, 3.5)]), renderer,
                               vehicle=_Vehicle())
    sensors = SensorManager(instance(), profile, backend)
    try:
        sensors.tick(DroneState(3.0, 3.0, 2.0), 1 / 30)
        production = sensors.production()
        assert renderer.passes, 'the camera seam was never reached'
        for sensor_id in ('front', 'sonar', 'gnss'):
            counts = production[sensor_id]
            assert counts['published'] == 1 and not counts['drops'], (sensor_id, counts)
    finally:
        sensors.close()


def test_a_missing_seam_fails_only_the_sensor_that_needed_it():
    """A backend with no vehicle datum still flies its cameras and rangefinders.

    The whole point of routing failures through the manager's accounting is
    that a provider nobody may block reports a shortfall instead of raising
    into the physics loop.
    """
    draft = camera_profile(16, 12, physics_hz=30.)
    draft['sensors'].append(dict(id='sonar', type='range.ultrasonic', parent='body',
                                 rate_hz=30., pose_parent=dict(pitch_deg=-90.)))
    draft['sensors'].append(dict(id='gnss', type='position.gnss', parent='body',
                                 rate_hz=30.))
    profile = DroneProfile.parse(draft)
    # No vehicle: the state sensor has nothing to read its datum from.
    backend = SimulatorBackend(SimpleNamespace(objects=[wall(4.5, 3.5)]), Renderer())
    sensors = SensorManager(instance(), profile, backend)
    try:
        sensors.tick(DroneState(3.0, 3.0, 2.0), 1 / 30)
        production = sensors.production()
        assert production['front']['published'] == 1
        assert production['sonar']['published'] == 1
        assert production['gnss']['published'] == 0
        assert production['gnss']['drops'] == 1
        assert 'no vehicle is attached' in production['gnss']['fault']
    finally:
        sensors.close()


def test_the_base_backend_names_the_seam_it_cannot_answer():
    base = BaseSensorBackend()
    with pytest.raises(RuntimeError, match='measures no ranges'):
        base.range_truth(np.eye(4), 'range.laser', {})
    with pytest.raises(RuntimeError, match='renders none'):
        base.render_views([])
    with pytest.raises(RuntimeError, match='no vehicle datum'):
        base.realism  # noqa: B018  -- reading the property is the check
    with pytest.raises(RuntimeError, match='no vehicle datum'):
        base.map_to_gps(0., 0., 0.)
    # The optional camera-lifecycle hooks are the ones a backend may ignore,
    # so they must be no-ops rather than refusals.
    assert base.prepare_profile(default_profile()) is None
    base.finish_profile(None, commit=True)
    base.drop_views(set())
