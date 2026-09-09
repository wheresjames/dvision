"""`dsim` as a sensor backend: the Panda3D-shaped half of sensing.

Everything the sensor plane needs that is specific to *this* simulator ends up
here -- the ray service that intersects the map's geometry, the renderer that
fills camera frames, and the running vehicle the state sensors read their
datum from. `SensorManager` talks to this through
:class:`dcmn.sensor_backend.SensorBackend` and to nothing else in ``dsim``,
which is what lets the scheduling, capture-id and transport machinery be
reused by a provider that is not this simulator.

The three seams were previously reached three different ways: an injected
``renderer``, a module-level import of ``dsim.range``, and a ``vehicle``
object. Composing them into one object is not a rename -- it is what makes the
surface countable, so a second backend can be checked against it.
"""

from __future__ import annotations

from dcmn.sensor_backend import BaseSensorBackend
from dsim.dsim import sim_yaw_to_compass_heading
from dsim.range import cast, scene_geometry


class SimulatorBackend(BaseSensorBackend):
    """Measurements from a running `DroneSimulator` and its renderer.

    ``renderer`` and ``vehicle`` are optional because a rig can legitimately
    have neither -- a range-only bench, a manifest test. Each absent one fails
    at the sensor that needed it, with the message naming what is missing,
    which `SensorManager` records as that sensor's fault rather than stopping
    the run.
    """

    def __init__(self, sim_map, renderer=None, vehicle=None):
        self.sim_map = sim_map
        self.renderer = renderer
        self.vehicle = vehicle

    # -- the world the beams meet -----------------------------------------

    def range_truth(self, pose_world, kind, model):
        # scene_geometry caches per object list, so this is a dict lookup on
        # every capture after the first.
        return cast(scene_geometry(self.sim_map), pose_world, kind, model)

    # -- the cameras -------------------------------------------------------

    def render_views(self, requests):
        if self.renderer is None:
            raise RuntimeError('a camera is scheduled but no renderer is attached')
        self.renderer.render_views(requests)

    def drop_views(self, camera_ids):
        if self.renderer is not None:
            self.renderer.drop_views(camera_ids)

    def prepare_profile(self, profile):
        # Not every renderer stages a swap; the ones that do allocate the new
        # generation's targets here so a failure happens before the publisher
        # has committed anything.
        prepare = getattr(self.renderer, 'prepare_profile', None)
        return None if prepare is None else prepare(profile)

    def finish_profile(self, prepared, *, commit):
        if prepared is not None:
            self.renderer.finish_profile(prepared, commit=commit)

    # -- the vehicle the state sensors read --------------------------------

    def compass_heading(self, yaw_deg):
        return sim_yaw_to_compass_heading(yaw_deg)

    @property
    def realism(self):
        return self._vehicle().realism

    @property
    def origin_alt_m(self):
        return self._vehicle().origin_alt_m

    def map_to_gps(self, x, y, z):
        return self._vehicle().map_to_gps(x, y, z)

    def _vehicle(self):
        if self.vehicle is None:
            raise RuntimeError('a state sensor is scheduled but no vehicle is attached')
        return self.vehicle
