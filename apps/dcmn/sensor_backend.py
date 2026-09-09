"""What the sensor plane needs from whatever is simulating the vehicle.

`SensorManager` owns the parts of sensing that are the same whatever produces
the measurements: when each sensor is due in simulated time, which capture id
a tick's records share, how sequences advance, what the record envelope looks
like, and how a drop is accounted for.

Three things *are* about the simulator, and they were reached three different
ways -- an injected renderer, a module-level import of the ray service, and a
``vehicle`` object the state sensors read. This protocol is those three seams
named as one surface:

* :meth:`range_truth` -- the world the beams meet.
* :meth:`render_views` -- the cameras, with the profile-swap hooks around them.
* the vehicle datum -- :meth:`compass_heading`, :attr:`realism`,
  :attr:`origin_alt_m` and :meth:`map_to_gps`, which the state sensors read.

A backend answers those and gets the whole sensor plane -- manifest, rings,
capture ids, health accounting -- for free. That is the point: a second
provider (a bridge to another simulator, or a real vehicle) should have to
supply measurements, not reimplement the transport.

Nothing here imports the simulator, so a headless provider can implement this
without pulling in a renderer or a Tk window.
"""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

import numpy as np


@runtime_checkable
class SensorBackend(Protocol):
    """The measurement surface behind :class:`~dsim.sensor_manager.SensorManager`.

    ``runtime_checkable`` only proves the methods exist -- Python cannot check
    a signature at runtime -- so it catches a backend that forgot a seam, not
    one that got a seam wrong. The conformance tests are what check meaning.
    """

    # -- the world the beams meet -----------------------------------------

    def range_truth(self, pose_world, kind: str, model) -> np.ndarray:
        """True first-surface range along every ray of one capture.

        One entry per ray in the sensor's own ray order, in metres, ``inf``
        where the beam left the world without hitting anything. Noise, the
        reducer and the confidence model are applied above this by
        ``dsim.sensor_models``, so a backend returns truth and never a
        measurement: two backends sharing this contract share the sensor's
        error behaviour, which is the only way their numbers are comparable.
        """

    # -- the cameras -------------------------------------------------------

    def render_views(self, requests: Sequence[tuple[str, Any, Any, Any]]) -> None:
        """Fill each request's frame buffer in one pass.

        Each request is ``(sensor_id, model, pose_world, frame)``, where
        ``frame`` is a writable view onto the video ring's next slot. One call
        covers every camera due at the same capture, so a synchronized pair
        cannot straddle two renders.
        """

    def drop_views(self, camera_ids) -> None:
        """Release whatever a removed camera was holding. May be a no-op."""

    def prepare_profile(self, profile) -> Any:
        """Do a profile swap's failable work before the publisher commits.

        Returns an opaque token handed back to :meth:`finish_profile`, or
        ``None`` when there is nothing to stage. A backend that cannot fail
        here need not implement it.
        """

    def finish_profile(self, prepared: Any, *, commit: bool) -> None:
        """Commit or discard what :meth:`prepare_profile` staged."""

    # -- the vehicle the state sensors read --------------------------------

    def compass_heading(self, yaw_deg: float) -> float:
        """The published compass heading for a private renderer yaw.

        The two are not the same number and their signs differ; see
        ``tests/test_dvision_coordinate_contract.py``. A backend with its own
        yaw convention converts here rather than anywhere else.
        """

    @property
    def realism(self):
        """The environment model the state sensors degrade themselves with."""

    @property
    def origin_alt_m(self) -> float:
        """Altitude of the map origin above mean sea level, in metres."""

    def map_to_gps(self, x: float, y: float, z: float) -> tuple[float, float, float]:
        """Map metres to ``(lat_deg, lon_deg, alt_m)``."""


class BaseSensorBackend:
    """Refuses every seam it was not given, by name.

    A backend that renders no cameras, or has no vehicle datum, is a normal
    thing -- a bench rig, a range-only bridge -- and it should fail where the
    missing capability is actually used, naming what is missing. Subclass this
    and override what you can answer; the rest raise ``RuntimeError`` at the
    one sensor that needed them, which `SensorManager` records as that
    sensor's fault without stopping the run.
    """

    def range_truth(self, pose_world, kind: str, model) -> np.ndarray:
        raise RuntimeError(f'{kind}: this backend measures no ranges')

    def render_views(self, requests) -> None:
        raise RuntimeError('a camera is scheduled but this backend renders none')

    def drop_views(self, camera_ids) -> None:
        """Nothing held, nothing to release."""

    def prepare_profile(self, profile):
        """No staged work, so the swap has nothing that can fail here."""
        return None

    def finish_profile(self, prepared, *, commit: bool) -> None:
        """Nothing was staged, so there is nothing to commit or discard."""

    def compass_heading(self, yaw_deg: float) -> float:
        raise RuntimeError('this backend publishes no compass heading')

    @property
    def realism(self):
        raise RuntimeError('this backend has no vehicle datum')

    @property
    def origin_alt_m(self) -> float:
        raise RuntimeError('this backend has no vehicle datum')

    def map_to_gps(self, x: float, y: float, z: float):
        raise RuntimeError('this backend has no vehicle datum')
