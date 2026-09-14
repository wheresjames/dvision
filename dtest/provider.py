"""A small deterministic sensor/pose provider: the standalone test and replay launcher.

It is a session provider exactly as dsim is -- it owns the neutral context
(session id, report root, data clock, frame, epochs, ideal pose), the sensor
plane and the module-bus ring -- but it needs no renderer, no world file and no
tour. Its scene is generated in memory: a few axis-aligned boxes, cast against
analytically. That is what lets dalg/dnav process tests run with world/tour
files and truth channels unavailable.

Everything it publishes goes through the same interfaces a real provider uses:
:class:`dcmn.context.Context` for session and pose, :class:`dcmn.sensors.
SensorPublisher` for lidar scans and camera frames (with capture-associated
pose metadata), and the instance module bus.

Test hooks mirror the provider-side events DV-MAPPING §9 distinguishes:
``announce_localization_reset``, ``announce_clock_reset``, ``sensor_reset``,
``set_sensors`` (a manifest change), ``pose_valid`` (an unavailable pose),
``mute`` (a sample gap), ``rollover`` (a new recording session) and ``close``.
``publish_reference_image`` / ``withdraw_reference_image`` mirror the optional
imagery channel (DV-MAPPING §7): a caller-supplied PNG and affine, or a plan
rendering of the scene's own boxes, on the plane no algorithm reads.

As a process::

    python dtest/provider.py --id NAME --report-dir DIR [--sensors scan,front]
        [--path x,y;x,y;...] [--speed 1.5] [--timeout 60]
"""
from __future__ import annotations

if __name__ == '__main__':
    import sys as _sys
    from pathlib import Path as _Path
    for _path in (str(_Path(__file__).resolve().parents[1]),
                  str(_Path(__file__).resolve().parents[1] / 'apps')):
        if _path not in _sys.path: _sys.path.insert(0, _path)

from dataclasses import dataclass
import io
import math
from pathlib import Path
import time
import uuid

import numpy as np

from dcmn.context import Context
from dcmn.module_bus import PymembusModuleBus, requests_shutdown
from dcmn.sensors import CAMERA_FRAME, LIDAR_FRAME, STATUS_INVALID, STATUS_VALID, SensorPublisher

#: Height of every box, metres. Above any flight altitude a test uses.
WALL_H = 3.0
PHYSICS_HZ = 30.
#: The default scene: a room spanning negative and positive coordinates, with
#: an interior wall and one doorway -- enough to force a detour and to exercise
#: negative grid origins.
ROOM = ((-12., -12., 18., -11.5), (-12., 17.5, 18., 18.),       # north, south walls
        (-12., -12., -11.5, 18.), (17.5, -12., 18., 18.),       # west, east walls
        (4., -12., 4.5, 1.), (4., 4., 4.5, 18.))                # divider with a doorway 1..4


@dataclass
class VehicleState:
    """What dsim.transforms needs: sim-internal yaw, 270 - compass heading."""
    x: float
    y: float
    z: float
    yaw_deg: float
    roll_deg: float = 0.
    pitch_deg: float = 0.

    @property
    def heading_deg(self): return (270. - self.yaw_deg) % 360.


def fixture_profile(sensors=('scan', 'front'), *, camera_hz=10., scan_hz=10., width=160, height=120):
    """A drone profile with a noiseless 2D lidar ``scan`` and a camera ``front``.

    The camera is always declared (a drone profile requires a primary camera);
    a lidar-only fixture simply never publishes frames on it unless asked.
    """
    from dsim.profiles import SCHEMA, DroneProfile
    rows = [dict(id='front', type='camera.rgb', enabled=True, rate_hz=camera_hz, parent='body',
                 pose_parent=dict(z_m=.1, pitch_deg=-8.),
                 model=dict(width_px=width, height_px=height, fov_h_deg=70., near_m=.15, far_m=60.))]
    if 'scan' in sensors:
        rows.append(dict(id='scan', type='lidar.scan2d', enabled=True, rate_hz=scan_hz, parent='body',
                         model=dict(fov_deg=360., samples=360, elevation_deg=0., min_range_m=.15,
                                    max_range_m=25., noise_std_m=0., quantization_m=0.,
                                    dropout_probability=0., limit_degradation=0.,
                                    confidence_model='range_linear')))
    return DroneProfile.parse(dict(schema=SCHEMA, name='fixture-' + '-'.join(sorted(sensors)) or 'fixture',
                                   physics_hz=PHYSICS_HZ, primary_camera='front', mounts=[], sensors=rows))


def cast(origins, directions, boxes, *, floor=False):
    """Distance along each unit direction to the first box face (or floor), inf if none."""
    origins = np.asarray(origins, np.float64); directions = np.asarray(directions, np.float64)
    ox, oy, oz = (origins[..., i] for i in range(3))
    dx, dy, dz = (directions[..., i] for i in range(3))
    best = np.full(np.broadcast(ox, dx).shape, np.inf)
    face = np.zeros(best.shape, np.int8); which = np.full(best.shape, -1, np.int16)
    with np.errstate(divide='ignore', invalid='ignore'):
        for index, (x0, y0, x1, y1) in enumerate(boxes):
            tx0, tx1 = (x0-ox)/dx, (x1-ox)/dx
            ty0, ty1 = (y0-oy)/dy, (y1-oy)/dy
            tminx, tmaxx = np.fmin(tx0, tx1), np.fmax(tx0, tx1)
            tminy, tmaxy = np.fmin(ty0, ty1), np.fmax(ty0, ty1)
            enter = np.fmax(tminx, tminy); leave = np.fmin(tmaxx, tmaxy)
            t = np.where(enter > 1e-9, enter, np.inf)
            z = oz + t*dz
            hit = (enter <= leave) & np.isfinite(t) & (z >= 0) & (z <= WALL_H) & (t < best)
            best = np.where(hit, t, best)
            face = np.where(hit, np.where(tminx > tminy, 0, 1), face)
            which = np.where(hit, index, which)
        if floor:
            t = np.where(dz < -1e-9, -oz/dz, np.inf)
            hit = t < best
            best = np.where(hit, t, best); which = np.where(hit, -2, which)
    return best, face, which


class FixtureProvider:
    """One provider instance. ``step`` advances the data clock by one physics tick."""

    def __init__(self, instance, report_root, *, sensors=('scan',), boxes=ROOM,
                 start=(0., 0., 1.5, 90.), camera_frames=None, provider_kind='ideal-fixture'):
        self.instance = instance
        self.boxes = tuple(boxes)
        self.sensor_ids = tuple(sensors)
        self.camera_frames = ('front' in sensors) if camera_frames is None else camera_frames
        self.time_s = 0.
        self.tick_index = 0
        self.localization_epoch = 0
        self.pose_valid = True
        self.publish_poses = True
        self.muted = set()
        self.state = VehicleState(start[0], start[1], start[2], 270. - start[3])
        self.path = []
        self.speed_mps = 1.5
        self.bus = PymembusModuleBus(instance, 'simulator', 'fixture-provider', create=True,
                                     sim_time=lambda: self.time_s)
        self.bus.connect()
        self.context = Context(instance)
        self.publisher = None
        self.profile = None
        self.imagery = None
        self.clock_epoch = uuid.uuid4().int & ((1 << 63) - 1)
        if sensors:
            self.profile = fixture_profile(sensors)
            self.publisher = SensorPublisher(instance, self.profile)
            self.clock_epoch = self.publisher.clock_epoch
            self.publisher.pose_kind = 'ideal'
        self.context.start(report_root, provider_kind=provider_kind, clock_epoch=self.clock_epoch, local_ned_origin=(0., 0., 0.))
        self.sequences = {}
        self._publish_pose()

    # -- motion -----------------------------------------------------------------

    def fly(self, points, speed_mps=1.5):
        """Follow a polyline of (x, y) points at constant speed, heading along it."""
        self.path = [tuple(map(float, p)) for p in points]; self.speed_mps = float(speed_mps)

    def teleport(self, x, y, heading_deg=None):
        self.state.x, self.state.y = float(x), float(y)
        if heading_deg is not None: self.state.yaw_deg = 270. - float(heading_deg)

    def _move(self, dt):
        if not self.path: return
        tx, ty = self.path[0]
        dx, dy = tx-self.state.x, ty-self.state.y
        distance = math.hypot(dx, dy)
        step = self.speed_mps*dt
        if distance <= step:
            self.state.x, self.state.y = tx, ty; self.path.pop(0)
        else:
            self.state.x += dx/distance*step; self.state.y += dy/distance*step
        if distance > 1e-6:
            self.state.yaw_deg = 270. - math.degrees(math.atan2(dx, -dy)) % 360.

    # -- publication --------------------------------------------------------------

    def body(self):
        st = self.state
        return dict(x_m=st.x, y_m=st.y, z_m=st.z, heading_deg=st.heading_deg,
                    roll_deg=st.roll_deg, pitch_deg=st.pitch_deg)

    def _publish_pose(self):
        if not self.publish_poses: return
        self.context.publish_pose(self.body(), self.time_s, localization_epoch=self.localization_epoch,
                                  clock_epoch=self.clock_epoch, valid=self.pose_valid)

    def step(self, dt=1./PHYSICS_HZ):
        self.time_s = round(self.time_s + dt, 9)
        self.tick_index += 1
        self._move(dt)
        if self.publisher is not None:
            self.publisher.localization_epoch = self.localization_epoch
            self.publisher.pose_valid = self.pose_valid
            sim_us = round(self.time_s*1e6)
            for sensor in self.profile.data['sensors']:
                divisor = round(PHYSICS_HZ/sensor['rate_hz'])
                if self.tick_index % divisor or sensor['id'] in self.muted: continue
                if sensor['type'] == 'lidar.scan2d': self._scan(sensor, sim_us)
                elif sensor['type'] == 'camera.rgb' and self.camera_frames: self._frame(sensor, sim_us)
        self._publish_pose()
        for event in self.bus.receive():
            if requests_shutdown(event): return False
        return True

    def run(self, seconds, dt=1./PHYSICS_HZ):
        for _ in range(max(1, round(seconds/dt))): self.step(dt)

    def _next(self, sid):
        return self.publisher.next_sequence(sid)

    def _pose_payload(self, sid):
        from dsim.transforms import pose_angles, resolve
        world = resolve(self.profile.data, sid, self.state)
        return world, dict(pose_world=world.tolist(), pose=pose_angles(world),
                           body=dict(self.body(), vx_mps=0., vy_mps=0., vz_mps=0.))

    def _scan(self, sensor, sim_us):
        from dsim import sensor_models
        from dsim.transforms import spherical_rays
        model = sensor['model']
        world, common = self._pose_payload(sensor['id'])
        calibration = sensor_models.calibration(sensor['type'], model)
        azimuths = calibration['angle_min_deg'] + np.arange(model['samples'])*calibration['angle_increment_deg']
        directions = spherical_rays(azimuths, model['elevation_deg']) @ world[:3, :3].T
        truth, _, _ = cast(world[:3, 3], directions, self.boxes)
        ranges, confidence = sensor_models.measure(truth, model, np.random.default_rng(self.tick_index))
        returns = int(np.isfinite(ranges).sum())
        sequence = self._next(sensor['id'])
        status = STATUS_VALID if returns else STATUS_INVALID
        self.publisher.write_array(sensor['id'], sequence, self.tick_index, sim_us,
                                   sensor_models.pack_array(ranges, confidence), status=status)
        self.publisher.write_compact(sensor['id'], sequence, self.tick_index, sim_us, LIDAR_FRAME,
            dict(schema='lidar.frame.v1', array_sequence=sequence, type=sensor['type'], returns=returns,
                 samples=int(ranges.size), calibration=calibration, min_range_m=model['min_range_m'],
                 max_range_m=model['max_range_m'], **common), status=status)

    def render(self, sensor):
        """A deterministic raycast image: textured walls, checkered floor, plain sky."""
        from dsim.transforms import pinhole_rays
        model = sensor['model']
        world, _ = self._pose_payload(sensor['id'])
        directions = pinhole_rays(model) @ world[:3, :3].T
        t, face, which = cast(world[:3, 3], directions, self.boxes, floor=True)
        hit = world[:3, 3] + directions*np.where(np.isfinite(t), t, 0.)[..., None]
        image = np.empty((*t.shape, 3), np.uint8); image[:] = (70, 90, 130)
        floor = which == -2
        checker = ((np.floor(hit[..., 0]) + np.floor(hit[..., 1])) % 2).astype(bool)
        image[floor & checker] = (150, 140, 120); image[floor & ~checker] = (95, 90, 80)
        wall = which >= 0
        along = np.where(face == 0, hit[..., 1], hit[..., 0])
        stripe = (np.floor(along*2) % 2).astype(bool)
        shade = np.where(face == 0, 1.0, .8)
        base = 170 + 20*(which % 3)
        for channel, scale in enumerate((1., .95, .9)):
            value = base*shade*scale*np.where(stripe, 1., .75)
            image[..., channel] = np.where(wall, np.clip(value, 0, 255), image[..., channel])
        return image

    def _frame(self, sensor, sim_us):
        sid = sensor['id']
        image = self.render(sensor)
        _, common = self._pose_payload(sid)
        slot = self.publisher.camera_slot(sid)
        slot[...] = image.reshape(slot.shape)
        del slot
        video_sequence = self.publisher.commit_camera(sid, sim_us)
        self.publisher.write_compact(sid, self._next(sid), self.tick_index, sim_us, CAMERA_FRAME,
            dict(schema='camera.frame.v1', video_sequence=video_sequence,
                 calibration_revision=self.profile.digest, sync_group=None, **common))

    # -- provider-side events -----------------------------------------------------------

    def announce_localization_reset(self):
        """A provider-announced localization discontinuity (DV-MAPPING Q5)."""
        self.localization_epoch += 1

    def announce_clock_reset(self):
        """A new clock epoch; data time restarts at zero."""
        self.clock_epoch = uuid.uuid4().int & ((1 << 63) - 1)
        self.time_s = 0.
        if self.publisher is not None:
            self.publisher.clock_epoch = self.clock_epoch
            self.publisher.apply(self.profile)

    def sensor_reset(self):
        if self.publisher is not None: self.publisher.reset()

    def set_sensors(self, sensors):
        """Commit a new sensor manifest generation with a different sensor set."""
        self.sensor_ids = tuple(sensors)
        self.profile = fixture_profile(sensors)
        self.camera_frames = 'front' in sensors
        self.publisher.apply(self.profile)

    def rollover(self, report_root):
        return self.context.rollover(report_root)

    # -- optional reference imagery -------------------------------------------

    def plan_png(self, *, metres_per_px=0.25):
        """A deterministic plan rendering of this provider's own scene.

        The image's placement is returned with its pixels -- the affine that
        maps pixel centres onto the scene's bounding box -- because a plan
        that could not say where it sits is a picture, not a reference
        image. Walls are the scene's boxes; nothing here consults a world
        file.
        """
        from PIL import Image, ImageDraw
        from dcmn import theme

        metres_per_px = float(metres_per_px)
        x0 = min(box[0] for box in self.boxes); y0 = min(box[1] for box in self.boxes)
        x1 = max(box[2] for box in self.boxes); y1 = max(box[3] for box in self.boxes)
        width = max(1, math.ceil((x1 - x0) / metres_per_px))
        height = max(1, math.ceil((y1 - y0) / metres_per_px))
        image = Image.new('RGB', (width, height), theme.CELL)
        draw = ImageDraw.Draw(image)
        for bx0, by0, bx1, by1 in self.boxes:
            draw.rectangle(((bx0 - x0) / metres_per_px, (by0 - y0) / metres_per_px,
                            (bx1 - x0) / metres_per_px - 1, (by1 - y0) / metres_per_px - 1),
                            fill=theme.WALL_FILL, outline=theme.WALL_EDGE)
        # Pixel (col, row) centres on (x0 + (col+.5)*mpp, y0 + (row+.5)*mpp).
        affine = [metres_per_px, 0., 0., metres_per_px,
                  x0 + metres_per_px / 2., y0 + metres_per_px / 2.]
        buffer = io.BytesIO(); image.save(buffer, 'PNG')
        return buffer.getvalue(), affine

    def publish_reference_image(self, image_id='world', *, png=None, affine=None,
                                source_category='surveyed_plan', metres_per_px=0.25,
                                localization_epoch=None, clock_epoch=None,
                                capture_s=None, registration=None,
                                altitude_m=None, floor=None):
        """Publish one reference image on the optional imagery plane.

        With no ``png`` the provider renders its scene as a plan and places
        it itself; with one, the caller owns the pixels *and* the affine,
        which is how tests publish rotated, reflected, deliberately
        misleading and plain malformed backgrounds. The frame epochs default
        to this session's current ones -- an image a test publishes is
        displayable immediately -- and can be overridden to publish an image
        that is already stale.
        """
        from dcmn.imagery import ImagePublisher
        if self.imagery is None:
            self.imagery = ImagePublisher(self.instance, producer='fixture-provider')
        if png is None:
            png, affine = self.plan_png(metres_per_px=metres_per_px)
        elif affine is None:
            raise ValueError('an explicit image needs an explicit affine')
        return self.imagery.publish(
            image_id, png, affine, source_category=source_category,
            frame_id='local',
            localization_epoch=self.localization_epoch if localization_epoch is None
            else int(localization_epoch),
            clock_domain_id=self.instance,
            clock_epoch=self.clock_epoch if clock_epoch is None else int(clock_epoch),
            capture_s=capture_s, registration=registration,
            altitude_m=altitude_m, floor=floor)

    def withdraw_reference_image(self, image_id='world'):
        """Take an image back off the plane: the explicit opposite of ambient."""
        return self.imagery.withdraw(image_id) if self.imagery is not None else False

    def close(self):
        self.context.close()
        if self.publisher is not None: self.publisher.close()
        if self.imagery is not None: self.imagery.close()
        self.bus.remove()


def parse_path(text):
    return [tuple(float(v) for v in point.split(',')) for point in text.split(';') if point.strip()]


def main(argv=None):
    import argparse
    import sys
    parser = argparse.ArgumentParser(description='deterministic sensor/pose provider (no world file)')
    parser.add_argument('--id', required=True)
    parser.add_argument('--report-dir', required=True)
    parser.add_argument('--sensors', default='scan', help='comma-separated: scan, front ("" for none)')
    parser.add_argument('--start', default='0,0,1.5,90', help='x,y,z,heading_deg')
    parser.add_argument('--path', default='', help='x,y;x,y;... to fly, at --speed')
    parser.add_argument('--speed', type=float, default=1.5)
    parser.add_argument('--rate', type=float, default=1.0, help='data seconds per wall second')
    parser.add_argument('--timeout', type=float, default=60.)
    parser.add_argument('--imagery', action='store_true',
                        help='publish the scene plan as optional reference imagery')
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    sensors = tuple(s for s in args.sensors.split(',') if s)
    provider = FixtureProvider(args.id, Path(args.report_dir), sensors=sensors,
                               start=tuple(float(v) for v in args.start.split(',')))
    if args.path: provider.fly(parse_path(args.path), args.speed)
    if args.imagery: provider.publish_reference_image()
    print(f'provider: {args.id} sensors={",".join(sensors) or "none"} report={args.report_dir}', flush=True)
    deadline = time.monotonic() + args.timeout
    dt = 1./PHYSICS_HZ
    try:
        while time.monotonic() < deadline:
            if not provider.step(dt): break
            time.sleep(dt/args.rate)
    except KeyboardInterrupt:
        pass
    finally:
        provider.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
