"""Public forward/right/up column-vector transforms (no Panda coordinates)."""
import math
import numpy as np


def rotation(roll_deg=0., pitch_deg=0., yaw_deg=0.):
    r, p, y = map(math.radians, (roll_deg, pitch_deg, yaw_deg))
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return (np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]) @
            np.array([[cp, 0, -sp], [0, 1, 0], [sp, 0, cp]]) @
            np.array([[1, 0, 0], [0, cr, sr], [0, -sr, cr]]))


def transform(pose):
    t = np.eye(4)
    t[:3, :3] = rotation(*(pose.get(k, 0.) for k in ('roll_deg', 'pitch_deg', 'yaw_deg')))
    t[:3, 3] = [pose.get(k, 0.) for k in ('x_m', 'y_m', 'z_m')]
    return t


def body_world(state):
    # Match dsim.sim_yaw_to_compass_heading; internal yaw increases left.
    h = math.radians((270. - state.yaw_deg) % 360.)
    basis = np.array([[math.sin(h), math.cos(h), 0],
                      [-math.cos(h), math.sin(h), 0], [0, 0, 1]])
    t = np.eye(4)
    t[:3, :3] = basis @ rotation(state.roll_deg, state.pitch_deg, 0.)
    t[:3, 3] = [state.x, state.y, state.z]
    return t


def mount_chain(profile, sensor_id):
    """The transform from the vehicle body to a component, and nothing else.

    This is the airframe geometry on its own: where a sensor sits and which
    way it looks relative to the vehicle, with no dependence on where the
    vehicle happens to be. A PTZ contributes its fixed placement and then its
    live state, in that order, so panning the head does not move its bracket.
    """
    nodes = {n['id']: n for n in (*profile['mounts'], *profile['sensors'])}
    chain = []
    current = sensor_id
    while current != 'body':
        node = nodes[current]
        local = transform(node['pose_parent'])
        if node['type'] == 'mount.ptz':
            st = node['state']
            local = local @ transform(dict(yaw_deg=st['pan_deg'], pitch_deg=st['tilt_deg'], roll_deg=st['roll_deg']))
        chain.append(local)
        current = node['parent']
    t = np.eye(4)
    for local in reversed(chain):
        t = t @ local
    return t


def resolve(profile, sensor_id, state):
    """The map-frame transform of a component at one vehicle state."""
    return body_world(state) @ mount_chain(profile, sensor_id)


def euler_angles(rotation_matrix):
    """Invert ``R = Rz(yaw) * Ry(pitch) * Rx(roll)`` back to degrees.

    Only the composition this module defines is inverted here; an ordinary
    right-handed Euler helper does not apply, because ``DV-SENSORS.md`` picks
    the roll and pitch matrices for their sign conventions rather than for a
    uniform handedness.
    """
    r = np.asarray(rotation_matrix)
    pitch = math.asin(float(np.clip(r[2, 0], -1.0, 1.0)))
    return dict(roll_deg=math.degrees(math.atan2(-r[2, 1], r[2, 2])),
                pitch_deg=math.degrees(pitch),
                yaw_deg=math.degrees(math.atan2(r[1, 0], r[0, 0])))


def pose_angles(t):
    """Map position and heading/pitch/roll of a sensor's public basis."""
    forward, right = t[:3, 0], t[:3, 1]
    h = math.atan2(forward[0], -forward[1])
    pitch = math.asin(float(np.clip(forward[2], -1, 1)))
    # Relative to the level basis at the extracted heading.
    level_right = np.array([math.cos(h), math.sin(h), 0])
    level_up = np.cross(forward, level_right)
    roll = math.atan2(-float(right @ level_up), float(right @ level_right))
    return dict(x_m=float(t[0, 3]), y_m=float(t[1, 3]), z_m=float(t[2, 3]),
                heading_deg=math.degrees(h) % 360, pitch_deg=math.degrees(pitch),
                roll_deg=math.degrees(roll))


def pinhole_rays(model, rows=None, columns=None):
    """Unit rays for a pinhole sensor, in its own forward/right/up axes.

    Optical coordinates are +X right, +Y down, +Z forward, and an optical
    vector ``(x, y, z)`` is the sensor-body vector ``(z, x, -y)``; a pixel at
    the principal point therefore looks straight along the sensor's +X. The
    result is ``(len(rows), len(columns), 3)``.
    """
    rows = np.arange(model['height_px']) if rows is None else np.asarray(rows)
    columns = np.arange(model['width_px']) if columns is None else np.asarray(columns)
    x = (columns - model['cx_px']) / model['fx_px']
    y = (rows - model['cy_px']) / model['fy_px']
    xx, yy = np.meshgrid(x, y)
    rays = np.stack((np.ones_like(xx), xx, -yy), axis=-1)
    return rays / np.linalg.norm(rays, axis=-1, keepdims=True)


def spherical_rays(azimuth_deg, elevation_deg=0.0):
    """Unit rays from sensor-frame azimuth/elevation, in forward/right/up axes.

    Azimuth follows the body yaw convention -- positive turns from forward
    (+X) toward right (+Y) -- and positive elevation lifts toward up (+Z).
    """
    az = np.radians(np.asarray(azimuth_deg, np.float64))
    el = np.radians(np.asarray(elevation_deg, np.float64))
    return np.stack((np.cos(el) * np.cos(az), np.cos(el) * np.sin(az),
                     np.broadcast_to(np.sin(el), np.shape(az * 1.0))), axis=-1)
