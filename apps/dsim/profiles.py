"""Versioned immutable hardware profiles and pre-allocation validation.

A drone profile is the hardware half of a run: what sensors exist, where they
are mounted, how fast they sample and what their models are. Realism settings
are the conditions half and live elsewhere. Parsing resolves every default and
derived value, so the object a caller holds is exactly what the manifest
publishes and the digest covers.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import math
from dvision2_common import validate_id

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = 'dvision2.drone-profile.v1'
MAX_MEMORY = 256 * 1024 * 1024
MAX_COMPONENTS = 64
MAX_ID_BYTES = 48
MAX_PROFILE_BYTES = 65536
RETENTION_S = 1.
RECORD_SIZE = 8192
MIN_RING_BYTES = 65536
RING_SLOT_OVERHEAD = 64
#: Registry allocation: nine keys of up to 64 KiB, plus create-time overhead.
REGISTRY_BYTES = 10 * (65536 + 64) + 4096
POSE_KEYS = ('x_m', 'y_m', 'z_m', 'roll_deg', 'pitch_deg', 'yaw_deg')
MOUNT_TYPES = ('mount.fixed', 'mount.ptz', 'rig.fixed')
PTZ_AXES = ('pan_deg', 'tilt_deg', 'roll_deg')
REDUCERS = ('nearest', 'farthest', 'median')
CONFIDENCE_MODELS = ('exact', 'range_linear')

#: Bytes per sample in a published range array: float32 metres plus a uint8
#: confidence, invalid samples being NaN with zero confidence.
ARRAY_BYTES_PER_SAMPLE = 5

#: Type-specific model defaults. Every field a type accepts appears here, so
#: an unknown field is a validation error rather than a silently ignored one.
#: The three scalar range types differ only through these presets: a narrow
#: short-range IR cone, a wide sonar cone with a blind zone, and a narrow
#: long-range laser with almost no angular spread.
RANGE_DEFAULTS = {
    'range.infrared': dict(beam_fov_deg=8., beam_samples=5, min_range_m=.05,
                           max_range_m=1.5, noise_std_m=.005, quantization_m=.001,
                           dropout_probability=.01, limit_degradation=1.,
                           reducer='nearest', confidence_model='range_linear'),
    'range.ultrasonic': dict(beam_fov_deg=25., beam_samples=9, min_range_m=.2,
                             max_range_m=7., noise_std_m=.01, quantization_m=.005,
                             dropout_probability=.02, limit_degradation=.5,
                             reducer='nearest', confidence_model='range_linear'),
    'range.laser': dict(beam_fov_deg=1., beam_samples=3, min_range_m=.03,
                        max_range_m=40., noise_std_m=.01, quantization_m=.001,
                        dropout_probability=.005, limit_degradation=.2,
                        reducer='nearest', confidence_model='range_linear'),
}
SCAN_DEFAULTS = dict(fov_deg=360., samples=720, elevation_deg=0., min_range_m=.15,
                     max_range_m=25., noise_std_m=.02, quantization_m=.005,
                     dropout_probability=.01, limit_degradation=0.,
                     confidence_model='range_linear')
RANGE_IMAGE_DEFAULTS = dict(width_px=64, height_px=48, min_range_m=.15,
                            max_range_m=20., noise_std_m=.03, quantization_m=.01,
                            dropout_probability=.02, limit_degradation=0.,
                            confidence_model='range_linear')
CAMERA_DEFAULTS = dict(width_px=640, height_px=480, near_m=.15, far_m=150.)

#: The inexpensive state sensors: they read vehicle state and the environment
#: rather than the scene, so their models are noise and quantization only.
#: Every field a type accepts appears here and anything else is rejected.
STATE_DEFAULTS = {
    'position.gnss': dict(),
    'motion.imu': dict(gyro_noise_std_dps=.05, accel_noise_std_mps2=.05,
                       gyro_bias_dps=0., accel_bias_mps2=0.),
    'altimeter.barometric': dict(noise_std_m=0., quantization_m=0.,
                                 sea_level_pressure_pa=101325.),
    'heading.magnetometer': dict(noise_std_deg=0.),
    'environment.temperature': dict(noise_std_c=.1, quantization_c=.01,
                                    lapse_rate_c_per_m=.0065),
}
SENSOR_TYPES = ('camera.rgb', 'lidar.scan2d', 'lidar.range_image',
                *RANGE_DEFAULTS, *STATE_DEFAULTS)
#: Sensors whose samples are large packed arrays on a dedicated ring.
ARRAY_TYPES = ('lidar.scan2d', 'lidar.range_image')
#: Sensors that read vehicle state rather than cast rays into the scene.
STATE_TYPES = tuple(STATE_DEFAULTS)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False, ensure_ascii=True)


def number(value, field, low=None, high=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{field}: finite number required')
    if low is not None and value < low or high is not None and value > high:
        raise ValueError(f'{field}: outside supported bounds')
    return float(value)


def positive(value, field):
    result = number(value, field)
    if result <= 0: raise ValueError(f'{field}: must be positive')
    return result


def counted(value, field, low=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < low:
        raise ValueError(f'{field}: integer of at least {low} required')
    return value


def choice(value, field, allowed):
    if value not in allowed: raise ValueError(f'{field}: one of {", ".join(allowed)} required')
    return value


def default_profile(width=640, height=480, rate_hz=30.):
    """The vehicle a run gets when nobody named a profile.

    The camera is today's historical one, unchanged, so default runs look the
    same. The state sensors beside it are the ones a real airframe of this
    class always carries: leaving them out would make "no GNSS fitted" the
    default, and every client that reads a position would be flying a vehicle
    with no receiver in it.
    """
    return dict(schema=SCHEMA, name='default', physics_hz=300., primary_camera='front', mounts=[],
                sensors=[dict(id='front', type='camera.rgb', enabled=True, rate_hz=rate_hz,
                              parent='body', pose_parent=dict(z_m=.1, pitch_deg=-5.),
                              model=dict(width_px=width, height_px=height, fov_h_deg=70., near_m=.15, far_m=150.)),
                         dict(id='gnss', type='position.gnss', enabled=True, rate_hz=10., parent='body'),
                         dict(id='imu', type='motion.imu', enabled=True, rate_hz=100., parent='body'),
                         dict(id='baro', type='altimeter.barometric', enabled=True, rate_hz=25., parent='body'),
                         dict(id='compass', type='heading.magnetometer', enabled=True, rate_hz=25., parent='body'),
                         dict(id='ambient', type='environment.temperature', enabled=True, rate_hz=1., parent='body')])


#: Model fields an editor must present as whole numbers rather than metres.
INTEGER_FIELDS = frozenset({'width_px', 'height_px', 'samples', 'beam_samples'})
#: Model fields with a fixed set of values rather than a number.
CHOICE_FIELDS = {'confidence_model': CONFIDENCE_MODELS, 'reducer': REDUCERS}


def model_fields(kind, model=None):
    """The ``model`` keys an editor should offer for one sensor type, in order.

    The lens pair is exactly-one-of, so the editor is shown whichever of
    ``fov_h_deg`` and ``fx_px`` the draft actually carries: offering both
    would present a form that cannot be valid.
    """
    lens = ('fx_px',) if model and 'fx_px' in model else ('fov_h_deg',)
    if kind == 'camera.rgb':
        return ('width_px', 'height_px', *lens, 'fy_px', 'cx_px', 'cy_px',
                'near_m', 'far_m')
    if kind == 'lidar.range_image':
        return ('width_px', 'height_px', *lens, 'fy_px', 'cx_px', 'cy_px',
                *(k for k in RANGE_IMAGE_DEFAULTS if not k.endswith('_px')))
    if kind == 'lidar.scan2d':
        return tuple(SCAN_DEFAULTS)
    if kind in RANGE_DEFAULTS:
        return tuple(RANGE_DEFAULTS[kind])
    return tuple(STATE_DEFAULTS.get(kind, ()))


def unique_id(stem, taken):
    """The first ``stem``, ``stem2``, ``stem3``... that nothing else is using."""
    if stem not in taken:
        return stem
    return next(f'{stem}{n}' for n in range(2, 1000) if f'{stem}{n}' not in taken)


def new_component(kind, taken=()):
    """A minimal valid component of one type, with an unused id.

    Everything else is left out so the loader fills it: a component created
    here and one loaded from a file resolve to the same thing.
    """
    stem = unique_id(kind.split('.')[-1].replace('_', ''), set(taken))
    if kind in MOUNT_TYPES:
        node = dict(id=stem, type=kind, parent='body', pose_parent={})
        if kind == 'mount.ptz':
            node['state'] = dict(pan_deg=0., tilt_deg=0., roll_deg=0.)
        return node
    node = dict(id=stem, type=kind, enabled=True, rate_hz=30., parent='body',
                pose_parent={})
    if kind == 'camera.rgb':
        node['model'] = dict(width_px=640, height_px=480, fov_h_deg=70.)
    elif kind == 'lidar.range_image':
        node['model'] = dict(fov_h_deg=70.)
    elif kind in STATE_DEFAULTS:
        node['rate_hz'] = 10.
    return node


def camera_profile(width=640, height=480, rate_hz=30., physics_hz=300.):
    """The default camera on its own, with none of the state sensors.

    A vehicle would not fly like this. It exists for callers that are testing
    one thing about a camera and do not want a hundred inertial samples a
    second arriving beside it.
    """
    profile = default_profile(width, height, rate_hz)
    profile['name'] = 'camera-only'
    profile['physics_hz'] = physics_hz
    profile['sensors'] = [s for s in profile['sensors'] if s['type'] == 'camera.rgb']
    return profile


def _pinhole(sid, m, defaults):
    """Resolve one pinhole model: exactly one of fov_h_deg or fx_px, then derive."""
    for key, default in defaults.items():
        m[key] = counted(m.get(key, default), f'{sid}.model.{key}') if key.endswith('_px') \
            else number(m.get(key, default), f'{sid}.model.{key}')
    # A resolved profile keeps fx and drops the redundant FOV, so saving and
    # reloading a resolved profile still satisfies exactly-one-of.
    if ('fx_px' in m) == ('fov_h_deg' in m):
        raise ValueError(f'{sid}.model: exactly one of fx_px or fov_h_deg required')
    if 'fov_h_deg' in m:
        fov = number(m.pop('fov_h_deg'), f'{sid}.model.fov_h_deg', .01, 179.)
        m['fx_px'] = m['width_px'] / (2 * math.tan(math.radians(fov) / 2))
    m['fx_px'] = positive(m['fx_px'], f'{sid}.model.fx_px')
    m['fy_px'] = positive(m.get('fy_px', m['fx_px']), f'{sid}.model.fy_px')
    m['cx_px'] = number(m.get('cx_px', m['width_px'] / 2), f'{sid}.model.cx_px')
    m['cy_px'] = number(m.get('cy_px', m['height_px'] / 2), f'{sid}.model.cy_px')


def _ray_model(sid, m, defaults):
    """Resolve the noise/gate fields every ray-based sensor shares."""
    m['min_range_m'] = number(m.get('min_range_m', defaults['min_range_m']), f'{sid}.model.min_range_m', 0.)
    m['max_range_m'] = positive(m.get('max_range_m', defaults['max_range_m']), f'{sid}.model.max_range_m')
    if m['min_range_m'] >= m['max_range_m']:
        raise ValueError(f'{sid}.model.max_range_m: must exceed min_range_m')
    m['noise_std_m'] = number(m.get('noise_std_m', defaults['noise_std_m']), f'{sid}.model.noise_std_m', 0.)
    m['quantization_m'] = number(m.get('quantization_m', defaults['quantization_m']), f'{sid}.model.quantization_m', 0.)
    m['dropout_probability'] = number(m.get('dropout_probability', defaults['dropout_probability']),
                                      f'{sid}.model.dropout_probability', 0., 1.)
    m['limit_degradation'] = number(m.get('limit_degradation', defaults['limit_degradation']),
                                    f'{sid}.model.limit_degradation', 0., 10.)
    m['confidence_model'] = choice(m.get('confidence_model', defaults['confidence_model']),
                                   f'{sid}.model.confidence_model', CONFIDENCE_MODELS)


def _resolve_model(sid, kind, m):
    if not isinstance(m, dict): raise ValueError(f'{sid}.model: object required')
    if kind == 'camera.rgb':
        allowed = {*CAMERA_DEFAULTS, 'fov_h_deg', 'fx_px', 'fy_px', 'cx_px', 'cy_px'}
        _reject_unknown(sid, m, allowed)
        _pinhole(sid, m, dict(width_px=CAMERA_DEFAULTS['width_px'], height_px=CAMERA_DEFAULTS['height_px']))
        m['near_m'] = positive(m.get('near_m', CAMERA_DEFAULTS['near_m']), f'{sid}.model.near_m')
        m['far_m'] = positive(m.get('far_m', CAMERA_DEFAULTS['far_m']), f'{sid}.model.far_m')
        if m['near_m'] >= m['far_m']: raise ValueError(f'{sid}.model.far_m: must exceed near_m')
        return
    if kind == 'lidar.scan2d':
        _reject_unknown(sid, m, set(SCAN_DEFAULTS))
        m['fov_deg'] = number(m.get('fov_deg', SCAN_DEFAULTS['fov_deg']), f'{sid}.model.fov_deg', .01, 360.)
        m['samples'] = counted(m.get('samples', SCAN_DEFAULTS['samples']), f'{sid}.model.samples', 2)
        m['elevation_deg'] = number(m.get('elevation_deg', SCAN_DEFAULTS['elevation_deg']),
                                    f'{sid}.model.elevation_deg', -89., 89.)
        _ray_model(sid, m, SCAN_DEFAULTS)
        return
    if kind == 'lidar.range_image':
        _reject_unknown(sid, m, {*RANGE_IMAGE_DEFAULTS, 'fov_h_deg', 'fx_px', 'fy_px', 'cx_px', 'cy_px'})
        _pinhole(sid, m, dict(width_px=RANGE_IMAGE_DEFAULTS['width_px'],
                              height_px=RANGE_IMAGE_DEFAULTS['height_px']))
        _ray_model(sid, m, RANGE_IMAGE_DEFAULTS)
        return
    if kind in STATE_DEFAULTS:
        defaults = STATE_DEFAULTS[kind]
        _reject_unknown(sid, m, set(defaults))
        for key, default in defaults.items():
            low = 0. if key.startswith(('noise', 'quantization', 'lapse', 'gyro_noise',
                                        'accel_noise', 'sea_level')) else None
            m[key] = number(m.get(key, default), f'{sid}.model.{key}', low)
        return
    defaults = RANGE_DEFAULTS[kind]
    _reject_unknown(sid, m, set(defaults))
    m['beam_fov_deg'] = number(m.get('beam_fov_deg', defaults['beam_fov_deg']), f'{sid}.model.beam_fov_deg', 0., 179.)
    m['beam_samples'] = counted(m.get('beam_samples', defaults['beam_samples']), f'{sid}.model.beam_samples')
    m['reducer'] = choice(m.get('reducer', defaults['reducer']), f'{sid}.model.reducer', REDUCERS)
    _ray_model(sid, m, defaults)


def _reject_unknown(sid, m, allowed):
    unknown = sorted(set(m) - set(allowed))
    if unknown: raise ValueError(f'{sid}.model.{unknown[0]}: unknown field for this sensor type')


def sample_count(sensor):
    """How many rays one capture of an array sensor produces."""
    m = sensor['model']
    return m['samples'] if sensor['type'] == 'lidar.scan2d' else m['width_px'] * m['height_px']


def scan_angles_deg(model):
    """``(angle_min_deg, angle_increment_deg)`` of a 2D scan.

    A full circle repeats its first ray, so it steps ``fov / samples``; a
    partial sector includes both endpoints and steps ``fov / (samples - 1)``.
    """
    if model['fov_deg'] >= 360.: return -180., 360. / model['samples']
    return -model['fov_deg'] / 2., model['fov_deg'] / (model['samples'] - 1)


def transport_plan(data):
    """Shared-memory each enabled sensor needs, in the units it is created with.

    Validation and channel creation both read this, so a profile can never pass
    a budget check under one arithmetic and be allocated under another.
    """
    cameras, arrays, compact_hz, render_hz = {}, {}, 0., 0.
    retention = data.get('transport', {}).get('retention_s', RETENTION_S)
    for sensor in data['sensors']:
        if not sensor['enabled']: continue
        rate, kind, m = sensor['rate_hz'], sensor['type'], sensor['model']
        slots = math.ceil(rate * retention) + 2
        # Every camera and array sensor also publishes one compact metadata
        # record per capture, so it costs the shared ring a slot as well.
        compact_hz += rate
        if kind == 'camera.rgb':
            render_hz += rate
            cameras[sensor['id']] = dict(
                slots=slots, bytes=slots * (m['width_px'] * m['height_px'] * 3 + 128) + 4096)
        elif kind in ARRAY_TYPES:
            record = 128 + sample_count(sensor) * ARRAY_BYTES_PER_SAMPLE
            arrays[sensor['id']] = dict(
                slots=slots, record_bytes=record,
                capacity=max(MIN_RING_BYTES, slots * (record + RING_SLOT_OVERHEAD)))
    sample_capacity = max(MIN_RING_BYTES,
                          (math.ceil(compact_hz * retention) + 2) * (RECORD_SIZE + RING_SLOT_OVERHEAD))
    total = (REGISTRY_BYTES + sample_capacity
             + sum(c['bytes'] for c in cameras.values())
             + sum(a['capacity'] for a in arrays.values()))
    return dict(cameras=cameras, arrays=arrays, sample_capacity=sample_capacity,
                compact_hz=compact_hz, render_hz=render_hz, total_bytes=total)


@dataclass(frozen=True)
class DroneProfile:
    """Canonical JSON is the immutable backing; callers receive isolated values."""
    encoded: str

    @property
    def data(self): return json.loads(self.encoded)
    @property
    def digest(self): return hashlib.sha256(self.encoded.encode()).hexdigest()
    @property
    def plan(self): return transport_plan(self.data)
    @property
    def memory_bytes(self): return self.plan['total_bytes']
    @property
    def memory_limit_bytes(self):
        return self.data.get('transport', {}).get('memory_limit_mib', 256.) * 1048576
    @property
    def primary(self):
        d = self.data
        return next(s for s in d['sensors'] if s['id'] == d['primary_camera'])
    def sensor(self, sensor_id):
        return next(s for s in self.data['sensors'] if s['id'] == sensor_id)
    def sync_group(self, name):
        return [s for s in self.data['sensors'] if s.get('sync_group') == name and s['enabled']]
    def save(self, path): Path(path).write_text(json.dumps(self.data, indent=2) + '\n')

    @classmethod
    def load(cls, name=None):
        if name is None: return cls.parse(default_profile())
        path = Path(name)
        if not path.is_file(): path = ROOT / 'assets/drone_profiles' / (str(name) + '.json')
        if path.stat().st_size > MAX_PROFILE_BYTES: raise ValueError('profile: exceeds 64 KiB')
        return cls.parse(json.loads(path.read_text()))

    @classmethod
    def parse(cls, value):
        d = json.loads(canonical(value))
        if not isinstance(d, dict) or d.get('schema') != SCHEMA: raise ValueError('schema: unsupported profile version')
        if not isinstance(d.get('name'), str) or not d['name']: raise ValueError('name: nonempty string required')
        transport = d.setdefault('transport', {})
        if not isinstance(transport, dict) or set(transport) - {'retention_s', 'memory_limit_mib'}:
            raise ValueError('transport: only retention_s and memory_limit_mib are supported')
        transport['retention_s'] = positive(transport.get('retention_s', RETENTION_S), 'transport.retention_s')
        transport['memory_limit_mib'] = positive(transport.get('memory_limit_mib', 256.), 'transport.memory_limit_mib')
        hz = positive(d.setdefault('physics_hz', 300.), 'physics_hz')
        mounts, sensors = d.setdefault('mounts', []), d.get('sensors')
        if not isinstance(mounts, list) or not isinstance(sensors, list): raise ValueError('mounts/sensors: arrays required')
        if len(mounts) + len(sensors) > MAX_COMPONENTS: raise ValueError(f'components: maximum {MAX_COMPONENTS}')
        nodes, groups = {}, {}
        for i, n in enumerate(mounts + sensors):
            field = f'components[{i}]'
            if not isinstance(n, dict): raise ValueError(f'{field}: object required')
            sid = n.get('id')
            if not isinstance(sid, str): raise ValueError(f'{field}.id: string required')
            validate_id(sid)
            if len(sid.encode()) > MAX_ID_BYTES or sid == 'body' or sid in nodes:
                raise ValueError(f'{field}.id: duplicate, reserved, or over {MAX_ID_BYTES} bytes')
            nodes[sid] = n
            if not isinstance(n.get('parent'), str): raise ValueError(f'{sid}.parent: required')
            pose = n.setdefault('pose_parent', {})
            if not isinstance(pose, dict) or set(pose) - set(POSE_KEYS): raise ValueError(f'{sid}.pose_parent: invalid fields')
            for k in POSE_KEYS: pose[k] = number(pose.get(k, 0.), f'{sid}.pose_parent.{k}')
            if i < len(mounts):
                choice(n.get('type'), f'{sid}.type', MOUNT_TYPES)
                if n['type'] == 'mount.ptz': _resolve_ptz(sid, n)
                elif 'state' in n or 'limits' in n:
                    raise ValueError(f'{sid}.state/limits: only mount.ptz supports these fields')
                continue
            choice(n.get('type'), f'{sid}.type', SENSOR_TYPES)
            if not isinstance(n.setdefault('enabled', True), bool): raise ValueError(f'{sid}.enabled: boolean required')
            rate = positive(n.setdefault('rate_hz', 30.), f'{sid}.rate_hz')
            if rate > hz or not math.isclose(hz / rate, round(hz / rate), abs_tol=1e-9):
                raise ValueError(f'{sid}.rate_hz: must be an integer divisor of physics_hz')
            _resolve_model(sid, n['type'], n.setdefault('model', {}))
            group = n.get('sync_group')
            if group is not None:
                if not isinstance(group, str) or not group: raise ValueError(f'{sid}.sync_group: nonempty string required')
                groups.setdefault(group, []).append(n)
        _check_parents(nodes, sensors)
        _check_groups(groups)
        if not isinstance(d.get('primary_camera'), str):
            raise ValueError('primary_camera: enabled RGB sensor required')
        primary = nodes.get(d.get('primary_camera'))
        if primary not in sensors or primary.get('type') != 'camera.rgb' or not primary['enabled']:
            raise ValueError('primary_camera: enabled RGB sensor required')
        plan = transport_plan(d)
        if plan['total_bytes'] > transport['memory_limit_mib'] * 1048576:
            raise ValueError(f"profile: {plan['total_bytes'] / 1048576:.1f} MiB exceeds the {transport['memory_limit_mib']:g} MiB sensor memory budget")
        for sid, ring in plan['arrays'].items():
            if ring['record_bytes'] + RING_SLOT_OVERHEAD >= ring['capacity']:
                raise ValueError(f'{sid}.model: one sample does not fit its ring')
        encoded = canonical(d)
        if len(encoded.encode()) > MAX_PROFILE_BYTES: raise ValueError('profile: exceeds 64 KiB')
        return cls(encoded)


def _resolve_ptz(sid, n):
    state, limits = n.setdefault('state', {}), n.setdefault('limits', {})
    if not isinstance(state, dict) or not isinstance(limits, dict):
        raise ValueError(f'{sid}.state/limits: objects required')
    if set(state) - set(PTZ_AXES) or set(limits) - set(PTZ_AXES):
        raise ValueError(f'{sid}.state/limits: only {", ".join(PTZ_AXES)} are supported')
    for k in PTZ_AXES:
        state[k] = number(state.get(k, 0.), f'{sid}.state.{k}')
        bounds = limits.get(k, [-180., 180.])
        if not isinstance(bounds, list) or len(bounds) != 2: raise ValueError(f'{sid}.limits.{k}: two bounds required')
        lo, hi = [number(v, f'{sid}.limits.{k}') for v in bounds]
        if lo > hi or not lo <= state[k] <= hi: raise ValueError(f'{sid}.state.{k}: outside limits')
        limits[k] = [lo, hi]


def _check_parents(nodes, sensors):
    """Exactly one path from every component to the vehicle body."""
    for sid, n in nodes.items():
        seen, parent = {sid}, n['parent']
        while parent != 'body':
            if parent not in nodes: raise ValueError(f'{sid}.parent: missing {parent}')
            if parent in seen: raise ValueError(f'{sid}.parent: cycle')
            if nodes[parent] in sensors: raise ValueError(f'{sid}.parent: sensors cannot parent components')
            seen.add(parent); parent = nodes[parent]['parent']


def _check_groups(groups):
    """Synchronized members must be comparable, or their geometry is undefined."""
    for name, members in groups.items():
        if len(members) < 2: raise ValueError(f'sync_group {name}: needs at least two members')
        first = members[0]
        if first['type'] != 'camera.rgb':
            raise ValueError(f'sync_group {name}: only camera.rgb members are supported')
        for other in members[1:]:
            if other['type'] != first['type'] or other['rate_hz'] != first['rate_hz'] or other['model'] != first['model']:
                raise ValueError(f'sync_group {name}: members need identical type, rate and lens model')
            if other['enabled'] != first['enabled']:
                raise ValueError(f'sync_group {name}: members must be enabled together')


def stereo_pair(base_id, *, baseline_m=.12, parent='body', rate_hz=30.,
                sync_group=None, pose=None, model=None):
    """Two cameras separated along the mount's right axis, as explicit poses.

    The baseline is stored as the two resolved poses rather than as a scalar,
    so a later asymmetric misalignment has somewhere to live.
    """
    group = sync_group or f'{base_id}_stereo'
    base = dict(pose or {})
    left, right = dict(base), dict(base)
    left['y_m'] = base.get('y_m', 0.) - baseline_m / 2
    right['y_m'] = base.get('y_m', 0.) + baseline_m / 2
    shape = model or dict(width_px=640, height_px=480, fov_h_deg=70., near_m=.15, far_m=150.)
    return [dict(id=f'{base_id}_left', type='camera.rgb', enabled=True, rate_hz=rate_hz,
                 parent=parent, pose_parent=left, model=dict(shape), sync_group=group),
            dict(id=f'{base_id}_right', type='camera.rgb', enabled=True, rate_hz=rate_hz,
                 parent=parent, pose_parent=right, model=dict(shape), sync_group=group)]
