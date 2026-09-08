"""Sensor v1 pymembus discovery, binary records, and matched RGB intake."""
from collections import OrderedDict
import json
import math
import struct
import time
import uuid
import numpy as np
from dvision2_common import load_pymembus, memkv_aligned_name_len, validate_id

SCHEMA = 'dvision2.sensor-manifest.v1'
MAX_MANIFEST = 65536
MAX_RECORD = 8192
# magic, version, header bytes, session UUID, generation, sequence, capture,
# sim microseconds, reset epoch, clock epoch, ID[48], payload schema, status, length
HEADER = struct.Struct('<4sHH16sQQQqQQ48sHHI')
REGISTRY_KEYS = ('sensors.schema', 'sensors.profile_name', 'sensors.profile_digest',
                 'sensors.manifest', 'sensors.generation', 'vehicle_id',
                 'provider_session_id', 'clock_domain_id', 'clock_epoch')

#: Payload schemas. Below 100 is strict compact JSON; 100 and above is a
#: packed little-endian array whose dtype and shape the manifest declares.
CAMERA_FRAME = 1
RANGE_SAMPLE = 2
LIDAR_FRAME = 3
GNSS_SAMPLE = 4
IMU_SAMPLE = 5
BAROMETER_SAMPLE = 6
MAGNETOMETER_SAMPLE = 7
TEMPERATURE_SAMPLE = 8
PACKED_ARRAY = 100
COMPACT_SCHEMA = {CAMERA_FRAME: 'camera.frame.v1', RANGE_SAMPLE: 'range.sample.v1',
                  LIDAR_FRAME: 'lidar.frame.v1', GNSS_SAMPLE: 'gnss.sample.v1',
                  IMU_SAMPLE: 'imu.sample.v1', BAROMETER_SAMPLE: 'barometer.sample.v1',
                  MAGNETOMETER_SAMPLE: 'magnetometer.sample.v1',
                  TEMPERATURE_SAMPLE: 'temperature.sample.v1'}
#: Profile sensor type -> the compact payload type it publishes.
STATE_PAYLOAD = {'position.gnss': GNSS_SAMPLE, 'motion.imu': IMU_SAMPLE,
                 'altimeter.barometric': BAROMETER_SAMPLE,
                 'heading.magnetometer': MAGNETOMETER_SAMPLE,
                 'environment.temperature': TEMPERATURE_SAMPLE}

STATUS_VALID = 1
STATUS_INVALID = 0


def json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False,
                      ensure_ascii=True).encode('utf-8')


def registry_name(instance):
    name = f'/dvision2.{validate_id(instance)}.sensors'
    if len(name.encode()) > 254: raise ValueError('instance: shared-memory name exceeds 254 bytes')
    return name


def data_name(instance, session, generation, suffix):
    name = f'/dvision2.{validate_id(instance)}.s{uuid.UUID(session).hex}.g{generation}.{suffix}'
    if len(name.encode()) > 254: raise ValueError('sensor channel name exceeds 254 bytes')
    return name


def encode_record(session, generation, sensor, sequence, capture, sim_us, payload,
                  *, reset_epoch=0, clock_epoch=0, payload_type=CAMERA_FRAME, status=STATUS_VALID):
    sid = validate_id(sensor).encode()
    if len(sid) > 48: raise ValueError('sensor id exceeds 48 bytes')
    raw = json_bytes(payload) if isinstance(payload, dict) else bytes(payload)
    return HEADER.pack(b'DVS1', 1, HEADER.size, uuid.UUID(session).bytes, generation,
                       sequence, capture, sim_us, reset_epoch, clock_epoch,
                       sid.ljust(48, b'\0'), payload_type, status, len(raw)) + raw


def decode_record(raw):
    if len(raw) < HEADER.size: raise ValueError('truncated sensor record')
    magic, version, size, session, gen, seq, cap, us, reset, clock, sid, typ, status, length = HEADER.unpack_from(raw)
    if magic != b'DVS1' or version != 1 or size != HEADER.size or len(raw) != size + length:
        raise ValueError('invalid sensor record header')
    payload = raw[size:]
    if typ < PACKED_ARRAY:
        payload = json.loads(payload, parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))
        if not isinstance(payload, dict): raise ValueError('compact payload must be an object')
    return dict(provider_session_id=uuid.UUID(bytes=session).hex, generation=gen,
                sensor_id=sid.rstrip(b'\0').decode('ascii'), sequence=seq,
                capture_id=cap, sim_time_us=us, reset_epoch=reset, clock_epoch=clock,
                payload_type=typ, status=status, payload=payload)


def unpack_array(payload, layout):
    """Split a packed array payload into the named fields the manifest declares."""
    raw, out, offset = memoryview(payload), {}, 0
    for field in layout:
        count = int(np.prod(field['shape']))
        dtype = np.dtype(field['dtype'])
        out[field['name']] = np.frombuffer(raw, dtype, count, offset).reshape(field['shape'])
        offset += count * dtype.itemsize
    return out


class RecordRing:
    """Generic binary access to pymembus' existing variable-record broadcast ring."""
    def __init__(self, name, size, *, create=False):
        self.pm = load_pymembus(); self.handle = self.pm.memmsg()
        if not hasattr(self.handle, 'write_bytes'):
            raise RuntimeError('pymembus byte APIs required; run scripts/build_sensor_pymembus.py --source PATH')
        if not self.handle.open(name, size, create, create):
            raise RuntimeError(f'{name}: {self.pm.last_error_message()}')
        self.name, self.size, self.overruns = name, size, 0
    def write(self, raw):
        if len(raw) + 64 >= self.size: raise ValueError('record exceeds ring capacity')
        if not self.handle.write_bytes(raw): raise RuntimeError(self.pm.last_error_message())
    def drain(self, limit=4096):
        # Bound work even if a concurrent publisher continuously fills the ring.
        for _ in range(limit):
            if not self.handle.poll(): break
            raw, overrun = self.handle.read_bytes_with_overrun(0)
            self.overruns += int(overrun)
            if raw: yield decode_record(raw)
    def close(self): self.handle.close()
    def unlink(self): self.pm.memmsg.remove(self.name)


class _Channels:
    """Every data area one manifest generation owns, created or destroyed together."""
    def __init__(self, pm):
        self.pm = pm; self.cameras = {}; self.arrays = {}; self.records = None
        self.camera_names = {}; self.record_name = None

    def destroy(self):
        for sensor_id, video in self.cameras.items():
            video.close(); self.pm.memvid.remove(self.camera_names[sensor_id])
        for ring in self.arrays.values():
            ring.close(); ring.unlink()
        if self.records is not None:
            self.records.close(); self.records.unlink()


class SensorPublisher:
    """Owns the registry and every generation-qualified channel a profile needs."""

    def __init__(self, instance, profile, *, session=None):
        self.pm = load_pymembus(); self.instance = instance
        self.session = session or uuid.uuid4().hex
        self.clock_epoch = uuid.uuid4().int & ((1 << 63) - 1)
        self.generation = 0; self.reset_epoch = 0
        self.channels = _Channels(self.pm); self.sequences = {}
        self.profile = None; self.manifest = {}
        self.registry = self.pm.memkv()
        name = registry_name(instance)
        self.pm.memkv.remove(name)
        if not self.registry.create(name, len(REGISTRY_KEYS), memkv_aligned_name_len(32, MAX_MANIFEST), MAX_MANIFEST, True):
            raise RuntimeError(self.pm.last_error_message())
        for i, k in enumerate(REGISTRY_KEYS):
            if not self.registry.setName(i, k): raise RuntimeError(self.pm.last_error_message())
        self.apply(profile)

    # -- lifecycle ---------------------------------------------------------

    def apply(self, profile):
        """Create generation N, commit its manifest, then release generation N-1."""
        from dsim.profiles import REGISTRY_BYTES
        plan = profile.plan
        if self.profile is not None and self.profile.memory_bytes + plan['total_bytes'] - REGISTRY_BYTES > profile.memory_limit_bytes:
            raise ValueError(f'Apply: active plus staged sensor memory exceeds {profile.memory_limit_bytes / 1048576:g} MiB')
        generation = self.generation + 1
        channels = _Channels(self.pm)
        try:
            manifest = self._build(profile, plan, generation, channels)
            encoded = json_bytes(manifest).decode()
            if len(encoded.encode()) > MAX_MANIFEST: raise ValueError('manifest exceeds 64 KiB')
            values = dict(zip(REGISTRY_KEYS, (
                SCHEMA, profile.data['name'], profile.digest, encoded, str(generation),
                self.instance, self.session, self.instance, str(self.clock_epoch))))
            if not self.registry.setAll(values): raise RuntimeError(self.pm.last_error_message())
        except Exception:
            channels.destroy()
            raise
        previous = self.channels
        self.channels, self.profile, self.manifest = channels, profile, manifest
        self.generation = generation
        self.sequences = {sid: 0 for sid in manifest['sensors']}
        previous.destroy()

    def _build(self, profile, plan, generation, channels):
        from dsim import sensor_models
        data = profile.data
        channels.record_name = data_name(self.instance, self.session, generation, 'sensor.samples')
        channels.records = RecordRing(channels.record_name, plan['sample_capacity'], create=True)
        sensors = {}
        for sensor in data['sensors']:
            if not sensor['enabled']: continue
            sid, kind, model = sensor['id'], sensor['type'], sensor['model']
            entry = dict(type=kind, enabled=True, rate_hz=sensor['rate_hz'],
                         parent=sensor['parent'], pose_parent=sensor['pose_parent'],
                         model=model, sync_group=sensor.get('sync_group'))
            if kind == 'camera.rgb':
                name = data_name(self.instance, self.session, generation, f'sensor.{sid}.video')
                video = self.pm.memvid()
                if not video.open(name, True, model['width_px'], model['height_px'],
                                  self.pm.video_format.rgb24, int(sensor['rate_hz']),
                                  plan['cameras'][sid]['slots']):
                    raise RuntimeError(f'{name}: {self.pm.last_error_message()}')
                channels.cameras[sid] = video; channels.camera_names[sid] = name
                entry.update(transport='video', channel=name, pixel_format='RGB24',
                             payload_type=CAMERA_FRAME, payload_schema=COMPACT_SCHEMA[CAMERA_FRAME],
                             calibration_revision=profile.digest)
            elif sid in plan['arrays']:
                ring = plan['arrays'][sid]
                name = data_name(self.instance, self.session, generation, f'sensor.{sid}.array')
                channels.arrays[sid] = RecordRing(name, ring['capacity'], create=True)
                entry.update(transport='array', channel=name, capacity=ring['capacity'],
                             fidelity=dict(material_independent=True, motion_distortion=False,
                                           multiple_returns=False),
                             record_bytes=ring['record_bytes'], payload_type=PACKED_ARRAY,
                             layout=sensor_models.array_layout(kind, model),
                             calibration=sensor_models.calibration(kind, model),
                             metadata_type=LIDAR_FRAME, metadata_schema=COMPACT_SCHEMA[LIDAR_FRAME])
            else:
                payload_type = STATE_PAYLOAD.get(kind, RANGE_SAMPLE)
                entry.update(transport='compact', channel=channels.record_name,
                             payload_type=payload_type,
                             payload_schema=COMPACT_SCHEMA[payload_type])
            sensors[sid] = entry
        return dict(schema=SCHEMA, provider_session_id=self.session, vehicle_id=self.instance,
                    generation=generation, clock_domain_id=self.instance, clock_epoch=self.clock_epoch,
                    primary_camera=data['primary_camera'], profile=data, profile_digest=profile.digest,
                    memory_bytes=plan['total_bytes'], sample_channel=channels.record_name,
                    sample_capacity=plan['sample_capacity'], sensors=sensors)

    def reset(self): self.reset_epoch += 1

    def close(self):
        self.channels.destroy()
        self.registry.close(); self.pm.memkv.remove(registry_name(self.instance))

    # -- publication -------------------------------------------------------

    @property
    def sensors(self): return self.manifest['sensors']

    def next_sequence(self, sensor_id):
        self.sequences[sensor_id] += 1
        return self.sequences[sensor_id]

    def camera_slot(self, sensor_id):
        """The writable RGB24 slot of a camera's ring, as a NumPy view.

        The view maps shared memory directly, so a caller must drop it before
        :meth:`commit_camera` advances the ring underneath it.
        """
        video = self.channels.cameras[sensor_id]
        return np.asarray(video[video.getPtr(0)])

    def commit_camera(self, sensor_id, sim_us):
        """Advance a camera's ring and return the exact committed frame sequence."""
        video = self.channels.cameras[sensor_id]
        slot = video.getPtr(0)
        video.setVpts(slot, sim_us); video.setApts(slot, sim_us); video.next(1)
        return video.getFrameSeq(slot)

    def write_compact(self, sensor_id, sequence, capture_id, sim_us, payload_type, payload,
                      *, status=STATUS_VALID):
        raw = self._record(sensor_id, sequence, capture_id, sim_us, payload_type, payload, status)
        if len(raw) > MAX_RECORD:
            raise ValueError(f'{sensor_id}: compact record exceeds {MAX_RECORD} bytes')
        self.channels.records.write(raw)

    def write_array(self, sensor_id, sequence, capture_id, sim_us, data, *, status=STATUS_VALID):
        self.channels.arrays[sensor_id].write(
            self._record(sensor_id, sequence, capture_id, sim_us, PACKED_ARRAY, data, status))

    def _record(self, sensor_id, sequence, capture_id, sim_us, payload_type, payload, status):
        return encode_record(self.session, self.generation, sensor_id, sequence, capture_id,
                             sim_us, payload, reset_epoch=self.reset_epoch,
                             clock_epoch=self.clock_epoch, payload_type=payload_type, status=status)


class SensorVideo:
    """Camera adapter exposes only copied, sequence-matched images to consumers.

    getSeq/getPtr/__getitem__ retain the applications' image intake interface;
    discovery and calibration have no legacy IPC aliases.
    """
    def __init__(self, instance, camera_id=None):
        self.instance=instance; self.pm=load_pymembus(); self.camera_id=camera_id
        self.video=self.records=None; self.identity=None; self.manifest={}
        self.pending=OrderedDict(); self.metadata=OrderedDict()
        self.frame=None; self.record=None; self.seq=0; self.revision=0; self.sequence_base=0
        self.last_probe=-1e9; self.association_drops=0; self.overruns=0
        self.last_seen_video=0; self.closed=False; self.selected=None
        self.reset_epoch = None

    def connect(self):
        now=time.monotonic()
        if now-self.last_probe < .25: return self.video is not None
        self.last_probe=now
        kv=self.pm.memkv()
        if not kv.open(registry_name(self.instance)):
            self._discard_transport()
            return False
        try: values=kv.getAll()
        finally: kv.close()
        if values.get('sensors.schema') != SCHEMA: return False
        manifest=json.loads(values['sensors.manifest'])
        identity=(values['provider_session_id'],int(values['sensors.generation']))
        if identity == self.identity and self.video is not None: return True
        self._discard_transport()
        selected=self.camera_id or manifest['primary_camera']
        entry=manifest['sensors'].get(selected)
        if entry is None or entry['transport'] != 'video': return False
        video=self.pm.memvid()
        if not video.open_existing(entry['channel']): return False
        try: records=RecordRing(manifest['sample_channel'],manifest['sample_capacity'])
        except RuntimeError:
            video.close(); return False
        self._close_handles()
        self.video,self.records=video,records
        self.identity,self.manifest,self.selected=identity,manifest,selected
        self.pending.clear();self.metadata.clear();self.frame=self.record=None
        self.last_seen_video=0;self.sequence_base=max(self.seq,self.sequence_base);self.seq=0;self.revision+=1
        return True

    def refresh(self):
        if self.closed: return
        self.connect()
        if self.video is None: return
        for rec in self.records.drain():
            if (rec['provider_session_id'],rec['generation']) != self.identity: continue
            if rec['sensor_id'] != self.selected or rec['payload_type'] != CAMERA_FRAME: continue
            if self.reset_epoch is not None and rec['reset_epoch'] != self.reset_epoch:
                self.association_drops += len(self.pending) + len(self.metadata)
                self.pending.clear(); self.metadata.clear()
                self.frame = self.record = None
                self.revision += 1
            self.reset_epoch = rec['reset_epoch']
            self.metadata[rec['payload']['video_sequence']]=rec
        self.overruns=self.records.overruns
        slot=self.video.getPtr(-1); before=self.video.getFrameSeq(slot)
        if before > self.last_seen_video:
            frame=np.array(self.video[slot],copy=True)
            after=self.video.getFrameSeq(slot)
            if before == after:
                self.pending[before]=frame;self.last_seen_video=before
        retention = self.manifest['profile'].get('transport', {}).get('retention_s', 1.)
        limit=math.ceil(self.getFps() * retention)+2
        for seq in list(self.pending):
            if seq in self.metadata:
                self.frame=self.pending.pop(seq);self.record=self.metadata.pop(seq)
                self.seq=self.sequence_base + seq
        for cache in (self.pending,self.metadata):
            while len(cache)>limit:
                cache.popitem(last=False);self.association_drops+=1

    def getSeq(self): self.refresh();return self.seq
    def getPtr(self, offset=-1): return 0
    def __getitem__(self,slot):
        if self.frame is None: raise RuntimeError('no matched camera frame')
        return self.frame
    def getWidth(self): return self.model['width_px']
    def getHeight(self): return self.model['height_px']
    def getFps(self): return self.entry['rate_hz']
    def getVpts(self,slot): return 0 if self.record is None else self.record['sim_time_us']
    def getSessionId(self): return self.revision
    @property
    def entry(self): return self.manifest['sensors'][self.selected]
    @property
    def model(self): return self.entry['model']
    def subscription(self, *, required=True):
        """What this handle is following, for a module's sensor health report."""
        if not self.manifest: return None
        return dict(sensor_id=self.selected, expected_hz=self.getFps(),
                    sync_group=self.entry.get('sync_group'),
                    generation=self.identity[1], provider_session_id=self.identity[0],
                    reset_epoch=self.reset_epoch, required=bool(required))

    def observation(self):
        """The last matched frame as an intake record, or None before the first.

        The per-sensor sequence rather than the reader's own running count:
        gaps in it are the frames this consumer never saw, which is the number
        its health report owes the operator.
        """
        if self.record is None: return None
        return dict(sensor_id=self.selected, sequence=self.record['sequence'],
                    sim_time_s=self.record['sim_time_us'] / 1e6,
                    generation=self.record['generation'],
                    capture_id=self.record['capture_id'],
                    overruns=self.overruns, drops=self.association_drops,
                    cache_bytes=self.cache_bytes)

    @property
    def cache_bytes(self):
        """Copied image and encoded metadata bytes, excluding Python overhead."""
        return (sum(frame.nbytes for frame in self.pending.values())
                + (0 if self.frame is None else self.frame.nbytes)
                + sum(len(json_bytes(record)) for record in self.metadata.values()))

    def capture_status(self, values=None):
        """Local algorithm input adapter, never published as vehicle status."""
        result=dict(values or {})
        if not self.manifest: return result
        m=self.model
        result.update({'camera.'+k:str(v) for k,v in m.items()})
        result['camera.fps']=str(self.getFps())
        result['camera.fov_h_deg']=str(math.degrees(2*math.atan(m['width_px']/(2*m['fx_px']))))
        result['camera.fov_v_deg']=str(math.degrees(2*math.atan(m['height_px']/(2*m['fy_px']))))
        if self.record:
            result['sim.time_s']=str(self.record['sim_time_us']/1e6)
            result.update({'drone.'+k:str(v) for k,v in self.record['payload']['body'].items()})
            result['sensor.pose']=self.record['payload']['pose']
            result['sensor.pose_world']=self.record['payload']['pose_world']
        return result
    def _close_handles(self):
        if self.video: self.video.close()
        if self.records: self.records.close()
    def _discard_transport(self):
        self._close_handles()
        self.video = self.records = None
        self.sequence_base = max(self.seq, self.sequence_base)
        self.seq = 0
        self.association_drops += len(self.pending) + len(self.metadata)
        self.pending.clear(); self.metadata.clear()
        self.frame = self.record = None
        self.manifest = {}; self.identity = None
        self.reset_epoch = None
        self.overruns = 0
    def close(self):
        self._close_handles();self.video=self.records=None;self.closed=True


def open_camera(instance, camera_id=None):
    video=SensorVideo(instance, camera_id)
    if video.connect(): return video
    video.close();return None


class SensorSamples:
    """Read-only intake for the shared compact ring and the array rings.

    Consumers that want numeric samples rather than pixels open this: it
    follows the registry the same way :class:`SensorVideo` does, and hands back
    decoded records grouped by sensor. Nothing in this repository is required
    to consume range or LiDAR yet; this is the reader the contract tests and
    the simulator's own diagnostics use.
    """

    def __init__(self, instance):
        self.instance=instance; self.pm=load_pymembus()
        self.records=None; self.arrays={}; self.identity=None; self.manifest={}
        self.last_probe=-1e9; self.overruns=0
        # Per-sensor sequence gaps. A ring's own overrun flag reports what the
        # transport noticed; a gap in the per-sensor sequence is what the
        # consumer actually missed, and the two are not the same number.
        self.last_sequence={}; self.skipped={}

    def connect(self):
        now=time.monotonic()
        if now-self.last_probe < .25: return self.records is not None
        self.last_probe=now
        kv=self.pm.memkv()
        if not kv.open(registry_name(self.instance)): return False
        try: values=kv.getAll()
        finally: kv.close()
        if values.get('sensors.schema') != SCHEMA: return False
        manifest=json.loads(values['sensors.manifest'])
        identity=(values['provider_session_id'],int(values['sensors.generation']))
        if identity == self.identity: return True
        records, arrays = None, {}
        try:
            records=RecordRing(manifest['sample_channel'],manifest['sample_capacity'])
            for sid, entry in manifest['sensors'].items():
                if entry['transport'] == 'array':
                    arrays[sid] = RecordRing(entry['channel'], entry['capacity'])
        except RuntimeError:
            if records is not None: records.close()
            for ring in arrays.values(): ring.close()
            return False
        self.close()
        self.records,self.arrays=records,arrays
        self.identity,self.manifest=identity,manifest
        self.last_sequence={}; self.skipped={sid:0 for sid in manifest['sensors']}
        return True

    def _account(self, record, stream):
        """Count the records this consumer never saw, per sensor.

        A LiDAR's metadata and its array share one sequence space, so the two
        streams are followed separately and their gaps summed. Sequences start
        at 1 in each generation, so a consumer is accountable for records
        published before it attached to that generation as well.
        """
        sid=record['sensor_id']; previous=self.last_sequence.get((stream,sid),0)
        if record['sequence'] > previous+1:
            self.skipped[sid]=self.skipped.get(sid,0)+record['sequence']-previous-1
        self.last_sequence[(stream,sid)]=record['sequence']
        return record

    def drain(self):
        """Compact records published since the last call, oldest first."""
        if not self.connect(): return []
        out=[self._account(r,'compact') for r in self.records.drain()
             if (r['provider_session_id'],r['generation'])==self.identity]
        self.overruns=self.records.overruns
        return out

    def drain_array(self, sensor_id):
        """Array records of one sensor, decoded into their named fields."""
        if not self.connect(): return []
        entry=self.manifest['sensors'][sensor_id]
        out=[]
        for record in self.arrays[sensor_id].drain():
            if (record['provider_session_id'],record['generation'])!=self.identity: continue
            record['fields']=unpack_array(record['payload'],entry['layout'])
            out.append(self._account(record,'array'))
        return out

    def close(self):
        if self.records: self.records.close()
        for ring in self.arrays.values(): ring.close()
        self.records=None; self.arrays={}
