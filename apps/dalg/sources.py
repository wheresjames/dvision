"""Build one evidence instance per available source, without scoring copies."""
from dataclasses import dataclass
from pathlib import Path
from dalg.evidence import CameraEvidence, LidarEvidence
from dalg.model import Frame, Intrinsics, Pose
from dalg.profiles import camera_evidence_algorithms, validate_sources
from dcmn.maps import MapPublisher


def resolved_settings(algorithm, settings, root):
    value = dict(settings)
    if algorithm == 'monocular_depth':
        path = Path(value.get('model_path', '')).expanduser()
        if not path.is_absolute(): path = Path(root)/path
        if not value.get('model_path') or not path.is_file():
            raise ValueError(f'monocular_depth model {str(path) if value.get("model_path") else "(no model_path)"} '
                             'is not installed; install it with scripts/install_dalg_depth_model.py')
        value['model_path'] = str(path)
    return value


def camera_frame(sample):
    def pose(value): return Pose(*(value[k] for k in ('x_m','y_m','z_m','heading_deg','roll_deg','pitch_deg')))
    return Frame(sample.sequence, sample.sim_time_s, sample.image,
                 pose(sample.payload['body']), camera=pose(sample.payload['pose']))


@dataclass
class SourceState:
    config: object
    stream: object
    evidence: object
    last_sequence: int = 0
    last_image: object = None
    preview: object = None
    samples: int = 0
    skipped: int = 0
    rejected: int = 0
    error: str = ''


class EvidenceSources:
    def __init__(self, instance, profile, session, streams, geometry, *, root=None,
                 context=None, generation=1, activate=True):
        validate_sources(profile.sources, {'sensors': session.devices})
        self.publisher = MapPublisher(instance, geometry, [dict(id=s.id, sensor=s.sensor,
            algorithm=s.algorithm, sensor_type=session.devices[s.sensor]['type']) for s in profile.sources],
            profile_name=profile.name, profile_digest=profile.digest, context=context,
            generation=generation, activate=False)
        self.states = {}
        self.context = dict(self.publisher.context)
        self.generation = session.identity; self.reset_epoch = session.reset_epoch
        try:
            for source in profile.sources:
                stream = streams[source.sensor]
                if source.algorithm in camera_evidence_algorithms():
                    intrinsics = Intrinsics(*(stream.model[k] for k in
                        ('width_px','height_px','fx_px','fy_px','cx_px','cy_px')))
                    evidence = CameraEvidence(geometry, intrinsics, source.sensor,
                        resolved_settings(source.algorithm, source.settings, root or Path('.')),
                        algorithm=source.algorithm, publisher=self.publisher)
                else: evidence = LidarEvidence(source, self.publisher)
                self.states[source.id] = SourceState(source, stream, evidence)
            if activate: self.publisher.activate()
        except Exception:
            self.publisher.close(); raise

    def publish(self, now, *, force=False):
        for state in self.states.values(): state.evidence.publish(now, force=force)

    def grids(self):
        return {sid:s.evidence.latest for sid,s in self.states.items() if s.evidence.latest is not None}

    def close(self): self.publisher.close()
