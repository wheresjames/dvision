"""Evidence source adapters: sensor samples in, stamped evidence grids out.

Nothing here reads a world file, a tour or a truth channel. Each adapter owns one
algorithm instance on the runtime geometry and publishes through the shared
generation publisher, so every source of a generation carries the same frame,
epochs and geometry revision.
"""
from dataclasses import replace

from dalg.grid import ObservedGrid
from dcmn.maps import EvidenceGrid, quantize


class GridEvidence:
    """The publication lifecycle shared by every evidence source."""
    def publish(self, now, *, force=False):
        if not force and self.last_publish is not None and now-self.last_publish < self.publisher.cadence_s:
            return
        grid = self.algorithm.grid
        occupancy = quantize(grid.result().probabilities)
        occupancy[~grid.observed] = 255
        stamps = grid.observed_ms.copy()[None]
        revision = self.publisher.publish(self.source, occupancy[None], stamps, now)
        self.latest = EvidenceGrid(self.geometry, occupancy[None], stamps,
                                  self.source, revision, now,
                                  entry=self.publisher.sources[self.source])
        if self.last_publish is not None and now > self.last_publish:
            self.rate_hz = 1.0 / (now-self.last_publish)
        self.last_publish = now
        self.published += 1


class CameraEvidence(GridEvidence):
    """One camera algorithm on runtime geometry; no world or mission input."""
    def __init__(self, geometry, intrinsics, sensor, settings, *, algorithm, publisher):
        from dalg.algo import ALGORITHMS
        from dalg.profiles import Source, camera_evidence_algorithms
        if algorithm not in camera_evidence_algorithms():
            raise ValueError(f'{algorithm} does not publish evidence')
        self.geometry = geometry
        self.algorithm_name = algorithm
        self.source = Source(sensor, algorithm).id
        self.algorithm = ALGORITHMS[algorithm](*geometry.extent_m, intrinsics,
                                               settings=settings, evidence=True)
        self.algorithm.grid = ObservedGrid(self.geometry)
        self.publisher = publisher
        self.latest = None
        self.last_publish = None
        self.published = 0
        self.rate_hz = 0.0

    def observe(self, frame):
        """Feed one frame captured at its own, capture-associated pose."""
        self.algorithm.grid.timestamp_s = frame.timestamp_s
        # The algorithm's cell indexing is local to the grid's origin, which may
        # be negative; only x/y translate. Camera height and attitude keep their
        # physical meaning above the ground.
        def local(pose):
            return replace(pose, x_m=pose.x_m-self.geometry.origin_x_m,
                           y_m=pose.y_m-self.geometry.origin_y_m)
        self.algorithm.observe(replace(frame, pose=local(frame.pose),
            camera=local(frame.camera_pose), range_m=None, range_confidence=None))
        # An algorithm that pairs frames fuses here, from frames already
        # captured only -- never from the future.
        fuse = getattr(self.algorithm, 'fuse_available', None)
        if fuse is not None: fuse()


class LidarEvidence(GridEvidence):
    """Same publication contract with calibrated scan rays as its input."""
    def __init__(self, source, publisher):
        from dalg.algo.lidar import LidarInverseModel
        self.geometry = publisher.geometry
        self.source = source.id
        self.algorithm = LidarInverseModel(self.geometry, source.settings)
        self.publisher = publisher
        self.latest = None
        self.last_publish = None
        self.published = 0
        self.rate_hz = 0.0

    def observe(self, sample):
        self.algorithm.observe(sample)
