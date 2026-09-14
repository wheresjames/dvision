"""Runtime mapping geometry: sizing defaults, snapping, caps and a measured budget."""
from __future__ import annotations

import math
import tempfile
import tracemalloc
import uuid
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dcmn.mapping import AllocationError, MappingConfig, contains

ROOT = Path(__file__).resolve().parents[1]


def test_the_side_multiplier_applies_to_side_length_not_area_or_radius():
    geometry = MappingConfig().resolve((0., 0.), (40., 0.))
    assert geometry.extent_m == (100., 100.)            # 2.5 x 40 m
    assert geometry.bounds_m() == (-30., -50., 70., 50.)  # centred between S and G


def test_the_minimum_and_the_margin_take_over_for_short_distances():
    assert MappingConfig().resolve((0., 0.), (0., 0.)).extent_m == (40., 40.)       # coincident
    assert MappingConfig().resolve((0., 0.), (10., 0.)).extent_m == (40., 40.)      # minimum
    config = MappingConfig(min_side_m=10., multiplier=1.)
    assert config.resolve((0., 0.), (30., 0.)).extent_m == (50., 50.)               # d + 2 * margin


def test_without_a_goal_the_square_is_around_the_first_pose_not_the_origin():
    geometry = MappingConfig().resolve((-100.3, 250.2))
    x0, y0, x1, y1 = geometry.bounds_m()
    assert x0 <= -120.3 and x1 >= -80.3 and y0 <= 230.2 and y1 >= 270.2
    assert contains(geometry, (-100.3, 250.2)) and not contains(geometry, (0., 0.))


def test_explicit_bounds_snap_outward_to_whole_cells_and_keep_negative_origins():
    geometry = MappingConfig(bounds=(-10.2, -5.1, 30.1, 25.), cell_m=.5).resolve((0., 0.), (500., 0.))
    assert geometry.bounds_m() == (-10.5, -5.5, 30.5, 25.)
    assert (geometry.width, geometry.height) == (82, 61)


@pytest.mark.parametrize('kwargs', [dict(cell_m=0.), dict(cell_m=math.nan), dict(dz_m=-1.),
                                    dict(bounds=(0., 0., 0., 1.)), dict(bounds=(0., 0., math.inf, 1.)),
                                    dict(multiplier=0.)])
def test_invalid_configuration_is_refused(kwargs):
    with pytest.raises(ValueError):
        MappingConfig(**kwargs)


def test_the_cell_cap_is_reported_with_requested_size_and_never_coarsened():
    with pytest.raises(AllocationError, match='limit is 4,000,000.*nothing is clipped or coarsened'):
        MappingConfig().resolve((0., 0.), (2000., 0.))


def test_admission_names_requested_and_allowed_sizes():
    config = MappingConfig(budget_bytes=64 << 20)
    geometry = config.resolve((0., 0.), (40., 0.))
    assert config.admit(geometry, 2) == config.estimate(geometry, 2)
    with pytest.raises(AllocationError, match=r'MiB.*budget 64.0 MiB'):
        config.admit(geometry, 2, retained_bytes=60 << 20)
    with pytest.raises(AllocationError, match='source'):
        MappingConfig(budget_bytes=1 << 20).admit(geometry, 1)


def test_the_default_budget_admits_hundreds_of_metres_but_not_kilometres():
    config = MappingConfig()
    config.admit(config.resolve((0., 0.), (180., 0.)), 2)       # a 450 m square, two sources
    with pytest.raises(AllocationError):
        config.admit(config.resolve((0., 0.), (400., 0.)), 2)   # a kilometre square


def _measured_peak(camera_algorithms, side):
    """tracemalloc peak of one mapping generation, its consumer, recorder and a plan."""
    from dalg.profiles import Profile, Source
    from dalg.sources import EvidenceSources
    from dcmn.archive import Recorder
    from dcmn.maps import EvidenceGrid, GridGeometry, MapSession
    from dnav.planners import build
    from dnav.policy import build_cost_map, load_policy

    geometry = GridGeometry.from_extent(side, side, .5)
    rows = (Source('scan', 'lidar_inverse'),) + tuple(Source('front', a) for a in camera_algorithms)
    model = dict(width_px=160, height_px=120, fx_px=114., fy_px=114., cx_px=80., cy_px=60.)
    session = SimpleNamespace(devices={'front': {'type': 'camera.rgb'}, 'scan': {'type': 'lidar.scan2d'}},
                              identity=('t', 1), reset_epoch=0)
    streams = {'front': SimpleNamespace(model=model), 'scan': SimpleNamespace()}
    tracemalloc.start()
    base = tracemalloc.get_traced_memory()[0]
    producer = EvidenceSources('mem-' + uuid.uuid4().hex[:8], Profile('m', rows, 'd'), session, streams, geometry)
    recorder = Recorder(Path(tempfile.mkdtemp()) / 'a', commit_s=60.)
    consumer = MapSession(producer.publisher.instance)
    try:
        for step in range(3):
            for state in producer.states.values():
                state.evidence.algorithm.grid.observed[:] = True
                state.evidence.algorithm.grid.observed_ms[:] = step + 1
            producer.publish(float(step + 1), force=True)
            recorder.record('evidence.published', {}, producer.grids())
        consumer.poll()
        free = {sid: EvidenceGrid(geometry, np.zeros(geometry.shape, np.uint8),
                                  np.ones(geometry.shape, np.uint32), sid, 1, 1.)
                for sid in consumer.sources}
        cost = build_cost_map(free, load_policy('default', ROOT))
        x0, y0, x1, y1 = geometry.bounds_m()
        route = build('astar').plan(cost, (x0 + .3, y0 + .3, 1.5), (x1 - .3, y1 - .3, 1.5), cost.policy)
        assert route.ok
        return tracemalloc.get_traced_memory()[1] - base, geometry, len(rows)
    finally:
        tracemalloc.stop()
        recorder.close(); consumer.close(); producer.close()


@pytest.mark.parametrize('cameras,side', [(('ground_plane',), 40.), (('ground_plane',), 120.),
    (('ground_plane', 'optical_flow_triangulation', 'feature_triangulation', 'sgbm', 'plane_sweep'), 60.)])
def test_the_admission_estimate_covers_the_measured_peak(cameras, side):
    """Default camera/lidar and a larger multi-source configuration (DV-MAPPING §9)."""
    peak, geometry, sources = _measured_peak(cameras, side)
    assert MappingConfig.estimate(geometry, sources) >= peak, (peak, MappingConfig.estimate(geometry, sources))
