"""The evidence plane: geometry, codec, discovery, revision and staleness.

The evidence contract's review gate is that a torn, oversize or geometry-inconsistent
record is rejected *with a message*, that revisions advance only on change, and
that a consumer opens the plane whatever order the processes started in.
"""

from __future__ import annotations

import uuid

import numpy as np
import pytest

from dcmn import maps
from dcmn.maps import (EVIDENCE_GRID, NEVER_OBSERVED, EvidenceGrid, GridGeometry,
                       MapPublisher, MapSession, decode_grid, encode_grid,
                       quantize, stamp_ms)
from dcmn.sensors import encode_record


def geometry(cell_m=.5, width_m=10., height_m=8.):
    return GridGeometry.from_extent(width_m, height_m, cell_m)


def instance(name):
    """A per-test instance id, so two tests never share a shared-memory plane."""
    return f'maps-{name}-{uuid.uuid4().hex[:8]}'


def populated(geo, *, probability=.9, sim_time_s=2.5):
    grid = EvidenceGrid.blank(geo)
    occupancy = grid.occupancy.copy()
    observed = grid.observed_ms.copy()
    occupancy[0, 1:4, 1:4] = quantize(probability)
    observed[0, 1:4, 1:4] = stamp_ms(sim_time_s)
    return occupancy, observed


# -- geometry ----------------------------------------------------------------

def test_geometry_rounds_an_extent_outward_and_sizes_its_record():
    geo = GridGeometry.from_extent(10.1, 8., .5)
    assert (geo.width, geo.height, geo.layers) == (21, 16, 1)
    assert geo.shape == (1, 16, 21)
    # Five bytes a cell -- one of occupancy, four of timestamp -- plus headers.
    assert geo.record_bytes == 128 + maps.GRID_HEADER.size + 21 * 16 * 5


def test_geometry_floors_rather_than_truncating_at_the_origin():
    """int() rounds toward zero, which would fold the west edge onto column 0."""
    geo = GridGeometry(0., 0., .5, 4, 4)
    assert geo.to_cell(.25, .25) == (0, 0)
    assert geo.to_cell(-.25, .25) is None
    assert geo.to_cell(.25, -.25) is None
    assert geo.to_cell(1.99, 1.99) == (3, 3)
    assert geo.to_cell(2.01, 1.) is None


def test_geometry_rejects_what_nothing_could_publish():
    for bad in (dict(cell_m=0.), dict(cell_m=-1.), dict(width=0), dict(layers=0),
                dict(dz_m=0.), dict(cell_m=float('nan'))):
        with pytest.raises(ValueError):
            GridGeometry(**{**dict(origin_x_m=0., origin_y_m=0., cell_m=.5,
                                   width=4, height=4), **bad})
    with pytest.raises(ValueError, match='over the'):
        GridGeometry(0., 0., .001, 4000, 4000)


def test_an_offset_origin_reads_back_in_absolute_metres():
    geo = GridGeometry(100., -50., .5, 8, 8)
    assert geo.bounds_m() == (100., -50., 104., -46.)
    assert geo.to_cell(100.25, -49.75) == (0, 0)
    assert geo.to_cell(99.9, -49.) is None
    assert geo.cell_centre_m(0, 0) == (100.25, -49.75)


# -- the sentinel ------------------------------------------------------------

def test_a_probability_never_quantizes_onto_the_never_observed_sentinel():
    values = quantize(np.linspace(0., 1., 1000))
    assert values.max() == 254 and values.min() == 0
    assert not (values == NEVER_OBSERVED).any()


def test_an_observation_at_simulated_zero_is_not_never_observed():
    """Timestamp zero is the sentinel, so the first tick has to round off it."""
    assert int(stamp_ms(0.)) == 1
    assert int(stamp_ms(.0004)) == 1


def test_a_grid_whose_two_sentinels_disagree_is_refused():
    geo = geometry()
    grid = EvidenceGrid.blank(geo)
    occupancy = grid.occupancy.copy()
    occupancy[0, 0, 0] = 10                       # observed, but timestamp is 0
    with pytest.raises(ValueError, match='never-observed'):
        EvidenceGrid(geo, occupancy, grid.observed_ms)


def test_never_observed_is_neither_free_nor_occupied():
    geo = geometry()
    occupancy, observed = populated(geo, probability=.05)
    grid = EvidenceGrid(geo, occupancy, observed)
    probability = grid.probability
    assert np.isnan(probability[grid.never_observed]).all()
    assert grid.never_observed.sum() == geo.cells - 9
    assert grid.probability_at(.75, .75) == pytest.approx(.05, abs=.005)
    assert grid.probability_at(4.75, 3.75) is None


# -- codec -------------------------------------------------------------------

def test_a_grid_survives_the_wire_unchanged():
    geo = geometry()
    occupancy, observed = populated(geo)
    payload = encode_grid(geo, occupancy, observed)
    assert len(payload) == maps.GRID_HEADER.size + geo.cells * 5
    back, occupancy_out, observed_out = decode_grid(payload)
    assert back == geo
    assert np.array_equal(occupancy_out, occupancy)
    assert np.array_equal(observed_out, observed)


@pytest.mark.parametrize('mangle,message', [
    (lambda p: p[:20], 'truncated'),
    (lambda p: p[:-1], 'record carries'),
    (lambda p: p + b'\0', 'record carries'),
    (lambda p: b'XXXX' + p[4:], 'magic'),
    (lambda p: p[:4] + b'\x09\x00' + p[6:], 'unsupported'),
])
def test_a_damaged_record_is_rejected_with_a_message(mangle, message):
    geo = geometry()
    payload = encode_grid(geo, *populated(geo))
    with pytest.raises(ValueError, match=message):
        decode_grid(mangle(payload))


def test_an_oversize_geometry_is_refused_before_anything_is_allocated():
    """A header may not talk a consumer into allocating whatever it likes."""
    header = maps.GRID_HEADER.pack(maps.GRID_MAGIC, maps.GRID_VERSION,
                                   maps.GRID_HEADER.size, 0., 0., .01, 0., 3.,
                                   100000, 100000, 1)
    with pytest.raises(ValueError, match='over the'):
        decode_grid(header + b'\0' * 16)


def test_a_grid_that_does_not_match_the_manifest_geometry_is_rejected():
    geo = geometry()
    payload = encode_grid(geo, *populated(geo))
    with pytest.raises(ValueError, match='does not match the manifest'):
        decode_grid(payload, expect=geometry(cell_m=.25))


def test_a_record_whose_sentinels_disagree_is_rejected_on_the_wire():
    geo = geometry()
    occupancy, observed = populated(geo)
    occupancy[0, 9, 9] = 200                      # observed, timestamp still 0
    payload = encode_grid(geo, occupancy, observed)
    with pytest.raises(ValueError, match='never-observed'):
        decode_grid(payload)


# -- the plane ---------------------------------------------------------------

@pytest.fixture
def plane():
    """A publisher and a session on the same instance, both cleaned up."""
    made = []
    def build(sources=('cam',), geo=None, **kwargs):
        name = instance('plane')
        geo = geo or geometry()
        publisher = MapPublisher(
            name, geo, [dict(id=sid, sensor=sid, sensor_type='camera.rgb',
                             algorithm='sgbm') for sid in sources], **kwargs)
        session = MapSession(name)
        made.append((publisher, session))
        return name, geo, publisher, session
    yield build
    for publisher, session in made:
        session.close(); publisher.close()


def test_a_consumer_discovers_a_producer_that_started_first(plane):
    name, geo, publisher, session = plane()
    assert session.connect()
    assert session.geometry == geo
    assert set(session.sources) == {'cam'}
    assert session.sources['cam']['payload_type'] == EVIDENCE_GRID
    assert session.sources['cam']['record_bytes'] == geo.record_bytes
    assert session.manifest['schema'] == maps.SCHEMA


def test_a_consumer_that_started_first_keeps_retrying(plane):
    name = instance('late')
    session = MapSession(name, probe_interval_s=0.)
    try:
        assert not session.connect()
        assert not session.connect()
        publisher = MapPublisher(name, geometry(),
                                 [dict(id='cam', sensor='front')])
        try:
            session._backoff_s = 0.
            assert session.connect()
            assert set(session.sources) == {'cam'}
        finally:
            publisher.close()
    finally:
        session.close()


def test_a_revision_advances_only_on_change_and_the_clock_always(plane):
    name, geo, publisher, session = plane()
    occupancy, observed = populated(geo)
    assert publisher.publish('cam', occupancy, observed, 1.) == 1
    assert publisher.publish('cam', occupancy, observed, 2.) == 1
    occupancy[0, 6, 6] = quantize(.2); observed[0, 6, 6] = stamp_ms(3.)
    assert publisher.publish('cam', occupancy, observed, 3.) == 2
    session.poll()
    state = session.states['cam']
    assert state.records == 3 and state.revision == 2
    assert state.sim_time_s == pytest.approx(3.)
    assert session.latest('cam').revision == 2


def test_a_map_goes_stale_on_simulated_time_not_wall_time(plane):
    name, geo, publisher, session = plane(staleness_horizon_s=5.)
    publisher.publish('cam', *populated(geo), 10.)
    session.poll()
    assert not session.stale('cam', 12.)
    assert session.age_s('cam', 12.) == pytest.approx(2.)
    assert session.stale('cam', 20.)
    assert session.report(20.)['sources']['cam']['stale'] is True


def test_a_source_that_has_published_nothing_is_stale_not_fresh(plane):
    name, geo, publisher, session = plane()
    assert session.connect()
    assert session.stale('cam', 0.)
    assert session.age_s('cam', 0.) is None


def test_a_departed_producer_becomes_a_stale_map_rather_than_a_vanished_one():
    """Forgetting a source when its producer exits leaves nothing to complain about."""
    name = instance('gone')
    publisher = MapPublisher(name, geometry(), [dict(id='cam')])
    session = MapSession(name, probe_interval_s=0.)
    try:
        publisher.publish('cam', *populated(publisher.geometry), 1.)
        session.poll()
        assert not session.stale('cam', 2.)
        publisher.close()
        session._backoff_s = 0.
        assert not session.connect()
        assert session.latest('cam') is not None
        assert session.stale('cam', 30.)
        assert session.report(30.)['connected'] is False
    finally:
        session.close()


def test_a_restarted_producer_is_a_new_session_and_clears_the_old_state():
    name = instance('restart')
    first = MapPublisher(name, geometry(), [dict(id='cam')])
    session = MapSession(name, probe_interval_s=0.)
    try:
        first.publish('cam', *populated(first.geometry), 1.)
        session.poll()
        assert session.latest('cam').revision == 1
        old_session_id = session.identity[0]
        first.close()
        second = MapPublisher(name, geometry(), [dict(id='cam')])
        try:
            session._backoff_s = 0.
            assert session.connect()
            assert session.identity[0] != old_session_id
            assert session.latest('cam') is None      # the old belief is not carried over
            second.publish('cam', *populated(second.geometry, probability=.1), 4.)
            session.poll()
            assert session.latest('cam').revision == 1
            assert session.discoveries == 2
        finally:
            second.close()
    finally:
        session.close()


def test_two_sources_are_two_grids_on_two_channels(plane):
    name, geo, publisher, session = plane(sources=('camera', 'lidar'))
    publisher.publish('camera', *populated(geo, probability=.9), 1.)
    publisher.publish('lidar', *populated(geo, probability=.1), 1.)
    session.poll()
    assert session.latest('camera').probability_at(.75, .75) > .5
    assert session.latest('lidar').probability_at(.75, .75) < .5
    channels = {entry['channel'] for entry in session.sources.values()}
    assert len(channels) == 2


def test_a_record_a_consumer_cannot_use_is_reported_not_swallowed(plane):
    name, geo, publisher, session = plane()
    assert session.connect()
    # A record with the right envelope and a payload from a different geometry:
    # exactly what a producer restarted against a new profile would emit if it
    # kept writing to a ring a consumer had already opened.
    other = geometry(cell_m=.25)
    publisher.rings['cam'].write(encode_record(
        publisher.session, publisher.generation, 'cam', 1, 1, 1_000_000,
        encode_grid(other, *populated(other)), payload_type=EVIDENCE_GRID))
    session.poll()
    assert session.latest('cam') is None
    assert session.states['cam'].rejected == 1
    assert 'does not match the manifest' in session.states['cam'].last_reason
    assert session.report(1.)['rejections']


def test_a_payload_of_the_wrong_type_is_named_in_the_rejection(plane):
    name, geo, publisher, session = plane()
    assert session.connect()
    publisher.rings['cam'].write(encode_record(
        publisher.session, publisher.generation, 'cam', 1, 1, 1_000_000,
        encode_grid(geo, *populated(geo)), payload_type=maps.EVIDENCE_PATCH))
    session.poll()
    assert 'is not 110' in session.states['cam'].last_reason
    assert session.latest('cam') is None


def test_a_ring_is_sized_from_the_declared_extent_not_the_sensor_cap(plane):
    """A 200x200 grid is ~200 KB; the compact ring's 8 KiB cap does not apply."""
    geo = GridGeometry.from_extent(100., 100., .5)
    name, _, publisher, session = plane(geo=geo)
    assert geo.record_bytes > 8192
    publisher.publish('cam', *populated(geo), 1.)
    session.poll()
    assert session.latest('cam').geometry == geo
    assert session.states['cam'].rejected == 0


def test_a_published_grid_round_trips_through_the_plane(plane):
    name, geo, publisher, session = plane()
    occupancy, observed = populated(geo, probability=.77, sim_time_s=4.25)
    publisher.publish('cam', occupancy, observed, 4.25)
    session.poll()
    grid = session.latest('cam')
    assert np.array_equal(grid.occupancy, occupancy)
    assert np.array_equal(grid.observed_ms, observed)
    assert grid.source == 'cam' and grid.sim_time_s == pytest.approx(4.25)
    assert grid.newest_observation_s() == pytest.approx(4.25, abs=.001)
    assert grid.entry['algorithm'] == 'sgbm'
    age = grid.age_s(6.25)
    assert np.isnan(age[grid.never_observed]).all()
    assert age[0, 2, 2] == pytest.approx(2., abs=.01)
