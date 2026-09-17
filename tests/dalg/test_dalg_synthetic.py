"""The evidence fixture: a known room, and nothing that pretends to be one.

A transport and UI test fixture, kept in test tooling (dtest) rather than in
dalg, with one standing rule -- it publishes a fixed known grid and must never grow
settings that make it look like an algorithm. The last test here is that rule.
"""

from __future__ import annotations

import inspect
import uuid

import numpy as np
import pytest

from dcmn.context import Context
from dtest import synthetic
from dtest.synthetic import (SOURCE_ID, SyntheticProducer, SyntheticRoom,
                             observer_m, room_occupancy)
from dcmn.maps import NEVER_OBSERVED, GridGeometry, MapSession


def geometry(cell_m=.5, extent_m=20.):
    return GridGeometry.from_extent(extent_m, extent_m, cell_m)


def test_the_room_is_enclosed_and_has_exactly_one_doorway():
    geo = geometry()
    obstacles = room_occupancy(geo)
    assert obstacles[0].all() and obstacles[-1].all()
    assert obstacles[:, 0].all() and obstacles[:, -1].all()
    # The divider spans the room but for one gap, and the gap is contiguous.
    column = obstacles[:, geo.width // 2]
    gap = np.flatnonzero(~column)
    assert gap.size > 1
    assert np.array_equal(gap, np.arange(gap[0], gap[-1] + 1))


@pytest.mark.parametrize('cell_m', (.25, .5, 1.))
def test_the_divider_stays_solid_at_every_resolution(cell_m):
    """A wall thinner than a cell is a wall a planner walks through."""
    geo = geometry(cell_m=cell_m)
    obstacles = room_occupancy(geo)
    middle = geo.width // 2
    assert obstacles[1, middle - 1:middle + 2].any()


def test_a_sweep_observes_from_the_fixed_point_and_leaves_a_shadow():
    geo = geometry()
    room = SyntheticRoom(geo)
    assert (room.occupancy == NEVER_OBSERVED).all()
    for step in range(12):
        room.sweep(1. + step)
    never = room.occupancy[0] == NEVER_OBSERVED
    # A full turn from inside the left half sees that half and, through the
    # doorway, a cone of the right one -- and never what the divider hides.
    ox, _ = observer_m(geo)
    left = slice(0, int(ox / geo.cell_m))
    assert not never[2:-2, left].any()
    assert never.any(), 'the divider casts no shadow'
    behind = never[:, geo.width // 2 + 2:]
    assert behind.any() and not behind.all()


def test_the_sweep_dates_what_it_sees_and_the_dates_move():
    geo = geometry()
    room = SyntheticRoom(geo)
    room.sweep(1.)
    first = room.observed_ms.copy()
    room.sweep(2.)
    stamps = set(np.unique(room.observed_ms)) - {0}
    assert stamps == {1000, 2000}
    assert not np.array_equal(first, room.observed_ms)


def test_the_room_is_valid_evidence_at_every_step():
    """The two sentinels have to agree on every cell, every publish."""
    room = SyntheticRoom(geometry())
    for step in range(14):
        room.sweep(1. + step)
        grid = room.grid()          # EvidenceGrid validates on construction
        assert grid.source == SOURCE_ID
        assert grid.observed.sum() == (grid.occupancy != NEVER_OBSERVED).sum()


def test_the_fixture_publishes_nothing_without_a_session_clock(capsys):
    """No session context is no clock, and the fixture says so instead of guessing."""
    instance = f'synth-{uuid.uuid4().hex[:8]}'
    producer = SyntheticProducer(instance, geometry())
    try:
        for _ in range(5): assert producer.step()
        assert producer.published == 0
        assert 'waiting for the session context' in capsys.readouterr().out
    finally:
        producer.close()


def test_the_fixture_publishes_on_the_real_plane_at_the_context_cadence(tmp_path):
    instance = f'synth-{uuid.uuid4().hex[:8]}'
    context = Context(instance)
    context.start(tmp_path, clock_epoch=11)
    producer = SyntheticProducer(instance, geometry())
    session = MapSession(instance)
    pose = dict(x_m=1., y_m=1., z_m=1., heading_deg=0., roll_deg=0., pitch_deg=0.)
    try:
        for tick in range(60):
            context.publish_pose(pose, tick * .25, clock_epoch=11)
            producer.step()
        session.poll()
        state = session.states[SOURCE_ID]
        # Fifteen data-clock seconds at a one-second cadence.
        assert producer.published == 15
        assert state.revision == 15
        assert state.sim_time_s == pytest.approx(14., abs=.01)
        assert not session.stale(SOURCE_ID, 14.75)
        assert session.stale(SOURCE_ID, 44.75)
        grid = session.latest(SOURCE_ID)
        assert grid.geometry == producer.geometry
        assert grid.never_observed.any() and grid.observed.any()
        # Labelled with the context it was published under, like any producer.
        assert session.context['clock_epoch'] == 11 and session.context['frame_id'] == 'local'
    finally:
        session.close(); producer.close(); context.close()


def test_the_fixture_has_no_settings_of_its_own():
    """The standing rule: a fixture that can be tuned is an algorithm in disguise."""
    room = inspect.signature(SyntheticRoom.__init__).parameters
    assert list(room) == ['self', 'geometry']
    producer = inspect.signature(SyntheticProducer.__init__).parameters
    # Geometry, cadence and provenance come from outside; nothing describes how
    # the fixture observes, because that is hard-coded and must stay so.
    assert set(producer) == {'self', 'instance', 'geometry', 'cadence_s',
                             'staleness_horizon_s', 'profile_name', 'profile_digest'}
    assert not hasattr(synthetic, 'CONFIGS')
