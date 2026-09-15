"""The optional reference-imagery plane, and everything a display does with it.

The contract under test is the reference-imagery plane's: an image is
decoration that can never become an input. Registration is exact under the full 2D affine --
translate, rotate, reflect, unequal scale -- and clipped, never stretched;
anything malformed, oversized, stale or lying about its checksum is refused
with a message or quietly not displayed; and nothing about imagery may change
one number the modules produce, which the three-background test says outright
by running the whole pipeline with no image, a correct one and a lying one
and demanding the same evidence and the same routes from all three.
"""

from __future__ import annotations

import hashlib
import io
import time
import uuid
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from dcmn import theme
from dcmn.archive import (ArchiveReader, Recorder, digest_image, displayed_background,
                          next_archive_dir, render_attempt)
from dcmn.imagery import (MAX_ENCODED_BYTES, MAX_IMAGES, REFERENCE_IMAGE,
                          SIMULATION_TRUTH, ImagePublisher, ImageSession,
                          ReferenceImage, affine_inverse, affine_xy, build_metadata,
                          decode_png, decode_revision, encode_revision, validate_affine)
from dcmn.map_pane import (SNAPSHOT_CELL_PX, Background, MapPane, Overlay,
                           ReferenceBackgroundHost, background_raster, background_shapes,
                           compose_background, never_cells, render_raster, snapshot_image)
from dcmn.maps import EvidenceGrid, GridGeometry, quantize, stamp_ms
from dcmn.sensors import encode_record
from dtest.provider import FixtureProvider
from dtest.tkfixture import hidden_tk

ROOT = Path(__file__).resolve().parents[1]
RED, GREEN = (255, 0, 0, 255), (0, 255, 0, 255)


def instance(name):
    """A per-test instance id, so two tests never share a shared-memory plane."""
    return f'imagery-{name}-{uuid.uuid4().hex[:8]}'


def png_bytes(width, height, pixels):
    """A PNG of solid transparent ground with the given RGBA pixels set."""
    image = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    for (col, row), colour in pixels.items():
        image.putpixel((col, row), colour)
    buffer = io.BytesIO(); image.save(buffer, 'PNG')
    return buffer.getvalue()


def reference(png, affine, *, revision=1, image_id='world', localization_epoch=0,
              clock_epoch=0, source_category=SIMULATION_TRUTH):
    """One validated revision, built the way the publisher builds one."""
    metadata = build_metadata(image_id, png, affine, revision=revision,
                              provider_session_id='test', source_category=source_category,
                              frame_id='local', localization_epoch=localization_epoch,
                              clock_domain_id='test', clock_epoch=clock_epoch)
    return ReferenceImage(image_id=image_id, revision=revision, metadata=metadata, png=png,
                          width=metadata['width'], height=metadata['height'],
                          affine=tuple(metadata['affine']),
                          image=decode_png(png, metadata['width'], metadata['height']))


def sample_grid(cell_m=1., width=4, height=2, observed=()):
    """Free floor where somebody has looked, never-observed where nobody has."""
    geo = GridGeometry(0., 0., cell_m, width, height)
    grid = EvidenceGrid.blank(geo)
    occupancy, observed_ms = grid.occupancy.copy(), grid.observed_ms.copy()
    for col, row in observed:
        occupancy[0, row, col] = quantize(.05)
        observed_ms[0, row, col] = stamp_ms(10.)
    return EvidenceGrid(geo, occupancy, observed_ms, 'cam', revision=1, sim_time_s=10.)


def consumer(instance_id):
    """A session with discovery backoff removed, so tests never wait on a clock."""
    return ImageSession(instance_id, probe_interval_s=0., max_probe_interval_s=0.)


# -- the affine ---------------------------------------------------------------

def test_the_affine_places_pixel_centres_and_inverts_exactly():
    translate = (1., 0., 0., 1., 3., -2.)
    assert affine_xy(translate, 0, 0) == (3., -2.)
    assert affine_xy(translate, 4, 7) == (7., 5.)
    assert np.allclose(affine_xy(affine_inverse(translate), 7., 5.), (4., 7.))
    # A quarter turn: columns run south, rows run west, and it is still rigid.
    rotate = (0., -1., 1., 0., 4., 4.)
    assert affine_xy(rotate, 0, 0) == (4., 4.)
    assert affine_xy(rotate, 2, 0) == (4., 6.)
    assert affine_xy(rotate, 0, 2) == (2., 4.)
    assert np.allclose(affine_xy(affine_inverse(rotate), 4., 6.), (2., 0.))
    # A mirror in x: the determinant says reflection, and it must be allowed.
    reflect = (-1., 0., 0., 1., 10., 0.)
    assert affine_xy(reflect, 0, 0) == (10., 0.)
    assert affine_xy(reflect, 3, 1) == (7., 1.)
    assert np.allclose(affine_xy(affine_inverse(reflect), 7., 1.), (3., 1.))
    # Pixels that are not square: 0.5 m a column, 1 m a row.
    scale = (.5, 0., 0., 1., .5, .5)
    assert affine_xy(scale, 0, 0) == (.5, .5)
    assert affine_xy(scale, 4, 1) == (2.5, 1.5)
    assert np.allclose(affine_xy(affine_inverse(scale), 2.5, 1.5), (4., 1.))


def test_the_affine_rejects_what_no_image_could_be_placed_by():
    for bad in ((0., 0., 0., 1., 0., 0.),          # degenerate: columns nowhere
                (1., 2., 2., 4., 0., 0.),           # determinant zero
                (1., 0., 0., 1., float('nan'), 0.),
                (1., 0., 0.),                       # not six numbers
                (1., 0., 0., 1., 0.)):
        with pytest.raises(ValueError):
            validate_affine(bad)


def test_the_footprint_is_the_half_pixel_beyond_the_pixels():
    corners = reference(png_bytes(3, 2, {}), (1., 0., 0., 1., 10., -5.)).footprint_m()
    assert np.allclose(corners, ((9.5, -5.5), (12.5, -5.5), (12.5, -3.5), (9.5, -3.5)))


# -- the plane ----------------------------------------------------------------

def test_a_late_reader_discovers_the_one_published_revision():
    plane = instance('late')
    png = png_bytes(2, 2, {(0, 0): RED})
    publisher = ImagePublisher(plane, producer='test')
    try:
        assert publisher.publish('world', png, (1., 0., 0., 1., .5, .5),
                                 source_category=SIMULATION_TRUTH) == 1
        session = consumer(plane)
        try:
            session.poll()
            found = session.latest('world')
            assert found is not None and found.revision == 1
            assert found.affine == (1., 0., 0., 1., .5, .5)
            assert found.png == png
            assert found.checksum == hashlib.sha256(png).hexdigest()
        finally:
            session.close()
    finally:
        publisher.close()


def test_revisions_advance_only_on_change_and_the_old_one_stays_staged():
    plane = instance('revisions')
    publisher = ImagePublisher(plane, producer='test')
    try:
        small = png_bytes(2, 2, {(0, 0): RED})
        assert publisher.publish('world', small, (1., 0., 0., 1., 0., 0.),
                                 source_category='surveyed_plan') == 1
        # The same picture again is a heartbeat, not a new revision.
        assert publisher.publish('world', small, (1., 0., 0., 1., 0., 0.),
                                 source_category='surveyed_plan') == 1
        # A moved image is a new revision.
        assert publisher.publish('world', small, (1., 0., 0., 1., 2., 0.),
                                 source_category='surveyed_plan') == 2
        session = consumer(plane)
        try:
            session.poll()
            state = session.states['world']
            assert state.current.revision == 2
            # §9's display cache: the newest revision and exactly one predecessor.
            assert state.staged is not None and state.staged.revision == 1
        finally:
            session.close()
    finally:
        publisher.close()


def test_withdrawal_takes_the_image_off_the_plane_for_every_reader():
    plane = instance('withdraw')
    publisher = ImagePublisher(plane, producer='test')
    try:
        publisher.publish('world', png_bytes(2, 2, {(0, 0): RED}),
                          (1., 0., 0., 1., 0., 0.), source_category='surveyed_plan')
        session = consumer(plane)
        try:
            session.poll()
            assert session.latest('world') is not None
            assert publisher.withdraw('world') is True
            assert publisher.withdraw('world') is False
            session.poll()
            assert session.latest('world') is None and 'world' not in session.states
            # A late republish is discovered again, wherever it lands.
            publisher.publish('world', png_bytes(2, 2, {(1, 1): GREEN}),
                             (1., 0., 0., 1., 5., -3.), source_category='surveyed_plan')
            session.poll()
            again = session.latest('world')
            assert again is not None and again.affine == (1., 0., 0., 1., 5., -3.)
        finally:
            session.close()
    finally:
        publisher.close()


def test_a_new_producer_is_a_wholesale_rediscovery_and_the_old_picture_lingers():
    plane = instance('restart')
    first = ImagePublisher(plane, producer='test')
    first.publish('world', png_bytes(2, 2, {(0, 0): RED}),
                  (1., 0., 0., 1., 0., 0.), source_category='surveyed_plan')
    session = consumer(plane)
    try:
        session.poll()
        assert session.latest('world') is not None
        # The producer dies: what it published is kept, not flushed -- a reader
        # that lost its picture because a producer departed would blame the wrong
        # thing.
        first.close()
        session.poll()
        assert session.latest('world') is not None
        second = ImagePublisher(plane, producer='test')
        try:
            second.publish('world', png_bytes(2, 2, {(0, 0): GREEN}),
                           (1., 0., 0., 1., 1., 1.), source_category='surveyed_plan')
            session.poll()
            replacement = session.latest('world')
            assert replacement is not None and replacement.affine == (1., 0., 0., 1., 1., 1.)
            assert replacement.metadata['provider_session_id'] == second.session
        finally:
            second.close()
    finally:
        session.close()


def test_only_imagery_matching_the_current_frame_is_displayed():
    plane = instance('frames')
    publisher = ImagePublisher(plane, producer='test')
    try:
        publisher.publish('world', png_bytes(2, 2, {(0, 0): RED}),
                          (1., 0., 0., 1., 0., 0.), source_category='surveyed_plan',
                          localization_epoch=3, clock_epoch=7)
        session = consumer(plane)
        try:
            session.poll()
            here = {'frame_id': 'local', 'localization_epoch': 3, 'clock_epoch': 7}
            assert session.display('world', here) is not None
            assert session.display('world', dict(here, localization_epoch=2)) is None
            assert session.display('world', dict(here, clock_epoch=8)) is None
            assert session.display('world', dict(here, frame_id='surveyed')) is None
            # A display that cannot check had no background, not a lucky one.
            assert session.display('world', {}) is None
            assert session.display('world', None) is None
            # The retention rule hides the image; the image itself is still there.
            assert session.latest('world') is not None
        finally:
            session.close()
    finally:
        publisher.close()


def test_matches_never_accepts_an_absent_context():
    ref = reference(png_bytes(2, 2, {(0, 0): RED}), (1., 0., 0., 1., 0., 0.))
    assert ref.matches({}) is False
    assert ref.matches(None) is False
    assert ref.matches({'frame_id': 'local', 'localization_epoch': 0, 'clock_epoch': 0})


def test_publication_refuses_what_no_consumer_could_show():
    plane = instance('refuse')
    publisher = ImagePublisher(plane, producer='test')
    try:
        with pytest.raises(ValueError, match='encoded'):
            publisher.publish('world', b'\x89PNG\r\n' + b'\0' * (MAX_ENCODED_BYTES + 1),
                             (1., 0., 0., 1., 0., 0.), source_category='surveyed_plan')
        with pytest.raises(ValueError, match='PNG'):
            publisher.publish('world', b'not a png at all',
                             (1., 0., 0., 1., 0., 0.), source_category='surveyed_plan')
        with pytest.raises(ValueError, match='singular'):
            publisher.publish('world', png_bytes(2, 2, {}),
                              (0., 0., 0., 1., 0., 0.), source_category='surveyed_plan')
        assert publisher.entries == {}
    finally:
        publisher.close()


def test_a_plane_holds_a_bounded_number_of_images():
    plane = instance('bound')
    publisher = ImagePublisher(plane, producer='test')
    try:
        small = png_bytes(1, 1, {(0, 0): RED})
        for index in range(MAX_IMAGES):
            publisher.publish(f'plan{index}', small, (1., 0., 0., 1., 0., 0.),
                              source_category='surveyed_plan')
        with pytest.raises(ValueError, match='at most'):
            publisher.publish('one-too-many', small, (1., 0., 0., 1., 0., 0.),
                              source_category='surveyed_plan')
    finally:
        publisher.close()


def test_an_image_that_outgrew_its_own_staging_is_told_to_withdraw():
    plane = instance('staging')
    publisher = ImagePublisher(plane, producer='test')
    try:
        publisher.publish('world', png_bytes(2, 2, {(0, 0): RED}),
                          (1., 0., 0., 1., 0., 0.), source_category='surveyed_plan')
        # The ring was sized from the first revision; a far larger one cannot
        # be quietly staged beside it.
        noise = Image.fromarray(np.random.default_rng(7).integers(
            0, 256, (300, 300, 3), dtype=np.uint8), 'RGB')
        buffer = io.BytesIO(); noise.save(buffer, 'PNG')
        with pytest.raises(ValueError, match='withdraw the image'):
            publisher.publish('world', buffer.getvalue(),
                             (1., 0., 0., 1., 0., 0.), source_category='surveyed_plan')
    finally:
        publisher.close()


# -- lying and torn records ---------------------------------------------------

def test_the_codec_refuses_what_nothing_could_display():
    small = png_bytes(2, 2, {(0, 0): RED})
    base = build_metadata('world', small, (1., 0., 0., 1., 0., 0.), revision=1,
                          provider_session_id='t', source_category='surveyed_plan')
    assert decode_revision(encode_revision(base, small)) == (base, small)
    for field, value, needle in (
            ('schema', 'something.else', 'schema'),
            ('encoding', 'jpeg', 'encoding'),
            ('width', 0, 'width'),
            ('revision', 0, 'revision'),
            ('checksum', '0' * 64, 'checksum'),
            ('affine', [0., 0., 0., 1., 0., 0.], 'singular')):
        with pytest.raises(ValueError, match=needle):
            decode_revision(encode_revision(dict(base, **{field: value}), small))
    with pytest.raises(ValueError, match='decoded'):
        decode_revision(encode_revision(dict(base, width=40000, height=40000), small))
    with pytest.raises(ValueError):                 # a torn payload
        decode_revision(encode_revision(base, small)[:10])


def test_a_consumer_rejects_lying_records_and_keeps_the_last_whole_one():
    plane = instance('lying')
    publisher = ImagePublisher(plane, producer='test')
    try:
        small = png_bytes(2, 2, {(0, 0): RED})
        publisher.publish('world', small, (1., 0., 0., 1., 0., 0.),
                          source_category='surveyed_plan')
        ring = publisher.rings['world']

        def inject(metadata, png, sequence):
            ring.write(encode_record(publisher.session, publisher.generation, 'world',
                                     sequence, sequence, 0, encode_revision(metadata, png),
                                     payload_type=REFERENCE_IMAGE,
                                     reset_epoch=metadata['localization_epoch'],
                                     clock_epoch=metadata['clock_epoch']))

        whole = build_metadata('world', small, (1., 0., 0., 1., 0., 0.), revision=2,
                               provider_session_id=publisher.session,
                               source_category='surveyed_plan')
        # A checksum that is not the bytes'.
        inject(dict(whole, checksum='0' * 64), small, 2)
        # A revision that disagrees with the record carrying it.
        inject(dict(whole, revision=5), small, 3)
        # An image that names another image.
        inject(dict(whole, image_id='other'), small, 4)
        # A payload that is not an envelope at all.
        ring.write(encode_record(publisher.session, publisher.generation, 'world',
                                 5, 5, 0, b'\x00\x01not-json',
                                 payload_type=REFERENCE_IMAGE, reset_epoch=0, clock_epoch=0))
        session = consumer(plane)
        try:
            session.poll()
            kept = session.latest('world')
            assert kept is not None and kept.revision == 1
            reasons = ' | '.join(reason for _, reason in session.rejections).lower()
            for needle in ('checksum', 'disagrees with the record sequence',
                           'names image', 'metadata bytes'):
                assert needle in reasons, reasons
            assert session.states['world'].rejected == 4
        finally:
            session.close()
    finally:
        publisher.close()


# -- registration under every tested transform --------------------------------

def expected_plane(rows):
    """A ``(height, width, 3)`` plane from per-row lists of RGB colours."""
    return np.array([[list(colour) for colour in row] for row in rows], np.uint8)


GROUND = [theme.rgb(theme.CANVAS)] * 4


def test_a_translation_places_pixels_under_cells_and_clips_beyond():
    ref = reference(png_bytes(2, 1, {(0, 0): RED, (1, 0): GREEN}), (1., 0., 0., 1., .5, .5))
    plane = background_raster(Background(ref, opacity=1.), GridGeometry(0., 0., 1., 4, 2))
    assert np.array_equal(plane, expected_plane([
        [list(RED[:3]), list(GREEN[:3])] + GROUND[:2],
        GROUND]))


def test_a_rotation_places_columns_south_and_rows_west():
    # x = -row + 3.5, y = col + 0.5: pixel (0,0) lands at cell (3,0), pixel
    # (1,0) one row below it. The whole rest of the grid stays ground.
    ref = reference(png_bytes(2, 1, {(0, 0): RED, (1, 0): GREEN}),
                    (0., -1., 1., 0., 3.5, .5))
    plane = background_raster(Background(ref, opacity=1.), GridGeometry(0., 0., 1., 4, 2))
    assert np.array_equal(plane, expected_plane([
        GROUND[:3] + [list(RED[:3])],
        GROUND[:3] + [list(GREEN[:3])]]))


def test_a_reflection_mirrors_the_columns():
    ref = reference(png_bytes(2, 1, {(0, 0): RED, (1, 0): GREEN}), (-1., 0., 0., 1., 1.5, .5))
    plane = background_raster(Background(ref, opacity=1.), GridGeometry(0., 0., 1., 4, 2))
    assert np.array_equal(plane, expected_plane([
        [list(GREEN[:3]), list(RED[:3])] + GROUND[:2],
        GROUND]))


def test_unequal_pixel_scales_sample_nearest_and_clip_at_the_footprint():
    # 0.5 m a column, 1 m a row: cells 0 and 1 cover two pixels each, the rest
    # of the row and all of the row below are outside the footprint.
    ref = reference(png_bytes(4, 1, {(0, 0): RED, (2, 0): GREEN}),
                    (.5, 0., 0., 1., .5, .5))
    plane = background_raster(Background(ref, opacity=1.), GridGeometry(0., 0., 1., 4, 2))
    assert np.array_equal(plane, expected_plane([
        [list(RED[:3]), list(GREEN[:3])] + GROUND[:2],
        GROUND]))


def test_opacity_blends_the_reference_over_the_ground_and_neither_extreme_hides_it():
    ref = reference(png_bytes(2, 1, {(0, 0): RED, (1, 0): GREEN}), (1., 0., 0., 1., .5, .5))
    geometry = GridGeometry(0., 0., 1., 2, 1)
    full = background_raster(Background(ref, opacity=1.), geometry)
    assert np.array_equal(full[0], np.array([RED[:3], GREEN[:3]], np.uint8))
    absent = background_raster(Background(ref, opacity=0.), geometry)
    assert np.array_equal(absent[0, 0], theme.rgb(theme.CANVAS))
    half = background_raster(Background(ref, opacity=.5), geometry)
    ground = np.array(theme.rgb(theme.CANVAS), np.float32)
    blended = np.round(np.array(RED[:3], np.float32) * .5 + ground * .5).astype(np.uint8)
    assert np.allclose(half[0, 0], blended, atol=1) and not np.array_equal(half[0, 0], full[0, 0])


def test_the_background_appears_only_where_nobody_has_looked():
    grid = sample_grid(width=4, height=1, observed=[(0, 0)])
    ref = reference(png_bytes(4, 1, {(0, 0): RED, (1, 0): GREEN, (2, 0): RED, (3, 0): GREEN}),
                    (1., 0., 0., 1., .5, .5))
    background = Background(ref, opacity=1.)
    raster = render_raster(grid, mode='occupancy')
    composed = compose_background(raster, background, grid.geometry,
                                  never_cells(grid, mode='occupancy'))
    # The measured cell keeps its measurement; the unmeasured ones show the picture.
    assert np.array_equal(composed[0, 0], raster[0, 0])
    assert tuple(composed[0, 1]) == GREEN[:3]
    # The never mode is the sentinel and a background may not erase it.
    assert never_cells(grid, mode='never') is None


def test_the_registration_marks_are_derived_from_the_affine():
    ref = reference(png_bytes(3, 2, {}), (1., 0., 0., 1., 10., -5.))
    shapes = list(background_shapes(Background(ref), cell_m=.5))
    assert len(shapes) == 6
    corners = ref.footprint_m()
    for shape, (start, end) in zip(shapes[:4], zip(corners, corners[1:] + corners[:1])):
        assert (shape.x0, shape.y0, shape.x1, shape.y1) == start + end
    # The axis L: columns in the accent colour, rows dimmed, from the first corner.
    assert shapes[4].outline == theme.ACCENT and shapes[5].outline == theme.DIM
    assert (shapes[4].x0, shapes[4].y0) == corners[0] == (shapes[5].x0, shapes[5].y0)
    assert shapes[4].x1 > shapes[4].x0 and shapes[5].y1 > shapes[5].y0


def test_the_pane_and_the_report_paint_one_picture():
    """The same description, painted twice: the widget and the PNG must agree."""
    grid = sample_grid(cell_m=1., width=4, height=2, observed=[(0, 0)])
    ref = reference(png_bytes(4, 2, {(1, 1): RED, (2, 0): GREEN}),
                    (1., 0., 0., 1., .5, .5))
    background = Background(ref, opacity=.8)
    overlay = Overlay(route=((0., 0.), (3., 1.)), vehicle=(0., 0., 90.), goal=(3., 1.))
    with hidden_tk() as root:
        pane = MapPane(root, width=240, height=160)
        pane.set_grid(grid, sim_now_s=10.)
        pane.set_overlay(route=overlay.route, vehicle=overlay.vehicle, goal=overlay.goal)
        pane.set_background(background)
        assert pane.refresh(force=True)
        from_pane = pane.snapshot()
    from_report = snapshot_image(grid, sim_now_s=10., overlay=overlay,
                                 background=background, cell_px=SNAPSHOT_CELL_PX)
    assert np.array_equal(np.asarray(from_pane), np.asarray(from_report))


# -- the host, and what it shows ------------------------------------------------

def test_the_host_shows_only_imagery_matching_the_current_frame(tmp_path):
    plane = instance('host')
    provider = FixtureProvider(plane, tmp_path/'session', sensors=('scan',))
    try:
        provider.publish_reference_image()
        with hidden_tk() as root:
            host = ReferenceBackgroundHost(root, plane, enabled=True)
            try:
                background = host.background()
                assert background is not None and background.image.image_id == 'world'
                assert background.label().startswith('reference only')
                # The toggle is a display decision the run has to hear about.
                heard = []
                host.on_change = lambda enabled, opacity: heard.append((enabled, opacity))
                host.var.set(False); host._toggled()
                assert heard == [(False, host.opacity)]
                assert host.background() is None            # off means off
                host.var.set(True); host._toggled()
                # The frame moves on; the picture does not follow it (§7).
                provider.announce_localization_reset(); provider.step()
                assert host.background() is None
            finally:
                host.close()
    finally:
        provider.close()


# -- provenance ----------------------------------------------------------------

def test_dnav_and_dalg_record_display_decisions_in_their_provenance():
    from dalg.profiles import load_profiles
    from dalg.run import DalgRun
    from dnav.plan import NavRun
    from dnav.policy import load_policy
    run = NavRun(instance('provenance'), ROOT, policy=load_policy('default', ROOT))
    run.note_display('reference-on', source='--show-reference')
    assert run.summary()['reference_display'] == dict(mode='reference-on',
                                                      source='--show-reference')
    assert any(event['type'] == 'display.reference' for event in run._events)
    run.close()

    alg = DalgRun(instance('provenance'), load_profiles(['lidar-baseline'], ROOT), ROOT)
    alg.note_display('reference-off', opacity=.8)
    assert alg.provenance['reference_display'] == dict(mode='reference-off', opacity=.8)
    alg.close()


# -- the archive: the exact revision a report used ------------------------------

def attempt_event(grid):
    return dict(schema='dvision2.planning-attempt.v1',
                pose={'x_m': 1., 'y_m': 1., 'z_m': 0., 'heading_deg': 90.},
                goal={'position': [3., 1.]},
                policy={'name': 'p', 'occupied_threshold': 100, 'inflation_m': .5,
                        'combine': 'max', 'schema_version': 1, 'digest': 'd'},
                planner='astar',
                route=dict(status='ok', reason='', planner='astar', waypoints=[
                    dict(x=1., y=1., z=0.), dict(x=3., y=1., z=0.)], cost=2.,
                    length_m=2., map_revision=1, policy_digest='d', plan_time_s=.001,
                    expanded=4, sim_time_s=10., goal=[3., 1., 0.], start=[1., 1., 0.],
                    diagnostics={}),
                inputs=dict(time_s=10., sources=['cam'], stale=[],
                            revisions={'cam': 1}))


def _record_attempt(recorder, grid, image=None):
    recorder.record('planning.attempt', attempt_event(grid), grids={'cam': grid})
    if image is None: return
    time.sleep(.5)             # the attempt commits on its own chunk, so losing
    recorder.record('report.background',    # the image can never cost the numbers
                    dict(time_s=10., image_id=image.image_id, revision=image.revision,
                         checksum=image.checksum, opacity=.8,
                         label='reference only · simulation_truth',
                         source_category=image.source_category),
                    images={'reference': image})


def test_a_report_background_round_trips_by_checksum(tmp_path):
    grid = sample_grid(cell_m=1., width=4, height=2, observed=[(0, 0)])
    ref = reference(png_bytes(4, 2, {(1, 1): RED}), (1., 0., 0., 1., .5, .5))
    recorder = Recorder(next_archive_dir(tmp_path/'dnav'), {'module': 'dnav'}, module='dnav')
    _record_attempt(recorder, grid, ref)
    assert recorder.close()['complete'] is True
    reader = ArchiveReader(recorder.directory)
    assert reader.validate()['complete'] is True
    archived = reader.image('reference', reader.events()[-1]['images']['reference'])
    assert archived.checksum == ref.checksum and archived.png == ref.png
    assert archived.affine == ref.affine and archived.metadata == ref.metadata


def test_the_standalone_render_registers_exactly_as_the_live_one(tmp_path):
    grid = sample_grid(cell_m=1., width=4, height=2, observed=[(0, 0)])
    ref = reference(png_bytes(4, 2, {(1, 1): RED, (2, 0): GREEN}),
                    (1., 0., 0., 1., .5, .5))
    background = Background(ref, opacity=.8)
    overlay = Overlay(route=((1., 1.), (3., 1.)), vehicle=(1., 1., 90.),
                      goal=(3., 1.), start=(1., 1.), inflation_m=.5)
    live = snapshot_image(grid, sim_now_s=10., overlay=overlay, background=background)
    bare = snapshot_image(grid, sim_now_s=10., overlay=overlay)
    recorder = Recorder(next_archive_dir(tmp_path/'dnav'), {'module': 'dnav'},
                        module='dnav', commit_s=.2)
    _record_attempt(recorder, grid, ref)
    assert recorder.close()['complete'] is True
    reader = ArchiveReader(recorder.directory)
    with_image = Image.open(render_attempt(reader, 1, tmp_path/'with.png'))
    assert np.array_equal(np.asarray(with_image), np.asarray(live))
    # The image is optional: losing it costs the background, not the numbers.
    blob_chunk = reader.events()[-1]['images']['reference']['chunk']
    (recorder.directory/blob_chunk).unlink()
    without_image = Image.open(render_attempt(ArchiveReader(recorder.directory), 1,
                                              tmp_path/'without.png'))
    assert np.array_equal(np.asarray(without_image), np.asarray(bare))


def test_displayed_background_takes_the_newest_event_even_when_its_bytes_are_gone(tmp_path):
    first = reference(png_bytes(4, 2, {(1, 1): RED}), (1., 0., 0., 1., .5, .5))
    second = reference(png_bytes(4, 2, {(1, 1): GREEN}), (1., 0., 0., 1., .5, .5))
    recorder = Recorder(next_archive_dir(tmp_path/'dnav'), {'module': 'dnav'},
                        module='dnav', commit_s=.1)
    recorder.record('report.background', dict(opacity=.8, checksum=first.checksum),
                    images={'reference': first})
    time.sleep(.3)                                  # two revisions, two chunks
    recorder.record('report.background', dict(opacity=.8, checksum=second.checksum),
                    images={'reference': second})
    assert recorder.close()['complete'] is True
    reader = ArchiveReader(recorder.directory)
    found, opacity = displayed_background(reader)
    assert found is not None and found.checksum == second.checksum and opacity == .8
    newest_chunk = reader.events()[-1]['images']['reference']['chunk']
    (recorder.directory/newest_chunk).unlink()
    gone, _ = displayed_background(ArchiveReader(recorder.directory))
    # The newest revision is the one the report used; an older picture would
    # misregister, so absence wins over the wrong image.
    assert gone is None


def test_digest_image_refuses_bytes_that_lie_about_their_checksum():
    ref = reference(png_bytes(2, 2, {(0, 0): RED}), (1., 0., 0., 1., 0., 0.))
    lying = ReferenceImage(image_id=ref.image_id, revision=ref.revision, metadata=ref.metadata,
                           png=png_bytes(2, 2, {(0, 0): GREEN}), width=ref.width,
                           height=ref.height, affine=ref.affine, image=ref.image)
    with pytest.raises(ValueError, match='checksum'):
        digest_image(lying)


# -- three backgrounds, one outcome --------------------------------------------

class NumericRig:
    """A provider, a dalg run and a dnav run on one instance, for numeric identity."""

    def __init__(self, tmp_path, name, *, imagery=None):
        from dalg.profiles import load_profiles
        from dalg.run import DalgRun
        from dnav.plan import NavRun
        from dnav.policy import load_policy
        self.instance = instance(name)
        self.provider = FixtureProvider(self.instance, tmp_path/name/'session',
                                        sensors=('scan',))
        if imagery is not None: imagery(self.provider)
        self.run = DalgRun(self.instance, load_profiles(['lidar-baseline'], ROOT), ROOT)
        self.nav = NavRun(self.instance, ROOT, policy=load_policy('default', ROOT),
                          goal=(8., 2.))
        # Discovery backoff is wall-clock; these loops run far faster than real time.
        self.nav.session.probe_interval_s = self.nav.session.max_probe_interval_s = 0.
        if self.nav.recorder is not None:
            self.nav.recorder.commit_s = .01   # commits keep up with these fast ticks

    def tick(self, count=1):
        for _ in range(count):
            self.provider.step()
            self.run.sensor_session.probe.last_probe = -1e9
            self.run.step()
            self.nav.step()

    def outcome(self):
        """The evidence and the route at the end, stripped of wall-clock timing.

        The planes are compared as exact bytes rather than through
        ``digest_grid``, whose metadata names this instance's channels and
        epochs -- identities that rightly differ between two runs. What must
        not differ is the evidence itself: revision, status, geometry and
        every occupancy and timestamp byte.
        """
        self.nav.replan(force=True)
        grids = {}
        for sid, grid in self.run.evidence_grids().items():
            grids[sid] = (int(grid.revision), int(grid.status), grid.geometry.as_dict(),
                          np.ascontiguousarray(grid.occupancy, np.uint8).tobytes(),
                          np.ascontiguousarray(grid.observed_ms, np.uint32).tobytes())
        route = self.nav.route.as_dict()
        route.pop('plan_time_s')
        return grids, route

    def last_attempt(self):
        """The forced final attempt's route, as the archive recorded it."""
        self.nav.replan(force=True)
        assert self.nav.recorder is not None
        wanted = self.nav.attempts
        deadline = time.monotonic() + 5.      # the commit thread is only milliseconds
        mine = []                            # behind at commit_s = .01
        while not mine and time.monotonic() < deadline:
            reader = ArchiveReader(self.nav.recorder.directory)
            mine = [event for event in reader.events()
                    if event['type'] == 'planning.attempt'
                    and event['data'].get('attempt') == wanted]
            if not mine: time.sleep(.05)
        assert mine, 'the final planning attempt never committed'
        data = mine[-1]['data']
        route = data['route']
        return (route['status'], tuple((w['x'], w['y'], w['z']) for w in route['waypoints']),
                route['cost'], route['length_m'],
                tuple(sorted(data['inputs']['revisions'].items())))

    def close(self):
        self.nav.close(); self.run.close(); self.provider.close()


def _covers(ref, x, y):
    corners = ref.footprint_m()
    xs = [corner[0] for corner in corners]; ys = [corner[1] for corner in corners]
    return min(xs) <= x <= max(xs) and min(ys) <= y <= max(ys)


def test_evidence_and_routes_are_identical_whatever_the_background(tmp_path):
    """None, a correct plan and a lying plan behind the operator: one outcome."""
    from dnav import route as R

    def misleading(provider):
        png, affine = provider.plan_png()
        # The true plan, placed forty metres away from where anything is.
        provider.publish_reference_image(png=png, affine=(affine[0], affine[1], affine[2],
                                                          affine[3], affine[4] + 40.,
                                                          affine[5] + 30.))

    conditions = {}
    for name, imagery in (('none', None),
                          ('correct', lambda provider: provider.publish_reference_image()),
                          ('misleading', misleading)):
        rig = NumericRig(tmp_path, name, imagery=imagery)
        try:
            if imagery is not None:
                # The condition must actually be live on the plane, or the
                # comparison would prove nothing by comparing two nothings.
                session = consumer(rig.instance)
                try:
                    session.poll()
                    latest = session.latest('world')
                    assert latest is not None, 'the published image never appeared'
                    assert _covers(latest, 8., 2.) == (name == 'correct')
                finally:
                    session.close()
            rig.provider.fly([(2., 0.), (2., 2.5)])
            rig.tick(120)
            outcome = rig.outcome()
            assert rig.nav.route.status == R.OK, rig.nav.route.reason
            assert rig.run.evidence_grids()
            conditions[name] = (outcome, rig.last_attempt())
            # A headless run never displays anything, and says so.
            assert rig.nav.summary()['reference_display'] is None
            assert rig.nav.summary()['reference_image'] is None
        finally:
            rig.close()
    baseline = conditions['none']
    for name in ('correct', 'misleading'):
        assert conditions[name] == baseline, f'a {name} background changed the numbers'
