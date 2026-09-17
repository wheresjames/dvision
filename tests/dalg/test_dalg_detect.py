"""Target detectors as pure parts: red_disc, projection, estimates, profiles and the detections plane.

The ``red_disc`` cases are ported from ``tests/daic/test_daic_detector.py`` onto
the copy under ``dalg.detect``, on the same synthetic frames, so the copy
provably finds what daic found; the new part is that every rejected blob says
why. Projection is checked against rays computed by hand, including the
camera's pitch and mount height. Nothing here imports daic.
"""
from __future__ import annotations

import json
import math
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import cv2
import numpy as np
import pytest

from dalg import detect as DET
from dalg.detect import DetectorResult, PixelDetection, split_settings
from dalg.detect.estimate import Estimates
from dalg.detect.projection import inputs, project, replay
from dalg.detect.red_disc import detect
from dalg.model import Intrinsics, Pose
from dalg.profiles import Detector, load_profile, load_profiles, require_evidence, validate_sources
from dcmn import detections as D
from dcmn.navigation import encode

ROOT = Path(__file__).resolve().parents[2]
#: daic's controller threshold the ported cases compared against.
MIN_CONFIDENCE = 0.3


def frame(h=480, w=640):
    return np.zeros((h, w, 3), dtype=np.uint8)


def red_circle(image, cx, cy, r, colour=(220, 20, 20)):
    cv2.circle(image, (cx, cy), r, colour, -1)


def best(image):
    result = detect(image)
    return result.detections[0] if result.detections else None


# -- red_disc: the ported cases ------------------------------------------------------------

def test_empty_and_missing_frames_find_nothing():
    assert best(frame(240, 320)) is None
    assert detect(None) == DetectorResult()


def test_a_red_circle_is_found_where_it_is_with_its_radius():
    image = frame()
    red_circle(image, 320, 240, 60)
    found = best(image)
    assert found is not None and found.label == 'landing_pad'
    assert found.confidence > MIN_CONFIDENCE
    assert abs(found.cx_px - 320) < 15 and abs(found.cy_px - 240) < 15
    assert abs(found.extent_px - 60) < 15


def test_a_white_centre_and_a_small_edge_artifact_do_not_move_the_centre():
    image = frame()
    red_circle(image, 320, 240, 70)
    cv2.circle(image, (320, 240), 16, (255, 255, 255), -1)
    found = best(image)
    assert abs(found.cx_px - 320) < 3 and abs(found.cy_px - 240) < 3 and abs(found.extent_px - 70) < 5
    red_circle(image, 402, 240, 5)
    found = best(image)
    assert abs(found.cx_px - 320) < 5 and abs(found.cy_px - 240) < 3


def test_the_larger_of_two_circles_is_chosen_and_the_other_says_why_not():
    image = frame()
    red_circle(image, 150, 240, 20)
    red_circle(image, 450, 240, 60)
    result = detect(image)
    assert abs(result.detections[0].cx_px - 450) < abs(result.detections[0].cx_px - 150)
    assert any('below the chosen blob' in c.reason and abs(c.cx_px - 150) < 5 for c in result.candidates)


@pytest.mark.parametrize('colour', [(20, 20, 220), (155, 100, 60), (230, 80, 20), (90, 55, 50)])
def test_blue_brick_orange_and_desaturated_red_are_not_pads(colour):
    image = frame()
    red_circle(image, 320, 240, 80, colour)
    assert best(image) is None


def test_a_shadowed_saturated_pad_and_a_wide_exposure_range_are_still_pads():
    image = frame()
    red_circle(image, 320, 240, 80, (90, 15, 15))
    assert best(image) is not None
    for scale in (0.25, 0.4, 0.6, 1.0, 1.4):
        image = frame()
        red_circle(image, 320, 240, 70, tuple(int(min(255, c * scale)) for c in (230, 25, 20)))
        assert best(image) is not None, scale


def test_an_oblique_pad_is_found_as_an_ellipse():
    for aspect in (2, 4, 8):
        image = frame()
        cv2.ellipse(image, (320, 240), (160, max(160 // aspect, 4)), 0, 0, 360, (220, 20, 20), -1)
        found = best(image)
        assert found is not None and abs(found.cx_px - 320) < 6 and abs(found.cy_px - 240) < 6
        assert found.diagnostics['aspect'] == pytest.approx(aspect, rel=0.2)


# -- red_disc: rejection reasons -------------------------------------------------------------

def test_a_ragged_star_is_rejected_for_its_solidity():
    image = frame()
    points = [[int(320 + (110 if i % 2 == 0 else 26) * math.cos(math.tau * i / 40)),
               int(240 + (110 if i % 2 == 0 else 26) * math.sin(math.tau * i / 40))] for i in range(40)]
    cv2.fillPoly(image, [np.array(points, dtype=np.int32)], (220, 20, 20))
    result = detect(image)
    assert not result.detections
    assert any(c.reason.startswith(('solidity', 'extent')) for c in result.candidates), result.candidates


def test_a_tiny_blob_and_an_edge_on_line_say_area_and_aspect():
    image = frame()
    red_circle(image, 100, 100, 2)
    cv2.ellipse(image, (400, 300), (200, 6), 0, 0, 360, (220, 20, 20), -1)
    reasons = [c.reason for c in detect(image).candidates]
    assert any(r.startswith('area') and '< 25' in r for r in reasons), reasons
    assert any(r.startswith('aspect') and '> 14' in r for r in reasons), reasons


# -- the registry ------------------------------------------------------------------------------

def test_a_new_detector_is_one_module_and_one_registry_line(monkeypatch):
    @dataclass(frozen=True)
    class StubSettings:
        colour: str = 'green'

    class Stub:
        name: ClassVar[str] = 'stub'
        labels: ClassVar[tuple[str, ...]] = ('thing',)
        sensor_type: ClassVar[str] = 'camera.rgb'
        settings_class: ClassVar[type | None] = StubSettings

        def __init__(self, config, intrinsics): self.config = config
        def detect(self, frame): return DetectorResult((PixelDetection('thing', 1., 2., 3., .5),))

    monkeypatch.setitem(DET.DETECTORS, 'stub', Stub)
    instance, common = DET.build('stub', dict(hz=2, colour='blue'), None)
    assert isinstance(instance, DET.Detector) and instance.config.colour == 'blue' and common.hz == 2
    row = Detector('primary_camera', 'stub', dict(colour='red'))
    validate_sources((), detectors=[row])
    assert row.labels == ('thing',)
    with pytest.raises(ValueError, match='unknown stub settings: shade'):
        split_settings('stub', dict(shade=1))


def test_common_settings_are_checked_by_name():
    for settings, message in ((dict(hz=0), 'hz'), (dict(confirm_n=0), 'confirm_n'), (dict(crop_px=4), 'crop_px'),
                              (dict(bogus=1), 'unknown red_disc settings: bogus')):
        with pytest.raises(ValueError, match=message):
            split_settings('red_disc', settings)
    with pytest.raises(ValueError, match='unsupported detector'):
        split_settings('blue_square', {})


# -- projection ------------------------------------------------------------------------------------

INTRINSICS = Intrinsics(160, 120, 100.0, 100.0, 80.0, 60.0)


def test_the_centre_pixel_of_a_pitched_camera_lands_at_height_over_tan_pitch():
    camera = Pose(2.0, 3.0, 1.6, 90.0, 0.0, -8.0)  # facing east, 8 degrees down, lens 1.6 m up
    result = project(80.0, 60.0, 10.0, camera, INTRINSICS)
    expected = 1.6 / math.tan(math.radians(8.0))
    assert result.projectable
    assert result.xy_m == pytest.approx((2.0 + expected, 3.0), abs=1e-9)
    assert result.range_m == pytest.approx(expected) and result.bearing_deg == pytest.approx(90.0)


def test_a_pixel_below_centre_is_nearer_and_one_to_the_side_turns_the_bearing():
    camera = Pose(0.0, 0.0, 2.0, 0.0, 0.0, 0.0)  # facing north (-y), level
    below = project(80.0, 60.0 + 100.0, 1.0, camera, INTRINSICS)  # 45 degrees down: 2 m ahead
    assert below.xy_m == pytest.approx((0.0, -2.0), abs=1e-9) and below.range_m == pytest.approx(2.0)
    right = project(80.0 + 100.0, 60.0 + 100.0, 1.0, camera, INTRINSICS)  # also 45 degrees right
    assert right.xy_m == pytest.approx((2.0, -2.0), abs=1e-9) and right.bearing_deg == pytest.approx(45.0)
    assert 0.0 < below.sigma_m < right.sigma_m, 'the further point is the less certain one'


def test_at_the_horizon_beyond_range_and_without_a_pose_are_not_projectable_with_reasons():
    camera = Pose(0.0, 0.0, 2.0, 0.0, 0.0, 0.0)
    assert project(80.0, 60.0, 1.0, camera, INTRINSICS).reason == 'at or above the horizon'
    assert project(80.0, 10.0, 1.0, camera, INTRINSICS).reason == 'at or above the horizon'
    far = project(80.0, 62.0, 1.0, camera, INTRINSICS, max_range_m=25.0)
    assert not far.projectable and far.reason == 'range 100.0 m > 25 m'
    assert project(80.0, 90.0, 1.0, None, INTRINSICS).reason == 'no camera pose with the frame'


def test_a_recorded_projection_replays_to_the_same_answer():
    camera = Pose(1.5, -2.0, 1.7, 33.0, 0.0, -12.0)
    recorded = json.loads(json.dumps(inputs(97.3, 88.1, 6.0, camera, INTRINSICS, z0_m=0.1, max_range_m=20.0)))
    assert replay(recorded) == project(97.3, 88.1, 6.0, camera, INTRINSICS, z0_m=0.1, max_range_m=20.0)


# -- estimates ---------------------------------------------------------------------------------------

def test_detections_within_the_gate_join_one_estimate_that_confirms_at_confirm_n():
    estimates = Estimates(gate_m=1.0, confirm_n=3, expire_s=10.0)
    first, events = estimates.associate('pad', 'cam-red', (4.0, 0.0), 1.0, 1.0, 'D1')
    assert [k for k, _ in events] == ['opened'] and first.id == 'pad-1'
    estimates.associate('pad', 'cam-red', (4.2, 0.0), 1.0, 2.0, 'D2')
    same, events = estimates.associate('pad', 'cam-red', (4.4, 0.0), 1.0, 3.0, 'D3')
    assert same is first and [k for k, _ in events] == ['updated', 'confirmed']
    assert first.xy_m == pytest.approx((4.2, 0.0)) and first.count == 3 and first.revision == 3
    assert first.spread_m == pytest.approx(math.sqrt(2 * 0.2 ** 2 / 3))
    other, events = estimates.associate('pad', 'cam-red', (9.0, 0.0), 1.0, 3.0, 'D4')
    assert other.id == 'pad-2' and [k for k, _ in events] == ['opened']


def test_confidence_weights_the_mean_and_labels_never_mix():
    estimates = Estimates(gate_m=5.0, confirm_n=2, expire_s=10.0)
    estimate, _ = estimates.associate('pad', 'c', (0.0, 0.0), 0.9, 0.0)
    estimates.associate('pad', 'c', (3.0, 0.0), 0.1, 1.0)
    assert estimate.xy_m == pytest.approx((0.3, 0.0))
    person, _ = estimates.associate('person', 'c', (0.5, 0.0), 1.0, 1.0)
    assert person is not estimate and person.id == 'person-1'


def test_unseen_estimates_go_stale_are_revived_by_a_sighting_and_an_epoch_clears_them():
    estimates = Estimates(gate_m=1.0, confirm_n=1, expire_s=5.0)
    estimate, _ = estimates.associate('pad', 'c', (1.0, 1.0), 1.0, 0.0)
    assert estimates.expire(4.9) == []
    events = estimates.expire(5.0)
    assert [k for k, _ in events] == ['stale'] and estimate.stale
    assert estimates.expire(20.0) == [], 'stale once, not every step'
    _, events = estimates.associate('pad', 'c', (1.1, 1.0), 1.0, 21.0)
    assert not estimate.stale and events[0][1]['revived'] is True
    cleared = estimates.clear('mapping reset')
    assert [(k, d['cause']) for k, d in cleared] == [('cleared', 'mapping reset')] and not estimates.items
    assert estimates.associate('pad', 'c', (1.0, 1.0), 1.0, 22.0)[0].id == 'pad-1', 'IDs restart with the epoch'


# -- profiles ---------------------------------------------------------------------------------------

def test_a_detector_profile_composes_beside_evidence_with_its_own_digest():
    detector = load_profile('red-disc', ROOT)
    assert not detector.sources and [d.id for d in detector.detectors] == ['primary_camera-red_disc']
    with pytest.raises(ValueError, match='detectors run beside evidence'):
        require_evidence(detector)
    composed = load_profiles(['lidar-baseline', 'red-disc'], ROOT)
    require_evidence(composed)
    assert composed.detectors == detector.detectors and len(composed.sources) == 1
    assert composed.digest != load_profile('lidar-baseline', ROOT).digest
    assert [c['name'] for c in composed.components] == ['lidar-baseline', 'red-disc']
    assert composed.detectors[0].labels == ('landing_pad',)


def test_detector_rows_are_validated_like_sources(tmp_path):
    def profile(rows):
        path = tmp_path / f'{uuid.uuid4().hex}.json'
        path.write_text(json.dumps(dict(name='t', sources=[], detectors=rows)))
        return path
    for rows, message in (([dict(sensor='front', detector='nope')], 'unsupported detector'),
                          ([dict(sensor='front', detector='red_disc', settings=dict(hz=-1))], 'hz'),
                          ([dict(sensor='front', detector='red_disc'), dict(sensor='front', detector='red_disc')],
                           'duplicate'),
                          ([dict(sensor='front', detector='red_disc', colour='red')], 'only sensor')):
        with pytest.raises(ValueError, match=message):
            load_profile(str(profile(rows)), ROOT)
    twins = load_profile(str(profile([dict(sensor='front', detector='red_disc'),
                                      dict(sensor='front', detector='red_disc', suffix='slow',
                                           settings=dict(hz=1))])), ROOT)
    assert [d.id for d in twins.detectors] == ['front-red_disc', 'front-red_disc-slow']
    assert twins.detectors[0].settings_digest != twins.detectors[1].settings_digest
    manifest = dict(sensors=dict(scan=dict(type='lidar.scan2d')))
    with pytest.raises(ValueError, match='needs camera.rgb'):
        validate_sources((), manifest, detectors=[Detector('scan', 'red_disc')])


def test_the_readiness_selector_matches_detector_names_and_labels():
    from dalg.run import matches_prepare
    composed = load_profiles(['lidar-baseline', 'red-disc'], ROOT)
    for selector in ('algorithm:red_disc', 'algorithm:landing_pad', 'algorithm:lidar_inverse'):
        assert matches_prepare(composed, [selector])
    assert not matches_prepare(load_profile('lidar-baseline', ROOT), ['algorithm:landing_pad'])


# -- the detections plane ---------------------------------------------------------------------------

def largest_snapshot():
    detector = dict(id='d' * D.MAX_TEXT, name='n' * D.MAX_TEXT, sensor='s' * D.MAX_TEXT,
                    labels=['l' * D.MAX_TEXT] * D.MAX_LABELS, settings_digest='x' * 64, state='running',
                    reason='r' * D.MAX_TEXT, processed=10 ** 9, skipped=10 ** 9, failed=10 ** 9, hz=12345.678,
                    configured_hz=5.0, latency_ms_p50=1.2345, latency_ms_p95=9.8765, last_frame_s=123456.789)
    detection = dict(id='D' * D.MAX_TEXT, detector='d' * D.MAX_TEXT, label='l' * D.MAX_TEXT, sequence=10 ** 9,
                     capture_time_s=123456.789012, cx_px=1234.5678, cy_px=1234.5678, extent_px=123.456789,
                     confidence=0.123456789, projectable=True, xy_m=[-12345.678912, 12345.678912],
                     range_m=24.99999999, bearing_deg=359.999999, sigma_m=12.3456789, estimate='e' * D.MAX_TEXT,
                     reason='')
    estimate = dict(id='e' * D.MAX_TEXT, label='l' * D.MAX_TEXT, detector='d' * D.MAX_TEXT,
                    xy_m=[-12345.678912, 12345.678912], spread_m=1.23456789, count=10 ** 9, revision=10 ** 9,
                    confirmed=True, stale=False, first_s=123456.789, last_s=123456.789, confidence=0.123456789)
    return dict(D.empty('v' * D.MAX_TEXT, 'p' * D.MAX_TEXT, 's' * D.MAX_TEXT, 123456.789,
                        context=dict(frame_id='local', localization_epoch=10 ** 9, clock_domain_id='c' * 64,
                                     clock_epoch=2 ** 62, mapping_epoch=10 ** 9)),
                revision=10 ** 9, detectors=[detector] * D.MAX_DETECTORS,
                detections=[detection] * D.MAX_DETECTIONS, estimates=[estimate] * D.MAX_ESTIMATES)


def test_the_largest_legal_detections_snapshot_fits_and_one_more_row_is_refused():
    value = D.validate(largest_snapshot())
    assert len(encode(value).encode()) <= 65536
    for key, cap in (('detectors', D.MAX_DETECTORS), ('detections', D.MAX_DETECTIONS),
                     ('estimates', D.MAX_ESTIMATES)):
        bigger = largest_snapshot()
        bigger[key] = bigger[key] + bigger[key][:1]
        with pytest.raises(ValueError, match=f'at most {cap}'):
            D.validate(bigger)


def test_malformed_rows_say_which_field_and_a_plane_round_trips():
    for change, message in ((dict(schema='x'), 'schema'), (dict(revision=-1), 'revision'),
                            (dict(detections=[dict(largest_snapshot()['detections'][0], projectable=False,
                                                   reason='')]), 'must say why'),
                            (dict(estimates=[dict(largest_snapshot()['estimates'][0], confirmed='yes')]),
                             'confirmed')):
        with pytest.raises(ValueError, match=message):
            D.validate(dict(largest_snapshot(), **change))
    vehicle = 'det-' + uuid.uuid4().hex[:8]
    publisher, reader = D.DetectionsPublisher(vehicle), D.DetectionsReader(vehicle)
    try:
        assert reader.read() == {} and reader.name == f'/dvision2.{vehicle}.detections.dalg'
        publisher.write(dict(D.empty(vehicle, 'dalg', 'session'), revision=3))
        assert reader.read()['revision'] == 3
        with pytest.raises(ValueError, match='live writer'):
            D.DetectionsPublisher(vehicle).start()
    finally:
        publisher.close()
    assert reader.read() == {}, 'a closed producer leaves no plane behind'


def test_the_profile_names_the_published_label_and_may_rename_a_single_label_detector(tmp_path, monkeypatch):
    shipped = load_profile('red-disc', ROOT).detectors[0]
    assert shipped.label == 'landing_pad' and shipped.labels == ('landing_pad',), 'stated in the file, not implied'

    def profile(row):
        path = tmp_path / f'{uuid.uuid4().hex}.json'
        path.write_text(json.dumps(dict(name='t', sources=[], detectors=[dict(sensor='front', detector='red_disc',
                                                                              **row)])))
        return load_profile(str(path), ROOT)
    renamed = profile(dict(label='red_pad')).detectors[0]
    assert renamed.labels == ('red_pad',) and renamed.native_labels == ('landing_pad',)
    assert renamed.published_label('landing_pad') == 'red_pad'
    assert profile({}).detectors[0].labels == ('landing_pad',), 'no label: the detector declares it'
    assert renamed.settings_digest != profile({}).detectors[0].settings_digest, 'what is published is in the digest'
    assert renamed.as_dict()['label'] == 'red_pad'
    for bad, message in ((dict(label='red pad'), 'must be 1-120'), (dict(label=7), 'must be text')):
        with pytest.raises(ValueError, match=message):
            profile(bad)

    class Twin:
        name: ClassVar[str] = 'twin'
        labels: ClassVar[tuple[str, ...]] = ('a', 'b')
        sensor_type: ClassVar[str] = 'camera.rgb'
        settings_class: ClassVar[type | None] = None
    monkeypatch.setitem(DET.DETECTORS, 'twin', Twin)
    with pytest.raises(ValueError, match='only rename a single-label detector'):
        validate_sources((), detectors=[Detector('front', 'twin', {}, label='x')])
    from dalg.run import matches_prepare
    from dalg.profiles import Profile
    composed = Profile('p', load_profile('lidar-baseline', ROOT).sources, 'd', detectors=(renamed,))
    assert matches_prepare(composed, ['algorithm:red_pad']) and not matches_prepare(composed, ['algorithm:landing_pad'])
