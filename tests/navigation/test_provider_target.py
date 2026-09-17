"""dnav's fallback goal: the map target the provider publishes, without reading the world."""
import pytest

from dcmn.provider_target import map_target
from dvision2_common import gps_to_local, local_to_gps

ORIGIN = dict(lat=52.52, lon=13.405, alt=34.0)


def snapshot(origin=(33.5, 17.5, 0.), epoch=0):
    return dict(localization_epoch=epoch, vehicle_transform=dict(
        schema='dvision2.local-ned-transform.v1', frame_id='local', localization_epoch=epoch, origin=list(origin)))


def status_for(x, y, width=67, height=35):
    """What dsim publishes for a map target at (x, y): its own map_to_gps, restated."""
    lat, lon, alt = local_to_gps(x - width / 2, height / 2 - y, 0., ORIGIN['lat'], ORIGIN['lon'], ORIGIN['alt'])
    return {'target.lat_deg': f'{lat:.7f}', 'target.lon_deg': f'{lon:.7f}', 'target.alt_m': f'{alt:.3f}',
            'origin.lat_deg': f"{ORIGIN['lat']:.7f}", 'origin.lon_deg': f"{ORIGIN['lon']:.7f}",
            'origin.alt_m': f"{ORIGIN['alt']:.3f}"}


def test_gps_to_local_inverts_local_to_gps():
    lat, lon, alt = local_to_gps(12.3, -45.6, 1.5, ORIGIN['lat'], ORIGIN['lon'], ORIGIN['alt'])
    assert gps_to_local(lat, lon, alt, ORIGIN['lat'], ORIGIN['lon'], ORIGIN['alt']) == pytest.approx((12.3, -45.6, 1.5))


def test_the_published_target_converts_to_map_metres():
    position, reason = map_target(status_for(52.444, 2.389), snapshot())
    assert reason == '' and position == pytest.approx([52.444, 2.389], abs=0.02)  # 7-decimal GNSS


def test_no_target_means_no_goal_with_a_reason():
    values = status_for(10., 10.)
    values['target.lat_deg'] = values['target.lon_deg'] = ''
    assert map_target(values, snapshot()) == (None, 'provider publishes no map target')
    assert map_target(None, snapshot())[1] == 'provider status plane unavailable'
    assert map_target(status_for(10., 10.), dict(localization_epoch=0))[1].startswith('session context has no')
    assert map_target(status_for(10., 10.), snapshot(epoch=0) | dict(localization_epoch=1))[0] is None
