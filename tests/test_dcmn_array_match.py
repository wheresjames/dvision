"""Array and metadata rendezvous, including independently delayed channels."""
import numpy as np
import pytest

from dcmn.sensors import SensorSession, LIDAR_FRAME
from dsim.profiles import DroneProfile, camera_profile
from tests.test_sensor_contract import manager


@pytest.fixture
def pair():
    draft = camera_profile(16, 12, physics_hz=30.)
    draft['sensors'].append(dict(id='scan', type='lidar.scan2d', parent='body',
                                 rate_hz=10., model=dict(samples=8)))
    sensors = manager(DroneProfile.parse(draft))
    session = SensorSession(sensors.publisher.instance)
    stream = session.open('scan', accounting='from_attach')
    try: yield sensors.publisher, session, stream
    finally: session.close(); sensors.close()


def array(pub, seq):
    fields = [np.full(f['shape'], seq, dtype=f['dtype'])
              for f in pub.sensors['scan']['layout']]
    pub.write_array('scan', seq, seq, seq*100000, b''.join(f.tobytes() for f in fields))


def metadata(pub, seq):
    pub.write_compact('scan', seq, seq, seq*100000, LIDAR_FRAME, dict(array_sequence=seq))


@pytest.mark.parametrize('order', ['together', 'array_first', 'metadata_first'])
def test_array_pairing(pair, order):
    pub, session, stream = pair
    if order == 'metadata_first':
        metadata(pub, 1); session.poll(); stream.refresh()
        assert stream.latest is None
        array(pub, 1)
    else:
        array(pub, 1)
        if order == 'array_first':
            session.poll(); stream.refresh(); assert stream.latest is None
        metadata(pub, 1)
    session.poll(); stream.refresh()
    assert stream.latest.capture_id == 1
    assert np.all(stream.latest.fields['range_m'] == 1)


def test_out_of_order_and_missing_array(pair):
    pub, session, stream = pair
    metadata(pub, 1); metadata(pub, 2)
    array(pub, 2)
    session.poll(); stream.refresh()
    assert stream.latest.sequence == 2
    assert 1 in stream.metadata
    array(pub, 1); session.poll(); stream.refresh()
    # The stale pair is never admitted and never replaces sequence 2,
    # but the loss is still counted.
    assert not stream.metadata
    assert stream.latest.sequence == 2
    assert stream.late_drops == 1


def test_reset_between_pair_discards_both_caches(pair):
    pub, session, stream = pair
    metadata(pub, 1); session.poll()
    array(pub, 2); stream.refresh()
    assert stream.pending and stream.metadata
    pub.reset()
    metadata(pub, 3); array(pub, 3)
    session.poll(); stream.refresh()
    assert stream.latest.sequence == 3
    assert stream.association_drops == 2
    assert not stream.pending and not stream.metadata


def test_closed_array_reopens_without_initial_loss(pair):
    pub, session, stream = pair
    metadata(pub, 1); array(pub, 1); session.poll(); stream.refresh()
    session.release('scan')
    assert stream.array is None
    for i in range(2, 25): metadata(pub, i); array(pub, i); session.poll()
    reopened = session.open('scan', accounting='from_attach')
    metadata(pub, 25); array(pub, 25); session.poll(); reopened.refresh()
    assert reopened.latest.sequence == 25
    assert reopened.skipped == 0
