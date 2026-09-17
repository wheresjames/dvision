"""A staged successor generation must survive its predecessor closing.

pymembus unlinks a shared-memory name when the handle that created it closes,
even after another store has re-created that name. dalg stages a new mapping
generation (coverage reset) and then closes the old publisher, so the old
handle's close deleted the new registry and consumers such as dnav kept a
stale map forever. These tests pin the recovery.
"""
import time
import uuid

import numpy as np

from dcmn.maps import GridGeometry, MapPublisher, MapSession, registry_name
from dvision2_common import load_pymembus

SOURCES = [dict(id='scan', sensor='scan', sensor_type='fixture', algorithm='synthetic')]


def context(instance, epoch):
    return dict(frame_id='local', localization_epoch=0, clock_domain_id=instance, clock_epoch=0,
                mapping_epoch=epoch, geometry_revision=epoch)


def poll_until(session, predicate, timeout_s=3.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        session._last_probe = -1e9
        session._backoff_s = 0.
        session.poll()
        if predicate(): return True
        time.sleep(0.02)
    return predicate()


def test_a_consumer_follows_a_new_generation_after_the_old_publisher_closes():
    instance = 'gen' + uuid.uuid4().hex[:8]
    geometry = GridGeometry.from_extent(10, 10, .5)
    occupancy = np.zeros(geometry.shape, np.uint8)
    observed = np.full(geometry.shape, 1000, np.uint32)
    old = MapPublisher(instance, geometry, SOURCES, context=context(instance, 1), generation=1)
    session = MapSession(instance, probe_interval_s=0.01)
    new = None
    try:
        old.publish('scan', occupancy, observed, 1.0)
        assert poll_until(session, lambda: session.identity is not None and session.latest('scan') is not None)
        # dalg's order: build and activate the successor, then retire the predecessor.
        new = MapPublisher(instance, geometry, SOURCES, context=context(instance, 2), generation=2, activate=False)
        new.activate()
        old.close()
        new.publish('scan', occupancy, observed, 2.0)
        assert poll_until(session, lambda: session.identity == (new.session, 2) and session.latest('scan') is not None), \
            f'consumer stuck on {session.identity}'
        assert session.latest('scan').sim_time_s == 2.0
    finally:
        session.close()
        if new is not None: new.close()
        old.close()
    probe = load_pymembus().memkv()
    assert not probe.open(registry_name(instance)), 'a clean shutdown leaves no registry behind'


def test_a_publisher_never_takes_the_name_back_from_a_newer_owner():
    instance = 'gen' + uuid.uuid4().hex[:8]
    geometry = GridGeometry.from_extent(10, 10, .5)
    first = MapPublisher(instance, geometry, SOURCES, context=context(instance, 1), generation=1)
    second = MapPublisher(instance, geometry, SOURCES, context=context(instance, 2), generation=2)
    try:
        assert first._ensure_registry() is False
        kv = load_pymembus().memkv()
        assert kv.open(registry_name(instance))
        try: assert kv.getAll().get('maps.generation') == '2'
        finally: kv.close()
    finally:
        second.close(); first.close()
