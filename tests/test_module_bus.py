from __future__ import annotations

import uuid

from dcmn.module_bus import (ModuleEvent, PymembusModuleBus, SHUTDOWN_EVENT,
                             requests_shutdown)
from dvision2_common import load_pymembus, shared_names


def test_event_validation_and_additive_fields():
    raw = (b'{"magic":"dvision2.module.v1","schema_version":1,'
           b'"event_id":"e","instance_id":"area1",'
           b'"source":{"role":"algorithm","implementation":"mock",'
           b'"process_id":"p"},"sequence":2,"sim_time_s":3.0,'
           b'"type":"run.ready","run_id":"r","payload":{},"future":1}')
    event = ModuleEvent.decode(raw)
    assert event is not None
    assert (event.role, event.type, event.run_id) == ("algorithm", "run.ready", "r")
    assert ModuleEvent.decode(b"not json") is None


def test_memmsg_broadcast_has_independent_subscribers():
    instance = f"bus-{uuid.uuid4().hex}"
    pm = load_pymembus()
    name = shared_names(instance)["events"]
    pm.memmsg.remove(name)
    owner = PymembusModuleBus(instance, "simulator", "test", create=True)
    first = PymembusModuleBus(instance, "algorithm", "one")
    second = PymembusModuleBus(instance, "recorder", "two")
    try:
        assert owner.connect() and first.connect() and second.connect()
        assert owner.publish("module.hello", payload={"state": "ready"})
        assert [e.type for e in first.receive()] == ["module.hello"]
        assert [e.type for e in second.receive()] == ["module.hello"]
    finally:
        second.close(); first.close(); owner.remove()


def test_shutdown_is_instance_wide_and_not_run_scoped():
    raw = (b'{"magic":"dvision2.module.v1","schema_version":1,'
           b'"event_id":"stop","instance_id":"area1",'
           b'"source":{"role":"simulator","implementation":"dsim",'
           b'"process_id":"p"},"sequence":3,"sim_time_s":4.0,'
           b'"type":"system.shutdown","payload":{"scope":"instance"}}')
    event = ModuleEvent.decode(raw)
    assert event is not None
    assert event.type == SHUTDOWN_EVENT
    assert event.run_id == ""
    assert requests_shutdown(event)
