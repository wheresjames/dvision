"""Transport-neutral module presence and run coordination over pymembus."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from dvision2_common import load_pymembus, shared_names, validate_id

MAGIC = "dvision2.module.v1"
SCHEMA_VERSION = 1
DEFAULT_SIZE = 262144
SHUTDOWN_EVENT = "system.shutdown"


def requests_shutdown(event: "ModuleEvent") -> bool:
    """Return whether an event requests an orderly instance-wide shutdown."""
    return event.type == SHUTDOWN_EVENT


class ModuleBus(Protocol):
    def connect(self) -> bool: ...
    def publish(self, event_type: str, *, run_id: str = "",
                payload: dict[str, Any] | None = None) -> bool: ...
    def receive(self) -> list["ModuleEvent"]: ...
    def close(self) -> None: ...


@dataclass(frozen=True)
class ModuleEvent:
    event_id: str
    instance_id: str
    role: str
    implementation: str
    process_id: str
    sequence: int
    sim_time_s: float
    type: str
    run_id: str
    payload: dict[str, Any]

    @classmethod
    def decode(cls, raw: str | bytes) -> "ModuleEvent | None":
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        try:
            value = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            return None
        if not isinstance(value, dict) or value.get("magic") != MAGIC:
            return None
        if value.get("schema_version") != SCHEMA_VERSION:
            return None
        source = value.get("source")
        payload = value.get("payload", {})
        required = ("event_id", "instance_id", "sequence", "sim_time_s", "type")
        if (not isinstance(source, dict) or not isinstance(payload, dict)
                or any(name not in value for name in required)):
            return None
        try:
            return cls(
                event_id=str(value["event_id"]),
                instance_id=validate_id(str(value["instance_id"])),
                role=str(source["role"]),
                implementation=str(source["implementation"]),
                process_id=str(source["process_id"]),
                sequence=int(value["sequence"]),
                sim_time_s=float(value["sim_time_s"]),
                type=str(value["type"]),
                run_id=str(value.get("run_id", "")), payload=payload)
        except (KeyError, TypeError, ValueError):
            return None


@dataclass
class PipelineMember:
    role: str
    implementation: str
    process_id: str
    protocol_version: int
    state: str
    ready: bool
    run_id: str
    capabilities: Any
    seen_monotonic: float
    #: The module's own account of whether it is keeping up, straight off the
    #: heartbeat. ``None`` from a module that does not report one.
    intake: Any = None


class PipelineView:
    """Expiring projection of complete hello/heartbeat snapshots."""
    def __init__(self, expiry_s: float = 3.0) -> None:
        self.expiry_s = expiry_s
        self._members: dict[str, PipelineMember] = {}

    def observe(self, event: ModuleEvent, now: float | None = None) -> None:
        if event.type == "module.goodbye":
            self._members.pop(event.process_id, None)
            return
        if event.type not in ("module.hello", "module.heartbeat"): return
        p = event.payload
        self._members[event.process_id] = PipelineMember(
            event.role, event.implementation, event.process_id, SCHEMA_VERSION,
            str(p.get("state", "")), bool(p.get("ready", False)), event.run_id,
            p.get("capabilities", ()), time.monotonic() if now is None else now,
            p.get("intake"))

    def members(self, now: float | None = None, *, include_expired=False):
        current = time.monotonic() if now is None else now
        result = []
        for member in self._members.values():
            age = max(0.0, current-member.seen_monotonic)
            if include_expired or age <= self.expiry_s: result.append((member, age))
        return sorted(result, key=lambda item: (item[0].role, item[0].implementation))

    def matching(self, role: str, selector: str = "", now: float | None = None):
        return [member for member, _ in self.members(now) if member.role == role and
                (not selector or member.implementation == selector or
                 member.process_id == selector or
                 (isinstance(member.capabilities, dict) and
                  selector in member.capabilities.get("algorithms", ()) ))]
class PymembusModuleBus:
    """One read/write endpoint on the per-instance broadcast ring."""

    def __init__(self, instance_id: str, role: str, implementation: str, *,
                 create: bool = False, size: int = DEFAULT_SIZE,
                 sim_time: Callable[[], float] = time.monotonic) -> None:
        self.instance_id = validate_id(instance_id)
        self.role = role
        self.implementation = implementation
        self.process_id = uuid.uuid4().hex
        self.create = create
        self.size = int(size)
        self.sim_time = sim_time
        self.sequence = 0
        self.overruns = 0
        self.session_id: int | None = None
        self._seen: set[str] = set()
        self._pm = load_pymembus()
        self._handle = None
        self._last_session_probe = -1e9

    def connect(self) -> bool:
        if self._handle is not None:
            return True
        handle = self._pm.memmsg()
        if not handle.open(shared_names(self.instance_id)["events"], self.size,
                           True, self.create):
            return False
        self._handle = handle
        self.session_id = int(handle.getSessionId())
        return True

    def publish(self, event_type: str, *, run_id: str = "",
                payload: dict[str, Any] | None = None) -> bool:
        if not self.connect():
            return False
        self.sequence += 1
        value = {
            "magic": MAGIC, "schema_version": SCHEMA_VERSION,
            "event_id": uuid.uuid4().hex, "instance_id": self.instance_id,
            "source": {"role": self.role, "implementation": self.implementation,
                       "process_id": self.process_id},
            "sequence": self.sequence, "sim_time_s": float(self.sim_time()),
            "type": event_type, "run_id": run_id, "payload": payload or {},
        }
        return bool(self._handle.write(json.dumps(
            value, sort_keys=True, separators=(",", ":"))))

    def receive(self) -> list[ModuleEvent]:
        if not self.connect():
            return []
        now = time.monotonic()
        if not self.create and now - self._last_session_probe >= 1.0:
            self._last_session_probe = now
            probe = self._pm.memmsg()
            name = shared_names(self.instance_id)["events"]
            if probe.open(name, self.size, True, False):
                session = int(probe.getSessionId())
                if session != self.session_id:
                    self._handle.close()
                    self._handle = probe
                    self.session_id = session
                    self._seen.clear()
                    probe = None
                if probe is not None:
                    probe.close()
        events: list[ModuleEvent] = []
        while self._handle.poll():
            raw, overrun = self._handle.read_with_overrun(0)
            if overrun:
                self.overruns += 1
            event = ModuleEvent.decode(raw)
            if event is None or event.instance_id != self.instance_id:
                continue
            if event.event_id in self._seen:
                continue
            self._seen.add(event.event_id)
            if len(self._seen) > 8192:
                self._seen.clear()
                self._seen.add(event.event_id)
            events.append(event)
        return events

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def remove(self) -> None:
        self.close()
        self._pm.memmsg.remove(shared_names(self.instance_id)["events"])
