"""Everything dmsn reads and writes outside itself, behind one small surface.

The runner talks only to a ports object. `LivePorts` is the real one -- the
session context, dway's execution snapshot, dnav's navigation snapshot, the
evidence plane, the provider's status plane, the module bus and dmsn's own two
retained planes. Tests substitute a fake with the same methods, which is what
lets debounce, authority loss and budgets be checked without a single process.

Reads never raise into the loop: an unavailable or malformed plane reads as
empty and says why in ``errors``.
"""
from __future__ import annotations

from typing import Any

from dcmn.context import Context
from dcmn.navigation import Snapshot


class LivePorts:
    def __init__(self, instance: str, *, planner: str = 'dnav', bus: bool = True, sim_time=None,
                 detections: str = 'dalg') -> None:
        from dcmn.detections import DetectionsReader
        from dcmn.maps import MapSession
        self.instance = instance
        self.context = Context(instance)
        self.execution = Snapshot(instance, 'execution')
        self.navigation = Snapshot(instance, planner=planner)
        self.status_out = Snapshot(instance, 'mission')
        self.intent_out = Snapshot(instance, 'mission_intent')
        self.maps = MapSession(instance)
        self.detections = DetectionsReader(instance, detections)
        self.errors: dict[str, str] = {}
        self.bus = None
        if bus:
            from dcmn.module_bus import PymembusModuleBus
            self.bus = PymembusModuleBus(instance, 'mission', 'dmsn', sim_time=sim_time or (lambda: 0.0))

    # -- ownership -----------------------------------------------------------

    def claim(self) -> None:
        """Own both mission planes, or raise ``ValueError`` if another dmsn does (Q11)."""
        self.status_out.start()
        self.previous_status = self.status_out.previous
        try: self.intent_out.start()
        except Exception:
            self.status_out.close()
            raise

    # -- reads ---------------------------------------------------------------

    def _guarded(self, name, read, default):
        try:
            value = read()
            self.errors.pop(name, None)
            return value
        except (ValueError, RuntimeError, OSError) as exc:
            self.errors[name] = str(exc)
            return default

    def read_context(self) -> dict[str, Any]:
        return self._guarded('context', self.context.read, {})

    def read_execution(self) -> dict[str, Any]:
        return self._guarded('execution', self.execution.read, {})

    def read_navigation(self) -> dict[str, Any]:
        return self._guarded('navigation', self.navigation.read, {})

    def read_detections(self) -> dict[str, Any]:
        """The detections plane, or ``{}``: only ever called for a mission with a target (Q27)."""
        return self._guarded('detections', self.detections.read, {})

    @property
    def detections_plane(self) -> str:
        return self.detections.name

    def poll_maps(self) -> None:
        self._guarded('maps', self.maps.poll, None)

    def grids(self) -> dict[str, Any]:
        grids = {}
        for sid in sorted(self.maps.states):
            grid = self.maps.latest(sid)
            if grid is not None: grids[sid] = grid
        return grids

    def coverage(self) -> tuple[float, float, float, float] | None:
        geometry = self.maps.geometry
        return None if geometry is None else tuple(float(v) for v in geometry.bounds_m())

    def provider_status(self) -> dict[str, str] | None:
        from dcmn.provider_target import read_status
        return self._guarded('provider', lambda: read_status(self.instance), None)

    # -- writes ----------------------------------------------------------------

    def set_goal(self, writer: str, position, *, handoff: bool = False) -> dict[str, Any] | None:
        return self.context.set_goal(writer, position, role='mission', handoff=handoff)

    def release_authority(self, writer: str) -> bool:
        try: return self.context.release_authority(writer)
        except (ValueError, RuntimeError): return False

    def request_reset(self, bounds, *, by: str = 'dmsn') -> int:
        return self.context.request_reset(bounds, by=by)

    def write_status(self, value: dict[str, Any]) -> None:
        self.status_out.write(value)

    def write_intent(self, value: dict[str, Any]) -> None:
        self.intent_out.write(value)

    # -- bus ---------------------------------------------------------------------

    def bus_publish(self, event_type: str, *, run_id: str = '', payload: dict[str, Any] | None = None) -> None:
        if self.bus is None: return
        try:
            if self.bus.connect(): self.bus.publish(event_type, run_id=run_id, payload=payload or {})
        except Exception as exc:  # the bus is coordination, never control
            self.errors['bus'] = f'{type(exc).__name__}: {exc}'

    def bus_receive(self) -> list:
        if self.bus is None: return []
        try:
            if not self.bus.connect(): return []
            return self.bus.receive()
        except Exception as exc:
            self.errors['bus'] = f'{type(exc).__name__}: {exc}'
            return []

    def close(self) -> None:
        for closer in (self.status_out.close, self.intent_out.close, self.maps.close):
            try: closer()
            except Exception: pass
        if self.bus is not None:
            try: self.bus.close()
            except Exception: pass


class IntentOutput:
    """The ``write`` surface `IntentWriter` needs, bound to a ports object."""

    def __init__(self, ports) -> None:
        self.ports = ports

    def write(self, value: dict[str, Any]) -> None:
        self.ports.write_intent(value)
