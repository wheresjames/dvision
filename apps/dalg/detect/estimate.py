"""A crude running estimate per target: association by distance and a weighted mean.

Deliberately simple. A projected detection joins the nearest estimate of the
same label from the same detector within ``gate_m``; otherwise it opens a new
one. Each estimate keeps a confidence-weighted mean position, its spread (the
weighted RMS distance of its observations from that mean), an observation
count and its first and last sighting in sim time. It is ``confirmed`` once it
has ``confirm_n`` observations, and ``stale`` once unseen for ``expire_s`` --
marked, never deleted, and fresh again if it is seen again.

IDs (``landing_pad-1``, ...; prefixed with the detector ID when several
detectors run) are stable for a mapping epoch. :meth:`clear`
drops every estimate when the epoch changes and returns the events that say so.
Confirmation that needs the vehicle to move -- go closer, look again -- is the
mission's business, not this module's.

Every change returns an event ``(kind, data)``; the host records them as
``estimate.opened`` / ``updated`` / ``confirmed`` / ``stale`` / ``cleared``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

#: Estimates kept per host; beyond it a detection that would open another is counted and dropped.
MAX_TRACKED = 256
MIN_WEIGHT = 1e-3


@dataclass
class Estimate:
    id: str
    label: str
    detector: str
    first_s: float
    last_s: float
    count: int = 0
    revision: int = 0
    confirmed: bool = False
    stale: bool = False
    weight: float = 0.0
    sum_x: float = 0.0
    sum_y: float = 0.0
    sum_r2: float = 0.0
    best_confidence: float = 0.0
    #: The detection IDs that made it, newest last (bounded).
    detections: list[str] = field(default_factory=list)

    @property
    def xy_m(self) -> tuple[float, float]:
        return (self.sum_x / self.weight, self.sum_y / self.weight)

    @property
    def spread_m(self) -> float:
        x, y = self.xy_m
        return math.sqrt(max(0.0, self.sum_r2 / self.weight - (x * x + y * y)))

    @property
    def confidence(self) -> float:
        return self.weight / max(self.count, 1)

    def as_dict(self) -> dict[str, Any]:
        x, y = self.xy_m
        return dict(id=self.id, label=self.label, detector=self.detector, xy_m=[round(x, 4), round(y, 4)],
                    spread_m=round(self.spread_m, 4), count=self.count, revision=self.revision,
                    confirmed=self.confirmed, stale=self.stale, first_s=round(self.first_s, 4),
                    last_s=round(self.last_s, 4), confidence=round(min(1.0, self.confidence), 4))


class Estimates:
    def __init__(self, *, gate_m: float, confirm_n: int, expire_s: float, id_prefix: str = '') -> None:
        self.gate_m, self.confirm_n, self.expire_s = float(gate_m), int(confirm_n), float(expire_s)
        #: Namespaces IDs when several detectors publish side by side.
        self.id_prefix = id_prefix
        self.items: dict[str, Estimate] = {}
        self.dropped = 0
        self._next: dict[str, int] = {}

    def _open(self, label: str, detector: str, time_s: float) -> Estimate:
        number = self._next.get(label, 0) + 1
        self._next[label] = number
        estimate = Estimate(f'{self.id_prefix}{label}-{number}', label, detector, time_s, time_s)
        self.items[estimate.id] = estimate
        return estimate

    def associate(self, label: str, detector: str, xy_m, confidence: float, time_s: float,
                  detection_id: str = '') -> tuple[Estimate | None, list[tuple[str, dict[str, Any]]]]:
        """Fold one projected detection in; ``(estimate, events)``, or ``(None, [])`` when full."""
        x, y = float(xy_m[0]), float(xy_m[1])
        near = [e for e in self.items.values() if e.label == label and e.detector == detector
                and math.dist(e.xy_m, (x, y)) <= self.gate_m]
        events: list[tuple[str, dict[str, Any]]] = []
        if near:
            estimate = min(near, key=lambda e: (math.dist(e.xy_m, (x, y)), e.id))
            before = estimate.as_dict()
        elif len(self.items) >= MAX_TRACKED:
            self.dropped += 1
            return None, []
        else:
            estimate = self._open(label, detector, time_s)
            before = None
        weight = max(float(confidence), MIN_WEIGHT)
        estimate.weight += weight
        estimate.sum_x += weight * x
        estimate.sum_y += weight * y
        estimate.sum_r2 += weight * (x * x + y * y)
        estimate.count += 1
        estimate.revision += 1
        estimate.last_s = float(time_s)
        estimate.best_confidence = max(estimate.best_confidence, float(confidence))
        if detection_id:
            estimate.detections.append(detection_id)
            del estimate.detections[:-64]
        revived = estimate.stale
        estimate.stale = False
        cause = dict(detection=detection_id, xy_m=[round(x, 4), round(y, 4)], confidence=round(float(confidence), 4))
        if before is None:
            events.append(('opened', dict(estimate=estimate.as_dict(), cause=cause)))
        else:
            events.append(('updated', dict(before=before, estimate=estimate.as_dict(), cause=cause, revived=revived)))
        if not estimate.confirmed and estimate.count >= self.confirm_n:
            estimate.confirmed = True
            events.append(('confirmed', dict(estimate=estimate.as_dict(), cause=cause)))
        return estimate, events

    def expire(self, time_s: float) -> list[tuple[str, dict[str, Any]]]:
        """Mark estimates unseen for ``expire_s`` stale."""
        events = []
        for estimate in self.items.values():
            if not estimate.stale and float(time_s) - estimate.last_s >= self.expire_s:
                estimate.stale = True
                estimate.revision += 1
                events.append(('stale', dict(estimate=estimate.as_dict(),
                                             cause=dict(unseen_s=round(float(time_s) - estimate.last_s, 4)))))
        return events

    def clear(self, cause: str) -> list[tuple[str, dict[str, Any]]]:
        """Forget everything (a mapping epoch ended); one ``cleared`` event per estimate."""
        events = [('cleared', dict(estimate=e.as_dict(), cause=cause)) for e in self.items.values()]
        self.items.clear()
        self._next.clear()
        return events

    def published(self, limit: int) -> list[dict[str, Any]]:
        """What the plane carries: confirmed first, then fresh, then the most recently seen."""
        ordered = sorted(self.items.values(), key=lambda e: (not e.confirmed, e.stale, -e.last_s, e.id))
        return [e.as_dict() for e in ordered[:limit]]
