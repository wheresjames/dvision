"""One module per mission type, and the protocol they all satisfy.

Following `dnav/planners` and `dalg/algo`: a new mission is a module here and a
line in ``MISSIONS``, and nothing above it changes. `editors` is what keeps it
that way for the window -- the Mission tab shows the tools a mission asks for
(``goal``, ``area``, ``waypoints``) without knowing which mission it is.

Missions are pure logic over :class:`~dmsn.inputs.Observation`: no planes, no
clocks, no windows. Take-off, going home and the finish action belong to the
runner, so a mission implements only its body.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Protocol, Sequence, runtime_checkable

from dcmn.mapview import Shape
from dmsn.inputs import Observation
from dmsn.intents import Decision
from dmsn.schema import ParamSchema

#: Map editors a mission may ask the Mission tab for.
EDITORS = ('goal', 'area', 'waypoints')
#: What a mission is told about, beside its observations.
EVENT_KINDS = ('click', 'retarget', 'paused', 'resumed', 'authority_lost')

OverlayShapes = list[Shape]


@dataclass(frozen=True)
class MissionEvent:
    kind: str
    xy: tuple[float, float] | None = None
    data: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in EVENT_KINDS: raise ValueError(f'unknown mission event {self.kind!r}')
        if self.xy is not None: object.__setattr__(self, 'xy', (float(self.xy[0]), float(self.xy[1])))

    def as_dict(self) -> dict[str, Any]:
        return dict(kind=self.kind, xy=None if self.xy is None else list(self.xy), data=dict(self.data))

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> 'MissionEvent':
        return cls(value['kind'], None if value.get('xy') is None else tuple(value['xy']), dict(value.get('data') or {}))


@runtime_checkable
class Mission(Protocol):
    name: ClassVar[str]
    label: ClassVar[str]
    schema: ClassVar[ParamSchema]
    editors: ClassVar[tuple[str, ...]]
    #: Whether the reference background opens enabled (Q10): on for operator missions.
    reference_default: ClassVar[bool]

    def start(self, params: dict[str, Any], obs: Observation, *, area: Sequence[tuple[float, float]] = (),
              waypoints: Sequence[tuple[float, float]] = (),
              strategy: Any = None) -> None: ...
    def step(self, obs: Observation) -> Decision: ...
    def on_event(self, event: MissionEvent) -> None: ...
    def overlay(self) -> OverlayShapes: ...
    def progress(self) -> dict[str, Any]: ...
    def summary(self) -> dict[str, Any]: ...


from dmsn.missions.explore import ExploreMission  # noqa: E402 - the registry follows the protocol it satisfies
from dmsn.missions.goto import GotoMission
from dmsn.missions.search import SearchMission
from dmsn.missions.waypoints import WaypointsMission

MISSIONS: dict[str, type] = {
    GotoMission.name: GotoMission,
    ExploreMission.name: ExploreMission,
    WaypointsMission.name: WaypointsMission,
    SearchMission.name: SearchMission,
}


def build(name: str) -> Mission:
    try: return MISSIONS[name]()
    except KeyError: raise ValueError(f'unknown mission {name!r}; known: {", ".join(sorted(MISSIONS))}') from None


def target_of(params: dict[str, Any] | None) -> dict[str, Any] | None:
    """The target a mission declared, or None: missions declare needs, never detectors (§13.6).

    A mission declares one by having ``target_label`` in its schema, and may
    add ``target_detector`` (a detector ID; empty accepts any) and
    ``min_confidence``.
    """
    params = params or {}
    if not params.get('target_label'): return None
    return dict(label=str(params['target_label']), detector=str(params.get('target_detector') or ''),
                min_confidence=float(params.get('min_confidence') or 0.0))


def schema_for(name: str) -> ParamSchema:
    try: return MISSIONS[name].schema
    except KeyError: raise ValueError(f'unknown mission {name!r}; known: {", ".join(sorted(MISSIONS))}') from None
