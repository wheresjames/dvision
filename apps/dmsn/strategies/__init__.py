"""Goal strategies: what an area mission asks for its next goal.

A strategy only proposes; the mission owns commitment, blacklisting and
termination. One module per strategy and a line in ``STRATEGIES``, as with
missions. Two exist: the nearest frontier (the baseline) and wall following
by goal stepping (the proof that the seam is real).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Protocol, Sequence, runtime_checkable

from dcmn.mapview import Shape
from dmsn.inputs import Observation
from dmsn.schema import ParamSchema


@dataclass(frozen=True)
class Candidate:
    xy: tuple[float, float]
    #: Why this point: shown as the goal's reason and kept in the archive.
    reason: str = ''


@runtime_checkable
class GoalStrategy(Protocol):
    name: ClassVar[str]
    schema: ClassVar[ParamSchema]

    def propose(self, obs: Observation, area: Sequence[tuple[float, float]],
                blacklist: Sequence[tuple[float, float]]) -> Candidate | None: ...
    def overlay(self) -> list[Shape]: ...


from dmsn.strategies.frontier import FrontierStrategy  # noqa: E402 - the registry follows the protocol
from dmsn.strategies.wall_follow import WallFollowStrategy  # noqa: E402

STRATEGIES: dict[str, type] = {
    FrontierStrategy.name: FrontierStrategy,
    WallFollowStrategy.name: WallFollowStrategy,
}


def build(name: str, params: dict[str, Any] | None = None) -> GoalStrategy:
    try: cls = STRATEGIES[name]
    except KeyError: raise ValueError(f'unknown strategy {name!r}; known: {", ".join(sorted(STRATEGIES)) or "none"}') from None
    strategy = cls()
    strategy.params = cls.schema.validate(params)
    return strategy
