"""What a mission decides, and how a vehicle action reaches dway.

A mission step returns exactly one decision:

========  ================================================================
`Goal`    set or keep this goal; the runner debounces unchanged goals
`Action`  an executor action by name (``launch``, ``hold``, ``resume``)
`Wait`    nothing to do; the reason becomes the blocking fact
`Done`    the mission is over: ``complete``, ``failed``, ``aborted``, ``budget``;
          ``finish`` may name the finish action (``land``) instead of the profile's
========  ================================================================

Decisions are plain values with a dictionary form, so the archive holds them
exactly and a replay compares them exactly. `IntentWriter` is the only way a
vehicle action leaves dmsn: a retained snapshot with a new revision per request
(:mod:`dcmn.mission`), which dway may refuse.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from dcmn.mission import INTENT_KINDS, INTENT_SCHEMA, OUTCOMES, clip

ACTIONS = tuple(k for k in INTENT_KINDS if k != 'none')


@dataclass(frozen=True)
class Goal:
    xy: tuple[float, float]
    tag: str = ''
    #: Operator intent (a click, a retarget, going home) is sent at once;
    #: a strategy's proposal waits out the debounce.
    operator: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, 'xy', (float(self.xy[0]), float(self.xy[1])))


@dataclass(frozen=True)
class Action:
    kind: str
    args: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in ACTIONS: raise ValueError(f'unknown executor action {self.kind!r}')


@dataclass(frozen=True)
class Wait:
    reason: str


@dataclass(frozen=True)
class Done:
    outcome: str
    reason: str
    #: The finish action this ending needs (``hold``, ``home``, ``land``); empty uses the profile's.
    finish: str = ''

    def __post_init__(self) -> None:
        if self.outcome not in OUTCOMES or not self.outcome: raise ValueError(f'invalid outcome {self.outcome!r}')
        if self.finish not in ('', 'hold', 'home', 'land'): raise ValueError(f'invalid finish action {self.finish!r}')


Decision = Goal | Action | Wait | Done


def to_dict(decision: Decision) -> dict[str, Any]:
    if isinstance(decision, Goal):
        return dict(type='goal', xy=[round(decision.xy[0], 6), round(decision.xy[1], 6)], tag=decision.tag,
                    operator=decision.operator)
    if isinstance(decision, Action): return dict(type='action', kind=decision.kind, args=dict(decision.args))
    if isinstance(decision, Wait): return dict(type='wait', reason=decision.reason)
    if isinstance(decision, Done):
        value = dict(type='done', outcome=decision.outcome, reason=decision.reason)
        if decision.finish: value['finish'] = decision.finish
        return value
    raise TypeError(f'not a mission decision: {decision!r}')


def from_dict(value: dict[str, Any]) -> Decision:
    kind = value.get('type')
    if kind == 'goal': return Goal(tuple(value['xy']), value.get('tag', ''), bool(value.get('operator')))
    if kind == 'action': return Action(value['kind'], dict(value.get('args') or {}))
    if kind == 'wait': return Wait(value['reason'])
    if kind == 'done': return Done(value['outcome'], value['reason'], value.get('finish', ''))
    raise ValueError(f'not a mission decision record: {value!r}')


class IntentWriter:
    """Publishes the retained intent; one revision per request."""

    def __init__(self, output, vehicle: str, session: str) -> None:
        self.output, self.vehicle, self.session = output, vehicle, session
        self.revision = 0
        self.sequence = 0
        self.last: dict[str, Any] | None = None

    def send(self, kind: str, time_s: float, reason: str = '', **args: Any) -> dict[str, Any]:
        if kind not in INTENT_KINDS: raise ValueError(f'unknown intent {kind!r}')
        self.revision += 1
        self.sequence += 1
        value = dict(schema=INTENT_SCHEMA, vehicle_id=self.vehicle, session=self.session,
                     sequence=self.sequence, time_s=float(time_s), revision=self.revision,
                     kind=kind, args=args, reason=clip(reason))
        self.output.write(value)
        self.last = value
        return value
