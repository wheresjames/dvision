"""goto: fly to one point, chosen by a click, typed coordinates or the provider's target.

The smallest real mission, and the one every other is measured against: set a
goal, wait for dway to report it complete, finish. A click or an operator
retarget while flying replaces the point at once. How the vehicle gets there
-- the route, the stops, the permission -- is dnav's and dway's business, and
the mission only reports what they say.
"""
from __future__ import annotations

from typing import Any, ClassVar, Sequence

from dcmn import theme
from dcmn.mapview import Shape
from dmsn.inputs import Observation
from dmsn.intents import Decision, Done, Goal, Wait
from dmsn.schema import Field, ParamSchema

SOURCES = ('click', 'xy', 'provider_target')


class GotoMission:
    name: ClassVar[str] = 'goto'
    label: ClassVar[str] = 'Go to a point'
    editors: ClassVar[tuple[str, ...]] = ('goal',)
    reference_default: ClassVar[bool] = True
    schema: ClassVar[ParamSchema] = ParamSchema((
        Field('goal_source', 'choice', default='click', choices=SOURCES,
              help='click: the map sets the goal; xy: the goal below; provider_target: the target '
                   'the simulator publishes (assisted, recorded)'),
        Field('goal', 'point', optional=True, unit='m',
              help='x,y in local map metres; a map click fills it in'),
    ))

    def __init__(self) -> None:
        self.params: dict[str, Any] = {}
        self.target: tuple[float, float] | None = None
        self.operator = False
        self.retargets = 0
        self.targets: list[tuple[float, float]] = []

    def start(self, params: dict[str, Any], obs: Observation, *, area: Sequence = (),
              waypoints: Sequence = (), strategy: Any = None) -> None:
        self.params = self.schema.validate(params)
        goal = self.params.get('goal')
        if self.params['goal_source'] in ('xy', 'click') and goal is not None:
            self._set(goal, operator=True)

    def _set(self, xy, *, operator: bool) -> None:
        self.target = (float(xy[0]), float(xy[1]))
        self.operator = operator
        self.targets.append(self.target)

    def on_event(self, event) -> None:
        if event.kind in ('click', 'retarget') and event.xy is not None:
            if self.target is not None: self.retargets += 1
            self._set(event.xy, operator=True)

    def step(self, obs: Observation) -> Decision:
        if self.target is None:
            if self.params.get('goal_source') == 'provider_target':
                if obs.provider_target is None:
                    return Wait(f'no provider target: {obs.provider_target_reason or "not published"}')
                self._set(obs.provider_target, operator=False)
            elif self.params.get('goal_source') == 'xy':
                return Wait('goal_source is xy but no goal is set')
            else:
                return Wait('click the map to set a goal')
        if obs.arrived(self.target):
            return Done('complete', f'arrived at {self.target[0]:.2f}, {self.target[1]:.2f} m')
        tag = 'provider target' if self.params.get('goal_source') == 'provider_target' and not self.retargets else 'goal'
        return Goal(self.target, tag, self.operator)

    def overlay(self) -> list[Shape]:
        shapes = []
        for x, y in self.targets[:-1]:
            r = 0.2
            shapes.append(Shape('oval', x - r, y - r, x + r, y + r, outline=theme.DIM, width=1.))
        return shapes

    def progress(self) -> dict[str, Any]:
        return dict(target=None if self.target is None else list(self.target), retargets=self.retargets)

    def summary(self) -> dict[str, Any]:
        return dict(goal_source=self.params.get('goal_source'), targets=[list(t) for t in self.targets],
                    retargets=self.retargets)
