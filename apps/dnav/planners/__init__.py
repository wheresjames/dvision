"""One module per planner, and the protocol they all satisfy.

`dalg` puts each algorithm in its own module and registers them in one table;
this follows that, and deliberately not `daic`'s one-file-with-everything. A
new planner -- hybrid A\\*, RRT\\*, field D\\* -- is a module here and a line in
``PLANNERS``, and nothing above it changes.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from dnav.planners.astar import AStarPlanner
from dnav.planners.controls import ControlPlanner
from dnav.route import Route


@runtime_checkable
class Planner(Protocol):
    """Everything dnav asks of a planner."""

    #: How the planner is named on the command line, in the UI and in a report.
    name: str

    def plan(self, cost_map, start, goal, policy) -> Route:
        """Plan from ``start`` to ``goal`` in map metres, or say why not."""
        ...


#: The control planner is listed alongside the real one on purpose. Every
#: measured thing in this repository is bracketed by a control, and a route is
#: no different: the straight line is the floor a plan has to beat, and a
#: planner that cannot beat it is one whose cost function is wrong.
PLANNERS: dict[str, type] = {
    AStarPlanner.name: AStarPlanner,
    ControlPlanner.name: ControlPlanner,
}

#: What dnav plans with unless told otherwise.
DEFAULT_PLANNER = AStarPlanner.name


def build(name: str):
    """One planner by name, rejecting an unknown one rather than defaulting."""
    if name not in PLANNERS:
        raise KeyError(f'unknown planner {name!r}; known: {", ".join(sorted(PLANNERS))}')
    return PLANNERS[name]()
