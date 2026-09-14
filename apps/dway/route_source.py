"""Where dway's route comes from: a static tour file, or dnav's retained snapshot.

The two sources are deliberately separate: only the tour adapter imports tour
code, and the dynamic source (:class:`dway.dynamic.DynamicRouteSource`) never
loads a tour or world map. A Mission keeps consuming the static tour exactly
as before; route edits in the tour editor cannot reach dynamic mode.
"""
from __future__ import annotations


class TourRouteSource:
    kind = 'tour'

    def __init__(self, path):
        from dway.tour import load_tour
        self.path = path
        self.tour = load_tour(path)

    def describe(self):
        return f'static tour {self.tour.tour_id}'


def route_source(args, profile=None):
    """The adapter selected by ``--mode``; dynamic mode needs a loaded profile."""
    if getattr(args, 'mode', 'tour') == 'dynamic':
        from dway.dynamic import DynamicRouteSource
        return DynamicRouteSource(args.id, args.planner, profile)
    return TourRouteSource(args.tour)
