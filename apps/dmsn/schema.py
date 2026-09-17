"""`ParamSchema`: one small typed field list behind the form, the profile and ``--set``.

A mission, a strategy and the profile's own settings each declare their
parameters once. The same declaration generates the Setup tab's form, validates
a profile file, and parses a command-line ``--set KEY=VALUE`` -- three consumers
that would otherwise drift into three slightly different ideas of what a valid
goal is. Deliberately not JSON Schema: a handful of kinds, a range, a unit and
a line of help is everything a mission needs, and all of it fits on a screen.

Kinds:

* scalars -- ``float``, ``int``, ``bool``, ``choice``, ``text``;
* geometry in local map metres -- ``point`` (``x,y``), ``bounds``
  (``xmin,ymin,xmax,ymax``), ``points`` and ``polygon`` (``x,y;x,y;...``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

from dcmn.mission import MAX_TEXT, MAX_VERTICES, MAX_WAYPOINTS

KINDS = ('float', 'int', 'bool', 'choice', 'text', 'point', 'bounds', 'points', 'polygon')
TRUE = ('1', 'true', 'yes', 'on')
FALSE = ('0', 'false', 'no', 'off')
#: What a form shows, and ``--set`` accepts, for "not set".
UNSET = ('', 'none', 'null', '-')


def _finite(value: Any, name: str) -> float:
    try: number = float(value)
    except (TypeError, ValueError): raise ValueError(f'{name} must be a number') from None
    if isinstance(value, bool) or not math.isfinite(number): raise ValueError(f'{name} must be a finite number')
    return number


def _pair(value: Any, name: str) -> list[float]:
    if isinstance(value, str): value = value.split(',')
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f'{name} must be x,y in metres')
    return [_finite(value[0], name), _finite(value[1], name)]


@dataclass(frozen=True)
class Field:
    """One parameter: its kind, default, limits and the line of help the form shows."""

    name: str
    kind: str
    default: Any = None
    minimum: float | None = None
    maximum: float | None = None
    unit: str = ''
    help: str = ''
    choices: tuple[str, ...] = ()
    #: Whether "not set" (``None``) is a valid value.
    optional: bool = False
    label: str = ''

    def __post_init__(self) -> None:
        if self.kind not in KINDS: raise ValueError(f'field {self.name}: unknown kind {self.kind!r}')
        if self.kind == 'choice' and not self.choices: raise ValueError(f'field {self.name}: a choice needs choices')

    @property
    def title(self) -> str:
        """The form label: the name in words, without a unit suffix the unit column already shows."""
        if self.label: return self.label
        name = self.name
        for suffix in ('_m', '_s', '_pct'):
            if self.unit and name.endswith(suffix): name = name[:-len(suffix)]
        return name.replace('_', ' ')

    @property
    def limit(self) -> int:
        return MAX_WAYPOINTS if self.kind == 'points' else MAX_VERTICES

    # -- values ------------------------------------------------------------

    def check(self, value: Any) -> Any:
        """The value in canonical form, or ``ValueError`` naming this field."""
        name = self.name
        if value is None or (isinstance(value, str) and value.strip().lower() in UNSET and self.kind != 'text'):
            if self.optional: return None
            raise ValueError(f'{name} is required')
        kind = self.kind
        if kind in ('float', 'int'):
            number = _finite(value, name)
            if kind == 'int':
                if number != int(number): raise ValueError(f'{name} must be a whole number')
                number = int(number)
            if self.minimum is not None and number < self.minimum:
                raise ValueError(f'{name} must be at least {self.minimum:g}{self.unit and " " + self.unit}')
            if self.maximum is not None and number > self.maximum:
                raise ValueError(f'{name} must be at most {self.maximum:g}{self.unit and " " + self.unit}')
            return number
        if kind == 'bool':
            if isinstance(value, bool): return value
            text = str(value).strip().lower()
            if text in TRUE: return True
            if text in FALSE: return False
            raise ValueError(f'{name} must be true or false')
        if kind == 'choice':
            if value not in self.choices: raise ValueError(f'{name} must be one of {", ".join(self.choices)}')
            return str(value)
        if kind == 'text':
            text = str(value)
            if len(text) > MAX_TEXT: raise ValueError(f'{name} is longer than {MAX_TEXT} characters')
            return text
        if kind == 'point':
            return _pair(value, name)
        if kind == 'bounds':
            if isinstance(value, str): value = value.split(',')
            if not isinstance(value, (list, tuple)) or len(value) != 4:
                raise ValueError(f'{name} must be xmin,ymin,xmax,ymax in metres')
            bounds = [_finite(v, name) for v in value]
            if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
                raise ValueError(f'{name} must have a positive extent')
            return bounds
        # points / polygon
        if isinstance(value, str):
            value = [piece for piece in value.split(';') if piece.strip()]
        if not isinstance(value, (list, tuple)): raise ValueError(f'{name} must be a list of x,y points')
        if len(value) > self.limit:
            raise ValueError(f'{name} has {len(value)} points; at most {self.limit} are allowed')
        points = [_pair(point, name) for point in value]
        if kind == 'polygon' and points:
            from dmsn.geometry import simple_polygon_error
            problem = simple_polygon_error(points)
            if problem: raise ValueError(f'{name}: {problem}')
        return points

    def parse(self, raw: str) -> Any:
        """Command-line or form text to a checked value."""
        return self.check(raw)

    def format(self, value: Any) -> str:
        """A value as the text a form entry shows and ``parse`` reads back."""
        if value is None: return ''
        if self.kind == 'bool': return 'true' if value else 'false'
        if self.kind in ('point', 'bounds'): return ','.join(f'{v:g}' for v in value)
        if self.kind in ('points', 'polygon'): return ';'.join(f'{x:g},{y:g}' for x, y in value)
        if self.kind == 'float': return f'{value:g}'
        return str(value)


class ParamSchema:
    """An ordered set of fields."""

    def __init__(self, fields: Iterable[Field] = ()) -> None:
        self.fields = tuple(fields)
        self.by_name = {f.name: f for f in self.fields}
        if len(self.by_name) != len(self.fields): raise ValueError('duplicate parameter names')

    def __contains__(self, name: str) -> bool:
        return name in self.by_name

    def __iter__(self):
        return iter(self.fields)

    def field(self, name: str) -> Field:
        try: return self.by_name[name]
        except KeyError: raise ValueError(f'unknown parameter {name!r}; known: {", ".join(self.by_name) or "none"}') from None

    def defaults(self) -> dict[str, Any]:
        return {f.name: f.default for f in self.fields}

    def validate(self, values: dict[str, Any] | None) -> dict[str, Any]:
        """Every field present and checked, in declaration order; unknown keys refused."""
        values = dict(values or {})
        unknown = set(values) - set(self.by_name)
        if unknown: raise ValueError(f'unknown parameters: {", ".join(sorted(unknown))}')
        result = {}
        for f in self.fields:
            result[f.name] = f.check(values[f.name] if f.name in values else f.default)
        return result

    def form_rows(self, values: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """What the Setup tab draws: one row per field, with its current text."""
        values = values or {}
        return [dict(name=f.name, title=f.title, kind=f.kind, unit=f.unit, help=f.help,
                     choices=f.choices, optional=f.optional,
                     text=f.format(values.get(f.name, f.default))) for f in self.fields]
