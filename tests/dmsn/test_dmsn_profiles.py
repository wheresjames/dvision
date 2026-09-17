"""Mission parameter schemas and profiles: one declaration behind the form, the file and --set."""
from __future__ import annotations

import json

import pytest

from dcmn.mission import MAX_VERTICES, MAX_WAYPOINTS
from dmsn import profiles as P
from dmsn.geometry import point_in_polygon, simple_polygon_error
from dmsn.missions import MISSIONS
from dmsn.schema import Field, ParamSchema


@pytest.fixture(autouse=True)
def user_dir(tmp_path, monkeypatch):
    monkeypatch.setenv('DVISION2_MISSION_DIR', str(tmp_path / 'user'))
    return tmp_path / 'user'


# -- ParamSchema ---------------------------------------------------------------------

def test_form_rows_come_from_the_schema_with_current_text():
    schema = ParamSchema((Field('speed', 'float', default=1.5, minimum=0, unit='m/s', help='how fast'),
                          Field('mode', 'choice', default='a', choices=('a', 'b')),
                          Field('goal', 'point', optional=True),
                          Field('on', 'bool', default=True)))
    rows = schema.form_rows(dict(goal=[1.0, -2.5]))
    assert [r['name'] for r in rows] == ['speed', 'mode', 'goal', 'on']
    assert rows[0]['unit'] == 'm/s' and rows[0]['help'] == 'how fast' and rows[0]['text'] == '1.5'
    assert rows[1]['choices'] == ('a', 'b')
    assert rows[2]['text'] == '1,-2.5'
    assert rows[3]['text'] == 'true'


@pytest.mark.parametrize('kind,raw,expected', [
    ('float', '2.5', 2.5), ('int', '7', 7), ('bool', 'yes', True), ('bool', 'off', False),
    ('point', '52.4, 2.4', [52.4, 2.4]), ('bounds', '0,0,10,5', [0., 0., 10., 5.]),
    ('points', '0,0;1,1;2,0', [[0., 0.], [1., 1.], [2., 0.]]),
])
def test_text_parses_to_the_canonical_value_and_formats_back(kind, raw, expected):
    field = Field('x', kind)
    value = field.parse(raw)
    assert value == expected
    assert field.parse(field.format(value)) == value


@pytest.mark.parametrize('field,raw,message', [
    (Field('x', 'float', minimum=0), '-1', 'at least'),
    (Field('x', 'float', maximum=5), '6', 'at most'),
    (Field('x', 'int'), '1.5', 'whole'),
    (Field('x', 'choice', choices=('a',)), 'b', 'one of'),
    (Field('x', 'point'), '1', 'x,y'),
    (Field('x', 'float'), 'nan', 'finite'),
    (Field('x', 'bounds'), '5,0,1,1', 'positive extent'),
    (Field('x', 'float'), '', 'required'),
    (Field('x', 'polygon'), '0,0;10,10;10,0;0,10', 'cross'),
])
def test_bad_values_are_refused_with_the_field_named(field, raw, message):
    with pytest.raises(ValueError, match=message):
        field.parse(raw)


def test_unknown_parameters_are_refused():
    with pytest.raises(ValueError, match='unknown parameters'):
        ParamSchema((Field('a', 'float', default=1.),)).validate(dict(a=1., b=2.))


def test_every_registered_mission_has_a_valid_default_profile():
    for name in MISSIONS:
        profile = P.blank(name)
        assert profile['mission'] == name
        assert P.validate(profile) == profile


# -- geometry ---------------------------------------------------------------------------

def test_simple_polygons_and_containment():
    square = [[0, 0], [10, 0], [10, 10], [0, 10]]
    assert simple_polygon_error(square) == ''
    assert 'cross' in simple_polygon_error([[0, 0], [10, 10], [10, 0], [0, 10]])
    assert 'at least 3' in simple_polygon_error([[0, 0], [1, 1]])
    assert point_in_polygon((5, 5), square) and point_in_polygon((0, 5), square)
    assert not point_in_polygon((11, 5), square)
    ell = [[0, 0], [10, 0], [10, 4], [4, 4], [4, 10], [0, 10]]
    assert simple_polygon_error(ell) == ''
    assert not point_in_polygon((8, 8), ell), 'the notch of a concave area is outside it'


# -- profiles ---------------------------------------------------------------------------

def test_choosing_a_strategy_starts_from_its_defaults_and_its_params_are_overridable():
    explore = P.blank('explore')
    assert explore['strategy'] == 'frontier' and explore['strategy_params'] == {}
    wall = P.apply_overrides(explore, ['strategy=wall_follow', 'strategy_params.hand=left'])
    assert wall['strategy_params'] == dict(step_m=4., hand='left', standoff_m=1.5, reach_m=6.)
    back = P.set_value(wall, 'strategy', 'frontier')
    assert back['strategy_params'] == {}, 'stale wall-follow parameters do not survive a strategy change'
    assert P.set_value(wall, 'mission', 'goto')['strategy'] == '', 'goto takes no strategy'
    with pytest.raises(ValueError, match='unknown strategy'):
        P.set_value(explore, 'strategy', 'spiral')
    with pytest.raises(ValueError, match='one of'):
        P.apply_overrides(wall, ['strategy_params.hand=both'])


def test_overrides_use_dotted_sections_and_bare_mission_params():
    profile = P.apply_overrides(P.blank('goto'), [
        'goal_source=xy', 'goal=52.4,2.4', 'budgets.time_s=600', 'finish_action=home',
        'debounce.min_commit_s=5', 'workspace=0,0,70,40', 'arena=maze_020'])
    assert profile['params'] == dict(goal_source='xy', goal=[52.4, 2.4])
    assert profile['budgets']['time_s'] == 600.
    assert profile['finish_action'] == 'home'
    assert profile['debounce']['min_commit_s'] == 5.
    assert profile['workspace'] == [0., 0., 70., 40.]
    for bad, message in (('nope=1', 'unknown profile key'), ('budgets.time_s=-1', 'at least'),
                         ('finish_action=fly', 'one of'), ('goal', 'KEY=VALUE')):
        with pytest.raises(ValueError, match=message):
            P.apply_overrides(profile, [bad])


def test_round_trip_keeps_the_digest_and_the_name_is_not_part_of_it(tmp_path):
    profile = P.apply_overrides(P.blank('goto', name='first'), ['goal_source=xy', 'goal=3,4'])
    path = P.save(profile, tmp_path / 'first.json')
    loaded, where = P.load(path)
    assert loaded == profile and where == path
    assert P.digest(loaded) == P.digest(profile)
    renamed = P.set_value(loaded, 'name', 'second')
    assert P.digest(renamed) == P.digest(profile)
    assert P.digest(P.set_value(loaded, 'budgets.time_s', 10)) != P.digest(profile)


def test_size_caps_hold_in_the_file_as_in_the_snapshot():
    too_many = [[float(i), 0.] for i in range(MAX_WAYPOINTS + 1)]
    with pytest.raises(ValueError, match='at most 256'):
        P.validate(dict(P.blank(), waypoints=too_many))
    ring = [[100 * __import__('math').cos(a), 100 * __import__('math').sin(a)]
            for a in [2 * 3.141592653589793 * i / (MAX_VERTICES + 1) for i in range(MAX_VERTICES + 1)]]
    with pytest.raises(ValueError, match='at most 256'):
        P.validate(dict(P.blank(), area=ring))
    assert P.validate(dict(P.blank(), area=ring[:MAX_VERTICES]))['area']


def test_unknown_keys_and_schemas_are_refused():
    with pytest.raises(ValueError, match='unknown profile keys'):
        P.validate(dict(P.blank(), surprise=1))
    with pytest.raises(ValueError, match='schema'):
        P.validate(dict(P.blank(), schema='dvision2.mission-profile.v0'))
    with pytest.raises(ValueError, match='unknown mission'):
        P.validate(dict(P.blank(), mission='teleport'))


def test_bare_names_resolve_user_directory_first_then_examples(user_dir):
    example = P.ASSET_DIR / 'maze_020-goto.json'
    assert example.exists(), 'the committed goto example is missing'
    assert P.resolve('maze_020-goto') == example
    user_copy = P.save(P.load(example)[0], user_dir / 'maze_020-goto.json')
    assert P.resolve('maze_020-goto') == user_copy
    labels = {(label, path.stem) for label, path in P.available()}
    assert ('user', 'maze_020-goto') in labels and ('example', 'maze_020-goto') in labels


def test_committed_examples_all_validate():
    paths = sorted(P.ASSET_DIR.glob('*.json'))
    assert paths
    for path in paths:
        profile, _ = P.load(path)
        assert json.loads(path.read_text())['mission'] == profile['mission']


def test_the_document_tracks_dirty_state_and_saves_to_the_user_directory(user_dir):
    document = P.ProfileDocument(P.load('maze_020-goto')[0], P.resolve('maze_020-goto'))
    assert not document.dirty
    assert document.set('budgets.time_s', '120') and document.dirty
    assert not document.set('budgets.time_s', 'soon') and 'number' in document.error
    assert document.value['budgets']['time_s'] == 120.
    saved = document.save(user_dir / 'mine.json')
    assert saved.exists() and not document.dirty and document.value['name'] == 'mine'
    document.set('finish_action', 'home')
    document.revert()
    assert document.value['finish_action'] == 'hold' and not document.dirty
