"""Source-only baseline profiles: loading, composition, selectors and precise errors.

The operational collection is exactly one baseline per
evidence algorithm, holding sources and settings only. Tours, map extents and
controls are not profile fields; old field names are rejected by name rather
than silently ignored, and nothing aliases the retired profile names.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from dalg.dalg import mapping_config, parse_args
from dalg.profiles import (PRIMARY_CAMERA, load_profile, load_profiles, preflight, profile_dir,
                           resolve_sources, source_configs)

ROOT = Path(__file__).resolve().parents[1]
BASELINES = ('features-baseline', 'ground-plane-baseline', 'lidar-baseline',
             'monocular-depth-baseline', 'optical-flow-baseline', 'plane-sweep-baseline',
             'sgbm-baseline')


def test_the_collection_is_exactly_one_baseline_per_evidence_algorithm():
    names = sorted(path.stem for path in profile_dir(ROOT).glob('*.json'))
    assert names == sorted(BASELINES)
    algorithms = {load_profile(name, ROOT).algorithm for name in BASELINES}
    assert algorithms == set(source_configs())


@pytest.mark.parametrize('name', BASELINES)
def test_every_baseline_holds_only_sources_and_binds_its_sensor(name):
    raw = json.loads((profile_dir(ROOT) / f'{name}.json').read_text())
    assert set(raw) == {'name', 'sources'}
    profile = load_profile(name, ROOT)
    assert profile.name == name and len(profile.sources) == 1
    source = profile.sources[0]
    assert source.sensor == ('scan' if source.algorithm == 'lidar_inverse' else PRIMARY_CAMERA)


def test_profile_names_and_paths():
    expected = load_profile('lidar-baseline', ROOT)
    assert profile_dir(ROOT) == ROOT / 'assets/algorithm_profiles'
    assert load_profile('lidar-baseline.json', ROOT) == expected
    assert load_profile('assets/algorithm_profiles/lidar-baseline.json', ROOT) == expected
    assert load_profile(str(expected.path), ROOT) == expected
    assert load_profiles(['lidar-baseline'], ROOT) == expected
    with pytest.raises(FileNotFoundError): load_profile('./lidar-baseline.json', ROOT)


@pytest.mark.parametrize('retired', ['sgbm-default', 'camera-lidar-maze020', 'constant-maze020',
                                     'exact-range-maze020', 'synthetic-maze020'])
def test_retired_profile_names_are_gone_without_aliases(retired):
    with pytest.raises(FileNotFoundError):
        load_profile(retired, ROOT)


@pytest.mark.parametrize('field', ['tour', 'map', 'algorithm', 'sensors', 'settings'])
def test_legacy_fields_are_rejected_by_name(tmp_path, field):
    path = tmp_path / 'legacy.json'
    path.write_text(json.dumps({'name': 'legacy', 'sources': [
        {'sensor': 'scan', 'algorithm': 'lidar_inverse', 'settings': {}}], field: 'x'}))
    with pytest.raises(ValueError, match=f'unsupported field.*{field}'):
        load_profile(str(path), ROOT)


def test_composition_keeps_every_source_and_its_provenance():
    profile = load_profiles(['lidar-baseline', 'ground-plane-baseline'], ROOT)
    assert [s.algorithm for s in profile.sources] == ['lidar_inverse', 'ground_plane']
    assert [c['name'] for c in profile.components] == ['lidar-baseline', 'ground-plane-baseline']
    assert profile.name == 'lidar-baseline+ground-plane-baseline'
    assert profile.digest == load_profiles(['lidar-baseline.json', 'ground-plane-baseline'], ROOT).digest


def test_duplicate_sources_are_rejected_before_and_after_selector_resolution(tmp_path):
    with pytest.raises(ValueError, match='duplicate'):
        load_profiles(['lidar-baseline', 'lidar-baseline'], ROOT)
    explicit = tmp_path / 'explicit.json'
    explicit.write_text(json.dumps({'name': 'explicit', 'sources': [
        {'sensor': 'nose', 'algorithm': 'sgbm', 'settings': {}}]}))
    # Distinct as written, identical once the selector binds to the manifest.
    profile = load_profiles(['sgbm-baseline', str(explicit)], ROOT)
    with pytest.raises(ValueError, match='duplicate'):
        resolve_sources(profile, {'primary_camera': 'nose', 'sensors': {}})
    # Never "the first camera": without a declared primary the selector stays unbound.
    unbound = resolve_sources(load_profile('sgbm-baseline', ROOT), {'sensors': {'a': {}}})
    assert unbound[0].sensor == PRIMARY_CAMERA


def test_invalid_settings_name_the_row():
    with pytest.raises(ValueError, match='source 1'):
        from dalg.profiles import Source, validate_sources
        validate_sources([Source('scan', 'lidar_inverse', {'free_log_odds': 1.0})])


def test_a_missing_model_is_a_precise_preflight_error_with_the_installer(tmp_path):
    path = tmp_path / 'depth.json'
    path.write_text(json.dumps({'name': 'depth', 'sources': [
        {'sensor': 'primary_camera', 'algorithm': 'monocular_depth',
         'settings': {'model_path': str(tmp_path / 'absent.onnx')}}]}))
    errors = preflight(load_profile(str(path), ROOT), ROOT)
    assert len(errors) == 1
    assert 'absent.onnx' in errors[0] and 'install_dalg_depth_model.py' in errors[0]
    assert preflight(load_profile('lidar-baseline', ROOT), ROOT) == []


def test_plural_cli_and_single_profile_compatibility():
    assert parse_args(['--id', 'test', '--profiles', 'a.json', 'b.json']).profiles == ['a.json', 'b.json']
    assert parse_args(['--edit', '--profiles', 'lidar-baseline']).profile == 'lidar-baseline'
    for argv in (['--edit', '--profiles', 'a', 'b'],
                 ['--id', 'test'],
                 ['--id', 'test', '--profile', 'a', '--profiles', 'b'],
                 ['--id', 'test', '--profile', 'a', '--camera-hz', '0'],
                 ['--id', 'test', '--profile', 'a', '--synthetic']):
        with pytest.raises(SystemExit): parse_args(argv)


def test_mapping_options_become_the_runtime_configuration():
    args = parse_args(['--id', 't', '--profile', 'lidar-baseline', '--bounds', '-10,-5,30,25',
                       '--cell-m', '0.25', '--dz-m', '2', '--mapping-budget-mib', '64'])
    config = mapping_config(args)
    assert config.bounds == (-10., -5., 30., 25.) and config.cell_m == .25
    assert config.dz_m == 2. and config.budget_bytes == 64 * 2**20
    with pytest.raises(ValueError):
        mapping_config(parse_args(['--id', 't', '--profile', 'x', '--bounds', '1,1,0,2']))


def test_plural_cli_loads_composition_into_run(monkeypatch):
    import dalg.dalg as app
    selected = []

    def make_run(instance, profile, root, **kwargs):
        selected.append((profile, kwargs))
        return SimpleNamespace(done=True, report_dir=None, close=lambda: None, state='RUNNING',
                               step=lambda: None, poll_delay=lambda: 0., finish=lambda **_: None)

    monkeypatch.setattr(app, 'DalgRun', make_run)
    assert app.main(['--id', 'test', '--no-ui', '--profiles',
                     'lidar-baseline.json', 'ground-plane-baseline', '--cell-m', '0.25']) == 0
    profile, kwargs = selected[0]
    assert len(profile.sources) == 2 and len(profile.components) == 2
    assert kwargs['mapping'].cell_m == .25 and kwargs['camera_hz'] == 5.


def test_a_missing_model_stops_the_launch_before_connecting(monkeypatch, tmp_path, capsys):
    import dalg.dalg as app
    path = tmp_path / 'depth.json'
    path.write_text(json.dumps({'name': 'depth', 'sources': [
        {'sensor': 'primary_camera', 'algorithm': 'monocular_depth',
         'settings': {'model_path': str(tmp_path / 'absent.onnx')}}]}))
    monkeypatch.setattr(app, 'DalgRun', lambda *a, **k: pytest.fail('must not construct a run'))
    assert app.main(['--id', 'test', '--no-ui', '--profile', str(path)]) == 1
    assert 'install_dalg_depth_model.py' in capsys.readouterr().err
