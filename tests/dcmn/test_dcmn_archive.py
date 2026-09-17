"""The numeric output archive: bounded, recoverable, and honest about what it lost."""
from __future__ import annotations

import json
import threading
import time

import numpy as np
import pytest

from dcmn.archive import ArchiveReader, IncompleteInput, Recorder, next_archive_dir
from dcmn.maps import EvidenceGrid, GridGeometry


def grid(value=0, *, source='scan', revision=1, t=1., size=(20, 20)):
    geometry = GridGeometry.from_extent(*size, .5, origin_x_m=-5., origin_y_m=-5.)
    g = EvidenceGrid.blank(geometry, source)
    g.occupancy[0, :value, :] = 200
    g.observed_ms[0, :value, :] = 1000 + value
    return EvidenceGrid(geometry, g.occupancy, g.observed_ms, source, revision, t)


def record_some(recorder, count=5):
    for index in range(count):
        recorder.record('evidence.published', dict(index=index), {'scan': grid(index % 3, revision=index % 3 + 1)})


def test_a_clean_archive_is_complete_and_stores_each_content_once(tmp_path):
    recorder = Recorder(tmp_path/'a', dict(module='test'), commit_s=.01)
    record_some(recorder, 9)
    report = recorder.close()
    assert report['complete'] and report['events_committed'] == 9 and report['state'] == 'finalized'
    reader = ArchiveReader(tmp_path/'a')
    check = reader.validate()
    assert check['complete'] and check['events'] == 9 and not check['errors']
    stored = [c for chunk in reader.chunks.values() for c in chunk['contents']]
    assert len(stored) == len(set(stored)) == 3
    rebuilt = reader.reconstruct(reader.events()[4])['reconstructed_grids']['scan']
    original = grid(4 % 3, revision=2)
    assert np.array_equal(rebuilt.occupancy, original.occupancy)
    assert np.array_equal(rebuilt.observed_ms, original.observed_ms)
    assert rebuilt.geometry.origin_x_m == -5.
    manifest = json.loads((tmp_path/'a/archive.json').read_text())
    assert 'uint8' in manifest['payload_semantics']['occupancy']


def test_payloads_are_numeric_only_and_load_without_pickle(tmp_path):
    recorder = Recorder(tmp_path/'a', commit_s=.01)
    record_some(recorder, 2); recorder.close()
    for chunk in (tmp_path/'a/chunks').glob('*.npz'):
        with np.load(chunk, allow_pickle=False) as data:
            assert all(data[key].dtype.kind in 'u' for key in data.files)


def test_the_caller_may_mutate_its_arrays_the_moment_record_returns(tmp_path):
    recorder = Recorder(tmp_path/'a', commit_s=5.)
    live = grid(2)
    recorder.record('evidence.published', {}, {'scan': live})
    live.occupancy[:] = 7
    recorder.close()
    reader = ArchiveReader(tmp_path/'a')
    assert (reader.reconstruct(reader.events()[0])['reconstructed_grids']['scan'].occupancy != 7).any()


def test_a_full_queue_drops_marks_the_gap_and_never_blocks(tmp_path):
    recorder = Recorder(tmp_path/'a', queue_bytes=12000, commit_s=60.)
    started = time.monotonic()
    accepted = [recorder.record('e', {}, {'scan': grid(i % 20, revision=i + 1)}) for i in range(10)]
    assert time.monotonic() - started < 1.
    assert None in accepted and accepted[0] == 1
    recorder.record('late', {})
    report = recorder.close()
    assert report['dropped'] > 0 and report['dropped_by']['queue full'] == report['dropped']
    assert not report['complete']
    check = ArchiveReader(tmp_path/'a').validate()
    assert not check['complete']
    assert any('recorder dropped sequences' in e for e in check['errors'])
    assert not any('unexplained' in e for e in check['errors'])


def test_the_disk_quota_stops_payloads_keeps_earlier_data_and_marks_the_archive(tmp_path):
    recorder = Recorder(tmp_path/'a', disk_bytes=(1 << 20) + 12000, commit_s=.001)
    sequences = []
    for i in range(40):
        sequences.append(recorder.record('e', {}, {'scan': grid(i % 20, revision=i + 1, size=(10, 10))}))
        time.sleep(.005)
    report = recorder.close()
    assert report['quota_exhausted'] and not report['complete']
    assert 'recording disk quota exceeded' in report['errors']
    reader = ArchiveReader(tmp_path/'a')
    check = reader.validate()
    assert check['events'] >= 1 and not check['complete']
    # What was committed before the limit is still whole.
    first = reader.events()[0]
    assert reader.reconstruct(first)['reconstructed_grids']['scan'].revision == 1


def test_a_corrupt_chunk_is_reported_not_trusted(tmp_path):
    recorder = Recorder(tmp_path/'a', commit_s=.01)
    record_some(recorder, 3); recorder.close()
    chunk = next((tmp_path/'a/chunks').glob('*.npz'))
    raw = bytearray(chunk.read_bytes()); raw[len(raw)//2] ^= 0xFF; chunk.write_bytes(bytes(raw))
    check = ArchiveReader(tmp_path/'a').validate()
    assert not check['complete']
    assert any('checksum mismatch' in e for e in check['errors'])


def test_a_missing_chunk_is_an_incomplete_input(tmp_path):
    recorder = Recorder(tmp_path/'a', commit_s=.01)
    record_some(recorder, 3); recorder.close()
    for chunk in (tmp_path/'a/chunks').glob('*.npz'): chunk.unlink()
    reader = ArchiveReader(tmp_path/'a')
    with pytest.raises(IncompleteInput):
        reader.reconstruct(reader.events()[0])
    check = reader.validate()
    assert not check['complete'] and len(check['incomplete_events']) == 3


def test_an_unclean_end_recovers_committed_chunks_and_ignores_the_torn_tail(tmp_path):
    recorder = Recorder(tmp_path/'a', commit_s=.01)
    record_some(recorder, 4)
    time.sleep(.3)
    # Simulate a crash: the worker never finalizes, and a torn line and an
    # uncommitted payload are left behind.
    recorder._abandoned = True
    recorder.stop.set(); recorder.thread.join(2)
    with (tmp_path/'a/index.jsonl').open('a') as index: index.write('{"sequence": 5, "ty')
    (tmp_path/'a/chunks/chunk-000999.npz').write_bytes(b'partial')
    reader = ArchiveReader(tmp_path/'a')
    check = reader.validate()
    assert not check['clean'] and not check['complete']
    assert check['events'] == 4 and not check['errors']
    assert any('truncated' in w for w in check['warnings'])
    assert any('never committed' in w for w in check['warnings'])


def test_a_finalized_archive_with_a_truncated_index_is_an_error(tmp_path):
    recorder = Recorder(tmp_path/'a', commit_s=.01)
    record_some(recorder, 3); recorder.close()
    path = tmp_path/'a/index.jsonl'
    path.write_text(path.read_text()[:-20])
    check = ArchiveReader(tmp_path/'a').validate()
    assert not check['complete'] and any('truncated' in e for e in check['errors'])


def test_a_planning_attempt_resolves_only_with_its_full_input_set(tmp_path):
    recorder = Recorder(tmp_path/'a', commit_s=.01)
    recorder.record('planning.attempt', dict(inputs=dict(sources=['scan']), pose={'x_m': 1}, goal={},
                    policy={}, planner='astar', route={}), {'scan': grid(2)})
    recorder.record('planning.attempt', dict(inputs=dict(sources=['scan', 'front']), pose={}, goal={},
                    policy={}, planner='astar', route={}), {'scan': grid(2)})
    recorder.close()
    reader = ArchiveReader(tmp_path/'a')
    attempt = reader.reconstruct_attempt(1)
    assert attempt['grids']['scan'].revision == 1 and attempt['pose'] == {'x_m': 1}
    with pytest.raises(IncompleteInput, match='front'):
        reader.reconstruct_attempt(2)
    check = reader.validate()
    assert check['reconstructable_attempts'] == [1] and 2 in check['incomplete_events']
    assert not check['complete']


def test_archives_are_append_once_and_rollover_opens_the_next(tmp_path):
    first = next_archive_dir(tmp_path/'dalg')
    Recorder(first).close()
    with pytest.raises(FileExistsError): Recorder(first)
    second = next_archive_dir(tmp_path/'dalg')
    assert second.name == 'archive-2'
    Recorder(second).close()
    assert ArchiveReader(first).validate()['complete'] and ArchiveReader(second).validate()['complete']


def test_a_shutdown_that_cannot_drain_says_so(tmp_path, monkeypatch):
    recorder = Recorder(tmp_path/'a', commit_s=.01, drain_s=.2)
    gate = threading.Event()
    original = recorder._commit
    monkeypatch.setattr(recorder, '_commit', lambda *a: gate.wait(5) or original(*a))
    record_some(recorder, 2)
    report = recorder.close()
    gate.set()
    assert report['state'] == 'unfinished' and not report['complete']
    assert any('deadline' in e for e in report['errors'])


def test_the_cli_validates_and_resolves(tmp_path, capsys):
    from dcmn.archive import main
    recorder = Recorder(tmp_path/'a', commit_s=.01)
    recorder.record('planning.attempt', dict(inputs=dict(sources=['scan']), pose={}, goal={},
                    policy={}, planner='astar', route={}), {'scan': grid(2)})
    recorder.close()
    assert main([str(tmp_path/'a')]) == 0
    assert json.loads(capsys.readouterr().out)['complete'] is True
    assert main([str(tmp_path/'a'), '--attempt', '1']) == 0
    assert json.loads(capsys.readouterr().out)['grids']['scan']['revision'] == 1
