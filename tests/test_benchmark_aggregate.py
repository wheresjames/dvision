"""Tests for the multi-run benchmark aggregator (Phase 5.8)."""

from __future__ import annotations

import json
from pathlib import Path

from tests.benchmark_aggregate import aggregate, _stats


def _write_run(report_dir: Path, *, landed: bool, final_state: str,
               landing_error_m: float, min_dist_to_target_m: float,
               detection_rate: float, hints: list[str],
               git_commit: str = "deadbeef") -> None:
    report_dir.mkdir(parents=True)
    summary = {
        "landed": landed,
        "final_state": final_state,
        "landing_error_m": landing_error_m,
        "min_dist_to_target_m": min_dist_to_target_m,
        "detection_rate": detection_rate,
        "diagnosis": {
            "classification_hints": [{"label": h, "evidence": "..."} for h in hints],
        },
    }
    metadata = {
        "map": "dsim/assets/maps/maze_001.txt",
        "duration_s": 90,
        "git_commit": git_commit,
    }
    (report_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (report_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


def test_stats_reports_median_and_range() -> None:
    assert _stats([1.0, 2.0, 3.0]) == {"median": 2.0, "min": 1.0, "max": 3.0, "n": 3}


def test_stats_ignores_missing_values() -> None:
    assert _stats([1.0, None, 3.0]) == {"median": 2.0, "min": 1.0, "max": 3.0, "n": 2}


def test_stats_handles_all_missing() -> None:
    assert _stats([None, None]) == {"median": None, "min": None, "max": None, "n": 0}


def test_aggregate_combines_landed_rate_and_distributions(tmp_path: Path) -> None:
    _write_run(tmp_path / "run1", landed=True, final_state="COMPLETE",
               landing_error_m=0.6, min_dist_to_target_m=0.6,
               detection_rate=1.0, hints=[])
    _write_run(tmp_path / "run2", landed=False, final_state="FAILSAFE",
               landing_error_m=4.6, min_dist_to_target_m=4.6,
               detection_rate=0.5, hints=["map noise / trap"])
    _write_run(tmp_path / "run3", landed=False, final_state="FAILSAFE",
               landing_error_m=5.9, min_dist_to_target_m=5.9,
               detection_rate=0.49, hints=["map noise / trap", "control stall"])

    agg = aggregate([tmp_path / "run1", tmp_path / "run2", tmp_path / "run3"])

    assert agg["n"] == 3
    assert agg["landed_count"] == 1
    assert agg["landed_rate"] == 1 / 3
    assert agg["final_states"] == {"COMPLETE": 1, "FAILSAFE": 2}
    assert agg["landing_error_m"]["median"] == 4.6
    assert agg["landing_error_m"]["min"] == 0.6
    assert agg["landing_error_m"]["max"] == 5.9
    assert agg["hint_counts"] == {"map noise / trap": 2, "control stall": 1}


def test_aggregate_skips_dirs_without_summary(tmp_path: Path) -> None:
    _write_run(tmp_path / "run1", landed=True, final_state="COMPLETE",
               landing_error_m=0.6, min_dist_to_target_m=0.6,
               detection_rate=1.0, hints=[])
    empty_dir = tmp_path / "run2-empty"
    empty_dir.mkdir()

    agg = aggregate([tmp_path / "run1", empty_dir, tmp_path / "does-not-exist"])

    assert agg["n"] == 1
    assert str(empty_dir) in agg["missing"]
    assert str(tmp_path / "does-not-exist") in agg["missing"]


def test_aggregate_with_no_valid_runs_reports_zero(tmp_path: Path) -> None:
    agg = aggregate([tmp_path / "nope"])

    assert agg["n"] == 0
    assert agg["runs"] == []
