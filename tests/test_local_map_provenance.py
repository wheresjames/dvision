import json
from pathlib import Path

from tests.local_map_provenance import analyze


def _record(front_occ_m, source=None, state="SEARCH") -> dict:
    local_map = {
        "front_occ_m": front_occ_m,
        "occupied_by_source": {
            "flow:expansion|front|ranged": 3,
        },
    }
    if source is not None:
        local_map["front_occ_source"] = source
    return {
        "t": 1.0,
        "state": state,
        "telem": {},
        "vision": {"local_map": local_map},
    }


def test_analyze_buckets_close_front_provenance(tmp_path: Path) -> None:
    log = tmp_path / "flight.jsonl"
    source = {
        "source": "flow:expansion",
        "sector": "front",
        "ranged": True,
        "age_ticks": 2,
        "value": 1.7,
    }
    records = [
        _record(0.8, source, "SEARCH"),
        _record(2.0, source, "SEARCH"),
        _record(0.7, {**source, "age_ticks": 4}, "APPROACH"),
        _record(0.6, None, "APPROACH"),
    ]
    log.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")

    result = analyze(log)

    assert result["front_occ_close_ticks"] == 3
    assert result["front_occ_without_provenance_ticks"] == 1
    assert result["front_occ_close_by_source"]["flow:expansion|front|ranged"] == 2
    assert result["front_occ_close_by_source"]["unknown"] == 1
    assert result["front_occ_close_by_state"]["APPROACH"] == 2
    assert result["front_occ_close_source_age"]["flow:expansion|front|ranged"]["median"] == 3.0
