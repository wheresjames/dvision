import json
from pathlib import Path

from tests.mini_slam_correlation import analyze


def _write_map(path: Path) -> None:
    path.write_text(
        "\n".join([
            "--- VARS",
            "+=drone",
            "0=wall",
            "--- MAP",
            "       ",
            " +  0  ",
            "       ",
        ]) + "\n",
        encoding="utf-8",
    )


def _record(x: float, y: float, yaw: float, front: float) -> dict:
    return {
        "t": 1.0,
        "state": "SEARCH",
        "telem": {
            "drone.x_m": str(x),
            "drone.y_m": str(y),
            "drone.heading_deg": str(yaw),
        },
        "vision": {
            "slam": {
                "method": "mini_slam:ok",
                "confidence": 1.0,
                "front": front,
                "front_left": 0.0,
                "front_right": 0.0,
                "front_range_m": None,
                "front_left_range_m": None,
                "front_right_range_m": None,
            },
            "local_map": {"front_occ_m": 0.7},
        },
    }


def test_analyze_counts_mini_slam_risk_correlation(tmp_path: Path) -> None:
    map_path = tmp_path / "map.txt"
    log_path = tmp_path / "flight.jsonl"
    _write_map(map_path)
    records = [
        _record(1.5, 1.5, 0.0, 0.7),    # wall ahead at x=4.5
        _record(1.5, 1.5, 180.0, 0.7),  # wall behind, not ahead
        _record(1.5, 1.5, 0.0, 0.0),    # no risk despite wall ahead
    ]
    log_path.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n",
        encoding="utf-8",
    )

    result = analyze(log_path, map_path)

    assert result["mini_slam_ticks"] == 3
    assert result["risk_ticks"] == 2
    assert result["risk_with_near_obstacle_ticks"] == 1
    assert result["risk_with_near_obstacle_rate"] == 0.5
    assert result["no_risk_with_near_obstacle_ticks"] == 1
    assert result["rangeless_risk_ticks"] == 2
