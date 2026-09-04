#!/usr/bin/env python3
"""Compare DALG summaries offline; deliberately no discovery or database."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

METRICS = ("occupied_iou", "free_iou", "coverage", "hallucination_rate")


def rows(paths):
    result = []
    for path in paths:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        algorithm = value["algorithm"]
        score = value["scores"][algorithm]
        result.append((value.get("profile", algorithm), algorithm, score))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="compare DALG summary.json files")
    parser.add_argument("summaries", nargs="+", type=Path)
    args = parser.parse_args(argv)
    data = rows(args.summaries)
    widths = [max(len("profile"), *(len(row[0]) for row in data)),
              max(len("algorithm"), *(len(row[1]) for row in data))]
    print(f"{'profile':<{widths[0]}}  {'algorithm':<{widths[1]}}  " +
          "  ".join(f"{name:>18}" for name in METRICS))
    for profile, algorithm, score in data:
        print(f"{profile:<{widths[0]}}  {algorithm:<{widths[1]}}  " +
              "  ".join(f"{float(score[name]):18.4f}" for name in METRICS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
