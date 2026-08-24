#!/usr/bin/env python3
"""Join aggregate-only outputs from the released resident and BLINK-v5 chains."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping


TOTALS = {"VStar": 191, "MMStar": 1500, "BLINK-v5": 1901, "ZoomBench": 845}


def read(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"not a JSON object: {path}")
    return value


def find_score(value: Any, total: int) -> tuple[int, int] | None:
    if isinstance(value, Mapping):
        correct = value.get("correct", value.get("num_correct"))
        observed = value.get("total", value.get("num_total"))
        if isinstance(correct, int) and not isinstance(correct, bool) and observed == total:
            return correct, observed
        for child in value.values():
            found = find_score(child, total)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = find_score(child, total)
            if found is not None:
                return found
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threebench", required=True, type=Path)
    parser.add_argument("--blink", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    three = read(args.threebench)
    blink = read(args.blink)
    benchmarks: dict[str, Any] = {}
    for name, total in TOTALS.items():
        source = blink if name == "BLINK-v5" else three
        score = find_score(source, total)
        if score is None:
            raise ValueError(f"missing strict {name} score")
        benchmarks[name] = {"correct": score[0], "total": score[1], "percent": score[0] / score[1] * 100.0}
    summary = {
        "schema_version": "hw_bjtu_opd_fourbench_summary_v1",
        "status": "complete",
        "benchmarks": benchmarks,
        "macro_percent": sum(item["percent"] for item in benchmarks.values()) / 4,
        "aggregate_only": True,
        "sample_level_output": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
