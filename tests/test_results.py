from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_result_counts_and_macros() -> None:
    value = json.loads((ROOT / "results/core_results.json").read_text(encoding="utf-8"))
    assert len(value["primary_blink_v5_matrix"]) == 7
    totals = {"VStar": 191, "MMStar": 1500, "BLINK-v5": 1901, "ZoomBench": 845}
    for row in value["primary_blink_v5_matrix"]:
        assert set(row["benchmarks"]) == set(totals)
        percents = []
        for name, score in row["benchmarks"].items():
            assert score["total"] == totals[name]
            assert abs(score["correct"] / score["total"] * 100 - score["percent"]) < 1e-12
            percents.append(score["percent"])
        assert abs(sum(percents) / 4 - row["macro_percent"]) < 1e-10


def test_latest_result_is_bound_to_fixed_27b_teacher() -> None:
    value = json.loads((ROOT / "results/core_results.json").read_text(encoding="utf-8"))
    latest = value["primary_blink_v5_matrix"][-1]
    assert latest["id"] == "sft9-vision6k-crop-sft27-teacher"
    assert latest["benchmarks"]["VStar"]["correct"] == 179
    assert latest["benchmarks"]["ZoomBench"]["correct"] == 537
    assert latest["macro_percent"] == 74.87136466710328
