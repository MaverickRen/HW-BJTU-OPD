from __future__ import annotations

import json
import re
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


def test_released_opd9_teacher_receipts_are_cross_bound() -> None:
    released = json.loads((ROOT / "results/released_opd9_teacher.json").read_text(encoding="utf-8"))
    config = json.loads((ROOT / "configs/opd9_teacher_artifact.json").read_text(encoding="utf-8"))
    artifacts = json.loads((ROOT / "results/artifact_manifest.json").read_text(encoding="utf-8"))
    evidence = json.loads((ROOT / "results/evidence_manifest.json").read_text(encoding="utf-8"))

    revision = released["artifact"]["revision"]
    shell = (ROOT / "scripts/evaluate_vstar.sh").read_text(encoding="utf-8")
    assert re.search(rf'^model_revision="{revision}"$', shell, re.MULTILINE)
    assert config["release_revision"] == revision
    checkpoint = artifacts["checkpoints"]["qwen35-9b-sft10k-visionopd6k-sft9b-teacher"]
    assert checkpoint["release_revision"] == revision
    assert evidence["released_artifact"]["revision"] == revision

    assert released["artifact"]["model_sha256"] == config["model"]["sha256"]
    assert checkpoint["weight_files"][0]["sha256"] == config["model"]["sha256"]
    scores = released["benchmarks"]
    assert {name: cell["correct"] for name, cell in scores.items()} == {
        "VStar": 176,
        "MMStar": 1159,
        "BLINK-v5": 1263,
        "ZoomBench": 525,
    }
    assert abs(sum(cell["percent"] for cell in scores.values()) / 4 - released["four_benchmark_macro_percent"]) < 1e-12

    zoom = scores["ZoomBench"]
    assert (
        zoom["dataset"]["source_parquet_sha256"] == "d44ebda2eda485cba055181f4e6dc50c42f81b5d0f7e936bf427fa01502a391a"
    )
    assert zoom["dataset"]["official_eval_commit"] == "fdc0ba1a3dee916d8c38304d543ad414879e0c99"
    assert zoom["semantic_judge"] == {
        "repo_id": "Qwen/Qwen3.5-27B",
        "revision": "fc05daec18b0a78c049392ed2e771dde82bdf654",
        "identity_sha256": "a9e26e80efcdf9eea4bb01544d817cd4fac0db4241c545ab24666b3ba49c003c",
        "acceptance_rule": "strip_lower_equals_yes",
    }


def test_public_vstar_gpu_validation_receipt_is_cross_bound() -> None:
    validation = json.loads((ROOT / "results/vstar_reproduction_validation.json").read_text(encoding="utf-8"))
    released = json.loads((ROOT / "results/released_opd9_teacher.json").read_text(encoding="utf-8"))
    evidence = json.loads((ROOT / "results/evidence_manifest.json").read_text(encoding="utf-8"))

    assert validation["status"] == "complete"
    assert validation["aggregate_only"] is True
    assert validation["sample_level_output"] is False
    assert validation["raw_predictions_persisted"] is False
    assert validation["artifact"] == {key: released["artifact"][key] for key in ("repo_id", "revision", "model_sha256")}

    public_gate = evidence["public_vstar_gate"]
    for key in (
        "dataset_repo_id",
        "dataset_revision",
        "source_parquet_sha256",
        "logical_rows_sha256",
        "image_manifest_sha256",
    ):
        receipt_key = {"dataset_repo_id": "repo_id", "dataset_revision": "revision"}.get(key, key)
        assert validation["dataset"][receipt_key] == public_gate[key]

    result = validation["result"]
    assert (result["correct"], result["total"]) == (176, 191)
    assert result["accuracy_percent"] == 176 / 191 * 100
    assert result["invalid_count"] == 0
    assert validation["reference"] == {
        "correct": public_gate["reference"]["correct"],
        "total": public_gate["reference"]["total"],
        "tolerance_correct": public_gate["similarity_tolerance_correct"],
    }
    assert validation["reproduction"] == {"correct_delta": 0, "similar": True}

    vstar_release = released["benchmarks"]["VStar"]
    assert validation["protocol"]["name"] == vstar_release["protocol"]
    assert validation["protocol"]["sha256"] == vstar_release["protocol_hash"]
    assert validation["hardware"] == {
        "nodes": 1,
        "gpu_model": "NVIDIA L20C",
        "gpu_count": 4,
        "tensor_parallel_size": 4,
    }

    receipt_text = (ROOT / "results/vstar_reproduction_validation.json").read_text(encoding="utf-8")
    assert "/minimax-3d-rw-backup/" not in receipt_text
    assert not re.search(r"j-[a-z0-9]+", receipt_text)
    assert not re.search(r"(?:hf_|ghp_)[A-Za-z0-9]{20,}", receipt_text)
    assert '"predictions"' not in receipt_text
