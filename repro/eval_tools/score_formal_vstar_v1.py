#!/usr/bin/env python3
"""Fail-closed, CPU-only VStar scoring and baseline comparison.

This tool is intentionally a small evidence boundary around the already
frozen workspace scorer.  It does not run inference, import CUDA/vLLM/Ray,
call a judge, access the network, or modify an existing artifact.  Both the
candidate and the official baseline must be answer JSONL files with the same
191 question ids.  The baseline artifact is checked against the baseline
answer before any comparison is emitted.

The command writes one deterministic JSON receipt with exclusive/create-once
semantics::

    python score_formal_vstar_v1.py \
        --candidate-answer candidate.jsonl \
        --official-baseline-answer public.jsonl \
        --official-baseline-artifact public.frozen_rule_score.json \
        --json-out formal_vstar_comparison.json

The score is the frozen first-option rule-only diagnostic.  It is not an LLM
judge result, and the paper number is reported only as a reference point.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import stat
import sys
from pathlib import Path
from typing import Any, Mapping


EXPECTED_TOTAL = 191
CATEGORY_TOTALS = {"direct_attributes": 115, "relative_position": 76}
PUBLIC_CHECKPOINT_CORRECT = 175
PAPER_REPORTED_CORRECT_EQUIVALENT = 181
STRICT_BEAT_CORRECT = 182
FROZEN_SCORER_SHA256 = (
    "51fb817d51c27a51e4dd82fec059a0ed0a548925d54bc6e2829c7c323e67364b"
)
FROZEN_SCORER_NAME = "summarize_b3_pilot384_results.py"
SCHEMA_VERSION = "formal_vstar_comparison_v1"
_QID_RE = re.compile(r"^[0-9]+$")


class FormalVStarError(RuntimeError):
    """A missing, malformed, changed, or inconsistent evidence input."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular_file(path: Path | str, *, label: str) -> Path:
    candidate = Path(path).expanduser().absolute()
    try:
        info = candidate.lstat()
    except OSError as error:
        raise FormalVStarError(f"{label} is unavailable: {candidate}") from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise FormalVStarError(f"{label} must be a non-symlink regular file: {candidate}")
    if info.st_size <= 0:
        raise FormalVStarError(f"{label} is empty: {candidate}")
    return candidate


def _load_frozen_scorer() -> tuple[Any, Path]:
    """Load the scorer beside this script and verify its exact source bytes."""

    scorer_path = Path(__file__).resolve().with_name(FROZEN_SCORER_NAME)
    scorer_path = _regular_file(scorer_path, label="frozen scorer")
    actual = _sha256(scorer_path)
    if actual != FROZEN_SCORER_SHA256:
        raise FormalVStarError(
            "frozen scorer SHA256 changed: "
            f"expected {FROZEN_SCORER_SHA256}, got {actual} ({scorer_path})"
        )
    spec = importlib.util.spec_from_file_location("frozen_vstar_scorer", scorer_path)
    if spec is None or spec.loader is None:
        raise FormalVStarError(f"cannot load frozen scorer: {scorer_path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as error:  # pragma: no cover - defensive import boundary
        raise FormalVStarError(f"cannot import frozen scorer: {scorer_path}") from error
    if not callable(getattr(module, "parse_result", None)):
        raise FormalVStarError("frozen scorer has no callable parse_result")
    return module, scorer_path


def _read_strict_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    """Read real JSONL; arrays disguised with a .jsonl suffix are rejected."""

    if path.suffix.lower() != ".jsonl":
        raise FormalVStarError(f"{label} must have a .jsonl suffix: {path}")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise FormalVStarError(f"cannot read {label}: {path}") from error
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise FormalVStarError(
                f"cannot parse {label} JSONL record {line_number}: {path}"
            ) from error
        if not isinstance(value, dict):
            raise FormalVStarError(
                f"{label} JSONL record {line_number} must be an object: {path}"
            )
        records.append(value)
    if len(records) != EXPECTED_TOTAL:
        raise FormalVStarError(
            f"{label} must contain exactly {EXPECTED_TOTAL} non-empty JSONL rows, "
            f"got {len(records)}: {path}"
        )
    return records


def _qid(value: Any, *, index: int, source: Path) -> str:
    if isinstance(value, bool) or value is None:
        raise FormalVStarError(f"question_id is missing/invalid at row {index + 1}: {source}")
    result = str(value).strip()
    if not result:
        raise FormalVStarError(f"question_id is empty at row {index + 1}: {source}")
    return result


def _qid_output(value: str) -> int | str:
    """Match the existing artifact convention for numeric question ids."""

    if _QID_RE.fullmatch(value) and (value == "0" or not value.startswith("0")):
        return int(value)
    return value


def _ordered_qids(values: set[str]) -> list[int | str]:
    numeric = [value for value in values if _QID_RE.fullmatch(value)]
    other = sorted(value for value in values if not _QID_RE.fullmatch(value))
    return [int(value) for value in sorted(numeric, key=int)] + other


def _correctness(
    rows: list[dict[str, Any]],
    *,
    source: Path,
    frozen: Any,
) -> dict[str, Any]:
    if len(rows) != EXPECTED_TOTAL:
        raise FormalVStarError(f"internal row-count mismatch for {source}")
    by_qid: dict[str, dict[str, Any]] = {}
    category_counts = {name: 0 for name in CATEGORY_TOTALS}
    category_correct = {name: 0 for name in CATEGORY_TOTALS}
    errors: set[str] = set()
    frozen_correct = 0
    for index, row in enumerate(rows):
        qid = _qid(row.get("question_id"), index=index, source=source)
        if qid in by_qid:
            raise FormalVStarError(f"duplicate question_id {qid!r}: {source}")
        category = row.get("category")
        if category not in CATEGORY_TOTALS:
            raise FormalVStarError(
                f"row {index + 1} has unsupported category {category!r}: {source}"
            )
        for field in ("response", "model_answer"):
            if field not in row:
                raise FormalVStarError(f"row {index + 1} lacks {field!r}: {source}")
        try:
            gold = frozen._first_option(row["response"], source=source, field="response")
            prediction = frozen._first_option(
                row["model_answer"], source=source, field="model_answer"
            )
        except Exception as error:
            raise FormalVStarError(
                f"frozen first-option scoring failed at question_id {qid!r}: {source}"
            ) from error
        correct = gold == prediction
        category_counts[category] += 1
        category_correct[category] += int(correct)
        frozen_correct += int(correct)
        if not correct:
            errors.add(qid)
        by_qid[qid] = {
            "question_id": qid,
            "category": category,
            "correct": correct,
        }
    if category_counts != CATEGORY_TOTALS:
        raise FormalVStarError(
            f"category totals must be {CATEGORY_TOTALS}, got {category_counts}: {source}"
        )
    return {
        "correct": frozen_correct,
        "total": EXPECTED_TOTAL,
        "accuracy": frozen_correct / EXPECTED_TOTAL,
        "accuracy_percent": frozen_correct / EXPECTED_TOTAL * 100.0,
        "categories": {
            category: {
                "correct": category_correct[category],
                "total": CATEGORY_TOTALS[category],
            }
            for category in CATEGORY_TOTALS
        },
        "error_question_ids": _ordered_qids(errors),
        "_by_qid": by_qid,
    }


def _parse_answer(
    path: Path | str,
    *,
    label: str,
    frozen: Any,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    source = _regular_file(path, label=label)
    rows = _read_strict_jsonl(source, label=label)
    try:
        parsed = frozen.parse_result(source, metric="vstar")
    except Exception as error:
        raise FormalVStarError(f"frozen scorer rejected {label}: {source}") from error
    if parsed.get("total") != EXPECTED_TOTAL:
        raise FormalVStarError(f"frozen scorer did not report 191 rows: {source}")
    if parsed.get("comparable_to_frozen_rule_only") is not True:
        raise FormalVStarError(f"{label} is not comparable frozen rule-only evidence: {source}")
    result = _correctness(rows, source=source, frozen=frozen)
    if parsed.get("correct") != result["correct"]:
        raise FormalVStarError(
            f"frozen scorer count disagrees with row-level count for {label}: {source}"
        )
    result["answer_jsonl"] = {
        "path": str(source),
        "rows": EXPECTED_TOTAL,
        "sha256": _sha256(source),
    }
    result.pop("_by_qid")
    return source, result, {**parsed, "rows": rows}


def _artifact_value(payload: Mapping[str, Any], path: tuple[str, ...], *, label: str) -> Any:
    value: Any = payload
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            raise FormalVStarError(f"official baseline artifact lacks {label}")
        value = value[key]
    return value


def _validate_official_artifact(
    artifact_path: Path | str,
    *,
    baseline_source: Path,
    baseline_result: Mapping[str, Any],
    scorer_path: Path,
) -> tuple[Path, dict[str, Any]]:
    source = _regular_file(artifact_path, label="official baseline artifact")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FormalVStarError(f"cannot parse official baseline artifact: {source}") from error
    if not isinstance(payload, dict):
        raise FormalVStarError("official baseline artifact must be a JSON object")
    if _artifact_value(payload, ("benchmark",), label="benchmark") != "vstar":
        raise FormalVStarError("official baseline artifact benchmark is not vstar")
    scorer_sha = _artifact_value(payload, ("scoring", "scorer_sha256"), label="scorer SHA256")
    if scorer_sha != FROZEN_SCORER_SHA256:
        raise FormalVStarError("official baseline artifact scorer SHA256 differs")
    scorer_declared = _artifact_value(payload, ("scoring", "scorer"), label="scorer path")
    if Path(str(scorer_declared)).name != scorer_path.name:
        raise FormalVStarError("official baseline artifact scorer path differs")
    answer_meta = _artifact_value(payload, ("evaluation", "answer_jsonl"), label="answer identity")
    if not isinstance(answer_meta, Mapping):
        raise FormalVStarError("official baseline artifact answer identity is malformed")
    declared_path = answer_meta.get("path")
    if not isinstance(declared_path, str) or Path(declared_path).expanduser().absolute() != baseline_source:
        raise FormalVStarError("official baseline artifact answer path differs")
    if answer_meta.get("rows") != EXPECTED_TOTAL:
        raise FormalVStarError("official baseline artifact answer row count differs")
    if answer_meta.get("sha256") != baseline_result["answer_jsonl"]["sha256"]:
        raise FormalVStarError("official baseline artifact answer SHA256 differs")
    artifact_result = _artifact_value(payload, ("result",), label="result")
    if not isinstance(artifact_result, Mapping):
        raise FormalVStarError("official baseline artifact result is malformed")
    for key in ("correct", "total", "categories", "error_question_ids"):
        if key not in artifact_result:
            raise FormalVStarError(f"official baseline artifact result lacks {key}")
    if artifact_result["correct"] != baseline_result["correct"]:
        raise FormalVStarError("official baseline artifact total correct differs")
    if artifact_result["total"] != EXPECTED_TOTAL:
        raise FormalVStarError("official baseline artifact total differs")
    if artifact_result["categories"] != baseline_result["categories"]:
        raise FormalVStarError("official baseline artifact category scores differ")
    if artifact_result["error_question_ids"] != baseline_result["error_question_ids"]:
        raise FormalVStarError("official baseline artifact error qids differ")
    return source, payload


def _comparison(
    candidate: Mapping[str, Any],
    baseline: Mapping[str, Any],
    *,
    candidate_rows: Mapping[str, Mapping[str, Any]],
    baseline_rows: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    candidate_ids = set(candidate_rows)
    baseline_ids = set(baseline_rows)
    if candidate_ids != baseline_ids:
        missing = _ordered_qids(baseline_ids - candidate_ids)
        extra = _ordered_qids(candidate_ids - baseline_ids)
        raise FormalVStarError(
            f"candidate question ids differ from official baseline (missing={missing}, extra={extra})"
        )
    gains: set[str] = set()
    regressions: set[str] = set()
    for qid in candidate_ids:
        candidate_correct = bool(candidate_rows[qid]["correct"])
        baseline_correct = bool(baseline_rows[qid]["correct"])
        if candidate_correct and not baseline_correct:
            gains.add(qid)
        elif baseline_correct and not candidate_correct:
            regressions.add(qid)

    def category_delta(category: str) -> dict[str, Any]:
        cand = candidate["categories"][category]
        base = baseline["categories"][category]
        return {
            "candidate_correct": cand["correct"],
            "baseline_correct": base["correct"],
            "correct_gain": cand["correct"] - base["correct"],
            "total": cand["total"],
            "accuracy_delta_pp": (cand["correct"] - base["correct"]) / cand["total"] * 100.0,
        }

    gain_categories = {category: [] for category in CATEGORY_TOTALS}
    regression_categories = {category: [] for category in CATEGORY_TOTALS}
    for qid in gains:
        gain_categories[candidate_rows[qid]["category"]].append(qid)
    for qid in regressions:
        regression_categories[candidate_rows[qid]["category"]].append(qid)
    return {
        "correct_gain": candidate["correct"] - baseline["correct"],
        "accuracy_delta_pp": (candidate["correct"] - baseline["correct"]) / EXPECTED_TOTAL * 100.0,
        "beats_official_baseline": candidate["correct"] > baseline["correct"],
        "gains": {
            "count": len(gains),
            "question_ids": _ordered_qids(gains),
            "by_category": {
                category: _ordered_qids(set(values))
                for category, values in gain_categories.items()
            },
        },
        "regressions": {
            "count": len(regressions),
            "question_ids": _ordered_qids(regressions),
            "by_category": {
                category: _ordered_qids(set(values))
                for category, values in regression_categories.items()
            },
        },
        "categories": {
            category: category_delta(category) for category in CATEGORY_TOTALS
        },
    }


def _public_references(candidate_correct: int) -> dict[str, Any]:
    return {
        "public_checkpoint": {
            "correct": PUBLIC_CHECKPOINT_CORRECT,
            "total": EXPECTED_TOTAL,
            "accuracy_percent": PUBLIC_CHECKPOINT_CORRECT / EXPECTED_TOTAL * 100.0,
            "candidate_correct_gain": candidate_correct - PUBLIC_CHECKPOINT_CORRECT,
            "candidate_beats": candidate_correct > PUBLIC_CHECKPOINT_CORRECT,
        },
        "paper_reported_equivalent": {
            "correct": PAPER_REPORTED_CORRECT_EQUIVALENT,
            "total": EXPECTED_TOTAL,
            "accuracy_percent": PAPER_REPORTED_CORRECT_EQUIVALENT / EXPECTED_TOTAL * 100.0,
            "candidate_correct_gain": candidate_correct - PAPER_REPORTED_CORRECT_EQUIVALENT,
            "candidate_reaches": candidate_correct >= PAPER_REPORTED_CORRECT_EQUIVALENT,
        },
        "strict_beat_threshold": {
            "correct": STRICT_BEAT_CORRECT,
            "total": EXPECTED_TOTAL,
            "accuracy_percent": STRICT_BEAT_CORRECT / EXPECTED_TOTAL * 100.0,
            "candidate_correct_gain": candidate_correct - STRICT_BEAT_CORRECT,
            "candidate_beats": candidate_correct >= STRICT_BEAT_CORRECT,
        },
    }


def build_receipt(
    candidate_answer: Path | str,
    official_baseline_answer: Path | str,
    official_baseline_artifact: Path | str,
) -> dict[str, Any]:
    """Score candidate and validate/compare it to the pinned official baseline."""

    frozen, scorer_path = _load_frozen_scorer()
    candidate_source, candidate_result, candidate_parsed = _parse_answer(
        candidate_answer, label="candidate answer", frozen=frozen
    )
    baseline_source, baseline_result, baseline_parsed = _parse_answer(
        official_baseline_answer, label="official baseline answer", frozen=frozen
    )
    artifact_source, artifact_payload = _validate_official_artifact(
        official_baseline_artifact,
        baseline_source=baseline_source,
        baseline_result=baseline_result,
        scorer_path=scorer_path,
    )
    candidate_rows = _correctness(
        candidate_parsed["rows"], source=candidate_source, frozen=frozen
    )["_by_qid"]
    baseline_rows = _correctness(
        baseline_parsed["rows"], source=baseline_source, frozen=frozen
    )["_by_qid"]
    comparison = _comparison(
        candidate_result,
        baseline_result,
        candidate_rows=candidate_rows,
        baseline_rows=baseline_rows,
    )
    candidate_output = dict(candidate_result)
    baseline_output = dict(baseline_result)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "strict_beat" if candidate_result["correct"] >= STRICT_BEAT_CORRECT else "scored_below_strict_beat",
        "benchmark": "vstar",
        "scoring": {
            "name": "frozen first-option rule-only diagnostic",
            "metric": "vstar",
            "scorer": str(scorer_path),
            "scorer_sha256": FROZEN_SCORER_SHA256,
            "cpu_only": True,
            "network": False,
            "llm_judge": False,
        },
        "candidate": candidate_output,
        "official_baseline": {
            "answer": baseline_output,
            "artifact": {
                "path": str(artifact_source),
                "sha256": _sha256(artifact_source),
                "result": artifact_payload["result"],
            },
        },
        "comparison": comparison,
        "reference_points": _public_references(candidate_result["correct"]),
        "warnings": [
            "VStar is scored by the frozen first-option rule-only diagnostic; no LLM judge was used.",
            "The paper-reported 181/191 value is an equivalent reference point, not a local rerun.",
            "The public checkpoint baseline is independently reproduced local evidence and is validated against its sealed artifact.",
        ],
    }


def _write_create_once(path: Path | str, payload: str) -> None:
    target = Path(path).expanduser().absolute()
    if target.exists() or target.is_symlink():
        if target.is_file() and not target.is_symlink() and target.read_text(encoding="utf-8") == payload:
            return
        raise FormalVStarError(f"output already exists with different bytes: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
    except FileExistsError as error:
        raise FormalVStarError(f"output was created concurrently: {target}") from error


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-answer", required=True, type=Path)
    parser.add_argument("--official-baseline-answer", required=True, type=Path)
    parser.add_argument("--official-baseline-artifact", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        receipt = build_receipt(
            args.candidate_answer,
            args.official_baseline_answer,
            args.official_baseline_artifact,
        )
        _write_create_once(
            args.json_out,
            json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
    except FormalVStarError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
