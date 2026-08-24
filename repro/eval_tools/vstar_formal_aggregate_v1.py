#!/usr/bin/env python3
"""Gold-isolated, aggregate-only VStar formal scorer.

The candidate answer is a private intermediate produced by the pinned
Vision-OPD evaluator.  This program consumes it together with the frozen
first-option scorer, but publishes only overall/direct/relative totals.  No
question text, image path, response, sample identifier, or prediction is
copied into the authority or printed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import score_formal_vstar_v1 as formal_score


SCHEMA_VERSION = "vstar_formal_aggregate_v1"
EXPECTED_UID = 30853
EXPECTED_TOTAL = 191
CATEGORY_TOTALS = {"direct_attributes": 115, "relative_position": 76}
PUBLIC_BASELINE_CORRECT = 175
PAPER_EQUIVALENT_CORRECT = 181
STRICT_BEAT_CORRECT = 182
WORKSPACE = Path("/minimax-3d-rw-backup/users/jiazhi/H_Workspace")
OUTPUT_ROOT = WORKSPACE / "Output"
FROZEN_SCORER_SHA256 = "51fb817d51c27a51e4dd82fec059a0ed0a548925d54bc6e2829c7c323e67364b"


class AggregateError(RuntimeError):
    """Malformed, incomplete or unsafe formal evidence."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular(path: Path, *, label: str) -> Path:
    candidate = path.expanduser().absolute()
    try:
        info = candidate.lstat()
    except OSError as error:
        raise AggregateError(f"{label} is unavailable") from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise AggregateError(f"{label} must be a single-link regular file")
    if info.st_size <= 0:
        raise AggregateError(f"{label} is empty")
    return candidate


def _load_candidate(path: Path) -> tuple[list[dict[str, Any]], str]:
    source = _regular(path, label="candidate answer")
    if source.suffix.lower() != ".jsonl":
        raise AggregateError("candidate answer must be JSONL")
    try:
        rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AggregateError("candidate answer cannot be parsed") from error
    if len(rows) != EXPECTED_TOTAL or any(not isinstance(row, dict) for row in rows):
        raise AggregateError("candidate answer must contain exactly 191 object rows")
    return rows, _sha256(source)


def aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Use the frozen scorer internally and return counts only."""

    frozen, scorer_path = formal_score._load_frozen_scorer()
    if _sha256(scorer_path) != FROZEN_SCORER_SHA256:
        raise AggregateError("frozen scorer source hash changed")
    seen: Counter[str] = Counter()
    correct: Counter[str] = Counter()
    for row in rows:
        category = row.get("category")
        if category not in CATEGORY_TOTALS:
            raise AggregateError("candidate category totals differ")
        if "response" not in row or "model_answer" not in row:
            raise AggregateError("candidate lacks scorer fields")
        try:
            gold = frozen._first_option(row["response"], source=Path("<private>"), field="response")
            predicted = frozen._first_option(row["model_answer"], source=Path("<private>"), field="model_answer")
        except Exception as error:
            raise AggregateError("frozen scorer rejected candidate answer") from error
        seen[category] += 1
        correct[category] += int(gold == predicted)
    if dict(seen) != CATEGORY_TOTALS:
        raise AggregateError("candidate category totals differ")
    total_correct = sum(correct.values())
    return {
        "correct": total_correct,
        "total": EXPECTED_TOTAL,
        "accuracy_percent": total_correct / EXPECTED_TOTAL * 100.0,
        "categories": {
            category: {"correct": correct[category], "total": CATEGORY_TOTALS[category]}
            for category in CATEGORY_TOTALS
        },
    }


def build_receipt(*, rows: Sequence[Mapping[str, Any]], model_tag: str, cache_key: str, candidate_sha256: str) -> dict[str, Any]:
    evaluation = aggregate_rows(rows)
    correct = evaluation["correct"]
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "strict_beat" if correct >= STRICT_BEAT_CORRECT else "scored_below_strict_beat",
        "benchmark": "vstar",
        "model_tag": model_tag,
        "protocol": {
            "seed": 42,
            "enable_thinking": False,
            "temperature": 0,
            "max_tokens": 32768,
            "cuda_visible_devices": "0,1,2,3,4,5,6,7",
            "tensor_parallel_size": 8,
            "gold_scope": "frozen_scorer_internal_only",
            "sample_level_output": False,
        },
        "cache": {"key": cache_key, "scope": "uid30853_private_model_backend_runtime"},
        "candidate_artifact": {"rows": EXPECTED_TOTAL, "sha256": candidate_sha256},
        "evaluation": evaluation,
        "reference_points": {
            "local_public_checkpoint": {"correct": PUBLIC_BASELINE_CORRECT, "total": EXPECTED_TOTAL, "gain": correct - PUBLIC_BASELINE_CORRECT},
            "paper_equivalent": {"correct": PAPER_EQUIVALENT_CORRECT, "total": EXPECTED_TOTAL, "gain": correct - PAPER_EQUIVALENT_CORRECT},
            "strict_beat": {"correct": STRICT_BEAT_CORRECT, "total": EXPECTED_TOTAL, "reached": correct >= STRICT_BEAT_CORRECT},
        },
    }


def _write_create_once(path: Path, value: Mapping[str, Any]) -> None:
    target = path.expanduser().absolute()
    try:
        target.relative_to(OUTPUT_ROOT)
    except ValueError as error:
        raise AggregateError("output must be below H_Workspace/Output") from error
    if target.exists() or target.is_symlink():
        raise AggregateError("aggregate authority already exists")
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-answer", required=True, type=Path)
    parser.add_argument("--model-tag", required=True)
    parser.add_argument("--cache-key", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if not args.model_tag or any(char.isspace() for char in args.model_tag):
            raise AggregateError("model tag is invalid")
        if not args.cache_key or any(char.isspace() for char in args.cache_key):
            raise AggregateError("cache key is invalid")
        output = args.output.expanduser().absolute()
        try:
            output.relative_to(OUTPUT_ROOT)
        except ValueError as error:
            raise AggregateError("output must be below H_Workspace/Output") from error
        if output.exists() or output.is_symlink():
            raise AggregateError("aggregate authority already exists")
        if args.dry_run:
            print(json.dumps({"schema_version": SCHEMA_VERSION, "dry_run": True, "reads_candidate": False, "writes_output": False, "aggregate_only": True}, sort_keys=True))
            return 0
        if os.geteuid() != EXPECTED_UID or os.getegid() != EXPECTED_UID:
            raise AggregateError("formal execution requires UID/GID 30853")
        rows, candidate_sha256 = _load_candidate(args.candidate_answer)
        receipt = build_receipt(rows=rows, model_tag=args.model_tag, cache_key=args.cache_key, candidate_sha256=candidate_sha256)
        _write_create_once(output, receipt)
        evaluation = receipt["evaluation"]
        direct = evaluation["categories"]["direct_attributes"]
        relative = evaluation["categories"]["relative_position"]
        print(f"VSTAR_FORMAL_AGGREGATE correct={evaluation['correct']} total={evaluation['total']} direct={direct['correct']}/{direct['total']} relative={relative['correct']}/{relative['total']} strict_beat={str(receipt['reference_points']['strict_beat']['reached']).lower()}")
        return 0
    except AggregateError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
