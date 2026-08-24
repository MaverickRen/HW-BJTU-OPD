#!/usr/bin/env python3
"""Strict, read-only gate for a pinned ZoomBench inference checkpoint.

Exit codes are part of the shell-wrapper contract:

* 0: the checkpoint is complete and strictly valid;
* 1: the benchmark contract is valid, but the checkpoint is incomplete/invalid;
* 2: invocation, benchmark, or verifier I/O failed.

The answer JSONL is never repaired or rewritten here.  The reference inference
program owns checkpoint recovery; this gate only decides whether another pass
is required and whether judging is safe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


EXPECTED_ROWS = 845
BENCHMARK_NAME = "zoombench"
ERROR_MARKERS = ("api_error", "future_error")
DETAIL_LIMIT = 20
SCHEMA_VERSION = "1.0"
IGNORED_ANSWER_KEYS = frozenset({"sample_uid", "model_answer"})


class ContractInputError(ValueError):
    """The benchmark or command invocation cannot support a safe decision."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def absolute(path: Path) -> str:
    return str(path.absolute())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_sample_uid(item: dict[str, Any], benchmark: str = BENCHMARK_NAME) -> str:
    """Mirror the UID contract in the pinned Vision-OPD inference program."""
    for key in ("sample_uid", "uid", "index", "question_id", "id"):
        value = item.get(key)
        if value is not None and str(value) != "":
            return f"{benchmark}:{key}:{value}"
    stable_obj = {
        "benchmark": benchmark,
        "images": item.get("images") or [],
        "query": item.get("query", ""),
    }
    raw = json.dumps(stable_obj, ensure_ascii=False, sort_keys=True)
    return "sha1:" + hashlib.sha1(raw.encode("utf-8")).hexdigest()


def benchmark_identity(record: dict[str, Any]) -> dict[str, Any]:
    """Return the benchmark-owned payload carried through by inference."""
    return {key: value for key, value in record.items() if key not in IGNORED_ANSWER_KEYS}


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def identity_mismatch_summary(
    expected: dict[str, Any], actual: dict[str, Any]
) -> dict[str, list[str]]:
    expected_keys = set(expected)
    actual_keys = set(actual)
    return {
        "missing_keys": sorted(expected_keys - actual_keys),
        "extra_keys": sorted(actual_keys - expected_keys),
        "changed_keys": sorted(
            key
            for key in expected_keys.intersection(actual_keys)
            if canonical_json(expected[key]) != canonical_json(actual[key])
        ),
    }


def load_benchmark(
    path: Path,
) -> tuple[list[dict[str, Any]], list[str], dict[str, dict[str, Any]], str]:
    if not path.is_file():
        raise ContractInputError(f"benchmark JSON does not exist: {absolute(path)}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ContractInputError(f"could not read benchmark JSON: {error}") from error
    if not isinstance(payload, list):
        raise ContractInputError("benchmark JSON must contain a list")
    if len(payload) != EXPECTED_ROWS:
        raise ContractInputError(
            f"benchmark must contain exactly {EXPECTED_ROWS} rows; got {len(payload)}"
        )

    rows: list[dict[str, Any]] = []
    uids: list[str] = []
    identities: dict[str, dict[str, Any]] = {}
    for row_number, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise ContractInputError(
                f"benchmark row {row_number} is not a JSON object"
            )
        uid = make_sample_uid(item)
        if not uid.strip():
            raise ContractInputError(f"benchmark row {row_number} has an empty UID")
        rows.append(item)
        uids.append(uid)
        identities[uid] = benchmark_identity(item)

    counts = Counter(uids)
    duplicates = sorted(uid for uid, count in counts.items() if count > 1)
    if duplicates:
        sample = ", ".join(repr(uid) for uid in duplicates[:DETAIL_LIMIT])
        raise ContractInputError(
            f"benchmark UIDs are not unique ({len(duplicates)} duplicated): {sample}"
        )
    try:
        digest = sha256_file(path)
    except OSError as error:
        raise ContractInputError(f"could not hash benchmark JSON: {error}") from error
    return rows, uids, identities, digest


def _sample_append(values: list[Any], value: Any) -> None:
    if len(values) < DETAIL_LIMIT:
        values.append(value)


def inspect_answers(
    path: Path,
    expected_uids: Sequence[str],
    expected_identities: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    expected_set = set(expected_uids)
    result: dict[str, Any] = {
        "path": absolute(path),
        "exists": path.is_file(),
        "expected_rows": EXPECTED_ROWS,
        "physical_line_count": 0,
        "json_object_count": 0,
        "unique_expected_uid_count": 0,
        "blank_line_count": 0,
        "invalid_json_count": 0,
        "non_object_count": 0,
        "missing_uid_count": 0,
        "duplicate_uid_count": 0,
        "unknown_uid_count": 0,
        "payload_mismatch_count": 0,
        "missing_expected_uid_count": EXPECTED_ROWS,
        "empty_answer_count": 0,
        "error_answer_count": 0,
        "retry_uid_count": EXPECTED_ROWS,
        "passed": False,
        "errors": [],
    }
    if not path.is_file():
        result["errors"].append("answer_jsonl_missing")
        result["missing_expected_uids_sample"] = list(expected_uids[:DETAIL_LIMIT])
        result["retry_uids_sample"] = list(expected_uids[:DETAIL_LIMIT])
        return result

    seen_counts: Counter[str] = Counter()
    acceptable_uids: set[str] = set()
    blank_lines: list[int] = []
    invalid_lines: list[dict[str, Any]] = []
    non_object_lines: list[int] = []
    missing_uid_lines: list[int] = []
    duplicate_uids: list[dict[str, Any]] = []
    unknown_uids: list[dict[str, Any]] = []
    payload_mismatches: list[dict[str, Any]] = []
    empty_answers: list[dict[str, Any]] = []
    error_answers: list[dict[str, Any]] = []

    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                result["physical_line_count"] += 1
                if not line.strip():
                    result["blank_line_count"] += 1
                    _sample_append(blank_lines, line_number)
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    result["invalid_json_count"] += 1
                    _sample_append(
                        invalid_lines,
                        {"line": line_number, "detail": str(error)},
                    )
                    continue
                if not isinstance(record, dict):
                    result["non_object_count"] += 1
                    _sample_append(non_object_lines, line_number)
                    continue
                result["json_object_count"] += 1

                raw_uid = record.get("sample_uid")
                uid = raw_uid.strip() if isinstance(raw_uid, str) else ""
                if not uid:
                    result["missing_uid_count"] += 1
                    _sample_append(missing_uid_lines, line_number)
                    continue

                seen_counts[uid] += 1
                if seen_counts[uid] > 1:
                    result["duplicate_uid_count"] += 1
                    _sample_append(
                        duplicate_uids,
                        {"line": line_number, "sample_uid": uid},
                    )
                if uid not in expected_set:
                    result["unknown_uid_count"] += 1
                    _sample_append(
                        unknown_uids,
                        {"line": line_number, "sample_uid": uid},
                    )

                payload_matches = False
                if uid in expected_set:
                    expected_identity = expected_identities[uid]
                    actual_identity = benchmark_identity(record)
                    payload_matches = canonical_json(
                        actual_identity
                    ) == canonical_json(expected_identity)
                    if not payload_matches:
                        result["payload_mismatch_count"] += 1
                        _sample_append(
                            payload_mismatches,
                            {
                                "line": line_number,
                                "sample_uid": uid,
                                **identity_mismatch_summary(
                                    expected_identity, actual_identity
                                ),
                            },
                        )

                answer = record.get("model_answer")
                if not isinstance(answer, str) or not answer.strip():
                    result["empty_answer_count"] += 1
                    _sample_append(
                        empty_answers,
                        {"line": line_number, "sample_uid": uid},
                    )
                    continue
                if any(marker in answer.casefold() for marker in ERROR_MARKERS):
                    result["error_answer_count"] += 1
                    _sample_append(
                        error_answers,
                        {"line": line_number, "sample_uid": uid},
                    )
                    continue
                if uid in expected_set and payload_matches:
                    acceptable_uids.add(uid)
    except (OSError, UnicodeError) as error:
        raise ContractInputError(f"could not read answer JSONL: {error}") from error

    seen_expected = expected_set.intersection(seen_counts)
    missing_expected = [uid for uid in expected_uids if uid not in seen_expected]
    retry_uids = [uid for uid in expected_uids if uid not in acceptable_uids]
    try:
        answer_bytes = path.stat().st_size
    except OSError as error:
        raise ContractInputError(f"could not stat answer JSONL: {error}") from error
    result.update(
        {
            "unique_uid_count": len(seen_counts),
            "unique_expected_uid_count": len(seen_expected),
            "missing_expected_uid_count": len(missing_expected),
            "retry_uid_count": len(retry_uids),
            "blank_lines_sample": blank_lines,
            "invalid_json_sample": invalid_lines,
            "non_object_lines_sample": non_object_lines,
            "missing_uid_lines_sample": missing_uid_lines,
            "duplicate_uids_sample": duplicate_uids,
            "unknown_uids_sample": unknown_uids,
            "payload_mismatches_sample": payload_mismatches,
            "missing_expected_uids_sample": missing_expected[:DETAIL_LIMIT],
            "empty_answers_sample": empty_answers,
            "error_answers_sample": error_answers,
            "retry_uids_sample": retry_uids[:DETAIL_LIMIT],
            "bytes": answer_bytes,
        }
    )

    checks = {
        "physical_line_count": result["physical_line_count"] == EXPECTED_ROWS,
        "json_object_count": result["json_object_count"] == EXPECTED_ROWS,
        "unique_expected_uid_count": result["unique_expected_uid_count"]
        == EXPECTED_ROWS,
        "blank_line_count": result["blank_line_count"] == 0,
        "invalid_json_count": result["invalid_json_count"] == 0,
        "non_object_count": result["non_object_count"] == 0,
        "missing_uid_count": result["missing_uid_count"] == 0,
        "duplicate_uid_count": result["duplicate_uid_count"] == 0,
        "unknown_uid_count": result["unknown_uid_count"] == 0,
        "payload_mismatch_count": result["payload_mismatch_count"] == 0,
        "missing_expected_uid_count": result["missing_expected_uid_count"] == 0,
        "empty_answer_count": result["empty_answer_count"] == 0,
        "error_answer_count": result["error_answer_count"] == 0,
        "retry_uid_count": result["retry_uid_count"] == 0,
    }
    result["checks"] = checks
    result["errors"] = [name for name, passed in checks.items() if not passed]
    result["passed"] = all(checks.values())
    try:
        result["sha256"] = sha256_file(path)
    except OSError as error:
        raise ContractInputError(f"could not hash answer JSONL: {error}") from error
    return result


def build_receipt(benchmark_json: Path, answer_jsonl: Path) -> tuple[dict[str, Any], int]:
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "zoombench_inference_strict_gate",
        "generated_at": utc_now(),
        "status": "input_error",
        "benchmark": {
            "path": absolute(benchmark_json),
            "expected_rows": EXPECTED_ROWS,
            "benchmark_name": BENCHMARK_NAME,
        },
        "answer": {"path": absolute(answer_jsonl), "passed": False},
        "errors": [],
    }
    try:
        _, expected_uids, expected_identities, benchmark_sha256 = load_benchmark(
            benchmark_json
        )
        receipt["benchmark"].update(
            {
                "actual_rows": len(expected_uids),
                "unique_uid_count": len(set(expected_uids)),
                "sha256": benchmark_sha256,
                "passed": True,
            }
        )
        answer = inspect_answers(answer_jsonl, expected_uids, expected_identities)
    except ContractInputError as error:
        receipt["errors"].append(str(error))
        return receipt, 2

    receipt["answer"] = answer
    receipt["errors"] = list(answer["errors"])
    receipt["status"] = "passed" if answer["passed"] else "incomplete"
    return receipt, 0 if answer["passed"] else 1


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, ensure_ascii=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, path)
        except BaseException:
            Path(temporary_name).unlink(missing_ok=True)
            raise
    except OSError as error:
        raise ContractInputError(f"could not write receipt: {error}") from error


def print_summary(receipt: dict[str, Any], stream: Any = sys.stdout) -> None:
    answer = receipt.get("answer", {})
    if receipt.get("status") == "input_error":
        details = "; ".join(str(item) for item in receipt.get("errors", []))
        print(f"ZoomBench verification input error: {details}", file=stream)
        return
    print(
        "ZoomBench verification: "
        f"status={receipt['status']} "
        f"lines={answer.get('physical_line_count', 0)}/{EXPECTED_ROWS} "
        "expected_unique_uids="
        f"{answer.get('unique_expected_uid_count', 0)}/{EXPECTED_ROWS} "
        f"missing={answer.get('missing_expected_uid_count', EXPECTED_ROWS)} "
        f"retry={answer.get('retry_uid_count', EXPECTED_ROWS)} "
        f"blank={answer.get('blank_line_count', 0)} "
        f"bad_json={answer.get('invalid_json_count', 0)} "
        f"duplicate={answer.get('duplicate_uid_count', 0)} "
        f"unknown={answer.get('unknown_uid_count', 0)} "
        f"payload_mismatch={answer.get('payload_mismatch_count', 0)} "
        f"empty={answer.get('empty_answer_count', 0)} "
        f"error={answer.get('error_answer_count', 0)}",
        file=stream,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strictly verify a pinned 845-row ZoomBench answer JSONL"
    )
    parser.add_argument("--benchmark-json", required=True, type=Path)
    parser.add_argument("--answer-jsonl", required=True, type=Path)
    parser.add_argument("--output", type=Path, help="Optional atomic JSON receipt")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    receipt, return_code = build_receipt(args.benchmark_json, args.answer_jsonl)
    if args.output is not None:
        try:
            atomic_write_json(args.output, receipt)
        except ContractInputError as error:
            if not args.quiet:
                print(f"ZoomBench verification input error: {error}", file=sys.stderr)
            return 2
    if not args.quiet:
        print_summary(receipt, sys.stderr if return_code == 2 else sys.stdout)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
