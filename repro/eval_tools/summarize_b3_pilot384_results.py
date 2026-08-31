#!/usr/bin/env python3
"""Fail-closed CPU-only summary for the B3 pilot evaluation.

The evaluator itself is deliberately outside this module.  This program only
reads completed, judged result files and the frozen Base/B1 evidence, then
writes a deterministic JSON and Markdown comparison.  It never imports
CUDA/vLLM/Ray and never starts a process other than the Python interpreter
running this file.

Candidate inputs may be either:

* an official Vision-OPD judge JSON/JSONL list with a ``judge`` field; or
* a small score object containing ``correct``/``total`` (or ``score``); or
* a MUIR-style CSV containing the ``none`` row and ``Overall`` column.

An unjudged answer JSONL is rejected.  This is intentional: an answer file
alone is not an accuracy result, and silently treating it as one would make a
training result impossible to reproduce.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import stat
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "b3_pilot384_result_summary_v1"
WORKSPACE = Path(__file__).resolve().parents[3]
METRICS = ("vstar", "mme-realworld-lite", "muir")
METRIC_LABELS = {
    "vstar": "VStar",
    "mme-realworld-lite": "MME-RealWorld-Lite",
    "muir": "MUIRBench",
}
EXPECTED_TOTALS = {"vstar": 191, "mme-realworld-lite": 1919, "muir": 2600}
RULE_ONLY_SCORING = "frozen first-letter rule-only diagnostic"
OFFICIAL_LLM_JUDGE_SCORING = "official LLM judge"
MUIR_SCORING = "official exact matching"
RULE_ONLY_EVIDENCE_KIND = "first_letter_rule_only"
MUIR_COMPLETION_SCHEMA = "fresh_muir_b3_completion_v8"
MUIR_COMPLETION_KIND = "muir_b3_native_completion_authority"
CONTROL_KEY = "b1_official_protocol_control"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

# These are the only historical numbers used by the default CLI.  VStar/MME
# are rule-only diagnostics from the frozen answer JSONL; MUIR is official
# exact matching from its frozen acc.csv.  Source hashes and row counts are
# checked before these values are exposed in a summary.
FIXED_BASELINES: Mapping[str, Mapping[str, Mapping[str, Any]]] = {
    "base": {
        "vstar": {
            "correct": 164,
            "total": 191,
            "source": "Output/opd_qwen35_9b/b0_base9b_20260806/eval/vision_opd_reference_ctx262k_v2/model_answer/vstar/Qwen3.5-9B_seed42_answer.jsonl",
            "sha256": "aeaeea16938049b4ca2fb1d1e5aa3be15a435249fb80ea47bf069dbfed44bf51",
            "scoring": "frozen first-letter rule-only diagnostic",
        },
        "mme-realworld-lite": {
            "correct": 1064,
            "total": 1919,
            "source": "Output/opd_qwen35_9b/b0_base9b_20260806/eval/vision_opd_reference_ctx262k_v2/model_answer/mme-realworld-lite/Qwen3.5-9B_seed42_answer.jsonl",
            "sha256": "85b3ec8dae0225e7638b9976bda0a022e74b6191691590bc9d3ab59a1aaceef5",
            "scoring": "frozen first-letter rule-only diagnostic",
        },
        "muir": {
            "correct": 922,
            "total": 2600,
            "source": "Output/opd_qwen35_9b/b0_base9b_20260806/eval_muir_fresh_v6_20260808/Qwen3.5-9B/T20260808-073749/Qwen3.5-9B_MUIRBench_acc.csv",
            "sha256": "d699f3509ce49972ae5d0a859d951444750c8f4bc225bca82f83cd8b5e543918",
            "scoring": "official exact matching",
        },
    },
    "b1": {
        "vstar": {
            "correct": 170,
            "total": 191,
            "source": "Output/opd_qwen35_9b/b1_ext27_full_vision6k_s42_launchfix1_20260807/eval/vision_opd_reference_ctx262k_v3_hf_normalized_v1/model_answer/vstar/Qwen3.5-9B-OPD-B1_seed42_answer.jsonl",
            "sha256": "d611fd8c4e896f059c57edb5d271baa9c90ce443fed23e6e9aabf407f6edbbb2",
            "scoring": "frozen first-letter rule-only diagnostic",
        },
        "mme-realworld-lite": {
            "correct": 970,
            "total": 1919,
            "source": "Output/opd_qwen35_9b/b1_ext27_full_vision6k_s42_launchfix1_20260807/eval/vision_opd_reference_ctx262k_v3_hf_normalized_v1/model_answer/mme-realworld-lite/Qwen3.5-9B-OPD-B1_seed42_answer.jsonl",
            "sha256": "fffd0eb9989bcff57f881ab4abcf9f9f866dcba01410f8c69d85b0ebd72376e4",
            "scoring": "frozen first-letter rule-only diagnostic",
        },
        "muir": {
            "correct": 1128,
            "total": 2600,
            "source": "Output/opd_qwen35_9b/b1_ext27_full_vision6k_s42_launchfix1_20260807/eval_muir_fresh_v7_20260808/Qwen3.5-9B-OPD-B1/T20260808-091318/Qwen3.5-9B-OPD-B1_MUIRBench_acc.csv",
            "sha256": "12c195f65b3dbcc826cb6d05fe8d722d7182790e4882557fd750c0539fd2a74b",
            "scoring": "official exact matching",
        },
    },
}


class SummaryError(RuntimeError):
    """A missing, malformed, or inconsistent evidence input."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_file(path: Path | str, *, label: str) -> Path:
    candidate = Path(path).expanduser().absolute()
    try:
        info = candidate.lstat()
    except OSError as error:
        raise SummaryError(f"{label} is unavailable: {candidate}") from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise SummaryError(f"{label} must be a non-symlink regular file: {candidate}")
    if info.st_size <= 0:
        raise SummaryError(f"{label} is empty: {candidate}")
    return candidate


def _finite_score(value: Any, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise SummaryError(f"{label} is not numeric: {value!r}") from error
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise SummaryError(f"{label} must be finite and in [0, 1], got {value!r}")
    return result


def _record(correct: int, total: int, *, source: Path, kind: str) -> dict[str, Any]:
    if not isinstance(correct, int) or isinstance(correct, bool) or correct < 0:
        raise SummaryError(f"invalid correct count in {source}: {correct!r}")
    if not isinstance(total, int) or isinstance(total, bool) or total <= 0:
        raise SummaryError(f"invalid total count in {source}: {total!r}")
    if correct > total:
        raise SummaryError(f"correct exceeds total in {source}: {correct}>{total}")
    return {
        "correct": correct,
        "total": total,
        "score": correct / total,
        "source": str(source),
        "source_sha256": _sha256(source),
        "input_kind": kind,
    }


def _annotate_score_object(
    result: dict[str, Any],
    payload: Mapping[str, Any],
    *,
    metric: str | None,
) -> dict[str, Any]:
    """Attach an explicit scoring protocol to a compact score receipt.

    A count object is not comparable merely because it claims a score.  The
    rule-only protocol additionally needs an explicit evidence-kind marker so
    that an official LLM-judge count cannot accidentally enter the frozen
    first-letter gate.
    """

    if result.get("input_kind") != "score_object" and "scoring" not in payload:
        return result
    scoring = payload.get("scoring")
    if not isinstance(scoring, str) or not scoring.strip():
        if metric == "muir" and _contains_exact_matching(payload):
            scoring = MUIR_SCORING
        else:
            scoring = "untyped score object"
    scoring = scoring.strip()
    evidence = payload.get("evidence_kind", payload.get("rule_only_evidence"))
    evidence_ok = evidence == RULE_ONLY_EVIDENCE_KIND or (
        isinstance(evidence, Mapping)
        and evidence.get("kind") == RULE_ONLY_EVIDENCE_KIND
    )
    result["scoring"] = scoring
    result["comparable_to_frozen_rule_only"] = bool(
        metric in {"vstar", "mme-realworld-lite"}
        and scoring == RULE_ONLY_SCORING
        and evidence_ok
    )
    return result


def _contains_exact_matching(payload: Any) -> bool:
    if isinstance(payload, Mapping):
        if payload.get("judge_model") == "exact_matching" or payload.get("judge") == "exact_matching":
            return True
        return any(_contains_exact_matching(value) for value in payload.values())
    if isinstance(payload, list):
        return any(_contains_exact_matching(value) for value in payload)
    return False


_FIRST_OPTION_RE = re.compile(
    r"^\s*[\(\[\{<\"']*\s*([A-E])(?=$|[\s\)\]\}>\"'.,:;])",
    re.IGNORECASE,
)
_OPTION_MARKER_RE = re.compile(r"\(([A-Z])\)")
_OPTION_PUNCT_RE = re.compile(r"([A-Z])[\.\)\s]")
_OPTION_ANY_RE = re.compile(r"([A-Z])")


def _first_option(value: Any, *, source: Path, field: str) -> str:
    text = str(value)
    if "<answer>" in text:
        start = text.find("<answer>")
        end = text.find("</answer>")
        if start != -1 and end != -1:
            text = text[start + len("<answer>") : end].strip()
    elif "Answer:" in text:
        text = text[text.find("Answer:") :].strip()
    # This is intentionally the same dispatch order as the frozen
    # Vision-OPD judge's extract_first_option: parenthesized choice, then a
    # choice followed by punctuation/space, then the first A-E fallback.
    match = _OPTION_MARKER_RE.search(text)
    if match is None:
        match = _OPTION_PUNCT_RE.search(text)
    if match is None:
        match = _OPTION_ANY_RE.search(text)
    if match is None:
        raise SummaryError(
            f"{field} must contain an uppercase option letter for first-letter scoring: {source}"
        )
    return match.group(1).upper()


def _count_rule_only(records: Iterable[Mapping[str, Any]], *, source: Path) -> tuple[int, int]:
    values = list(records)
    if not values:
        raise SummaryError(f"rule-only result is empty: {source}")
    if any("response" not in item or "model_answer" not in item for item in values):
        raise SummaryError(
            f"first-letter rule-only evidence requires response/model_answer fields: {source}"
        )
    correct = sum(
        _first_option(item["response"], source=source, field="response")
        == _first_option(item["model_answer"], source=source, field="model_answer")
        for item in values
    )
    return int(correct), len(values)


def _count_judged(records: Iterable[Mapping[str, Any]], *, source: Path) -> tuple[int, int]:
    values = list(records)
    if not values:
        raise SummaryError(f"judged result is empty: {source}")
    if any("judge" not in item for item in values):
        raise SummaryError(
            f"unjudged answer file rejected (every record needs 'judge'): {source}"
        )
    correct = sum(str(item.get("judge", "")).strip().lower() == "yes" for item in values)
    recognized = sum(
        str(item.get("judge", "")).strip().lower() in {"yes", "no"} for item in values
    )
    if recognized != len(values):
        raise SummaryError(f"judge values must be Yes/No only: {source}")
    return int(correct), len(values)


def _json_records(
    source: Path,
    payload: Any,
    *,
    expected_total: int,
    metric: str | None = None,
) -> dict[str, Any]:
    if isinstance(payload, list):
        values = list(payload)
        if values and all(isinstance(item, Mapping) and "judge" in item for item in values):
            correct, total = _count_judged(values, source=source)
            result = _record(correct, total, source=source, kind="official_judge_json")
            result["scoring"] = OFFICIAL_LLM_JUDGE_SCORING
            result["comparable_to_frozen_rule_only"] = False
            return result
        if values and all(
            isinstance(item, Mapping)
            and "response" in item
            and "model_answer" in item
            for item in values
        ):
            if metric not in {"vstar", "mme-realworld-lite"}:
                raise SummaryError(
                    "first-letter rule-only records are only valid for VStar/MME: "
                    f"{source}"
                )
            correct, total = _count_rule_only(values, source=source)
            result = _record(correct, total, source=source, kind="rule_only_answer_json")
            result["scoring"] = RULE_ONLY_SCORING
            result["comparable_to_frozen_rule_only"] = True
            return result
        _count_judged(values, source=source)
        raise SummaryError(f"unjudged answer file rejected: {source}")
    if not isinstance(payload, dict):
        raise SummaryError(f"JSON result must be an object or judged list: {source}")

    # Common compact score receipts.  Nested ``metrics``/``result`` objects are
    # accepted, but only one unambiguous score is allowed.
    for nested_key in ("metrics", "result", "payload", "score"):
        nested = payload.get(nested_key)
        if isinstance(nested, dict) and nested is not payload:
            try:
                result = _json_records(
                    source, nested, expected_total=expected_total, metric=metric
                )
                return _annotate_score_object(result, payload, metric=metric)
            except SummaryError:
                pass
    if "correct" in payload and "total" in payload:
        try:
            correct = int(payload["correct"])
            total = int(payload["total"])
        except (TypeError, ValueError) as error:
            raise SummaryError(f"correct/total must be integers: {source}") from error
        result = _record(correct, total, source=source, kind="score_object")
        if "score" in payload and abs(_finite_score(payload["score"], label="score") - result["score"]) > 1e-12:
            raise SummaryError(f"score disagrees with correct/total: {source}")
        return _annotate_score_object(result, payload, metric=metric)
    score_keys = [key for key in ("score", "accuracy", "overall", "Overall", "overall_acc") if key in payload]
    if len(score_keys) == 1:
        score = _finite_score(payload[score_keys[0]], label=score_keys[0])
        total = expected_total
        correct_float = score * total
        correct = round(correct_float)
        if abs(correct_float - correct) > 1e-8:
            raise SummaryError(f"score does not correspond to a count over {total}: {source}")
        return _annotate_score_object(
            _record(correct, total, source=source, kind="score_object"),
            payload,
            metric=metric,
        )
    # ``status.json`` receipts from the native evaluator commonly use keys
    # such as ``split=none|Overall``.  Search only for explicit overall keys;
    # category-level metrics are not accepted as a substitute.
    overall_keys = {"split=none|Overall", "split=none|overall", "overall", "Overall"}
    found: list[Any] = []

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key in overall_keys:
                    found.append(value)
                elif isinstance(value, (dict, list)):
                    visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(payload)
    if len(found) == 1:
        score = _finite_score(found[0], label="overall score")
        correct_float = score * expected_total
        correct = round(correct_float)
        if abs(correct_float - correct) > 1e-8:
            raise SummaryError(
                f"overall score does not correspond to a count over {expected_total}: {source}"
            )
        return _annotate_score_object(
            _record(correct, expected_total, source=source, kind="score_object"),
            payload,
            metric=metric,
        )
    raise SummaryError(f"no unambiguous score or judged records found: {source}")


def _csv_score(source: Path) -> dict[str, Any]:
    with source.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or "Overall" not in rows[0]:
        raise SummaryError(f"CSV must contain an Overall column and one data row: {source}")
    row = next((item for item in rows if str(item.get("split", "none")) == "none"), rows[0])
    score = _finite_score(row.get("Overall"), label="Overall")
    total = EXPECTED_TOTALS["muir"]
    correct_float = score * total
    correct = round(correct_float)
    if abs(correct_float - correct) > 1e-8:
        raise SummaryError(f"Overall does not correspond to a count over {total}: {source}")
    result = _record(correct, total, source=source, kind="muir_acc_csv")
    result["scoring"] = MUIR_SCORING
    result["comparable_to_frozen_rule_only"] = False
    result["completion_authority_bound"] = False
    return result


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    body = {key: item for key, item in value.items() if key != "seal_sha256"}
    return hashlib.sha256(
        json.dumps(
            body,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _validate_muir_completion_authority(
    authority: Any,
    *,
    source: Path,
) -> dict[str, Any]:
    """Validate the B3 MUIR completion envelope and bind it to *source*.

    The native MUIR completion contract seals the model/protocol identity and
    records the exact result artifact.  A score without that edge remains
    display-only and cannot promote a candidate.  This intentionally accepts
    no inferred or path-only authority.
    """

    if not isinstance(authority, Mapping):
        raise SummaryError("MUIR completion authority must be an object")
    value = dict(authority)
    seal = value.get("seal_sha256")
    if not isinstance(seal, str) or SHA256_RE.fullmatch(seal) is None:
        raise SummaryError("MUIR completion authority has no valid seal_sha256")
    try:
        canonical = _canonical_sha256(value)
    except (TypeError, ValueError) as error:
        raise SummaryError("MUIR completion authority is not canonical JSON") from error
    if canonical != seal:
        raise SummaryError("MUIR completion authority seal differs")
    if (
        value.get("schema_version") != MUIR_COMPLETION_SCHEMA
        or value.get("kind") != MUIR_COMPLETION_KIND
        or value.get("status") != "passed"
    ):
        raise SummaryError("MUIR completion authority schema/status differs")
    model = value.get("model")
    if not isinstance(model, Mapping) or not all(
        isinstance(model.get(key), str) and model.get(key)
        for key in ("id", "tag", "checkpoint_name")
    ):
        raise SummaryError("MUIR completion authority has no complete model identity")
    protocol = value.get("protocol")
    if not isinstance(protocol, Mapping) or any(
        (
            protocol.get("dataset") != "MUIRBench",
            protocol.get("judge") != "exact_matching",
            protocol.get("reuse") is not False,
            protocol.get("reuse_aux") != "none",
        )
    ):
        raise SummaryError("MUIR completion authority protocol identity differs")
    static = value.get("postflight_static_contract")
    static_output = static.get("output") if isinstance(static, Mapping) else None
    if not isinstance(static_output, Mapping) or static_output.get("dataset") != "MUIRBench":
        raise SummaryError("MUIR completion authority output identity differs")

    lifecycle = value.get("lifecycle")
    result = lifecycle.get("result") if isinstance(lifecycle, Mapping) else None
    artifact = result.get("artifact") if isinstance(result, Mapping) else None
    if not isinstance(artifact, Mapping):
        raise SummaryError("MUIR completion authority has no result artifact identity")
    artifact_path = artifact.get("path")
    artifact_sha = artifact.get("sha256")
    if not isinstance(artifact_path, str) or not artifact_path:
        raise SummaryError("MUIR completion authority result path is missing")
    if Path(artifact_path).expanduser().absolute() != source:
        raise SummaryError("MUIR completion authority result path differs from candidate")
    if not isinstance(artifact_sha, str) or artifact_sha != _sha256(source):
        raise SummaryError("MUIR completion authority result hash differs from candidate")
    return {
        "schema_version": value["schema_version"],
        "kind": value["kind"],
        "model": {
            "id": model["id"],
            "tag": model["tag"],
            "checkpoint_name": model["checkpoint_name"],
        },
        "result_artifact": {
            "path": str(source),
            "sha256": artifact_sha,
        },
        "seal_sha256": seal,
    }


def _load_completion_authority(path: Path | str) -> dict[str, Any]:
    source = _safe_file(path, label="MUIR completion authority")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SummaryError(f"cannot parse MUIR completion authority: {source}") from error
    if not isinstance(payload, dict):
        raise SummaryError(f"MUIR completion authority must be an object: {source}")
    return payload


def _embedded_completion_authority(payload: Any) -> Any:
    if isinstance(payload, Mapping):
        authority = payload.get("completion_authority")
        if authority is not None:
            return authority
    return None


def _load_json_payload(source: Path) -> Any:
    """Read JSON and pretty JSON arrays, then fall back to real JSONL."""

    try:
        text = source.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise SummaryError(f"cannot read JSON result: {source}") from error
    try:
        return json.loads(text)
    except json.JSONDecodeError as whole_file_error:
        if source.suffix.lower() != ".jsonl":
            raise SummaryError(f"cannot parse JSON result: {source}") from whole_file_error
        records: list[Any] = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as line_error:
                raise SummaryError(
                    f"cannot parse JSONL record {line_number}: {source}"
                ) from line_error
        if not records:
            raise SummaryError(f"JSONL result is empty: {source}") from whole_file_error
        return records


def parse_result(
    path: Path | str,
    *,
    metric: str,
    completion_authority: Path | str | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Parse one completed metric result without GPU/model dependencies."""
    if metric not in METRICS:
        raise SummaryError(f"unknown metric: {metric}")
    source = _safe_file(path, label=f"{metric} result")
    payload: Any = None
    if source.suffix.lower() == ".csv":
        result = _csv_score(source)
    else:
        payload = _load_json_payload(source)
        result = _json_records(
            source,
            payload,
            expected_total=EXPECTED_TOTALS[metric],
            metric=metric,
        )
        if metric == "muir":
            result.setdefault("scoring", MUIR_SCORING)
            result["completion_authority_bound"] = False
    expected = EXPECTED_TOTALS[metric]
    if result["total"] != expected:
        raise SummaryError(f"{metric} total must be {expected}, got {result['total']} in {source}")
    if metric == "muir":
        authority: Any = completion_authority
        if isinstance(authority, (str, Path)):
            authority = _load_completion_authority(authority)
        if authority is None:
            authority = _embedded_completion_authority(payload)
        if authority is not None:
            result["completion_authority"] = _validate_muir_completion_authority(
                authority,
                source=source,
            )
            result["completion_authority_bound"] = True
    result["metric"] = metric
    return result


def _fixed_baseline(label: str, *, workspace: Path) -> dict[str, dict[str, Any]]:
    if label not in FIXED_BASELINES:
        raise SummaryError(f"unknown fixed baseline: {label}")
    output: dict[str, dict[str, Any]] = {}
    for metric, pinned in FIXED_BASELINES[label].items():
        source = _safe_file(workspace / pinned["source"], label=f"{label} {metric} baseline")
        digest = _sha256(source)
        if digest != pinned["sha256"]:
            raise SummaryError(
                f"{label} {metric} baseline hash changed: expected {pinned['sha256']}, got {digest}"
            )
        total = int(pinned["total"])
        if metric in ("vstar", "mme-realworld-lite"):
            # Only row count is checked for answer JSONL.  Their historical
            # scores are intentionally pinned from the paired diagnostic.
            rows = [line for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
            if len(rows) != total:
                raise SummaryError(f"{label} {metric} baseline row count changed: {len(rows)} != {total}")
        result = {
            "metric": metric,
            "correct": int(pinned["correct"]),
            "total": total,
            "score": int(pinned["correct"]) / total,
            "source": str(source),
            "source_sha256": digest,
            "input_kind": "frozen_baseline_evidence",
            "scoring": pinned["scoring"],
        }
        output[metric] = result
    return output


def _delta(candidate: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, Any]:
    value = float(candidate["score"]) - float(baseline["score"])
    return {"delta": value, "delta_pp": value * 100.0, "beats": value > 0.0}


def _protocol_gate_eligible(metric: str, candidate: Mapping[str, Any]) -> tuple[bool, str]:
    if metric in {"vstar", "mme-realworld-lite"}:
        if candidate.get("comparable_to_frozen_rule_only") is True:
            return True, "candidate uses frozen first-letter rule-only evidence"
        return False, (
            "candidate is not frozen first-letter rule-only evidence; official LLM-judge "
            "scores are display-only"
        )
    if metric == "muir":
        if candidate.get("scoring") != MUIR_SCORING:
            return False, "candidate MUIR scoring is not official exact matching"
        if candidate.get("completion_authority_bound") is not True:
            return False, (
                "candidate MUIR result has no sealed completion authority identity bound "
                "to the result artifact"
            )
        return True, "candidate MUIR result has a sealed completion authority identity"
    return False, f"no comparable protocol is defined for {metric}"


def build_summary(
    candidate: Mapping[str, Mapping[str, Any]],
    *,
    baselines: Mapping[str, Mapping[str, Mapping[str, Any]]],
    control_alias: str = "b1",
) -> dict[str, Any]:
    """Build a deterministic summary from already parsed metric records."""
    missing = [metric for metric in METRICS if metric not in candidate]
    if missing:
        raise SummaryError(f"candidate metrics missing: {', '.join(missing)}")
    for metric in METRICS:
        if int(candidate[metric]["total"]) != EXPECTED_TOTALS[metric]:
            raise SummaryError(f"candidate {metric} total mismatch")
    if control_alias not in baselines:
        raise SummaryError(f"control alias baseline missing: {control_alias}")
    comparisons: dict[str, Any] = {}
    protocol_eligibility: dict[str, Any] = {}
    for metric in METRICS:
        eligible, reason = _protocol_gate_eligible(metric, candidate[metric])
        protocol_eligibility[metric] = {"eligible": eligible, "reason": reason}
        comparisons[metric] = {
            "base": _delta(candidate[metric], baselines["base"][metric]),
            "b1": _delta(candidate[metric], baselines["b1"][metric]),
            CONTROL_KEY: _delta(candidate[metric], baselines[control_alias][metric]),
            "protocol_gate_eligible": eligible,
        }
    # Keep the aggregate computation explicit so a display-only official judge
    # can never turn into a passed promotion status through a raw delta.
    beats = {
        "base": all(
            comparisons[metric]["base"]["beats"]
            and comparisons[metric]["protocol_gate_eligible"]
            for metric in METRICS
        ),
        "b1": all(
            comparisons[metric]["b1"]["beats"]
            and comparisons[metric]["protocol_gate_eligible"]
            for metric in METRICS
        ),
        CONTROL_KEY: all(
            comparisons[metric][CONTROL_KEY]["beats"]
            and comparisons[metric]["protocol_gate_eligible"]
            for metric in METRICS
        ),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if beats[CONTROL_KEY] else "not_promoted",
        "candidate": {metric: dict(candidate[metric]) for metric in METRICS},
        "baselines": {
            "base": {metric: dict(baselines["base"][metric]) for metric in METRICS},
            "b1": {metric: dict(baselines["b1"][metric]) for metric in METRICS},
        },
        "controls": {
            CONTROL_KEY: {
                "alias_of": control_alias,
                "independent_model_evidence": False,
                "description": "B1 under the frozen official-protocol control; this is not an independent Vision-OPD model result.",
                "metrics": {metric: dict(baselines[control_alias][metric]) for metric in METRICS},
            }
        },
        "comparisons": comparisons,
        "promotion_gates": {
            "beats_base_all_three": beats["base"],
            "beats_b1_all_three": beats["b1"],
            "beats_b1_official_protocol_control_all_three": beats[CONTROL_KEY],
            "protocol_eligibility": protocol_eligibility,
            "strict_all_three_required": True,
        },
        "warnings": [
            "b1_official_protocol_control is B1 under the frozen official protocol, not an independent Vision-OPD model result.",
            "VStar/MME official LLM-judge results are displayed for diagnosis only; promotion requires the same frozen first-letter rule-only evidence as Base/B1.",
            "MUIR promotion is fail-closed unless a sealed completion authority binds the exact result artifact and model/protocol identity.",
        ],
    }


def render_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# B3 pilot384 result summary",
        "",
        f"- Status: **{summary['status']}**",
        "- Promotion gate: candidate must strictly beat `b1_official_protocol_control` on all three comparable metrics.",
        "- `b1_official_protocol_control`: B1 official-protocol alias, not an independent Vision-OPD model.",
        "",
        "| Benchmark | B3 candidate | Base | Δ vs Base | B1 | Δ vs B1 | b1_official_protocol_control | Δ vs control |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for metric in METRICS:
        candidate = summary["candidate"][metric]
        base = summary["baselines"]["base"][metric]
        b1 = summary["baselines"]["b1"][metric]
        control = summary["controls"][CONTROL_KEY]["metrics"][metric]
        comparisons = summary["comparisons"][metric]
        lines.append(
            "| {label} | {cand:.2f}% ({cc}/{ct}) | {base:.2f}% | {db:+.2f} pp | {b1:.2f}% | {d1:+.2f} pp | {ctrl:.2f}% | {dc:+.2f} pp |".format(
                label=METRIC_LABELS[metric],
                cand=100 * candidate["score"],
                cc=candidate["correct"],
                ct=candidate["total"],
                base=100 * base["score"],
                db=comparisons["base"]["delta_pp"],
                b1=100 * b1["score"],
                d1=comparisons["b1"]["delta_pp"],
                ctrl=100 * control["score"],
                dc=comparisons[CONTROL_KEY]["delta_pp"],
            )
        )
    lines += [
        "",
        "## Gates",
        "",
        f"- Beats Base on all three: `{summary['promotion_gates']['beats_base_all_three']}`",
        f"- Beats B1 on all three: `{summary['promotion_gates']['beats_b1_all_three']}`",
        f"- Beats b1_official_protocol_control on all three: `{summary['promotion_gates']['beats_b1_official_protocol_control_all_three']}`",
        "",
        "## Evidence notes",
        "",
    ]
    lines.extend(f"- {warning}" for warning in summary["warnings"])
    lines.append("")
    return "\n".join(lines)


def _write_create_once(path: Path | str, payload: str) -> None:
    target = Path(path).expanduser().absolute()
    if target.exists() or target.is_symlink():
        if target.is_file() and target.read_text(encoding="utf-8") == payload:
            return
        raise SummaryError(f"output already exists with different bytes: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(payload, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b3-vstar", required=True, type=Path)
    parser.add_argument("--b3-mme", required=True, type=Path)
    parser.add_argument("--b3-muir", required=True, type=Path)
    parser.add_argument(
        "--muir-completion-authority",
        type=Path,
        help="sealed B3 MUIR completion authority bound to the exact --b3-muir artifact",
    )
    parser.add_argument("--json-out", required=True, type=Path)
    parser.add_argument("--markdown-out", required=True, type=Path)
    parser.add_argument("--workspace", type=Path, default=WORKSPACE)
    args = parser.parse_args(argv)
    try:
        workspace = args.workspace.expanduser().absolute()
        candidate = {
            "vstar": parse_result(args.b3_vstar, metric="vstar"),
            "mme-realworld-lite": parse_result(args.b3_mme, metric="mme-realworld-lite"),
            "muir": parse_result(
                args.b3_muir,
                metric="muir",
                completion_authority=args.muir_completion_authority,
            ),
        }
        baselines = {
            "base": _fixed_baseline("base", workspace=workspace),
            "b1": _fixed_baseline("b1", workspace=workspace),
        }
        summary = build_summary(candidate, baselines=baselines)
        _write_create_once(args.json_out, json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        _write_create_once(args.markdown_out, render_markdown(summary))
    except SummaryError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
