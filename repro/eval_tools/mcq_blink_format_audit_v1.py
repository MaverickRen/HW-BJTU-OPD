#!/usr/bin/env python3
"""Audit BLINK v5 answer formatting without changing the v5 score.

The audit reruns the deterministic checkpoint-comparison v5 request against an
OpenAI-compatible endpoint and keeps the raw answer in a private NDJSON
sidecar.  Gold answers, questions, options, and image paths are kept in memory
only.  The aggregate contains counts and hashes, never sample-level data.

``--limit`` is intentionally an explicit diagnostic mode.  A run without it
must consume all 1901 rows; a full-run aggregate is therefore never silently
published from a partial result.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import stat
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from threading import local
from typing import Any, Callable


try:
    import mcq_blink_checkpoint_comparison_aggregate_v5 as _v5
except ModuleNotFoundError as error:  # pragma: no cover - direct script import
    if error.name != "mcq_blink_checkpoint_comparison_aggregate_v5":
        raise
    _v5_path = Path(__file__).with_name("mcq_blink_checkpoint_comparison_aggregate_v5.py")
    _v5_spec = importlib.util.spec_from_file_location(
        "mcq_blink_checkpoint_comparison_aggregate_v5", _v5_path
    )
    if _v5_spec is None or _v5_spec.loader is None:
        raise ImportError("v5 scorer is unavailable") from error
    _v5 = importlib.util.module_from_spec(_v5_spec)
    sys.modules[_v5_spec.name] = _v5
    _v5_spec.loader.exec_module(_v5)


WORKSPACE = _v5.WORKSPACE
OUTPUT_ROOT = _v5.OUTPUT_ROOT
DATASET = _v5.DATASET
DATASET_MD5 = dict(_v5.DATASET_MD5)
EXPECTED_TOTAL = _v5.EXPECTED_TOTAL
DATASET_ROWS = dict(_v5.DATASET_ROWS)
DEFAULT_WORKERS = _v5.DEFAULT_WORKERS
PRESET_NAME = _v5.PRESET_NAME
FROZEN_PRESET = dict(_v5.FROZEN_PRESET)
SCHEMA_VERSION = "mcq_blink_format_audit_v1"
CODE_VERSION = SCHEMA_VERSION

AuditError = _v5.MCQAggregateError
LocalInputError = _v5.LocalInputError

_HEX64 = re.compile(r"[0-9a-f]{64}\Z")

# Deliberately narrower than a general-purpose answer extractor.  A rescue is
# useful for diagnosing formatting only when the model itself made a unique,
# explicit final selection.  In particular, a bare A/B/C/D buried in reasoning
# is never rescued.
_BOXED_RE = re.compile(r"\\boxed\s*\{\s*([A-D])\s*\}", re.IGNORECASE)
_JSON_OBJECT_RE = re.compile(r"\{\s*\"answer\"\s*:\s*\"([A-D])\"\s*\}", re.IGNORECASE)
_FINAL_LINE_RE = re.compile(
    r"(?is)^\s*(?:final\s+(?:answer|choice|option)\s*[:=\-]?\s*)?"
    r"[\[\(\{`'\"*]*([A-D])[\]\}\)'\"*.,;:!?]*\s*$"
)
_JSON_FENCE_RE = re.compile(r"(?is)^\s*```(?:json)?\s*\n(.*?)\n?```\s*$")


@dataclass(frozen=True)
class AuditRecord:
    """Private row result; only a redacted projection is persisted."""

    row_index: int
    response_content: str | None
    finish_reason: str
    v5_prediction: str | None
    category: str
    supplemental_prediction: str | None = None
    supplemental_method: str | None = None
    provider_failed: bool = False
    failure_kind: str | None = None
    # Gold is deliberately not part of this dataclass.  ``audit_records``
    # computes score counts from the private row/sample pair and then discards
    # that pair before publication.


@dataclass(frozen=True)
class _PrivateResult:
    record: AuditRecord
    gold: str


def _valid_labels(options: Mapping[str, Any] | Sequence[tuple[str, Any]]) -> set[str]:
    if isinstance(options, Mapping):
        labels = [str(label) for label in options]
    else:
        labels = [str(label) for label, _ in options]
    if len(labels) < 2 or len(labels) != len(set(labels)):
        return set()
    if any(label not in _v5.CHOICES for label in labels):
        return set()
    return set(labels)


def _explicit_state(text: str, valid: set[str]) -> tuple[str | None, bool, bool]:
    """Return ``(candidate, seen, conflict)`` for v5 marker/terminal syntax.

    ``seen`` and ``conflict`` are intentionally separate: a boxed/JSON rescue
    must not mistake a contradictory pair of explicit markers for the absence
    of a marker.
    """

    markers = [m.group(1).upper() for m in _v5._v3._ANSWER_MARKER_RE.finditer(text)]
    terminal = _v5._v3._TERMINAL_RE.search(text)
    if terminal:
        markers.append(terminal.group(1).upper())
    if not markers:
        return None, False, False
    if any(value not in valid for value in markers):
        return None, True, True
    values = set(markers)
    return (next(iter(values)), True, False) if len(values) == 1 else (None, True, True)


def _parse_json_supplement(text: str, valid: set[str]) -> str | None:
    """Parse only a unique answer JSON object at the end of the response."""

    # v5 already accepts a complete object/fenced object.  This intentionally
    # adds only the safe ``reasoning\n{...}`` shape, where the object is the
    # final non-whitespace token and there is no competing explicit marker.
    candidate_text = text.strip()
    fenced = _JSON_FENCE_RE.fullmatch(candidate_text)
    if fenced:
        candidate_text = fenced.group(1).strip()
    try:
        value = json.loads(candidate_text)
    except (TypeError, ValueError, json.JSONDecodeError):
        value = None
    if isinstance(value, dict) and set(value) == {"answer"}:
        answer = value.get("answer")
        if isinstance(answer, str) and answer.upper() in valid:
            return answer.upper()

    # A trailing object after a prose prefix is accepted only if the prefix
    # does not contain an answer/choice marker or a boxed answer.  This keeps
    # ``reasoning ... answer is B ... {answer:A}`` invalid rather than guessing.
    match = re.search(r"(?is)(\{\s*\"answer\"\s*:\s*\"([A-D])\"\s*\})\s*$", text)
    if not match:
        return None
    prefix = text[: match.start(1)]
    if _v5._v3._ANSWER_MARKER_RE.search(prefix) or _BOXED_RE.search(prefix):
        return None
    answer = match.group(2).upper()
    return answer if answer in valid else None


def supplemental_parse(
    response_content: Any,
    options: Mapping[str, Any] | Sequence[tuple[str, Any]],
) -> tuple[str | None, str | None]:
    """Return ``(candidate, method)`` for a conservative format-only rescue.

    The order is JSON, boxed, then terminal.  Multiple/contradictory explicit
    forms are rejected.  This helper never consults option text and therefore
    cannot convert a semantic free-form answer into a score.
    """

    if not isinstance(response_content, str):
        return None, None
    text = response_content.strip()
    if not text:
        return None, None
    valid = _valid_labels(options)
    if not valid:
        return None, None

    json_candidate = _parse_json_supplement(text, valid)
    if json_candidate is not None:
        # A different explicit answer anywhere else makes JSON unsafe.
        explicit, seen, conflict = _explicit_state(text, valid)
        if not conflict and (not seen or explicit == json_candidate):
            return json_candidate, "json"

    boxed = [value.upper() for value in _BOXED_RE.findall(text)]
    if boxed and len(set(boxed)) == 1 and boxed[0] in valid:
        explicit, seen, conflict = _explicit_state(text, valid)
        # ``\boxed{A}`` is the selection itself.  A conflicting recognized
        # marker is not rescued; an absent marker is safe.
        if not conflict and (not seen or explicit == boxed[0]):
            return boxed[0], "boxed"

    # Only an entire final non-empty line is eligible.  This does not rescue a
    # trailing option letter in the middle of a sentence.
    final_line = next((line for line in reversed(text.splitlines()) if line.strip()), "")
    match = _FINAL_LINE_RE.fullmatch(final_line)
    if match:
        candidate = match.group(1).upper()
        if candidate in valid:
            explicit, seen, conflict = _explicit_state(text, valid)
            if not conflict and (not seen or explicit == candidate):
                return candidate, "terminal"
    return None, None


# Friendly aliases used by CPU tests and downstream diagnostics.
parse_supplemental = supplemental_parse
parse_safe_supplement = supplemental_parse


def classify_v5(
    response_content: Any,
    options: Mapping[str, Any] | Sequence[tuple[str, Any]],
    v5_prediction: str | None,
    *,
    provider_failed: bool = False,
    failure_kind: str | None = None,
) -> str:
    """Classify the baseline v5 result without exposing raw text in labels."""

    if provider_failed:
        return f"provider_error_{failure_kind or 'unknown'}"
    if v5_prediction is not None:
        return "v5_parsed"
    if response_content is None:
        return "missing_content"
    if not isinstance(response_content, str):
        return "non_string_content"
    if not response_content.strip():
        return "empty_content"
    valid = _valid_labels(options)
    explicit, seen, conflict = _v5._explicit_parse(response_content, options)
    if seen and conflict:
        # Distinguish strict JSON failures from marker disagreement where this
        # is possible; both remain invalid under v5.
        stripped = response_content.strip()
        if stripped.startswith("{") or _JSON_FENCE_RE.fullmatch(stripped):
            return "v5_invalid_json"
        return "v5_conflicting_or_invalid_explicit"
    local_candidate = _v5._v4.can_infer(response_content, options)
    if explicit is not None and local_candidate is not None and explicit != local_candidate:
        return "v5_parser_disagreement"
    return "v5_no_safe_match"


# Alternate spelling retained as a small compatibility convenience.
classify_category = classify_v5


def _finish_reason(choice: Any) -> str:
    return _v5._v2._finish_bucket(_v5._v2._field(choice, "finish_reason"))


def _infer_audit(
    samples: Sequence[Any],
    *,
    args: argparse.Namespace,
    client_factory: Callable[[], Any] | None = None,
) -> list[_PrivateResult]:
    """Run v5 requests and retain raw content only in memory until writing."""

    thread_local = local()

    def client() -> Any:
        value = getattr(thread_local, "client", None)
        if value is None:
            value = client_factory() if client_factory is not None else _v5._new_client(args)
            thread_local.client = value
        return value

    def one(item: tuple[int, Any]) -> _PrivateResult:
        row_index, sample = item
        try:
            request = _v5.build_request(sample, model_id=args.model_id)
            response = client().chat.completions.create(**request)
            choice = _v5._v2._choice(response)
            message = _v5._v2._message(response)
            if choice is None or message is None:
                record = AuditRecord(
                    row_index=row_index,
                    response_content=None,
                    finish_reason="other",
                    v5_prediction=None,
                    category="malformed_response",
                )
                return _PrivateResult(record, str(sample.gold))
            content = _v5._v2._response_content(response)
            prediction = _v5.can_infer(content, sample.options)
            prediction = prediction if isinstance(prediction, str) else None
            finish = _finish_reason(choice)
            failure = False
            kind = None
        except Exception as error:
            # Keep endpoint exception text out of the private sidecar.  Known
            # OpenAI failures retain only the protocol-safe kind.
            kind = _v5._v2._provider_failure_kind(error)
            if kind is None:
                kind = "unknown"
            content = None
            prediction = None
            finish = "other"
            failure = True

        category = classify_v5(
            content, sample.options, prediction,
            provider_failed=failure,
            failure_kind=kind,
        )
        supplement, method = supplemental_parse(content, sample.options)
        # A supplement is diagnostic only for rows baseline v5 left invalid.
        if prediction is not None:
            supplement, method = None, None
        record = AuditRecord(
            row_index=row_index,
            response_content=content if isinstance(content, str) else None,
            finish_reason=finish,
            v5_prediction=prediction,
            category=category,
            supplemental_prediction=supplement,
            supplemental_method=method,
            provider_failed=failure,
            failure_kind=kind if failure else None,
        )
        return _PrivateResult(record, str(sample.gold))

    results: dict[int, _PrivateResult] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(one, item) for item in enumerate(samples)]
        for future in as_completed(futures):
            result = future.result()
            results[result.record.row_index] = result
    return [results[index] for index in sorted(results)]


infer_audit = _infer_audit


def _record_sidecar(record: AuditRecord) -> dict[str, Any]:
    """Project an in-memory record to the privacy-reviewed NDJSON schema."""

    return {
        "row_index": record.row_index,
        "response_content": record.response_content,
        "finish_reason": record.finish_reason,
        "v5_prediction": record.v5_prediction,
        "category": record.category,
        "supplemental_prediction": record.supplemental_prediction,
        "supplemental_method": record.supplemental_method,
    }


def sidecar_records(records: Sequence[AuditRecord]) -> list[dict[str, Any]]:
    return [_record_sidecar(record) for record in sorted(records, key=lambda x: x.row_index)]


def _assert_output_path(path: Path) -> Path:
    """Require an output below the normal H_Workspace/Output tree."""

    target = path.resolve()
    root = OUTPUT_ROOT.resolve()
    try:
        target.relative_to(root)
    except ValueError:
        raise AuditError("output must be below H_Workspace/Output") from None
    if target == root or target.exists() or target.is_symlink():
        raise AuditError("audit output already exists or is the output root")
    cursor = root
    try:
        root_info = root.lstat()
    except OSError as error:
        raise AuditError("audit output root is unavailable") from error
    if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode):
        raise AuditError("audit output root must be a real directory")
    for component in target.relative_to(root).parts[:-1]:
        cursor = cursor / component
        if cursor.exists() or cursor.is_symlink():
            info = cursor.lstat()
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                raise AuditError("audit output parent must contain no symlinks")
    if not target.parent.is_dir() or target.parent.is_symlink():
        raise AuditError("audit output parent must be an existing real directory")
    return target


def _write_private_once(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise AuditError("audit output already exists")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def write_sidecar(path: Path, records: Sequence[AuditRecord]) -> str:
    """Write mode-0600 NDJSON and return its SHA256."""

    path = _assert_output_path(path)
    lines = [
        json.dumps(_record_sidecar(record), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for record in sorted(records, key=lambda x: x.row_index)
    ]
    payload = ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")
    _write_private_once(path, payload)
    return hashlib.sha256(payload).hexdigest()


def _code_identity() -> dict[str, Any]:
    digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return {
        "version": CODE_VERSION,
        "audit_sha256": digest,
        "v5": _v5._code_identity(),
    }


def aggregate_audit(
    private_results: Sequence[_PrivateResult] | Sequence[tuple[AuditRecord, str]],
    *,
    model_id: str,
    sidecar_sha256: str,
    prompt_hash: str | None = None,
    image_manifest_hash: str | None = None,
    limited: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    """Build aggregate-only counts; no raw response or per-row gold survives."""

    normalized: list[_PrivateResult] = []
    for item in private_results:
        if isinstance(item, _PrivateResult):
            normalized.append(item)
        else:
            normalized.append(_PrivateResult(item[0], str(item[1])))
    total = len(normalized)
    if total == 0:
        raise AuditError("audit requires at least one row")
    if not isinstance(sidecar_sha256, str) or _HEX64.fullmatch(sidecar_sha256) is None:
        raise AuditError("sidecar SHA256 is malformed")
    baseline_correct = sum(
        int(result.record.v5_prediction is not None and result.record.v5_prediction == result.gold)
        for result in normalized
    )
    invalid = [result for result in normalized if result.record.v5_prediction is None]
    rescued = [
        result for result in invalid
        if result.record.supplemental_prediction in _v5.CHOICES
    ]
    rescued_correct = sum(int(result.record.supplemental_prediction == result.gold) for result in rescued)
    rescued_wrong = len(rescued) - rescued_correct
    new_correct = baseline_correct + rescued_correct
    categories = Counter(result.record.category for result in normalized)
    invalid_categories = Counter(result.record.category for result in invalid)
    finish = Counter(result.record.finish_reason for result in normalized)
    methods = Counter(result.record.supplemental_method for result in rescued)
    v5_predictions = Counter(result.record.v5_prediction for result in normalized if result.record.v5_prediction)
    body: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "code_version": CODE_VERSION,
        "dataset": {
            "name": DATASET,
            "rows": EXPECTED_TOTAL,
            "observed_rows": total,
            "md5": DATASET_MD5[DATASET],
        },
        "model_id": model_id,
        "preset": PRESET_NAME,
        "frozen_generation": dict(FROZEN_PRESET),
        "formal_full_run": bool(not limited and total == EXPECTED_TOTAL),
        "limited": bool(limited),
        "limit": limit,
        "baseline_v5": {
            "correct": baseline_correct,
            "total": total,
            "accuracy": baseline_correct / total,
            "invalid_count": len(invalid),
            "invalid_categories": dict(sorted(invalid_categories.items())),
            "category_counts": dict(sorted(categories.items())),
            "finish_reason_counts": dict(sorted(finish.items())),
            "selected_option_counts": dict(sorted(v5_predictions.items())),
        },
        "supplemental_format_only": {
            "parser_order": ["json", "boxed", "terminal"],
            "eligible_invalid_count": len(invalid),
            "rescued_count": len(rescued),
            "rescued_correct": rescued_correct,
            "rescued_wrong": rescued_wrong,
            "rescued_by_method": dict(sorted(methods.items())),
            "new_correct": new_correct,
            "new_accuracy": new_correct / total,
            "changes_baseline": True,
        },
        "sidecar": {
            "sha256": sidecar_sha256,
            "mode": "0600",
            "fields": [
                "row_index", "response_content", "finish_reason", "v5_prediction",
                "category", "supplemental_prediction", "supplemental_method",
            ],
        },
        "hashes": {
            "sidecar_sha256": sidecar_sha256,
            **({"prompt_sha256": prompt_hash} if prompt_hash else {}),
            **({"image_manifest_sha256": image_manifest_hash} if image_manifest_hash else {}),
        },
        "privacy": {
            "gold_saved": False,
            "question_saved": False,
            "options_saved": False,
            "aggregate_raw_responses_saved": False,
            "sidecar_contains_raw_response": True,
        },
        "code": _code_identity(),
    }
    body["seal_sha256"] = hashlib.sha256(
        json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return body


def write_aggregate(path: Path, aggregate: Mapping[str, Any]) -> None:
    path = _assert_output_path(path)
    seal = aggregate.get("seal_sha256")
    unsigned = {key: value for key, value in aggregate.items() if key != "seal_sha256"}
    expected = hashlib.sha256(
        json.dumps(unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if seal != expected:
        raise AuditError("aggregate seal is malformed")
    payload = (json.dumps(aggregate, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    _write_private_once(path, payload)


def _validate_preset(args: argparse.Namespace) -> None:
    if args.preset != PRESET_NAME:
        raise AuditError("only the frozen v5 checkpoint-comparison preset is supported")
    for key, expected in FROZEN_PRESET.items():
        if getattr(args, key) != expected:
            raise AuditError(f"frozen preset control differs: {key}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blink-tsv", "--blink_tsv", "--blink", dest="blink_tsv", required=True, type=Path)
    parser.add_argument("--sidecar", "--output-sidecar", "--output_sidecar", dest="sidecar", required=True, type=Path)
    parser.add_argument("--aggregate", "--aggregate-output", "--aggregate_output", dest="aggregate", required=True, type=Path)
    parser.add_argument("--model-id", "--model_id", "--model", dest="model_id", required=True)
    parser.add_argument("--api-base", "--api_base", required=True)
    parser.add_argument("--api-key", "--api_key", default="EMPTY")
    parser.add_argument("--preset", default=PRESET_NAME)
    parser.add_argument("--thinking", "--enable-thinking", "--enable_thinking", nargs="?", const=True, default=False, type=_v5._parse_bool)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", "--top_p", type=float, default=1.0)
    parser.add_argument("--top-k", "--top_k", type=int, default=-1)
    parser.add_argument("--min-p", "--min_p", type=float, default=0.0)
    parser.add_argument("--presence-penalty", "--presence_penalty", type=float, default=0.0)
    parser.add_argument("--repetition-penalty", "--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--max-tokens", "--max_tokens", type=int, default=32768)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", "--worker", dest="workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--structured-json", "--structured_json", nargs="?", const=True, default=False, type=_v5._parse_bool)
    parser.add_argument("--image-root", "--image_root", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None, help="diagnostic row limit; omit for the required full 1901-row run")
    parser.add_argument("--dry-run", "--dry_run", action="store_true")
    return parser


def _dry_run_metadata(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "code_version": CODE_VERSION,
        "dry_run": True,
        "reads_data": False,
        "connects_api": False,
        "writes_output": False,
        "dataset": DATASET,
        "total": EXPECTED_TOTAL,
        "required_full_rows": EXPECTED_TOTAL,
        "preset": PRESET_NAME,
        "frozen_generation": dict(FROZEN_PRESET),
        "limit": args.limit,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        _validate_preset(args)
        if args.workers < 1 or args.workers > 256:
            raise AuditError("workers must be in [1, 256]")
        if args.limit is not None and not 1 <= args.limit <= EXPECTED_TOTAL:
            raise AuditError(f"limit must be in [1, {EXPECTED_TOTAL}]")
        if args.sidecar.resolve() == args.aggregate.resolve():
            raise AuditError("sidecar and aggregate outputs must differ")
        _v5._validate_api_base(args.api_base)
        if args.dry_run:
            print(json.dumps(_dry_run_metadata(args), sort_keys=True))
            return 0

        blink = _v5._absolute(args.blink_tsv)
        _v5._regular_file(blink, label="BLINK TSV", suffix=".tsv")
        sidecar_path = _assert_output_path(args.sidecar)
        aggregate_path = _assert_output_path(args.aggregate)
        # The v5 reader verifies the complete official TSV before limiting, so
        # --limit cannot accidentally run against a truncated/reordered file.
        samples, _digest = _v5._read_dataset(blink, image_root=args.image_root)
        if len(samples) != EXPECTED_TOTAL:
            raise AuditError("BLINK dataset must contain exactly 1901 rows")
        limit = args.limit
        selected = tuple(samples if limit is None else samples[:limit])
        model_client = _v5._new_client(args)
        _v5._preflight_model_id(args, model_client)
        private_results = _infer_audit(selected, args=args)
        records = [result.record for result in private_results]
        prompt_hash = _v5._v2._hash_texts(_v5.build_blink_prompt(sample) for sample in samples)
        image_hash = _v5.image_manifest_sha256(samples)
        sidecar_sha = write_sidecar(sidecar_path, records)
        aggregate = aggregate_audit(
            private_results,
            model_id=args.model_id,
            sidecar_sha256=sidecar_sha,
            prompt_hash=prompt_hash,
            image_manifest_hash=image_hash,
            limited=limit is not None and limit < EXPECTED_TOTAL,
            limit=limit,
        )
        write_aggregate(aggregate_path, aggregate)
        print(json.dumps({
            "schema_version": SCHEMA_VERSION,
            "rows": len(records),
            "v5_invalid": aggregate["baseline_v5"]["invalid_count"],
            "rescued_correct": aggregate["supplemental_format_only"]["rescued_correct"],
            "rescued_wrong": aggregate["supplemental_format_only"]["rescued_wrong"],
            "new_accuracy": aggregate["supplemental_format_only"]["new_accuracy"],
            "sidecar": str(sidecar_path),
            "aggregate": str(aggregate_path),
        }, sort_keys=True))
        return 0
    except AuditError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
