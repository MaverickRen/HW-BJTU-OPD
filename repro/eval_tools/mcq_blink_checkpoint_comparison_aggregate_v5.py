#!/usr/bin/env python3
"""Aggregate-only deterministic BLINK checkpoint-comparison evaluator (v5).

This protocol is explicitly for deterministic checkpoint comparison, not an
official leaderboard reproduction.  It keeps the native local
``ImageMCQDataset`` prompt, uses greedy non-thinking generation, and combines
the v3 explicit parser with the v4 local matching-util parser.  If the two
parsers disagree, the answer is invalid.  No judge or LLM fallback is used.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import local
from typing import Any, Callable


try:
    import mcq_blink_official_aggregate_v4 as _v4
except ModuleNotFoundError as error:  # pragma: no cover - direct script import
    if error.name != "mcq_blink_official_aggregate_v4":
        raise
    _v4_path = Path(__file__).with_name("mcq_blink_official_aggregate_v4.py")
    _v4_spec = importlib.util.spec_from_file_location(
        "mcq_blink_official_aggregate_v4", _v4_path
    )
    if _v4_spec is None or _v4_spec.loader is None:
        raise ImportError("v4 scorer is unavailable") from error
    _v4 = importlib.util.module_from_spec(_v4_spec)
    sys.modules[_v4_spec.name] = _v4
    _v4_spec.loader.exec_module(_v4)


_v3 = _v4._v3
_v2 = _v4._v2
WORKSPACE = _v4.WORKSPACE
OUTPUT_ROOT = _v4.OUTPUT_ROOT
DATASET = _v4.DATASET
EXPECTED_TOTAL = _v4.EXPECTED_TOTAL
DATASET_ROWS = dict(_v4.DATASET_ROWS)
DATASET_MD5 = dict(_v4.DATASET_MD5)
DEFAULT_WORKERS = _v4.DEFAULT_WORKERS
CHOICES = frozenset("ABCD")

SCHEMA_VERSION = "mcq_blink_checkpoint_comparison_aggregate_v5"
CODE_VERSION = SCHEMA_VERSION
PRESET_NAME = "blink_deterministic_checkpoint_comparison_v5"
FROZEN_PRESET = {
    "thinking": False,
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": -1,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "repetition_penalty": 1.0,
    "max_tokens": 32768,
    "structured_json": False,
    "seed": 42,
}

MCQAggregateError = _v4.MCQAggregateError
LocalInputError = _v4.LocalInputError
InferenceOutcome = _v4.InferenceOutcome

_FENCE_RE = re.compile(r"\A```(?:json)?\s*\n?(.*?)\n?```\Z", re.IGNORECASE | re.DOTALL)

# Reuse the exact native ImageMCQDataset prompt and image ordering from v3/v4.
_canonical = _v4._canonical
_absolute = _v4._absolute
_validate_api_base = _v4._validate_api_base
_parse_bool = _v4._parse_bool
_regular_file = _v4._regular_file
_read_dataset = _v4._read_dataset
build_blink_prompt = _v4.build_blink_prompt
build_messages = _v4.build_messages
_new_client = _v4._new_client
image_manifest_sha256 = _v4.image_manifest_sha256
aggregate_outcomes = _v2.aggregate_outcomes
aggregate_counts = aggregate_outcomes


def build_request(
    row: Any,
    *,
    model_id: str,
    image_uris: Sequence[str] | None = None,
    thinking: bool = False,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = -1,
    min_p: float = 0.0,
    presence_penalty: float = 0.0,
    repetition_penalty: float = 1.0,
    max_tokens: int = 32768,
    structured_json: bool = False,
    seed: int = 42,
) -> dict[str, Any]:
    controls = locals()
    for name, expected in FROZEN_PRESET.items():
        if controls[name] != expected:
            raise MCQAggregateError(f"frozen preset control differs: {name}")
    if structured_json:
        raise MCQAggregateError("v5 uses native ImageMCQDataset prompt without structured JSON")
    return {
        "model": model_id,
        "messages": build_messages(row, image_uris),
        "temperature": temperature,
        "top_p": top_p,
        "presence_penalty": presence_penalty,
        "max_tokens": max_tokens,
        "seed": seed,
        "extra_body": {
            "top_k": top_k,
            "min_p": min_p,
            "repetition_penalty": repetition_penalty,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    }


def _strict_json_answer(text: str, valid: set[str]) -> tuple[str | None, bool, bool]:
    candidate_text = text
    fenced = _FENCE_RE.fullmatch(candidate_text)
    if fenced:
        candidate_text = fenced.group(1).strip()
    if not (candidate_text.startswith("{") and candidate_text.endswith("}")):
        return None, False, False

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    try:
        value = json.loads(candidate_text, object_pairs_hook=pairs)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None, True, True
    if not isinstance(value, dict) or set(value) != {"answer"}:
        return None, True, True
    answer = value.get("answer")
    if isinstance(answer, str) and answer in valid:
        return answer, True, False
    return None, True, True


def _explicit_parse(
    answer: str,
    choices: Mapping[str, Any] | Sequence[tuple[str, str]],
) -> tuple[str | None, bool, bool]:
    """Return (candidate, saw_explicit_form, conflict_or_invalid_form)."""

    option_map = _v4._options(choices)
    if option_map is None:
        return None, True, True
    valid = set(option_map)
    text = answer.strip()

    candidate, seen, conflict = _strict_json_answer(text, valid)
    if seen:
        return candidate, seen, conflict

    markers = [match.group(1).upper() for match in _v3._ANSWER_MARKER_RE.finditer(text)]
    terminal = _v3._TERMINAL_RE.search(text)
    explicit_candidates = list(markers)
    if terminal:
        explicit_candidates.append(terminal.group(1).upper())
    if not explicit_candidates:
        return None, False, False
    values = set(explicit_candidates)
    if len(values) != 1 or not values.issubset(valid):
        return None, True, True
    return explicit_candidates[0], True, False


def _union_parse(
    answer: Any,
    choices: Mapping[str, Any] | Sequence[tuple[str, str]],
) -> str | None:
    if not isinstance(answer, str):
        return None
    explicit, seen, conflict = _explicit_parse(answer, choices)
    if conflict:
        return None
    local = _v4.can_infer(answer, choices)
    if explicit is not None and local is not None and explicit != local:
        return None
    if explicit is not None:
        return explicit
    return local if not seen else None


can_infer = _union_parse
parse_choice = can_infer
parse_answer = can_infer
parse_model_answer = can_infer


def _code_identity() -> dict[str, Any]:
    def digest(path: Path, label: str) -> str:
        return _v2._file_sha256(path, label=label)

    source_hashes = {
        "v1": digest(Path(_v2._v1.__file__), "v1 scorer source"),
        "v2": digest(Path(_v2.__file__), "v2 scorer source"),
        "v3": digest(Path(_v3.__file__), "v3 scorer source"),
        "v4": digest(Path(_v4.__file__), "v4 scorer source"),
        "v5": digest(Path(__file__), "v5 scorer source"),
    }
    return {
        "version": CODE_VERSION,
        "sha256": hashlib.sha256(_canonical(source_hashes)).hexdigest(),
        "runner_sha256": source_hashes["v5"],
        "dependency_hashes": dict(source_hashes),
        "source_hashes": source_hashes,
    }


def _source_sha256() -> str:
    return str(_code_identity()["sha256"])


def _preflight_model_id(args: argparse.Namespace, client: Any | None = None) -> dict[str, Any]:
    return _v4._preflight_model_id(args, client)


preflight_model_id = _preflight_model_id


def _infer(
    samples: Sequence[_v2.Sample],
    *,
    args: argparse.Namespace,
    client_factory: Callable[[], Any] | None = None,
) -> list[InferenceOutcome]:
    thread_local = local()

    def client() -> Any:
        value = getattr(thread_local, "client", None)
        if value is None:
            value = client_factory() if client_factory is not None else _new_client(args)
            thread_local.client = value
        return value

    def one(sample: _v2.Sample) -> InferenceOutcome:
        request = build_request(sample, model_id=args.model_id)
        try:
            response = client().chat.completions.create(**request)
        except Exception as error:
            failure_kind = _v2._provider_failure_kind(error)
            if failure_kind is None:
                raise LocalInputError("provider call returned a non-OpenAI error") from None
            return InferenceOutcome(
                dataset=DATASET, gold=sample.gold, prediction=None,
                provider_failed=True, failure_kind=failure_kind,
            )
        choice = _v2._choice(response)
        if choice is None or _v2._message(response) is None:
            raise LocalInputError("provider returned a malformed choice")
        content = _v2._response_content(response)
        return InferenceOutcome(
            dataset=DATASET,
            gold=sample.gold,
            prediction=can_infer(content, sample.options),
            finish_reason=_v2._finish_bucket(_v2._field(choice, "finish_reason")),
            content_present=content is not None,
            reasoning_present=_v2._response_reasoning_present(response),
            completion_tokens=_v2._completion_tokens(response),
        )

    results: list[InferenceOutcome] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(one, sample): sample for sample in samples}
        for future in as_completed(futures):
            results.append(future.result())
    return results


def _protocol(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "preset": PRESET_NAME,
        "thinking": False,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "repetition_penalty": 1.0,
        "max_tokens": 32768,
        "seed": 42,
        "workers": args.workers,
        "dataset": DATASET,
        "prompt_source": "VLMEvalKit.ImageMCQDataset",
        "prompt_only": True,
        "response_format": "native_image_mcq_v3_explicit_union_v4_matching_util",
        "image_order": "source_tsv_order",
        "gold_scope": "scorer_memory_only",
        "sample_level_output": False,
        "leaderboard_claim": False,
        "comparison_purpose": "deterministic_checkpoint_comparison",
    }


_PROTOCOL_KEYS = frozenset(_protocol(argparse.Namespace(workers=1)))


def _sanitize_protocol(protocol: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(protocol, Mapping) or set(protocol) - _PROTOCOL_KEYS:
        raise MCQAggregateError("protocol contains an unsupported field")
    result = dict(protocol)
    if (
        result.get("dataset") != DATASET
        or result.get("preset") != PRESET_NAME
        or result.get("prompt_source") != "VLMEvalKit.ImageMCQDataset"
        or result.get("leaderboard_claim") is not False
        or result.get("comparison_purpose") != "deterministic_checkpoint_comparison"
    ):
        raise MCQAggregateError("checkpoint-comparison protocol identity differs")
    if result.get("thinking") is not False or result.get("prompt_only") is not True:
        raise MCQAggregateError("protocol mode differs")
    return result


def _validate_args(args: argparse.Namespace, *, inspect_inputs: bool) -> tuple[Path, Path]:
    if args.workers < 1 or args.workers > 256:
        raise MCQAggregateError("workers must be in [1, 256]")
    if args.output == _absolute(OUTPUT_ROOT) or args.output.is_dir():
        raise MCQAggregateError("output must be a receipt file")
    _v2._assert_output_scope(args.output)
    _validate_api_base(args.api_base)
    if not isinstance(args.model_id, str) or not args.model_id.strip():
        raise MCQAggregateError("model id must be non-empty")
    blink = _absolute(args.blink_tsv)
    output = _absolute(args.output)
    if inspect_inputs:
        _regular_file(blink, label="BLINK TSV", suffix=".tsv")
    return blink, output


def _validate_preset(args: argparse.Namespace) -> None:
    if args.preset != PRESET_NAME:
        raise MCQAggregateError("only the frozen v5 checkpoint-comparison preset is supported")
    for key, expected in FROZEN_PRESET.items():
        if getattr(args, key) != expected:
            raise MCQAggregateError(f"frozen preset control differs: {key}")


def _receipt(
    *,
    aggregate: Mapping[str, Any],
    prompt_hash: str,
    model_id: str,
    protocol: Mapping[str, Any],
    model_preflight: Mapping[str, Any],
    image_manifest_hash: str | None,
) -> dict[str, Any]:
    clean_aggregate = _v2._sanitize_aggregate(aggregate)
    if set(clean_aggregate.get("datasets", {})) != {DATASET}:
        raise MCQAggregateError("aggregate dataset identity differs")
    clean_protocol = _sanitize_protocol(protocol)
    clean_preflight = _v2._sanitize_model_preflight(model_preflight)
    _v2._validate_hash(prompt_hash, label="prompt hash")
    if image_manifest_hash is not None:
        _v2._validate_hash(image_manifest_hash, label="image manifest hash")
    code_identity = _code_identity()
    data_hash = _v2._sha256_bytes(_canonical(DATASET_MD5))
    model_hash = _v2._sha256_bytes(model_id.encode("utf-8"))
    protocol_hash = _v2._sha256_bytes(_canonical(clean_protocol))
    hashes = {"data": data_hash, "prompt": prompt_hash, "model": model_hash, "protocol": protocol_hash}
    if image_manifest_hash is not None:
        hashes["images"] = image_manifest_hash
    telemetry = {
        "api_failure_count": int(clean_aggregate.get("api_failure_count", 0)),
        "http_failure_count": int(clean_aggregate.get("http_failure_count", 0)),
        "non_http_api_failure_count": int(clean_aggregate.get("non_http_api_failure_count", 0)),
        "api_error_count": int(clean_aggregate.get("api_error_count", 0)),
        "http_error_count": int(clean_aggregate.get("http_error_count", 0)),
        "invalid_format_count": int(clean_aggregate.get("invalid_format_count", 0)),
        "parsed_count": int(clean_aggregate.get("parsed_count", 0)),
        "unparsed_count": int(clean_aggregate.get("unparsed_count", 0)),
        "content_present_count": int(clean_aggregate.get("content_present_count", 0)),
        "reasoning_present_count": int(clean_aggregate.get("reasoning_present_count", 0)),
        "finish_reason_counts": dict(clean_aggregate.get("finish_reason_counts", {})),
        "completion_tokens": dict(clean_aggregate.get("completion_tokens", {})),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "code_version": CODE_VERSION,
        "code_sha256": code_identity["sha256"],
        "code": code_identity,
        "dependency_hashes": dict(code_identity["dependency_hashes"]),
        "datasets": {DATASET: {"total": EXPECTED_TOTAL, "md5": DATASET_MD5[DATASET]}},
        "correct": int(clean_aggregate["correct"]),
        "total": int(clean_aggregate["total"]),
        "accuracy": float(clean_aggregate["accuracy"]),
        "invalid_count": int(clean_aggregate["invalid_count"]),
        "api_failure_count": int(clean_aggregate.get("api_failure_count", 0)),
        "http_failure_count": int(clean_aggregate.get("http_failure_count", 0)),
        "non_http_api_failure_count": int(clean_aggregate.get("non_http_api_failure_count", 0)),
        "api_error_count": int(clean_aggregate.get("api_error_count", 0)),
        "http_error_count": int(clean_aggregate.get("http_error_count", 0)),
        "invalid_format_count": int(clean_aggregate["invalid_format_count"]),
        "parsed_count": int(clean_aggregate["parsed_count"]),
        "unparsed_count": int(clean_aggregate["unparsed_count"]),
        "content_present_count": int(clean_aggregate.get("content_present_count", 0)),
        "reasoning_present_count": int(clean_aggregate.get("reasoning_present_count", 0)),
        "finish_reason_counts": dict(clean_aggregate.get("finish_reason_counts", {})),
        "completion_tokens": dict(clean_aggregate.get("completion_tokens", {})),
        "completion_tokens_p50": clean_aggregate.get("completion_tokens_p50"),
        "completion_tokens_p90": clean_aggregate.get("completion_tokens_p90"),
        "completion_tokens_max": clean_aggregate.get("completion_tokens_max"),
        "selected_option_counts": dict(clean_aggregate["selected_option_counts"]),
        "scores": clean_aggregate["datasets"],
        "model_id": model_id,
        "model_id_preflight": clean_preflight,
        "preset": PRESET_NAME,
        "protocol": clean_protocol,
        "telemetry": telemetry,
        "hashes": hashes,
        "data_hash": data_hash,
        "prompt_hash": prompt_hash,
        "model_hash": model_hash,
        "protocol_hash": protocol_hash,
        **({"image_manifest_hash": image_manifest_hash} if image_manifest_hash is not None else {}),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blink-tsv", "--blink_tsv", "--blink", dest="blink_tsv", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-id", "--model_id", "--model", dest="model_id", required=True)
    parser.add_argument("--api-base", "--api_base", required=True)
    parser.add_argument("--api-key", "--api_key", default="EMPTY")
    parser.add_argument("--preset", default=PRESET_NAME, choices=(PRESET_NAME,))
    parser.add_argument("--thinking", "--enable-thinking", "--enable_thinking", nargs="?", const=True, default=False, type=_parse_bool)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", "--top_p", type=float, default=1.0)
    parser.add_argument("--top-k", "--top_k", type=int, default=-1)
    parser.add_argument("--min-p", "--min_p", type=float, default=0.0)
    parser.add_argument("--presence-penalty", "--presence_penalty", type=float, default=0.0)
    parser.add_argument("--repetition-penalty", "--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--max-tokens", "--max_tokens", type=int, default=32768)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", "--worker", dest="workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--structured-json", "--structured_json", nargs="?", const=True, default=False, type=_parse_bool)
    parser.add_argument("--image-root", "--image_root", type=Path, default=None)
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
        "preset": PRESET_NAME,
        "protocol": _protocol(args),
        "hashes_deferred": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        _validate_preset(args)
        blink, output = _validate_args(args, inspect_inputs=False)
        if args.dry_run:
            print(json.dumps(_dry_run_metadata(args), sort_keys=True))
            return 0
        model_preflight = _preflight_model_id(args)
        _regular_file(blink, label="BLINK TSV", suffix=".tsv")
        samples, _digest = _read_dataset(blink, image_root=args.image_root)
        image_hash = image_manifest_sha256(samples)
        prompt_hash = _v2._hash_texts(build_blink_prompt(sample) for sample in samples)
        protocol = _protocol(args)
        outcomes = _infer(samples, args=args)
        aggregate = aggregate_outcomes(
            outcomes,
            expected_total=EXPECTED_TOTAL,
            expected_dataset_totals=DATASET_ROWS,
        )
        receipt = _receipt(
            aggregate=aggregate,
            prompt_hash=prompt_hash,
            model_id=args.model_id,
            protocol=protocol,
            model_preflight=model_preflight,
            image_manifest_hash=image_hash,
        )
        _v2._write_create_once(output, receipt)
        print(
            f"MCQ_BLINK_CHECKPOINT_COMPARISON_V5 correct={aggregate['correct']} "
            f"total={aggregate['total']} invalid_count={aggregate['invalid_count']} "
            f"parsed={aggregate['parsed_count']}"
        )
        return 0
    except MCQAggregateError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
