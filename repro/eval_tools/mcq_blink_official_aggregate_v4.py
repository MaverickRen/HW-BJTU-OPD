#!/usr/bin/env python3
"""Aggregate-only BLINK evaluator using the local VLMEvalKit matcher (v4).

This is a new runner layered on the read-only v3 helpers.  Its parser ports
the local ``matching_util.can_infer_option`` followed by
``can_infer_text``: punctuation is tokenized, a unique option token is
accepted only in the final five-token range, the local verbose answer marker
is supported, and a length-gated unique option text is the final fallback.
Rows may contain two, three, or four A--D options.  Refusals, ``Z``, malformed
rows, and ambiguous answers are invalid; no judge or LLM fallback is used.

Only aggregate counters, telemetry, and hashes are written to the receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import local
from typing import Any, Callable


try:
    import mcq_blink_official_aggregate_v3 as _v3
except ModuleNotFoundError as error:  # pragma: no cover - direct script import
    if error.name != "mcq_blink_official_aggregate_v3":
        raise
    _v3_path = Path(__file__).with_name("mcq_blink_official_aggregate_v3.py")
    _v3_spec = importlib.util.spec_from_file_location(
        "mcq_blink_official_aggregate_v3", _v3_path
    )
    if _v3_spec is None or _v3_spec.loader is None:
        raise ImportError("v3 scorer is unavailable") from error
    _v3 = importlib.util.module_from_spec(_v3_spec)
    sys.modules[_v3_spec.name] = _v3
    _v3_spec.loader.exec_module(_v3)


_v2 = _v3._v2
WORKSPACE = _v3.WORKSPACE
OUTPUT_ROOT = _v3.OUTPUT_ROOT
DATASET = _v3.DATASET
EXPECTED_TOTAL = _v3.EXPECTED_TOTAL
DATASET_ROWS = dict(_v3.DATASET_ROWS)
DATASET_MD5 = dict(_v3.DATASET_MD5)
DEFAULT_WORKERS = _v3.DEFAULT_WORKERS
CHOICES = frozenset("ABCD")

SCHEMA_VERSION = "mcq_blink_official_aggregate_v4"
CODE_VERSION = SCHEMA_VERSION
PRESET_NAME = "qwen3.5_official_nonthinking_blink_v4_matching_util"
FROZEN_PRESET = dict(_v3.FROZEN_PRESET)

MCQAggregateError = _v3.MCQAggregateError
LocalInputError = _v3.LocalInputError
InferenceOutcome = _v3.InferenceOutcome

_VERBOSE_ANSWER_RE = re.compile(r"(?i)(?:correct\s+)?answer\s+is\s+\**([ABCD])\**")
_PUNCTUATION = ".()[],:;!*#{}"
_REFUSAL_MESSAGES = (
    "Sorry, I can't help with images of people yet.",
    "I can't process this file.",
    "I'm sorry, but without the image provided",
    "Cannot determine the answer",
)


# Read-only protocol/input helpers from v3.  v4 owns the parser, receipt
# identity, protocol identity, and runner entry point below.
_canonical = _v3._canonical
_absolute = _v3._absolute
_validate_api_base = _v3._validate_api_base
_parse_bool = _v3._parse_bool
_regular_file = _v3._regular_file
_read_dataset = _v3._read_dataset
build_blink_prompt = _v3.build_blink_prompt
build_messages = _v3.build_messages
_new_client = _v3._new_client
image_manifest_sha256 = _v3.image_manifest_sha256
aggregate_outcomes = _v2.aggregate_outcomes
aggregate_counts = aggregate_outcomes


def build_request(
    row: Any,
    *,
    model_id: str,
    image_uris: Sequence[str] | None = None,
    thinking: bool = False,
    temperature: float = 0.7,
    top_p: float = 0.8,
    top_k: int = 20,
    min_p: float = 0.0,
    presence_penalty: float = 1.5,
    repetition_penalty: float = 1.0,
    max_tokens: int = 2048,
    structured_json: bool = False,
    seed: int = 42,
) -> dict[str, Any]:
    controls = locals()
    for name, expected in FROZEN_PRESET.items():
        if controls[name] != expected:
            raise MCQAggregateError(f"frozen preset control differs: {name}")
    if structured_json:
        raise MCQAggregateError("BLINK v4 uses the raw VLMEvalKit prompt")
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


def _options(choices: Mapping[str, Any] | Sequence[tuple[str, str]]) -> dict[str, str] | None:
    if isinstance(choices, Mapping):
        pairs = tuple((str(label), str(value)) for label, value in choices.items())
    else:
        pairs = tuple((str(label), str(value)) for label, value in choices)
    labels = tuple(label for label, _ in pairs)
    if (
        len(labels) < 2
        or len(labels) != len(set(labels))
        or any(not label or label not in CHOICES for label in labels)
    ):
        return None
    return dict(pairs)


def _count_choice(tokens: Sequence[str], choices: Mapping[str, str]) -> int:
    # matching_util.count_choice counts each candidate label at most once;
    # repeated occurrences of the same label do not make the answer multiway.
    return sum(1 for label in choices if label in tokens)


def _tokenize(answer: str) -> list[str]:
    answer_mod = answer
    for char in _PUNCTUATION:
        answer_mod = answer_mod.replace(char, " ")
    return [value.strip() for value in answer_mod.split()]


def _forced_invalid_option(answer: str, choices: Mapping[str, str]) -> bool:
    """Identify local Z/refusal outcomes before text fallback can run."""

    if "Failed to obtain answer via API" in answer:
        return True
    if any(message in answer for message in _REFUSAL_MESSAGES):
        return True
    tokens = _tokenize(answer)
    return _count_choice(tokens, choices) == 0 and (tokens.count("Z") + tokens.count("")) == 1


def _can_infer_option(answer: str, choices: Mapping[str, str]) -> str | None:
    """Port matching_util.can_infer_option with row-local labels and safe Z."""

    if _forced_invalid_option(answer, choices):
        return None

    splits = _tokenize(answer)
    count = _count_choice(splits, choices)

    if count == 1:
        for label in choices:
            # This intentionally mirrors local matching_util's strict
            # ``index > len(tokens)-5`` final-token test.
            if "A" in splits and len(splits) > 3 and os.environ.get("VERBOSE", 0):
                return None
            if label in splits and splits.index(label) > (len(splits) - 5):
                return label
    elif count == 0 and (splits.count("Z") + splits.count("")) == 1:
        return None

    match = _VERBOSE_ANSWER_RE.search(answer or "")
    if match and match.group(1).upper() in choices:
        return match.group(1).upper()
    return None


def _can_infer_text(answer: str, choices: Mapping[str, str]) -> str | None:
    """Port matching_util.can_infer_text without mutating caller data."""

    answer_lower = answer.lower()
    if len(answer_lower) > 2 * sum(len(str(value)) for value in choices.values()):
        return None
    candidates = [
        label for label, value in choices.items() if str(value).lower() in answer_lower
    ]
    return candidates[0] if len(candidates) == 1 else None


def can_infer(
    answer: Any,
    choices: Mapping[str, Any] | Sequence[tuple[str, str]],
) -> str | None:
    """Match local VLMEvalKit option/text semantics, returning None if invalid."""

    option_map = _options(choices)
    if option_map is None:
        return None
    text = str(answer)
    if _forced_invalid_option(text, option_map):
        return None
    return _can_infer_option(text, option_map) or _can_infer_text(text, option_map)


parse_choice = can_infer
parse_answer = can_infer
parse_model_answer = can_infer


def _code_identity() -> dict[str, Any]:
    """Bind all four scorer sources without exposing source contents."""

    def digest(path: Path, label: str) -> str:
        return _v2._file_sha256(path, label=label)

    source_hashes = {
        "v1": digest(Path(_v2._v1.__file__), "v1 scorer source"),
        "v2": digest(Path(_v2.__file__), "v2 scorer source"),
        "v3": digest(Path(_v3.__file__), "v3 scorer source"),
        "v4": digest(Path(__file__), "v4 scorer source"),
    }
    return {
        "version": CODE_VERSION,
        "sha256": hashlib.sha256(_canonical(source_hashes)).hexdigest(),
        "runner_sha256": source_hashes["v4"],
        "dependency_hashes": dict(source_hashes),
        "source_hashes": source_hashes,
    }


def _source_sha256() -> str:
    return str(_code_identity()["sha256"])


def _preflight_model_id(args: argparse.Namespace, client: Any | None = None) -> dict[str, Any]:
    return _v3._preflight_model_id(args, client)


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
                dataset=DATASET,
                gold=sample.gold,
                prediction=None,
                provider_failed=True,
                failure_kind=failure_kind,
            )
        choice = _v2._choice(response)
        if choice is None or _v2._message(response) is None:
            raise LocalInputError("provider returned a malformed choice")
        content = _v2._response_content(response)
        prediction = can_infer(content, sample.options)
        return InferenceOutcome(
            dataset=DATASET,
            gold=sample.gold,
            prediction=prediction,
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
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "repetition_penalty": 1.0,
        "max_tokens": 2048,
        "seed": 42,
        "workers": args.workers,
        "dataset": DATASET,
        "prompt_source": "VLMEvalKit.ImageMCQDataset",
        "prompt_only": True,
        "response_format": "raw_vlmevalkit_prompt_matching_util_exact_matching",
        "image_order": "source_tsv_order",
        "gold_scope": "scorer_memory_only",
        "sample_level_output": False,
    }


_PROTOCOL_KEYS = frozenset(_protocol(argparse.Namespace(workers=1)))


def _sanitize_protocol(protocol: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(protocol, Mapping) or set(protocol) - _PROTOCOL_KEYS:
        raise MCQAggregateError("protocol contains an unsupported field")
    result = dict(protocol)
    if result.get("dataset") != DATASET or result.get("preset") != PRESET_NAME:
        raise MCQAggregateError("protocol identity differs")
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
        raise MCQAggregateError("only the frozen Qwen3.5 non-thinking BLINK v4 preset is supported")
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
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", "--top_p", type=float, default=0.8)
    parser.add_argument("--top-k", "--top_k", type=int, default=20)
    parser.add_argument("--min-p", "--min_p", type=float, default=0.0)
    parser.add_argument("--presence-penalty", "--presence_penalty", type=float, default=1.5)
    parser.add_argument("--repetition-penalty", "--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--max-tokens", "--max_tokens", type=int, default=2048)
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
            f"MCQ_BLINK_OFFICIAL_V4 correct={aggregate['correct']} "
            f"total={aggregate['total']} invalid_count={aggregate['invalid_count']} "
            f"parsed={aggregate['parsed_count']}"
        )
        return 0
    except MCQAggregateError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
