#!/usr/bin/env python3
"""Aggregate-only BLINK evaluator with the local VLMEvalKit protocol (v3).

This evaluator is intentionally independent from the running v1/v2 files.  It
imports their read-only helpers for the pinned BLINK data contract, local
model-id preflight, privacy-safe telemetry, image manifest hashing, and
durable create-once publication, but it freezes a separate non-thinking
protocol:

* the text prompt is the local ``ImageMCQDataset`` prompt;
* Qwen3.5 instruct-mode sampling is temperature=0.7, top_p=0.8, top_k=20,
  min_p=0, presence_penalty=1.5, repetition_penalty=1;
* the answer parser is deterministic exact matching only; and
* only aggregate counters, telemetry counters, and cryptographic bindings are
  persisted.  Questions, images, gold answers, predictions, response text,
  and reasoning text never enter a receipt or stdout.

``--dry-run`` is a no-read/no-connect/no-write operation.  A real run uses a
local OpenAI-compatible endpoint and publishes one receipt below ``Output``.
It never calls an external judge or LLM.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from threading import local
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable


try:
    import mcq_choice_formal_aggregate_v2 as _v2
except ModuleNotFoundError as error:  # pragma: no cover - direct script import
    if error.name != "mcq_choice_formal_aggregate_v2":
        raise
    _v2_path = Path(__file__).with_name("mcq_choice_formal_aggregate_v2.py")
    _v2_spec = importlib.util.spec_from_file_location(
        "mcq_choice_formal_aggregate_v2", _v2_path
    )
    if _v2_spec is None or _v2_spec.loader is None:
        raise ImportError("v2 scorer is unavailable") from error
    _v2 = importlib.util.module_from_spec(_v2_spec)
    sys.modules[_v2_spec.name] = _v2
    _v2_spec.loader.exec_module(_v2)


WORKSPACE = _v2.WORKSPACE
OUTPUT_ROOT = _v2.OUTPUT_ROOT
CHOICES = _v2.CHOICES
DATASET = "BLINK"
EXPECTED_TOTAL = _v2.EXPECTED_BLINK_ROWS
DATASET_ROWS = {DATASET: EXPECTED_TOTAL}
DATASET_MD5 = {DATASET: _v2.BLINK_MD5}
DEFAULT_WORKERS = _v2.DEFAULT_WORKERS

SCHEMA_VERSION = "mcq_blink_official_aggregate_v3"
CODE_VERSION = SCHEMA_VERSION
PRESET_NAME = "qwen3.5_official_nonthinking_blink"

# Qwen3.5 README instruct/general-task recommendation.  2048 is deliberately
# sufficient for a direct BLINK MCQ response while preventing the long,
# unconstrained completions that invalidated the earlier run.
FROZEN_PRESET = {
    "thinking": False,
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 1.5,
    "repetition_penalty": 1.0,
    "max_tokens": 2048,
    "structured_json": False,
    "seed": 42,
}

_WHITESPACE_RE = re.compile(r"\s+")
_ANSWER_MARKER_RE = re.compile(
    r"(?is)\b(?:final\s+)?(?:answer|choice|option)\b\s*"
    r"(?:is\s*)?[:=\-]?\s*[\[\(\{`'\"*]*([A-D])"
    r"(?=$|[\s\]})'\"*.,;:)])"
)
_TERMINAL_RE = re.compile(
    r"(?is)(?:^|\n)\s*[\[\(\{`'\"*]*([A-D])"
    r"[\]})'\"*.,;:!?]*\s*$"
)


MCQAggregateError = _v2.MCQAggregateError
LocalInputError = _v2.LocalInputError
InferenceOutcome = _v2.InferenceOutcome


def _canonical(value: Any) -> bytes:
    return _v2._canonical(value)


def _absolute(path: Path | str) -> Path:
    return _v2._absolute(path)


def _validate_api_base(value: str) -> str:
    return _v2._validate_api_base(value)


def _parse_bool(value: str | bool) -> bool:
    return _v2._parse_bool(value)


def _regular_file(path: Path | str, *, label: str, suffix: str | None = None) -> Path:
    return _v2._regular_file(path, label=label, suffix=suffix)


def _read_dataset(path: Path, *, image_root: Path | None = None):
    return _v2._read_dataset(
        path, DATASET, verify_hash=True, image_root=image_root
    )


def _normalize_text(value: str) -> str:
    return _WHITESPACE_RE.sub(" ", value.casefold()).strip()


def _row_parts(row: Any) -> tuple[str, tuple[tuple[str, str], ...], str | None, tuple[str, ...]]:
    if isinstance(row, _v2.Sample):
        return row.question, row.options, row.hint, row.images
    question = row.get("question") if isinstance(row, Mapping) else None
    if not isinstance(question, str):
        raise MCQAggregateError("question must be text")
    options = _v2._option_values(row)
    hint_value = row.get("hint")
    hint = hint_value if isinstance(hint_value, str) and hint_value else None
    images = tuple(str(value) for value in _v2._row_images(row))
    return question, options, hint, images


def build_blink_prompt(row: Any) -> str:
    """Reproduce ``ImageMCQDataset.build_prompt`` for BLINK exactly."""

    question, options, hint, _ = _row_parts(row)
    prompt = ""
    if hint is not None:
        prompt += f"Hint: {hint}\n"
    prompt += f"Question: {question}\n"
    if options:
        prompt += "Options:\n"
        for label, value in options:
            prompt += f"{label}. {value}\n"
        prompt += "Please select the correct answer from the options above. \n"
    return prompt


def build_messages(row: Any, image_uris: Sequence[str] | None = None) -> list[dict[str, Any]]:
    """Build ordered multimodal messages without changing the BLINK prompt."""

    _, _, _, images = _row_parts(row)
    uris = list(image_uris) if image_uris is not None else [_v2._image_data_uri(image) for image in images]
    if len(uris) != len(images):
        raise MCQAggregateError("image URI count differs from the input image count")
    content = [{"type": "image_url", "image_url": {"url": uri}} for uri in uris]
    content.append({"type": "text", "text": build_blink_prompt(row)})
    return [{"role": "user", "content": content}]


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
        raise MCQAggregateError("BLINK v3 uses the raw VLMEvalKit prompt")
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


def _unique_option_text(answer: str, options: Sequence[tuple[str, str]]) -> str | None:
    normalized_answer = _normalize_text(answer)
    if not normalized_answer:
        return None
    if len(normalized_answer) > 2 * sum(len(_normalize_text(value)) for _, value in options):
        return None
    matches: list[str] = []
    for label, value in options:
        normalized_value = _normalize_text(value)
        if len(normalized_value) < 2:
            continue
        if normalized_value in normalized_answer:
            matches.append(label)
    return matches[0] if len(matches) == 1 else None


def can_infer(answer: Any, choices: Mapping[str, Any] | Sequence[tuple[str, str]]) -> str | None:
    """Deterministically map a BLINK response to one option or ``None``.

    Accepted forms are a terminal choice (``B``/``(B)``/``B.``), one unique
    ``answer``/``choice``/``option`` marker, or one unique option text.  A
    contradictory marker, multiple option texts, Z, or an incidental letter
    in reasoning is deliberately unparsed.
    """

    if not isinstance(answer, str):
        return None
    text = answer.strip()
    if not text:
        return None
    if isinstance(choices, Mapping):
        options = tuple((str(label), str(value)) for label, value in choices.items())
    else:
        options = tuple((str(label), str(value)) for label, value in choices)
    labels = tuple(label for label, _ in options)
    # BLINK rows do not all have four choices.  Validate the row's actual
    # labels, then restrict every inference path to that row-local set.
    if (
        len(labels) < 2
        or any(not label or label not in CHOICES for label in labels)
        or len(set(labels)) != len(labels)
    ):
        return None
    valid = set(labels)

    terminal = _TERMINAL_RE.search(text)
    if terminal and terminal.group(1) in valid:
        return terminal.group(1)

    markers = [match.group(1) for match in _ANSWER_MARKER_RE.finditer(text)]
    marker_values = set(markers)
    if len(marker_values) == 1 and markers[0] in valid:
        return markers[0]
    if marker_values:
        return None
    return _unique_option_text(text, options)


parse_choice = can_infer
parse_answer = can_infer
parse_model_answer = can_infer


def _code_identity() -> dict[str, Any]:
    """Hash v1, v2, and this independent v3 runner without source output."""

    def digest(path: Path, label: str) -> str:
        return _v2._file_sha256(path, label=label)

    v1_path = Path(_v2._v1.__file__)
    v2_path = Path(_v2.__file__)
    source_hashes = {
        "v1": digest(v1_path, "v1 scorer source"),
        "v2": digest(v2_path, "v2 scorer source"),
        "v3": digest(Path(__file__), "v3 runner source"),
    }
    return {
        "version": CODE_VERSION,
        "sha256": hashlib.sha256(_canonical(source_hashes)).hexdigest(),
        "runner_sha256": source_hashes["v3"],
        "dependency_hashes": {"v1": source_hashes["v1"], "v2": source_hashes["v2"]},
        "source_hashes": source_hashes,
    }


def _source_sha256() -> str:
    return str(_code_identity()["sha256"])


def _preflight_model_id(args: argparse.Namespace, client: Any | None = None) -> dict[str, Any]:
    return _v2._preflight_model_id(args, client)


preflight_model_id = _preflight_model_id


def _new_client(args: argparse.Namespace) -> Any:
    return _v2._new_client(args)


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
        finish_reason = _v2._finish_bucket(_v2._field(choice, "finish_reason"))
        return InferenceOutcome(
            dataset=DATASET,
            gold=sample.gold,
            prediction=prediction,
            finish_reason=finish_reason,
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


aggregate_outcomes = _v2.aggregate_outcomes
aggregate_counts = aggregate_outcomes
image_manifest_sha256 = _v2._blink_image_manifest_sha256


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
        "response_format": "raw_vlmevalkit_prompt_deterministic_exact_matching",
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
        raise MCQAggregateError("only the frozen Qwen3.5 non-thinking BLINK preset is supported")
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
        samples, digest = _read_dataset(blink, image_root=args.image_root)
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
        # Keep the v2 durable O_EXCL/O_NOFOLLOW/fsync implementation, but do
        # not call the v2 receipt builder or alter its source/hash contract.
        _v2._write_create_once(output, receipt)
        print(
            f"MCQ_BLINK_OFFICIAL_V3 correct={aggregate['correct']} "
            f"total={aggregate['total']} invalid_count={aggregate['invalid_count']} "
            f"parsed={aggregate['parsed_count']}"
        )
        return 0
    except MCQAggregateError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
