#!/usr/bin/env python3
"""Formal aggregate-only MMStar/BLINK choice evaluation (v2).

This runner deliberately builds on :mod:`mcq_choice_formal_aggregate_v1` for
the pinned TSV reader, prompt, parser, image handling, and scoring contract.
The v2 run is frozen to the official Qwen3.5 thinking MCQ controls and adds
aggregate-only provider telemetry.  In particular, no response or reasoning
text is retained in an outcome, receipt, exception, or stdout line.

``--dry-run`` remains a no-read/no-connect/no-write operation.  A real run
first checks that the requested model id is an exact id advertised by the
local OpenAI-compatible server, then performs inference and writes one
create-once aggregate receipt below ``H_Workspace/Output``.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import math
import os
import stat
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.util
from pathlib import Path
from threading import local
from typing import Any, Callable, Iterable, Mapping, NamedTuple, Sequence
from urllib.parse import urlparse

try:
    import mcq_choice_formal_aggregate_v1 as _v1
except ModuleNotFoundError as error:  # pragma: no cover - importlib test path
    if error.name != "mcq_choice_formal_aggregate_v1":
        raise
    _v1_path = Path(__file__).with_name("mcq_choice_formal_aggregate_v1.py")
    _v1_spec = importlib.util.spec_from_file_location("mcq_choice_formal_aggregate_v1", _v1_path)
    if _v1_spec is None or _v1_spec.loader is None:
        raise ImportError("v1 scorer is unavailable") from error
    _v1 = importlib.util.module_from_spec(_v1_spec)
    sys.modules[_v1_spec.name] = _v1
    _v1_spec.loader.exec_module(_v1)


# Re-export the v1 data and prompt/scoring contract.  Keeping these aliases
# avoids a second implementation which could silently drift from the formal
# MMStar/BLINK run while v1 may be executing in another process.
WORKSPACE = _v1.WORKSPACE
OUTPUT_ROOT = _v1.OUTPUT_ROOT
CHOICES = _v1.CHOICES
DATASET_ROWS = _v1.DATASET_ROWS
EXPECTED_TOTAL = _v1.EXPECTED_TOTAL
EXPECTED_MMSTAR_ROWS = _v1.EXPECTED_MMSTAR_ROWS
EXPECTED_BLINK_ROWS = _v1.EXPECTED_BLINK_ROWS
DATASET_MD5 = _v1.DATASET_MD5
MMSTAR_MD5 = _v1.MMSTAR_MD5
BLINK_MD5 = _v1.BLINK_MD5
OFFICIAL_INSTRUCTION = _v1.OFFICIAL_INSTRUCTION
STRUCTURED_JSON_SCHEMA = _v1.STRUCTURED_JSON_SCHEMA
Sample = _v1.Sample
MCQAggregateError = _v1.MCQAggregateError
_absolute = _v1._absolute
_regular_file = _v1._regular_file
_md5 = _v1._md5
_canonical = _v1._canonical
_sha256_bytes = _v1._sha256_bytes
_hash_texts = _v1._hash_texts
_parse_list_field = _v1._parse_list_field
_row_images = _v1._row_images
_resolve_images = _v1._resolve_images
_option_values = _v1._option_values
_read_dataset = _v1._read_dataset
load_tsv_records = _v1.load_tsv_records
build_prompt = _v1.build_prompt
parse_choice = _v1.parse_choice
parse_answer = _v1.parse_answer
extract_answer = _v1.extract_answer
parse_json_answer = _v1.parse_json_answer
parse_model_answer = _v1.parse_model_answer
_image_data_uri = _v1._image_data_uri
build_messages = _v1.build_messages


SCHEMA_VERSION = "mcq_choice_formal_aggregate_v2"
CODE_VERSION = SCHEMA_VERSION
PRESET_NAME = "qwen3.5_official_thinking_mcq"
FROZEN_PRESET_NAME = PRESET_NAME

# These are intentionally literals, rather than defaults inherited from v1.
# The official thinking MCQ run is prompt-only: structured output is not sent
# to the server, so the answer-field instruction in the prompt is authoritative.
FROZEN_PRESET: Mapping[str, Any] = {
    "thinking": True,
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 1.5,
    "repetition_penalty": 1.0,
    "max_tokens": 32768,
    "structured_json": False,
    "seed": 42,
}
OFFICIAL_THINKING_PRESET = FROZEN_PRESET
DEFAULT_WORKERS = _v1.DEFAULT_WORKERS
LOCAL_HOSTS = _v1.LOCAL_HOSTS


class InferenceOutcome(NamedTuple):
    """Private per-row result; only its aggregate projection is persisted."""

    dataset: str
    gold: str
    prediction: str | None
    provider_failed: bool = False
    failure_kind: str | None = None
    finish_reason: str | None = None
    content_present: bool = False
    reasoning_present: bool = False
    completion_tokens: int | None = None


class LocalInputError(MCQAggregateError):
    """A local request/response-shape/input error; never score as provider failure."""


class _ProviderFailure(Exception):
    """Private marker for an explicitly recognized OpenAI provider failure."""

    def __init__(self, failure_kind: str) -> None:
        super().__init__()
        self.failure_kind = failure_kind


_MISSING = object()


def _file_sha256(path: Path, *, label: str) -> str:
    try:
        digest = hashlib.sha256()
        with path.open("rb", buffering=0) as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError as error:  # pragma: no cover - installation guard
        raise MCQAggregateError(f"{label} is unavailable") from None


def _code_identity() -> dict[str, Any]:
    """Hash v2 and its v1 dependency without exposing source text."""

    runner_sha = _file_sha256(Path(__file__), label="v2 runner source")
    dependency_raw = getattr(_v1, "__file__", None)
    if not isinstance(dependency_raw, str) or not dependency_raw:
        raise MCQAggregateError("v1 scorer source is unavailable")
    dependency_path = Path(dependency_raw)
    dependency_sha = _file_sha256(dependency_path, label="v1 scorer source")
    source_hashes = {"v1": dependency_sha, "v2": runner_sha}
    combined_sha = hashlib.sha256(_canonical(source_hashes)).hexdigest()
    return {
        "version": CODE_VERSION,
        "sha256": combined_sha,
        "runner_sha256": runner_sha,
        "dependency_hashes": {"v1": dependency_sha},
        "source_hashes": source_hashes,
    }


def _source_sha256() -> str:
    """Return the combined v1+v2 identity hash."""

    return str(_code_identity()["sha256"])


def _field(value: Any, name: str, default: Any = None) -> Any:
    """Read an SDK object or a synthetic mapping without copying its content."""

    if isinstance(value, Mapping):
        return value.get(name, default)
    try:
        return getattr(value, name)
    except Exception:
        return default


def _choice(response: Any) -> Any | None:
    choices = _field(response, "choices")
    if isinstance(choices, (str, bytes)) or not isinstance(choices, Sequence) or not choices:
        return None
    return choices[0]


def _message(response: Any) -> Any | None:
    choice = _choice(response)
    return _field(choice, "message") if choice is not None else None


def _response_content(response: Any) -> Any:
    message = _message(response)
    return _field(message, "content") if message is not None else None


def _response_reasoning_present(response: Any) -> bool:
    """Count either vLLM reasoning field, without retaining either value.

    Current local vLLM exposes ``message.reasoning`` while older OpenAI
    compatible responses used ``reasoning_content``.  Presence is an OR: if
    both fields exist (even with different values), this still contributes
    exactly one count and neither value is copied into an outcome.
    """

    message = _message(response)
    if message is None:
        return False
    reasoning = _field(message, "reasoning", _MISSING)
    legacy = _field(message, "reasoning_content", _MISSING)
    return (reasoning is not _MISSING and reasoning is not None) or (
        legacy is not _MISSING and legacy is not None
    )


def _response_reasoning(response: Any) -> bool:
    """Compatibility alias returning only the privacy-safe presence bit."""

    return _response_reasoning_present(response)


def _response_text(response: Any) -> str | None:
    content = _response_content(response)
    return content if isinstance(content, str) else None


def _finish_bucket(value: Any) -> str:
    if value == "stop":
        return "stop"
    if value == "length":
        return "length"
    return "other"


def _completion_tokens(response: Any) -> int | None:
    usage = _field(response, "usage")
    value = _field(usage, "completion_tokens") if usage is not None else None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float) and not value.is_integer():
        return None
    integer = int(value)
    return integer if integer >= 0 else None


def _status_code(error: BaseException) -> int | None:
    value = _field(error, "status_code")
    if value is None:
        response = _field(error, "response")
        value = _field(response, "status_code")
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if 100 <= value <= 599 else None


_OPENAI_PROVIDER_NAMES = frozenset(
    {
        "APIError",
        "APIConnectionError",
        "APITimeoutError",
        "APIStatusError",
        "BadRequestError",
        "AuthenticationError",
        "PermissionDeniedError",
        "NotFoundError",
        "ConflictError",
        "UnprocessableEntityError",
        "RateLimitError",
        "InternalServerError",
    }
)


def _is_openai_provider_exception(error: BaseException) -> bool:
    """Recognize only OpenAI SDK transport/provider exception classes."""

    error_type = type(error)
    for candidate in error_type.__mro__:
        module = candidate.__module__
        if (module == "openai" or module.startswith("openai.")) and candidate.__name__ in _OPENAI_PROVIDER_NAMES:
            return True
    return False


def _provider_failure_kind(error: BaseException) -> str | None:
    if not _is_openai_provider_exception(error):
        return None
    return "http" if _status_code(error) is not None else "api"


def _failure_kind(error: BaseException) -> str:
    """Classify a recognized provider exception without exposing its text."""

    return _provider_failure_kind(error) or "api"


def _coerce_outcome(value: InferenceOutcome | Mapping[str, Any] | Sequence[Any]) -> InferenceOutcome:
    """Accept private test fixtures and the v2 internal outcome type."""

    if isinstance(value, InferenceOutcome):
        return value
    if isinstance(value, Mapping):
        provider_failed = value.get("provider_failed", False)
        if not isinstance(provider_failed, bool):
            raise MCQAggregateError("in-memory provider status is malformed")
        return InferenceOutcome(
            dataset=value.get("dataset"),
            gold=value.get("gold"),
            prediction=value.get("prediction"),
            provider_failed=provider_failed,
            failure_kind=value.get("failure_kind"),
            finish_reason=value.get("finish_reason"),
            content_present=value.get("content_present", False),
            reasoning_present=value.get("reasoning_present", False),
            completion_tokens=value.get("completion_tokens"),
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) == 4:
            dataset, gold, prediction, provider_failed = value
            return InferenceOutcome(dataset, gold, prediction, bool(provider_failed))
        if len(value) == 8:
            dataset, gold, prediction, provider_failed, finish_reason, content, reasoning, tokens = value
            return InferenceOutcome(
                dataset, gold, prediction, bool(provider_failed), None,
                finish_reason, bool(content), bool(reasoning), tokens,
            )
        if len(value) == 9:
            return InferenceOutcome(*value)  # type: ignore[arg-type]
        if len(value) == 10:
            # Also accept an explicit HTTP/API pair in CPU fixtures.  The
            # aggregate projection remains the canonical representation.
            dataset, gold, prediction, provider_failed, failure_kind, finish_reason, content, reasoning, tokens, _ = value
            return InferenceOutcome(dataset, gold, prediction, bool(provider_failed), failure_kind, finish_reason, bool(content), bool(reasoning), tokens)
    raise MCQAggregateError("in-memory provider outcome is malformed")


def _nearest_rank(values: Sequence[int], fraction: float) -> int | None:
    if not values:
        return None
    rank = max(1, int(math.ceil(fraction * len(values))))
    return sorted(values)[rank - 1]


def _token_stats(tokens: Sequence[int]) -> dict[str, Any]:
    ordered = sorted(tokens)
    return {
        "count": len(ordered),
        "percentile_method": "nearest_rank",
        "p50": _nearest_rank(ordered, 0.50),
        "p90": _nearest_rank(ordered, 0.90),
        "max": max(ordered) if ordered else None,
    }


def aggregate_outcomes(
    outcomes: Sequence[InferenceOutcome | Mapping[str, Any] | Sequence[Any]],
    *,
    expected_total: int | None = None,
    expected_dataset_totals: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Reduce private outcomes to scores and aggregate-only telemetry."""

    selected: Counter[str] = Counter()
    dataset_selected: dict[str, Counter[str]] = {}
    dataset_total: Counter[str] = Counter()
    dataset_correct: Counter[str] = Counter()
    dataset_api_failure: Counter[str] = Counter()
    dataset_http_failure: Counter[str] = Counter()
    dataset_invalid_format: Counter[str] = Counter()
    finish = Counter({"stop": 0, "length": 0, "other": 0})
    dataset_finish: dict[str, Counter[str]] = {}
    correct = api_failure = http_failure = invalid_format = 0
    content_present = reasoning_present = parsed = 0
    tokens: list[int] = []
    normalized: list[InferenceOutcome] = []

    for raw in outcomes:
        outcome = _coerce_outcome(raw)
        normalized.append(outcome)
        if not isinstance(outcome.dataset, str) or not outcome.dataset:
            raise MCQAggregateError("in-memory dataset label is malformed")
        if outcome.gold not in CHOICES:
            raise MCQAggregateError("in-memory gold answer is malformed")
        if not isinstance(outcome.provider_failed, bool):
            raise MCQAggregateError("in-memory provider status is malformed")
        if not isinstance(outcome.content_present, bool) or not isinstance(outcome.reasoning_present, bool):
            raise MCQAggregateError("in-memory presence telemetry is malformed")
        if outcome.failure_kind not in {None, "api", "http"}:
            raise MCQAggregateError("in-memory provider failure kind is malformed")
        if outcome.provider_failed and outcome.prediction is not None:
            raise MCQAggregateError("provider failure cannot contain a prediction")
        if not outcome.provider_failed and outcome.prediction is not None and outcome.prediction not in CHOICES:
            raise MCQAggregateError("in-memory prediction is malformed")
        if outcome.completion_tokens is not None and (
            isinstance(outcome.completion_tokens, bool)
            or not isinstance(outcome.completion_tokens, int)
            or outcome.completion_tokens < 0
        ):
            raise MCQAggregateError("in-memory completion token count is malformed")

        dataset_total[outcome.dataset] += 1
        dataset_selected.setdefault(outcome.dataset, Counter())
        dataset_finish.setdefault(outcome.dataset, Counter({"stop": 0, "length": 0, "other": 0}))
        if not outcome.provider_failed:
            bucket = _finish_bucket(outcome.finish_reason)
            finish[bucket] += 1
            dataset_finish[outcome.dataset][bucket] += 1
        content_present += int(outcome.content_present)
        reasoning_present += int(outcome.reasoning_present)
        if outcome.completion_tokens is not None:
            tokens.append(outcome.completion_tokens)

        if outcome.provider_failed:
            api_failure += 1
            dataset_api_failure[outcome.dataset] += 1
            if outcome.failure_kind == "http":
                http_failure += 1
                dataset_http_failure[outcome.dataset] += 1
            continue
        if outcome.prediction not in CHOICES:
            invalid_format += 1
            dataset_invalid_format[outcome.dataset] += 1
            continue
        parsed += 1
        selected[outcome.prediction] += 1  # type: ignore[index]
        dataset_selected[outcome.dataset][outcome.prediction] += 1  # type: ignore[index]
        correct += int(outcome.gold == outcome.prediction)
        dataset_correct[outcome.dataset] += int(outcome.gold == outcome.prediction)

    total = len(normalized)
    if expected_total is not None and total != expected_total:
        raise MCQAggregateError("aggregate row count differs from the formal total")
    if expected_dataset_totals is not None and dict(dataset_total) != dict(expected_dataset_totals):
        raise MCQAggregateError("aggregate dataset row counts differ from the formal totals")

    datasets: dict[str, Any] = {}
    for dataset in dataset_total:
        datasets[dataset] = {
            "correct": dataset_correct[dataset],
            "total": dataset_total[dataset],
            "accuracy": dataset_correct[dataset] / dataset_total[dataset],
            "invalid_count": dataset_api_failure[dataset] + dataset_invalid_format[dataset],
            "api_failure_count": dataset_api_failure[dataset],
            "http_failure_count": dataset_http_failure[dataset],
            "invalid_format_count": dataset_invalid_format[dataset],
            "parsed_count": sum(dataset_selected[dataset].values()),
            "content_present_count": sum(
                int(item.dataset == dataset and item.content_present) for item in normalized
            ),
            "reasoning_present_count": sum(
                int(item.dataset == dataset and item.reasoning_present) for item in normalized
            ),
            "finish_reason_counts": dict(dataset_finish[dataset]),
            "selected_option_counts": {choice: dataset_selected[dataset][choice] for choice in CHOICES},
            "completion_tokens": _token_stats([
                item.completion_tokens for item in normalized
                if item.dataset == dataset and item.completion_tokens is not None
            ]),
        }
    stats = _token_stats(tokens)
    return {
        "correct": correct,
        "total": total,
        "accuracy": (correct / total) if total else 0.0,
        "invalid_count": api_failure + invalid_format,
        "api_failure_count": api_failure,
        "http_failure_count": http_failure,
        "non_http_api_failure_count": api_failure - http_failure,
        "api_error_count": api_failure - http_failure,
        "http_error_count": http_failure,
        "invalid_format_count": invalid_format,
        "parsed_count": parsed,
        "unparsed_count": total - parsed,
        "content_present_count": content_present,
        "reasoning_present_count": reasoning_present,
        "content_count": content_present,
        "reasoning_count": reasoning_present,
        "finish_reason_counts": {name: finish[name] for name in ("stop", "length", "other")},
        "finish_reason_stop_count": finish["stop"],
        "finish_reason_length_count": finish["length"],
        "finish_reason_other_count": finish["other"],
        "completion_tokens": stats,
        # Flat aliases make the receipt easy to query while retaining the
        # grouped representation for consumers that prefer a namespace.
        "completion_tokens_p50": stats["p50"],
        "completion_tokens_p90": stats["p90"],
        "completion_tokens_max": stats["max"],
        "selected_option_counts": {choice: selected[choice] for choice in CHOICES},
        "datasets": datasets,
    }


aggregate_counts = aggregate_outcomes


def build_request(
    row: Sample | Mapping[str, Any],
    *,
    model_id: str,
    image_uris: Sequence[str] | None = None,
    thinking: bool = True,
    temperature: float = 1.0,
    top_p: float = 0.95,
    top_k: int = 20,
    min_p: float = 0.0,
    presence_penalty: float = 1.5,
    repetition_penalty: float = 1.0,
    max_tokens: int = 32768,
    structured_json: bool = False,
    seed: int = 42,
) -> dict[str, Any]:
    """Build the prompt-only request using the frozen Qwen3.5 controls."""

    controls = {
        "thinking": thinking,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "min_p": min_p,
        "presence_penalty": presence_penalty,
        "repetition_penalty": repetition_penalty,
        "max_tokens": max_tokens,
        "structured_json": structured_json,
        "seed": seed,
    }
    if structured_json:
        raise MCQAggregateError("the frozen thinking MCQ preset is prompt-only")
    for name, expected in FROZEN_PRESET.items():
        if controls[name] != expected:
            raise MCQAggregateError(f"frozen preset control differs: {name}")
    return _v1.build_request(
        row,
        model_id=model_id,
        image_uris=image_uris,
        thinking=thinking,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        presence_penalty=presence_penalty,
        repetition_penalty=repetition_penalty,
        max_tokens=max_tokens,
        structured_json=False,
        seed=seed,
    )


def _validate_api_base(value: str) -> str:
    return _v1._validate_api_base(value)


def _parse_bool(value: str | bool) -> bool:
    return _v1._parse_bool(value)


def _assert_output_scope(path: Path, *, allow_missing: bool = True) -> Path:
    """Validate output ancestors against this v2 module's output root."""

    target = _absolute(path)
    root = _absolute(OUTPUT_ROOT)
    try:
        relative = target.relative_to(root)
    except ValueError:
        raise MCQAggregateError("output must be below H_Workspace/Output") from None
    try:
        root_info = root.lstat()
    except FileNotFoundError:
        if not allow_missing:
            raise MCQAggregateError("output root is unavailable") from None
        root_info = None
    except OSError:
        raise MCQAggregateError("output root is unavailable") from None
    if root_info is not None and (stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode)):
        raise MCQAggregateError("output root must be a real directory")
    cursor = root
    for component in relative.parts[:-1]:
        cursor = cursor / component
        try:
            info = cursor.lstat()
        except FileNotFoundError:
            break
        except OSError:
            raise MCQAggregateError("output parent is unavailable") from None
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise MCQAggregateError("output parent must contain no symlinks")
    try:
        existing = target.exists() or target.is_symlink()
    except OSError:
        raise MCQAggregateError("output target is unavailable") from None
    if existing:
        raise MCQAggregateError("aggregate receipt already exists")
    return target


def _validate_preset(args: argparse.Namespace) -> None:
    if getattr(args, "preset", PRESET_NAME) != PRESET_NAME:
        raise MCQAggregateError("only the frozen Qwen3.5 thinking MCQ preset is supported")
    for key, expected in FROZEN_PRESET.items():
        actual = getattr(args, key)
        if actual != expected:
            raise MCQAggregateError(f"frozen preset control differs: {key}")


def _validate_args(args: argparse.Namespace, *, inspect_inputs: bool) -> tuple[Path, Path, Path]:
    _validate_preset(args)
    if args.workers < 1 or args.workers > 256:
        raise MCQAggregateError("workers must be in [1, 256]")
    if len(set(args.datasets)) != len(args.datasets):
        raise MCQAggregateError("datasets must contain no duplicates")
    mmstar = _absolute(args.mmstar_tsv)
    blink = _absolute(args.blink_tsv)
    output = _absolute(args.output)
    if output == _absolute(OUTPUT_ROOT) or output.is_dir():
        raise MCQAggregateError("output must be a receipt file")
    _assert_output_scope(output)
    _validate_api_base(args.api_base)
    if inspect_inputs:
        _regular_file(mmstar, label="MMStar TSV", suffix=".tsv")
        _regular_file(blink, label="BLINK TSV", suffix=".tsv")
    if not isinstance(args.model_id, str) or not args.model_id.strip():
        raise MCQAggregateError("model id must be non-empty")
    return mmstar, blink, output


def _protocol(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "preset": PRESET_NAME,
        "thinking": True,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "repetition_penalty": 1.0,
        "max_tokens": 32768,
        "seed": 42,
        "workers": args.workers,
        "datasets": list(args.datasets),
        "structured_json": False,
        "prompt_only": True,
        "response_format": "prompt_only_explicit_answer_or_terminal_A_D",
        "image_order": "source_tsv_order",
        "gold_scope": "scorer_memory_only",
        "sample_level_output": False,
    }


def _new_client(args: argparse.Namespace) -> Any:
    try:
        from openai import OpenAI
    except Exception as error:  # pragma: no cover - runtime guard
        raise MCQAggregateError("openai package is required for a real run") from error
    return OpenAI(
        api_key=args.api_key,
        base_url=_validate_api_base(args.api_base),
        timeout=3600,
        max_retries=0,
    )


def _model_ids(payload: Any) -> list[str]:
    data = _field(payload, "data")
    if data is None and isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        data = payload
    if isinstance(data, (str, bytes)) or not isinstance(data, Sequence):
        raise MCQAggregateError("model preflight response is malformed")
    ids = []
    for item in data:
        model_id = _field(item, "id")
        if isinstance(model_id, str) and model_id:
            ids.append(model_id)
    return ids


def _preflight_model_id(args: argparse.Namespace, client: Any | None = None) -> dict[str, Any]:
    """Require the requested id to equal a model id advertised by the server."""

    value = client if client is not None else _new_client(args)
    try:
        advertised = _model_ids(value.models.list())
    except MCQAggregateError:
        raise
    except Exception as error:
        # Do not emit provider exception text; it can contain request details.
        raise MCQAggregateError("model id preflight failed") from error
    if args.model_id not in advertised:
        raise MCQAggregateError("requested model id is not an exact server model id")
    return {
        "status": "passed",
        "exact_match": True,
        "requested_model_id": args.model_id,
        "server_model_id": args.model_id,
    }


preflight_model_id = _preflight_model_id


def _infer(
    samples: Sequence[Sample],
    *,
    args: argparse.Namespace,
    client_factory: Callable[[], Any] | None = None,
) -> list[InferenceOutcome]:
    """Infer rows while retaining only non-text telemetry in each outcome."""

    thread_local = local()

    def client() -> Any:
        value = getattr(thread_local, "client", None)
        if value is None:
            value = client_factory() if client_factory is not None else _new_client(args)
            thread_local.client = value
        return value

    def one(sample: Sample) -> InferenceOutcome:
        # Request construction and image encoding are local operations.  They
        # must abort the formal run, rather than being misreported as an API
        # failure.  Only an explicitly recognized OpenAI SDK exception from
        # the provider call is converted into a scored failure.
        request = build_request(sample, model_id=args.model_id, image_uris=None)
        try:
            response = client().chat.completions.create(**request)
        except Exception as error:
            failure_kind = _provider_failure_kind(error)
            if failure_kind is None:
                raise LocalInputError("provider call returned a non-OpenAI error") from None
            raise _ProviderFailure(failure_kind) from None
        choice = _choice(response)
        if choice is None:
            raise LocalInputError("provider returned no choice")
        if _message(response) is None:
            raise LocalInputError("provider returned a malformed choice")
        finish_reason = _field(choice, "finish_reason")
        content = _response_content(response)
        prediction = parse_choice(content)
        return InferenceOutcome(
            dataset=sample.dataset,
            gold=sample.gold,
            prediction=prediction,
            finish_reason=_finish_bucket(finish_reason),
            content_present=content is not None,
            reasoning_present=_response_reasoning_present(response),
            completion_tokens=_completion_tokens(response),
        )

    results: list[InferenceOutcome] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(one, sample): sample for sample in samples}
        for future in as_completed(futures):
            sample = futures[future]
            try:
                results.append(future.result())
            except _ProviderFailure as error:
                # Never serialize the exception or sample.  The kind is only
                # an aggregate HTTP-vs-API bucket.
                results.append(
                    InferenceOutcome(
                        dataset=sample.dataset,
                        gold=sample.gold,
                        prediction=None,
                        provider_failed=True,
                        failure_kind=error.failure_kind,
                    )
                )
            except LocalInputError:
                raise
            except MCQAggregateError:
                raise
            except Exception:
                # A failure outside the explicitly recognized OpenAI
                # hierarchy is a local/runtime abort, not a provider count.
                raise LocalInputError("inference failed locally") from None
    return results


_PROTOCOL_KEYS = frozenset(
    {
        "preset",
        "thinking",
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "presence_penalty",
        "repetition_penalty",
        "max_tokens",
        "seed",
        "workers",
        "datasets",
        "structured_json",
        "prompt_only",
        "response_format",
        "image_order",
        "gold_scope",
        "sample_level_output",
    }
)
_MODEL_PREFLIGHT_KEYS = frozenset(
    {"status", "exact_match", "requested_model_id", "server_model_id"}
)
_AGGREGATE_KEYS = frozenset(
    {
        "correct",
        "total",
        "accuracy",
        "invalid_count",
        "api_failure_count",
        "http_failure_count",
        "non_http_api_failure_count",
        "api_error_count",
        "http_error_count",
        "invalid_format_count",
        "parsed_count",
        "unparsed_count",
        "content_present_count",
        "reasoning_present_count",
        "content_count",
        "reasoning_count",
        "finish_reason_counts",
        "finish_reason_stop_count",
        "finish_reason_length_count",
        "finish_reason_other_count",
        "completion_tokens",
        "completion_tokens_p50",
        "completion_tokens_p90",
        "completion_tokens_max",
        "selected_option_counts",
        "datasets",
    }
)
_DATASET_SCORE_KEYS = frozenset(
    {
        "correct",
        "total",
        "accuracy",
        "invalid_count",
        "api_failure_count",
        "http_failure_count",
        "invalid_format_count",
        "parsed_count",
        "content_present_count",
        "reasoning_present_count",
        "finish_reason_counts",
        "selected_option_counts",
        "completion_tokens",
    }
)
_TOKEN_KEYS = frozenset({"count", "percentile_method", "p50", "p90", "max"})
_HEX64 = set("0123456789abcdef")
PROTOCOL_ALLOWLIST = _PROTOCOL_KEYS
MODEL_PREFLIGHT_ALLOWLIST = _MODEL_PREFLIGHT_KEYS
AGGREGATE_DATASET_ALLOWLIST = _DATASET_SCORE_KEYS


def _strict_mapping(value: Any, allowed: frozenset[str], *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise MCQAggregateError(f"{label} is malformed")
    result: dict[str, Any] = {}
    for key, child in value.items():
        if not isinstance(key, str) or key not in allowed:
            raise MCQAggregateError(f"{label} contains an unsupported field")
        result[key] = child
    return result


def _validate_hash(value: Any, *, label: str, length: int = 64) -> str:
    if not isinstance(value, str) or len(value) != length or any(char not in _HEX64 for char in value.lower()):
        raise MCQAggregateError(f"{label} is malformed")
    return value


def _sanitize_protocol(protocol: Mapping[str, Any]) -> dict[str, Any]:
    result = _strict_mapping(protocol, _PROTOCOL_KEYS, label="protocol")
    for key in (
        "preset", "response_format", "image_order", "gold_scope",
    ):
        if key in result and not isinstance(result[key], str):
            raise MCQAggregateError("protocol is malformed")
    for key in ("thinking", "structured_json", "prompt_only", "sample_level_output"):
        if key in result and not isinstance(result[key], bool):
            raise MCQAggregateError("protocol is malformed")
    datasets = result.get("datasets")
    if datasets is not None:
        if isinstance(datasets, (str, bytes)) or not isinstance(datasets, Sequence):
            raise MCQAggregateError("protocol datasets are malformed")
        if any(dataset not in DATASET_ROWS for dataset in datasets):
            raise MCQAggregateError("protocol datasets are malformed")
        result["datasets"] = list(datasets)
    return result


def _sanitize_model_preflight(value: Mapping[str, Any]) -> dict[str, Any]:
    result = _strict_mapping(value, _MODEL_PREFLIGHT_KEYS, label="model preflight")
    if result.get("status") is not None and not isinstance(result["status"], str):
        raise MCQAggregateError("model preflight is malformed")
    if result.get("exact_match") is not None and not isinstance(result["exact_match"], bool):
        raise MCQAggregateError("model preflight is malformed")
    for key in ("requested_model_id", "server_model_id"):
        if key in result and (not isinstance(result[key], str) or not result[key]):
            raise MCQAggregateError("model preflight is malformed")
    return result


def _sanitize_token_stats(value: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    result = _strict_mapping(value, _TOKEN_KEYS, label=label)
    if "percentile_method" in result and not isinstance(result["percentile_method"], str):
        raise MCQAggregateError(f"{label} is malformed")
    for key in ("count", "p50", "p90", "max"):
        if key in result and result[key] is not None and (
            isinstance(result[key], bool) or not isinstance(result[key], int) or result[key] < 0
        ):
            raise MCQAggregateError(f"{label} is malformed")
    return result


def _sanitize_dataset_scores(value: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping):
        raise MCQAggregateError("aggregate datasets are malformed")
    datasets: dict[str, dict[str, Any]] = {}
    for dataset, score in value.items():
        if dataset not in DATASET_ROWS:
            raise MCQAggregateError("aggregate datasets contain an unsupported field")
        clean = _strict_mapping(score, _DATASET_SCORE_KEYS, label=f"{dataset} score")
        for key, number in clean.items():
            if key == "accuracy":
                if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(float(number)):
                    raise MCQAggregateError(f"{dataset} score is malformed")
            elif key.endswith("_count") or key in {"correct", "total"}:
                if isinstance(number, bool) or not isinstance(number, int) or number < 0:
                    raise MCQAggregateError(f"{dataset} score is malformed")
        if "finish_reason_counts" in clean:
            finish = _strict_mapping(
                clean["finish_reason_counts"], frozenset({"stop", "length", "other"}),
                label=f"{dataset} finish reasons",
            )
            for count in finish.values():
                if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                    raise MCQAggregateError(f"{dataset} finish reasons are malformed")
            clean["finish_reason_counts"] = finish
        if "selected_option_counts" in clean:
            options = _strict_mapping(
                clean["selected_option_counts"], frozenset(CHOICES),
                label=f"{dataset} selected options",
            )
            for count in options.values():
                if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                    raise MCQAggregateError(f"{dataset} selected options are malformed")
            clean["selected_option_counts"] = options
        if "completion_tokens" in clean:
            clean["completion_tokens"] = _sanitize_token_stats(
                clean["completion_tokens"], label=f"{dataset} completion tokens"
            )
        datasets[dataset] = clean
    return datasets


def _sanitize_aggregate(aggregate: Mapping[str, Any]) -> dict[str, Any]:
    result = _strict_mapping(aggregate, _AGGREGATE_KEYS, label="aggregate")
    required = {"correct", "total", "accuracy", "invalid_count", "invalid_format_count", "selected_option_counts", "datasets"}
    if not required.issubset(result):
        raise MCQAggregateError("aggregate is incomplete")
    for key, number in result.items():
        if key == "accuracy":
            if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(float(number)):
                raise MCQAggregateError("aggregate is malformed")
        elif key.endswith("_count") or key in {"correct", "total"}:
            if isinstance(number, bool) or not isinstance(number, int) or number < 0:
                raise MCQAggregateError("aggregate is malformed")
    result["datasets"] = _sanitize_dataset_scores(result.get("datasets", {}))
    if "finish_reason_counts" in result:
        result["finish_reason_counts"] = _strict_mapping(
            result["finish_reason_counts"], frozenset({"stop", "length", "other"}),
            label="aggregate finish reasons",
        )
    if "selected_option_counts" in result:
        result["selected_option_counts"] = _strict_mapping(
            result["selected_option_counts"], frozenset(CHOICES),
            label="aggregate selected options",
        )
    if "completion_tokens" in result:
        result["completion_tokens"] = _sanitize_token_stats(
            result["completion_tokens"], label="aggregate completion tokens"
        )
    return result


def _source_image_bytes(image: str) -> bytes:
    if image.startswith("data:"):
        try:
            header, payload = image.split(",", 1)
            if ";base64" not in header.lower():
                raise ValueError
            return base64.b64decode(payload, validate=True)
        except (ValueError, binascii.Error):
            raise LocalInputError("an embedded image has malformed bytes") from None
    if urlparse(image).scheme in {"http", "https"}:
        raise LocalInputError("remote image URLs are not permitted")
    try:
        source = _regular_file(image, label="BLINK image")
        with source.open("rb", buffering=0) as stream:
            return stream.read()
    except MCQAggregateError:
        raise
    except OSError:
        raise LocalInputError("a BLINK image is unavailable") from None


def _blink_image_manifest_sha256(samples: Sequence[Sample]) -> str | None:
    """Hash ordered source image bytes for BLINK without recording paths."""

    blink_samples = [sample for sample in samples if sample.dataset == "BLINK"]
    if not blink_samples:
        return None
    digest = hashlib.sha256()
    digest.update(b"BLINK_SOURCE_IMAGE_BYTES_MANIFEST_V1\0")
    image_count = 0
    for sample in blink_samples:
        for image in sample.images:
            payload = _source_image_bytes(image)
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
            image_count += 1
    digest.update(image_count.to_bytes(8, "big"))
    return digest.hexdigest()


image_manifest_sha256 = _blink_image_manifest_sha256


def _receipt(
    *,
    aggregate: Mapping[str, Any],
    dataset_md5: Mapping[str, str],
    prompt_hash: str,
    model_id: str,
    protocol: Mapping[str, Any],
    model_preflight: Mapping[str, Any] | None = None,
    code_sha256: str | None = None,
    image_manifest_hash: str | None = None,
) -> dict[str, Any]:
    if not isinstance(dataset_md5, Mapping):
        raise MCQAggregateError("dataset MD5 metadata is malformed")
    clean_md5: dict[str, str] = {}
    for dataset, digest in dataset_md5.items():
        if dataset not in DATASET_ROWS:
            raise MCQAggregateError("dataset MD5 metadata contains an unsupported field")
        clean_md5[dataset] = _validate_hash(digest, label=f"{dataset} MD5", length=32)
    clean_aggregate = _sanitize_aggregate(aggregate)
    clean_protocol = _sanitize_protocol(protocol)
    clean_preflight = _sanitize_model_preflight(
        model_preflight or {"status": "passed", "exact_match": True, "server_model_id": model_id}
    )
    if not isinstance(model_id, str) or not model_id:
        raise MCQAggregateError("model id is malformed")
    _validate_hash(prompt_hash, label="prompt hash")
    if image_manifest_hash is not None:
        _validate_hash(image_manifest_hash, label="image manifest hash")
    data_hash = _sha256_bytes(_canonical(clean_md5))
    model_hash = _sha256_bytes(model_id.encode("utf-8"))
    protocol_hash = _sha256_bytes(_canonical(clean_protocol))
    hashes = {"data": data_hash, "prompt": prompt_hash, "model": model_hash, "protocol": protocol_hash}
    if image_manifest_hash is not None:
        hashes["images"] = image_manifest_hash
    code_identity = _code_identity()
    if code_sha256 is not None:
        _validate_hash(code_sha256, label="code hash")
        code_identity["sha256"] = code_sha256
    telemetry = {
        "api_failure_count": int(clean_aggregate.get("api_failure_count", 0)),
        "http_failure_count": int(clean_aggregate.get("http_failure_count", 0)),
        "non_http_api_failure_count": int(clean_aggregate.get("non_http_api_failure_count", clean_aggregate.get("api_failure_count", 0))),
        "api_error_count": int(clean_aggregate.get("api_error_count", clean_aggregate.get("non_http_api_failure_count", clean_aggregate.get("api_failure_count", 0)))),
        "http_error_count": int(clean_aggregate.get("http_error_count", clean_aggregate.get("http_failure_count", 0))),
        "finish_reason_counts": dict(clean_aggregate.get("finish_reason_counts", {"stop": 0, "length": 0, "other": 0})),
        "content_present_count": int(clean_aggregate.get("content_present_count", 0)),
        "reasoning_present_count": int(clean_aggregate.get("reasoning_present_count", 0)),
        "parsed_count": int(clean_aggregate.get("parsed_count", 0)),
        "unparsed_count": int(clean_aggregate.get("unparsed_count", clean_aggregate.get("total", 0) - clean_aggregate.get("parsed_count", 0))),
        "completion_tokens": dict(clean_aggregate.get("completion_tokens", {"count": 0, "p50": None, "p90": None, "max": None})),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "code_version": CODE_VERSION,
        "code_sha256": code_identity["sha256"],
        "code_sha": code_identity["sha256"],
        "code": code_identity,
        "dependency_hashes": dict(code_identity["dependency_hashes"]),
        "datasets": {name: {"total": DATASET_ROWS[name], "md5": clean_md5[name]} for name in clean_md5},
        "correct": int(clean_aggregate["correct"]),
        "total": int(clean_aggregate["total"]),
        "accuracy": float(clean_aggregate["accuracy"]),
        "invalid_count": int(clean_aggregate["invalid_count"]),
        "api_failure_count": telemetry["api_failure_count"],
        "http_failure_count": telemetry["http_failure_count"],
        "non_http_api_failure_count": telemetry["non_http_api_failure_count"],
        "api_error_count": telemetry["api_error_count"],
        "http_error_count": telemetry["http_error_count"],
        "invalid_format_count": int(clean_aggregate["invalid_format_count"]),
        "parsed_count": telemetry["parsed_count"],
        "unparsed_count": telemetry["unparsed_count"],
        "content_present_count": telemetry["content_present_count"],
        "reasoning_present_count": telemetry["reasoning_present_count"],
        "finish_reason_counts": telemetry["finish_reason_counts"],
        "completion_tokens": telemetry["completion_tokens"],
        "completion_tokens_p50": telemetry["completion_tokens"].get("p50"),
        "completion_tokens_p90": telemetry["completion_tokens"].get("p90"),
        "completion_tokens_max": telemetry["completion_tokens"].get("max"),
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


def _write_create_once(path: Path, receipt: Mapping[str, Any]) -> None:
    """Durably create one receipt, never replacing an existing artifact.

    The file is created directly with ``O_EXCL|O_NOFOLLOW`` under an opened
    parent directory.  Every partial ``os.write`` is retried, the file and
    parent directory are fsynced, and any failure after creation removes the
    incomplete receipt.  All OS errors are converted to a short aggregate
    error so no traceback can expose local paths or other details.
    """

    try:
        encoded = (json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    except Exception:
        raise MCQAggregateError("aggregate receipt encoding failed") from None

    try:
        target = _assert_output_scope(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Re-check after mkdir so an ancestor cannot be swapped for a symlink.
        target = _assert_output_scope(target)
        parent_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        parent_fd = os.open(target.parent, parent_flags)
    except MCQAggregateError:
        raise
    except OSError:
        raise MCQAggregateError("aggregate receipt parent is unavailable") from None

    descriptor: int | None = None
    created = False

    def cleanup() -> None:
        if not created:
            return
        try:
            os.unlink(target.name, dir_fd=parent_fd)
        except OSError:
            pass

    try:
        descriptor = os.open(
            target.name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=parent_fd,
        )
        created = True
        offset = 0
        while offset < len(encoded):
            written = os.write(descriptor, encoded[offset:])
            if written <= 0:
                raise OSError
            offset += written
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.fsync(parent_fd)
    except FileExistsError:
        cleanup()
        raise MCQAggregateError("aggregate receipt already exists") from None
    except OSError:
        cleanup()
        raise MCQAggregateError("aggregate receipt could not be durably written") from None
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        try:
            os.close(parent_fd)
        except OSError:
            pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mmstar-tsv", "--mmstar_tsv", "--mmstar", dest="mmstar_tsv", required=True, type=Path)
    parser.add_argument("--blink-tsv", "--blink_tsv", "--blink", dest="blink_tsv", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-id", "--model_id", "--model", dest="model_id", required=True)
    parser.add_argument("--api-base", "--api_base", required=True)
    parser.add_argument("--api-key", "--api_key", default="EMPTY")
    parser.add_argument("--preset", default=PRESET_NAME, choices=(PRESET_NAME,))
    parser.add_argument("--thinking", "--enable-thinking", "--enable_thinking", nargs="?", const=True, default=True, type=_parse_bool)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", "--top_p", type=float, default=0.95)
    parser.add_argument("--top-k", "--top_k", type=int, default=20)
    parser.add_argument("--min-p", "--min_p", type=float, default=0.0)
    parser.add_argument("--presence-penalty", "--presence_penalty", type=float, default=1.5)
    parser.add_argument("--repetition-penalty", "--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--max-tokens", "--max_tokens", type=int, default=32768)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", "--worker", dest="workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--structured-json", "--structured_json", nargs="?", const=True, default=False, type=_parse_bool)
    parser.add_argument("--datasets", nargs="+", choices=tuple(DATASET_ROWS), default=list(DATASET_ROWS))
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
        "datasets": {name: DATASET_ROWS[name] for name in args.datasets},
        "preset": PRESET_NAME,
        "protocol": _protocol(args),
        "hashes_deferred": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        # Validate scopes without touching input files first.  The exact
        # server model-id preflight must precede every TSV/image read.
        mmstar, blink, output = _validate_args(args, inspect_inputs=False)
        if args.dry_run:
            print(json.dumps(_dry_run_metadata(args), sort_keys=True))
            return 0

        # The model preflight intentionally happens before loading any rows or
        # images.  A non-exact model id cannot produce a formal receipt.
        model_preflight = _preflight_model_id(args)
        _regular_file(mmstar, label="MMStar TSV", suffix=".tsv")
        _regular_file(blink, label="BLINK TSV", suffix=".tsv")
        paths = {"MMStar": mmstar, "BLINK": blink}
        samples: list[Sample] = []
        dataset_md5: dict[str, str] = {}
        for dataset in args.datasets:
            loaded, digest = _read_dataset(paths[dataset], dataset, image_root=args.image_root)
            samples.extend(loaded)
            dataset_md5[dataset] = digest
        image_manifest_hash = _blink_image_manifest_sha256(samples)
        prompt_hash = _hash_texts(build_prompt(sample) for sample in samples)
        protocol = _protocol(args)
        outcomes = _infer(samples, args=args)
        aggregate = aggregate_outcomes(
            outcomes,
            expected_total=sum(DATASET_ROWS[name] for name in args.datasets),
            expected_dataset_totals={name: DATASET_ROWS[name] for name in args.datasets},
        )
        receipt = _receipt(
            aggregate=aggregate,
            dataset_md5=dataset_md5,
            prompt_hash=prompt_hash,
            model_id=args.model_id,
            protocol=protocol,
            model_preflight=model_preflight,
            code_sha256=_source_sha256(),
            image_manifest_hash=image_manifest_hash,
        )
        _write_create_once(output, receipt)
        dataset_summary = " ".join(
            f"{name.lower()}={aggregate['datasets'][name]['correct']}/{aggregate['datasets'][name]['total']}"
            for name in args.datasets
        )
        print(
            f"MCQ_CHOICE_FORMAL_V2 correct={aggregate['correct']} total={aggregate['total']} "
            f"invalid_count={aggregate['invalid_count']} parsed={aggregate['parsed_count']} {dataset_summary}"
        )
        return 0
    except MCQAggregateError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
