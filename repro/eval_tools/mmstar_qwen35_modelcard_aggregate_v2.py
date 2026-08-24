#!/usr/bin/env python3
"""Aggregate-only MMStar protocol matching the local Qwen3.5 model card.

The scorer uses the pinned 1500-row MMStar TSV and the model-card thinking
sampling controls.  The prompt is the dataset question/options prompt followed
by the model-card MCQ JSON answer instruction.  Only ``message.content`` is
parsed.  Reasoning fields are deliberately ignored (their presence is a
non-sensitive aggregate telemetry counter).  Questions, images, labels,
responses and row identifiers never leave process memory.

This is an intentionally separate protocol from the VLMEvalKit matcher.  It
is suitable for a fixed-seed calibration against the locally documented
Qwen3.5-9B MMStar result; it does not make a leaderboard claim.
"""

from __future__ import annotations

import argparse
import ast
import base64
import csv
import hashlib
import importlib.util
import json
import math
import os
import stat
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from threading import local
from typing import Any, Callable, Mapping, NamedTuple, Sequence
from urllib.parse import urlparse


WORKSPACE = Path("/minimax-3d-rw-backup/users/jiazhi/H_Workspace")
OUTPUT_ROOT = WORKSPACE / "Output"
TOOLS_DIR = Path(__file__).resolve().parent
_reader_path = TOOLS_DIR / "mcq_choice_formal_aggregate_v1.py"
_spec = importlib.util.spec_from_file_location("_mmstar_mcq_reader_v1", _reader_path)
if _spec is None or _spec.loader is None:  # pragma: no cover
    raise ImportError("pinned MMStar reader is unavailable")
_reader = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _reader
_spec.loader.exec_module(_reader)

Sample = _reader.Sample
MCQAggregateError = _reader.MCQAggregateError
CHOICES = tuple("ABCD")
MMSTAR_DATASET = "MMStar"
MMSTAR_ROWS = 1500
MMSTAR_MD5 = "e1ecd2140806c1b1bbf54b43372efb9e"
DATASET_ROWS = {MMSTAR_DATASET: MMSTAR_ROWS}
DATASET_MD5 = {MMSTAR_DATASET: MMSTAR_MD5}
EXPECTED_TOTAL = MMSTAR_ROWS
DEFAULT_WORKERS = 32
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}

SCHEMA_VERSION = "mmstar_qwen35_modelcard_aggregate_v2"
CODE_VERSION = SCHEMA_VERSION
PURPOSE = "qwen35_modelcard_mmstar_seed0_calibration"
PRESET_NAME = "qwen3.5_modelcard_mmstar_thinking_json_prompt"
MODEL_CARD_INSTRUCTION = (
    'Please show your choice in the `answer` field with only the choice letter, '
    'e.g., `"answer": "C"`.'
)
OFFICIAL_INSTRUCTION = MODEL_CARD_INSTRUCTION
FROZEN_PRESET: Mapping[str, Any] = {
    "thinking": True,
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 1.5,
    "repetition_penalty": 1.0,
    "max_tokens": 32768,
    "seed": 0,
    "structured_json": False,
}
OFFICIAL_THINKING_PRESET = FROZEN_PRESET
PARSER_CATEGORIES = (
    "json_answer",
    "explicit_answer_or_choice",
    "terminal_letter",
    "missing_content",
    "blank_content",
    "invalid_content",
    "duplicate_json_key",
)

@dataclass(frozen=True)
class ParsedContent:
    prediction: str | None
    category: str


class InferenceOutcome(NamedTuple):
    dataset: str
    gold: str
    prediction: str | None
    provider_failed: bool = False
    failure_kind: str | None = None
    finish_reason: str | None = None
    content_present: bool = False
    reasoning_present: bool = False
    completion_tokens: int | None = None
    parse_category: str = "invalid_content"


class LocalInputError(MCQAggregateError):
    pass


class _ProviderFailure(Exception):
    def __init__(self, kind: str) -> None:
        super().__init__()
        self.kind = kind


_MISSING = object()
_HEX64 = frozenset("0123456789abcdef")


def _field(value: Any, name: str, default: Any = None) -> Any:
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
    item = _choice(response)
    return _field(item, "message") if item is not None else None


def _read_dataset(path: Path | str, *, image_root: Path | None = None) -> tuple[list[Sample], str]:
    try:
        samples, digest = _reader._read_dataset(path, MMSTAR_DATASET, verify_hash=True, image_root=image_root)
    except Exception as error:
        if isinstance(error, MCQAggregateError):
            raise
        raise MCQAggregateError("MMStar TSV is malformed") from None
    if digest != MMSTAR_MD5 or len(samples) != MMSTAR_ROWS:
        raise MCQAggregateError("MMStar TSV identity differs from pinned 1500-row file")
    return samples, digest


def load_mmstar_dataset(path: Path | str, *, image_root: Path | None = None) -> tuple[list[Sample], str]:
    return _read_dataset(path, image_root=image_root)


def load_tsv_records(path: Path | str, dataset: str | None = None, *, image_root: Path | None = None) -> tuple[list[Sample], str]:
    if dataset is not None and dataset != MMSTAR_DATASET:
        raise MCQAggregateError("only MMStar is supported")
    return _read_dataset(path, image_root=image_root)


def build_prompt(row: Sample | Mapping[str, Any]) -> str:
    """Build raw question/options text, then append the model-card instruction."""

    if isinstance(row, Sample):
        question, options, hint = row.question, row.options, row.hint
    else:
        question = row.get("question")
        if not isinstance(question, str):
            raise MCQAggregateError("question must be text")
        options = tuple((label, str(row[label])) for label in CHOICES if row.get(label) not in (None, ""))
        hint = row.get("hint") if isinstance(row.get("hint"), str) and row.get("hint") else None
    lines: list[str] = []
    if hint is not None:
        lines.append(f"Hint: {hint}")
    lines.append(f"Question: {question}")
    if options:
        lines.append("Options:")
        lines.extend(f"{label}. {value}" for label, value in options)
        lines.append("Please select the correct answer from the options above. ")
    lines.append(MODEL_CARD_INSTRUCTION)
    return "\n".join(lines)


def _image_uri(path_or_uri: str) -> str:
    if path_or_uri.startswith("data:"):
        return path_or_uri
    if urlparse(path_or_uri).scheme in {"http", "https"}:
        raise MCQAggregateError("remote image URLs are not permitted")
    try:
        from PIL import Image
        with Image.open(path_or_uri) as source:
            image = source.convert("RGB")
        output = BytesIO()
        image.save(output, format="PNG")
    except Exception:
        raise MCQAggregateError("an input image could not be encoded") from None
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii")


def build_messages(row: Sample, image_uris: Sequence[str] | None = None) -> list[dict[str, Any]]:
    images = list(row.images)
    uris = list(image_uris) if image_uris is not None else [_image_uri(image) for image in images]
    if len(uris) != len(images):
        raise MCQAggregateError("image URI count differs from input")
    content: list[dict[str, Any]] = [{"type": "image_url", "image_url": {"url": uri}} for uri in uris]
    content.append({"type": "text", "text": build_prompt(row)})
    return [{"role": "user", "content": content}]


def build_request(
    row: Sample, *, model_id: str, image_uris: Sequence[str] | None = None,
    chat_template: str | None = None, **controls: Any,
) -> dict[str, Any]:
    frozen = dict(FROZEN_PRESET)
    given = {**frozen, **controls}
    for name, expected in frozen.items():
        if given.get(name) != expected:
            raise MCQAggregateError(f"frozen model-card control differs: {name}")
    request = {
        "model": model_id,
        "messages": build_messages(row, image_uris),
        "temperature": 1.0,
        "top_p": 0.95,
        "presence_penalty": 1.5,
        "max_tokens": 32768,
        "seed": 0,
        "extra_body": {
            "top_k": 20,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
            "chat_template_kwargs": {"enable_thinking": True},
        },
    }
    # The OpenAI Python SDK deliberately rejects unknown top-level kwargs.
    # vLLM's OpenAI-compatible server merges ``extra_body`` into the HTTP
    # ChatCompletionRequest, so carry this vLLM request field there. Keep this
    # conditional so the default request remains equivalent to the frozen
    # model-card protocol.
    if chat_template is not None:
        if not isinstance(chat_template, str):
            raise MCQAggregateError("chat template must be text")
        request["extra_body"]["chat_template"] = chat_template
    return request


def _strict_json_answer(text: str) -> tuple[str | None, str | None]:
    """Return (letter, category) for a complete unique JSON object only."""

    candidate = text.strip()
    if candidate.startswith("```") and candidate.endswith("```"):
        lines = candidate.splitlines()
        if len(lines) >= 3 and lines[0].strip().lower() in {"```", "```json"} and lines[-1].strip() == "```":
            candidate = "\n".join(lines[1:-1]).strip()
    if not (candidate.startswith("{") and candidate.endswith("}")):
        return None, None
    duplicate = False

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        nonlocal duplicate
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                duplicate = True
            result[key] = value
        return result

    try:
        value = json.loads(candidate, object_pairs_hook=pairs)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None, "invalid_content"
    if duplicate:
        return None, "duplicate_json_key"
    if isinstance(value, dict) and set(value) == {"answer"}:
        answer = value.get("answer")
        if isinstance(answer, str) and answer.upper() in CHOICES:
            return answer.upper(), "json_answer"
    return None, "invalid_content"


def _parse_content(content: Any) -> ParsedContent:
    """Use the pinned parser that produced the 79.33 model-card calibration.

    ``mcq_choice_formal_aggregate_v1.parse_choice`` is the authority.  The
    additional checks below only classify its accepted result for aggregate
    telemetry; they never broaden or narrow the selected prediction.
    """

    if content is _MISSING or content is None:
        return ParsedContent(None, "missing_content")
    if not isinstance(content, str):
        return ParsedContent(None, "invalid_content")
    text = content.strip()
    if not text:
        return ParsedContent(None, "blank_content")
    prediction = _reader.parse_choice(content)
    if prediction not in CHOICES:
        _, json_category = _strict_json_answer(text)
        category = "duplicate_json_key" if json_category == "duplicate_json_key" else "invalid_content"
        return ParsedContent(None, category)

    json_choice, _ = _strict_json_answer(text)
    if json_choice == prediction:
        return ParsedContent(prediction, "json_answer")

    candidate = text
    fence = _reader._FENCE_RE.fullmatch(candidate)
    if fence:
        candidate = fence.group(1).strip()
    marked = _reader._ANSWER_MARKER_RE.findall(candidate)
    if marked and len(set(marked)) == 1 and marked[0] == prediction:
        return ParsedContent(prediction, "explicit_answer_or_choice")
    tail = _reader._TERMINAL_CHOICE_RE.search(candidate)
    if tail and tail.group(1) == prediction:
        return ParsedContent(prediction, "terminal_letter")
    # Defensive category for a future pinned-reader extension.  The authority
    # remains its selected A-D prediction rather than this telemetry label.
    return ParsedContent(prediction, "explicit_answer_or_choice")


def parse_content(content: Any) -> ParsedContent:
    """Parse model content, treating parser exceptions as invalid answers.

    Unexpected answer content is a prediction-format error, not a local
    runner error.  Keep parser failures in the private invalid bucket so one
    malformed model answer cannot abort the formal aggregate.
    """

    try:
        return _parse_content(content)
    except Exception:
        return ParsedContent(None, "invalid_content")


def parse_choice(content: Any) -> str | None:
    """Compatibility scalar API: return only the selected letter."""

    return parse_content(content).prediction


parse_answer = parse_choice
extract_answer = parse_choice


def parse_answer_channels(response: Any) -> dict[str, Any]:
    """Expose content-only parsing telemetry without returning any text."""

    message = _message(response)
    content = _field(message, "content", _MISSING) if message is not None else _MISSING
    parsed = parse_content(content)
    reasoning_present = bool(message is not None and (
        _field(message, "reasoning", _MISSING) is not _MISSING
        or _field(message, "reasoning_content", _MISSING) is not _MISSING
    ))
    return {
        "prediction": parsed.prediction,
        "parsed": parsed.prediction in CHOICES,
        "category": parsed.category,
        "content_present": content is not _MISSING and content is not None,
        "reasoning_present": reasoning_present,
        "reasoning_ignored": reasoning_present,
    }


def _finish_bucket(value: Any) -> str:
    return value if value in {"stop", "length"} else "other"


def _completion_tokens(response: Any) -> int | None:
    usage = _field(response, "usage")
    value = _field(usage, "completion_tokens") if usage is not None else None
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _nearest(values: Sequence[int], fraction: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))]


def _token_stats(values: Sequence[int]) -> dict[str, Any]:
    return {"count": len(values), "percentile_method": "nearest_rank", "p50": _nearest(values, .50), "p90": _nearest(values, .90), "max": max(values) if values else None}


def _provider_failure_kind(error: BaseException) -> str | None:
    names = {"APIError", "APIConnectionError", "APITimeoutError", "APIStatusError", "BadRequestError", "AuthenticationError", "PermissionDeniedError", "NotFoundError", "ConflictError", "UnprocessableEntityError", "RateLimitError", "InternalServerError"}
    status = _field(error, "status_code")
    if status is None:
        status = _field(_field(error, "response"), "status_code")
    for cls in type(error).__mro__:
        if (cls.__module__ == "openai" or cls.__module__.startswith("openai.")) and cls.__name__ in names:
            return "http" if isinstance(status, int) and 100 <= status <= 599 else "api"
    return None


def _api_base(value: str) -> str:
    from urllib.parse import urlparse
    parsed = urlparse(value)
    if parsed.scheme != "http" or parsed.hostname not in LOCAL_HOSTS or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise MCQAggregateError("API base must be a local HTTP endpoint")
    return value.rstrip("/")


def _new_client(args: argparse.Namespace) -> Any:
    try:
        from openai import OpenAI
        return OpenAI(api_key=args.api_key, base_url=_api_base(args.api_base), timeout=3600, max_retries=0)
    except MCQAggregateError:
        raise
    except Exception:
        raise MCQAggregateError("openai package is required") from None


def _preflight_model_id(args: argparse.Namespace, client: Any | None = None) -> dict[str, Any]:
    value = client if client is not None else _new_client(args)
    try:
        advertised = _field(value.models.list(), "data")
        ids = [_field(item, "id") for item in advertised] if isinstance(advertised, Sequence) else []
    except Exception:
        raise MCQAggregateError("model id preflight failed") from None
    if args.model_id not in ids:
        raise MCQAggregateError("requested model id is not an exact server model id")
    return {"status": "passed", "exact_match": True, "requested_model_id": args.model_id, "server_model_id": args.model_id}


def _infer(samples: Sequence[Sample], *, args: argparse.Namespace, client_factory: Callable[[], Any] | None = None) -> list[InferenceOutcome]:
    thread_local = local()
    # The caller loads and hashes the file once before inference. The fallback
    # keeps this helper safe for direct callers that only provide the CLI path
    # (and avoids reading the file once per worker).
    chat_template = getattr(args, "_chat_template_text", None)
    if chat_template is None and getattr(args, "chat_template_file", None) is not None:
        chat_template, _ = read_chat_template_file(args.chat_template_file)

    def client() -> Any:
        current = getattr(thread_local, "client", None)
        if current is None:
            current = client_factory() if client_factory is not None else _new_client(args)
            thread_local.client = current
        return current
    def one(sample: Sample) -> InferenceOutcome:
        request = build_request(sample, model_id=args.model_id, chat_template=chat_template)
        try:
            response = client().chat.completions.create(**request)
        except Exception as error:
            kind = _provider_failure_kind(error)
            if kind is None:
                raise LocalInputError("provider call returned a non-OpenAI error") from None
            raise _ProviderFailure(kind) from None
        message = _message(response)
        content = _field(message, "content", _MISSING) if message is not None else _MISSING
        parsed = parse_content(content)
        reasoning_present = message is not None and (_field(message, "reasoning", _MISSING) is not _MISSING or _field(message, "reasoning_content", _MISSING) is not _MISSING)
        item = _choice(response)
        return InferenceOutcome(sample.dataset, sample.gold, parsed.prediction, False, None, _finish_bucket(_field(item, "finish_reason")), content is not _MISSING and content is not None, bool(reasoning_present), _completion_tokens(response), parsed.category)
    results: list[InferenceOutcome] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(one, sample): sample for sample in samples}
        for future in as_completed(futures):
            sample = futures[future]
            try:
                results.append(future.result())
            except _ProviderFailure as error:
                results.append(InferenceOutcome(sample.dataset, sample.gold, None, True, error.kind, parse_category="api_failure"))
            except (LocalInputError, MCQAggregateError):
                raise
            except Exception:
                raise LocalInputError("inference failed locally") from None
    return results


def _coerce(value: InferenceOutcome | Mapping[str, Any] | Sequence[Any]) -> InferenceOutcome:
    if isinstance(value, InferenceOutcome):
        return value
    if isinstance(value, Mapping):
        return InferenceOutcome(value.get("dataset"), value.get("gold"), value.get("prediction"), bool(value.get("provider_failed", False)), value.get("failure_kind"), value.get("finish_reason"), bool(value.get("content_present", False)), bool(value.get("reasoning_present", False)), value.get("completion_tokens"), value.get("parse_category", "invalid_content"))
    if isinstance(value, Sequence) and len(value) == 10:
        return InferenceOutcome(*value)  # type: ignore[arg-type]
    raise MCQAggregateError("in-memory outcome is malformed")


def aggregate_outcomes(outcomes: Sequence[InferenceOutcome | Mapping[str, Any] | Sequence[Any]], *, expected_total: int | None = None) -> dict[str, Any]:
    selected: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    total = correct = parsed = invalid = api_failure = http_failure = content_present = reasoning_present = 0
    finish: Counter[str] = Counter({"stop": 0, "length": 0, "other": 0})
    tokens: list[int] = []
    for raw in outcomes:
        item = _coerce(raw)
        if item.dataset != MMSTAR_DATASET or item.gold not in CHOICES:
            raise MCQAggregateError("in-memory dataset or gold is malformed")
        if item.provider_failed and item.prediction is not None:
            raise MCQAggregateError("provider failure has a prediction")
        if item.failure_kind not in {None, "api", "http"}:
            raise MCQAggregateError("in-memory failure kind is malformed")
        if not isinstance(item.content_present, bool) or not isinstance(item.reasoning_present, bool):
            raise MCQAggregateError("in-memory channel telemetry is malformed")
        if item.parse_category not in set(PARSER_CATEGORIES) | {"api_failure"}:
            raise MCQAggregateError("in-memory parse category is malformed")
        # A malformed prediction can only represent an unparseable model
        # answer.  Normalize it to the same private invalid bucket as
        # missing/blank content instead of aborting publication.
        if not item.provider_failed and item.prediction is not None and item.prediction not in CHOICES:
            item = item._replace(prediction=None, parse_category="invalid_content")
        total += 1
        content_present += int(item.content_present)
        reasoning_present += int(item.reasoning_present)
        categories[item.parse_category] += 1
        if item.completion_tokens is not None:
            if isinstance(item.completion_tokens, bool) or not isinstance(item.completion_tokens, int) or item.completion_tokens < 0:
                raise MCQAggregateError("in-memory completion token count is malformed")
            tokens.append(item.completion_tokens)
        if item.provider_failed:
            api_failure += 1
            http_failure += int(item.failure_kind == "http")
            continue
        finish[_finish_bucket(item.finish_reason)] += 1
        if item.prediction not in CHOICES:
            invalid += 1
            continue
        parsed += 1
        selected[item.prediction] += 1
        correct += int(item.prediction == item.gold)
    if expected_total is not None and total != expected_total:
        raise MCQAggregateError("aggregate row count differs from formal total")
    stats = _token_stats(tokens)
    invalid_categories = {key: categories[key] for key in PARSER_CATEGORIES if key not in {"json_answer", "explicit_answer_or_choice", "terminal_letter"}}
    score = {
        "correct": correct, "total": total, "accuracy": correct / total if total else 0.0,
        "invalid_count": invalid + api_failure, "invalid_predictions": invalid,
        "api_failure_count": api_failure, "http_failure_count": http_failure,
        "invalid_format_count": invalid, "parsed_count": parsed, "content_present_count": content_present,
        "reasoning_present_count": reasoning_present, "invalid_format_category_counts": invalid_categories,
        "parse_category_counts": {key: categories[key] for key in PARSER_CATEGORIES},
        "finish_reason_counts": dict(finish), "completion_tokens": stats,
        "selected_option_counts": {choice: selected[choice] for choice in CHOICES},
    }
    return {**score, "datasets": {MMSTAR_DATASET: score}, "non_http_api_failure_count": api_failure - http_failure,
            "unparsed_count": total - parsed, "completion_tokens_p50": stats["p50"], "completion_tokens_p90": stats["p90"], "completion_tokens_max": stats["max"]}


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_sha256() -> str:
    return hashlib.sha256(_canonical({"runner": _file_sha256(Path(__file__)), "reader": _file_sha256(_reader_path)})).hexdigest()


def _hash_texts(texts: Sequence[str] | Any) -> str:
    digest = hashlib.sha256()
    for text in texts:
        encoded = text.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _protocol(
    args: argparse.Namespace, *, chat_template_provenance: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    protocol = {"preset": PRESET_NAME, **dict(FROZEN_PRESET), "workers": args.workers, "dataset": MMSTAR_DATASET,
                "prompt_template": "raw_question_options_then_model_card_instruction", "model_card_instruction_sha256": hashlib.sha256(MODEL_CARD_INSTRUCTION.encode()).hexdigest(),
                "answer_source": "content_only", "answer_precedence": ["pinned_mcq_choice_formal_v1_json_marker_terminal"],
                "invalid_format_categories": list(PARSER_CATEGORIES[3:]), "gold_scope": "scorer_memory_only", "sample_level_output": False}
    # Do not add a protocol field when no override was supplied: this keeps
    # the historical protocol hash exactly unchanged for default runs.
    if chat_template_provenance is not None:
        if set(chat_template_provenance) != {"path", "sha256"}:
            raise MCQAggregateError("chat template provenance is malformed")
        path = chat_template_provenance["path"]
        digest = chat_template_provenance["sha256"]
        if not isinstance(path, str) or not os.path.isabs(path) or not isinstance(digest, str) or len(digest) != 64:
            raise MCQAggregateError("chat template provenance is malformed")
        protocol["chat_template_file"] = {"path": path, "sha256": digest}
    return protocol


def _receipt(*, aggregate: Mapping[str, Any], prompt_hash: str, model_id: str, protocol: Mapping[str, Any], preflight: Mapping[str, Any]) -> dict[str, Any]:
    if model_id != preflight.get("server_model_id") or preflight.get("status") != "passed" or preflight.get("exact_match") is not True:
        raise MCQAggregateError("model preflight is not an exact pass")
    if not isinstance(prompt_hash, str) or len(prompt_hash) != 64:
        raise MCQAggregateError("prompt hash is malformed")
    data_hash = hashlib.sha256(_canonical({MMSTAR_DATASET: MMSTAR_MD5})).hexdigest()
    protocol_hash = hashlib.sha256(_canonical(protocol)).hexdigest()
    code_hash = _source_sha256()
    clean = dict(aggregate)
    receipt = {"schema_version": SCHEMA_VERSION, "code_version": CODE_VERSION, "code_sha256": code_hash,
            "datasets": {MMSTAR_DATASET: {"total": MMSTAR_ROWS, "md5": MMSTAR_MD5}},
            "correct": clean["correct"], "total": clean["total"], "accuracy": float(clean["accuracy"]),
            "invalid_count": clean["invalid_count"], "invalid_predictions": clean["invalid_predictions"],
            "invalid_format_count": clean["invalid_format_count"],
            "api_failure_count": clean["api_failure_count"], "http_failure_count": clean["http_failure_count"],
            "parsed_count": clean["parsed_count"], "parse_category_counts": clean["parse_category_counts"],
            "invalid_format_category_counts": clean["invalid_format_category_counts"],
            "content_present_count": clean["content_present_count"], "reasoning_present_count": clean["reasoning_present_count"],
            "finish_reason_counts": clean["finish_reason_counts"], "completion_tokens": clean["completion_tokens"],
            "selected_option_counts": clean["selected_option_counts"], "scores": {MMSTAR_DATASET: clean["datasets"][MMSTAR_DATASET]},
            "model_id": model_id, "model_id_preflight": dict(preflight), "purpose": PURPOSE, "leaderboard_claim": False,
            "preset": PRESET_NAME, "protocol": dict(protocol), "hashes": {"data": data_hash, "prompt": prompt_hash, "protocol": protocol_hash, "code": code_hash},
            "data_hash": data_hash, "prompt_hash": prompt_hash, "protocol_hash": protocol_hash, "model_hash": hashlib.sha256(model_id.encode()).hexdigest()}
    if "chat_template_file" in protocol:
        # Keep a convenient aggregate-level provenance copy in addition to
        # binding it into the protocol hash.
        receipt["chat_template_file"] = dict(protocol["chat_template_file"])
    return receipt


def _absolute(path: Path | str) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def read_chat_template_file(path: Path | str) -> tuple[str, dict[str, str]]:
    """Read a UTF-8 vLLM chat template and return text plus safe provenance."""

    absolute = _absolute(path)
    try:
        raw = absolute.read_bytes()
    except OSError:
        raise MCQAggregateError("chat template file is unavailable") from None
    try:
        template = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise MCQAggregateError("chat template file is not UTF-8") from None
    return template, {"path": str(absolute), "sha256": hashlib.sha256(raw).hexdigest()}


def _assert_output_scope(path: Path, *, allow_missing: bool = True) -> Path:
    target, root = _absolute(path), _absolute(OUTPUT_ROOT)
    try:
        target.relative_to(root)
    except ValueError:
        raise MCQAggregateError("output must be below H_Workspace/Output") from None
    try:
        root_info = root.lstat()
    except FileNotFoundError:
        root_info = None
    except OSError:
        raise MCQAggregateError("output root is unavailable") from None
    if root_info is not None and (stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode)):
        raise MCQAggregateError("output root must be a real directory")
    cursor = root
    relative = target.relative_to(root)
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
    if target.exists() or target.is_symlink():
        raise MCQAggregateError("aggregate receipt already exists")
    return target


def _write_create_once(path: Path, value: Mapping[str, Any]) -> None:
    target = _assert_output_scope(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    parent_fd = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    fd: int | None = None
    try:
        fd = os.open(target.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=parent_fd)
        os.write(fd, (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode())
        os.fsync(fd)
        os.close(fd); fd = None; os.fsync(parent_fd)
    except FileExistsError:
        raise MCQAggregateError("aggregate receipt already exists") from None
    except OSError:
        raise MCQAggregateError("aggregate receipt could not be written") from None
    finally:
        if fd is not None:
            os.close(fd)
        os.close(parent_fd)


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool): return value
    if value.lower() in {"true", "1", "yes"}: return True
    if value.lower() in {"false", "0", "no"}: return False
    raise argparse.ArgumentTypeError("expected a boolean")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mmstar-tsv", dest="mmstar_tsv", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-id", dest="model_id", required=True)
    parser.add_argument("--api-base", dest="api_base", required=True)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--thinking", type=_parse_bool, default=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", dest="top_p", type=float, default=.95)
    parser.add_argument("--top-k", dest="top_k", type=int, default=20)
    parser.add_argument("--min-p", dest="min_p", type=float, default=0.0)
    parser.add_argument("--presence-penalty", dest="presence_penalty", type=float, default=1.5)
    parser.add_argument("--repetition-penalty", dest="repetition_penalty", type=float, default=1.0)
    parser.add_argument("--max-tokens", dest="max_tokens", type=int, default=32768)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--structured-json", dest="structured_json", type=_parse_bool, default=False)
    parser.add_argument(
        "--chat-template-file", dest="chat_template_file", type=Path, default=None,
        help="optional UTF-8 vLLM chat template sent as top-level chat_template",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace, *, inspect_inputs: bool) -> None:
    for key, expected in FROZEN_PRESET.items():
        if getattr(args, key) != expected:
            raise MCQAggregateError(f"frozen model-card control differs: {key}")
    if not 1 <= args.workers <= 256: raise MCQAggregateError("workers must be in [1,256]")
    _api_base(args.api_base)
    _assert_output_scope(args.output)
    if inspect_inputs:
        _read_dataset(args.mmstar_tsv)


def _dry_run(args: argparse.Namespace) -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "code_version": CODE_VERSION, "dry_run": True, "reads_data": False, "connects_api": False, "writes_output": False, "dataset": MMSTAR_DATASET, "rows": MMSTAR_ROWS, "preset": PRESET_NAME, "purpose": PURPOSE, "protocol": _protocol(args), "hashes_deferred": True}


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        _validate_args(args, inspect_inputs=False)
        if args.dry_run:
            print(json.dumps(_dry_run(args), sort_keys=True))
            return 0
        samples, digest = _read_dataset(args.mmstar_tsv)
        preflight = _preflight_model_id(args)
        prompt_hash = _hash_texts(build_prompt(sample) for sample in samples)
        chat_template_provenance = None
        if args.chat_template_file is not None:
            chat_template, chat_template_provenance = read_chat_template_file(args.chat_template_file)
            # Keep the text transient and avoid a second read inside workers.
            args._chat_template_text = chat_template
        outcomes = _infer(samples, args=args)
        # API/HTTP failures are not benchmark answers.  Keep them available to
        # the in-memory reducer for diagnostics, but never publish a partial
        # or failure-contaminated authority receipt.
        if any(item.provider_failed for item in outcomes):
            raise MCQAggregateError("provider API/HTTP failure; receipt not published")
        aggregate = aggregate_outcomes(outcomes, expected_total=MMSTAR_ROWS)
        receipt = _receipt(
            aggregate=aggregate, prompt_hash=prompt_hash, model_id=args.model_id,
            protocol=_protocol(args, chat_template_provenance=chat_template_provenance),
            preflight=preflight,
        )
        _write_create_once(args.output, receipt)
        print(f"MMSTAR_QWEN35_MODELCARD correct={aggregate['correct']} total={aggregate['total']} invalid_predictions={aggregate['invalid_predictions']} invalid_count={aggregate['invalid_count']} parsed={aggregate['parsed_count']}")
        return 0
    except MCQAggregateError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
