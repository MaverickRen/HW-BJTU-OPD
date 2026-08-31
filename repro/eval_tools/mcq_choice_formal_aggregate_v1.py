#!/usr/bin/env python3
"""Formal aggregate-only MMStar/BLINK choice evaluation.

The input files are the pinned VLMEvalKit TSV files.  A row is converted to
the same question/options prompt used by ``ImageMCQDataset`` and the exact
Qwen3.5 answer-field instruction is appended.  The model is queried through
a local OpenAI-compatible chat endpoint.  Gold answers and model answers are
kept in memory only; the only persistent artifact is a create-once aggregate
receipt below ``H_Workspace/Output``.

``--dry-run`` is intentionally a no-read/no-connect/no-write mode.  It is
useful on machines where the dataset and endpoint are not mounted yet.
"""

from __future__ import annotations

import argparse
import ast
import base64
import csv
import hashlib
import json
import os
import re
import stat
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlparse


WORKSPACE = Path(os.environ.get("OPD_QWEN35_WORKSPACE", Path(__file__).resolve().parents[3]))
OUTPUT_ROOT = WORKSPACE / "Output"
CHOICES = ("A", "B", "C", "D")
DATASET_ROWS = {"MMStar": 1500, "BLINK": 1901}
EXPECTED_TOTAL = sum(DATASET_ROWS.values())
EXPECTED_MMSTAR_ROWS = DATASET_ROWS["MMStar"]
EXPECTED_BLINK_ROWS = DATASET_ROWS["BLINK"]
DATASET_MD5 = {
    "MMStar": "e1ecd2140806c1b1bbf54b43372efb9e",
    "BLINK": "d5e8af148b10ac69f535ff7b23f3f989",
}
MMSTAR_MD5 = DATASET_MD5["MMStar"]
BLINK_MD5 = DATASET_MD5["BLINK"]
OFFICIAL_INSTRUCTION = (
    'Please show your choice in the `answer` field with only the choice letter, '
    'e.g., `"answer": "C"`.'
)
SCHEMA_VERSION = "mcq_choice_formal_aggregate_v1"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 1.0
DEFAULT_TOP_K = 20
DEFAULT_MIN_P = 0.0
DEFAULT_PRESENCE_PENALTY = 0.0
DEFAULT_REPETITION_PENALTY = 1.0
DEFAULT_MAX_TOKENS = 1024
DEFAULT_WORKERS = 32
DEFAULT_SEED = 42
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}
_FENCE_RE = re.compile(r"\A```(?:json)?\s*\n?(.*?)\n?```\Z", re.IGNORECASE | re.DOTALL)
_ANSWER_MARKER_RE = re.compile(
    r"[\"'`]*(?i:(?:final\s+)?answer|choice)[\"'`]*\s*"
    r"(?i:is)?\s*[:=]?\s*"
    r"[\"'`*({[]*([A-D])[\"'`*)}\].,;:]*"
)
_TERMINAL_CHOICE_RE = re.compile(
    r"(?is)(?:^|\n)\s*(?:(?i:final\s+answer|answer|choice|option)\s*"
    r"(?:(?i:is)\s*)?[:=]?\s*)?[\"'`*({[]*([A-D])[\"'`*)}\].,;:]*\s*$"
)
STRUCTURED_JSON_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string", "enum": list(CHOICES)}},
    "required": ["answer"],
    "additionalProperties": False,
}


class MCQAggregateError(RuntimeError):
    """Malformed input, provider result, or unsafe output path."""


@dataclass(frozen=True)
class Sample:
    """Private in-memory scorer sample; never serialize this object."""

    dataset: str
    images: tuple[str, ...]
    question: str
    options: tuple[tuple[str, str], ...]
    gold: str
    hint: str | None = None


def _absolute(path: Path | str) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _regular_file(path: Path | str, *, label: str, suffix: str | None = None) -> Path:
    candidate = _absolute(path)
    try:
        info = candidate.lstat()
    except OSError as error:
        raise MCQAggregateError(f"{label} is unavailable") from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise MCQAggregateError(f"{label} must be a single-link regular file")
    if info.st_size <= 0:
        raise MCQAggregateError(f"{label} is empty")
    if suffix is not None and candidate.suffix.lower() != suffix.lower():
        raise MCQAggregateError(f"{label} must use {suffix}")
    return candidate


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb", buffering=0) as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hash_texts(texts: Iterable[str]) -> str:
    """Hash ordered prompts without retaining them in the receipt."""

    digest = hashlib.sha256()
    for text in texts:
        encoded = text.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _parse_list_field(value: Any) -> list[str]:
    """Mirror VLMEvalKit's list-like TSV fields without using eval."""

    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    if not isinstance(value, str):
        return []
    value = value.strip()
    if not value:
        return []
    if value.startswith("[") and value.endswith("]"):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            raise MCQAggregateError("an image list field is malformed") from None
        if not isinstance(parsed, (list, tuple)):
            raise MCQAggregateError("an image list field is malformed")
        return [str(item) for item in parsed]
    return [value]


def _row_images(row: Mapping[str, Any]) -> list[str]:
    # VLMEvalKit local TSVs normally contain image_path.  Supporting image as
    # a fallback keeps this reader compatible with localized TSV variants.
    field = row.get("image_path")
    embedded = field is None or (isinstance(field, str) and not field.strip())
    if embedded:
        field = row.get("image")
    images = _parse_list_field(field)
    if not images or any(not image.strip() for image in images):
        raise MCQAggregateError("each MCQ row must contain at least one image")
    if embedded:
        # MMStar's pinned VLMEvalKit TSV stores raw JPEG base64 rather than an
        # image_path.  Normalize it in memory without writing decoded samples.
        # Keep failures generic so raw image bytes can never enter logs.
        prefixes = {
            "/9j/": "image/jpeg",
            "iVBOR": "image/png",
            "R0lGOD": "image/gif",
            "UklGR": "image/webp",
        }
        normalized: list[str] = []
        for image in images:
            if image.startswith("data:"):
                normalized.append(image)
                continue
            mime = next((kind for prefix, kind in prefixes.items() if image.startswith(prefix)), None)
            if mime is None:
                raise MCQAggregateError("an embedded image field has an unsupported encoding")
            normalized.append(f"data:{mime};base64,{image}")
        images = normalized
    # Do not sort or deduplicate: BLINK's multi-image order is authoritative.
    return images


def _resolve_images(
    images: Sequence[str],
    *,
    dataset: str,
    tsv_path: Path,
    image_root: Path | None,
) -> tuple[str, ...]:
    resolved: list[str] = []
    default_root = tsv_path.parent / "images" / dataset
    for image in images:
        if image.startswith("data:") or urlparse(image).scheme in {"http", "https"}:
            resolved.append(image)
            continue
        path = Path(image).expanduser()
        if not path.is_absolute():
            path = (image_root / path) if image_root is not None else (default_root / path)
        resolved.append(str(path))
    return tuple(resolved)


def _option_values(row: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    options: list[tuple[str, str]] = []
    for label in CHOICES:
        value = row.get(label)
        if value is None or (isinstance(value, str) and not value.strip()):
            continue
        options.append((label, str(value)))
    if not options:
        raise MCQAggregateError("each MCQ row must contain options")
    return tuple(options)


def _read_dataset(
    path: Path | str,
    dataset: str,
    *,
    verify_hash: bool = True,
    image_root: Path | None = None,
) -> tuple[list[Sample], str]:
    """Read one VLMEvalKit TSV and return private samples plus its MD5."""

    if dataset not in DATASET_ROWS:
        raise MCQAggregateError("unsupported dataset")
    source = _regular_file(path, label=f"{dataset} TSV", suffix=".tsv")
    digest = _md5(source)
    if verify_hash and digest != DATASET_MD5[dataset]:
        raise MCQAggregateError(f"{dataset} TSV MD5 differs from the pinned file")

    samples: list[Sample] = []
    try:
        # The pinned TSVs also contain base64 image columns.  Python's default
        # 128-KiB CSV field limit is therefore too small even though this
        # evaluator consumes the authoritative ``image_path`` column.
        field_limit = sys.maxsize
        while True:
            try:
                csv.field_size_limit(field_limit)
                break
            except OverflowError:  # pragma: no cover - 32-bit C long guard
                field_limit //= 10
        stream = source.open("r", encoding="utf-8", newline="")
        with stream:
            reader = csv.DictReader(stream, delimiter="\t")
            if not reader.fieldnames or "question" not in reader.fieldnames:
                raise MCQAggregateError(f"{dataset} TSV has no question column")
            for row in reader:
                if None in row:
                    raise MCQAggregateError(f"{dataset} TSV has malformed columns")
                question = row.get("question")
                gold = row.get("answer")
                if not isinstance(question, str) or not isinstance(gold, str):
                    raise MCQAggregateError(f"{dataset} TSV has malformed scoring fields")
                gold = gold.strip().upper()
                if gold not in CHOICES:
                    raise MCQAggregateError(f"{dataset} TSV has a non-choice gold answer")
                hint_value = row.get("hint")
                hint = hint_value if isinstance(hint_value, str) and hint_value else None
                images = _resolve_images(
                    _row_images(row),
                    dataset=dataset,
                    tsv_path=source,
                    image_root=image_root,
                )
                for image in images:
                    if image.startswith("data:"):
                        continue
                    if urlparse(image).scheme in {"http", "https"}:
                        raise MCQAggregateError("remote image URLs are not permitted")
                    _regular_file(image, label=f"{dataset} image")
                samples.append(
                    Sample(
                        dataset=dataset,
                        images=images,
                        question=question,
                        options=_option_values(row),
                        gold=gold,
                        hint=hint,
                    )
                )
    except (UnicodeError, csv.Error) as error:
        raise MCQAggregateError(f"{dataset} TSV cannot be decoded") from error
    if len(samples) != DATASET_ROWS[dataset]:
        raise MCQAggregateError(f"{dataset} TSV row count differs from the pinned file")
    return samples, digest


def build_prompt(row: Sample | Mapping[str, Any]) -> str:
    """Build the dataset prompt and append the exact official instruction."""

    if isinstance(row, Sample):
        question, options, hint = row.question, row.options, row.hint
    else:
        question = row.get("question")
        if not isinstance(question, str):
            raise MCQAggregateError("question must be text")
        options = _option_values(row)
        hint_value = row.get("hint")
        hint = hint_value if isinstance(hint_value, str) and hint_value else None
    lines: list[str] = []
    if hint is not None:
        lines.append(f"Hint: {hint}")
    lines.append(f"Question: {question}")
    if options:
        lines.append("Options:")
        lines.extend(f"{label}. {value}" for label, value in options)
        lines.append("Please select the correct answer from the options above. ")
    lines.append(OFFICIAL_INSTRUCTION)
    return "\n".join(lines)


load_tsv_records = _read_dataset


def _image_data_uri(path_or_uri: str) -> str:
    if path_or_uri.startswith("data:"):
        return path_or_uri
    if urlparse(path_or_uri).scheme in {"http", "https"}:
        raise MCQAggregateError("remote image URLs are not permitted")
    try:
        from PIL import Image
    except Exception as error:  # pragma: no cover - runtime environment guard
        raise MCQAggregateError("Pillow is required for image encoding") from error
    try:
        with Image.open(path_or_uri) as source:
            image = source.convert("RGB")
        output = BytesIO()
        image.save(output, format="PNG")
    except Exception as error:
        raise MCQAggregateError("an input image could not be decoded") from error
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii")


def build_messages(
    row: Sample | Mapping[str, Any], image_uris: Sequence[str] | None = None
) -> list[dict[str, Any]]:
    """Return OpenAI multimodal content, preserving every image's order."""

    images = list(row.images) if isinstance(row, Sample) else _row_images(row)
    uris = list(image_uris) if image_uris is not None else [_image_data_uri(image) for image in images]
    if len(uris) != len(images):
        raise MCQAggregateError("image URI count differs from the input image count")
    content: list[dict[str, Any]] = [
        {"type": "image_url", "image_url": {"url": uri}} for uri in uris
    ]
    content.append({"type": "text", "text": build_prompt(row)})
    return [{"role": "user", "content": content}]


def build_request(
    row: Sample | Mapping[str, Any],
    *,
    model_id: str,
    image_uris: Sequence[str] | None = None,
    thinking: bool = False,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    top_k: int = DEFAULT_TOP_K,
    min_p: float = DEFAULT_MIN_P,
    presence_penalty: float = DEFAULT_PRESENCE_PENALTY,
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    structured_json: bool = True,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Build one local OpenAI-compatible request with all tunable controls."""

    extra_body: dict[str, Any] = {
        "top_k": top_k,
        "min_p": min_p,
        "repetition_penalty": repetition_penalty,
        "chat_template_kwargs": {"enable_thinking": bool(thinking)},
    }
    if structured_json:
        extra_body["structured_outputs"] = {
            "json": STRUCTURED_JSON_SCHEMA,
            "disable_additional_properties": True,
        }
    return {
        "model": model_id,
        "messages": build_messages(row, image_uris),
        "temperature": temperature,
        "top_p": top_p,
        "presence_penalty": presence_penalty,
        "max_tokens": max_tokens,
        "seed": seed,
        # vLLM/OpenAI-compatible extensions belong in extra_body.
        "extra_body": extra_body,
    }


def parse_choice(content: Any) -> str | None:
    """Parse an explicit uppercase choice without semantic judging.

    A strict JSON object is preferred.  The exact official Qwen example is a
    JSON member fragment (``"answer": "C"``), so a unique explicit answer or
    choice marker and a standalone terminal uppercase choice are also valid.
    Option-text matching, case folding and semantic judging are forbidden.
    """

    if not isinstance(content, str):
        return None
    text = content.strip()
    match = _FENCE_RE.fullmatch(text)
    if match:
        text = match.group(1).strip()
    def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    if text.startswith("{") and text.endswith("}"):
        try:
            value = json.loads(text, object_pairs_hook=_pairs)
        except (TypeError, ValueError, json.JSONDecodeError):
            value = None
        if isinstance(value, dict) and set(value) == {"answer"}:
            answer = value.get("answer")
            if isinstance(answer, str) and answer in CHOICES:
                return answer
            return None

    marked = _ANSWER_MARKER_RE.findall(text)
    if marked and len(set(marked)) == 1:
        return marked[0]
    terminal = _TERMINAL_CHOICE_RE.search(text)
    return terminal.group(1) if terminal else None


# Explicit aliases make the parser convenient to use from small CPU tests.
parse_answer = parse_choice
extract_answer = parse_choice
parse_json_answer = parse_choice
parse_model_answer = parse_choice


def _response_text(response: Any) -> str | None:
    try:
        content = response.choices[0].message.content
    except Exception:
        return None
    return content if isinstance(content, str) else None


def aggregate_outcomes(
    outcomes: Sequence[tuple[str, str, str | None, bool]],
    *,
    expected_total: int | None = None,
    expected_dataset_totals: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Reduce private outcomes to aggregate-only overall and dataset counts."""

    selected: Counter[str] = Counter()
    dataset_selected: dict[str, Counter[str]] = {}
    dataset_total: Counter[str] = Counter()
    dataset_correct: Counter[str] = Counter()
    dataset_api_failure: Counter[str] = Counter()
    dataset_invalid_format: Counter[str] = Counter()
    correct = api_failure = invalid_format = 0
    for dataset, gold, prediction, provider_failed in outcomes:
        if not isinstance(dataset, str) or not dataset:
            raise MCQAggregateError("in-memory dataset label is malformed")
        if gold not in CHOICES:
            raise MCQAggregateError("in-memory gold answer is malformed")
        if not isinstance(provider_failed, bool):
            raise MCQAggregateError("in-memory provider status is malformed")
        dataset_total[dataset] += 1
        dataset_selected.setdefault(dataset, Counter())
        if provider_failed:
            if prediction is not None:
                raise MCQAggregateError("provider failure cannot contain a prediction")
            api_failure += 1
            dataset_api_failure[dataset] += 1
            continue
        if prediction not in CHOICES:
            invalid_format += 1
            dataset_invalid_format[dataset] += 1
            continue
        selected[prediction] += 1
        dataset_selected[dataset][prediction] += 1
        correct += int(gold == prediction)
        dataset_correct[dataset] += int(gold == prediction)
    total = len(outcomes)
    if expected_total is not None and total != expected_total:
        raise MCQAggregateError("aggregate row count differs from the formal total")
    if expected_dataset_totals is not None and dict(dataset_total) != dict(expected_dataset_totals):
        raise MCQAggregateError("aggregate dataset row counts differ from the formal totals")
    counts = {choice: selected[choice] for choice in CHOICES}
    datasets = {
        dataset: {
            "correct": dataset_correct[dataset],
            "total": dataset_total[dataset],
            "accuracy": dataset_correct[dataset] / dataset_total[dataset],
            "invalid_count": (
                dataset_api_failure[dataset] + dataset_invalid_format[dataset]
            ),
            "api_failure_count": dataset_api_failure[dataset],
            "invalid_format_count": dataset_invalid_format[dataset],
            "selected_option_counts": {
                choice: dataset_selected[dataset][choice] for choice in CHOICES
            },
        }
        for dataset in dataset_total
    }
    return {
        "correct": correct,
        "total": total,
        "accuracy": (correct / total) if total else 0.0,
        "invalid_count": api_failure + invalid_format,
        "api_failure_count": api_failure,
        "invalid_format_count": invalid_format,
        "selected_option_counts": counts,
        "datasets": datasets,
    }


aggregate_counts = aggregate_outcomes


def _validate_api_base(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise MCQAggregateError("api base must be an http(s) URL")
    if parsed.hostname not in LOCAL_HOSTS:
        raise MCQAggregateError("api base must be local loopback")
    return value.rstrip("/")


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError("expected true/false")


def _assert_output_scope(path: Path, *, allow_missing: bool = True) -> Path:
    """Validate every existing ancestor without following symlinks."""

    target = _absolute(path)
    root = _absolute(OUTPUT_ROOT)
    try:
        relative = target.relative_to(root)
    except ValueError as error:
        raise MCQAggregateError("output must be below H_Workspace/Output") from error
    try:
        root_info = root.lstat()
    except FileNotFoundError:
        if not allow_missing:
            raise MCQAggregateError("output root is unavailable") from None
        root_info = None
    except OSError as error:
        raise MCQAggregateError("output root is unavailable") from error
    if root_info is not None and (stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode)):
        raise MCQAggregateError("output root must be a real directory")

    cursor = root
    # Check all existing parent components.  A missing descendant is fine;
    # _write_create_once creates it only after this scope check.
    for component in relative.parts[:-1]:
        cursor = cursor / component
        try:
            info = cursor.lstat()
        except FileNotFoundError:
            break
        except OSError as error:
            raise MCQAggregateError("output parent is unavailable") from error
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise MCQAggregateError("output parent must contain no symlinks")
    if target.exists() or target.is_symlink():
        raise MCQAggregateError("aggregate receipt already exists")
    return target


def _validate_args(args: argparse.Namespace, *, inspect_inputs: bool) -> tuple[Path, Path, Path]:
    if args.workers < 1 or args.workers > 256:
        raise MCQAggregateError("workers must be in [1, 256]")
    if args.max_tokens < 1:
        raise MCQAggregateError("max-tokens must be positive")
    if args.seed < 0 or args.seed > 2**32 - 1:
        raise MCQAggregateError("seed must be in [0, 2^32-1]")
    if not 0 <= args.top_p <= 1:
        raise MCQAggregateError("top-p must be in [0, 1]")
    if not 0 <= args.min_p <= 1:
        raise MCQAggregateError("min-p must be in [0, 1]")
    if args.top_k < 0:
        raise MCQAggregateError("top-k must be non-negative")
    if not -2 <= args.presence_penalty <= 2:
        raise MCQAggregateError("presence-penalty must be in [-2, 2]")
    if not isinstance(args.model_id, str) or not args.model_id.strip():
        raise MCQAggregateError("model id must be non-empty")
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
    return mmstar, blink, output


def _protocol(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "thinking": bool(args.thinking),
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": args.min_p,
        "presence_penalty": args.presence_penalty,
        "repetition_penalty": args.repetition_penalty,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "workers": args.workers,
        "datasets": list(args.datasets),
        "response_format": (
            "vllm_structured_json_answer_A_D"
            if args.structured_json
            else "prompt_only_explicit_answer_or_terminal_A_D"
        ),
        "image_order": "source_tsv_order",
        "gold_scope": "scorer_memory_only",
        "sample_level_output": False,
    }


def _receipt(
    *,
    aggregate: Mapping[str, Any],
    dataset_md5: Mapping[str, str],
    prompt_hash: str,
    model_id: str,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    data_hash = _sha256_bytes(_canonical(dict(dataset_md5)))
    model_hash = _sha256_bytes(model_id.encode("utf-8"))
    protocol_hash = _sha256_bytes(_canonical(protocol))
    hashes = {
        "data": data_hash,
        "prompt": prompt_hash,
        "model": model_hash,
        "protocol": protocol_hash,
    }
    # Keep the public schema deliberately aggregate-only.  Dataset names,
    # row counts and file MD5s are provenance, never row-level evidence.
    return {
        "schema_version": SCHEMA_VERSION,
        "datasets": {
            name: {"total": DATASET_ROWS[name], "md5": dataset_md5[name]}
            for name in dataset_md5
        },
        "correct": int(aggregate["correct"]),
        "total": int(aggregate["total"]),
        "accuracy": float(aggregate["accuracy"]),
        "invalid_count": int(aggregate["invalid_count"]),
        "api_failure_count": int(aggregate["api_failure_count"]),
        "invalid_format_count": int(aggregate["invalid_format_count"]),
        "selected_option_counts": dict(aggregate["selected_option_counts"]),
        "scores": dict(aggregate["datasets"]),
        "model_id": model_id,
        "protocol": dict(protocol),
        "hashes": hashes,
        "data_hash": data_hash,
        "prompt_hash": prompt_hash,
        "model_hash": model_hash,
        "protocol_hash": protocol_hash,
    }


def _write_create_once(path: Path, receipt: Mapping[str, Any]) -> None:
    target = _assert_output_scope(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Re-check after mkdir so a pre-existing ancestor cannot be swapped for a
    # symlink between validation and the create-once operation.
    _assert_output_scope(target)
    try:
        parent_info = target.parent.lstat()
    except OSError as error:
        raise MCQAggregateError("output parent is unavailable") from error
    if stat.S_ISLNK(parent_info.st_mode) or not stat.S_ISDIR(parent_info.st_mode):
        raise MCQAggregateError("output parent must be a real directory")
    encoded = (json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    try:
        descriptor = os.open(
            target,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
        )
    except FileExistsError as error:
        raise MCQAggregateError("aggregate receipt already exists") from error
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mmstar-tsv", "--mmstar_tsv", "--mmstar", dest="mmstar_tsv", required=True, type=Path)
    parser.add_argument("--blink-tsv", "--blink_tsv", "--blink", dest="blink_tsv", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-id", "--model_id", "--model", dest="model_id", required=True)
    parser.add_argument("--api-base", "--api_base", required=True)
    parser.add_argument("--api-key", "--api_key", default="EMPTY")
    parser.add_argument(
        "--thinking", "--enable-thinking", "--enable_thinking", nargs="?", const=True,
        default=False, type=_parse_bool,
    )
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-p", "--top_p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--top-k", "--top_k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--min-p", "--min_p", type=float, default=DEFAULT_MIN_P)
    parser.add_argument("--presence-penalty", "--presence_penalty", type=float, default=DEFAULT_PRESENCE_PENALTY)
    parser.add_argument(
        "--repetition-penalty", "--repetition_penalty", type=float,
        default=DEFAULT_REPETITION_PENALTY,
    )
    parser.add_argument("--max-tokens", "--max_tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--workers", "--worker", dest="workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--structured-json", "--structured_json", nargs="?", const=True,
        default=True, type=_parse_bool,
    )
    parser.add_argument(
        "--datasets", nargs="+", choices=tuple(DATASET_ROWS),
        default=list(DATASET_ROWS),
    )
    parser.add_argument("--image-root", "--image_root", type=Path, default=None)
    parser.add_argument("--dry-run", "--dry_run", action="store_true")
    return parser


def _dry_run_metadata(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "dry_run": True,
        "reads_data": False,
        "connects_api": False,
        "writes_output": False,
        "datasets": {name: DATASET_ROWS[name] for name in args.datasets},
        "protocol": _protocol(args),
        "hashes_deferred": True,
    }


def _infer(
    samples: Sequence[Sample],
    *,
    args: argparse.Namespace,
) -> list[tuple[str, str, str | None, bool]]:
    try:
        from openai import OpenAI
    except Exception as error:  # pragma: no cover - runtime guard
        raise MCQAggregateError("openai package is required for a real run") from error

    from threading import local

    thread_local = local()

    def client() -> Any:
        value = getattr(thread_local, "client", None)
        if value is None:
            value = OpenAI(
                api_key=args.api_key,
                base_url=_validate_api_base(args.api_base),
                timeout=3600,
                max_retries=0,
            )
            thread_local.client = value
        return value

    def one(sample: Sample) -> tuple[str, str, str | None, bool]:
        request = build_request(
            sample,
            model_id=args.model_id,
            thinking=args.thinking,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            min_p=args.min_p,
            presence_penalty=args.presence_penalty,
            repetition_penalty=args.repetition_penalty,
            max_tokens=args.max_tokens,
            structured_json=args.structured_json,
            seed=args.seed,
        )
        response = client().chat.completions.create(**request)
        return sample.dataset, sample.gold, parse_choice(_response_text(response)), False

    results: list[tuple[str, str, str | None, bool]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(one, sample): sample for sample in samples}
        for future in as_completed(futures):
            sample = futures[future]
            try:
                results.append(future.result())
            except Exception:
                # API failures are private invalid outcomes; never expose a
                # sample index, question, image, or provider text.
                results.append((sample.dataset, sample.gold, None, True))
    return results


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        mmstar, blink, output = _validate_args(args, inspect_inputs=not args.dry_run)
        if args.dry_run:
            print(json.dumps(_dry_run_metadata(args), sort_keys=True))
            return 0

        paths = {"MMStar": mmstar, "BLINK": blink}
        samples: list[Sample] = []
        dataset_md5: dict[str, str] = {}
        for dataset in args.datasets:
            loaded, digest = _read_dataset(
                paths[dataset], dataset, image_root=args.image_root
            )
            samples.extend(loaded)
            dataset_md5[dataset] = digest
        prompts = [build_prompt(sample) for sample in samples]
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
            prompt_hash=_hash_texts(prompts),
            model_id=args.model_id,
            protocol=protocol,
        )
        _write_create_once(output, receipt)
        # Safe aggregate-only stdout; no row data is ever printed.
        dataset_summary = " ".join(
            f"{name.lower()}={aggregate['datasets'][name]['correct']}/"
            f"{aggregate['datasets'][name]['total']}"
            for name in args.datasets
        )
        print(
            f"MCQ_CHOICE_FORMAL correct={aggregate['correct']} "
            f"total={aggregate['total']} invalid_count={aggregate['invalid_count']} "
            f"{dataset_summary}"
        )
        return 0
    except MCQAggregateError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
