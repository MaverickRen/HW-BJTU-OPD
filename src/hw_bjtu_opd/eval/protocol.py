"""Portable benchmark contracts used by :mod:`scripts.evaluate`.

This module is deliberately stdlib-only.  It does not import model-serving or
dataset frameworks, so it can be used by preflight and tests on CPU-only
machines.  The full benchmark prompt/parser remains in ``repro/eval_tools``;
the portable entry point uses the same first-option semantics for VStar and
the same answer-field parser for the MCQ benchmarks.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BENCHMARKS: dict[str, dict[str, Any]] = {
    "vstar": {
        "label": "VStar",
        "rows": 191,
        "dataset_repo": "lmms-lab/vstar-bench",
        "dataset_revision": "b44023b4dca749ed8a76b85eb576627d05a1c174",
        "protocol": "vstar_frozen_first_option_v1",
        "default_limit": 8,
    },
    "mmstar": {
        "label": "MMStar",
        "rows": 1500,
        "dataset_repo": "VLMEvalKit MMStar.tsv",
        "dataset_revision": "md5:e1ecd2140806c1b1bbf54b43372efb9e",
        "protocol": "mmstar_qwen35_modelcard_thinking_v2",
        "default_limit": 0,
    },
    "blink": {
        "label": "BLINK-v5",
        "rows": 1901,
        "dataset_repo": "VLMEvalKit BLINK.tsv",
        "dataset_revision": "md5:d5e8af148b10ac69f535ff7b23f3f989",
        "protocol": "blink_deterministic_checkpoint_comparison_v5",
        "default_limit": 0,
    },
    "zoombench": {
        "label": "ZoomBench",
        "rows": 845,
        "dataset_repo": "inclusionAI/ZoomBench",
        "dataset_revision": "b788097e57d30510c6877824833234a73bf80d25",
        "protocol": "zoombench_score_aggregate_v1",
        "default_limit": 0,
    },
}
BENCHMARK_ALIASES = {
    "vstar": "vstar",
    "v*": "vstar",
    "mmstar": "mmstar",
    "blink": "blink",
    "blink-v5": "blink",
    "zoombench": "zoombench",
    "zoom": "zoombench",
}
CHOICES = tuple("ABCDE")
_ANSWER_MARKER = re.compile(
    r"(?is)\b(?:final\s+)?(?:answer|choice|option)\b\s*(?:is\s*)?[:=\-]?\s*[\[({`'\"*]*([A-E])\b"
)
_TERMINAL = re.compile(r"(?is)(?:^|\n)\s*[\[({`'\"*]*([A-E])[\]})'\"*.,;:!?]*\s*$")
_JSON_FENCE = re.compile(r"\A```(?:json)?\s*\n?(.*?)\n?```\Z", re.IGNORECASE | re.DOTALL)
_VSTAR_MARKER = re.compile(r"\(([A-Z])\)")
_VSTAR_PUNCT = re.compile(r"([A-Z])[\.\)\s]")
_VSTAR_ANY = re.compile(r"([A-Z])")


class EvaluationError(RuntimeError):
    """A user-facing, fail-closed evaluation error."""


@dataclass(frozen=True)
class Record:
    """One in-memory benchmark row; raw responses are never in receipts."""

    index: int
    images: tuple[str, ...]
    query: str
    gold: str
    category: str = "unknown"
    options: tuple[tuple[str, str], ...] = ()


def canonical_benchmark(name: str) -> str:
    try:
        return BENCHMARK_ALIASES[name.strip().lower()]
    except (AttributeError, KeyError) as exc:
        raise EvaluationError(f"unknown benchmark {name!r}; choose vstar, mmstar, blink or zoombench") from exc


def _read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as exc:
        raise EvaluationError(f"dataset is missing: {path}") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"dataset is not valid JSON: {path}") from exc


def _as_rows(value: Any, path: Path) -> list[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        for key in ("data", "samples", "rows", "items"):
            if isinstance(value.get(key), list):
                value = value[key]
                break
    if not isinstance(value, list) or any(not isinstance(item, Mapping) for item in value):
        raise EvaluationError(f"dataset must be a JSON array of objects: {path}")
    return list(value)


def _image_list(row: Mapping[str, Any], root: Path) -> tuple[str, ...]:
    raw = row.get("images", row.get("image", row.get("image_path")))
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, Sequence) or isinstance(raw, (bytes, bytearray)):
        return ()
    result: list[str] = []
    root = root.resolve()
    for item in raw:
        if not isinstance(item, str) or not item.strip():
            continue
        candidate = Path(item)
        if candidate.is_absolute():
            raise EvaluationError("benchmark image paths must be relative to the dataset")
        unresolved = root / candidate
        current = unresolved
        while current != root:
            if current.is_symlink():
                raise EvaluationError(f"benchmark image path contains a symlink: {item}")
            current = current.parent
        candidate = unresolved.resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise EvaluationError(f"benchmark image path escapes the dataset: {item}") from exc
        result.append(str(candidate))
    return tuple(result)


def _options(row: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    result: list[tuple[str, str]] = []
    raw = row.get("options")
    if isinstance(raw, Mapping):
        result.extend((str(key).upper(), str(value)) for key, value in raw.items() if str(key).upper() in CHOICES)
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        result.extend((CHOICES[index], str(value)) for index, value in enumerate(raw[: len(CHOICES)]))
    for label in CHOICES:
        if label in row and row[label] not in (None, "") and not any(key == label for key, _ in result):
            result.append((label, str(row[label])))
    return tuple(result)


def load_records(path: Path | str, benchmark: str, *, limit: int | None = None) -> tuple[list[Record], str]:
    """Load the portable JSON form and return rows plus a stable file hash.

    The historical prepared evaluator uses JSON with ``images/query/response``
    fields.  Keeping this reader narrow makes malformed or accidentally
    private-path data fail before a request is sent.
    """

    canonical_benchmark(benchmark)
    source = Path(path).expanduser().resolve()
    if source.suffix.lower() in {".jsonl", ".ndjson"}:
        try:
            rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise EvaluationError(f"dataset is not valid JSONL: {source}") from exc
        rows = _as_rows(rows, source)
    else:
        if source.suffix.lower() == ".tsv":
            try:
                with source.open("r", encoding="utf-8", newline="") as handle:
                    rows = list(csv.DictReader(handle, delimiter="\t"))
            except (OSError, UnicodeError, csv.Error) as exc:
                raise EvaluationError(f"dataset is not valid TSV: {source}") from exc
            rows = _as_rows(rows, source)
        else:
            rows = _as_rows(_read_json(source), source)
    if limit is not None:
        if limit < 1:
            raise EvaluationError("--limit must be positive")
        rows = rows[:limit]
    records: list[Record] = []
    for index, row in enumerate(rows):
        query = row.get("query", row.get("text", row.get("question", "")))
        gold = row.get("response", row.get("label", row.get("answer", "")))
        if not isinstance(query, str) or not isinstance(gold, (str, int)):
            raise EvaluationError(f"row {index + 1} lacks query/answer text: {source}")
        records.append(
            Record(
                index=index,
                images=_image_list(row, source.parent),
                query=query.strip(),
                gold=str(gold).strip(),
                category=str(row.get("category", "unknown") or "unknown"),
                options=_options(row),
            )
        )
    if not records:
        raise EvaluationError(f"dataset contains no rows: {source}")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    return records, digest


def parse_choice(text: Any, *, valid: Iterable[str] = CHOICES) -> str | None:
    """Parse a single option using the frozen explicit/terminal convention."""

    if not isinstance(text, str):
        return None
    allowed = {str(item).upper() for item in valid}
    candidate = text.strip()
    match = _JSON_FENCE.fullmatch(candidate)
    if match:
        candidate = match.group(1).strip()
    if candidate.startswith("{") and candidate.endswith("}"):
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            return None
        value = payload.get("answer") if isinstance(payload, Mapping) else None
        return value.upper() if isinstance(value, str) and value.upper() in allowed else None
    markers = [item.upper() for item in _ANSWER_MARKER.findall(candidate)]
    if markers:
        return markers[-1] if len(set(markers)) == 1 and markers[-1] in allowed else None
    terminal = _TERMINAL.search(candidate)
    if terminal and terminal.group(1).upper() in allowed:
        return terminal.group(1).upper()
    # VStar's original scorer uses the first standalone option token.  Keep
    # this fallback explicit and conservative for the portable quick check.
    tokens = re.findall(r"(?<![A-Z])([A-E])(?![A-Z])", candidate.upper())
    return tokens[0] if len(tokens) == 1 and tokens[0] in allowed else None


def parse_vstar_choice(text: Any) -> str | None:
    """Exact frozen VStar first-option dispatch used for the reported score."""

    if not isinstance(text, str):
        return None
    candidate = text
    if "<answer>" in candidate:
        start = candidate.find("<answer>")
        end = candidate.find("</answer>")
        if start != -1 and end != -1:
            candidate = candidate[start + len("<answer>") : end].strip()
    elif "Answer:" in candidate:
        candidate = candidate[candidate.find("Answer:") :].strip()
    for pattern in (_VSTAR_MARKER, _VSTAR_PUNCT, _VSTAR_ANY):
        match = pattern.search(candidate)
        if match is not None and match.group(1) in CHOICES:
            return match.group(1)
    return None


def parse_prediction(benchmark: str, text: Any, options: Mapping[str, str]) -> str | None:
    """Dispatch to the released parser without silently changing protocols."""

    name = canonical_benchmark(benchmark)
    if name == "vstar":
        return parse_vstar_choice(text)
    # MMStar's released content parser is the strict JSON/explicit/terminal
    # convention implemented by parse_choice.  BLINK's option-text fallback is
    # intentionally left to the source-frozen full evaluator.
    if name == "mmstar":
        return parse_choice(text, valid=options or dict.fromkeys("ABCD", ""))
    if name == "blink":
        raise EvaluationError("portable BLINK execution is not enabled; use the frozen full evaluator")
    raise EvaluationError("ZoomBench requires the pinned 27B semantic judge; use the frozen full evaluator")


def score_records(
    records: Sequence[Record],
    predictions: Sequence[str | None],
    *,
    benchmark: str = "vstar",
) -> dict[str, Any]:
    if len(records) != len(predictions):
        raise EvaluationError("prediction count differs from dataset rows")
    name = canonical_benchmark(benchmark)
    gold_parser = parse_vstar_choice if name == "vstar" else parse_choice
    golds = [gold_parser(row.gold) for row in records]
    if any(gold is None for gold in golds):
        raise EvaluationError(f"{name}: dataset contains an invalid gold answer")
    correct = sum(int(gold == prediction) for gold, prediction in zip(golds, predictions))
    return {
        "correct": correct,
        "total": len(records),
        "accuracy_percent": 100.0 * correct / len(records) if records else 0.0,
        "invalid_count": sum(int(prediction is None) for prediction in predictions),
    }


def build_plan(
    *,
    benchmark: str,
    model: str | None,
    data: str | None,
    output: str | None,
    limit: int | None,
    api_base: str | None,
    execute: bool,
) -> dict[str, Any]:
    name = benchmark if benchmark == "all" else canonical_benchmark(benchmark)
    if name == "all":
        spec = {
            "label": "VStar/MMStar/BLINK-v5/ZoomBench",
            "rows": sum(item["rows"] for item in BENCHMARKS.values()),
            "protocol": "four_benchmark_suite_v1",
            "dataset_repo": "multiple pinned sources",
            "dataset_revision": None,
            "default_limit": 0,
        }
    else:
        spec = BENCHMARKS[name]
    return {
        "schema_version": "hw_bjtu_opd_eval_plan_v1",
        "status": "execute" if execute else "dry_run",
        "benchmark": name,
        "label": spec["label"],
        "expected_rows": spec["rows"],
        "requested_rows": limit if limit is not None else (spec["default_limit"] or spec["rows"]),
        "model": model,
        "data": str(Path(data).expanduser()) if data else None,
        "output": str(Path(output).expanduser()) if output else None,
        "api_base": api_base,
        "protocol": spec["protocol"],
        "dataset": {"repo_id": spec["dataset_repo"], "revision": spec["dataset_revision"]},
        "gpu_required": execute,
        "raw_predictions_persisted": False,
    }
