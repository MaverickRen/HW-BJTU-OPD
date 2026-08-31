"""Implementation for the small public evaluation and preflight CLIs."""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import mimetypes
import os
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from typing import Any

from .protocol import (
    BENCHMARKS,
    EvaluationError,
    Record,
    build_plan,
    canonical_benchmark,
    load_records,
    parse_prediction,
    score_records,
)

_OPTIONAL_MODULES = {
    "Pillow": "PIL",
    "openai": "openai",
    "transformers": "transformers",
    "torch": "torch",
    "vllm": "vllm",
}
_EVALUATOR_FILES = {
    "vstar": ("repro/eval_tools/vstar_formal_aggregate_v1.py", "repro/eval_tools/score_formal_vstar_v1.py"),
    "mmstar": (
        "repro/eval_tools/mmstar_qwen35_modelcard_aggregate_v2.py",
        "repro/eval_tools/mcq_choice_formal_aggregate_v1.py",
    ),
    "blink": (
        "repro/eval_tools/mcq_blink_checkpoint_comparison_aggregate_v5.py",
        "repro/eval_tools/mcq_blink_official_aggregate_v4.py",
        "repro/eval_tools/mcq_blink_official_aggregate_v3.py",
        "repro/eval_tools/mcq_choice_formal_aggregate_v2.py",
        "repro/eval_tools/mcq_choice_formal_aggregate_v1.py",
    ),
    "zoombench": (
        "repro/eval_tools/zoombench_formal_aggregate_v1.py",
        "repro/eval_tools/prepare_zoombench.py",
    ),
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _file_info(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    if not path.is_file() or path.is_symlink():
        return {"path": str(path), "exists": True, "regular": False}
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": str(path), "exists": True, "regular": True, "sha256": digest, "bytes": path.stat().st_size}


def _closure(root: Path, benchmark: str) -> dict[str, Any]:
    files = []
    missing = []
    for relative in _EVALUATOR_FILES[benchmark]:
        path = root / relative
        item = _file_info(path)
        files.append(item)
        if not item.get("regular"):
            missing.append(relative)
    return {"files": files, "missing": missing, "complete": not missing}


def _dependency_status() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, module in _OPTIONAL_MODULES.items():
        result[name] = {"module": module, "installed": importlib.util.find_spec(module) is not None}
    return result


def preflight(
    *,
    root: Path | None = None,
    benchmark: str = "vstar",
    model: Path | None = None,
    data: Path | None = None,
    strict: bool = False,
    execute: bool = False,
) -> dict[str, Any]:
    """Return a structured preflight report without importing GPU libraries."""

    root = (root or _repo_root()).resolve()
    names = tuple(BENCHMARKS) if benchmark == "all" else (canonical_benchmark(benchmark),)
    report: dict[str, Any] = {
        "schema_version": "hw_bjtu_opd_preflight_v1",
        "status": "ready",
        "root": str(root),
        "cpu_only": True,
        "execute_requested": execute,
        "benchmarks": {},
        "dependencies": _dependency_status(),
        "model": None,
        "data": None,
        "errors": [],
        "warnings": [],
    }
    if model is not None:
        model = model.expanduser().resolve()
        report["model"] = {"path": str(model), "exists": model.is_dir(), "config": (model / "config.json").is_file()}
        if not model.is_dir() or not (model / "config.json").is_file():
            report["errors" if strict else "warnings"].append(
                "model checkpoint must be a directory containing config.json"
            )
    elif execute:
        report["errors" if strict else "warnings"].append(
            "--model was not supplied; pass the local checkpoint used by the server"
        )

    for name in names:
        closure = _closure(root, name)
        report["benchmarks"][name] = {
            "label": BENCHMARKS[name]["label"],
            "expected_rows": BENCHMARKS[name]["rows"],
            "protocol": BENCHMARKS[name]["protocol"],
            "evaluator": closure,
        }
        if not closure["complete"]:
            report["errors" if strict else "warnings"].append(f"{name}: evaluator dependency closure is incomplete")

    if data is not None:
        data = data.expanduser().resolve()
        report["data"] = {"path": str(data), "exists": data.exists(), "directory": data.is_dir()}
        if not data.exists():
            report["errors" if strict else "warnings"].append(f"dataset is missing: {data}")
        elif execute:
            for name in names:
                try:
                    _data_for_benchmark(data, name)
                except EvaluationError as exc:
                    report["errors" if strict else "warnings"].append(str(exc))
    elif execute:
        report["errors" if strict else "warnings"].append(
            "--data was not supplied; pass a prepared benchmark JSON/JSONL file"
        )

    if execute:
        unsupported = [name for name in names if name in {"blink", "zoombench"}]
        if unsupported:
            report["errors" if strict else "warnings"].append(
                "portable execution does not support: " + ", ".join(unsupported)
            )
        for dependency in ("Pillow", "vllm"):
            if not report["dependencies"][dependency]["installed"]:
                report["errors" if strict else "warnings"].append(f"execute-time dependency is missing: {dependency}")

    if strict and report["errors"]:
        report["status"] = "failed"
    elif report["warnings"]:
        report["status"] = "needs_resources"
    return report


def _mime_data_uri(path: Path, *, benchmark: str) -> str:
    try:
        if benchmark == "vstar":
            from PIL import Image

            image = Image.open(path).convert("RGB")
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            payload = buffer.getvalue()
            while len(payload) > 20 * 1024 * 1024 and min(image.size) > 100:
                image = image.resize(
                    (int(image.size[0] * 0.75), int(image.size[1] * 0.75)),
                    Image.Resampling.LANCZOS,
                )
                buffer = BytesIO()
                image.save(buffer, format="PNG")
                payload = buffer.getvalue()
            mime = "image/png"
        else:
            payload = path.read_bytes()
            mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    except (ImportError, OSError) as exc:
        raise EvaluationError(f"image is unavailable: {path}") from exc
    return f"data:{mime};base64,{base64.b64encode(payload).decode('ascii')}"


def _request_url(api_base: str) -> str:
    value = api_base.rstrip("/")
    parsed = urllib.parse.urlparse(value)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise EvaluationError("--api-base must be a loopback HTTP endpoint")
    if value.endswith("/chat/completions"):
        return value
    if value.endswith("/v1"):
        return value + "/chat/completions"
    return value + "/v1/chat/completions"


def _api_request(*, api_base: str, api_key: str, model_id: str, row: Record, benchmark: str, timeout: float) -> str:
    content: list[dict[str, Any]] = []
    for image in row.images:
        content.append({"type": "image_url", "image_url": {"url": _mime_data_uri(Path(image), benchmark=benchmark)}})
    content.append({"type": "text", "text": row.query})
    payload: dict[str, Any] = {
        "model": model_id,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": 32768,
        "seed": 42,
        # This function emits raw HTTP JSON. OpenAI's Python client would
        # merge its ``extra_body`` argument into the request, so vLLM-specific
        # fields must already be top-level here.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if benchmark == "mmstar":
        content[-1]["text"] = (
            row.query
            + '\nPlease show your choice in the `answer` field with only the choice letter, e.g., `"answer": "C"`.'
        )
        payload["temperature"] = 1.0
        payload["top_p"] = 0.95
        payload["seed"] = 0
        payload["top_k"] = 20
        payload["min_p"] = 0.0
        payload["presence_penalty"] = 1.5
        payload["chat_template_kwargs"] = {"enable_thinking": True}
    request = urllib.request.Request(
        _request_url(api_base),
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key or 'EMPTY'}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.HTTPError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"model API request failed for row {row.index + 1}: {exc}") from exc
    try:
        value = result["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise EvaluationError("model API returned no message content") from exc
    if not isinstance(value, str):
        raise EvaluationError("model API message content is not text")
    return value


def _data_for_benchmark(data: Path, benchmark: str) -> Path:
    if data.is_file():
        return data
    candidates = {
        "vstar": ("vstar.json", "vstar.jsonl"),
        "mmstar": ("mmstar.json", "mmstar.jsonl", "mmstar.tsv"),
        "blink": ("blink.json", "blink.jsonl", "blink.tsv"),
        "zoombench": ("zoombench.json", "zoombench.jsonl"),
    }[benchmark]
    for candidate in candidates:
        path = data / candidate
        if path.is_file():
            return path
    raise EvaluationError(f"no {benchmark} dataset found below {data}")


def evaluate(
    *,
    benchmark: str,
    model: Path | None,
    data: Path,
    output: Path,
    model_id: str,
    api_base: str,
    api_key: str,
    limit: int | None,
    timeout: float = 300.0,
    workers: int = 8,
) -> dict[str, Any]:
    """Run one or all benchmark requests and persist only aggregate counts."""

    names = tuple(BENCHMARKS) if benchmark == "all" else (canonical_benchmark(benchmark),)
    if any(name in {"blink", "zoombench"} for name in names):
        raise EvaluationError(
            "BLINK-v5 and ZoomBench require the source-frozen full evaluator; "
            "the portable low-cost entry supports VStar and MMStar only"
        )
    if workers < 1:
        raise EvaluationError("--workers must be positive")
    result: dict[str, Any] = {
        "schema_version": "hw_bjtu_opd_evaluation_v1",
        "model_id": model_id,
        "model_artifact": model.name if model else None,
        "raw_predictions_persisted": False,
        "benchmarks": {},
    }
    for name in names:
        source = _data_for_benchmark(data, name)
        # A user-provided limit is useful for smoke tests.  ``all`` defaults to
        # the exact row count; the single VStar command defaults to eight.
        row_limit = limit if limit is not None else (BENCHMARKS[name]["default_limit"] or BENCHMARKS[name]["rows"])
        records, digest = load_records(source, name, limit=row_limit)
        if len(records) != row_limit:
            raise EvaluationError(f"{name}: expected {row_limit} rows, found {len(records)}")
        for row in records:
            if len(row.images) != 1:
                raise EvaluationError(f"{name}: row {row.index + 1} must contain exactly one image")
            image = Path(row.images[0])
            if image.is_symlink() or not image.is_file():
                raise EvaluationError(f"{name}: row {row.index + 1} image is unavailable")
        with ThreadPoolExecutor(max_workers=workers) as executor:
            predictions = list(
                executor.map(
                    lambda row, benchmark_name=name: _api_request(
                        api_base=api_base,
                        api_key=api_key,
                        model_id=model_id,
                        row=row,
                        benchmark=benchmark_name,
                        timeout=timeout,
                    ),
                    records,
                )
            )
        parsed = [parse_prediction(name, value, dict(row.options)) for value, row in zip(predictions, records)]
        scores = score_records(records, parsed, benchmark=name)
        result["benchmarks"][name] = {
            **scores,
            "dataset": {"name": source.name, "sha256": digest},
            "protocol": BENCHMARKS[name]["protocol"],
            "generation": {
                "seed": 42 if name == "vstar" else 0,
                "enable_thinking": name == "mmstar",
                "temperature": 0 if name == "vstar" else 1.0,
                "max_tokens": 32768,
            },
            "quick": len(records) < BENCHMARKS[name]["rows"],
        }
    result["status"] = "complete"
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return result


def render_plan(
    *,
    benchmark: str,
    model: Path | None,
    data: Path | None,
    output: Path | None,
    model_id: str,
    api_base: str | None,
    limit: int | None,
    execute: bool,
) -> dict[str, Any]:
    return build_plan(
        benchmark=benchmark,
        model=str(model) if model else None,
        data=str(data) if data else None,
        output=str(output) if output else None,
        limit=limit,
        api_base=api_base,
        execute=execute,
    )
