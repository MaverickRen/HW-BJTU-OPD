#!/usr/bin/env python3
"""Fail-closed launcher for the source-pinned official QwenLM judge.

The official judge intentionally turns a request that fails three times into
the literal answer ``No``.  That is unsafe for a formal evaluation because an
outage is then indistinguishable from an incorrect model answer.  This driver
loads the pinned official module and replaces only its API transport helper:
successful responses keep the official ordering and text semantics, while any
exhausted request raises and prevents the official output file from being
published.

The API key is accepted only through
``VISION_OPD_REFERENCE_JUDGE_API_KEY`` (with the legacy environment fallbacks)
and is injected into ``sys.argv`` in-process.  It is never a process command
line argument.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import stat
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterable, Sequence


API_KEY_ENV = "VISION_OPD_REFERENCE_JUDGE_API_KEY"


class JudgeAPIError(RuntimeError):
    """At least one judge request exhausted all retries."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_pinned_official(path: Path, expected_sha256: str) -> ModuleType:
    path = Path(os.path.abspath(path))
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise JudgeAPIError(f"official judge source is not a regular file: {path}")
    if path.resolve(strict=True) != path:
        raise JudgeAPIError(f"official judge source contains a symlink or alias: {path}")
    actual = _sha256(path)
    if actual != expected_sha256:
        raise JudgeAPIError(
            f"official judge source SHA256 differs: expected {expected_sha256}, got {actual}"
        )
    spec = importlib.util.spec_from_file_location("vision_opd_pinned_judge_qwenlm", path)
    if spec is None or spec.loader is None:
        raise JudgeAPIError(f"could not load official judge source: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def hardened_judge_via_api(
    prompts: Sequence[str],
    api_base: str,
    api_key: str,
    judge_model: str,
    judge_max_tokens: int,
    parallel_workers: int = 32,
    *,
    client_factory: Callable[[], Any] | None = None,
    completed_iterator: Callable[[Iterable[Any]], Iterable[Any]] | None = None,
    retry_delay_seconds: float = 1.0,
) -> list[str]:
    """Mirror the official API judge, but raise after an exhausted request.

    ``client_factory`` and ``completed_iterator`` are dependency-injection
    seams for CPU-only fault tests.  Production defaults are the same OpenAI
    client, completion order, timeout, temperature, token limit, and three
    attempts used by the pinned official implementation.
    """

    if client_factory is None:
        from openai import OpenAI

        client_factory = lambda: OpenAI(api_key=api_key, base_url=api_base, timeout=600)
    if completed_iterator is None:
        from tqdm import tqdm

        completed_iterator = lambda futures: tqdm(
            as_completed(futures), total=len(futures), desc="LLM Judge"
        )

    thread_local = threading.local()
    results = [""] * len(prompts)

    def get_client() -> Any:
        client = getattr(thread_local, "client", None)
        if client is None:
            client = client_factory()
            thread_local.client = client
        return client

    def call_one(index: int, prompt: str) -> tuple[int, str]:
        client = get_client()
        last_error: BaseException | None = None
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=judge_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=judge_max_tokens,
                )
                return index, (response.choices[0].message.content or "").strip()
            except Exception as error:  # Match the official retry boundary.
                last_error = error
                if attempt < 2:
                    time.sleep(retry_delay_seconds)
        error_name = type(last_error).__name__ if last_error is not None else "unknown"
        raise JudgeAPIError(
            f"judge API request {index} failed after 3 attempts ({error_name})"
        ) from last_error

    executor = ThreadPoolExecutor(max_workers=parallel_workers)
    futures = [executor.submit(call_one, index, prompt) for index, prompt in enumerate(prompts)]
    try:
        for future in completed_iterator(futures):
            index, text = future.result()
            results[index] = text
    except BaseException:
        for future in futures:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)
    return results


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-script", type=Path, required=True)
    parser.add_argument("--official-sha256", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--judge-max-tokens", type=int, default=2048)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if len(args.official_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in args.official_sha256
    ):
        raise JudgeAPIError("--official-sha256 must be a lowercase SHA256 digest")

    module = _load_pinned_official(args.official_script, args.official_sha256)
    module.judge_via_api = hardened_judge_via_api
    api_key = os.environ.get(
        API_KEY_ENV,
        os.environ.get("JUDGE_API_KEY", os.environ.get("OPENAI_API_KEY", "EMPTY")),
    )
    original_argv = sys.argv
    try:
        sys.argv = [
            str(args.official_script),
            "--benchmark",
            args.benchmark,
            "--model",
            args.model,
            "--api_base",
            args.api_base,
            "--api_key",
            api_key,
            "--judge_model",
            args.judge_model,
            "--judge_max_tokens",
            str(args.judge_max_tokens),
        ]
        result = module.main()
    finally:
        sys.argv = original_argv
    return int(result or 0)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except JudgeAPIError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
