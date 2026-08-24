#!/usr/bin/env python3
"""Fail-closed checks for JSON, result arithmetic, secrets and public path leaks."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {"", ".md", ".py", ".sh", ".json", ".toml", ".txt", ".jinja", ".patch", ".example"}
SECRET_PATTERNS = {
    "github_pat": re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    "hf_token": re.compile(r"hf_[A-Za-z0-9]{20,}"),
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}
PUBLIC_PREFIXES = ("README.md", "docs/", "configs/", "scripts/", "src/", "results/", "hf_card/")
INTERNAL_PATHS = ("/minimax-3d-rw-backup/users/", "/home/jiazhi/", "/root/")
PATH_SCANNER_IMPLEMENTATIONS = {"scripts/verify_repo.py", "scripts/materialize_sft_dataset.py"}


def files() -> list[Path]:
    result: list[Path] = []
    for current, directories, names in os.walk(ROOT, followlinks=False):
        current_path = Path(current)
        directories[:] = [
            name
            for name in directories
            if name not in {".git", ".venv", ".pytest_cache", "__pycache__", ".ruff_cache", ".mypy_cache"}
        ]
        for name in names:
            path = current_path / name
            if path.is_symlink():
                raise ValueError(f"repository contains a symlink: {path.relative_to(ROOT)}")
            if path.suffix.lower() in TEXT_SUFFIXES or path.name in {"LICENSE", ".gitignore", ".gitattributes"}:
                result.append(path)
    return sorted(result)


def main() -> int:
    errors: list[str] = []
    for path in files():
        relative = path.relative_to(ROOT).as_posix()
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeError:
            continue
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                errors.append(f"{label} pattern in {relative}")
        if relative not in PATH_SCANNER_IMPLEMENTATIONS and any(
            relative == prefix or relative.startswith(prefix) for prefix in PUBLIC_PREFIXES
        ):
            for marker in INTERNAL_PATHS:
                if marker in text:
                    errors.append(f"internal path {marker!r} in public surface {relative}")
        if path.suffix == ".json":
            try:
                json.loads(text)
            except json.JSONDecodeError as exc:
                errors.append(f"invalid JSON {relative}: {exc}")
    results = json.loads((ROOT / "results/core_results.json").read_text(encoding="utf-8"))
    for row in results["primary_blink_v5_matrix"]:
        observed = sum(item["correct"] / item["total"] * 100.0 for item in row["benchmarks"].values()) / 4
        if abs(observed - row["macro_percent"]) > 1e-10:
            errors.append(f"macro differs for {row['id']}: {observed} vs {row['macro_percent']}")
    patch = ROOT / "patches/verl-qwen35-opd.patch"
    if not patch.is_file() or patch.stat().st_size < 10_000:
        errors.append("veRL release patch is missing or unexpectedly small")
    if errors:
        print(json.dumps({"status": "failed", "errors": errors}, indent=2), file=sys.stderr)
        return 2
    print(json.dumps({"status": "passed", "text_files": len(files()), "primary_result_rows": 7}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
