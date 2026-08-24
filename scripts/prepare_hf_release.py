#!/usr/bin/env python3
"""Assemble one hard-linked Hugging Face release tree without copying weights."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Sequence

from _common import ReleaseError, require_dir, require_file, write_json


MODEL_FILES = {
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors.index.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "video_preprocessor_config.json",
    "vocab.json",
}


def selected_model_files(root: Path) -> list[Path]:
    files = [path for path in root.iterdir() if path.is_file() and not path.is_symlink()]
    selected = [path for path in files if path.name in MODEL_FILES or path.name.endswith(".safetensors")]
    if not any(path.name.endswith(".safetensors") for path in selected):
        raise ReleaseError(f"model has no safetensors: {root}")
    if not any(path.name == "config.json" for path in selected):
        raise ReleaseError(f"model has no config.json: {root}")
    return sorted(selected)


def link_file(source: Path, target: Path) -> None:
    if source.is_symlink() or not source.is_file():
        raise ReleaseError(f"source is not a regular file: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.is_symlink() or not target.is_file() or target.stat().st_size != source.stat().st_size:
            raise ReleaseError(f"unsafe or mismatched resumed target: {target}")
        return
    os.link(source, target, follow_symlinks=False)


def link_tree(source: Path, target: Path) -> tuple[int, int]:
    count = 0
    size = 0
    for current, directories, files in os.walk(source, followlinks=False):
        current_path = Path(current)
        directories[:] = sorted(name for name in directories if not (current_path / name).is_symlink())
        for name in sorted(files):
            item = current_path / name
            if item.is_symlink():
                raise ReleaseError(f"dataset tree contains a symlink: {item}")
            destination = target / item.relative_to(source)
            link_file(item, destination)
            count += 1
            size += item.stat().st_size
    return count, size


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--output", required=True, type=Path)
    value.add_argument("--model-9b", required=True, type=Path)
    value.add_argument("--model-27b", required=True, type=Path)
    value.add_argument("--dataset", required=True, type=Path)
    value.add_argument("--card", required=True, type=Path)
    value.add_argument("--dataset-card", required=True, type=Path)
    value.add_argument("--model-9b-card", required=True, type=Path)
    value.add_argument("--model-27b-card", required=True, type=Path)
    value.add_argument("--license", required=True, type=Path)
    value.add_argument("--execute", action="store_true")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        model9 = selected_model_files(require_dir(args.model_9b, "9B model"))
        model27 = selected_model_files(require_dir(args.model_27b, "27B model"))
        require_dir(args.dataset, "portable dataset")
        require_file(args.dataset / "manifest.json", "portable dataset manifest")
        require_file(args.card, "Hugging Face README")
        require_file(args.dataset_card, "dataset card")
        require_file(args.model_9b_card, "9B model card")
        require_file(args.model_27b_card, "27B model card")
        require_file(args.license, "license")
        plan = {
            "status": "dry_run" if not args.execute else "assembling",
            "model_9b_files": len(model9),
            "model_9b_bytes": sum(path.stat().st_size for path in model9),
            "model_27b_files": len(model27),
            "model_27b_bytes": sum(path.stat().st_size for path in model27),
            "dataset": str(args.dataset),
            "output": str(args.output),
        }
        if not args.execute:
            print(json.dumps(plan, indent=2, sort_keys=True))
            return 0
        if args.output.is_symlink():
            raise ReleaseError("release output cannot be a symlink")
        args.output.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.card, args.output / "README.md")
        shutil.copy2(args.license, args.output / "LICENSE")
        (args.output / "datasets/sft-v1-10k").mkdir(parents=True, exist_ok=True)
        (args.output / "checkpoints/qwen35-9b-sft-v1-10k").mkdir(parents=True, exist_ok=True)
        (args.output / "checkpoints/qwen35-27b-sft-v1-10k").mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.dataset_card, args.output / "datasets/sft-v1-10k/README.md")
        shutil.copy2(args.model_9b_card, args.output / "checkpoints/qwen35-9b-sft-v1-10k/README.md")
        shutil.copy2(args.model_27b_card, args.output / "checkpoints/qwen35-27b-sft-v1-10k/README.md")
        for source in model9:
            link_file(source, args.output / "checkpoints/qwen35-9b-sft-v1-10k" / source.name)
        for source in model27:
            link_file(source, args.output / "checkpoints/qwen35-27b-sft-v1-10k" / source.name)
        dataset_count, dataset_bytes = link_tree(args.dataset, args.output / "datasets/sft-v1-10k")
        manifest = {
            "schema_version": "hw_bjtu_opd_hf_release_v1",
            "status": "complete",
            "checkpoints": {
                "qwen35-9b-sft-v1-10k": {"files": len(model9), "bytes": sum(path.stat().st_size for path in model9)},
                "qwen35-27b-sft-v1-10k": {"files": len(model27), "bytes": sum(path.stat().st_size for path in model27)},
            },
            "dataset": {"name": "sft-v1-10k", "files": dataset_count, "bytes": dataset_bytes},
            "source_paths_recorded": False,
            "credentials_recorded": False,
        }
        write_json(args.output / "release_manifest.json", manifest)
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0
    except (ReleaseError, OSError, ValueError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
