#!/usr/bin/env python3
"""Create a portable, path-sanitized SFT_V1 10K snapshot with its media."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from _common import ReleaseError, require_file, sha256_file, write_json


EXPECTED_ROWS = 10_000
EXPECTED_SOURCE_SHA256 = "9f56d58c076c255df3bc660ba3c193b1cff8dd69c51ad2f73c844f5f2a8c49b0"
INTERNAL_PREFIXES = (
    "/minimax-3d-rw-backup/users/",
    "/home/jiazhi/",
    "/root/",
)
SECRET_PATTERNS = (
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
)


def plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)) or type(value).__name__ == "ndarray":
        return [plain(item) for item in list(value)]
    if hasattr(value, "as_py"):
        return plain(value.as_py())
    return value


def has_internal_path(value: str) -> bool:
    return any(prefix in value for prefix in INTERNAL_PREFIXES)


def scrub_metadata(value: Any) -> Any:
    """Remove machine-local provenance paths while preserving scientific metadata."""

    value = plain(value)
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            if isinstance(item, str) and has_internal_path(item) and "path" in key.lower():
                continue
            cleaned[key] = scrub_metadata(item)
        return cleaned
    if isinstance(value, list):
        return [scrub_metadata(item) for item in value]
    if isinstance(value, str) and has_internal_path(value):
        raise ReleaseError("an internal path remains in non-path metadata")
    return value


def safe_suffix(path: Path) -> str:
    suffix = path.suffix.lower()
    return suffix if suffix in {".png", ".jpg", ".jpeg", ".webp", ".bmp"} else ".img"


class MediaStore:
    def __init__(self, root: Path, mode: str) -> None:
        self.root = root
        self.mode = mode
        self.by_source: dict[str, str] = {}
        self.content_files: dict[str, Path] = {}
        self.total_bytes = 0

    def add(self, source_value: str) -> str:
        if source_value in self.by_source:
            return self.by_source[source_value]
        source = Path(source_value)
        require_file(source, "referenced image")
        digest = sha256_file(source)
        relative = Path("media") / digest[:2] / f"{digest}{safe_suffix(source)}"
        target = self.root / relative
        prior = self.content_files.get(digest)
        if prior is None:
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if target.is_symlink() or not target.is_file() or sha256_file(target) != digest:
                    raise ReleaseError(f"unsafe or mismatched resumed media file: {target}")
            elif self.mode == "hardlink":
                os.link(source, target, follow_symlinks=False)
            else:
                shutil.copy2(source, target, follow_symlinks=False)
            self.content_files[digest] = target
            self.total_bytes += target.stat().st_size
        else:
            relative = prior.relative_to(self.root)
        portable = relative.as_posix()
        self.by_source[source_value] = portable
        return portable


def rewrite_image_list(value: Any, media: MediaStore) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for raw in plain(value) or []:
        if not isinstance(raw, dict):
            raise ReleaseError("image entry is not an object")
        source = raw.get("path") or raw.get("image")
        if not isinstance(source, str) or not source:
            raise ReleaseError("image entry has no path")
        relative = media.add(source)
        clean = {key: scrub_metadata(item) for key, item in raw.items() if key not in {"path", "image"}}
        clean.update({"path": relative, "image": relative})
        result.append(clean)
    return result


def sanitize_extra(value: Any) -> dict[str, Any]:
    extra = plain(value) or {}
    if not isinstance(extra, dict):
        raise ReleaseError("extra_info is not an object")
    # These duplicate source-image paths are not consumed by SFT and are the
    # only path-bearing payload outside the model input columns.
    extra.pop("original_images", None)
    verification = extra.get("verification_metadata_json")
    if isinstance(verification, str):
        try:
            parsed = json.loads(verification)
        except json.JSONDecodeError as exc:
            raise ReleaseError("invalid verification_metadata_json") from exc
        extra["verification_metadata_json"] = json.dumps(
            scrub_metadata(parsed), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
    return scrub_metadata(extra)


def assert_public(value: Any) -> None:
    if isinstance(value, dict):
        for item in value.values():
            assert_public(item)
    elif isinstance(value, list):
        for item in value:
            assert_public(item)
    elif isinstance(value, str):
        if has_internal_path(value):
            raise ReleaseError("sanitized dataset still contains an internal absolute path")
        if any(pattern.search(value) for pattern in SECRET_PATTERNS):
            raise ReleaseError("sanitized dataset contains a credential-like string")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--source-parquet", required=True, type=Path)
    value.add_argument("--output", required=True, type=Path)
    value.add_argument("--mode", choices=("hardlink", "copy"), default="hardlink")
    value.add_argument("--execute", action="store_true")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        require_file(args.source_parquet, "source parquet")
        source_sha = sha256_file(args.source_parquet)
        if source_sha != EXPECTED_SOURCE_SHA256:
            raise ReleaseError(f"unexpected source parquet SHA256: {source_sha}")
        if not args.execute:
            print(
                json.dumps(
                    {
                        "status": "dry_run",
                        "source_sha256": source_sha,
                        "expected_rows": EXPECTED_ROWS,
                        "output": str(args.output),
                        "copy_mode": args.mode,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.output.is_symlink():
            raise ReleaseError("output cannot be a symlink")
        args.output.mkdir(parents=True, exist_ok=True)
        frame = pd.read_parquet(args.source_parquet)
        if len(frame) != EXPECTED_ROWS:
            raise ReleaseError(f"expected {EXPECTED_ROWS} rows, observed {len(frame)}")
        media = MediaStore(args.output, args.mode)
        rows: list[dict[str, Any]] = []
        license_counts: Counter[str] = Counter()
        source_counts: Counter[str] = Counter()
        for record in frame.to_dict(orient="records"):
            clean = {key: plain(item) for key, item in record.items()}
            clean["images"] = rewrite_image_list(record["images"], media)
            clean["bbox_images"] = rewrite_image_list(record["bbox_images"], media)
            clean["extra_info"] = sanitize_extra(record["extra_info"])
            assert_public(clean)
            license_counts[str(clean["extra_info"].get("source_license"))] += 1
            source_counts[str(clean["extra_info"].get("source_dataset"))] += 1
            rows.append(clean)
        target = args.output / "train_10000.parquet"
        if target.exists() or target.is_symlink():
            raise ReleaseError(f"create-once output parquet already exists: {target}")
        pd.DataFrame.from_records(rows).to_parquet(target, index=False, compression="zstd")
        output_sha = sha256_file(target)
        manifest = {
            "schema_version": "hw_bjtu_opd_sft_v1_10k_portable_v1",
            "status": "published",
            "rows": EXPECTED_ROWS,
            "source_parquet_sha256": source_sha,
            "portable_parquet": {
                "path": "train_10000.parquet",
                "bytes": target.stat().st_size,
                "sha256": output_sha,
            },
            "media": {
                "references": len(media.by_source),
                "unique_content_files": len(media.content_files),
                "bytes": media.total_bytes,
                "paths_relative_to_parquet": True,
            },
            "composition": {
                "fine_grained_single": 3800,
                "general_knowledge": 2600,
                "multi_image_reasoning": 3600,
            },
            "source_counts": dict(sorted(source_counts.items())),
            "row_level_license_counts": dict(sorted(license_counts.items())),
            "dataset_level_license_resolution": {
                "zhenjiemao__aRefCOCO": "cc-by-4.0",
                "zhenjiemao/aRefCOCO": "cc-by-4.0",
                "TIGER-Lab/Mantis-Instruct": "apache-2.0",
                "UCSC-VLAA/VLM-CapCurriculum-VisualReasoning-Data": "apache-2.0",
                "datajuicer/VeriSciQA": "cc-by-sa-4.0",
            },
            "decontamination": {
                "benchmarks": ["VStarBench", "MMStar", "BLINK", "ZoomBench"],
                "hard_overlap": "zero",
                "phash_radius": 4,
                "b28_excluded": True,
                "public_vision_opd_excluded": True,
            },
            "internal_absolute_paths": 0,
            "credentials": 0,
        }
        write_json(args.output / "manifest.json", manifest)
        print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except (ReleaseError, OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
