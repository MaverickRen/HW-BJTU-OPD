#!/usr/bin/env python3
"""Shared constants and helpers for the pinned Vision-OPD-6K snapshot."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator


REPO_ID = "yuanqianhao/Vision-OPD-6K"
REVISION = "eb5c1c2e7b9a7b6a619efe4161c7369c71bf8af4"
REVISION_SHORT = "eb5c1c2"
EXPECTED_ROWS = 6_241
SEED = 42

H_WORKSPACE = Path(__file__).resolve().parents[3]
DEFAULT_RAW_ROOT = H_WORKSPACE / "Dataset/raw/Vision-OPD-6K" / REVISION_SHORT
DEFAULT_MEDIA_ROOT = H_WORKSPACE / "Dataset/media/Vision-OPD-6K" / REVISION_SHORT
DEFAULT_OUTPUT_DIR = H_WORKSPACE / "Dataset/processed/b1_vision_opd_6k"
DEFAULT_MANIFEST_DIR = H_WORKSPACE / "Dataset/manifests"

REMOVE_HINT = (
    "Only focus on the objects inside the red bounding box in the image "
    "to answer this question."
)

EXPECTED_RAW_SIZES = {
    "images/images.tar.gz00": 5_368_709_120,
    "images/images.tar.gz01": 5_368_709_120,
    "images/images.tar.gz02": 5_368_709_120,
    "images/images.tar.gz03": 5_368_709_120,
    "images/images.tar.gz04": 5_368_709_120,
    "images/images.tar.gz05": 1_496_961_022,
    "original_images/original_images.tar.gz": 6_249_850_743,
    "teacher_images/teacher_images.tar.gz": 2_942_713_517,
    "train.jsonl": 4_566_587,
}

# Git-LFS object IDs are the SHA-256 of the downloaded payload. The three small
# Git-managed files were hashed from this pinned revision after download.
EXPECTED_RAW_SHA256 = {
    ".gitattributes": "9e75dd981de037ec3769f24f790e126bc5a160b6871f510214e68dc70649aeeb",
    "README.md": "2d00d4a1ddd41a2c69c1f002f2dcec94e88668c0272af320ac17117ff27f3eab",
    "images/images.tar.gz00": "e72239bb03d393886e84aa2758eabcad387b03e86a7d7ce7238178f5832f52d0",
    "images/images.tar.gz01": "3e0872e7ae37cc4b94019ec0f23ae274ec5c1f7dbebfa087e6219e0da9301979",
    "images/images.tar.gz02": "b0bd79677a7439e87745f39716da57b85796dc241974032831ef67772f96be1a",
    "images/images.tar.gz03": "4b230e8f814b22117126064e8eb6d4d7ba6f639860c6982a387ce8bb814e99ee",
    "images/images.tar.gz04": "d7872e12c1ab49ca4fd0773f1e6aa621a7c3ef8355ca7e72fc57e8ce6292cfc6",
    "images/images.tar.gz05": "44844e97bc43ee0bf8546ae50ca5b709decb22c144bfda410f3ddfe8c546060a",
    "original_images/original_images.tar.gz": "fdd112788d07956bdfe160c8d5b93c071f5e097439c4dfea2c02f01b781bc123",
    "teacher_images/teacher_images.tar.gz": "f2fb6541e8e1ea4e33114aff9d511c5e8ce764c0972798151c8d5b1b5b91883e",
    "train.jsonl": "8ad2fb81da0f6fba1766545dc5f84cc2250e48704738757461b2d75aa31821df",
}


def clean_question(problem: str) -> str:
    """Match the question cleanup performed by the upstream Vision-OPD script."""
    text = (problem or "").replace("<image>", "").strip()
    text = text.replace(f"\n\n{REMOVE_HINT}", "")
    text = text.replace(REMOVE_HINT, "")
    return text.strip()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                raise ValueError(f"Blank JSONL line at zero-based row {index}: {path}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"Expected an object at zero-based row {index}: {path}")
            yield index, value


def safe_relative_path(value: str) -> Path:
    """Convert a POSIX dataset path while rejecting absolute/traversal paths."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"Invalid relative path: {value!r}")
    posix_path = PurePosixPath(value)
    if posix_path.is_absolute() or ".." in posix_path.parts:
        raise ValueError(f"Unsafe relative path: {value!r}")
    parts = [part for part in posix_path.parts if part not in ("", ".")]
    if not parts:
        raise ValueError(f"Empty relative path after normalization: {value!r}")
    return Path(*parts)


def resolve_media_path(media_root: Path, value: str) -> Path:
    path = (media_root / safe_relative_path(value)).resolve()
    root = media_root.resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"Path escapes media root: {value!r}")
    return path


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    atomic_write_text(path, payload)


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def summarize_numbers(values: Iterable[int]) -> dict[str, float | int | None]:
    ordered = sorted(values)
    if not ordered:
        return {"min": None, "max": None, "median": None}
    middle = len(ordered) // 2
    if len(ordered) % 2:
        median: float | int = ordered[middle]
    else:
        median = (ordered[middle - 1] + ordered[middle]) / 2
    return {"min": ordered[0], "max": ordered[-1], "median": median}
