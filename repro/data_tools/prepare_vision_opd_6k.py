#!/usr/bin/env python3
"""Reproducibly extract and convert the pinned Vision-OPD-6K snapshot.

The raw Hugging Face snapshot is never modified or deleted. Split archives are
streamed directly into the separate media tree, so no temporary joined archive
is needed.
"""

from __future__ import annotations

import argparse
import io
import os
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from vision_opd_common import (
    DEFAULT_MEDIA_ROOT,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RAW_ROOT,
    EXPECTED_RAW_SIZES,
    EXPECTED_ROWS,
    REPO_ID,
    REVISION,
    atomic_write_json,
    canonical_json,
    clean_question,
    iter_jsonl,
    resolve_media_path,
    sha256_bytes,
    sha256_file,
)


class ConcatenatedReader(io.RawIOBase):
    """A non-seekable binary stream backed by ordered archive parts."""

    def __init__(self, paths: Sequence[Path]) -> None:
        super().__init__()
        if not paths:
            raise ValueError("At least one archive part is required")
        self.paths = list(paths)
        self._index = 0
        self._handle: BinaryIO | None = None

    def readable(self) -> bool:
        return True

    def _open_next(self) -> bool:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        if self._index >= len(self.paths):
            return False
        self._handle = self.paths[self._index].open("rb")
        self._index += 1
        return True

    def readinto(self, buffer: bytearray | memoryview) -> int:
        view = memoryview(buffer).cast("B")
        while True:
            if self._handle is None and not self._open_next():
                return 0
            assert self._handle is not None
            count = self._handle.readinto(view)
            if count:
                return count
            if not self._open_next():
                return 0

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        super().close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--media-root", type=Path, default=DEFAULT_MEDIA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the exact pinned HF revision into raw-root before preparing.",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Use an already extracted media tree and only rebuild train.parquet.",
    )
    return parser.parse_args()


def download_snapshot(raw_root: Path) -> None:
    raw_root.mkdir(parents=True, exist_ok=True)
    command = [
        "hf",
        "download",
        REPO_ID,
        "--repo-type",
        "dataset",
        "--revision",
        REVISION,
        "--local-dir",
        str(raw_root),
    ]
    print(f"Downloading {REPO_ID}@{REVISION} into {raw_root}", flush=True)
    subprocess.run(command, check=True)


def verify_raw_snapshot(raw_root: Path) -> dict[str, int]:
    observed: dict[str, int] = {}
    for relative, expected_size in EXPECTED_RAW_SIZES.items():
        path = raw_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Missing raw snapshot file: {path}")
        size = path.stat().st_size
        if size != expected_size:
            raise ValueError(
                f"Raw file size mismatch for {path}: expected {expected_size}, got {size}"
            )
        observed[relative] = size
    return observed


def normalized_member_path(member_name: str, expected_prefix: str) -> Path | None:
    posix = PurePosixPath(member_name)
    if posix.is_absolute() or ".." in posix.parts:
        raise ValueError(f"Unsafe archive member: {member_name!r}")
    parts = [part for part in posix.parts if part not in ("", ".")]
    if parts and parts[0] == expected_prefix:
        parts = parts[1:]
    if not parts:
        return None
    return Path(*parts)


def extract_archive(
    archive_parts: Sequence[Path], destination: Path, expected_prefix: str
) -> dict[str, int]:
    """Stream a gzip tar (possibly split across files) into destination."""
    destination.mkdir(parents=True, exist_ok=True)
    files = 0
    directories = 0
    uncompressed_bytes = 0
    seen: set[Path] = set()

    with ConcatenatedReader(archive_parts) as stream:
        with tarfile.open(fileobj=stream, mode="r|gz") as archive:
            for member in archive:
                relative = normalized_member_path(member.name, expected_prefix)
                if relative is None:
                    continue
                if relative in seen:
                    raise ValueError(f"Duplicate archive member after normalization: {relative}")
                seen.add(relative)
                target = destination / relative

                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    directories += 1
                    continue
                if not member.isfile():
                    raise ValueError(
                        f"Unsupported non-regular archive member {member.name!r} "
                        f"(type={member.type!r})"
                    )

                target.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise OSError(f"Unable to read archive member: {member.name}")
                descriptor, temporary_name = tempfile.mkstemp(
                    prefix=f".{target.name}.", suffix=".extracting", dir=target.parent
                )
                temporary_path = Path(temporary_name)
                try:
                    with os.fdopen(descriptor, "wb") as output, source:
                        shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
                        output.flush()
                        os.fsync(output.fileno())
                    if temporary_path.stat().st_size != member.size:
                        raise OSError(
                            f"Extracted size mismatch for {member.name}: "
                            f"expected {member.size}, got {temporary_path.stat().st_size}"
                        )
                    os.chmod(temporary_path, member.mode & 0o777)
                    os.replace(temporary_path, target)
                    try:
                        os.utime(target, (member.mtime, member.mtime))
                    except (OSError, OverflowError):
                        pass
                except BaseException:
                    temporary_path.unlink(missing_ok=True)
                    raise

                files += 1
                uncompressed_bytes += member.size

    return {
        "archive_parts": len(archive_parts),
        "files": files,
        "directories": directories,
        "uncompressed_bytes": uncompressed_bytes,
    }


def extract_media(raw_root: Path, media_root: Path) -> dict[str, dict[str, int]]:
    student_parts = sorted((raw_root / "images").glob("images.tar.gz*"))
    expected_student_parts = [raw_root / "images" / f"images.tar.gz0{i}" for i in range(6)]
    if student_parts != expected_student_parts:
        raise ValueError(
            "Student archive parts are missing or unexpected: "
            f"expected {expected_student_parts}, got {student_parts}"
        )

    plan = [
        ("images", student_parts, media_root / "images"),
        (
            "teacher_images",
            [raw_root / "teacher_images/teacher_images.tar.gz"],
            media_root / "teacher_images",
        ),
        (
            "original_images",
            [raw_root / "original_images/original_images.tar.gz"],
            media_root / "original_images",
        ),
    ]
    results: dict[str, dict[str, int]] = {}
    for prefix, parts, destination in plan:
        print(
            f"Extracting {prefix}: {len(parts)} archive part(s) -> {destination}",
            flush=True,
        )
        results[prefix] = extract_archive(parts, destination, prefix)
        print(f"Extracted {results[prefix]['files']} {prefix} files", flush=True)
    return results


def image_refs(paths: list[str], media_root: Path) -> list[dict[str, str]]:
    # Keep upstream's `path` key while also supplying `image`, which the pinned
    # qwen_vl_utils consumed by the local veRL checkout requires.
    return [
        {
            "path": str(resolve_media_path(media_root, path)),
            "image": str(resolve_media_path(media_root, path)),
        }
        for path in paths
    ]


def build_record(index: int, item: dict[str, Any], media_root: Path) -> dict[str, Any]:
    required = ("images", "teacher_images", "original_images", "bbox", "problem", "answer")
    missing = [field for field in required if field not in item]
    if missing:
        raise KeyError(f"Row {index} is missing required fields: {missing}")

    images = item["images"]
    teacher_images = item["teacher_images"]
    original_images = item["original_images"]
    if not all(isinstance(value, list) and value for value in (images, teacher_images, original_images)):
        raise TypeError(f"Row {index} image fields must be non-empty lists")
    for relative in [*images, *teacher_images, *original_images]:
        resolved = resolve_media_path(media_root, relative)
        if not resolved.is_file():
            raise FileNotFoundError(f"Row {index} references a missing media file: {resolved}")

    source_record_json = canonical_json(item)
    source_record_sha256 = sha256_bytes(source_record_json.encode("utf-8"))
    source_id = f"vision_opd_6k_{index:04d}_{source_record_sha256[:12]}"
    source_extra = item.get("extra_info") or {}

    return {
        "source_id": source_id,
        "source_record_sha256": source_record_sha256,
        "data_source": "zwz_rl_vqa_bbox_teacher",
        "prompt": [{"role": "user", "content": item["problem"]}],
        "images": image_refs(images, media_root),
        "bbox_images": image_refs(teacher_images, media_root),
        "ability": "visual_question_answering",
        "reward_model": {"style": "none", "ground_truth": item["answer"]},
        "extra_info": {
            "answer": item["answer"],
            "question": clean_question(item["problem"]),
            "source_extra_info": {
                "answer": str(source_extra.get("answer", "")),
                "question": str(source_extra.get("question", "")),
            },
            "row_index": index,
            "source_revision": REVISION,
            "original_images": image_refs(original_images, media_root),
            "bbox": [int(value) for value in item["bbox"]],
            "relative_images": list(images),
            "relative_teacher_images": list(teacher_images),
            "relative_original_images": list(original_images),
            "source_record_json": source_record_json,
        },
    }


def parquet_schema() -> pa.Schema:
    image_ref = pa.struct(
        [
            pa.field("path", pa.string(), nullable=False),
            pa.field("image", pa.string(), nullable=False),
        ]
    )
    source_extra_info = pa.struct(
        [
            pa.field("answer", pa.string(), nullable=False),
            pa.field("question", pa.string(), nullable=False),
        ]
    )
    schema = pa.schema(
        [
            pa.field("source_id", pa.string(), nullable=False),
            pa.field("source_record_sha256", pa.string(), nullable=False),
            pa.field("data_source", pa.string(), nullable=False),
            pa.field(
                "prompt",
                pa.list_(
                    pa.struct(
                        [
                            pa.field("role", pa.string(), nullable=False),
                            pa.field("content", pa.string(), nullable=False),
                        ]
                    )
                ),
                nullable=False,
            ),
            pa.field("images", pa.list_(image_ref), nullable=False),
            pa.field("bbox_images", pa.list_(image_ref), nullable=False),
            pa.field("ability", pa.string(), nullable=False),
            pa.field(
                "reward_model",
                pa.struct(
                    [
                        pa.field("style", pa.string(), nullable=False),
                        pa.field("ground_truth", pa.string(), nullable=False),
                    ]
                ),
                nullable=False,
            ),
            pa.field(
                "extra_info",
                pa.struct(
                    [
                        pa.field("answer", pa.string(), nullable=False),
                        pa.field("question", pa.string(), nullable=False),
                        pa.field("source_extra_info", source_extra_info, nullable=False),
                        pa.field("row_index", pa.int64(), nullable=False),
                        pa.field("source_revision", pa.string(), nullable=False),
                        pa.field("original_images", pa.list_(image_ref), nullable=False),
                        pa.field("bbox", pa.list_(pa.int64()), nullable=False),
                        pa.field("relative_images", pa.list_(pa.string()), nullable=False),
                        pa.field("relative_teacher_images", pa.list_(pa.string()), nullable=False),
                        pa.field("relative_original_images", pa.list_(pa.string()), nullable=False),
                        pa.field("source_record_json", pa.string(), nullable=False),
                    ]
                ),
                nullable=False,
            ),
        ],
        metadata={
            b"dataset": REPO_ID.encode(),
            b"revision": REVISION.encode(),
            b"expected_rows": str(EXPECTED_ROWS).encode(),
        },
    )
    return schema


def write_parquet(raw_root: Path, media_root: Path, output_dir: Path) -> dict[str, Any]:
    jsonl_path = raw_root / "train.jsonl"
    records = [build_record(index, item, media_root) for index, item in iter_jsonl(jsonl_path)]
    if len(records) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} JSONL rows, got {len(records)}")
    if len({record["source_id"] for record in records}) != EXPECTED_ROWS:
        raise ValueError("Derived source_id values are not unique")

    table = pa.Table.from_pylist(records, schema=parquet_schema())
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "train.parquet"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_dir
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        pq.write_table(
            table,
            temporary_path,
            compression="zstd",
            compression_level=9,
            use_dictionary=True,
            write_statistics=True,
            data_page_version="2.0",
        )
        os.replace(temporary_path, output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return {
        "path": str(output_path),
        "rows": table.num_rows,
        "columns": table.num_columns,
        "bytes": output_path.stat().st_size,
        "sha256": sha256_file(output_path),
    }


def main() -> None:
    args = parse_args()
    raw_root = args.raw_root.resolve()
    media_root = args.media_root.resolve()
    output_dir = args.output_dir.resolve()

    if args.download:
        download_snapshot(raw_root)
    raw_sizes = verify_raw_snapshot(raw_root)
    print(f"Pinned raw snapshot verified ({sum(raw_sizes.values()):,} bytes)", flush=True)

    if args.skip_extract:
        extraction: dict[str, dict[str, int]] | str = "skipped-by-request"
    else:
        media_root.mkdir(parents=True, exist_ok=True)
        extraction = extract_media(raw_root, media_root)

    parquet = write_parquet(raw_root, media_root, output_dir)
    receipt = {
        "dataset": REPO_ID,
        "revision": REVISION,
        "raw_root": str(raw_root),
        "media_root": str(media_root),
        "output_dir": str(output_dir),
        "raw_sizes": raw_sizes,
        "extraction": extraction,
        "parquet": parquet,
    }
    receipt_path = output_dir / "prepare_receipt.json"
    atomic_write_json(receipt_path, receipt)
    print(f"Prepared {parquet['rows']} rows at {parquet['path']}", flush=True)
    print(f"Preparation receipt: {receipt_path}", flush=True)


if __name__ == "__main__":
    main()
