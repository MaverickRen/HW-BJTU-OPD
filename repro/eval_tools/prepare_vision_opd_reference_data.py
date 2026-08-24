#!/usr/bin/env python3
"""Atomically prepare the two pinned Vision-OPD reference benchmarks.

The upstream evaluator is retained as the conversion implementation, but its
Hugging Face downloads are forced to exact commits and all work happens in a
fresh sibling staging directory.  The final data root is published with
``renameat2(RENAME_NOREPLACE)`` only after row counts, UIDs, image decodes and
source parquet bytes have been audited.

Without ``--execute`` this program is a side-effect-free plan printer.
"""

from __future__ import annotations

import argparse
import ctypes
import fcntl
import hashlib
import io
import json
import os
import stat
import subprocess
import sys
import types
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence


SCHEMA_VERSION = "vision_opd_reference_preparation_v2"
REFERENCE_COMMIT = "c8a8fdd1f88eef1b5ef4fe6a8d64eb0272917471"
REFERENCE_PREPARE_SHA256 = "8e71bb3f04c741434ab505acfc4d2b6107cefec2864cdf909b2ff0e4fad79c5a"
EXPECTED_UID = 30853
RENAME_NOREPLACE = 1
AT_FDCWD = -100
ARCHIVED_TOOL_RELATIVE = "provenance/prepare_vision_opd_reference_data.py"
RECEIPT_RELATIVE = "preparation_receipt.json"


class PreparationError(RuntimeError):
    """A fail-closed preparation or verification error."""


@dataclass(frozen=True)
class SourceSpec:
    benchmark: str
    repo_id: str
    revision: str
    row_count: int
    json_name: str
    image_dir: str
    source_dir: str


SOURCES: tuple[SourceSpec, ...] = (
    SourceSpec(
        benchmark="vstar",
        repo_id="lmms-lab/vstar-bench",
        revision="b44023b4dca749ed8a76b85eb576627d05a1c174",
        row_count=191,
        json_name="vstar.json",
        image_dir="VStar_images",
        source_dir="vstar_data",
    ),
    SourceSpec(
        benchmark="mme-realworld-lite",
        repo_id="yifanzhang114/MME-RealWorld-lite-lmms-eval",
        revision="f6b0dc81ba4d3c39bd9b4e544578198d365ac084",
        row_count=1919,
        json_name="MME_RealWorld_Lite.json",
        image_dir="MME_RealWorld_Lite_images",
        source_dir="yifanzhang114_MME-RealWorld-lite-lmms-eval",
    ),
)
SOURCE_BY_REPO = {item.repo_id: item for item in SOURCES}


@dataclass(frozen=True)
class StableBytes:
    """Bytes and identity read from one unchanged, non-symlink regular file."""

    data: bytes
    sha256: str
    size: int
    device: int
    inode: int
    mode: int
    mtime_ns: int


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _pretty_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode(
            "utf-8"
        )
        + b"\n"
    )


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _stable_read_bytes(path: Path, *, label: str) -> StableBytes:
    """Read a regular file once and prove its path/descriptor stayed identical."""

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        before_path = path.lstat()
        descriptor = os.open(path, flags)
    except OSError as error:
        raise PreparationError(f"cannot open {label} {path}: {error}") from error
    chunks: list[bytes] = []
    try:
        before_fd = os.fstat(descriptor)
        if stat.S_ISLNK(before_path.st_mode) or not stat.S_ISREG(before_fd.st_mode):
            raise PreparationError(f"{label} is not a non-symlink regular file: {path}")
        if (before_path.st_dev, before_path.st_ino) != (before_fd.st_dev, before_fd.st_ino):
            raise PreparationError(f"{label} identity changed while opening: {path}")
        while True:
            chunk = os.read(descriptor, 8 * 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after_fd = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        after_path = path.lstat()
    except OSError as error:
        raise PreparationError(f"{label} disappeared after reading: {path}: {error}") from error
    fields = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns")
    if any(getattr(before_fd, key) != getattr(after_fd, key) for key in fields):
        raise PreparationError(f"{label} changed while reading: {path}")
    if any(getattr(after_fd, key) != getattr(after_path, key) for key in fields):
        raise PreparationError(f"{label} path changed after reading: {path}")
    data = b"".join(chunks)
    if len(data) != before_fd.st_size:
        raise PreparationError(f"short read for {label}: {path}")
    return StableBytes(
        data=data,
        sha256=_sha256_bytes(data),
        size=len(data),
        device=before_fd.st_dev,
        inode=before_fd.st_ino,
        mode=before_fd.st_mode,
        mtime_ns=before_fd.st_mtime_ns,
    )


def _hash_file(path: Path) -> tuple[str, int]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        before = path.lstat()
        descriptor = os.open(path, flags)
    except OSError as error:
        raise PreparationError(f"cannot open regular file {path}: {error}") from error
    digest = hashlib.sha256()
    size = 0
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise PreparationError(f"not a regular file: {path}")
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise PreparationError(f"file identity changed while opening: {path}")
        while True:
            chunk = os.read(descriptor, 8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    fields = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns")
    if any(getattr(opened, key) != getattr(after, key) for key in fields):
        raise PreparationError(f"file changed while hashing: {path}")
    if size != opened.st_size:
        raise PreparationError(f"short read while hashing: {path}")
    try:
        final_path = path.lstat()
    except OSError as error:
        raise PreparationError(f"file disappeared after hashing: {path}: {error}") from error
    if any(getattr(after, key) != getattr(final_path, key) for key in fields):
        raise PreparationError(f"file path changed after hashing: {path}")
    return digest.hexdigest(), size


def _record(path: Path, root: Path) -> dict[str, Any]:
    digest, size = _hash_file(path)
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "bytes": size,
        "sha256": digest,
    }


def _atomic_create(path: Path, payload: bytes, mode: int = 0o440) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, mode)
    except OSError as error:
        raise PreparationError(f"cannot create {path}: {error}") from error
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise PreparationError(f"short write while creating {path}")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISDIR(info.st_mode):
            raise PreparationError(f"cannot fsync non-directory: {path}")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _walk_regular_files(root: Path) -> tuple[list[Path], list[Path]]:
    """Return deterministic files/directories, rejecting links and special nodes."""

    _require_directory(root, label="inventory root")
    files: list[Path] = []
    directories: list[Path] = [root]
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            with os.scandir(directory) as iterator:
                entries = sorted(iterator, key=lambda item: item.name)
        except OSError as error:
            raise PreparationError(f"cannot enumerate directory {directory}: {error}") from error
        for entry in entries:
            path = Path(entry.path)
            try:
                info = entry.stat(follow_symlinks=False)
            except OSError as error:
                raise PreparationError(f"cannot stat inventory entry {path}: {error}") from error
            if stat.S_ISLNK(info.st_mode):
                raise PreparationError(f"symlink is forbidden in prepared tree: {path}")
            if stat.S_ISDIR(info.st_mode):
                directories.append(path)
                pending.append(path)
            elif stat.S_ISREG(info.st_mode):
                files.append(path)
            else:
                raise PreparationError(f"special node is forbidden in prepared tree: {path}")
    return sorted(files), sorted(directories)


def _durably_fsync_tree(root: Path) -> None:
    """Fsync every file, then every directory bottom-up, without following links."""

    files, directories = _walk_regular_files(root)
    for path in files:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode):
                raise PreparationError(f"non-regular file appeared during fsync: {path}")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    for path in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        _fsync_dir(path)


def _freeze_tree_permissions(root: Path) -> None:
    files, directories = _walk_regular_files(root)
    for path in files:
        os.chmod(path, 0o440, follow_symlinks=False)
    for path in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        os.chmod(path, 0o550, follow_symlinks=False)


def _require_frozen_tree(root: Path) -> None:
    files, directories = _walk_regular_files(root)
    for path in [*files, *directories]:
        if path.lstat().st_mode & 0o222:
            raise PreparationError(f"published tree entry is writable: {path}")


def _rename_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise PreparationError("libc renameat2 is unavailable; refusing non-atomic publication")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        AT_FDCWD,
        os.fsencode(source),
        AT_FDCWD,
        os.fsencode(destination),
        RENAME_NOREPLACE,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        raise PreparationError(
            f"atomic no-replace publication failed {source} -> {destination}: "
            f"{os.strerror(error_number)}"
        )


def _require_regular(path: Path, *, label: str) -> None:
    try:
        info = path.lstat()
    except OSError as error:
        raise PreparationError(f"{label} is unavailable: {path}: {error}") from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise PreparationError(f"{label} is not a non-symlink regular file: {path}")


def _require_directory(path: Path, *, label: str) -> None:
    try:
        info = path.lstat()
    except OSError as error:
        raise PreparationError(f"{label} is unavailable: {path}: {error}") from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise PreparationError(f"{label} is not a non-symlink directory: {path}")
    if path.resolve(strict=True) != path:
        raise PreparationError(f"{label} contains a symlink or alias: {path}")


def _under(path: Path, root: Path, *, label: str) -> None:
    try:
        path.relative_to(root)
    except ValueError as error:
        raise PreparationError(f"{label} escapes workspace: {path}") from error


def _git_head(reference_root: Path) -> str:
    try:
        result = subprocess.run(
            [
                "/usr/bin/git",
                "-c",
                f"safe.directory={reference_root}",
                "-C",
                str(reference_root),
                "rev-parse",
                "--verify",
                "HEAD^{commit}",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
            env={
                "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                "LANG": "C",
                "LC_ALL": "C",
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_TERMINAL_PROMPT": "0",
            },
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise PreparationError(f"cannot inspect reference commit: {error}") from error
    return result.stdout.strip()


def _load_official(reference_root: Path) -> Any:
    if _git_head(reference_root) != REFERENCE_COMMIT:
        raise PreparationError("Vision-OPD reference commit differs from the pinned commit")
    prepare_path = reference_root / "eval" / "prepare_data.py"
    snapshot = _stable_read_bytes(prepare_path, label="pinned prepare_data.py")
    if snapshot.sha256 != REFERENCE_PREPARE_SHA256:
        raise PreparationError(
            "prepare_data.py hash differs: "
            f"{snapshot.sha256} != {REFERENCE_PREPARE_SHA256}"
        )
    module_name = "vision_opd_pinned_prepare_data"
    module = types.ModuleType(module_name)
    module.__file__ = str(prepare_path)
    module.__package__ = ""
    try:
        code = compile(snapshot.data, str(prepare_path), "exec", dont_inherit=True)
        exec(code, module.__dict__)
    except Exception as error:
        raise PreparationError(f"cannot execute pinned prepare source bytes: {error}") from error
    return module


def _materialize_cached_snapshot(
    *, cache_root: Path, source: SourceSpec, destination: Path
) -> list[dict[str, Any]]:
    """Copy one exact HF snapshot into staging without retaining cache symlinks."""

    storage = cache_root / f"datasets--{source.repo_id.replace('/', '--')}"
    snapshot = storage / "snapshots" / source.revision
    blobs = storage / "blobs"
    _require_directory(cache_root, label="pinned Hugging Face cache root")
    _require_directory(storage, label=f"{source.benchmark} cache repository")
    _require_directory(snapshot, label=f"{source.benchmark} cached snapshot")
    _require_directory(blobs, label=f"{source.benchmark} cache blobs")
    if destination.exists() or destination.is_symlink():
        raise PreparationError(f"offline snapshot destination already exists: {destination}")
    os.mkdir(destination, 0o750)

    inventory: list[dict[str, Any]] = []
    pending: list[tuple[Path, Path]] = [(snapshot, destination)]
    while pending:
        source_dir, destination_dir = pending.pop()
        try:
            with os.scandir(source_dir) as iterator:
                entries = sorted(iterator, key=lambda item: item.name)
        except OSError as error:
            raise PreparationError(f"cannot enumerate cached snapshot {source_dir}: {error}") from error
        for entry in entries:
            source_path = Path(entry.path)
            relative = source_path.relative_to(snapshot)
            destination_path = destination / relative
            try:
                info = entry.stat(follow_symlinks=False)
            except OSError as error:
                raise PreparationError(f"cannot stat cached snapshot entry {source_path}: {error}") from error
            if stat.S_ISDIR(info.st_mode):
                os.mkdir(destination_path, 0o750)
                pending.append((source_path, destination_path))
                continue
            if stat.S_ISLNK(info.st_mode):
                try:
                    byte_source = source_path.resolve(strict=True)
                    byte_source.relative_to(blobs)
                except (OSError, ValueError) as error:
                    raise PreparationError(
                        f"cached snapshot symlink escapes its repository blobs: {source_path}"
                    ) from error
                snapshot_bytes = _stable_read_bytes(
                    byte_source, label=f"{source.benchmark} cached blob"
                )
            elif stat.S_ISREG(info.st_mode):
                snapshot_bytes = _stable_read_bytes(
                    source_path, label=f"{source.benchmark} cached snapshot file"
                )
            else:
                raise PreparationError(f"special cached snapshot entry is forbidden: {source_path}")
            _atomic_create(destination_path, snapshot_bytes.data, 0o440)
            inventory.append(
                {
                    "relative_path": relative.as_posix(),
                    "bytes": snapshot_bytes.size,
                    "sha256": snapshot_bytes.sha256,
                }
            )
    if not inventory:
        raise PreparationError(f"cached snapshot contains no files: {snapshot}")
    return sorted(inventory, key=lambda item: item["relative_path"])


def _pinned_downloader(
    real_snapshot_download: Callable[..., str],
    staging_root: Path,
    *,
    workspace_root: Path | None = None,
    events: list[dict[str, Any]] | None = None,
) -> Callable[..., str]:
    def download(repo_id: str, *args: Any, **kwargs: Any) -> str:
        source = SOURCE_BY_REPO.get(repo_id)
        if source is None:
            raise PreparationError(f"unpinned dataset repository requested: {repo_id}")
        if args:
            raise PreparationError(f"unexpected positional snapshot_download arguments for {repo_id}")
        if kwargs.get("repo_type") != "dataset":
            raise PreparationError(f"snapshot_download repo_type must be dataset: {repo_id}")
        requested_revision = kwargs.pop("revision", source.revision)
        if requested_revision != source.revision:
            raise PreparationError(
                f"dataset revision differs for {repo_id}: {requested_revision} != {source.revision}"
            )
        local_dir = Path(kwargs.get("local_dir", ""))
        expected_dir = staging_root / source.source_dir
        if not local_dir.is_absolute():
            local_dir = Path(os.path.abspath(local_dir))
        if local_dir != expected_dir:
            raise PreparationError(
                f"dataset local_dir differs for {repo_id}: {local_dir} != {expected_dir}"
            )
        kwargs["local_dir"] = str(expected_dir)
        kwargs["revision"] = source.revision
        pinned_cache_raw = os.environ.get("OPD_HF_PINNED_CACHE")
        if pinned_cache_raw:
            if os.environ.get("HF_HUB_OFFLINE") != "1":
                raise PreparationError("OPD_HF_PINNED_CACHE requires HF_HUB_OFFLINE=1")
            pinned_cache = Path(os.path.abspath(pinned_cache_raw))
            if workspace_root is None:
                raise PreparationError("offline pinned cache requires a workspace root")
            _under(pinned_cache, workspace_root, label="pinned Hugging Face cache")
            cache_files = _materialize_cached_snapshot(
                cache_root=pinned_cache,
                source=source,
                destination=expected_dir,
            )
            if events is not None:
                events.append(
                    {
                        "repo_id": source.repo_id,
                        "revision": source.revision,
                        "mode": "pinned_shared_cache_offline",
                        "cache_root": str(pinned_cache),
                        "cache_snapshot_relative": (
                            f"datasets--{source.repo_id.replace('/', '--')}/snapshots/"
                            f"{source.revision}"
                        ),
                        "cache_files": cache_files,
                    }
                )
            return str(expected_dir)
        result = real_snapshot_download(repo_id, **kwargs)
        if events is not None:
            events.append(
                {
                    "repo_id": source.repo_id,
                    "revision": source.revision,
                    "mode": "huggingface_hub_network",
                    "cache_root": None,
                    "cache_snapshot_relative": None,
                    "cache_files": [],
                }
            )
        return result

    return download


def _sample_uid(item: dict[str, Any], benchmark: str) -> str:
    for key in ("sample_uid", "uid", "index", "question_id", "id"):
        value = item.get(key)
        if value is not None and str(value) != "":
            return f"{benchmark}:{key}:{value}"
    stable = {
        "benchmark": benchmark,
        "images": item.get("images") or [],
        "query": item.get("query", ""),
    }
    raw = json.dumps(stable, ensure_ascii=False, sort_keys=True)
    return "sha1:" + hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _image_record_and_decode(path: Path, root: Path) -> dict[str, Any]:
    """Hash and fully decode the exact same immutable byte snapshot."""

    snapshot = _stable_read_bytes(path, label="benchmark image")
    if snapshot.size <= 0:
        raise PreparationError(f"image is empty: {path}")
    try:
        from PIL import Image, ImageFile

        ImageFile.LOAD_TRUNCATED_IMAGES = False
        with Image.open(io.BytesIO(snapshot.data)) as image:
            image.verify()
        with Image.open(io.BytesIO(snapshot.data)) as image:
            image.load()
            if image.width <= 0 or image.height <= 0:
                raise PreparationError(f"image has invalid dimensions: {path}")
    except PreparationError:
        raise
    except Exception as error:
        raise PreparationError(f"image failed full decode: {path}: {error}") from error
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "bytes": snapshot.size,
        "sha256": snapshot.sha256,
    }


def _rewrite_and_validate_rows(
    *,
    rows: Any,
    source: SourceSpec,
    staging_root: Path,
    final_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not isinstance(rows, list) or len(rows) != source.row_count:
        actual = len(rows) if isinstance(rows, list) else "non-list"
        raise PreparationError(
            f"{source.benchmark} row count differs: {actual} != {source.row_count}"
        )
    rewritten: list[dict[str, Any]] = []
    image_records: list[dict[str, Any]] = []
    seen_uids: set[str] = set()
    seen_images: set[Path] = set()
    expected_image_root = staging_root / source.image_dir
    _require_directory(expected_image_root, label=f"{source.benchmark} image directory")

    for row_number, original in enumerate(rows, start=1):
        if not isinstance(original, dict):
            raise PreparationError(f"{source.benchmark} row {row_number} is not an object")
        row = dict(original)
        query = row.get("query")
        response = row.get("response")
        if not isinstance(query, str) or not query.strip():
            raise PreparationError(f"{source.benchmark} row {row_number} has empty query")
        if not isinstance(response, str) or not response.strip():
            raise PreparationError(f"{source.benchmark} row {row_number} has empty response")
        images = row.get("images")
        if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], str):
            raise PreparationError(f"{source.benchmark} row {row_number} must have one image")
        image_path = Path(images[0])
        if not image_path.is_absolute():
            image_path = staging_root / image_path
        image_path = Path(os.path.abspath(image_path))
        try:
            relative = image_path.relative_to(staging_root)
        except ValueError as error:
            raise PreparationError(
                f"{source.benchmark} row {row_number} image escapes staging: {image_path}"
            ) from error
        if relative.parts[0] != source.image_dir:
            raise PreparationError(
                f"{source.benchmark} row {row_number} image is outside {source.image_dir}"
            )
        _require_regular(image_path, label=f"{source.benchmark} row {row_number} image")
        row["images"] = [str(final_root / relative)]
        uid = _sample_uid(row, source.benchmark)
        if uid in seen_uids:
            raise PreparationError(f"{source.benchmark} duplicate UID: {uid}")
        seen_uids.add(uid)
        if image_path in seen_images:
            raise PreparationError(f"{source.benchmark} duplicate image path: {image_path}")
        seen_images.add(image_path)
        image_record = _image_record_and_decode(image_path, staging_root)
        image_record["sample_uid"] = uid
        image_records.append(image_record)
        rewritten.append(row)

    actual_files, _ = _walk_regular_files(expected_image_root)
    actual_file_set = set(actual_files)
    if actual_file_set != seen_images:
        missing = sorted(str(path) for path in seen_images - actual_file_set)[:10]
        extra = sorted(str(path) for path in actual_file_set - seen_images)[:10]
        raise PreparationError(
            f"{source.benchmark} image inventory differs; missing={missing}, extra={extra}"
        )
    return rewritten, image_records


def _source_inventory(
    staging_root: Path, source: SourceSpec
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    root = staging_root / source.source_dir
    _require_directory(root, label=f"{source.benchmark} pinned source directory")
    files, _ = _walk_regular_files(root)
    parquets = [path for path in files if path.suffix == ".parquet"]
    if not parquets:
        raise PreparationError(f"{source.benchmark} source contains no parquet files")
    records = [_record(path, staging_root) for path in files]
    records_by_path = {record["relative_path"]: record for record in records}
    parquet_records = [records_by_path[path.relative_to(staging_root).as_posix()] for path in parquets]
    return records, parquet_records


def _seal(payload: dict[str, Any]) -> dict[str, Any]:
    envelope: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "vision_opd_reference_preparation",
        "status": "passed",
        "payload": payload,
    }
    envelope["seal_sha256"] = _sha256_bytes(_canonical_bytes(envelope))
    return envelope


def _validate_envelope(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PreparationError("preparation receipt is not an object")
    expected = {"schema_version", "kind", "status", "payload", "seal_sha256"}
    if set(value) != expected:
        raise PreparationError("preparation receipt envelope keys differ")
    if value["schema_version"] != SCHEMA_VERSION:
        raise PreparationError("preparation receipt schema differs")
    if value["kind"] != "vision_opd_reference_preparation" or value["status"] != "passed":
        raise PreparationError("preparation receipt kind/status differs")
    base = {key: value[key] for key in expected if key != "seal_sha256"}
    if value["seal_sha256"] != _sha256_bytes(_canonical_bytes(base)):
        raise PreparationError("preparation receipt seal is invalid")
    if not isinstance(value["payload"], dict):
        raise PreparationError("preparation receipt payload is not an object")
    return value


def _read_json_snapshot(path: Path, *, label: str) -> tuple[Any, StableBytes]:
    snapshot = _stable_read_bytes(path, label=label)
    try:
        value = json.loads(snapshot.data.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise PreparationError(f"{label} is not valid UTF-8 JSON: {path}: {error}") from error
    return value, snapshot


def _read_json(path: Path, *, label: str) -> Any:
    return _read_json_snapshot(path, label=label)[0]


def _record_matches(root: Path, record: Any, *, label: str) -> dict[str, Any]:
    required = {"relative_path", "bytes", "sha256"}
    if not isinstance(record, dict) or not required.issubset(set(record)):
        raise PreparationError(f"{label} record is malformed")
    relative = record.get("relative_path")
    if not isinstance(relative, str) or relative.startswith("/") or ".." in Path(relative).parts:
        raise PreparationError(f"{label} relative path is unsafe")
    path = root / relative
    _require_regular(path, label=label)
    current = _record(path, root)
    expected = {key: record[key] for key in ("relative_path", "bytes", "sha256")}
    if current != expected:
        raise PreparationError(f"{label} bytes changed: {relative}")
    return current


def _image_record_matches(root: Path, record: Any, *, label: str) -> None:
    required = {"relative_path", "bytes", "sha256", "sample_uid"}
    if not isinstance(record, dict) or not required.issubset(set(record)):
        raise PreparationError(f"{label} record is malformed")
    relative = record.get("relative_path")
    if not isinstance(relative, str) or relative.startswith("/") or ".." in Path(relative).parts:
        raise PreparationError(f"{label} relative path is unsafe")
    current = _image_record_and_decode(root / relative, root)
    expected = {key: record[key] for key in ("relative_path", "bytes", "sha256")}
    if current != expected:
        raise PreparationError(f"{label} bytes changed: {relative}")


def verify_published(data_root: Path) -> dict[str, Any]:
    _require_directory(data_root, label="published Vision-OPD data root")
    _require_frozen_tree(data_root)
    receipt_path = data_root / RECEIPT_RELATIVE
    raw_receipt, receipt_snapshot = _read_json_snapshot(
        receipt_path, label="preparation receipt"
    )
    value = _validate_envelope(raw_receipt)
    payload = value["payload"]
    if set(payload) != {
        "generated_at_utc",
        "attempt_id",
        "published_data_root",
        "reference",
        "acquisition",
        "tool",
        "datasets",
    }:
        raise PreparationError("preparation receipt payload fields differ")
    if payload.get("published_data_root") != str(data_root):
        raise PreparationError("preparation receipt published data root differs")
    if payload.get("reference") != {
        "commit": REFERENCE_COMMIT,
        "prepare_data_sha256": REFERENCE_PREPARE_SHA256,
    }:
        raise PreparationError("preparation receipt reference pin differs")
    acquisition = payload.get("acquisition")
    if not isinstance(acquisition, list) or len(acquisition) != len(SOURCES):
        raise PreparationError("preparation acquisition event count differs")
    acquisition_by_repo: dict[str, dict[str, Any]] = {}
    workspace_root = data_root.parents[2]
    for source, event in zip(SOURCES, acquisition, strict=True):
        if not isinstance(event, dict) or set(event) != {
            "repo_id",
            "revision",
            "mode",
            "cache_root",
            "cache_snapshot_relative",
            "cache_files",
        }:
            raise PreparationError(f"{source.benchmark} acquisition event is malformed")
        if event.get("repo_id") != source.repo_id or event.get("revision") != source.revision:
            raise PreparationError(f"{source.benchmark} acquisition identity differs")
        mode = event.get("mode")
        cache_files = event.get("cache_files")
        if mode == "huggingface_hub_network":
            if event.get("cache_root") is not None or event.get("cache_snapshot_relative") is not None:
                raise PreparationError(f"{source.benchmark} network acquisition cache fields differ")
            if cache_files != []:
                raise PreparationError(f"{source.benchmark} network cache inventory differs")
        elif mode == "pinned_shared_cache_offline":
            cache_root = event.get("cache_root")
            expected_relative = (
                f"datasets--{source.repo_id.replace('/', '--')}/snapshots/{source.revision}"
            )
            if not isinstance(cache_root, str) or not Path(cache_root).is_absolute():
                raise PreparationError(f"{source.benchmark} offline cache root is unsafe")
            _under(Path(cache_root), workspace_root, label=f"{source.benchmark} offline cache")
            if event.get("cache_snapshot_relative") != expected_relative:
                raise PreparationError(f"{source.benchmark} offline cache snapshot differs")
            if not isinstance(cache_files, list) or not cache_files:
                raise PreparationError(f"{source.benchmark} offline cache inventory is empty")
            seen_cache_paths: set[str] = set()
            for record in cache_files:
                required = {"relative_path", "bytes", "sha256"}
                if not isinstance(record, dict) or set(record) != required:
                    raise PreparationError(f"{source.benchmark} offline cache record is malformed")
                relative = record.get("relative_path")
                if (
                    not isinstance(relative, str)
                    or relative.startswith("/")
                    or ".." in Path(relative).parts
                    or relative in seen_cache_paths
                ):
                    raise PreparationError(f"{source.benchmark} offline cache path is unsafe")
                seen_cache_paths.add(relative)
        else:
            raise PreparationError(f"{source.benchmark} acquisition mode differs")
        acquisition_by_repo[source.repo_id] = event
    tool = payload.get("tool")
    if not isinstance(tool, dict) or set(tool) != {"entry_path", "entry_snapshot", "archived"}:
        raise PreparationError("preparation tool provenance is malformed")
    entry_snapshot = tool.get("entry_snapshot")
    if not isinstance(entry_snapshot, dict) or set(entry_snapshot) != {
        "bytes",
        "sha256",
        "device",
        "inode",
        "mode",
        "mtime_ns",
    }:
        raise PreparationError("preparation entry snapshot is malformed")
    archived = tool.get("archived")
    if not isinstance(archived, dict) or archived.get("relative_path") != ARCHIVED_TOOL_RELATIVE:
        raise PreparationError("archived preparation tool path differs")
    _record_matches(data_root, archived, label="archived preparation tool")
    if (archived.get("bytes"), archived.get("sha256")) != (
        entry_snapshot.get("bytes"),
        entry_snapshot.get("sha256"),
    ):
        raise PreparationError("archived preparation tool differs from entry snapshot")
    datasets = payload.get("datasets")
    if not isinstance(datasets, dict) or set(datasets) != {item.benchmark for item in SOURCES}:
        raise PreparationError("preparation receipt dataset set differs")
    expected_inventory = {RECEIPT_RELATIVE, ARCHIVED_TOOL_RELATIVE}
    for source in SOURCES:
        item = datasets[source.benchmark]
        if not isinstance(item, dict) or set(item) != {
            "repo_id",
            "revision",
            "row_count",
            "unique_uid_count",
            "image_count",
            "json",
            "source_files",
            "source_parquets",
            "images",
        }:
            raise PreparationError(f"{source.benchmark} dataset receipt fields differ")
        expected_identity = {
            "repo_id": source.repo_id,
            "revision": source.revision,
            "row_count": source.row_count,
            "unique_uid_count": source.row_count,
            "image_count": source.row_count,
        }
        for key, expected in expected_identity.items():
            if item.get(key) != expected:
                raise PreparationError(
                    f"{source.benchmark} preparation identity differs for {key}"
                )
        json_record = item.get("json")
        if not isinstance(json_record, dict) or json_record.get("relative_path") != source.json_name:
            raise PreparationError(f"{source.benchmark} JSON path differs")
        _record_matches(data_root, json_record, label=f"{source.benchmark} JSON")
        expected_inventory.add(item["json"]["relative_path"])
        source_files = item.get("source_files")
        source_records = item.get("source_parquets")
        image_records = item.get("images")
        if not isinstance(source_files, list) or not source_files:
            raise PreparationError(f"{source.benchmark} source file receipt is empty")
        if not isinstance(source_records, list) or not source_records:
            raise PreparationError(f"{source.benchmark} source parquet receipt is empty")
        if not isinstance(image_records, list) or len(image_records) != source.row_count:
            raise PreparationError(f"{source.benchmark} image receipt count differs")
        source_paths: set[str] = set()
        source_records_by_relative: dict[str, dict[str, Any]] = {}
        for record in source_files:
            _record_matches(data_root, record, label=f"{source.benchmark} source file")
            relative = record["relative_path"]
            if not relative.startswith(source.source_dir + "/"):
                raise PreparationError(f"{source.benchmark} source file path differs")
            if relative in source_paths:
                raise PreparationError(f"{source.benchmark} duplicate source file receipt")
            source_paths.add(relative)
            source_records_by_relative[relative] = record
            expected_inventory.add(relative)
        event = acquisition_by_repo[source.repo_id]
        if event["mode"] == "pinned_shared_cache_offline":
            materialized = {
                relative.removeprefix(source.source_dir + "/"): {
                    "relative_path": relative.removeprefix(source.source_dir + "/"),
                    "bytes": record["bytes"],
                    "sha256": record["sha256"],
                }
                for relative, record in source_records_by_relative.items()
            }
            cached = {record["relative_path"]: record for record in event["cache_files"]}
            if materialized != cached:
                raise PreparationError(
                    f"{source.benchmark} materialized source differs from cached snapshot"
                )
        parquet_paths: set[str] = set()
        for record in source_records:
            _record_matches(data_root, record, label=f"{source.benchmark} source parquet")
            relative = record["relative_path"]
            if not relative.endswith(".parquet") or relative not in source_paths:
                raise PreparationError(
                    f"{source.benchmark} source parquet is not in source inventory"
                )
            parquet_paths.add(relative)
        if len(parquet_paths) != len(source_records):
            raise PreparationError(f"{source.benchmark} duplicate source parquet receipt")
        image_paths: set[str] = set()
        for record in image_records:
            _image_record_matches(data_root, record, label=f"{source.benchmark} image")
            relative = record["relative_path"]
            if not relative.startswith(source.image_dir + "/"):
                raise PreparationError(f"{source.benchmark} image path differs")
            if relative in image_paths:
                raise PreparationError(f"{source.benchmark} duplicate image receipt")
            image_paths.add(relative)
            expected_inventory.add(relative)
        benchmark_rows, benchmark_snapshot = _read_json_snapshot(
            data_root / source.json_name, label=f"{source.benchmark} JSON"
        )
        if (benchmark_snapshot.size, benchmark_snapshot.sha256) != (
            json_record.get("bytes"),
            json_record.get("sha256"),
        ):
            raise PreparationError(f"{source.benchmark} JSON changed while verifying")
        if not isinstance(benchmark_rows, list) or len(benchmark_rows) != source.row_count:
            raise PreparationError(f"{source.benchmark} published row count differs")
        uids = []
        for row_number, (row, image_record) in enumerate(
            zip(benchmark_rows, image_records, strict=True), start=1
        ):
            if not isinstance(row, dict):
                raise PreparationError(
                    f"{source.benchmark} published row {row_number} is not an object"
                )
            query = row.get("query")
            response = row.get("response")
            images = row.get("images")
            if not isinstance(query, str) or not query.strip():
                raise PreparationError(
                    f"{source.benchmark} published row {row_number} has empty query"
                )
            if not isinstance(response, str) or not response.strip():
                raise PreparationError(
                    f"{source.benchmark} published row {row_number} has empty response"
                )
            expected_image = str(data_root / image_record["relative_path"])
            if images != [expected_image]:
                raise PreparationError(
                    f"{source.benchmark} published row {row_number} image binding differs"
                )
            uid = _sample_uid(row, source.benchmark)
            if image_record.get("sample_uid") != uid:
                raise PreparationError(
                    f"{source.benchmark} published row {row_number} UID/image binding differs"
                )
            uids.append(uid)
        if len(set(uids)) != source.row_count:
            raise PreparationError(f"{source.benchmark} published UIDs are not unique")
    actual_files, _ = _walk_regular_files(data_root)
    actual_inventory = {path.relative_to(data_root).as_posix() for path in actual_files}
    if actual_inventory != expected_inventory:
        missing = sorted(expected_inventory - actual_inventory)[:10]
        extra = sorted(actual_inventory - expected_inventory)[:10]
        raise PreparationError(
            f"published file inventory differs; missing={missing}, extra={extra}"
        )
    final_receipt = _stable_read_bytes(receipt_path, label="preparation receipt postflight")
    if not _same_snapshot(receipt_snapshot, final_receipt):
        raise PreparationError("preparation receipt changed during verification")
    return {
        "status": "reverified",
        "data_root": str(data_root),
        "receipt": str(receipt_path),
        "receipt_bytes": receipt_snapshot.size,
        "receipt_sha256": receipt_snapshot.sha256,
        "seal_sha256": value["seal_sha256"],
    }


def _lock(lock_path: Path) -> int:
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as error:
        raise PreparationError(f"cannot open preparation lock {lock_path}: {error}") from error
    try:
        info = os.fstat(descriptor)
        path_info = lock_path.lstat()
        if not stat.S_ISREG(info.st_mode) or (info.st_dev, info.st_ino) != (
            path_info.st_dev,
            path_info.st_ino,
        ):
            raise PreparationError("preparation lock path/descriptor identity differs")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise PreparationError(f"another official preparation owns {lock_path}") from error
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _entry_snapshot_payload(snapshot: StableBytes) -> dict[str, Any]:
    return {
        "bytes": snapshot.size,
        "sha256": snapshot.sha256,
        "device": snapshot.device,
        "inode": snapshot.inode,
        "mode": snapshot.mode,
        "mtime_ns": snapshot.mtime_ns,
    }


def _same_snapshot(left: StableBytes, right: StableBytes) -> bool:
    return _entry_snapshot_payload(left) == _entry_snapshot_payload(right) and left.data == right.data


def _reject_stale_staging(data_root: Path) -> None:
    prefix = f".{data_root.name}.staging."
    try:
        stale = sorted(path for path in data_root.parent.iterdir() if path.name.startswith(prefix))
    except OSError as error:
        raise PreparationError(f"cannot inspect stale staging roots: {error}") from error
    if stale:
        rendered = ", ".join(str(path) for path in stale[:10])
        raise PreparationError(
            "preserved staging evidence blocks a new attempt; inspect and explicitly "
            f"quarantine it first: {rendered}"
        )


def _expected_inventory_from_datasets(datasets: dict[str, Any]) -> set[str]:
    expected = {RECEIPT_RELATIVE, ARCHIVED_TOOL_RELATIVE}
    for source in SOURCES:
        item = datasets[source.benchmark]
        expected.add(item["json"]["relative_path"])
        expected.update(record["relative_path"] for record in item["source_files"])
        expected.update(record["relative_path"] for record in item["images"])
    return expected


def _verify_staging_records(staging_root: Path, datasets: dict[str, Any]) -> None:
    """Re-read every recorded byte immediately before durable publication."""

    for source in SOURCES:
        item = datasets[source.benchmark]
        _record_matches(staging_root, item["json"], label=f"{source.benchmark} staged JSON")
        for record in item["source_files"]:
            _record_matches(
                staging_root, record, label=f"{source.benchmark} staged source file"
            )
        for record in item["images"]:
            _image_record_matches(
                staging_root, record, label=f"{source.benchmark} staged image"
            )


def prepare(
    *,
    workspace_root: Path,
    reference_root: Path,
    data_root: Path,
    lock_path: Path,
    entry_tool_snapshot: StableBytes | None = None,
) -> dict[str, Any]:
    _require_directory(workspace_root, label="workspace root")
    _require_directory(reference_root, label="Vision-OPD reference root")
    _require_directory(data_root.parent, label="evaluation dataset parent")
    _require_directory(lock_path.parent, label="workspace lock directory")
    for path, label in (
        (reference_root, "reference root"),
        (data_root, "data root"),
        (lock_path, "lock path"),
    ):
        _under(path, workspace_root, label=label)

    lock_descriptor = _lock(lock_path)
    try:
        if data_root.exists() or data_root.is_symlink():
            return verify_published(data_root)

        _reject_stale_staging(data_root)

        tool_path = Path(os.path.abspath(__file__))
        current_entry = _stable_read_bytes(tool_path, label="preparation entry tool")
        if entry_tool_snapshot is None:
            entry_tool_snapshot = current_entry
        elif not _same_snapshot(entry_tool_snapshot, current_entry):
            raise PreparationError("preparation tool changed between process entry and execution")

        now = datetime.now(timezone.utc)
        attempt_id = now.strftime("%Y%m%dT%H%M%SZ") + f"-{os.getpid()}"
        staging_root = data_root.parent / f".{data_root.name}.staging.{attempt_id}"
        try:
            os.mkdir(staging_root, 0o750)
        except OSError as error:
            raise PreparationError(f"cannot create fresh staging root {staging_root}: {error}") from error

        published = False
        try:
            official = _load_official(reference_root)
            acquisition_events: list[dict[str, Any]] = []
            official.snapshot_download = _pinned_downloader(
                official.snapshot_download,
                staging_root,
                workspace_root=workspace_root,
                events=acquisition_events,
            )
            raw_rows = {
                "vstar": official.prepare_vstar(staging_root),
                "mme-realworld-lite": official.prepare_mme_realworld_lite(staging_root),
            }

            datasets: dict[str, Any] = {}
            for source in SOURCES:
                rows, image_records = _rewrite_and_validate_rows(
                    rows=raw_rows[source.benchmark],
                    source=source,
                    staging_root=staging_root,
                    final_root=data_root,
                )
                json_path = staging_root / source.json_name
                _atomic_create(json_path, _pretty_bytes(rows), 0o440)
                source_files, source_parquets = _source_inventory(staging_root, source)
                datasets[source.benchmark] = {
                    "repo_id": source.repo_id,
                    "revision": source.revision,
                    "row_count": source.row_count,
                    "unique_uid_count": source.row_count,
                    "image_count": source.row_count,
                    "json": _record(json_path, staging_root),
                    "source_files": source_files,
                    "source_parquets": source_parquets,
                    "images": image_records,
                }

            current_before_archive = _stable_read_bytes(
                tool_path, label="preparation entry tool before archive"
            )
            if not _same_snapshot(entry_tool_snapshot, current_before_archive):
                raise PreparationError("preparation tool changed before provenance archive")
            provenance_dir = staging_root / "provenance"
            os.mkdir(provenance_dir, 0o750)
            archived_tool_path = staging_root / ARCHIVED_TOOL_RELATIVE
            _atomic_create(archived_tool_path, entry_tool_snapshot.data, 0o440)
            receipt = _seal(
                {
                    "generated_at_utc": now.isoformat().replace("+00:00", "Z"),
                    "attempt_id": attempt_id,
                    "published_data_root": str(data_root),
                    "reference": {
                        "commit": REFERENCE_COMMIT,
                        "prepare_data_sha256": REFERENCE_PREPARE_SHA256,
                    },
                    "acquisition": acquisition_events,
                    "tool": {
                        "entry_path": str(tool_path),
                        "entry_snapshot": _entry_snapshot_payload(entry_tool_snapshot),
                        "archived": _record(archived_tool_path, staging_root),
                    },
                    "datasets": datasets,
                }
            )
            receipt_path = staging_root / RECEIPT_RELATIVE
            _atomic_create(receipt_path, _pretty_bytes(receipt), 0o440)
            actual_files, _ = _walk_regular_files(staging_root)
            actual_inventory = {
                path.relative_to(staging_root).as_posix() for path in actual_files
            }
            expected_inventory = _expected_inventory_from_datasets(datasets)
            if actual_inventory != expected_inventory:
                missing = sorted(expected_inventory - actual_inventory)[:10]
                extra = sorted(actual_inventory - expected_inventory)[:10]
                raise PreparationError(
                    f"prepublication file inventory differs; missing={missing}, extra={extra}"
                )
            current_before_publish = _stable_read_bytes(
                tool_path, label="preparation entry tool before publication"
            )
            if not _same_snapshot(entry_tool_snapshot, current_before_publish):
                raise PreparationError("preparation tool changed before publication")
            _verify_staging_records(staging_root, datasets)
            _freeze_tree_permissions(staging_root)
            _durably_fsync_tree(staging_root)
            post_fsync_files, _ = _walk_regular_files(staging_root)
            post_fsync_inventory = {
                path.relative_to(staging_root).as_posix() for path in post_fsync_files
            }
            if post_fsync_inventory != expected_inventory:
                raise PreparationError("prepared inventory changed during durable fsync")
            _rename_noreplace(staging_root, data_root)
            published = True
            try:
                _fsync_dir(data_root.parent)
            except Exception as error:
                raise PreparationError(
                    "atomic rename succeeded but parent fsync failed; destination may be "
                    f"published and must be reverified before retry: {data_root}: {error}"
                ) from error
        except Exception as error:
            if published:
                if isinstance(error, PreparationError):
                    raise
                raise PreparationError(
                    f"post-publication failure; reverify destination {data_root}: {error}"
                ) from error
            if isinstance(error, PreparationError):
                raise PreparationError(f"{error}; preserved staging: {staging_root}") from error
            raise PreparationError(
                f"official preparation failed: {error}; preserved staging: {staging_root}"
            ) from error

        result = verify_published(data_root)
        result["status"] = "created"
        return result
    finally:
        try:
            fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
        finally:
            os.close(lock_descriptor)


def _parser() -> argparse.ArgumentParser:
    script = Path(__file__).resolve()
    workspace = script.parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=workspace)
    parser.add_argument(
        "--reference-root", type=Path, default=workspace / "Codes" / "Vision-OPD-reference"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=workspace / "Dataset" / "eval" / "vision_opd_reference_c8a8fdd",
    )
    parser.add_argument(
        "--lock-path",
        type=Path,
        default=workspace / "Locks" / "vision_opd_reference_prepare.lock",
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--execute", action="store_true")
    action.add_argument("--verify-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    workspace = Path(os.path.abspath(args.workspace_root))
    reference = Path(os.path.abspath(args.reference_root))
    data_root = Path(os.path.abspath(args.data_root))
    lock_path = Path(os.path.abspath(args.lock_path))
    if not args.execute and not args.verify_only:
        plan = {
            "mode": "dry-run",
            "side_effects": "none",
            "workspace_root": str(workspace),
            "reference_root": str(reference),
            "data_root": str(data_root),
            "lock_path": str(lock_path),
            "reference_commit": REFERENCE_COMMIT,
            "reference_prepare_sha256": REFERENCE_PREPARE_SHA256,
            "datasets": [
                {
                    "benchmark": source.benchmark,
                    "repo_id": source.repo_id,
                    "revision": source.revision,
                    "expected_rows": source.row_count,
                }
                for source in SOURCES
            ],
            "publication": "fresh sibling staging + renameat2(RENAME_NOREPLACE)",
        }
        print(json.dumps(plan, ensure_ascii=False, sort_keys=True, indent=2))
        return 0
    if args.verify_only:
        try:
            result = verify_published(data_root)
        except PreparationError as error:
            print(f"ERROR: {error}", file=sys.stderr)
            return 1
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
        return 0
    if os.geteuid() != EXPECTED_UID or os.getegid() != EXPECTED_UID:
        print(
            f"ERROR: execute requires uid:gid {EXPECTED_UID}:{EXPECTED_UID}; "
            f"got {os.geteuid()}:{os.getegid()}",
            file=sys.stderr,
        )
        return 2
    try:
        entry_tool_snapshot = _stable_read_bytes(
            Path(os.path.abspath(__file__)), label="preparation process entry tool"
        )
        result = prepare(
            workspace_root=workspace,
            reference_root=reference,
            data_root=data_root,
            lock_path=lock_path,
            entry_tool_snapshot=entry_tool_snapshot,
        )
    except PreparationError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
