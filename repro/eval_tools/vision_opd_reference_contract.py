#!/usr/bin/env python3
"""Create-once contracts for the pinned official Vision-OPD evaluator.

The program never downloads data, starts a model, calls an API, repairs an
answer file, or edits an upstream artifact.  It validates immutable inputs,
publishes deterministic sealed receipts, and exposes fail-closed artifact
staging/publication primitives for the pinned lifecycle wrapper.

Four subcommands form the evaluation lineage chain::

    data -> run -> answers -> judge

The additional ``model``, ``judge-runtime``, and ``matrix`` subcommands freeze
a standalone judge inventory, a recoverable serving plan, and the final
cross-target judge matrix respectively.  Artifact helper subcommands validate
paths and partial checkpoints, hardlink/copy immutable inputs, and publish via
Linux ``renameat2(RENAME_NOREPLACE)``.

All artifact paths must stay below ``--workspace-root``.  Symlinks are
rejected (including symlinks in path ancestors).  Every receipt is fsynced as
a sibling candidate before no-replace publication.  Re-running a command is
allowed only when the recomputed receipt is byte-for-byte identical; differing
candidates are retained as conflict evidence.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
import re
import secrets
import stat
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlsplit


SCHEMA_VERSION = "1.0"
WORKSPACE = Path(__file__).resolve().parents[3]
DEFAULT_REFERENCE_ROOT = WORKSPACE / "Codes" / "Vision-OPD-reference"
REFERENCE_COMMIT = "c8a8fdd1f88eef1b5ef4fe6a8d64eb0272917471"
REFERENCE_FILES = {
    "eval/prepare_data.py": "8e71bb3f04c741434ab505acfc4d2b6107cefec2864cdf909b2ff0e4fad79c5a",
    "eval/infer.py": "bb379999932658907196cdc98d22c60d63e3308cb5a867317481c4a85af70374",
    "eval/judge_qwenlm.py": "abbe11dacf7fae19728ca16407a02c91d04a9bc8ea72edd3b4a91b6224f4b670",
    "eval/cal_acc.py": "695dbddc3e63a1b9f8971c0d414d963a5da94776863d58589feaa4a1c6b0f025",
}
PREPARER_SHA256 = "b765dd7f9d06397e4606d9f40af0468c02704821083cca25a02cbaf779a05d2f"
PREPARATION_SCHEMA_VERSION = "vision_opd_reference_preparation_v2"
PREPARATION_RECEIPT_NAME = "preparation_receipt.json"
PREPARATION_SOURCES = {
    "vstar": {
        "repo_id": "lmms-lab/vstar-bench",
        "revision": "b44023b4dca749ed8a76b85eb576627d05a1c174",
        "row_count": 191,
    },
    "mme-realworld-lite": {
        "repo_id": "yifanzhang114/MME-RealWorld-lite-lmms-eval",
        "revision": "f6b0dc81ba4d3c39bd9b4e544578198d365ac084",
        "row_count": 1919,
    },
}
BENCHMARKS = {
    "vstar": {"json_name": "vstar.json"},
    "mme-realworld-lite": {"json_name": "MME_RealWorld_Lite.json"},
}
RESERVED_BENCHMARK_KEYS = frozenset(
    {"model_answer", "extracted_answer", "judge", "judge_source"}
)
ANSWER_ADDED_KEYS = frozenset({"sample_uid", "model_answer"})
JUDGE_ADDED_KEYS = frozenset({"extracted_answer", "judge", "judge_source"})
ANSWER_ERROR_PREFIXES = ("[api_error]", "[future_error]")
OFFICIAL_SEED = 42
OFFICIAL_SEED_LABEL = "seed42"
OFFICIAL_MAX_TOKENS = 32768
OFFICIAL_JUDGE_MAX_TOKENS = 2048
OFFICIAL_JUDGE_SOURCES = frozenset({"mathruler", "first letter", "llm"})
OFFICIAL_JUDGE_DATA_PARALLEL_SIZE = 1
OFFICIAL_JUDGE_MAX_MODEL_LEN = 65536
OFFICIAL_JUDGE_MAX_NUM_SEQS = 32
OFFICIAL_JUDGE_GPU_MEMORY_UTILIZATION = 0.85
OFFICIAL_JUDGE_MM_IMAGE_LIMIT = 16
JUDGE_CONTEXT_GATE_KIND = "vision_opd_reference_judge_context_gate"
JUDGE_CONTEXT_GATE_DISPATCH_ORDER = ["mathruler", "first letter", "llm"]
JUDGE_CONTEXT_GATE_PROMPT_TEMPLATE_SHA256 = (
    "f5c6097ef2082324b1d5bc7ffc407898181ef33965bdf19ff490dbbf006b1cb5"
)
JUDGE_CONTEXT_GATE_TOKENIZER_ASSETS = frozenset(
    {
        "chat_template.jinja",
        "config.json",
        "merges.txt",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    }
)
REQUIRED_JUDGE_RUNTIME_SOURCES = frozenset(
    {
        "serve",
        "reference_eval_runner",
        "contract_helper",
        "runtime_auditor",
        "hardened_judge_driver",
        "judge_context_gate",
        "judge_matrix_lifecycle",
        "official_judge",
    }
)
DETAIL_LIMIT = 20
AT_FDCWD = -100
RENAME_NOREPLACE = 1


class ContractError(RuntimeError):
    """Fail-closed validation or publication error."""


class ReceiptConflict(ContractError):
    """A create-once path already contains different bytes."""


def _absolute(path: Path | str) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )


def _pretty_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_sha256(value: Any) -> str:
    return _sha256_bytes(_canonical_bytes(value))


def _safe_root(path: Path | str, *, label: str) -> Path:
    root = _absolute(path)
    try:
        resolved = root.resolve(strict=True)
        info = root.lstat()
    except OSError as error:
        raise ContractError(f"{label} is unavailable: {root}: {error}") from error
    if resolved != root or stat.S_ISLNK(info.st_mode):
        raise ContractError(f"{label} contains a symlink or alias: {root}")
    if not stat.S_ISDIR(info.st_mode):
        raise ContractError(f"{label} is not a directory: {root}")
    return root


def _within(path: Path, root: Path, *, label: str) -> None:
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ContractError(f"{label} escapes its trusted root: {path} not below {root}") from error


def _safe_existing(
    path: Path | str,
    root: Path,
    *,
    label: str,
    kind: str,
) -> Path:
    candidate = _absolute(path)
    _within(candidate, root, label=label)
    try:
        info = candidate.lstat()
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise ContractError(f"{label} is unavailable: {candidate}: {error}") from error
    if resolved != candidate or stat.S_ISLNK(info.st_mode):
        raise ContractError(f"{label} contains a symlink or alias: {candidate}")
    if kind == "file" and not stat.S_ISREG(info.st_mode):
        raise ContractError(f"{label} is not a regular file: {candidate}")
    if kind == "dir" and not stat.S_ISDIR(info.st_mode):
        raise ContractError(f"{label} is not a directory: {candidate}")
    return candidate


def _make_safe_parent(path: Path | str, workspace: Path) -> Path:
    candidate = _absolute(path)
    _within(candidate, workspace, label="receipt path")
    if candidate == workspace:
        raise ContractError("receipt path cannot be the workspace root")
    relative_parent = candidate.parent.relative_to(workspace)
    cursor = workspace
    for component in relative_parent.parts:
        cursor = cursor / component
        try:
            os.mkdir(cursor, 0o750)
        except FileExistsError:
            pass
        _safe_existing(cursor, workspace, label="receipt parent", kind="dir")
    if candidate.is_symlink():
        raise ContractError(f"receipt path is a symlink: {candidate}")
    return candidate


def _make_safe_directory(path: Path | str, workspace: Path) -> Path:
    """Create one directory chain without accepting aliases or special files."""

    candidate = _absolute(path)
    _within(candidate, workspace, label="output directory")
    if candidate == workspace:
        return workspace
    relative = candidate.relative_to(workspace)
    cursor = workspace
    for component in relative.parts:
        cursor = cursor / component
        try:
            os.mkdir(cursor, 0o750)
        except FileExistsError:
            pass
        _safe_existing(cursor, workspace, label="output directory", kind="dir")
    return candidate


def _gate_output_leaf(
    path: Path | str,
    workspace: Path,
    *,
    create_parent: bool,
) -> Path:
    """Reject a leaf or existing ancestor that is a symlink/special file."""

    candidate = _absolute(path)
    _within(candidate, workspace, label="output leaf")
    if candidate == workspace:
        raise ContractError("output leaf cannot be the workspace root")
    if create_parent:
        _make_safe_directory(candidate.parent, workspace)
    else:
        _safe_existing(candidate.parent, workspace, label="output parent", kind="dir")
    try:
        info = candidate.lstat()
    except FileNotFoundError:
        return candidate
    except OSError as error:
        raise ContractError(f"could not inspect output leaf {candidate}: {error}") from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ContractError(f"output leaf is a symlink or special file: {candidate}")
    if candidate.resolve(strict=True) != candidate:
        raise ContractError(f"output leaf contains a symlink or alias: {candidate}")
    return candidate


def gate_artifact_paths(
    *,
    workspace: Path,
    directories: Sequence[Path | str],
    leaves: Sequence[Path | str],
    create: bool,
) -> dict[str, Any]:
    checked_directories: list[str] = []
    checked_leaves: list[str] = []
    for path in directories:
        directory = (
            _make_safe_directory(path, workspace)
            if create
            else _safe_existing(path, workspace, label="output directory", kind="dir")
        )
        checked_directories.append(str(directory))
    for path in leaves:
        checked_leaves.append(
            str(_gate_output_leaf(path, workspace, create_parent=create))
        )
    return {
        "directories": checked_directories,
        "leaves": checked_leaves,
        "create": create,
    }


def _stream_file(path: Path) -> tuple[str, int]:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        before = path.lstat()
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ContractError(f"could not open regular file {path}: {error}") from error
    digest = hashlib.sha256()
    size = 0
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or (before.st_dev, before.st_ino) != (
            opened.st_dev,
            opened.st_ino,
        ):
            raise ContractError(f"file changed identity while opening: {path}")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    stable_fields = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns")
    if any(getattr(opened, field) != getattr(after, field) for field in stable_fields):
        raise ContractError(f"file changed while hashing: {path}")
    if size != opened.st_size:
        raise ContractError(f"short read while hashing: {path}")
    return digest.hexdigest(), size


def _read_bytes(path: Path) -> tuple[bytes, str]:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        before = path.lstat()
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ContractError(f"could not open {path}: {error}") from error
    chunks: list[bytes] = []
    digest = hashlib.sha256()
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or (before.st_dev, before.st_ino) != (
            opened.st_dev,
            opened.st_ino,
        ):
            raise ContractError(f"file changed identity while opening: {path}")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
            digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    stable_fields = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns")
    if any(getattr(opened, field) != getattr(after, field) for field in stable_fields):
        raise ContractError(f"file changed while reading: {path}")
    data = b"".join(chunks)
    if len(data) != opened.st_size:
        raise ContractError(f"short read: {path}")
    return data, digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    digest, size = _stream_file(path)
    return {"path": str(path), "bytes": size, "sha256": digest}


def _load_json(path: Path, *, label: str) -> tuple[Any, dict[str, Any]]:
    raw, digest = _read_bytes(path)
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ContractError(f"{label} is not valid UTF-8 JSON: {path}: {error}") from error
    return payload, {"path": str(path), "bytes": len(raw), "sha256": digest}


def _seal(kind: str, payload: dict[str, Any]) -> dict[str, Any]:
    envelope: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": kind,
        "status": "passed",
        "payload": payload,
    }
    envelope["seal_sha256"] = _canonical_sha256(envelope)
    return envelope


def _validate_envelope(value: Any, *, expected_kind: str, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{label} must be a JSON object")
    expected_keys = {"schema_version", "kind", "status", "payload", "seal_sha256"}
    if set(value) != expected_keys:
        raise ContractError(f"{label} has unexpected envelope keys")
    if value["schema_version"] != SCHEMA_VERSION:
        raise ContractError(f"{label} schema version differs")
    if value["kind"] != expected_kind or value["status"] != "passed":
        raise ContractError(f"{label} kind/status differs")
    if not isinstance(value["payload"], dict):
        raise ContractError(f"{label} payload is not an object")
    base = {key: value[key] for key in expected_keys if key != "seal_sha256"}
    if value["seal_sha256"] != _canonical_sha256(base):
        raise ContractError(f"{label} seal is invalid")
    return value


def _load_receipt(
    path: Path | str,
    workspace: Path,
    *,
    expected_kind: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    receipt = _safe_existing(path, workspace, label=expected_kind, kind="file")
    value, record = _load_json(receipt, label=expected_kind)
    return _validate_envelope(value, expected_kind=expected_kind, label=expected_kind), record


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as error:
        raise ContractError(f"could not fsync artifact directory {path}: {error}") from error


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish with Linux renameat2(RENAME_NOREPLACE)."""

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise ContractError("libc renameat2 is unavailable; refusing non-atomic publication")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        AT_FDCWD,
        os.fsencode(source),
        AT_FDCWD,
        os.fsencode(destination),
        RENAME_NOREPLACE,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileExistsError(error_number, os.strerror(error_number), destination)
    raise ContractError(
        f"renameat2(RENAME_NOREPLACE) failed for {source} -> {destination}: "
        f"{os.strerror(error_number)}"
    )


def _candidate_path(target: Path) -> Path:
    return target.parent / (
        f".{target.name}.candidate.{os.getpid()}.{secrets.token_hex(8)}"
    )


def _write_candidate(target: Path, payload: bytes, *, mode: int) -> Path:
    candidate = _candidate_path(target)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(candidate, flags, mode)
    except OSError as error:
        raise ContractError(f"could not create publication candidate {candidate}: {error}") from error
    complete = False
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise ContractError(f"short write while staging {candidate}")
            offset += written
        os.fsync(descriptor)
        complete = True
    finally:
        os.close(descriptor)
    if not complete:
        raise ContractError(f"publication candidate is incomplete and retained: {candidate}")
    _fsync_directory(candidate.parent)
    return candidate


def _publish_candidate(
    *,
    candidate: Path,
    target: Path,
    workspace: Path,
    payload: bytes,
    seal_sha256: str | None = None,
) -> dict[str, Any]:
    candidate = _safe_existing(
        candidate, workspace, label="publication candidate", kind="file"
    )
    target = _gate_output_leaf(target, workspace, create_parent=False)
    if candidate.parent != target.parent:
        raise ContractError("publication candidate must be a sibling of its target")
    proposed_sha = _sha256_bytes(payload)
    try:
        _rename_noreplace(candidate, target)
    except FileExistsError:
        existing = _safe_existing(target, workspace, label="existing artifact", kind="file")
        current, current_sha = _read_bytes(existing)
        if current != payload:
            raise ReceiptConflict(
                f"no-replace publication conflict at {target}: "
                f"existing_sha256={current_sha}, proposed_sha256={proposed_sha}, "
                f"evidence_path={candidate}"
            )
        try:
            candidate.unlink()
        except OSError as error:
            raise ContractError(
                f"identical candidate could not be removed {candidate}: {error}"
            ) from error
        _fsync_directory(target.parent)
        result = {
            "publication": "reverified",
            "path": str(target),
            "bytes": len(payload),
            "sha256": current_sha,
        }
    else:
        _fsync_directory(target.parent)
        result = {
            "publication": "created",
            "path": str(target),
            "bytes": len(payload),
            "sha256": proposed_sha,
        }
    if seal_sha256 is not None:
        result["seal_sha256"] = seal_sha256
    return result


def _publish_create_once(
    path: Path | str,
    workspace: Path,
    envelope: dict[str, Any],
) -> dict[str, Any]:
    target = _make_safe_parent(path, workspace)
    payload = _pretty_bytes(envelope)
    candidate = _write_candidate(target, payload, mode=0o440)
    return _publish_candidate(
        candidate=candidate,
        target=target,
        workspace=workspace,
        payload=payload,
        seal_sha256=envelope["seal_sha256"],
    )


def stage_file(
    *, workspace: Path, source: Path | str, destination: Path | str
) -> dict[str, Any]:
    source_path = _safe_existing(source, workspace, label="stage source", kind="file")
    destination_path = _make_safe_parent(destination, workspace)
    if destination_path.exists() or destination_path.is_symlink():
        raise ContractError(f"stage destination already exists: {destination_path}")
    payload, source_sha = _read_bytes(source_path)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    method = "hardlink"
    try:
        os.link(source_path, destination_path, follow_symlinks=False)
        descriptor = os.open(destination_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as link_error:
        method = "copy"
        try:
            descriptor = os.open(destination_path, flags, 0o400)
        except OSError as error:
            raise ContractError(
                f"could not stage {source_path} at {destination_path}: "
                f"hardlink={link_error}; copy={error}"
            ) from error
        try:
            offset = 0
            while offset < len(payload):
                written = os.write(descriptor, payload[offset:])
                if written <= 0:
                    raise ContractError(f"short write while staging {destination_path}")
                offset += written
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    _fsync_directory(destination_path.parent)
    staged, staged_sha = _read_bytes(destination_path)
    if staged != payload or staged_sha != source_sha:
        raise ContractError(f"staged input differs from source: {destination_path}")
    return {
        "method": method,
        "source": str(source_path),
        "destination": str(destination_path),
        "bytes": len(payload),
        "sha256": source_sha,
    }


def publish_file(
    *, workspace: Path, source: Path | str, destination: Path | str
) -> dict[str, Any]:
    source_path = _safe_existing(source, workspace, label="publish source", kind="file")
    target = _make_safe_parent(destination, workspace)
    payload, _ = _read_bytes(source_path)
    candidate = _write_candidate(target, payload, mode=0o440)
    return _publish_candidate(
        candidate=candidate,
        target=target,
        workspace=workspace,
        payload=payload,
    )


def validate_reference_sources(reference_root: Path | str) -> dict[str, Any]:
    root = _safe_root(reference_root, label="Vision-OPD reference root")
    try:
        completed = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={root}",
                "-C",
                str(root),
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
        raise ContractError(f"could not inspect Vision-OPD reference commit: {error}") from error
    commit = completed.stdout.strip()
    if commit != REFERENCE_COMMIT:
        raise ContractError(f"Vision-OPD reference commit differs: {commit} != {REFERENCE_COMMIT}")
    files: list[dict[str, Any]] = []
    for relative, expected_sha in sorted(REFERENCE_FILES.items()):
        source = _safe_existing(root / relative, root, label=f"reference source {relative}", kind="file")
        record = _file_record(source)
        record["relative_path"] = relative
        if record["sha256"] != expected_sha:
            raise ContractError(
                f"Vision-OPD reference source hash differs for {relative}: "
                f"{record['sha256']} != {expected_sha}"
            )
        files.append(record)
    return {"root": str(root), "commit": commit, "files": files}


def validate_preparation_binding(
    *,
    workspace: Path,
    data_root: Path | str,
    preparation_receipt: Path | str,
    preparer_path: Path | str,
) -> dict[str, Any]:
    """Bind the schema-v2 preparer and its exact published receipt.

    The lifecycle invokes the pinned preparer's exhaustive ``--verify-only``
    pass immediately before this contract.  This validator independently
    freezes the tool bytes, receipt bytes/seal, both dataset revisions/counts,
    and the archived-tool identity so downstream receipts cannot be mixed with
    another preparation.
    """

    root = _safe_existing(data_root, workspace, label="prepared data root", kind="dir")
    receipt = _safe_existing(
        preparation_receipt, root, label="preparation receipt", kind="file"
    )
    if receipt != root / PREPARATION_RECEIPT_NAME:
        raise ContractError(
            f"preparation receipt must be {root / PREPARATION_RECEIPT_NAME}"
        )
    preparer = _safe_existing(
        preparer_path, workspace, label="pinned preparation tool", kind="file"
    )
    preparer_record = _file_record(preparer)
    if preparer_record["sha256"] != PREPARER_SHA256:
        raise ContractError(
            "preparation tool SHA256 differs: "
            f"{preparer_record['sha256']} != {PREPARER_SHA256}"
        )
    value, receipt_record = _load_json(receipt, label="preparation receipt")
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "kind",
        "status",
        "payload",
        "seal_sha256",
    }:
        raise ContractError("preparation receipt envelope fields differ")
    if value.get("schema_version") != PREPARATION_SCHEMA_VERSION:
        raise ContractError("preparation receipt schema version differs")
    if value.get("kind") != "vision_opd_reference_preparation" or value.get(
        "status"
    ) != "passed":
        raise ContractError("preparation receipt kind/status differs")
    base = {
        key: value[key]
        for key in ("schema_version", "kind", "status", "payload")
    }
    if value.get("seal_sha256") != _canonical_sha256(base):
        raise ContractError("preparation receipt seal is invalid")
    payload = value.get("payload")
    if not isinstance(payload, dict) or payload.get("published_data_root") != str(root):
        raise ContractError("preparation receipt data-root binding differs")
    if payload.get("reference") != {
        "commit": REFERENCE_COMMIT,
        "prepare_data_sha256": REFERENCE_FILES["eval/prepare_data.py"],
    }:
        raise ContractError("preparation receipt reference-source binding differs")
    tool = payload.get("tool")
    if not isinstance(tool, dict):
        raise ContractError("preparation receipt tool provenance is malformed")
    snapshot = tool.get("entry_snapshot")
    archived = tool.get("archived")
    if not isinstance(snapshot, dict) or not isinstance(archived, dict):
        raise ContractError("preparation receipt tool snapshots are malformed")
    if tool.get("entry_path") != str(preparer) or archived.get(
        "relative_path"
    ) != "provenance/prepare_vision_opd_reference_data.py":
        raise ContractError("preparation receipt tool path provenance differs")
    if snapshot.get("sha256") != PREPARER_SHA256 or archived.get(
        "sha256"
    ) != PREPARER_SHA256:
        raise ContractError("preparation receipt is not bound to the pinned preparer")
    if snapshot.get("bytes") != preparer_record["bytes"] or archived.get(
        "bytes"
    ) != preparer_record["bytes"]:
        raise ContractError("preparation receipt preparer byte count differs")
    datasets = payload.get("datasets")
    if not isinstance(datasets, dict) or set(datasets) != set(PREPARATION_SOURCES):
        raise ContractError("preparation receipt dataset set differs")
    identities: list[dict[str, Any]] = []
    for benchmark, expected in sorted(PREPARATION_SOURCES.items()):
        item = datasets.get(benchmark)
        if not isinstance(item, dict):
            raise ContractError(f"preparation dataset is malformed: {benchmark}")
        identity = {
            "benchmark": benchmark,
            "repo_id": item.get("repo_id"),
            "revision": item.get("revision"),
            "row_count": item.get("row_count"),
        }
        if {key: identity[key] for key in ("repo_id", "revision", "row_count")} != expected:
            raise ContractError(f"preparation dataset identity differs: {benchmark}")
        if item.get("unique_uid_count") != expected["row_count"] or item.get(
            "image_count"
        ) != expected["row_count"]:
            raise ContractError(f"preparation dataset counts differ: {benchmark}")
        identities.append(identity)
    return {
        "preparer": preparer_record,
        "receipt": {
            **receipt_record,
            "seal_sha256": value["seal_sha256"],
            "schema_version": value["schema_version"],
        },
        "datasets": identities,
    }


def make_sample_uid(item: dict[str, Any], benchmark: str) -> str:
    """Mirror ``eval/infer.py::make_sample_uid`` at the pinned commit."""
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


def _benchmark_identity(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if key not in ANSWER_ADDED_KEYS}


def _load_benchmark_rows(
    *,
    benchmark: str,
    benchmark_json: Path | str,
    data_root: Path | str,
    workspace: Path,
    hash_images: bool,
) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    if benchmark not in BENCHMARKS:
        raise ContractError(f"unsupported benchmark: {benchmark}")
    root = _safe_existing(data_root, workspace, label="data root", kind="dir")
    json_path = _safe_existing(benchmark_json, root, label="benchmark JSON", kind="file")
    expected_name = BENCHMARKS[benchmark]["json_name"]
    if json_path.name != expected_name:
        raise ContractError(f"benchmark JSON must be named {expected_name}, got {json_path.name}")
    value, json_record = _load_json(json_path, label="benchmark JSON")
    if not isinstance(value, list) or not value:
        raise ContractError("benchmark JSON must contain a non-empty list")

    rows: list[dict[str, Any]] = []
    uids: list[str] = []
    image_bindings: list[dict[str, Any]] = []
    unique_image_paths: dict[str, Path] = {}
    for row_number, item in enumerate(value, start=1):
        if not isinstance(item, dict):
            raise ContractError(f"benchmark row {row_number} is not an object")
        reserved = sorted(RESERVED_BENCHMARK_KEYS.intersection(item))
        if reserved:
            raise ContractError(f"benchmark row {row_number} contains reserved keys: {reserved}")
        query = item.get("query")
        response = item.get("response")
        if not isinstance(query, str) or not query.strip():
            raise ContractError(f"benchmark row {row_number} has an empty/non-string query")
        if not isinstance(response, str) or not response.strip():
            raise ContractError(f"benchmark row {row_number} has an empty/non-string response")
        images = item.get("images")
        if not isinstance(images, list) or not images or not isinstance(images[0], str) or not images[0].strip():
            raise ContractError(f"benchmark row {row_number} has no valid first image")
        requested = Path(images[0])
        if not requested.is_absolute():
            requested = json_path.parent / requested
        image_path = _safe_existing(requested, root, label=f"row {row_number} first image", kind="file")
        uid = make_sample_uid(item, benchmark)
        if not uid.strip():
            raise ContractError(f"benchmark row {row_number} has an empty UID")
        rows.append(item)
        uids.append(uid)
        unique_image_paths[str(image_path)] = image_path
        image_bindings.append({"sample_uid": uid, "path": str(image_path)})

    duplicates = sorted(uid for uid, count in Counter(uids).items() if count > 1)
    if duplicates:
        raise ContractError(f"benchmark UIDs are not unique: {duplicates[:DETAIL_LIMIT]}")

    images: list[dict[str, Any]] = []
    image_sha_by_path: dict[str, str] = {}
    for image_path in sorted(unique_image_paths.values(), key=lambda item: str(item)):
        if hash_images:
            record = _file_record(image_path)
            if record["bytes"] <= 0:
                raise ContractError(f"first image is empty: {image_path}")
        else:
            info = image_path.stat()
            if info.st_size <= 0:
                raise ContractError(f"first image is empty: {image_path}")
            record = {"path": str(image_path), "bytes": info.st_size}
        record["relative_path"] = image_path.relative_to(root).as_posix()
        images.append(record)
        if "sha256" in record:
            image_sha_by_path[str(image_path)] = record["sha256"]
    if hash_images:
        for binding in image_bindings:
            binding["sha256"] = image_sha_by_path[binding["path"]]

    summary = {
        "name": benchmark,
        "data_root": str(root),
        "json": json_record,
        "row_count": len(rows),
        "uids": uids,
        "uids_sha256": _canonical_sha256(uids),
        "unique_uid_count": len(uids),
        "first_image_count": len(image_bindings),
        "unique_first_image_count": len(images),
        "first_images": images,
        "first_image_bindings": image_bindings,
        "first_image_bindings_sha256": _canonical_sha256(image_bindings),
    }
    return rows, uids, summary


def build_data_payload(
    *,
    workspace: Path,
    reference_root: Path | str,
    benchmark: str,
    benchmark_json: Path | str,
    data_root: Path | str,
    preparation_receipt: Path | str,
    preparer_path: Path | str,
) -> dict[str, Any]:
    preparation = validate_preparation_binding(
        workspace=workspace,
        data_root=data_root,
        preparation_receipt=preparation_receipt,
        preparer_path=preparer_path,
    )
    _, _, benchmark_summary = _load_benchmark_rows(
        benchmark=benchmark,
        benchmark_json=benchmark_json,
        data_root=data_root,
        workspace=workspace,
        hash_images=True,
    )
    expected_count = PREPARATION_SOURCES[benchmark]["row_count"]
    if benchmark_summary["row_count"] != expected_count:
        raise ContractError(
            f"prepared benchmark row count differs: "
            f"{benchmark_summary['row_count']} != {expected_count}"
        )
    return {
        "benchmark": benchmark_summary,
        "preparation": preparation,
        "reference": validate_reference_sources(reference_root),
    }


def _model_inventory(model_path: Path | str, workspace: Path) -> dict[str, Any]:
    root = _safe_existing(model_path, workspace, label="model directory", kind="dir")
    files: list[dict[str, Any]] = []
    directories: list[str] = ["."]

    def walk(directory: Path, relative: Path) -> None:
        before = directory.stat(follow_symlinks=False)
        try:
            entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
        except OSError as error:
            raise ContractError(f"could not scan model directory {directory}: {error}") from error
        names_before = [entry.name for entry in entries]
        for entry in entries:
            path = directory / entry.name
            try:
                info = entry.stat(follow_symlinks=False)
            except OSError as error:
                raise ContractError(f"could not stat model entry {path}: {error}") from error
            if stat.S_ISLNK(info.st_mode):
                raise ContractError(f"model inventory contains a symlink: {path}")
            child_relative = relative / entry.name
            if stat.S_ISDIR(info.st_mode):
                directories.append(child_relative.as_posix())
                walk(path, child_relative)
            elif stat.S_ISREG(info.st_mode):
                safe_path = _safe_existing(path, root, label="model file", kind="file")
                record = _file_record(safe_path)
                record["relative_path"] = child_relative.as_posix()
                del record["path"]
                files.append(record)
            else:
                raise ContractError(f"model inventory contains a special file: {path}")
        try:
            names_after = sorted(entry.name for entry in os.scandir(directory))
            after = directory.stat(follow_symlinks=False)
        except OSError as error:
            raise ContractError(f"model directory changed while scanning {directory}: {error}") from error
        stable = ("st_dev", "st_ino", "st_mode", "st_mtime_ns")
        if names_before != names_after or any(getattr(before, key) != getattr(after, key) for key in stable):
            raise ContractError(f"model directory changed while inventorying: {directory}")

    walk(root, Path())
    if not files:
        raise ContractError(f"model directory contains no regular files: {root}")
    inventory_body = {"directories": directories, "files": files}
    return {
        "path": str(root),
        "directory_count": len(directories),
        "file_count": len(files),
        "total_bytes": sum(item["bytes"] for item in files),
        **inventory_body,
        "inventory_sha256": _canonical_sha256(inventory_body),
    }


def _validate_reference_binding(payload: dict[str, Any], current: dict[str, Any], *, label: str) -> None:
    if payload.get("reference") != current:
        raise ContractError(f"{label} reference-source binding differs")


def _load_data_lineage(
    path: Path | str,
    workspace: Path,
    reference: dict[str, Any],
    *,
    rehash_images: bool,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[str]]:
    receipt, record = _load_receipt(path, workspace, expected_kind="vision_opd_reference_data")
    payload = receipt["payload"]
    _validate_reference_binding(payload, reference, label="data receipt")
    benchmark = payload.get("benchmark")
    if not isinstance(benchmark, dict):
        raise ContractError("data receipt has no benchmark binding")
    preparation = payload.get("preparation")
    if not isinstance(preparation, dict):
        raise ContractError("data receipt has no preparation binding")
    receipt_binding = preparation.get("receipt")
    preparer_binding = preparation.get("preparer")
    if not isinstance(receipt_binding, dict) or not isinstance(preparer_binding, dict):
        raise ContractError("data receipt preparation binding is malformed")
    current_preparation = validate_preparation_binding(
        workspace=workspace,
        data_root=benchmark.get("data_root", ""),
        preparation_receipt=receipt_binding.get("path", ""),
        preparer_path=preparer_binding.get("path", ""),
    )
    if current_preparation != preparation:
        raise ContractError("data receipt preparation binding differs")
    name = benchmark.get("name")
    data_root = benchmark.get("data_root")
    json_binding = benchmark.get("json")
    if name not in BENCHMARKS or not isinstance(data_root, str) or not isinstance(json_binding, dict):
        raise ContractError("data receipt benchmark binding is malformed")
    rows, uids, current_summary = _load_benchmark_rows(
        benchmark=name,
        benchmark_json=json_binding.get("path", ""),
        data_root=data_root,
        workspace=workspace,
        hash_images=rehash_images,
    )
    if rehash_images:
        if current_summary != benchmark:
            raise ContractError("data receipt no longer matches benchmark/media bytes")
    else:
        if current_summary["json"] != json_binding:
            raise ContractError("benchmark JSON no longer matches data receipt")
        for key in ("name", "data_root", "row_count", "uids", "uids_sha256", "unique_uid_count"):
            if current_summary.get(key) != benchmark.get(key):
                raise ContractError(f"benchmark metadata differs from data receipt: {key}")
    return receipt, record, rows, uids


def _clean_text(value: str, *, label: str, pattern: str | None = None) -> str:
    if not isinstance(value, str) or not value.strip() or any(ord(char) < 32 for char in value):
        raise ContractError(f"{label} is empty or contains control characters")
    value = value.strip()
    if pattern is not None and re.fullmatch(pattern, value) is None:
        raise ContractError(f"{label} has an unsafe format: {value!r}")
    return value


def build_run_payload(
    *,
    workspace: Path,
    reference_root: Path | str,
    data_receipt: Path | str,
    model_path: Path | str,
    model_id: str,
    model_tag: str,
    seed: int,
    seed_label: str,
    enable_thinking: bool,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    reference = validate_reference_sources(reference_root)
    data_doc, data_record, _, _ = _load_data_lineage(
        data_receipt, workspace, reference, rehash_images=True
    )
    benchmark = data_doc["payload"]["benchmark"]["name"]
    model_id = _clean_text(model_id, label="model ID")
    model_tag = _clean_text(model_tag, label="model tag", pattern=r"[A-Za-z0-9][A-Za-z0-9._-]{0,191}")
    if seed != OFFICIAL_SEED or seed_label != OFFICIAL_SEED_LABEL:
        raise ContractError(f"official seed contract requires {OFFICIAL_SEED_LABEL} ({OFFICIAL_SEED})")
    if not model_tag.endswith(f"_{seed_label}"):
        raise ContractError(f"model tag must end with _{seed_label}")
    if enable_thinking:
        raise ContractError("official Vision-OPD contract requires non-thinking inference")
    if temperature != 0:
        raise ContractError("official Vision-OPD contract requires temperature=0")
    if max_tokens != OFFICIAL_MAX_TOKENS:
        raise ContractError(f"official Vision-OPD contract requires max_tokens={OFFICIAL_MAX_TOKENS}")
    return {
        "benchmark": benchmark,
        "data_receipt": {
            **data_record,
            "seal_sha256": data_doc["seal_sha256"],
        },
        "model": {
            "id": model_id,
            "tag": model_tag,
            "inventory": _model_inventory(model_path, workspace),
        },
        "inference": {
            "enable_thinking": False,
            "temperature": 0,
            "max_tokens": OFFICIAL_MAX_TOKENS,
            "seed": OFFICIAL_SEED,
            "seed_label": OFFICIAL_SEED_LABEL,
        },
        "reference": reference,
    }


def build_model_payload(
    *,
    workspace: Path,
    reference_root: Path | str,
    model_path: Path | str,
    model_id: str,
    model_tag: str,
) -> dict[str, Any]:
    """Freeze a standalone full model inventory for a shared local judge.

    Unlike a ``run`` receipt this has no benchmark or inference binding.  It is
    used by the judge-matrix lifecycle to prove that every target was judged by
    one create-once set of local weights.
    """
    reference = validate_reference_sources(reference_root)
    model_id = _clean_text(model_id, label="model ID")
    model_tag = _clean_text(
        model_tag, label="model tag", pattern=r"[A-Za-z0-9][A-Za-z0-9._-]{0,191}"
    )
    return {
        "model": {
            "id": model_id,
            "tag": model_tag,
            "inventory": _model_inventory(model_path, workspace),
        },
        "reference": reference,
    }


def _receipt_binding(document: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    return {**record, "seal_sha256": document["seal_sha256"]}


def _load_model_lineage(
    path: Path | str,
    workspace: Path,
    reference: dict[str, Any],
    *,
    verify_inventory: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    receipt, record = _load_receipt(
        path, workspace, expected_kind="vision_opd_reference_model"
    )
    payload = receipt["payload"]
    _validate_reference_binding(payload, reference, label="judge model receipt")
    model = payload.get("model")
    if not isinstance(model, dict) or set(model) != {"id", "tag", "inventory"}:
        raise ContractError("judge model receipt is malformed")
    model_id = _clean_text(model.get("id", ""), label="judge model ID")
    model_tag = _clean_text(
        model.get("tag", ""),
        label="judge model tag",
        pattern=r"[A-Za-z0-9][A-Za-z0-9._-]{0,191}",
    )
    inventory = model.get("inventory")
    if not isinstance(inventory, dict):
        raise ContractError("judge model receipt has no inventory")
    required_inventory = {
        "path",
        "directory_count",
        "file_count",
        "total_bytes",
        "directories",
        "files",
        "inventory_sha256",
    }
    if set(inventory) != required_inventory:
        raise ContractError("judge model inventory has unexpected fields")
    if verify_inventory and _model_inventory(inventory.get("path", ""), workspace) != inventory:
        raise ContractError("judge model directory no longer matches its receipt")
    normalized = {"id": model_id, "tag": model_tag, "inventory": inventory}
    if model != normalized:
        raise ContractError("judge model receipt identity is not normalized")
    return receipt, record, normalized


def _load_run_lineage(
    path: Path | str,
    workspace: Path,
    reference: dict[str, Any],
    *,
    verify_model: bool,
    rehash_images: bool,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[str]]:
    receipt, record = _load_receipt(path, workspace, expected_kind="vision_opd_reference_run")
    payload = receipt["payload"]
    _validate_reference_binding(payload, reference, label="run contract")
    data_binding = payload.get("data_receipt")
    model = payload.get("model")
    inference = payload.get("inference")
    if not isinstance(data_binding, dict) or not isinstance(model, dict) or not isinstance(inference, dict):
        raise ContractError("run contract is malformed")
    data_doc, data_record, rows, uids = _load_data_lineage(
        data_binding.get("path", ""), workspace, reference, rehash_images=rehash_images
    )
    expected_data_binding = {**data_record, "seal_sha256": data_doc["seal_sha256"]}
    if data_binding != expected_data_binding:
        raise ContractError("run contract data-receipt binding differs")
    if payload.get("benchmark") != data_doc["payload"]["benchmark"]["name"]:
        raise ContractError("run contract benchmark differs from data receipt")
    expected_inference = {
        "enable_thinking": False,
        "temperature": 0,
        "max_tokens": OFFICIAL_MAX_TOKENS,
        "seed": OFFICIAL_SEED,
        "seed_label": OFFICIAL_SEED_LABEL,
    }
    if inference != expected_inference:
        raise ContractError("run contract inference settings differ from official settings")
    if verify_model:
        inventory = model.get("inventory")
        if not isinstance(inventory, dict):
            raise ContractError("run contract model inventory is absent")
        if _model_inventory(inventory.get("path", ""), workspace) != inventory:
            raise ContractError("model directory no longer matches run contract")
    return receipt, record, rows, uids


def _inspect_answers(
    path: Path | str,
    workspace: Path,
    benchmark: str,
    rows: Sequence[dict[str, Any]],
    uids: Sequence[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    answer_path = _safe_existing(path, workspace, label="official infer JSONL", kind="file")
    raw, digest = _read_bytes(answer_path)
    try:
        text = raw.decode("utf-8")
    except UnicodeError as error:
        raise ContractError(f"answer JSONL is not UTF-8: {error}") from error
    physical_lines = text.splitlines()
    if len(physical_lines) != len(rows):
        raise ContractError(f"answer JSONL line count differs: {len(physical_lines)} != {len(rows)}")
    expected = {uid: _benchmark_identity(row) for uid, row in zip(uids, rows)}
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(physical_lines, start=1):
        if not line.strip():
            raise ContractError(f"answer JSONL line {line_number} is blank")
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ContractError(f"answer JSONL line {line_number} is invalid: {error}") from error
        if not isinstance(record, dict):
            raise ContractError(f"answer JSONL line {line_number} is not an object")
        uid = record.get("sample_uid")
        if not isinstance(uid, str) or not uid.strip():
            raise ContractError(f"answer JSONL line {line_number} has no string sample_uid")
        if uid in seen:
            raise ContractError(f"answer JSONL has duplicate UID: {uid}")
        seen.add(uid)
        if uid not in expected:
            raise ContractError(f"answer JSONL has unknown UID: {uid}")
        if _benchmark_identity(record) != expected[uid]:
            raise ContractError(f"answer JSONL benchmark payload differs for UID: {uid}")
        expected_keys = set(expected[uid]).union(ANSWER_ADDED_KEYS)
        if set(record) != expected_keys:
            raise ContractError(f"answer JSONL keys differ for UID: {uid}")
        answer = record.get("model_answer")
        if not isinstance(answer, str) or not answer.strip():
            raise ContractError(f"answer JSONL has empty model_answer for UID: {uid}")
        normalized = answer.lstrip().casefold()
        if normalized.startswith(ANSWER_ERROR_PREFIXES):
            raise ContractError(f"answer JSONL contains an API/future error for UID: {uid}")
        records.append(record)
    missing = [uid for uid in uids if uid not in seen]
    if missing:
        raise ContractError(f"answer JSONL is missing UIDs: {missing[:DETAIL_LIMIT]}")
    return records, {
        "path": str(answer_path),
        "bytes": len(raw),
        "sha256": digest,
        "physical_line_count": len(physical_lines),
        "json_object_count": len(records),
        "unique_uid_count": len(seen),
        "uid_order_sha256": _canonical_sha256([record["sample_uid"] for record in records]),
    }


def inspect_partial_answers(
    *,
    workspace: Path,
    reference_root: Path | str,
    run_contract: Path | str,
    answer_jsonl: Path | str,
    model_id: str,
    model_tag: str,
) -> dict[str, Any]:
    """Validate an upstream-resumable JSONL without repairing any byte.

    A partial checkpoint is accepted only when every physical line is valid
    JSON, every UID is unique and belongs to the exact frozen benchmark, and
    every record is byte-semantically the official benchmark row plus exactly
    ``sample_uid`` and a string ``model_answer``.  Empty/error answers remain
    acceptable here because the pinned upstream inferencer explicitly retries
    them; a final answers receipt still rejects them.
    """

    reference = validate_reference_sources(reference_root)
    run_doc, run_record, rows, uids = _load_run_lineage(
        run_contract,
        workspace,
        reference,
        verify_model=True,
        rehash_images=True,
    )
    model = run_doc["payload"].get("model")
    if not isinstance(model, dict):
        raise ContractError("run contract has no model binding")
    model_id = _clean_text(model_id, label="partial answer model ID")
    model_tag = _clean_text(
        model_tag,
        label="partial answer model tag",
        pattern=r"[A-Za-z0-9][A-Za-z0-9._-]{0,191}",
    )
    if model.get("id") != model_id or model.get("tag") != model_tag:
        raise ContractError("partial answer model identity differs from run contract")
    benchmark = run_doc["payload"].get("benchmark")
    answer_path = _safe_existing(
        answer_jsonl, workspace, label="partial answer JSONL", kind="file"
    )
    expected_name = f"{model_tag}_answer.jsonl"
    if answer_path.name != expected_name or answer_path.parent.name != benchmark:
        raise ContractError(
            "partial answer path does not match frozen benchmark/model tag"
        )
    raw, digest = _read_bytes(answer_path)
    try:
        text = raw.decode("utf-8")
    except UnicodeError as error:
        raise ContractError(f"partial answer JSONL is not UTF-8: {error}") from error
    lines = text.splitlines()
    expected = {uid: _benchmark_identity(row) for uid, row in zip(uids, rows)}
    seen: set[str] = set()
    retryable = 0
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise ContractError(f"partial answer JSONL line {line_number} is blank")
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ContractError(
                f"partial answer JSONL line {line_number} is invalid: {error}"
            ) from error
        if not isinstance(record, dict):
            raise ContractError(f"partial answer JSONL line {line_number} is not an object")
        uid = record.get("sample_uid")
        if not isinstance(uid, str) or not uid.strip():
            raise ContractError(
                f"partial answer JSONL line {line_number} has no string sample_uid"
            )
        if uid in seen:
            raise ContractError(f"partial answer JSONL has duplicate UID: {uid}")
        if uid not in expected:
            raise ContractError(f"partial answer JSONL has unknown UID: {uid}")
        seen.add(uid)
        if _benchmark_identity(record) != expected[uid]:
            raise ContractError(f"partial answer benchmark payload differs for UID: {uid}")
        if set(record) != set(expected[uid]).union(ANSWER_ADDED_KEYS):
            raise ContractError(f"partial answer JSONL keys differ for UID: {uid}")
        answer = record.get("model_answer")
        if not isinstance(answer, str):
            raise ContractError(f"partial answer model_answer is not a string for UID: {uid}")
        normalized = answer.lstrip().casefold()
        if not answer.strip() or normalized.startswith(ANSWER_ERROR_PREFIXES):
            retryable += 1
    if not set(seen).issubset(expected):
        raise ContractError("partial answer UIDs are not a subset of the frozen benchmark")
    return {
        "benchmark": benchmark,
        "model": {"id": model_id, "tag": model_tag},
        "run_contract": _receipt_binding(run_doc, run_record),
        "answer_jsonl": {
            "path": str(answer_path),
            "bytes": len(raw),
            "sha256": digest,
            "physical_line_count": len(lines),
            "unique_uid_count": len(seen),
            "expected_uid_count": len(uids),
            "retryable_record_count": retryable,
            "uid_order_sha256": _canonical_sha256(
                [json.loads(line)["sample_uid"] for line in lines]
            ),
        },
    }


def build_answers_payload(
    *,
    workspace: Path,
    reference_root: Path | str,
    run_contract: Path | str,
    answer_jsonl: Path | str,
) -> dict[str, Any]:
    reference = validate_reference_sources(reference_root)
    run_doc, run_record, rows, uids = _load_run_lineage(
        run_contract,
        workspace,
        reference,
        verify_model=True,
        rehash_images=True,
    )
    _, answer_record = _inspect_answers(
        answer_jsonl,
        workspace,
        run_doc["payload"]["benchmark"],
        rows,
        uids,
    )
    return {
        "benchmark": run_doc["payload"]["benchmark"],
        "run_contract": {**run_record, "seal_sha256": run_doc["seal_sha256"]},
        "answer_jsonl": answer_record,
        "model": {
            "id": run_doc["payload"]["model"]["id"],
            "tag": run_doc["payload"]["model"]["tag"],
        },
        "reference": reference,
    }


def _load_answers_lineage(
    path: Path | str,
    workspace: Path,
    reference: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    receipt, record = _load_receipt(path, workspace, expected_kind="vision_opd_reference_answers")
    payload = receipt["payload"]
    _validate_reference_binding(payload, reference, label="answers receipt")
    run_binding = payload.get("run_contract")
    answer_binding = payload.get("answer_jsonl")
    if not isinstance(run_binding, dict) or not isinstance(answer_binding, dict):
        raise ContractError("answers receipt is malformed")
    run_doc, run_record, rows, uids = _load_run_lineage(
        run_binding.get("path", ""),
        workspace,
        reference,
        verify_model=False,
        rehash_images=False,
    )
    if run_binding != {**run_record, "seal_sha256": run_doc["seal_sha256"]}:
        raise ContractError("answers receipt run-contract binding differs")
    records, current_answer = _inspect_answers(
        answer_binding.get("path", ""),
        workspace,
        run_doc["payload"]["benchmark"],
        rows,
        uids,
    )
    if current_answer != answer_binding:
        raise ContractError("answer JSONL no longer matches answers receipt")
    return receipt, record, records


def _validate_api_base(value: str) -> str:
    value = _clean_text(value, label="judge API base")
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ContractError("judge API base must be an absolute http(s) URL")
    if parsed.username is not None or parsed.password is not None:
        raise ContractError("judge API base must not contain credentials")
    if parsed.query or parsed.fragment:
        raise ContractError("judge API base must not contain query/fragment data")
    return value.rstrip("/")


def _parse_cuda_devices(value: str) -> list[int]:
    value = _clean_text(value, label="CUDA devices")
    parts = value.split(",")
    if any(re.fullmatch(r"[0-7]", item) is None for item in parts):
        raise ContractError("CUDA devices must be a comma-separated subset of physical GPUs 0-7")
    devices = [int(item) for item in parts]
    if len(set(devices)) != len(devices):
        raise ContractError("CUDA devices contain duplicates")
    return devices


def _normalize_source_specs(
    specs: Sequence[Sequence[str]], workspace: Path
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for spec in specs:
        if len(spec) != 2:
            raise ContractError("runtime source spec must contain NAME PATH")
        name = _clean_text(
            spec[0], label="runtime source name", pattern=r"[a-z][a-z0-9_]{0,63}"
        )
        if name in seen:
            raise ContractError(f"duplicate runtime source name: {name}")
        seen.add(name)
        source = _safe_existing(spec[1], workspace, label=f"runtime source {name}", kind="file")
        records.append({"name": name, **_file_record(source)})
    if seen != REQUIRED_JUDGE_RUNTIME_SOURCES:
        raise ContractError(
            "judge runtime sources differ: "
            f"expected={sorted(REQUIRED_JUDGE_RUNTIME_SOURCES)} observed={sorted(seen)}"
        )
    return sorted(records, key=lambda item: item["name"])


def _normalize_expected_pairs(
    specs: Sequence[Sequence[str]], workspace: Path
) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str]] = set()
    seen_paths: set[str] = set()
    for spec in specs:
        if len(spec) != 3:
            raise ContractError("expected pair spec must contain LABEL BENCHMARK JUDGE_RECEIPT")
        label = _clean_text(
            spec[0], label="matrix target label", pattern=r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}"
        )
        benchmark = spec[1]
        if benchmark not in BENCHMARKS:
            raise ContractError(f"unsupported matrix benchmark: {benchmark}")
        key = (label, benchmark)
        if key in seen_keys:
            raise ContractError(f"duplicate expected matrix pair: {label}/{benchmark}")
        seen_keys.add(key)
        receipt = _absolute(spec[2])
        _within(receipt, workspace, label="planned judge receipt")
        _safe_existing(receipt.parent, workspace, label="planned judge receipt parent", kind="dir")
        if receipt.is_symlink():
            raise ContractError(f"planned judge receipt is a symlink: {receipt}")
        if receipt.exists():
            _safe_existing(receipt, workspace, label="planned judge receipt", kind="file")
        receipt_text = str(receipt)
        if receipt_text in seen_paths:
            raise ContractError(f"duplicate planned judge receipt path: {receipt}")
        seen_paths.add(receipt_text)
        pairs.append({"label": label, "benchmark": benchmark, "judge_receipt": receipt_text})
    if not pairs:
        raise ContractError("judge runtime requires at least one expected pair")
    return sorted(pairs, key=lambda item: (item["label"], item["benchmark"]))


def _revalidate_gate_file_record(
    value: Any,
    workspace: Path,
    *,
    label: str,
    expected_relative_path: str | None = None,
    trusted_root: Path | None = None,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{label} file record is malformed")
    expected_keys = {"path", "bytes", "sha256"}
    if expected_relative_path is not None:
        expected_keys.add("relative_path")
    if set(value) != expected_keys:
        raise ContractError(f"{label} file record fields differ")
    if expected_relative_path is not None and value.get("relative_path") != expected_relative_path:
        raise ContractError(f"{label} relative path differs")
    path = _safe_existing(
        value.get("path", ""), trusted_root or workspace, label=label, kind="file"
    )
    current = _file_record(path)
    if expected_relative_path is not None:
        current["relative_path"] = expected_relative_path
    if value != current:
        raise ContractError(f"{label} bytes no longer match context gate")
    return current


def _load_judge_context_gate_lineage(
    path: Path | str,
    workspace: Path,
    reference: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    receipt, record = _load_receipt(path, workspace, expected_kind=JUDGE_CONTEXT_GATE_KIND)
    payload = receipt["payload"]
    _validate_reference_binding(payload, reference, label="judge context gate")
    expected_top = {
        "protocol",
        "judge_model",
        "tokenizer",
        "sources",
        "software",
        "pairs",
        "aggregate",
        "reference",
    }
    if set(payload) != expected_top:
        raise ContractError("judge context gate payload fields differ")

    expected_protocol = {
        "dispatch_order": JUDGE_CONTEXT_GATE_DISPATCH_ORDER,
        "prompt_template_sha256": JUDGE_CONTEXT_GATE_PROMPT_TEMPLATE_SHA256,
        "chat_messages": [{"role": "user", "content": "<PROMPT_TEMPLATE>"}],
        "apply_chat_template": {
            "tokenize": True,
            "add_generation_prompt": True,
            "enable_thinking": False,
            "truncation": False,
        },
        "judge_max_output_tokens": OFFICIAL_JUDGE_MAX_TOKENS,
        "judge_max_model_len": OFFICIAL_JUDGE_MAX_MODEL_LEN,
        "acceptance": "prompt_tokens + judge_max_output_tokens <= judge_max_model_len",
        "no_truncation": True,
    }
    protocol = payload.get("protocol")
    if not isinstance(protocol, dict):
        raise ContractError("judge context gate protocol is malformed")
    if protocol != expected_protocol:
        raise ContractError("judge context gate protocol differs from ctx65k policy")

    judge_model = payload.get("judge_model")
    if not isinstance(judge_model, dict) or set(judge_model) != {
        "contract",
        "id",
        "tag",
        "inventory",
    }:
        raise ContractError("judge context gate model binding is malformed")
    model_binding = judge_model["contract"]
    if not isinstance(model_binding, dict):
        raise ContractError("judge context gate has no model receipt binding")
    model_doc, model_record, model = _load_model_lineage(
        model_binding.get("path", ""), workspace, reference, verify_inventory=False
    )
    if model_binding != _receipt_binding(model_doc, model_record):
        raise ContractError("judge context gate model receipt binding differs")
    if judge_model != {
        "contract": model_binding,
        "id": model["id"],
        "tag": model["tag"],
        "inventory": model["inventory"],
    }:
        raise ContractError("judge context gate model inventory differs")

    tokenizer = payload.get("tokenizer")
    if not isinstance(tokenizer, dict) or set(tokenizer) != {
        "loader",
        "model_inventory_sha256",
        "assets",
        "assets_sequence_sha256",
        "chat_template",
    }:
        raise ContractError("judge context gate tokenizer binding is malformed")
    expected_loader_keys = {
        "class",
        "local_files_only",
        "trust_remote_code",
        "use_fast",
    }
    loader = tokenizer["loader"]
    if (
        not isinstance(loader, dict)
        or set(loader) != expected_loader_keys
        or loader.get("local_files_only") is not True
        or loader.get("trust_remote_code") is not False
        or loader.get("use_fast") is not True
        or not isinstance(loader.get("class"), str)
        or not loader["class"]
    ):
        raise ContractError("judge context gate tokenizer loader differs")
    if tokenizer["model_inventory_sha256"] != model["inventory"]["inventory_sha256"]:
        raise ContractError("judge context gate tokenizer/model inventory differs")
    assets = tokenizer["assets"]
    if not isinstance(assets, list) or {
        item.get("relative_path") for item in assets if isinstance(item, dict)
    } != JUDGE_CONTEXT_GATE_TOKENIZER_ASSETS:
        raise ContractError("judge context gate tokenizer asset set differs")
    normalized_assets: list[dict[str, Any]] = []
    for asset in assets:
        relative = asset.get("relative_path")
        normalized_assets.append(
            _revalidate_gate_file_record(
                asset,
                workspace,
                label=f"judge tokenizer asset {relative}",
                expected_relative_path=relative,
            )
        )
    normalized_assets.sort(key=lambda item: item["relative_path"])
    if assets != normalized_assets:
        raise ContractError("judge context gate tokenizer assets are not normalized")
    inventory_files = {
        item.get("relative_path"): item
        for item in model["inventory"].get("files", [])
        if isinstance(item, dict)
    }
    for asset in normalized_assets:
        frozen = inventory_files.get(asset["relative_path"])
        if (
            not isinstance(frozen, dict)
            or frozen.get("bytes") != asset["bytes"]
            or frozen.get("sha256") != asset["sha256"]
        ):
            raise ContractError("judge tokenizer asset differs from model inventory")
    expected_asset_sequence = _canonical_sha256(
        [
            {
                "relative_path": item["relative_path"],
                "bytes": item["bytes"],
                "sha256": item["sha256"],
            }
            for item in normalized_assets
        ]
    )
    if tokenizer["assets_sequence_sha256"] != expected_asset_sequence:
        raise ContractError("judge context gate tokenizer asset sequence differs")
    chat_template = tokenizer["chat_template"]
    if not isinstance(chat_template, dict) or set(chat_template) != {
        "bytes",
        "sha256",
        "file_sha256",
        "matches_tokenizer_config",
        "matches_chat_template_jinja",
    }:
        raise ContractError("judge context gate chat-template binding is malformed")
    chat_asset = next(
        item for item in normalized_assets if item["relative_path"] == "chat_template.jinja"
    )
    if (
        chat_template.get("bytes") != chat_asset["bytes"]
        or chat_template.get("sha256") != chat_asset["sha256"]
        or chat_template.get("file_sha256") != chat_asset["sha256"]
        or chat_template.get("matches_tokenizer_config") is not True
        or chat_template.get("matches_chat_template_jinja") is not True
    ):
        raise ContractError("judge context gate chat-template hashes differ")

    sources = payload.get("sources")
    if not isinstance(sources, dict) or set(sources) != {
        "context_gate",
        "official_judge",
        "mathruler",
    }:
        raise ContractError("judge context gate source binding is malformed")
    _revalidate_gate_file_record(
        sources["context_gate"], workspace, label="judge context gate source"
    )
    official_record = _revalidate_gate_file_record(
        sources["official_judge"],
        workspace,
        label="official judge source",
        trusted_root=Path(reference["root"]),
    )
    official_reference = next(
        item for item in reference["files"] if item["relative_path"] == "eval/judge_qwenlm.py"
    )
    if (
        official_record["path"] != official_reference["path"]
        or official_record["bytes"] != official_reference["bytes"]
        or official_record["sha256"] != official_reference["sha256"]
    ):
        raise ContractError("judge context gate official source differs from reference")
    mathruler = sources["mathruler"]
    if not isinstance(mathruler, dict) or set(mathruler) != {
        "distribution",
        "version",
        "sources",
        "sources_sequence_sha256",
    }:
        raise ContractError("judge context gate MathRuler binding is malformed")
    if mathruler.get("distribution") != "mathruler" or not isinstance(
        mathruler.get("version"), str
    ):
        raise ContractError("judge context gate MathRuler identity differs")
    math_sources = mathruler.get("sources")
    expected_math_names = {"__init__.py", "grader.py", "math_normalize.py"}
    if not isinstance(math_sources, list) or {
        item.get("relative_path") for item in math_sources if isinstance(item, dict)
    } != expected_math_names:
        raise ContractError("judge context gate MathRuler source set differs")
    current_math_sources: list[dict[str, Any]] = []
    for source in math_sources:
        relative = source.get("relative_path")
        current_math_sources.append(
            _revalidate_gate_file_record(
                source,
                workspace,
                label=f"MathRuler source {relative}",
                expected_relative_path=relative,
            )
        )
    if math_sources != current_math_sources:
        raise ContractError("judge context gate MathRuler sources are not normalized")
    expected_math_sequence = _canonical_sha256(
        [
            {
                "relative_path": item["relative_path"],
                "bytes": item["bytes"],
                "sha256": item["sha256"],
            }
            for item in current_math_sources
        ]
    )
    if mathruler["sources_sequence_sha256"] != expected_math_sequence:
        raise ContractError("judge context gate MathRuler source sequence differs")

    software = payload.get("software")
    if not isinstance(software, dict) or set(software) != {
        "transformers",
        "tokenizers",
        "mathruler",
    } or any(not isinstance(item, str) or not item for item in software.values()):
        raise ContractError("judge context gate software versions are malformed")
    if software["mathruler"] != mathruler["version"]:
        raise ContractError("judge context gate MathRuler versions differ")

    pairs = payload.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ContractError("judge context gate has no pairs")
    pair_keys: set[tuple[str, str]] = set()
    answer_paths: set[str] = set()
    total_rows = 0
    total_llm = 0
    max_prompt = 0
    for pair in pairs:
        if not isinstance(pair, dict) or set(pair) != {
            "label",
            "benchmark",
            "counts",
            "prompt_tokens",
            "violations",
            "answers_receipt",
        }:
            raise ContractError("judge context gate pair is malformed")
        label = pair.get("label")
        benchmark = pair.get("benchmark")
        if (
            not isinstance(label, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", label) is None
            or benchmark not in BENCHMARKS
        ):
            raise ContractError("judge context gate pair identity differs")
        key = (label, benchmark)
        if key in pair_keys:
            raise ContractError(f"duplicate judge context gate pair: {label}/{benchmark}")
        pair_keys.add(key)
        answer_binding = pair.get("answers_receipt")
        if not isinstance(answer_binding, dict):
            raise ContractError("judge context gate pair has no answers binding")
        answers_doc, answers_record, answers = _load_answers_lineage(
            answer_binding.get("path", ""), workspace, reference
        )
        if answer_binding != _receipt_binding(answers_doc, answers_record):
            raise ContractError("judge context gate answers binding differs")
        if answers_doc["payload"].get("benchmark") != benchmark:
            raise ContractError("judge context gate answers benchmark differs")
        if answer_binding["path"] in answer_paths:
            raise ContractError("judge context gate reuses one answers receipt")
        answer_paths.add(answer_binding["path"])
        counts = pair.get("counts")
        if not isinstance(counts, dict) or set(counts) != {
            "rows",
            "mathruler",
            "first_letter",
            "llm",
            "mathruler_exceptions",
        }:
            raise ContractError("judge context gate dispatch counts are malformed")
        if any(not isinstance(value, int) or value < 0 for value in counts.values()):
            raise ContractError("judge context gate dispatch counts are invalid")
        if counts["rows"] != len(answers) or (
            counts["mathruler"] + counts["first_letter"] + counts["llm"]
        ) != counts["rows"]:
            raise ContractError("judge context gate dispatch counts do not cover answers")
        prompt = pair.get("prompt_tokens")
        if not isinstance(prompt, dict) or set(prompt) != {
            "count",
            "min",
            "max",
            "max_with_output",
            "violation_count",
            "token_count_sequence_sha256",
            "row_dispatch_token_sequence_sha256",
        }:
            raise ContractError("judge context gate prompt counts are malformed")
        if any(
            not isinstance(prompt.get(name), int) or prompt[name] < 0
            for name in ("count", "min", "max", "max_with_output", "violation_count")
        ):
            raise ContractError("judge context gate prompt counts are invalid")
        for name in ("token_count_sequence_sha256", "row_dispatch_token_sequence_sha256"):
            if re.fullmatch(r"[0-9a-f]{64}", str(prompt.get(name, ""))) is None:
                raise ContractError("judge context gate token sequence digest is malformed")
        if (
            prompt["count"] != counts["llm"]
            or prompt["max_with_output"] != prompt["max"] + OFFICIAL_JUDGE_MAX_TOKENS
            or prompt["max_with_output"] > OFFICIAL_JUDGE_MAX_MODEL_LEN
            or prompt["violation_count"] != 0
            or pair.get("violations") != []
        ):
            raise ContractError("judge context gate did not pass ctx65k without truncation")
        total_rows += counts["rows"]
        total_llm += counts["llm"]
        max_prompt = max(max_prompt, prompt["max"])
    if pairs != sorted(pairs, key=lambda item: (item["label"], item["benchmark"])):
        raise ContractError("judge context gate pairs are not normalized")

    aggregate = payload.get("aggregate")
    expected_pair_sequence = _canonical_sha256(
        [
            {
                "label": pair["label"],
                "benchmark": pair["benchmark"],
                "token_count_sequence_sha256": pair["prompt_tokens"][
                    "token_count_sequence_sha256"
                ],
            }
            for pair in pairs
        ]
    )
    expected_aggregate = {
        "pair_count": len(pairs),
        "row_count": total_rows,
        "llm_prompt_count": total_llm,
        "max_prompt_tokens": max_prompt,
        "max_with_output": max_prompt + OFFICIAL_JUDGE_MAX_TOKENS,
        "violation_count": 0,
        "pair_token_count_sequence_sha256": expected_pair_sequence,
    }
    if aggregate != expected_aggregate:
        raise ContractError("judge context gate aggregate differs from its pairs")
    return receipt, record, payload


def build_judge_runtime_payload(
    *,
    workspace: Path,
    reference_root: Path | str,
    judge_model_contract: Path | str,
    judge_context_gate: Path | str,
    judge_model_id: str,
    judge_api_base: str,
    cuda_devices: str,
    tensor_parallel_size: int,
    source_specs: Sequence[Sequence[str]],
    expected_pair_specs: Sequence[Sequence[str]],
) -> dict[str, Any]:
    """Freeze one recoverable judge server and its complete planned matrix."""

    reference = validate_reference_sources(reference_root)
    model_doc, model_record, model = _load_model_lineage(
        judge_model_contract, workspace, reference, verify_inventory=False
    )
    gate_doc, gate_record, gate = _load_judge_context_gate_lineage(
        judge_context_gate, workspace, reference
    )
    model_binding = _receipt_binding(model_doc, model_record)
    if gate["judge_model"]["contract"] != model_binding:
        raise ContractError("judge runtime and context gate model bindings differ")
    judge_model_id = _clean_text(judge_model_id, label="judge model ID")
    if judge_model_id != model["id"]:
        raise ContractError("judge runtime model ID differs from judge model receipt")
    judge_api_base = _validate_api_base(judge_api_base)
    devices = _parse_cuda_devices(cuda_devices)
    if tensor_parallel_size < 1 or tensor_parallel_size > len(devices):
        raise ContractError("judge tensor parallel size exceeds selected CUDA devices")
    sources = _normalize_source_specs(source_specs, workspace)
    expected_pairs = _normalize_expected_pairs(expected_pair_specs, workspace)
    expected_pair_keys = {
        (pair["label"], pair["benchmark"]) for pair in expected_pairs
    }
    gate_pair_keys = {(pair["label"], pair["benchmark"]) for pair in gate["pairs"]}
    if gate_pair_keys != expected_pair_keys:
        raise ContractError("judge runtime pair set differs from context gate")
    gate_source = gate["sources"]["context_gate"]
    runtime_gate_source = next(
        item for item in sources if item["name"] == "judge_context_gate"
    )
    if {
        "path": runtime_gate_source["path"],
        "bytes": runtime_gate_source["bytes"],
        "sha256": runtime_gate_source["sha256"],
    } != gate_source:
        raise ContractError("judge runtime context-gate source differs from gate receipt")
    return {
        "judge_model_contract": model_binding,
        "judge_context_gate": _receipt_binding(gate_doc, gate_record),
        "judge_model_identity": {
            "id": model["id"],
            "tag": model["tag"],
            "inventory_path": model["inventory"]["path"],
            "inventory_sha256": model["inventory"]["inventory_sha256"],
            "file_count": model["inventory"]["file_count"],
            "total_bytes": model["inventory"]["total_bytes"],
        },
        "judge_protocol": {
            "model_id": judge_model_id,
            "api_base": judge_api_base,
            "temperature": 0,
            "max_tokens": OFFICIAL_JUDGE_MAX_TOKENS,
        },
        "serve_config": {
            "cuda_visible_devices": devices,
            "tensor_parallel_size": tensor_parallel_size,
            "data_parallel_size": OFFICIAL_JUDGE_DATA_PARALLEL_SIZE,
            "max_model_len": OFFICIAL_JUDGE_MAX_MODEL_LEN,
            "max_num_seqs": OFFICIAL_JUDGE_MAX_NUM_SEQS,
            "gpu_memory_utilization": OFFICIAL_JUDGE_GPU_MEMORY_UTILIZATION,
            "limit_mm_per_prompt": {"image": OFFICIAL_JUDGE_MM_IMAGE_LIMIT},
            "default_chat_template_kwargs": {"enable_thinking": False},
        },
        "sources": sources,
        "expected_pairs": expected_pairs,
        "reference": reference,
    }


def _load_judge_runtime_lineage(
    path: Path | str,
    workspace: Path,
    reference: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    receipt, record = _load_receipt(
        path, workspace, expected_kind="vision_opd_reference_judge_runtime"
    )
    payload = receipt["payload"]
    _validate_reference_binding(payload, reference, label="judge runtime receipt")
    model_binding = payload.get("judge_model_contract")
    if not isinstance(model_binding, dict):
        raise ContractError("judge runtime has no judge model binding")
    model_doc, model_record, model = _load_model_lineage(
        model_binding.get("path", ""), workspace, reference, verify_inventory=False
    )
    if model_binding != _receipt_binding(model_doc, model_record):
        raise ContractError("judge runtime model-receipt binding differs")
    gate_binding = payload.get("judge_context_gate")
    if not isinstance(gate_binding, dict):
        raise ContractError("judge runtime has no context-gate binding")
    gate_doc, gate_record, gate = _load_judge_context_gate_lineage(
        gate_binding.get("path", ""), workspace, reference
    )
    if gate_binding != _receipt_binding(gate_doc, gate_record):
        raise ContractError("judge runtime context-gate receipt binding differs")
    if gate["judge_model"]["contract"] != model_binding:
        raise ContractError("judge runtime context gate/model binding differs")
    identity = payload.get("judge_model_identity")
    expected_identity = {
        "id": model["id"],
        "tag": model["tag"],
        "inventory_path": model["inventory"]["path"],
        "inventory_sha256": model["inventory"]["inventory_sha256"],
        "file_count": model["inventory"]["file_count"],
        "total_bytes": model["inventory"]["total_bytes"],
    }
    if identity != expected_identity:
        raise ContractError("judge runtime model inventory identity differs")
    protocol = payload.get("judge_protocol")
    if not isinstance(protocol, dict) or protocol.get("model_id") != model["id"]:
        raise ContractError("judge runtime protocol/model identity differs")
    if protocol.get("temperature") != 0 or protocol.get("max_tokens") != OFFICIAL_JUDGE_MAX_TOKENS:
        raise ContractError("judge runtime protocol differs from official settings")
    if _validate_api_base(protocol.get("api_base", "")) != protocol.get("api_base"):
        raise ContractError("judge runtime API base is not normalized")
    serve = payload.get("serve_config")
    sources = payload.get("sources")
    pairs = payload.get("expected_pairs")
    if not isinstance(serve, dict) or not isinstance(sources, list) or not isinstance(pairs, list):
        raise ContractError("judge runtime receipt is malformed")
    devices = serve.get("cuda_visible_devices")
    tensor_parallel_size = serve.get("tensor_parallel_size")
    if (
        not isinstance(devices, list)
        or not devices
        or any(not isinstance(item, int) or item < 0 or item > 7 for item in devices)
        or len(set(devices)) != len(devices)
        or not isinstance(tensor_parallel_size, int)
        or tensor_parallel_size < 1
        or tensor_parallel_size > len(devices)
    ):
        raise ContractError("judge runtime CUDA/TP configuration is malformed")
    expected_serve = {
        "cuda_visible_devices": devices,
        "tensor_parallel_size": tensor_parallel_size,
        "data_parallel_size": OFFICIAL_JUDGE_DATA_PARALLEL_SIZE,
        "max_model_len": OFFICIAL_JUDGE_MAX_MODEL_LEN,
        "max_num_seqs": OFFICIAL_JUDGE_MAX_NUM_SEQS,
        "gpu_memory_utilization": OFFICIAL_JUDGE_GPU_MEMORY_UTILIZATION,
        "limit_mm_per_prompt": {"image": OFFICIAL_JUDGE_MM_IMAGE_LIMIT},
        "default_chat_template_kwargs": {"enable_thinking": False},
    }
    if serve != expected_serve:
        raise ContractError("judge runtime serve config differs from ctx65k policy")
    if {item.get("name") for item in sources if isinstance(item, dict)} != REQUIRED_JUDGE_RUNTIME_SOURCES:
        raise ContractError("judge runtime source set differs")
    if not pairs:
        raise ContractError("judge runtime has no expected pairs")
    if {(pair.get("label"), pair.get("benchmark")) for pair in pairs} != {
        (pair["label"], pair["benchmark"]) for pair in gate["pairs"]
    }:
        raise ContractError("judge runtime pair set differs from context gate")
    return receipt, record, payload


def _inspect_judge(
    path: Path | str,
    workspace: Path,
    answers: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    judge_path = _safe_existing(path, workspace, label="official judge JSON", kind="file")
    value, file_record = _load_json(judge_path, label="official judge JSON")
    if not isinstance(value, list) or len(value) != len(answers):
        actual = len(value) if isinstance(value, list) else "non-list"
        raise ContractError(f"judge JSON row count differs: {actual} != {len(answers)}")
    seen: set[str] = set()
    correct = 0
    source_counts: Counter[str] = Counter()
    category_stats: dict[str, dict[str, int]] = {}
    for index, (item, answer) in enumerate(zip(value, answers), start=1):
        if not isinstance(item, dict):
            raise ContractError(f"judge JSON row {index} is not an object")
        uid = item.get("sample_uid")
        if not isinstance(uid, str) or not uid.strip():
            raise ContractError(f"judge JSON row {index} has no string sample_uid")
        if uid in seen:
            raise ContractError(f"judge JSON has duplicate UID: {uid}")
        seen.add(uid)
        if uid != answer["sample_uid"]:
            raise ContractError(f"judge JSON order/UID differs at row {index}")
        expected_keys = set(answer).union(JUDGE_ADDED_KEYS)
        if set(item) != expected_keys:
            raise ContractError(f"judge JSON keys differ for UID: {uid}")
        for key, expected_value in answer.items():
            if item.get(key) != expected_value:
                raise ContractError(f"judge JSON answer payload differs for UID {uid}, key {key}")
        judge = item.get("judge")
        source = item.get("judge_source")
        if not isinstance(judge, str) or not judge.strip():
            raise ContractError(f"judge JSON has empty judge for UID: {uid}")
        if not isinstance(source, str) or not source.strip():
            raise ContractError(f"judge JSON has empty judge_source for UID: {uid}")
        source = source.strip()
        if source not in OFFICIAL_JUDGE_SOURCES:
            raise ContractError(f"judge JSON has unsupported judge_source for UID {uid}: {source}")
        source_counts[source] += 1
        exact_yes = judge.strip().lower() == "yes"
        if exact_yes:
            correct += 1
        category = str(item.get("category", "unknown") or "unknown")
        bucket = category_stats.setdefault(category, {"correct": 0, "total": 0})
        bucket["total"] += 1
        if exact_yes:
            bucket["correct"] += 1
    total = len(value)
    metrics = {
        "definition": "str(judge).strip().lower() == 'yes' (pinned cal_acc.py)",
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else 0.0,
        "accuracy_percent": (100.0 * correct / total) if total else 0.0,
        "fraction": f"{correct}/{total}",
        "judge_source_counts": dict(sorted(source_counts.items())),
        "category_counts": dict(sorted(category_stats.items())),
    }
    file_record.update(
        {
            "row_count": total,
            "unique_uid_count": len(seen),
            "uid_order_sha256": _canonical_sha256([item["sample_uid"] for item in value]),
        }
    )
    return file_record, metrics


def build_judge_payload(
    *,
    workspace: Path,
    reference_root: Path | str,
    answers_receipt: Path | str,
    judge_json: Path | str,
    judge_model_contract: Path | str,
    judge_runtime_contract: Path | str | None,
    judge_model_id: str,
    judge_api_base: str,
    judge_temperature: float,
    judge_max_tokens: int,
) -> dict[str, Any]:
    reference = validate_reference_sources(reference_root)
    answers_doc, answers_record, answers = _load_answers_lineage(
        answers_receipt, workspace, reference
    )
    model_doc, model_record, model = _load_model_lineage(
        judge_model_contract, workspace, reference, verify_inventory=False
    )
    answers_binding = _receipt_binding(answers_doc, answers_record)
    model_binding = _receipt_binding(model_doc, model_record)
    judge_model_id = _clean_text(judge_model_id, label="judge model ID")
    if judge_model_id != model["id"]:
        raise ContractError("judge model ID differs from judge model receipt")
    judge_api_base = _validate_api_base(judge_api_base)
    if judge_temperature != 0:
        raise ContractError("official judge contract requires temperature=0")
    if judge_max_tokens != OFFICIAL_JUDGE_MAX_TOKENS:
        raise ContractError(
            f"official judge contract requires max_tokens={OFFICIAL_JUDGE_MAX_TOKENS}"
        )
    runtime_binding: dict[str, Any] | None = None
    if judge_runtime_contract is not None:
        runtime_doc, runtime_record, runtime = _load_judge_runtime_lineage(
            judge_runtime_contract, workspace, reference
        )
        if runtime.get("judge_model_contract") != model_binding:
            raise ContractError("judge runtime and judge model receipt bindings differ")
        expected_protocol = {
            "model_id": judge_model_id,
            "api_base": judge_api_base,
            "temperature": 0,
            "max_tokens": OFFICIAL_JUDGE_MAX_TOKENS,
        }
        if runtime.get("judge_protocol") != expected_protocol:
            raise ContractError("judge invocation differs from frozen judge runtime")
        gate_binding = runtime.get("judge_context_gate")
        if not isinstance(gate_binding, dict):
            raise ContractError("judge runtime has no context-gate binding")
        _, _, gate = _load_judge_context_gate_lineage(
            gate_binding.get("path", ""), workspace, reference
        )
        matching_gate_pairs = [
            pair
            for pair in gate["pairs"]
            if pair["benchmark"] == answers_doc["payload"]["benchmark"]
            and pair["answers_receipt"] == answers_binding
        ]
        if len(matching_gate_pairs) != 1:
            raise ContractError("judge answers are not uniquely covered by context gate")
        runtime_binding = _receipt_binding(runtime_doc, runtime_record)
    judge_record, metrics = _inspect_judge(judge_json, workspace, answers)
    return {
        "benchmark": answers_doc["payload"]["benchmark"],
        "answers_receipt": answers_binding,
        "judge_model_contract": model_binding,
        "judge_model_identity": {
            "id": model["id"],
            "tag": model["tag"],
            "inventory_path": model["inventory"]["path"],
            "inventory_sha256": model["inventory"]["inventory_sha256"],
            "file_count": model["inventory"]["file_count"],
            "total_bytes": model["inventory"]["total_bytes"],
        },
        "judge_runtime_contract": runtime_binding,
        "judge_json": judge_record,
        "judge_protocol": {
            "model_id": judge_model_id,
            "api_base": judge_api_base,
            "temperature": 0,
            "max_tokens": OFFICIAL_JUDGE_MAX_TOKENS,
        },
        "official_exact_yes": metrics,
        "reference": reference,
    }


def _load_judge_lineage(
    path: Path | str,
    workspace: Path,
    reference: dict[str, Any],
    *,
    expected_model_binding: dict[str, Any],
    expected_model_identity: dict[str, Any],
    expected_runtime_binding: dict[str, Any],
    expected_protocol: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    receipt, record = _load_receipt(
        path, workspace, expected_kind="vision_opd_reference_judge"
    )
    payload = receipt["payload"]
    _validate_reference_binding(payload, reference, label="judge receipt")
    if payload.get("judge_model_contract") != expected_model_binding:
        raise ContractError("judge receipt model binding differs from matrix runtime")
    if payload.get("judge_model_identity") != expected_model_identity:
        raise ContractError("judge receipt model inventory identity differs from matrix runtime")
    if payload.get("judge_runtime_contract") != expected_runtime_binding:
        raise ContractError("judge receipt runtime binding differs from matrix runtime")
    if payload.get("judge_protocol") != expected_protocol:
        raise ContractError("judge receipt protocol differs from matrix runtime")
    answers_binding = payload.get("answers_receipt")
    judge_binding = payload.get("judge_json")
    if not isinstance(answers_binding, dict) or not isinstance(judge_binding, dict):
        raise ContractError("judge receipt is malformed")
    answers_doc, answers_record, answers = _load_answers_lineage(
        answers_binding.get("path", ""), workspace, reference
    )
    if answers_binding != _receipt_binding(answers_doc, answers_record):
        raise ContractError("judge receipt answers binding differs")
    current_judge, metrics = _inspect_judge(
        judge_binding.get("path", ""), workspace, answers
    )
    if current_judge != judge_binding:
        raise ContractError("judge JSON no longer matches judge receipt")
    if metrics != payload.get("official_exact_yes"):
        raise ContractError("judge receipt metrics no longer match judge JSON")
    if payload.get("benchmark") != answers_doc["payload"].get("benchmark"):
        raise ContractError("judge receipt benchmark differs from answers")
    return receipt, record, payload


def build_matrix_payload(
    *,
    workspace: Path,
    reference_root: Path | str,
    judge_runtime_contract: Path | str,
    judge_receipt_specs: Sequence[Sequence[str]],
) -> dict[str, Any]:
    """Freeze the complete, comparable result matrix under one runtime plan."""

    reference = validate_reference_sources(reference_root)
    runtime_doc, runtime_record, runtime = _load_judge_runtime_lineage(
        judge_runtime_contract, workspace, reference
    )
    observed = _normalize_expected_pairs(judge_receipt_specs, workspace)
    expected = runtime["expected_pairs"]
    if observed != expected:
        raise ContractError("final matrix pair set differs from frozen judge runtime")
    runtime_binding = _receipt_binding(runtime_doc, runtime_record)
    model_binding = runtime["judge_model_contract"]
    protocol = runtime["judge_protocol"]
    gate_binding = runtime["judge_context_gate"]
    gate_doc, gate_record, gate = _load_judge_context_gate_lineage(
        gate_binding.get("path", ""), workspace, reference
    )
    if gate_binding != _receipt_binding(gate_doc, gate_record):
        raise ContractError("matrix runtime/context-gate binding differs")
    gate_pairs = {
        (pair["label"], pair["benchmark"]): pair for pair in gate["pairs"]
    }
    pairs: list[dict[str, Any]] = []
    for pair in observed:
        judge_doc, judge_record, judge_payload = _load_judge_lineage(
            pair["judge_receipt"],
            workspace,
            reference,
            expected_model_binding=model_binding,
            expected_model_identity=runtime["judge_model_identity"],
            expected_runtime_binding=runtime_binding,
            expected_protocol=protocol,
        )
        if judge_payload.get("benchmark") != pair["benchmark"]:
            raise ContractError(
                f"judge receipt benchmark differs for {pair['label']}/{pair['benchmark']}"
            )
        gate_pair = gate_pairs.get((pair["label"], pair["benchmark"]))
        if gate_pair is None or judge_payload.get("answers_receipt") != gate_pair.get(
            "answers_receipt"
        ):
            raise ContractError(
                f"judge receipt answers differ from context gate for "
                f"{pair['label']}/{pair['benchmark']}"
            )
        pairs.append(
            {
                "label": pair["label"],
                "benchmark": pair["benchmark"],
                "judge_receipt": _receipt_binding(judge_doc, judge_record),
                "official_exact_yes": judge_payload["official_exact_yes"],
            }
        )
    return {
        "judge_runtime_contract": runtime_binding,
        "judge_context_gate": gate_binding,
        "judge_model_contract": model_binding,
        "judge_model_identity": runtime["judge_model_identity"],
        "judge_protocol": protocol,
        "serve_config": runtime["serve_config"],
        "sources": runtime["sources"],
        "pairs": pairs,
        "reference": reference,
    }


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "false":
        return False
    if normalized == "true":
        return True
    raise argparse.ArgumentTypeError("expected true or false")


def _common(subparser: argparse.ArgumentParser) -> None:
    subparser.add_argument("--workspace-root", type=Path, default=WORKSPACE)
    subparser.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCE_ROOT)
    subparser.add_argument("--receipt", type=Path, required=True)
    subparser.add_argument("--quiet", action="store_true")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    data = commands.add_parser("data", help="freeze benchmark JSON and every actual first image")
    _common(data)
    data.add_argument("--benchmark", choices=tuple(BENCHMARKS), required=True)
    data.add_argument("--benchmark-json", type=Path, required=True)
    data.add_argument("--data-root", type=Path, required=True)
    data.add_argument("--preparation-receipt", type=Path, required=True)
    data.add_argument("--preparer-path", type=Path, required=True)

    run = commands.add_parser("run", help="freeze official inference and complete model inventory")
    _common(run)
    run.add_argument("--data-receipt", type=Path, required=True)
    run.add_argument("--model-path", type=Path, required=True)
    run.add_argument("--model-id", required=True)
    run.add_argument("--model-tag", required=True)
    run.add_argument("--seed", type=int, default=OFFICIAL_SEED)
    run.add_argument("--seed-label", default=OFFICIAL_SEED_LABEL)
    run.add_argument("--enable-thinking", type=_parse_bool, default=False)
    run.add_argument("--temperature", type=float, default=0)
    run.add_argument("--max-tokens", type=int, default=OFFICIAL_MAX_TOKENS)

    model = commands.add_parser("model", help="freeze a standalone complete model inventory")
    _common(model)
    model.add_argument("--model-path", type=Path, required=True)
    model.add_argument("--model-id", required=True)
    model.add_argument("--model-tag", required=True)

    runtime = commands.add_parser(
        "judge-runtime", help="freeze one judge model/server plan and expected matrix"
    )
    _common(runtime)
    runtime.add_argument("--judge-model-contract", type=Path, required=True)
    runtime.add_argument("--judge-context-gate", type=Path, required=True)
    runtime.add_argument("--judge-model-id", required=True)
    runtime.add_argument("--judge-api-base", required=True)
    runtime.add_argument("--cuda-devices", required=True)
    runtime.add_argument("--tensor-parallel-size", type=int, required=True)
    runtime.add_argument("--source", action="append", nargs=2, default=[], metavar=("NAME", "PATH"))
    runtime.add_argument(
        "--expected-pair",
        action="append",
        nargs=3,
        default=[],
        metavar=("LABEL", "BENCHMARK", "JUDGE_RECEIPT"),
    )

    answers = commands.add_parser("answers", help="freeze a complete official infer JSONL")
    _common(answers)
    answers.add_argument("--run-contract", type=Path, required=True)
    answers.add_argument("--answer-jsonl", type=Path, required=True)

    judge = commands.add_parser("judge", help="freeze official judge JSON and exact-Yes accuracy")
    _common(judge)
    judge.add_argument("--answers-receipt", type=Path, required=True)
    judge.add_argument("--judge-json", type=Path, required=True)
    judge.add_argument("--judge-model-contract", type=Path, required=True)
    judge.add_argument("--judge-runtime-contract", type=Path)
    judge.add_argument("--judge-model-id", required=True)
    judge.add_argument("--judge-api-base", required=True)
    judge.add_argument("--judge-temperature", type=float, default=0)
    judge.add_argument("--judge-max-tokens", type=int, default=OFFICIAL_JUDGE_MAX_TOKENS)

    matrix = commands.add_parser(
        "matrix", help="freeze all judge receipts under one exact judge runtime"
    )
    _common(matrix)
    matrix.add_argument("--judge-runtime-contract", type=Path, required=True)
    matrix.add_argument(
        "--judge-receipt",
        action="append",
        nargs=3,
        default=[],
        metavar=("LABEL", "BENCHMARK", "JUDGE_RECEIPT"),
    )

    paths = commands.add_parser(
        "artifact-paths",
        help="reject symlink/special output leaves and ancestors, optionally creating directories",
    )
    paths.add_argument("--workspace-root", type=Path, default=WORKSPACE)
    paths.add_argument("--directory", type=Path, action="append", default=[])
    paths.add_argument("--leaf", type=Path, action="append", default=[])
    paths.add_argument("--create", action="store_true")
    paths.add_argument("--quiet", action="store_true")

    partial = commands.add_parser(
        "partial-answers",
        help="strictly validate an incomplete official answer JSONL before resume",
    )
    partial.add_argument("--workspace-root", type=Path, default=WORKSPACE)
    partial.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCE_ROOT)
    partial.add_argument("--run-contract", type=Path, required=True)
    partial.add_argument("--answer-jsonl", type=Path, required=True)
    partial.add_argument("--model-id", required=True)
    partial.add_argument("--model-tag", required=True)
    partial.add_argument("--quiet", action="store_true")

    stage = commands.add_parser(
        "stage-file", help="hardlink/copy one immutable input into a safe staging tree"
    )
    stage.add_argument("--workspace-root", type=Path, default=WORKSPACE)
    stage.add_argument("--source", type=Path, required=True)
    stage.add_argument("--destination", type=Path, required=True)
    stage.add_argument("--quiet", action="store_true")

    publish = commands.add_parser(
        "publish-file",
        help="fsync and atomically publish a file with renameat2 NOREPLACE",
    )
    publish.add_argument("--workspace-root", type=Path, default=WORKSPACE)
    publish.add_argument("--source", type=Path, required=True)
    publish.add_argument("--destination", type=Path, required=True)
    publish.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        workspace = _safe_root(args.workspace_root, label="workspace root")
        if args.command == "artifact-paths":
            result = gate_artifact_paths(
                workspace=workspace,
                directories=args.directory,
                leaves=args.leaf,
                create=args.create,
            )
            if not args.quiet:
                print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
            return 0
        if args.command == "partial-answers":
            result = inspect_partial_answers(
                workspace=workspace,
                reference_root=args.reference_root,
                run_contract=args.run_contract,
                answer_jsonl=args.answer_jsonl,
                model_id=args.model_id,
                model_tag=args.model_tag,
            )
            if not args.quiet:
                print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
            return 0
        if args.command == "stage-file":
            result = stage_file(
                workspace=workspace,
                source=args.source,
                destination=args.destination,
            )
            if not args.quiet:
                print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
            return 0
        if args.command == "publish-file":
            result = publish_file(
                workspace=workspace,
                source=args.source,
                destination=args.destination,
            )
            if not args.quiet:
                print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
            return 0
        if args.command == "data":
            payload = build_data_payload(
                workspace=workspace,
                reference_root=args.reference_root,
                benchmark=args.benchmark,
                benchmark_json=args.benchmark_json,
                data_root=args.data_root,
                preparation_receipt=args.preparation_receipt,
                preparer_path=args.preparer_path,
            )
            envelope = _seal("vision_opd_reference_data", payload)
        elif args.command == "run":
            payload = build_run_payload(
                workspace=workspace,
                reference_root=args.reference_root,
                data_receipt=args.data_receipt,
                model_path=args.model_path,
                model_id=args.model_id,
                model_tag=args.model_tag,
                seed=args.seed,
                seed_label=args.seed_label,
                enable_thinking=args.enable_thinking,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            envelope = _seal("vision_opd_reference_run", payload)
        elif args.command == "model":
            payload = build_model_payload(
                workspace=workspace,
                reference_root=args.reference_root,
                model_path=args.model_path,
                model_id=args.model_id,
                model_tag=args.model_tag,
            )
            envelope = _seal("vision_opd_reference_model", payload)
        elif args.command == "judge-runtime":
            payload = build_judge_runtime_payload(
                workspace=workspace,
                reference_root=args.reference_root,
                judge_model_contract=args.judge_model_contract,
                judge_context_gate=args.judge_context_gate,
                judge_model_id=args.judge_model_id,
                judge_api_base=args.judge_api_base,
                cuda_devices=args.cuda_devices,
                tensor_parallel_size=args.tensor_parallel_size,
                source_specs=args.source,
                expected_pair_specs=args.expected_pair,
            )
            envelope = _seal("vision_opd_reference_judge_runtime", payload)
        elif args.command == "answers":
            payload = build_answers_payload(
                workspace=workspace,
                reference_root=args.reference_root,
                run_contract=args.run_contract,
                answer_jsonl=args.answer_jsonl,
            )
            envelope = _seal("vision_opd_reference_answers", payload)
        elif args.command == "judge":
            payload = build_judge_payload(
                workspace=workspace,
                reference_root=args.reference_root,
                answers_receipt=args.answers_receipt,
                judge_json=args.judge_json,
                judge_model_contract=args.judge_model_contract,
                judge_runtime_contract=args.judge_runtime_contract,
                judge_model_id=args.judge_model_id,
                judge_api_base=args.judge_api_base,
                judge_temperature=args.judge_temperature,
                judge_max_tokens=args.judge_max_tokens,
            )
            envelope = _seal("vision_opd_reference_judge", payload)
        else:
            payload = build_matrix_payload(
                workspace=workspace,
                reference_root=args.reference_root,
                judge_runtime_contract=args.judge_runtime_contract,
                judge_receipt_specs=args.judge_receipt,
            )
            envelope = _seal("vision_opd_reference_judge_matrix", payload)
        result = _publish_create_once(args.receipt, workspace, envelope)
    except ContractError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    if not args.quiet:
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
