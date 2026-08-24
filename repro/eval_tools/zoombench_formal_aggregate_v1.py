#!/usr/bin/env python3
"""Publish a gold-isolated, aggregate-only ZoomBench authority.

The two GPU lifecycle scripts own candidate inference and the fixed local
judge.  This helper is deliberately CPU-only: it validates pinned metadata,
reads only the official judge label from the private result, and publishes a
single create-once aggregate receipt.  It never retains or prints questions,
images, gold answers, predictions, responses, or sample identifiers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


WORKSPACE = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = WORKSPACE / "Output"
DATASET_MANIFEST = WORKSPACE / "Dataset" / "eval" / "ZoomBench" / "manifest.json"
EXPECTED_UID = 30853
EXPECTED_TOTAL = 845
SCHEMA_VERSION = "zoombench_formal_aggregate_v1"
EXPECTED_DATASET_ID = "inclusionAI/ZoomBench"
EXPECTED_DATASET_REVISION = "b788097e57d30510c6877824833234a73bf80d25"
EXPECTED_OFFICIAL_EVAL_COMMIT = "fdc0ba1a3dee916d8c38304d543ad414879e0c99"
EXPECTED_REFERENCE_COMMIT = "c8a8fdd1f88eef1b5ef4fe6a8d64eb0272917471"
EXPECTED_MANIFEST_SHA256 = "7c01d8b6db61834f3e8e550e0bae1a3475e555e6e54959087b72dae3d14ddc72"
EXPECTED_SOURCE_PARQUET_SHA256 = "d44ebda2eda485cba055181f4e6dc50c42f81b5d0f7e936bf427fa01502a391a"
EXPECTED_CANDIDATE_MAX_TOKENS = 256
EXPECTED_JUDGE_MAX_TOKENS = 32
EXPECTED_JUDGE_PROTOCOL = (
    "rule/MathRuler pass followed by fixed local semantic Qwen judge, then accuracy"
)
EXPECTED_CUDA_DEVICES = "0,1,2,3,4,5,6,7"
EXPECTED_CANDIDATE_TP = 8
EXPECTED_CANDIDATE_DP = 1
EXPECTED_JUDGE_TP = 2
EXPECTED_JUDGE_DP = 4
CHECKPOINT_IDENTITY_SCHEMA = "zoombench_checkpoint_identity_v1"
CHECKPOINT_CRITICAL_FILES = (
    "config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "added_tokens.json",
    "spiece.model",
    "vocab.json",
    "merges.txt",
    "tokenizer.model",
    "special_tokens_map.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "generation_config.json",
)
TOKENIZER_CRITICAL_FILES = frozenset(
    {
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "added_tokens.json",
        "spiece.model",
        "vocab.json",
        "merges.txt",
        "tokenizer.model",
        "special_tokens_map.json",
    }
)
CHECKPOINT_EXECUTION_ASSET_SUFFIXES = (
    ".cfg",
    ".ini",
    ".json",
    ".json5",
    ".toml",
    ".yaml",
    ".yml",
    ".jinja",
    ".jinja2",
    ".template",
    ".tmpl",
    ".py",
)
CHECKPOINT_EXECUTION_ASSET_TOKENS = (
    "config",
    "processor",
    "template",
)
ORDER_METADATA_KEYS = ("sample_uid", "index", "id", "question_type")

# These are intentionally explicit pins.  The shell orchestrator checks the
# same values before either GPU phase; the scorer checks them again before
# publishing the authority, so a mixed source lifecycle fails closed.
CRITICAL_SOURCE_HASHES: dict[str, tuple[Path, str]] = {
    "eval_env": (
        Path("Codes/opd-qwen35/eval_tools/eval_env.sh"),
        "1f6b5b0e502e0a68d19028e6c4c20ffa870430b1e8b2aa41cee2a5126659baf5",
    ),
    "run_one_model_eval": (
        Path("Codes/opd-qwen35/eval_tools/run_one_model_eval.sh"),
        "5d7c4867ca0d8abd0a9108400b036fe423978ee1ab0830cc5f59caa5c9d936d9",
    ),
    "run_zoombench": (
        Path("Codes/opd-qwen35/eval_tools/run_zoombench.sh"),
        "c103c1355b47e8a81b378b09164c8342a1b1a4ff41793dbe215e73efe9555ff0",
    ),
    "verify_zoombench_inference": (
        Path("Codes/opd-qwen35/eval_tools/verify_zoombench_inference.py"),
        "66b61ca08a383bec88feef28cfa270141dc4e2a6807abd31bd5e81b144c1033c",
    ),
    "run_zoombench_judge_matrix": (
        Path("Codes/opd-qwen35/eval_tools/run_zoombench_judge_matrix.sh"),
        "dd3892e33d8f3e92dc41fa5a7388574098e1a288d965f62b7d0eefecdc2d979d",
    ),
    "serve_qwen35": (
        Path("Codes/opd-qwen35/eval_tools/serve_qwen35.sh"),
        "e685b999ab846de7967dacd08a3a7124184b0212b7610da4aac4b48ce24effea",
    ),
    "prepare_zoombench": (
        Path("Codes/opd-qwen35/eval_tools/prepare_zoombench.py"),
        "f60e4fc6255c6d4083933f706de835c1fd30c2d87e16502945af9d8625481ef8",
    ),
    "zoombench_manifest_contract": (
        Path("Codes/opd-qwen35/eval_tools/manifests/zoombench.json"),
        "e730048f8d0ebc16e8698779d2271a14ff7d09398419018da9f650a7a98f37e7",
    ),
    "vision_opd_infer": (
        Path("Codes/Vision-OPD-reference/eval/infer.py"),
        "bb379999932658907196cdc98d22c60d63e3308cb5a867317481c4a85af70374",
    ),
    "vision_opd_judge": (
        Path("Codes/Vision-OPD-reference/eval/judge_qwenlm.py"),
        "abbe11dacf7fae19728ca16407a02c91d04a9bc8ea72edd3b4a91b6224f4b670",
    ),
    "vision_opd_cal_acc": (
        Path("Codes/Vision-OPD-reference/eval/cal_acc.py"),
        "695dbddc3e63a1b9f8971c0d414d963a5da94776863d58589feaa4a1c6b0f025",
    ),
}


class AggregateError(RuntimeError):
    """Malformed, incomplete, or mixed lifecycle evidence."""


def _absolute(path: Path | str) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _checkpoint_identity(path: Path, *, label: str) -> dict[str, Any]:
    """Hash a complete model identity without loading model tensors.

    The identity covers every safetensors shard (path, size, and SHA-256) and
    the critical config/tokenizer files.  Missing optional tokenizer assets are
    recorded explicitly, while config.json and at least one tokenizer asset
    are required for a usable Hugging Face checkpoint.  No tensor contents are
    decoded and no identity is printed by the formal lifecycle.
    """

    root = _assert_below_without_symlinks(path, WORKSPACE, label=f"{label} checkpoint")
    try:
        info = root.lstat()
    except OSError as error:
        raise AggregateError(f"{label} checkpoint is unavailable") from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise AggregateError(f"{label} checkpoint must be a real directory")

    def is_execution_asset(relative_name: str) -> bool:
        relative = Path(relative_name)
        basename = relative.name.lower()
        parts = tuple(part.lower() for part in relative.parts)
        return (
            basename in CHECKPOINT_CRITICAL_FILES
            or basename.endswith(CHECKPOINT_EXECUTION_ASSET_SUFFIXES)
            or any(
                token in part
                for part in parts
                for token in CHECKPOINT_EXECUTION_ASSET_TOKENS
            )
        )

    safetensors: list[dict[str, Any]] = []
    for candidate in sorted(root.rglob("*")):
        try:
            candidate_info = candidate.lstat()
        except OSError as error:
            raise AggregateError(f"{label} checkpoint cannot be inspected") from error
        if stat.S_ISLNK(candidate_info.st_mode):
            raise AggregateError(f"{label} checkpoint contains a symlink")
        if stat.S_ISDIR(candidate_info.st_mode):
            continue
        if not stat.S_ISREG(candidate_info.st_mode):
            if is_execution_asset(candidate.relative_to(root).as_posix()):
                raise AggregateError(
                    f"{label} checkpoint execution asset is not a regular file: "
                    f"{candidate.relative_to(root).as_posix()}"
                )
            continue
        relative = candidate.relative_to(root).as_posix()
        if relative.endswith(".safetensors"):
            if candidate_info.st_nlink != 1:
                raise AggregateError(f"{label} safetensors shard has unexpected hard links")
            safetensors.append(
                {
                    "path": relative,
                    "bytes": candidate_info.st_size,
                    "sha256": _sha256(candidate),
                }
            )
    if not safetensors:
        raise AggregateError(f"{label} checkpoint has no safetensors shards")

    critical: dict[str, Any] = {}
    tokenizer_present = False
    for relative in CHECKPOINT_CRITICAL_FILES:
        candidate = root / relative
        if not candidate.exists():
            critical[relative] = None
            continue
        checked = _regular(
            candidate,
            label=f"{label} checkpoint file {relative}",
            trusted_root=root,
            max_bytes=512 * 1024 * 1024,
        )
        if relative in TOKENIZER_CRITICAL_FILES:
            tokenizer_present = True
        critical[relative] = {
            "bytes": checked.stat().st_size,
            "sha256": _sha256(checked),
        }
    # trust_remote_code permits arbitrary local Python and custom processor /
    # template / config assets to affect execution.  Hash every such asset,
    # including assets in nested directories, so the identity cannot silently
    # survive a local-code or preprocessing mutation.
    for candidate in sorted(root.rglob("*")):
        try:
            candidate_info = candidate.lstat()
        except OSError as error:
            raise AggregateError(f"{label} checkpoint cannot be inspected") from error
        if stat.S_ISLNK(candidate_info.st_mode):
            if is_execution_asset(candidate.relative_to(root).as_posix()):
                raise AggregateError(
                    f"{label} checkpoint execution asset is symlinked: "
                    f"{candidate.relative_to(root).as_posix()}"
                )
            continue
        if not stat.S_ISREG(candidate_info.st_mode):
            continue
        relative_name = candidate.relative_to(root).as_posix()
        if not is_execution_asset(relative_name) or relative_name in critical:
            continue
        checked = _regular(
            candidate,
            label=f"{label} checkpoint file {relative_name}",
            trusted_root=root,
            max_bytes=512 * 1024 * 1024,
        )
        critical[relative_name] = {
            "bytes": checked.stat().st_size,
            "sha256": _sha256(checked),
        }
    if critical["config.json"] is None:
        raise AggregateError(f"{label} checkpoint is missing config.json")
    if not tokenizer_present:
        raise AggregateError(f"{label} checkpoint is missing tokenizer assets")

    material = {
        "schema_version": CHECKPOINT_IDENTITY_SCHEMA,
        "safetensors": safetensors,
        "critical_files": critical,
    }
    return {
        "schema_version": CHECKPOINT_IDENTITY_SCHEMA,
        "path": str(root),
        "safetensors": safetensors,
        "critical_files": critical,
        "safetensors_count": len(safetensors),
        "identity_sha256": hashlib.sha256(_canonical_json(material)).hexdigest(),
    }


def _load_expected_identity(path: Path, *, label: str) -> dict[str, Any]:
    try:
        info = path.lstat()
    except OSError as error:
        raise AggregateError(f"{label} identity is unavailable") from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise AggregateError(f"{label} identity must be a regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AggregateError(f"{label} identity cannot be parsed") from error
    if not isinstance(value, dict) or value.get("schema_version") != CHECKPOINT_IDENTITY_SCHEMA:
        raise AggregateError(f"{label} identity schema differs")
    if not isinstance(value.get("identity_sha256"), str):
        raise AggregateError(f"{label} identity digest is missing")
    return value


def _assert_below_without_symlinks(path: Path | str, root: Path, *, label: str) -> Path:
    candidate = _absolute(path)
    trusted = _absolute(root)
    try:
        relative = candidate.relative_to(trusted)
    except ValueError as error:
        raise AggregateError(f"{label} escapes its trusted root") from error
    try:
        root_info = trusted.lstat()
    except OSError as error:
        raise AggregateError(f"{label} trusted root is unavailable") from error
    if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode):
        raise AggregateError(f"{label} trusted root is not a real directory")
    cursor = trusted
    for component in relative.parts:
        cursor = cursor / component
        try:
            info = cursor.lstat()
        except FileNotFoundError:
            break
        except OSError as error:
            raise AggregateError(f"{label} cannot be inspected") from error
        if stat.S_ISLNK(info.st_mode):
            raise AggregateError(f"{label} contains a symlink")
    return candidate


def _regular(path: Path, *, label: str, trusted_root: Path, max_bytes: int = 256 * 1024 * 1024) -> Path:
    candidate = _assert_below_without_symlinks(path, trusted_root, label=label)
    try:
        info = candidate.lstat()
    except OSError as error:
        raise AggregateError(f"{label} is unavailable") from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise AggregateError(f"{label} must be a single-link regular file")
    if info.st_size <= 0 or info.st_size > max_bytes:
        raise AggregateError(f"{label} has an invalid size")
    return candidate


def _binding(path: Path, *, label: str, trusted_root: Path) -> dict[str, Any]:
    source = _regular(path, label=label, trusted_root=trusted_root)
    return {"path": str(_absolute(source)), "bytes": source.stat().st_size, "sha256": _sha256(source)}


def _validate_source_hashes(*, expected_aggregator_sha256: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for name, (relative, expected) in CRITICAL_SOURCE_HASHES.items():
        path = _regular(WORKSPACE / relative, label=f"source {name}", trusted_root=WORKSPACE)
        actual = _sha256(path)
        if actual != expected:
            raise AggregateError(f"source hash mismatch: {name}")
        result[name] = actual
    if re.fullmatch(r"[0-9a-f]{64}", expected_aggregator_sha256) is None:
        raise AggregateError("formal aggregator source hash is invalid")
    aggregator = _regular(
        Path(__file__), label="source formal_aggregator", trusted_root=WORKSPACE
    )
    actual_aggregator_sha256 = _sha256(aggregator)
    if actual_aggregator_sha256 != expected_aggregator_sha256:
        raise AggregateError("source hash mismatch: formal_aggregator")
    result["formal_aggregator"] = actual_aggregator_sha256
    orchestrator = _regular(
        WORKSPACE / "Codes/opd-qwen35/eval_tools/run_zoombench_formal_aggregate_v1.sh",
        label="source formal_orchestrator",
        trusted_root=WORKSPACE,
    )
    result["formal_orchestrator"] = _sha256(orchestrator)
    return result


def _validate_manifest(path: Path) -> dict[str, Any]:
    source = _regular(path, label="ZoomBench dataset manifest", trusted_root=WORKSPACE)
    if _absolute(source) != _absolute(DATASET_MANIFEST):
        raise AggregateError("dataset manifest is not the pinned ZoomBench manifest path")
    digest = _sha256(source)
    if digest != EXPECTED_MANIFEST_SHA256:
        raise AggregateError("dataset manifest hash differs from the pinned release")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AggregateError("dataset manifest cannot be parsed") from error
    if not isinstance(value, dict):
        raise AggregateError("dataset manifest must be a metadata object")
    dataset = value.get("dataset")
    if not isinstance(dataset, dict):
        # The generated manifest is the authoritative shape; this branch is
        # useful only for making malformed synthetic metadata fail closed.
        required = {"benchmark", "dataset_id", "dataset_revision", "rows"}
        if not required.issubset(value):
            raise AggregateError("dataset manifest metadata is incomplete")
        metadata = value
        dataset_id = value.get("dataset_id")
        revision = value.get("dataset_revision")
        rows = value.get("rows")
    else:
        metadata = value
        dataset_id = dataset.get("id")
        revision = dataset.get("revision")
        rows = dataset.get("expected_rows")
    if dataset_id != EXPECTED_DATASET_ID or revision != EXPECTED_DATASET_REVISION:
        raise AggregateError("dataset id or revision differs from pinned ZoomBench release")
    if rows != EXPECTED_TOTAL:
        raise AggregateError("dataset denominator differs from 845")
    if value.get("benchmark") not in (None, "ZoomBench"):
        raise AggregateError("dataset benchmark differs")
    if value.get("materialization_complete") is not True:
        raise AggregateError("dataset materialization is incomplete")
    if value.get("full_images") != EXPECTED_TOTAL:
        raise AggregateError("full-image count differs from 845")
    if value.get("source_parquet_sha256") != EXPECTED_SOURCE_PARQUET_SHA256:
        raise AggregateError("source parquet hash differs from pinned release")
    if value.get("official_eval_commit_audited") != EXPECTED_OFFICIAL_EVAL_COMMIT:
        raise AggregateError("official ZoomBench evaluator commit differs")
    if value.get("vision_opd_reference_commit") != EXPECTED_REFERENCE_COMMIT:
        raise AggregateError("Vision-OPD reference commit differs")
    if not str(value.get("primary_protocol", "")).startswith("full image only;"):
        raise AggregateError("ZoomBench primary protocol is not full-image-only")
    benchmark_json_value = value.get("benchmark_json")
    benchmark_json = DATASET_MANIFEST.parent / "zoombench.json"
    if not isinstance(benchmark_json_value, str) or _absolute(Path(benchmark_json_value)) != _absolute(benchmark_json):
        raise AggregateError("materialized ZoomBench JSON path differs from the pinned manifest")
    benchmark_json = _regular(
        benchmark_json,
        label="materialized ZoomBench JSON",
        trusted_root=WORKSPACE,
        max_bytes=4 * 1024 * 1024 * 1024,
    )
    benchmark_json_sha256 = value.get("benchmark_json_sha256")
    if not isinstance(benchmark_json_sha256, str) or len(benchmark_json_sha256) != 64:
        raise AggregateError("pinned benchmark JSON hash is missing")
    actual_benchmark_json_sha256 = _sha256(benchmark_json)
    if actual_benchmark_json_sha256 != benchmark_json_sha256:
        raise AggregateError("materialized ZoomBench JSON hash differs from manifest")
    # Only metadata and cryptographic bindings are returned.  Rows and images
    # are not parsed by this authority.
    return {
        "path": str(_absolute(source)),
        "bytes": source.stat().st_size,
        "sha256": digest,
        "dataset_id": dataset_id,
        "revision": revision,
        "expected_rows": rows,
        "source_parquet_sha256": value.get("source_parquet_sha256", EXPECTED_SOURCE_PARQUET_SHA256),
        "official_eval_commit": value.get("official_eval_commit_audited", EXPECTED_OFFICIAL_EVAL_COMMIT),
        "reference_commit": value.get("vision_opd_reference_commit", EXPECTED_REFERENCE_COMMIT),
        "full_image_only": "crop image is oracle diagnostic and excluded" in str(value.get("primary_protocol", "")),
        "benchmark_json": {
            "path": str(_absolute(benchmark_json)),
            "bytes": benchmark_json.stat().st_size,
            "sha256": actual_benchmark_json_sha256,
            "manifest_benchmark_json_sha256": benchmark_json_sha256,
        },
    }


def _validate_audit_root(path: Path) -> dict[str, Any]:
    root = _assert_below_without_symlinks(path, OUTPUT_ROOT, label="judge audit root")
    try:
        info = root.lstat()
    except OSError as error:
        raise AggregateError("judge audit root is unavailable") from error
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise AggregateError("judge audit root must be a real directory")
    children = []
    try:
        for item in root.iterdir():
            item_info = item.lstat()
            if stat.S_ISLNK(item_info.st_mode):
                raise AggregateError("judge audit root contains a symlink")
            if stat.S_ISDIR(item_info.st_mode):
                children.append(item.name)
    except OSError as error:
        raise AggregateError("judge audit root cannot be inspected") from error
    if len(children) != 1:
        raise AggregateError("judge audit root must contain exactly one matrix attempt")
    return {"path": str(root), "attempt_count": 1}


def _read_status_env(path: Path, *, trusted_root: Path, label: str) -> dict[str, str]:
    """Read only the small, shell-quoted lifecycle status metadata.

    The CPU seal uses this as an evidence gate.  It intentionally never walks
    or parses any answer/judge payload while checking that the two GPU phases
    really reached their durable ``complete`` states.
    """

    source = _regular(path, label=label, trusted_root=trusted_root, max_bytes=64 * 1024)
    values: dict[str, str] = {}
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise AggregateError(f"{label} cannot be read") from error
    for line in lines:
        if not line or "=" not in line:
            raise AggregateError(f"{label} contains malformed metadata")
        key, encoded = line.split("=", 1)
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) or key in values:
            raise AggregateError(f"{label} contains malformed metadata key")
        try:
            decoded = shlex.split(encoded, comments=False, posix=True)
        except ValueError as error:
            raise AggregateError(f"{label} contains malformed shell quoting") from error
        if len(decoded) != 1:
            raise AggregateError(f"{label} contains malformed metadata value")
        values[key] = decoded[0]
    if values.get("state") != "complete":
        raise AggregateError(f"{label} is not complete")
    return values


def _single_child_status(root: Path, *, child_name: str, label: str) -> tuple[Path, dict[str, str]]:
    """Locate exactly one ``<child>/status.env`` below a lifecycle root."""

    parent = _assert_below_without_symlinks(root / child_name, root, label=f"{label} lifecycle")
    try:
        parent_info = parent.lstat()
    except OSError as error:
        raise AggregateError(f"{label} lifecycle is unavailable") from error
    if stat.S_ISLNK(parent_info.st_mode) or not stat.S_ISDIR(parent_info.st_mode):
        raise AggregateError(f"{label} lifecycle is not a real directory")
    children: list[Path] = []
    try:
        for item in parent.iterdir():
            info = item.lstat()
            if stat.S_ISLNK(info.st_mode):
                raise AggregateError(f"{label} lifecycle contains a symlink")
            if stat.S_ISDIR(info.st_mode):
                children.append(item)
    except OSError as error:
        raise AggregateError(f"{label} lifecycle cannot be inspected") from error
    if len(children) != 1:
        raise AggregateError(f"{label} lifecycle must contain exactly one attempt")
    status = children[0] / "status.env"
    return status, _read_status_env(status, trusted_root=root, label=f"{label} status")


def _validate_resume_statuses(
    *,
    run_root: Path,
    audit_root: Path,
    model_id: str,
    model_path: Path,
    judge_model_id: str,
    judge_model_path: Path,
) -> None:
    """Fail closed unless both existing GPU lifecycles are complete."""

    candidate_status_path, candidate = _single_child_status(
        run_root, child_name="_runner", label="candidate"
    )
    del candidate_status_path
    expected_candidate = {
        "model_id": model_id,
        "model_path": str(_absolute(model_path)),
        "work_dir": str(_absolute(run_root)),
        "cuda_visible_devices": EXPECTED_CUDA_DEVICES,
        "tp": str(EXPECTED_CANDIDATE_TP),
        "dp": str(EXPECTED_CANDIDATE_DP),
        "profiles": "zoom-infer",
    }
    for key, expected in expected_candidate.items():
        if candidate.get(key) != expected:
            raise AggregateError(f"candidate status metadata differs: {key}")

    audit_attempts = _assert_below_without_symlinks(audit_root, OUTPUT_ROOT, label="audit root")
    try:
        children = []
        for item in audit_attempts.iterdir():
            info = item.lstat()
            if stat.S_ISLNK(info.st_mode):
                raise AggregateError("judge audit root contains a symlink")
            if stat.S_ISDIR(info.st_mode):
                children.append(item)
    except OSError as error:
        raise AggregateError("judge audit root cannot be inspected") from error
    if len(children) != 1:
        raise AggregateError("judge audit root must contain exactly one matrix attempt")
    judge_status = _read_status_env(
        children[0] / "status.env", trusted_root=audit_root, label="judge status"
    )
    expected_judge = {
        "judge_model_id": judge_model_id,
        "judge_model_path": str(_absolute(judge_model_path)),
        "cuda_visible_devices": EXPECTED_CUDA_DEVICES,
        "tp": str(EXPECTED_JUDGE_TP),
        "dp": str(EXPECTED_JUDGE_DP),
        "targets": "candidate",
    }
    for key, expected in expected_judge.items():
        if judge_status.get(key) != expected:
            raise AggregateError(f"judge status metadata differs: {key}")


def _validate_matrix_protocol(
    *,
    path: Path,
    run_root: Path,
    model_id: str,
    model_tag: str,
    judge_model_path: Path,
    judge_model_id: str,
    judge_port: int,
) -> dict[str, Any]:
    source = _regular(path, label="ZoomBench judge protocol", trusted_root=run_root)
    expected_path = _absolute(run_root) / "zoombench" / "judge_protocol.json"
    if _absolute(source) != expected_path:
        raise AggregateError("judge protocol is not the fixed candidate path")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AggregateError("judge protocol cannot be parsed") from error
    if not isinstance(value, dict):
        raise AggregateError("judge protocol must be metadata")
    if value.get("target_label") != "candidate":
        raise AggregateError("judge protocol target label differs")
    if value.get("target_model_id") != model_id or value.get("judge_model_id") != judge_model_id:
        raise AggregateError("judge protocol model identity differs")
    expected_judge_path = _absolute(judge_model_path)
    if _absolute(Path(str(value.get("judge_model_path", "")))) != expected_judge_path:
        raise AggregateError("judge protocol checkpoint differs")
    config = _regular(expected_judge_path / "config.json", label="judge config", trusted_root=WORKSPACE)
    if value.get("judge_config_sha256") != _sha256(config):
        raise AggregateError("judge config hash differs")
    if value.get("judge_port") != judge_port:
        raise AggregateError("judge port differs")
    if value.get("judge_cuda_visible_devices") != EXPECTED_CUDA_DEVICES:
        raise AggregateError("judge CUDA devices differ")
    if value.get("judge_tensor_parallel_size") != EXPECTED_JUDGE_TP or value.get("judge_data_parallel_size") != EXPECTED_JUDGE_DP:
        raise AggregateError("judge parallelism differs")
    if value.get("temperature") != 0 or value.get("enable_thinking") is not False:
        raise AggregateError("judge sampling protocol differs")
    if value.get("vision_opd_reference_commit") != EXPECTED_REFERENCE_COMMIT:
        raise AggregateError("judge reference commit differs")
    if value.get("protocol") != EXPECTED_JUDGE_PROTOCOL:
        raise AggregateError("judge protocol description differs")
    if re.fullmatch(r"[A-Za-z0-9._-]+_seed42", model_tag or "") is None:
        raise AggregateError("candidate model tag is invalid")
    protocol_binding = _binding(source, label="judge protocol", trusted_root=run_root)
    protocol_binding.update(
        {
            "target_label": "candidate",
            "target_model_id": model_id,
            "judge_model_id": judge_model_id,
            "cuda_visible_devices": EXPECTED_CUDA_DEVICES,
            "tensor_parallel_size": EXPECTED_JUDGE_TP,
            "data_parallel_size": EXPECTED_JUDGE_DP,
        }
    )
    return protocol_binding


def _metadata_only(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Retain only non-content row metadata while parsing private artifacts."""

    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in ORDER_METADATA_KEYS:
            if key in result:
                raise AggregateError(f"private row has duplicate metadata key: {key}")
            result[key] = value
    return result


def _judge_row_only(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = _metadata_only(pairs)
    for key, value in pairs:
        if key == "judge":
            if key in result:
                raise AggregateError("private judge row has duplicate judge labels")
            result[key] = value
    return result


def _row_order_digest(rows: Sequence[Mapping[str, Any]], *, label: str) -> str:
    """Bind ordered rows using a digest, without returning identifiers."""

    digest = hashlib.sha256()
    seen: set[str] = set()
    for position, row in enumerate(rows):
        metadata = {key: row[key] for key in ORDER_METADATA_KEYS if key in row}
        if not metadata:
            raise AggregateError(f"{label} row metadata is missing")
        token = hashlib.sha256(_canonical_json(metadata)).hexdigest()
        if token in seen:
            raise AggregateError(f"{label} row metadata is duplicated")
        seen.add(token)
        digest.update(position.to_bytes(8, "big"))
        digest.update(bytes.fromhex(token))
    return digest.hexdigest()


def _load_answer_order(path: Path, *, trusted_root: Path) -> tuple[dict[str, Any], str]:
    source = _regular(path, label="private ZoomBench answer checkpoint", trusted_root=trusted_root)
    if source.suffix.lower() != ".jsonl":
        raise AggregateError("private ZoomBench answer checkpoint must be JSONL")
    rows: list[dict[str, Any]] = []
    try:
        with source.open("r", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    raise AggregateError("private ZoomBench answer checkpoint contains a blank row")
                value = json.loads(line, object_pairs_hook=_metadata_only)
                if not isinstance(value, dict):
                    raise AggregateError("private ZoomBench answer row is not an object")
                rows.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AggregateError("private ZoomBench answer checkpoint cannot be parsed") from error
    if len(rows) != EXPECTED_TOTAL:
        raise AggregateError(f"private ZoomBench answer checkpoint must contain exactly {EXPECTED_TOTAL} rows")
    return (
        {
            "path": str(_absolute(source)),
            "bytes": source.stat().st_size,
            "sha256": _sha256(source),
            "rows": EXPECTED_TOTAL,
        },
        _row_order_digest(rows, label="private ZoomBench answer"),
    )


def aggregate_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Count only the official scalar ``judge`` labels."""

    if len(records) != EXPECTED_TOTAL:
        raise AggregateError(f"judge result must contain exactly {EXPECTED_TOTAL} rows")
    correct = 0
    for row in records:
        if not isinstance(row, Mapping):
            raise AggregateError("judge result rows must be objects")
        if "judge" not in row:
            raise AggregateError("judge label is missing")
        value = row.get("judge", "")
        if not isinstance(value, (str, int, float, bool)):
            raise AggregateError("judge label has an unsupported type")
        correct += int(str(value).strip().lower() == "yes")
    return {"correct": correct, "total": EXPECTED_TOTAL, "accuracy_percent": correct / EXPECTED_TOTAL * 100.0}


def _load_judge(
    path: Path, *, trusted_root: Path = OUTPUT_ROOT
) -> tuple[list[dict[str, Any]], str, str]:
    source = _regular(path, label="private ZoomBench judge result", trusted_root=trusted_root)
    if source.suffix.lower() not in {".jsonl", ".json"}:
        raise AggregateError("private judge result must be JSON or JSONL")
    try:
        raw = source.read_text(encoding="utf-8")
        try:
            value: Any = json.loads(raw, object_pairs_hook=_judge_row_only)
        except json.JSONDecodeError:
            if source.suffix.lower() != ".jsonl":
                raise AggregateError("private judge result is not a JSON array")
            value = [
                json.loads(line, object_pairs_hook=_judge_row_only)
                for line in raw.splitlines()
                if line.strip()
            ]
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AggregateError("private judge result cannot be parsed") from error
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise AggregateError("private judge result must be a list of objects")
    order_digest = _row_order_digest(value, label="private ZoomBench judge")
    # Metadata is used only for the private order binding above.  The
    # aggregation layer receives judge labels alone and cannot accidentally
    # propagate even hashed row metadata into a public result.
    judge_rows = [{"judge": row["judge"]} if "judge" in row else {} for row in value]
    return judge_rows, _sha256(source), order_digest


def _write_all(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        try:
            written = os.write(descriptor, payload[offset:])
        except OSError as error:
            raise AggregateError("aggregate authority write failed") from error
        if written <= 0 or written > len(payload) - offset:
            raise AggregateError("aggregate authority write made no progress")
        offset += written


def _write_create_once(path: Path, value: Mapping[str, Any]) -> None:
    target = _assert_below_without_symlinks(path, OUTPUT_ROOT, label="aggregate output")
    if target.exists() or target.is_symlink():
        raise AggregateError("aggregate authority already exists")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        # Re-check after mkdir: the lexical preflight intentionally allows the
        # final missing components, but an attacker must not substitute a
        # symlink while the parent is being materialized.
        target = _assert_below_without_symlinks(
            target, OUTPUT_ROOT, label="aggregate output"
        )
        parent = target.parent
        parent_info = parent.lstat()
        if stat.S_ISLNK(parent_info.st_mode) or not stat.S_ISDIR(parent_info.st_mode):
            raise AggregateError("aggregate output parent is not a real directory")
    except AggregateError:
        raise
    except OSError as error:
        raise AggregateError("aggregate authority parent is unavailable") from error

    encoded = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor: int | None = None
    temporary: str | None = None
    try:
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{target.name}.", dir=os.fspath(parent)
        )
        _write_all(descriptor, encoded)
        try:
            os.fsync(descriptor)
        except OSError as error:
            raise AggregateError("aggregate authority fsync failed") from error
        try:
            os.close(descriptor)
        except OSError as error:
            descriptor = None
            raise AggregateError("aggregate authority close failed") from error
        descriptor = None
        try:
            # link() is atomic and create-once: unlike rename/replace it cannot
            # overwrite a concurrently created authority.
            os.link(temporary, target, follow_symlinks=False)
        except FileExistsError as error:
            raise AggregateError("aggregate authority already exists") from error
        except OSError as error:
            raise AggregateError("aggregate authority could not be published") from error
        try:
            os.unlink(temporary)
        except OSError as error:
            raise AggregateError("aggregate authority temporary cleanup failed") from error
        temporary = None
    except AggregateError:
        raise
    except OSError as error:
        raise AggregateError("aggregate authority could not be written") from error
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        if temporary is not None:
            try:
                os.unlink(temporary)
            except OSError:
                pass


def build_receipt(
    *,
    model_id: str,
    model_tag: str,
    judge_model_id: str,
    cache_key: str,
    manifest: Mapping[str, Any],
    audit_root: Mapping[str, Any],
    protocol: Mapping[str, Any],
    candidate_identity: Mapping[str, Any],
    judge_identity: Mapping[str, Any],
    answer_binding: Mapping[str, Any],
    answer_order_digest: str,
    judge_path: Path,
    judge_sha256: str,
    judge_order_digest: str,
    evaluation: Mapping[str, Any],
    source_hashes: Mapping[str, str],
    orchestrator_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "scored",
        "benchmark": "ZoomBench",
        "model_id": model_id,
        "model_tag": model_tag,
        "dataset": dict(manifest),
        "candidate_inference": {
            "seed": 42,
            "enable_thinking": False,
            "temperature": 0,
            "max_tokens": EXPECTED_CANDIDATE_MAX_TOKENS,
            "input": "full image only",
        },
        "judge": {
            "model_id": judge_model_id,
            "temperature": 0,
            "enable_thinking": False,
            "max_tokens": EXPECTED_JUDGE_MAX_TOKENS,
            "matrix_protocol": dict(protocol),
        },
        "checkpoint_identity": {
            "candidate": dict(candidate_identity),
            "judge": dict(judge_identity),
        },
        "runtime": {
            "candidate_cuda_visible_devices": EXPECTED_CUDA_DEVICES,
            "candidate_tensor_parallel_size": EXPECTED_CANDIDATE_TP,
            "candidate_data_parallel_size": EXPECTED_CANDIDATE_DP,
            "judge_tensor_parallel_size": EXPECTED_JUDGE_TP,
            "judge_data_parallel_size": EXPECTED_JUDGE_DP,
            "serial_candidate_then_judge": True,
            "sample_level_output": False,
        },
        "cache": {"key": cache_key, "scope": "uid30853_private_zoombench_runtime"},
        "source_hashes": {**dict(source_hashes), "formal_orchestrator": orchestrator_sha256},
        "judge_artifact": {
            "path": str(_absolute(judge_path)),
            "rows": EXPECTED_TOTAL,
            "sha256": judge_sha256,
            "row_metadata_order_sha256": judge_order_digest,
        },
        "answer_artifact": {
            **dict(answer_binding),
            "row_metadata_order_sha256": answer_order_digest,
        },
        "audit_root": dict(audit_root),
        "evaluation": dict(evaluation),
        "authority": {
            "aggregate_only": True,
            "private_fields_discarded": True,
            "local_qwen_judge_caveat": (
                "The fixed local Qwen3.5 judge shares the Qwen family with the candidate/teacher; "
                "report this family/style bias and do not call it an independent external judge."
            ),
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge-json", type=Path)
    parser.add_argument("--model-id")
    parser.add_argument("--model-tag")
    parser.add_argument(
        "--judge-model-path",
        type=Path,
        default=WORKSPACE / "Ckpt" / "Qwen3.5-27B",
    )
    parser.add_argument("--judge-model-id", default="Qwen3.5-27B-ZoomJudge")
    parser.add_argument("--judge-port", default=18319, type=int)
    parser.add_argument("--cache-key", default="dry-run")
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--audit-root", type=Path)
    parser.add_argument(
        "--matrix-protocol",
        "--matrix-aggregate",
        dest="matrix_protocol",
        type=Path,
    )
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        default=DATASET_MANIFEST,
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--expected-candidate-identity", type=Path)
    parser.add_argument("--expected-judge-identity", type=Path)
    parser.add_argument("--checkpoint-identity", type=Path)
    parser.add_argument("--aggregator-sha256")
    parser.add_argument("--orchestrator-sha256")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--resume-seal",
        action="store_true",
        help="seal an existing complete candidate/judge lifecycle without GPU work",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.checkpoint_identity is not None:
            identity = _checkpoint_identity(args.checkpoint_identity, label="checkpoint")
            print(json.dumps(identity, ensure_ascii=False, sort_keys=True))
            return 0

        required = {
            "judge_json": args.judge_json,
            "model_tag": args.model_tag,
            "run_root": args.run_root,
            "audit_root": args.audit_root,
            "matrix_protocol": args.matrix_protocol,
            "output": args.output,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise AggregateError(f"missing required aggregate arguments: {', '.join(missing)}")
        if not args.dry_run and not args.model_id:
            raise AggregateError("candidate model id is required for execution")
        if args.model_id and re.fullmatch(r"[A-Za-z0-9._/+:-]+", args.model_id) is None:
            raise AggregateError("candidate model id is invalid")
        if re.fullmatch(r"[A-Za-z0-9._-]+_seed42", args.model_tag or "") is None:
            raise AggregateError("candidate model tag is invalid")
        if re.fullmatch(r"[A-Za-z0-9._/+:-]+", args.judge_model_id or "") is None:
            raise AggregateError("judge model id is invalid")
        if not args.cache_key or any(char.isspace() for char in args.cache_key):
            raise AggregateError("cache key is invalid")
        if not args.dry_run and re.fullmatch(r"[0-9a-f]{64}", args.aggregator_sha256 or "") is None:
            raise AggregateError("formal aggregator source hash is invalid")
        if not args.dry_run and re.fullmatch(r"[0-9a-f]{64}", args.orchestrator_sha256 or "") is None:
            raise AggregateError("formal orchestrator source hash is invalid")
        if args.judge_port < 1024 or args.judge_port > 65535:
            raise AggregateError("judge port is invalid")
        run_root = _assert_below_without_symlinks(args.run_root, OUTPUT_ROOT, label="run root")
        audit_root = _assert_below_without_symlinks(args.audit_root, OUTPUT_ROOT, label="audit root")
        output = _assert_below_without_symlinks(args.output, OUTPUT_ROOT, label="aggregate output")
        judge_path = _assert_below_without_symlinks(args.judge_json, OUTPUT_ROOT, label="judge JSON")
        # A dry-run intentionally has no run-root yet.  Check the prospective
        # path lexically below Output; execute mode additionally requires the
        # existing run-root itself to be the trusted ancestor.
        protocol_path = _assert_below_without_symlinks(
            args.matrix_protocol,
            OUTPUT_ROOT if args.dry_run else run_root,
            label="judge protocol",
        )
        manifest_path = _assert_below_without_symlinks(args.dataset_manifest, WORKSPACE, label="dataset manifest")
        expected_tag = args.model_id.replace("/", "_") + "_seed42" if args.model_id else None
        if expected_tag is not None and args.model_tag != expected_tag:
            raise AggregateError("candidate model tag must match the ZoomBench output path")
        expected_judge = run_root / "zoombench" / "judge" / "zoombench" / f"{args.model_tag}_answer.jsonl"
        if judge_path != expected_judge:
            raise AggregateError("judge JSON is not the fixed ZoomBench path")
        expected_output = run_root / "zoombench_formal_aggregate.json"
        if output != expected_output:
            raise AggregateError("aggregate output is not the fixed run-root path")
        if output.exists() or output.is_symlink():
            raise AggregateError("aggregate authority already exists")
        if args.dry_run:
            print(json.dumps({
                "aggregate_only": True,
                "dry_run": True,
                "reads_judge": False,
                "reads_manifest": False,
                "reads_protocol": False,
                "writes_output": False,
                "schema_version": SCHEMA_VERSION,
            }, sort_keys=True))
            return 0
        if os.geteuid() != EXPECTED_UID or os.getegid() != EXPECTED_UID:
            raise AggregateError("formal execution requires UID/GID 30853")
        if not run_root.is_dir():
            raise AggregateError("run root is unavailable")
        if args.model_path is None:
            raise AggregateError("candidate model path is required for execution")
        if args.expected_candidate_identity is None or args.expected_judge_identity is None:
            raise AggregateError("execute requires preflight checkpoint identity bindings")
        candidate_identity = _checkpoint_identity(args.model_path, label="candidate")
        judge_identity = _checkpoint_identity(args.judge_model_path, label="judge")
        expected_candidate_identity = _load_expected_identity(
            args.expected_candidate_identity, label="candidate"
        )
        expected_judge_identity = _load_expected_identity(
            args.expected_judge_identity, label="judge"
        )
        if candidate_identity != expected_candidate_identity:
            raise AggregateError("candidate checkpoint identity changed during lifecycle")
        if judge_identity != expected_judge_identity:
            raise AggregateError("judge checkpoint identity changed during lifecycle")
        source_hashes = _validate_source_hashes(
            expected_aggregator_sha256=args.aggregator_sha256
        )
        if source_hashes["formal_orchestrator"] != args.orchestrator_sha256:
            raise AggregateError("formal orchestrator source hash differs from live wrapper")
        manifest = _validate_manifest(manifest_path)
        audit = _validate_audit_root(audit_root)
        protocol = _validate_matrix_protocol(
            path=protocol_path,
            run_root=run_root,
            model_id=args.model_id,
            model_tag=args.model_tag,
            judge_model_path=args.judge_model_path,
            judge_model_id=args.judge_model_id,
            judge_port=args.judge_port,
        )
        if args.resume_seal:
            _validate_resume_statuses(
                run_root=run_root,
                audit_root=audit_root,
                model_id=args.model_id,
                model_path=args.model_path,
                judge_model_id=args.judge_model_id,
                judge_model_path=args.judge_model_path,
            )
        answer_path = run_root / "zoombench" / "model_answer" / "zoombench" / f"{args.model_tag}_answer.jsonl"
        answer_binding, answer_order_digest = _load_answer_order(answer_path, trusted_root=run_root)
        rows, judge_sha256, judge_order_digest = _load_judge(judge_path)
        if answer_order_digest != judge_order_digest:
            raise AggregateError("judge row metadata/order does not match candidate answer checkpoint")
        evaluation = aggregate_records(rows)
        receipt = build_receipt(
            model_id=args.model_id,
            model_tag=args.model_tag,
            judge_model_id=args.judge_model_id,
            cache_key=args.cache_key,
            manifest=manifest,
            audit_root=audit,
            protocol=protocol,
            candidate_identity=candidate_identity,
            judge_identity=judge_identity,
            answer_binding=answer_binding,
            answer_order_digest=answer_order_digest,
            judge_path=judge_path,
            judge_sha256=judge_sha256,
            judge_order_digest=judge_order_digest,
            evaluation=evaluation,
            source_hashes=source_hashes,
            orchestrator_sha256=args.orchestrator_sha256,
        )
        _write_create_once(output, receipt)
        print(
            "ZOOMBENCH_FORMAL_AGGREGATE "
            f"correct={evaluation['correct']} total={evaluation['total']} "
            f"accuracy_percent={evaluation['accuracy_percent']:.4f}"
        )
        return 0
    except AggregateError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
