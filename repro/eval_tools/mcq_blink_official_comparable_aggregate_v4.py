#!/usr/bin/env python3
"""BLINK v4 aggregate-only contract with an independent trust root.

The v4 entry point never trusts ambient imports.  Its compatibility reader is
loaded from one exact absolute path and its dependency is preloaded from one
exact absolute path.  The external trust-root manifest pins those files and
the v4 runner/artifact/server sources; the sealer additionally pins the trust
root's own SHA256, removing a runner/manifest self-reference loophole.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import stat
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


WORKSPACE = Path("/minimax-3d-rw-backup/users/jiazhi/H_Workspace")
OUTPUT_ROOT = WORKSPACE / "Output"
TOOLS_ROOT = WORKSPACE / "Codes/opd-qwen35/eval_tools"
V4_PATH = TOOLS_ROOT / "mcq_blink_official_comparable_aggregate_v4.py"
V3_PATH = TOOLS_ROOT / "mcq_blink_official_comparable_aggregate_v3.py"
V2_PATH = TOOLS_ROOT / "mcq_blink_official_comparable_aggregate_v2.py"
TRUST_ROOT_PATH = TOOLS_ROOT / "blink_v4_trust_root.json"
ARTIFACT_PATH = TOOLS_ROOT / "blink_v4_artifact.py"
RUNNER_PATH = TOOLS_ROOT / "run_blink_official_comparable_v4.sh"
SERVER_PATH = TOOLS_ROOT / "serve_qwen35_blink_formal_v4.sh"
SEALER_PATH = TOOLS_ROOT / "seal_blink_v4.py"
DATASET_PATH = WORKSPACE / "Dataset/eval/BLINK.tsv"
DATASET = "BLINK"
DATASET_ROWS = 1901
DATASET_MD5 = "d5e8af148b10ac69f535ff7b23f3f989"
DATASET_SHA256 = "fa2190bcb4e80d25af9ff778caae60ee02ae37b9d0f40bdbc3483cfdf328107c"
DATASET_TSV_SIZE = 538366044
DATASET_IMAGE_COUNT = 3675
DATASET_IMAGE_MANIFEST_SHA256 = "e5f03b4c37edfdeba24e8aa199dd3c517afd2fb0508515a7f5d09e950f09de09"
SCHEMA = "blink_official_comparable_aggregate_v4"
PROTOCOL_EXACT = "blink_vlmevalkit_exact_matching_official_comparable_v4"
MODEL_MANIFEST_SCHEMA = "blink_model_artifact_manifest_v4"
CUDA_VISIBLE_DEVICES = "0,1,2,3,4,5,6,7"
TP_SIZE = 8
DP_SIZE = 1
MAX_MODEL_LEN = 65536
MAX_NUM_SEQS = 4
CLIENT_WORKERS = 4
REQUEST_TIMEOUT = 3600.0
MAX_RETRIES = 0
SERVE_FACADE = SERVER_PATH.name
RUNTIME_ENV = "UV_Env/verl-opd-qwen35"
GPU_LOCK = WORKSPACE / "Locks/opd_gpu_0_7.lock"


class ContractError(RuntimeError):
    pass


def normalize_finish_reason(value: Any) -> str:
    if not isinstance(value, str):
        return "unknown"
    key = value.strip().casefold().replace("-", "_").replace(" ", "_")
    return {
        "stop": "stop", "eos": "stop", "completed": "stop",
        "length": "length", "max_tokens": "length", "max_completion_tokens": "length",
        "tool_calls": "tool_calls", "function_call": "tool_calls",
        "content_filter": "content_filter", "safety": "content_filter",
    }.get(key, "unknown")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    try:
        info = path.lstat()
    except OSError as exc:
        raise ContractError(f"missing fixed source: {path}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ContractError(f"fixed source is not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_fixed(name: str, path: Path) -> Any:
    """Load only the exact file named by the v4 trust boundary."""
    if Path(path).resolve() != path.absolute() or path.is_symlink() or not path.is_file():
        raise ContractError(f"fixed import path is not exact: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ContractError(f"cannot load fixed dependency: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    loaded_path = Path(str(getattr(module, "__file__", ""))).resolve()
    if loaded_path != path.resolve():
        raise ContractError(f"ambient import path mismatch: {name}")
    return module


def _fixed_v3() -> Any:
    if Path(__file__).resolve() != V4_PATH.resolve():
        raise ContractError("v4 aggregate __file__ is not the fixed path")
    # Preload the v2 dependency under its canonical name, then load v3 from
    # the fixed path.  v3's read-only reader consequently cannot be shadowed
    # by an ambient sys.modules entry or PYTHONPATH module.
    v2 = _load_fixed("mcq_blink_official_comparable_aggregate_v2", V2_PATH)
    return _load_fixed("_blink_v3_fixed_reader", V3_PATH)


def _trust_root() -> dict[str, Any]:
    if TRUST_ROOT_PATH.is_symlink() or not TRUST_ROOT_PATH.is_file():
        raise ContractError("v4 trust root is not a regular fixed file")
    try:
        value = json.loads(TRUST_ROOT_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError("v4 trust root is malformed") from exc
    if not isinstance(value, dict) or value.get("schema") != "blink_v4_trust_root":
        raise ContractError("v4 trust root schema differs")
    root_sha = value.get("manifest_sha256")
    body = {key: child for key, child in value.items() if key != "manifest_sha256"}
    if not isinstance(root_sha, str) or _sha_bytes(_canonical(body)) != root_sha:
        raise ContractError("v4 trust root self-seal mismatch")
    files = value.get("files")
    if not isinstance(files, dict) or not files:
        raise ContractError("v4 trust root file inventory is empty")
    for rel, expected in files.items():
        if not isinstance(rel, str) or not isinstance(expected, str) or len(expected) != 64:
            raise ContractError("v4 trust root source record is malformed")
        path = WORKSPACE / rel
        if _sha_file(path) != expected:
            raise ContractError(f"v4 trust root source hash mismatch: {rel}")
    return value


def source_identity() -> dict[str, Any]:
    root = _trust_root()
    v3 = _fixed_v3()
    # The loaded module's file identity is checked in _load_fixed; retain only
    # digests and fixed paths in the aggregate, never source text.
    return {
        "trust_root_sha256": root["manifest_sha256"],
        "trust_root_path": str(TRUST_ROOT_PATH),
        "fixed_v3_path": str(V3_PATH),
        "fixed_v2_path": str(V2_PATH),
        "aggregate_code_sha256": _sha_file(V4_PATH),
        "trust_root_files": dict(root["files"]),
        "reader_module_file": str(Path(v3.__file__).resolve()),
    }


def _read_samples(path: Path = DATASET_PATH) -> tuple[Any, ...]:
    reader = _fixed_v3()
    raw = reader._read_samples(path)
    if len(raw) != DATASET_ROWS:
        raise ContractError("BLINK row-count contract mismatch")
    return tuple(raw)


def _input_binding(path: Path, samples: Sequence[Any]) -> dict[str, Any]:
    reader = _fixed_v3()
    binding = dict(reader.dataset_binding(path, samples))
    # Bind the known official source identity independently of a self-authored
    # receipt.  The reader still verifies the row count and source MD5.
    if path.resolve() != DATASET_PATH.resolve():
        raise ContractError("BLINK v4 accepts only the fixed official TSV")
    if _sha_file(path) != DATASET_SHA256 or binding.get("tsv_size") != DATASET_TSV_SIZE:
        raise ContractError("official BLINK TSV identity mismatch")
    if binding.get("image_count") != DATASET_IMAGE_COUNT or binding.get("image_manifest_sha256") != DATASET_IMAGE_MANIFEST_SHA256:
        raise ContractError("official BLINK image-content identity mismatch")
    binding["tsv_md5"] = DATASET_MD5
    binding["tsv_sha256"] = DATASET_SHA256
    binding["schema"] = "blink_input_binding_v4"
    unsigned = {key: binding[key] for key in binding if key != "binding_sha256"}
    binding["binding_sha256"] = _sha_bytes(_canonical(unsigned))
    return binding


def _read_manifest(path: Path, expected: str) -> dict[str, Any]:
    artifact = _load_fixed("blink_v4_artifact_fixed", ARTIFACT_PATH)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError("model artifact manifest is unavailable") from exc
    if not isinstance(value, dict) or value.get("schema") != MODEL_MANIFEST_SCHEMA:
        raise ContractError("model artifact manifest schema differs")
    if artifact.verify_manifest(value) != expected:
        raise ContractError("model artifact manifest binding differs")
    return value


def _read_warmup(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError("warmup status is unavailable") from exc
    if not isinstance(value, dict) or value.get("required") is not True or value.get("status") != "passed" or value.get("scoring") is not False:
        raise ContractError("non-scoring multimodal warmup is not proven")
    return value


def aggregate(
    outcomes: Sequence[Any], *, model_id: str, model_artifact_sha256: str,
    artifact_manifest_path: str, input_binding: Mapping[str, Any],
    warmup: Mapping[str, Any], source: Mapping[str, Any],
) -> dict[str, Any]:
    if len(outcomes) != DATASET_ROWS:
        raise ContractError("aggregate requires exactly 1901 outcomes")
    if not isinstance(model_id, str) or not model_id:
        raise ContractError("model id is required")
    if not isinstance(model_artifact_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", model_artifact_sha256):
        raise ContractError("artifact manifest digest is required")
    if input_binding.get("schema") != "blink_input_binding_v4" or input_binding.get("tsv_md5") != DATASET_MD5 or input_binding.get("tsv_sha256") != DATASET_SHA256 or input_binding.get("tsv_size") != DATASET_TSV_SIZE or input_binding.get("image_count") != DATASET_IMAGE_COUNT or input_binding.get("image_manifest_sha256") != DATASET_IMAGE_MANIFEST_SHA256:
        raise ContractError("official BLINK input binding differs")
    if _sha_bytes(_canonical({key: input_binding[key] for key in input_binding if key != "binding_sha256"})) != input_binding.get("binding_sha256"):
        raise ContractError("input binding seal mismatch")
    if warmup.get("required") is not True or warmup.get("status") != "passed" or warmup.get("scoring") is not False:
        raise ContractError("warmup boundary is not satisfied")
    counts: dict[str, int] = {}
    finish: dict[str, int] = {}
    correct = 0
    for outcome in outcomes:
        prediction = str(outcome.prediction)
        counts[prediction] = counts.get(prediction, 0) + 1
        reason = normalize_finish_reason(getattr(outcome, "finish_reason", "unknown"))
        finish[reason] = finish.get(reason, 0) + 1
        correct += int(outcome.gold == outcome.prediction and prediction in {"A", "B", "C", "D"})
    body = {
        "schema": SCHEMA,
        "protocol_label": PROTOCOL_EXACT,
        "dataset": {"name": DATASET, "rows": DATASET_ROWS, "md5": DATASET_MD5, "tsv_sha256": DATASET_SHA256, "tsv_size": DATASET_TSV_SIZE, "image_count": DATASET_IMAGE_COUNT, "image_manifest_sha256": DATASET_IMAGE_MANIFEST_SHA256},
        "counts": {"total": len(outcomes), "correct": correct, "accuracy": correct / DATASET_ROWS, "prediction": counts, "finish_reason": finish},
        "candidate": {
            "model_id": model_id, "artifact_manifest_sha256": model_artifact_sha256, "artifact_manifest_path": artifact_manifest_path,
            "artifact_manifest_schema": MODEL_MANIFEST_SCHEMA, "cuda_visible_devices": CUDA_VISIBLE_DEVICES,
            "tensor_parallel_size": TP_SIZE, "data_parallel_size": DP_SIZE, "max_model_len": MAX_MODEL_LEN,
            "max_num_seqs": MAX_NUM_SEQS, "client_workers": CLIENT_WORKERS, "request_timeout_seconds": REQUEST_TIMEOUT,
            "max_retries": MAX_RETRIES, "serve_facade": SERVE_FACADE, "runtime_environment": RUNTIME_ENV,
        },
        "input_binding": dict(input_binding),
        "warmup": dict(warmup),
        "source_identity": dict(source),
        "aggregate_only": True, "request_logging": False, "raw_responses_saved": False,
        "sample_level_data_saved": False, "sample_persistence": "disabled",
    }
    body["seal_sha256"] = _sha_bytes(_canonical(body))
    return body


def verify_seal(receipt: Mapping[str, Any]) -> bool:
    seal = receipt.get("seal_sha256")
    if not isinstance(seal, str) or len(seal) != 64:
        return False
    body = {key: value for key, value in receipt.items() if key != "seal_sha256"}
    return _sha_bytes(_canonical(body)) == seal


def write_once(path: Path, receipt: Mapping[str, Any]) -> None:
    if not verify_seal(receipt) or path.exists() or path.is_symlink() or not path.parent.is_dir() or path.parent.is_symlink():
        raise ContractError("aggregate create-once/seal boundary failed")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(_canonical(receipt) + b"\n"); stream.flush(); os.fsync(stream.fileno())
    except Exception:
        try: path.unlink()
        except OSError: pass
        raise


def validate_run_root(run_root: Path, output: Path) -> None:
    root = Path(os.path.abspath(os.fspath(run_root)))
    target = Path(os.path.abspath(os.fspath(output)))
    current = Path(root.anchor or "/")
    for part in root.relative_to(current).parts:
        current /= part
        if current.is_symlink():
            raise ContractError("run-root ancestor is a symlink")
    current = Path(target.anchor or "/")
    for part in target.relative_to(current).parts:
        current /= part
        if current.is_symlink():
            raise ContractError("aggregate ancestor is a symlink")
    try:
        root.relative_to(OUTPUT_ROOT)
        target.relative_to(root)
    except ValueError as exc:
        raise ContractError("run-root/output escapes fixed Output root") from exc
    if root == OUTPUT_ROOT or target != root / "blink_official_comparable_aggregate.json":
        raise ContractError("run-root/output path is not fixed")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BLINK official-comparable aggregate-only v4")
    parser.add_argument("--blink-tsv", type=Path, default=DATASET_PATH)
    parser.add_argument("--candidate-api-base", required=True)
    parser.add_argument("--candidate-model-id", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact-manifest", type=Path, required=True)
    parser.add_argument("--model-artifact-sha256", required=True)
    parser.add_argument("--warmup-status", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.dry_run:
            print(json.dumps({"schema": SCHEMA, "dataset": DATASET, "rows": DATASET_ROWS, "tsv_size": DATASET_TSV_SIZE, "image_count": DATASET_IMAGE_COUNT, "image_manifest_sha256": DATASET_IMAGE_MANIFEST_SHA256, "protocol": PROTOCOL_EXACT,
                              "max_model_len": MAX_MODEL_LEN, "max_num_seqs": MAX_NUM_SEQS, "workers": CLIENT_WORKERS,
                              "timeout": REQUEST_TIMEOUT, "retry": MAX_RETRIES, "aggregate_only": True,
                              "trust_root": str(TRUST_ROOT_PATH), "reads_data": False, "writes_output": False}, sort_keys=True))
            return 0
        if args.output != args.run_root / "blink_official_comparable_aggregate.json":
            raise ContractError("aggregate output path is not fixed to run root")
        if args.status != args.run_root / "lifecycle/status.json":
            raise ContractError("status path is not fixed to run root")
        validate_run_root(args.run_root, args.output)
        if args.blink_tsv.resolve() != DATASET_PATH.resolve():
            raise ContractError("BLINK v4 dataset path is fixed")
        source = source_identity()
        _read_manifest(args.artifact_manifest, args.model_artifact_sha256)
        warmup = _read_warmup(args.warmup_status)
        samples = _read_samples(args.blink_tsv)
        binding = _input_binding(args.blink_tsv, samples)
        reader = _fixed_v3()
        outcomes = reader.evaluate_samples(samples, candidate_api_base=args.candidate_api_base, candidate_model_id=args.candidate_model_id, timeout=REQUEST_TIMEOUT)
        receipt = aggregate(outcomes, model_id=args.candidate_model_id, model_artifact_sha256=args.model_artifact_sha256,
                            artifact_manifest_path="lifecycle/model_artifact_manifest.json", input_binding=binding, warmup=warmup, source=source)
        write_once(args.output, receipt)
        print(json.dumps({"protocol_label": receipt["protocol_label"], "rows": receipt["counts"]["total"], "correct": receipt["counts"]["correct"], "seal_sha256": receipt["seal_sha256"]}, sort_keys=True))
        return 0
    except ContractError as exc:
        print(f"BLINK v4 contract failed closed: {exc}", file=sys.stderr)
        return 2
    except Exception:
        print("BLINK v4 contract failed closed: internal validation failure", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
