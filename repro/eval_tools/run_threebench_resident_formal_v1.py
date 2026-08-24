#!/usr/bin/env python3
"""Resident, aggregate-only coordinator for VStar/MMStar/ZoomBench.

The coordinator owns one candidate Qwen3.5-9B TP8 service and sends the three
candidate protocols to that same loopback endpoint in a fixed serial order.
The candidate service is stopped before the existing ZoomBench 27B judge
matrix is started.  Existing benchmark aggregators and judge scripts remain
the protocol authorities; this file is only a lifecycle/coordinator boundary.

``--dry-run`` is deliberately pure.  It renders a plan from constants and
arguments only: it does not inspect checkpoints, datasets, Output, locks or
source files, and it does not spawn a subprocess.  Execute mode is fail-closed
and create-once.  Public coordinator receipts contain only aggregate paths,
counts/hashes and protocol metadata; benchmark samples and raw responses stay
in the private child protocol lifecycle and are never printed by this file.
"""

from __future__ import annotations

import argparse
import contextlib
import errno
import fcntl
import hashlib
import json
import os
import re
import signal
import shutil
import stat
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "threebench_resident_formal_v1"
SERVER_MANIFEST_SCHEMA = "threebench_resident_server_manifest_v1"
RESULT_SCHEMA = "threebench_resident_result_v1"
STATUS_SCHEMA = "threebench_resident_status_v1"

REQUIRED_UID = 30853
REQUIRED_GID = 30853
GPU_IDS = tuple(range(8))
# Keep the literal visible in source and receipts: physical CUDA 0..7 only.
CUDA_DEVICES = "0,1,2,3,4,5,6,7"
CUDA_VISIBLE_DEVICES = CUDA_DEVICES
TP_SIZE = 8
DP_SIZE = 1
# ZoomBench's full-image prompts are the largest of the three candidate
# request families.  One value is advertised in the manifest and passed to
# vLLM for all three clients; request max_tokens remain benchmark-specific.
MAX_MODEL_LEN = 262144
SERVER_MAX_NUM_SEQS = 32
SERVER_GPU_MEMORY_UTILIZATION = "0.85"
MODEL_TAG_SUFFIX = "_seed42"

MMSTAR_MAX_TOKENS = 32768
MMSTAR_WORKERS = 32
VSTAR_MAX_TOKENS = 32768
ZOOMBENCH_MAX_TOKENS = 256
ZOOMBENCH_JUDGE_MAX_TOKENS = 32
ZOOMBENCH_JUDGE_TP = 2
ZOOMBENCH_JUDGE_DP = 4
ZOOMBENCH_JUDGE_MODEL_ID = "Qwen3.5-27B-ZoomJudge"
# Existing ZoomBench protocol: candidate TP8/DP1, then judge TP2/DP4, with
# the candidate service stopped between phases.  GPU ownership uses
# fcntl.flock(LOCK_EX|LOCK_NB) on ``opd_gpu_0_7.lock``.

LOCAL_HOST = "127.0.0.1"
DEFAULT_CANDIDATE_PORT = 18418
DEFAULT_JUDGE_PORT = 18419
READINESS_TIMEOUT_SECONDS = 1800
READINESS_POLL_SECONDS = 5

HEX64 = re.compile(r"^[0-9a-f]{64}$")
SAFE_MODEL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/+:-]{0,127}$")
SAFE_TAG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}_seed42$")
SAFE_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _workspace_root() -> Path:
    override = os.environ.get("OPD_QWEN35_WORKSPACE")
    if override:
        return Path(override)
    # .../H_Workspace/Codes/opd-qwen35/eval_tools/<this file>
    return Path(__file__).resolve().parents[3]


WORKSPACE_ROOT = _workspace_root()
OUTPUT_ROOT = WORKSPACE_ROOT / "Output"
TOOLS_ROOT = WORKSPACE_ROOT / "Codes/opd-qwen35/eval_tools"
REFERENCE_ROOT = WORKSPACE_ROOT / "Codes/Vision-OPD-reference"
LOCK_PATH = WORKSPACE_ROOT / "Locks/opd_gpu_0_7.lock"
SERVE_SCRIPT = TOOLS_ROOT / "serve_qwen35_formal_shared_cache_v1.sh"
# This is the architecture/runtime/backend cache namespace already pinned as
# the default by ``serve_qwen35_formal_shared_cache_v1.sh``.  Candidate
# weights, model IDs, HOME, benchmark artifacts, and result roots remain
# private per invocation; only compiled kernels and FlashInfer runtime cubins
# are shared.  The global GPU0-7 lease serializes every resident user of it.
SHARED_KERNEL_CACHE_ROOT = WORKSPACE_ROOT / (
    "Cache/vllm_formal_shared/"
    "mcc3a793f5ed8cb5098e120ed-rebdbf1149fa89d154c1c7304-flashinfer"
)
SHARED_KERNEL_CONFIG_ROOT = WORKSPACE_ROOT / "Cache/vllm_formal_shared/mmstar_base_seed42/home/.config"
SHARED_FLASHINFER_CACHE_ROOT = SHARED_KERNEL_CACHE_ROOT / "flashinfer"
SHARED_KERNEL_CACHE_KEY = hashlib.sha256(str(SHARED_KERNEL_CACHE_ROOT).encode()).hexdigest()[:32]
# This is the non-sensitive identity of the pre-warmed resident compile
# artifacts.  It deliberately excludes model paths, model IDs, weights and the
# full vLLM environment; those remain formal per-run provenance or may contain
# credentials/host-specific values.  The values are taken from the matching
# ``cache_key_factors.json`` entry under SHARED_KERNEL_CACHE_ROOT.
SHARED_KERNEL_COMPILE_IDENTITY: Mapping[str, Any] = {
    "architecture": "Qwen3_5ForConditionalGeneration",
    "code_hash": "81565e68b4091c872b3bd48f4275eb3ee8b671ce6633bd6b05ab7cee756dab28",
    "compiler_hash": "32492a154f",
    "config_hash": "2f5ac0ef64",
    "cuda_main_version": "12.9",
    "cuda_target_device": "cuda",
    "vllm_version": "0.18.0",
    "tensor_parallel_size": TP_SIZE,
    "data_parallel_size": DP_SIZE,
    "compile_backend": "inductor",
    "compile_cache_save_format": "binary",
    "compile_range": [1, 8192],
    "fuse_allreduce_rms": False,
    "flashinfer_autotune": False,
}
# The 27B ZoomBench judge is a different topology from the candidate service.
# Keep its compile/runtime state in a separate, fixed namespace so a candidate
# TP8/DP1 cache can never be mistaken for judge TP2/DP4 artifacts.  The matrix
# launcher currently rewrites XDG_CACHE_HOME/HF_HOME/TMPDIR inside its serving
# helper, therefore VLLM_CACHE_ROOT is also bound explicitly below; vLLM uses
# it as the authoritative torch-compile cache root.
JUDGE_CACHE_SCHEMA = "threebench_zoombench_judge_cache_v1"
# The v1 namespace is sealed against the previous judge launcher source hash.
# Keep it immutable for auditability and cold-start a new namespace after the
# launcher began propagating VLLM_ENGINE_READY_TIMEOUT_S=1200.
JUDGE_CACHE_ROOT = WORKSPACE_ROOT / (
    "Cache/vllm_formal_shared/zoombench_judge_qwen35_27b_tp2dp4_v2_timeout1200"
)
JUDGE_CACHE_CHILDREN = (
    "home",
    "xdg_cache",
    "vllm",
    "xdg_config",
    "flashinfer",
    "torchinductor",
    "triton",
    "cuda",
    "tmp",
)
JUDGE_CACHE_MANIFEST_NAME = "sealed_manifest.json"
JUDGE_MAX_MODEL_LEN = 32768
JUDGE_MAX_NUM_SEQS = 32
JUDGE_GPU_MEMORY_UTILIZATION = "0.85"
JUDGE_VLLM_VERSION = "0.18.0"
JUDGE_COMPILER_HASH = "32492a154f"
JUDGE_CODE_HASH = "81565e68b4091c872b3bd48f4275eb3ee8b671ce6633bd6b05ab7cee756dab28"
JUDGE_VLLM_CONFIG_HASH = "96be66b954"
JUDGE_CUDA_MAIN_VERSION = "12.9"
JUDGE_CUDA_TARGET_DEVICE = "cuda"
JUDGE_ARCHITECTURE = "Qwen3_5ForConditionalGeneration"
JUDGE_COMPILE_CONFIG: Mapping[str, Any] = {
    "pass_config": {"fuse_allreduce_rms": False},
}
JUDGE_KERNEL_CONFIG: Mapping[str, Any] = {
    "enable_flashinfer_autotune": False,
}
JUDGE_COMPILE_RANGE = [1, 8192]
MMSTAR_AGGREGATOR = TOOLS_ROOT / "mmstar_qwen35_modelcard_aggregate_v2.py"
VSTAR_AGGREGATOR = TOOLS_ROOT / "vstar_formal_aggregate_v1.py"
VSTAR_RUNNER = TOOLS_ROOT / "run_vision_opd_reference_eval.sh"
ZOOMBENCH_RUNNER = TOOLS_ROOT / "run_zoombench.sh"
ZOOMBENCH_PREPARER = TOOLS_ROOT / "prepare_zoombench.py"
ZOOMBENCH_VERIFIER = TOOLS_ROOT / "verify_zoombench_inference.py"
ZOOMBENCH_JUDGE_MATRIX = TOOLS_ROOT / "run_zoombench_judge_matrix.sh"
ZOOMBENCH_AGGREGATOR = TOOLS_ROOT / "zoombench_formal_aggregate_v1.py"
ZOOMBENCH_DATA_ROOT = WORKSPACE_ROOT / "Dataset/eval/ZoomBench"
MMSTAR_TSV = WORKSPACE_ROOT / "Dataset/eval/MMStar.tsv"
ZOOMBENCH_DATASET_MANIFEST = ZOOMBENCH_DATA_ROOT / "manifest.json"
OFFICIAL_QWEN35_9B_CHAT_TEMPLATE = WORKSPACE_ROOT / "Ckpt/Qwen3.5-9B/chat_template.jinja"
OFFICIAL_QWEN35_9B_CHAT_TEMPLATE_SHA256 = "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715"
PYTHON_BIN = WORKSPACE_ROOT / "UV_Env/verl-opd-qwen35/bin/python"
VLMEVAL_PYTHON = WORKSPACE_ROOT / "UV_Env/vlmevalkit-opd/bin/python"

# Source pins are intentionally literal, so dry-run does not need to inspect
# source files.  Execute rechecks every pin immediately before launching any
# protocol process.  These are the audited files used by the existing formal
# runners, not a second implementation of their benchmark semantics.
SOURCE_HASHES: Mapping[str, str] = {
    "serve_qwen35_formal_shared_cache_v1.sh": "336baaa33628905ef8fbf6928bf91c870b7f8cb93d331569f363adcaf0db9453",
    "mmstar_qwen35_modelcard_aggregate_v2.py": "514c8d09258186327b179d171b14555a548c10a79ce735c08d575629307940e5",
    "vstar_formal_aggregate_v1.py": "db452ae9f5d6f2b80dfc38b6e80d9d6efd25e8dac571e1aa0eb8a6d3b5ad3047",
    "run_vision_opd_reference_eval.sh": "98ea8a104cdc96e51ebc0e408f83cd47f39fcbd19512b3a9d55aa83358be0375",
    "run_zoombench.sh": "c103c1355b47e8a81b378b09164c8342a1b1a4ff41793dbe215e73efe9555ff0",
    "prepare_zoombench.py": "f60e4fc6255c6d4083933f706de835c1fd30c2d87e16502945af9d8625481ef8",
    "verify_zoombench_inference.py": "66b61ca08a383bec88feef28cfa270141dc4e2a6807abd31bd5e81b144c1033c",
    "run_zoombench_judge_matrix.sh": "dd3892e33d8f3e92dc41fa5a7388574098e1a288d965f62b7d0eefecdc2d979d",
    "run_zoombench_formal_aggregate_v1.sh": "e39ef08b8b7384a96d21dedbcb826d873449a992f8b2e970da6e61d0a5d50d45",
    "zoombench_formal_aggregate_v1.py": "4b5fd3683cc42cfd2818bd5b47fda710c6f33072d0218c50cf5f3051c04b142f",
    "vision_opd_reference_infer.py": "bb379999932658907196cdc98d22c60d63e3308cb5a867317481c4a85af70374",
    "vision_opd_reference_judge.py": "abbe11dacf7fae19728ca16407a02c91d04a9bc8ea72edd3b4a91b6224f4b670",
    "vision_opd_reference_cal_acc.py": "695dbddc3e63a1b9f8971c0d414d963a5da94776863d58589feaa4a1c6b0f025",
}


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    slug: str
    protocol_hash: str
    max_tokens: int
    enable_thinking: bool
    authority_name: str
    authority_schema: str


BENCHMARKS: tuple[BenchmarkSpec, ...] = (
    BenchmarkSpec(
        "VStar",
        "vstar",
        # This is the hash used by the existing formal aggregate protocol.
        "55fe9e9013c5a38e20e29446f07106ef9ab482be0f79332676aef9a4bfa07d98",
        VSTAR_MAX_TOKENS,
        False,
        "formal_aggregate.json",
        "vstar_formal_aggregate_v1",
    ),
    BenchmarkSpec(
        "MMStar",
        "mmstar",
        "3d1baa4687ad3b5607cd622d0fef4e88f60f0f5c14bbe54d7b0ce0d1de221c17",
        MMSTAR_MAX_TOKENS,
        True,
        "mmstar_qwen35_modelcard_formal_aggregate.json",
        "mmstar_qwen35_modelcard_aggregate_v2",
    ),
    BenchmarkSpec(
        "ZoomBench",
        "zoombench",
        "de79d40ac9916300db8a139a851727bf6bcb4fb016e9c659bf77609f9cb19f5a",
        ZOOMBENCH_MAX_TOKENS,
        False,
        "zoombench_formal_aggregate.json",
        "zoombench_formal_aggregate_v1",
    ),
)
BENCHMARK_BY_SLUG = {item.slug: item for item in BENCHMARKS}


class CoordinatorError(RuntimeError):
    """An integrity, lifecycle or protocol failure."""


class _ResidentSignal(CoordinatorError):
    """A process signal converted into the normal fail-closed error path."""

    def __init__(self, signum: int) -> None:
        self.signum = signum
        super().__init__(f"resident coordinator interrupted by signal {signum}")


class _ExecutionState:
    """Live process/receipt state shared with the top-level signal handler.

    Every subprocess started by the coordinator owns a new process group.  A
    signal handler cannot safely perform blocking cleanup itself, so it raises
    ``_ResidentSignal`` and lets the ordinary ``_execute`` exception/finally
    path stop all registered groups and write the failed lifecycle receipt.
    """

    def __init__(self) -> None:
        self.lifecycle: Path | None = None
        self.model_id: str | None = None
        self.manifest_sha256: str | None = None
        self.processes: list[subprocess.Popen[bytes]] = []
        self.cleaning = False
        self.finalizing = False
        self.signal_number: int | None = None

    def register(self, process: subprocess.Popen[bytes]) -> None:
        if process not in self.processes:
            self.processes.append(process)

    def unregister(self, process: subprocess.Popen[bytes] | None) -> None:
        if process is None:
            return
        self.processes = [item for item in self.processes if item is not process]

    def cleanup(self) -> None:
        if self.cleaning:
            return
        self.cleaning = True
        # Preserve registration order (candidate first, then protocol/judge
        # children), while tolerating a child being removed during teardown.
        for process in list(self.processes):
            _stop_process(process)
        self.processes.clear()


_ACTIVE_EXECUTION: _ExecutionState | None = None


def _register_process(process: subprocess.Popen[bytes]) -> None:
    if _ACTIVE_EXECUTION is not None:
        _ACTIVE_EXECUTION.register(process)


def _unregister_process(process: subprocess.Popen[bytes] | None) -> None:
    if _ACTIVE_EXECUTION is not None:
        _ACTIVE_EXECUTION.unregister(process)


def _resident_signal_handler(signum: int, _frame: Any) -> None:
    """Convert HUP/TERM/INT into a cleanup-aware exception.

    Once cleanup or the final publication section has begun, additional
    signals are ignored so they cannot interrupt process-group teardown or
    leave a half-written lifecycle receipt.
    """

    state = _ACTIVE_EXECUTION
    if state is not None:
        if state.signal_number is not None or state.cleaning or state.finalizing:
            return
        state.signal_number = signum
    raise _ResidentSignal(signum)


def _install_resident_signal_handlers() -> dict[int, Any]:
    previous: dict[int, Any] = {}
    for signum in (signal.SIGHUP, signal.SIGTERM, signal.SIGINT):
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, _resident_signal_handler)
    return previous


def _restore_resident_signal_handlers(previous: Mapping[int, Any]) -> None:
    for signum, handler in previous.items():
        signal.signal(signum, handler)


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha_text(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        info = path.lstat()
    except OSError as exc:
        raise CoordinatorError(f"source is unavailable: {path}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise CoordinatorError(f"source must be a regular file: {path}")
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise CoordinatorError(f"source cannot be read: {path}") from exc
    return digest.hexdigest()


def _source_paths() -> Mapping[str, Path]:
    return {
        "serve_qwen35_formal_shared_cache_v1.sh": SERVE_SCRIPT,
        "mmstar_qwen35_modelcard_aggregate_v2.py": MMSTAR_AGGREGATOR,
        "vstar_formal_aggregate_v1.py": VSTAR_AGGREGATOR,
        "run_vision_opd_reference_eval.sh": VSTAR_RUNNER,
        "run_zoombench.sh": ZOOMBENCH_RUNNER,
        "prepare_zoombench.py": ZOOMBENCH_PREPARER,
        "verify_zoombench_inference.py": ZOOMBENCH_VERIFIER,
        "run_zoombench_judge_matrix.sh": ZOOMBENCH_JUDGE_MATRIX,
        "run_zoombench_formal_aggregate_v1.sh": TOOLS_ROOT / "run_zoombench_formal_aggregate_v1.sh",
        "zoombench_formal_aggregate_v1.py": ZOOMBENCH_AGGREGATOR,
        "vision_opd_reference_infer.py": REFERENCE_ROOT / "eval/infer.py",
        "vision_opd_reference_judge.py": REFERENCE_ROOT / "eval/judge_qwenlm.py",
        "vision_opd_reference_cal_acc.py": REFERENCE_ROOT / "eval/cal_acc.py",
    }


def verify_source_pins() -> dict[str, str]:
    actual: dict[str, str] = {}
    for name, expected in SOURCE_HASHES.items():
        path = _source_paths()[name]
        digest = sha_file(path)
        if digest != expected:
            raise CoordinatorError(f"source hash mismatch: {name}")
        actual[name] = digest
    return actual


def endpoint(port: int) -> str:
    if not isinstance(port, int) or not 1024 <= port <= 65535:
        raise CoordinatorError("port must be in [1024, 65535]")
    return f"http://{LOCAL_HOST}:{port}/v1"


def _lexical_absolute(value: str | Path) -> Path:
    # Unlike Path.resolve(), this intentionally does not inspect the target or
    # follow symlinks.  It is safe for the pure dry-run path.
    return Path(os.path.abspath(os.fspath(value)))


def _under(path: Path, parent: Path, *, label: str) -> None:
    try:
        path.relative_to(parent)
    except ValueError as exc:
        raise CoordinatorError(f"{label} must be below {parent}: {path}") from exc
    if path == parent:
        raise CoordinatorError(f"{label} cannot be {parent}")


def model_tag_for(model_id: str) -> str:
    if SAFE_MODEL_ID.fullmatch(model_id) is None:
        raise CoordinatorError("model-id is unsafe")
    tag = model_id.replace("/", "_") + MODEL_TAG_SUFFIX
    if SAFE_TAG.fullmatch(tag) is None:
        raise CoordinatorError("derived model tag is unsafe")
    return tag


def request_protocols() -> dict[str, dict[str, Any]]:
    """Return immutable request-level controls advertised in the manifest."""

    return {
        "vstar": {
            "enable_thinking": False,
            "temperature": 0,
            "max_tokens": VSTAR_MAX_TOKENS,
            "seed": 42,
            "reasoning_parser": None,
            "gold_scope": "frozen_scorer_internal_only",
            "sample_level_output": False,
        },
        "mmstar": {
            "enable_thinking": True,
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 1.5,
            "repetition_penalty": 1.0,
            "max_tokens": MMSTAR_MAX_TOKENS,
            "seed": 0,
            "reasoning_parser": "qwen3",
            "response_source": "content_only",
            "sample_level_output": False,
        },
        "zoombench": {
            "enable_thinking": False,
            "temperature": 0,
            "max_tokens": ZOOMBENCH_MAX_TOKENS,
            "seed": 42,
            "reasoning_parser": None,
            "input": "full image only",
            "sample_level_output": False,
        },
    }


def build_server_manifest(
    *,
    model_path: str | Path,
    model_id: str,
    model_tag: str,
    candidate_port: int,
    cache_key: str,
    source_hashes: Mapping[str, str],
) -> dict[str, Any]:
    """Build a self-sealed, sample-free shared candidate server manifest."""

    if SAFE_MODEL_ID.fullmatch(model_id) is None:
        raise CoordinatorError("model-id is unsafe")
    if SAFE_TAG.fullmatch(model_tag) is None or model_tag != model_tag_for(model_id):
        raise CoordinatorError("model-tag must be the exact model-id_seed42 tag")
    body: dict[str, Any] = {
        "schema_version": SERVER_MANIFEST_SCHEMA,
        "coordinator_schema": SCHEMA_VERSION,
        "candidate": {
            "model_id": model_id,
            "model_tag": model_tag,
            "model_path": str(_lexical_absolute(model_path)),
        },
        "endpoint": {
            "base_url": endpoint(candidate_port),
            "host": LOCAL_HOST,
            "port": candidate_port,
            "served_model_id": model_id,
        },
        "hardware": {
            "cuda_visible_devices": CUDA_VISIBLE_DEVICES,
            "gpu_ids": list(GPU_IDS),
            "tensor_parallel_size": TP_SIZE,
            "data_parallel_size": DP_SIZE,
            "max_model_len": MAX_MODEL_LEN,
            "max_num_seqs": SERVER_MAX_NUM_SEQS,
        },
        "server": {
            "backend": "vllm",
            "serve_script": SERVE_SCRIPT.name,
            "default_enable_thinking": False,
            "reasoning_parser": "qwen3",
            "chat_template": {
                "path": str(OFFICIAL_QWEN35_9B_CHAT_TEMPLATE),
                "sha256": OFFICIAL_QWEN35_9B_CHAT_TEMPLATE_SHA256,
                "purpose": "official_qwen35_inference_template_override",
            },
            "gpu_memory_utilization": SERVER_GPU_MEMORY_UTILIZATION,
            "cache_key": cache_key,
            "shared_kernel_cache": {
                "key": SHARED_KERNEL_CACHE_KEY,
                "scope": "uid_private_architecture_runtime_backend_only",
                "compile_root": str(SHARED_KERNEL_CACHE_ROOT),
                "xdg_config_home": str(SHARED_KERNEL_CONFIG_ROOT),
                "flashinfer_workspace_base": str(SHARED_FLASHINFER_CACHE_ROOT),
                "compile_identity": dict(SHARED_KERNEL_COMPILE_IDENTITY),
                "serialized_by_gpu_lock": True,
                "contains_model_weights": False,
                "contains_benchmark_samples": False,
            },
        },
        "request_protocols": request_protocols(),
        "benchmarks_serial_order": [item.slug for item in BENCHMARKS],
        "aggregate_only": True,
        "request_logging": False,
        "raw_responses_saved": False,
        "sample_level_data_saved": False,
        "sample_persistence": "disabled",
        "source_hashes": dict(source_hashes),
    }
    body["manifest_sha256"] = sha_text(body)
    return body


def validate_server_manifest(value: Mapping[str, Any]) -> str:
    if value.get("schema_version") != SERVER_MANIFEST_SCHEMA:
        raise CoordinatorError("shared server manifest schema differs")
    seal = value.get("manifest_sha256")
    body = {key: item for key, item in value.items() if key != "manifest_sha256"}
    if not isinstance(seal, str) or not HEX64.fullmatch(seal) or sha_text(body) != seal:
        raise CoordinatorError("shared server manifest seal differs")
    endpoint_info = value.get("endpoint")
    if not isinstance(endpoint_info, Mapping) or endpoint_info.get("host") != LOCAL_HOST:
        raise CoordinatorError("shared endpoint is not loopback")
    candidate = value.get("candidate")
    port = endpoint_info.get("port") if isinstance(endpoint_info, Mapping) else None
    if not isinstance(candidate, Mapping) or not isinstance(port, int) or endpoint_info.get("base_url") != endpoint(port) or endpoint_info.get("served_model_id") != candidate.get("model_id"):
        raise CoordinatorError("shared endpoint/model binding differs")
    server = value.get("server")
    expected_kernel_cache = {
        "key": SHARED_KERNEL_CACHE_KEY,
        "scope": "uid_private_architecture_runtime_backend_only",
        "compile_root": str(SHARED_KERNEL_CACHE_ROOT),
        "xdg_config_home": str(SHARED_KERNEL_CONFIG_ROOT),
        "flashinfer_workspace_base": str(SHARED_FLASHINFER_CACHE_ROOT),
        "compile_identity": dict(SHARED_KERNEL_COMPILE_IDENTITY),
        "serialized_by_gpu_lock": True,
        "contains_model_weights": False,
        "contains_benchmark_samples": False,
    }
    if not isinstance(server, Mapping) or server.get("shared_kernel_cache") != expected_kernel_cache:
        raise CoordinatorError("shared kernel cache binding differs")
    expected_chat_template = {
        "path": str(OFFICIAL_QWEN35_9B_CHAT_TEMPLATE),
        "sha256": OFFICIAL_QWEN35_9B_CHAT_TEMPLATE_SHA256,
        "purpose": "official_qwen35_inference_template_override",
    }
    if server.get("chat_template") != expected_chat_template:
        raise CoordinatorError("official inference chat template binding differs")
    hardware = value.get("hardware")
    if not isinstance(hardware, Mapping) or hardware.get("cuda_visible_devices") != CUDA_VISIBLE_DEVICES or hardware.get("gpu_ids") != list(GPU_IDS) or hardware.get("tensor_parallel_size") != TP_SIZE or hardware.get("data_parallel_size") != DP_SIZE or hardware.get("max_model_len") != MAX_MODEL_LEN:
        raise CoordinatorError("shared server hardware contract differs")
    if value.get("benchmarks_serial_order") != [item.slug for item in BENCHMARKS]:
        raise CoordinatorError("resident benchmark order differs")
    if value.get("aggregate_only") is not True or value.get("request_logging") is not False or value.get("raw_responses_saved") is not False or value.get("sample_level_data_saved") is not False or value.get("sample_persistence") != "disabled":
        raise CoordinatorError("shared server privacy boundary differs")
    return seal


def build_result(
    *,
    model_id: str,
    model_tag: str,
    run_root: Path,
    audit_root: Path | None = None,
    manifest: Mapping[str, Any],
    authorities: Mapping[str, Mapping[str, Any]],
    judge: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the public coordinator result without copying benchmark content."""

    if model_tag != model_tag_for(model_id):
        raise CoordinatorError("result model tag does not match exact model id")
    manifest_sha = validate_server_manifest(manifest)
    clean_authorities: dict[str, dict[str, Any]] = {}
    for bench in BENCHMARKS:
        record = authorities.get(bench.slug)
        if not isinstance(record, Mapping):
            raise CoordinatorError(f"missing {bench.name} authority record")
        if record.get("schema_version") != bench.authority_schema:
            raise CoordinatorError(f"{bench.name} authority schema differs")
        clean_authorities[bench.slug] = {
            "status": "complete",
            "authority_path": str(record.get("authority_path", "")),
            "authority_sha256": record.get("authority_sha256"),
            "schema_version": record.get("schema_version"),
            "protocol_hash": bench.protocol_hash,
            "shared_server_manifest_sha256": manifest_sha,
            "sample_artifacts_removed": record.get("sample_artifacts_removed") is True,
            "removed_artifacts": list(record.get("removed_artifacts", [])),
        }
        if record.get("sample_artifacts_removed") is not True or not isinstance(record.get("removed_artifacts", []), list):
            raise CoordinatorError(f"{bench.name} sample artifact cleanup was not certified")
        if not isinstance(clean_authorities[bench.slug]["authority_sha256"], str) or not HEX64.fullmatch(clean_authorities[bench.slug]["authority_sha256"]):
            raise CoordinatorError(f"{bench.name} authority hash is malformed")
    clean_judge = {
        "status": "complete",
        "model_id": judge.get("model_id"),
        "model_path": judge.get("model_path"),
        "tensor_parallel_size": ZOOMBENCH_JUDGE_TP,
        "data_parallel_size": ZOOMBENCH_JUDGE_DP,
        "authority_path": judge.get("authority_path"),
        "authority_sha256": judge.get("authority_sha256"),
        "shared_server_manifest_sha256": manifest_sha,
        "sample_artifacts_removed": judge.get("sample_artifacts_removed") is True,
        "removed_artifacts": list(judge.get("removed_artifacts", [])),
    }
    expected_judge_path = str(WORKSPACE_ROOT / "Ckpt/Qwen3.5-27B")
    if judge.get("model_id") != ZOOMBENCH_JUDGE_MODEL_ID or judge.get("model_path") != expected_judge_path:
        raise CoordinatorError("ZoomBench judge model identity differs")
    if judge.get("sample_artifacts_removed") is not True or not isinstance(judge.get("removed_artifacts", []), list):
        raise CoordinatorError("ZoomBench judge artifact cleanup was not certified")
    queue_audit_root = audit_root or run_root.with_name(run_root.name + "_audit")
    queue_target = {
        "model_id": model_id,
        "model_tag": model_tag,
        "model_path": str(manifest["candidate"]["model_path"]),
    }
    queue_mapping = {
        "vstar": {
            **queue_target,
            "root": str(run_root / "vstar"),
            "aggregate_path": str(run_root / "vstar" / BENCHMARK_BY_SLUG["vstar"].authority_name),
            "authority_sha256": clean_authorities["vstar"]["authority_sha256"],
            "protocol_hash": BENCHMARK_BY_SLUG["vstar"].protocol_hash,
        },
        "mmstar": {
            **queue_target,
            "root": str(run_root / "mmstar"),
            "aggregate_path": str(run_root / "mmstar" / BENCHMARK_BY_SLUG["mmstar"].authority_name),
            "status_env": str(run_root / "mmstar" / "lifecycle/status.env"),
            "authority_sha256": clean_authorities["mmstar"]["authority_sha256"],
            "protocol_hash": BENCHMARK_BY_SLUG["mmstar"].protocol_hash,
        },
        "zoombench": {
            **queue_target,
            "root": str(run_root),
            "audit_root": str(queue_audit_root),
            "aggregate_path": str(run_root / BENCHMARK_BY_SLUG["zoombench"].authority_name),
            "candidate_status_env": str(run_root / "_runner/resident_candidate/status.env"),
            "judge_status_root": str(queue_audit_root),
            "authority_sha256": clean_authorities["zoombench"]["authority_sha256"],
            "protocol_hash": BENCHMARK_BY_SLUG["zoombench"].protocol_hash,
        },
    }
    return {
        "schema_version": RESULT_SCHEMA,
        "status": "complete",
        "candidate_model_id": model_id,
        "candidate_model_tag": model_tag,
        "run_root": str(run_root),
        "shared_server_manifest_path": "lifecycle/shared_server_manifest.json",
        "shared_server_manifest_sha256": manifest_sha,
        "benchmarks": clean_authorities,
        "zoom_judge": clean_judge,
        "resident_candidate_server": {
            "cuda_visible_devices": CUDA_VISIBLE_DEVICES,
            "tensor_parallel_size": TP_SIZE,
            "data_parallel_size": DP_SIZE,
            "max_model_len": MAX_MODEL_LEN,
            "endpoint": endpoint(int(manifest["endpoint"]["port"])),
            "stopped_before_judge": True,
        },
        "aggregate_only": True,
        "request_logging": False,
        "raw_responses_saved": False,
        "sample_level_data_saved": False,
        "sample_persistence": "disabled",
        "sample_artifacts_removed": True,
        # This is a read-only hand-off description for a later gate/queue.
        # It does not alter or invoke that queue and carries no sample data.
        "fourbench_queue_mapping": queue_mapping,
        "fourbench_queue_missing_benchmarks": ["blink"],
    }


def render_plan(
    *,
    model_path: str,
    model_id: str,
    run_root: str,
    audit_root: str,
    candidate_port: int,
    judge_port: int,
) -> list[dict[str, Any]]:
    """Render pure aggregate-only plan records for dry-run and CPU tests."""

    model_tag = model_tag_for(model_id)
    root = _lexical_absolute(run_root)
    audit = _lexical_absolute(audit_root)
    rows: list[dict[str, Any]] = []
    for index, bench in enumerate(BENCHMARKS, start=1):
        rows.append(
            {
                "sequence": index,
                "benchmark": bench.name,
                "slug": bench.slug,
                "model_id": model_id,
                "model_tag": model_tag,
                "model_path": str(_lexical_absolute(model_path)),
                "candidate_endpoint": endpoint(candidate_port),
                "candidate_server": "one resident TP8 service",
                "candidate_thinking": bench.enable_thinking,
                "request_reasoning_parser": "qwen3" if bench.slug == "mmstar" else None,
                "max_tokens": bench.max_tokens,
                "max_model_len": MAX_MODEL_LEN,
                "physical_cuda": CUDA_VISIBLE_DEVICES,
                "tensor_parallel_size": TP_SIZE,
                "data_parallel_size": DP_SIZE,
                "gpu_lock": str(LOCK_PATH),
                "run_root": str(root / bench.slug),
                # ZoomBench's fixed aggregator publishes at WORK_DIR root;
                # its private candidate/judge artifacts live below
                # WORK_DIR/zoombench.
                "authority": str(root / bench.authority_name) if bench.slug == "zoombench" else str(root / bench.slug / bench.authority_name),
                "status": "pending",
                "aggregate_only": True,
                "sample_level_output": False,
            }
        )
    rows.append(
        {
            "sequence": 4,
            "phase": "zoom_judge",
            "benchmark": "ZoomBench",
            "judge_model_id": ZOOMBENCH_JUDGE_MODEL_ID,
            "judge_endpoint": endpoint(judge_port),
            "physical_cuda": CUDA_VISIBLE_DEVICES,
            "judge_tensor_parallel_size": ZOOMBENCH_JUDGE_TP,
            "judge_data_parallel_size": ZOOMBENCH_JUDGE_DP,
            "candidate_server_stopped": True,
            "audit_root": str(audit),
            "aggregate_only": True,
            "sample_level_output": False,
        }
    )
    return rows


def _write_create_once(path: Path, value: Mapping[str, Any], *, mode: int = 0o600) -> None:
    if path.exists() or path.is_symlink():
        raise CoordinatorError(f"create-once path already exists: {path}")
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    if parent.is_symlink() or not parent.is_dir():
        raise CoordinatorError(f"output parent is unsafe: {parent}")
    encoded = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    descriptor: int | None = None
    try:
        descriptor = os.open(os.fspath(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, mode)
        offset = 0
        while offset < len(encoded):
            count = os.write(descriptor, encoded[offset:])
            if count <= 0:
                raise CoordinatorError(f"write made no progress: {path}")
            offset += count
        os.fsync(descriptor)
    except FileExistsError as exc:
        raise CoordinatorError(f"create-once path already exists: {path}") from exc
    except OSError as exc:
        raise CoordinatorError(f"cannot write {path}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    os.chmod(path, mode)


def _write_status(path: Path, value: Mapping[str, Any]) -> None:
    """Atomically update tiny status metadata; never include sample fields."""

    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise CoordinatorError(f"status path is unsafe: {path}")
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise CoordinatorError(f"status parent is unsafe: {path.parent}")
    encoded = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor: int | None = None
    try:
        descriptor = os.open(os.fspath(temporary), os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        os.write(descriptor, encoded)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.replace(temporary, path)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _write_compat_status_env(path: Path, values: Mapping[str, str]) -> None:
    """Publish queue-compatible lifecycle metadata without sample fields."""

    if path.exists() or path.is_symlink():
        raise CoordinatorError(f"compatibility status already exists: {path}")
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise CoordinatorError(f"compatibility status parent is unsafe: {path.parent}")
    allowed = {"state", "model_id", "model_path", "work_dir", "cuda_visible_devices", "tp", "dp", "profiles", "aggregate_only", "sample_level_output"}
    if set(values) - allowed:
        raise CoordinatorError("compatibility status contains unsupported fields")
    # The existing queue decodes the
    # tiny printf-%q subset (comma/space/semicolon/backslash), not shell
    # quoting.  Paths/model IDs are already validated labels below.
    def percent_q(value: str) -> str:
        return value.replace("\\", "\\\\").replace(",", "\\,").replace(" ", "\\ ").replace(";", "\\;")

    encoded = "".join(f"{key}={percent_q(str(values[key]))}\n" for key in values).encode("utf-8")
    descriptor: int | None = None
    try:
        descriptor = os.open(os.fspath(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        offset = 0
        while offset < len(encoded):
            count = os.write(descriptor, encoded[offset:])
            if count <= 0:
                raise CoordinatorError(f"compatibility status write made no progress: {path}")
            offset += count
        os.fsync(descriptor)
    except OSError as exc:
        raise CoordinatorError(f"compatibility status could not be written: {path}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _ensure_fresh_directory(path: Path, *, mode: int = 0o700) -> None:
    if path.exists() or path.is_symlink():
        raise CoordinatorError(f"create-once directory already exists: {path}")
    path.mkdir(parents=True, mode=mode)
    os.chmod(path, mode)
    if path.is_symlink() or not path.is_dir():
        raise CoordinatorError(f"created directory is unsafe: {path}")


def _private_cache_path(run_root: Path, model_id: str) -> Path:
    digest = hashlib.sha256(f"{run_root}\0{model_id}".encode()).hexdigest()[:24]
    return WORKSPACE_ROOT / "Cache/vllm_formal_shared" / f"threebench_resident_v1-{digest}"


def _prepare_cache(path: Path) -> str:
    parent = path.parent
    if parent.is_symlink() or not parent.is_dir():
        raise CoordinatorError(f"vLLM cache parent is unavailable: {parent}")
    _ensure_fresh_directory(path)
    return hashlib.sha256(str(path).encode()).hexdigest()[:32]


def build_judge_cache_identity(model_path: str | Path) -> dict[str, Any]:
    """Build the non-sample identity for the fixed 27B TP2/DP4 judge cache."""

    path = _lexical_absolute(model_path)
    try:
        info = path.lstat()
    except OSError as exc:
        raise CoordinatorError(f"judge model path is unavailable: {path}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise CoordinatorError(f"judge model path must be a real directory: {path}")
    config = path / "config.json"
    config_sha = sha_file(config)
    # These source hashes are intentionally part of the sealed identity.  A
    # future launcher/compiler change must create or explicitly bootstrap a
    # matching namespace rather than silently reusing incompatible artifacts.
    serve_sha = sha_file(TOOLS_ROOT / "serve_qwen35.sh")
    matrix_sha = sha_file(ZOOMBENCH_JUDGE_MATRIX)
    return {
        "model_id": ZOOMBENCH_JUDGE_MODEL_ID,
        "model_path": str(path),
        "model_config_sha256": config_sha,
        "architecture": JUDGE_ARCHITECTURE,
        "code_hash": JUDGE_CODE_HASH,
        "vllm_version": JUDGE_VLLM_VERSION,
        "compiler_hash": JUDGE_COMPILER_HASH,
        "config_hash": JUDGE_VLLM_CONFIG_HASH,
        "cuda_main_version": JUDGE_CUDA_MAIN_VERSION,
        "cuda_target_device": JUDGE_CUDA_TARGET_DEVICE,
        "cuda_visible_devices": CUDA_VISIBLE_DEVICES,
        "tensor_parallel_size": ZOOMBENCH_JUDGE_TP,
        "data_parallel_size": ZOOMBENCH_JUDGE_DP,
        "max_model_len": JUDGE_MAX_MODEL_LEN,
        "max_num_seqs": JUDGE_MAX_NUM_SEQS,
        "gpu_memory_utilization": JUDGE_GPU_MEMORY_UTILIZATION,
        "compile_backend": "inductor",
        "compile_cache_save_format": "binary",
        "compile_range": list(JUDGE_COMPILE_RANGE),
        "compile_config": json.loads(json.dumps(JUDGE_COMPILE_CONFIG, sort_keys=True)),
        "kernel_config": json.loads(json.dumps(JUDGE_KERNEL_CONFIG, sort_keys=True)),
        "serve_script_sha256": serve_sha,
        "judge_matrix_sha256": matrix_sha,
    }


def _judge_cache_paths(cache_root: str | Path = JUDGE_CACHE_ROOT) -> dict[str, Path]:
    root = _lexical_absolute(cache_root)
    paths = {name: root / name for name in JUDGE_CACHE_CHILDREN}
    paths["root"] = root
    paths["manifest"] = root / JUDGE_CACHE_MANIFEST_NAME
    return paths


def _assert_no_symlink_ancestors(path: Path) -> None:
    current = _lexical_absolute(path)
    while True:
        try:
            info = current.lstat()
        except FileNotFoundError:
            info = None
        except OSError as exc:
            raise CoordinatorError(f"cache namespace ancestor cannot be inspected: {current}") from exc
        if info is not None and stat.S_ISLNK(info.st_mode):
            raise CoordinatorError(f"cache namespace ancestor is a symlink: {current}")
        if current == current.parent:
            return
        current = current.parent


def _ensure_private_directory(path: Path, *, label: str) -> None:
    """Create one namespace component, rejecting pre-existing unsafe state."""

    _assert_no_symlink_ancestors(path)
    existed = path.exists() or path.is_symlink()
    if existed:
        _validate_private_cache_directory(path, label=label)
        return
    try:
        path.mkdir(parents=True, mode=0o700, exist_ok=False)
    except FileExistsError as exc:
        raise CoordinatorError(f"cache namespace raced during creation: {path}") from exc
    os.chmod(path, 0o700)
    _validate_private_cache_directory(path, label=label)


def _validate_cache_tree(root: Path, *, label: str, private_root: bool) -> None:
    """Reject symlinks and foreign owners throughout a cache-only tree."""

    if private_root:
        _validate_private_cache_directory(root, label=label)
    else:
        try:
            info = root.lstat()
        except OSError as exc:
            raise CoordinatorError(f"{label} is unavailable: {root}") from exc
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise CoordinatorError(f"{label} must be a real directory: {root}")
        if info.st_uid != os.getuid() or info.st_gid != os.getgid():
            raise CoordinatorError(f"{label} ownership gate failed: {root}")
    try:
        iterator = os.walk(root, topdown=True, followlinks=False)
        for current, directories, files in iterator:
            for name in [*directories, *files]:
                child = Path(current) / name
                try:
                    info = child.lstat()
                except OSError as exc:
                    raise CoordinatorError(f"{label} entry cannot be inspected: {child}") from exc
                if stat.S_ISLNK(info.st_mode):
                    raise CoordinatorError(f"{label} contains a symlink: {child}")
                if info.st_uid != os.getuid() or info.st_gid != os.getgid():
                    raise CoordinatorError(f"{label} entry ownership gate failed: {child}")
    except OSError as exc:
        raise CoordinatorError(f"{label} cannot be traversed: {root}") from exc


def _judge_cache_manifest_body(
    paths: Mapping[str, Path],
    identity: Mapping[str, Any],
    *,
    bootstrap: Mapping[str, Any],
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema_version": JUDGE_CACHE_SCHEMA,
        "namespace": str(paths["root"]),
        "cache_key": sha_text(identity)[:32],
        "identity": dict(identity),
        "roots": {name: str(paths[name]) for name in JUDGE_CACHE_CHILDREN},
        "bootstrap": dict(bootstrap),
    }
    body["manifest_sha256"] = sha_text(body)
    return body


def _validate_judge_cache_manifest(
    manifest_path: Path,
    *,
    paths: Mapping[str, Path],
    identity: Mapping[str, Any],
) -> str:
    _assert_no_symlink_ancestors(paths["root"])
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise CoordinatorError(f"judge cache manifest is unavailable: {manifest_path}")
    try:
        info = manifest_path.lstat()
    except OSError as exc:
        raise CoordinatorError(f"judge cache manifest cannot be inspected: {manifest_path}") from exc
    if info.st_uid != os.getuid() or info.st_gid != os.getgid() or stat.S_IMODE(info.st_mode) != 0o600:
        raise CoordinatorError("judge cache manifest ownership/mode gate failed")
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CoordinatorError("judge cache manifest is not valid JSON") from exc
    if not isinstance(value, Mapping) or value.get("schema_version") != JUDGE_CACHE_SCHEMA:
        raise CoordinatorError("judge cache manifest schema differs")
    expected_keys = {"schema_version", "namespace", "cache_key", "identity", "roots", "bootstrap", "manifest_sha256"}
    if set(value) != expected_keys:
        raise CoordinatorError("judge cache manifest fields differ")
    seal = value.get("manifest_sha256")
    body = {key: item for key, item in value.items() if key != "manifest_sha256"}
    if not isinstance(seal, str) or not HEX64.fullmatch(seal) or sha_text(body) != seal:
        raise CoordinatorError("judge cache manifest seal differs")
    if value.get("namespace") != str(paths["root"]) or value.get("cache_key") != sha_text(identity)[:32] or value.get("identity") != dict(identity):
        raise CoordinatorError("judge cache identity differs")
    expected_roots = {name: str(paths[name]) for name in JUDGE_CACHE_CHILDREN}
    if value.get("roots") != expected_roots or not isinstance(value.get("bootstrap"), Mapping):
        raise CoordinatorError("judge cache roots differ")
    for name in JUDGE_CACHE_CHILDREN:
        _validate_private_cache_directory(paths[name], label=f"judge cache {name}")
    return seal


def _cache_children_nonempty(paths: Mapping[str, Path]) -> bool:
    for name in JUDGE_CACHE_CHILDREN:
        child = paths[name]
        try:
            if any(child.iterdir()):
                return True
        except OSError as exc:
            raise CoordinatorError(f"judge cache child cannot be inspected: {child}") from exc
    return False


def _prepare_judge_cache_namespace(
    model_path: str | Path,
    *,
    cache_root: str | Path = JUDGE_CACHE_ROOT,
) -> dict[str, Any]:
    """Create/validate the sealed fixed judge namespace before serving."""

    identity = build_judge_cache_identity(model_path)
    paths = _judge_cache_paths(cache_root)
    _ensure_private_directory(paths["root"], label="judge cache namespace")
    for name in JUDGE_CACHE_CHILDREN:
        _ensure_private_directory(paths[name], label=f"judge cache {name}")
    manifest = paths["manifest"]
    if manifest.exists() or manifest.is_symlink():
        seal = _validate_judge_cache_manifest(manifest, paths=paths, identity=identity)
        _validate_cache_tree(paths["root"], label="judge cache namespace", private_root=True)
        return {"identity": dict(identity), "paths": paths, "manifest": manifest, "manifest_sha256": seal}
    if _cache_children_nonempty(paths):
        raise CoordinatorError("unsealed judge cache contains existing artifacts")
    value = _judge_cache_manifest_body(paths, identity, bootstrap={"mode": "cold_start"})
    _write_create_once(manifest, value, mode=0o600)
    seal = _validate_judge_cache_manifest(manifest, paths=paths, identity=identity)
    _validate_cache_tree(paths["root"], label="judge cache namespace", private_root=True)
    return {"identity": dict(identity), "paths": paths, "manifest": manifest, "manifest_sha256": seal}


def _completed_status_state(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise CoordinatorError(f"bootstrap status is unavailable: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise CoordinatorError(f"bootstrap status cannot be read: {path}") from exc
    if path.suffix == ".json":
        try:
            value = json.loads(text)
        except ValueError as exc:
            raise CoordinatorError("bootstrap status is not valid JSON") from exc
        state = value.get("state") if isinstance(value, Mapping) else None
    else:
        state = next((line.split("=", 1)[1] for line in text.splitlines() if line.startswith("state=")), None)
    if state != "complete":
        raise CoordinatorError("bootstrap source is not complete")
    return str(state)


def bootstrap_judge_cache_from_completed_run(
    *,
    model_path: str | Path,
    source_status: str | Path,
    source_roots: Mapping[str, str | Path],
    source_relative: Mapping[str, str] | None = None,
    cache_root: str | Path = JUDGE_CACHE_ROOT,
) -> dict[str, Any]:
    """Copy only audited compile/runtime trees and seal a new judge namespace.

    ``source_roots`` must name cache-only directories (never a run/output
    root).  ``source_relative`` permits bootstrapping one exact vLLM compile
    key below the destination ``vllm`` root without copying unrelated keys.
    """

    identity = build_judge_cache_identity(model_path)
    status_path = _lexical_absolute(source_status)
    _completed_status_state(status_path)
    status_info = status_path.lstat()
    if status_info.st_uid != os.getuid() or status_info.st_gid != os.getgid():
        raise CoordinatorError("bootstrap status ownership gate failed")
    paths = _judge_cache_paths(cache_root)
    _assert_no_symlink_ancestors(paths["root"])
    if paths["root"].exists() or paths["root"].is_symlink():
        raise CoordinatorError("judge cache bootstrap destination already exists")
    relative_map = dict(source_relative or {})
    unknown = set(source_roots) - set(JUDGE_CACHE_CHILDREN)
    if unknown or set(relative_map) - set(source_roots):
        raise CoordinatorError("bootstrap source roots are unsupported")
    for name, relative in relative_map.items():
        rel = Path(relative)
        if rel.is_absolute() or ".." in rel.parts or not str(rel):
            raise CoordinatorError("bootstrap destination relative path is unsafe")
    checked_sources: dict[str, Path] = {}
    for name, source in source_roots.items():
        source_path = _lexical_absolute(source)
        _validate_cache_tree(source_path, label=f"bootstrap source {name}", private_root=False)
        checked_sources[name] = source_path

    _ensure_private_directory(paths["root"], label="judge cache namespace")
    for name in JUDGE_CACHE_CHILDREN:
        destination_root = paths[name]
        _ensure_private_directory(destination_root, label=f"judge cache {name}")
        source_path = checked_sources.get(name)
        if source_path is None:
            continue
        relative = Path(relative_map.get(name, ""))
        destination = destination_root / relative if str(relative) else destination_root
        if destination != destination_root:
            destination.parent.mkdir(mode=0o700, parents=True, exist_ok=False)
            os.chmod(destination.parent, 0o700)
        try:
            # All source symlinks and foreign owners were rejected above.
            shutil.copytree(
                source_path,
                destination,
                symlinks=False,
                dirs_exist_ok=destination == destination_root,
            )
        except (OSError, shutil.Error) as exc:
            raise CoordinatorError(f"bootstrap copy failed for {name}") from exc
        os.chmod(destination_root, 0o700)
        if destination.parent != destination_root:
            os.chmod(destination.parent, 0o700)

    _validate_cache_tree(paths["root"], label="judge cache namespace", private_root=True)
    bootstrap = {
        "mode": "audited_copy",
        "source_status": str(status_path),
        "source_status_sha256": sha_file(status_path),
        "source_roots": {name: str(path) for name, path in checked_sources.items()},
        "source_relative": {name: str(Path(relative)) for name, relative in relative_map.items()},
    }
    value = _judge_cache_manifest_body(paths, identity, bootstrap=bootstrap)
    _write_create_once(paths["manifest"], value, mode=0o600)
    seal = _validate_judge_cache_manifest(paths["manifest"], paths=paths, identity=identity)
    return {"identity": dict(identity), "paths": paths, "manifest": paths["manifest"], "manifest_sha256": seal}


def _validate_execution_args(
    *, model_path: Path, model_id: str, run_root: Path, audit_root: Path, candidate_port: int, judge_port: int
) -> str:
    _under(model_path, WORKSPACE_ROOT, label="model path")
    _under(run_root, OUTPUT_ROOT, label="run root")
    _under(audit_root, OUTPUT_ROOT, label="audit root")
    if run_root == audit_root:
        raise CoordinatorError("run root and audit root must differ")
    if SAFE_MODEL_ID.fullmatch(model_id) is None:
        raise CoordinatorError("model-id is unsafe")
    tag = model_tag_for(model_id)
    if candidate_port == judge_port:
        raise CoordinatorError("candidate and judge ports must differ")
    endpoint(candidate_port)
    endpoint(judge_port)
    if os.getuid() != REQUIRED_UID or os.getgid() != REQUIRED_GID:
        raise CoordinatorError(f"formal execution requires UID/GID {REQUIRED_UID}/{REQUIRED_GID}")
    try:
        info = model_path.lstat()
    except OSError as exc:
        raise CoordinatorError("candidate checkpoint is unavailable") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode) or not (model_path / "config.json").is_file():
        raise CoordinatorError("candidate checkpoint requires a real directory/config.json")
    if not (WORKSPACE_ROOT / "Ckpt/Qwen3.5-27B/config.json").is_file():
        raise CoordinatorError("ZoomBench judge checkpoint is unavailable")
    if (
        OFFICIAL_QWEN35_9B_CHAT_TEMPLATE.is_symlink()
        or not OFFICIAL_QWEN35_9B_CHAT_TEMPLATE.is_file()
        or sha_file(OFFICIAL_QWEN35_9B_CHAT_TEMPLATE)
        != OFFICIAL_QWEN35_9B_CHAT_TEMPLATE_SHA256
    ):
        raise CoordinatorError("official Qwen3.5-9B inference chat template differs")
    if not LOCK_PATH.is_file() or LOCK_PATH.is_symlink():
        raise CoordinatorError("GPU0-7 lock is unavailable")
    return tag


@contextlib.contextmanager
def _gpu_lease() -> Any:
    descriptor: int | None = None
    try:
        descriptor = os.open(os.fspath(LOCK_PATH), os.O_RDONLY | os.O_NOFOLLOW)
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
        if exc.errno in {errno.EACCES, errno.EAGAIN}:
            raise CoordinatorError("GPU0-7 lock is busy") from exc
        raise CoordinatorError("GPU0-7 lock cannot be acquired") from exc
    try:
        yield descriptor
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _safe_env(base: Mapping[str, str], *, runtime_root: Path) -> dict[str, str]:
    env = dict(base)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": CUDA_VISIBLE_DEVICES,
            "OPD_FORMAL_AGGREGATE_ONLY": "1",
            "OPD_FORMAL_GPU_LOCK": str(LOCK_PATH),
            "HOME": str(runtime_root / "home"),
            "XDG_CONFIG_HOME": str(runtime_root / "xdg_config"),
            "MPLCONFIGDIR": str(runtime_root / "mplconfig"),
            "FLASHINFER_WORKSPACE_BASE": str(runtime_root / "flashinfer"),
            "TORCHINDUCTOR_CACHE_DIR": str(runtime_root / "torchinductor"),
            "TRITON_CACHE_DIR": str(runtime_root / "triton"),
            # PA interactive pods expose both eth0 and a same-address veth.
            # NCCL 2.27 can fail bootstrap auto-selection in that layout with
            # `no socket interface found`; pin the routable pod interface for
            # candidate and judge collectives.  This changes only transport,
            # not benchmark generation or scoring semantics.
            "NCCL_SOCKET_IFNAME": "eth0",
            "NO_PROXY": "127.0.0.1,localhost",
            "no_proxy": "127.0.0.1,localhost",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    for name in ("HOME", "XDG_CONFIG_HOME", "MPLCONFIGDIR", "FLASHINFER_WORKSPACE_BASE", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"):
        Path(env[name]).mkdir(parents=True, exist_ok=True)
        os.chmod(env[name], 0o700)
    return env


def _judge_cache_env(
    base: Mapping[str, str],
    *,
    cache_root: str | Path = JUDGE_CACHE_ROOT,
    identity: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Bind every judge cache variable to the already-sealed namespace."""

    paths = _judge_cache_paths(cache_root)
    _validate_judge_cache_manifest(
        paths["manifest"],
        paths=paths,
        identity=dict(identity) if identity is not None else build_judge_cache_identity(WORKSPACE_ROOT / "Ckpt/Qwen3.5-27B"),
    )
    _validate_cache_tree(paths["root"], label="judge cache namespace", private_root=True)
    env = dict(base)
    env.update(
        {
            "HOME": str(paths["home"]),
            "XDG_CACHE_HOME": str(paths["xdg_cache"]),
            # The matrix's generic serving helper rewrites XDG_CACHE_HOME;
            # vLLM's explicit root is the stable compile-cache authority.
            "VLLM_CACHE_ROOT": str(paths["vllm"]),
            "XDG_CONFIG_HOME": str(paths["xdg_config"]),
            "FLASHINFER_WORKSPACE_BASE": str(paths["flashinfer"]),
            "TORCHINDUCTOR_CACHE_DIR": str(paths["torchinductor"]),
            "TRITON_CACHE_DIR": str(paths["triton"]),
            "CUDA_CACHE_PATH": str(paths["cuda"]),
            "TMPDIR": str(paths["tmp"]),
        }
    )
    return env


def _validate_private_cache_directory(path: Path, *, label: str) -> None:
    """Require one fixed cache component to be a private directory of this UID."""

    try:
        info = path.lstat()
    except OSError as exc:
        raise CoordinatorError(f"{label} is unavailable: {path}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise CoordinatorError(f"{label} must be a real directory: {path}")
    if info.st_uid != os.getuid() or info.st_gid != os.getgid() or stat.S_IMODE(info.st_mode) != 0o700:
        raise CoordinatorError(f"{label} ownership/mode gate failed")


def _validate_shared_kernel_cache_roots(
    compile_root: Path | None = None,
    config_root: Path | None = None,
    flashinfer_root: Path | None = None,
) -> None:
    """Require fixed compile/config/FlashInfer roots to be private and real.

    The cache is an acceleration input only.  It is never allowed to contain
    model weights or benchmark artifacts, and the formal GPU lease guarantees
    that resident evaluations do not mutate it concurrently.
    """

    compile_root = SHARED_KERNEL_CACHE_ROOT if compile_root is None else compile_root
    config_root = SHARED_KERNEL_CONFIG_ROOT if config_root is None else config_root
    flashinfer_root = SHARED_FLASHINFER_CACHE_ROOT if flashinfer_root is None else flashinfer_root
    _validate_private_cache_directory(compile_root, label="shared compile cache")
    _validate_private_cache_directory(config_root, label="shared vLLM config cache")
    _validate_private_cache_directory(flashinfer_root, label="shared FlashInfer cache")


def _run_logged(command: Sequence[str], *, env: Mapping[str, str], log_path: Path, cwd: Path | None = None) -> None:
    with log_path.open("ab") as stream:
        try:
            # Every protocol/judge command owns a process group so the
            # coordinator can terminate it when its SSH session receives
            # HUP/TERM.  ``subprocess.run`` does not expose a live child for
            # that fail-closed cleanup path.
            process = subprocess.Popen(
                command,
                env=dict(env),
                cwd=os.fspath(cwd) if cwd else None,
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            _register_process(process)
            try:
                returncode = process.wait()
            except BaseException:
                _stop_process(process)
                raise
            finally:
                _unregister_process(process)
        except OSError as exc:
            raise CoordinatorError(f"protocol command unavailable: {command[0]}") from exc
    if returncode != 0:
        raise CoordinatorError(f"protocol command failed (rc={returncode}): {Path(command[0]).name}")


def _start_candidate_server(*, model_path: Path, model_id: str, port: int, cache_root: Path, log_path: Path, runtime_root: Path) -> subprocess.Popen[bytes]:
    _validate_shared_kernel_cache_roots()
    env = _safe_env(os.environ, runtime_root=runtime_root)
    env.update(
        {
            "SERVE_ENV": str(WORKSPACE_ROOT / "UV_Env/verl-opd-qwen35"),
            "OPD_FORMAL_CACHE_ROOT": str(cache_root),
            "OPD_FORMAL_COMPILE_CACHE_ROOT": str(SHARED_KERNEL_CACHE_ROOT),
            # Do not inherit a caller's judge VLLM_CACHE_ROOT into the
            # candidate TP8/DP1 service; its compile namespace is explicit.
            "VLLM_CACHE_ROOT": str(SHARED_KERNEL_CACHE_ROOT / "xdg"),
            "XDG_CONFIG_HOME": str(SHARED_KERNEL_CONFIG_ROOT),
            "FLASHINFER_WORKSPACE_BASE": str(SHARED_FLASHINFER_CACHE_ROOT),
            "EVAL_MODEL_ID": model_id,
            "EVAL_API_HOST": LOCAL_HOST,
            "EVAL_API_PORT": str(port),
            "EVAL_TP_SIZE": str(TP_SIZE),
            "EVAL_DP_SIZE": str(DP_SIZE),
            "EVAL_GPU_MEMORY_UTILIZATION": SERVER_GPU_MEMORY_UTILIZATION,
            "EVAL_MAX_MODEL_LEN": str(MAX_MODEL_LEN),
            "EVAL_MAX_NUM_SEQS": str(SERVER_MAX_NUM_SEQS),
            "EVAL_MM_LIMITS": '{"image":16}',
        }
    )
    command = [
        str(SERVE_SCRIPT),
        str(model_path),
        "--chat-template",
        str(OFFICIAL_QWEN35_9B_CHAT_TEMPLATE),
        "--reasoning-parser",
        "qwen3",
    ]
    stream = log_path.open("ab")
    try:
        process = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    except OSError as exc:
        stream.close()
        raise CoordinatorError("candidate service could not be started") from exc
    # The descriptor is intentionally retained by the child; closing the
    # parent copy does not affect the process group's log stream.
    stream.close()
    _register_process(process)
    return process


def _stop_process(process: subprocess.Popen[bytes] | None) -> None:
    if process is None:
        return
    # The process-group leader can exit while a descendant remains alive.
    # Always signal the group first instead of using poll() as an early exit.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + 120
    while process.poll() is None and time.monotonic() < deadline:
        time.sleep(1)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=30)
    except (subprocess.TimeoutExpired, ChildProcessError):
        pass


def _json_url(url: str) -> Mapping[str, Any] | None:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            value = json.loads(response.read().decode("utf-8"))
    except (OSError, ValueError, urllib.error.URLError):
        return None
    return value if isinstance(value, Mapping) else None


def _http_ok(url: str) -> bool:
    """Probe a health endpoint without assuming it returns JSON."""

    request = urllib.request.Request(url, headers={"Accept": "*/*"})
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            response.read(1024)
        return True
    except (OSError, urllib.error.URLError):
        return False


def _wait_ready(process: subprocess.Popen[bytes], *, port: int, model_id: str) -> None:
    deadline = time.monotonic() + READINESS_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise CoordinatorError("candidate service exited before readiness")
        health = _http_ok(f"http://{LOCAL_HOST}:{port}/health")
        models = _json_url(f"http://{LOCAL_HOST}:{port}/v1/models")
        model_ids = {
            str(item.get("id"))
            for item in (models or {}).get("data", [])
            if isinstance(item, Mapping) and isinstance(item.get("id"), str)
        }
        if health and models is not None and model_id in model_ids:
            return
        time.sleep(READINESS_POLL_SECONDS)
    raise CoordinatorError("candidate service readiness timed out")


def _authority_record(path: Path, bench: BenchmarkSpec, *, model_id: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise CoordinatorError(f"{bench.name} authority is unavailable")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CoordinatorError(f"{bench.name} authority is malformed") from exc
    if not isinstance(value, Mapping):
        raise CoordinatorError(f"{bench.name} authority is not an object")
    schema = value.get("schema") if bench.slug == "blink" else value.get("schema_version")
    if schema != bench.authority_schema:
        raise CoordinatorError(f"{bench.name} authority schema differs")
    if value.get("model_id") not in {None, model_id}:
        # VStar and ZoomBench publish model_id at top level.  MMStar's fixed
        # authority also publishes it; omission is tolerated for old schemas.
        candidate = value.get("candidate")
        if not isinstance(candidate, Mapping) or candidate.get("model_id") != model_id:
            raise CoordinatorError(f"{bench.name} authority model identity differs")
    if bench.slug in {"vstar", "zoombench"}:
        expected_tag = model_tag_for(model_id)
        if value.get("model_tag") != expected_tag:
            raise CoordinatorError(f"{bench.name} authority model tag differs")
    if bench.slug == "vstar" and value.get("status") not in {"strict_beat", "scored_below_strict_beat"}:
        raise CoordinatorError("VStar authority did not report a successful score")
    if bench.slug == "zoombench" and value.get("status") != "scored":
        raise CoordinatorError("ZoomBench authority did not report a successful score")
    protocol_hash: str | None = None
    if isinstance(value.get("protocol_hash"), str):
        protocol_hash = value["protocol_hash"]
    elif bench.slug == "vstar" and isinstance(value.get("protocol"), Mapping):
        protocol_hash = sha_text(value["protocol"])
    elif bench.slug == "zoombench":
        runtime = value.get("runtime")
        candidate = value.get("candidate_inference")
        judge = value.get("judge")
        if isinstance(runtime, Mapping) and isinstance(candidate, Mapping) and isinstance(judge, Mapping):
            protocol: dict[str, Any] = {
                "candidate_cuda_visible_devices": CUDA_VISIBLE_DEVICES,
                "candidate_data_parallel_size": DP_SIZE,
                "candidate_enable_thinking": False,
                "candidate_max_tokens": ZOOMBENCH_MAX_TOKENS,
                "candidate_temperature": 0,
                "candidate_tensor_parallel_size": TP_SIZE,
                "judge_data_parallel_size": ZOOMBENCH_JUDGE_DP,
                "judge_enable_thinking": False,
                "judge_max_tokens": ZOOMBENCH_JUDGE_MAX_TOKENS,
                "judge_temperature": 0,
                "judge_tensor_parallel_size": ZOOMBENCH_JUDGE_TP,
                "sample_level_output": False,
                "serial_candidate_then_judge": True,
            }
            protocol_hash = sha_text(protocol)
    if protocol_hash != bench.protocol_hash:
        raise CoordinatorError(f"{bench.name} authority protocol differs")
    protocol = value.get("protocol")
    protocol_sample = protocol.get("sample_level_output") if isinstance(protocol, Mapping) else None
    runtime = value.get("runtime")
    runtime_sample = runtime.get("sample_level_output") if isinstance(runtime, Mapping) else None
    authority = value.get("authority")
    authority_only = authority.get("aggregate_only") if isinstance(authority, Mapping) else None
    aggregate_only = (
        value.get("aggregate_only") is True
        or value.get("sample_level_output") is False
        or protocol_sample is False
        or runtime_sample is False
        or authority_only is True
    )
    if not aggregate_only:
        raise CoordinatorError(f"{bench.name} authority is not aggregate-only")
    return {
        "authority_path": str(path),
        "authority_sha256": sha_file(path),
        "schema_version": schema,
    }


def _single_child(path: Path) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise CoordinatorError(f"audit root is unavailable: {path}")
    children = list(path.iterdir())
    if len(children) != 1 or children[0].is_symlink() or not children[0].is_dir():
        raise CoordinatorError("judge audit root must contain exactly one lifecycle")
    return children[0]


def _remove_known_sample_artifacts(
    paths: Sequence[Path], *, trusted_root: Path, require_empty_parents: bool = True
) -> list[str]:
    """Remove only exact, newly-created protocol artifact paths.

    This is deliberately not a recursive cleanup.  Every path is checked to
    be below this invocation's fresh run root, a regular non-symlink file, and
    one of the coordinator's fixed candidate/judge/score locations.  Parent
    directories are removed only when empty; an unexpected sibling is a
    terminal integrity error rather than a reason to widen deletion.
    """

    root = _lexical_absolute(trusted_root)
    unique: list[Path] = []
    seen: set[Path] = set()
    for supplied in paths:
        path = _lexical_absolute(supplied)
        _under(path, root, label="sample artifact")
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)

    removed: list[str] = []
    parent_dirs: set[Path] = set()
    for path in unique:
        if (
            not path.exists()
            and not path.is_symlink()
            and not path.parent.exists()
            and not path.parent.is_symlink()
        ):
            # A protocol may have emitted no artifact in this fixed slot;
            # absence is itself the desired post-authority state.
            continue
        # Do not unlink through a runner-created symlinked directory.  The
        # run root is fresh, but every parent is rechecked before deletion so
        # a compromised child protocol cannot redirect this narrow cleanup.
        ancestor = path.parent
        while True:
            try:
                ancestor_info = ancestor.lstat()
            except OSError as exc:
                raise CoordinatorError(f"sample artifact parent cannot be inspected: {ancestor}") from exc
            if stat.S_ISLNK(ancestor_info.st_mode) or not stat.S_ISDIR(ancestor_info.st_mode):
                raise CoordinatorError(f"sample artifact parent is unsafe: {ancestor}")
            if ancestor == root:
                break
            try:
                ancestor = ancestor.parent
                path.relative_to(root)
            except (ValueError, OSError) as exc:
                raise CoordinatorError(f"sample artifact parent escaped trusted root: {path}") from exc
        if not path.exists() and not path.is_symlink():
            parent_dirs.add(path.parent)
            continue
        parent_dirs.add(path.parent)
        try:
            info = path.lstat()
        except OSError as exc:
            raise CoordinatorError(f"sample artifact cannot be inspected: {path}") from exc
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise CoordinatorError(f"sample artifact is not a single-link regular file: {path}")
        try:
            path.unlink()
        except OSError as exc:
            raise CoordinatorError(f"sample artifact could not be removed: {path}") from exc
        if path.exists() or path.is_symlink():
            raise CoordinatorError(f"sample artifact remains after removal: {path}")
        removed.append(str(path.relative_to(root)))

    if not require_empty_parents:
        return removed

    # Verify the exact artifact parents.  If a runner emitted an unaccounted
    # sibling, fail closed instead of deleting it by pattern or recursion.
    for directory in sorted(parent_dirs, key=lambda item: len(item.parts), reverse=True):
        if not directory.exists():
            continue
        if directory.is_symlink() or not directory.is_dir():
            raise CoordinatorError(f"sample artifact parent is unsafe: {directory}")
        try:
            leftovers = list(directory.iterdir())
        except OSError as exc:
            raise CoordinatorError(f"sample artifact parent cannot be inspected: {directory}") from exc
        if leftovers:
            raise CoordinatorError(f"unaccounted sample artifact remains: {leftovers[0]}")
        try:
            directory.rmdir()
        except OSError as exc:
            raise CoordinatorError(f"empty sample artifact parent could not be removed: {directory}") from exc
        if directory.exists() or directory.is_symlink():
            raise CoordinatorError(f"sample artifact parent remains: {directory}")
    return removed


def _vstar_sample_artifacts(*, run_root: Path, model_tag: str) -> list[Path]:
    return [
        run_root / "model_answer/vstar" / f"{model_tag}_answer.jsonl",
        run_root / "judge/vstar" / f"{model_tag}_answer.jsonl",
        run_root / "score/vstar" / f"{model_tag}_cal_acc.log",
    ]


def _zoombench_sample_artifacts(*, run_root: Path, audit_root: Path, model_tag: str) -> list[Path]:
    # Validate the matrix shape here even though its log is cleaned in its
    # own trusted root below.
    _single_child(audit_root)
    return [
        run_root / "zoombench/model_answer/zoombench" / f"{model_tag}_answer.jsonl",
        run_root / "zoombench/judge/zoombench" / f"{model_tag}_answer.jsonl",
        run_root / "zoombench/score/zoombench" / f"{model_tag}_cal_acc.log",
    ]


def _judge_audit_sample_artifacts(audit_root: Path) -> tuple[Path, Path]:
    attempt = _single_child(audit_root)
    return audit_root, attempt / "candidate_judge_score.log"


def _run_zoom_judge(
    *,
    env: Mapping[str, str],
    run_root: Path,
    audit_root: Path,
    model_id: str,
    judge_port: int,
    log_path: Path,
) -> tuple[Path, Path]:
    target = f"candidate|{model_id}|{run_root}"
    command = [
        str(ZOOMBENCH_JUDGE_MATRIX),
        "--judge-model-path",
        str(WORKSPACE_ROOT / "Ckpt/Qwen3.5-27B"),
        "--judge-model-id",
        ZOOMBENCH_JUDGE_MODEL_ID,
        "--cuda-devices",
        CUDA_VISIBLE_DEVICES,
        "--tp",
        str(ZOOMBENCH_JUDGE_TP),
        "--dp",
        str(ZOOMBENCH_JUDGE_DP),
        "--port",
        str(judge_port),
        "--audit-root",
        str(audit_root),
        "--target",
        target,
    ]
    _run_logged(command, env=env, log_path=log_path)
    lifecycle = _single_child(audit_root)
    status = lifecycle / "status.env"
    if not status.is_file() or status.is_symlink():
        raise CoordinatorError("judge lifecycle status is unavailable")
    matrix_protocol = run_root / "zoombench/judge_protocol.json"
    if matrix_protocol.is_symlink() or not matrix_protocol.is_file():
        raise CoordinatorError("ZoomBench judge protocol is unavailable")
    judge_json = run_root / "zoombench/judge/zoombench" / f"{model_id.replace('/', '_')}_seed42_answer.jsonl"
    if judge_json.is_symlink() or not judge_json.is_file():
        raise CoordinatorError("ZoomBench judge result is unavailable")
    return matrix_protocol, judge_json


def _execute_impl(
    *,
    model_path: Path,
    model_id: str,
    run_root: Path,
    audit_root: Path,
    candidate_port: int,
    judge_port: int,
) -> dict[str, Any]:
    state = _ACTIVE_EXECUTION
    if state is None:
        # Keep the implementation directly callable by embedders/tests while
        # the public wrapper remains the place that installs signal handlers.
        state = _ExecutionState()
    state.model_id = model_id
    model_tag = _validate_execution_args(
        model_path=model_path,
        model_id=model_id,
        run_root=run_root,
        audit_root=audit_root,
        candidate_port=candidate_port,
        judge_port=judge_port,
    )
    source_hashes = verify_source_pins()
    _ensure_fresh_directory(run_root)
    _ensure_fresh_directory(audit_root)
    lifecycle = run_root / "lifecycle"
    lifecycle.mkdir(mode=0o700)
    os.chmod(lifecycle, 0o700)
    cache_root = _private_cache_path(run_root, model_id)
    cache_key = _prepare_cache(cache_root)
    manifest = build_server_manifest(
        model_path=model_path,
        model_id=model_id,
        model_tag=model_tag,
        candidate_port=candidate_port,
        cache_key=cache_key,
        source_hashes=source_hashes,
    )
    manifest_sha = validate_server_manifest(manifest)
    # Publish these pointers before the first lifecycle write so a signal
    # during setup can still be reflected as ``failed`` by the wrapper.
    state.lifecycle = lifecycle
    state.manifest_sha256 = manifest_sha
    manifest_path = lifecycle / "shared_server_manifest.json"
    _write_create_once(manifest_path, manifest)
    _write_status(
        lifecycle / "status.json",
        {
            "schema_version": STATUS_SCHEMA,
            "state": "starting",
            "candidate_model_id": model_id,
            "cuda_visible_devices": CUDA_VISIBLE_DEVICES,
            "tensor_parallel_size": TP_SIZE,
            "data_parallel_size": DP_SIZE,
            "max_model_len": MAX_MODEL_LEN,
            "shared_server_manifest_sha256": manifest_sha,
            "aggregate_only": True,
        },
    )
    runtime_root = lifecycle / "runtime"
    runtime_root.mkdir(mode=0o700)
    os.chmod(runtime_root, 0o700)
    server_log = lifecycle / "candidate_server.log"
    candidate_process: subprocess.Popen[bytes] | None = None
    authorities: dict[str, Mapping[str, Any]] = {}
    judge_record: dict[str, Any] = {
        "model_id": ZOOMBENCH_JUDGE_MODEL_ID,
        "model_path": str(WORKSPACE_ROOT / "Ckpt/Qwen3.5-27B"),
    }
    try:
        with _gpu_lease():
            candidate_process = _start_candidate_server(
                model_path=model_path,
                model_id=model_id,
                port=candidate_port,
                cache_root=cache_root,
                log_path=server_log,
                runtime_root=runtime_root,
            )
            # Register again here for CPU/mocked starters; the real starter
            # also registers at creation time and the state deduplicates it.
            state.register(candidate_process)
            _wait_ready(candidate_process, port=candidate_port, model_id=model_id)
            endpoint_url = endpoint(candidate_port)
            common_env = _safe_env(os.environ, runtime_root=runtime_root)
            common_env.update(
                {
                    "EVAL_MODEL_ID": model_id,
                    "EVAL_MODEL_TAG": model_tag,
                    "EVAL_API_BASE": endpoint_url,
                    "EVAL_API_KEY": os.environ.get("EVAL_API_KEY", "sk-local-opd"),
                    "EVAL_WORK_DIR": str(run_root),
                    "EVAL_SEED": "42",
                }
            )
            for bench in BENCHMARKS:
                bench_root = run_root / bench.slug
                bench_root.mkdir(mode=0o700)
                os.chmod(bench_root, 0o700)
                status_path = bench_root / "status.json"
                _write_status(
                    status_path,
                    {
                        "schema_version": STATUS_SCHEMA,
                        "state": "evaluating",
                        "benchmark": bench.name,
                        "candidate_model_id": model_id,
                        "shared_server_manifest_sha256": manifest_sha,
                        "aggregate_only": True,
                        "sample_level_output": False,
                        "sample_artifacts_removed": False,
                        "sample_persistence": "pending_cleanup",
                    },
                )
                bench_log = bench_root / "protocol.log"
                if bench.slug == "vstar":
                    vstar_env = {
                        **common_env,
                        "VISION_OPD_REFERENCE_PYTHON": str(PYTHON_BIN),
                        "VISION_OPD_CONTRACT_PYTHON": str(PYTHON_BIN),
                        "VISION_OPD_REFERENCE_API_KEY": common_env["EVAL_API_KEY"],
                    }
                    _run_logged(
                        [
                            str(VSTAR_RUNNER),
                            "preflight",
                            "--benchmark",
                            "vstar",
                            "--run-root",
                            str(bench_root),
                            "--model-path",
                            str(model_path),
                            "--model-id",
                            model_id,
                            "--model-tag",
                            model_tag,
                        ],
                        env=vstar_env,
                        log_path=bench_log,
                    )
                    _run_logged(
                        [
                            str(VSTAR_RUNNER),
                            "infer",
                            "--benchmark",
                            "vstar",
                            "--run-root",
                            str(bench_root),
                            "--model-path",
                            str(model_path),
                            "--model-id",
                            model_id,
                            "--model-tag",
                            model_tag,
                            "--api-base",
                            endpoint_url,
                        ],
                        env=vstar_env,
                        log_path=bench_log,
                    )
                    candidate = bench_root / "model_answer/vstar" / f"{model_tag}_answer.jsonl"
                    authority = bench_root / bench.authority_name
                    _run_logged(
                        [
                            str(PYTHON_BIN),
                            "-B",
                            str(VSTAR_AGGREGATOR),
                            "--candidate-answer",
                            str(candidate),
                            "--model-tag",
                            model_tag,
                            "--cache-key",
                            cache_key,
                            "--output",
                            str(authority),
                        ],
                        env=common_env,
                        log_path=bench_log,
                    )
                elif bench.slug == "mmstar":
                    authority = bench_root / bench.authority_name
                    _run_logged(
                        [
                            str(PYTHON_BIN),
                            "-B",
                            str(MMSTAR_AGGREGATOR),
                            "--mmstar-tsv",
                            str(MMSTAR_TSV),
                            "--output",
                            str(authority),
                            "--model-id",
                            model_id,
                            "--api-base",
                            endpoint_url,
                            "--workers",
                            str(MMSTAR_WORKERS),
                            "--max-tokens",
                            str(MMSTAR_MAX_TOKENS),
                        ],
                        env=common_env,
                        log_path=bench_log,
                    )
                else:
                    zoom_env = {
                        **common_env,
                        "ZOOMBENCH_MODE": "prepare",
                        "ZOOMBENCH_DATA_ROOT": str(ZOOMBENCH_DATA_ROOT),
                        # The audited ZoomBench layout is WORK_DIR/zoombench;
                        # the judge matrix receives WORK_DIR as its target.
                        "ZOOMBENCH_RUN_ROOT": str(bench_root),
                        "ZOOMBENCH_MAX_TOKENS": str(ZOOMBENCH_MAX_TOKENS),
                        "EVAL_API_NPROC": str(MMSTAR_WORKERS),
                        "DRY_RUN": "0",
                    }
                    _run_logged([str(ZOOMBENCH_RUNNER)], env=zoom_env, log_path=bench_log)
                    _run_logged([str(ZOOMBENCH_RUNNER)], env={**zoom_env, "ZOOMBENCH_MODE": "infer"}, log_path=bench_log)
                    queue_runner_root = run_root / "_runner"
                    _ensure_fresh_directory(queue_runner_root)
                    queue_candidate_root = queue_runner_root / "resident_candidate"
                    _ensure_fresh_directory(queue_candidate_root)
                    _write_compat_status_env(
                        queue_candidate_root / "status.env",
                        {
                            "state": "complete",
                            "model_id": model_id,
                            "model_path": str(model_path),
                            "work_dir": str(run_root),
                            "cuda_visible_devices": CUDA_VISIBLE_DEVICES,
                            "tp": str(TP_SIZE),
                            "dp": str(DP_SIZE),
                            "profiles": "zoom-infer",
                            "aggregate_only": "1",
                            "sample_level_output": "0",
                        },
                    )
                    # ZoomBench's aggregate authority is intentionally not
                    # published until the post-candidate 27B judge phase.
                    # Keep an independent candidate-complete status now and
                    # continue to the shared-server shutdown below.
                    _write_status(
                        status_path,
                        {
                            "schema_version": STATUS_SCHEMA,
                            "state": "candidate_complete",
                            "benchmark": bench.name,
                            "candidate_model_id": model_id,
                            "candidate_endpoint": endpoint_url,
                            "shared_server_manifest_sha256": manifest_sha,
                            "aggregate_only": True,
                            "sample_level_output": False,
                            "sample_artifacts_removed": False,
                            "sample_persistence": "pending_cleanup",
                        },
                    )
                    continue
                authority_record = _authority_record(authority, bench, model_id=model_id)
                if bench.slug == "vstar":
                    removed = _remove_known_sample_artifacts(
                        _vstar_sample_artifacts(run_root=bench_root, model_tag=model_tag),
                        trusted_root=bench_root,
                    )
                else:
                    # MMStar's aggregate helper reduces each response in
                    # memory and does not publish candidate/judge/score
                    # artifacts.  Keep an explicit empty cleanup certificate.
                    removed = []
                authority_record["sample_artifacts_removed"] = True
                authority_record["removed_artifacts"] = removed
                authorities[bench.slug] = authority_record
                if bench.slug == "mmstar":
                    # The existing four-bench queue recognizes MMStar via a
                    # tiny lifecycle status.env. This is aggregate metadata;
                    # the direct MMStar aggregator does not publish samples.
                    mmstar_lifecycle = bench_root / "lifecycle"
                    mmstar_lifecycle.mkdir(mode=0o700)
                    os.chmod(mmstar_lifecycle, 0o700)
                    _write_compat_status_env(
                        mmstar_lifecycle / "status.env",
                        {
                            "state": "complete",
                            "model_id": model_id,
                            "model_path": str(model_path),
                            "work_dir": str(bench_root),
                            "cuda_visible_devices": CUDA_VISIBLE_DEVICES,
                            "tp": str(TP_SIZE),
                            "dp": str(DP_SIZE),
                            "profiles": "mmstar",
                            "aggregate_only": "1",
                            "sample_level_output": "0",
                        },
                    )
                _write_status(
                    status_path,
                    {
                        "schema_version": STATUS_SCHEMA,
                        "state": "complete",
                        "benchmark": bench.name,
                        "candidate_model_id": model_id,
                        "authority_path": str(authority),
                        "authority_sha256": authorities[bench.slug]["authority_sha256"],
                        "shared_server_manifest_sha256": manifest_sha,
                        "aggregate_only": True,
                        "sample_level_output": False,
                        "sample_artifacts_removed": True,
                        "removed_artifacts": removed,
                    },
                )
            _stop_process(candidate_process)
            _unregister_process(candidate_process)
            state.unregister(candidate_process)
            candidate_process = None
            _write_status(
                lifecycle / "status.json",
                {
                    "schema_version": STATUS_SCHEMA,
                    "state": "candidate_stopped",
                    "candidate_model_id": model_id,
                    "cuda_visible_devices": CUDA_VISIBLE_DEVICES,
                    "tensor_parallel_size": TP_SIZE,
                    "data_parallel_size": DP_SIZE,
                    "max_model_len": MAX_MODEL_LEN,
                    "shared_server_manifest_sha256": manifest_sha,
                    "aggregate_only": True,
                    "sample_artifacts_removed": False,
                    "sample_persistence": "pending_cleanup",
                },
            )
            judge_model_path = WORKSPACE_ROOT / "Ckpt/Qwen3.5-27B"
            judge_cache = _prepare_judge_cache_namespace(judge_model_path)
            judge_env = _judge_cache_env(
                _safe_env(os.environ, runtime_root=runtime_root),
                cache_root=JUDGE_CACHE_ROOT,
                identity=judge_cache["identity"],
            )
            judge_env.update(
                {
                    "EVAL_MODEL_ID": ZOOMBENCH_JUDGE_MODEL_ID,
                    "EVAL_SEED": "42",
                    "DRY_RUN": "0",
                    "ZOOMBENCH_JUDGE_MAX_TOKENS": str(ZOOMBENCH_JUDGE_MAX_TOKENS),
                    "EVAL_WORK_DIR": str(run_root),
                    "OPD_JUDGE_CACHE_MANIFEST": str(judge_cache["manifest"]),
                    "OPD_JUDGE_CACHE_MANIFEST_SHA256": str(judge_cache["manifest_sha256"]),
                }
            )
            matrix_protocol, judge_json = _run_zoom_judge(
                env=judge_env,
                run_root=run_root,
                audit_root=audit_root,
                model_id=model_id,
                judge_port=judge_port,
                log_path=lifecycle / "zoom_judge.log",
            )
            # The serving subprocess is the only writer during this phase;
            # re-seal the namespace boundary before any authority/result can
            # be published if it created a symlink or foreign-owned entry.
            _validate_judge_cache_manifest(
                judge_cache["manifest"],
                paths=judge_cache["paths"],
                identity=judge_cache["identity"],
            )
            _validate_cache_tree(judge_cache["paths"]["root"], label="judge cache namespace", private_root=True)
            candidate_identity = lifecycle / "candidate_checkpoint_identity.json"
            judge_identity = lifecycle / "judge_checkpoint_identity.json"
            _run_logged(
                [str(PYTHON_BIN), "-B", str(ZOOMBENCH_AGGREGATOR), "--checkpoint-identity", str(model_path)],
                env=judge_env,
                log_path=candidate_identity,
            )
            _run_logged(
                [str(PYTHON_BIN), "-B", str(ZOOMBENCH_AGGREGATOR), "--checkpoint-identity", str(judge_model_path)],
                env=judge_env,
                log_path=judge_identity,
            )
            # The fixed formal aggregator's run-root is the parent WORK_DIR;
            # it owns the candidate ``zoombench/`` subtree and publishes its
            # authority at WORK_DIR/zoombench_formal_aggregate.json.
            zoom_authority = run_root / "zoombench_formal_aggregate.json"
            _run_logged(
                [
                    str(PYTHON_BIN),
                    "-B",
                    str(ZOOMBENCH_AGGREGATOR),
                    "--judge-json",
                    str(judge_json),
                    "--model-id",
                    model_id,
                    "--model-tag",
                    model_tag,
                    "--model-path",
                    str(model_path),
                    "--judge-model-path",
                    str(judge_model_path),
                    "--judge-model-id",
                    ZOOMBENCH_JUDGE_MODEL_ID,
                    "--expected-candidate-identity",
                    str(candidate_identity),
                    "--expected-judge-identity",
                    str(judge_identity),
                    "--aggregator-sha256",
                    SOURCE_HASHES["zoombench_formal_aggregate_v1.py"],
                    "--orchestrator-sha256",
                    SOURCE_HASHES["run_zoombench_formal_aggregate_v1.sh"],
                    "--judge-port",
                    str(judge_port),
                    "--cache-key",
                    cache_key,
                    "--run-root",
                    str(run_root),
                    "--audit-root",
                    str(audit_root),
                    "--matrix-protocol",
                    str(matrix_protocol),
                    "--dataset-manifest",
                    str(ZOOMBENCH_DATASET_MANIFEST),
                    "--output",
                    str(zoom_authority),
                ],
                env=judge_env,
                log_path=lifecycle / "zoom_aggregate.log",
            )
            zoom_record = _authority_record(zoom_authority, BENCHMARK_BY_SLUG["zoombench"], model_id=model_id)
            removed_zoom = _remove_known_sample_artifacts(
                _zoombench_sample_artifacts(run_root=run_root, audit_root=audit_root, model_tag=model_tag),
                trusted_root=run_root,
            )
            audit_root_for_cleanup, audit_score = _judge_audit_sample_artifacts(audit_root)
            removed_audit = _remove_known_sample_artifacts([audit_score], trusted_root=audit_root_for_cleanup, require_empty_parents=False)
            zoom_record["sample_artifacts_removed"] = True
            zoom_record["removed_artifacts"] = removed_zoom + [f"audit:{item}" for item in removed_audit]
            authorities["zoombench"] = zoom_record
            judge_record.update({"authority_path": str(zoom_authority), "authority_sha256": authorities["zoombench"]["authority_sha256"], "sample_artifacts_removed": True, "removed_artifacts": zoom_record["removed_artifacts"]})
            _write_status(
                run_root / "zoombench/status.json",
                {
                    "schema_version": STATUS_SCHEMA,
                    "state": "complete",
                    "benchmark": "ZoomBench",
                    "candidate_model_id": model_id,
                    "authority_path": str(zoom_authority),
                    "authority_sha256": authorities["zoombench"]["authority_sha256"],
                    "judge_model_id": ZOOMBENCH_JUDGE_MODEL_ID,
                    "shared_server_manifest_sha256": manifest_sha,
                    "aggregate_only": True,
                    "sample_level_output": False,
                    "sample_artifacts_removed": True,
                    "removed_artifacts": zoom_record["removed_artifacts"],
                },
            )
            result = build_result(model_id=model_id, model_tag=model_tag, run_root=run_root, audit_root=audit_root, manifest=manifest, authorities=authorities, judge=judge_record)
            _write_create_once(run_root / "threebench_resident_result.json", result)
            _write_status(lifecycle / "status.json", {**result, "schema_version": STATUS_SCHEMA, "state": "complete"})
            return result
    except Exception as exc:
        if state is not None:
            state.cleanup()
        else:
            _stop_process(candidate_process)
        # A signal can arrive in the tiny interval after result creation but
        # before the final status write.  Remove that exact private receipt so
        # every interrupted execution is observably failed and unpublished.
        unpublished_result = run_root / "threebench_resident_result.json"
        try:
            if unpublished_result.is_file() and not unpublished_result.is_symlink():
                unpublished_result.unlink()
        except OSError:
            # Do not widen cleanup or replace an unexpected path; status is
            # still marked failed below and the integrity error remains
            # visible to the operator.
            pass
        try:
            _write_status(
                lifecycle / "status.json",
                {
                    "schema_version": STATUS_SCHEMA,
                    "state": "failed",
                    "candidate_model_id": model_id,
                    "detail": type(exc).__name__,
                    "shared_server_manifest_sha256": manifest_sha,
                    "aggregate_only": True,
                },
            )
        except Exception:
            pass
        if isinstance(exc, CoordinatorError):
            raise
        raise CoordinatorError("resident coordinator failed") from exc


def _execute(
    *,
    model_path: Path,
    model_id: str,
    run_root: Path,
    audit_root: Path,
    candidate_port: int,
    judge_port: int,
) -> dict[str, Any]:
    """Run one resident evaluation with fail-closed signal handling."""

    global _ACTIVE_EXECUTION
    previous_handlers = _install_resident_signal_handlers()
    previous_state = _ACTIVE_EXECUTION
    state = _ExecutionState()
    _ACTIVE_EXECUTION = state
    try:
        try:
            return _execute_impl(
                model_path=model_path,
                model_id=model_id,
                run_root=run_root,
                audit_root=audit_root,
                candidate_port=candidate_port,
                judge_port=judge_port,
            )
        except Exception as exc:
            # Setup failures occur before _execute_impl enters its lifecycle
            # try/except.  Once a lifecycle exists, keep the same failed
            # status contract used by protocol/runtime failures.
            if state.lifecycle is not None and state.manifest_sha256 is not None:
                try:
                    _write_status(
                        state.lifecycle / "status.json",
                        {
                            "schema_version": STATUS_SCHEMA,
                            "state": "failed",
                            "candidate_model_id": model_id,
                            "detail": type(exc).__name__,
                            "shared_server_manifest_sha256": state.manifest_sha256,
                            "aggregate_only": True,
                        },
                    )
                except Exception:
                    pass
            raise
    finally:
        # This is also a last-resort guard for failures during setup, before
        # the implementation has entered its lifecycle try/except block.
        state.cleanup()
        _ACTIVE_EXECUTION = previous_state
        _restore_resident_signal_handlers(previous_handlers)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="print the fixed plan without reading or writing")
    mode.add_argument("--execute", action="store_true", help="execute one target with a resident candidate server")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--audit-root")
    parser.add_argument("--candidate-port", type=int, default=DEFAULT_CANDIDATE_PORT)
    parser.add_argument("--judge-port", type=int, default=DEFAULT_JUDGE_PORT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    run_root = _lexical_absolute(args.run_root)
    audit_root = _lexical_absolute(args.audit_root or f"{args.run_root}_audit")
    if args.dry_run:
        # Scope checks are lexical only here; nonexistent checkpoints/roots
        # are allowed, but a typo cannot render a plan outside this workspace.
        try:
            _under(_lexical_absolute(args.model_path), WORKSPACE_ROOT, label="model path")
            _under(run_root, OUTPUT_ROOT, label="run root")
            _under(audit_root, OUTPUT_ROOT, label="audit root")
        except CoordinatorError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        for item in render_plan(
            model_path=args.model_path,
            model_id=args.model_id,
            run_root=args.run_root,
            audit_root=args.audit_root or f"{args.run_root}_audit",
            candidate_port=args.candidate_port,
            judge_port=args.judge_port,
        ):
            print("RESIDENT_PLAN " + json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        print(
            "RESIDENT_PLAN "
            + json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "shared_server": True,
                    "candidate_service_count": 1,
                    "candidate_server_stopped_before_judge": True,
                    "max_model_len": MAX_MODEL_LEN,
                    "gpu_ids": list(GPU_IDS),
                    "aggregate_only": True,
                    "request_logging": False,
                    "raw_responses_saved": False,
                    "sample_level_data_saved": False,
                    "effects": "no filesystem, lock, subprocess, GPU, endpoint, dataset, checkpoint, or authority read/write",
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 0
    try:
        result = _execute(
            model_path=_lexical_absolute(args.model_path),
            model_id=args.model_id,
            run_root=run_root,
            audit_root=audit_root,
            candidate_port=args.candidate_port,
            judge_port=args.judge_port,
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        return 0
    except (CoordinatorError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
