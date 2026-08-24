#!/usr/bin/env python3
"""Serial formal four-benchmark queue for the two B57 dual4 exports.

The training jobs use two disjoint four-GPU groups, but formal evaluation is
deliberately run on one physical CUDA 0..7 allocation.  This boundary keeps
the existing protocol owners intact: VStar/MMStar/ZoomBench are delegated to
``run_threebench_resident_formal_v1.sh`` and BLINK-v5 is delegated to
``run_blink_checkpoint_comparison_v5.sh``.  No benchmark implementation is
duplicated here.

``--dry-run`` is pure and does not inspect the model or training trees.  In
execute mode model readiness is polled before any GPU child is launched.  A
completed component is reused on resume; failed/partial component roots are
preserved and a bounded retry namespace is selected for the next attempt.
Only aggregate/status metadata is published by this coordinator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import stat
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


TOOLS_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = Path(
    os.environ.get("OPD_QWEN35_WORKSPACE", "/minimax-3d-rw-backup/users/jiazhi/H_Workspace")
).absolute()
OUTPUT_ROOT = WORKSPACE_ROOT / "Output/opd_qwen35_9b"
DEFAULT_RUN_ROOT = OUTPUT_ROOT / "b57_dual4_models_fourbench_v1"
DEFAULT_QUEUE_ROOT = DEFAULT_RUN_ROOT

RESIDENT_RUNNER = TOOLS_ROOT / "run_threebench_resident_formal_v1.sh"
BLINK_RUNNER = TOOLS_ROOT / "run_blink_checkpoint_comparison_v5.sh"
OFFICIAL_QWEN35_9B_CHAT_TEMPLATE = WORKSPACE_ROOT / "Ckpt/Qwen3.5-9B/chat_template.jinja"

BENCHMARK_ORDER = ("VStar", "MMStar", "BLINK-v5", "ZoomBench")
EXPECTED_TOTALS = {"VStar": 191, "MMStar": 1500, "BLINK-v5": 1901, "ZoomBench": 845}
PROTOCOLS = {
    "VStar": "vstar_frozen_first_option_v1",
    "MMStar": "mmstar_qwen35_modelcard_thinking_v2",
    "BLINK-v5": "blink_deterministic_checkpoint_comparison_v5",
    "ZoomBench": "zoombench_score_aggregate_v1",
}
PROTOCOL_HASHES = {
    "VStar": "55fe9e9013c5a38e20e29446f07106ef9ab482be0f79332676aef9a4bfa07d98",
    "MMStar": "3d1baa4687ad3b5607cd622d0fef4e88f60f0f5c14bbe54d7b0ce0d1de221c17",
    "BLINK-v5": "ab5754c61c01c3c761c9fd72ae37480163884e894ffdb38cb805fe96b54204dc",
    "ZoomBench": "de79d40ac9916300db8a139a851727bf6bcb4fb016e9c659bf77609f9cb19f5a",
}

# Frozen BLINK-v5 control.  The displayed 59.13 is intentionally retained as
# a protocol reference, not recomputed/rounded into the candidate score.
BLINK_V5_RAW9_REFERENCE = {"correct": 1124, "total": 1901, "percent": 1124 / 1901 * 100.0, "reported_percent": 59.13}
BLINK_V5_PRESET = "blink_deterministic_checkpoint_comparison_v5"
BLINK_V5_SCHEMA = "mcq_blink_checkpoint_comparison_aggregate_v5"
GPU_IDS = tuple(range(8))
CUDA_VISIBLE_DEVICES = ",".join(str(item) for item in GPU_IDS)
REQUIRED_UID = REQUIRED_GID = 30853
MAX_RETRIES = 999
DEFAULT_WAIT_TIMEOUT_SECONDS = 7 * 24 * 60 * 60
DEFAULT_POLL_SECONDS = 30.0

SCHEMA_VERSION = "b57_dual4_models_fourbench_v1"
SUMMARY_SCHEMA = "b57_dual4_models_fourbench_summary_v1"
STATUS_SCHEMA = "b57_dual4_models_fourbench_status_v1"


@dataclass(frozen=True)
class ModelSpec:
    slug: str
    name: str
    model_id: str
    model_path: Path
    training_root: Path


MODEL_SPECS = (
    ModelSpec(
        slug="b57_10k_init_vision6k_crop_raw9_teacher_dual4_s65_v1",
        name="B57 10K Vision6K crop, Raw9 teacher, dual4 s65",
        model_id="Qwen3.5-9B-B57-10K-Vision6K-Crop-Raw9Teacher-Dual4-S65-v1",
        model_path=OUTPUT_ROOT / "b57_10k_init_vision6k_crop_raw9_teacher_dual4_s65_v1/merged/final_hf_official_chat_v1",
        training_root=OUTPUT_ROOT / "b57_10k_init_vision6k_crop_raw9_teacher_dual4_s65_v1",
    ),
    ModelSpec(
        slug="b57_10k_init_vision6k_crop_b57_27b_teacher_dual4_s65_v1",
        name="B57 10K Vision6K crop, B57-27B teacher, dual4 s65",
        model_id="Qwen3.5-9B-B57-10K-Vision6K-Crop-B57-27BTeacher-Dual4-S65-v1",
        model_path=OUTPUT_ROOT / "b57_10k_init_vision6k_crop_b57_27b_teacher_dual4_s65_v1/merged/final_hf_official_chat_v1",
        training_root=OUTPUT_ROOT / "b57_10k_init_vision6k_crop_b57_27b_teacher_dual4_s65_v1",
    ),
)
TARGETS = MODEL_SPECS
MODELS = MODEL_SPECS

# Three ports are reserved per model.  They are distinct across models even
# though the models are evaluated serially, which makes an accidental stale
# service fail closed instead of silently serving the other checkpoint.
BASE_PORT = 18618
PORT_STRIDE = 10


class QueueError(RuntimeError):
    """Fail-closed queue or metadata error."""


class _Signal(QueueError):
    pass


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular_json(path: Path) -> Mapping[str, Any] | None:
    try:
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size <= 0:
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
        return None
    return value if isinstance(value, Mapping) else None


def _path_string(path: Path) -> str:
    return str(path.absolute())


def _ports(index: int) -> dict[str, int]:
    base = BASE_PORT + index * PORT_STRIDE
    return {"candidate": base, "judge": base + 1, "blink": base + 2}


def _model_root(queue_root: Path, target: ModelSpec) -> Path:
    return queue_root / target.slug


def _component_base(queue_root: Path, target: ModelSpec, component: str) -> Path:
    return _model_root(queue_root, target) / component


def _retry_path(base: Path, index: int) -> Path:
    return base if index == 0 else base.with_name(f"{base.name}_retry{index}")


def _find_complete_root(base: Path, validator: Any) -> Path | None:
    """Find the canonical/latest complete root without following symlinks."""

    candidates = [base] + [_retry_path(base, index) for index in range(1, MAX_RETRIES + 1)]
    for candidate in candidates:
        if candidate.is_symlink() or not candidate.is_dir():
            continue
        try:
            if validator(candidate):
                return candidate
        except OSError:
            continue
    return None


def _reserve_root(base: Path) -> Path:
    for index in range(MAX_RETRIES + 1):
        candidate = _retry_path(base, index)
        if not candidate.exists() and not candidate.is_symlink():
            return candidate
    raise QueueError(f"retry namespace exhausted: {base}")


def _score(value: Mapping[str, Any], total: int) -> dict[str, Any] | None:
    candidates: list[Mapping[str, Any]] = [value]
    for key in ("evaluation", "counts", "scores", "aggregate"):
        child = value.get(key)
        if isinstance(child, Mapping):
            candidates.append(child)
    for candidate in candidates:
        correct = candidate.get("correct", candidate.get("num_correct"))
        observed = candidate.get("total", candidate.get("num_total", total))
        if isinstance(correct, bool) or not isinstance(correct, int) or observed != total or not 0 <= correct <= total:
            continue
        return {"correct": correct, "total": total, "percent": correct / total * 100.0}
    return None


def _resident_complete(root: Path, target: ModelSpec) -> bool:
    result = _regular_json(root / "threebench_resident_result.json")
    if not result or result.get("schema_version") != "threebench_resident_result_v1" or result.get("status") != "complete":
        return False
    if result.get("candidate_model_id") != target.model_id or result.get("aggregate_only") is not True or result.get("sample_level_data_saved") is not False:
        return False
    benchmarks = result.get("benchmarks")
    if not isinstance(benchmarks, Mapping):
        return False
    for name, slug in (("VStar", "vstar"), ("MMStar", "mmstar"), ("ZoomBench", "zoombench")):
        item = benchmarks.get(slug)
        if not isinstance(item, Mapping) or item.get("status") != "complete" or item.get("protocol_hash") != PROTOCOL_HASHES[name]:
            return False
        authority_path = item.get("authority_path")
        if not isinstance(authority_path, str):
            return False
        authority = _regular_json(Path(authority_path))
        if authority is None:
            return False
        if name == "VStar" and authority.get("status") not in {"strict_beat", "scored_below_strict_beat"}:
            return False
        if name == "ZoomBench" and authority.get("status") != "scored":
            return False
        if name == "MMStar" and authority.get("schema_version") != "mmstar_qwen35_modelcard_aggregate_v2":
            return False
        if name == "MMStar":
            # A prior run returned 1500 HTTP-success records whose answer
            # content was absent in every response.  Counting those as a
            # completed 0/1500 benchmark silently validates a transport/
            # response-schema failure, not model quality.  Formal MMStar must
            # therefore have essentially complete content and parse coverage.
            mmstar = authority.get("scores", {}).get("MMStar", authority)
            if not isinstance(mmstar, Mapping):
                return False
            if mmstar.get("api_failure_count") != 0 or mmstar.get("http_failure_count") != 0:
                return False
            if not isinstance(mmstar.get("content_present_count"), int) or mmstar["content_present_count"] < 1400:
                return False
            if not isinstance(mmstar.get("parsed_count"), int) or mmstar["parsed_count"] < 1400:
                return False
        if _score(authority, EXPECTED_TOTALS[name]) is None:
            # Older authorities may not expose counts at top level, but they
            # must still carry a strict 1500/191/845 score somewhere.
            if name == "MMStar":
                scores = authority.get("scores")
                if not isinstance(scores, Mapping) or _score(scores, EXPECTED_TOTALS[name]) is None:
                    return False
            else:
                return False
    return True


def _blink_complete(root: Path, target: ModelSpec) -> bool:
    aggregate = _regular_json(root / "blink_deterministic_checkpoint_comparison_v5.json")
    status = _regular_json(root / "lifecycle/status.json")
    if not aggregate or not status:
        return False
    if status.get("state") != "complete" or status.get("protocol") != BLINK_V5_PRESET or status.get("model_id") != target.model_id:
        return False
    if aggregate.get("schema_version") != BLINK_V5_SCHEMA or aggregate.get("code_version") != BLINK_V5_SCHEMA:
        return False
    if aggregate.get("preset") != BLINK_V5_PRESET or aggregate.get("model_id") != target.model_id or aggregate.get("total") != EXPECTED_TOTALS["BLINK-v5"]:
        return False
    if aggregate.get("protocol_hash") != PROTOCOL_HASHES["BLINK-v5"]:
        return False
    return _score(aggregate, EXPECTED_TOTALS["BLINK-v5"]) is not None


def _authority_score(root: Path, name: str) -> dict[str, Any] | None:
    if name == "BLINK-v5":
        return _score(_regular_json(root / "blink_deterministic_checkpoint_comparison_v5.json") or {}, EXPECTED_TOTALS[name])
    path = {
        "VStar": root / "vstar/formal_aggregate.json",
        "MMStar": root / "mmstar/mmstar_qwen35_modelcard_formal_aggregate.json",
        "ZoomBench": root / "zoombench_formal_aggregate.json",
    }[name]
    value = _regular_json(path)
    if not value:
        return None
    score = _score(value, EXPECTED_TOTALS[name])
    if score is not None:
        return score
    nested = value.get("scores")
    return _score(nested, EXPECTED_TOTALS[name]) if isinstance(nested, Mapping) else None


def _model_status(target: ModelSpec) -> tuple[str, str, dict[str, Any]]:
    """Inspect training/export readiness without reading model weights."""

    training = target.training_root
    status_path = training / "artifacts/status.json"
    status = _regular_json(status_path)
    if status is not None:
        state = status.get("state")
        if state == "failed":
            return "failed", "training status is failed", {"status_path": str(status_path), "training_state": state}
        if state != "complete":
            return "pending", f"training state is {state!r}", {"status_path": str(status_path), "training_state": state}
    elif training.exists():
        return "pending", "training completion status is not published", {"status_path": str(status_path), "training_state": None}
    model = target.model_path
    try:
        info = model.lstat()
    except OSError:
        return "pending", "merged model directory is absent", {"model_path": str(model)}
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        return "failed", "merged model path is not a real directory", {"model_path": str(model)}
    try:
        names = {entry.name for entry in model.iterdir()}
    except OSError as exc:
        return "pending", f"merged model directory is unreadable: {exc}", {"model_path": str(model)}
    if any(name.endswith(".incomplete") or name.startswith(".merge_") for name in names):
        return "pending", "merged model has an incomplete publication", {"model_path": str(model)}
    required = {"config.json", "tokenizer.json", "tokenizer_config.json"}
    missing = sorted(required - names)
    if missing:
        return "pending", "merged model is missing required files: " + ", ".join(missing), {"model_path": str(model), "missing": missing}
    shards = sorted(name for name in names if name.startswith("model") and (name.endswith(".safetensors") or name.endswith(".safetensors.index.json")))
    if not shards:
        return "pending", "merged model has no safetensors weight shard", {"model_path": str(model)}
    for name in ["config.json", "tokenizer.json", "tokenizer_config.json", *[item for item in shards if item.endswith(".safetensors")]]:
        path = model / name
        try:
            child = path.lstat()
            if stat.S_ISLNK(child.st_mode) or not stat.S_ISREG(child.st_mode) or child.st_size <= 0:
                return "pending", f"merged model file is incomplete: {name}", {"model_path": str(model)}
        except OSError:
            return "pending", f"merged model file is unavailable: {name}", {"model_path": str(model)}
    return "ready", "training status and merged HF artifact are complete", {"model_path": str(model), "weight_shards": shards, "training_state": "complete"}


def _target_label(target: ModelSpec, index: int, queue_root: Path) -> dict[str, Any]:
    ports = _ports(index)
    root = _model_root(queue_root, target)
    return {
        "slug": target.slug,
        "name": target.name,
        "model_id": target.model_id,
        "model_path": str(target.model_path),
        "training_root": str(target.training_root),
        "run_namespace": str(root),
        "ports": ports,
        "benchmark_order": list(BENCHMARK_ORDER),
        "expected_totals": dict(EXPECTED_TOTALS),
        "protocols": dict(PROTOCOLS),
        "serial_order": ["resident(VStar,MMStar,ZoomBench)", "BLINK-v5"],
        "gpu_ids": list(GPU_IDS),
        "cuda_visible_devices": CUDA_VISIBLE_DEVICES,
        "aggregate_only": True,
        "sample_level_output": False,
    }


def resident_command(target: ModelSpec, root: Path, audit_root: Path, *, ports: Mapping[str, int]) -> list[str]:
    return [
        str(RESIDENT_RUNNER), "--execute", "--model-path", str(target.model_path), "--model-id", target.model_id,
        "--run-root", str(root), "--audit-root", str(audit_root), "--candidate-port", str(ports["candidate"]), "--judge-port", str(ports["judge"]),
    ]


def blink_command(target: ModelSpec, root: Path, *, port: int) -> list[str]:
    return [
        str(BLINK_RUNNER),
        "--execute",
        "--model-path",
        str(target.model_path),
        "--model-id",
        target.model_id,
        "--run-root",
        str(root),
        "--port",
        str(port),
        "--chat-template-file",
        str(OFFICIAL_QWEN35_9B_CHAT_TEMPLATE),
    ]


def render_plan(*, queue_root: Path = DEFAULT_RUN_ROOT, targets: Sequence[ModelSpec] = MODEL_SPECS) -> dict[str, Any]:
    """Build a pure plan; intentionally no filesystem/runtime inspection."""

    rows = []
    commands = []
    for index, target in enumerate(targets):
        label = _target_label(target, index, queue_root)
        base_resident = _component_base(queue_root, target, "threebench")
        base_blink = _component_base(queue_root, target, "blink-v5")
        row = {**label, "resident_root": str(base_resident), "resident_audit_root": str(base_resident.with_name("threebench_audit")), "blink_root": str(base_blink), "status": "pending"}
        rows.append(row)
        commands.append({"target": target.slug, "resident": resident_command(target, base_resident, base_resident.with_name("threebench_audit"), ports=_ports(index)), "blink_v5": blink_command(target, base_blink, port=_ports(index)["blink"])})
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "dry_run",
        "aggregate_only": True,
        "sample_level_output": False,
        "serial": True,
        "model_serial": True,
        "gpu_ids": list(GPU_IDS),
        "cuda_visible_devices": CUDA_VISIBLE_DEVICES,
        "benchmark_order": list(BENCHMARK_ORDER),
        "expected_totals": dict(EXPECTED_TOTALS),
        "protocols": dict(PROTOCOLS),
        "blink_v5_raw9_reference": dict(BLINK_V5_RAW9_REFERENCE),
        "blink_v5_protocol_hash": PROTOCOL_HASHES["BLINK-v5"],
        "queue_root": str(queue_root),
        "summary_path": str(queue_root / "summary.json"),
        "status_path": str(queue_root / "status.json"),
        "target_order": [target.slug for target in targets],
        "targets": rows,
        "commands": commands,
        "effects": "no filesystem/runtime reads or writes; no subprocesses; no GPU or checkpoint access",
    }


def dry_run(*, queue_root: Path = DEFAULT_RUN_ROOT, targets: Sequence[ModelSpec] = MODEL_SPECS) -> dict[str, Any]:
    return render_plan(queue_root=queue_root, targets=targets)


def _plan(queue_root: Path = DEFAULT_RUN_ROOT) -> dict[str, Any]:
    """Compatibility helper used by CPU-only pipeline contract tests."""

    return render_plan(queue_root=queue_root)


def _metadata_public(value: Any, path: str = "root") -> None:
    forbidden = {"question", "image", "gold", "prediction", "response", "raw_response", "sample_uid", "sample_id"}
    if isinstance(value, Mapping):
        for key, child in value.items():
            lowered = str(key).casefold()
            if lowered in forbidden or ("sample" in lowered and lowered not in {"sample_level_output", "sample_count"}):
                raise QueueError(f"non-aggregate field in metadata: {path}.{key}")
            _metadata_public(child, f"{path}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            _metadata_public(child, f"{path}[{index}]")


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise QueueError(f"metadata path is unsafe: {path}")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.parent.is_symlink():
        raise QueueError(f"metadata parent is unsafe: {path.parent}")
    payload = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor: int | None = None
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        os.write(descriptor, payload)
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


class _ExecutionState:
    def __init__(self) -> None:
        self.processes: list[subprocess.Popen[Any]] = []
        self.signal_number: int | None = None
        self.cleaning = False

    def cleanup(self) -> None:
        if self.cleaning:
            return
        self.cleaning = True
        for process in list(self.processes):
            _stop_process(process)
        self.processes.clear()


_ACTIVE: _ExecutionState | None = None


def _on_signal(signum: int, _frame: Any) -> None:
    state = _ACTIVE
    if state is not None and state.signal_number is None and not state.cleaning:
        state.signal_number = signum
    raise _Signal(f"queue interrupted by signal {signum}")


def _stop_process(process: subprocess.Popen[Any] | None) -> None:
    if process is None or not isinstance(getattr(process, "pid", None), int):
        return
    pid = process.pid
    try:
        os.killpg(pid, signal.SIGTERM)
    except OSError:
        pass
    try:
        process.wait(timeout=2)
    except (OSError, subprocess.TimeoutExpired):
        pass
    try:
        os.killpg(pid, signal.SIGKILL)
    except OSError:
        pass
    try:
        process.wait(timeout=2)
    except (OSError, subprocess.TimeoutExpired):
        pass


def _run_child(command: Sequence[str], *, state: _ExecutionState, label: str, log_path: Path) -> None:
    if state.processes:
        raise QueueError(f"parallel child submission attempted before {label}")
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": CUDA_VISIBLE_DEVICES, "OPD_FORMAL_AGGREGATE_ONLY": "1", "OPD_PIPELINE_AGGREGATE_ONLY": "1", "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"})
    log_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with log_path.open("ab") as stream:
        try:
            process = subprocess.Popen(list(command), env=env, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
        except OSError as exc:
            raise QueueError(f"{label} child is unavailable") from exc
        state.processes.append(process)
        try:
            code = process.wait()
        finally:
            _stop_process(process)
            if process in state.processes:
                state.processes.remove(process)
    if code != 0:
        raise QueueError(f"{label} child failed with return code {code}")


def _bench_cells(target: ModelSpec, resident_root: Path | None, blink_root: Path | None) -> dict[str, Any]:
    cells: dict[str, Any] = {}
    for name in ("VStar", "MMStar", "ZoomBench"):
        score = _authority_score(resident_root, name) if resident_root is not None else None
        cells[name] = {"status": "complete" if resident_root is not None and _resident_complete(resident_root, target) else "pending", "complete": resident_root is not None and _resident_complete(resident_root, target), "total": EXPECTED_TOTALS[name], "protocol": PROTOCOLS[name], "protocol_hash": PROTOCOL_HASHES[name], **(score or {})}
        if resident_root is not None:
            cells[name]["formal_root"] = str(resident_root)
    score = _authority_score(blink_root, "BLINK-v5") if blink_root is not None else None
    cells["BLINK-v5"] = {"status": "complete" if blink_root is not None else "pending", "complete": blink_root is not None, "total": EXPECTED_TOTALS["BLINK-v5"], "protocol": PROTOCOLS["BLINK-v5"], "protocol_hash": PROTOCOL_HASHES["BLINK-v5"], **(score or {})}
    if blink_root is not None:
        cells["BLINK-v5"]["formal_root"] = str(blink_root)
    return cells


def _target_record(target: ModelSpec, index: int, queue_root: Path, *, resident_root: Path | None, blink_root: Path | None, actions: Mapping[str, str], readiness: Mapping[str, Any]) -> dict[str, Any]:
    cells = _bench_cells(target, resident_root, blink_root)
    complete = all(cell.get("complete") is True for cell in cells.values())
    return {
        **_target_label(target, index, queue_root),
        "status": "complete" if complete else "failed",
        "eligible_for_model_judgment": complete,
        "benchmarks": cells,
        "actions": dict(actions),
        "readiness": dict(readiness),
        "fourbench_macro_percent": (sum(float(cell["percent"]) for cell in cells.values()) / 4.0 if complete and all("percent" in cell for cell in cells.values()) else None),
    }


def _initial_summary(queue_root: Path, targets: Sequence[ModelSpec]) -> dict[str, Any]:
    return {
        "schema_version": SUMMARY_SCHEMA,
        "status": "running",
        "aggregate_only": True,
        "sample_level_output": False,
        "serial": True,
        "model_serial": True,
        "gpu_ids": list(GPU_IDS),
        "cuda_visible_devices": CUDA_VISIBLE_DEVICES,
        "benchmark_order": list(BENCHMARK_ORDER),
        "expected_totals": dict(EXPECTED_TOTALS),
        "protocols": dict(PROTOCOLS),
        "protocol_hashes": dict(PROTOCOL_HASHES),
        "blink_v5_raw9_reference": dict(BLINK_V5_RAW9_REFERENCE),
        "queue_root": str(queue_root),
        "target_order": [target.slug for target in targets],
        "targets": [],
        "events": [],
        "resume_count": 0,
    }


def _load_summary(path: Path) -> dict[str, Any] | None:
    value = _regular_json(path)
    if value is None or value.get("schema_version") != SUMMARY_SCHEMA:
        return None
    if value.get("aggregate_only") is not True or value.get("sample_level_output") is not False or value.get("benchmark_order") != list(BENCHMARK_ORDER):
        raise QueueError("existing summary violates aggregate four-benchmark contract")
    return dict(value)


def _wait_models(targets: Sequence[ModelSpec], *, timeout_seconds: float, poll_seconds: float) -> dict[str, Mapping[str, Any]]:
    started = time.monotonic()
    while True:
        states: dict[str, Mapping[str, Any]] = {}
        failed: list[str] = []
        pending: list[str] = []
        for target in targets:
            state, reason, details = _model_status(target)
            states[target.slug] = {"state": state, "reason": reason, **details}
            if state == "failed":
                failed.append(f"{target.slug}: {reason}")
            elif state != "ready":
                pending.append(f"{target.slug}: {reason}")
        if failed:
            raise QueueError("model/training failed: " + "; ".join(failed))
        if not pending:
            return states
        if timeout_seconds >= 0 and time.monotonic() - started >= timeout_seconds:
            raise QueueError("model artifacts remain incomplete after wait timeout: " + "; ".join(pending))
        time.sleep(max(0.01, poll_seconds))


def model_complete(target: ModelSpec) -> bool:
    return _model_status(target)[0] == "ready"


wait_for_models = _wait_models


def execute(*, queue_root: Path = DEFAULT_RUN_ROOT, targets: Sequence[ModelSpec] = MODEL_SPECS, timeout_seconds: float = DEFAULT_WAIT_TIMEOUT_SECONDS, poll_seconds: float = DEFAULT_POLL_SECONDS, no_wait: bool = False) -> dict[str, Any]:
    if os.getuid() != REQUIRED_UID or os.getgid() != REQUIRED_GID:
        raise QueueError(f"formal execution requires UID/GID {REQUIRED_UID}/{REQUIRED_GID}")
    queue_root = queue_root.absolute()
    if queue_root == OUTPUT_ROOT or OUTPUT_ROOT not in queue_root.parents:
        raise QueueError(f"queue root must be below Output/opd_qwen35_9b: {queue_root}")
    summary_path = queue_root / "summary.json"
    status_path = queue_root / "status.json"
    previous = _load_summary(summary_path)
    if previous is not None and previous.get("status") == "complete":
        records = previous.get("targets")
        if isinstance(records, list) and len(records) == len(targets) and all(isinstance(row, Mapping) and row.get("eligible_for_model_judgment") is True for row in records):
            return previous
    readiness = {target.slug: {"state": "deferred", "reason": "not inspected"} for target in targets}
    if not no_wait:
        readiness = dict(_wait_models(targets, timeout_seconds=timeout_seconds, poll_seconds=poll_seconds))
    else:
        for target in targets:
            state, reason, details = _model_status(target)
            if state != "ready":
                raise QueueError(f"--no-wait requires complete model: {target.slug}: {reason}")
            readiness[target.slug] = {"state": state, "reason": reason, **details}
    summary = previous or _initial_summary(queue_root, targets)
    summary["resume_count"] = int(summary.get("resume_count", 0)) + (1 if previous else 0)
    summary["status"] = "running"
    summary["readiness"] = readiness
    summary["targets"] = []
    summary["events"] = list(summary.get("events", [])) if isinstance(summary.get("events"), list) else []
    _metadata_public(summary)
    queue_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    _atomic_json(summary_path, summary)
    _atomic_json(status_path, {"schema_version": STATUS_SCHEMA, "state": "running", "summary_path": str(summary_path), "serial": True, "gpu_ids": list(GPU_IDS), "target_order": [target.slug for target in targets], "aggregate_only": True, "sample_level_output": False})
    state = _ExecutionState()
    global _ACTIVE
    previous_active = _ACTIVE
    old_handlers = {signum: signal.getsignal(signum) for signum in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM)}
    for signum in old_handlers:
        signal.signal(signum, _on_signal)
    _ACTIVE = state
    try:
        for index, target in enumerate(targets):
            ports = _ports(index)
            model_root = _model_root(queue_root, target)
            if model_root.is_symlink() or (model_root.exists() and not model_root.is_dir()):
                raise QueueError(f"model run namespace is unsafe: {model_root}")
            model_root.mkdir(parents=True, exist_ok=True, mode=0o700)
            if model_root.is_symlink():
                raise QueueError(f"model run namespace became a symlink: {model_root}")
            (model_root / "lifecycle").mkdir(mode=0o700, exist_ok=True)
            resident_base = _component_base(queue_root, target, "threebench")
            blink_base = _component_base(queue_root, target, "blink-v5")
            resident_root = _find_complete_root(resident_base, lambda path, t=target: _resident_complete(path, t))
            blink_root = _find_complete_root(blink_base, lambda path, t=target: _blink_complete(path, t))
            actions = {"resident": "skip_complete" if resident_root else "pending", "blink_v5": "skip_complete" if blink_root else "pending"}
            if resident_root is None:
                resident_root = _reserve_root(resident_base)
                audit_root = resident_root.with_name(resident_root.name.replace("threebench", "threebench_audit", 1))
                summary["events"].append({"target": target.slug, "component": "resident", "state": "starting", "run_root": str(resident_root), "audit_root": str(audit_root), "ports": ports})
                _atomic_json(summary_path, summary)
                _run_child(resident_command(target, resident_root, audit_root, ports=ports), state=state, label=f"resident {target.slug}", log_path=model_root / "lifecycle/resident.log")
                if not _resident_complete(resident_root, target):
                    raise QueueError(f"resident runner returned without complete VStar/MMStar/ZoomBench evidence: {target.slug}")
                actions["resident"] = "execute"
                summary["events"].append({"target": target.slug, "component": "resident", "state": "complete", "run_root": str(resident_root)})
            if blink_root is None:
                blink_root = _reserve_root(blink_base)
                summary["events"].append({"target": target.slug, "component": "BLINK-v5", "state": "starting", "run_root": str(blink_root), "port": ports["blink"]})
                _atomic_json(summary_path, summary)
                _run_child(blink_command(target, blink_root, port=ports["blink"]), state=state, label=f"BLINK-v5 {target.slug}", log_path=model_root / "lifecycle/blink-v5.log")
                if not _blink_complete(blink_root, target):
                    raise QueueError(f"BLINK-v5 runner returned without complete 1901-row evidence: {target.slug}")
                actions["blink_v5"] = "execute"
                summary["events"].append({"target": target.slug, "component": "BLINK-v5", "state": "complete", "run_root": str(blink_root)})
            record = _target_record(target, index, queue_root, resident_root=resident_root, blink_root=blink_root, actions=actions, readiness=readiness[target.slug])
            if record["eligible_for_model_judgment"] is not True:
                raise QueueError(f"target is missing one or more complete benchmark cells: {target.slug}")
            summary["targets"].append(record)
            summary["events"].append({"target": target.slug, "state": "complete"})
            summary["completed_targets"] = [row["slug"] for row in summary["targets"]]
            _atomic_json(summary_path, summary)
        summary["status"] = "complete"
        summary["summary_sha256"] = _sha({key: value for key, value in summary.items() if key != "summary_sha256"})
        _metadata_public(summary)
        _atomic_json(summary_path, summary)
        _atomic_json(status_path, {"schema_version": STATUS_SCHEMA, "state": "complete", "summary_path": str(summary_path), "summary_sha256": summary["summary_sha256"], "completed_targets": summary["completed_targets"], "serial": True, "aggregate_only": True, "sample_level_output": False})
        return summary
    except BaseException as exc:
        summary["status"] = "failed"
        summary["error"] = str(exc)
        summary["error_type"] = type(exc).__name__
        try:
            _metadata_public(summary)
            _atomic_json(summary_path, summary)
            _atomic_json(status_path, {"schema_version": STATUS_SCHEMA, "state": "failed", "summary_path": str(summary_path), "error_type": type(exc).__name__, "serial": True, "aggregate_only": True, "sample_level_output": False})
        except Exception:
            pass
        raise
    finally:
        state.cleanup()
        _ACTIVE = previous_active
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="render a pure plan")
    mode.add_argument("--execute", action="store_true", help="wait for complete model exports and run the formal suite")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--wait-timeout-seconds", type=float, default=DEFAULT_WAIT_TIMEOUT_SECONDS)
    parser.add_argument("--poll-seconds", type=float, default=DEFAULT_POLL_SECONDS)
    parser.add_argument("--no-wait", action="store_true", help="fail immediately if either export is not complete")
    parser.add_argument("--target", action="append", help="target slug; repeat to select a subset")
    return parser


def _select(values: Sequence[str] | None) -> tuple[ModelSpec, ...]:
    if not values:
        return MODEL_SPECS
    by_slug = {key: target for target in MODEL_SPECS for key in (target.slug, target.training_root.name)}
    result = []
    for value in values:
        if value not in by_slug:
            raise QueueError(f"unknown target slug: {value}")
        result.append(by_slug[value])
    return tuple(result)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        targets = _select(args.target)
        if args.dry_run:
            print(json.dumps(render_plan(queue_root=args.run_root, targets=targets), ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            return 0
        if args.wait_timeout_seconds < 0 or args.poll_seconds <= 0:
            raise QueueError("wait timeout must be non-negative and poll seconds must be positive")
        result = execute(queue_root=args.run_root, targets=targets, timeout_seconds=args.wait_timeout_seconds, poll_seconds=args.poll_seconds, no_wait=args.no_wait)
        print(json.dumps({"schema_version": SUMMARY_SCHEMA, "status": result.get("status"), "summary": str(args.run_root / "summary.json"), "completed_targets": result.get("completed_targets", [])}, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        return 0
    except (QueueError, OSError, ValueError, TypeError) as exc:
        print(json.dumps({"schema_version": SCHEMA_VERSION, "status": "failed_closed", "aggregate_only": True, "sample_level_output": False, "error": str(exc)}, ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
