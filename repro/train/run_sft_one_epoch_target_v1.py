#!/usr/bin/env python3
"""Train and export one requested Qwen3.5 LoRA-SFT target on GPUs 0--7.

The four supported targets form the controlled comparison requested on
2026-08-16: raw 9B/27B on the complete B28 panel and raw 9B/27B on the
published B57 10k panel.  Every target sees every selected row exactly once.
The script owns no resource provisioning; execute only uses the existing
eight-GPU lease in ``Locks/opd_gpu_0_7.lock``.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


WORKSPACE = Path("/minimax-3d-rw-backup/users/jiazhi/H_Workspace")
TRAIN_DIR = Path(__file__).resolve().parent
OUTPUT_PARENT = WORKSPACE / "Output/sft_one_epoch_matrix_v1"
LOCK = WORKSPACE / "Locks/opd_gpu_0_7.lock"
RUNTIME = WORKSPACE / "UV_Env/verl-opd-qwen35/bin/python"
VERL = WORKSPACE / "Codes/verl"
MERGER = TRAIN_DIR / "merge_qwen35_sharded_lora_to_hf_v1.py"
CUDA = "0,1,2,3,4,5,6,7"
EXPECTED_UID = 30853
WORLD_SIZE = 8
SEED = 42
LORA_RANK = 16
LORA_ALPHA = 16
PREFLIGHT_ROOT = WORKSPACE / "Output/requested_sft_matrix_v1"
RUNTIME_ROOT = Path(f"/tmp/opd-sft-one-epoch-{EXPECTED_UID}")


class TargetError(RuntimeError):
    pass


@dataclass(frozen=True)
class Target:
    slug: str
    model_size: str
    model_path: str
    dataset_name: str
    parquet: str
    parquet_sha256: str
    manifests: tuple[str, ...]
    rows: int
    global_batch: int
    steps: int
    learning_rate: str
    max_length: int
    custom_dataset: str
    custom_class: str
    dynamic_batch: bool
    length_bucket: bool
    shuffle: bool


_B28_ROOT = WORKSPACE / "Dataset/b58_b28_sft_1536_exact_republish_v1"
_B57_ROOT = WORKSPACE / "Dataset/b57_balanced_fine_multi_sft_10k_v9_retry1"
_B28_ADAPTER = TRAIN_DIR / "b28_b26_safe_fine_multi_1536_v1_dataset.py"
_B57_ADAPTER = TRAIN_DIR / "b54_10k_sft_dataset_v1.py"


def _target(slug: str, model_size: str, dataset_name: str) -> Target:
    if model_size not in {"9b", "27b"}:
        raise TargetError(f"unsupported model size: {model_size}")
    model_path = WORKSPACE / f"Ckpt/Qwen3.5-{'9B' if model_size == '9b' else '27B'}"
    if dataset_name == "b28":
        return Target(
            slug=slug,
            model_size=model_size,
            model_path=str(model_path),
            dataset_name="B28/B26 safe fine-multi 1536",
            parquet=str(_B28_ROOT / "processed/train_1536.parquet"),
            parquet_sha256="9d4de6b1e4a0e3efe5a398a91dc33f93e154dbb0eed9f29bf746b7cd13d512a9",
            manifests=(
                str(_B28_ROOT / "manifest/build_receipt.json"),
                str(_B28_ROOT / "manifest/selection_receipt.json"),
            ),
            rows=1536,
            global_batch=48,
            steps=32,
            learning_rate="2e-6",
            max_length=16384 if model_size == "9b" else 9216,
            custom_dataset=str(_B28_ADAPTER),
            custom_class="B28B26SafeFineMulti1536Dataset",
            dynamic_batch=False,
            length_bucket=False,
            shuffle=True,
        )
    if dataset_name == "b57":
        return Target(
            slug=slug,
            model_size=model_size,
            model_path=str(model_path),
            dataset_name="B57 balanced fine/general/multi 10k v9 retry1",
            parquet=str(_B57_ROOT / "processed/train_10000.parquet"),
            parquet_sha256="9f56d58c076c255df3bc660ba3c193b1cff8dd69c51ad2f73c844f5f2a8c49b0",
            manifests=(
                str(_B57_ROOT / "manifest/build_receipt.json"),
                str(_B57_ROOT / "manifest/final_gate.json"),
            ),
            rows=10000,
            global_batch=80,
            steps=125,
            learning_rate="2e-5" if model_size == "9b" else "1e-5",
            max_length=16384 if model_size == "9b" else 9216,
            custom_dataset=str(_B57_ADAPTER),
            custom_class="B54TenKSFTDataset",
            dynamic_batch=True,
            length_bucket=True,
            shuffle=False,
        )
    raise TargetError(f"unsupported dataset: {dataset_name}")


TARGETS = {
    "b28-raw9b-1epoch": _target("b28-raw9b-1epoch", "9b", "b28"),
    "b28-raw27b-1epoch": _target("b28-raw27b-1epoch", "27b", "b28"),
    "b57-raw9b-1epoch": _target("b57-raw9b-1epoch", "9b", "b57"),
    "b57-raw27b-1epoch": _target("b57-raw27b-1epoch", "27b", "b57"),
}


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def regular(path: Path, label: str) -> os.stat_result:
    try:
        info = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise TargetError(f"missing {label}: {path}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise TargetError(f"{label} is not a regular file: {path}")
    return info


def directory(path: Path, label: str) -> os.stat_result:
    try:
        info = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise TargetError(f"missing {label}: {path}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise TargetError(f"{label} is not a real directory: {path}")
    return info


def read_json(path: Path, label: str) -> dict[str, Any]:
    regular(path, label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TargetError(f"invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise TargetError(f"{label} is not an object: {path}")
    seal = value.get("seal_sha256")
    if isinstance(seal, str):
        body = {key: item for key, item in value.items() if key != "seal_sha256"}
        if hashlib.sha256(canonical(body)).hexdigest() != seal:
            raise TargetError(f"{label} seal differs: {path}")
    return value


def write_json_create_once(path: Path, body: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = dict(body)
    clean["seal_sha256"] = hashlib.sha256(canonical(clean)).hexdigest()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        os.write(descriptor, canonical(clean) + b"\n")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def validate_render_preflight(target: Target) -> dict[str, Any]:
    short = "b28" if target.parquet == str(_B28_ROOT / "processed/train_1536.parquet") else "b57"
    receipt_path = PREFLIGHT_ROOT / f"data_render_preflight_{short}_9216_v1.json"
    value = read_json(receipt_path, "9216-token render preflight")
    receipt_target = value.get("target") or {}
    dataset = value.get("dataset") or {}
    implementation = value.get("dataset_implementation") or {}
    processor = value.get("processor") or {}
    statistics = value.get("statistics") or {}
    input_tokens = statistics.get("input_tokens") or {}
    audit_source = value.get("audit_source") or {}
    source_path = Path(str(audit_source.get("path", "")))
    if (
        value.get("schema_version") != "requested_sft_render_9216_v1"
        or value.get("status") != "passed"
        or value.get("strict_max_length") != 9_216
        or value.get("cpu_only") is not True
        or value.get("gpu_used") is not False
        or value.get("aggregate_only") is not True
        or value.get("sample_payload_written") is not False
        or value.get("proves_9b_16384_arm") is not True
        or value.get("proves_27b_9216_arm") is not True
        or receipt_target.get("slug") != short
        or receipt_target.get("parquet") != target.parquet
        or receipt_target.get("parquet_sha256") != target.parquet_sha256
        or receipt_target.get("rows") != target.rows
        or receipt_target.get("dataset_impl") != target.custom_dataset
        or receipt_target.get("dataset_class") != target.custom_class
        or dataset.get("path") != target.parquet
        or dataset.get("sha256") != target.parquet_sha256
        or dataset.get("rows") != target.rows
        or implementation.get("path") != target.custom_dataset
        or implementation.get("sha256") != sha256_file(Path(target.custom_dataset))
        or implementation.get("class") != target.custom_class
        or processor.get("identical_processor_descriptors") is not True
        or sorted(processor.get("applies_to_models", [])) != sorted(
            [str(WORKSPACE / "Ckpt/Qwen3.5-9B"), str(WORKSPACE / "Ckpt/Qwen3.5-27B")]
        )
        or statistics.get("rows_seen") != target.rows
        or statistics.get("rows_rendered") != target.rows
        or statistics.get("rows_failed") != 0
        or not isinstance(input_tokens.get("max"), int)
        or input_tokens.get("max") > 9_216
    ):
        raise TargetError(f"9216-token render preflight is not a passing exact receipt: {receipt_path}")
    regular(source_path, "render preflight source")
    if audit_source.get("sha256") != sha256_file(source_path):
        raise TargetError("render preflight source SHA differs")
    return {
        "path": str(receipt_path),
        "sha256": sha256_file(receipt_path),
        "status": "passed",
        "strict_max_length": 9_216,
        "rows_rendered": target.rows,
        "rows_failed": 0,
        "max_input_tokens": int(input_tokens["max"]),
        "processor_descriptors_identical_for_9b_27b": True,
        "audit_source": {"path": str(source_path), "sha256": audit_source["sha256"]},
    }


def validate(target: Target, run_root: Path, *, execute: bool) -> dict[str, Any]:
    if target.rows != target.global_batch * target.steps:
        raise TargetError("target does not expose exactly one epoch")
    parquet = Path(target.parquet)
    regular(parquet, "training parquet")
    observed = sha256_file(parquet)
    if observed != target.parquet_sha256:
        raise TargetError(f"training parquet SHA differs: {observed}")
    manifests: list[dict[str, Any]] = []
    for raw in target.manifests:
        path = Path(raw)
        value = read_json(path, "dataset manifest")
        if value.get("status") not in {"published", "passed", "passed_cpu_build_audit"}:
            raise TargetError(f"dataset manifest status is not released: {path}")
        manifests.append({"path": str(path), "sha256": sha256_file(path), "status": value.get("status")})
    model = Path(target.model_path)
    directory(model, "base model")
    for name in ("config.json", "tokenizer_config.json", "chat_template.jinja", "model.safetensors.index.json"):
        regular(model / name, f"base model {name}")
    adapter = Path(target.custom_dataset)
    regular(adapter, "custom dataset adapter")
    if not RUNTIME.exists() or not os.access(RUNTIME, os.X_OK):
        raise TargetError(f"training runtime is unavailable: {RUNTIME}")
    regular(MERGER, "LoRA merger")
    directory(VERL, "veRL root")
    lock_info = regular(LOCK, "GPU 0-7 lock")
    render_preflight = validate_render_preflight(target)
    if run_root.parent != OUTPUT_PARENT or run_root == OUTPUT_PARENT or run_root.exists() or run_root.is_symlink():
        raise TargetError(f"run root must be a new direct child of {OUTPUT_PARENT}: {run_root}")
    if execute and (os.getuid(), os.getgid()) != (EXPECTED_UID, EXPECTED_UID):
        raise TargetError(f"execute requires uid/gid {EXPECTED_UID}")
    return {
        "schema_version": "sft_one_epoch_target_v1",
        "target": asdict(target),
        "one_epoch": True,
        "rows_exposed": target.rows,
        "optimizer_steps": target.steps,
        "world_size": WORLD_SIZE,
        "physical_gpus": list(range(WORLD_SIZE)),
        "additional_resources_requested": False,
        "dataset": {"path": str(parquet), "sha256": observed, "manifests": manifests},
        "model": {"path": str(model), "config_sha256": sha256_file(model / "config.json")},
        "custom_dataset": {"path": str(adapter), "sha256": sha256_file(adapter), "class": target.custom_class},
        "render_preflight": render_preflight,
        "lock": {"path": str(LOCK), "device": lock_info.st_dev, "inode": lock_info.st_ino},
        "run_root": str(run_root),
    }


def build_train_command(target: Target, run_root: Path) -> list[str]:
    command = [
        str(RUNTIME), "-m", "torch.distributed.run", "--standalone", "--nnodes=1", "--nproc-per-node=8",
        "-m", "verl.trainer.sft_trainer",
        f"data.train_files={target.parquet}", "data.val_files=null", f"data.train_max_samples={target.rows}",
        f"data.train_batch_size={target.global_batch}", "data.micro_batch_size_per_gpu=1", "data.messages_key=prompt",
        "+data.image_key=images", f"data.custom_cls.path=file://{target.custom_dataset}", f"data.custom_cls.name={target.custom_class}",
        "data.enable_thinking_default=False", "data.ignore_input_ids_mismatch=True", f"data.max_length={target.max_length}",
        "data.truncation=error", "data.pad_mode=no_padding", f"data.use_dynamic_bsz={'True' if target.dynamic_batch else 'False'}",
        f"data.max_token_len_per_gpu={target.max_length}", "data.num_workers=8", f"+data.shuffle={'True' if target.shuffle else 'False'}",
        f"model.path={target.model_path}", "model.enable_gradient_checkpointing=True", "model.use_remove_padding=True",
        f"model.lora_rank={LORA_RANK}", f"model.lora_alpha={LORA_ALPHA}", "model.target_modules=all-linear",
        "engine=fsdp", "engine.strategy=fsdp", "engine.fsdp_size=8", f"engine.seed={SEED}",
        "engine.use_torch_compile=False", "engine.dtype=bfloat16", f"optim.lr={target.learning_rate}",
        "optim.lr_warmup_steps_ratio=0.0", "optim.lr_scheduler_type=cosine", "optim.clip_grad=1.0",
        f"trainer.seed={SEED}", "trainer.n_gpus_per_node=8", f"trainer.total_training_steps={target.steps}",
        "trainer.total_epochs=1", "trainer.save_freq=-1", f"+trainer.save_steps=[{target.steps}]", "trainer.test_freq=-1",
        "trainer.resume_mode=disable", "trainer.logger=['console']", f"trainer.default_local_dir={run_root / 'checkpoints'}",
        "trainer.project_name=sft_one_epoch_matrix_v1", f"trainer.experiment_name={target.slug}",
        "checkpoint.save_contents=['model','extra']", "checkpoint.load_contents=['model','extra']", "+checkpoint.strict=True",
        f"hydra.run.dir={run_root}/hydra_rank_${{oc.env:LOCAL_RANK,0}}", "hydra.job.chdir=False",
    ]
    if target.length_bucket:
        command.extend(("+data.length_bucket_batch=True", f"+data.length_bucket_seed={SEED}", "+data.length_bucket_image_token_proxy=1024"))
    return command


def prepare_runtime_dirs() -> dict[str, str]:
    """Create short, uid-owned cache paths before importing Triton/FLA.

    The interactive GPU pod is entered through a root SSH shell and the
    launcher is then executed with ``setpriv``.  Inheriting ``HOME=/root``
    makes Triton's early backend probe fail closed to CPU for uid 30853.  FLA
    caches that result and later tries to enter ``torch.cpu.device`` for CUDA
    tensors.  Keep every runtime path local, writable and short so the backend
    probe deterministically resolves to CUDA on all eight ranks.
    """

    paths = {
        "HOME": RUNTIME_ROOT / "home",
        "TMPDIR": RUNTIME_ROOT / "tmp",
        "XDG_CACHE_HOME": RUNTIME_ROOT / "xdg_cache",
        "XDG_CONFIG_HOME": RUNTIME_ROOT / "xdg_config",
        "TORCHINDUCTOR_CACHE_DIR": RUNTIME_ROOT / "torchinductor",
        "TRITON_CACHE_DIR": RUNTIME_ROOT / "triton",
        "MPLCONFIGDIR": RUNTIME_ROOT / "mplconfig",
    }
    RUNTIME_ROOT.mkdir(mode=0o700, parents=False, exist_ok=True)
    root_info = os.stat(RUNTIME_ROOT, follow_symlinks=False)
    if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode):
        raise TargetError(f"runtime root is not a real directory: {RUNTIME_ROOT}")
    if (root_info.st_uid, root_info.st_gid) != (EXPECTED_UID, EXPECTED_UID):
        raise TargetError(f"runtime root owner differs: {RUNTIME_ROOT}")
    for path in paths.values():
        path.mkdir(mode=0o700, exist_ok=True)
        info = os.stat(path, follow_symlinks=False)
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise TargetError(f"runtime path is not a real directory: {path}")
        if (info.st_uid, info.st_gid) != (EXPECTED_UID, EXPECTED_UID):
            raise TargetError(f"runtime path owner differs: {path}")
    return {key: str(path) for key, path in paths.items()}


def runtime_env(*, cuda: bool) -> dict[str, str]:
    value = os.environ.copy()
    value.update(
        {
            "CUDA_VISIBLE_DEVICES": CUDA if cuda else "",
            "NVIDIA_VISIBLE_DEVICES": CUDA if cuda else "",
            "PYTHONPATH": str(VERL),
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "HF_HUB_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "LOGNAME": f"opd_uid_{EXPECTED_UID}",
            "USER": f"opd_uid_{EXPECTED_UID}",
            **prepare_runtime_dirs(),
        }
    )
    return value


def gpu_backend_audit() -> dict[str, Any]:
    probe = (
        "import json, torch; import fla.utils as u; "
        "print(json.dumps({'torch_cuda': torch.cuda.is_available(), "
        "'device_count': torch.cuda.device_count(), 'fla_device': u.device, "
        "'fla_platform': u.device_platform, "
        "'fla_torch_lib': u.device_torch_lib.__name__}))"
    )
    result = subprocess.run(
        [str(RUNTIME), "-B", "-c", probe],
        cwd=str(VERL),
        env=runtime_env(cuda=True),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )
    try:
        payload = json.loads(result.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise TargetError(f"CUDA/FLA backend preflight was not JSON: {result.stderr[-1000:]}") from exc
    if (
        result.returncode != 0
        or payload.get("torch_cuda") is not True
        or payload.get("device_count") != WORLD_SIZE
        or payload.get("fla_device") != "cuda"
        or payload.get("fla_platform") != "cuda"
        or payload.get("fla_torch_lib") != "torch.cuda"
    ):
        raise TargetError(f"CUDA/FLA backend preflight failed: {payload}")
    return payload


def gpu_audit() -> dict[str, Any]:
    binary = shutil.which("nvidia-smi")
    if binary is None:
        raise TargetError("nvidia-smi is unavailable")
    result = subprocess.run(
        [binary, "--query-gpu=index", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    indices = sorted(int(line.strip()) for line in result.stdout.splitlines() if line.strip()) if result.returncode == 0 else []
    if indices != list(range(WORLD_SIZE)):
        raise TargetError(f"GPU audit saw {indices}, expected 0..7")
    return {"indices": indices, "cuda_visible_devices": CUDA}


def run_logged(command: Sequence[str], log: Path, *, cwd: Path, env: Mapping[str, str]) -> None:
    with log.open("xb") as stream:
        result = subprocess.run(list(command), cwd=str(cwd), env=dict(env), stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, check=False)
    if result.returncode != 0:
        raise TargetError(f"command failed with status {result.returncode}; see {log}")


def export_checkpoint(target: Target, run_root: Path) -> dict[str, Any]:
    checkpoint = run_root / "checkpoints" / f"global_step_{target.steps}"
    directory(checkpoint, "terminal checkpoint")
    regular(checkpoint / "lora_train_meta.json", "terminal LoRA metadata")
    staging = run_root / ".fsdp_export_final.incomplete"
    merged = run_root / "merged/final_hf_official_chat_v1"
    if staging.exists() or staging.is_symlink() or merged.exists() or merged.is_symlink():
        raise TargetError("export target already exists")
    fsdp_command = [
        str(RUNTIME), "-B", "-m", "verl.model_merger", "merge", "--backend", "fsdp",
        "--local_dir", str(checkpoint), "--target_dir", str(staging),
    ]
    run_logged(fsdp_command, run_root / "artifacts/fsdp_export.log", cwd=VERL, env=runtime_env(cuda=False))
    adapter = staging / "lora_adapter"
    directory(adapter, "exported LoRA adapter")
    merge_command = [
        str(RUNTIME), "-B", str(MERGER), "--execute", "--base-model", target.model_path,
        "--adapter", str(adapter), "--output", str(merged), "--checkpoint-meta", str(checkpoint / "lora_train_meta.json"),
    ]
    run_logged(merge_command, run_root / "artifacts/hf_merge.log", cwd=TRAIN_DIR, env=runtime_env(cuda=False))
    directory(merged, "merged HF model")
    regular(merged / "config.json", "merged config")
    weight_files = sorted(merged.glob("*.safetensors"))
    if not weight_files:
        raise TargetError("merged HF model has no safetensors")
    inventory = {
        "checkpoint": str(checkpoint),
        "merged_model": str(merged),
        "weight_files": [{"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in weight_files],
        "merge_receipt_sha256": sha256_file(merged / "merge_receipt.json"),
        "successful_staging_removed": True,
    }
    # Exact, self-created, reproducible intermediate; the FSDP checkpoint and
    # verified final model remain.  Removing it avoids retaining another full
    # 9B/27B copy for every arm.
    shutil.rmtree(staging)
    return inventory


def execute(target: Target, run_root: Path, plan: Mapping[str, Any]) -> dict[str, Any]:
    OUTPUT_PARENT.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(LOCK, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0))
    started = datetime.now(timezone.utc).isoformat()
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        runtime = gpu_audit()
        runtime["backend"] = gpu_backend_audit()
        run_root.mkdir(mode=0o700)
        (run_root / "artifacts").mkdir(mode=0o700)
        write_json_create_once(run_root / "artifacts/launch_plan.json", dict(plan))
        command = build_train_command(target, run_root)
        run_logged(command, run_root / "artifacts/train.log", cwd=VERL, env=runtime_env(cuda=True))
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)
    exported = export_checkpoint(target, run_root)
    result = {
        **dict(plan),
        "status": "complete",
        "started_at_utc": started,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_runtime": runtime,
        "train_command": command,
        "export": exported,
        "gpu_used": True,
    }
    write_json_create_once(run_root / "artifacts/completion.json", result)
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    mode = value.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    value.add_argument("--target", choices=tuple(TARGETS), required=True)
    value.add_argument("--run-root", type=Path)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    target = TARGETS[args.target]
    run_root = (args.run_root or (OUTPUT_PARENT / target.slug)).absolute()
    try:
        plan = validate(target, run_root, execute=args.execute)
        plan["train_command"] = build_train_command(target, run_root)
        if args.dry_run:
            print(json.dumps({**plan, "status": "dry_run_passed", "writes_performed": 0, "gpu_queried": False}, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        result = execute(target, run_root, plan)
        print(json.dumps({"status": result["status"], "target": target.slug, "merged_model": result["export"]["merged_model"]}, sort_keys=True))
        return 0
    except (TargetError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "failed", "target": args.target, "error": str(exc)}, ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
