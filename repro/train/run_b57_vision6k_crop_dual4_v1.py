#!/usr/bin/env python3
"""Run two fixed-teacher Vision-OPD crop experiments on disjoint 4-GPU sets.

This is an execution adapter for one 8-GPU interactive pod.  Each arm uses
the legacy Vision-OPD colocated FSDP teacher path, so the student and teacher
share four visible GPUs.  The two arms use independent local Ray runtimes and
cache trees and therefore can run concurrently without sharing CUDA devices.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


WORKSPACE = Path("/minimax-3d-rw-backup/users/jiazhi/H_Workspace")
REFERENCE = WORKSPACE / "Codes/Vision-OPD-reference"
VERL_ROOT = WORKSPACE / "Codes/verl"
PYTHON = WORKSPACE / "UV_Env/verl-opd-qwen35/bin/python"
DATA = WORKSPACE / "Dataset/processed/b1_vision_opd_6k/train_decontaminated.parquet"
TEMPLATE = REFERENCE / "chat_templates/perception_chat_template_qwen35.jinja"
STUDENT = (
    WORKSPACE
    / "Output/sft_one_epoch_matrix_v1/b57-raw9b-1epoch/merged/final_hf_official_chat_v1"
)
RAW9_TEACHER = WORKSPACE / "Ckpt/Qwen3.5-9B"
B57_27B_TEACHER = (
    WORKSPACE
    / "Output/sft_one_epoch_matrix_v1/b57-raw27b-1epoch/merged/final_hf_official_chat_v1"
)
OUTPUT_PARENT = WORKSPACE / "Output/opd_qwen35_9b"

DATA_ROWS = 6_241
TRAIN_BATCH_SIZE = 96
TOTAL_STEPS = 65
SAVE_FREQ = 13
SEED = 42
# The successful 8-GPU control expands each driver batch by rollout.n=8,
# yielding 768 trajectories and 96 local trajectories per actor rank.  With
# four actor ranks each rank receives 192 trajectories, so use a local PPO
# mini-batch of 192 to preserve one optimizer update per driver step.
PPO_MINI_BATCH_SIZE = TRAIN_BATCH_SIZE * 8 // 4


@dataclass(frozen=True)
class Arm:
    key: str
    teacher: Path
    cuda_devices: str
    run_name: str
    ray_root: Path

    @property
    def run_dir(self) -> Path:
        return OUTPUT_PARENT / self.run_name


ARMS = (
    Arm(
        key="raw9_teacher",
        teacher=RAW9_TEACHER,
        cuda_devices="0,1,2,3",
        run_name="b57_10k_init_vision6k_crop_raw9_teacher_dual4_s65_v1",
        ray_root=Path("/tmp/b57-vision6k-crop-raw9-teacher-ray"),
    ),
    Arm(
        key="b57_27b_teacher",
        teacher=B57_27B_TEACHER,
        cuda_devices="4,5,6,7",
        run_name="b57_10k_init_vision6k_crop_b57_27b_teacher_dual4_s65_v1",
        ray_root=Path("/tmp/b57-vision6k-crop-b57-27b-teacher-ray"),
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require_inputs() -> dict[str, Any]:
    required_files = (PYTHON, DATA, TEMPLATE, STUDENT / "config.json")
    required_files += tuple(arm.teacher / "config.json" for arm in ARMS)
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing required inputs: {missing}")
    if not os.access(PYTHON, os.X_OK):
        raise RuntimeError(f"training Python is not executable: {PYTHON}")
    if not OUTPUT_PARENT.is_dir():
        raise RuntimeError(f"missing output parent: {OUTPUT_PARENT}")
    return {
        "student": str(STUDENT),
        "student_config_sha256": sha256(STUDENT / "config.json"),
        "data": str(DATA),
        "data_rows": DATA_ROWS,
        "teacher_configs": {
            arm.key: sha256(arm.teacher / "config.json") for arm in ARMS
        },
    }


def build_command(arm: Arm) -> list[str]:
    run_dir = arm.run_dir
    checkpoints = run_dir / "checkpoints"
    rollouts = run_dir / "rollouts"
    hydra = run_dir / "hydra"
    return [
        str(PYTHON),
        "-m",
        "verl.trainer.main_ppo",
        "--config-name",
        "vopd",
        f'data.train_files=["{DATA}"]',
        "data.val_files=[]",
        f"data.train_max_samples={DATA_ROWS}",
        "data.filter_overlong_prompts=False",
        "data.max_prompt_length=8192",
        "data.max_response_length=1024",
        "data.truncation=error",
        "data.shuffle=True",
        f"data.seed={SEED}",
        "data.trust_remote_code=True",
        "data.return_multi_modal_inputs=True",
        "data.image_key=images",
        f"data.train_batch_size={TRAIN_BATCH_SIZE}",
        "data.dataloader_num_workers=8",
        f"actor_rollout_ref.model.path={STUDENT}",
        "actor_rollout_ref.model.trust_remote_code=True",
        "actor_rollout_ref.model.use_remove_padding=True",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        "actor_rollout_ref.rollout.n=8",
        "actor_rollout_ref.actor.optim.lr=2e-6",
        "actor_rollout_ref.actor.optim.lr_warmup_steps=10",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={PPO_MINI_BATCH_SIZE}",
        "actor_rollout_ref.actor.use_dynamic_bsz=True",
        "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=9216",
        "actor_rollout_ref.actor.fsdp_config.param_offload=True",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True",
        "actor_rollout_ref.actor.clip_ratio_high=0.3",
        "actor_rollout_ref.actor.clip_ratio_low=0.2",
        "actor_rollout_ref.actor.use_kl_loss=False",
        "actor_rollout_ref.actor.policy_loss.loss_mode=vopd",
        "actor_rollout_ref.actor.calculate_entropy=False",
        "actor_rollout_ref.actor.self_distillation.distillation_topk=100",
        "actor_rollout_ref.actor.self_distillation.max_reprompt_len=10240",
        "actor_rollout_ref.actor.self_distillation.is_clip=2.0",
        "actor_rollout_ref.actor.self_distillation.teacher_always_on=True",
        "actor_rollout_ref.actor.self_distillation.teacher_model_source=fixed",
        f"actor_rollout_ref.actor.self_distillation.teacher_model_path={arm.teacher}",
        "actor_rollout_ref.actor.self_distillation.teacher_regularization=ema",
        "actor_rollout_ref.actor.self_distillation.teacher_update_rate=0.0",
        "actor_rollout_ref.actor.self_distillation.teacher_image_key=bbox_images",
        "actor_rollout_ref.actor.self_distillation.fallback_to_policy_loss_on_missing_teacher=False",
        "actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=True",
        "actor_rollout_ref.actor.self_distillation.alpha=0.5",
        "actor_rollout_ref.actor.self_distillation.include_environment_feedback=False",
        "algorithm.rollout_correction.rollout_is=token",
        "algorithm.rollout_correction.rollout_is_threshold=2.0",
        "algorithm.adv_estimator=grpo",
        "algorithm.norm_adv_by_std_in_grpo=False",
        "algorithm.use_kl_in_reward=False",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.7",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.rollout.max_num_batched_tokens=9216",
        "actor_rollout_ref.rollout.max_model_len=9216",
        "+actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.pass_config.fuse_allreduce_rms=False",
        "+actor_rollout_ref.rollout.engine_kwargs.vllm.kernel_config.enable_flashinfer_autotune=False",
        "actor_rollout_ref.rollout.response_length=1024",
        "actor_rollout_ref.rollout.calculate_log_probs=True",
        "actor_rollout_ref.rollout.agent.num_workers=4",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.ref.fsdp_config.param_offload=True",
        "reward_model.enable=False",
        "reward_model.use_reward_loop=False",
        "custom_reward_function.path=null",
        f"critic.model.path={STUDENT}",
        "trainer.project_name=Vision-OPD",
        f"trainer.group_name={arm.run_name}",
        f"trainer.experiment_name={arm.run_name}",
        'trainer.logger=["console","tensorboard"]',
        "trainer.n_gpus_per_node=4",
        "trainer.nnodes=1",
        f"trainer.save_freq={SAVE_FREQ}",
        "trainer.test_freq=-1",
        "trainer.max_actor_ckpt_to_keep=null",
        "trainer.total_epochs=1",
        f"trainer.total_training_steps={TOTAL_STEPS}",
        "trainer.resume_mode=auto",
        "trainer.val_before_train=False",
        f"trainer.default_local_dir={checkpoints}",
        f"trainer.rollout_data_dir={rollouts}",
        f"hydra.run.dir={hydra}",
        "ray_kwargs.ray_init.num_cpus=48",
        "+ray_kwargs.ray_init.num_gpus=4",
        "+ray_kwargs.ray_init.address=local",
        "+ray_kwargs.ray_init.include_dashboard=False",
        f"+ray_kwargs.ray_init._temp_dir={arm.ray_root}",
        f"actor_rollout_ref.model.custom_chat_template_file={TEMPLATE}",
    ]


def arm_environment(arm: Arm) -> dict[str, str]:
    env = os.environ.copy()
    cache = arm.run_dir / "cache"
    short_tmp = Path("/tmp") / ("vopd-r9" if arm.key == "raw9_teacher" else "vopd-r27")
    env.update(
        {
            "HOME": str(cache / "home"),
            "USER": "jiazhi",
            "LOGNAME": "jiazhi",
            "CUDA_VISIBLE_DEVICES": arm.cuda_devices,
            "NVIDIA_VISIBLE_DEVICES": arm.cuda_devices,
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            # Canoe exports bond1, but this interactive pod exposes eth0.
            # Pin the actual pod interface so NCCL/Gloo bootstrap succeeds.
            "NCCL_SOCKET_IFNAME": "eth0",
            "GLOO_SOCKET_IFNAME": "eth0",
            "NCCL_DEBUG": "WARN",
            "PYTHONPATH": str(REFERENCE),
            "PYTHONUNBUFFERED": "1",
            "PYTHONNOUSERSITE": "1",
            "VLLM_USE_V1": "1",
            "VLLM_NO_USAGE_STATS": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "XDG_CACHE_HOME": str(cache / "xdg"),
            "XDG_CONFIG_HOME": str(cache / "config"),
            "HF_HOME": str(cache / "hf"),
            "HF_DATASETS_CACHE": str(cache / "hf" / "datasets"),
            "TRANSFORMERS_CACHE": str(cache / "hf" / "transformers"),
            "VLLM_CACHE_ROOT": str(cache / "vllm"),
            "CUDA_CACHE_PATH": str(cache / "cuda"),
            "FLASHINFER_WORKSPACE_BASE": str(cache / "flashinfer"),
            "TORCHINDUCTOR_CACHE_DIR": str(cache / "torchinductor"),
            "TRITON_CACHE_DIR": str(cache / "triton"),
            # vLLM uses AF_UNIX sockets and Linux caps sun_path at 107 bytes.
            "TMPDIR": str(short_tmp),
        }
    )
    return env


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def prepare_arm(arm: Arm, inputs: dict[str, Any]) -> tuple[list[str], Any]:
    run_dir = arm.run_dir
    for child in ("checkpoints", "rollouts", "hydra", "artifacts", "cache"):
        (run_dir / child).mkdir(parents=True, exist_ok=True)
    cache = run_dir / "cache"
    for child in (
        "home",
        "tmp",
        "xdg",
        "config",
        "hf/datasets",
        "hf/transformers",
        "vllm",
        "cuda",
        "flashinfer",
        "torchinductor",
        "triton",
    ):
        (cache / child).mkdir(parents=True, exist_ok=True)
    (Path("/tmp") / ("vopd-r9" if arm.key == "raw9_teacher" else "vopd-r27")).mkdir(
        parents=True, exist_ok=True
    )
    command = build_command(arm)
    write_json(
        run_dir / "artifacts/launch_plan.json",
        {
            "schema_version": "b57_vision6k_crop_dual4_v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "arm": arm.key,
            "student": str(STUDENT),
            "teacher": str(arm.teacher),
            "teacher_mode": "fixed",
            "student_image_key": "images",
            "teacher_image_key": "bbox_images",
            "cuda_devices": arm.cuda_devices,
            "world_size": 4,
            "steps": TOTAL_STEPS,
            "global_batch": TRAIN_BATCH_SIZE,
            "ppo_mini_batch_size_per_actor_rank": PPO_MINI_BATCH_SIZE,
            "optimizer_updates_per_driver_step": 1,
            "inputs": inputs,
            "argv": command,
        },
    )
    log = (run_dir / "artifacts/trainer.log").open("ab", buffering=0)
    return command, log


def merge_arm(arm: Arm) -> int:
    actor = arm.run_dir / f"checkpoints/global_step_{TOTAL_STEPS}/actor"
    target = arm.run_dir / "merged/final_hf_official_chat_v1"
    log_path = arm.run_dir / "artifacts/merge.log"
    if (target / "config.json").is_file() and any(target.glob("model*.safetensors*")):
        return 0
    if not actor.is_dir():
        return 90
    target.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(PYTHON),
        "-m",
        "verl.model_merger",
        "merge",
        "--backend",
        "fsdp",
        "--local_dir",
        str(actor),
        "--target_dir",
        str(target),
        "--use_cpu_initialization",
    ]
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "NVIDIA_VISIBLE_DEVICES": "",
            "PYTHONPATH": str(VERL_ROOT),
            "PYTHONUNBUFFERED": "1",
        }
    )
    with log_path.open("ab", buffering=0) as log:
        return subprocess.run(command, cwd=VERL_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode


def execute() -> int:
    inputs = require_inputs()
    children: dict[str, tuple[Arm, subprocess.Popen[bytes], Any]] = {}
    for arm in ARMS:
        command, log = prepare_arm(arm, inputs)
        process = subprocess.Popen(
            command,
            cwd=REFERENCE,
            env=arm_environment(arm),
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        children[arm.key] = (arm, process, log)
        write_json(
            arm.run_dir / "artifacts/status.json",
            {"state": "running", "pid": process.pid, "updated_at": time.time()},
        )

    interrupted = False
    try:
        while any(process.poll() is None for _, process, _ in children.values()):
            time.sleep(10)
    except KeyboardInterrupt:
        interrupted = True
        for _, process, _ in children.values():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for _, process, _ in children.values():
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
    finally:
        for _, _, log in children.values():
            log.close()

    result: dict[str, Any] = {"interrupted": interrupted, "arms": {}}
    failed = interrupted
    for key, (arm, process, _) in children.items():
        train_rc = process.returncode
        merge_rc = merge_arm(arm) if train_rc == 0 else None
        state = "complete" if train_rc == 0 and merge_rc == 0 else "failed"
        result["arms"][key] = {
            "state": state,
            "train_returncode": train_rc,
            "merge_returncode": merge_rc,
            "run_dir": str(arm.run_dir),
            "model": str(arm.run_dir / "merged/final_hf_official_chat_v1"),
        }
        write_json(arm.run_dir / "artifacts/status.json", result["arms"][key])
        failed = failed or state != "complete"
    write_json(OUTPUT_PARENT / "b57_vision6k_crop_dual4_v1_summary.json", result)
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    inputs = require_inputs()
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "dry_run",
                    "inputs": inputs,
                    "arms": [
                        {
                            "key": arm.key,
                            "teacher": str(arm.teacher),
                            "cuda_devices": arm.cuda_devices,
                            "run_dir": str(arm.run_dir),
                            "argv": build_command(arm),
                        }
                        for arm in ARMS
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    return execute()


if __name__ == "__main__":
    raise SystemExit(main())
