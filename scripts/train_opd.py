#!/usr/bin/env python3
"""Launch the released full-8-GPU Vision-OPD-6K crop distillation recipe."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

from _common import ReleaseError, load_config, printable, require_dir, require_file, run, training_env, write_json


SCHEMA = "hw_bjtu_opd_vision6k_crop_v1"


def build_command(
    cfg: dict,
    *,
    python: Path,
    student: Path,
    teacher: Path,
    data: Path,
    output: Path,
    chat_template: Path,
) -> list[str]:
    if int(cfg["world_size"]) != 8:
        raise ReleaseError("the released result is an 8-GPU run; changing world size changes the optimization geometry")
    if cfg["teacher_mode"] != "fixed" or float(cfg["teacher_update_rate"]) != 0.0:
        raise ReleaseError("released teacher must remain fixed (teacher_update_rate=0)")
    if cfg["student_image_key"] != "images" or cfg["teacher_image_key"] != "bbox_images":
        raise ReleaseError("released crop OPD view contract changed")
    steps = int(cfg["total_training_steps"])
    command = [
        str(python),
        "-m",
        "verl.trainer.main_ppo",
        "--config-name",
        "vopd",
        f'data.train_files=["{data}"]',
        "data.val_files=[]",
        f"data.train_max_samples={int(cfg['dataset_rows'])}",
        "data.filter_overlong_prompts=False",
        f"data.max_prompt_length={int(cfg['max_prompt_length'])}",
        f"data.max_response_length={int(cfg['max_response_length'])}",
        "data.truncation=error",
        "data.shuffle=True",
        f"data.seed={int(cfg['seed'])}",
        "data.trust_remote_code=True",
        "data.return_multi_modal_inputs=True",
        f"data.image_key={cfg['student_image_key']}",
        f"data.train_batch_size={int(cfg['train_batch_size'])}",
        "data.dataloader_num_workers=8",
        f"actor_rollout_ref.model.path={student}",
        "actor_rollout_ref.model.trust_remote_code=True",
        "actor_rollout_ref.model.use_remove_padding=True",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        f"actor_rollout_ref.rollout.n={int(cfg['rollout_n'])}",
        f"actor_rollout_ref.actor.optim.lr={cfg['learning_rate']}",
        f"actor_rollout_ref.actor.optim.lr_warmup_steps={int(cfg['warmup_steps'])}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={int(cfg['ppo_mini_batch_size'])}",
        "actor_rollout_ref.actor.use_dynamic_bsz=True",
        f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu={int(cfg['max_model_length'])}",
        "actor_rollout_ref.actor.fsdp_config.param_offload=True",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True",
        f"actor_rollout_ref.actor.clip_ratio_high={cfg['clip_ratio_high']}",
        f"actor_rollout_ref.actor.clip_ratio_low={cfg['clip_ratio_low']}",
        "actor_rollout_ref.actor.use_kl_loss=False",
        "actor_rollout_ref.actor.policy_loss.loss_mode=vopd",
        "actor_rollout_ref.actor.calculate_entropy=False",
        f"actor_rollout_ref.actor.self_distillation.distillation_topk={int(cfg['distillation_topk'])}",
        "actor_rollout_ref.actor.self_distillation.max_reprompt_len=10240",
        "actor_rollout_ref.actor.self_distillation.is_clip=2.0",
        "actor_rollout_ref.actor.self_distillation.teacher_always_on=True",
        "actor_rollout_ref.actor.self_distillation.teacher_model_source=fixed",
        f"actor_rollout_ref.actor.self_distillation.teacher_model_path={teacher}",
        "actor_rollout_ref.actor.self_distillation.teacher_regularization=ema",
        "actor_rollout_ref.actor.self_distillation.teacher_update_rate=0.0",
        f"actor_rollout_ref.actor.self_distillation.teacher_image_key={cfg['teacher_image_key']}",
        "actor_rollout_ref.actor.self_distillation.fallback_to_policy_loss_on_missing_teacher=False",
        "actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=True",
        f"actor_rollout_ref.actor.self_distillation.alpha={cfg['distillation_alpha']}",
        "actor_rollout_ref.actor.self_distillation.include_environment_feedback=False",
        "algorithm.rollout_correction.rollout_is=token",
        "algorithm.rollout_correction.rollout_is_threshold=2.0",
        "algorithm.adv_estimator=grpo",
        "algorithm.norm_adv_by_std_in_grpo=False",
        "algorithm.use_kl_in_reward=False",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={cfg['rollout_gpu_memory_utilization']}",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1",
        f"actor_rollout_ref.rollout.max_num_batched_tokens={int(cfg['max_model_length'])}",
        f"actor_rollout_ref.rollout.max_model_len={int(cfg['max_model_length'])}",
        "+actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.pass_config.fuse_allreduce_rms=False",
        "+actor_rollout_ref.rollout.engine_kwargs.vllm.kernel_config.enable_flashinfer_autotune=False",
        f"actor_rollout_ref.rollout.response_length={int(cfg['max_response_length'])}",
        "actor_rollout_ref.rollout.calculate_log_probs=True",
        "actor_rollout_ref.rollout.agent.num_workers=8",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.ref.fsdp_config.param_offload=True",
        "reward_model.enable=False",
        "reward_model.use_reward_loop=False",
        "custom_reward_function.path=null",
        f"critic.model.path={student}",
        "trainer.project_name=Vision-OPD",
        f"trainer.group_name={cfg['name']}",
        f"trainer.experiment_name={cfg['name']}",
        'trainer.logger=["console","tensorboard"]',
        "trainer.n_gpus_per_node=8",
        "trainer.nnodes=1",
        f"trainer.save_freq={int(cfg['save_frequency'])}",
        "trainer.test_freq=-1",
        "trainer.max_actor_ckpt_to_keep=null",
        "trainer.total_epochs=1",
        f"trainer.total_training_steps={steps}",
        "trainer.resume_mode=auto",
        "trainer.val_before_train=False",
        f"trainer.default_local_dir={output / 'checkpoints'}",
        f"trainer.rollout_data_dir={output / 'rollouts'}",
        f"hydra.run.dir={output / 'hydra'}",
        "ray_kwargs.ray_init.num_cpus=48",
        "+ray_kwargs.ray_init.num_gpus=8",
        "+ray_kwargs.ray_init.address=local",
        "+ray_kwargs.ray_init.include_dashboard=False",
        f"+ray_kwargs.ray_init._temp_dir=/tmp/hw-bjtu-opd-{os.getuid()}",
        f"actor_rollout_ref.model.custom_chat_template_file={chat_template}",
    ]
    return command


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--config", required=True, type=Path)
    value.add_argument("--student", required=True, type=Path)
    value.add_argument("--teacher", required=True, type=Path)
    value.add_argument("--data", required=True, type=Path)
    value.add_argument("--output", required=True, type=Path)
    value.add_argument("--verl-root", required=True, type=Path)
    value.add_argument("--vision-opd-root", required=True, type=Path)
    value.add_argument("--python", required=True, type=Path)
    value.add_argument("--chat-template", type=Path)
    value.add_argument("--execute", action="store_true")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        cfg = load_config(args.config, SCHEMA)
        template = args.chat_template or args.vision_opd_root / "chat_templates/perception_chat_template_qwen35.jinja"
        require_file(args.python, "Python entrypoint")
        require_dir(args.student, "student model")
        require_dir(args.teacher, "teacher model")
        require_file(args.student / "config.json", "student config")
        require_file(args.teacher / "config.json", "teacher config")
        require_file(args.data, "Vision-OPD parquet")
        require_dir(args.verl_root, "veRL checkout")
        require_dir(args.vision_opd_root, "Vision-OPD checkout")
        require_file(template, "Qwen3.5 perception chat template")
        if args.output.is_symlink():
            raise ReleaseError("output cannot be a symlink")
        command = build_command(
            cfg,
            python=args.python.absolute(),
            student=args.student.absolute(),
            teacher=args.teacher.absolute(),
            data=args.data.absolute(),
            output=args.output.absolute(),
            chat_template=template.absolute(),
        )
        plan = {
            "schema_version": "hw_bjtu_opd_crop_launch_v1",
            "status": "validated",
            "config": cfg,
            "answer_used_by_opd": False,
            "reward_model_enabled": False,
            "teacher_fixed": True,
            "world_size": 8,
            "additional_resources_requested": False,
            "command": command,
            "shell_command": printable(command),
        }
        if not args.execute:
            print(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        args.output.mkdir(parents=True, exist_ok=True)
        write_json(args.output / "launch_plan.json", plan)
        env = training_env(args.verl_root, args.vision_opd_root)
        env.update(
            {
                "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
                "NVIDIA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
                "VLLM_USE_V1": "1",
                "VLLM_NO_USAGE_STATS": "1",
            }
        )
        run(command, cwd=args.verl_root, env=env, log=args.output / "trainer.log")
        write_json(args.output / "completion.json", {**plan, "status": "complete"})
        print(json.dumps({"status": "complete", "output": str(args.output)}, sort_keys=True))
        return 0
    except (ReleaseError, OSError, ValueError, KeyError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
