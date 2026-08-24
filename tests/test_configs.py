from __future__ import annotations

import json
from pathlib import Path

import train_opd
import train_sft


ROOT = Path(__file__).resolve().parents[1]


def config(name: str) -> dict:
    return json.loads((ROOT / "configs" / name).read_text(encoding="utf-8"))


def test_sft_configs_are_one_epoch_and_eight_gpu() -> None:
    for name in ("sft_v1_10k_9b.json", "sft_v1_10k_27b.json"):
        value = config(name)
        assert value["world_size"] == 8
        assert value["epochs"] == 1
        assert value["global_batch_size"] * value["optimizer_steps"] == value["dataset_rows"] == 10_000
        command = train_sft.build_command(
            value,
            python=Path("/python"),
            model=Path("/model"),
            data=Path("/data.parquet"),
            output=Path("/output"),
            adapter=Path("/adapter.py"),
        )
        joined = "\n".join(command)
        assert "--nproc-per-node=8" in command
        assert "data.train_max_samples=10000" in command
        assert "model.lora_rank=16" in command
        assert "trainer.total_training_steps=125" in command
        assert "data.length_bucket_batch=True" in joined


def test_opd_configs_preserve_fixed_privileged_teacher_contract() -> None:
    for name in ("opd_vision6k_crop_sft9_teacher.json", "opd_vision6k_crop_sft27_teacher.json"):
        value = config(name)
        command = train_opd.build_command(
            value,
            python=Path("/python"),
            student=Path("/student"),
            teacher=Path("/teacher"),
            data=Path("/vision6k.parquet"),
            output=Path("/output"),
            chat_template=Path("/chat.jinja"),
        )
        assert value["world_size"] == 8
        assert value["student_image_key"] == "images"
        assert value["teacher_image_key"] == "bbox_images"
        assert value["teacher_update_rate"] == 0
        assert "actor_rollout_ref.actor.self_distillation.teacher_update_rate=0.0" in command
        assert "actor_rollout_ref.actor.self_distillation.teacher_image_key=bbox_images" in command
        assert "reward_model.enable=False" in command
        assert "actor_rollout_ref.rollout.agent.num_workers=8" in command
        assert "trainer.total_training_steps=65" in command
