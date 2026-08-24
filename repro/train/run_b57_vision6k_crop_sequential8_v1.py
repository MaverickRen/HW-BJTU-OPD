#!/usr/bin/env python3
"""Run the two B57-initialized Vision-OPD crop arms on all eight GPUs.

Each selected arm is trained and merged before the process exits.  Selecting
``all`` runs the raw-9B-teacher arm first and the B57-SFT-27B-teacher arm
second.  The command is the proven legacy Vision-OPD recipe at world size 8;
only student/teacher identities and output namespaces differ from the control.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import run_b57_vision6k_crop_dual4_v1 as base


WORLD_SIZE = 8
PPO_MINI_BATCH_SIZE = 96
SAVE_FREQ = 8

ARMS = (
    base.Arm(
        key="raw9_teacher",
        teacher=base.RAW9_TEACHER,
        cuda_devices="0,1,2,3,4,5,6,7",
        run_name="b57_10k_init_vision6k_crop_raw9_teacher_full8_s65_v1",
        ray_root=Path("/tmp/b57-v6k-r9-full8-ray"),
    ),
    base.Arm(
        key="b57_27b_teacher",
        teacher=base.B57_27B_TEACHER,
        cuda_devices="0,1,2,3,4,5,6,7",
        run_name="b57_10k_init_vision6k_crop_b57_27b_teacher_full8_s65_v1",
        ray_root=Path("/tmp/b57-v6k-r27-full8-ray"),
    ),
)


def command_for(arm: base.Arm) -> list[str]:
    command = base.build_command(arm)
    replacements = {
        f"actor_rollout_ref.actor.ppo_mini_batch_size={base.PPO_MINI_BATCH_SIZE}":
            f"actor_rollout_ref.actor.ppo_mini_batch_size={PPO_MINI_BATCH_SIZE}",
        "actor_rollout_ref.rollout.agent.num_workers=4":
            f"actor_rollout_ref.rollout.agent.num_workers={WORLD_SIZE}",
        "trainer.n_gpus_per_node=4": f"trainer.n_gpus_per_node={WORLD_SIZE}",
        f"trainer.save_freq={base.SAVE_FREQ}": f"trainer.save_freq={SAVE_FREQ}",
        "+ray_kwargs.ray_init.num_gpus=4":
            f"+ray_kwargs.ray_init.num_gpus={WORLD_SIZE}",
    }
    observed: set[str] = set()
    result: list[str] = []
    for item in command:
        if item in replacements:
            observed.add(item)
            item = replacements[item]
        result.append(item)
    missing = set(replacements) - observed
    if missing:
        raise RuntimeError(f"full8 command adaptation keys missing: {sorted(missing)}")
    return result


def prepare(arm: base.Arm, inputs: dict[str, Any]) -> tuple[list[str], Any]:
    for child in ("checkpoints", "rollouts", "hydra", "artifacts", "cache"):
        (arm.run_dir / child).mkdir(parents=True, exist_ok=True)
    cache = arm.run_dir / "cache"
    for child in (
        "home", "xdg", "config", "hf/datasets", "hf/transformers", "vllm",
        "cuda", "flashinfer", "torchinductor", "triton",
    ):
        (cache / child).mkdir(parents=True, exist_ok=True)
    short_tmp = Path("/tmp") / _short_tmp_name(arm)
    short_tmp.mkdir(parents=True, exist_ok=True)
    command = command_for(arm)
    base.write_json(
        arm.run_dir / "artifacts/launch_plan.json",
        {
            "schema_version": "b57_vision6k_crop_sequential8_v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "arm": arm.key,
            "student": str(base.STUDENT),
            "teacher": str(arm.teacher),
            "teacher_mode": "fixed",
            "student_image_key": "images",
            "teacher_image_key": "bbox_images",
            "cuda_devices": arm.cuda_devices,
            "world_size": WORLD_SIZE,
            "steps": base.TOTAL_STEPS,
            "global_batch": base.TRAIN_BATCH_SIZE,
            "rollout_n": 8,
            "ppo_mini_batch_size_per_actor_rank": PPO_MINI_BATCH_SIZE,
            "optimizer_updates_per_driver_step": 1,
            "inputs": inputs,
            "argv": command,
        },
    )
    log = (arm.run_dir / "artifacts/trainer.log").open("ab", buffering=0)
    return command, log


def environment_for(arm: base.Arm) -> dict[str, str]:
    env = base.arm_environment(arm)
    env["TMPDIR"] = str(Path("/tmp") / _short_tmp_name(arm))
    return env


def _short_tmp_name(arm: base.Arm) -> str:
    """Return a short, identity-preserving Unix-socket namespace."""

    names = {
        "raw9_teacher": "vopd8-r9",
        "b57_27b_teacher": "vopd8-r27",
        "b57_10k_sft_9b_teacher": "vopd8-b579",
    }
    try:
        return names[arm.key]
    except KeyError as exc:
        raise RuntimeError(f"unknown full8 arm key: {arm.key}") from exc


def run_arm(arm: base.Arm, inputs: dict[str, Any]) -> dict[str, Any]:
    target = arm.run_dir / "merged/final_hf_official_chat_v1"
    if (target / "config.json").is_file() and any(target.glob("model*.safetensors*")):
        record = {
            "state": "complete", "train_returncode": 0, "merge_returncode": 0,
            "run_dir": str(arm.run_dir), "model": str(target), "action": "reuse_complete",
        }
        base.write_json(arm.run_dir / "artifacts/status.json", record)
        return record
    command, log = prepare(arm, inputs)
    process = subprocess.Popen(
        command,
        cwd=base.REFERENCE,
        env=environment_for(arm),
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    base.write_json(
        arm.run_dir / "artifacts/status.json",
        {"state": "running", "pid": process.pid, "updated_at": time.time()},
    )
    try:
        train_rc = process.wait()
    except KeyboardInterrupt:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            train_rc = process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            train_rc = process.wait()
    finally:
        log.close()
    merge_rc = base.merge_arm(arm) if train_rc == 0 else None
    state = "complete" if train_rc == 0 and merge_rc == 0 else "failed"
    record = {
        "state": state,
        "train_returncode": train_rc,
        "merge_returncode": merge_rc,
        "run_dir": str(arm.run_dir),
        "model": str(target),
        "action": "execute",
    }
    base.write_json(arm.run_dir / "artifacts/status.json", record)
    if state != "complete":
        raise RuntimeError(f"{arm.key} failed: train={train_rc}, merge={merge_rc}")
    return record


def select(value: str) -> tuple[base.Arm, ...]:
    if value == "all":
        return ARMS
    return tuple(arm for arm in ARMS if arm.key == value)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--arm", choices=("raw9_teacher", "b57_27b_teacher", "all"), default="all"
    )
    args = parser.parse_args(argv)
    inputs = base.require_inputs()
    arms = select(args.arm)
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "dry_run",
                    "world_size": WORLD_SIZE,
                    "inputs": inputs,
                    "arms": [
                        {
                            "key": arm.key,
                            "student": str(base.STUDENT),
                            "teacher": str(arm.teacher),
                            "run_dir": str(arm.run_dir),
                            "argv": command_for(arm),
                        }
                        for arm in arms
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    summary: dict[str, Any] = {"status": "running", "arms": {}}
    try:
        for arm in arms:
            summary["arms"][arm.key] = run_arm(arm, inputs)
        summary["status"] = "complete"
        return 0
    except BaseException as exc:
        summary["status"] = "failed"
        summary["error"] = str(exc)
        raise
    finally:
        base.write_json(base.OUTPUT_PARENT / "b57_vision6k_crop_sequential8_v1_summary.json", summary)


if __name__ == "__main__":
    raise SystemExit(main())
