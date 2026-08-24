#!/usr/bin/env python3
"""Launch the exact one-epoch SFT_V1 10K recipe on one existing 8-GPU node."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from _common import (
    ReleaseError,
    load_config,
    printable,
    require_dir,
    require_file,
    run,
    sha256_file,
    training_env,
    write_json,
)


SCHEMA = "hw_bjtu_opd_sft_v1"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ADAPTER = REPO_ROOT / "src/hw_bjtu_opd/data/sft_dataset.py"


def build_command(
    config: dict,
    *,
    python: Path,
    model: Path,
    data: Path,
    output: Path,
    adapter: Path,
) -> list[str]:
    world_size = int(config["world_size"])
    if world_size != 8:
        raise ReleaseError("the released recipe is fixed to the user-authorized 8 GPUs")
    rows = int(config["dataset_rows"])
    batch = int(config["global_batch_size"])
    steps = int(config["optimizer_steps"])
    if batch * steps != rows or int(config["epochs"]) != 1:
        raise ReleaseError("SFT config no longer represents exactly one epoch")
    command = [
        str(python),
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc-per-node={world_size}",
        "-m",
        "verl.trainer.sft_trainer",
        f"data.train_files={data}",
        "data.val_files=null",
        f"data.train_max_samples={rows}",
        f"data.train_batch_size={batch}",
        f"data.micro_batch_size_per_gpu={int(config['micro_batch_size_per_gpu'])}",
        "data.messages_key=prompt",
        "+data.image_key=images",
        f"data.custom_cls.path=file://{adapter}",
        "data.custom_cls.name=B54TenKSFTDataset",
        "data.enable_thinking_default=False",
        "data.ignore_input_ids_mismatch=True",
        f"data.max_length={int(config['max_length'])}",
        "data.truncation=error",
        "data.pad_mode=no_padding",
        f"data.use_dynamic_bsz={bool(config['dynamic_batch'])}",
        f"data.max_token_len_per_gpu={int(config['max_length'])}",
        "data.num_workers=8",
        f"+data.shuffle={bool(config['shuffle'])}",
        f"model.path={model}",
        "model.enable_gradient_checkpointing=True",
        "model.use_remove_padding=True",
        f"model.lora_rank={int(config['lora_rank'])}",
        f"model.lora_alpha={int(config['lora_alpha'])}",
        f"model.target_modules={config['lora_target_modules']}",
        "engine=fsdp",
        "engine.strategy=fsdp",
        f"engine.fsdp_size={world_size}",
        f"engine.seed={int(config['seed'])}",
        "engine.use_torch_compile=False",
        f"engine.dtype={config['dtype']}",
        f"optim.lr={config['learning_rate']}",
        "optim.lr_warmup_steps_ratio=0.0",
        "optim.lr_scheduler_type=cosine",
        "optim.clip_grad=1.0",
        f"trainer.seed={int(config['seed'])}",
        f"trainer.n_gpus_per_node={world_size}",
        f"trainer.total_training_steps={steps}",
        "trainer.total_epochs=1",
        "trainer.save_freq=-1",
        f"+trainer.save_steps=[{steps}]",
        "trainer.test_freq=-1",
        "trainer.resume_mode=disable",
        "trainer.logger=['console']",
        f"trainer.default_local_dir={output / 'checkpoints'}",
        "trainer.project_name=HW_BJTU_OPD_SFT_V1",
        f"trainer.experiment_name={config['name']}",
        "checkpoint.save_contents=['model','extra']",
        "checkpoint.load_contents=['model','extra']",
        "+checkpoint.strict=True",
        f"hydra.run.dir={output}/hydra_rank_${{oc.env:LOCAL_RANK,0}}",
        "hydra.job.chdir=False",
    ]
    if bool(config["length_bucket"]):
        command.extend(
            [
                "+data.length_bucket_batch=True",
                f"+data.length_bucket_seed={int(config['seed'])}",
                f"+data.length_bucket_image_token_proxy={int(config['length_bucket_image_token_proxy'])}",
            ]
        )
    return command


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", required=True, type=Path)
    result.add_argument("--model", required=True, type=Path)
    result.add_argument("--data", required=True, type=Path)
    result.add_argument("--output", required=True, type=Path)
    result.add_argument("--verl-root", required=True, type=Path)
    result.add_argument("--python", required=True, type=Path)
    result.add_argument("--dataset-adapter", type=Path, default=DEFAULT_ADAPTER)
    result.add_argument("--execute", action="store_true", help="otherwise only emit a validated plan")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        config = load_config(args.config, SCHEMA)
        require_file(args.python, "Python entrypoint")
        require_dir(args.model, "base model")
        require_file(args.model / "config.json", "base model config")
        require_file(args.data, "SFT parquet")
        require_dir(args.verl_root, "veRL checkout")
        require_file(args.dataset_adapter, "SFT dataset adapter")
        observed = sha256_file(args.data)
        if observed != config["dataset_sha256"]:
            raise ReleaseError(f"dataset SHA256 differs: {observed}")
        if args.output.exists() or args.output.is_symlink():
            raise ReleaseError(f"create-once output already exists: {args.output}")
        command = build_command(
            config,
            python=args.python.absolute(),
            model=args.model.absolute(),
            data=args.data.absolute(),
            output=args.output.absolute(),
            adapter=args.dataset_adapter.absolute(),
        )
        plan = {
            "schema_version": "hw_bjtu_opd_sft_launch_v1",
            "status": "validated",
            "config": config,
            "dataset_sha256": observed,
            "world_size": 8,
            "additional_resources_requested": False,
            "command": command,
            "shell_command": printable(command),
        }
        if not args.execute:
            print(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        args.output.mkdir(parents=True)
        write_json(args.output / "launch_plan.json", plan)
        run(
            command,
            cwd=args.verl_root,
            env=training_env(args.verl_root),
            log=args.output / "train.log",
        )
        write_json(args.output / "completion.json", {**plan, "status": "complete"})
        print(json.dumps({"status": "complete", "output": str(args.output)}, sort_keys=True))
        return 0
    except (ReleaseError, OSError, ValueError, KeyError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
