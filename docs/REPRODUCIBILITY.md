# Reproducibility contract

## Pinned source revisions

| Dependency | Revision |
|---|---|
| Vision-OPD | `c8a8fdd1f88eef1b5ef4fe6a8d64eb0272917471` |
| veRL | `c282ca53a025f00687f53b55b0eb890bf92a9840` plus `patches/verl-qwen35-opd.patch` |
| VLMEvalKit | `09874c7a69c2a3c7c60ace141525c1552a2c1095` |

`scripts/bootstrap.sh` checks out these exact revisions and applies the local Qwen3.5/privileged-teacher/FSDP fixes. The `repro/` directory preserves the exact launchers, mergers and frozen evaluator code that produced the recorded artifacts. The clean `scripts/` entrypoints remove machine-specific paths while retaining the same hyperparameters.

## Latest experiment contract

- Student initialization: merged Qwen3.5-9B SFT_V1 10K.
- Teacher: merged Qwen3.5-27B SFT_V1 10K, fixed for the entire run.
- OPD data: `yuanqianhao/Vision-OPD-6K`, pinned and decontaminated to 6,241 rows.
- Student view: full image with target region marked by a red box (`images`).
- Teacher view: actual target-region crop (`bbox_images`).
- Driver batch: 96; rollout `n=8`; 65 steps; seed 42.
- Optimizer: LR `2e-6`, ten warmup steps, one optimizer update per driver step.
- Distillation: top-k 100, alpha 0.5, fixed teacher update rate 0.
- Hardware: exactly the user-authorized 8 GPUs; no additional resources.

Although the configuration field is named `teacher_regularization=ema` for upstream compatibility, `teacher_update_rate=0.0` means the released teacher is not EMA-updated.

## Why the four-GPU and eight-GPU recipes are separate

The four-GPU prototype changed the local PPO mini-batch geometry to preserve one optimizer update per step. It is not guaranteed to be numerically identical to the full-eight run. Reported final results use the eight-GPU recipe in `configs/opd_vision6k_crop_sft27_teacher.json`.

## Evaluation policy

Every checkpoint decision must include V*, MMStar, BLINK and ZoomBench. The scripts keep raw predictions private and publish aggregate counts only. A cell is complete only when its expected total is exact and its frozen protocol gate passes.
