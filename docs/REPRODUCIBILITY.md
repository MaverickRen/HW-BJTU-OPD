# Reproducibility contract

## Public source revisions

| Dependency | Revision and role |
|---|---|
| Vision-OPD | `c8a8fdd1f88eef1b5ef4fe6a8d64eb0272917471`; OPD runtime uses its vendored veRL |
| veRL | public base `11c94ad2354456d9bfa93c558e05e9430cd731b2` plus `patches/verl-qwen35-opd.patch`; SFT runtime |
| VLMEvalKit | `09874c7a69c2a3c7c60ace141525c1552a2c1095`; advanced benchmark tooling |
| Vision-OPD-6K | `eb5c1c2e7b9a7b6a619efe4161c7369c71bf8af4`; 6,241 OPD rows |
| VStar | `b44023b4dca749ed8a76b85eb576627d05a1c174`; 191-row public evaluation gate |

`scripts/bootstrap.sh` checks out only these fetchable revisions. The veRL patch
is a complete diff from the public base, including the three former local
commits and the final tracked training changes. A fresh checkout must pass
`git apply --check`; the old non-public SHA is no longer part of the contract.

## Released 9B-teacher OPD arm

- Artifact ID: `qwen35-9b-vision-opd-sft9-teacher-full8-s65-v1`.
- Hugging Face repo:
  `HWBJTUOPD/Qwen3.5-9B-SFT10K-VisionOPD6K-SFT9BTeacher`.
- Frozen release revision: `6fc7d1ed7c509572898a32ff9de6cff19e8455f0`.
- Student initialization: merged Qwen3.5-9B SFT_V1 10K.
- Teacher initialization: the same merged Qwen3.5-9B SFT_V1 10K.
- Teacher: fixed for the run (`teacher_update_rate=0.0`).
- Student view: red-box full image (`images`).
- Teacher view: actual target crop (`bbox_images`).
- Data: 6,241 Vision-OPD-6K rows; decontamination excluded zero rows.
- Driver batch 96, rollout `n=8`, 65 steps, seed 42.
- LR `2e-6`, ten warmup steps, one optimizer update per driver step.
- Prompt/response/model lengths: 8,192/1,024/9,216.
- Distillation top-k 100, alpha 0.5; PPO clip 0.2--0.3.
- Hardware: one node with eight GPUs.

The upstream-compatible field `teacher_regularization=ema` remains in the
launch command, but a zero teacher update rate means the released teacher is
not EMA-updated. `reward_model.enable=False`; OPD does not optimize against the
dataset answer labels.

## Artifact identity

The seven required model files and their sizes/SHA256 values are frozen in
`configs/opd9_teacher_artifact.json`. The principal weight identity is:

```text
model.safetensors
bytes: 18819722392
sha256: c86054edddaf186b5a0754fed55e4d8e80108ba2081ff7e6ba7c2d3e589ccdc7
```

`scripts/prepare_opd_release.py --verify-only` rehashes every required model,
processor, tokenizer, configuration, and chat-template file. Evaluation fails
before model startup if any file differs.

## Public evaluation gate

`scripts/evaluate_vstar.sh --execute` is the supported result reproduction:

1. download the released model;
2. verify all seven artifact files;
3. download VStar at its pinned dataset revision;
4. validate the source Parquet, materialized JSON, logical rows, exact image
   file set, and every image content hash;
5. start a loopback-only vLLM service;
6. run the historical seed-42, non-thinking, temperature-0 VStar contract with
   a 32,768-token maximum output inside a 65,536-token server context;
7. apply the frozen first-option scorer and persist aggregate counts only.

The historical topology was TP8 and produced `176/191`. The public wrapper can
use one sufficiently large GPU to reduce cost and labels `173..179/191` as a
similar reproduction. An 8-row quick run validates plumbing only.

The successful environment intentionally combines vLLM 0.18 with Transformers
5.5 for Qwen3.5 support. Because vLLM's wheel metadata still declares
`transformers<5`, `scripts/install_eval.sh` installs vLLM and its dependencies
first, then applies and verifies the exact Transformers/tokenizers override.
The same installer fixes `quack-kernels==0.5.0` with
`nvidia-cutlass-dsl==4.5.3`: CUTLASS DSL 4.6 and later removed the `ThrMma`
compatibility symbol used by that QuACK release. Installation fails closed if
the package versions, paired DSL library, or import boundary differs. Runtime
compilation/IPC caches use a short node-local directory to avoid network
filesystem locking and Unix socket path limits.

The complete four-benchmark chain is retained in `repro/eval_tools` for audit,
but it is not the low-cost public gate. It additionally needs official MMStar,
BLINK, and ZoomBench assets, eight candidate GPUs, and a separate 27B
ZoomBench judge. Raw predictions and benchmark gold payloads are not released.

## SFT training and export

The released SFT checkpoints can be used directly. To regenerate them, use
`scripts/train_sft.py` with `configs/sft_v1_10k_9b.json` or
`configs/sft_v1_10k_27b.json`, the patched standalone veRL checkout, and the
portable SFT data. Both the original research Parquet SHA and the rewritten
portable Parquet SHA are accepted and recorded in the launch receipt.

For example, the 9B arm is:

```bash
python scripts/train_sft.py \
  --config configs/sft_v1_10k_9b.json \
  --model /path/to/Qwen3.5-9B \
  --data artifacts/hw-bjtu-opd/datasets/sft-v1-10k/train_10000.parquet \
  --output outputs/qwen35-9b-sft-v1-10k \
  --verl-root third_party/verl \
  --python "$PWD/.venv/bin/python" \
  --execute
```

Use `scripts/export_sft_checkpoint.sh` to recover and merge the terminal LoRA
checkpoint. OPD export instead uses `scripts/export_opd_checkpoint.sh` with
`VISION_OPD_ROOT` set, so merge code comes from the same vendored runtime that
trained the actor.
