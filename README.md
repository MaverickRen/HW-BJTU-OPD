# HW-BJTU-OPD

Qwen3.5 multimodal SFT, crop-based online perception distillation (OPD), checkpoint export and frozen four-benchmark evaluation.

The latest completed experiment is exactly:

- **Student initialization:** Qwen3.5-9B after one epoch of SFT_V1 10K.
- **OPD data:** decontaminated Vision-OPD-6K, 6,241 rows.
- **Student view:** full image with the target region marked by a red box.
- **Teacher view:** the actual cropped target region.
- **Teacher:** Qwen3.5-27B after the same SFT_V1 10K, fixed (`teacher_update_rate=0`).
- **Run:** 8 GPUs, batch 96, rollout `n=8`, 65 steps, seed 42.

It reaches **93.72 V***, **76.20 MMStar**, **66.02 BLINK-v5**, **63.55 ZoomBench**, and **74.87 four-benchmark macro**. This is the strongest macro result in the current controlled matrix.

## Repository contents

```text
configs/                 exact SFT and OPD configurations
scripts/                 portable data/train/export/evaluation entrypoints
src/hw_bjtu_opd/data/    relative-path-aware multimodal SFT adapter
repro/                   source-frozen launchers, mergers and evaluators
patches/                 veRL Qwen3.5/OPD patch and its CPU contract tests
results/                 machine-readable aggregate results
docs/                    data, protocol and reproducibility notes
hf_card/                 public Hugging Face release card
```

Large datasets, checkpoints, benchmark payloads, predictions, caches and credentials are deliberately excluded from Git. The companion public artifact repository is:

```text
HWBJTUOPD/HW-BJTU-OPD
```

It contains the portable SFT_V1 10K snapshot and the merged 9B/27B SFT checkpoints in subdirectories.

## Results

All values are percentages; counts and full protocol metadata are in [`results/core_results.json`](results/core_results.json).

| Model / experiment | V* | MMStar | BLINK-v5 | ZoomBench | Macro |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-9B raw | 84.82 | 78.93 | 59.13 | 51.01 | 68.47 |
| Qwen3.5-27B raw | 86.39 | 79.33 | 50.97 | 57.63 | 68.58 |
| Qwen3.5-9B SFT_V1 10K | 85.86 | 77.47 | 64.97 | 52.90 | 70.30 |
| Qwen3.5-27B SFT_V1 10K | 86.39 | 79.87 | 62.65 | 58.70 | 71.90 |
| 9B SFT + Vision6K Crop + raw 9B teacher | 92.67 | 76.20 | 64.18 | 61.07 | 73.53 |
| 9B SFT + Vision6K Crop + 9B SFT teacher | 92.15 | **77.27** | **66.44** | 62.13 | 74.50 |
| 9B SFT + Vision6K Crop + 27B SFT teacher | **93.72** | 76.20 | 66.02 | **63.55** | **74.87** |

Vision-OPD reference checkpoints are reported separately because the available local record uses another BLINK protocol:

- Official Vision-OPD-9B `6e41541`: V* `175/191 = 91.62%`; the other three formal cells are missing.
- Locally trained Vision-OPD-9B B1 crop step65: V* 89.01, MMStar 78.33, **BLINK-exact** 40.77, ZoomBench 60.24.

`BLINK-v5` and `BLINK-exact/v14` are not interchangeable. See [`docs/RESULTS.md`](docs/RESULTS.md) before comparing rows.

## 1. Environment

The successful environment used Python 3.12 and the exact package versions in [`requirements-repro.txt`](requirements-repro.txt), including PyTorch 2.10.0, Transformers 5.5.0, vLLM 0.18.0, FlashAttention 2.8.3.post1 and Ray 2.53.0. Use a CUDA driver/toolkit compatible with the selected PyTorch wheel.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-repro.txt

# Fetch exact upstream revisions, apply the local veRL patch and install it.
PYTHON_BIN="$PWD/.venv/bin/python" scripts/bootstrap.sh
```

If FlashAttention must compile locally, install it after PyTorch with `--no-build-isolation`. `scripts/bootstrap.sh` pins:

- veRL `c282ca53a025f00687f53b55b0eb890bf92a9840` plus the included patch;
- Vision-OPD `c8a8fdd1f88eef1b5ef4fe6a8d64eb0272917471`;
- VLMEvalKit `09874c7a69c2a3c7c60ace141525c1552a2c1095`.

The patch is part of the training implementation, not an optional performance tweak. It adds the fixed privileged teacher view, Qwen3.5 distillation control flow, FSDP export fixes and CPU contract tests used by these runs.

## 2. Download released artifacts

```bash
hf download HWBJTUOPD/HW-BJTU-OPD --local-dir artifacts/hw-bjtu-opd
```

Important paths after download:

```text
artifacts/hw-bjtu-opd/
├── checkpoints/qwen35-9b-sft-v1-10k/
├── checkpoints/qwen35-27b-sft-v1-10k/
└── datasets/sft-v1-10k/
    ├── train_10000.parquet
    ├── manifest.json
    └── media/...
```

The portable Parquet has a new hash because its absolute media paths were rewritten. `manifest.json` binds it to the exact original research Parquet SHA256 `9f56d58c…a8c49b0`. See [`docs/DATA.md`](docs/DATA.md).

## 3. Train SFT_V1 10K

Download the raw Qwen3.5 base model first. The launcher is dry-run by default and only starts training with `--execute`.

### Qwen3.5-9B

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

This exposes all 10,000 rows exactly once: global batch 80, 125 optimizer steps, LoRA rank/alpha 16, LR `2e-5`, max length 16,384, dynamic batches and deterministic length bucketing.

### Qwen3.5-27B

Use the same command with:

```text
--config configs/sft_v1_10k_27b.json
--model /path/to/Qwen3.5-27B
--output outputs/qwen35-27b-sft-v1-10k
```

The 27B recipe uses LR `1e-5` and max length 9,216; all other controlled settings remain aligned.

### Export and merge SFT

For the terminal 9B checkpoint, for example:

```bash
scripts/export_sft_checkpoint.sh \
  --python "$PWD/.venv/bin/python" \
  --verl-root third_party/verl \
  --checkpoint outputs/qwen35-9b-sft-v1-10k/checkpoints/global_step_125 \
  --base-model /path/to/Qwen3.5-9B \
  --output outputs/qwen35-9b-sft-v1-10k-merged
```

The exporter first uses `verl.model_merger` to recover the LoRA adapter from FSDP and then uses the included Qwen3.5 sharded merger. The latter verifies the exact tensor-key universe and rejects residual LoRA tensors.

## 4. Prepare Vision-OPD-6K

The pinned source is `yuanqianhao/Vision-OPD-6K`. Build and decontaminate it against your local official benchmark denylist:

```bash
export PYTHONPATH="$PWD/repro/data_tools"

python repro/data_tools/prepare_vision_opd_6k.py \
  --download \
  --raw-root data/vision-opd-6k/raw \
  --media-root data/vision-opd-6k/media \
  --output-dir data/vision-opd-6k/processed

python repro/data_tools/validate_vision_opd_6k.py \
  --raw-root data/vision-opd-6k/raw \
  --media-root data/vision-opd-6k/media \
  --output-dir data/vision-opd-6k/processed \
  --manifest-dir data/vision-opd-6k/manifest \
  --denylist-dir /path/to/four-benchmark-denylists
```

The OPD launcher expects `data/vision-opd-6k/processed/train_decontaminated.parquet` with exactly 6,241 rows.

## 5. Train crop OPD

The released final configuration uses the two merged SFT checkpoints as student and teacher. It uses **only the existing eight GPUs**.

```bash
python scripts/train_opd.py \
  --config configs/opd_vision6k_crop_sft27_teacher.json \
  --student artifacts/hw-bjtu-opd/checkpoints/qwen35-9b-sft-v1-10k \
  --teacher artifacts/hw-bjtu-opd/checkpoints/qwen35-27b-sft-v1-10k \
  --data data/vision-opd-6k/processed/train_decontaminated.parquet \
  --output outputs/sft9-vision6k-crop-sft27-teacher \
  --verl-root third_party/verl \
  --vision-opd-root third_party/Vision-OPD \
  --python "$PWD/.venv/bin/python" \
  --execute
```

To reproduce the 9B-teacher comparison, switch to `configs/opd_vision6k_crop_sft9_teacher.json` and point `--teacher` to the 9B SFT checkpoint.

The OPD stage does **not** train on dataset answers. `reward_model.enable=False`, no reward function is configured, and the objective is teacher-token distillation over model rollouts. Crop OPD means the student and teacher receive different image fields; the teacher truly receives the cropped region, not merely the red-box image.

Export the terminal OPD actor:

```bash
PYTHON_BIN="$PWD/.venv/bin/python" VERL_ROOT="$PWD/third_party/verl" \
  scripts/export_opd_checkpoint.sh \
  --checkpoint outputs/sft9-vision6k-crop-sft27-teacher/checkpoints/global_step_65/actor \
  --output outputs/sft9-vision6k-crop-sft27-teacher-hf
```

## 6. Evaluate V*, MMStar, BLINK and ZoomBench

The exact aggregate-only protocol code is under `repro/eval_tools`. Benchmark payloads are not redistributed. Place official VStar/MMStar/BLINK/ZoomBench assets under your workspace as described by the upstream projects, and configure the frozen ZoomBench judge.

```bash
export OPD_QWEN35_WORKSPACE=/path/to/H_Workspace
export TRAIN_PYTHON="$PWD/.venv/bin/python"
export OPD_GPU_LOCK="$OPD_QWEN35_WORKSPACE/Locks/opd_gpu_0_7.lock"

# Omit --execute for a validated dry run.
scripts/evaluate_fourbench.sh \
  --model-path outputs/sft9-vision6k-crop-sft27-teacher-hf \
  --model-id Qwen3.5-9B-SFT10K-Vision6K-Crop-SFT27Teacher \
  --run-root "$OPD_QWEN35_WORKSPACE/Output/release_eval" \
  --execute
```

Evaluation is serial on the same eight GPUs: one resident model service for V*/MMStar/ZoomBench, then the frozen BLINK-v5 service. Expected totals are 191, 1,500, 1,901 and 845. Any invalid BLINK output counts as wrong. The raw prediction files remain private; the final `summary.json` is aggregate-only.

## Reproducibility and licensing

- [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) records the exact source, view, optimizer and evaluation contracts.
- [`docs/RESULTS.md`](docs/RESULTS.md) explains result authorities and the two BLINK protocols.
- [`docs/DATA.md`](docs/DATA.md) records data composition, decontamination and path sanitization.
- [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) records upstream projects and dataset licenses.

Never commit access tokens, `.env` files, raw predictions, benchmark gold data or model weights to this Git repository.
