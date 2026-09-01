# HW-BJTU-OPD

Reproducible Qwen3.5 multimodal SFT and crop-based online perception
distillation (OPD), with public checkpoints, pinned data, and a simple
aggregate-only evaluation entry.

[![verify](https://github.com/MaverickRen/HW-BJTU-OPD/actions/workflows/verify.yml/badge.svg)](https://github.com/MaverickRen/HW-BJTU-OPD/actions/workflows/verify.yml)

## Verified release status

Public access and end-to-end reproduction were re-verified on 2026-09-01:

- The final 9B SFT-teacher Vision-OPD checkpoint is public and ungated at
  revision `6fc7d1ed7c509572898a32ff9de6cff19e8455f0`.
- The 9B/27B SFT initialization checkpoints and portable SFT_V1 10K data are
  public and ungated at revision `83362b995e3d3bf6789268e655cb286b925af215`.
- A clean TP4 run on four NVIDIA L20C GPUs reproduced the released VStar score
  exactly: **176/191 (92.1466%)**, with zero invalid outputs. The sanitized
  receipt is
  [`results/vstar_reproduction_validation.json`](results/vstar_reproduction_validation.json).
- The GitHub Actions CPU contracts, tests, lint and shell checks pass on the
  public `main` branch.

## Public release

The newly released model is the requested **second-highest macro** experiment:

- Student: Qwen3.5-9B after SFT_V1 10K.
- Teacher: the same Qwen3.5-9B SFT_V1 10K checkpoint, fixed throughout OPD.
- Data: 6,241 rows from pinned Vision-OPD-6K.
- Views: red-box full image for the student; actual target crop for the teacher.
- Run: eight GPUs, batch 96, rollout `n=8`, 65 steps, seed 42.

Published artifacts:

- **Final OPD model:**
  [`HWBJTUOPD/Qwen3.5-9B-SFT10K-VisionOPD6K-SFT9BTeacher`](https://huggingface.co/HWBJTUOPD/Qwen3.5-9B-SFT10K-VisionOPD6K-SFT9BTeacher)
- **SFT checkpoints and portable SFT_V1 10K data:**
  [`HWBJTUOPD/HW-BJTU-OPD`](https://huggingface.co/HWBJTUOPD/HW-BJTU-OPD)
- **Source:**
  [`MaverickRen/HW-BJTU-OPD`](https://github.com/MaverickRen/HW-BJTU-OPD)

The final OPD model is a closed seven-file Hugging Face artifact. Its
`model.safetensors` is 18,819,722,392 bytes with SHA256
`c86054edddaf186b5a0754fed55e4d8e80108ba2081ff7e6ba7c2d3e589ccdc7`.

## Fastest result reproduction

This is the supported, low-cost public gate. It downloads the exact model and
the pinned 191-row VStar snapshot, verifies all model/data hashes, runs a
loopback-only vLLM service, and persists aggregate counts only.

```bash
git clone https://github.com/MaverickRen/HW-BJTU-OPD.git
cd HW-BJTU-OPD

python3.12 -m venv .venv
source .venv/bin/activate
scripts/install_eval.sh

scripts/evaluate_vstar.sh --execute
```

Use the installer rather than invoking `pip install -r requirements-eval.txt`
directly. vLLM 0.18 declares `transformers<5`, while its audited Qwen3.5
runtime here uses Transformers 5.5; the script resolves vLLM first and applies
that tested override explicitly. It also pins the QuACK/CUTLASS DSL pair whose
unbounded upstream upgrade otherwise fails during vLLM worker initialization.

The default uses one CUDA GPU; at least 48 GB of GPU memory is recommended.
The model download is about 18.84 GB. Allow at least 45 GB of free disk during
the first install and download; the installer does not retain the large wheel
cache. For the historical tensor-parallel topology, use:

```bash
scripts/evaluate_vstar.sh \
  --gpus 0,1,2,3,4,5,6,7 \
  --tp-size 8 \
  --execute
```

Expected VStar result: **176/191 = 92.1466%**. The script marks results within
three correct answers as similar because changing GPU topology or kernels can
change a few greedy outputs. `--quick 8` is available to check plumbing, but
it is not a result reproduction. Omitting `--execute` prints a no-write plan.

A fresh install on 2026-08-31 reproduced **176/191** with TP4 on four NVIDIA
L20C GPUs. The aggregate-only, path-free environment and result receipt is
[`results/vstar_reproduction_validation.json`](results/vstar_reproduction_validation.json).

The exact historical run used TP8 and the versions in
[`requirements-eval.txt`](requirements-eval.txt). This environment has been
separated from the larger training environment to keep installation small.

## Results

All values below are percentages. Exact counts and protocol IDs are in
[`results/core_results.json`](results/core_results.json); the released row also
has a sanitized, aggregate-only receipt in
[`results/released_opd9_teacher.json`](results/released_opd9_teacher.json).

| Model / experiment | VStar | MMStar | BLINK-v5 | ZoomBench | Macro |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-9B raw | 84.82 | 78.93 | 59.13 | 51.01 | 68.47 |
| Qwen3.5-27B raw | 86.39 | 79.33 | 50.97 | 57.63 | 68.58 |
| Qwen3.5-9B SFT_V1 10K | 85.86 | 77.47 | 64.97 | 52.90 | 70.30 |
| Qwen3.5-27B SFT_V1 10K | 86.39 | 79.87 | 62.65 | 58.70 | 71.90 |
| 9B SFT + Vision6K crop + raw 9B teacher | 92.67 | 76.20 | 64.18 | 61.07 | 73.53 |
| **9B SFT + Vision6K crop + 9B SFT teacher (released)** | **92.15** | **77.27** | **66.44** | **62.13** | **74.50** |
| 9B SFT + Vision6K crop + 27B SFT teacher | 93.72 | 76.20 | 66.02 | 63.55 | 74.87 |

The released row is second by macro and has the strongest MMStar and BLINK-v5
cells among the three OPD arms. `BLINK-v5` is a frozen local checkpoint
comparison protocol, not an official BLINK leaderboard result. See
[`docs/RESULTS.md`](docs/RESULTS.md) before comparing BLINK variants.

## Repository layout

```text
apps/                    Streamlit inspection tools for real OPD samples
configs/                 exact SFT, OPD, and artifact contracts
scripts/                 portable preparation/train/export/evaluation commands
src/hw_bjtu_opd/         relative-path data adapter and lightweight evaluator
repro/                   source-frozen historical launchers and evaluators
patches/                 complete public-base veRL patch and CPU contract tests
results/                 aggregate results and artifact manifests
docs/                    data, result, and reproducibility details
hf_card/                 Hugging Face release cards
```

Large checkpoints, benchmark payloads, caches, raw predictions, and
credentials are excluded from Git.

## Training environment

Training is substantially more expensive than evaluation and uses the fuller
environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-repro.txt

PYTHON_BIN="$PWD/.venv/bin/python" scripts/bootstrap.sh
```

`scripts/bootstrap.sh` uses only public, fetchable source revisions:

- veRL base `11c94ad2354456d9bfa93c558e05e9430cd731b2`, plus the complete
  [`patches/verl-qwen35-opd.patch`](patches/verl-qwen35-opd.patch);
- Vision-OPD `c8a8fdd1f88eef1b5ef4fe6a8d64eb0272917471`;
- VLMEvalKit `09874c7a69c2a3c7c60ace141525c1552a2c1095`.

The standalone patched veRL checkout is the SFT runtime. OPD intentionally
uses the pinned Vision-OPD checkout's vendored veRL, matching the successful
run. This distinction fixes the former README pin/runtime mismatch.

## Download the SFT initialization and data

```bash
hf download HWBJTUOPD/HW-BJTU-OPD --local-dir artifacts/hw-bjtu-opd
```

```text
artifacts/hw-bjtu-opd/
├── checkpoints/qwen35-9b-sft-v1-10k/
├── checkpoints/qwen35-27b-sft-v1-10k/
└── datasets/sft-v1-10k/
    ├── train_10000.parquet
    ├── manifest.json
    └── media/...
```

The portable SFT Parquet SHA256 is
`bcc980bf809f6905fa9aab978e59ee8884c45258f6e69c289bf63547aa2dc859`.
Its hash differs from the research copy because machine-local image paths were
rewritten to relative paths; both accepted identities are recorded by the SFT
launchers.

## Prepare Vision-OPD-6K

The project does not duplicate roughly 37 GB of upstream Vision-OPD media.
Instead, the preparation script downloads the exact public revision and
verifies its source files before extraction:

```bash
export PYTHONPATH="$PWD/repro/data_tools"

python repro/data_tools/prepare_vision_opd_6k.py \
  --download \
  --raw-root data/vision-opd-6k/raw \
  --media-root data/vision-opd-6k/media \
  --output-dir data/vision-opd-6k/processed
```

The released decontamination audit excluded zero rows, so the generated
6,241-row `processed/train.parquet` has the same training rows as the
historical `train_decontaminated.parquet`. Its byte hash is expected to differ
across machines because the Parquet records resolved media roots. The pinned
source revision, source checksums, row count, views, and row order are the
portable identity. See [`docs/DATA.md`](docs/DATA.md) for the optional full
denylist audit.

## Inspect real Vision-OPD samples

The Streamlit viewer reads either the pinned raw JSONL or the prepared Parquet
without copying image payloads. It displays the original image, the student's
red-box full image, the teacher's target crop, the exact question, bbox and
audit-only gold answer side by side:

```bash
python -m pip install -e '.[viewer]'

streamlit run apps/opd_data_viewer.py -- \
  --data data/vision-opd-6k/raw/train.jsonl \
  --media-root data/vision-opd-6k/media
```

The same viewer accepts `data/vision-opd-6k/processed/train.parquet`. The
`--media-root` override makes copied Parquet snapshots portable by remapping
their recorded `relative_*` media fields. Images are loaded lazily, so browsing
does not place the roughly 37 GB compressed source archive in memory.

## Reproduce the released OPD training arm

Launchers are dry-run by default and require `--execute` to write or train.
From-scratch OPD reproduction requires eight GPUs and both initial 9B SFT
roles point to the same released checkpoint:

```bash
python scripts/train_opd.py \
  --config configs/opd_vision6k_crop_sft9_teacher.json \
  --student artifacts/hw-bjtu-opd/checkpoints/qwen35-9b-sft-v1-10k \
  --teacher artifacts/hw-bjtu-opd/checkpoints/qwen35-9b-sft-v1-10k \
  --data data/vision-opd-6k/processed/train.parquet \
  --output outputs/sft9-vision6k-crop-sft9-teacher \
  --vision-opd-root third_party/Vision-OPD \
  --python "$PWD/.venv/bin/python" \
  --execute
```

The OPD stage does not train against dataset answers: the reward model is
disabled and the objective distils teacher token distributions over model
rollouts. The teacher receives `bbox_images`; the student receives `images`.
`teacher_update_rate=0` keeps the teacher fixed.

Export the final actor with the same vendored runtime:

```bash
PYTHON_BIN="$PWD/.venv/bin/python" \
VISION_OPD_ROOT="$PWD/third_party/Vision-OPD" \
  scripts/export_opd_checkpoint.sh \
  --checkpoint outputs/sft9-vision6k-crop-sft9-teacher/checkpoints/global_step_65/actor \
  --output outputs/sft9-vision6k-crop-sft9-teacher-hf
```

SFT training and merge commands are documented in
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md); most users can use the
released merged initialization directly.

## Full four-benchmark evaluation

The simple public path deliberately targets VStar. The frozen full chain under
`repro/eval_tools/` is retained for audit and includes VStar, MMStar, BLINK-v5,
and ZoomBench, but it is an advanced historical workflow: it requires all
official benchmark assets, eight candidate GPUs, and a separate Qwen3.5-27B
ZoomBench semantic judge pinned at
`fc05daec18b0a78c049392ed2e771dde82bdf654`. It should not be mistaken for a
low-cost one-command test. Raw predictions and benchmark gold data are not
redistributed; exact aggregate, dataset, evaluator, and judge receipts are in
[`results/released_opd9_teacher.json`](results/released_opd9_teacher.json).

## Reproducibility and licenses

- [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) defines source, runtime,
  training, and evaluation contracts.
- [`docs/DATA.md`](docs/DATA.md) records data identities and path portability.
- [`docs/RESULTS.md`](docs/RESULTS.md) records result authorities and protocol
  distinctions.
- [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) records upstream licenses.

The code and released model are Apache-2.0, subject to upstream Qwen terms.
Dataset and benchmark assets retain their own licenses. Never commit access
tokens, `.env` files, benchmark gold data, raw predictions, or model weights
to this Git repository.
