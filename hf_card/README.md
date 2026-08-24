---
license: apache-2.0
library_name: transformers
pipeline_tag: image-text-to-text
tags:
- qwen3.5
- vision-language
- supervised-fine-tuning
- online-perception-distillation
- multi-image
- fine-grained-vision
base_model:
- Qwen/Qwen3.5-9B
- Qwen/Qwen3.5-27B
---

# HW-BJTU-OPD release

This public artifact repository contains:

```text
checkpoints/
├── qwen35-9b-sft-v1-10k/
└── qwen35-27b-sft-v1-10k/
datasets/
└── sft-v1-10k/
```

Both checkpoints are merged Hugging Face models produced by one epoch of LoRA SFT on the same 10,000-example multimodal dataset. The repository also contains the portable dataset snapshot and all model-input media. Code, exact configurations and the four-benchmark protocol are in the companion `HW-BJTU-OPD` source repository.

## Download

```bash
hf download HWBJTUOPD/HW-BJTU-OPD --local-dir hw-bjtu-opd
```

Load one variant by pointing Transformers or vLLM at its subdirectory:

```python
from transformers import AutoModelForImageTextToText, AutoProcessor

path = "hw-bjtu-opd/checkpoints/qwen35-9b-sft-v1-10k"
processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(
    path, torch_dtype="auto", device_map="auto", trust_remote_code=True
)
```

## SFT recipe

| Variant | Rows | Epochs | Global batch | Steps | LR | Max length | LoRA rank/alpha |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.5-9B SFT_V1 | 10,000 | 1 | 80 | 125 | 2e-5 | 16,384 | 16/16 |
| Qwen3.5-27B SFT_V1 | 10,000 | 1 | 80 | 125 | 1e-5 | 9,216 | 16/16 |

Both use seed 42, bf16, FSDP over eight GPUs, all-linear LoRA, complete-answer loss masking, dynamic batches and deterministic length bucketing.

## Four-benchmark results

The BLINK column is the local deterministic checkpoint-comparison v5 protocol, not an official leaderboard claim.

| Checkpoint | V* | MMStar | BLINK-v5 | ZoomBench | Macro |
|---|---:|---:|---:|---:|---:|
| Raw Qwen3.5-9B | 84.82 | 78.93 | 59.13 | 51.01 | 68.47 |
| Qwen3.5-9B SFT_V1 10K | 85.86 | 77.47 | 64.97 | 52.90 | 70.30 |
| Raw Qwen3.5-27B | 86.39 | 79.33 | 50.97 | 57.63 | 68.58 |
| Qwen3.5-27B SFT_V1 10K | 86.39 | 79.87 | 62.65 | 58.70 | 71.90 |

These SFT models are also the student/teacher initializations for the best current OPD experiment: 9B SFT student + Vision-OPD-6K crop + fixed 27B SFT teacher, which scores 93.72/76.20/66.02/63.55 and a 74.87 macro under the same four protocols. That OPD checkpoint is not part of this artifact release.

## Dataset

SFT_V1 10K contains 3,800 fine-grained single-image examples, 2,600 general visual knowledge/reasoning examples and 3,600 multi-image reasoning examples. It excludes B28 and public Vision-OPD rows and passed a zero-hard-overlap gate against VStarBench, MMStar, BLINK and ZoomBench.

The public Parquet uses relative content-addressed media paths. Its `manifest.json` records the portable hash and binds it to the original research Parquet SHA256. Answers are present because this is SFT data. OPD itself does not use answer labels.

## Licenses and limitations

The model/code release is Apache-2.0 subject to the accompanying Qwen terms. Dataset rows retain source-specific licensing and provenance. The relevant source licenses include Apache-2.0, CC-BY-4.0 and CC-BY-SA-4.0; those licenses are not replaced by the repository-level license.

The data is research-oriented and may contain source-dataset biases or annotation errors. The four-benchmark results are checkpoint-comparison measurements under frozen local protocols; they should not be presented as official leaderboard submissions.
