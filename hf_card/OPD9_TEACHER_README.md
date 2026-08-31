---
license: apache-2.0
library_name: transformers
pipeline_tag: image-text-to-text
base_model: Qwen/Qwen3.5-9B
tags:
- qwen3.5
- vision-language
- online-perception-distillation
- multi-image
---

# Qwen3.5-9B SFT10K + Vision-OPD6K (9B SFT teacher)

This is the final OPD student from the **second-highest four-benchmark macro**
arm in the controlled comparison. A Qwen3.5-9B student and a fixed
Qwen3.5-9B teacher were both initialized from the same 10K SFT checkpoint;
the released model is the student after 65 Vision-OPD update steps. It is not
the pre-OPD SFT checkpoint.

Source, exact configuration, artifact hashes, and the public evaluator are in
[MaverickRen/HW-BJTU-OPD](https://github.com/MaverickRen/HW-BJTU-OPD).

## Reproduce the public VStar cell

The repository provides a one-command test that downloads this model and the
exact pinned 191-row VStar snapshot, verifies every model/data hash, starts a
loopback-only vLLM server, evaluates, and writes aggregate counts only.

```bash
git clone https://github.com/MaverickRen/HW-BJTU-OPD.git
cd HW-BJTU-OPD
python3.12 -m venv .venv
source .venv/bin/activate
scripts/install_eval.sh
scripts/evaluate_vstar.sh --execute
```

`install_eval.sh` handles the audited vLLM 0.18 / Transformers 5.5 metadata
override used by the successful Qwen3.5 runtime.

The low-cost topology is one CUDA GPU with at least 48 GB memory. For the
historical TP8 topology, use:

```bash
scripts/evaluate_vstar.sh --gpus 0,1,2,3,4,5,6,7 --tp-size 8 --execute
```

The published reference is `176/191 = 92.1466%`. The script reports a
reproduction as similar when it is within three correct answers; small
topology/kernel differences can change a few greedy outputs. A quick plumbing
check is available with `--quick 8`, but it is not a result reproduction.

## Verified four-benchmark result

| VStar | MMStar | BLINK-v5 | ZoomBench | Macro |
|---:|---:|---:|---:|---:|
| 176/191 (92.15%) | 1159/1500 (77.27%) | 1263/1901 (66.44%) | 525/845 (62.13%) | **74.4955%** |

BLINK-v5 is the repository's deterministic checkpoint-comparison protocol,
not an official leaderboard submission. Full four-benchmark evaluation is an
advanced, high-resource path because ZoomBench uses a separate frozen 27B
semantic judge (`Qwen/Qwen3.5-27B` at
`fc05daec18b0a78c049392ed2e771dde82bdf654`). VStar is the supported low-cost
public reproduction gate.

## Training contract

- Data: `yuanqianhao/Vision-OPD-6K` at
  `eb5c1c2e7b9a7b6a619efe4161c7369c71bf8af4`, 6,241 rows.
- Runtime: `Vision-OPD` at
  `c8a8fdd1f88eef1b5ef4fe6a8d64eb0272917471` and its vendored veRL.
- Student view: full image with the target region marked by a red box
  (`images`).
- Teacher view: the actual cropped target region (`bbox_images`).
- Fixed 9B SFT teacher: `teacher_update_rate=0`.
- Eight GPUs, seed 42, rollout `n=8`, train batch 96, 65 steps.
- LR `2e-6`, 10 warmup steps; lengths 8,192/1,024/9,216.
- Top-k distillation 100, alpha 0.5; PPO clip range 0.2--0.3.

`artifact_manifest.json` records the exact seven-file artifact. The released
`model.safetensors` is 18,819,722,392 bytes with SHA256
`c86054edddaf186b5a0754fed55e4d8e80108ba2081ff7e6ba7c2d3e589ccdc7`.

## License

Apache-2.0 for this release, subject to upstream Qwen terms and the licenses of
the referenced data and benchmark projects.
