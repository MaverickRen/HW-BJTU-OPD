#!/usr/bin/env bash
# Export a terminal veRL LoRA-FSDP checkpoint and merge it into its HF base.
set -euo pipefail

usage() {
  echo "Usage: $0 --python PY --verl-root DIR --checkpoint DIR --base-model DIR --output DIR" >&2
  exit 2
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin='' verl_root='' checkpoint='' base_model='' output=''
while [[ $# -gt 0 ]]; do
  case "$1" in
    --python) python_bin="${2:-}"; shift 2 ;;
    --verl-root) verl_root="${2:-}"; shift 2 ;;
    --checkpoint) checkpoint="${2:-}"; shift 2 ;;
    --base-model) base_model="${2:-}"; shift 2 ;;
    --output) output="${2:-}"; shift 2 ;;
    *) usage ;;
  esac
done
[[ -n "$python_bin" && -n "$verl_root" && -n "$checkpoint" && -n "$base_model" && -n "$output" ]] || usage
[[ -x "$python_bin" && -d "$verl_root" && -d "$checkpoint" && -d "$base_model" ]] || usage
[[ ! -e "$output" && ! -L "$output" ]] || { echo "Output already exists: $output" >&2; exit 2; }

staging="${output}.fsdp-export-incomplete"
[[ ! -e "$staging" && ! -L "$staging" ]] || { echo "Staging already exists: $staging" >&2; exit 2; }

PYTHONPATH="$verl_root${PYTHONPATH:+:$PYTHONPATH}" \
  "$python_bin" -B -m verl.model_merger merge \
  --backend fsdp --local_dir "$checkpoint" --target_dir "$staging"

"$python_bin" -B "$repo_root/repro/train/merge_qwen35_sharded_lora_to_hf_v1.py" \
  --execute --base-model "$base_model" --adapter "$staging/lora_adapter" \
  --output "$output" --checkpoint-meta "$checkpoint/lora_train_meta.json"

printf 'Merged SFT checkpoint: %s\n' "$output"
