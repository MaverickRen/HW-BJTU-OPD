#!/usr/bin/env bash
# Run the frozen V*/MMStar/ZoomBench resident chain, then frozen BLINK-v5.
set -euo pipefail

usage() {
  echo "Usage: $0 --model-path DIR --model-id ID --run-root DIR [--execute]" >&2
  exit 2
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${OPD_QWEN35_WORKSPACE:?set OPD_QWEN35_WORKSPACE to the H_Workspace root}"
: "${TRAIN_PYTHON:?set TRAIN_PYTHON to the pinned environment Python}"

model_path='' model_id='' run_root='' mode=--dry-run
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-path) model_path="${2:-}"; shift 2 ;;
    --model-id) model_id="${2:-}"; shift 2 ;;
    --run-root) run_root="${2:-}"; shift 2 ;;
    --execute) mode=--execute; shift ;;
    *) usage ;;
  esac
done
[[ -n "$model_path" && -n "$model_id" && -n "$run_root" ]] || usage

OPD_QWEN35_WORKSPACE="$OPD_QWEN35_WORKSPACE" \
  "$TRAIN_PYTHON" -B "$repo_root/repro/eval_tools/run_threebench_resident_formal_v1.py" \
  "$mode" --model-path "$model_path" --model-id "$model_id" \
  --run-root "$run_root/threebench" --audit-root "$run_root/threebench_audit" \
  --candidate-port 18618 --judge-port 18619

OPD_QWEN35_WORKSPACE="$OPD_QWEN35_WORKSPACE" \
TRAIN_PYTHON="$TRAIN_PYTHON" \
  "$repo_root/scripts/run_blink_v5_portable.sh" \
  --model-path "$model_path" --model-id "$model_id" \
  --run-root "$run_root/blink-v5" --port 18620 "$mode"

if [[ "$mode" == --execute ]]; then
  "$TRAIN_PYTHON" -B "$repo_root/scripts/summarize_fourbench.py" \
    --threebench "$run_root/threebench/threebench_resident_result.json" \
    --blink "$run_root/blink-v5/blink_deterministic_checkpoint_comparison_v5.json" \
    --output "$run_root/summary.json"
fi
