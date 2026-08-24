#!/usr/bin/env bash

set -euo pipefail

OPD_WORKSPACE=/minimax-3d-rw-backup/users/jiazhi/H_Workspace
SERVE_ENV="${SERVE_ENV:-$OPD_WORKSPACE/UV_Env/verl-opd-qwen35}"
VLLM_BIN="$SERVE_ENV/bin/vllm"
MODEL_PATH=${1:-$OPD_WORKSPACE/Ckpt/Qwen3.5-9B}
if [[ $# -gt 0 ]]; then
  shift
fi

if [[ ! -x "$VLLM_BIN" ]]; then
  echo "vLLM executable is missing: $VLLM_BIN" >&2
  exit 1
fi
if [[ ! -f "$MODEL_PATH/config.json" ]]; then
  echo "Model or merged checkpoint is invalid: $MODEL_PATH" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLLM_USE_V1=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HOME="$OPD_WORKSPACE/Cache/huggingface_opd"
export XDG_CACHE_HOME="$OPD_WORKSPACE/Cache/xdg_opd"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-$OPD_WORKSPACE/Cache/flashinfer_opd}"
export CUDA_CACHE_PATH="$OPD_WORKSPACE/Cache/cuda_opd"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$OPD_WORKSPACE/Cache/torchinductor_opd}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$OPD_WORKSPACE/Cache/triton_opd}"
export TMPDIR="$OPD_WORKSPACE/Output/tmp/vllm-eval"
EVAL_MM_LIMITS_VALUE=${EVAL_MM_LIMITS:-}
if [[ -z "$EVAL_MM_LIMITS_VALUE" ]]; then
  EVAL_MM_LIMITS_VALUE='{"image":16}'
fi
if [[ -z "${LOGNAME:-}${USER:-}${LNAME:-}${USERNAME:-}" ]]; then
  export LOGNAME="opd_uid_$(id -u)"
fi
mkdir -p \
  "$HF_HOME" "$XDG_CACHE_HOME" "$FLASHINFER_WORKSPACE_BASE" "$CUDA_CACHE_PATH" \
  "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$TMPDIR"

COMMAND=(
  "$VLLM_BIN" serve "$MODEL_PATH"
  --served-model-name "${EVAL_MODEL_ID:-Qwen3.5-9B}"
  --host "${EVAL_API_HOST:-127.0.0.1}"
  --port "${EVAL_API_PORT:-8000}"
  --tensor-parallel-size "${EVAL_TP_SIZE:-1}"
  --data-parallel-size "${EVAL_DP_SIZE:-1}"
  --gpu-memory-utilization "${EVAL_GPU_MEMORY_UTILIZATION:-0.85}"
  --max-model-len "${EVAL_MAX_MODEL_LEN:-32768}"
  --max-num-seqs "${EVAL_MAX_NUM_SEQS:-32}"
  --limit-mm-per-prompt "$EVAL_MM_LIMITS_VALUE"
  --compilation-config '{"pass_config":{"fuse_allreduce_rms":false}}'
  --kernel-config '{"enable_flashinfer_autotune":false}'
  --trust-remote-code
  "$@"
)

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf 'vLLM serve command:'
  printf ' %q' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi

exec "${COMMAND[@]}"
