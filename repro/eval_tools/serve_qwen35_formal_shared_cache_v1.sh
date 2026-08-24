#!/usr/bin/env bash
# vLLM server entrypoint for the formal aggregate runner.  Cache directories
# are supplied by the runner and are private, ownership-gated, and keyed by
# model/runtime/backend identity.  This file never writes evaluation answers.
set -euo pipefail

OPD_WORKSPACE=/minimax-3d-rw-backup/users/jiazhi/H_Workspace
SERVE_ENV="${SERVE_ENV:-$OPD_WORKSPACE/UV_Env/verl-opd-qwen35}"
VLLM_BIN="$SERVE_ENV/bin/vllm"
MODEL_PATH="${1:-}"
[[ -n "$MODEL_PATH" ]] || { echo "model path is required" >&2; exit 2; }
shift || true
[[ -x "$VLLM_BIN" ]] || { echo "vLLM executable is unavailable" >&2; exit 1; }
[[ -f "$MODEL_PATH/config.json" ]] || { echo "model checkpoint is invalid" >&2; exit 1; }
[[ -n "${OPD_FORMAL_CACHE_ROOT:-}" ]] || { echo "formal cache root is required" >&2; exit 2; }

cache_root="$(realpath -m -- "$OPD_FORMAL_CACHE_ROOT")"
case "$cache_root" in
  "$OPD_WORKSPACE/Cache/vllm_formal_shared/"*) ;;
  *) echo "formal cache root is outside the fixed workspace cache" >&2; exit 2 ;;
esac
mkdir -p -- "$cache_root"
chmod 0700 -- "$cache_root"
owner_uid="$(stat -c '%u' -- "$cache_root")"
owner_mode="$(stat -c '%a' -- "$cache_root")"
[[ "$owner_uid" == "$(id -u)" && "$owner_mode" == 700 ]] || {
  echo "formal cache ownership/mode gate failed" >&2; exit 2;
}

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export VLLM_USE_V1=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
compile_cache_root="${OPD_FORMAL_COMPILE_CACHE_ROOT:-$OPD_WORKSPACE/Cache/vllm_formal_shared/mcc3a793f5ed8cb5098e120ed-rebdbf1149fa89d154c1c7304-flashinfer}"
compile_cache_root="$(realpath -m -- "$compile_cache_root")"
case "$compile_cache_root" in
  "$OPD_WORKSPACE/Cache/vllm_formal_shared/"*) ;;
  *) echo "formal compile cache root is outside the fixed workspace cache" >&2; exit 2 ;;
esac
mkdir -p -- "$compile_cache_root"
chmod 0700 -- "$compile_cache_root"
[[ "$(stat -c '%u' -- "$compile_cache_root")" == "$(id -u)" ]] || {
  echo "formal compile cache ownership gate failed" >&2; exit 2;
}
export HF_HOME="$cache_root/huggingface"
export XDG_CACHE_HOME="$compile_cache_root/xdg"
export CUDA_CACHE_PATH="$compile_cache_root/cuda"
export TORCHINDUCTOR_CACHE_DIR="$compile_cache_root/torchinductor"
export TRITON_CACHE_DIR="$compile_cache_root/triton"
cache_tmp_key="$(printf '%s' "$cache_root" | sha256sum | cut -c1-16)"
# vLLM uses TMPDIR for Unix-domain sockets.  Keep this direct path short
# enough for Linux sockaddr_un.sun_path (107 bytes including child names),
# while preserving a deterministic, UID-private directory per cache root.
export TMPDIR="/tmp/opd-vllm-$cache_tmp_key"
mkdir -p -- "$HF_HOME" "$XDG_CACHE_HOME" "$CUDA_CACHE_PATH" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$TMPDIR"
for path in "$HF_HOME" "$XDG_CACHE_HOME" "$CUDA_CACHE_PATH" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$TMPDIR"; do
  chmod 0700 -- "$path"
  [[ "$(stat -c '%u' -- "$path")" == "$(id -u)" ]] || { echo "formal cache child ownership gate failed" >&2; exit 2; }
done

EVAL_MM_LIMITS_VALUE=${EVAL_MM_LIMITS:-}
if [[ -z "$EVAL_MM_LIMITS_VALUE" ]]; then
  EVAL_MM_LIMITS_VALUE='{"image":16}'
fi
COMMAND=(
  "$VLLM_BIN" serve "$MODEL_PATH"
  --served-model-name "${EVAL_MODEL_ID:-Qwen3.5-9B}"
  --host "${EVAL_API_HOST:-127.0.0.1}"
  --port "${EVAL_API_PORT:-8000}"
  --tensor-parallel-size "${EVAL_TP_SIZE:-8}"
  --data-parallel-size "${EVAL_DP_SIZE:-1}"
  --gpu-memory-utilization "${EVAL_GPU_MEMORY_UTILIZATION:-0.85}"
  --max-model-len "${EVAL_MAX_MODEL_LEN:-262144}"
  --max-num-seqs "${EVAL_MAX_NUM_SEQS:-32}"
  --limit-mm-per-prompt "$EVAL_MM_LIMITS_VALUE"
  --compilation-config '{"pass_config":{"fuse_allreduce_rms":false}}'
  --kernel-config '{"enable_flashinfer_autotune":false}'
  --trust-remote-code
  "$@"
)
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf 'formal shared-cache vLLM command:'; printf ' %q' "${COMMAND[@]}"; printf '\n'
  exit 0
fi
exec "${COMMAND[@]}"
