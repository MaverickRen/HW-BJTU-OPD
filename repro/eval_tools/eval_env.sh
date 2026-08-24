#!/usr/bin/env bash

set -euo pipefail

export OPD_WORKSPACE=/minimax-3d-rw-backup/users/jiazhi/H_Workspace
export VLMEVAL_ROOT="$OPD_WORKSPACE/Codes/VLMEvalKit"
export VLMEVAL_ENV="$OPD_WORKSPACE/UV_Env/vlmevalkit-opd"
export VLMEVAL_PYTHON="$VLMEVAL_ENV/bin/python"
export LMUData="$OPD_WORKSPACE/Dataset/eval"
export HF_HOME="$OPD_WORKSPACE/Dataset/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export XDG_CACHE_HOME="$OPD_WORKSPACE/Dataset/.cache/vlmevalkit_opd"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-$OPD_WORKSPACE/Cache/flashinfer_opd}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$OPD_WORKSPACE/Cache/torchinductor_opd}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$OPD_WORKSPACE/Cache/triton_opd}"
export TMPDIR="$OPD_WORKSPACE/Output/tmp/vlmevalkit"
export PYTHONPATH="$VLMEVAL_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export VLMEVAL_LOCAL_MEDIA=0

# PyTorch 2.x still calls getpass.getuser() for a global inductor header cache
# even when TORCHINDUCTOR_CACHE_DIR is explicit.  Batch UIDs may have no passwd
# entry, so provide the least invasive fallback that getpass understands.
if [[ -z "${LOGNAME:-}${USER:-}${LNAME:-}${USERNAME:-}" ]]; then
  export LOGNAME="opd_uid_$(id -u)"
fi

export EVAL_API_BASE="${EVAL_API_BASE:-http://127.0.0.1:8000/v1}"
export EVAL_MODEL_ID="${EVAL_MODEL_ID:-Qwen3.5-9B}"
export EVAL_API_KEY="${EVAL_API_KEY:-sk-local-opd}"
export EVAL_WORK_DIR="${EVAL_WORK_DIR:-$OPD_WORKSPACE/Output/opd_qwen35_9b/b0_base9b_20260806/eval}"

case ",${NO_PROXY:-}," in
  *,127.0.0.1,localhost,*) ;;
  *) export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost" ;;
esac

mkdir -p \
  "$LMUData" \
  "$HF_HUB_CACHE" \
  "$HF_DATASETS_CACHE" \
  "$XDG_CACHE_HOME" \
  "$FLASHINFER_WORKSPACE_BASE" \
  "$TORCHINDUCTOR_CACHE_DIR" \
  "$TRITON_CACHE_DIR" \
  "$TMPDIR" \
  "$EVAL_WORK_DIR"
