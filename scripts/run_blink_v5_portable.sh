#!/usr/bin/env bash
# Reproduce the deterministic BLINK protocol where raw Qwen3.5-9B scored
# 1124/1901 = 59.1268%.  This is a checkpoint-comparison protocol, not a
# leaderboard claim.
set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE="${OPD_QWEN35_WORKSPACE:?set OPD_QWEN35_WORKSPACE}"
PYTHON="${TRAIN_PYTHON:-$WORKSPACE/UV_Env/verl-opd-qwen35/bin/python}"
VLLM="${VLLM_BIN:-$(dirname "$PYTHON")/vllm}"
SCORER="$REPO_ROOT/repro/eval_tools/mcq_blink_checkpoint_comparison_aggregate_v5.py"
AUDITOR="$REPO_ROOT/repro/eval_tools/mcq_blink_format_audit_v1.py"
DATA="$WORKSPACE/Dataset/eval/BLINK.tsv"
LOCK="${OPD_GPU_LOCK:-$WORKSPACE/Locks/opd_gpu_0_7.lock}"
MODEL=''
MODEL_ID=''
RUN_ROOT=''
PORT=18085
TP_SIZE=8
DP_SIZE=1
CHAT_TEMPLATE_FILE=''
MODE=''
AUDIT=0

die() { printf 'BLINK checkpoint-v5 ERROR: %s\n' "$*" >&2; exit 2; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-path) MODEL="${2:-}"; shift 2 ;;
    --model-id) MODEL_ID="${2:-}"; shift 2 ;;
    --run-root) RUN_ROOT="${2:-}"; shift 2 ;;
    --port) PORT="${2:-}"; shift 2 ;;
    --tp-size) TP_SIZE="${2:-}"; shift 2 ;;
    --dp-size) DP_SIZE="${2:-}"; shift 2 ;;
    --chat-template-file) CHAT_TEMPLATE_FILE="${2:-}"; shift 2 ;;
    --format-audit) AUDIT=1; shift ;;
    --dry-run) MODE=dry-run; shift ;;
    --execute) MODE=execute; shift ;;
    *) die "unknown option: $1" ;;
  esac
done
[[ -n "$MODEL" && -n "$MODEL_ID" && -n "$RUN_ROOT" && -n "$MODE" ]] || die 'model-path/model-id/run-root/mode are required'
[[ "$MODEL_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._/+:-]{0,127}$ ]] || die 'unsafe model id'
[[ "$PORT" =~ ^[0-9]+$ && "$PORT" -ge 1024 && "$PORT" -le 65535 ]] || die 'invalid port'
[[ "$TP_SIZE" =~ ^[1-9][0-9]*$ && "$DP_SIZE" =~ ^[1-9][0-9]*$ ]] || die 'invalid TP/DP size'
(( TP_SIZE * DP_SIZE == 8 )) || die 'TP_SIZE * DP_SIZE must equal the existing 8-GPU allocation'
MODEL="$(realpath -m -- "$MODEL")"
RUN_ROOT="$(realpath -ms -- "$RUN_ROOT")"
if [[ -n "$CHAT_TEMPLATE_FILE" ]]; then CHAT_TEMPLATE_FILE="$(realpath -m -- "$CHAT_TEMPLATE_FILE")"; fi
case "$MODEL/" in "$WORKSPACE/"*) ;; *) die 'model is outside workspace' ;; esac
case "$RUN_ROOT/" in "$WORKSPACE/Output/"*) ;; *) die 'run root is outside Output' ;; esac
[[ -d "$MODEL" && ! -L "$MODEL" && -f "$MODEL/config.json" ]] || die 'model directory is unavailable'
if [[ -n "$CHAT_TEMPLATE_FILE" ]]; then
  case "$CHAT_TEMPLATE_FILE" in "$WORKSPACE"/*) ;; *) die 'chat template is outside workspace' ;; esac
  [[ -f "$CHAT_TEMPLATE_FILE" && ! -L "$CHAT_TEMPLATE_FILE" ]] || die 'chat template file is unavailable'
fi
[[ -f "$SCORER" && -f "$AUDITOR" && -f "$DATA" && -f "$LOCK" ]] || die 'fixed protocol input is unavailable'
[[ ! -e "$RUN_ROOT" && ! -L "$RUN_ROOT" ]] || die 'run root already exists'

if [[ "$MODE" == dry-run ]]; then
  if (( AUDIT == 1 )); then
    "$PYTHON" -B "$AUDITOR" --blink-tsv "$DATA" \
      --sidecar "$RUN_ROOT/private_responses.ndjson" --aggregate "$RUN_ROOT/blink_format_audit_v1.json" \
      --model-id "$MODEL_ID" --api-base "http://127.0.0.1:$PORT/v1" --thinking false --temperature 0 \
      --top-p 1 --top-k -1 --min-p 0 --presence-penalty 0 --repetition-penalty 1 \
      --max-tokens 32768 --seed 42 --workers 32 --dry-run
  else
    "$PYTHON" -B "$SCORER" --blink-tsv "$DATA" --output "$RUN_ROOT/blink_deterministic_checkpoint_comparison_v5.json" --model-id "$MODEL_ID" --api-base "http://127.0.0.1:$PORT/v1" --thinking false --temperature 0 --top-p 1 --top-k -1 --min-p 0 --presence-penalty 0 --repetition-penalty 1 --max-tokens 32768 --seed 42 --workers 32 --dry-run
  fi
  exit 0
fi

exec 8<"$LOCK"
/usr/bin/flock 8
mkdir -- "$RUN_ROOT" "$RUN_ROOT/lifecycle"
chmod 700 "$RUN_ROOT" "$RUN_ROOT/lifecycle"
STATUS="$RUN_ROOT/lifecycle/status.json"
SERVE_LOG="$RUN_ROOT/lifecycle/serve.log"
AGG_LOG="$RUN_ROOT/lifecycle/aggregate.log"
OUTPUT="$RUN_ROOT/blink_deterministic_checkpoint_comparison_v5.json"
SIDECAR=''
PROTOCOL='blink_deterministic_checkpoint_comparison_v5'
if (( AUDIT == 1 )); then
  OUTPUT="$RUN_ROOT/blink_format_audit_v1.json"
  SIDECAR="$RUN_ROOT/private_responses.ndjson"
  PROTOCOL='blink_format_audit_v1_over_frozen_checkpoint_v5'
fi
SERVER_PID=''
complete=0

# Execute may be entered from an administrative shell whose HOME/config/cache
# roots may not be writable by the execution uid. Bind every cache explicitly so
# vLLM's architecture-inspection subprocess cannot fall back to /root.
RUNTIME_ROOT="$RUN_ROOT/lifecycle/runtime"
PRIVATE_HOME="$RUNTIME_ROOT/home"
XDG_CONFIG_ROOT="$RUNTIME_ROOT/xdg_config"
MPLCONFIG_ROOT="$XDG_CONFIG_ROOT/matplotlib"
# Keep the vLLM ZeroMQ IPC path well below Linux's 107-byte sun_path limit.
# The global GPU lock makes this stable per-port directory single-writer.
EXECUTION_UID="$(id -u)"
TMP_ROOT="/tmp/opd-blink-v5-${EXECUTION_UID}-$PORT"
SHARED_CACHE_ROOT="$WORKSPACE/Cache/requested_sft_matrix_v1/blink_v5_shared"
HF_CACHE_ROOT="$SHARED_CACHE_ROOT/huggingface"
XDG_CACHE_ROOT="$SHARED_CACHE_ROOT/xdg"
VLLM_CACHE_ROOT_FIXED="$SHARED_CACHE_ROOT/vllm"
CUDA_CACHE_ROOT="$SHARED_CACHE_ROOT/cuda"
FLASHINFER_ROOT="$SHARED_CACHE_ROOT/flashinfer"
TORCHINDUCTOR_ROOT="$SHARED_CACHE_ROOT/torchinductor"
TRITON_ROOT="$SHARED_CACHE_ROOT/triton"
TORCH_EXTENSIONS_ROOT="$SHARED_CACHE_ROOT/torch_extensions"
mkdir -p -- \
  "$PRIVATE_HOME" "$XDG_CONFIG_ROOT" "$MPLCONFIG_ROOT" "$TMP_ROOT" \
  "$HF_CACHE_ROOT" "$XDG_CACHE_ROOT" "$VLLM_CACHE_ROOT_FIXED" \
  "$CUDA_CACHE_ROOT" "$FLASHINFER_ROOT" "$TORCHINDUCTOR_ROOT" \
  "$TRITON_ROOT" "$TORCH_EXTENSIONS_ROOT"
chmod 700 -- \
  "$RUNTIME_ROOT" "$PRIVATE_HOME" "$XDG_CONFIG_ROOT" "$MPLCONFIG_ROOT" "$TMP_ROOT" \
  "$SHARED_CACHE_ROOT" "$HF_CACHE_ROOT" "$XDG_CACHE_ROOT" "$VLLM_CACHE_ROOT_FIXED" \
  "$CUDA_CACHE_ROOT" "$FLASHINFER_ROOT" "$TORCHINDUCTOR_ROOT" \
  "$TRITON_ROOT" "$TORCH_EXTENSIONS_ROOT"

write_status() {
  local state="$1" detail="$2"
  "$PYTHON" -B - "$STATUS" "$state" "$detail" "$MODEL" "$MODEL_ID" "$PROTOCOL" <<'PY'
import json, os, pathlib, sys
path, state, detail, model, model_id, protocol = sys.argv[1:]
tmp = pathlib.Path(path + '.tmp')
tp_size, dp_size = map(int, os.environ.get('BLINK_PARALLELISM', '8,1').split(','))
template_path = os.environ.get('BLINK_CHAT_TEMPLATE_FILE') or None
template_sha256 = None
if template_path is not None:
    import hashlib
    template_sha256 = hashlib.sha256(pathlib.Path(template_path).read_bytes()).hexdigest()
tmp.write_text(json.dumps({'schema_version':'blink_checkpoint_comparison_v5_status','state':state,'detail':detail,'model_path':model,'model_id':model_id,'protocol':protocol,'raw_qwen35_9b_reference':{'correct':1124,'total':1901,'accuracy':1124/1901},'gpu_count':8,'tensor_parallel_size':tp_size,'data_parallel_size':dp_size,'chat_template_file':template_path,'chat_template_sha256':template_sha256,'additional_resources_requested':False},sort_keys=True,separators=(',',':'))+'\n')
os.chmod(tmp,0o600); os.replace(tmp,path)
PY
}
cleanup() {
  local rc=$?
  set +e
  trap - EXIT INT TERM HUP
  if [[ -n "$SERVER_PID" ]] && kill -0 -- "-$SERVER_PID" 2>/dev/null; then kill -TERM -- "-$SERVER_PID" 2>/dev/null; fi
  [[ -z "$SERVER_PID" ]] || wait "$SERVER_PID" 2>/dev/null || true
  (( complete == 1 )) || write_status failed "checkpoint-v5 failed with status $rc"
  /usr/bin/flock -u 8 2>/dev/null || true
  exec 8>&-
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

export BLINK_PARALLELISM="$TP_SIZE,$DP_SIZE"
export BLINK_CHAT_TEMPLATE_FILE="$CHAT_TEMPLATE_FILE"
CHAT_TEMPLATE_ARGS=()
if [[ -n "$CHAT_TEMPLATE_FILE" ]]; then CHAT_TEMPLATE_ARGS=(--chat-template "$CHAT_TEMPLATE_FILE"); fi
write_status starting "launching exact 8-GPU checkpoint-comparison service (TP${TP_SIZE}xDP${DP_SIZE})"
/usr/bin/setsid env \
  HOME="$PRIVATE_HOME" XDG_CONFIG_HOME="$XDG_CONFIG_ROOT" MPLCONFIGDIR="$MPLCONFIG_ROOT" TMPDIR="$TMP_ROOT" \
  HF_HOME="$HF_CACHE_ROOT" XDG_CACHE_HOME="$XDG_CACHE_ROOT" VLLM_CACHE_ROOT="$VLLM_CACHE_ROOT_FIXED" \
  CUDA_CACHE_PATH="$CUDA_CACHE_ROOT" FLASHINFER_WORKSPACE_BASE="$FLASHINFER_ROOT" \
  TORCHINDUCTOR_CACHE_DIR="$TORCHINDUCTOR_ROOT" TRITON_CACHE_DIR="$TRITON_ROOT" TORCH_EXTENSIONS_DIR="$TORCH_EXTENSIONS_ROOT" \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false LOGNAME="opd_uid_${EXECUTION_UID}" USER="opd_uid_${EXECUTION_UID}" \
  "$VLLM" serve "$MODEL" --served-model-name "$MODEL_ID" --host 127.0.0.1 --port "$PORT" \
  --tensor-parallel-size "$TP_SIZE" --data-parallel-size "$DP_SIZE" --gpu-memory-utilization 0.85 --max-model-len 65536 --max-num-seqs 32 \
  --limit-mm-per-prompt '{"image":16}' --compilation-config '{"pass_config":{"fuse_allreduce_rms":false}}' \
  --kernel-config '{"enable_flashinfer_autotune":false}' --trust-remote-code --reasoning-parser qwen3 --no-enable-log-requests \
  "${CHAT_TEMPLATE_ARGS[@]}" \
  >"$SERVE_LOG" 2>&1 &
SERVER_PID=$!

ready=0
deadline=$((SECONDS+1800))
while (( SECONDS < deadline )); do
  kill -0 "$SERVER_PID" 2>/dev/null || break
  if /usr/bin/curl --noproxy '*' -fsS --max-time 5 "http://127.0.0.1:$PORT/health" >/dev/null && \
     /usr/bin/curl --noproxy '*' -fsS --max-time 5 "http://127.0.0.1:$PORT/v1/models" | "$PYTHON" -c "import json,sys; x=json.load(sys.stdin); raise SystemExit(0 if '$MODEL_ID' in {str(v.get('id')) for v in x.get('data',[]) if isinstance(v,dict)} else 1)"; then
    ready=1
    break
  fi
  sleep 5
done
(( ready == 1 )) || die 'vLLM readiness failed'

write_status evaluating 'running all 1901 rows; invalid predictions count as wrong'
if (( AUDIT == 1 )); then
  env HOME="$PRIVATE_HOME" XDG_CONFIG_HOME="$XDG_CONFIG_ROOT" MPLCONFIGDIR="$MPLCONFIG_ROOT" TMPDIR="$TMP_ROOT" \
    HF_HOME="$HF_CACHE_ROOT" XDG_CACHE_HOME="$XDG_CACHE_ROOT" \
    "$PYTHON" -B "$AUDITOR" --blink-tsv "$DATA" --sidecar "$SIDECAR" --aggregate "$OUTPUT" \
    --model-id "$MODEL_ID" --api-base "http://127.0.0.1:$PORT/v1" --thinking false --temperature 0 \
    --top-p 1 --top-k -1 --min-p 0 --presence-penalty 0 --repetition-penalty 1 \
    --max-tokens 32768 --seed 42 --workers 32 >"$AGG_LOG" 2>&1
  [[ -s "$SIDECAR" ]] || die 'private audit sidecar is missing'
  [[ "$(stat -c '%a' "$SIDECAR")" == 600 ]] || die 'private audit sidecar mode differs'
else
  env HOME="$PRIVATE_HOME" XDG_CONFIG_HOME="$XDG_CONFIG_ROOT" MPLCONFIGDIR="$MPLCONFIG_ROOT" TMPDIR="$TMP_ROOT" \
    HF_HOME="$HF_CACHE_ROOT" XDG_CACHE_HOME="$XDG_CACHE_ROOT" \
    "$PYTHON" -B "$SCORER" --blink-tsv "$DATA" --output "$OUTPUT" --model-id "$MODEL_ID" --api-base "http://127.0.0.1:$PORT/v1" \
    --thinking false --temperature 0 --top-p 1 --top-k -1 --min-p 0 --presence-penalty 0 --repetition-penalty 1 --max-tokens 32768 --seed 42 --workers 32 \
    >"$AGG_LOG" 2>&1
fi
[[ -s "$OUTPUT" ]] || die 'aggregate output is missing'
write_status complete 'all 1901 rows completed under frozen checkpoint-v5 generation'
complete=1
"$PYTHON" -B - "$OUTPUT" "$AUDIT" <<'PY'
import json, pathlib, sys
x=json.loads(pathlib.Path(sys.argv[1]).read_text())
if int(sys.argv[2]):
    print(json.dumps({'protocol':x['schema_version'],'baseline_v5':x['baseline_v5'],'supplemental_format_only':x['supplemental_format_only']},sort_keys=True))
else:
    print(json.dumps({'protocol':x['preset'],'correct':x['correct'],'total':x['total'],'accuracy':x['accuracy']},sort_keys=True))
PY
