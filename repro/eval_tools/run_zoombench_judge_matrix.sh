#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/eval_env.sh"

JUDGE_MODEL_PATH="$OPD_WORKSPACE/Ckpt/Qwen3.5-27B"
JUDGE_MODEL_ID="Qwen3.5-27B-ZoomJudge"
CUDA_DEVICES="0,1,2,3,4,5,6,7"
TP_SIZE=2
DP_SIZE=4
PORT=19000
STARTUP_TIMEOUT=1200
POLL_INTERVAL=5
AUDIT_ROOT="$OPD_WORKSPACE/Output/opd_qwen35_9b/_setup/eval/zoombench_judge"
TARGET_SPECS=()

usage() {
  cat <<'EOF'
Usage: run_zoombench_judge_matrix.sh [options]

  --judge-model-path PATH   Fixed local Qwen3.5-27B checkpoint
  --judge-model-id ID       Served judge ID
  --cuda-devices LIST       CUDA_VISIBLE_DEVICES
  --tp N                    Judge tensor-parallel size
  --dp N                    Judge data-parallel size
  --port N                  Dedicated judge port (default: 19000)
  --target SPEC             Repeatable LABEL|TARGET_MODEL_ID|TARGET_EVAL_WORK_DIR
  --audit-root PATH         Judge service audit logs under H_Workspace/Output
  --startup-timeout SEC     Readiness timeout
  --poll-interval SEC       Health poll interval

Without --target, the base, teacher, and B1 work directories/model IDs are used.
Set DRY_RUN=1 to write and print commands without starting a server or judging.
EOF
}

need_value() {
  [[ $# -ge 2 && -n "$2" ]] || { echo "Missing value for $1" >&2; exit 2; }
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --judge-model-path) need_value "$@"; JUDGE_MODEL_PATH="$2"; shift 2 ;;
    --judge-model-id) need_value "$@"; JUDGE_MODEL_ID="$2"; shift 2 ;;
    --cuda-devices) need_value "$@"; CUDA_DEVICES="$2"; shift 2 ;;
    --tp) need_value "$@"; TP_SIZE="$2"; shift 2 ;;
    --dp) need_value "$@"; DP_SIZE="$2"; shift 2 ;;
    --port) need_value "$@"; PORT="$2"; shift 2 ;;
    --target) need_value "$@"; TARGET_SPECS+=("$2"); shift 2 ;;
    --audit-root) need_value "$@"; AUDIT_ROOT="$2"; shift 2 ;;
    --startup-timeout) need_value "$@"; STARTUP_TIMEOUT="$2"; shift 2 ;;
    --poll-interval) need_value "$@"; POLL_INTERVAL="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ ${#TARGET_SPECS[@]} -eq 0 ]]; then
  TARGET_SPECS=(
    "base|Qwen3.5-9B|$OPD_WORKSPACE/Output/opd_qwen35_9b/b0_base9b_20260806/eval"
    "teacher|Qwen3.5-27B|$OPD_WORKSPACE/Output/opd_qwen35_9b/t0_teacher27b_20260806/eval"
    "b1|Qwen3.5-9B-OPD-B1|$OPD_WORKSPACE/Output/opd_qwen35_9b/b1_ext27_full_vision6k_s42_launchfix1_20260807/eval"
  )
fi

require_uint() {
  local name="$1" value="$2" minimum="$3" maximum="$4"
  if [[ ! "$value" =~ ^[0-9]+$ ]] || (( value < minimum || value > maximum )); then
    echo "$name must be an integer in [$minimum, $maximum]; got: $value" >&2
    exit 2
  fi
}
require_uint --tp "$TP_SIZE" 1 8
require_uint --dp "$DP_SIZE" 1 8
require_uint --port "$PORT" 1024 65535
require_uint --startup-timeout "$STARTUP_TIMEOUT" 1 86400
require_uint --poll-interval "$POLL_INTERVAL" 1 60

if [[ ! "$CUDA_DEVICES" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "--cuda-devices must be a comma-separated numeric list" >&2
  exit 2
fi
IFS=',' read -r -a CUDA_ARRAY <<<"$CUDA_DEVICES"
declare -A CUDA_SEEN=()
for gpu in "${CUDA_ARRAY[@]}"; do
  [[ -z "${CUDA_SEEN[$gpu]:-}" ]] || { echo "Duplicate CUDA device: $gpu" >&2; exit 2; }
  CUDA_SEEN[$gpu]=1
done
if (( TP_SIZE * DP_SIZE > ${#CUDA_ARRAY[@]} )); then
  echo "TP*DP=$((TP_SIZE * DP_SIZE)) exceeds ${#CUDA_ARRAY[@]} visible GPUs" >&2
  exit 2
fi

resolve_path() {
  "$VLMEVAL_PYTHON" -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "$1"
}
JUDGE_MODEL_PATH="$(resolve_path "$JUDGE_MODEL_PATH")"
AUDIT_ROOT="$(resolve_path "$AUDIT_ROOT")"
case "$JUDGE_MODEL_PATH/" in "$OPD_WORKSPACE"/*) ;; *) echo "Judge path is outside H_Workspace" >&2; exit 2 ;; esac
case "$AUDIT_ROOT/" in "$OPD_WORKSPACE/Output"/*) ;; *) echo "Audit root is outside H_Workspace/Output" >&2; exit 2 ;; esac
[[ -f "$JUDGE_MODEL_PATH/config.json" ]] || { echo "Invalid judge checkpoint: $JUDGE_MODEL_PATH" >&2; exit 2; }

TARGET_LABELS=()
TARGET_MODEL_IDS=()
TARGET_WORK_DIRS=()
declare -A LABEL_SEEN=()
for spec in "${TARGET_SPECS[@]}"; do
  IFS='|' read -r label target_model_id target_work_dir extra <<<"$spec"
  if [[ -z "$label" || -z "$target_model_id" || -z "$target_work_dir" || -n "$extra" ]]; then
    echo "Invalid --target; expected LABEL|MODEL_ID|WORK_DIR: $spec" >&2
    exit 2
  fi
  [[ "$label" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "Unsafe target label: $label" >&2; exit 2; }
  [[ -z "${LABEL_SEEN[$label]:-}" ]] || { echo "Duplicate target label: $label" >&2; exit 2; }
  LABEL_SEEN[$label]=1
  target_work_dir="$(resolve_path "$target_work_dir")"
  case "$target_work_dir/" in
    "$OPD_WORKSPACE/Output"/*) ;;
    *) echo "Target work dir is outside H_Workspace/Output: $target_work_dir" >&2; exit 2 ;;
  esac
  TARGET_LABELS+=("$label")
  TARGET_MODEL_IDS+=("$target_model_id")
  TARGET_WORK_DIRS+=("$target_work_dir")
done

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_ROOT="$AUDIT_ROOT/$STAMP"
SERVE_LOG="$RUN_ROOT/serve.log"
HEALTH_LOG="$RUN_ROOT/health.log"
STATUS_FILE="$RUN_ROOT/status.env"
COMMAND_FILE="$RUN_ROOT/commands.sh"
mkdir -p "$RUN_ROOT"

write_status() {
  local state="$1" detail="${2:-}" temporary="$STATUS_FILE.tmp"
  {
    printf 'state=%q\n' "$state"
    printf 'detail=%q\n' "$detail"
    printf 'updated_utc=%q\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'judge_model_id=%q\n' "$JUDGE_MODEL_ID"
    printf 'judge_model_path=%q\n' "$JUDGE_MODEL_PATH"
    printf 'cuda_visible_devices=%q\n' "$CUDA_DEVICES"
    printf 'tp=%q\n' "$TP_SIZE"
    printf 'dp=%q\n' "$DP_SIZE"
    printf 'port=%q\n' "$PORT"
    printf 'targets=%q\n' "${TARGET_LABELS[*]}"
    printf 'self_judge_risk=%q\n' 'teacher target is judged by the same Qwen3.5-27B weights; all targets share family bias'
  } >"$temporary"
  mv "$temporary" "$STATUS_FILE"
}

{
  echo '#!/usr/bin/env bash'
  echo 'set -euo pipefail'
  printf 'DRY_RUN=${DRY_RUN:-0} %q' "$SCRIPT_DIR/run_zoombench_judge_matrix.sh"
  printf ' --judge-model-path %q --judge-model-id %q' "$JUDGE_MODEL_PATH" "$JUDGE_MODEL_ID"
  printf ' --cuda-devices %q --tp %q --dp %q --port %q' "$CUDA_DEVICES" "$TP_SIZE" "$DP_SIZE" "$PORT"
  printf ' --audit-root %q' "$AUDIT_ROOT"
  printf ' --startup-timeout %q --poll-interval %q' "$STARTUP_TIMEOUT" "$POLL_INTERVAL"
  for i in "${!TARGET_LABELS[@]}"; do
    printf ' --target %q' "${TARGET_LABELS[$i]}|${TARGET_MODEL_IDS[$i]}|${TARGET_WORK_DIRS[$i]}"
  done
  printf '\n'
} >"$COMMAND_FILE"
chmod 0600 "$COMMAND_FILE"
write_status initialized "fixed judge matrix audit bundle created"

JUDGE_COMMON_ENV=(
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
  VLLM_ENGINE_READY_TIMEOUT_S="$STARTUP_TIMEOUT"
  EVAL_MODEL_ID="$JUDGE_MODEL_ID"
  EVAL_API_HOST=127.0.0.1
  EVAL_API_PORT="$PORT"
  EVAL_TP_SIZE="$TP_SIZE"
  EVAL_DP_SIZE="$DP_SIZE"
)
JUDGE_API_BASE="http://127.0.0.1:$PORT/v1"

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  {
    echo "DRY RUN ONLY: no judge server or judge requests will be started."
    echo "WARNING: teacher/Qwen3.5-27B is self-judged; all three targets share judge-family bias."
    env "${JUDGE_COMMON_ENV[@]}" DRY_RUN=1 "$SCRIPT_DIR/serve_qwen35.sh" \
      "$JUDGE_MODEL_PATH" --default-chat-template-kwargs '{"enable_thinking":false}'
    for i in "${!TARGET_LABELS[@]}"; do
      env EVAL_MODEL_ID="${TARGET_MODEL_IDS[$i]}" \
        EVAL_WORK_DIR="${TARGET_WORK_DIRS[$i]}" \
        ZOOMBENCH_MODE=judge-score \
        ZOOMBENCH_JUDGE_API_BASE="$JUDGE_API_BASE" \
        ZOOMBENCH_JUDGE_API_KEY=REDACTED_DRY_RUN_KEY \
        ZOOMBENCH_JUDGE_MODEL="$JUDGE_MODEL_ID" \
        DRY_RUN=1 "$SCRIPT_DIR/run_zoombench.sh"
    done
  } 2>&1 | tee "$RUN_ROOT/dry_run.log"
  printf 'dry-run placeholder: judge vLLM was not started\n' >"$SERVE_LOG"
  for label in "${TARGET_LABELS[@]}"; do
    printf 'dry-run placeholder: target %s was not judged\n' "$label" >"$RUN_ROOT/${label}_judge_score.log"
  done
  write_status dry_run_complete "no judge server or requests were started"
  exit 0
fi

for i in "${!TARGET_LABELS[@]}"; do
  model_tag="${TARGET_MODEL_IDS[$i]//\//_}_seed${EVAL_SEED:-42}"
  answer_path="${TARGET_WORK_DIRS[$i]}/zoombench/model_answer/zoombench/${model_tag}_answer.jsonl"
  [[ -f "$answer_path" ]] || { echo "Missing ZoomBench inference output: $answer_path" >&2; exit 1; }
done
command -v curl >/dev/null || { echo "curl is required" >&2; exit 1; }
command -v setsid >/dev/null || { echo "setsid is required" >&2; exit 1; }
"$VLMEVAL_PYTHON" - "$PORT" <<'PY'
import socket, sys
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind(("127.0.0.1", int(sys.argv[1])))
PY

SERVER_PID=""
cleanup() {
  local rc=$?
  trap - EXIT INT TERM HUP
  if [[ -n "$SERVER_PID" ]] && kill -0 -- "-$SERVER_PID" 2>/dev/null; then
    kill -TERM -- "-$SERVER_PID" 2>/dev/null || true
    for _ in {1..60}; do
      kill -0 -- "-$SERVER_PID" 2>/dev/null || break
      sleep 1
    done
    kill -KILL -- "-$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  if (( rc == 0 )); then
    write_status complete "all targets judged and scored with fixed judge; server stopped"
  else
    write_status failed "exit code $rc; judge server stopped"
  fi
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

write_status starting "launching fixed local Qwen3.5-27B judge"
setsid env "${JUDGE_COMMON_ENV[@]}" "$SCRIPT_DIR/serve_qwen35.sh" \
  "$JUDGE_MODEL_PATH" --default-chat-template-kwargs '{"enable_thinking":false}' \
  >"$SERVE_LOG" 2>&1 &
SERVER_PID=$!
printf '%s launched judge process group %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$SERVER_PID" >"$HEALTH_LOG"

deadline=$((SECONDS + STARTUP_TIMEOUT))
ready=0
while (( SECONDS < deadline )); do
  kill -0 "$SERVER_PID" 2>/dev/null || { echo "Judge server exited before readiness" >&2; exit 1; }
  health_tmp="$RUN_ROOT/health.tmp"
  models_tmp="$RUN_ROOT/models.tmp"
  if curl -fsS --max-time 5 "http://127.0.0.1:$PORT/health" >"$health_tmp" \
    && curl -fsS --max-time 5 "http://127.0.0.1:$PORT/v1/models" >"$models_tmp" \
    && "$VLMEVAL_PYTHON" - "$models_tmp" "$JUDGE_MODEL_ID" <<'PY'
import json, pathlib, sys
payload = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
ids = {str(x.get("id")) for x in payload.get("data", []) if isinstance(x, dict)}
raise SystemExit(0 if sys.argv[2] in ids else 1)
PY
  then
    mv "$health_tmp" "$RUN_ROOT/health.response"
    mv "$models_tmp" "$RUN_ROOT/models.json"
    ready=1
    break
  fi
  rm -f "$health_tmp" "$models_tmp"
  printf '%s waiting for judge /health and /v1/models\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$HEALTH_LOG"
  sleep "$POLL_INTERVAL"
done
(( ready == 1 )) || { echo "Timed out waiting for fixed judge; see $SERVE_LOG" >&2; exit 1; }

write_status judging "fixed judge ready; processing all target outputs"
for i in "${!TARGET_LABELS[@]}"; do
  label="${TARGET_LABELS[$i]}"
  target_model_id="${TARGET_MODEL_IDS[$i]}"
  target_work_dir="${TARGET_WORK_DIRS[$i]}"
  target_log="$target_work_dir/zoombench/judge_score.log"
  mkdir -p "$target_work_dir/zoombench"
  env EVAL_MODEL_ID="$target_model_id" EVAL_WORK_DIR="$target_work_dir" \
    ZOOMBENCH_MODE=judge-score \
    ZOOMBENCH_JUDGE_API_BASE="$JUDGE_API_BASE" \
    ZOOMBENCH_JUDGE_API_KEY="${ZOOMBENCH_JUDGE_API_KEY:-sk-local-opd}" \
    ZOOMBENCH_JUDGE_MODEL="$JUDGE_MODEL_ID" \
    "$SCRIPT_DIR/run_zoombench.sh" 2>&1 | tee "$RUN_ROOT/${label}_judge_score.log" "$target_log"

  "$VLMEVAL_PYTHON" - "$target_work_dir/zoombench/judge_protocol.json" \
    "$label" "$target_model_id" "$JUDGE_MODEL_ID" "$JUDGE_MODEL_PATH" "$PORT" \
    "$CUDA_DEVICES" "$TP_SIZE" "$DP_SIZE" <<'PY'
import hashlib, json, os, pathlib, sys
path, label, target, judge, judge_path, port, cuda_devices, tp, dp = sys.argv[1:]
config = pathlib.Path(judge_path) / "config.json"
payload = {
    "target_label": label,
    "target_model_id": target,
    "judge_model_id": judge,
    "judge_model_path": judge_path,
    "judge_config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
    "judge_port": int(port),
    "judge_cuda_visible_devices": cuda_devices,
    "judge_tensor_parallel_size": int(tp),
    "judge_data_parallel_size": int(dp),
    "temperature": 0,
    "enable_thinking": False,
    "vision_opd_reference_commit": "c8a8fdd1f88eef1b5ef4fe6a8d64eb0272917471",
    "protocol": "rule/MathRuler pass followed by fixed local semantic Qwen judge, then accuracy",
    "self_judge_risk": (
        "The Qwen3.5-27B teacher target is judged by the same model weights; "
        "all targets may also benefit from Qwen-family stylistic bias. Report this limitation."
    ),
}
target_path = pathlib.Path(path)
target_path.parent.mkdir(parents=True, exist_ok=True)
temporary = target_path.with_suffix(target_path.suffix + ".tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, target_path)
PY
done
