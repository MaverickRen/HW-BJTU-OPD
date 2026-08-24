#!/usr/bin/env bash

set -euo pipefail
umask 077

# A fail-closed wrapper around the source-pinned Vision-OPD evaluator.  This is
# deliberately independent of eval_env.sh: that file creates cache/output
# directories when sourced, which would make DRY_RUN mutate the workspace.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(realpath -m "$SCRIPT_DIR/../../..")"
REFERENCE_ROOT="$WORKSPACE_ROOT/Codes/Vision-OPD-reference"
OUTPUT_ROOT="$WORKSPACE_ROOT/Output"
DATA_ROOT="$WORKSPACE_ROOT/Dataset/eval/vision_opd_reference_c8a8fdd"
CONTRACT_HELPER="$SCRIPT_DIR/vision_opd_reference_contract.py"
HARDENED_JUDGE_DRIVER="$SCRIPT_DIR/judge_qwenlm_fail_closed.py"
PREPARER="$SCRIPT_DIR/prepare_vision_opd_reference_data.py"
CONTRACT_PYTHON="${VISION_OPD_CONTRACT_PYTHON:-$(command -v python3 || true)}"
REFERENCE_PYTHON="${VISION_OPD_REFERENCE_PYTHON:-$WORKSPACE_ROOT/UV_Env/verl-opd-qwen35/bin/python}"

EXPECTED_REFERENCE_COMMIT="c8a8fdd1f88eef1b5ef4fe6a8d64eb0272917471"
EXPECTED_PREPARE_SHA256="8e71bb3f04c741434ab505acfc4d2b6107cefec2864cdf909b2ff0e4fad79c5a"
EXPECTED_INFER_SHA256="bb379999932658907196cdc98d22c60d63e3308cb5a867317481c4a85af70374"
EXPECTED_JUDGE_SHA256="abbe11dacf7fae19728ca16407a02c91d04a9bc8ea72edd3b4a91b6224f4b670"
EXPECTED_SCORE_SHA256="695dbddc3e63a1b9f8971c0d414d963a5da94776863d58589feaa4a1c6b0f025"
EXPECTED_HARDENED_JUDGE_SHA256="bd873d7ea58aad676b805ce45faa5e96314817e81840fd803c8926e337c17a5f"
EXPECTED_PREPARER_SHA256="b765dd7f9d06397e4606d9f40af0468c02704821083cca25a02cbaf779a05d2f"
EXPECTED_CONTRACT_SHA256="bee94feda73270e8e2a28e29a4cc882b27c05734d5e6d9540b8bbf3a2c1f2daa"

SEED=42
SEED_LABEL=seed42
ENABLE_THINKING=false
INFER_TEMPERATURE=0
INFER_MAX_TOKENS=32768
INFER_MAX_RETRIES=3
INFER_PARALLEL_WORKERS=256
JUDGE_TEMPERATURE=0
JUDGE_MAX_TOKENS=2048

case "${DRY_RUN:-0}" in
  0|1) ;;
  *) echo "ERROR: DRY_RUN must be 0 or 1" >&2; exit 2 ;;
esac

usage() {
  cat <<'EOF'
Usage:
  run_vision_opd_reference_eval.sh MODE --benchmark BENCHMARK \
    --run-root PATH --model-path PATH --model-id ID --model-tag TAG [options]

MODE:
  preflight     Freeze/revalidate data and run contracts only.
  answers       Freeze/revalidate data, run, and a complete answer artifact.
  prepare       Download/convert official data, then freeze data/model/run.
  infer         Freeze inputs, resume official inference, strictly gate answers.
  judge         Strictly gate answers, run the official API judge, gate judge JSON.
  score         Strictly gate answers/judge JSON and save official cal_acc output.
  judge-score   Run judge and score in sequence.

Required for every mode:
  --benchmark NAME       vstar or mme-realworld-lite
  --run-root PATH        Explicit run directory strictly below H_Workspace/Output
  --model-path PATH      Explicit checkpoint path frozen by the contract helper
  --model-id ID          Explicit OpenAI-compatible served model ID
  --model-tag TAG        Explicit safe output tag ending in _seed42

Additional requirements:
  infer:
    --api-base URL
  judge, score, judge-score:
    --judge-api-base URL --judge-model ID --judge-model-contract PATH
    --judge-runtime-contract PATH is additionally required by a judge matrix.

API keys are accepted only through environment variables and are never written
to a receipt or dry-run output:
  VISION_OPD_REFERENCE_API_KEY       (fallback: OPENAI_API_KEY, then EMPTY)
  VISION_OPD_REFERENCE_JUDGE_API_KEY (fallback: JUDGE_API_KEY,
                                      OPENAI_API_KEY, then EMPTY)

The data root is fixed at:
  H_Workspace/Dataset/eval/vision_opd_reference_c8a8fdd

Set DRY_RUN=1 to print the complete source-pinned command plan.  Dry-run does
not create directories or files and does not make network/API/GPU calls.
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 2
}

need_value() {
  [[ $# -ge 2 && -n "$2" ]] || die "missing value for $1"
}

MODE="${1:-}"
case "$MODE" in
  preflight|answers|prepare|infer|judge|score|judge-score) shift ;;
  -h|--help) usage; exit 0 ;;
  "") usage >&2; die "MODE is required" ;;
  *) usage >&2; die "unsupported MODE: $MODE" ;;
esac

BENCHMARK=""
RUN_ROOT=""
MODEL_PATH=""
MODEL_ID=""
MODEL_TAG=""
API_BASE=""
JUDGE_API_BASE=""
JUDGE_MODEL=""
JUDGE_MODEL_CONTRACT=""
JUDGE_RUNTIME_CONTRACT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --benchmark) need_value "$@"; BENCHMARK="$2"; shift 2 ;;
    --run-root) need_value "$@"; RUN_ROOT="$2"; shift 2 ;;
    --model-path) need_value "$@"; MODEL_PATH="$2"; shift 2 ;;
    --model-id) need_value "$@"; MODEL_ID="$2"; shift 2 ;;
    --model-tag) need_value "$@"; MODEL_TAG="$2"; shift 2 ;;
    --api-base) need_value "$@"; API_BASE="$2"; shift 2 ;;
    --judge-api-base) need_value "$@"; JUDGE_API_BASE="$2"; shift 2 ;;
    --judge-model) need_value "$@"; JUDGE_MODEL="$2"; shift 2 ;;
    --judge-model-contract) need_value "$@"; JUDGE_MODEL_CONTRACT="$2"; shift 2 ;;
    --judge-runtime-contract) need_value "$@"; JUDGE_RUNTIME_CONTRACT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; die "unknown option: $1" ;;
  esac
done

[[ -n "$BENCHMARK" ]] || die "--benchmark is required"
[[ -n "$RUN_ROOT" ]] || die "--run-root is required"
[[ -n "$MODEL_PATH" ]] || die "--model-path is required"
[[ -n "$MODEL_ID" ]] || die "--model-id is required"
[[ -n "$MODEL_TAG" ]] || die "--model-tag is required"

case "$BENCHMARK" in
  vstar) BENCHMARK_JSON_NAME="vstar.json" ;;
  mme-realworld-lite) BENCHMARK_JSON_NAME="MME_RealWorld_Lite.json" ;;
  *) die "unsupported benchmark: $BENCHMARK (expected vstar or mme-realworld-lite)" ;;
esac

[[ "$MODEL_ID" =~ ^[A-Za-z0-9._/:+-]+$ ]] || die "unsafe --model-id: $MODEL_ID"
[[ "$MODEL_TAG" =~ ^[A-Za-z0-9._-]+$ ]] || die "unsafe --model-tag: $MODEL_TAG"
[[ "$MODEL_TAG" == *_seed42 ]] || die "--model-tag must end in _seed42"

RUN_ROOT="$(realpath -ms "$RUN_ROOT")"
MODEL_PATH="$(realpath -m "$MODEL_PATH")"
[[ "$RUN_ROOT" != "$OUTPUT_ROOT" ]] || die "--run-root cannot be H_Workspace/Output itself"
case "$RUN_ROOT" in
  "$OUTPUT_ROOT"/*) ;;
  *) die "--run-root must be strictly below $OUTPUT_ROOT: $RUN_ROOT" ;;
esac

case "$MODE" in
  infer)
    [[ -n "$API_BASE" ]] || die "--api-base is required for infer"
    ;;
  judge|score|judge-score)
    [[ -n "$JUDGE_API_BASE" ]] || die "--judge-api-base is required for $MODE"
    [[ -n "$JUDGE_MODEL" ]] || die "--judge-model is required for $MODE"
    [[ -n "$JUDGE_MODEL_CONTRACT" ]] || die "--judge-model-contract is required for $MODE"
    ;;
esac

if [[ -n "$JUDGE_MODEL_CONTRACT" ]]; then
  JUDGE_MODEL_CONTRACT="$(realpath -ms "$JUDGE_MODEL_CONTRACT")"
  case "$JUDGE_MODEL_CONTRACT" in
    "$WORKSPACE_ROOT"/*) ;;
    *) die "--judge-model-contract must be under $WORKSPACE_ROOT" ;;
  esac
fi
if [[ -n "$JUDGE_RUNTIME_CONTRACT" ]]; then
  JUDGE_RUNTIME_CONTRACT="$(realpath -ms "$JUDGE_RUNTIME_CONTRACT")"
  case "$JUDGE_RUNTIME_CONTRACT" in
    "$WORKSPACE_ROOT"/*) ;;
    *) die "--judge-runtime-contract must be under $WORKSPACE_ROOT" ;;
  esac
fi

case "$API_BASE" in
  ""|http://*|https://*) ;;
  *) die "--api-base must use http:// or https://" ;;
esac
case "$JUDGE_API_BASE" in
  ""|http://*|https://*) ;;
  *) die "--judge-api-base must use http:// or https://" ;;
esac
if [[ -n "$JUDGE_MODEL" && ! "$JUDGE_MODEL" =~ ^[A-Za-z0-9._/:+-]+$ ]]; then
  die "unsafe --judge-model: $JUDGE_MODEL"
fi

command -v git >/dev/null || die "git is required"
command -v sha256sum >/dev/null || die "sha256sum is required"
[[ -n "$CONTRACT_PYTHON" && -x "$CONTRACT_PYTHON" ]] || die "contract Python is unavailable"
[[ -f "$CONTRACT_HELPER" && ! -L "$CONTRACT_HELPER" ]] || {
  die "contract helper is missing or a symlink: $CONTRACT_HELPER"
}
[[ -f "$HARDENED_JUDGE_DRIVER" && ! -L "$HARDENED_JUDGE_DRIVER" ]] || {
  die "hardened judge driver is missing or a symlink: $HARDENED_JUDGE_DRIVER"
}
[[ -f "$PREPARER" && ! -L "$PREPARER" ]] || {
  die "pinned preparation tool is missing or a symlink: $PREPARER"
}
[[ -d "$REFERENCE_ROOT/.git" ]] || die "reference repository is missing: $REFERENCE_ROOT"

verify_reference_source() {
  local actual_commit actual_sha path expected_sha
  # The Canoe container may run under a batch UID different from the shared
  # checkout owner.  Command-local safe.directory keeps this read-only check
  # portable without mutating user/global git configuration.
  actual_commit="$(git -c safe.directory="$REFERENCE_ROOT" -C "$REFERENCE_ROOT" rev-parse HEAD)"
  [[ "$actual_commit" == "$EXPECTED_REFERENCE_COMMIT" ]] || {
    die "reference commit mismatch: expected $EXPECTED_REFERENCE_COMMIT, got $actual_commit"
  }
  while [[ $# -gt 0 ]]; do
    path="$1"
    expected_sha="$2"
    shift 2
    [[ -f "$path" ]] || die "reference source is missing: $path"
    actual_sha="$(sha256sum "$path")"
    actual_sha="${actual_sha%% *}"
    [[ "$actual_sha" == "$expected_sha" ]] || {
      die "reference source hash mismatch for $path: expected $expected_sha, got $actual_sha"
    }
  done
}

verify_reference_source \
  "$REFERENCE_ROOT/eval/prepare_data.py" "$EXPECTED_PREPARE_SHA256" \
  "$REFERENCE_ROOT/eval/infer.py" "$EXPECTED_INFER_SHA256" \
  "$REFERENCE_ROOT/eval/judge_qwenlm.py" "$EXPECTED_JUDGE_SHA256" \
  "$REFERENCE_ROOT/eval/cal_acc.py" "$EXPECTED_SCORE_SHA256"

actual_hardened_judge_sha256="$(sha256sum "$HARDENED_JUDGE_DRIVER")"
actual_hardened_judge_sha256="${actual_hardened_judge_sha256%% *}"
[[ "$actual_hardened_judge_sha256" == "$EXPECTED_HARDENED_JUDGE_SHA256" ]] || {
  die "hardened judge driver hash mismatch: expected $EXPECTED_HARDENED_JUDGE_SHA256, got $actual_hardened_judge_sha256"
}
actual_preparer_sha256="$(sha256sum "$PREPARER")"
actual_preparer_sha256="${actual_preparer_sha256%% *}"
[[ "$actual_preparer_sha256" == "$EXPECTED_PREPARER_SHA256" ]] || {
  die "preparation tool hash mismatch: expected $EXPECTED_PREPARER_SHA256, got $actual_preparer_sha256"
}
actual_contract_sha256="$(sha256sum "$CONTRACT_HELPER")"
actual_contract_sha256="${actual_contract_sha256%% *}"
[[ "$actual_contract_sha256" == "$EXPECTED_CONTRACT_SHA256" ]] || {
  die "contract helper hash mismatch: expected $EXPECTED_CONTRACT_SHA256, got $actual_contract_sha256"
}

if [[ "${DRY_RUN:-0}" != 1 ]]; then
  [[ -x "$REFERENCE_PYTHON" ]] || die "reference Python is unavailable: $REFERENCE_PYTHON"
  [[ -d "$MODEL_PATH" && -f "$MODEL_PATH/config.json" ]] || {
    die "invalid model checkpoint (directory/config.json required): $MODEL_PATH"
  }
fi

BENCHMARK_JSON="$DATA_ROOT/$BENCHMARK_JSON_NAME"
ANSWER_JSONL="$RUN_ROOT/model_answer/$BENCHMARK/${MODEL_TAG}_answer.jsonl"
JUDGE_JSON="$RUN_ROOT/judge/$BENCHMARK/${MODEL_TAG}_answer.jsonl"
SCORE_LOG="$RUN_ROOT/score/$BENCHMARK/${MODEL_TAG}_cal_acc.log"
PREPARATION_RECEIPT="$DATA_ROOT/preparation_receipt.json"
PREPARATION_LOCK="$WORKSPACE_ROOT/Locks/vision_opd_reference_prepare.lock"
DATA_RECEIPT="$OUTPUT_ROOT/vision_opd_reference_contracts/data/$BENCHMARK/data.json"
RUN_RECEIPT="$RUN_ROOT/contracts/$BENCHMARK/$MODEL_TAG/run.json"
ANSWERS_RECEIPT="$RUN_ROOT/contracts/$BENCHMARK/$MODEL_TAG/answers.json"
JUDGE_RECEIPT="$RUN_ROOT/contracts/$BENCHMARK/$MODEL_TAG/judge.json"
ARTIFACT_LOCK_PATH="$WORKSPACE_ROOT/Locks/vision_opd_reference_${BENCHMARK}_${MODEL_TAG}.lock"
STAGING_ROOT="$RUN_ROOT/.staging"

DATA_CONTRACT_CMD=(
  "$CONTRACT_PYTHON" "$CONTRACT_HELPER" data
  --workspace-root "$WORKSPACE_ROOT"
  --reference-root "$REFERENCE_ROOT"
  --benchmark "$BENCHMARK"
  --benchmark-json "$BENCHMARK_JSON"
  --data-root "$DATA_ROOT"
  --preparation-receipt "$PREPARATION_RECEIPT"
  --preparer-path "$PREPARER"
  --receipt "$DATA_RECEIPT"
)
RUN_CONTRACT_CMD=(
  "$CONTRACT_PYTHON" "$CONTRACT_HELPER" run
  --workspace-root "$WORKSPACE_ROOT"
  --reference-root "$REFERENCE_ROOT"
  --data-receipt "$DATA_RECEIPT"
  --model-path "$MODEL_PATH"
  --model-id "$MODEL_ID"
  --model-tag "$MODEL_TAG"
  --seed "$SEED"
  --seed-label "$SEED_LABEL"
  --enable-thinking "$ENABLE_THINKING"
  --temperature "$INFER_TEMPERATURE"
  --max-tokens "$INFER_MAX_TOKENS"
  --receipt "$RUN_RECEIPT"
)
ANSWERS_GATE_CMD=(
  "$CONTRACT_PYTHON" "$CONTRACT_HELPER" answers
  --workspace-root "$WORKSPACE_ROOT"
  --reference-root "$REFERENCE_ROOT"
  --run-contract "$RUN_RECEIPT"
  --answer-jsonl "$ANSWER_JSONL"
  --receipt "$ANSWERS_RECEIPT"
)
PARTIAL_ANSWERS_CMD=(
  "$CONTRACT_PYTHON" "$CONTRACT_HELPER" partial-answers
  --workspace-root "$WORKSPACE_ROOT"
  --reference-root "$REFERENCE_ROOT"
  --run-contract "$RUN_RECEIPT"
  --answer-jsonl "$ANSWER_JSONL"
  --model-id "$MODEL_ID"
  --model-tag "$MODEL_TAG"
  --quiet
)
JUDGE_GATE_CMD=(
  "$CONTRACT_PYTHON" "$CONTRACT_HELPER" judge
  --workspace-root "$WORKSPACE_ROOT"
  --reference-root "$REFERENCE_ROOT"
  --answers-receipt "$ANSWERS_RECEIPT"
  --judge-json "$JUDGE_JSON"
  --judge-model-contract "$JUDGE_MODEL_CONTRACT"
  --judge-model-id "$JUDGE_MODEL"
  --judge-api-base "$JUDGE_API_BASE"
  --judge-temperature "$JUDGE_TEMPERATURE"
  --judge-max-tokens "$JUDGE_MAX_TOKENS"
  --receipt "$JUDGE_RECEIPT"
)
if [[ -n "$JUDGE_RUNTIME_CONTRACT" ]]; then
  JUDGE_GATE_CMD+=(--judge-runtime-contract "$JUDGE_RUNTIME_CONTRACT")
fi
ARTIFACT_PATH_GATE_CMD=(
  "$CONTRACT_PYTHON" "$CONTRACT_HELPER" artifact-paths
  --workspace-root "$WORKSPACE_ROOT"
  --create
  --directory "$RUN_ROOT"
  --directory "$STAGING_ROOT"
  --leaf "$ANSWER_JSONL"
  --leaf "$ANSWER_JSONL.tmp"
  --leaf "$JUDGE_JSON"
  --leaf "$SCORE_LOG"
  --leaf "$DATA_RECEIPT"
  --leaf "$RUN_RECEIPT"
  --leaf "$ANSWERS_RECEIPT"
  --leaf "$JUDGE_RECEIPT"
  --quiet
)
PREPARER_VERIFY_CMD=(
  "$REFERENCE_PYTHON" "$PREPARER"
  --workspace-root "$WORKSPACE_ROOT"
  --reference-root "$REFERENCE_ROOT"
  --data-root "$DATA_ROOT"
  --lock-path "$PREPARATION_LOCK"
  --verify-only
)
PREPARER_EXECUTE_CMD=(
  "$REFERENCE_PYTHON" "$PREPARER"
  --workspace-root "$WORKSPACE_ROOT"
  --reference-root "$REFERENCE_ROOT"
  --data-root "$DATA_ROOT"
  --lock-path "$PREPARATION_LOCK"
  --execute
)

INFER_API_KEY="${VISION_OPD_REFERENCE_API_KEY:-${OPENAI_API_KEY:-EMPTY}}"
JUDGE_API_KEY_VALUE="${VISION_OPD_REFERENCE_JUDGE_API_KEY:-${JUDGE_API_KEY:-${OPENAI_API_KEY:-EMPTY}}}"
INFER_CMD=(
  "$REFERENCE_PYTHON" "$REFERENCE_ROOT/eval/infer.py"
  --benchmark "$BENCHMARK"
  --benchmark_json "$BENCHMARK_JSON"
  --out_dir model_answer
  --model_name "$MODEL_TAG"
  --seed "$SEED"
  --api_base "$API_BASE"
  --api_key "$INFER_API_KEY"
  --model_id "$MODEL_ID"
  --max_tokens "$INFER_MAX_TOKENS"
  --max_retries "$INFER_MAX_RETRIES"
  --parallel_workers "$INFER_PARALLEL_WORKERS"
  --enable_thinking False
)
INFER_DRY_CMD=(
  "$REFERENCE_PYTHON" "$REFERENCE_ROOT/eval/infer.py"
  --benchmark "$BENCHMARK"
  --benchmark_json "$BENCHMARK_JSON"
  --out_dir model_answer
  --model_name "$MODEL_TAG"
  --seed "$SEED"
  --api_base "$API_BASE"
  --api_key '<redacted-env:VISION_OPD_REFERENCE_API_KEY>'
  --model_id "$MODEL_ID"
  --max_tokens "$INFER_MAX_TOKENS"
  --max_retries "$INFER_MAX_RETRIES"
  --parallel_workers "$INFER_PARALLEL_WORKERS"
  --enable_thinking False
)
JUDGE_CMD=(
  "$REFERENCE_PYTHON" "$HARDENED_JUDGE_DRIVER"
  --official-script "$REFERENCE_ROOT/eval/judge_qwenlm.py"
  --official-sha256 "$EXPECTED_JUDGE_SHA256"
  --benchmark "$BENCHMARK"
  --model "$MODEL_TAG"
  --api-base "$JUDGE_API_BASE"
  --judge-model "$JUDGE_MODEL"
  --judge-max-tokens "$JUDGE_MAX_TOKENS"
)
JUDGE_DRY_CMD=(
  env 'VISION_OPD_REFERENCE_JUDGE_API_KEY=<redacted-env>'
  "$REFERENCE_PYTHON" "$HARDENED_JUDGE_DRIVER"
  --official-script "$REFERENCE_ROOT/eval/judge_qwenlm.py"
  --official-sha256 "$EXPECTED_JUDGE_SHA256"
  --benchmark "$BENCHMARK"
  --model "$MODEL_TAG"
  --api-base "$JUDGE_API_BASE"
  --judge-model "$JUDGE_MODEL"
  --judge-max-tokens "$JUDGE_MAX_TOKENS"
)
SCORE_CMD=(
  "$REFERENCE_PYTHON" "$REFERENCE_ROOT/eval/cal_acc.py"
  --benchmark "$BENCHMARK"
  --judge_json "$JUDGE_JSON"
  --benchmark_json "$BENCHMARK_JSON"
)

print_cwd_command() {
  local label="$1"
  shift
  printf 'DRY_RUN[%s]: cd %q &&' "$label" "$RUN_ROOT"
  printf ' %q' "$@"
  printf '\n'
}

print_command() {
  local label="$1"
  shift
  printf 'DRY_RUN[%s]:' "$label"
  printf ' %q' "$@"
  printf '\n'
}

print_score_command() {
  printf 'DRY_RUN[score]: cd %q &&' "$RUN_ROOT"
  printf ' %q' "${SCORE_CMD[@]}"
  printf ' 2>&1 | tee %q\n' "$SCORE_LOG.candidate-run.<mktemp>"
}

dry_run_plan() {
  printf '%s\n' \
    "DRY_RUN[protocol]: reference_commit=$EXPECTED_REFERENCE_COMMIT" \
    "DRY_RUN[protocol]: prepare_sha256=$EXPECTED_PREPARE_SHA256" \
    "DRY_RUN[protocol]: infer_sha256=$EXPECTED_INFER_SHA256" \
    "DRY_RUN[protocol]: judge_sha256=$EXPECTED_JUDGE_SHA256" \
    "DRY_RUN[protocol]: hardened_judge_sha256=$EXPECTED_HARDENED_JUDGE_SHA256" \
    "DRY_RUN[protocol]: preparer_sha256=$EXPECTED_PREPARER_SHA256 preparation_schema=vision_opd_reference_preparation_v2" \
    "DRY_RUN[protocol]: contract_sha256=$EXPECTED_CONTRACT_SHA256" \
    "DRY_RUN[protocol]: score_sha256=$EXPECTED_SCORE_SHA256" \
    "DRY_RUN[protocol]: seed=$SEED seed_label=$SEED_LABEL enable_thinking=$ENABLE_THINKING infer_temperature=$INFER_TEMPERATURE infer_max_tokens=$INFER_MAX_TOKENS judge_temperature=$JUDGE_TEMPERATURE judge_max_tokens=$JUDGE_MAX_TOKENS" \
    "DRY_RUN[paths]: data_root=$DATA_ROOT" \
    "DRY_RUN[paths]: preparation_receipt=$PREPARATION_RECEIPT data_contract=$DATA_RECEIPT" \
    "DRY_RUN[paths]: run_root=$RUN_ROOT" \
    "DRY_RUN[paths]: answer_jsonl=$ANSWER_JSONL" \
    "DRY_RUN[paths]: judge_json=$JUDGE_JSON" \
    "DRY_RUN[paths]: score_log=$SCORE_LOG" \
    "DRY_RUN[paths]: artifact_lock=$ARTIFACT_LOCK_PATH (NOT opened or locked) staging_root=$STAGING_ROOT" \
    "DRY_RUN[resume_policy]: complete output is strictly gated and skipped; receipt-free partial JSONL resumes only after exact UID/schema/model validation; any drift is terminal" \
    "DRY_RUN[publication]: judge runs from hardlink/copy sibling staging; judge/score/receipts use fsync plus renameat2 NOREPLACE and retain conflicts"

  print_command artifact_path_gate_after_lock "${ARTIFACT_PATH_GATE_CMD[@]}"

  case "$MODE" in
    preflight)
      print_command preparation_verify_only "${PREPARER_VERIFY_CMD[@]}"
      print_command data_contract "${DATA_CONTRACT_CMD[@]}"
      print_command run_contract "${RUN_CONTRACT_CMD[@]}"
      ;;
    answers)
      print_command preparation_verify_only "${PREPARER_VERIFY_CMD[@]}"
      print_command data_contract "${DATA_CONTRACT_CMD[@]}"
      print_command run_contract "${RUN_CONTRACT_CMD[@]}"
      print_command answers_strict "${ANSWERS_GATE_CMD[@]}"
      ;;
    prepare)
      print_command preparation_execute_or_reverify "${PREPARER_EXECUTE_CMD[@]}"
      print_command preparation_verify_only "${PREPARER_VERIFY_CMD[@]}"
      print_command data_contract "${DATA_CONTRACT_CMD[@]}"
      print_command run_contract "${RUN_CONTRACT_CMD[@]}"
      ;;
    infer)
      print_command preparation_verify_only "${PREPARER_VERIFY_CMD[@]}"
      print_command data_contract "${DATA_CONTRACT_CMD[@]}"
      print_command run_contract "${RUN_CONTRACT_CMD[@]}"
      print_command answers_preflight_strict_if_present "${ANSWERS_GATE_CMD[@]}"
      print_command partial_answers_strict_before_resume "${PARTIAL_ANSWERS_CMD[@]}"
      print_cwd_command infer_if_answers_not_complete "${INFER_DRY_CMD[@]}"
      print_command answers_postflight_strict "${ANSWERS_GATE_CMD[@]}"
      ;;
    judge)
      print_command preparation_verify_only "${PREPARER_VERIFY_CMD[@]}"
      print_command data_contract "${DATA_CONTRACT_CMD[@]}"
      print_command run_contract "${RUN_CONTRACT_CMD[@]}"
      print_command answers_strict "${ANSWERS_GATE_CMD[@]}"
      print_command judge_preflight_strict_if_present "${JUDGE_GATE_CMD[@]}"
      print_command judge_stage_input "$CONTRACT_PYTHON" "$CONTRACT_HELPER" stage-file \
        --workspace-root "$WORKSPACE_ROOT" --source "$ANSWER_JSONL" \
        --destination "$STAGING_ROOT/.judge.<mktemp>/model_answer/$BENCHMARK/${MODEL_TAG}_answer.jsonl"
      printf 'DRY_RUN[judge_if_not_complete]: cd %q &&' "$STAGING_ROOT/.judge.<mktemp>"
      printf ' %q' "${JUDGE_DRY_CMD[@]}"
      printf '\n'
      print_command judge_publish_noreplace "$CONTRACT_PYTHON" "$CONTRACT_HELPER" publish-file \
        --workspace-root "$WORKSPACE_ROOT" \
        --source "$STAGING_ROOT/.judge.<mktemp>/judge/$BENCHMARK/${MODEL_TAG}_answer.jsonl" \
        --destination "$JUDGE_JSON"
      print_command judge_postflight_strict "${JUDGE_GATE_CMD[@]}"
      ;;
    score)
      print_command preparation_verify_only "${PREPARER_VERIFY_CMD[@]}"
      print_command data_contract "${DATA_CONTRACT_CMD[@]}"
      print_command run_contract "${RUN_CONTRACT_CMD[@]}"
      print_command answers_strict "${ANSWERS_GATE_CMD[@]}"
      print_command judge_strict "${JUDGE_GATE_CMD[@]}"
      print_score_command
      print_command score_publish_noreplace "$CONTRACT_PYTHON" "$CONTRACT_HELPER" publish-file \
        --workspace-root "$WORKSPACE_ROOT" --source "$SCORE_LOG.candidate-run.<mktemp>" \
        --destination "$SCORE_LOG"
      ;;
    judge-score)
      print_command preparation_verify_only "${PREPARER_VERIFY_CMD[@]}"
      print_command data_contract "${DATA_CONTRACT_CMD[@]}"
      print_command run_contract "${RUN_CONTRACT_CMD[@]}"
      print_command answers_strict "${ANSWERS_GATE_CMD[@]}"
      print_command judge_preflight_strict_if_present "${JUDGE_GATE_CMD[@]}"
      print_command judge_stage_input "$CONTRACT_PYTHON" "$CONTRACT_HELPER" stage-file \
        --workspace-root "$WORKSPACE_ROOT" --source "$ANSWER_JSONL" \
        --destination "$STAGING_ROOT/.judge.<mktemp>/model_answer/$BENCHMARK/${MODEL_TAG}_answer.jsonl"
      printf 'DRY_RUN[judge_if_not_complete]: cd %q &&' "$STAGING_ROOT/.judge.<mktemp>"
      printf ' %q' "${JUDGE_DRY_CMD[@]}"
      printf '\n'
      print_command judge_publish_noreplace "$CONTRACT_PYTHON" "$CONTRACT_HELPER" publish-file \
        --workspace-root "$WORKSPACE_ROOT" \
        --source "$STAGING_ROOT/.judge.<mktemp>/judge/$BENCHMARK/${MODEL_TAG}_answer.jsonl" \
        --destination "$JUDGE_JSON"
      print_command judge_postflight_strict "${JUDGE_GATE_CMD[@]}"
      print_score_command
      print_command score_publish_noreplace "$CONTRACT_PYTHON" "$CONTRACT_HELPER" publish-file \
        --workspace-root "$WORKSPACE_ROOT" --source "$SCORE_LOG.candidate-run.<mktemp>" \
        --destination "$SCORE_LOG"
      ;;
  esac
}

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  dry_run_plan
  exit 0
fi

[[ -x /usr/bin/flock ]] || die "/usr/bin/flock is required"
[[ -d "$WORKSPACE_ROOT/Locks" && ! -L "$WORKSPACE_ROOT/Locks" ]] || {
  die "workspace Locks directory is missing or a symlink"
}
if [[ -e "$ARTIFACT_LOCK_PATH" || -L "$ARTIFACT_LOCK_PATH" ]]; then
  [[ -f "$ARTIFACT_LOCK_PATH" && ! -L "$ARTIFACT_LOCK_PATH" ]] || {
    die "artifact lock is a symlink or special file: $ARTIFACT_LOCK_PATH"
  }
fi
exec 9>>"$ARTIFACT_LOCK_PATH"
[[ -f "$ARTIFACT_LOCK_PATH" && ! -L "$ARTIFACT_LOCK_PATH" ]] || {
  die "artifact lock changed identity while opening: $ARTIFACT_LOCK_PATH"
}
if ! /usr/bin/flock -n 9; then
  die "artifact writer is already active for $BENCHMARK/$MODEL_TAG: $ARTIFACT_LOCK_PATH"
fi

"${ARTIFACT_PATH_GATE_CMD[@]}"

freeze_data_and_run() {
  "${PREPARER_VERIFY_CMD[@]}"
  "${DATA_CONTRACT_CMD[@]}"
  "${RUN_CONTRACT_CMD[@]}"
}

strict_answers_gate() {
  "${ANSWERS_GATE_CMD[@]}"
}

strict_judge_gate() {
  "${JUDGE_GATE_CMD[@]}"
}

run_inference_or_revalidate() {
  # A published receipt makes the output immutable.  If either the receipt or
  # its artifact drifted, the strict gate must terminate before upstream can
  # compact/rewrite anything.
  if [[ -e "$ANSWERS_RECEIPT" || -L "$ANSWERS_RECEIPT" ]]; then
    strict_answers_gate
    echo "Complete answer output revalidated without overwrite: $ANSWER_JSONL"
    return
  fi

  # With no published receipt, a complete output is frozen and skipped.  A
  # partial JSONL is allowed to fall through to upstream's resumable writer.
  if [[ -e "$ANSWER_JSONL" || -L "$ANSWER_JSONL" ]]; then
    if strict_answers_gate; then
      echo "Complete answer output frozen and revalidated without overwrite: $ANSWER_JSONL"
      return
    fi
    "${PARTIAL_ANSWERS_CMD[@]}"
    echo "Strict partial answer checkpoint accepted for upstream resume: $ANSWER_JSONL"
  fi
  (
    cd "$RUN_ROOT"
    "${INFER_CMD[@]}"
  )
  "${ARTIFACT_PATH_GATE_CMD[@]}"
  strict_answers_gate
}

run_judge_or_revalidate() {
  strict_answers_gate
  if [[ -e "$JUDGE_RECEIPT" || -L "$JUDGE_RECEIPT" ]]; then
    strict_judge_gate
    echo "Complete judge output revalidated without overwrite: $JUDGE_JSON"
    return
  fi
  if [[ -e "$JUDGE_JSON" || -L "$JUDGE_JSON" ]]; then
    strict_judge_gate
    echo "Complete judge output frozen and revalidated without overwrite: $JUDGE_JSON"
    return
  fi

  local stage_root stage_answer stage_judge stage_receipt
  local -a stage_gate
  stage_root="$(mktemp -d "$STAGING_ROOT/.judge.${BENCHMARK}.${MODEL_TAG}.XXXXXX")"
  stage_answer="$stage_root/model_answer/$BENCHMARK/${MODEL_TAG}_answer.jsonl"
  stage_judge="$stage_root/judge/$BENCHMARK/${MODEL_TAG}_answer.jsonl"
  stage_receipt="$stage_root/staged_judge_receipt.json"
  "$CONTRACT_PYTHON" "$CONTRACT_HELPER" artifact-paths \
    --workspace-root "$WORKSPACE_ROOT" --create \
    --directory "$stage_root" \
    --leaf "$stage_answer" --leaf "$stage_judge" --leaf "$stage_receipt" --quiet
  "$CONTRACT_PYTHON" "$CONTRACT_HELPER" stage-file \
    --workspace-root "$WORKSPACE_ROOT" \
    --source "$ANSWER_JSONL" --destination "$stage_answer" --quiet
  if ! (
    cd "$stage_root"
    env VISION_OPD_REFERENCE_JUDGE_API_KEY="$JUDGE_API_KEY_VALUE" "${JUDGE_CMD[@]}"
  ); then
    echo "Judge failed; staging evidence retained at $stage_root" >&2
    return 1
  fi
  stage_gate=(
    "$CONTRACT_PYTHON" "$CONTRACT_HELPER" judge
    --workspace-root "$WORKSPACE_ROOT"
    --reference-root "$REFERENCE_ROOT"
    --answers-receipt "$ANSWERS_RECEIPT"
    --judge-json "$stage_judge"
    --judge-model-contract "$JUDGE_MODEL_CONTRACT"
    --judge-model-id "$JUDGE_MODEL"
    --judge-api-base "$JUDGE_API_BASE"
    --judge-temperature "$JUDGE_TEMPERATURE"
    --judge-max-tokens "$JUDGE_MAX_TOKENS"
    --receipt "$stage_receipt"
    --quiet
  )
  if [[ -n "$JUDGE_RUNTIME_CONTRACT" ]]; then
    stage_gate+=(--judge-runtime-contract "$JUDGE_RUNTIME_CONTRACT")
  fi
  if ! "${stage_gate[@]}"; then
    echo "Staged judge validation failed; evidence retained at $stage_root" >&2
    return 1
  fi
  if ! "$CONTRACT_PYTHON" "$CONTRACT_HELPER" publish-file \
    --workspace-root "$WORKSPACE_ROOT" \
    --source "$stage_judge" --destination "$JUDGE_JSON" --quiet; then
    echo "Judge publication conflicted; evidence retained at $stage_root" >&2
    return 1
  fi
  strict_judge_gate
  case "$stage_root" in
    "$STAGING_ROOT"/.judge.*)
      [[ -d "$stage_root" && ! -L "$stage_root" ]] || {
        echo "Refusing to clean unsafe successful stage: $stage_root" >&2
        return 1
      }
      rm -rf -- "$stage_root"
      ;;
    *)
      echo "Refusing to clean stage outside staging root: $stage_root" >&2
      return 1
      ;;
  esac
}

run_score() {
  strict_answers_gate
  strict_judge_gate
  local score_stage
  score_stage="$(mktemp "$SCORE_LOG.candidate-run.XXXXXX")"
  if ! (
    cd "$RUN_ROOT"
    "${SCORE_CMD[@]}"
  ) 2>&1 | tee "$score_stage"; then
    echo "Score command failed; staging evidence retained at $score_stage" >&2
    return 1
  fi
  if ! "$CONTRACT_PYTHON" "$CONTRACT_HELPER" publish-file \
    --workspace-root "$WORKSPACE_ROOT" \
    --source "$score_stage" --destination "$SCORE_LOG" --quiet; then
    echo "Score publication conflicted; evidence retained at $score_stage" >&2
    return 1
  fi
  rm -f -- "$score_stage"
}

case "$MODE" in
  preflight)
    freeze_data_and_run
    ;;
  answers)
    freeze_data_and_run
    strict_answers_gate
    ;;
  prepare)
    "${PREPARER_EXECUTE_CMD[@]}"
    "${PREPARER_VERIFY_CMD[@]}"
    "${DATA_CONTRACT_CMD[@]}"
    "${RUN_CONTRACT_CMD[@]}"
    ;;
  infer)
    freeze_data_and_run
    run_inference_or_revalidate
    ;;
  judge)
    freeze_data_and_run
    run_judge_or_revalidate
    ;;
  score)
    freeze_data_and_run
    run_score
    ;;
  judge-score)
    freeze_data_and_run
    run_judge_or_revalidate
    run_score
    ;;
esac
