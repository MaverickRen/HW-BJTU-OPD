#!/usr/bin/env bash
# Formal full-image ZoomBench lifecycle: candidate first, fixed local judge
# second, one GPU0-7 lease, and aggregate-only public stdout.
set -euo pipefail
umask 077

SCRIPT_DIR="$(cd -P -- "$(dirname -- "$0")" && pwd -P)"
WORKSPACE_ROOT="$(realpath -m -- "$SCRIPT_DIR/../../..")"
OUTPUT_ROOT="$WORKSPACE_ROOT/Output"
PYTHON_BIN="$WORKSPACE_ROOT/UV_Env/verl-opd-qwen35/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then PYTHON_BIN="$(command -v python3 || true)"; fi
RUN_ONE_MODEL="$SCRIPT_DIR/run_one_model_eval.sh"
RUN_ZOOMBENCH="$SCRIPT_DIR/run_zoombench.sh"
VERIFY_ZOOMBENCH="$SCRIPT_DIR/verify_zoombench_inference.py"
JUDGE_MATRIX="$SCRIPT_DIR/run_zoombench_judge_matrix.sh"
EVAL_ENV="$SCRIPT_DIR/eval_env.sh"
SERVE_SCRIPT="$SCRIPT_DIR/serve_qwen35.sh"
PREPARE_ZOOMBENCH="$SCRIPT_DIR/prepare_zoombench.py"
MANIFEST_CONTRACT="$SCRIPT_DIR/manifests/zoombench.json"
REFERENCE_ROOT="$WORKSPACE_ROOT/Codes/Vision-OPD-reference"
REFERENCE_INFER="$REFERENCE_ROOT/eval/infer.py"
REFERENCE_JUDGE="$REFERENCE_ROOT/eval/judge_qwenlm.py"
REFERENCE_CAL_ACC="$REFERENCE_ROOT/eval/cal_acc.py"
AGGREGATOR="$SCRIPT_DIR/zoombench_formal_aggregate_v1.py"
ORCHESTRATOR="$SCRIPT_DIR/run_zoombench_formal_aggregate_v1.sh"
MANIFEST="$WORKSPACE_ROOT/Dataset/eval/ZoomBench/manifest.json"
BENCHMARK_JSON="$WORKSPACE_ROOT/Dataset/eval/ZoomBench/zoombench.json"
LOCK_PATH="$WORKSPACE_ROOT/Locks/opd_gpu_0_7.lock"

DATASET_ID="inclusionAI/ZoomBench"
DATASET_REVISION="b788097e57d30510c6877824833234a73bf80d25"
OFFICIAL_EVAL_COMMIT="fdc0ba1a3dee916d8c38304d543ad414879e0c99"
REFERENCE_COMMIT="c8a8fdd1f88eef1b5ef4fe6a8d64eb0272917471"
EXPECTED_ROWS=845
MANIFEST_SHA256="7c01d8b6db61834f3e8e550e0bae1a3475e555e6e54959087b72dae3d14ddc72"
SOURCE_PARQUET_SHA256="d44ebda2eda485cba055181f4e6dc50c42f81b5d0f7e936bf427fa01502a391a"
CUDA_DEVICES="0,1,2,3,4,5,6,7"
CANDIDATE_TP=8
CANDIDATE_DP=1
JUDGE_TP=2
JUDGE_DP=4
CANDIDATE_MAX_TOKENS=256
JUDGE_MAX_TOKENS=32
TRUST_REMOTE_CODE=true
CANDIDATE_PORT=18318
JUDGE_PORT=18319

RUN_ONE_MODEL_SHA256="5d7c4867ca0d8abd0a9108400b036fe423978ee1ab0830cc5f59caa5c9d936d9"
RUN_ZOOMBENCH_SHA256="c103c1355b47e8a81b378b09164c8342a1b1a4ff41793dbe215e73efe9555ff0"
VERIFY_ZOOMBENCH_SHA256="66b61ca08a383bec88feef28cfa270141dc4e2a6807abd31bd5e81b144c1033c"
JUDGE_MATRIX_SHA256="dd3892e33d8f3e92dc41fa5a7388574098e1a288d965f62b7d0eefecdc2d979d"
EVAL_ENV_SHA256="1f6b5b0e502e0a68d19028e6c4c20ffa870430b1e8b2aa41cee2a5126659baf5"
SERVE_SCRIPT_SHA256="e685b999ab846de7967dacd08a3a7124184b0212b7610da4aac4b48ce24effea"
PREPARE_ZOOMBENCH_SHA256="f60e4fc6255c6d4083933f706de835c1fd30c2d87e16502945af9d8625481ef8"
MANIFEST_CONTRACT_SHA256="e730048f8d0ebc16e8698779d2271a14ff7d09398419018da9f650a7a98f37e7"
REFERENCE_INFER_SHA256="bb379999932658907196cdc98d22c60d63e3308cb5a867317481c4a85af70374"
REFERENCE_JUDGE_SHA256="abbe11dacf7fae19728ca16407a02c91d04a9bc8ea72edd3b4a91b6224f4b670"
REFERENCE_CAL_ACC_SHA256="695dbddc3e63a1b9f8971c0d414d963a5da94776863d58589feaa4a1c6b0f025"
AGGREGATOR_SHA256="4b5fd3683cc42cfd2818bd5b47fda710c6f33072d0218c50cf5f3051c04b142f"

MODEL_PATH=""
MODEL_ID=""
MODEL_TAG=""
JUDGE_MODEL_PATH="$WORKSPACE_ROOT/Ckpt/Qwen3.5-27B"
JUDGE_MODEL_ID="Qwen3.5-27B-ZoomJudge"
JUDGE_MODEL_TAG=""
RUN_ROOT=""
AUDIT_ROOT=""
MODE=""

die() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }
need_value() { [[ $# -ge 2 && -n "$2" ]] || die "missing value for $1"; }
usage() {
  cat <<'EOF'
Usage: run_zoombench_formal_aggregate_v1.sh --model-path PATH --model-id ID
  --run-root PATH --audit-root PATH [--model-tag ID_seed42]
  [--judge-model-path PATH] [--judge-model-id ID]
  [--candidate-port N] [--judge-port N] [--dry-run|--execute|--resume-seal]

The candidate uses run_one_model_eval.sh --profiles zoom-infer, then the
fixed run_zoombench_judge_matrix.sh. GPUs are physical 0-7 only and phases
are serial. Execute is create-once and requires UID/GID 30853. Dry-run is
side-effect free; execute stdout is one aggregate-only line. Resume-seal
reuses an existing complete candidate/judge lifecycle and performs only the
CPU aggregate/seal step; it never acquires the GPU lease or starts inference.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-path) need_value "$@"; MODEL_PATH="$2"; shift 2 ;;
    --model-id) need_value "$@"; MODEL_ID="$2"; shift 2 ;;
    --model-tag) need_value "$@"; MODEL_TAG="$2"; shift 2 ;;
    --judge-model-path) need_value "$@"; JUDGE_MODEL_PATH="$2"; shift 2 ;;
    --judge-model-id) need_value "$@"; JUDGE_MODEL_ID="$2"; shift 2 ;;
    --judge-model-tag) need_value "$@"; JUDGE_MODEL_TAG="$2"; shift 2 ;;
    --run-root) need_value "$@"; RUN_ROOT="$2"; shift 2 ;;
    --audit-root) need_value "$@"; AUDIT_ROOT="$2"; shift 2 ;;
    --candidate-port) need_value "$@"; CANDIDATE_PORT="$2"; shift 2 ;;
    --judge-port) need_value "$@"; JUDGE_PORT="$2"; shift 2 ;;
    --dry-run)
      [[ -z "$MODE" ]] || die "choose exactly one of --dry-run or --execute"
      MODE=dry-run; shift ;;
    --execute)
      [[ -z "$MODE" ]] || die "choose exactly one of --dry-run or --execute"
      MODE=execute; shift ;;
    --resume-seal)
      [[ -z "$MODE" ]] || die "choose exactly one of --dry-run, --execute, or --resume-seal"
      MODE=resume-seal; shift ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; die "unknown option: $1" ;;
  esac
done
if [[ -z "$MODE" ]]; then
  case "${DRY_RUN:-1}" in 1) MODE=dry-run ;; 0) MODE=execute ;; *) die "DRY_RUN must be 0 or 1" ;; esac
fi
[[ "$MODE" == dry-run || "$MODE" == execute || "$MODE" == resume-seal ]] || die "invalid mode"
[[ -n "$MODEL_PATH" && -n "$MODEL_ID" ]] || die "model path and model id are required"
[[ -n "$RUN_ROOT" && -n "$AUDIT_ROOT" ]] || die "run-root and audit-root are required"
[[ "$MODEL_ID" =~ ^[A-Za-z0-9._/+:-]+$ ]] || die "unsafe candidate model id"
[[ "$JUDGE_MODEL_ID" =~ ^[A-Za-z0-9._/+:-]+$ ]] || die "unsafe judge model id"
if [[ -n "$JUDGE_MODEL_TAG" ]]; then
  [[ "$JUDGE_MODEL_TAG" =~ ^[A-Za-z0-9._-]+$ ]] || die "unsafe judge model tag"
fi

MODEL_PATH="$(realpath -m -- "$MODEL_PATH")"
JUDGE_MODEL_PATH="$(realpath -m -- "$JUDGE_MODEL_PATH")"
RUN_ROOT="$(realpath -ms -- "$RUN_ROOT")"
AUDIT_ROOT="$(realpath -ms -- "$AUDIT_ROOT")"
DERIVED_MODEL_TAG="$(printf '%s' "$MODEL_ID" | tr '/' '_')_seed42"
if [[ -z "$MODEL_TAG" ]]; then MODEL_TAG="$DERIVED_MODEL_TAG"; fi
[[ "$MODEL_TAG" == "$DERIVED_MODEL_TAG" ]] || die "model-tag must equal $DERIVED_MODEL_TAG"
[[ "$MODEL_TAG" =~ ^[A-Za-z0-9._-]+_seed42$ ]] || die "unsafe model tag"
require_port() { [[ "$2" =~ ^[0-9]+$ && "$2" -ge 1024 && "$2" -le 65535 ]] || die "$1 is invalid"; }
require_port candidate-port "$CANDIDATE_PORT"
require_port judge-port "$JUDGE_PORT"
(( CANDIDATE_PORT != JUDGE_PORT )) || die "ports must differ"

case "$MODEL_PATH/" in "$WORKSPACE_ROOT"/*) ;; *) die "model path must be below H_Workspace" ;; esac
case "$JUDGE_MODEL_PATH/" in "$WORKSPACE_ROOT"/*) ;; *) die "judge path must be below H_Workspace" ;; esac
case "$RUN_ROOT/" in "$OUTPUT_ROOT"/*) ;; *) die "run-root must be below H_Workspace/Output" ;; esac
case "$AUDIT_ROOT/" in "$OUTPUT_ROOT"/*) ;; *) die "audit-root must be below H_Workspace/Output" ;; esac
[[ "$RUN_ROOT" != "$OUTPUT_ROOT" && "$AUDIT_ROOT" != "$OUTPUT_ROOT" ]] || die "roots cannot be Output itself"
[[ "$RUN_ROOT" != "$AUDIT_ROOT" ]] || die "roots must differ"
[[ -d "$MODEL_PATH" && -f "$MODEL_PATH/config.json" ]] || die "candidate checkpoint is invalid"
[[ -d "$JUDGE_MODEL_PATH" && -f "$JUDGE_MODEL_PATH/config.json" ]] || die "judge checkpoint is invalid"
[[ -x "$PYTHON_BIN" ]] || die "formal Python runtime is unavailable"
command -v sha256sum >/dev/null || die "sha256sum is required"

for required_file in "$RUN_ONE_MODEL" "$RUN_ZOOMBENCH" "$VERIFY_ZOOMBENCH" "$JUDGE_MATRIX" "$EVAL_ENV" \
  "$SERVE_SCRIPT" "$PREPARE_ZOOMBENCH" "$MANIFEST_CONTRACT" "$REFERENCE_INFER" "$REFERENCE_JUDGE" \
  "$REFERENCE_CAL_ACC" "$AGGREGATOR"; do
  [[ -f "$required_file" && ! -L "$required_file" ]] || die "required source/manifest unavailable"
done
verify_sha256() {
  local path="$1" expected="$2" actual
  actual="$(sha256sum -- "$path" | awk '{print $1}')"
  [[ "$actual" == "$expected" ]] || die "source hash mismatch"
}
verify_sha256 "$RUN_ONE_MODEL" "$RUN_ONE_MODEL_SHA256"
verify_sha256 "$RUN_ZOOMBENCH" "$RUN_ZOOMBENCH_SHA256"
verify_sha256 "$VERIFY_ZOOMBENCH" "$VERIFY_ZOOMBENCH_SHA256"
verify_sha256 "$JUDGE_MATRIX" "$JUDGE_MATRIX_SHA256"
verify_sha256 "$EVAL_ENV" "$EVAL_ENV_SHA256"
verify_sha256 "$SERVE_SCRIPT" "$SERVE_SCRIPT_SHA256"
verify_sha256 "$PREPARE_ZOOMBENCH" "$PREPARE_ZOOMBENCH_SHA256"
verify_sha256 "$MANIFEST_CONTRACT" "$MANIFEST_CONTRACT_SHA256"
verify_sha256 "$REFERENCE_INFER" "$REFERENCE_INFER_SHA256"
verify_sha256 "$REFERENCE_JUDGE" "$REFERENCE_JUDGE_SHA256"
verify_sha256 "$REFERENCE_CAL_ACC" "$REFERENCE_CAL_ACC_SHA256"
[[ "$AGGREGATOR_SHA256" != __AGGREGATOR_SHA256__ ]] || die "aggregator source pin is unset"
verify_sha256 "$AGGREGATOR" "$AGGREGATOR_SHA256"
ORCHESTRATOR_SHA256="$(sha256sum -- "$ORCHESTRATOR" | awk '{print $1}')"

BENCHMARK_JSON_SHA256="deferred_until_execute"

MANIFEST_PREFIX="$(printf '%s' "$MANIFEST_SHA256" | cut -c1-12)"
CACHE_KEY="zoombench-dry-run-manifest$MANIFEST_PREFIX-runtime-v1"
CANDIDATE_JSON="$RUN_ROOT/zoombench/model_answer/zoombench/"$MODEL_TAG"_answer.jsonl"
JUDGE_JSON="$RUN_ROOT/zoombench/judge/zoombench/${MODEL_TAG}_answer.jsonl"
MATRIX_PROTOCOL="$RUN_ROOT/zoombench/judge_protocol.json"
AGGREGATE="$RUN_ROOT/zoombench_formal_aggregate.json"

if [[ "$MODE" == dry-run ]]; then
  printf '%s\n' \
    "DRY_RUN[protocol]: benchmark=ZoomBench rows=$EXPECTED_ROWS full_image_only=true dataset=$DATASET_ID revision=$DATASET_REVISION benchmark_json_sha256=deferred_until_execute candidate_seed=42 candidate_enable_thinking=false candidate_temperature=0 candidate_max_tokens=$CANDIDATE_MAX_TOKENS judge_temperature=0 judge_enable_thinking=false judge_max_tokens=$JUDGE_MAX_TOKENS trust_remote_code=true" \
    "DRY_RUN[hardware]: physical_cuda=$CUDA_DEVICES candidate_tp=$CANDIDATE_TP candidate_dp=$CANDIDATE_DP judge_tp=$JUDGE_TP judge_dp=$JUDGE_DP serial=true lock=$LOCK_PATH" \
    "DRY_RUN[composition]: candidate=$RUN_ONE_MODEL --profiles zoom-infer -> $RUN_ZOOMBENCH; judge=$JUDGE_MATRIX" \
    "DRY_RUN[paths]: fresh_run_root=$RUN_ROOT fresh_audit_root=$AUDIT_ROOT candidate_answer=$RUN_ROOT/zoombench/model_answer/zoombench/${MODEL_TAG}_answer.jsonl judge_json=$JUDGE_JSON matrix_protocol=$MATRIX_PROTOCOL aggregate_authority=$AGGREGATE" \
    "DRY_RUN[hashes]: orchestrator=$ORCHESTRATOR_SHA256 eval_env=$EVAL_ENV_SHA256 run_one=$RUN_ONE_MODEL_SHA256 zoombench=$RUN_ZOOMBENCH_SHA256 verifier=$VERIFY_ZOOMBENCH_SHA256 judge_matrix=$JUDGE_MATRIX_SHA256 infer=$REFERENCE_INFER_SHA256 judge=$REFERENCE_JUDGE_SHA256 cal_acc=$REFERENCE_CAL_ACC_SHA256 aggregator=$AGGREGATOR_SHA256 manifest=$MANIFEST_SHA256" \
    "DRY_RUN[caveat]: fixed local Qwen3.5 judge shares Qwen-family/style bias; report it as non-independent" \
    "DRY_RUN[effects]: no directory, lock, process/GPU query, server, HTTP request, inference, judging, scoring, or sample-level output"
  exit 0
fi
if [[ "$MODE" == resume-seal ]]; then
  [[ "$(id -u)" == 30853 && "$(id -g)" == 30853 ]] || die "formal resume requires UID/GID 30853"
  [[ -d "$RUN_ROOT" && ! -L "$RUN_ROOT" ]] || die "existing run-root is unavailable"
  [[ -d "$AUDIT_ROOT" && ! -L "$AUDIT_ROOT" ]] || die "existing audit-root is unavailable"
  [[ ! -e "$AGGREGATE" && ! -L "$AGGREGATE" ]] || die "aggregate authority already exists (create-once)"
  [[ -f "$CANDIDATE_JSON" && ! -L "$CANDIDATE_JSON" ]] || die "candidate answer checkpoint is unavailable"
  [[ -f "$JUDGE_JSON" && ! -L "$JUDGE_JSON" ]] || die "judge result is unavailable"
  [[ -f "$MATRIX_PROTOCOL" && ! -L "$MATRIX_PROTOCOL" ]] || die "judge protocol is unavailable"

  status_complete() {
    local status_path="$1"
    [[ -f "$status_path" && ! -L "$status_path" ]] || return 1
    awk -F= '
      $1 == "state" { count += 1; value = $2 }
      END { exit !(count == 1 && value == "complete") }
    ' "$status_path"
  }
  [[ -d "$RUN_ROOT/_runner" && ! -L "$RUN_ROOT/_runner" ]] || die "candidate lifecycle metadata is unavailable"
  CANDIDATE_STATUS_PATH="$(find -P "$RUN_ROOT/_runner" -mindepth 2 -maxdepth 2 -type f -name status.env -print)"
  [[ "$(find -P "$RUN_ROOT/_runner" -mindepth 2 -maxdepth 2 -type f -name status.env -print | wc -l)" -eq 1 ]] || die "candidate lifecycle must contain one status.env"
  status_complete "$CANDIDATE_STATUS_PATH" || die "candidate lifecycle is not complete"
  JUDGE_STATUS_PATH="$(find -P "$AUDIT_ROOT" -mindepth 2 -maxdepth 2 -type f -name status.env -print)"
  [[ "$(find -P "$AUDIT_ROOT" -mindepth 2 -maxdepth 2 -type f -name status.env -print | wc -l)" -eq 1 ]] || die "judge lifecycle must contain one status.env"
  status_complete "$JUDGE_STATUS_PATH" || die "judge lifecycle is not complete"

  PRIVATE_RESUME_CANDIDATE_IDENTITY="$(mktemp /tmp/opd-zoombench-resume-candidate-identity.XXXXXX)"
  PRIVATE_RESUME_JUDGE_IDENTITY="$(mktemp /tmp/opd-zoombench-resume-judge-identity.XXXXXX)"
  PRIVATE_RESUME_AGGREGATE_LOG="$(mktemp /tmp/opd-zoombench-resume-aggregate.XXXXXX)"
  chmod 0600 "$PRIVATE_RESUME_CANDIDATE_IDENTITY" "$PRIVATE_RESUME_JUDGE_IDENTITY" "$PRIVATE_RESUME_AGGREGATE_LOG"
  cleanup_resume() {
    local rc=$?
    rm -f -- "$PRIVATE_RESUME_CANDIDATE_IDENTITY" "$PRIVATE_RESUME_JUDGE_IDENTITY" \
      "$PRIVATE_RESUME_AGGREGATE_LOG" 2>/dev/null || true
    exit "$rc"
  }
  trap cleanup_resume EXIT INT TERM HUP

  if ! "$PYTHON_BIN" -B "$AGGREGATOR" --checkpoint-identity "$MODEL_PATH" \
    >"$PRIVATE_RESUME_CANDIDATE_IDENTITY" 2>/dev/null; then
    printf 'ERROR: candidate checkpoint identity preflight failed\n' >&2
    exit 1
  fi
  RESUME_CANDIDATE_DIGEST="$("$PYTHON_BIN" -B - "$PRIVATE_RESUME_CANDIDATE_IDENTITY" <<'PY'
import json, pathlib, sys
value = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
print(value["identity_sha256"])
PY
  )"
  [[ "$RESUME_CANDIDATE_DIGEST" =~ ^[0-9a-f]{64}$ ]] || die "candidate checkpoint identity digest is malformed"
  if ! "$PYTHON_BIN" -B "$AGGREGATOR" --checkpoint-identity "$JUDGE_MODEL_PATH" \
    >"$PRIVATE_RESUME_JUDGE_IDENTITY" 2>/dev/null; then
    printf 'ERROR: judge checkpoint identity preflight failed\n' >&2
    exit 1
  fi
  RESUME_JUDGE_DIGEST="$("$PYTHON_BIN" -B - "$PRIVATE_RESUME_JUDGE_IDENTITY" <<'PY'
import json, pathlib, sys
value = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
print(value["identity_sha256"])
PY
  )"
  [[ "$RESUME_JUDGE_DIGEST" =~ ^[0-9a-f]{64}$ ]] || die "judge checkpoint identity digest is malformed"
  CACHE_KEY="zoombench-c$(printf '%s' "$RESUME_CANDIDATE_DIGEST" | cut -c1-24)-j$(printf '%s' "$RESUME_JUDGE_DIGEST" | cut -c1-24)-manifest$MANIFEST_PREFIX-runtime-v1"

  if ! aggregate_output="$("$PYTHON_BIN" -B "$AGGREGATOR" \
    --resume-seal --judge-json "$JUDGE_JSON" --model-id "$MODEL_ID" --model-tag "$MODEL_TAG" \
    --model-path "$MODEL_PATH" --judge-model-path "$JUDGE_MODEL_PATH" --judge-model-id "$JUDGE_MODEL_ID" \
    --expected-candidate-identity "$PRIVATE_RESUME_CANDIDATE_IDENTITY" \
    --expected-judge-identity "$PRIVATE_RESUME_JUDGE_IDENTITY" \
    --aggregator-sha256 "$AGGREGATOR_SHA256" --judge-port "$JUDGE_PORT" --cache-key "$CACHE_KEY" \
    --orchestrator-sha256 "$ORCHESTRATOR_SHA256" --run-root "$RUN_ROOT" --audit-root "$AUDIT_ROOT" \
    --matrix-protocol "$MATRIX_PROTOCOL" --dataset-manifest "$MANIFEST" --output "$AGGREGATE" \
    2>"$PRIVATE_RESUME_AGGREGATE_LOG")"; then
    printf 'ERROR: CPU aggregate resume/seal failed\n' >&2
    sed -n '1p' "$PRIVATE_RESUME_AGGREGATE_LOG" >&2 || true
    exit 1
  fi
  printf '%s\n' "$aggregate_output"
  exit 0
fi
[[ "$MODE" == execute ]] || die "invalid mode"
[[ "$(id -u)" == 30853 && "$(id -g)" == 30853 ]] || die "formal execution requires UID/GID 30853"
# Only execute may inspect or hash the materialized manifest/benchmark JSON.
for required_file in "$MANIFEST" "$BENCHMARK_JSON"; do
  [[ -f "$required_file" && ! -L "$required_file" ]] || die "required manifest/benchmark JSON unavailable"
done
verify_sha256 "$MANIFEST" "$MANIFEST_SHA256"
BENCHMARK_JSON_SHA256="$(sha256sum -- "$BENCHMARK_JSON" | awk '{print $1}')"
"$PYTHON_BIN" - "$MANIFEST" "$BENCHMARK_JSON" "$BENCHMARK_JSON_SHA256" \
  "$DATASET_ID" "$DATASET_REVISION" "$REFERENCE_COMMIT" "$OFFICIAL_EVAL_COMMIT" \
  "$EXPECTED_ROWS" "$SOURCE_PARQUET_SHA256" <<'PY'
import json, pathlib, sys
manifest_path, benchmark_json, benchmark_json_sha, dataset_id, revision, reference_commit, official_commit, rows, parquet_sha = sys.argv[1:]
value = json.loads(pathlib.Path(manifest_path).read_text(encoding="utf-8"))
expected_benchmark_json = pathlib.Path(benchmark_json).resolve()
manifest_benchmark_json = pathlib.Path(str(value.get("benchmark_json", ""))).resolve()
ok = (
    manifest_benchmark_json == expected_benchmark_json
    and value.get("benchmark_json_sha256") == benchmark_json_sha
    and value.get("dataset_id") == dataset_id
    and value.get("dataset_revision") == revision
    and value.get("rows") == int(rows)
    and value.get("materialization_complete") is True
    and value.get("source_parquet_sha256") == parquet_sha
    and value.get("vision_opd_reference_commit") == reference_commit
    and value.get("official_eval_commit_audited") == official_commit
    and value.get("primary_protocol", "").startswith("full image only")
)
raise SystemExit(0 if ok else 2)
PY
[[ -f "$LOCK_PATH" && ! -L "$LOCK_PATH" ]] || die "GPU lease file is unavailable"
[[ ! -e "$RUN_ROOT" && ! -L "$RUN_ROOT" ]] || die "run-root already exists (create-once)"
[[ ! -e "$AUDIT_ROOT" && ! -L "$AUDIT_ROOT" ]] || die "audit-root already exists (create-once)"

exec 9<"$LOCK_PATH"
flock -n 9 || die "GPU0-7 lease is busy"
PRIVATE_CANDIDATE_LOG="$(mktemp /tmp/opd-zoombench-candidate.XXXXXX)"
PRIVATE_JUDGE_LOG="$(mktemp /tmp/opd-zoombench-judge.XXXXXX)"
PRIVATE_AGGREGATE_LOG="$(mktemp /tmp/opd-zoombench-aggregate.XXXXXX)"
PRIVATE_CANDIDATE_IDENTITY="$(mktemp /tmp/opd-zoombench-candidate-identity.XXXXXX)"
PRIVATE_JUDGE_IDENTITY="$(mktemp /tmp/opd-zoombench-judge-identity.XXXXXX)"
cleanup_private() {
  local rc=$?
  rm -f -- "$PRIVATE_CANDIDATE_LOG" "$PRIVATE_JUDGE_LOG" "$PRIVATE_AGGREGATE_LOG" \
    "$PRIVATE_CANDIDATE_IDENTITY" "$PRIVATE_JUDGE_IDENTITY" 2>/dev/null || true
  exit "$rc"
}
trap cleanup_private EXIT INT TERM HUP

if ! "$PYTHON_BIN" -B "$AGGREGATOR" --checkpoint-identity "$MODEL_PATH" \
  >"$PRIVATE_CANDIDATE_IDENTITY" 2>/dev/null; then
  printf 'ERROR: candidate checkpoint identity preflight failed\n' >&2
  exit 1
fi
mapfile -t IDENTITY_DIGESTS < <("$PYTHON_BIN" -B - "$PRIVATE_CANDIDATE_IDENTITY" <<'PY'
import json, pathlib, sys
value = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
print(value["identity_sha256"])
PY
)
[[ ${#IDENTITY_DIGESTS[@]} -eq 1 && ${#IDENTITY_DIGESTS[0]} -eq 64 ]] || {
  printf 'ERROR: candidate checkpoint identity digest is malformed\n' >&2
  exit 1
}
if ! env DRY_RUN=0 EVAL_MODEL_TAG="$MODEL_TAG" EVAL_SEED=42 \
  ZOOMBENCH_MAX_TOKENS="$CANDIDATE_MAX_TOKENS" EVAL_WORK_DIR="$RUN_ROOT" \
  "$RUN_ONE_MODEL" --model-path "$MODEL_PATH" --model-id "$MODEL_ID" --work-dir "$RUN_ROOT" \
  --cuda-devices "$CUDA_DEVICES" --tp "$CANDIDATE_TP" --dp "$CANDIDATE_DP" \
  --port "$CANDIDATE_PORT" --profiles zoom-infer >"$PRIVATE_CANDIDATE_LOG" 2>&1; then
  printf 'ERROR: candidate ZoomBench lifecycle failed; private audit retained under run root\n' >&2
  exit 1
fi
[[ -f "$CANDIDATE_JSON" && ! -L "$CANDIDATE_JSON" ]] || die "candidate produced no checkpoint"

if ! "$PYTHON_BIN" -B "$AGGREGATOR" --checkpoint-identity "$JUDGE_MODEL_PATH" \
  >"$PRIVATE_JUDGE_IDENTITY" 2>/dev/null; then
  printf 'ERROR: judge checkpoint identity preflight failed\n' >&2
  exit 1
fi
mapfile -t JUDGE_IDENTITY_DIGESTS < <("$PYTHON_BIN" -B - "$PRIVATE_JUDGE_IDENTITY" <<'PY'
import json, pathlib, sys
value = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
print(value["identity_sha256"])
PY
)
[[ ${#JUDGE_IDENTITY_DIGESTS[@]} -eq 1 && ${#JUDGE_IDENTITY_DIGESTS[0]} -eq 64 ]] || {
  printf 'ERROR: judge checkpoint identity digest is malformed\n' >&2
  exit 1
}
CACHE_KEY="zoombench-c${IDENTITY_DIGESTS[0]:0:24}-j${JUDGE_IDENTITY_DIGESTS[0]:0:24}-manifest$MANIFEST_PREFIX-runtime-v1"

if ! env DRY_RUN=0 EVAL_SEED=42 ZOOMBENCH_JUDGE_MAX_TOKENS="$JUDGE_MAX_TOKENS" \
  TRUST_REMOTE_CODE="$TRUST_REMOTE_CODE" "$JUDGE_MATRIX" \
  --judge-model-path "$JUDGE_MODEL_PATH" --judge-model-id "$JUDGE_MODEL_ID" \
  --cuda-devices "$CUDA_DEVICES" --tp "$JUDGE_TP" --dp "$JUDGE_DP" --port "$JUDGE_PORT" \
  --audit-root "$AUDIT_ROOT" --target "candidate|$MODEL_ID|$RUN_ROOT" \
  >"$PRIVATE_JUDGE_LOG" 2>&1; then
  printf 'ERROR: ZoomBench judge lifecycle failed; private audit retained under audit root\n' >&2
  exit 1
fi
[[ -f "$JUDGE_JSON" && -f "$MATRIX_PROTOCOL" ]] || die "judge produced no complete metadata"

# Bind the formal Zoom judge controls into the pinned matrix protocol before
# the child receipt consumes it.  The explicit env value is therefore
# auditable even though the generic matrix writer does not know this contract.
"$PYTHON_BIN" - "$MATRIX_PROTOCOL" "$JUDGE_MAX_TOKENS" <<'PY'
import json, os, pathlib, sys
path = pathlib.Path(sys.argv[1])
value = json.loads(path.read_text(encoding="utf-8"))
if not isinstance(value, dict):
    raise SystemExit("judge protocol is not an object")
value["judge_max_tokens"] = int(sys.argv[2])
value["trust_remote_code"] = True
temporary = path.with_suffix(path.suffix + ".tmp")
temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, path)
PY

if ! aggregate_output="$("$PYTHON_BIN" -B "$AGGREGATOR" \
  --judge-json "$JUDGE_JSON" --model-id "$MODEL_ID" --model-tag "$MODEL_TAG" \
  --model-path "$MODEL_PATH" \
  --judge-model-path "$JUDGE_MODEL_PATH" --judge-model-id "$JUDGE_MODEL_ID" \
  --expected-candidate-identity "$PRIVATE_CANDIDATE_IDENTITY" \
  --expected-judge-identity "$PRIVATE_JUDGE_IDENTITY" \
  --aggregator-sha256 "$AGGREGATOR_SHA256" \
  --judge-port "$JUDGE_PORT" --cache-key "$CACHE_KEY" \
  --orchestrator-sha256 "$ORCHESTRATOR_SHA256" \
  --run-root "$RUN_ROOT" --audit-root "$AUDIT_ROOT" --matrix-protocol "$MATRIX_PROTOCOL" \
  --dataset-manifest "$MANIFEST" --output "$AGGREGATE" 2>"$PRIVATE_AGGREGATE_LOG")"; then
  printf 'ERROR: aggregate authority failed; private audit retained under run root\n' >&2
  exit 1
fi
printf '%s\n' "$aggregate_output"
