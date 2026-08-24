#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/eval_env.sh"

REFERENCE_ROOT="$OPD_WORKSPACE/Codes/Vision-OPD-reference"
EXPECTED_REFERENCE_COMMIT=c8a8fdd1f88eef1b5ef4fe6a8d64eb0272917471
ACTUAL_REFERENCE_COMMIT="$(git -C "$REFERENCE_ROOT" rev-parse HEAD)"
if [[ "$ACTUAL_REFERENCE_COMMIT" != "$EXPECTED_REFERENCE_COMMIT" ]]; then
  echo "Vision-OPD reference commit mismatch: expected $EXPECTED_REFERENCE_COMMIT, got $ACTUAL_REFERENCE_COMMIT" >&2
  exit 1
fi

MODE="${ZOOMBENCH_MODE:-all}"
case "$MODE" in
  all|prepare|infer|judge|score|judge-score) ;;
  *) echo "Unknown ZOOMBENCH_MODE=$MODE; expected all, prepare, infer, judge, score, or judge-score" >&2; exit 2 ;;
esac

INFER_MAX_PASSES="${ZOOMBENCH_INFER_MAX_PASSES:-4}"
if [[ ! "$INFER_MAX_PASSES" =~ ^[0-9]+$ ]] \
  || (( INFER_MAX_PASSES < 1 || INFER_MAX_PASSES > 20 )); then
  echo "ZOOMBENCH_INFER_MAX_PASSES must be an integer in [1, 20]; got: $INFER_MAX_PASSES" >&2
  exit 2
fi

DATA_ROOT="${ZOOMBENCH_DATA_ROOT:-$LMUData/ZoomBench}"
BENCHMARK_JSON="$DATA_ROOT/zoombench.json"
RUN_ROOT="${ZOOMBENCH_RUN_ROOT:-$EVAL_WORK_DIR/zoombench}"
MODEL_TAG="${EVAL_MODEL_TAG:-${EVAL_MODEL_ID//\//_}_seed${EVAL_SEED:-42}}"
ANSWER_JSONL="$RUN_ROOT/model_answer/zoombench/${MODEL_TAG}_answer.jsonl"
JUDGE_JSON="$RUN_ROOT/judge/zoombench/${MODEL_TAG}_answer.jsonl"

PREPARE_COMMAND=(
  "$VLMEVAL_PYTHON" "$SCRIPT_DIR/prepare_zoombench.py"
  --dataset-root "$DATA_ROOT"
  --denylist-root "$OPD_WORKSPACE/Dataset/denylist/eval_primary"
)

INFER_COMMAND=(
  "$VLMEVAL_PYTHON" "$REFERENCE_ROOT/eval/infer.py"
  --benchmark zoombench
  --benchmark_json "$BENCHMARK_JSON"
  --out_dir model_answer
  --model_name "$MODEL_TAG"
  --seed "${EVAL_SEED:-42}"
  --api_base "$EVAL_API_BASE"
  --api_key "$EVAL_API_KEY"
  --model_id "$EVAL_MODEL_ID"
  --max_tokens "${ZOOMBENCH_MAX_TOKENS:-256}"
  --max_retries "${EVAL_RETRY:-3}"
  --parallel_workers "${EVAL_API_NPROC:-16}"
  --enable_thinking False
)

VERIFY_COMMAND=(
  "$VLMEVAL_PYTHON" "$SCRIPT_DIR/verify_zoombench_inference.py"
  --benchmark-json "$BENCHMARK_JSON"
  --answer-jsonl "$ANSWER_JSONL"
)

JUDGE_COMMAND=(
  "${ZOOMBENCH_JUDGE_PYTHON:-$VLMEVAL_PYTHON}" "$REFERENCE_ROOT/eval/judge_qwenlm.py"
  --benchmark zoombench
  --model "$MODEL_TAG"
  --judge_max_tokens "${ZOOMBENCH_JUDGE_MAX_TOKENS:-32}"
)
if [[ -n "${ZOOMBENCH_JUDGE_API_BASE:-}" ]]; then
  if [[ -z "${ZOOMBENCH_JUDGE_MODEL:-}" ]]; then
    echo "ZOOMBENCH_JUDGE_MODEL is required with ZOOMBENCH_JUDGE_API_BASE" >&2
    exit 2
  fi
  JUDGE_COMMAND+=(
    --api_base "$ZOOMBENCH_JUDGE_API_BASE"
    --api_key "${ZOOMBENCH_JUDGE_API_KEY:-EMPTY}"
    --judge_model "$ZOOMBENCH_JUDGE_MODEL"
  )
elif [[ -n "${ZOOMBENCH_JUDGE_MODEL_PATH:-}" ]]; then
  JUDGE_COMMAND+=(--judge_model_path "$ZOOMBENCH_JUDGE_MODEL_PATH")
elif [[ "${DRY_RUN:-0}" != 1 && ( "$MODE" == all || "$MODE" == judge || "$MODE" == judge-score ) ]]; then
  echo "ZoomBench follows the audited semantic-judge protocol." >&2
  echo "Set ZOOMBENCH_JUDGE_API_BASE + ZOOMBENCH_JUDGE_MODEL, or ZOOMBENCH_JUDGE_MODEL_PATH." >&2
  exit 2
fi

SCORE_COMMAND=(
  "$VLMEVAL_PYTHON" "$REFERENCE_ROOT/eval/cal_acc.py"
  --benchmark zoombench
  --judge_json "$JUDGE_JSON"
  --benchmark_json "$BENCHMARK_JSON"
)

print_command() {
  printf '%s:' "$1"
  shift
  printf ' %q' "$@"
  printf '\n'
}

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  echo "ZoomBench is a separate benchmark path; it is not VLMEvalKit-native."
  echo "Primary protocol uses only full images. Crop images are oracle annotations and are not sent."
  if [[ "$MODE" == all || "$MODE" == prepare ]]; then
    print_command prepare "${PREPARE_COMMAND[@]}"
  fi
  if [[ "$MODE" == all || "$MODE" == infer ]]; then
    echo "infer-loop: up to $INFER_MAX_PASSES checkpoint-resume passes in the same served-model session"
    print_command verify-before-infer "${VERIFY_COMMAND[@]}"
    print_command infer-pass "${INFER_COMMAND[@]}"
    print_command verify-after-each-pass "${VERIFY_COMMAND[@]}"
  fi
  if [[ "$MODE" == all || "$MODE" == judge || "$MODE" == judge-score ]]; then
    print_command verify-before-judge "${VERIFY_COMMAND[@]}"
    if [[ ${#JUDGE_COMMAND[@]} -gt 8 ]]; then
      print_command judge "${JUDGE_COMMAND[@]}"
    else
      echo "judge: <configure ZOOMBENCH_JUDGE_API_BASE/MODEL or ZOOMBENCH_JUDGE_MODEL_PATH>"
    fi
  fi
  if [[ "$MODE" == all || "$MODE" == score || "$MODE" == judge-score ]]; then
    print_command score "${SCORE_COMMAND[@]}"
  fi
  exit 0
fi

run_verified_inference() {
  local pass verify_rc infer_rc

  if "${VERIFY_COMMAND[@]}"; then
    echo "ZoomBench checkpoint is already strictly complete; inference is not repeated."
    return 0
  else
    verify_rc=$?
  fi
  if (( verify_rc != 1 )); then
    echo "ZoomBench verifier could not establish the benchmark contract; refusing inference." >&2
    return "$verify_rc"
  fi

  for ((pass = 1; pass <= INFER_MAX_PASSES; pass++)); do
    echo "ZoomBench inference pass $pass/$INFER_MAX_PASSES"
    if "${INFER_COMMAND[@]}"; then
      infer_rc=0
    else
      infer_rc=$?
      echo "ZoomBench inference command exited nonzero on pass $pass (rc=$infer_rc); verifying its checkpoint before deciding whether to resume." >&2
    fi

    if "${VERIFY_COMMAND[@]}"; then
      if (( infer_rc == 0 )); then
        echo "ZoomBench inference passed the strict 845-row gate on pass $pass."
      else
        echo "ZoomBench checkpoint passed the strict 845-row gate despite inference command exit code $infer_rc on pass $pass."
      fi
      return 0
    else
      verify_rc=$?
    fi
    if (( verify_rc != 1 )); then
      echo "ZoomBench verifier failed on pass $pass with exit code $verify_rc." >&2
      return "$verify_rc"
    fi
    if (( pass < INFER_MAX_PASSES )); then
      if (( infer_rc == 0 )); then
        echo "ZoomBench checkpoint remains incomplete; retrying only missing/error rows via upstream resume."
      else
        echo "ZoomBench checkpoint remains incomplete after inference rc=$infer_rc; continuing bounded checkpoint resume."
      fi
    fi
  done

  echo "ZoomBench checkpoint failed the strict gate after $INFER_MAX_PASSES inference passes." >&2
  return 1
}

mkdir -p "$RUN_ROOT"
cd "$RUN_ROOT"

if [[ "$MODE" == all || "$MODE" == prepare ]]; then
  "${PREPARE_COMMAND[@]}"
fi
if [[ "$MODE" == all || "$MODE" == infer ]]; then
  [[ -f "$BENCHMARK_JSON" ]] || { echo "Missing $BENCHMARK_JSON; run prepare first" >&2; exit 1; }
  run_verified_inference
fi
if [[ "$MODE" == all || "$MODE" == judge || "$MODE" == judge-score ]]; then
  [[ -f "$ANSWER_JSONL" ]] || { echo "Missing $ANSWER_JSONL; run inference first" >&2; exit 1; }
  echo "Verifying strict ZoomBench inference completeness before judge."
  "${VERIFY_COMMAND[@]}"
  "${JUDGE_COMMAND[@]}"
fi
if [[ "$MODE" == all || "$MODE" == score || "$MODE" == judge-score ]]; then
  [[ -f "$JUDGE_JSON" ]] || { echo "Missing $JUDGE_JSON; run judge first" >&2; exit 1; }
  "${SCORE_COMMAND[@]}"
fi
