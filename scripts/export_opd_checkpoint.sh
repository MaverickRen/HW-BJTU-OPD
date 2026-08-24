#!/usr/bin/env bash
# Export the actor from a terminal veRL OPD FSDP checkpoint to Hugging Face.
set -euo pipefail

if [[ $# -ne 4 || "$1" != --checkpoint || "$3" != --output ]]; then
  echo "Usage: PYTHON_BIN=... VERL_ROOT=... $0 --checkpoint ACTOR_DIR --output HF_DIR" >&2
  exit 2
fi
: "${PYTHON_BIN:?set PYTHON_BIN}"
: "${VERL_ROOT:?set VERL_ROOT}"
checkpoint="$2"
output="$4"
[[ -d "$checkpoint" && ! -e "$output" && ! -L "$output" ]] || {
  echo "Checkpoint missing or output already exists" >&2
  exit 2
}
PYTHONPATH="$VERL_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON_BIN" -B -m verl.model_merger merge \
  --backend fsdp --local_dir "$checkpoint" --target_dir "$output"
