#!/usr/bin/env bash
# Thin entrypoint for the resident coordinator.  The Python coordinator owns
# all validation, dry-run purity, GPU lease and serial lifecycle semantics.
# Fixed contract is CUDA_DEVICES="0,1,2,3,4,5,6,7", TP_SIZE=8, DP_SIZE=1;
# candidate max-model-len is unified by the Python manifest (262144).
set -euo pipefail
SCRIPT_DIR="$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
WORKSPACE_ROOT="$(realpath -m -- "$SCRIPT_DIR/../../..")"
PYTHON_BIN="$WORKSPACE_ROOT/UV_Env/verl-opd-qwen35/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then PYTHON_BIN="$(command -v python3 || true)"; fi
[[ -n "$PYTHON_BIN" ]] || { printf 'ERROR: Python is unavailable\n' >&2; exit 2; }
exec "$PYTHON_BIN" -B "$SCRIPT_DIR/run_threebench_resident_formal_v1.py" "$@"
