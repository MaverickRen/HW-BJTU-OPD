#!/usr/bin/env bash
# Install the exact public inference environment, including its audited
# vLLM/Transformers metadata override.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python}"
runtime_user="${USER:-${LOGNAME:-opd-eval}}"
export USER="$runtime_user" LOGNAME="$runtime_user"

command -v "$python_bin" >/dev/null || {
  echo "Python is required" >&2
  exit 2
}

"$python_bin" - <<'PY'
import sys

if sys.prefix == sys.base_prefix:
    raise SystemExit("activate a Python virtual environment before running scripts/install_eval.sh")
PY

"$python_bin" -m pip install --no-cache-dir -r "$repo_root/requirements-eval.txt"
"$python_bin" -m pip install --no-cache-dir --upgrade --no-deps \
  transformers==5.5.0 tokenizers==0.22.2 huggingface-hub==1.21.0

"$python_bin" - <<'PY'
import importlib.metadata as metadata

expected = {
    "torch": "2.10.0",
    "nvidia-nccl-cu12": "2.27.5",
    "transformers": "5.5.0",
    "tokenizers": "0.22.2",
    "vllm": "0.18.0",
    "flashinfer-python": "0.6.6",
    "quack-kernels": "0.5.0",
    "nvidia-cutlass-dsl": "4.5.3",
    "nvidia-cutlass-dsl-libs-base": "4.5.3",
    "huggingface-hub": "1.21.0",
}
observed = {name: metadata.version(name) for name in expected}
different = {
    name: {"expected": expected[name], "observed": version}
    for name, version in observed.items()
    if version != expected[name]
}
if different:
    raise SystemExit(f"evaluation environment differs: {different}")

# This import exercises the compatibility boundary that an unconstrained
# CUTLASS DSL upgrade breaks before vLLM can finish initializing its workers.
import cutlass.cute.core as cute_core
import quack.layout_utils  # noqa: F401
import transformers  # noqa: F401
import vllm  # noqa: F401

if not hasattr(cute_core, "ThrMma"):
    raise SystemExit("CUTLASS DSL is incompatible: cutlass.cute.core.ThrMma is missing")
print("Verified evaluation package versions:", observed)
PY

# vLLM 0.18 predates the released Qwen3.5 Transformers implementation. The
# one audited metadata exception is deliberate; reject every additional
# resolver conflict instead of hiding it.
pip_check="$("$python_bin" -m pip check 2>&1 || true)"
expected_conflict='vllm 0.18.0 has requirement transformers<5,>=4.56.0, but you have transformers 5.5.0.'
if [[ "$pip_check" != "$expected_conflict" ]]; then
  echo "Unexpected pip check result:" >&2
  printf '%s\n' "$pip_check" >&2
  exit 2
fi
echo "Verified the single documented vLLM/Transformers metadata exception."
