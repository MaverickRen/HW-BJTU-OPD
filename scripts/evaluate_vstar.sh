#!/usr/bin/env bash
# One-command public reproduction of the reported 176/191 VStar cell.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model_repo="HWBJTUOPD/Qwen3.5-9B-SFT10K-VisionOPD6K-SFT9BTeacher"
model_revision="6fc7d1ed7c509572898a32ff9de6cff19e8455f0"
work_dir="$repo_root/artifacts/vstar-reproduction"
gpus="${CUDA_VISIBLE_DEVICES:-0}"
tp_size=1
port=18618
limit=191
mode=dry-run
model_id="HW-BJTU-OPD-9B-SFT9Teacher"

usage() {
  echo "Usage: $0 [--work-dir DIR] [--gpus 0[,1...]] [--tp-size N] [--quick N] [--execute]" >&2
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --work-dir) work_dir="${2:-}"; shift 2 ;;
    --model-repo) model_repo="${2:-}"; shift 2 ;;
    --model-revision) model_revision="${2:-}"; shift 2 ;;
    --gpus) gpus="${2:-}"; shift 2 ;;
    --tp-size) tp_size="${2:-}"; shift 2 ;;
    --port) port="${2:-}"; shift 2 ;;
    --quick) limit="${2:-}"; shift 2 ;;
    --execute) mode=execute; shift ;;
    -h|--help) usage ;;
    *) usage ;;
  esac
done

[[ "$tp_size" =~ ^[1-9][0-9]*$ ]] || usage
[[ "$limit" =~ ^[1-9][0-9]*$ && "$limit" -le 191 ]] || usage
[[ "$port" =~ ^[0-9]+$ && "$port" -ge 1024 && "$port" -le 65535 ]] || usage
[[ "$gpus" =~ ^[0-9]+(,[0-9]+)*$ ]] || usage
gpu_count=$(( $(tr -cd ',' <<<"$gpus" | wc -c) + 1 ))
(( tp_size <= gpu_count )) || { echo "--tp-size exceeds --gpus count" >&2; exit 2; }

python_bin="${PYTHON_BIN:-python}"
vllm_bin="${VLLM_BIN:-vllm}"
model_dir="$work_dir/model"
data_dir="$work_dir/data"
run_dir="$work_dir/run"
result="$run_dir/vstar-${limit}.json"
serve_log="$run_dir/vllm-${limit}.log"

if [[ "$mode" == dry-run ]]; then
  "$python_bin" "$repo_root/scripts/evaluate.py" \
    --benchmark vstar --model "$model_dir" --model-id "$model_id" \
    --data "$data_dir/vstar.json" --output "$result" --limit "$limit" \
    --api-base "http://127.0.0.1:$port/v1"
  exit 0
fi

command -v "$python_bin" >/dev/null || { echo "Python is required" >&2; exit 2; }
command -v hf >/dev/null || { echo "hf CLI is required" >&2; exit 2; }
command -v "$vllm_bin" >/dev/null || { echo "vLLM is required; install requirements-eval.txt" >&2; exit 2; }
command -v setsid >/dev/null || { echo "setsid is required (util-linux)" >&2; exit 2; }
command -v mktemp >/dev/null || { echo "mktemp is required (coreutils)" >&2; exit 2; }
[[ ! -L "$work_dir" ]] || { echo "work directory cannot be a symlink" >&2; exit 2; }
mkdir -p "$work_dir" "$run_dir"
[[ ! -e "$result" ]] || { echo "result already exists: $result" >&2; exit 2; }

if [[ ! -f "$model_dir/model.safetensors" ]]; then
  hf download "$model_repo" --revision "$model_revision" --local-dir "$model_dir"
fi
"$python_bin" "$repo_root/scripts/prepare_opd_release.py" \
  --source "$model_dir" --verify-only
"$python_bin" "$repo_root/scripts/prepare_vstar.py" --output "$data_dir"

nccl_so="$("$python_bin" - <<'PY'
from importlib.metadata import distribution
from pathlib import Path

package = distribution("nvidia-nccl-cu12")
matches = [
    Path(package.locate_file(item))
    for item in package.files or ()
    if str(item).endswith("nvidia/nccl/lib/libnccl.so.2")
]
if len(matches) != 1 or matches[0].is_symlink() or not matches[0].is_file():
    raise SystemExit("could not resolve the pinned nvidia-nccl-cu12 library")
print(matches[0])
PY
)"

runtime_tmp="$(mktemp -d /tmp/hw-opd.XXXXXX)"
runtime_user="${USER:-${LOGNAME:-opd-eval}}"
server_pid=''
cleanup() {
  rc=$?
  trap - EXIT INT TERM HUP
  if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
    kill -TERM -- "-$server_pid" 2>/dev/null || kill -TERM "$server_pid" 2>/dev/null || true
    for _ in $(seq 1 15); do
      kill -0 "$server_pid" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "$server_pid" 2>/dev/null; then
      kill -KILL -- "-$server_pid" 2>/dev/null || kill -KILL "$server_pid" 2>/dev/null || true
    fi
    wait "$server_pid" 2>/dev/null || true
  fi
  rm -rf -- "$runtime_tmp"
  exit "$rc"
}
trap cleanup EXIT INT TERM HUP

# Compilation and IPC caches are latency-sensitive and can stall badly on a
# network-mounted work directory. Keep them on the node-local, short-lived
# runtime path; downloaded model/data and final results remain in work_dir.
runtime="$runtime_tmp/cache"
mkdir -p "$runtime"/{hf,xdg,cuda,torchinductor,triton}

setsid env -u NCCL_SOCKET_IFNAME -u VLLM_BIN \
  USER="$runtime_user" LOGNAME="$runtime_user" \
  VLLM_NCCL_SO_PATH="$nccl_so" \
  CUDA_VISIBLE_DEVICES="$gpus" HF_HOME="$runtime/hf" XDG_CACHE_HOME="$runtime/xdg" \
  CUDA_CACHE_PATH="$runtime/cuda" TORCHINDUCTOR_CACHE_DIR="$runtime/torchinductor" \
  TRITON_CACHE_DIR="$runtime/triton" FLASHINFER_WORKSPACE_BASE="$runtime/flashinfer" \
  TMPDIR="$runtime_tmp" \
  "$vllm_bin" serve "$model_dir" \
  --served-model-name "$model_id" --host 127.0.0.1 --port "$port" \
  --tensor-parallel-size "$tp_size" --gpu-memory-utilization 0.85 \
  --max-model-len 65536 --max-num-seqs 8 --limit-mm-per-prompt '{"image":1}' \
  --chat-template "$model_dir/chat_template.jinja" --trust-remote-code \
  --compilation-config '{"pass_config":{"fuse_allreduce_rms":false}}' \
  --kernel-config '{"enable_flashinfer_autotune":false}' --no-enable-log-requests \
  >"$serve_log" 2>&1 &
server_pid=$!

ready=0
for _ in $(seq 1 360); do
  kill -0 "$server_pid" 2>/dev/null || break
  if "$python_bin" - "$port" "$model_id" <<'PY' >/dev/null 2>&1
import json, sys, urllib.request
port, model = sys.argv[1:]
with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=3) as response:
    values = json.load(response).get("data", [])
raise SystemExit(0 if model in {str(item.get("id")) for item in values} else 1)
PY
  then
    ready=1
    break
  fi
  sleep 5
done
(( ready == 1 )) || { echo "vLLM failed readiness; see $serve_log" >&2; exit 2; }

"$python_bin" "$repo_root/scripts/evaluate.py" \
  --benchmark vstar --model "$model_dir" --model-id "$model_id" \
  --data "$data_dir/vstar.json" --output "$result" --limit "$limit" \
  --workers 8 --api-base "http://127.0.0.1:$port/v1" --execute >/dev/null

"$python_bin" - "$result" "$model_repo" "$model_revision" "$limit" <<'PY'
import json, os, pathlib, sys
path = pathlib.Path(sys.argv[1])
value = json.loads(path.read_text())
value["artifact"] = {
    "repo_id": sys.argv[2],
    "revision": sys.argv[3],
    "model_sha256": "c86054edddaf186b5a0754fed55e4d8e80108ba2081ff7e6ba7c2d3e589ccdc7",
}
full = int(sys.argv[4]) == 191
if full:
    cell = value["benchmarks"]["vstar"]
    delta = int(cell["correct"]) - 176
    value["published_reference"] = {"correct": 176, "total": 191, "tolerance_correct": 3}
    value["reproduction"] = {"correct_delta": delta, "similar": abs(delta) <= 3}
temporary = path.with_name(f".{path.name}.final-{os.getpid()}")
with temporary.open("x", encoding="utf-8") as stream:
    json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())
os.replace(temporary, path)
print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
raise SystemExit(0 if not full or value["reproduction"]["similar"] else 3)
PY
