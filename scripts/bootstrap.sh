#!/usr/bin/env bash
# Fetch the exact upstream revisions and apply the recorded Qwen3.5/OPD patch.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
third_party="${1:-$repo_root/third_party}"
python_bin="${PYTHON_BIN:-python3}"

# Public, fetchable upstream base.  The release patch contains both the three
# historical local commits and the final working-tree changes used by SFT.
verl_commit=11c94ad2354456d9bfa93c558e05e9430cd731b2
vision_opd_commit=c8a8fdd1f88eef1b5ef4fe6a8d64eb0272917471
vlmevalkit_commit=09874c7a69c2a3c7c60ace141525c1552a2c1095

mkdir -p "$third_party"

clone_at() {
  local url="$1" destination="$2" commit="$3" allow_dirty="${4:-0}"
  if [[ ! -d "$destination/.git" ]]; then
    git clone --filter=blob:none "$url" "$destination"
    git -C "$destination" fetch --depth=1 origin "$commit"
    git -C "$destination" checkout --detach "$commit"
    return
  fi
  [[ "$allow_dirty" == 1 || -z "$(git -C "$destination" status --porcelain)" ]] || {
    echo "Refusing to alter a dirty dependency checkout: $destination" >&2
    exit 2
  }
  [[ "$(git -C "$destination" rev-parse HEAD)" == "$commit" ]] || {
    echo "Existing dependency checkout is not at the pinned commit: $destination" >&2
    exit 2
  }
}

clone_at https://github.com/verl-project/verl.git "$third_party/verl" "$verl_commit" 1
clone_at https://github.com/VisionOPD/Vision-OPD.git "$third_party/Vision-OPD" "$vision_opd_commit"
clone_at https://github.com/open-compass/VLMEvalKit.git "$third_party/VLMEvalKit" "$vlmevalkit_commit"

if ! git -C "$third_party/verl" apply --reverse --check "$repo_root/patches/verl-qwen35-opd.patch" >/dev/null 2>&1; then
  git -C "$third_party/verl" apply --check "$repo_root/patches/verl-qwen35-opd.patch"
  git -C "$third_party/verl" apply "$repo_root/patches/verl-qwen35-opd.patch"
fi
cp -R "$repo_root/patches/verl-tests/tests/." "$third_party/verl/tests/"

# The standalone patched checkout is the exact SFT runtime.  OPD uses the
# vendored veRL in the pinned Vision-OPD checkout, matching the successful run.
if [[ "${INSTALL_EDITABLE:-1}" == 1 ]]; then
  "$python_bin" -m pip install -e "$third_party/verl" --no-deps
fi

git -C "$third_party/verl" diff --check
printf 'Prepared pinned dependencies in %s\n' "$third_party"
