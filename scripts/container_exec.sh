#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$repo/scripts/cluster_env.sh"
[[ -f "$MATH_RL_IMAGE" ]] || { echo "Missing image: $MATH_RL_IMAGE" >&2; exit 1; }
cd "$repo"
if [[ -f configs/container.sha256 ]]; then
  read -r expected_hash _ < configs/container.sha256
  actual_hash=$(sha256sum "$MATH_RL_IMAGE")
  [[ "${actual_hash%% *}" == "$expected_hash" ]] || { echo 'Selected container checksum mismatch.' >&2; exit 1; }
fi
gpu_args=()
if [[ "${MATH_RL_GPU:-0}" == 1 ]]; then gpu_args=(--nv); fi
ray_args=()
if [[ -n "${RAY_TMPDIR:-}" ]]; then ray_args=(--bind "$RAY_TMPDIR:$RAY_TMPDIR"); fi
exec "$MATH_RL_RUNTIME" exec "${gpu_args[@]}" \
  "${ray_args[@]}" \
  --bind "$MATH_RL_ROOT:$MATH_RL_ROOT" --bind "$repo:$repo" \
  --bind "${TMPDIR:-/tmp}:${TMPDIR:-/tmp}" \
  --pwd "$repo" "$MATH_RL_IMAGE" "$@"
