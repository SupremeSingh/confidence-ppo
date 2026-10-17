#!/usr/bin/env bash
# Run from Duke: bash scripts/gsm8k_pilot.sh submit|report [OUT] [DATASET_REVISION]
set -euo pipefail
mode="${1:-submit}"
if [[ "$mode" == run ]]; then
  cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
else
  cd "$(dirname "${BASH_SOURCE[0]}")/.."
fi
out="${2:-outputs/gsm8k-grpo-learning-check-v1}"
case "$mode" in
submit)
  [[ -f .runtime-ready ]] || { echo 'Complete runtime setup first.' >&2; exit 1; }
  command -v sbatch >/dev/null
  mkdir "$out"
  out="$(cd "$out" && pwd)"
  python3 - "$out" "${3:-main}" <<'PY'
import json
from pathlib import Path
import sys
(Path(sys.argv[1]) / 'request.json').write_text(json.dumps({'revision': sys.argv[2]}) + '\n')
PY
  job=$(sbatch --parsable --nodes=1 --ntasks=1 --cpus-per-task=8 \
    --mem=128G --time=04:00:00 --partition=compsci-gpu --gres=gpu:a5000:2 \
    --job-name=gsm8k-grpo-check --output="$out/slurm-%j.out" \
    scripts/gsm8k_pilot.sh run "$out")
  printf '%s\n' "${job%%;*}" > "$out/job-id.txt"
  echo "Submitted GSM8K learning check: ${job%%;*}"
  echo "Logs: $out/preparation.log and $out/grpo-seed301.log"
  ;;
run)
  : "${SLURM_JOB_ID:?Run via submit}"
  out="$(cd "$out" && pwd)"
  source scripts/cluster_env.sh
  export MATH_RL_GPU=1 MATH_RL_SEED=301 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
  export PYTHONPATH="$PWD/src" PYTHONUNBUFFERED=1 RAY_ADDRESS=local
  export RAY_TMPDIR
  RAY_TMPDIR=$(mktemp -d /tmp/gsm8k-grpo-XXXXXX)
  unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE
  srun bash scripts/container_exec.sh "$PWD/.venv/bin/python" \
    scripts/gsm8k_pilot.py run --out "$out" > "$out/preparation.log" 2>&1
  ;;
report)
  python3 scripts/gsm8k_pilot.py report --out "$out"
  ;;
*) echo 'Usage: bash scripts/gsm8k_pilot.sh submit|report [OUT] [DATASET_REVISION]' >&2; exit 2 ;;
esac
