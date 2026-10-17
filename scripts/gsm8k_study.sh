#!/usr/bin/env bash
# Duke: bash scripts/gsm8k_study.sh pilot|status|full|report [OUT] [REVISION]
set -euo pipefail
mode="${1:-pilot}"
case "$mode" in
  prepare|smoke|run-neutral|run-full|gate|final-report) cd "${SLURM_SUBMIT_DIR:?Submit from repository root}" ;;
  *) cd "$(dirname "${BASH_SOURCE[0]}")/.." ;;
esac
out="${2:-outputs/gsm8k-screened-grpo-v1}"

submit() {
  local stage="$1" dependency="$2" gpu_count="$3" array="$4"
  local opts=(--parsable --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=128G --time=12:00:00
    --job-name="gsm8k-$stage" --output="$out/logs/$stage-%A_%a.out")
  if [[ "$gpu_count" == 0 ]]; then
    opts+=(--partition=compsci)
  else
    opts+=(--partition=compsci-gpu --gres="gpu:a5000:$gpu_count")
  fi
  [[ -z "$dependency" ]] || opts+=(--dependency="afterok:$dependency" --kill-on-invalid-dep=yes)
  [[ -z "$array" ]] || opts+=(--array="$array")
  local job
  job=$(sbatch "${opts[@]}" scripts/gsm8k_study.sh "$stage" "$out")
  job="${job%%;*}"
  printf '%s\t%s\n' "$stage" "$job" >> "$out/jobs.tsv"
  echo "Submitted $stage: $job" >&2
  echo "$job"
}

case "$mode" in
pilot)
  [[ -f .runtime-ready ]] || { echo 'Complete runtime setup first.' >&2; exit 1; }
  command -v sbatch >/dev/null
  mkdir "$out"
  out="$(cd "$out" && pwd)"
  mkdir "$out/logs"
  python3 - "$out" "${3:-main}" <<'PY'
import json
from pathlib import Path
import sys
(Path(sys.argv[1]) / 'request.json').write_text(json.dumps({'revision': sys.argv[2]}) + '\n')
PY
  preparation=$(submit prepare '' 1 '')
  smoke_job=$(submit smoke "$preparation" 2 '')
  neutral=$(submit run-neutral "$smoke_job" 2 '0-2%2')
  submit gate "$neutral" 0 ''
  echo "Workflow: screening -> GPU smoke -> 3 neutral seeds -> learning gate."
  echo "Inspect: bash scripts/gsm8k_study.sh status $out"
  ;;
full)
  [[ -f .runtime-ready ]] || { echo 'Complete runtime setup first.' >&2; exit 1; }
  out="$(cd "$out" && pwd)"
  # CPU-only check: recompute gate from saved answers/audits and verify hashes.
  python3 scripts/gsm8k_study.py full-check --out "$out"
  mkdir "$out/full-submission"
  remaining=$(submit run-full '' 2 '0-14%2')
  submit final-report "$remaining" 0 ''
  echo 'The comparison uses 15 fresh runs; pilot seeds are excluded from the comparison.'
  ;;
status)
  echo 'Submitted jobs:'
  cat "$out/jobs.tsv"
  if [[ -f "$out/screening-summary.json" ]]; then cat "$out/screening-summary.json"; fi
  if [[ -f "$out/gate-report.txt" ]]; then
    cat "$out/gate-report.txt"
  else
    echo "Gate not available yet. Inspect $out/logs/ for progress or failures."
  fi
  ;;
report)
  if [[ -f "$out/report.txt" ]]; then cat "$out/report.txt";
  elif [[ -f "$out/gate-report.txt" ]]; then cat "$out/gate-report.txt";
  else echo 'No completed report. Use status and inspect Slurm logs.' >&2; exit 1; fi
  ;;
prepare|smoke|run-neutral|run-full|gate|final-report)
  out="$(cd "$out" && pwd)"
  source scripts/cluster_env.sh
  export MATH_RL_GPU=1
  [[ "$mode" != gate && "$mode" != final-report ]] || export MATH_RL_GPU=0
  export PYTHONPATH="$PWD/src" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
  export PYTHONUNBUFFERED=1 RAY_ADDRESS=local
  export RAY_TMPDIR
  RAY_TMPDIR=$(mktemp -d /tmp/gsm8k-study-XXXXXX)
  unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE
  stage="$mode"
  [[ "$stage" != final-report ]] || stage=report
  srun bash scripts/container_exec.sh "$PWD/.venv/bin/python" \
    scripts/gsm8k_study.py "$stage" --out "$out" --task "${SLURM_ARRAY_TASK_ID:-0}"
  ;;
*) echo 'Usage: bash scripts/gsm8k_study.sh pilot|status|full|report [OUT] [REVISION]' >&2; exit 2 ;;
esac
