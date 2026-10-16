#!/usr/bin/env bash
# Run from this standalone repository root: bash scripts/submit.sh [smoke|full] [OUT]
set -euo pipefail
mode="${1:-smoke}"; out="${2:-outputs/confidence-v1}"
[[ "$mode" == smoke || "$mode" == full ]] || { echo 'Mode must be smoke or full'; exit 2; }
[[ -f scripts/study.py && -f inputs/assets.json ]] || { echo 'Run from the standalone repository root after importing assets.' >&2; exit 2; }
mkdir -p "$out/logs"; out="$(cd "$out" && pwd)"
submit() {
 local stage="$1" dep="$2" gpu="$3" array="$4"
 local opts=(--parsable --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=128G --time=12:00:00
 --job-name="confidence-$stage" --output="$out/logs/$stage-%A_%a.out")
 if [[ "$gpu" == 1 ]]; then opts+=(--partition=compsci-gpu --gres=gpu:a5000:2); else opts+=(--partition=compsci); fi
 [[ -z "$dep" ]] || opts+=(--dependency="afterok:$dep" --kill-on-invalid-dep=yes)
 [[ -z "$array" ]] || opts+=(--array="$array")
 local job; job="$(sbatch "${opts[@]}" scripts/run.sh "$stage" "$out" "$gpu")"; job="${job%%;*}"
 printf '%s\t%s\n' "$stage" "$job" >> "$out/jobs.tsv"; echo "Submitted $stage: $job" >&2; echo "$job"
}
if [[ "$mode" == smoke ]]; then
 dep=''; [[ -f .runtime-ready ]] || dep="$(submit setup '' 0 '')"
 [[ -f "$out/manifest.json" ]] || dep="$(submit prepare "$dep" 0 '')"
 submit smoke "$dep" 1 ''
else
 [[ -f .runtime-ready ]] || { echo 'Setup must complete first'; exit 2; }
 python3 -c 'import json,sys; assert json.load(open(sys.argv[1]))["passed"]' "$out/smoke.json"
 job="$(submit run '' 1 '0-14%2')"; submit report "$job" 0 ''
fi
