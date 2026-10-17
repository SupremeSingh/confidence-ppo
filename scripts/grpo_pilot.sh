#!/usr/bin/env bash
# Isolated neutral GRPO pilot: submit [OUT] [ACTOR_LR] or report [OUT].
set -euo pipefail
mode="${1:-submit}"
if [[ "$mode" == run ]]; then
  cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
else
  cd "$(dirname "${BASH_SOURCE[0]}")/.."
fi
out="${2:-outputs/confidence-grpo-neutral-v1}"

case "$mode" in
submit)
  [[ -f .runtime-ready ]] || { echo 'Complete runtime setup first.' >&2; exit 1; }
  command -v sbatch >/dev/null
  # Atomic creation prevents duplicate submissions to the same destination.
  mkdir "$out"
  out="$(cd "$out" && pwd)"
  python3 - "$out" "${3:-1e-6}" <<'PY'
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

source = Path('outputs/confidence-v1').resolve()
out = Path(sys.argv[1])
actor_lr = float(sys.argv[2])
if not math.isfinite(actor_lr) or actor_lr <= 0:
    raise ValueError('Actor learning rate must be finite and positive')
manifest = json.loads((source / 'manifest.json').read_text())
(out / 'data').mkdir()
hashes = {}
for split in ('train', 'val'):
    relative = f'neutral/data/{split}.parquet'
    destination = out / 'data' / f'{split}.parquet'
    shutil.copyfile(source / relative, destination)
    checksum = hashlib.sha256(destination.read_bytes()).hexdigest()
    if checksum != manifest['data_hashes'][relative]:
        raise ValueError(f'Prepared data changed: {relative}')
    hashes[f'data/{split}.parquet'] = checksum
(out / 'pilot.json').write_text(json.dumps(dict(
    method='grpo', style='neutral', seed=301, steps=20,
    answers_per_update=64, actor_lr=actor_lr, evaluation_split='val',
    source=str(source), data_hashes=hashes,
    scope='Validation pilot; one seed; no test-set evaluation.'
), indent=2) + '\n')
PY
  job=$(sbatch --parsable --nodes=1 --ntasks=1 --cpus-per-task=8 \
    --mem=128G --time=04:00:00 --partition=compsci-gpu --gres=gpu:a5000:2 \
    --job-name=confidence-grpo-pilot --output="$out/slurm-%j.out" \
    scripts/grpo_pilot.sh run "$out")
  printf '%s\n' "${job%%;*}" > "$out/job-id.txt"
  echo "Submitted GRPO pilot: ${job%%;*}"
  echo "Training log: $out/grpo-seed301.log"
  ;;
run)
  : "${SLURM_JOB_ID:?Run via the submit command}"
  out="$(cd "$out" && pwd)"
  [[ -f "$out/pilot.json" ]] || { echo 'Missing pilot manifest.' >&2; exit 1; }
  # Slurm receives the output directory; settings come from its saved manifest.
  actor_lr=$(python3 - "$out/pilot.json" <<'PY'
import json
import math
from pathlib import Path
import sys
lr = float(json.loads(Path(sys.argv[1]).read_text())['actor_lr'])
if not math.isfinite(lr) or lr <= 0:
    raise ValueError('Invalid actor learning rate in pilot manifest')
print(format(lr, '.12g'))
PY
  )
  source scripts/cluster_env.sh
  export MATH_RL_GPU=1 MATH_RL_SEED=301
  export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1
  export PYTHONPATH="$PWD/src" RAY_ADDRESS=local
  export RAY_TMPDIR
  RAY_TMPDIR=$(mktemp -d /tmp/cp-grpo-XXXXXX)
  unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE
  srun bash scripts/container_exec.sh "$PWD/.venv/bin/python" \
    -m math_rl.math_main \
    experiment.method=grpo experiment.ppo_profile=legacy \
    algorithm.adv_estimator=grpo \
    data.train_files="$out/data/train.parquet" \
    data.val_files="$out/data/val.parquet" \
    experiment.test_file="$out/data/val.parquet" \
    experiment.neutral_test_file="$out/data/val.parquet" \
    experiment.evaluation_split=val data.seed=301 \
    actor_rollout_ref.actor.optim.lr="$actor_lr" \
    trainer.total_training_steps=20 trainer.save_freq=20 \
    trainer.default_local_dir="$out/grpo-seed301" \
    trainer.experiment_name=confidence-grpo-neutral-seed301 \
    hydra.run.dir="$out/hydra-seed301" \
    > "$out/grpo-seed301.log" 2>&1
  bash scripts/grpo_pilot.sh report "$out"
  ;;
report)
  python3 - "$out" <<'PY'
import json
from pathlib import Path
import sys

out = Path(sys.argv[1])
p = out / 'grpo-seed301/result.json'
if not p.exists():
    sys.exit(f'Pilot not complete. Inspect {out / "grpo-seed301.log"} and slurm-*.out.')
r = json.loads(p.read_text())
pilot = json.loads((out / 'pilot.json').read_text())
if (r['method'], r['seed'], r['updates'], r['answers'], r['evaluation_split']) != ('grpo', 301, 20, 1280, 'val'):
    sys.exit('Unexpected method, seed, budget or evaluation split.')
before = 100 * r['initial_validation']['accuracy']
after = 100 * r['final_validation']['accuracy']
text = (f'Neutral GRPO validation pilot; seed 301\n'
        f'Actor learning rate: {pilot["actor_lr"]:g}\n'
        f'Updates: {r["updates"]}; answers: {r["answers"]}\n'
        f'Validation accuracy: {before:.2f}% -> {after:.2f}%\n'
        f'Gain: {after-before:+.2f} percentage points\n'
        'One seed and 64 validation questions; diagnostic evidence only.\n')
(out / 'report.txt').write_text(text)
print(text, end='')
PY
  ;;
*) echo "Usage: bash scripts/grpo_pilot.sh submit [OUT] [ACTOR_LR] | report [OUT]" >&2; exit 2 ;;
esac
