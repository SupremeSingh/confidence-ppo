#!/usr/bin/env bash
# Run inside the existing pinned Duke container, in a compute allocation.
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Use the setup Slurm job.' >&2; exit 1; }
[[ -f inputs/assets.json ]] || { echo 'Import assets first.' >&2; exit 1; }
python3 -m venv --system-site-packages --without-pip .venv
python3 -m pip --python .venv/bin/python install pip
.venv/bin/python -m pip install --no-deps 'verl @ git+https://github.com/verl-project/verl.git@8d9e350ea58c7ad4b50dd14d9dcb50577242c55f'
.venv/bin/python -m pip install -e '.[dev]'
python3 -m venv --without-pip .venv-ppo-verifier
.venv/bin/python -m pip --python .venv-ppo-verifier/bin/python install -r requirements-verifier.txt
.venv/bin/python -m pip freeze > configs/environment.txt
.venv/bin/python -m pip --python .venv-ppo-verifier/bin/python freeze > configs/ppo-verifier-environment.txt
PYTHONPATH=src .venv/bin/python -m pytest -q
PYTHONPATH=src .venv-ppo-verifier/bin/python -c 'from math_rl.math_verify_reward import provenance; print(provenance())'
.venv/bin/python -c 'import torch; assert torch.__version__.split("+")[0] == "2.6.0", torch.__version__'
printf 'ready\n' > .runtime-ready
