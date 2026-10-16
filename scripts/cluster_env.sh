#!/usr/bin/env bash
# Source on the host before setup, submissions, or container commands.
export MATH_RL_ROOT="${MATH_RL_ROOT:-/usr/xtmp/ms785}"
export MATH_RL_IMAGE="${MATH_RL_IMAGE:-$MATH_RL_ROOT/containers/verl-v041.sif}"
if [[ -z "${MATH_RL_RUNTIME:-}" ]]; then
  if command -v apptainer >/dev/null 2>&1; then
    MATH_RL_RUNTIME=apptainer
  elif command -v singularity >/dev/null 2>&1; then
    MATH_RL_RUNTIME=singularity
  else
    echo 'Load the site Singularity/Apptainer module first (module avail).' >&2
    return 1
  fi
fi
export MATH_RL_RUNTIME
export HF_HOME="$MATH_RL_ROOT/cache/huggingface"
export APPTAINER_CACHEDIR="$MATH_RL_ROOT/cache/apptainer"
export SINGULARITY_CACHEDIR="$MATH_RL_ROOT/cache/singularity"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-$MATH_RL_ROOT/cache/apptainer-tmp}"
export SINGULARITY_TMPDIR="$APPTAINER_TMPDIR"
# mksquashfs is memory hungry; two workers are plenty for this one-time pull.
export APPTAINER_MKSQUASHFS_ARGS="${APPTAINER_MKSQUASHFS_ARGS:--processors 2}"
export SINGULARITY_MKSQUASHFS_ARGS="${SINGULARITY_MKSQUASHFS_ARGS:--processors 2}"
export PIP_CACHE_DIR="$MATH_RL_ROOT/cache/pip"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  export TMPDIR="${SLURM_TMPDIR:-/tmp/math-rl-${USER}-${SLURM_JOB_ID}}"
  mkdir -p "$TMPDIR"
fi
mkdir -p "$MATH_RL_ROOT/containers" "$HF_HOME" "$APPTAINER_CACHEDIR" "$SINGULARITY_CACHEDIR" "$APPTAINER_TMPDIR" "$SINGULARITY_TMPDIR" "$PIP_CACHE_DIR"
