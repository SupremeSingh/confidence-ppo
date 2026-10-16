"""Small, CPU-only provenance records; no credentials or full environment dump."""
import hashlib
import json
import os
from pathlib import Path
import subprocess


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot(root):
    root = Path(root)
    files = {}
    for pattern in ("src/**/*.py", "scripts/*.py", "scripts/*.sh", "configs/*.yaml", "pyproject.toml", "requirements*.txt"):
        for path in sorted(root.glob(pattern)):
            files[str(path.relative_to(root))] = sha256(path)
    for name in ("configs/assets.json", "configs/environment.txt", "configs/container.sha256",
                 "configs/ppo-verifier-environment.txt", "data/gsm8k/train.parquet", "data/gsm8k/val.parquet"):
        path = root / name
        if path.exists():
            files[name] = sha256(path)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {"git_commit": commit, "files": files,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "container_path": os.environ.get("MATH_RL_IMAGE")}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
