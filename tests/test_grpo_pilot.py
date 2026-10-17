"""Check isolated GRPO submission and Slurm execution without GPUs."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


SOURCE = Path(__file__).resolve().parents[1]


def setup_repo(tmp_path):
    repo = tmp_path / 'repo'
    (repo / 'scripts').mkdir(parents=True)
    shutil.copyfile(SOURCE / 'scripts/grpo_pilot.sh', repo / 'scripts/grpo_pilot.sh')
    (repo / '.runtime-ready').touch()
    data = repo / 'outputs/confidence-v1/neutral/data'
    data.mkdir(parents=True)
    hashes = {}
    for split in ('train', 'val'):
        p = data / f'{split}.parquet'
        p.write_bytes(f'{split} snapshot'.encode())
        hashes[f'neutral/data/{p.name}'] = hashlib.sha256(p.read_bytes()).hexdigest()
    (repo / 'outputs/confidence-v1/manifest.json').write_text(json.dumps({'data_hashes': hashes}))
    binary = tmp_path / 'bin'
    binary.mkdir()
    log = tmp_path / 'calls.json'
    fake = binary / 'sbatch'
    fake.write_text('#!/usr/bin/env python3\nimport json,os,sys\n'
                    'from pathlib import Path\n'
                    'Path(os.environ["CALLS"]).write_text(json.dumps(sys.argv[1:]))\n'
                    'print("123;cluster")\n')
    fake.chmod(0o755)
    env = dict(os.environ, PATH=str(binary) + ':' + os.environ['PATH'], CALLS=str(log))
    return repo, env, log


def test_submission_copies_verified_data_and_rejects_reuse(tmp_path):
    repo, env, log = setup_repo(tmp_path)
    command = ['bash', 'scripts/grpo_pilot.sh', 'submit']
    subprocess.run(command, cwd=repo, env=env, check=True, capture_output=True)
    args = json.loads(log.read_text())
    assert '--gres=gpu:a5000:2' in args
    assert args[-3:-1] == ['scripts/grpo_pilot.sh', 'run']
    out = repo / 'outputs/confidence-grpo-neutral-v1'
    assert (out / 'job-id.txt').read_text() == '123\n'
    assert (out / 'data/train.parquet').read_bytes() == b'train snapshot'
    assert json.loads((out / 'pilot.json').read_text())['evaluation_split'] == 'val'
    log.unlink()
    assert subprocess.run(command, cwd=repo, env=env, capture_output=True).returncode != 0
    assert not log.exists()


def test_changed_source_data_prevents_submission(tmp_path):
    repo, env, log = setup_repo(tmp_path)
    (repo / 'outputs/confidence-v1/neutral/data/train.parquet').write_bytes(b'changed')
    result = subprocess.run(['bash', 'scripts/grpo_pilot.sh', 'submit'],
                            cwd=repo, env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert 'Prepared data changed' in result.stderr
    assert not log.exists()


def test_slurm_spooled_script_uses_submit_directory_and_grpo(tmp_path):
    repo, env, log = setup_repo(tmp_path)
    out = repo / 'outputs/pilot'
    out.mkdir()
    (out / 'pilot.json').write_text('{}')
    case = out / 'grpo-seed301'
    case.mkdir()
    (case / 'result.json').write_text(json.dumps(dict(
        method='grpo', seed=301, updates=20, answers=1280, evaluation_split='val',
        initial_validation={'accuracy': .5}, final_validation={'accuracy': .5625})))
    (repo / 'scripts/cluster_env.sh').write_text('# Existing runtime\n')
    srun = Path(env['PATH'].split(':')[0]) / 'srun'
    srun.write_text('#!/usr/bin/env python3\nimport json,os,sys\n'
                    'from pathlib import Path\n'
                    'Path(os.environ["CALLS"]).write_text(json.dumps(sys.argv[1:]))\n')
    srun.chmod(0o755)
    spool = tmp_path / 'slurm-script'
    shutil.copyfile(repo / 'scripts/grpo_pilot.sh', spool)
    subprocess.run(['bash', str(spool), 'run', str(out)], cwd=tmp_path,
                   env=dict(env, SLURM_SUBMIT_DIR=str(repo), SLURM_JOB_ID='123'),
                   check=True, capture_output=True)
    args = json.loads(log.read_text())
    assert 'experiment.method=grpo' in args
    assert 'algorithm.adv_estimator=grpo' in args
    assert f'experiment.test_file={out}/data/val.parquet' in args
    assert 'actor_rollout_ref.actor.optim.lr=1e-6' in args
    assert 'Gain: +6.25 percentage points' in (out / 'report.txt').read_text()
