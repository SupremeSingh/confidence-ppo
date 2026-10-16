import json
import os
from pathlib import Path
import shutil
import subprocess


def test_initial_submission_chains_private_setup_prepare_smoke(tmp_path):
    repo=tmp_path/'standalone';(repo/'scripts').mkdir(parents=True);(repo/'inputs').mkdir()
    source=Path(__file__).resolve().parents[1]
    shutil.copyfile(source/'scripts/submit.sh',repo/'scripts/submit.sh')
    (repo/'scripts/study.py').touch();(repo/'inputs/assets.json').write_text('{}')
    bin_dir=tmp_path/'bin';bin_dir.mkdir();log=tmp_path/'calls'
    fake=bin_dir/'sbatch'
    fake.write_text('#!/usr/bin/env python3\nimport json,os,sys\nfrom pathlib import Path\np=Path(os.environ["CALLS"])\nrows=p.read_text().splitlines() if p.exists() else []\nwith p.open("a") as f:f.write(json.dumps(sys.argv[1:])+"\\n")\nprint(100+len(rows))\n')
    fake.chmod(0o755)
    subprocess.run(['bash','scripts/submit.sh'],cwd=repo,check=True,env=dict(os.environ,PATH=str(bin_dir)+':'+os.environ['PATH'],CALLS=str(log)),capture_output=True,text=True)
    calls=[json.loads(l) for l in log.read_text().splitlines()]
    assert len(calls)==3
    assert '--dependency=afterok:100' in calls[1]
    assert '--dependency=afterok:101' in calls[2]
    assert '--gres=gpu:a5000:2' not in calls[0]
    assert '--gres=gpu:a5000:2' in calls[2]
    assert not any('rl_ms_project' in arg for call in calls for arg in call)
