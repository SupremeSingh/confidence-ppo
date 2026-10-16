"""VERL batch-reward hook. Keep symbolic dependencies outside the Torch environment."""
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
EXCLUSION_POLICY = 'Only correct/incorrect verifier statuses with nonempty tokens enter critic fitting; all others excluded, not zero-labelled'


def verifier_command():
    python = ROOT / ".venv-ppo-verifier/bin/python"
    if not python.is_file():
        raise RuntimeError("Run scripts/setup_ppo.sh inside the container first")
    return [str(python), "-m", "math_rl.verifier_batch"]


def compute_score(data_sources, solution_strs, ground_truths, extra_infos=None, *, exclude_errors=False, rule=None):
    if not (len(data_sources) == len(solution_strs) == len(ground_truths)):
        raise ValueError("Reward batch lengths differ")
    if any(source not in {"gsm8k", "math_numeric", "numina_numeric"} for source in data_sources):
        raise ValueError("Expected an explicit numeric math data source")
    if not solution_strs:
        return []
    rows = [dict(response=text, ground_truth=gold) for text, gold in zip(solution_strs, ground_truths)]
    if rule is not None:
        for row in rows:
            row['rule'] = rule
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    # One child per batch, not per token/answer. Library timeouts remain per answer.
    try:
        command = verifier_command() + (["--exclude-errors"] if exclude_errors else [])
        result = subprocess.run(command, input=json.dumps(rows), text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
                                timeout=15 + (16 if rule else 8) * len(rows), cwd=ROOT, env=env)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Isolated verifier failed:\n{exc.stderr[-4000:]}") from exc
    except subprocess.TimeoutExpired:
        if not exclude_errors:
            raise
        if len(rows) == 1:
            return [dict(score=None, verifier_status='worker_timeout', extracted='', verifier_error='TimeoutExpired')]
        # Isolate the slow answer; don't discard the rest of its batch.
        return [compute_score([source], [text], [gold], exclude_errors=True, rule=rule)[0]
                for source, text, gold in zip(data_sources, solution_strs, ground_truths)]
    scores = json.loads(result.stdout)
    if len(scores) != len(rows):
        raise RuntimeError("Verifier returned an incomplete batch")
    return scores
