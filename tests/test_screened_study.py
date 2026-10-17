import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from math_rl.study_checks import evaluation_seed, group_diagnostics, sampled_evaluation

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('gsm8k_study', ROOT / 'scripts/gsm8k_study.py')
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def test_validation_is_reserved_before_screening_and_pools_are_disjoint():
    examples = [dict(question=f'How much is {i} plus one?', answer=f'#### {i+1}') for i in range(20)]
    examples.append(dict(question=' HOW much is 0 plus one? ', answer='#### 1'))
    p = dict(study.PROTOCOL, val_count=4, candidate_limit=16)
    val, candidates = study.choose_pools(examples, lambda q: True, p)
    assert len(val) == 4 and len(candidates) == 16
    assert len({study.normalize(q['question']) for q in val+candidates}) == 20
    assert set(q['id'] for q in val).isdisjoint(q['id'] for q in candidates)
    assert (val, candidates) == study.choose_pools(examples, lambda q: True, p)
    with pytest.raises(ValueError, match='eligible'):
        study.choose_pools(examples, lambda q: False, p)


@pytest.mark.parametrize('correct,expected', [(0, False), (1, False), (2, True), (6, True), (7, False), (8, False)])
def test_screening_boundaries(correct, expected):
    scores = [dict(score=int(i < correct), verifier_status='correct' if i < correct else 'incorrect') for i in range(8)]
    assert study.qualifies(scores) is expected
    scores[-1] = dict(score=0, verifier_status='parse_failure')
    assert not study.qualifies(scores)


def test_incomplete_screening_and_score_mismatches_fail():
    with pytest.raises(ValueError, match='Incomplete'):
        study.qualifies([])
    with pytest.raises(ValueError, match='mismatch'):
        study.qualifies([dict(score=1, verifier_status='incorrect')] * 8)


def test_final_tasks_use_fresh_matched_seeds_and_rotate_order(tmp_path):
    tasks = study.full_tasks()
    assert len(set(tasks)) == 15
    assert set(study.PROTOCOL['seeds']).isdisjoint(study.PROTOCOL['pilot_seeds'])
    assert tasks[:3] == [('neutral', 301), ('confident', 301), ('critical', 301)]
    assert tasks[3:6] == [('confident', 302), ('critical', 302), ('neutral', 302)]
    cmd = study.run_command(tmp_path, 'confident', 302)
    assert 'algorithm.adv_estimator=grpo' in cmd
    assert 'trainer.total_training_steps=40' in cmd
    assert 'experiment.sampled_train_probe_n=8' in cmd
    assert f'experiment.neutral_test_file={tmp_path}/neutral/data/val.parquet' in cmd
    assert 'actor_rollout_ref.actor.optim.lr=1e-6' in cmd
    smoke = study.run_command(tmp_path, 'neutral', 19, smoke=True)
    assert 'trainer.total_training_steps=2' in smoke
    assert f'experiment.neutral_test_file={tmp_path}/smoke/data/val.parquet' in smoke


def evidence(greedy=1, sampled=1):
    return dict(audits=dict(complete=True, positive_lr=True, nonzero_gradients=True, policy_change_proxy=True),
                comparisons=dict(practice=dict(gain_pp=greedy), sampled_practice=dict(gain_pp=sampled)))


def test_gate_requires_consistent_greedy_gain_and_mean_sampled_gain():
    assert study.learning_gate([evidence(), evidence(), evidence(sampled=-.1)])['passed']
    assert not study.learning_gate([evidence(), evidence(), evidence(greedy=0)])['passed']
    assert not study.learning_gate([evidence(sampled=-1)] * 3)['passed']
    bad = evidence()
    bad['audits']['positive_lr'] = False
    assert not study.learning_gate([evidence(), evidence(), bad])['passed']
    with pytest.raises(ValueError, match='three'):
        study.learning_gate([evidence()])


def test_interleaved_groups_and_distinct_per_sample_seeds():
    ids = ['a', 'b', 'c'] * 4
    scores = [0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0]
    assert group_diagnostics(ids, scores) == dict(groups=3, mixed=1, all_correct=1, all_incorrect=1)
    with pytest.raises(ValueError, match='Incomplete'):
        group_diagnostics(ids[:-1], scores[:-1])
    seeds = [evaluation_seed('q1', i, 73129) for i in range(8)]
    assert len(set(seeds)) == 8
    assert seeds[0] != evaluation_seed('q2', 0, 73129)
    assert seeds[0] != evaluation_seed('q1', 0, 9181)


def test_sampled_evaluation_restores_engine_and_config_after_failure():
    seen = []
    def generate(*args, **kwargs):
        seen.append(kwargs['sampling_params'])
        raise RuntimeError('simulated generation failure')
    engine = SimpleNamespace(generate=generate)
    val = dict(temperature=0., top_p=.9, top_k=50)
    rollout = SimpleNamespace(config=SimpleNamespace(val_kwargs=val), inference_engine=engine)
    common = SimpleNamespace(seed=301, n=4)
    with pytest.raises(RuntimeError, match='simulated'):
        with sampled_evaluation(rollout, [12, 13]):
            assert val == dict(temperature=1., top_p=1., top_k=-1)
            engine.generate(prompts=['same prompt', 'same prompt'], sampling_params=common)
    assert [p.seed for p in seen[0]] == [12, 13]
    assert all(p.n == 1 for p in seen[0])
    assert common.seed == 301 and common.n == 4
    assert engine.generate is generate
    assert val == dict(temperature=0., top_p=.9, top_k=50)


def test_bootstrap_constant_and_paired_null():
    assert study.bootstrap([[0]*8]*5, draws=100)['interval_97_5_pp'] == [0., 0.]
    assert study.bootstrap([[.1]*8]*5, draws=100)['mean_pp'] == pytest.approx(10)


def test_submission_dependency_chain_and_no_duplicate_writers(tmp_path):
    repo = tmp_path / 'repo'
    (repo / 'scripts').mkdir(parents=True)
    (repo / 'outputs').mkdir()
    (repo / '.runtime-ready').touch()
    shutil.copyfile(ROOT / 'scripts/gsm8k_study.sh', repo / 'scripts/gsm8k_study.sh')
    binary = tmp_path / 'bin'
    binary.mkdir()
    log = tmp_path / 'calls'
    fake = binary / 'sbatch'
    fake.write_text('#!/usr/bin/env python3\nimport os,sys,json\nfrom pathlib import Path\n'
                    'p=Path(os.environ["CALLS"])\nrows=p.read_text().splitlines() if p.exists() else []\n'
                    'with p.open("a") as f: f.write(json.dumps(sys.argv[1:])+"\\n")\nprint(100+len(rows))\n')
    fake.chmod(0o755)
    env = dict(os.environ, PATH=str(binary)+':'+os.environ['PATH'], CALLS=str(log))
    cmd = ['bash', 'scripts/gsm8k_study.sh', 'pilot']
    subprocess.run(cmd, cwd=repo, env=env, check=True, capture_output=True)
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(calls) == 4
    assert '--gres=gpu:a5000:1' in calls[0]
    assert '--dependency=afterok:100' in calls[1]
    assert '--gres=gpu:a5000:2' in calls[1]
    assert '--dependency=afterok:101' in calls[2] and '--array=0-2%2' in calls[2]
    assert '--dependency=afterok:102' in calls[3] and '--partition=compsci' in calls[3]
    assert not any('rl_ms_project' in arg for call in calls for arg in call)
    assert subprocess.run(cmd, cwd=repo, env=env, capture_output=True).returncode != 0
    assert len(log.read_text().splitlines()) == 4
    # A failed gate cannot submit the full array.
    (repo / 'scripts/gsm8k_study.py').write_text('raise SystemExit(2)\n')
    full = ['bash', 'scripts/gsm8k_study.sh', 'full']
    assert subprocess.run(full, cwd=repo, env=env, capture_output=True).returncode != 0
    assert len(log.read_text().splitlines()) == 4
    # A passed gate submits exactly 15 fresh runs and a dependent report.
    (repo / 'scripts/gsm8k_study.py').write_text('pass\n')
    subprocess.run(full, cwd=repo, env=env, check=True, capture_output=True)
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert '--array=0-14%2' in calls[4]
    assert '--dependency=afterok:104' in calls[5]
    assert subprocess.run(full, cwd=repo, env=env, capture_output=True).returncode != 0
    assert len(log.read_text().splitlines()) == 6


def test_sampled_answers_reject_reused_or_mismatched_seeds(tmp_path):
    q = dict(id='q1', ground_truth='4')
    (tmp_path / 'questions.json').write_text(json.dumps(dict(train=[q])))
    case = tmp_path / 'neutral/grpo-seed201'
    case.mkdir(parents=True)
    rows = [dict(**q, score=1, verifier_status='correct', sample_index=i,
                 sampling_seed=evaluation_seed('q1', i, 73129)) for i in range(8)]
    path = case / 'base-sampled-training-probe-test.jsonl'
    def save():
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    save()
    assert len(study.answers(tmp_path, 'neutral', 201, 'base-sampled-training-probe', 'train', 8)) == 8
    rows[-1]['sampling_seed'] = rows[0]['sampling_seed']
    save()
    with pytest.raises(ValueError, match='seeds'):
        study.answers(tmp_path, 'neutral', 201, 'base-sampled-training-probe', 'train', 8)


def test_saved_run_gate_and_incomplete_budgets(tmp_path):
    """Exercise actual artifact pairing and the gate, not just gate booleans."""
    from math_rl.study_checks import evaluation_seed
    questions = [dict(id=f'q{i}', ground_truth='4') for i in range(4)]
    (tmp_path / 'questions.json').write_text(json.dumps(dict(train=questions, val=questions)))
    (tmp_path / 'manifest.json').write_text(json.dumps(dict(code_hashes={})))

    for seed in study.PROTOCOL['pilot_seeds']:
        case = tmp_path / 'neutral' / f'grpo-seed{seed}'
        case.mkdir(parents=True)
        result = dict(method='grpo', seed=seed, updates=40, answers=2560,
                      evaluation_split='val', training_gpu_hours=1.)
        (case / 'result.json').write_text(json.dumps(result))
        config = dict(actor_rollout_ref=dict(actor=dict(optim=dict(lr=1e-6))),
                      experiment=dict(sampled_train_probe_n=8, sampled_train_probe_seed=73129))
        (case / 'run.json').write_text(json.dumps(dict(config=config, provenance=dict(files={}))))
        metrics = [dict(step=i, metrics={'actor/lr': 1e-6, 'actor/grad_norm': .3,
                                       'actor/ppo_kl': .00001}) for i in range(1, 41)]
        groups = [dict(step=i, groups=16, mixed=8, all_correct=4, all_incorrect=4)
                  for i in range(1, 41)]
        for name, rows in [('training-metrics', metrics), ('grpo-groups', groups)]:
            (case / f'{name}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        for label, n in [('training-probe', 1), ('sampled-training-probe', 8),
                         ('validation-neutral', 1), ('validation', 1)]:
            for stage in ('base', 'final'):
                rows = []
                for q in questions:
                    for index in range(n):
                        score = int(stage == 'final')
                        r = dict(**q, score=score, verifier_status='correct' if score else 'incorrect',
                                 response_tokens=10, capped=False)
                        if n > 1:
                            r.update(sample_index=index, sampling_seed=evaluation_seed(q['id'], index, 73129))
                        rows.append(r)
                (case / f'{stage}-{label}-test.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))

    assert study.gate(tmp_path, require_pass=True)['passed']
    case = tmp_path / 'neutral/grpo-seed201'
    r = study.read(case / 'result.json')
    r['answers'] = 1280
    (case / 'result.json').write_text(json.dumps(r))
    with pytest.raises(ValueError, match='mismatched run'):
        study.gate(tmp_path, require_pass=True)
