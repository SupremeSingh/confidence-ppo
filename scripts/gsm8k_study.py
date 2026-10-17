"""Screen GSM8K once, gate neutral GRPO learning, then compare prompt styles."""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from math_rl.provenance import sha256, snapshot, write_json
from math_rl.prompts import make_prompt
from math_rl.reward import parse_number

STYLES = {
    'neutral': 'Work through the problem carefully.',
    'confident': 'Express confidence in your reasoning while working through the problem. Check calculations and correct mistakes when needed.',
    'critical': 'Examine your reasoning critically while working through the problem. Check calculations and correct mistakes when needed.',
}
PROTOCOL = dict(
    name='gsm8k-screened-grpo-v1', train_count=128, val_count=512,
    candidate_limit=512, screening_n=8, min_correct=2, max_correct=6,
    screening_seed=9181, selection_seed=31031, sampled_probe_seed=73129,
    sampled_probe_n=8, steps=40, answers_per_update=64, actor_lr=1e-6,
    pilot_seeds=[201, 202, 203], seeds=[301, 302, 303, 304, 305],
    temperature=1.0, top_p=1.0, top_k=-1, max_prompt_tokens=512, max_response_tokens=2048,
    bootstrap_draws=5000,
    gate='Complete finite audits, positive actor LR and gradients, a nonzero policy-change proxy; '
         'positive greedy practice gain in each of 3 seeds and positive mean independent sampled practice gain. '
         'No validation-accuracy gate. This is an optimization check, not significance or generalization.',
    scope='Intermediate observed neutral success selected with 8 samples; noisy screening; '
          'validation is a development set; official test untouched; exploratory comparisons.',
)


def read(path):
    return json.loads(Path(path).read_text())


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def normalize(text):
    return ''.join(unicodedata.normalize('NFKC', text).split()).casefold()


def choose_pools(examples, fits, protocol=PROTOCOL):
    """Reserve random validation BEFORE any outcome-based training selection."""
    order = list(range(len(examples)))
    random.Random(protocol['selection_seed']).shuffle(order)
    eligible, seen = [], set()
    needed = protocol['val_count'] + protocol['candidate_limit']
    for index in order:
        example = examples[index]
        key = normalize(example['question'])
        if key in seen:
            continue
        _, marker, answer = example['answer'].rpartition('####')
        if not marker or parse_number(answer.strip()) is None:
            raise ValueError(f'Invalid GSM8K reference at source row {index}')
        if not all(fits(instruction + '\n\n' + example['question']) for instruction in STYLES.values()):
            continue
        seen.add(key)
        eligible.append(dict(id=f'gsm8k/train/{index}', source_index=index,
                             question=example['question'], ground_truth=answer.strip()))
        if len(eligible) == needed:
            break
    if len(eligible) != needed:
        raise ValueError(f'Need {needed} eligible distinct questions; found {len(eligible)}')
    cut = protocol['val_count']
    return eligible[:cut], eligible[cut:]


def qualifies(scores, protocol=PROTOCOL):
    if len(scores) != protocol['screening_n']:
        raise ValueError('Incomplete screening group')
    valid = all(r['verifier_status'] in ('correct', 'incorrect') for r in scores)
    if any(r['score'] != int(r['verifier_status'] == 'correct') for r in scores):
        raise ValueError('Screening score/status mismatch')
    return valid and protocol['min_correct'] <= sum(r['score'] for r in scores) <= protocol['max_correct']


def full_tasks(protocol=PROTOCOL):
    return [(list(STYLES)[(j + i) % 3], seed)
            for i, seed in enumerate(protocol['seeds']) for j in range(3)]


def dataset_rows(questions, style, split):
    return [dict(data_source='gsm8k', prompt=make_prompt(STYLES[style] + '\n\n' + q['question']),
                 ability='math', reward_model=dict(style='rule', ground_truth=q['ground_truth']),
                 extra_info=dict(index=q['source_index'], prompt_id=q['id'], split=split, level=0))
            for q in questions]


def prepare(out):
    from datasets import Dataset, load_dataset
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from math_rl.prompts import encode_completion
    from math_rl.ppo_reward import compute_score

    if (out / 'screening-answers.jsonl').exists() or (out / 'manifest.json').exists():
        raise ValueError('Preparation already started; use a fresh output directory')
    started = time.perf_counter()
    code_hashes = snapshot(ROOT)['files']
    assets = read(ROOT / 'inputs/assets.json')
    for name, checksum in assets['files'].items():
        if sha256(ROOT / name) != checksum:
            raise ValueError(f'Imported asset changed: {name}')
    revision = HfApi().dataset_info('openai/gsm8k', revision=read(out / 'request.json')['revision']).sha
    if not revision or len(revision) != 40:
        raise ValueError('Cannot resolve immutable dataset revision')
    source = load_dataset('openai/gsm8k', 'main', split='train', revision=revision)
    tokenizer = AutoTokenizer.from_pretrained(ROOT / 'models/qwen-math', local_files_only=True)
    def fits(question):
        return 0 < len(encode_completion(tokenizer, question)[1]) <= PROTOCOL['max_prompt_tokens']
    validation, candidates = choose_pools(source, fits)
    write_json(out / 'pools.json', dict(revision=revision, validation=validation, candidates=candidates,
        protocol=PROTOCOL, styles=STYLES, official_test_used=False))
    # Reserve and save validation before loading the generator or grading any candidate.
    llm = LLM(model=str(ROOT / 'models/qwen-math'), dtype='bfloat16', tensor_parallel_size=1,
              gpu_memory_utilization=.5, enforce_eager=True, max_model_len=2560,
              max_num_batched_tokens=2560, max_num_seqs=8, seed=PROTOCOL['screening_seed'])
    selected, statuses, screened, sample_count = [], Counter(), 0, 0
    with (out / 'screening-answers.jsonl').open('x') as handle:
        for start in range(0, len(candidates), 16):
            batch = candidates[start:start + 16]
            prompts = [dict(prompt_token_ids=encode_completion(tokenizer, STYLES['neutral'] + '\n\n' + q['question'])[1]) for q in batch]
            params = [SamplingParams(n=8, temperature=1.0, top_p=1.0, top_k=-1, max_tokens=2048,
                                     seed=PROTOCOL['screening_seed'] + start + i) for i in range(len(batch))]
            generated = llm.generate(prompts=prompts, sampling_params=params, use_tqdm=False)
            if len(generated) != len(batch) or any(len(g.outputs) != 8 for g in generated):
                raise ValueError('Incomplete screening generation')
            texts = [s.text for g in generated for s in g.outputs]
            gold = [q['ground_truth'] for q in batch for _ in range(8)]
            grades = compute_score(['gsm8k'] * len(texts), texts, gold)
            for i, (q, generation) in enumerate(zip(batch, generated)):
                scores = grades[i * 8:(i + 1) * 8]
                accepted = qualifies(scores)
                chosen = accepted and len(selected) < PROTOCOL['train_count']
                if chosen:
                    selected.append(q)
                for j, (sample, grade) in enumerate(zip(generation.outputs, scores)):
                    row = dict(id=q['id'], sample_index=j, ground_truth=q['ground_truth'], response=sample.text,
                               response_tokens=len(sample.token_ids),
                               capped=len(sample.token_ids) == 2048 and bool(sample.token_ids)
                                   and sample.token_ids[-1] != tokenizer.eos_token_id,
                               qualifies=accepted, selected=chosen, **grade)
                    statuses[grade['verifier_status']] += 1
                    handle.write(json.dumps(row) + '\n')
                screened += 1
                sample_count += 8
            handle.flush()
            print(f'Screened {screened}/{len(candidates)}; selected {len(selected)}/128', flush=True)
            if len(selected) == PROTOCOL['train_count']:
                break
    screening = dict(candidates_screened=screened, answers=sample_count, selected=len(selected),
        verifier_statuses=dict(statuses), wall_seconds=time.perf_counter() - started,
        gpu_hours=(time.perf_counter() - started) / 3600,
        error_policy='Candidates with any non-correct/incorrect verifier status are ineligible; statuses retained.',
        stopping='Stop after first completed batch with 128 eligible questions; maximum 512 candidates.')
    write_json(out / 'screening-summary.json', screening)
    if len(selected) != PROTOCOL['train_count']:
        raise ValueError('Fewer than 128 qualified questions; do not relax thresholds silently')
    if snapshot(ROOT)['files'] != code_hashes:
        raise ValueError('Code/runtime changed during screening; use a fresh study directory')
    hashes = {}
    for style in STYLES:
        data = out / style / 'data'
        data.mkdir(parents=True)
        for split, questions in [('train', selected), ('val', validation)]:
            path = data / f'{split}.parquet'
            Dataset.from_list(dataset_rows(questions, style, split)).to_parquet(str(path))
            hashes[str(path.relative_to(out))] = sha256(path)
    smoke_data = out / 'smoke/data'
    smoke_data.mkdir(parents=True)
    for split, questions in [('train', selected[:16]), ('val', validation[:16])]:
        path = smoke_data / f'{split}.parquet'
        Dataset.from_list(dataset_rows(questions, 'neutral', split)).to_parquet(str(path))
        hashes[str(path.relative_to(out))] = sha256(path)
    write_json(out / 'questions.json', dict(train=selected, val=validation))
    write_json(out / 'manifest.json', dict(protocol=PROTOCOL, styles=STYLES, revision=revision,
        official_test_used=False, screening=screening, data_hashes=hashes,
        questions_sha256=sha256(out / 'questions.json'), pools_sha256=sha256(out / 'pools.json'),
        screening_answers_sha256=sha256(out / 'screening-answers.jsonl'),
        code_hashes=code_hashes, input_files=assets['files'],
        full_tasks=full_tasks(), scope=PROTOCOL['scope']))
    print('Screening complete. Neutral training may proceed.', flush=True)


def verify(out):
    m = read(out / 'manifest.json')
    if m['protocol'] != PROTOCOL or m['styles'] != STYLES or m['full_tasks'] != [list(x) for x in full_tasks()]:
        raise ValueError('Protocol changed; use a new study directory')
    for name, checksum in m['code_hashes'].items():
        if sha256(ROOT / name) != checksum:
            raise ValueError(f'Code/runtime changed after preparation: {name}')
    for name, checksum in m['input_files'].items():
        if sha256(ROOT / name) != checksum:
            raise ValueError(f'Imported asset changed: {name}')
    for name, checksum in m['data_hashes'].items():
        if sha256(out / name) != checksum:
            raise ValueError(f'Study data changed: {name}')
    for name, key in [('questions.json', 'questions_sha256'), ('pools.json', 'pools_sha256'),
                      ('screening-answers.jsonl', 'screening_answers_sha256')]:
        if sha256(out / name) != m[key]:
            raise ValueError(f'Frozen study artifact changed: {name}')
    return m


def run_command(out, style, seed, smoke=False):
    """All settings that differ from the pinned runtime are explicit here."""
    base = out / ('smoke' if smoke else style)
    case = base / f'grpo-seed{seed}'
    steps = 2 if smoke else 40
    neutral = base / 'data/val.parquet' if smoke else out / 'neutral/data/val.parquet'
    return [sys.executable, '-m', 'math_rl.math_main',
        'experiment.method=grpo', 'algorithm.adv_estimator=grpo', 'experiment.ppo_profile=legacy',
        'algorithm.norm_adv_by_std_in_grpo=true',
        f'data.train_files={base / "data/train.parquet"}', f'data.val_files={base / "data/val.parquet"}',
        'experiment.evaluation_split=val', f'experiment.test_file={base / "data/val.parquet"}',
        f'experiment.neutral_test_file={neutral}',
        f'experiment.train_probe_file={base / "data/train.parquet"}',
        'experiment.sampled_train_probe_n=8', 'experiment.sampled_train_probe_seed=73129',
        'experiment.audit_metrics=true', 'actor_rollout_ref.actor.optim.lr=1e-6',
        f'data.seed={seed}', f'trainer.total_training_steps={steps}', f'trainer.save_freq={steps}',
        'trainer.test_freq=20', 'trainer.val_before_train=false',
        f'trainer.default_local_dir={case}', f'trainer.experiment_name=gsm8k-{style}-seed{seed}',
        f'hydra.run.dir={base / (case.name + "-hydra")}']


def launch(out, phase, task):
    m = verify(out)
    if phase == 'neutral':
        if not read(out / 'smoke.json')['passed']:
            raise ValueError('GPU smoke check must pass first')
        tasks = [('neutral', seed) for seed in PROTOCOL['pilot_seeds']]
    else:
        gate(out, require_pass=True)
        tasks = m['full_tasks']
    if not 0 <= task < len(tasks):
        raise ValueError('Task index outside the frozen task list')
    style, seed = tasks[task]
    case = out / style / f'grpo-seed{seed}'
    if case.exists():
        raise ValueError('Run already exists; partial runs do not resume')
    env = dict(os.environ, PYTHONPATH=str(ROOT / 'src'), MATH_RL_SEED=str(seed), RAY_ADDRESS='local')
    env.pop('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE', None)
    started = time.perf_counter()
    with (out / f'{style}-grpo-seed{seed}.log').open('x') as log:
        subprocess.run(run_command(out, style, seed), cwd=ROOT, env=env,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    seconds = time.perf_counter() - started
    write_json(case / 'execution.json', dict(driver_seconds=seconds, gpu_hours=seconds * 2 / 3600,
        scope='Driver wall time including preflight, initialization, explicit evaluation and training; '
              'excludes scheduler wait and container startup.'))


def smoke(out):
    verify(out)
    with (out / 'smoke.log').open('x') as log:
        subprocess.run(run_command(out, 'neutral', 19, smoke=True), cwd=ROOT,
            env=dict(os.environ, PYTHONPATH=str(ROOT / 'src'), MATH_RL_SEED='19', RAY_ADDRESS='local'),
            stdout=log, stderr=subprocess.STDOUT, check=True)
    case = out / 'smoke/grpo-seed19'
    r = read(case / 'result.json')
    if (r['updates'], r['answers'], r['initial_sampled_train_probe']['questions'],
            r['final_sampled_train_probe']['answers']) != (2, 128, 16, 128):
        raise ValueError('GPU smoke budget or sampled evaluation incomplete')
    metrics = [r for r in jsonl(case / 'training-metrics.jsonl') if 'actor/grad_norm' in r['metrics']]
    groups = jsonl(case / 'grpo-groups.jsonl')
    if [r['step'] for r in metrics] != [1, 2] or [r['step'] for r in groups] != [1, 2]:
        raise ValueError('GPU smoke audits incomplete')
    from math_rl.ppo_checks import checkpoint_complete
    if not checkpoint_complete(case / 'global_step_2', world_size=2, roles=('actor',)):
        raise ValueError('GPU smoke actor checkpoint incomplete')
    write_json(out / 'smoke.json', dict(passed=True, updates=2, answers=128,
        checks='Sampled worker, independent sample seeds, telemetry, reward/preflight and checkpoint path'))
    print('GPU smoke check passed. Three neutral seeds may proceed.')


def answers(out, style, seed, label, split, samples=1):
    path = out / style / f'grpo-seed{seed}' / f'{label}-test.jsonl'
    rows = jsonl(path)
    expected = [(q['id'], q['ground_truth']) for q in read(out / 'questions.json')[split] for _ in range(samples)]
    if [(r['id'], r['ground_truth']) for r in rows] != expected:
        raise ValueError(f'Evaluation pairing mismatch: {path}')
    if any(r['score'] not in (0, 1) or r['score'] != int(r['verifier_status'] == 'correct') for r in rows):
        raise ValueError(f'Invalid correctness/status: {path}')
    if samples > 1 and [r['sample_index'] for r in rows] != list(range(samples)) * (len(rows) // samples):
        raise ValueError('Invalid sampled evaluation indices')
    if samples > 1:
        from math_rl.study_checks import evaluation_seed
        if any(r['sampling_seed'] != evaluation_seed(r['id'], r['sample_index'], PROTOCOL['sampled_probe_seed']) for r in rows):
            raise ValueError('Unexpected sampled evaluation seeds')
    return rows


def run_evidence(out, style, seed):
    case = out / style / f'grpo-seed{seed}'
    r = read(case / 'result.json')
    if (r['method'], r['seed'], r['updates'], r['answers'], r['evaluation_split']) != ('grpo', seed, 40, 2560, 'val'):
        raise ValueError(f'Incomplete or mismatched run: {case}')
    run = read(case / 'run.json')
    m = read(out / 'manifest.json')
    if any(run['provenance']['files'].get(k) != v for k, v in m['code_hashes'].items()):
        raise ValueError(f'Run code differs from screening protocol: {case}')
    config = run['config']
    if (config['actor_rollout_ref']['actor']['optim']['lr'] != 1e-6
            or config['experiment']['sampled_train_probe_n'] != 8
            or config['experiment']['sampled_train_probe_seed'] != PROTOCOL['sampled_probe_seed']):
        raise ValueError('Unexpected learning rate or sampled evaluation settings')
    metrics = [row for row in jsonl(case / 'training-metrics.jsonl') if 'actor/grad_norm' in row['metrics']]
    groups = jsonl(case / 'grpo-groups.jsonl')
    expected = list(range(1, 41))
    complete = [r['step'] for r in metrics] == expected and [r['step'] for r in groups] == expected
    if not complete or any(g['groups'] != 16 or sum(g[k] for k in ('mixed', 'all_correct', 'all_incorrect')) != 16 for g in groups):
        raise ValueError(f'Incomplete update/group audits: {case}')
    if any(not math.isfinite(v) for row in metrics for v in row['metrics'].values()):
        raise ValueError('Nonfinite training metrics')
    lr_positive = all(row['metrics'].get('actor/lr', 0) > 0 for row in metrics)
    gradients = any(row['metrics']['actor/grad_norm'] > 0 for row in metrics)
    movement = any(abs(row['metrics'].get('actor/ppo_kl', 0)) > 0
                   or row['metrics'].get('actor/pg_clipfrac', 0) > 0 for row in metrics)
    comparisons = {}
    for name, label, split, n in [('practice', 'training-probe', 'train', 1),
                                  ('sampled_practice', 'sampled-training-probe', 'train', 8),
                                  ('validation', 'validation-neutral', 'val', 1),
                                  ('assigned_validation', 'validation', 'val', 1)]:
        before = answers(out, style, seed, 'base-' + label, split, n)
        after = answers(out, style, seed, 'final-' + label, split, n)
        a, b = sum(x['score'] for x in before) / len(before), sum(x['score'] for x in after) / len(after)
        comparisons[name] = dict(before=a, after=b, gain_pp=(b-a)*100,
            mean_final_tokens=sum(x['response_tokens'] for x in after) / len(after),
            final_truncation=sum(x['capped'] for x in after) / len(after),
            final_statuses=dict(Counter(x['verifier_status'] for x in after)))
    return dict(comparisons=comparisons, audits=dict(complete=complete, positive_lr=lr_positive,
        nonzero_gradients=gradients, policy_change_proxy=movement),
        group_counts={k: sum(g[k] for g in groups) for k in ('groups', 'mixed', 'all_correct', 'all_incorrect')},
        groups_first_5={k: sum(g[k] for g in groups[:5]) for k in ('mixed', 'all_correct', 'all_incorrect')},
        groups_last_5={k: sum(g[k] for g in groups[-5:]) for k in ('mixed', 'all_correct', 'all_incorrect')},
        training_gpu_hours=r['training_gpu_hours'],
        metrics_first_5=metrics[:5], metrics_last_5=metrics[-5:])


def learning_gate(evidence):
    if len(evidence) != 3:
        raise ValueError('Learning gate requires all three neutral runs')
    checks = dict(
        complete_finite_active_runs=all(all(e['audits'].values()) for e in evidence),
        positive_greedy_practice_gain_every_seed=all(e['comparisons']['practice']['gain_pp'] > 0 for e in evidence),
        positive_mean_independent_sampled_practice_gain=
            sum(e['comparisons']['sampled_practice']['gain_pp'] for e in evidence) / 3 > 0)
    return dict(passed=all(checks.values()), checks=checks, criterion=PROTOCOL['gate'])


def gate(out, require_pass=False):
    evidence = {str(seed): run_evidence(out, 'neutral', seed) for seed in PROTOCOL['pilot_seeds']}
    result = learning_gate(list(evidence.values()))
    result.update(evidence=evidence, manifest_sha256=sha256(out / 'manifest.json'))
    # The full array also rechecks evidence, but only the gate job writes this file.
    if not require_pass:
        write_json(out / 'gate.json', result)
        lines = ['Neutral GRPO learning gate: ' + ('PASSED' if result['passed'] else 'FAILED'),
                 'Seed     practice greedy gain    fresh sampled gain    neutral validation gain (pp)']
        for seed, e in evidence.items():
            c = e['comparisons']
            lines.append(f'{seed} {c["practice"]["gain_pp"]:>20.2f} {c["sampled_practice"]["gain_pp"]:>21.2f} {c["validation"]["gain_pp"]:>25.2f}')
            g = e['group_counts']
            lines.append(f'  groups: mixed={g["mixed"]}/{g["groups"]}, all correct={g["all_correct"]}, all incorrect={g["all_incorrect"]}; audits={e["audits"]}')
        lines += [PROTOCOL['gate'], 'Practice improvement establishes optimization only; validation remains exploratory.']
        text = '\n'.join(lines) + '\n'
        (out / 'gate-report.txt').write_text(text)
        print(text, end='')
    if require_pass and not result['passed']:
        raise ValueError('Neutral learning gate failed; style comparison blocked')
    return result


def bootstrap(matrix, draws=5000):
    import numpy as np
    x = np.asarray(matrix, dtype=float)
    rng = np.random.default_rng(7)
    values = [x[rng.integers(x.shape[0], size=x.shape[0])][:, rng.integers(x.shape[1], size=x.shape[1])].mean()*100
              for _ in range(draws)]
    return dict(mean_pp=float(x.mean()*100), interval_95_pp=np.quantile(values, [.025, .975]).tolist(),
                interval_97_5_pp=np.quantile(values, [.0125, .9875]).tolist())


def report(out):
    pilot = gate(out, require_pass=True)
    gains, evidence, statistics = {}, {}, {}
    for style in STYLES:
        for seed in PROTOCOL['seeds']:
            evidence[f'{style}/{seed}'] = run_evidence(out, style, seed)
        for evaluation, label in [('neutral', 'validation-neutral'), ('assigned', 'validation')]:
            matrix = []
            for seed in PROTOCOL['seeds']:
                before = answers(out, style, seed, 'base-' + label, 'val')
                after = answers(out, style, seed, 'final-' + label, 'val')
                matrix.append([b['score'] - a['score'] for a, b in zip(before, after)])
            gains[style, evaluation] = matrix
            statistics[f'{style}/{evaluation}'] = bootstrap(matrix)
    contrasts = {}
    for style in ('confident', 'critical'):
        for evaluation in ('neutral', 'assigned'):
            difference = [[a-b for a, b in zip(x, y)] for x, y in
                          zip(gains[style, evaluation], gains['neutral', evaluation])]
            contrasts[f'{style}-minus-neutral/{evaluation}'] = bootstrap(difference)
    summary = dict(statistics=statistics, contrasts=contrasts, runs=evidence,
        screening=read(out / 'screening-summary.json'),
        pilot_runs=pilot['evidence'],
        training_cost_gpu_hours=dict(
            pilot=sum(e['training_gpu_hours'] for e in pilot['evidence'].values()),
            comparison=sum(e['training_gpu_hours'] for e in evidence.values())),
        primary='Held-out neutral-prompt learning-gain contrasts; 97.5% paired seed/question bootstrap intervals.',
        scope=PROTOCOL['scope'],
        cost_scope='Screening cost separate; training GPU hours exclude explicit before/after evaluation and setup.',
        probe_status='Deferred: define error labels, matched prefixes, question splits and control tasks before probing. '
                     'Group fractions are mechanism hypotheses, not causal evidence.')
    write_json(out / 'summary.json', summary)
    # Same randomly chosen question/seed pairs across styles, with no outcome
    # filtering. Save assigned-style answers for manual style/grading review.
    questions = read(out / 'questions.json')['val']
    pairs = [(seed, i) for seed in PROTOCOL['seeds'] for i in range(len(questions))]
    review_pairs = random.Random(43).sample(pairs, min(10, len(pairs)))
    review = []
    for style in STYLES:
        for seed, i in review_pairs:
            for stage in ('base', 'final'):
                rows = answers(out, style, seed, stage + '-validation', 'val')
                review.append(dict(style=style, instruction=STYLES[style], seed=seed, stage=stage,
                                   question=questions[i]['question'], **rows[i]))
    with (out / 'review.jsonl').open('w') as handle:
        for row in review:
            handle.write(json.dumps(row) + '\n')
    lines = ['Screened GSM8K GRPO style comparison (exploratory development-set results)',
             'Mean neutral-prompt learning gains (pp):']
    for style in STYLES:
        lines.append(f'  {style}: {statistics[style + "/neutral"]["mean_pp"]:+.2f}')
    for style in ('confident', 'critical'):
        c = contrasts[style + '-minus-neutral/neutral']
        lines.append(f'{style} minus neutral: {c["mean_pp"]:+.2f} pp; 97.5% interval {c["interval_97_5_pp"]}')
    lines += [PROTOCOL['scope'], 'See summary.json for assigned-style gains, audits, lengths, statuses and costs.',
              'review.jsonl contains randomly selected matched answers for manual style and grading review.']
    text = '\n'.join(lines) + '\n'
    (out / 'report.txt').write_text(text)
    print(text, end='')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['prepare', 'smoke', 'run-neutral', 'run-full', 'gate', 'full-check', 'report'])
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--task', type=int, default=0)
    a = p.parse_args()
    out = a.out.resolve()
    if a.stage == 'prepare':
        prepare(out)
    elif a.stage == 'smoke':
        smoke(out)
    elif a.stage.startswith('run-'):
        launch(out, a.stage[4:], a.task)
    elif a.stage == 'full-check':
        verify(out)
        gate(out, require_pass=True)
    elif a.stage == 'gate':
        if not gate(out)['passed']:
            p.exit(2, 'Learning gate failed. Do not launch the style comparison.\n')
    else:
        report(out)


if __name__ == '__main__':
    main()
