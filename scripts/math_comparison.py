"""Run and report matched PPO, PPO-ridge, PPO-LSTD(.99) and GRPO on MATH."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from math_rl.prompts import make_prompt
from math_rl.provenance import sha256, snapshot, write_json
from math_rl.reward import parse_number
from evaluate_pilots import paired_counts

METHODS = ('ppo', 'ridge', 'lstd', 'grpo')


def validate_questions(questions):
    ids, wording = set(), set()
    for q in questions:
        text = ''.join(unicodedata.normalize('NFKC', q['question']).split()).casefold()
        if q['id'] in ids or text in wording:
            raise ValueError('Duplicate question ID/wording across MATH splits')
        ids.add(q['id'])
        wording.add(text)
        if (q['split'] not in ('train', 'val', 'test') or q['level'] not in (4, 5)
                or parse_number(q['ground_truth']) is None or not q['id'].startswith('math/')):
            raise ValueError('Expected saved MATH levels 4/5 numeric questions')
        expected = 'test' if q['split'] == 'test' else 'train'
        if q['official_split'] != expected:
            raise ValueError('Official test questions cannot enter training/validation')
    counts = Counter(q['split'] for q in questions)
    if counts['train'] < 16 or not counts['val'] or not counts['test']:
        raise ValueError('Insufficient saved train/validation/test questions')
    return dict(counts)


def prepare(source, out):
    """Reuse the question split, never the old trajectories or frozen-policy values."""
    from datasets import Dataset
    from transformers import AutoTokenizer
    from math_rl.prompts import encode_completion

    source = source.resolve()
    manifest_path = source / 'data/manifest.json'
    question_path = source / 'data/questions.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest['config']['protocol'] != 'frozen-math-critics-v1':
        raise ValueError('Pass the completed MATH-specific fitting output directory')
    if manifest['questions_sha256'] != sha256(question_path) or not (source / 'report.txt').is_file():
        raise ValueError('Incomplete MATH fit or changed question snapshot')
    saved = json.loads(question_path.read_text())
    questions = saved['questions']
    counts = validate_questions(questions)
    if saved.get('reserved_math500_questions') != 500:
        raise ValueError('Expected the MATH-500 exclusion in the saved split')
    model_hashes = {}
    for file, digest in manifest['files'].items():
        if '/models/qwen-math/' in file:
            local = ROOT / 'models/qwen-math' / Path(file).name
            if sha256(local) != digest:
                raise ValueError(f'Base model changed since the offline fit: {local.name}')
            model_hashes[local.name] = digest
    if not any(name.endswith('.safetensors') for name in model_hashes):
        raise ValueError('Offline source does not identify base model weights')
    tokenizer = AutoTokenizer.from_pretrained(ROOT / 'models/qwen-math', local_files_only=True)
    data = out / 'data'
    data.mkdir()
    for split in ('train', 'val', 'test'):
        rows = []
        for q in questions:
            if q['split'] != split:
                continue
            _, tokens = encode_completion(tokenizer, q['question'])
            if not 0 < len(tokens) <= 512:
                raise ValueError('Saved question no longer fits the prompt contract')
            rows.append(dict(data_source='math_numeric', prompt=make_prompt(q['question']), ability='math',
                reward_model=dict(style='rule', ground_truth=q['ground_truth']),
                extra_info=dict(index=q['source_index'], prompt_id=q['id'], split=split, level=q['level'])))
        Dataset.from_list(rows).to_parquet(str(data / f'{split}.parquet'))
    (data / 'questions.json').write_bytes(question_path.read_bytes())
    return dict(source=str(source), source_manifest_sha256=sha256(manifest_path),
        questions_sha256=sha256(question_path), counts=counts, model_hashes=model_hashes,
        data_hashes={p.name: sha256(p) for p in data.glob('*.parquet')},
        scope='Same inspected MATH question splits; exploratory online comparison, not a new untouched benchmark',
        offline_heads='Not loaded: online heads are refitted; feature and replay settings are in this manifest')


def command(out, method, seed, steps, feature_mode='actor', replay_capacity=0,
            replay_max_age=4, replay_max_transitions=131072, ppo_profile='legacy',
            evaluation_split='test', case_name=None):
    from math_rl.ppo_baseline import PPO_PROFILES
    case = out / (case_name or f'{method}-seed{seed}')
    extras = []
    if method == 'ppo' and ppo_profile != 'legacy':
        profile = PPO_PROFILES[ppo_profile]
        extras = [f'experiment.ppo_profile={ppo_profile}',
            f'critic.optim.lr={profile["lr"]}', f'critic.ppo_epochs={profile["epochs"]}',
            'critic.loss_agg_mode=seq-mean-token-mean', 'critic.use_dynamic_bsz=false']
    return [sys.executable, '-m', 'math_rl.math_main',
        f'experiment.method={method}',
        f'actor_rollout_ref.model.critic_feature_mode={feature_mode if method in ("lstd", "ridge") else "actor"}',
        f'experiment.replay_capacity={replay_capacity if method in ("lstd", "ridge") else 0}',
        f'experiment.replay_max_age={replay_max_age}',
        f'experiment.replay_max_transitions={replay_max_transitions}',
        f'algorithm.adv_estimator={"grpo" if method == "grpo" else "gae"}',
        f'data.train_files={out / "data/train.parquet"}', f'data.val_files={out / "data/val.parquet"}',
        f'experiment.test_file={out / ("data/" + evaluation_split + ".parquet")}',
        f'experiment.evaluation_split={evaluation_split}', f'data.seed={seed}',
        f'trainer.total_training_steps={steps}', f'trainer.save_freq={steps}',
        f'trainer.default_local_dir={case}', f'trainer.experiment_name=math-{method}-seed{seed}',
        f'hydra.run.dir={out / (case.name + "-hydra")}'] + extras


def read_answers(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Empty or duplicate test answers')
    for r in rows:
        if r['score'] not in (0, 1) or r['score'] != int(r['verifier_status'] == 'correct'):
            raise ValueError('Invalid evaluation score/status')
    return rows


def report(out):
    manifest = json.loads((out / 'manifest.json').read_text())
    records, comparisons, missing = {}, {}, []
    expected = [q for q in json.loads((out / 'data/questions.json').read_text())['questions'] if q['split'] == 'test']
    for seed in manifest['seeds']:
        final = {}
        initial = {}
        for method in manifest['methods']:
            name = f'{method}-seed{seed}'
            case = out / name
            if not (case / 'result.json').exists():
                missing.append(name)
                continue
            record = json.loads((case / 'result.json').read_text())
            if record['updates'] != manifest['steps'] or record['answers'] != manifest['steps'] * 64:
                raise ValueError('Unequal or incomplete training budgets')
            base, answers = (read_answers(case / f'{label}-test.jsonl') for label in ('base', 'final'))
            for rows in (base, answers):
                if [(r['id'], r['ground_truth']) for r in rows] != [(q['id'], q['ground_truth']) for q in expected]:
                    raise ValueError('Different test IDs, order or references')
            record['versus_base'] = paired_counts([r['score'] for r in base], [r['score'] for r in answers])
            record['verifier_statuses'] = dict(Counter(r['verifier_status'] for r in answers))
            record['per_level'] = {str(level): dict(questions=sum(r['level'] == level for r in answers),
                accuracy=sum(r['score'] for r in answers if r['level'] == level) / sum(r['level'] == level for r in answers))
                for level in (4, 5) if any(r['level'] == level for r in answers)}
            log = (out / f'{name}.log').read_text()
            memory = [float(v) for v in re.findall(r'perf/max_memory_allocated_gb:([0-9.eE+-]+)', log)]
            record['logged_max_memory_allocated_gb'] = max(memory) if memory else None
            fit_log = case / 'linear-critic/metrics.jsonl'
            if method in ('lstd', 'ridge'):
                fits = [json.loads(line) for line in fit_log.read_text().splitlines()]
                if [r['step'] for r in fits] != list(range(1, manifest['steps'] + 1)):
                    raise ValueError('Missing per-iteration linear critic fits')
                record['critic_fit_seconds'] = sum(r['fit_seconds'] for r in fits)
                record['last_out_of_fold_brier'] = fits[-1]['out_of_fold_brier']
            records[name] = record
            final[method] = [r['score'] for r in answers]
            initial[method] = [r['score'] for r in base]
        comparisons[str(seed)] = {f'{method}_versus_{baseline}': paired_counts(final[baseline], final[method])
            for baseline in ('ppo', 'grpo') for method in ('ridge', 'lstd')
            if method in final and baseline in final}
        if 'lstd' in final and 'ridge' in final:
            comparisons[str(seed)]['lstd_versus_ridge'] = paired_counts(final['ridge'], final['lstd'])
        comparisons[str(seed)]['identical_base_scores'] = all(
            scores == next(iter(initial.values())) for scores in initial.values()) if initial else None
    summary = dict(runs=records, comparisons=comparisons, missing=missing, complete=not missing,
        scope=manifest['data']['scope'],
        limitations='Pilot budgets; imperfect reward audit; one seed by default. Paired p-values exploratory and unadjusted. Offline prediction quality is not PPO evidence.')
    write_json(out / 'summary.json', summary)
    lines = ['MATH levels 4/5 numeric subset: online actor training',
        f"Linear critic settings: {manifest.get('linear_critic', {'feature_mode': 'actor', 'replay_capacity': 0})}",
        f"{manifest['steps']} updates x 64 answers per method/seed; same 2 GPUs, reward, prompts and decoding.",
        'method/seed          base accuracy   final accuracy   change (pp)   training seconds   GPU hours']
    for name, r in records.items():
        lines.append(f"{name:<20} {r['initial_test']['accuracy']:.3f}           {r['final_test']['accuracy']:.3f}"
            f"            {r['versus_base']['accuracy_change_percentage_points']:+.2f}"
            f"          {r['training_seconds']:.1f}              {r['training_gpu_hours']:.3f}")
    lines += ['Training cost includes feature capture/transport, fitting, validation and checkpoints.',
              'See summary.json for paired comparisons, level results, truncation and verifier statuses.',
              summary['scope'], summary['limitations']]
    if missing:
        lines.append('INCOMPLETE: ' + ', '.join(missing))
    (out / 'report.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    return summary


def run(args):
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    state = dict(state='preparing', runs={})
    write_json(out / 'status.json', state)
    try:
        data = prepare(args.source, out)
        replay_options = dict(feature_mode=getattr(args, 'critic_feature_mode', 'actor'),
            replay_capacity=getattr(args, 'critic_replay_capacity', 0),
            replay_max_age=getattr(args, 'critic_replay_max_age', 4),
            replay_max_transitions=getattr(args, 'critic_replay_max_transitions', 131072))
        manifest = dict(linear_critic=replay_options, protocol='math-online-v1', data=data, seeds=args.seeds, methods=args.methods,
            steps=args.steps, critic_lambda=.99, critic_alpha=.01, actor_gae_lambda=.95,
            answers_per_update=64, provenance=snapshot(ROOT))
        write_json(out / 'manifest.json', manifest)
        for seed in args.seeds:
            for method in args.methods:
                name = f'{method}-seed{seed}'
                state.update(state='running', current=name)
                write_json(out / 'status.json', state)
                started = time.perf_counter()
                env = dict(os.environ, RAY_ADDRESS='local', MATH_RL_SEED=str(seed),
                           RAY_TMPDIR=f'/tmp/mrl-{os.environ.get("SLURM_JOB_ID", os.getpid())}-{method}-{seed}')
                Path(env['RAY_TMPDIR']).mkdir(exist_ok=True)
                with (out / f'{name}.log').open('x') as log:
                    result = subprocess.run(command(out, method, seed, args.steps, **replay_options), cwd=ROOT, env=env,
                                            stdout=log, stderr=subprocess.STDOUT)
                state['runs'][name] = dict(exit_code=result.returncode, process_seconds=time.perf_counter() - started)
                write_json(out / 'status.json', state)
                # Independent methods continue even when one fails; never report a
                # partial comparison as successful or hide the failed method.
        summary = report(out)
        success = summary['complete'] and all(r['exit_code'] == 0 for r in state['runs'].values())
        state['state'] = 'complete' if success else 'failed'
        write_json(out / 'status.json', state)
        if not success:
            raise RuntimeError(f'Comparison incomplete; inspect {out / "status.json"}')
    except BaseException as exc:
        state.update(state='failed', error=f'{type(exc).__name__}: {exc}')
        write_json(out / 'status.json', state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path, help='Completed outputs/math-fit-... directory')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--seeds', nargs='+', type=int, default=[42])
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    parser.add_argument('--critic-feature-mode', choices=['actor', 'frozen'], default='actor')
    parser.add_argument('--critic-replay-capacity', type=int, default=0, help='FIFO capacity in complete answers; zero disables replay')
    parser.add_argument('--critic-replay-max-age', type=int, default=4, help='Maximum policy-update age')
    parser.add_argument('--critic-replay-max-transitions', type=int, default=131072)
    parser.add_argument('--report', action='store_true')
    args = parser.parse_args()
    if args.steps < 1 or len(set(args.seeds)) != len(args.seeds) or len(set(args.methods)) != len(args.methods):
        parser.error('Use positive steps and unique seeds/methods')
    if args.critic_replay_capacity < 0 or (args.critic_replay_capacity and
            (args.critic_feature_mode != 'frozen' or args.critic_replay_capacity < 64)):
        parser.error('Replay requires frozen features and capacity >= 64')
    if args.critic_replay_max_age < 0 or args.critic_replay_max_transitions < 131072:
        parser.error('Use nonnegative age and at least 131072 transitions')
    report(args.out) if args.report else run(args)


if __name__ == '__main__':
    main()
