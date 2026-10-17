"""Read a completed GRPO pilot's logs and answers without GPU dependencies."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
from statistics import mean


KEYS = (
    'actor/lr', 'actor/grad_norm', 'actor/ppo_kl', 'actor/pg_clipfrac',
    'actor/entropy', 'critic/score/mean', 'response_length/mean',
    'response_length/clip_ratio',
)
NUMBER = r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'


def log_metrics(text):
    rows = []
    for line in text.splitlines():
        line = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', line)
        if 'actor/grad_norm' not in line:
            continue
        row = {}
        for key in KEYS:
            match = re.search(re.escape(key) + r'\s*[:=]\s*(' + NUMBER + ')', line)
            if match:
                row[key] = float(match[1])
        if row:
            rows.append(row)
    return rows


def answer_changes(before, after):
    if not before or len(before) != len(after):
        raise ValueError('Empty or unequal evaluation answer counts')
    if len({r['id'] for r in before}) != len(before):
        raise ValueError('Duplicate evaluation IDs')
    counts = Counter()
    for a, b in zip(before, after):
        if (a['id'], a['ground_truth']) != (b['id'], b['ground_truth']):
            raise ValueError('Unmatched evaluation answers')
        for row in (a, b):
            if row['score'] not in (0, 1) or row['score'] != int(row['verifier_status'] == 'correct'):
                raise ValueError('Invalid evaluation score/status')
        counts[int(a['score']), int(b['score'])] += 1
    return counts


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def diagnose(out):
    case = out / 'grpo-seed301'
    result = case / 'result.json'
    if not result.exists():
        raise ValueError('Pilot result missing; inspect training and Slurm logs first')
    record = json.loads(result.read_text())
    print(f"Completed {record['method']}: {record['updates']} updates, {record['answers']} answers")
    rows = log_metrics((out / 'grpo-seed301.log').read_text())
    print(f'\nTraining metric lines found: {len(rows)} (expected 20)')
    print('Metric                         first <=5      last <=5       min        max')
    for key in KEYS:
        first = [r[key] for r in rows[:5] if key in r]
        last = [r[key] for r in rows[-5:] if key in r]
        values = [r[key] for r in rows if key in r]
        if first and last:
            print(f'{key:<30} {mean(first):>10.6g} {mean(last):>12.6g}'
                  f' {min(values):>10.6g} {max(values):>10.6g}')
        else:
            print(f'{key:<30} unavailable')
    print('Logged values may be rounded; nonzero gradients do not prove parameter updates.')

    before = read_rows(case / 'base-validation-test.jsonl')
    after = read_rows(case / 'final-validation-test.jsonl')
    flips = answer_changes(before, after)
    print(f'\nValidation questions: {len(before)}')
    print(f'Wrong -> correct: {flips[0, 1]}; correct -> wrong: {flips[1, 0]}')
    print(f'Correct both times: {flips[1, 1]}; wrong both times: {flips[0, 0]}')
    for label, answers in [('Before', before), ('After', after)]:
        print(f'{label}: mean tokens={mean(r["response_tokens"] for r in answers):.1f}; '
              f'capped={sum(bool(r["capped"]) for r in answers)}/{len(answers)}; '
              f'verifier statuses={dict(Counter(r["verifier_status"] for r in answers))}')

    # VERL generation dumps identify repeated prompts by their input text.
    # Group inside each update file, never across different training steps.
    mixed = total = skipped = 0
    files = sorted((case / 'rollouts').glob('*.jsonl'))
    for path in files:
        groups = defaultdict(list)
        dump = read_rows(path)
        if not dump or any(not isinstance(r.get('input'), str) or r.get('score') not in (0, 1) for r in dump):
            skipped += 1
            continue
        for row in dump:
            groups[row['input']].append(row['score'])
        if any(len(scores) != 4 for scores in groups.values()):
            skipped += 1
            continue
        total += len(groups)
        mixed += sum(len(set(scores)) > 1 for scores in groups.values())
    if total:
        print(f'\nMixed-reward groups: {mixed}/{total} ({100*mixed/total:.1f}%) '
              f'in recognized rollout files; skipped files: {skipped}')
    else:
        print('\nMixed-reward groups unavailable: no recognized four-answer rollout dumps.')
    print('No jobs submitted and no training files modified.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('outputs/confidence-grpo-neutral-v1'))
    args = parser.parse_args()
    try:
        diagnose(args.out)
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(1, f'Diagnostic failed: {exc}\n')
