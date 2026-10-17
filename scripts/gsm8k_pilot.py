"""Small GSM8K GRPO learning check, with practice and held-out evaluations."""
import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from math_rl.data import make_row
from math_rl.provenance import sha256, snapshot, write_json


def select_rows(examples, token_count, train_count=32, val_count=256):
    """Deterministic, disjoint samples from official training data only."""
    indices = list(range(len(examples)))
    random.Random(301).shuffle(indices)
    seen, selected = set(), []
    for index in indices:
        example = examples[index]
        question = example['question']
        key = ''.join(unicodedata.normalize('NFKC', question).split()).casefold()
        if key in seen:
            continue
        row = make_row(example, index)
        text = 'Work through the problem carefully.\n\n' + question
        if not 0 < token_count(text) <= 512:
            continue
        row['prompt'][1]['content'] = text
        split = 'train' if len(selected) < train_count else 'val'
        # GSM8K has no MATH difficulty level; 0 means unspecified here.
        row['extra_info'].update(split=split, level=0)
        seen.add(key)
        selected.append(row)
        if len(selected) == train_count + val_count:
            break
    if len(selected) != train_count + val_count:
        raise ValueError('Insufficient distinct questions within the prompt budget')
    return selected[:train_count], selected[train_count:]


def run(out):
    from datasets import Dataset, load_dataset
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer
    from math_rl.prompts import encode_completion
    from math_comparison import command

    request = json.loads((out / 'request.json').read_text())
    revision = HfApi().dataset_info('openai/gsm8k', revision=request['revision']).sha
    if not revision or len(revision) != 40:
        raise ValueError('Cannot resolve an immutable GSM8K revision')
    examples = load_dataset('openai/gsm8k', 'main', split='train', revision=revision)
    tokenizer = AutoTokenizer.from_pretrained(ROOT / 'models/qwen-math', local_files_only=True)
    train, val = select_rows(examples, lambda q: len(encode_completion(tokenizer, q)[1]))
    data = out / 'data'
    data.mkdir()  # Refuse to overwrite an earlier preparation.
    for split, rows in [('train', train), ('val', val)]:
        Dataset.from_list(rows).to_parquet(str(data / f'{split}.parquet'))
    write_json(out / 'manifest.json', dict(
        protocol='gsm8k-grpo-learning-check-v1', dataset='openai/gsm8k', revision=revision,
        source_split='train', official_test_used=False, train_count=len(train), val_count=len(val),
        method='grpo', seed=301, steps=20, answers_per_update=64, actor_lr=1e-6,
        training_probe='Same 32 training questions; measures practice-set improvement, not generalization.',
        selection='Seeded shuffle, normalized wording deduplication and prompt-length filtering; no accuracy-based selection.',
        data_hashes={p.name: sha256(p) for p in data.glob('*.parquet')},
        selected_ids={name: [r['extra_info']['prompt_id'] for r in rows]
                      for name, rows in [('train', train), ('val', val)]},
        provenance=snapshot(ROOT)))
    cmd = command(out, 'grpo', 301, 20, evaluation_split='val') + [
        f'experiment.neutral_test_file={data / "val.parquet"}',
        f'experiment.train_probe_file={data / "train.parquet"}',
        'actor_rollout_ref.actor.optim.lr=1e-6']
    with (out / 'grpo-seed301.log').open('x') as log:
        subprocess.run(cmd, cwd=ROOT, env=dict(os.environ, PYTHONPATH=str(ROOT / 'src')),
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    report(out)


def report(out):
    path = out / 'grpo-seed301/result.json'
    if not path.exists():
        raise ValueError(f'No completed result. Inspect {out}/preparation.log, grpo-seed301.log and slurm-*.out.')
    r = json.loads(path.read_text())
    if (r['method'], r['updates'], r['answers'], r['evaluation_split']) != ('grpo', 20, 1280, 'val'):
        raise ValueError('Unexpected method, training budget or evaluation split')
    lines = ['GSM8K neutral GRPO learning check; seed 301; actor LR 1e-6',
             '20 updates; 1,280 training answers; official test set unused.']
    for label, start, end in [
        ('Practice questions (32; seen during training)', 'initial_train_probe', 'final_train_probe'),
        ('Held-out validation (256)', 'initial_validation', 'final_validation')]:
        before, after = 100 * r[start]['accuracy'], 100 * r[end]['accuracy']
        lines.append(f'{label}: {before:.2f}% -> {after:.2f}% ({after-before:+.2f} pp)')
    lines.append('Practice-set gain tests optimization; held-out gain tests generalization. One seed is diagnostic only.')
    text = '\n'.join(lines) + '\n'
    (out / 'report.txt').write_text(text)
    print(text, end='')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['run', 'report'])
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.stage == 'run':
        run(args.out.resolve())
    else:
        report(args.out.resolve())


if __name__ == '__main__':
    main()
