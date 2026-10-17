import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    'gsm8k_pilot', Path(__file__).resolve().parents[1] / 'scripts/gsm8k_pilot.py')
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def test_selection_has_disjoint_ids_and_wording_and_current_metadata():
    examples = [dict(question=f'What is {i} plus one?', answer=f'Solution\n#### {i+1}')
                for i in range(20)]
    examples.append(dict(question=' WHAT is 0 plus one? ', answer='#### 1'))
    train, val = pilot.select_rows(examples, lambda q: 10, train_count=4, val_count=16)
    rows = train + val
    assert len({r['extra_info']['prompt_id'] for r in rows}) == 20
    assert len({''.join(r['prompt'][1]['content'].split()).casefold() for r in rows}) == 20
    assert all(r['extra_info']['split'] == 'train' for r in train)
    assert all(r['extra_info']['split'] == 'val' for r in val)
    assert all(r['extra_info']['level'] == 0 and r['data_source'] == 'gsm8k' for r in rows)
    assert all(r['prompt'][1]['content'].startswith('Work through the problem carefully.') for r in rows)
    assert (train, val) == pilot.select_rows(examples, lambda q: 10, 4, 16)


def test_overlong_questions_cannot_silently_reduce_counts():
    examples = [dict(question='question', answer='#### 4')]
    with pytest.raises(ValueError, match='Insufficient'):
        pilot.select_rows(examples, lambda q: 513, 1, 1)


def test_report_separates_practice_improvement_from_generalization(tmp_path):
    case = tmp_path / 'grpo-seed301'
    case.mkdir()
    result = dict(method='grpo', updates=20, answers=1280, evaluation_split='val',
                  initial_train_probe={'accuracy': .5}, final_train_probe={'accuracy': .75},
                  initial_validation={'accuracy': .5}, final_validation={'accuracy': .5})
    (case / 'result.json').write_text(json.dumps(result))
    pilot.report(tmp_path)
    text = (tmp_path / 'report.txt').read_text()
    assert 'Practice questions (32; seen during training): 50.00% -> 75.00% (+25.00 pp)' in text
    assert 'Held-out validation (256): 50.00% -> 50.00% (+0.00 pp)' in text
    result['answers'] = 64
    (case / 'result.json').write_text(json.dumps(result))
    with pytest.raises(ValueError, match='Unexpected'):
        pilot.report(tmp_path)
