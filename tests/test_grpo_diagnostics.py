import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    'diagnose_grpo', Path(__file__).resolve().parents[1] / 'scripts/diagnose_grpo.py')
diagnostics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostics)


def test_log_metrics_keep_scientific_learning_rates_and_signed_kl():
    rows = diagnostics.log_metrics(
        'configuration actor/lr:1e-6\n'
        '\x1b[32mstep:1 - actor/grad_norm:0.65 - actor/lr:1e-6 - actor/ppo_kl:-0.000\x1b[0m\n'
        'step:2 - actor/grad_norm:0.2 - actor/lr:1e-6 - critic/score/mean:0.5\n')
    assert len(rows) == 2
    assert rows[0]['actor/lr'] == 1e-6
    assert rows[1]['critic/score/mean'] == .5


def test_answer_changes_distinguish_net_gain_from_churn():
    def row(i, score):
        return dict(id=i, ground_truth='4', score=score,
                    verifier_status='correct' if score else 'incorrect')
    before = [row(1, 0), row(2, 1), row(3, 1), row(4, 1)]
    after = [row(1, 1), row(2, 0), row(3, 0), row(4, 1)]
    counts = diagnostics.answer_changes(before, after)
    assert counts[0, 1] == 1
    assert counts[1, 0] == 2
    with pytest.raises(ValueError, match='Unmatched'):
        diagnostics.answer_changes(before, after[::-1])
