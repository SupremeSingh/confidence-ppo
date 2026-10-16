"""Predeclared conventional-PPO candidates and auditable critic diagnostics."""
import math
from pathlib import Path
import json

import torch

PPO_PROFILES = {
    'normalized': dict(lr=1e-5, epochs=2),
    'higher_lr': dict(lr=5e-5, epochs=2),
    'more_fitting': dict(lr=1e-5, epochs=4),
}


def zero_value_head(model):
    heads = [m for name, m in model.named_modules()
             if name.split('.')[-1] == 'score' and isinstance(m, torch.nn.Linear) and m.out_features == 1]
    if len(heads) != 1:
        raise ValueError('Expected exactly one scalar Qwen score head')
    with torch.no_grad():
        heads[0].weight.zero_()
        if heads[0].bias is not None:
            heads[0].bias.zero_()


def critic_diagnostics(before, after, returns, mask):
    """Equal-answer MSE on the SAME detached raw GAE targets, not whitened advantages."""
    before, after, returns, mask = [x.detach().cpu().float() for x in (before, after, returns, mask)]
    if not (before.shape == after.shape == returns.shape == mask.shape):
        raise ValueError('Critic diagnostic shape mismatch')
    if not all(torch.isfinite(x).all() for x in (before, after, returns, mask)) or (mask.sum(-1) <= 0).any():
        raise ValueError('Invalid critic diagnostic values')
    def mse(x):
        return float((((x - returns).square() * mask).sum(-1) / mask.sum(-1)).mean())
    return dict(mse_before=mse(before), mse_after=mse(after),
                prediction_change=float(((after-before).abs()*mask).sum()/mask.sum()),
                initial_value_abs_max=float(before[mask.bool()].abs().max()),
                target_mean=float(returns[mask.bool()].mean()), answers=len(mask))


def audit_health(directory, steps, world_size=2):
    files = sorted(Path(directory).glob('rank-*.jsonl'))
    rows = []
    complete = len(files) == world_size
    for file in files:
        records = [json.loads(s) for s in file.read_text().splitlines()]
        complete &= [r['update'] for r in records] == list(range(1, steps + 1))
        rows.extend(records)
    finite = bool(rows) and all(all(math.isfinite(r[k]) for k in
        ('mse_before', 'mse_after', 'prediction_change', 'grad_norm_max', 'lr')) for r in rows)
    gates = dict(complete=bool(complete), finite=finite,
        zero_initial_values=bool(rows) and all(r['initial_value_abs_max'] <= 1e-6 for r in rows if r['update'] == 1),
        nonzero_updates=bool(rows) and all(r['lr'] > 0 for r in rows) and any(r['grad_norm_max'] > 0 and r['prediction_change'] > 0 for r in rows),
        targets_fit_better=finite and sum(r['mse_after'] for r in rows) < sum(r['mse_before'] for r in rows))
    audit_seconds = sum(max((r.get('audit_seconds', 0.) for r in rows if r['update'] == step), default=0.)
                        for step in range(1, steps + 1))
    return dict(pass_checks=all(gates.values()), gates=gates,
                extra_forward_wall_seconds_estimate=audit_seconds,
                mean_mse_before=sum(r['mse_before'] for r in rows)/len(rows) if rows else None,
                mean_mse_after=sum(r['mse_after'] for r in rows)/len(rows) if rows else None,
                scope='Training-target fit check, not held-out critic accuracy')


def select_profile(candidates):
    eligible = [c for c in candidates if c['health']['pass_checks']
                and c['final_validation']['accuracy'] >= c['initial_validation']['accuracy']]
    if not eligible:
        raise RuntimeError('No PPO candidate passed critic health and validation non-regression; comparison not launched')
    return min(eligible, key=lambda c: (-c['final_validation']['accuracy'], list(PPO_PROFILES).index(c['profile'])))['profile']
