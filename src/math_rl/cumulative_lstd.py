"""Raw-sum LSTD with a fixed feature basis and stable question cross-fitting.

Old-policy transitions are retained as sufficient statistics, not an on-policy
estimator. No answer buffer, mean normalization, adaptive ridge or pseudoinverse.
"""
import time
import numpy as np
import torch
from math_rl.online_critic import question_folds
from math_rl.lstd import eligibility


def encoder_epoch(step, interval):
    if step < 1 or interval < 0:
        raise ValueError('Positive update and nonnegative interval required')
    return (step - 1) // interval if interval else 0


def raw_design(states):
    x = torch.as_tensor(states).detach().cpu().double()
    if x.ndim != 2 or not len(x) or not torch.isfinite(x).all():
        raise ValueError('Expected finite, nonempty pre-action features')
    # Fixed coordinates: per-update standardization would invalidate old A/b.
    return torch.cat((x, torch.ones(len(x), 1, dtype=x.dtype)), 1)


def trajectory_statistics(states, outcome, trace_lambda=0.):
    if outcome not in (0., 1.) or not 0 <= trace_lambda <= 1:
        raise ValueError('Binary reward and lambda in [0,1] required')
    phi = raw_design(states)
    following = torch.cat((phi[1:], torch.zeros_like(phi[:1])))
    z = phi if trace_lambda == 0 else eligibility(phi, trace_lambda, torch.zeros(phi.shape[1], dtype=phi.dtype))
    return z.T @ (phi - following), z[-1] * outcome, len(phi)


class CumulativeLSTD:
    def __init__(self, question_ids, seed=42, epsilon=.01, trace_lambda=0., reset_interval=0, method="lstd"):
        if not np.isfinite(epsilon) or epsilon <= 0 or not 0 <= trace_lambda <= 1 or reset_interval < 0:
            raise ValueError('Invalid cumulative LSTD settings')
        if method not in ('lstd', 'ridge'):
            raise ValueError('Unknown cumulative method')
        self.method = method
        ids = sorted(set(question_ids))
        self.assignment = dict(zip(ids, question_folds(ids, seed).tolist()))
        self.epsilon, self.trace_lambda, self.reset_interval = epsilon, trace_lambda, reset_interval
        self.step, self.epoch = 0, -1
        self.stats = None

    def update(self, features, rewards, mask, questions, step, feature_epoch=0):
        started = time.perf_counter()
        if step != self.step + 1:
            raise ValueError('Cumulative updates must be consecutive, without duplicates or resume')
        epoch = encoder_epoch(step, self.reset_interval)
        if feature_epoch not in (0, epoch):
            raise ValueError('Encoder/statistics epoch mismatch')
        rewards, mask = rewards.detach().cpu(), mask.detach().cpu()
        if rewards.shape != mask.shape or len(features) != len(mask) or len(questions) != len(mask):
            raise ValueError('Feature/reward/mask batch mismatch')
        lengths = mask.sum(-1).long()
        expected = torch.arange(mask.shape[1])[None] < lengths[:, None]
        if (lengths < 1).any() or not torch.equal(mask, expected.to(mask.dtype)):
            raise ValueError('Expected nonempty contiguous masks')
        outcomes = rewards.sum(-1)
        terminal = torch.zeros_like(rewards)
        terminal[torch.arange(len(mask)), lengths - 1] = outcomes
        if not torch.equal(terminal, rewards) or not ((outcomes == 0) | (outcomes == 1)).all():
            raise ValueError('Only binary terminal rewards without KL shaping are supported')
        states = [raw_design(f) for f in features]
        k = states[0].shape[1]
        if any(len(f) != int(n) or f.shape[1] != k for f, n in zip(states, lengths)):
            raise ValueError('Feature count/dimension differs from actions')
        if any(q not in self.assignment for q in questions):
            raise ValueError('Question outside the sealed training fold assignment')
        reset = epoch != self.epoch
        if reset:
            self.stats = [dict(a=torch.zeros(k, k, dtype=torch.float64), b=torch.zeros(k, dtype=torch.float64),
                               transitions=0, answers=0) for _ in range(2)]
        elif self.stats[0]['b'].numel() != k:
            raise ValueError('Feature dimension changed without a reset')
        for phi, outcome, q in zip(states, outcomes.tolist(), questions):
            following = torch.cat((phi[1:], torch.zeros_like(phi[:1])))
            z = phi if self.trace_lambda == 0 else eligibility(phi, self.trace_lambda, torch.zeros(k, dtype=phi.dtype))
            stats = self.stats[self.assignment[q]]
            if self.method == 'ridge':
                stats['a'] += phi.T @ phi
                stats['b'] += phi.sum(0) * outcome
            else:
                stats['a'] += z.T @ (phi - following)
                stats['b'] += z[-1] * outcome
            stats['transitions'] += len(phi)
            stats['answers'] += 1
        values = torch.zeros_like(rewards, dtype=torch.float32)
        heads, checks = [], []
        for fold in range(2):
            stats = self.stats[1 - fold]  # No current OR historical labels from predicted questions.
            system = stats['a'] + self.epsilon * torch.eye(k, dtype=torch.float64)
            weights = torch.linalg.solve(system, stats['b'])
            residual = torch.linalg.vector_norm(system @ weights - stats['b'])
            relative = residual / (torch.linalg.matrix_norm(system) * torch.linalg.vector_norm(weights)
                                   + torch.linalg.vector_norm(stats['b'])).clamp_min(1e-30)
            if not torch.isfinite(weights).all() or relative > 1e-8:
                raise RuntimeError('Cumulative LSTD solve failed; no silent regularization change')
            for i, (phi, q) in enumerate(zip(states, questions)):
                if self.assignment[q] == fold:
                    values[i, :len(phi)] = (phi @ weights).float()
            heads.append(dict(weights=weights, predicted_fold=fold, feature_epoch=feature_epoch))
            checks.append(dict(predicted_fold=fold, transitions=stats['transitions'], answers=stats['answers'],
                effective_mean_regularization=self.epsilon / max(stats['transitions'], 1),
                relative_residual=float(relative), empty_fit=stats['transitions'] == 0,
                weight_norm=float(torch.linalg.vector_norm(weights))))
        if not torch.isfinite(values).all():
            raise RuntimeError('Nonfinite cumulative predictions')
        self.step, self.epoch = step, epoch
        valid = values[mask.bool()]
        target = outcomes[:, None].expand_as(mask)[mask.bool()]
        return values, heads, dict(method=self.method, fitting_mode='cumulative_raw_sum', step=step,
            feature_epoch=feature_epoch, statistics_epoch=epoch, reset=reset, epsilon=self.epsilon,
            intercept_penalized=True, critic_lambda=self.trace_lambda, folds=checks,
            fit_seconds=time.perf_counter()-started, transitions=int(lengths.sum()), answers=len(states),
            out_of_fold_brier=float((valid-target).square().mean()),
            outside_unit_interval=float(((valid < 0) | (valid > 1)).float().mean()),
            scope='Uncorrected mixture of behavior policies within each statistics epoch')

    def state_dict(self):
        return dict(method=self.method, step=self.step, epoch=self.epoch, stats=self.stats, assignment=self.assignment,
                    epsilon=self.epsilon, trace_lambda=self.trace_lambda, reset_interval=self.reset_interval)
