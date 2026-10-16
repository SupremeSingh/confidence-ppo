"""Question-cross-fitted linear values for PPO (gamma=1).

Inputs are detached final-normalized actor states BEFORE each response token.
Optional recent replay requires a frozen encoder; actor updates remain on fresh data.
"""
from contextlib import contextmanager
import hashlib
import time

import numpy as np
import torch

from math_rl.lstd import accumulate, design, empty_statistics, solve


@contextmanager
def capture_prefixes(model, lengths):
    """Capture the existing unpadded, microbatch=1 Qwen forward, without another pass.

    lengths contains (nonpadding prompt tokens, valid response tokens) in local
    microbatch order. This deliberately rejects dynamic batching and SP elsewhere.
    """
    norms = [m for name, m in model.named_modules() if name.endswith('.model.norm') or name == 'model.norm']
    if len(norms) != 1:
        raise ValueError('Expected exactly one Qwen final model.norm')
    features = []

    def capture(module, inputs, hidden):
        if len(features) >= len(lengths):
            raise ValueError('Unexpected extra actor forward while capturing features')
        prompt, response = lengths[len(features)]
        if prompt < 1 or response < 1 or hidden.ndim != 3 or hidden.shape[:2] != (1, prompt + response):
            raise ValueError('Actor feature shape/order differs from the pinned unpadded forward')
        # Last prompt token predicts action 0; never include that action in its state.
        states = hidden[0, prompt - 1:prompt + response - 1].detach().float().cpu().numpy().copy()
        if not np.isfinite(states).all():
            raise ValueError('Nonfinite actor features')
        features.append(states)

    handle = norms[0].register_forward_hook(capture)
    try:
        yield features
        if len(features) != len(lengths):
            raise ValueError('Missing actor feature forwards')
    finally:
        handle.remove()


def question_folds(question_ids, seed):
    """Balanced, deterministic two-fold split; every answer to a question stays together."""
    ids = sorted(set(question_ids), key=lambda q: hashlib.sha256(f'{seed}:{q}'.encode()).hexdigest())
    if len(ids) < 4:
        raise ValueError('Cross-fitting needs at least four distinct questions')
    assignment = {q: i % 2 for i, q in enumerate(ids)}
    return np.array([assignment[q] for q in question_ids])


def fit_head(states, outcomes, method, alpha, trace_lambda):
    if method not in ('ridge', 'lstd') or not np.isfinite(alpha) or alpha <= 0:
        raise ValueError('Online fits require ridge/lstd and positive fixed regularization')
    if not states or len(states) != len(outcomes) or any(r not in (0., 1.) for r in outcomes):
        raise ValueError('Expected nonempty binary-outcome trajectories')
    x = torch.cat(states).float()
    if x.ndim != 2 or not torch.isfinite(x).all() or any(len(s) == 0 for s in states):
        raise ValueError('Invalid feature trajectories')
    mean = x.mean(0).double()
    scale = x.std(0, unbiased=False).clamp_min(1e-5).double()
    del x
    stats = empty_statistics(len(mean))
    for state, reward in zip(states, outcomes, strict=True):
        if method == 'lstd':
            # accumulate ignores the terminal context and explicitly zeroes phi_next.
            terminal = torch.zeros_like(state[:1])
            accumulate(stats, torch.cat((state, terminal)), reward, mean, scale,
                       trace_lambda=trace_lambda, include_ridge=False)
        else:
            for chunk in state.split(2048):
                phi = design(chunk, mean, scale)
                stats['gram'] += phi.T @ phi
                stats['returns_rhs'] += phi.sum(0) * reward
            stats['transitions'] += len(state)
            stats['trajectories'] += 1
    weights, diagnostics = solve(stats, method, alpha)
    if diagnostics['relative_residual'] > 1e-8:
        raise RuntimeError('Linear critic solve failed its residual check')
    return dict(weights=weights, mean=mean, scale=scale), dict(diagnostics,
        transitions=stats['transitions'], trajectories=stats['trajectories'])


def cross_fitted_values(features, rewards, mask, question_ids, method, alpha=.01, trace_lambda=.99, seed=42, replay=None, step=None):
    """Predict each question using only other questions' outcomes.

    Without replay, all fitting data comes from the current batch. Replay includes
    recent older policies and therefore does not estimate a pure on-policy value.

    Includes zero-reward parse failures, just like online PPO/GRPO. Raw linear
    predictions are neither clipped nor passed through a sigmoid. EOS and the
    length cap are terminal under this capped-response task.
    """
    started = time.perf_counter()
    rewards, mask = rewards.detach().cpu(), mask.detach().cpu()
    if rewards.shape != mask.shape or len(features) != len(mask) or len(question_ids) != len(mask):
        raise ValueError('Feature/reward/mask batch mismatch')
    lengths = mask.sum(-1).long()
    expected = torch.arange(mask.shape[1])[None, :] < lengths[:, None]
    if not torch.equal(mask.bool(), expected) or not ((mask == 0) | (mask == 1)).all() or (lengths < 1).any():
        raise ValueError('Expected nonempty contiguous response masks')
    outcomes = rewards.sum(-1)
    terminal_rewards = torch.zeros_like(rewards)
    terminal_rewards[torch.arange(len(mask)), lengths - 1] = outcomes
    if not torch.equal(rewards, terminal_rewards) or not ((outcomes == 0) | (outcomes == 1)).all():
        raise ValueError('This critic requires binary terminal rewards, gamma=1 and no KL shaping')
    states = [torch.as_tensor(x).detach().cpu().float() for x in features]
    if any(len(x) != int(n) for x, n in zip(states, lengths, strict=True)):
        raise ValueError('Feature count must equal valid response actions')
    folds = question_folds(question_ids, seed)
    if replay is not None:
        if step is None:
            raise ValueError('Replay requires a behavior-policy step')
        replay.append(states, outcomes.tolist(), question_ids, step)
    values = torch.zeros_like(rewards, dtype=torch.float32)
    heads, diagnostics = [], []
    for fold in range(2):
        train = np.flatnonzero(folds != fold).tolist()
        test = np.flatnonzero(folds == fold).tolist()
        held_out = {question_ids[i] for i in test}
        if replay is None:
            fit_states, fit_rewards = [states[i] for i in train], outcomes[train].tolist()
            fit_questions = {question_ids[i] for i in train}
            history = 0
        else:
            # Exclude ALL older answers to held-out questions, even when folds change.
            eligible = [e for e in replay.entries if e.question not in held_out]
            fit_states, fit_rewards = [e.states for e in eligible], [e.reward for e in eligible]
            fit_questions = {e.question for e in eligible}
            history = sum(e.step < step for e in eligible)
        head, checks = fit_head(fit_states, fit_rewards, method, alpha, trace_lambda)
        for i in test:
            values[i, :lengths[i]] = (design(states[i], head['mean'], head['scale']) @ head['weights']).float()
        heads.append(dict(head, predicted_fold=fold, train_questions=sorted(fit_questions)))
        diagnostics.append(dict(checks, predicted_fold=fold,
            historical_answers=history, train_questions=len(fit_questions), test_questions=len({question_ids[i] for i in test})))
    valid = values[mask.bool()]
    targets = outcomes[:, None].expand_as(values)[mask.bool()]
    if not torch.isfinite(valid).all():
        raise RuntimeError('Nonfinite cross-fitted values')
    metrics = dict(method=method, alpha=alpha, critic_lambda=trace_lambda if method == 'lstd' else 1.,
        fit_seconds=time.perf_counter() - started, folds=diagnostics, answers=len(states),
        transitions=int(lengths.sum()), mean_value=float(valid.mean()), min_value=float(valid.min()),
        max_value=float(valid.max()), outside_unit_interval=float(((valid < 0) | (valid > 1)).float().mean()),
        out_of_fold_brier=float((valid - targets).square().mean()), reward_rate=float(outcomes.mean()))
    if replay is not None:
        metrics['replay'] = replay.metrics()
    return values.detach(), heads, metrics


@torch.inference_mode()
def frozen_prefixes(encoder, input_ids, attention_mask, response_width, layer="final"):
    """One causal encoder pass per unpadded answer, retaining pre-action states."""
    device = next(encoder.parameters()).device
    features = []
    for ids, mask in zip(input_ids, attention_mask, strict=True):
        prompt = int(mask[:-response_width].sum())
        response = int(mask[-response_width:].sum())
        if prompt < 1 or response < 1:
            raise ValueError('Empty frozen-feature prompt or response')
        tokens = ids[mask.bool()].unsqueeze(0).to(device)
        if layer == 'final':
            hidden = encoder(input_ids=tokens, use_cache=False, return_dict=True).last_hidden_state
        elif layer == 'two_thirds':
            from math_rl.critic_layers import layer_spec
            captured = []
            def capture(module, inputs, output):
                captured.append(output[0] if isinstance(output, tuple) else output)
            hook = encoder.layers[layer_spec(len(encoder.layers))['two_thirds'] - 1].register_forward_hook(capture)
            try:
                encoder(input_ids=tokens, use_cache=False, return_dict=True)
            finally:
                hook.remove()
            if len(captured) != 1:
                raise ValueError('Expected exactly one intermediate feature capture')
            hidden = captured[0]
        else:
            raise ValueError('Unknown critic feature layer')
        states = hidden[0, prompt-1:prompt+response-1].float().cpu().numpy().copy()
        if len(states) != response or not np.isfinite(states).all():
            raise ValueError('Invalid frozen features')
        features.append(states)
    return features
