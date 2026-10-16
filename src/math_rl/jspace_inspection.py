"""Read-only J-lens diagnostics; concept scores are not correctness probabilities."""
import json
import re

import numpy as np
import torch

CONCEPTS = {
    'error': ['error', 'wrong', 'incorrect', 'mistake', 'fault', 'fail', 'failure'],
    'uncertainty': ['uncertain', 'unsure', 'maybe', 'perhaps', 'doubt'],
    'revision': ['however', 'but', 'instead', 'reconsider', 'check'],
}


def concept_ids(tokenizer, vocab_size):
    """Only exact single-token words, including case/whitespace variants.

    Never approximate multi-token phrases by adding unrelated subtoken scores.
    Save the actual IDs/decoded strings so coverage is inspectable.
    """
    groups = {name: [] for name in CONCEPTS}
    coverage = {name: [] for name in CONCEPTS}
    for token in range(vocab_size):
        text = tokenizer.decode([token])
        for name, words in CONCEPTS.items():
            if text.strip().casefold() in words:
                groups[name].append(token)
                coverage[name].append({'id': token, 'text': text})
    return groups, coverage


@torch.inference_mode()
def direct_readout(hidden, jacobian, unembedding, gain, eps, groups, top_k=10):
    """Official readout formula U RMSNorm(Jh); no sparse pursuit or fitted reward head."""
    # None accepts a vector already transported by official JacobianLens.transport.
    transported = hidden.float() if jacobian is None else hidden.float() @ jacobian.float().T
    normalized = transported * torch.rsqrt(transported.square().mean(-1, keepdim=True) + eps)
    logits = (normalized * gain.float()) @ unembedding.float().T
    logp = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    values, ids = logp.topk(min(top_k, logits.shape[-1]), dim=-1)
    masses = {name: logp[:, tokens].exp().sum(-1) if tokens else torch.zeros(len(hidden), device=hidden.device)
              for name, tokens in groups.items()}
    if not torch.isfinite(logp).all():
        raise ValueError('Nonfinite J-lens scores')
    return ids.cpu(), values.exp().cpu(), {name: value.cpu() for name, value in masses.items()}


def linear_contributions(record, hidden, count=12):
    """Exact standardized-coordinate contributions to a saved linear ridge head."""
    weight = record['state']['weight'].reshape(-1)
    bias = float(record['state']['bias'].reshape(-1)[0])
    contributions = ((hidden.float() - record['mean']) / record['scale']) * weight
    value = contributions.sum(-1) + bias
    ids = contributions.abs().topk(min(count, hidden.shape[-1]), dim=-1).indices
    return value, [dict(bias=bias, dimensions=index.tolist(),
                       contributions=row[index].tolist(),
                       other=float(row.sum() - row[index].sum()))
                   for row, index in zip(contributions, ids)]


def visible_mentions(text):
    return {name: bool(re.search(r'\b(?:' + '|'.join(words) + r')\b', text, re.I))
            for name, words in CONCEPTS.items()}


def association(rows, iterations=1000):
    """One score per answer, then paired/question-clustered descriptive bootstrap.

    All sampled nonzero prefixes averaged per answer; repeats were removed upstream.
    Restricting away visible mentions is diagnostic, not a causal adjustment.
    """
    from math_rl.critic_probe import auc
    output = {}
    for block in sorted({r['block'] for r in rows}):
        for group in CONCEPTS:
            for restriction in ('all', 'no_visible_mention'):
                answers = {}
                for r in rows:
                    if r['block'] != block or r['position'] == 0:
                        continue
                    if restriction == 'no_visible_mention' and r['mentions'][group]:
                        continue
                    key = (r['question'], r['response'])
                    answers.setdefault(key, {'y': r['correct'], 'scores': []})['scores'].append(r['mass'][group])
                if not answers:
                    continue
                keys = list(answers)
                score = np.array([np.mean(answers[k]['scores']) for k in keys])
                wrong = np.array([1-answers[k]['y'] for k in keys], dtype=bool)
                qs = np.array([k[0] for k in keys])
                unique = np.unique(qs)
                def difference(indices):
                    y, x = wrong[indices], score[indices]
                    return float(x[y].mean()-x[~y].mean()) if y.any() and (~y).any() else None
                rng = np.random.default_rng(514)
                clusters = [np.flatnonzero(qs == q) for q in unique]
                deltas = []
                for _ in range(iterations):
                    d = difference(np.concatenate([clusters[i] for i in rng.integers(len(clusters), size=len(clusters))]))
                    if d is not None:
                        deltas.append(d)
                output[f'block{block}/{group}/{restriction}'] = dict(
                    answers=len(keys), questions=len(unique), wrong_answers=int(wrong.sum()),
                    failure_auroc=auc(wrong.astype(int), score),
                    wrong_minus_correct_mean_mass=difference(np.arange(len(keys))),
                    descriptive_95_interval=np.quantile(deltas, [.025, .975]).tolist() if deltas else None)
    return output


def html_payload(value):
    """Embed untrusted answer text as inert JSON, never as executable HTML."""
    return json.dumps(value, ensure_ascii=True, allow_nan=False).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
