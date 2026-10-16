"""Protocol helpers for a question-disjoint, full-readout J-lens probe study."""
import hashlib
import json
import re
import unicodedata

import numpy as np
import torch

from math_rl.reference_audit import reference_decision

ARMS = ('direct_jlens_concat', 'hidden_concat', 'rms_hidden_concat',
        'hidden_two_thirds', 'final')
HEADS = ('linear_logit', 'mlp2', 'resnet10')
SPLITS = ('train', 'val', 'test')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def family_key(text):
    """Conservative number-substitution deduplication, not semantic deduplication."""
    text = unicodedata.normalize('NFKC', text).lower()
    text = re.sub(r'\d+(?:[.,]\d+)*', ' NUMBER ', text)
    return re.sub(r'\W+', ' ', text).strip()


def candidate(row):
    """Reference-only eligibility; never consult generated answers or rewards."""
    if (row.get('source') != 'cn_k12' or row.get('synthetic') is not False
            or row.get('question_type') != 'math-word-problem'
            or row.get('problem_is_valid') != 'Yes' or row.get('solution_is_valid') != 'Yes'):
        return None, 'source_or_metadata'
    problem = row['problem'].strip()
    if re.search(r'!\[|\\includegraphics|https?://|\b(?:diagram|figure|shown below|prove that)\b', problem, re.I):
        return None, 'non_text_or_proof'
    decision = reference_decision(problem, row['solution'], str(row.get('answer', '')))
    if not decision['eligible']:
        return None, '|'.join(decision['reasons'])
    return dict(question=problem, ground_truth=decision['numeric_reference'],
                reference_solution=row['solution'], reference_audit=decision), None


def assign_splits(rows, counts):
    """Stable split assignment without outcome-based balancing."""
    ordered = sorted(rows, key=lambda q: digest(['large-jlens-v1', q['family']]))
    if len(ordered) < sum(counts.values()):
        raise ValueError(f'Only {len(ordered)} eligible families; need {sum(counts.values())}. Do not silently relax filters.')
    result = []; cursor = 0
    for split, count in counts.items():
        for q in ordered[cursor:cursor+count]:
            result.append(dict(q, split=split, index=len(result)))
        cursor += count
    return result


def prefix_positions(length, question, response):
    if length < 1:
        raise ValueError('No prefix for an empty response')
    rng = np.random.default_rng(np.random.SeedSequence([1729, question, response]))
    return [0] + (rng.integers(1, length, 7).tolist() if length > 1 else [0]*7)


def streaming_normalization(x, batch_size=4096):
    """Training-only mean and ONE isotropic scale, without a full float64 copy."""
    n, d = x.shape
    if not n:
        raise ValueError('Empty training features')
    total = np.zeros(d, dtype=np.float64); squares = 0.
    for start in range(0, n, batch_size):
        batch = np.asarray(x[start:start+batch_size], dtype=np.float64)
        if not np.isfinite(batch).all():
            raise ValueError('Nonfinite features')
        total += batch.sum(0); squares += np.square(batch).sum()
    mean = total/n
    variance = max(0., squares/(n*d) - np.square(mean).mean())
    return torch.tensor(mean, dtype=torch.float32), torch.full((d,), max(variance**.5, 1e-8))


def paired_errors(y, a, b, questions, accuracy=False, repetitions=4000):
    """Paired question bootstrap; average errors across head seeds first."""
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != len(y):
        raise ValueError('Unpaired predictions')
    if accuracy:
        diff = (((a >= .5) == y).astype(float) - ((b >= .5) == y)).mean(0)
    else:
        diff = (np.square(a-y) - np.square(b-y)).mean(0)
    unique, group = np.unique(questions, return_inverse=True)
    per_question = np.bincount(group, weights=diff)/np.bincount(group)
    rng = np.random.default_rng(913)
    boot = np.array([rng.choice(per_question, len(unique), replace=True).mean() for _ in range(repetitions)])
    return dict(mean=float(per_question.mean()), questions=len(unique),
                interval_95=np.quantile(boot, [.025, .975]).tolist(),
                interval_97_5=np.quantile(boot, [.0125, .9875]).tolist())
