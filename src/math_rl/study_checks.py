"""CPU-only helpers for auditable GRPO diagnostics and sampled evaluation."""
from collections import Counter, defaultdict
from contextlib import contextmanager
from copy import deepcopy
import hashlib


def evaluation_seed(question_id, sample_index, seed):
    digest = hashlib.sha256(f'{seed}:{question_id}:{sample_index}'.encode()).digest()
    return int.from_bytes(digest[:4], 'big') % (2**31)


def group_diagnostics(ids, scores, expected_size=4):
    if len(ids) != len(scores):
        raise ValueError('Group/reward count mismatch')
    groups = defaultdict(list)
    for identity, score in zip(ids, scores):
        if score not in (0, 1):
            raise ValueError('Expected binary correctness rewards')
        groups[str(identity)].append(score)
    if not groups or any(len(x) != expected_size for x in groups.values()):
        raise ValueError('Incomplete GRPO prompt groups')
    kinds = Counter('all_correct' if all(x) else 'all_incorrect' if not any(x) else 'mixed'
                    for x in groups.values())
    return dict(groups=len(groups), **{k: kinds[k] for k in ('mixed', 'all_correct', 'all_incorrect')})


@contextmanager
def sampled_evaluation(rollout, seeds):
    """Override only evaluation: one independently seeded sample per repeated row.

    The pinned trainer repeats rows before generation. Supplying the same vLLM
    seed for every repeated row could produce identical answers, so each row
    gets a stable seed derived from its question ID and sample index.
    """
    val = rollout.config.val_kwargs
    saved = {k: val[k] for k in ('temperature', 'top_p', 'top_k')}
    engine = rollout.inference_engine
    generate = engine.generate

    def seeded_generate(*args, **kwargs):
        prompts = kwargs.get('prompts', args[0] if args else None)
        if prompts is None or len(prompts) != len(seeds):
            raise ValueError('Sample seed/prompt count mismatch')
        common = kwargs['sampling_params']
        params = []
        for seed in seeds:
            p = deepcopy(common)
            p.seed = int(seed)
            p.n = 1
            params.append(p)
        kwargs['sampling_params'] = params
        return generate(*args, **kwargs)

    val['temperature'], val['top_p'], val['top_k'] = 1.0, 1.0, -1
    engine.generate = seeded_generate
    try:
        yield
    finally:
        engine.generate = generate
        for key, value in saved.items():
            val[key] = value
