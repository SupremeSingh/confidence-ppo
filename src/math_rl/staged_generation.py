"""Frozen-policy staged rollouts. Inserted context is never a sampled action."""
VERSION = 'staged-plan-v2'
END = '<END_STAGE>'


def excluded_tokens(row, invalid_ids):
    """Preserve an unusable attempt for audit, never repair its token sequence."""
    return dict(response=row.get('response', '') + '\n[Generation excluded: invalid token ID]',
        token_ids=[], score=None, verifier_status='invalid_token_id', finish_reason='abort',
        final_stage_text='', generation_protocol=VERSION,
        generation_error=dict(invalid_ids=sorted(set(invalid_ids)), original=row))


def check_cached(row, valid_ids):
    if row.get('verifier_status') == 'invalid_token_id':
        if row.get('score') is not None or row.get('token_ids') or not row.get('generation_error'):
            raise ValueError('Malformed excluded generation')
        return row
    validate(row)
    invalid = set(row['context_token_ids']) - valid_ids
    return excluded_tokens(row, invalid) if invalid else row


def decision_states(states, row):
    """Select pre-action states from full-context causal features, plus terminal."""
    positions = validate(row)
    if len(states) != len(row['context_token_ids']) + 1:
        raise ValueError('Staged feature/context length mismatch')
    return states[positions + [len(row['context_token_ids'])]]


def stages(planning):
    if planning:
        return [('facts', 'What we know:', 96), ('plan', 'What we will do:', 96),
                ('solution', 'Solution:', 1792), ('final', 'Final answer:', 64)]
    return [('solution', 'Solution:', 1984), ('final', 'Final answer:', 64)]


def prompt(tokenizer, question, planning):
    instruction = ('Solve the question in stages. The controller supplies each section heading. '
        'Write only the content of the current section, then write <END_STAGE>. '
        'Do not repeat headings or write later sections. ')
    if planning:
        instruction += ('For What we know, list given facts, constraints and the requested quantity; invent nothing. '
                        'For What we will do, give a short proposed method without the final result. ')
    instruction += (
        'For Solution, work through the calculation. '
        'For Final answer, state one unambiguous final numeric answer. A short sentence or boxed answer is acceptable.')
    text = instruction + '\n\nQuestion: ' + question + '\n\nResponse:'
    return text, tokenizer.encode(text, add_special_tokens=False)


def validate(row):
    context, mask = row['context_token_ids'], row['action_mask']
    if len(context) != len(mask) or any(type(m) is not bool for m in mask):
        raise ValueError('Invalid staged action mask')
    if [t for t, action in zip(context, mask) if action] != row['token_ids']:
        raise ValueError('Staged sampled tokens do not match context mask')
    positions = [i for i, action in enumerate(mask) if action]
    if positions != row['action_positions']:
        raise ValueError('Invalid staged decision positions')
    cursor = 0
    actions = 0
    if not row['stages']:
        raise ValueError('Missing stage schedule')
    for stage in row['stages']:
        if not (stage['header_start'] == cursor <= stage['action_start'] <= stage['end'] <= len(context)):
            raise ValueError('Invalid stage boundaries')
        if any(mask[cursor:stage['action_start']]) or not all(mask[stage['action_start']:stage['end']]):
            raise ValueError('Injected headers marked as sampled actions')
        if stage['end'] - stage['action_start'] > stage['budget'] or stage['action_offset'] != actions:
            raise ValueError('Invalid stage action budget or offset')
        if stage['sampled_tokens'] != stage['end'] - stage['action_start']:
            raise ValueError('Incorrect sampled-token count')
        actions += stage['end'] - stage['action_start']
        cursor = stage['end']
    if cursor != len(context) or sum(s['budget'] for s in row['stages']) != 2048:
        raise ValueError('Incomplete staged trajectory or wrong total budget')
    expected = stages(row['stages'][0]['name'] == 'facts')
    if [(s['name'], s['header'], s['budget']) for s in row['stages']] != expected:
        raise ValueError('Unknown staged schedule')
    return positions


def rollout(engine, tokenizer, prompt_ids, planning, seed, sampling_class, valid_ids=None):
    valid_ids = set(tokenizer.get_vocab().values()) if valid_ids is None else valid_ids
    if set(prompt_ids) - valid_ids:
        raise ValueError('Prompt contains tokens absent from the tokenizer')
    context, mask, log = [], [], []
    for index, (name, header, budget) in enumerate(stages(planning)):
        header_start = len(context)
        injected = tokenizer.encode('\n\n' + header + '\n', add_special_tokens=False)
        if set(injected) - valid_ids:
            raise ValueError('Controller header contains invalid tokens')
        context.extend(injected)
        mask.extend([False] * len(injected))
        start = len(context)
        params = sampling_class(n=1, temperature=1., top_p=1., top_k=-1,
            max_tokens=budget, seed=seed + index, stop=[END],
            include_stop_str_in_output=True)
        result = engine.generate([{'prompt_token_ids': prompt_ids + context}], params)[0].outputs[0]
        if result.finish_reason not in ('stop', 'length'):
            raise RuntimeError('Unexpected stage termination')
        ids = list(result.token_ids)
        invalid = set(ids) - valid_ids
        if invalid:
            # Output vocabularies can contain IDs the tokenizer cannot accept as
            # the next stage's input. Do not decode, retry or silently remove them.
            return excluded_tokens(dict(response=tokenizer.decode(context, skip_special_tokens=True),
                context_token_ids=context + ids, action_mask=mask + [True] * len(ids),
                failed_stage=name, completed_stages=log), invalid)
        context.extend(ids)
        mask.extend([True] * len(ids))
        log.append(dict(name=name, header=header, budget=budget, header_start=header_start,
            action_start=start, end=len(context), action_offset=sum(mask[:start]),
            finish_reason=result.finish_reason, stop_reason=result.stop_reason,
            text=tokenizer.decode(ids, skip_special_tokens=True), sampled_tokens=len(ids)))
    final = log[-1]['text'].split(END, 1)[0].strip()
    row = dict(response=tokenizer.decode(context, skip_special_tokens=True),
        context_token_ids=context, action_mask=mask,
        action_positions=[i for i, action in enumerate(mask) if action],
        token_ids=[t for t, action in zip(context, mask) if action], stages=log,
        inserted_tokens=sum(not v for v in mask),
        final_stage_text=final, finish_reason='length' if any(s['finish_reason'] == 'length' for s in log) else 'stop',
        stop_reason=log[-1]['stop_reason'], generation_protocol=VERSION)
    validate(row)
    return row


def stage_audit(records):
    """Count controller stops and content failures without interpreting reasoning."""
    summary = {}
    for row in records:
        for stage in row.get('stages') or []:
            counts = summary.setdefault(stage['name'], dict(attempts=0, capped=0,
                empty=0, marker_stop=0, other_stop=0, sampled_tokens=0))
            counts['attempts'] += 1
            counts['capped'] += stage['finish_reason'] == 'length'
            counts['empty'] += not stage['text'].split(END, 1)[0].strip()
            counts['marker_stop'] += stage['stop_reason'] == END
            counts['other_stop'] += stage['finish_reason'] == 'stop' and stage['stop_reason'] != END
            counts['sampled_tokens'] += stage['sampled_tokens']
    return summary


def generate(out, config, questions, root, atomic_json):
    import json
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    from math_rl.ppo_reward import compute_score
    if all((out / 'trajectories' / f'{i:05d}.json').exists() for i in range(len(questions))):
        return
    tokenizer = AutoTokenizer.from_pretrained(root / 'models/qwen-math', local_files_only=True)
    valid_ids = set(tokenizer.get_vocab().values())
    planning = config['prompt_style'] == 'plan'
    header_tokens = sum(len(tokenizer.encode('\n\n' + header + '\n', add_special_tokens=False))
                        for _, header, _ in stages(planning))
    context_limit = max(len(q['prompt_token_ids']) for q in questions) + 2048 + header_tokens
    engine = LLM(model=str(root / 'models/qwen-math'), dtype='bfloat16', tensor_parallel_size=1,
        max_model_len=context_limit, gpu_memory_utilization=.6, max_num_seqs=4,
        max_num_batched_tokens=context_limit, enforce_eager=True, seed=42, generation_config='vllm')
    (out / 'pending').mkdir(exist_ok=True)
    for i, q in enumerate(questions):
        target = out / 'trajectories' / f'{i:05d}.json'
        if target.exists():
            continue
        rows = []
        for sample in range(config['responses']):
            cached = out / 'pending' / f'{i:05d}-{sample}.json'
            if cached.exists():
                saved = json.loads(cached.read_text())
                if saved['question_id'] != q['id']:
                    raise ValueError('Staged resume question mismatch')
                row = check_cached(saved['response'], valid_ids)
            else:
                row = rollout(engine, tokenizer, q['prompt_token_ids'], planning,
                              config['generation_seed'] + i * 100 + sample * 10, SamplingParams, valid_ids)
                atomic_json(cached, dict(question_id=q['id'], response=row))
            rows.append(row)
        scorable = [r for r in rows if r.get('verifier_status') != 'invalid_token_id']
        scores = compute_score(['math_numeric'] * len(scorable),
            ['Final answer: ' + r['final_stage_text'] for r in scorable],
            [q['ground_truth']] * len(scorable), exclude_errors=True) if scorable else []
        for row, score in zip(scorable, scores, strict=True):
            row.update(score, scoring_scope='final stage only')
            if not row['final_stage_text'] or not row['token_ids']:
                row.update(score=None, verifier_status='empty_final_stage')
        atomic_json(target, dict(question_id=q['id'], responses=rows))
        for sample in range(config['responses']):
            (out / 'pending' / f'{i:05d}-{sample}.json').unlink()
        print(f'Staged generation/scoring {i + 1}/{len(questions)}', flush=True)
