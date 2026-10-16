"""Fresh answer-verifier audit gates, separate from task difficulty."""


def final_answer_gates(metadata, scoring, reviewed, agreement):
    """A reliable verifier and a suitably difficult dataset are separate claims."""
    fresh = (metadata.get('mode') == 'audit' and metadata.get('audit_offset') == 160
             and len(set(metadata.get('excluded_development_ids', []))) == 160
             and len(set(metadata.get('prompt_ids', []))) * 2 == scoring['responses']
             and not set(metadata.get('prompt_ids', [])) & set(metadata.get('excluded_development_ids', [])))
    statuses = scoring.get('status_counts', {})
    runtime_ok = sum(statuses.values()) == scoring['responses'] and not any(count for status, count in statuses.items()
                         if status not in ('correct', 'incorrect', 'parse_failure', 'unsupported_parse'))
    reliability = dict(fresh_audit=fresh,
                       all_responses_reviewed=reviewed == scoring['responses'] and reviewed >= 200,
                       agreement_at_least_99_percent=agreement is not None and agreement >= 0.99,
                       no_verifier_runtime_errors=runtime_ok,
                       truncation_at_most_5_percent=scoring['truncation_rate'] <= 0.05)
    difficulty = dict(reward_rate_5_to_70_percent=0.05 <= scoring['diagnostic_reward_rate'] <= 0.70,
                      mixed_pairs_at_least_10_percent=scoring['mixed_pair_rate'] >= 0.10)
    return dict(reliability_gates=reliability, difficulty_gates=difficulty,
                verifier_audit_pass=all(reliability.values()),
                stage1_pass=all(reliability.values()) and all(difficulty.values()))
