"""JSON batch in/out for the isolated, pinned Math-Verify interpreter."""
from contextlib import redirect_stdout
import json
import sys

from math_rl.math_verify_reward import RULE_VERSION, grade, provenance


def score_batch(rows, exclude_errors=False):
    provenance()
    scores = []
    for row in rows:
        grader = grade
        if row.get('rule') not in (None, RULE_VERSION):
            from math_rl.conclusion_reward import RULE, grade_conclusion
            if row['rule'] != RULE:
                raise ValueError('Unknown verifier rule')
            grader = grade_conclusion
        if exclude_errors:
            from math_rl.reward import parse_number
            if parse_number(row['ground_truth']) is None:
                raise ValueError('Invalid reference; refusing to exclude a dataset/configuration error')
            try:
                result = grader(row["response"], row["ground_truth"])
            except (MemoryError, OSError, ImportError):
                raise
            except Exception as exc:
                result = dict(diagnostic_reward=0, extracted=None, status='grader_error', error=type(exc).__name__)
        else:
            result = grader(row["response"], row["ground_truth"])
        # Bounded verification can be inconclusive on model-generated expressions.
        # Keep that status distinct from incorrect; do not swallow runtime errors.
        if exclude_errors:
            scores.append(dict(score=result['diagnostic_reward'] if result['status'] in {'correct', 'incorrect'} else None,
                               verifier_status=result['status'], extracted=result['extracted'] or '',
                               **({'audit': {k: result[k] for k in ('selection_route', 'selected_text',
                                   'legacy_status', 'legacy_reward', 'legacy_extracted', 'rule_version')}}
                                  if 'rule_version' in result else {}),
                               **({'verifier_error': result['error']} if 'error' in result else {})))
            continue
        if result["status"] not in {"correct", "incorrect", "parse_failure", "unsupported_parse",
                                    "parse_timeout", "verify_timeout"}:
            raise RuntimeError(f"Verifier failed: {result}")
        scores.append(dict(score=result["diagnostic_reward"],
                           verifier_status=result["status"],
                           extracted=result["extracted"] or ""))
    return scores


def main():
    if sys.argv[1:] == ["--provenance"]:
        print(json.dumps(provenance()))
        return
    rows = json.load(sys.stdin)
    # Parser diagnostics cannot corrupt the JSON protocol.
    with redirect_stdout(sys.stderr):
        scores = score_batch(rows, exclude_errors=sys.argv[1:] == ['--exclude-errors'])
    json.dump(scores, sys.stdout, allow_nan=False)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
