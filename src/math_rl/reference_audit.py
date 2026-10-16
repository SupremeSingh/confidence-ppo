"""Conservative single-number eligibility from complete reference solutions.

Outcome-independent filtering, not a proof of reference correctness. Multiple
boxes are rejected even if repeated: do not silently choose the last box.
"""
import re
from math_rl.reward import parse_number


def reference_decision(problem, solution, saved=None):
    boxes = []
    malformed = False
    for match in re.finditer(r'\\(?:boxed|fbox)\s*\{', solution):
        start = match.end()
        depth = 1
        for i in range(start, len(solution)):
            depth += (solution[i] == '{') - (solution[i] == '}')
            if depth == 0:
                boxes.append(solution[start:i])
                break
        else:
            malformed = True
    reasons = []
    if malformed:
        reasons.append('malformed_reference_box')
    if len(boxes) != 1:
        reasons.append('reference_must_have_exactly_one_box')
    number = parse_number(boxes[0]) if len(boxes) == 1 else None
    if number is None:
        reasons.append('reference_not_single_decimal')
    # A box holding one root must not pass a prompt requesting the entire set.
    if re.search(r'(?:find|enter|give|list|determine)\s+all\s+(?:the\s+)?(?:possible\s+)?(?:values|solutions|roots)|separated\s+by\s+commas', problem, re.I):
        reasons.append('question_requests_multiple_answers')
    if saved is not None and number is not None and parse_number(saved) != number:
        reasons.append('saved_reference_mismatch')
    return dict(eligible=not reasons, reasons=reasons, reference_boxes=boxes,
                numeric_reference=str(number) if number is not None else None)
