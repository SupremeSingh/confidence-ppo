"""Opt-in numeric conclusion selection; Math-Verify remains the equivalence checker.

Selection never sees gold. Preserve legacy scores and selected text for audit.
This is an experimental rule, not a semantic correctness or reasoning verifier.
"""
import re
from math_rl.math_verify_reward import grade, MAX_CHARS

RULE = 'math-verify-conclusion-v2'
NUMBER = r'[+-]?(?:\d+(?:\.\d+)?|\.\d+)'


def select_conclusion(response):
    # Ignore generated code/output when selecting a prose conclusion; never execute it.
    text = re.sub(r'```.*?(?:```|\Z)', '', response, flags=re.S)
    # Prefer the last explicitly named final-answer line, not an earlier calculation.
    lines = list(re.finditer(r'(?im)^[ \t]*(?:\*\*)?Final answer(?:\*\*)?[ \t]*:[ \t]*(?:\*\*)?[ \t]*(.+)$', text))
    if lines:
        if len(lines) != 1:
            return '', 'ambiguous_conclusion'
        return lines[-1].group(1).strip(), 'final_answer_line'
    if r'\boxed' in text:
        return None, 'legacy_box_selection'
    # A concluding assignment can be followed by unrelated numbers (base-7 example).
    # Accept one scalar assignment immediately after a conclusion cue, in the last
    # paragraph only. Multiple concluding assignments or later answer cues abstain.
    paragraph = text.strip().split('\n\n')[-1]
    cue = r'(?:we (?:find|conclude) that|therefore[, ]*|thus[, ]*|hence[, ]*)'
    assignment = r'(?:\$|\\\()[ \t]*[A-Za-z][ \t]*=[ \t]*(' + NUMBER + r')[ \t]*(?:\$|\\\))'
    matches = list(re.finditer(cue + r'[ \t]*' + assignment, paragraph, re.I))
    if matches:
        if len(matches) != 1 or re.search(r'\b(?:answer|instead|actually|however)\b', paragraph[matches[0].end():], re.I):
            return '', 'ambiguous_conclusion'
        return matches[0].group(1), 'concluding_numeric_assignment'
    return None, 'legacy_fallback'


def grade_conclusion(response, ground_truth):
    legacy = grade(response, ground_truth)
    candidate, route = select_conclusion(response) if len(response) <= MAX_CHARS else (None, 'input_too_long')
    if route == 'ambiguous_conclusion':
        result = dict(extracted=None, diagnostic_reward=0, status='ambiguous_conclusion')
    else:
        result = grade(candidate, ground_truth) if candidate is not None else legacy
    return dict(result, selection_route=route, selected_text=candidate,
                legacy_status=legacy['status'], legacy_reward=legacy['diagnostic_reward'],
                legacy_extracted=legacy['extracted'], rule_version=RULE)
