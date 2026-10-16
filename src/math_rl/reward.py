"""Legacy Stage 0 reward; not the Math-Verify audit scorer."""
import re
from decimal import Decimal

# Accept decimal numbers and correctly grouped thousands separators.
NUMBER = re.compile(r"[+-]?(?:[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+)(?:\.[0-9]+)?")


def parse_number(text):
    text = str(text).strip()
    if not NUMBER.fullmatch(text):
        return None
    return Decimal(text.replace(",", ""))


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    target = parse_number(ground_truth)
    if target is None:
        raise ValueError(f"Invalid ground truth: {ground_truth!r}")
    # The entire final non-empty line must follow our answer contract.
    lines = solution_str.strip().splitlines()
    if not lines:
        return 0.0
    line = lines[-1].strip()
    if not line.startswith("Answer:"):
        return 0.0
    predicted = parse_number(line[len("Answer:"):])
    return float(predicted is not None and predicted == target)
