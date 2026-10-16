"""Frozen Stage 1/PPO completion prefix and the preserved Stage 0 prompt."""
MODEL_ID = "Qwen/Qwen2.5-Math-1.5B"
INSTRUCTION = (
    "Solve the math problem step by step. End with a line containing only "
    "Answer: <number>, where <number> is a decimal number. "
    "Do not put the final answer in boxed notation or append units or punctuation."
)


def make_prompt(question):
    return [{"role": "system", "content": INSTRUCTION},
            {"role": "user", "content": question}]


def validate_prompt(messages):
    if len(messages) != 2 or messages[0] != {"role": "system", "content": INSTRUCTION} or messages[1]["role"] != "user":
        raise ValueError("Stale prompt contract: rerun scripts/prepare_data.py and re-audit")


def display_prompt(messages):
    return "\n\n".join(f"[{m['role']}]\n{m['content']}" for m in messages)


def validate_assets(assets):
    if assets.get("model_id") != MODEL_ID:
        raise ValueError(f"Expected base model {MODEL_ID}; preserve old assets separately and reprepare")
    for key in ("model_revision", "dataset_revision"):
        value = assets.get(key, "")
        if len(value) != 40 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"{key} must be an immutable commit SHA")


# Frozen Stage 1 prefix. Legacy make_prompt above belongs to the old smoke path.
AUDIT_INSTRUCTION = r"Solve the math problem step by step. Put your final numeric answer inside a single \boxed{...}."


def encode_completion(tokenizer, question):
    text = AUDIT_INSTRUCTION + "\n\nQuestion: " + question + "\n\nSolution:"
    return text, tokenizer.encode(text, add_special_tokens=False)


PLAN_INSTRUCTION = (
    r"First give a brief plan of 2-3 sentences describing how to solve the problem. "
    r"Then carry out the plan step by step. Put your final numeric answer inside a single \boxed{...}."
)


def encode_planning(tokenizer, question):
    text = PLAN_INSTRUCTION + "\n\nQuestion: " + question + "\n\nSolution:"
    return text, tokenizer.encode(text, add_special_tokens=False)


def encode_plan_sections(tokenizer, question):
    instruction = (PLAN_INSTRUCTION + ' Use a Plan: heading for the plan and then a Solution: '
                   'heading on its own line before carrying out the plan.')
    text = instruction + '\n\nQuestion: ' + question + '\n\nPlan:'
    return text, tokenizer.encode(text, add_special_tokens=False)


def encode_structured(tokenizer, question, planning):
    if planning:
        instruction = (
            'Solve the problem using these sections in order.\n'
            'What we know: List the given facts, constraints and requested quantity in at most three short bullets. '
            'Do not invent facts.\n'
            'What we will do: Give at most three short steps explaining your proposed method. '
            'Do not calculate the final answer yet.\n'
            'Solution: Carry out the method and check the calculation.\n')
    else:
        instruction = 'Solve the math problem step by step.\n'
    instruction += r'End with one line: Final answer: \boxed{number}. Put only your final numeric answer in the box.'
    text = instruction + '\n\nQuestion: ' + question + '\n\nResponse:'
    return text, tokenizer.encode(text, add_special_tokens=False)
