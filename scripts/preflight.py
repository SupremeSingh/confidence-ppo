"""Fail before GPU work when the prepared experiment is incompatible."""
import json
import argparse
from importlib.metadata import distribution
from pathlib import Path
from math_rl.prompts import validate_assets, validate_prompt


def main():
    parser = argparse.ArgumentParser()
    method = parser.add_mutually_exclusive_group()
    method.add_argument("--ppo", action="store_true")
    method.add_argument("--grpo", action="store_true")
    parser.add_argument("--expected-gpus", type=int)
    parser.add_argument("--train-file", type=Path, default=Path('data/gsm8k/train.parquet'))
    parser.add_argument("--val-file", type=Path, default=Path('data/gsm8k/val.parquet'))
    args = parser.parse_args()
    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer
    root = Path(__file__).resolve().parents[1]
    for name in ("configs/environment.txt", "configs/container.sha256"):
        if not (root / name).is_file():
            raise RuntimeError(f"Missing runtime provenance: {name}; follow README setup")
    validate_assets(json.loads((root / "configs/assets.json").read_text()))
    direct = json.loads(distribution("verl").read_text("direct_url.json") or "{}")
    if direct.get("vcs_info", {}).get("commit_id") != "8d9e350ea58c7ad4b50dd14d9dcb50577242c55f":
        raise RuntimeError("VERL is not installed at the pinned commit; rerun container setup")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8:
        raise RuntimeError("Request an Ampere GPU, such as --gres=gpu:a5000:2")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 unavailable in this GPU/runtime combination")
    if args.expected_gpus is not None and torch.cuda.device_count() != args.expected_gpus:
        raise RuntimeError("Visible GPU count differs from trainer.n_gpus_per_node")
    tokenizer = AutoTokenizer.from_pretrained(root / "models/qwen-math", local_files_only=True)
    questions = []
    for path in (args.train_file, args.val_file):
        rows = Dataset.from_parquet(str(root / path))
        for row in rows:
            validate_prompt(row["prompt"])
        questions.append({" ".join(row["prompt"][1]["content"].split()) for row in rows})
        if not (args.ppo or args.grpo):
            rendered = tokenizer.apply_chat_template(rows[0]["prompt"], tokenize=False, add_generation_prompt=True)
            if "\\boxed" in rendered:
                raise RuntimeError("Tokenizer injected a conflicting boxed-answer instruction")
    if questions[0] & questions[1]:
        raise RuntimeError("Training and validation contain overlapping questions")
    if args.ppo or args.grpo:
        from math_rl.ppo_reward import compute_score
        scores = compute_score(["gsm8k", "gsm8k"], [r"Answer: \boxed{4}", "The answer is 5."], ["4", "4"])
        if [s["score"] for s in scores] != [1, 0]:
            raise RuntimeError("Math-Verify integration check failed")
        from math_rl.ppo_checks import numerical_checks, grpo_numerical_checks
        (grpo_numerical_checks if args.grpo else numerical_checks)()
    print(json.dumps({"gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
                      "cuda": torch.version.cuda, "bf16": True}))


if __name__ == "__main__":
    main()
