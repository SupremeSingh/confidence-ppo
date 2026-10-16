from math_rl.prompts import make_prompt
from math_rl.reward import parse_number


def make_row(example, index):
    # GSM8K separates its final numeric answer with ####.
    _, separator, answer = example["answer"].rpartition("####")
    if not separator or parse_number(answer.strip()) is None:
        raise ValueError(f"Missing reference answer at row {index}")
    return {
        "data_source": "gsm8k",
        "prompt": make_prompt(example["question"]),
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": answer.strip()},
        "extra_info": {"index": index, "split": "train", "prompt_id": f"gsm8k/train/{index}"},
    }


def prepare_data(output_dir, revision, seed=42):
    from datasets import Dataset, load_dataset

    output_dir.mkdir(parents=True, exist_ok=True)
    source = load_dataset("openai/gsm8k", "main", split="train", revision=revision)
    source = source.map(make_row, with_indices=True, remove_columns=source.column_names)
    source = source.shuffle(seed=seed)
    # Disjoint subsets of training data; leave the official test set untouched.
    for name, start, stop in [("train", 0, 128), ("val", 128, 160)]:
        subset = Dataset.from_list([source[i] for i in range(start, stop)])
        subset.to_parquet(str(output_dir / f"{name}.parquet"))
