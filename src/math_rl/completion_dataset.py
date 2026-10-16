"""VERL dataset using the exact Stage 1 completion tokens, without a chat template."""
import copy

from datasets import concatenate_datasets, Dataset as HFDataset
import torch
from torch.utils.data import Dataset

from math_rl.prompts import encode_completion, validate_prompt


class CompletionDataset(Dataset):
    def __init__(self, data_files, tokenizer, config, processor=None):
        if processor is not None:
            raise ValueError("This experiment supports text-only Qwen")
        paths = [data_files] if isinstance(data_files, str) else list(data_files)
        source = concatenate_datasets([HFDataset.from_parquet(str(p)) for p in paths])
        self.rows = []
        self.max_length = config.max_prompt_length
        self.pad_id = tokenizer.pad_token_id
        if self.pad_id is None:
            raise ValueError("Tokenizer must define pad_token_id")
        excluded = 0
        for row in source:
            validate_prompt(row["prompt"])
            text, tokens = encode_completion(tokenizer, row["prompt"][1]["content"])
            if len(tokens) > self.max_length:
                if not config.filter_overlong_prompts:
                    raise ValueError("Prompt exceeds token budget")
                excluded += 1
                continue
            if not tokens:
                raise ValueError("Empty prompt")
            row.pop("prompt")
            row["raw_prompt_ids"] = tokens
            row["index"] = row["extra_info"]["index"]
            row["extra_info"]["completion_prompt"] = text
            self.rows.append(row)
        if not self.rows:
            raise ValueError("No usable prompts")
        print(f"Completion dataset: {len(self.rows)} prompts; {excluded} overlong excluded")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = copy.deepcopy(self.rows[index])
        tokens = row["raw_prompt_ids"]
        padding = self.max_length - len(tokens)
        row["input_ids"] = torch.tensor([self.pad_id] * padding + tokens, dtype=torch.long)
        row["attention_mask"] = torch.tensor([0] * padding + [1] * len(tokens), dtype=torch.long)
        row["position_ids"] = (row["attention_mask"].cumsum(-1) - 1).clamp(min=0)
        return row
