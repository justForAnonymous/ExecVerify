#!/usr/bin/env python3
import argparse
import json
import os
from typing import Any, Dict

from datasets import load_dataset

try:
    from verl.utils.hdfs_io import copy as hdfs_copy, makedirs as hdfs_makedirs
except ImportError:
    hdfs_copy = None
    hdfs_makedirs = None


DATASET_ID = "PRIME-RL/Eurus-2-RL-Data"

SYSTEM_PROMPT = """
You are a programming expert.
"""


def ensure_serializable(obj: Any) -> str:
    if isinstance(obj, str):
        return obj
    return json.dumps(obj, ensure_ascii=False)


def build_prompt(example_prompt: list) -> list:
    base_prompt_str = example_prompt[1]['content']
    base_prompt_str = "Try to generate code to solve the problem: \n" + base_prompt_str + """

Validate your generated code by the provided input and output pairs.
Output format:
<reasoning>
Your step-by-step reasoning.
</reasoning>
<answer>
```python
The generated code.
```
</answer>
"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": base_prompt_str},
    ]

def convert_example(example: Dict[str, Any], idx: int, split: str) -> Dict[str, Any]:
    gt = ensure_serializable(example["reward_model"]["ground_truth"])
    return {
        "data_source": example.get("data_source", DATASET_ID),
        "ability": example.get("ability", "code"),
        "prompt": build_prompt(example["prompt"]),
        "reward_model": {
            "style": "prime_code",
            "ground_truth": gt,
        },
        "extra_info": {
            **example.get("extra_info", {}),
            "split": split,
            "index": idx,
            "timeout": 10,           
        },
    }


def preprocess(args: argparse.Namespace) -> None:
    os.makedirs(args.local_dir, exist_ok=True)

    raw_ds = load_dataset(DATASET_ID)
    train_ds = raw_ds["train"].filter(lambda row: row.get("ability") == "code")
    val_ds = raw_ds["validation"].filter(lambda row: row.get("ability") == "code")

    train_ds = train_ds.map(lambda ex, idx: convert_example(ex, idx, "train"), with_indices=True)
    val_ds = val_ds.map(lambda ex, idx: convert_example(ex, idx, "validation"), with_indices=True)

    train_path = os.path.join(args.local_dir, "train.parquet")
    val_path = os.path.join(args.local_dir, "val.parquet")
    train_ds.to_parquet(train_path)
    val_ds.to_parquet(val_path)

    if args.hdfs_dir:
        if hdfs_makedirs is None or hdfs_copy is None:
            raise RuntimeError("verl.utils.hdfs_io cannot be found, cannot write to HDFS")
        hdfs_makedirs(args.hdfs_dir)
        hdfs_copy(src=args.local_dir, dst=args.hdfs_dir)

    print(f"Saved train -> {train_path}")
    print(f"Saved val   -> {val_path}")
    if args.hdfs_dir:
        print(f"HDFS copy -> {args.hdfs_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/eurus_prime", type=os.path.expanduser)
    parser.add_argument("--hdfs_dir", default=None, type=str)
    return parser.parse_args()


if __name__ == "__main__":
    preprocess(parse_args())