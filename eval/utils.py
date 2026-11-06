"""I/O helpers and the two prompt templates used in the paper (zero-shot)."""
import json
import os
import random
from pathlib import Path
from typing import Any, Iterable, Union

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def load_jsonl(file: Union[str, Path]) -> Iterable[Any]:
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                print("Error in loading:", line)
                raise


def save_jsonl(samples, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def lower_keys(example):
    return {k.lower(): v for k, v in example.items()}


# {input} is replaced by the problem statement.
PROMPT_TEMPLATES = {
    # Qwen2.5-Math ChatML template (also applied to base models as the "format mismatch" setting)
    "qwen25-math-cot": (
        "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
        "<|im_start|>user\n{input}<|im_end|>\n"
        "<|im_start|>assistant\n"
    ),
    # Raw-text template; wrap with --apply_chat_template for chat models such as LLaMA-3.1-8B-Instruct
    "mathstral": (
        "{input}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
    ),
}


def construct_prompt(question, prompt_type):
    return PROMPT_TEMPLATES[prompt_type].format(input=question).strip(" ")
