"""Download SimpleRL math dataset (level 3-5, qwen-boxed) and convert it
into a chatml-friendly prompt format.

Source dataset:
    https://huggingface.co/datasets/hkust-nlp/SimpleRL-Zoo-Data
    subset: simplelr_qwen_level3to5

Source format problem:
    The shipped `prompt` column is a list of one user message whose `content`
    string already includes <|im_start|>system ... <|im_end|>\n<|im_start|>user
    ... assistant\n template tokens. When verl's tokenizer applies its own
    chat template on top, those template tokens are duplicated.

Fix applied here:
    Rewrite the `prompt` column as a clean two-message list
        [
            {"role": "system", "content": <reasoning instruction>},
            {"role": "user",   "content": <raw question>},
        ]
    Other columns (question, answer, ground_truth, ability, etc.) are kept
    untouched so reward and metadata logic still works.

Output:
    data/simplelr_math_35/{train,test}.parquet  (relative to this file)

Usage:
    cd RSP/dapo
    python data/prepare_data.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download


REPO_ID = "hkust-nlp/SimpleRL-Zoo-Data"
SUBSET = "simplelr_qwen_level3to5"

SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."


def transform_prompt(question: str) -> np.ndarray:
    """Build a [system, user] message list from a raw question string."""
    return np.array(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": str(question)},
        ],
        dtype=object,
    )


def process_split(split: str, out_dir: Path) -> Path:
    print(f"[{split}] downloading from {REPO_ID}/{SUBSET}/{split}.parquet ...")
    local = hf_hub_download(
        repo_id=REPO_ID,
        filename=f"{SUBSET}/{split}.parquet",
        repo_type="dataset",
    )
    df = pd.read_parquet(local)

    # train split has a 'question' column directly; test split keeps
    # the raw question inside extra_info['question'].
    if "question" in df.columns:
        questions = df["question"]
    elif "extra_info" in df.columns:
        questions = df["extra_info"].apply(lambda x: x["question"])
    else:
        raise RuntimeError(
            f"unexpected schema: no 'question' or 'extra_info'. columns={list(df.columns)}"
        )

    print(f"[{split}] {len(df)} rows; rewriting prompt column ...")
    df["prompt"] = questions.apply(transform_prompt)

    out_path = out_dir / f"{split}.parquet"
    df.to_parquet(out_path, index=False)
    print(f"[{split}] -> {out_path}")
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "simplelr_math_35",
        help="Output directory (default: data/simplelr_math_35)",
    )
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {args.out_dir}")

    for split in ("train", "test"):
        process_split(split, args.out_dir)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
