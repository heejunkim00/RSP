"""Download OpenR1-Math-46k and save (data_source, problem, answer) as parquet.

    python sft/prepare_data.py            # -> sft/data/openr1_math_46k.parquet (45,792 problems)
"""
import argparse
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="sft/data/openr1_math_46k.parquet", type=str)
    args = parser.parse_args()

    src = hf_hub_download("Elliott/Openr1-Math-46k-8192", "openr1.parquet", repo_type="dataset")
    df = pd.read_parquet(src, columns=["data_source", "prompt", "reward_model"])
    out = pd.DataFrame({
        "data_source": df["data_source"],
        "problem": df["prompt"].apply(lambda msgs: msgs[-1]["content"]),
        "answer": df["reward_model"].apply(lambda r: r["ground_truth"]),
    })
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(f"Saved {len(out)} problems to {args.out}")


if __name__ == "__main__":
    main()
