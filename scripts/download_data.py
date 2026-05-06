"""Download evaluation datasets used in the RSP paper.

Downloads to eval/data/{dataset}/test.jsonl in jsonl format compatible
with eval/data_loader.py.

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --datasets math500,gsm8k
    python scripts/download_data.py --output_dir eval/data
"""
import argparse
import json
from pathlib import Path
from datasets import load_dataset


def lower_keys(example):
    return {k.lower(): v for k, v in example.items()}


def save_jsonl(examples, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def download_math500(output_dir):
    """MATH-500 (test split, 500 problems)."""
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    examples = [lower_keys(ex) for ex in dataset]
    out = f"{output_dir}/math500/test.jsonl"
    save_jsonl(examples, out)
    print(f"  -> {out}  ({len(examples)} examples)")


def download_gsm8k(output_dir):
    """GSM8K (test split, 1319 problems)."""
    dataset = load_dataset("gsm8k", "main", split="test")
    examples = [lower_keys(ex) for ex in dataset]
    examples = [{**ex, "idx": i} for i, ex in enumerate(examples)]
    out = f"{output_dir}/gsm8k/test.jsonl"
    save_jsonl(examples, out)
    print(f"  -> {out}  ({len(examples)} examples)")


def download_aime24(output_dir):
    """AIME 2024 (30 problems) from AI-MO/aimo-validation-aime, filtered to 2024."""
    dataset = load_dataset("AI-MO/aimo-validation-aime", split="train")
    examples = [lower_keys(ex) for ex in dataset
                if "2024" in str(ex.get("url", ""))]
    # Mirror 'problem' as 'question' for downstream code that references either.
    for ex in examples:
        if "question" not in ex and "problem" in ex:
            ex["question"] = ex["problem"]
    out = f"{output_dir}/aime24/test.jsonl"
    save_jsonl(examples, out)
    print(f"  -> {out}  ({len(examples)} examples)")


def main():
    parser = argparse.ArgumentParser(description="Download RSP evaluation datasets.")
    parser.add_argument("--output_dir", default="eval/data", type=str,
                        help="Output directory (default: eval/data)")
    parser.add_argument("--datasets", default="math500,gsm8k,aime24", type=str,
                        help="Comma-separated subset of {math500, gsm8k, aime24}")
    args = parser.parse_args()

    names = [n.strip() for n in args.datasets.split(",")]
    print(f"Downloading datasets: {names}")
    print(f"Output directory: {args.output_dir}")

    if "math500" in names:
        download_math500(args.output_dir)
    if "gsm8k" in names:
        download_gsm8k(args.output_dir)
    if "aime24" in names:
        download_aime24(args.output_dir)

    print("Done.")


if __name__ == "__main__":
    main()
