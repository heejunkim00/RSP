"""Download the SFT evaluation benchmarks (test.jsonl files from the Qwen2.5-Math evaluation suite).

    python sft/download_eval_data.py      # -> sft/eval_data/<benchmark>/test.jsonl
"""
import argparse
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/QwenLM/Qwen2.5-Math/main/evaluation/data"
BENCHMARKS = ["minerva_math", "gaokao2023en", "olympiadbench", "aime24", "amc23"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="sft/eval_data", type=Path)
    args = ap.parse_args()
    for b in BENCHMARKS:
        out = args.out_dir / b / "test.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(f"{BASE}/{b}/test.jsonl") as r:
            out.write_bytes(r.read())
        print(f"  -> {out}")


if __name__ == "__main__":
    main()
